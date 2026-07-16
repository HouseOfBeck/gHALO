#include "mpi_backend.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unistd.h>
#include <vector>

namespace ghalo {
namespace {

std::array<int, 2> halo_compatible_dims(int ranks) {
  int cols = 1;
  int remaining = ranks;

  if (ranks % 25 == 0) {
    cols = 5;
    remaining = ranks / 5;
  } else if (ranks % 9 == 0) {
    cols = 3;
    remaining = ranks / 3;
  }

  while (remaining % 2 == 0 && cols * cols < ranks && remaining != 1) {
    remaining /= 2;
    cols *= 2;
  }

  const int rows = ranks / cols;
  return {rows, cols};
}

int cart_rank(MPI_Comm comm, int row, int col) {
  int rank = MPI_PROC_NULL;
  int coords[2] = {row, col};
  MPI_Cart_rank(comm, coords, &rank);
  return rank;
}

} // namespace

MPIEnvironment::MPIEnvironment(int& argc, char**& argv) {
  MPI_Init(&argc, &argv);
}

MPIEnvironment::~MPIEnvironment() {
  int finalized = 0;
  MPI_Finalized(&finalized);
  if (!finalized) {
    MPI_Finalize();
  }
}

int MPIEnvironment::rank() const {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  return rank;
}

void MPIEnvironment::abort(int error_code) const {
  MPI_Abort(MPI_COMM_WORLD, error_code);
  std::abort();
}

MPIBackend::MPIBackend() {
  initialize_topology();
  initialize_metadata();
}

MPIBackend::~MPIBackend() {
  if (cart_comm_ != MPI_COMM_NULL) {
    MPI_Comm_free(&cart_comm_);
  }
}

std::string MPIBackend::name() const { return "MPIBackend"; }

std::string MPIBackend::algorithm() const { return "sendrecv"; }

TopologyInfo MPIBackend::topology() const { return topology_; }

BackendMetadata MPIBackend::metadata() const { return metadata_; }

int MPIBackend::rank() const { return topology_.world_rank; }

int MPIBackend::size() const { return topology_.world_size; }

bool MPIBackend::is_root() const { return rank() == 0; }

void MPIBackend::setup(std::size_t halo_words) {
  halo_words_ = halo_words;
  const std::size_t buffer_words = 3 * halo_words_;

  hins_.assign(buffer_words, 0.0F);
  hons_.assign(buffer_words, 0.0F);
  hiew_.assign(buffer_words, 0.0F);

  if (hoew_.size() != buffer_words) {
    hoew_.assign(buffer_words, static_cast<float>(rank()));
  }
}

void MPIBackend::exchange() {
  const int n = static_cast<int>(halo_words_);
  const int two_n = 2 * n;

  std::copy(hoew_.begin(), hoew_.end(), hins_.begin());

  MPI_Sendrecv(hins_.data(), n, MPI_FLOAT, topology_.south, 9901, hons_.data(),
               n, MPI_FLOAT, topology_.north, 9901, cart_comm_,
               MPI_STATUS_IGNORE);

  MPI_Sendrecv(hins_.data() + n, two_n, MPI_FLOAT, topology_.north, 9902,
               hons_.data() + n, two_n, MPI_FLOAT, topology_.south, 9902,
               cart_comm_, MPI_STATUS_IGNORE);

  std::copy(hons_.begin(), hons_.end(), hiew_.begin());

  MPI_Sendrecv(hiew_.data(), n, MPI_FLOAT, topology_.west, 9903, hoew_.data(),
               n, MPI_FLOAT, topology_.east, 9903, cart_comm_,
               MPI_STATUS_IGNORE);

  MPI_Sendrecv(hiew_.data() + n, two_n, MPI_FLOAT, topology_.east, 9904,
               hoew_.data() + n, two_n, MPI_FLOAT, topology_.west, 9904,
               cart_comm_, MPI_STATUS_IGNORE);
}

void MPIBackend::barrier() { MPI_Barrier(cart_comm_); }

double MPIBackend::now() const { return MPI_Wtime(); }

double MPIBackend::max_time(double local_seconds) {
  double global_seconds = 0.0;
  MPI_Allreduce(&local_seconds, &global_seconds, 1, MPI_DOUBLE, MPI_MAX,
                cart_comm_);
  return global_seconds;
}

std::vector<IterationTimingReduction> MPIBackend::max_time_ranks(
    const std::vector<double>& local_seconds) {
  struct TimeRank {
    double seconds;
    int rank;
  };
  std::vector<TimeRank> local;
  std::vector<TimeRank> global(local_seconds.size());
  local.reserve(local_seconds.size());
  for (const double seconds : local_seconds) {
    local.push_back({seconds, topology_.world_rank});
  }
  MPI_Allreduce(local.data(), global.data(), static_cast<int>(local.size()),
                MPI_DOUBLE_INT, MPI_MAXLOC, cart_comm_);
  std::vector<IterationTimingReduction> reductions;
  reductions.reserve(global.size());
  for (const auto& entry : global) {
    reductions.push_back({entry.seconds, entry.rank});
  }
  return reductions;
}

void MPIBackend::initialize_topology() {
  MPI_Comm_rank(MPI_COMM_WORLD, &topology_.world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &topology_.world_size);

  const auto dims_pair = halo_compatible_dims(topology_.world_size);
  int dims[2] = {dims_pair[0], dims_pair[1]};
  int periods[2] = {1, 1};
  constexpr int reorder = 0;

  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm_);
  if (cart_comm_ == MPI_COMM_NULL) {
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
  }

  MPI_Comm_rank(cart_comm_, &topology_.cart_rank);

  int coords[2] = {0, 0};
  MPI_Cart_coords(cart_comm_, topology_.cart_rank, 2, coords);

  topology_.rows = dims[0];
  topology_.cols = dims[1];
  topology_.row = coords[0];
  topology_.col = coords[1];
  topology_.north = cart_rank(cart_comm_, (topology_.row + 1) % topology_.rows,
                              topology_.col);
  topology_.south =
      cart_rank(cart_comm_,
                (topology_.row + topology_.rows - 1) % topology_.rows,
                topology_.col);
  topology_.east = cart_rank(cart_comm_, topology_.row,
                             (topology_.col + 1) % topology_.cols);
  topology_.west = cart_rank(
      cart_comm_, topology_.row,
      (topology_.col + topology_.cols - 1) % topology_.cols);
}

void MPIBackend::initialize_metadata() {
  metadata_.memory_location = "host";

  int version_length = 0;
  char version[MPI_MAX_LIBRARY_VERSION_STRING] = {};
  MPI_Get_library_version(version, &version_length);
  metadata_.mpi_library_version = std::string(version, version_length);

  MPI_Comm local_comm = MPI_COMM_NULL;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &local_comm);

  int local_rank = 0;
  MPI_Comm_rank(local_comm, &local_rank);

  char hostname[256] = {};
  if (gethostname(hostname, sizeof(hostname) - 1) != 0) {
    std::snprintf(hostname, sizeof(hostname), "unknown");
  }

  constexpr int hostname_length = 256;
  std::vector<char> hostnames(static_cast<std::size_t>(topology_.world_size) *
                             hostname_length);
  MPI_Allgather(hostname, hostname_length, MPI_CHAR, hostnames.data(),
                hostname_length, MPI_CHAR, MPI_COMM_WORLD);

  RankMetadata local;
  local.world_rank = topology_.world_rank;
  local.local_rank = local_rank;
  local.cart_rank = topology_.cart_rank;
  local.row = topology_.row;
  local.col = topology_.col;

  struct PackedRank {
    int world_rank;
    int local_rank;
    int cart_rank;
    int row;
    int col;
    int hip_device_index;
  };

  PackedRank packed{local.world_rank, local.local_rank, local.cart_rank,
                    local.row,        local.col,        local.hip_device_index};
  std::vector<PackedRank> packed_ranks(topology_.world_size);
  MPI_Allgather(&packed, static_cast<int>(sizeof(PackedRank)), MPI_BYTE,
                packed_ranks.data(), static_cast<int>(sizeof(PackedRank)),
                MPI_BYTE, MPI_COMM_WORLD);

  metadata_.ranks.clear();
  metadata_.ranks.reserve(static_cast<std::size_t>(topology_.world_size));
  for (int i = 0; i < topology_.world_size; ++i) {
    RankMetadata rank;
    rank.world_rank = packed_ranks[static_cast<std::size_t>(i)].world_rank;
    rank.local_rank = packed_ranks[static_cast<std::size_t>(i)].local_rank;
    rank.cart_rank = packed_ranks[static_cast<std::size_t>(i)].cart_rank;
    rank.row = packed_ranks[static_cast<std::size_t>(i)].row;
    rank.col = packed_ranks[static_cast<std::size_t>(i)].col;
    rank.hip_device_index =
        packed_ranks[static_cast<std::size_t>(i)].hip_device_index;
    rank.hostname =
        std::string(hostnames.data() + static_cast<std::size_t>(i) *
                                         hostname_length);
    metadata_.ranks.push_back(rank);
  }

  MPI_Comm_free(&local_comm);
}

} // namespace ghalo
