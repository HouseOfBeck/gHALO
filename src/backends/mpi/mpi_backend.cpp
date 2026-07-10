#include "mpi_backend.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>

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
}

MPIBackend::~MPIBackend() {
  if (cart_comm_ != MPI_COMM_NULL) {
    MPI_Comm_free(&cart_comm_);
  }
}

std::string MPIBackend::name() const { return "MPIBackend"; }

std::string MPIBackend::algorithm() const { return "sendrecv"; }

TopologyInfo MPIBackend::topology() const { return topology_; }

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

} // namespace ghalo
