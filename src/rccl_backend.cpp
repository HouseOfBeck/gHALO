#include "ghalo/rccl_backend.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <unistd.h>
#include <vector>

namespace ghalo {
namespace {

void mpi_check(int error, const char* operation) {
  if (error != MPI_SUCCESS) {
    char text[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(error, text, &length);
    std::ostringstream message;
    message << "MPI call failed: " << operation << ": "
            << std::string(text, length);
    throw std::runtime_error(message.str());
  }
}

std::string hostname() {
  char name[256] = {};
  if (gethostname(name, sizeof(name) - 1) != 0) {
    return "unknown";
  }
  return name;
}

std::string getenv_string(const char* name) {
  const char* value = std::getenv(name);
  return value == nullptr ? std::string{} : std::string(value);
}

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

  return {ranks / cols, cols};
}

int cart_rank(MPI_Comm comm, int row, int col) {
  int rank = MPI_PROC_NULL;
  int coords[2] = {row, col};
  mpi_check(MPI_Cart_rank(comm, coords, &rank), "MPI_Cart_rank");
  return rank;
}

std::vector<std::string> split_visible_devices(const std::string& value) {
  std::vector<std::string> tokens;
  std::stringstream stream(value);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (!token.empty()) {
      tokens.push_back(token);
    }
  }
  return tokens;
}

std::string selected_visible_token(const std::string& rocr_visible_devices,
                                   int selected_device) {
  const auto tokens = split_visible_devices(rocr_visible_devices);
  if (selected_device >= 0 &&
      selected_device < static_cast<int>(tokens.size())) {
    return tokens[static_cast<std::size_t>(selected_device)];
  }
  return std::to_string(selected_device);
}

float pattern_value(int source_rank, int segment, int halo_words, int index) {
  return static_cast<float>(source_rank * 100000 + segment * 10000 +
                            halo_words * 10 + (index % 10));
}

std::string hip_runtime_version(int rank, int local_rank, int device) {
  int version = 0;
  const hipError_t error = hipRuntimeGetVersion(&version);
  if (error != hipSuccess) {
    std::ostringstream message;
    message << "HIP call failed: hipRuntimeGetVersion: "
            << hipGetErrorName(error) << ": " << hipGetErrorString(error)
            << " (rank=" << rank << " local_rank=" << local_rank
            << " hip_device=" << device << ")";
    throw std::runtime_error(message.str());
  }
  return std::to_string(version);
}

std::string rccl_version(int rank, int local_rank, int device) {
  int version = 0;
  const ncclResult_t error = ncclGetVersion(&version);
  if (error != ncclSuccess) {
    std::ostringstream message;
    message << "RCCL call failed: ncclGetVersion: "
            << ncclGetErrorString(error) << " (rank=" << rank
            << " local_rank=" << local_rank << " hip_device=" << device
            << ")";
    throw std::runtime_error(message.str());
  }
  return std::to_string(version);
}

std::string rocm_version() {
  const std::string loaded = getenv_string("GHALO_LOADED_ROCM_MODULE");
  if (!loaded.empty()) {
    return loaded;
  }
  return getenv_string("ROCM_PATH");
}

std::string values_string(const std::vector<float>& values) {
  std::ostringstream out;
  out << "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    out << values[i];
  }
  out << "]";
  return out.str();
}

} // namespace

class RCCLBackend::DeviceBuffer final {
public:
  DeviceBuffer() = default;
  ~DeviceBuffer() noexcept { reset(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void allocate(std::size_t bytes, const char* label, int rank, int local_rank,
                int device) {
    reset();
    const hipError_t error = hipMalloc(&ptr_, bytes);
    if (error != hipSuccess) {
      std::ostringstream message;
      message << "HIP call failed: hipMalloc " << label << ": "
              << hipGetErrorName(error) << ": " << hipGetErrorString(error)
              << " (rank=" << rank << " local_rank=" << local_rank
              << " hip_device=" << device << ")";
      throw std::runtime_error(message.str());
    }
    bytes_ = bytes;
  }

  void reset() noexcept {
    if (ptr_ != nullptr) {
      (void)hipFree(ptr_);
      ptr_ = nullptr;
      bytes_ = 0;
    }
  }

  void* data() const { return ptr_; }
  std::size_t bytes() const { return bytes_; }

private:
  void* ptr_ = nullptr;
  std::size_t bytes_ = 0;
};

RCCLBackend::RCCLBackend(bool validate, bool allow_oversubscription)
    : validate_(validate), allow_oversubscription_(allow_oversubscription) {
  initialize_topology();
  initialize_local_rank();
  select_device();

  hip_check(hipStreamCreate(&stream_), "hipStreamCreate");
  initialize_rccl_communicator();
  initialize_metadata();
  print_stage_b_startup();
}

RCCLBackend::~RCCLBackend() {
  if (stream_ != nullptr) {
    (void)hipStreamSynchronize(stream_);
  }
  if (rccl_comm_ != nullptr) {
    (void)ncclCommDestroy(rccl_comm_);
    rccl_comm_ = nullptr;
  }
  hons_.reset();
  hins_.reset();
  hoew_.reset();
  if (stream_ != nullptr) {
    (void)hipStreamDestroy(stream_);
    stream_ = nullptr;
  }
  if (local_comm_ != MPI_COMM_NULL) {
    (void)MPI_Comm_free(&local_comm_);
    local_comm_ = MPI_COMM_NULL;
  }
  if (cart_comm_ != MPI_COMM_NULL) {
    (void)MPI_Comm_free(&cart_comm_);
    cart_comm_ = MPI_COMM_NULL;
  }
}

std::string RCCLBackend::name() const { return "RCCLBackend Stage B"; }

std::string RCCLBackend::algorithm() const { return "rccl-north-south-only"; }

TopologyInfo RCCLBackend::topology() const { return topology_; }

BackendMetadata RCCLBackend::metadata() const { return metadata_; }

int RCCLBackend::rank() const { return topology_.world_rank; }

int RCCLBackend::size() const { return topology_.world_size; }

bool RCCLBackend::is_root() const { return topology_.world_rank == 0; }

void RCCLBackend::setup(std::size_t halo_words) {
  halo_words_ = halo_words;
  allocate_buffers(halo_words_);
  if (validate_) {
    validate_north_south();
  }
}

void RCCLBackend::exchange() { fail_not_implemented(); }

void RCCLBackend::barrier() {
  mpi_check(MPI_Barrier(cart_comm_), "MPI_Barrier");
}

double RCCLBackend::now() const { return MPI_Wtime(); }

double RCCLBackend::max_time(double local_seconds) {
  double global_seconds = 0.0;
  mpi_check(MPI_Allreduce(&local_seconds, &global_seconds, 1, MPI_DOUBLE,
                          MPI_MAX, cart_comm_),
            "MPI_Allreduce max_time");
  return global_seconds;
}

void RCCLBackend::run_development_validation(
    const std::vector<std::size_t>& halo_lengths) {
  if (!validate_) {
    throw std::runtime_error(
        "RCCL Stage B requires validation; remove --no-validate");
  }

  for (const auto halo_words : halo_lengths) {
    setup(halo_words);
  }

  int local_ok = 1;
  int global_ok = 0;
  mpi_check(MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN,
                          cart_comm_),
            "MPI_Allreduce RCCL Stage B final status");
  if (rank() == 0) {
    if (global_ok == 1) {
      std::cout << "RCCL north/south validation PASSED\n";
    } else {
      std::cout << "RCCL north/south validation FAILED\n";
    }
  }
  if (global_ok != 1) {
    throw std::runtime_error("RCCL north/south validation failed");
  }
}

bool RCCLBackend::exchange_implemented() const { return false; }

void RCCLBackend::initialize_topology() {
  mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &topology_.world_rank),
            "MPI_Comm_rank");
  mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &topology_.world_size),
            "MPI_Comm_size");

  const auto dims_pair = halo_compatible_dims(topology_.world_size);
  int dims[2] = {dims_pair[0], dims_pair[1]};
  int periods[2] = {1, 1};
  constexpr int reorder = 0;
  mpi_check(MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder,
                            &cart_comm_),
            "MPI_Cart_create");
  if (cart_comm_ == MPI_COMM_NULL) {
    throw std::runtime_error("MPI_Cart_create returned MPI_COMM_NULL");
  }

  mpi_check(MPI_Comm_rank(cart_comm_, &topology_.cart_rank),
            "MPI_Comm_rank cart");
  int coords[2] = {0, 0};
  mpi_check(MPI_Cart_coords(cart_comm_, topology_.cart_rank, 2, coords),
            "MPI_Cart_coords");

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

void RCCLBackend::initialize_local_rank() {
  mpi_check(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                                MPI_INFO_NULL, &local_comm_),
            "MPI_Comm_split_type");
  mpi_check(MPI_Comm_rank(local_comm_, &local_rank_), "MPI_Comm_rank local");
  mpi_check(MPI_Comm_size(local_comm_, &local_size_), "MPI_Comm_size local");
  hostname_ = hostname();
}

void RCCLBackend::select_device() {
  hip_check(hipGetDeviceCount(&visible_device_count_), "hipGetDeviceCount");
  if (visible_device_count_ <= 0) {
    throw std::runtime_error(
        "RCCL backend requested, but HIP reports no visible devices");
  }

  selected_device_ = local_rank_ % visible_device_count_;
  rocr_visible_devices_ = getenv_string("ROCR_VISIBLE_DEVICES");
  const std::string selected_token =
      selected_visible_token(rocr_visible_devices_, selected_device_);

  constexpr int token_length = 128;
  char local_token[token_length] = {};
  std::snprintf(local_token, sizeof(local_token), "%s",
                selected_token.c_str());
  std::vector<char> selected_tokens(static_cast<std::size_t>(local_size_) *
                                   token_length);
  mpi_check(MPI_Allgather(local_token, token_length, MPI_CHAR,
                          selected_tokens.data(), token_length, MPI_CHAR,
                          local_comm_),
            "MPI_Allgather selected GPU tokens");

  if (!allow_oversubscription_) {
    for (int i = 0; i < local_size_; ++i) {
      if (i == local_rank_) {
        continue;
      }
      const std::string other(selected_tokens.data() +
                              static_cast<std::size_t>(i) * token_length);
      if (other == selected_token) {
        std::ostringstream message;
        message << "invalid rank-to-device mapping on host " << hostname_
                << ": local rank " << local_rank_ << " and local rank " << i
                << " both selected visible HIP device token " << selected_token
                << "; pass --allow-gpu-oversubscription to override";
        throw std::runtime_error(message.str());
      }
    }
  }

  hip_check(hipSetDevice(selected_device_), "hipSetDevice");
  hipDeviceProp_t properties{};
  hip_check(hipGetDeviceProperties(&properties, selected_device_),
            "hipGetDeviceProperties");
  device_name_ = properties.name;
}

void RCCLBackend::initialize_rccl_communicator() {
  ncclUniqueId id{};
  if (rank() == 0) {
    rccl_check(ncclGetUniqueId(&id), "ncclGetUniqueId");
  }
  mpi_check(MPI_Bcast(&id, static_cast<int>(sizeof(id)), MPI_BYTE, 0,
                      MPI_COMM_WORLD),
            "MPI_Bcast ncclUniqueId");
  rccl_check(ncclCommInitRank(&rccl_comm_, topology_.world_size, id,
                              topology_.world_rank),
             "ncclCommInitRank");
}

void RCCLBackend::initialize_metadata() {
  metadata_.memory_location = "device";
  metadata_.device_map = "local-rank";
  metadata_.gpu_aware_mpi_requested = false;
  metadata_.validation_enabled = validate_;
  metadata_.validation_passed = !validate_;
  metadata_.rccl_stage = "north-south-only";
  metadata_.hip_runtime_version =
      hip_runtime_version(rank(), local_rank_, selected_device_);
  metadata_.rccl_version = rccl_version(rank(), local_rank_, selected_device_);
  metadata_.rocm_version = rocm_version();
  metadata_.rccl_plugin_root = getenv_string("OLCF_OFI_NCCL_ROOT");

  int version_length = 0;
  char version[MPI_MAX_LIBRARY_VERSION_STRING] = {};
  mpi_check(MPI_Get_library_version(version, &version_length),
            "MPI_Get_library_version");
  metadata_.mpi_library_version = std::string(version, version_length);

  struct PackedRank {
    int world_rank;
    int local_rank;
    int cart_rank;
    int row;
    int col;
    int hip_device_index;
  };

  PackedRank local{topology_.world_rank, local_rank_, topology_.cart_rank,
                   topology_.row,        topology_.col, selected_device_};
  std::vector<PackedRank> packed_ranks(topology_.world_size);
  mpi_check(MPI_Allgather(&local, static_cast<int>(sizeof(PackedRank)),
                          MPI_BYTE, packed_ranks.data(),
                          static_cast<int>(sizeof(PackedRank)), MPI_BYTE,
                          MPI_COMM_WORLD),
            "MPI_Allgather rank metadata");

  constexpr int text_length = 256;
  char local_hostname[text_length] = {};
  char local_device[text_length] = {};
  char local_rocr[text_length] = {};
  std::snprintf(local_hostname, sizeof(local_hostname), "%s",
                hostname_.c_str());
  std::snprintf(local_device, sizeof(local_device), "%s",
                device_name_.c_str());
  std::snprintf(local_rocr, sizeof(local_rocr), "%s",
                rocr_visible_devices_.c_str());

  std::vector<char> hostnames(static_cast<std::size_t>(topology_.world_size) *
                             text_length);
  std::vector<char> devices(static_cast<std::size_t>(topology_.world_size) *
                            text_length);
  std::vector<char> rocrs(static_cast<std::size_t>(topology_.world_size) *
                          text_length);
  mpi_check(MPI_Allgather(local_hostname, text_length, MPI_CHAR,
                          hostnames.data(), text_length, MPI_CHAR,
                          MPI_COMM_WORLD),
            "MPI_Allgather hostnames");
  mpi_check(MPI_Allgather(local_device, text_length, MPI_CHAR, devices.data(),
                          text_length, MPI_CHAR, MPI_COMM_WORLD),
            "MPI_Allgather device names");
  mpi_check(MPI_Allgather(local_rocr, text_length, MPI_CHAR, rocrs.data(),
                          text_length, MPI_CHAR, MPI_COMM_WORLD),
            "MPI_Allgather ROCR_VISIBLE_DEVICES");

  metadata_.ranks.clear();
  metadata_.ranks.reserve(static_cast<std::size_t>(topology_.world_size));
  for (int i = 0; i < topology_.world_size; ++i) {
    const auto index = static_cast<std::size_t>(i);
    RankMetadata rank_metadata;
    rank_metadata.world_rank = packed_ranks[index].world_rank;
    rank_metadata.local_rank = packed_ranks[index].local_rank;
    rank_metadata.cart_rank = packed_ranks[index].cart_rank;
    rank_metadata.row = packed_ranks[index].row;
    rank_metadata.col = packed_ranks[index].col;
    rank_metadata.hip_device_index = packed_ranks[index].hip_device_index;
    rank_metadata.hostname =
        std::string(hostnames.data() + index * text_length);
    rank_metadata.hip_device_name =
        std::string(devices.data() + index * text_length);
    rank_metadata.rocr_visible_devices =
        std::string(rocrs.data() + index * text_length);
    metadata_.ranks.push_back(rank_metadata);
  }
}

void RCCLBackend::allocate_buffers(std::size_t halo_words) {
  const std::size_t bytes = 3 * halo_words * sizeof(float);
  if (!hins_) {
    hins_ = std::make_unique<DeviceBuffer>();
    hons_ = std::make_unique<DeviceBuffer>();
    hoew_ = std::make_unique<DeviceBuffer>();
  }
  hins_->allocate(bytes, "hins", rank(), local_rank_, selected_device_);
  hons_->allocate(bytes, "hons", rank(), local_rank_, selected_device_);
  hoew_->allocate(bytes, "hoew", rank(), local_rank_, selected_device_);
}

void RCCLBackend::fill_pattern() {
  const std::size_t count = 3 * halo_words_;
  std::vector<float> host(count);
  for (std::size_t i = 0; i < count; ++i) {
    const int segment = i < halo_words_ ? 1 : 2;
    host[i] = pattern_value(rank(), segment, static_cast<int>(halo_words_),
                            static_cast<int>(i));
  }
  hip_check(hipMemcpyAsync(hoew_->data(), host.data(), count * sizeof(float),
                           hipMemcpyHostToDevice, stream_),
            "hipMemcpyAsync validation pattern");
  hip_check(hipStreamSynchronize(stream_),
            "hipStreamSynchronize after validation pattern");
}

void RCCLBackend::copy_hoew_to_hins() {
  hip_check(hipMemcpyAsync(hins_->data(), hoew_->data(), hoew_->bytes(),
                           hipMemcpyDeviceToDevice, stream_),
            "hipMemcpyAsync hoew -> hins");
  hip_check(hipStreamSynchronize(stream_),
            "hipStreamSynchronize after hoew -> hins");
}

void RCCLBackend::exchange_north_south() {
  const std::size_t n = halo_words_;
  const std::size_t two_n = 2 * halo_words_;
  auto* hins = static_cast<float*>(hins_->data());
  auto* hons = static_cast<float*>(hons_->data());

  if (n == 0) {
    return;
  }

  rccl_check(ncclGroupStart(), "ncclGroupStart north/south");
  rccl_check(ncclSend(hins, n, ncclFloat, topology_.south, rccl_comm_, stream_),
             "ncclSend north/south N");
  rccl_check(ncclRecv(hons, n, ncclFloat, topology_.north, rccl_comm_, stream_),
             "ncclRecv north/south N");
  rccl_check(ncclSend(hins + n, two_n, ncclFloat, topology_.north, rccl_comm_,
                      stream_),
             "ncclSend north/south 2N");
  rccl_check(ncclRecv(hons + n, two_n, ncclFloat, topology_.south, rccl_comm_,
                      stream_),
             "ncclRecv north/south 2N");
  rccl_check(ncclGroupEnd(), "ncclGroupEnd north/south");
  hip_check(hipStreamSynchronize(stream_),
            "hipStreamSynchronize after north/south RCCL");
}

void RCCLBackend::validate_north_south() {
  fill_pattern();
  copy_hoew_to_hins();
  exchange_north_south();

  const std::size_t count = 3 * halo_words_;
  std::vector<float> host(count);
  hip_check(hipMemcpyAsync(host.data(), hons_->data(), count * sizeof(float),
                           hipMemcpyDeviceToHost, stream_),
            "hipMemcpyAsync validation hons");
  hip_check(hipStreamSynchronize(stream_),
            "hipStreamSynchronize after validation hons copy");

  bool local_valid = true;
  std::string first_mismatch;
  auto check_value = [&](const char* direction, float actual, float expected,
                         std::size_t index) {
    if (local_valid && actual != expected) {
      const std::size_t sample_count = std::min<std::size_t>(8, host.size());
      std::vector<float> prefix(host.begin(), host.begin() + sample_count);
      std::ostringstream message;
      message << "RCCL north/south validation failed on rank " << rank()
              << " direction " << direction << " index " << index
              << ": expected " << expected << ", actual " << actual
              << "; cart_dims=(" << topology_.rows << "," << topology_.cols
              << ") cart_coords=(" << topology_.row << "," << topology_.col
              << ") neighbors north=" << topology_.north
              << " south=" << topology_.south << " east=" << topology_.east
              << " west=" << topology_.west
              << " hons_prefix=" << values_string(prefix);
      first_mismatch = message.str();
      local_valid = false;
    }
  };

  for (std::size_t i = 0; i < halo_words_; ++i) {
    check_value("north", host[i],
                pattern_value(topology_.north, 1, static_cast<int>(halo_words_),
                              static_cast<int>(i)),
                i);
  }
  for (std::size_t i = halo_words_; i < count; ++i) {
    check_value("south", host[i],
                pattern_value(topology_.south, 2, static_cast<int>(halo_words_),
                              static_cast<int>(i)),
                i);
  }

  if (!local_valid) {
    std::cerr << first_mismatch << '\n';
  }

  int local_ok = local_valid ? 1 : 0;
  int global_ok = 0;
  mpi_check(MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN,
                          cart_comm_),
            "MPI_Allreduce RCCL north/south validation");
  if (global_ok != 1) {
    throw std::runtime_error("RCCL north/south validation failed");
  }
  metadata_.validation_passed = true;
}

void RCCLBackend::print_stage_b_startup() const {
  if (rank() != 0) {
    return;
  }
  std::cout << "backend: RCCLBackend Stage B\n"
            << "rccl_stage: north-south-only\n"
            << "rocm_version: " << metadata_.rocm_version << "\n"
            << "hip_runtime_version: " << metadata_.hip_runtime_version << "\n"
            << "rccl_version: " << metadata_.rccl_version << "\n"
            << "mpi_world_size: " << topology_.world_size << "\n"
            << "cartesian_dimensions: " << topology_.rows << " x "
            << topology_.cols << "\n";
  if (!metadata_.rccl_plugin_root.empty()) {
    std::cout << "rccl_plugin_root: " << metadata_.rccl_plugin_root << "\n";
  }
  if (!metadata_.ranks.empty() && metadata_.ranks.size() <= 16) {
    for (const auto& rank_metadata : metadata_.ranks) {
      std::cout << "rank_detail global_rank=" << rank_metadata.world_rank
                << " local_rank=" << rank_metadata.local_rank
                << " hostname=" << rank_metadata.hostname
                << " selected_hip_device="
                << rank_metadata.hip_device_index << " gpu_name=\""
                << rank_metadata.hip_device_name << "\"\n";
    }
  }
}

void RCCLBackend::hip_check(hipError_t error, const char* operation) const {
  if (error != hipSuccess) {
    std::ostringstream message;
    message << "HIP call failed: " << operation << ": "
            << hipGetErrorName(error) << ": " << hipGetErrorString(error)
            << " (rank=" << rank() << " local_rank=" << local_rank_
            << " hip_device=" << selected_device_ << ")";
    throw std::runtime_error(message.str());
  }
}

void RCCLBackend::rccl_check(ncclResult_t error, const char* operation) const {
  if (error != ncclSuccess) {
    std::ostringstream message;
    message << "RCCL call failed: " << operation << ": "
            << ncclGetErrorString(error) << " (rank=" << rank()
            << " local_rank=" << local_rank_
            << " hip_device=" << selected_device_ << ")";
    throw std::runtime_error(message.str());
  }
}

void RCCLBackend::fail_not_implemented() const {
  throw std::runtime_error(
      "RCCL backend currently implements north/south validation only; full "
      "halo exchange is not implemented");
}

} // namespace ghalo
