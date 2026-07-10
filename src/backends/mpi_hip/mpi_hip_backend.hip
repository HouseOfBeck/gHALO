#include "mpi_hip_backend.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <unistd.h>
#include <vector>

namespace ghalo {
namespace {

void hip_check(hipError_t error, const char* call) {
  if (error != hipSuccess) {
    std::ostringstream message;
    message << "HIP call failed: " << call << ": " << hipGetErrorName(error)
            << ": " << hipGetErrorString(error);
    throw std::runtime_error(message.str());
  }
}

void mpi_check(int error, const char* call) {
  if (error != MPI_SUCCESS) {
    char text[MPI_MAX_ERROR_STRING] = {};
    int length = 0;
    MPI_Error_string(error, text, &length);
    std::ostringstream message;
    message << "MPI call failed: " << call << ": "
            << std::string(text, length);
    throw std::runtime_error(message.str());
  }
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

  const int rows = ranks / cols;
  return {rows, cols};
}

int cart_rank(MPI_Comm comm, int row, int col) {
  int rank = MPI_PROC_NULL;
  int coords[2] = {row, col};
  mpi_check(MPI_Cart_rank(comm, coords, &rank), "MPI_Cart_rank");
  return rank;
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

__host__ __device__ float pattern_value(int source_rank, int segment,
                                        int halo_words, int index) {
  return static_cast<float>(source_rank * 100000 + segment * 10000 +
                            halo_words * 10 + (index % 10));
}

__global__ void fill_pattern_kernel(float* data, int halo_words, int rank) {
  const int total = 3 * halo_words;
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total) {
    return;
  }
  const int segment = index < halo_words ? 1 : 2;
  data[index] = pattern_value(rank, segment, halo_words, index);
}

__global__ void fill_rank_kernel(float* data, int count, int rank) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    data[index] = static_cast<float>(rank);
  }
}

void launch_fill_pattern(void* data, std::size_t halo_words, int rank) {
  const int count = static_cast<int>(3 * halo_words);
  const int block = 256;
  const int grid = (count + block - 1) / block;
  hipLaunchKernelGGL(fill_pattern_kernel, dim3(grid), dim3(block), 0, 0,
                     static_cast<float*>(data),
                     static_cast<int>(halo_words), rank);
  hip_check(hipGetLastError(), "fill_pattern_kernel launch");
}

void launch_fill_rank(void* data, std::size_t count, int rank) {
  const int elements = static_cast<int>(count);
  const int block = 256;
  const int grid = (elements + block - 1) / block;
  hipLaunchKernelGGL(fill_rank_kernel, dim3(grid), dim3(block), 0, 0,
                     static_cast<float*>(data), elements, rank);
  hip_check(hipGetLastError(), "fill_rank_kernel launch");
}

std::string hip_runtime_version() {
  int version = 0;
  hip_check(hipRuntimeGetVersion(&version), "hipRuntimeGetVersion");
  std::ostringstream out;
  out << version;
  return out.str();
}

} // namespace

HIPBuffer::~HIPBuffer() { reset(); }

HIPBuffer::HIPBuffer(HIPBuffer&& other) noexcept
    : ptr_(other.ptr_), bytes_(other.bytes_) {
  other.ptr_ = nullptr;
  other.bytes_ = 0;
}

HIPBuffer& HIPBuffer::operator=(HIPBuffer&& other) noexcept {
  if (this != &other) {
    reset();
    ptr_ = other.ptr_;
    bytes_ = other.bytes_;
    other.ptr_ = nullptr;
    other.bytes_ = 0;
  }
  return *this;
}

void HIPBuffer::allocate(std::size_t bytes) {
  if (bytes_ == bytes && ptr_ != nullptr) {
    return;
  }
  reset();
  hip_check(hipMalloc(&ptr_, bytes), "hipMalloc");
  bytes_ = bytes;
}

void HIPBuffer::reset() {
  if (ptr_ != nullptr) {
    hipFree(ptr_);
    ptr_ = nullptr;
    bytes_ = 0;
  }
}

void* HIPBuffer::data() const { return ptr_; }

std::size_t HIPBuffer::bytes() const { return bytes_; }

MPIHIPBackend::MPIHIPBackend(std::string device_map, bool validate,
                             bool allow_oversubscription)
    : device_map_(std::move(device_map)), validate_(validate),
      allow_oversubscription_(allow_oversubscription) {
  if (device_map_ != "local-rank") {
    throw std::invalid_argument("unsupported device map: " + device_map_);
  }
  initialize_topology();
  initialize_device();
  initialize_metadata();
  probe_device_mpi();
}

MPIHIPBackend::~MPIHIPBackend() {
  if (local_comm_ != MPI_COMM_NULL) {
    MPI_Comm_free(&local_comm_);
  }
  if (cart_comm_ != MPI_COMM_NULL) {
    MPI_Comm_free(&cart_comm_);
  }
}

std::string MPIHIPBackend::name() const { return "MPIHIPBackend"; }

std::string MPIHIPBackend::algorithm() const { return "sendrecv"; }

TopologyInfo MPIHIPBackend::topology() const { return topology_; }

BackendMetadata MPIHIPBackend::metadata() const { return metadata_; }

int MPIHIPBackend::rank() const { return topology_.world_rank; }

int MPIHIPBackend::size() const { return topology_.world_size; }

bool MPIHIPBackend::is_root() const { return rank() == 0; }

void MPIHIPBackend::setup(std::size_t halo_words) {
  halo_words_ = halo_words;
  const std::size_t buffer_words = 3 * halo_words_;
  const std::size_t bytes = buffer_words * sizeof(float);

  hins_.allocate(bytes);
  hons_.allocate(bytes);
  hiew_.allocate(bytes);
  hoew_.allocate(bytes);

  launch_fill_rank(hoew_.data(), buffer_words, rank());
  hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize after setup fill");

  if (validate_) {
    validate_exchange();
  }
}

void MPIHIPBackend::exchange() {
  const int n = static_cast<int>(halo_words_);
  const int two_n = 2 * n;
  const std::size_t bytes = 3 * halo_words_ * sizeof(float);

  hip_check(hipMemcpy(hins_.data(), hoew_.data(), bytes,
                      hipMemcpyDeviceToDevice),
            "hipMemcpy hoew -> hins");

  auto* hins = static_cast<float*>(hins_.data());
  auto* hons = static_cast<float*>(hons_.data());
  auto* hiew = static_cast<float*>(hiew_.data());
  auto* hoew = static_cast<float*>(hoew_.data());

  mpi_check(MPI_Sendrecv(hins, n, MPI_FLOAT, topology_.south, 9901, hons, n,
                         MPI_FLOAT, topology_.north, 9901, cart_comm_,
                         MPI_STATUS_IGNORE),
            "MPI_Sendrecv north/south N");

  mpi_check(MPI_Sendrecv(hins + n, two_n, MPI_FLOAT, topology_.north, 9902,
                         hons + n, two_n, MPI_FLOAT, topology_.south, 9902,
                         cart_comm_, MPI_STATUS_IGNORE),
            "MPI_Sendrecv north/south 2N");

  hip_check(hipMemcpy(hiew_.data(), hons_.data(), bytes,
                      hipMemcpyDeviceToDevice),
            "hipMemcpy hons -> hiew");

  mpi_check(MPI_Sendrecv(hiew, n, MPI_FLOAT, topology_.west, 9903, hoew, n,
                         MPI_FLOAT, topology_.east, 9903, cart_comm_,
                         MPI_STATUS_IGNORE),
            "MPI_Sendrecv east/west N");

  mpi_check(MPI_Sendrecv(hiew + n, two_n, MPI_FLOAT, topology_.east, 9904,
                         hoew + n, two_n, MPI_FLOAT, topology_.west, 9904,
                         cart_comm_, MPI_STATUS_IGNORE),
            "MPI_Sendrecv east/west 2N");
}

void MPIHIPBackend::barrier() { mpi_check(MPI_Barrier(cart_comm_), "MPI_Barrier"); }

double MPIHIPBackend::now() const { return MPI_Wtime(); }

double MPIHIPBackend::max_time(double local_seconds) {
  double global_seconds = 0.0;
  mpi_check(MPI_Allreduce(&local_seconds, &global_seconds, 1, MPI_DOUBLE,
                          MPI_MAX, cart_comm_),
            "MPI_Allreduce");
  return global_seconds;
}

void MPIHIPBackend::initialize_topology() {
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

void MPIHIPBackend::initialize_device() {
  mpi_check(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                                MPI_INFO_NULL, &local_comm_),
            "MPI_Comm_split_type");
  mpi_check(MPI_Comm_rank(local_comm_, &local_rank_), "MPI_Comm_rank local");
  mpi_check(MPI_Comm_size(local_comm_, &local_size_), "MPI_Comm_size local");

  hip_check(hipGetDeviceCount(&visible_device_count_), "hipGetDeviceCount");
  if (visible_device_count_ <= 0) {
    throw std::runtime_error(
        "GPU-aware MPI backend requested, but HIP reports no visible devices");
  }

  selected_device_ = local_rank_ % visible_device_count_;
  rocr_visible_devices_ = getenv_string("ROCR_VISIBLE_DEVICES");
  selected_visible_token_ =
      selected_visible_token(rocr_visible_devices_, selected_device_);

  constexpr int token_length = 128;
  char local_token[token_length] = {};
  std::snprintf(local_token, sizeof(local_token), "%s",
                selected_visible_token_.c_str());
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
      if (other == selected_visible_token_) {
        std::ostringstream message;
        message << "invalid rank-to-device mapping on host " << hostname()
                << ": local rank " << local_rank_ << " and local rank " << i
                << " both selected visible HIP device token "
                << selected_visible_token_
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
  hostname_ = hostname();
}

void MPIHIPBackend::initialize_metadata() {
  metadata_.memory_location = "device";
  metadata_.device_map = device_map_;
  metadata_.gpu_aware_mpi_requested = true;
  metadata_.validation_enabled = validate_;
  metadata_.validation_passed = !validate_;
  metadata_.hip_runtime_version = hip_runtime_version();

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
    RankMetadata rank;
    rank.world_rank = packed_ranks[index].world_rank;
    rank.local_rank = packed_ranks[index].local_rank;
    rank.cart_rank = packed_ranks[index].cart_rank;
    rank.row = packed_ranks[index].row;
    rank.col = packed_ranks[index].col;
    rank.hip_device_index = packed_ranks[index].hip_device_index;
    rank.hostname = std::string(hostnames.data() + index * text_length);
    rank.hip_device_name = std::string(devices.data() + index * text_length);
    rank.rocr_visible_devices = std::string(rocrs.data() + index * text_length);
    metadata_.ranks.push_back(rank);
  }
}

void MPIHIPBackend::probe_device_mpi() {
  HIPBuffer send;
  HIPBuffer recv;
  send.allocate(sizeof(float));
  recv.allocate(sizeof(float));
  launch_fill_rank(send.data(), 1, rank());
  hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize before MPI probe");

  mpi_check(MPI_Sendrecv(send.data(), 1, MPI_FLOAT, topology_.east, 9910,
                         recv.data(), 1, MPI_FLOAT, topology_.west, 9910,
                         cart_comm_, MPI_STATUS_IGNORE),
            "MPI_Sendrecv GPU-aware probe");

  float value = -1.0F;
  hip_check(hipMemcpy(&value, recv.data(), sizeof(float),
                      hipMemcpyDeviceToHost),
            "hipMemcpy GPU-aware probe result");
  const float expected = static_cast<float>(topology_.west);
  if (value != expected) {
    std::ostringstream message;
    message << "GPU-aware MPI probe failed on rank " << rank()
            << ": expected value from west rank " << topology_.west
            << ", got " << value;
    throw std::runtime_error(message.str());
  }
}

void MPIHIPBackend::validate_exchange() {
  launch_fill_pattern(hoew_.data(), halo_words_, rank());
  hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize before validation");

  exchange();

  const std::size_t count = 3 * halo_words_;
  std::vector<float> hons(count);
  std::vector<float> hoew(count);
  hip_check(hipMemcpy(hons.data(), hons_.data(), count * sizeof(float),
                      hipMemcpyDeviceToHost),
            "hipMemcpy validation hons");
  hip_check(hipMemcpy(hoew.data(), hoew_.data(), count * sizeof(float),
                      hipMemcpyDeviceToHost),
            "hipMemcpy validation hoew");

  const int north_east =
      cart_rank(cart_comm_, (topology_.row + 1) % topology_.rows,
                (topology_.col + 1) % topology_.cols);
  const int south_west =
      cart_rank(cart_comm_,
                (topology_.row + topology_.rows - 1) % topology_.rows,
                (topology_.col + topology_.cols - 1) % topology_.cols);

  auto check_value = [&](const char* direction, float actual, float expected,
                         std::size_t index) {
    if (actual != expected) {
      std::ostringstream message;
      message << "validation failed on rank " << rank() << " direction "
              << direction << " index " << index << ": expected " << expected
              << ", actual " << actual;
      throw std::runtime_error(message.str());
    }
  };

  for (std::size_t i = 0; i < halo_words_; ++i) {
    check_value("north", hons[i],
                pattern_value(topology_.north, 1, static_cast<int>(halo_words_),
                              static_cast<int>(i)),
                i);
    check_value("east-final", hoew[i],
                pattern_value(north_east, 1, static_cast<int>(halo_words_),
                              static_cast<int>(i)),
                i);
  }
  for (std::size_t i = halo_words_; i < count; ++i) {
    check_value("south", hons[i],
                pattern_value(topology_.south, 2, static_cast<int>(halo_words_),
                              static_cast<int>(i)),
                i);
    check_value("west-final", hoew[i],
                pattern_value(south_west, 2, static_cast<int>(halo_words_),
                              static_cast<int>(i)),
                i);
  }

  int local_ok = 1;
  int global_ok = 0;
  mpi_check(MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN,
                          cart_comm_),
            "MPI_Allreduce validation status");
  if (global_ok != 1) {
    throw std::runtime_error("GPU-aware MPI validation failed");
  }
  metadata_.validation_passed = true;

  launch_fill_rank(hoew_.data(), count, rank());
  hip_check(hipDeviceSynchronize(), "hipDeviceSynchronize after validation");
}

} // namespace ghalo
