#include "ghalo/rccl_backend.hpp"

#include "ghalo/exchange.hpp"

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

RCCLBackend::SyncMode parse_sync_mode(const std::string& value,
                                      bool stage_b_only) {
  if (value == "conservative") {
    return RCCLBackend::SyncMode::Conservative;
  }
  if (value == "stream-ordered") {
    if (stage_b_only) {
      throw std::runtime_error(
          "--rccl-sync-mode stream-ordered requires the full RCCL backend; "
          "Stage B supports conservative north/south validation only");
    }
    return RCCLBackend::SyncMode::StreamOrdered;
  }
  throw std::invalid_argument("invalid RCCL sync mode: " + value);
}

std::string sync_mode_name(RCCLBackend::SyncMode mode) {
  switch (mode) {
  case RCCLBackend::SyncMode::Conservative:
    return "conservative";
  case RCCLBackend::SyncMode::StreamOrdered:
    return "stream-ordered";
  }
  return "unknown";
}

std::string synchronization_model(RCCLBackend::SyncMode mode,
                                  bool stage_b_only) {
  if (stage_b_only) {
    return "single_stream_synchronization";
  }
  switch (mode) {
  case RCCLBackend::SyncMode::Conservative:
    return "three_stream_synchronizations";
  case RCCLBackend::SyncMode::StreamOrdered:
    return "one_final_stream_sync";
  }
  return {};
}

std::string active_phase_names(RCCLBackend::SyncMode mode) {
  switch (mode) {
  case RCCLBackend::SyncMode::Conservative:
    return "north_south_communication,north_south_sync,transpose_copy,"
           "transpose_sync,east_west_communication,east_west_sync";
  case RCCLBackend::SyncMode::StreamOrdered:
    return "north_south_communication_enqueue,transpose_copy_enqueue,"
           "east_west_communication_enqueue,final_stream_sync";
  }
  return {};
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

RCCLBackend::RCCLBackend(bool validate, bool allow_oversubscription,
                         bool stage_b_only, const std::string& sync_mode)
    : validate_(validate), allow_oversubscription_(allow_oversubscription),
      stage_b_only_(stage_b_only),
      sync_mode_(parse_sync_mode(sync_mode, stage_b_only)) {
  initialize_topology();
  initialize_local_rank();
  select_device();

  hip_check(hipStreamCreate(&stream_), "hipStreamCreate");
  initialize_rccl_communicator();
  initialize_metadata();
  print_startup();
}

RCCLBackend::~RCCLBackend() {
  print_validation_summary();
  if (stream_ != nullptr) {
    (void)hipStreamSynchronize(stream_);
  }
  if (rccl_comm_ != nullptr) {
    (void)ncclCommDestroy(rccl_comm_);
    rccl_comm_ = nullptr;
  }
  hons_.reset();
  hins_.reset();
  hiew_.reset();
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

std::string RCCLBackend::name() const {
  return stage_b_only_ ? "RCCLBackend Stage B" : "RCCLBackend";
}

std::string RCCLBackend::algorithm() const {
  return stage_b_only_ ? "rccl-north-south-only" : "rccl";
}

TopologyInfo RCCLBackend::topology() const { return topology_; }

BackendMetadata RCCLBackend::metadata() const { return metadata_; }

int RCCLBackend::rank() const { return topology_.world_rank; }

int RCCLBackend::size() const { return topology_.world_size; }

bool RCCLBackend::is_root() const { return topology_.world_rank == 0; }

void RCCLBackend::setup(std::size_t halo_words) {
  halo_words_ = halo_words;
  allocate_buffers(halo_words_);
  if (validate_) {
    if (stage_b_only_) {
      validate_north_south();
    } else {
      validate_full_exchange();
    }
  }
}

void RCCLBackend::exchange() {
  if (stage_b_only_) {
    fail_not_implemented();
  }
  if (sync_mode_ == SyncMode::StreamOrdered) {
    exchange_stream_ordered();
    return;
  }
  exchange_conservative();
}

void RCCLBackend::exchange_conservative() {
  copy_hoew_to_hins();
  if (!phase_timing_collecting_) {
    exchange_north_south();
    copy_hons_to_hiew();
    exchange_east_west();
    return;
  }

  double start = now();
  enqueue_north_south();
  phase_timing_.north_south_communication_seconds += now() - start;

  // Keep the initial RCCL backend correctness-first: RCCL completion on the
  // stream is the synchronization relationship before HIP reads hons_, and
  // before RCCL later reads hiew_ after the transpose copy.
  start = now();
  synchronize_stream("hipStreamSynchronize after north/south RCCL");
  phase_timing_.north_south_sync_seconds += now() - start;

  start = now();
  enqueue_hons_to_hiew_copy();
  phase_timing_.transpose_copy_seconds += now() - start;

  start = now();
  synchronize_stream("hipStreamSynchronize after hons -> hiew");
  phase_timing_.transpose_sync_seconds += now() - start;

  start = now();
  enqueue_east_west();
  phase_timing_.east_west_communication_seconds += now() - start;

  start = now();
  synchronize_stream("hipStreamSynchronize after east/west RCCL");
  phase_timing_.east_west_sync_seconds += now() - start;
}

void RCCLBackend::exchange_stream_ordered() {
  if (phase_timing_collecting_) {
    exchange_stream_ordered_with_phase_timing();
    return;
  }
  enqueue_hoew_to_hins_copy();
  enqueue_north_south();
  enqueue_hons_to_hiew_copy();
  enqueue_east_west();
  synchronize_stream("hipStreamSynchronize after stream-ordered RCCL exchange");
}

void RCCLBackend::exchange_stream_ordered_with_phase_timing() {
  enqueue_hoew_to_hins_copy();

  double start = now();
  enqueue_north_south();
  phase_timing_.north_south_communication_enqueue_seconds += now() - start;

  start = now();
  enqueue_hons_to_hiew_copy();
  phase_timing_.transpose_copy_enqueue_seconds += now() - start;

  start = now();
  enqueue_east_west();
  phase_timing_.east_west_communication_enqueue_seconds += now() - start;

  start = now();
  synchronize_stream("hipStreamSynchronize after stream-ordered RCCL exchange");
  phase_timing_.final_stream_sync_seconds += now() - start;
}

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

bool RCCLBackend::exchange_implemented() const { return !stage_b_only_; }

bool RCCLBackend::supports_phase_timing() const { return !stage_b_only_; }

void RCCLBackend::set_phase_timing_enabled(bool enabled) {
  if (enabled && stage_b_only_) {
    throw std::runtime_error(
        "phase timing is not supported by RCCL Stage B debug mode");
  }
  phase_timing_enabled_ = enabled;
  metadata_.phase_timing_enabled = enabled;
  if (enabled) {
    metadata_.phase_timing_source = "MPI_Wtime";
    metadata_.phase_timing_aggregation = "maximum local average across ranks";
    metadata_.phase_timing_active_phases = active_phase_names(sync_mode_);
  } else {
    metadata_.phase_timing_source.clear();
    metadata_.phase_timing_aggregation.clear();
    metadata_.phase_timing_active_phases.clear();
    phase_timing_collecting_ = false;
  }
}

bool RCCLBackend::phase_timing_enabled() const {
  return phase_timing_enabled_;
}

void RCCLBackend::reset_phase_timing() {
  phase_timing_ = {};
  phase_timing_collecting_ = phase_timing_enabled_;
}

PhaseTimingResult RCCLBackend::phase_timing_result(int iterations,
                                                   double total_seconds) {
  if (iterations <= 0) {
    throw std::invalid_argument("phase timing iterations must be positive");
  }
  phase_timing_collecting_ = false;

  PhaseTimingResult result;
  if (sync_mode_ == SyncMode::StreamOrdered) {
    result.north_south_communication_enqueue_seconds = reduced_phase_average(
        phase_timing_.north_south_communication_enqueue_seconds, iterations);
    result.transpose_copy_enqueue_seconds = reduced_phase_average(
        phase_timing_.transpose_copy_enqueue_seconds, iterations);
    result.east_west_communication_enqueue_seconds = reduced_phase_average(
        phase_timing_.east_west_communication_enqueue_seconds, iterations);
    result.final_stream_sync_seconds =
        reduced_phase_average(phase_timing_.final_stream_sync_seconds,
                              iterations);
    result.phase_sum_seconds = phase_timing_sum(result);
    result.total_exchange_seconds = total_seconds;
    result.total_minus_sum_of_phase_maxima_seconds =
        total_seconds - result.phase_sum_seconds;
    result.unattributed_seconds =
        result.total_minus_sum_of_phase_maxima_seconds;
    return result;
  }
  result.north_south_communication_seconds = reduced_phase_average(
      phase_timing_.north_south_communication_seconds, iterations);
  result.north_south_sync_seconds =
      reduced_phase_average(phase_timing_.north_south_sync_seconds,
                            iterations);
  result.transpose_copy_seconds =
      reduced_phase_average(phase_timing_.transpose_copy_seconds, iterations);
  result.transpose_sync_seconds =
      reduced_phase_average(phase_timing_.transpose_sync_seconds, iterations);
  result.east_west_communication_seconds =
      reduced_phase_average(phase_timing_.east_west_communication_seconds,
                            iterations);
  result.east_west_sync_seconds =
      reduced_phase_average(phase_timing_.east_west_sync_seconds, iterations);
  result.phase_sum_seconds = phase_timing_sum(result);
  result.total_exchange_seconds = total_seconds;
  result.total_minus_sum_of_phase_maxima_seconds =
      total_seconds - result.phase_sum_seconds;
  result.unattributed_seconds =
      result.total_minus_sum_of_phase_maxima_seconds;
  return result;
}

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
  metadata_.rccl_stage = stage_b_only_ ? "north-south-only" : "full";
  metadata_.rccl_sync_mode = sync_mode_name(sync_mode_);
  metadata_.synchronization_model =
      synchronization_model(sync_mode_, stage_b_only_);
  metadata_.hip_runtime_version =
      hip_runtime_version(rank(), local_rank_, selected_device_);
  metadata_.rccl_version = rccl_version(rank(), local_rank_, selected_device_);
  metadata_.rocm_version = rocm_version();
  metadata_.rccl_plugin_root = getenv_string("OLCF_OFI_NCCL_ROOT");
  metadata_.transport_provider = getenv_string("FI_PROVIDER");

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
  const std::size_t bytes = halo_buffer_words(halo_words) * sizeof(float);
  if (!hins_) {
    hins_ = std::make_unique<DeviceBuffer>();
    hons_ = std::make_unique<DeviceBuffer>();
    hiew_ = std::make_unique<DeviceBuffer>();
    hoew_ = std::make_unique<DeviceBuffer>();
  }
  hins_->allocate(bytes, "hins", rank(), local_rank_, selected_device_);
  hons_->allocate(bytes, "hons", rank(), local_rank_, selected_device_);
  hiew_->allocate(bytes, "hiew", rank(), local_rank_, selected_device_);
  hoew_->allocate(bytes, "hoew", rank(), local_rank_, selected_device_);
}

void RCCLBackend::fill_pattern() {
  const std::size_t count = halo_buffer_words(halo_words_);
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

void RCCLBackend::fill_rank() {
  const std::size_t count = halo_buffer_words(halo_words_);
  std::vector<float> host(count, static_cast<float>(rank()));
  hip_check(hipMemcpyAsync(hoew_->data(), host.data(), count * sizeof(float),
                           hipMemcpyHostToDevice, stream_),
            "hipMemcpyAsync rank fill");
  hip_check(hipStreamSynchronize(stream_),
            "hipStreamSynchronize after rank fill");
}

void RCCLBackend::copy_hoew_to_hins() {
  enqueue_hoew_to_hins_copy();
  synchronize_stream("hipStreamSynchronize after hoew -> hins");
}

void RCCLBackend::enqueue_hoew_to_hins_copy() {
  hip_check(hipMemcpyAsync(hins_->data(), hoew_->data(), hoew_->bytes(),
                           hipMemcpyDeviceToDevice, stream_),
            "hipMemcpyAsync hoew -> hins");
}

void RCCLBackend::enqueue_north_south() {
  const std::size_t n = halo_words_;
  const std::size_t two_n = 2 * halo_words_;
  const std::size_t n_offset = halo_n_offset();
  const std::size_t two_n_offset = halo_two_n_offset(halo_words_);
  auto* hins = static_cast<float*>(hins_->data());
  auto* hons = static_cast<float*>(hons_->data());

  if (n == 0) {
    return;
  }

  rccl_check(ncclGroupStart(), "ncclGroupStart north/south");
  rccl_check(ncclSend(hins + n_offset, n, ncclFloat, topology_.south,
                      rccl_comm_, stream_),
             "ncclSend north/south N");
  rccl_check(ncclRecv(hons + n_offset, n, ncclFloat, topology_.north,
                      rccl_comm_, stream_),
             "ncclRecv north/south N");
  rccl_check(ncclSend(hins + two_n_offset, two_n, ncclFloat, topology_.north,
                      rccl_comm_, stream_),
             "ncclSend north/south 2N");
  rccl_check(ncclRecv(hons + two_n_offset, two_n, ncclFloat, topology_.south,
                      rccl_comm_, stream_),
             "ncclRecv north/south 2N");
  rccl_check(ncclGroupEnd(), "ncclGroupEnd north/south");
}

void RCCLBackend::exchange_north_south() {
  enqueue_north_south();
  synchronize_stream("hipStreamSynchronize after north/south RCCL");
}

void RCCLBackend::enqueue_hons_to_hiew_copy() {
  hip_check(hipMemcpyAsync(hiew_->data(), hons_->data(), hons_->bytes(),
                           hipMemcpyDeviceToDevice, stream_),
            "hipMemcpyAsync hons -> hiew");
}

void RCCLBackend::copy_hons_to_hiew() {
  enqueue_hons_to_hiew_copy();
  synchronize_stream("hipStreamSynchronize after hons -> hiew");
}

void RCCLBackend::enqueue_east_west() {
  const std::size_t n = halo_words_;
  const std::size_t two_n = 2 * halo_words_;
  const std::size_t n_offset = halo_n_offset();
  const std::size_t two_n_offset = halo_two_n_offset(halo_words_);
  auto* hiew = static_cast<float*>(hiew_->data());
  auto* hoew = static_cast<float*>(hoew_->data());

  if (n == 0) {
    return;
  }

  rccl_check(ncclGroupStart(), "ncclGroupStart east/west");
  rccl_check(ncclSend(hiew + n_offset, n, ncclFloat, topology_.west,
                      rccl_comm_, stream_),
             "ncclSend east/west N");
  rccl_check(ncclRecv(hoew + n_offset, n, ncclFloat, topology_.east,
                      rccl_comm_, stream_),
             "ncclRecv east/west N");
  rccl_check(ncclSend(hiew + two_n_offset, two_n, ncclFloat, topology_.east,
                      rccl_comm_, stream_),
             "ncclSend east/west 2N");
  rccl_check(ncclRecv(hoew + two_n_offset, two_n, ncclFloat, topology_.west,
                      rccl_comm_, stream_),
             "ncclRecv east/west 2N");
  rccl_check(ncclGroupEnd(), "ncclGroupEnd east/west");
}

void RCCLBackend::exchange_east_west() {
  enqueue_east_west();
  synchronize_stream("hipStreamSynchronize after east/west RCCL");
}

void RCCLBackend::synchronize_stream(const char* operation) {
  hip_check(hipStreamSynchronize(stream_), operation);
}

double RCCLBackend::reduced_phase_average(double local_total_seconds,
                                          int iterations) const {
  const double local_average =
      local_total_seconds / static_cast<double>(iterations);
  double global_average = 0.0;
  mpi_check(MPI_Allreduce(&local_average, &global_average, 1, MPI_DOUBLE,
                          MPI_MAX, cart_comm_),
            "MPI_Allreduce RCCL phase timing");
  return global_average;
}

void RCCLBackend::validate_north_south() {
  fill_pattern();
  copy_hoew_to_hins();
  exchange_north_south();

  const std::size_t count = halo_buffer_words(halo_words_);
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
    validation_failed_ = true;
    throw std::runtime_error("RCCL north/south validation failed");
  }
  metadata_.validation_passed = true;
  ++validated_halo_count_;
  fill_rank();
}

void RCCLBackend::validate_full_exchange() {
  fill_pattern();
  exchange();

  const std::size_t count = halo_buffer_words(halo_words_);
  std::vector<float> hons(count);
  std::vector<float> hoew(count);
  hip_check(hipMemcpyAsync(hons.data(), hons_->data(), count * sizeof(float),
                           hipMemcpyDeviceToHost, stream_),
            "hipMemcpyAsync validation hons");
  hip_check(hipMemcpyAsync(hoew.data(), hoew_->data(), count * sizeof(float),
                           hipMemcpyDeviceToHost, stream_),
            "hipMemcpyAsync validation hoew");
  hip_check(hipStreamSynchronize(stream_),
            "hipStreamSynchronize after full validation copies");

  const int north_east =
      cart_rank(cart_comm_, (topology_.row + 1) % topology_.rows,
                (topology_.col + 1) % topology_.cols);
  const int south_west =
      cart_rank(cart_comm_,
                (topology_.row + topology_.rows - 1) % topology_.rows,
                (topology_.col + topology_.cols - 1) % topology_.cols);

  bool local_valid = true;
  std::string first_mismatch;
  auto check_value = [&](const char* direction, float actual, float expected,
                         std::size_t index, int neighbor) {
    if (local_valid && actual != expected) {
      const std::size_t sample_count = std::min<std::size_t>(8, hoew.size());
      std::vector<float> hoew_prefix(hoew.begin(),
                                     hoew.begin() + sample_count);
      std::vector<float> hons_prefix(hons.begin(),
                                     hons.begin() + sample_count);
      std::ostringstream message;
      message << "RCCL full halo validation failed on rank " << rank()
              << " mode " << sync_mode_name(sync_mode_)
              << " local_rank " << local_rank_ << " direction " << direction
              << " index " << index << ": expected " << expected
              << ", actual " << actual << ", neighbor " << neighbor
              << "; cart_dims=(" << topology_.rows << "," << topology_.cols
              << ") cart_coords=(" << topology_.row << "," << topology_.col
              << ") neighbors north=" << topology_.north
              << " south=" << topology_.south << " east=" << topology_.east
              << " west=" << topology_.west
              << " hons_prefix=" << values_string(hons_prefix)
              << " hoew_prefix=" << values_string(hoew_prefix);
      first_mismatch = message.str();
      local_valid = false;
    }
  };

  for (std::size_t i = 0; i < halo_words_; ++i) {
    check_value("north", hons[i],
                pattern_value(topology_.north, 1, static_cast<int>(halo_words_),
                              static_cast<int>(i)),
                i, topology_.north);
    check_value("east", hoew[i],
                pattern_value(north_east, 1, static_cast<int>(halo_words_),
                              static_cast<int>(i)),
                i, north_east);
  }
  for (std::size_t i = halo_words_; i < count; ++i) {
    check_value("south", hons[i],
                pattern_value(topology_.south, 2, static_cast<int>(halo_words_),
                              static_cast<int>(i)),
                i, topology_.south);
    check_value("west", hoew[i],
                pattern_value(south_west, 2, static_cast<int>(halo_words_),
                              static_cast<int>(i)),
                i, south_west);
  }

  if (!local_valid) {
    std::cerr << first_mismatch << '\n';
  }

  int local_ok = local_valid ? 1 : 0;
  int global_ok = 0;
  mpi_check(MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN,
                          cart_comm_),
            "MPI_Allreduce RCCL full validation");
  if (global_ok != 1) {
    validation_failed_ = true;
    if (rank() == 0) {
      std::cout << "RCCL full halo validation FAILED\n";
    }
    throw std::runtime_error("RCCL full halo validation failed");
  }
  metadata_.validation_passed = true;
  ++validated_halo_count_;
  fill_rank();
}

void RCCLBackend::print_validation_summary() const {
  if (validation_summary_printed_ || stage_b_only_ || !validate_ ||
      validation_failed_ || validated_halo_count_ == 0 || rank() != 0) {
    return;
  }
  std::cout << "RCCL full halo validation PASSED for "
            << validated_halo_count_ << " halo size"
            << (validated_halo_count_ == 1 ? "" : "s")
            << " in " << sync_mode_name(sync_mode_) << " mode\n";
  validation_summary_printed_ = true;
}

void RCCLBackend::print_startup() const {
  if (rank() != 0) {
    return;
  }
  std::cout << "backend: " << name() << "\n"
            << "rccl_stage: " << metadata_.rccl_stage << "\n"
            << "RCCL sync mode: " << metadata_.rccl_sync_mode << "\n"
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
      "RCCL Stage B debug mode implements north/south validation only; full "
      "halo exchange is available without --rccl-stage-b");
}

} // namespace ghalo
