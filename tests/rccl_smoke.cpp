#include <hip/hip_runtime.h>
#include <mpi.h>

#if __has_include(<rccl/rccl.h>)
#include <rccl/rccl.h>
#elif __has_include(<rccl.h>)
#include <rccl.h>
#else
#include <nccl.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

struct Options {
  std::size_t count = 1024;
  int iterations = 1;
  bool validate = true;
  bool verbose = false;
  bool help = false;
};

struct RankContext {
  int world_rank = 0;
  int world_size = 1;
  int local_rank = 0;
  int local_size = 1;
  int selected_device = 0;
  std::string hostname;
  std::string device_name;
};

[[noreturn]] void usage_error(const std::string& message) {
  throw std::invalid_argument(message);
}

void print_usage(std::ostream& out) {
  out << "Usage: ghalo_rccl_smoke [--count N] [--iterations N]\n"
      << "                        [--validate] [--verbose] [--help]\n";
}

std::size_t parse_size(const std::string& value, const char* option) {
  if (value.empty()) {
    usage_error(std::string(option) + " must be a positive integer");
  }
  for (const char character : value) {
    if (character < '0' || character > '9') {
      usage_error(std::string(option) + " must be a positive integer");
    }
  }
  std::size_t consumed = 0;
  const auto parsed = std::stoull(value, &consumed);
  if (consumed != value.size() || parsed == 0) {
    usage_error(std::string(option) + " must be a positive integer");
  }
  return static_cast<std::size_t>(parsed);
}

int parse_int(const std::string& value, const char* option) {
  const auto parsed = parse_size(value, option);
  if (parsed > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    usage_error(std::string(option) + " is too large");
  }
  return static_cast<int>(parsed);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      options.help = true;
      return options;
    }
    if (arg == "--count") {
      if (i + 1 >= argc) {
        usage_error("--count requires a value");
      }
      options.count = parse_size(argv[++i], "--count");
    } else if (arg == "--iterations") {
      if (i + 1 >= argc) {
        usage_error("--iterations requires a value");
      }
      options.iterations = parse_int(argv[++i], "--iterations");
    } else if (arg == "--validate") {
      options.validate = true;
    } else if (arg == "--verbose") {
      options.verbose = true;
    } else {
      usage_error("unknown argument: " + arg);
    }
  }
  return options;
}

std::string hostname() {
  char name[256] = {};
  if (gethostname(name, sizeof(name) - 1) != 0) {
    return "unknown";
  }
  return name;
}

std::string mpi_error_string(int error) {
  char text[MPI_MAX_ERROR_STRING] = {};
  int length = 0;
  MPI_Error_string(error, text, &length);
  return std::string(text, length);
}

std::string rank_prefix(const RankContext& context) {
  std::ostringstream out;
  out << "rank=" << context.world_rank << " local_rank=" << context.local_rank
      << " hip_device=" << context.selected_device;
  return out.str();
}

void mpi_check(int error, const char* operation) {
  if (error != MPI_SUCCESS) {
    std::ostringstream message;
    message << "MPI call failed: " << operation << ": "
            << mpi_error_string(error);
    throw std::runtime_error(message.str());
  }
}

void hip_check(hipError_t error, const char* operation,
               const RankContext& context) {
  if (error != hipSuccess) {
    std::ostringstream message;
    message << "HIP call failed: " << operation << ": "
            << hipGetErrorName(error) << ": " << hipGetErrorString(error)
            << " (" << rank_prefix(context) << ")";
    throw std::runtime_error(message.str());
  }
}

void rccl_check(ncclResult_t error, const char* operation,
                const RankContext& context) {
  if (error != ncclSuccess) {
    std::ostringstream message;
    message << "RCCL call failed: " << operation << ": "
            << ncclGetErrorString(error) << " (" << rank_prefix(context)
            << ")";
    throw std::runtime_error(message.str());
  }
}

std::string hip_runtime_version(const RankContext& context) {
  int version = 0;
  hip_check(hipRuntimeGetVersion(&version), "hipRuntimeGetVersion", context);
  return std::to_string(version);
}

std::string rccl_runtime_version(const RankContext& context) {
  int version = 0;
  rccl_check(ncclGetVersion(&version), "ncclGetVersion", context);
  return std::to_string(version);
}

std::int32_t pattern_value(int rank, std::size_t index) {
  static_assert(sizeof(std::int32_t) == sizeof(int));
  return static_cast<std::int32_t>(rank * 1000000 +
                                   static_cast<int>(index));
}

void validate_pattern_range(const Options& options,
                            const RankContext& context) {
  const auto max_value =
      static_cast<long long>(context.world_size - 1) * 1000000LL +
      static_cast<long long>(options.count - 1);
  if (max_value > std::numeric_limits<std::int32_t>::max()) {
    std::ostringstream message;
    message << "validation pattern would exceed int32_t range for world_size="
            << context.world_size << " and count=" << options.count;
    throw std::runtime_error(message.str());
  }
}

class LocalCommunicator final {
public:
  LocalCommunicator() {
    mpi_check(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0,
                                  MPI_INFO_NULL, &comm_),
              "MPI_Comm_split_type");
  }

  ~LocalCommunicator() {
    if (comm_ != MPI_COMM_NULL) {
      (void)MPI_Comm_free(&comm_);
    }
  }

  MPI_Comm get() const { return comm_; }

private:
  MPI_Comm comm_ = MPI_COMM_NULL;
};

class DeviceBuffer final {
public:
  DeviceBuffer() = default;

  explicit DeviceBuffer(std::size_t bytes, const RankContext& context) {
    allocate(bytes, context);
  }

  ~DeviceBuffer() { reset(); }

  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  void allocate(std::size_t bytes, const RankContext& context) {
    reset();
    hip_check(hipMalloc(&ptr_, bytes), "hipMalloc", context);
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

class HipStream final {
public:
  explicit HipStream(const RankContext& context) {
    hip_check(hipStreamCreate(&stream_), "hipStreamCreate", context);
  }

  ~HipStream() {
    if (stream_ != nullptr) {
      (void)hipStreamDestroy(stream_);
    }
  }

  hipStream_t get() const { return stream_; }

private:
  hipStream_t stream_ = nullptr;
};

class RcclCommunicator final {
public:
  RcclCommunicator(const ncclUniqueId& id, const RankContext& context) {
    rccl_check(ncclCommInitRank(&comm_, context.world_size, id,
                                context.world_rank),
               "ncclCommInitRank", context);
  }

  ~RcclCommunicator() {
    if (comm_ != nullptr) {
      (void)ncclCommDestroy(comm_);
    }
  }

  ncclComm_t get() const { return comm_; }

private:
  ncclComm_t comm_ = nullptr;
};

RankContext initialize_rank_context() {
  RankContext context;
  mpi_check(MPI_Comm_rank(MPI_COMM_WORLD, &context.world_rank),
            "MPI_Comm_rank");
  mpi_check(MPI_Comm_size(MPI_COMM_WORLD, &context.world_size),
            "MPI_Comm_size");
  context.hostname = hostname();

  LocalCommunicator local_comm;
  mpi_check(MPI_Comm_rank(local_comm.get(), &context.local_rank),
            "MPI_Comm_rank local");
  mpi_check(MPI_Comm_size(local_comm.get(), &context.local_size),
            "MPI_Comm_size local");

  int device_count = 0;
  hip_check(hipGetDeviceCount(&device_count), "hipGetDeviceCount", context);
  if (device_count <= 0) {
    throw std::runtime_error("no HIP devices are visible to this rank");
  }
  if (context.local_size > device_count) {
    std::ostringstream message;
    message << "invalid rank-to-device mapping on host " << context.hostname
            << ": local ranks=" << context.local_size
            << ", visible HIP devices=" << device_count;
    throw std::runtime_error(message.str());
  }

  context.selected_device = context.local_rank % device_count;
  hip_check(hipSetDevice(context.selected_device), "hipSetDevice", context);

  hipDeviceProp_t properties{};
  hip_check(hipGetDeviceProperties(&properties, context.selected_device),
            "hipGetDeviceProperties", context);
  context.device_name = properties.name;
  return context;
}

void print_rank_details(const RankContext& context, bool verbose,
                        const std::string& hip_version,
                        const std::string& rccl_version) {
  if (context.world_rank == 0) {
    std::cout << "gHALO RCCL smoke test\n"
              << "world_size=" << context.world_size
              << " hip_runtime_version=" << hip_version
              << " rccl_version=" << rccl_version << '\n';
  }

  if (!verbose) {
    return;
  }

  for (int rank = 0; rank < context.world_size; ++rank) {
    mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier verbose");
    if (rank == context.world_rank) {
      std::cout << "rank_detail"
                << " hostname=" << context.hostname
                << " global_rank=" << context.world_rank
                << " local_rank=" << context.local_rank
                << " selected_hip_device=" << context.selected_device
                << " gpu_name=\"" << context.device_name << "\""
                << " world_size=" << context.world_size
                << " hip_runtime_version=" << hip_version
                << " rccl_version=" << rccl_version << '\n';
      std::cout.flush();
    }
  }
  mpi_check(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier verbose final");
}

void initialize_buffers(DeviceBuffer& send_buffer, DeviceBuffer& recv_buffer,
                        std::size_t count, const RankContext& context) {
  std::vector<std::int32_t> host_send(count);
  std::vector<std::int32_t> host_recv(count, -1);
  for (std::size_t i = 0; i < count; ++i) {
    host_send[i] = pattern_value(context.world_rank, i);
  }

  hip_check(hipMemcpy(send_buffer.data(), host_send.data(),
                      count * sizeof(std::int32_t), hipMemcpyHostToDevice),
            "hipMemcpy send initialization", context);
  hip_check(hipMemcpy(recv_buffer.data(), host_recv.data(),
                      count * sizeof(std::int32_t), hipMemcpyHostToDevice),
            "hipMemcpy receive sentinel initialization", context);
}

bool validate_receive(const DeviceBuffer& recv_buffer, std::size_t count,
                      int recv_from, const RankContext& context) {
  std::vector<std::int32_t> host_recv(count);
  hip_check(hipMemcpy(host_recv.data(), recv_buffer.data(),
                      count * sizeof(std::int32_t), hipMemcpyDeviceToHost),
            "hipMemcpy receive validation", context);

  for (std::size_t i = 0; i < count; ++i) {
    const auto expected = pattern_value(recv_from, i);
    if (host_recv[i] != expected) {
      std::cerr << "RCCL smoke validation mismatch:"
                << " rank=" << context.world_rank
                << " source_rank=" << recv_from
                << " destination_rank=" << context.world_rank
                << " index=" << i << " expected=" << expected
                << " actual=" << host_recv[i] << '\n';
      return false;
    }
  }
  return true;
}

bool run_smoke_exchange(const Options& options, const RankContext& context,
                        ncclComm_t comm, hipStream_t stream,
                        DeviceBuffer& send_buffer,
                        DeviceBuffer& recv_buffer) {
  const int send_to = (context.world_rank + 1) % context.world_size;
  const int recv_from =
      (context.world_rank - 1 + context.world_size) % context.world_size;
  bool local_valid = true;

  for (int iteration = 0; iteration < options.iterations; ++iteration) {
    initialize_buffers(send_buffer, recv_buffer, options.count, context);

    rccl_check(ncclGroupStart(), "ncclGroupStart", context);
    rccl_check(ncclSend(send_buffer.data(), options.count, ncclInt, send_to,
                        comm, stream),
               "ncclSend", context);
    rccl_check(ncclRecv(recv_buffer.data(), options.count, ncclInt,
                        recv_from, comm, stream),
               "ncclRecv", context);
    rccl_check(ncclGroupEnd(), "ncclGroupEnd", context);
    hip_check(hipStreamSynchronize(stream), "hipStreamSynchronize", context);

    if (options.validate) {
      local_valid =
          validate_receive(recv_buffer, options.count, recv_from, context) &&
          local_valid;
    }
  }

  return local_valid;
}

int run(int argc, char** argv) {
  const auto options = parse_options(argc, argv);
  if (options.help) {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      print_usage(std::cout);
    }
    return 0;
  }

  auto context = initialize_rank_context();
  validate_pattern_range(options, context);
  const auto hip_version = hip_runtime_version(context);
  const auto rccl_version = rccl_runtime_version(context);
  print_rank_details(context, options.verbose, hip_version, rccl_version);

  ncclUniqueId id{};
  if (context.world_rank == 0) {
    rccl_check(ncclGetUniqueId(&id), "ncclGetUniqueId", context);
  }
  mpi_check(MPI_Bcast(&id, static_cast<int>(sizeof(id)), MPI_BYTE, 0,
                      MPI_COMM_WORLD),
            "MPI_Bcast ncclUniqueId");

  HipStream stream(context);
  RcclCommunicator comm(id, context);
  DeviceBuffer send_buffer(options.count * sizeof(std::int32_t), context);
  DeviceBuffer recv_buffer(options.count * sizeof(std::int32_t), context);

  const bool local_valid =
      run_smoke_exchange(options, context, comm.get(), stream.get(),
                         send_buffer, recv_buffer);
  const int local_failed = local_valid ? 0 : 1;
  int global_failed = 0;
  mpi_check(MPI_Allreduce(&local_failed, &global_failed, 1, MPI_INT, MPI_MAX,
                          MPI_COMM_WORLD),
            "MPI_Allreduce validation status");

  if (context.world_rank == 0) {
    if (global_failed == 0) {
      std::cout << "RCCL smoke validation PASSED\n";
    } else {
      std::cout << "RCCL smoke validation FAILED\n";
    }
  }
  return global_failed == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
  int initialized = 0;
  try {
    mpi_check(MPI_Init(&argc, &argv), "MPI_Init");
    initialized = 1;
    const int status = run(argc, argv);
    mpi_check(MPI_Finalize(), "MPI_Finalize");
    initialized = 0;
    return status;
  } catch (const std::exception& error) {
    int rank = 0;
    if (initialized != 0) {
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    std::cerr << "gHALO RCCL smoke error on rank " << rank << ": "
              << error.what() << '\n';
    if (initialized != 0) {
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return 1;
  }
}
