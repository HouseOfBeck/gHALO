#pragma once

#include "ghalo/backend.hpp"

#include <hip/hip_runtime.h>
#include <mpi.h>

#if __has_include(<rccl/rccl.h>)
#include <rccl/rccl.h>
#elif __has_include(<rccl.h>)
#include <rccl.h>
#else
#include <nccl.h>
#endif

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace ghalo {

class RCCLBackend final : public Backend {
public:
  enum class SyncMode { Conservative, StreamOrdered };

  RCCLBackend(bool validate, bool allow_oversubscription, bool stage_b_only,
              const std::string& sync_mode);
  ~RCCLBackend() override;

  RCCLBackend(const RCCLBackend&) = delete;
  RCCLBackend& operator=(const RCCLBackend&) = delete;

  std::string name() const override;
  std::string algorithm() const override;
  TopologyInfo topology() const override;
  BackendMetadata metadata() const override;

  int rank() const override;
  int size() const override;
  bool is_root() const override;

  void setup(std::size_t halo_words) override;
  void exchange() override;
  void barrier() override;
  double now() const override;
  double max_time(double local_seconds) override;
  void run_development_validation(
      const std::vector<std::size_t>& halo_lengths) override;
  bool supports_phase_timing() const override;
  void set_phase_timing_enabled(bool enabled) override;
  bool phase_timing_enabled() const override;
  void reset_phase_timing() override;
  PhaseTimingResult phase_timing_result(int iterations,
                                        double total_seconds) override;

  bool exchange_implemented() const;

private:
  class DeviceBuffer;
  struct PhaseTimingAccumulator {
    double north_south_communication_seconds{};
    double north_south_sync_seconds{};
    double transpose_copy_seconds{};
    double transpose_sync_seconds{};
    double east_west_communication_seconds{};
    double east_west_sync_seconds{};
    double north_south_communication_enqueue_seconds{};
    double transpose_copy_enqueue_seconds{};
    double east_west_communication_enqueue_seconds{};
    double final_stream_sync_seconds{};
  };

  void initialize_topology();
  void initialize_local_rank();
  void select_device();
  void initialize_rccl_communicator();
  void initialize_metadata();
  void allocate_buffers(std::size_t halo_words);
  void fill_pattern();
  void fill_rank();
  void enqueue_hoew_to_hins_copy();
  void copy_hoew_to_hins();
  void enqueue_north_south();
  void exchange_north_south();
  void enqueue_hons_to_hiew_copy();
  void copy_hons_to_hiew();
  void enqueue_east_west();
  void exchange_east_west();
  void exchange_conservative();
  void exchange_stream_ordered();
  void exchange_stream_ordered_with_phase_timing();
  void synchronize_stream(const char* operation);
  double reduced_phase_average(double local_total_seconds,
                               int iterations) const;
  void validate_north_south();
  void validate_full_exchange();
  void print_validation_summary() const;
  void print_startup() const;
  void hip_check(hipError_t error, const char* operation) const;
  void rccl_check(ncclResult_t error, const char* operation) const;
  [[noreturn]] void fail_not_implemented() const;

  MPI_Comm cart_comm_ = MPI_COMM_NULL;
  MPI_Comm local_comm_ = MPI_COMM_NULL;
  ncclComm_t rccl_comm_ = nullptr;
  hipStream_t stream_ = nullptr;
  TopologyInfo topology_{};
  BackendMetadata metadata_{};
  std::size_t halo_words_ = 0;
  bool validate_ = true;
  bool allow_oversubscription_ = false;
  bool stage_b_only_ = false;
  SyncMode sync_mode_ = SyncMode::Conservative;
  bool phase_timing_enabled_ = false;
  bool phase_timing_collecting_ = false;
  bool validation_failed_ = false;
  mutable bool validation_summary_printed_ = false;
  int validated_halo_count_ = 0;
  int visible_device_count_ = 0;
  int local_rank_ = 0;
  int local_size_ = 1;
  int selected_device_ = -1;
  std::string hostname_;
  std::string device_name_;
  std::string rocr_visible_devices_;
  std::unique_ptr<DeviceBuffer> hins_;
  std::unique_ptr<DeviceBuffer> hons_;
  std::unique_ptr<DeviceBuffer> hiew_;
  std::unique_ptr<DeviceBuffer> hoew_;
  PhaseTimingAccumulator phase_timing_;
};

} // namespace ghalo
