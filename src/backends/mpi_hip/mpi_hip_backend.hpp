#pragma once

#include "ghalo/backend.hpp"

#include <mpi.h>

#include <cstddef>
#include <string>

namespace ghalo {

class HIPBuffer final {
public:
  HIPBuffer() = default;
  ~HIPBuffer() noexcept;

  HIPBuffer(const HIPBuffer&) = delete;
  HIPBuffer& operator=(const HIPBuffer&) = delete;

  HIPBuffer(HIPBuffer&& other) noexcept;
  HIPBuffer& operator=(HIPBuffer&& other) noexcept;

  void allocate(std::size_t bytes);
  void reset() noexcept;
  void* data() const;
  std::size_t bytes() const;

private:
  void* ptr_ = nullptr;
  std::size_t bytes_ = 0;
};

class MPIHIPBackend final : public Backend {
public:
  MPIHIPBackend(std::string device_map, bool validate,
                bool allow_oversubscription);
  ~MPIHIPBackend() override;

  MPIHIPBackend(const MPIHIPBackend&) = delete;
  MPIHIPBackend& operator=(const MPIHIPBackend&) = delete;

  std::string name() const override;
  std::string algorithm() const override;
  TopologyInfo topology() const override;
  BackendMetadata metadata() const override;

  int rank() const override;
  int size() const override;
  bool is_root() const override;

  void setup(std::size_t halo_words) override;
  void validate_current_halo() override;
  void exchange() override;
  void barrier() override;
  double now() const override;
  double max_time(double local_seconds) override;
  std::vector<IterationTimingReduction> max_time_ranks(
      const std::vector<double>& local_seconds) override;
  bool supports_phase_timing() const override;
  void set_phase_timing_enabled(bool enabled) override;
  bool phase_timing_enabled() const override;
  void reset_phase_timing() override;
  PhaseTimingResult phase_timing_result(int iterations,
                                        double total_seconds) override;

private:
  struct PhaseTimingAccumulator {
    double input_device_copy_seconds{};
    double north_south_mpi_seconds{};
    double north_south_sync_seconds{};
    double transpose_device_copy_seconds{};
    double transpose_copy_sync_seconds{};
    double east_west_mpi_seconds{};
    double east_west_sync_seconds{};
  };

  void initialize_topology();
  void initialize_device();
  void initialize_metadata();
  void probe_device_mpi();
  void validate_exchange();
  void perform_exchange(bool prepare_hins_from_hoew);
  void copy_hoew_to_hins();
  void exchange_north_south();
  void copy_hons_to_hiew();
  void synchronize_after_north_south_mpi();
  void copy_hons_to_hiew_data();
  void synchronize_after_transpose_copy();
  void exchange_east_west();
  double reduced_phase_average(double local_total_seconds,
                               int iterations) const;

  MPI_Comm cart_comm_ = MPI_COMM_NULL;
  MPI_Comm local_comm_ = MPI_COMM_NULL;
  TopologyInfo topology_{};
  BackendMetadata metadata_{};
  std::string device_map_;
  bool validate_ = true;
  bool allow_oversubscription_ = false;
  int local_rank_ = 0;
  int local_size_ = 1;
  int visible_device_count_ = 0;
  int selected_device_ = 0;
  std::string selected_visible_token_;
  std::string hostname_;
  std::string device_name_;
  std::string rocr_visible_devices_;
  std::size_t halo_words_ = 0;
  bool phase_timing_enabled_ = false;
  bool phase_timing_collecting_ = false;
  PhaseTimingAccumulator phase_timing_;
  HIPBuffer hins_;
  HIPBuffer hons_;
  HIPBuffer hiew_;
  HIPBuffer hoew_;
};

} // namespace ghalo
