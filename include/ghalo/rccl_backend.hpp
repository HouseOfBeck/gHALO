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
  RCCLBackend(bool validate, bool allow_oversubscription);
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

  bool exchange_implemented() const;

private:
  class DeviceBuffer;

  void initialize_topology();
  void initialize_local_rank();
  void select_device();
  void initialize_rccl_communicator();
  void initialize_metadata();
  void allocate_buffers(std::size_t halo_words);
  void fill_pattern();
  void copy_hoew_to_hins();
  void exchange_north_south();
  void validate_north_south();
  void print_stage_b_startup() const;
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
  int visible_device_count_ = 0;
  int local_rank_ = 0;
  int local_size_ = 1;
  int selected_device_ = -1;
  std::string hostname_;
  std::string device_name_;
  std::string rocr_visible_devices_;
  std::unique_ptr<DeviceBuffer> hins_;
  std::unique_ptr<DeviceBuffer> hons_;
  std::unique_ptr<DeviceBuffer> hoew_;
};

} // namespace ghalo
