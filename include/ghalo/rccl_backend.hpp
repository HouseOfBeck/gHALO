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
#include <string>

namespace ghalo {

class RCCLBackend final : public Backend {
public:
  RCCLBackend();
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

  bool exchange_implemented() const;

private:
  void initialize_topology();
  void initialize_local_rank();
  void select_device();
  void initialize_rccl_communicator();
  [[noreturn]] void fail_not_implemented() const;

  MPI_Comm cart_comm_ = MPI_COMM_NULL;
  MPI_Comm local_comm_ = MPI_COMM_NULL;
  ncclComm_t rccl_comm_ = nullptr;
  hipStream_t stream_ = nullptr;
  TopologyInfo topology_{};
  BackendMetadata metadata_{};
  std::size_t halo_words_ = 0;
  int local_rank_ = 0;
  int local_size_ = 1;
  int selected_device_ = -1;
};

} // namespace ghalo
