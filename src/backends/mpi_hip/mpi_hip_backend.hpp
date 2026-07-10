#pragma once

#include "ghalo/backend.hpp"

#include <mpi.h>

#include <cstddef>
#include <string>

namespace ghalo {

class HIPBuffer final {
public:
  HIPBuffer() = default;
  ~HIPBuffer();

  HIPBuffer(const HIPBuffer&) = delete;
  HIPBuffer& operator=(const HIPBuffer&) = delete;

  HIPBuffer(HIPBuffer&& other) noexcept;
  HIPBuffer& operator=(HIPBuffer&& other) noexcept;

  void allocate(std::size_t bytes);
  void reset();
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
  void exchange() override;
  void barrier() override;
  double now() const override;
  double max_time(double local_seconds) override;

private:
  void initialize_topology();
  void initialize_device();
  void initialize_metadata();
  void probe_device_mpi();
  void validate_exchange();

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
  HIPBuffer hins_;
  HIPBuffer hons_;
  HIPBuffer hiew_;
  HIPBuffer hoew_;
};

} // namespace ghalo
