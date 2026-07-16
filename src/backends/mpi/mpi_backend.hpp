#pragma once

#include "ghalo/backend.hpp"

#include <mpi.h>

#include <cstddef>
#include <string>
#include <vector>

namespace ghalo {

class MPIEnvironment final {
public:
  MPIEnvironment(int& argc, char**& argv);
  ~MPIEnvironment();

  MPIEnvironment(const MPIEnvironment&) = delete;
  MPIEnvironment& operator=(const MPIEnvironment&) = delete;

  int rank() const;
  void abort(int error_code) const;
};

class MPIBackend final : public Backend {
public:
  MPIBackend();
  ~MPIBackend() override;

  MPIBackend(const MPIBackend&) = delete;
  MPIBackend& operator=(const MPIBackend&) = delete;

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
  std::vector<IterationTimingReduction> max_time_ranks(
      const std::vector<double>& local_seconds) override;

private:
  void initialize_topology();
  void initialize_metadata();

  MPI_Comm cart_comm_ = MPI_COMM_NULL;
  TopologyInfo topology_{};
  BackendMetadata metadata_{};
  std::size_t halo_words_ = 0;
  std::vector<float> hins_;
  std::vector<float> hons_;
  std::vector<float> hiew_;
  std::vector<float> hoew_;
};

} // namespace ghalo
