#include "ghalo/rccl_backend.hpp"

#include <stdexcept>

namespace ghalo {

RCCLBackend::RCCLBackend() {
  MPI_Comm_rank(MPI_COMM_WORLD, &topology_.world_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &topology_.world_size);
  metadata_.memory_location = "device";
  metadata_.gpu_aware_mpi_requested = false;
  metadata_.device_map = "local-rank";
}

RCCLBackend::~RCCLBackend() {
  if (rccl_comm_ != nullptr) {
    (void)ncclCommDestroy(rccl_comm_);
  }
  if (stream_ != nullptr) {
    (void)hipStreamDestroy(stream_);
  }
  if (local_comm_ != MPI_COMM_NULL) {
    (void)MPI_Comm_free(&local_comm_);
  }
  if (cart_comm_ != MPI_COMM_NULL) {
    (void)MPI_Comm_free(&cart_comm_);
  }
}

std::string RCCLBackend::name() const { return "RCCLBackend"; }

std::string RCCLBackend::algorithm() const {
  return "rccl-experimental-not-implemented";
}

TopologyInfo RCCLBackend::topology() const { return topology_; }

BackendMetadata RCCLBackend::metadata() const { return metadata_; }

int RCCLBackend::rank() const { return topology_.world_rank; }

int RCCLBackend::size() const { return topology_.world_size; }

bool RCCLBackend::is_root() const { return topology_.world_rank == 0; }

void RCCLBackend::setup(std::size_t halo_words) {
  halo_words_ = halo_words;
  fail_not_implemented();
}

void RCCLBackend::exchange() { fail_not_implemented(); }

void RCCLBackend::barrier() { MPI_Barrier(MPI_COMM_WORLD); }

double RCCLBackend::now() const { return MPI_Wtime(); }

double RCCLBackend::max_time(double local_seconds) {
  double global_seconds = 0.0;
  MPI_Allreduce(&local_seconds, &global_seconds, 1, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);
  return global_seconds;
}

bool RCCLBackend::exchange_implemented() const { return false; }

void RCCLBackend::initialize_topology() {}

void RCCLBackend::initialize_local_rank() {}

void RCCLBackend::select_device() {}

void RCCLBackend::initialize_rccl_communicator() {}

void RCCLBackend::fail_not_implemented() const {
  throw std::runtime_error("RCCL backend exchange is not implemented yet");
}

} // namespace ghalo
