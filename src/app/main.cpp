#include "ghalo/benchmark.hpp"
#include "ghalo/cli.hpp"
#include "ghalo/output.hpp"

#include "mpi_backend.hpp"
#ifdef GHALO_HAVE_MPI_HIP
#include "mpi_hip_backend.hpp"
#endif
#ifdef GHALO_HAVE_RCCL
#include "ghalo/rccl_backend.hpp"
#endif

#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {

std::unique_ptr<ghalo::Backend> make_backend(const ghalo::CliOptions& options) {
  if (options.backend == "mpi") {
    return std::make_unique<ghalo::MPIBackend>();
  }
  if (options.backend == "mpi-hip") {
#ifdef GHALO_HAVE_MPI_HIP
    return std::make_unique<ghalo::MPIHIPBackend>(
        options.device_map, options.validate, options.allow_gpu_oversubscription);
#else
    throw std::runtime_error(
        "backend mpi-hip requested, but gHALO was not built with "
        "GHALO_ENABLE_MPI_HIP=ON");
#endif
  }
  if (options.backend == "rccl") {
#ifdef GHALO_HAVE_RCCL
    return std::make_unique<ghalo::RCCLBackend>();
#else
    throw std::runtime_error(
        "backend rccl requested, but gHALO was not built with "
        "GHALO_ENABLE_RCCL=ON");
#endif
  }
  throw std::invalid_argument("unknown backend: " + options.backend);
}

} // namespace

int main(int argc, char** argv) {
  ghalo::MPIEnvironment mpi(argc, argv);

  try {
    auto options = ghalo::parse_cli_options(argc, argv);
    if (options.show_help) {
      if (mpi.rank() == 0) {
        ghalo::print_usage(std::cout);
      }
      return 0;
    }

    auto backend = make_backend(options);
    backend->set_phase_timing_enabled(options.phase_timing);

    ghalo::BenchmarkConfig config;
    config.target_seconds = options.target_seconds;

    const auto results = ghalo::run_benchmark(*backend, config);

    if (backend->is_root()) {
      ghalo::write_console(std::cout, results);
      ghalo::write_csv(options.csv_path, results);
      ghalo::write_json(options.json_path, results);
    }
  } catch (const std::exception& error) {
    std::cerr << "gHALO error on rank " << mpi.rank() << ": " << error.what()
              << '\n';
    mpi.abort(1);
  } catch (...) {
    std::cerr << "gHALO error on rank " << mpi.rank()
              << ": unknown exception\n";
    mpi.abort(1);
  }

  return 0;
}
