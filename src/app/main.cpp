#include "ghalo/benchmark.hpp"
#include "ghalo/output.hpp"
#include "ghalo/version.hpp"

#include "mpi_backend.hpp"
#ifdef GHALO_HAVE_MPI_HIP
#include "mpi_hip_backend.hpp"
#endif

#include <exception>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

struct CliOptions {
  std::string backend = "mpi";
  std::string csv_path = "ghalo.csv";
  std::string json_path = "ghalo.json";
  std::string device_map = "local-rank";
  double target_seconds = 3.0;
  bool show_help = false;
  bool validate = true;
  bool allow_gpu_oversubscription = false;
};

void print_usage(std::ostream& out) {
  out << "Usage: ghalo [--backend mpi|mpi-hip] [--csv PATH] [--json PATH]\n"
      << "             [--target-seconds SECONDS] [--device-map local-rank]\n"
      << "             [--validate|--no-validate] [--allow-gpu-oversubscription]\n";
}

CliOptions parse_args(int argc, char** argv) {
  CliOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      options.show_help = true;
      return options;
    }
    if (arg == "--backend" && i + 1 < argc) {
      options.backend = argv[++i];
    } else if (arg == "--csv" && i + 1 < argc) {
      options.csv_path = argv[++i];
    } else if (arg == "--json" && i + 1 < argc) {
      options.json_path = argv[++i];
    } else if (arg == "--target-seconds" && i + 1 < argc) {
      options.target_seconds = std::stod(argv[++i]);
    } else if (arg == "--device-map" && i + 1 < argc) {
      options.device_map = argv[++i];
    } else if (arg == "--validate") {
      options.validate = true;
    } else if (arg == "--no-validate") {
      options.validate = false;
    } else if (arg == "--allow-gpu-oversubscription") {
      options.allow_gpu_oversubscription = true;
    } else {
      throw std::invalid_argument("unknown or incomplete argument: " + arg);
    }
  }
  return options;
}

std::unique_ptr<ghalo::Backend> make_backend(const CliOptions& options) {
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
  throw std::invalid_argument("unknown backend: " + options.backend);
}

} // namespace

int main(int argc, char** argv) {
  ghalo::MPIEnvironment mpi(argc, argv);

  try {
    auto options = parse_args(argc, argv);
    if (options.show_help) {
      if (mpi.rank() == 0) {
        print_usage(std::cout);
      }
      return 0;
    }

    auto backend = make_backend(options);

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
