#include "ghalo/benchmark.hpp"
#include "ghalo/output.hpp"
#include "ghalo/version.hpp"

#include "mpi_backend.hpp"

#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct CliOptions {
  std::string csv_path = "ghalo.csv";
  std::string json_path = "ghalo.json";
  double target_seconds = 3.0;
  bool show_help = false;
};

void print_usage(std::ostream& out) {
  out << "Usage: ghalo [--csv PATH] [--json PATH] [--target-seconds SECONDS]\n";
}

CliOptions parse_args(int argc, char** argv) {
  CliOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      options.show_help = true;
      return options;
    }
    if (arg == "--csv" && i + 1 < argc) {
      options.csv_path = argv[++i];
    } else if (arg == "--json" && i + 1 < argc) {
      options.json_path = argv[++i];
    } else if (arg == "--target-seconds" && i + 1 < argc) {
      options.target_seconds = std::stod(argv[++i]);
    } else {
      throw std::invalid_argument("unknown or incomplete argument: " + arg);
    }
  }
  return options;
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

    ghalo::MPIBackend backend;

    ghalo::BenchmarkConfig config;
    config.target_seconds = options.target_seconds;

    const auto results = ghalo::run_benchmark(backend, config);

    if (backend.is_root()) {
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
