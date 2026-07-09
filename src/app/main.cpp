#include "ghalo/benchmark.hpp"
#include "ghalo/output.hpp"
#include "ghalo/version.hpp"

#include "mpi_backend.hpp"

#include <exception>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct CliOptions {
  std::string csv_path = "ghalo.csv";
  std::string json_path = "ghalo.json";
  double target_seconds = 3.0;
};

void print_usage(std::ostream& out) {
  out << "Usage: ghalo [--csv PATH] [--json PATH] [--target-seconds SECONDS]\n";
}

CliOptions parse_args(int argc, char** argv) {
  CliOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_usage(std::cout);
      std::exit(0);
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
  try {
    auto options = parse_args(argc, argv);
    ghalo::MPIBackend backend(argc, argv);

    ghalo::BenchmarkConfig config;
    config.target_seconds = options.target_seconds;

    const auto results = ghalo::run_benchmark(backend, config);

    if (backend.is_root()) {
      ghalo::write_console(std::cout, results);
      ghalo::write_csv(options.csv_path, results);
      ghalo::write_json(options.json_path, results);
    }
  } catch (const std::exception& error) {
    std::cerr << "gHALO error: " << error.what() << '\n';
    return 1;
  }

  return 0;
}
