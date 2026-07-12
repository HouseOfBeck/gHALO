#include "ghalo/cli.hpp"

#include <ostream>
#include <stdexcept>

namespace ghalo {

void print_usage(std::ostream& out) {
  out << "Usage: ghalo [--backend mpi|mpi-hip|rccl] [--csv PATH] [--json PATH]\n"
      << "             [--target-seconds SECONDS] [--device-map local-rank]\n"
      << "             [--validate|--no-validate] [--allow-gpu-oversubscription]\n"
      << "             [--phase-timing] [--rccl-stage-b]\n"
      << "             [--rccl-sync-mode conservative|stream-ordered]\n";
}

CliOptions parse_cli_options(int argc, char** argv) {
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
    } else if (arg == "--rccl-sync-mode" && i + 1 < argc) {
      options.rccl_sync_mode = argv[++i];
      if (options.rccl_sync_mode != "conservative" &&
          options.rccl_sync_mode != "stream-ordered") {
        throw std::invalid_argument(
            "invalid --rccl-sync-mode value: " + options.rccl_sync_mode +
            " (expected conservative or stream-ordered)");
      }
    } else if (arg == "--validate") {
      options.validate = true;
    } else if (arg == "--no-validate") {
      options.validate = false;
    } else if (arg == "--allow-gpu-oversubscription") {
      options.allow_gpu_oversubscription = true;
    } else if (arg == "--phase-timing") {
      options.phase_timing = true;
    } else if (arg == "--rccl-stage-b") {
      options.rccl_stage_b = true;
    } else {
      throw std::invalid_argument("unknown or incomplete argument: " + arg);
    }
  }
  return options;
}

} // namespace ghalo
