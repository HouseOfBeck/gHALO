#pragma once

#include "ghalo/halo_range.hpp"

#include <cstddef>
#include <iosfwd>
#include <string>

namespace ghalo {

struct CliOptions {
  std::string backend = "mpi";
  std::string csv_path = "ghalo.csv";
  std::string json_path = "ghalo.json";
  std::string device_map = "local-rank";
  std::string rccl_sync_mode = "conservative";
  std::size_t min_halo = default_min_halo;
  std::size_t max_halo = default_max_halo;
  std::size_t halo_multiplier = default_halo_multiplier;
  double target_seconds = 3.0;
  bool show_help = false;
  bool validate = true;
  bool allow_gpu_oversubscription = false;
  bool phase_timing = false;
  bool rccl_stage_b = false;
};

void print_usage(std::ostream& out);
CliOptions parse_cli_options(int argc, char** argv);

} // namespace ghalo
