#pragma once

#include <iosfwd>
#include <string>

namespace ghalo {

struct CliOptions {
  std::string backend = "mpi";
  std::string csv_path = "ghalo.csv";
  std::string json_path = "ghalo.json";
  std::string device_map = "local-rank";
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
