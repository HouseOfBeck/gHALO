#pragma once

#include "ghalo/backend.hpp"
#include "ghalo/halo_range.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace ghalo {

struct BenchmarkConfig {
  std::size_t min_halo = default_min_halo;
  std::size_t max_halo = default_max_halo;
  std::size_t halo_multiplier = default_halo_multiplier;
  std::size_t samples_per_halo = 1;
  std::vector<std::size_t> halo_lengths{};
  double target_seconds = 3.0;
  bool record_iteration_times = false;
  bool record_iteration_phase_times = false;
  bool record_stalled_rank_times = false;
  double iteration_stall_threshold_us = 0.0;
  int calibration_iterations = 5;
};

struct IterationTimingRecord {
  std::size_t halo_words{};
  std::size_t sample_index{1};
  std::size_t iteration_index{1};
  double global_max_iteration_seconds{};
  int max_rank{};
  std::string backend;
  std::string rccl_sync_mode;
  int world_size{};
};

struct IterationPhaseTimingRecord {
  std::string backend;
  std::string rccl_sync_mode;
  std::string backend_schema;
  int world_size{};
  std::size_t halo_words{};
  std::size_t sample_index{1};
  std::size_t sample_count{1};
  std::size_t iteration_index{1};
  int iterations_in_sample{};
  double stall_threshold_us{};
  double total_iteration_seconds{};
  int total_iteration_max_rank{};
  std::string phase_name;
  double phase_seconds{};
  int phase_max_rank{};
};

struct BenchmarkResult {
  std::string backend;
  std::string algorithm;
  std::size_t halo_words{};
  std::size_t word_bytes{};
  std::size_t n_message_bytes{};
  std::size_t two_n_message_bytes{};
  std::size_t total_exchange_bytes_per_rank{};
  int iterations{};
  std::size_t sample_index{1};
  std::size_t sample_count{1};
  double max_average_seconds{};
  double max_total_seconds{};
  TopologyInfo topology;
  BackendMetadata metadata;
  std::optional<PhaseTimingResult> phase_timing;
  bool iteration_timing_enabled = false;
  double iteration_stall_threshold_us = 0.0;
  std::size_t iteration_total_observed = 0;
  std::size_t iteration_records_emitted = 0;
  std::size_t iteration_stall_count = 0;
  std::vector<IterationTimingRecord> iteration_times;
  bool iteration_phase_timing_enabled = false;
  std::size_t iteration_phase_records_emitted = 0;
  std::string iteration_phase_backend_schema;
  double iteration_phase_stall_threshold_us = 0.0;
  std::vector<IterationPhaseTimingRecord> iteration_phase_times;
  bool stalled_rank_timing_enabled = false;
  std::size_t stalled_rank_iterations_recorded = 0;
  std::size_t stalled_rank_rows_emitted = 0;
  std::string stalled_rank_schema;
  double stalled_rank_stall_threshold_us = 0.0;
  std::vector<StalledRankTimingRecord> stalled_rank_times;
};

std::vector<BenchmarkResult> run_benchmark(Backend& backend,
                                           const BenchmarkConfig& config);
std::vector<std::size_t> generate_halo_lengths(std::size_t min_halo,
                                               std::size_t max_halo,
                                               std::size_t multiplier);

} // namespace ghalo
