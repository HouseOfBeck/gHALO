#pragma once

#include "ghalo/backend.hpp"
#include "ghalo/halo_range.hpp"

#include <cstddef>
#include <optional>
#include <vector>

namespace ghalo {

struct BenchmarkConfig {
  std::size_t min_halo = default_min_halo;
  std::size_t max_halo = default_max_halo;
  std::size_t halo_multiplier = default_halo_multiplier;
  std::size_t samples_per_halo = 1;
  std::vector<std::size_t> halo_lengths{};
  double target_seconds = 3.0;
  int calibration_iterations = 5;
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
};

std::vector<BenchmarkResult> run_benchmark(Backend& backend,
                                           const BenchmarkConfig& config);
std::vector<std::size_t> generate_halo_lengths(std::size_t min_halo,
                                               std::size_t max_halo,
                                               std::size_t multiplier);

} // namespace ghalo
