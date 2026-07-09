#pragma once

#include "ghalo/backend.hpp"

#include <cstddef>
#include <vector>

namespace ghalo {

struct BenchmarkConfig {
  std::vector<std::size_t> halo_lengths{2, 4, 8, 16, 32, 64,
                                        128, 256, 512, 1024};
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
  double max_average_seconds{};
  double max_total_seconds{};
  TopologyInfo topology;
};

std::vector<BenchmarkResult> run_benchmark(Backend& backend,
                                           const BenchmarkConfig& config);

} // namespace ghalo
