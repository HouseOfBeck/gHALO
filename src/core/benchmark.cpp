#include "ghalo/benchmark.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace ghalo {
namespace {

double time_exchanges(Backend& backend, int iterations) {
  const double start = backend.now();
  for (int i = 0; i < iterations; ++i) {
    backend.exchange();
  }
  return backend.now() - start;
}

} // namespace

std::vector<BenchmarkResult> run_benchmark(Backend& backend,
                                           const BenchmarkConfig& config) {
  if (config.target_seconds <= 0.0) {
    throw std::invalid_argument("target_seconds must be positive");
  }
  if (config.calibration_iterations <= 0) {
    throw std::invalid_argument("calibration_iterations must be positive");
  }

  std::vector<BenchmarkResult> results;
  results.reserve(config.halo_lengths.size());

  for (const std::size_t halo_words : config.halo_lengths) {
    backend.setup(halo_words);

    // One unmeasured exchange preserves the original HALO steady-state method:
    // backend setup, first-use costs, and request/window creation are excluded.
    backend.exchange();

    backend.barrier();
    const double calibration_local =
        time_exchanges(backend, config.calibration_iterations);
    const double calibration_max = backend.max_time(calibration_local);
    const double calibration_average =
        calibration_max / static_cast<double>(config.calibration_iterations);

    const int estimated_iterations =
        static_cast<int>(std::llround(config.target_seconds /
                                      std::max(calibration_average, 1.0e-12)));
    const int iterations =
        std::max(config.calibration_iterations, estimated_iterations);

    if (backend.phase_timing_enabled()) {
      backend.reset_phase_timing();
    }

    backend.barrier();
    const double measured_local = time_exchanges(backend, iterations);
    const double measured_max = backend.max_time(measured_local);

    BenchmarkResult result;
    result.backend = backend.name();
    result.algorithm = backend.algorithm();
    result.halo_words = halo_words;
    result.word_bytes = sizeof(float);
    result.n_message_bytes = halo_words * result.word_bytes;
    result.two_n_message_bytes = 2 * halo_words * result.word_bytes;
    result.total_exchange_bytes_per_rank = 6 * halo_words * result.word_bytes;
    result.iterations = iterations;
    result.max_total_seconds = measured_max;
    result.max_average_seconds = measured_max / static_cast<double>(iterations);
    result.topology = backend.topology();
    result.metadata = backend.metadata();
    if (backend.phase_timing_enabled()) {
      result.phase_timing =
          backend.phase_timing_result(iterations, result.max_average_seconds);
      result.metadata = backend.metadata();
    }
    results.push_back(result);
  }

  return results;
}

} // namespace ghalo
