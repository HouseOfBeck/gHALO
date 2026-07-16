#include "ghalo/benchmark.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace ghalo {
namespace {

struct TimedExchangeResult {
  double local_total_seconds{};
  std::vector<double> local_iteration_seconds;
  std::size_t total_observed{};
  std::size_t records_emitted{};
  std::size_t stall_count{};
  std::vector<IterationTimingRecord> records;
};

double time_exchanges(Backend& backend, int iterations) {
  const double start = backend.now();
  for (int i = 0; i < iterations; ++i) {
    backend.exchange();
  }
  return backend.now() - start;
}

TimedExchangeResult time_exchanges_with_iteration_records(
    Backend& backend, int iterations, std::size_t halo_words,
    std::size_t sample_index, double threshold_us) {
  TimedExchangeResult result;
  result.total_observed = static_cast<std::size_t>(iterations);
  result.local_iteration_seconds.reserve(static_cast<std::size_t>(iterations));
  result.records.reserve(static_cast<std::size_t>(iterations));

  const bool emit_all = threshold_us <= 0.0;
  const double threshold_seconds = threshold_us * 1.0e-6;
  const auto backend_name = backend.name();
  const auto metadata = backend.metadata();
  const auto topology = backend.topology();

  for (int i = 0; i < iterations; ++i) {
    const double start = backend.now();
    backend.exchange();
    const double local_seconds = backend.now() - start;
    result.local_total_seconds += local_seconds;
    result.local_iteration_seconds.push_back(local_seconds);
  }

  const auto reductions = backend.max_time_ranks(result.local_iteration_seconds);
  if (reductions.size() != result.local_iteration_seconds.size()) {
    throw std::runtime_error(
        "backend returned mismatched iteration timing reductions");
  }

  for (std::size_t i = 0; i < reductions.size(); ++i) {
    const auto& reduced = reductions[i];
    const bool is_stall =
        !emit_all && reduced.seconds >= threshold_seconds;
    if (is_stall) {
      ++result.stall_count;
    }
    if (emit_all || is_stall) {
      result.records.push_back(
          IterationTimingRecord{halo_words,
                                sample_index,
                                i + 1,
                                reduced.seconds,
                                reduced.rank,
                                backend_name,
                                metadata.rccl_sync_mode,
                                topology.world_size});
      ++result.records_emitted;
    }
  }
  return result;
}

} // namespace

std::vector<std::size_t> generate_halo_lengths(std::size_t min_halo,
                                               std::size_t max_halo,
                                               std::size_t multiplier) {
  if (min_halo == 0) {
    throw std::invalid_argument("min_halo must be positive");
  }
  if (max_halo == 0) {
    throw std::invalid_argument("max_halo must be positive");
  }
  if (multiplier <= 1) {
    throw std::invalid_argument("halo_multiplier must be greater than 1");
  }
  if (max_halo < min_halo) {
    throw std::invalid_argument(
        "max_halo must be greater than or equal to min_halo");
  }

  std::vector<std::size_t> lengths;
  for (std::size_t halo = min_halo; halo <= max_halo;) {
    lengths.push_back(halo);
    if (halo > std::numeric_limits<std::size_t>::max() / multiplier) {
      if (halo != max_halo) {
        throw std::invalid_argument("halo range overflows size_t");
      }
      break;
    }
    const std::size_t next = halo * multiplier;
    if (next <= halo) {
      throw std::invalid_argument("halo range does not progress");
    }
    if (next > max_halo) {
      break;
    }
    halo = next;
  }
  return lengths;
}

std::vector<BenchmarkResult> run_benchmark(Backend& backend,
                                           const BenchmarkConfig& config) {
  if (config.target_seconds <= 0.0) {
    throw std::invalid_argument("target_seconds must be positive");
  }
  if (config.calibration_iterations <= 0) {
    throw std::invalid_argument("calibration_iterations must be positive");
  }
  if (config.samples_per_halo == 0) {
    throw std::invalid_argument("samples_per_halo must be positive");
  }
  if (config.iteration_stall_threshold_us < 0.0) {
    throw std::invalid_argument(
        "iteration_stall_threshold_us must be nonnegative");
  }

  const std::vector<std::size_t> halo_lengths =
      config.halo_lengths.empty()
          ? generate_halo_lengths(config.min_halo, config.max_halo,
                                  config.halo_multiplier)
          : config.halo_lengths;

  std::vector<BenchmarkResult> results;
  results.reserve(halo_lengths.size() * config.samples_per_halo);

  for (const std::size_t halo_words : halo_lengths) {
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

    for (std::size_t sample = 1; sample <= config.samples_per_halo; ++sample) {
      backend.validate_current_halo();

      if (backend.phase_timing_enabled()) {
        backend.reset_phase_timing();
      }

      backend.barrier();
      TimedExchangeResult iteration_timing;
      double measured_local = 0.0;
      if (config.record_iteration_times) {
        iteration_timing = time_exchanges_with_iteration_records(
            backend, iterations, halo_words, sample,
            config.iteration_stall_threshold_us);
        measured_local = iteration_timing.local_total_seconds;
      } else {
        measured_local = time_exchanges(backend, iterations);
      }
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
      result.sample_index = sample;
      result.sample_count = config.samples_per_halo;
      result.iteration_timing_enabled = config.record_iteration_times;
      result.iteration_stall_threshold_us =
          config.iteration_stall_threshold_us;
      result.iteration_total_observed = iteration_timing.total_observed;
      result.iteration_records_emitted = iteration_timing.records_emitted;
      result.iteration_stall_count = iteration_timing.stall_count;
      result.iteration_times = std::move(iteration_timing.records);
      result.max_total_seconds = measured_max;
      result.max_average_seconds =
          measured_max / static_cast<double>(iterations);
      result.topology = backend.topology();
      result.metadata = backend.metadata();
      if (backend.phase_timing_enabled()) {
        result.phase_timing =
            backend.phase_timing_result(iterations, result.max_average_seconds);
        result.metadata = backend.metadata();
      }
      results.push_back(result);
    }
  }

  return results;
}

} // namespace ghalo
