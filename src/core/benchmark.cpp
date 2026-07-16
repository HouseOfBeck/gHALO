#include "ghalo/benchmark.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace ghalo {
namespace {

struct TimedExchangeResult {
  double local_total_seconds{};
  std::vector<double> local_iteration_seconds;
  std::vector<PhaseTimingResult> local_iteration_phases;
  std::size_t total_observed{};
  std::size_t records_emitted{};
  std::size_t stall_count{};
  std::vector<IterationTimingRecord> records;
  std::vector<IterationPhaseTimingRecord> phase_records;
  std::vector<StalledIterationLocalRecord> stalled_local_records;
  std::vector<StalledRankTimingRecord> stalled_rank_records;
};

struct PhaseValue {
  std::string name;
  double seconds{};
};

double time_exchanges(Backend& backend, int iterations) {
  const double start = backend.now();
  for (int i = 0; i < iterations; ++i) {
    backend.exchange();
  }
  return backend.now() - start;
}

std::string iteration_phase_schema(const std::string& backend_name,
                                   const BackendMetadata& metadata) {
  if (metadata.rccl_sync_mode == "stream-ordered") {
    return "rccl-stream-ordered";
  }
  if (backend_name.find("RCCL") != std::string::npos ||
      metadata.rccl_sync_mode == "conservative") {
    return "rccl-conservative";
  }
  if (backend_name.find("MPIHIP") != std::string::npos ||
      metadata.memory_location == "device") {
    return "mpi-hip";
  }
  return "unknown";
}

std::vector<PhaseValue> phase_values_for_schema(
    const std::string& schema, const PhaseTimingResult& phase) {
  if (schema == "rccl-stream-ordered") {
    return {
        {"north_south_enqueue",
         phase.north_south_communication_enqueue_seconds},
        {"transpose_enqueue", phase.transpose_copy_enqueue_seconds},
        {"east_west_enqueue", phase.east_west_communication_enqueue_seconds},
        {"final_stream_sync", phase.final_stream_sync_seconds},
    };
  }
  if (schema == "rccl-conservative") {
    return {
        {"north_south_communication",
         phase.north_south_communication_seconds},
        {"north_south_sync", phase.north_south_sync_seconds},
        {"transpose_copy", phase.transpose_copy_seconds},
        {"transpose_sync", phase.transpose_sync_seconds},
        {"east_west_communication",
         phase.east_west_communication_seconds},
        {"east_west_sync", phase.east_west_sync_seconds},
    };
  }
  if (schema == "mpi-hip") {
    return {
        {"input_device_copy", phase.input_device_copy_seconds},
        {"north_south_mpi", phase.north_south_mpi_seconds},
        {"north_south_sync", phase.north_south_sync_seconds},
        {"transpose_device_copy", phase.transpose_device_copy_seconds},
        {"transpose_copy_sync", phase.transpose_copy_sync_seconds},
        {"east_west_mpi", phase.east_west_mpi_seconds},
        {"east_west_sync", phase.east_west_sync_seconds},
    };
  }
  return {};
}

TimedExchangeResult time_exchanges_with_iteration_records(
    Backend& backend, int iterations, std::size_t halo_words,
    std::size_t sample_index, std::size_t sample_count, double threshold_us,
    bool record_phase_times, bool record_stalled_rank_times) {
  TimedExchangeResult result;
  result.total_observed = static_cast<std::size_t>(iterations);
  result.local_iteration_seconds.reserve(static_cast<std::size_t>(iterations));
  if (record_phase_times) {
    result.local_iteration_phases.reserve(static_cast<std::size_t>(iterations));
  }
  result.records.reserve(static_cast<std::size_t>(iterations));

  const bool emit_all = threshold_us <= 0.0;
  const double threshold_seconds = threshold_us * 1.0e-6;
  const auto backend_name = backend.name();
  const auto metadata = backend.metadata();
  const auto topology = backend.topology();
  const auto phase_schema = iteration_phase_schema(backend_name, metadata);

  for (int i = 0; i < iterations; ++i) {
    if (record_phase_times) {
      backend.begin_iteration_phase_timing();
    }
    const double start = backend.now();
    backend.exchange();
    const double local_seconds = backend.now() - start;
    result.local_total_seconds += local_seconds;
    result.local_iteration_seconds.push_back(local_seconds);
    if (record_phase_times) {
      result.local_iteration_phases.push_back(
          backend.end_iteration_phase_timing(local_seconds));
    }
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
      if (record_stalled_rank_times) {
        PhaseTimingResult local_phase;
        if (record_phase_times && i < result.local_iteration_phases.size()) {
          local_phase = result.local_iteration_phases[i];
        }
        result.stalled_local_records.push_back(
            StalledIterationLocalRecord{i + 1,
                                        result.local_iteration_seconds[i],
                                        reduced.seconds,
                                        reduced.rank,
                                        local_phase});
      }
    }
  }

  if (record_phase_times && !result.records.empty()) {
    const auto phase_template =
        phase_values_for_schema(phase_schema, result.local_iteration_phases[0]);
    for (const auto& phase_entry : phase_template) {
      std::vector<double> local_phase_seconds;
      local_phase_seconds.reserve(result.records.size());
      for (const auto& record : result.records) {
        const auto values = phase_values_for_schema(
            phase_schema,
            result.local_iteration_phases[record.iteration_index - 1]);
        const auto iter =
            std::find_if(values.begin(), values.end(), [&](const auto& value) {
              return value.name == phase_entry.name;
            });
        local_phase_seconds.push_back(iter == values.end() ? 0.0
                                                           : iter->seconds);
      }
      const auto phase_reductions =
          backend.max_time_ranks(local_phase_seconds);
      if (phase_reductions.size() != result.records.size()) {
        throw std::runtime_error(
            "backend returned mismatched iteration phase timing reductions");
      }
      for (std::size_t i = 0; i < result.records.size(); ++i) {
        const auto& total = result.records[i];
        const auto& phase_reduction = phase_reductions[i];
        result.phase_records.push_back(IterationPhaseTimingRecord{
            backend_name,
            metadata.rccl_sync_mode,
            phase_schema,
            topology.world_size,
            halo_words,
            sample_index,
            sample_count,
            total.iteration_index,
            iterations,
            threshold_us,
            total.global_max_iteration_seconds,
            total.max_rank,
            phase_entry.name,
            phase_reduction.seconds,
            phase_reduction.rank});
      }
    }
  }
  if (record_stalled_rank_times) {
    result.stalled_rank_records = backend.gather_stalled_rank_timings(
        halo_words, sample_index, sample_count, iterations, threshold_us,
        phase_schema, result.stalled_local_records);
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
  if (config.record_iteration_phase_times) {
    if (!config.record_iteration_times) {
      throw std::invalid_argument(
          "record_iteration_phase_times requires record_iteration_times");
    }
    if (!backend.supports_phase_timing()) {
      throw std::invalid_argument(
          "iteration phase timing requires a backend with phase timing support");
    }
  }
  if (config.record_stalled_rank_times) {
    if (!config.record_iteration_times || !config.record_iteration_phase_times) {
      throw std::invalid_argument(
          "record_stalled_rank_times requires record_iteration_times and "
          "record_iteration_phase_times");
    }
    if (!backend.supports_phase_timing()) {
      throw std::invalid_argument(
          "stalled rank timing requires a backend with phase timing support");
    }
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
            config.samples_per_halo, config.iteration_stall_threshold_us,
            config.record_iteration_phase_times,
            config.record_stalled_rank_times);
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
      result.iteration_phase_timing_enabled =
          config.record_iteration_phase_times;
      result.iteration_phase_stall_threshold_us =
          config.iteration_stall_threshold_us;
      result.iteration_phase_backend_schema =
          iteration_phase_schema(result.backend, backend.metadata());
      result.iteration_phase_records_emitted =
          iteration_timing.phase_records.size();
      result.iteration_phase_times =
          std::move(iteration_timing.phase_records);
      result.stalled_rank_timing_enabled = config.record_stalled_rank_times;
      result.stalled_rank_iterations_recorded =
          iteration_timing.stalled_local_records.size();
      result.stalled_rank_rows_emitted =
          iteration_timing.stalled_rank_records.size();
      result.stalled_rank_schema = result.iteration_phase_backend_schema;
      result.stalled_rank_stall_threshold_us =
          config.iteration_stall_threshold_us;
      result.stalled_rank_times =
          std::move(iteration_timing.stalled_rank_records);
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
