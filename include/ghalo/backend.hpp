#pragma once

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace ghalo {

struct TopologyInfo {
  int world_size{};
  int world_rank{};
  int cart_rank{};
  int rows{};
  int cols{};
  int row{};
  int col{};
  int north{};
  int south{};
  int east{};
  int west{};
};

struct RankMetadata {
  int world_rank{};
  int local_rank{-1};
  int cart_rank{};
  int row{};
  int col{};
  int hip_device_index{-1};
  std::string hostname;
  std::string hip_device_name;
  std::string rocr_visible_devices;
};

struct BackendMetadata {
  std::string mpi_library_version;
  std::string hip_runtime_version;
  std::string rccl_version;
  std::string rocm_version;
  std::string rccl_stage;
  std::string rccl_sync_mode;
  std::string rccl_plugin_root;
  std::string memory_location = "host";
  std::string device_map;
  std::string phase_timing_source;
  std::string phase_timing_aggregation;
  std::string phase_timing_active_phases;
  std::string synchronization_model;
  std::string transport_provider;
  bool gpu_aware_mpi_requested = false;
  bool validation_enabled = false;
  bool validation_passed = false;
  bool phase_timing_enabled = false;
  std::vector<RankMetadata> ranks;
};

struct PhaseTimingResult {
  double input_device_copy_seconds{};
  double north_south_mpi_seconds{};
  double north_south_communication_seconds{};
  double north_south_sync_seconds{};
  double transpose_device_copy_seconds{};
  double transpose_copy_seconds{};
  double transpose_copy_sync_seconds{};
  double transpose_sync_seconds{};
  double east_west_mpi_seconds{};
  double east_west_communication_seconds{};
  double east_west_sync_seconds{};
  double north_south_communication_enqueue_seconds{};
  double transpose_copy_enqueue_seconds{};
  double east_west_communication_enqueue_seconds{};
  double final_stream_sync_seconds{};
  double phase_sum_seconds{};
  double unattributed_seconds{};
  double total_exchange_seconds{};
  double total_minus_sum_of_phase_maxima_seconds{};
};

struct IterationTimingReduction {
  double seconds{};
  int rank{};
};

struct StalledIterationLocalRecord {
  std::size_t iteration_index{1};
  double local_total_iteration_seconds{};
  double global_max_iteration_seconds{};
  int global_max_iteration_rank{};
  PhaseTimingResult local_phase;
};

struct StalledRankTimingRecord {
  std::string backend;
  std::string rccl_sync_mode;
  std::string schema;
  int world_size{};
  std::size_t halo_words{};
  std::size_t sample_index{1};
  std::size_t sample_count{1};
  std::size_t iteration_index{1};
  int iterations_in_sample{};
  double stall_threshold_us{};
  int world_rank{};
  int local_rank{-1};
  std::string hostname;
  int selected_hip_device{-1};
  int cart_rank{};
  int cart_row{};
  int cart_col{};
  double local_total_iteration_seconds{};
  double global_max_iteration_seconds{};
  int global_max_iteration_rank{};
  PhaseTimingResult local_phase;
};

inline double phase_timing_sum(const PhaseTimingResult& phase) {
  const bool has_generic_fields =
      phase.north_south_communication_seconds != 0.0 ||
      phase.transpose_copy_seconds != 0.0 ||
      phase.transpose_sync_seconds != 0.0 ||
      phase.east_west_communication_seconds != 0.0;
  const bool has_stream_ordered_fields =
      phase.north_south_communication_enqueue_seconds != 0.0 ||
      phase.transpose_copy_enqueue_seconds != 0.0 ||
      phase.east_west_communication_enqueue_seconds != 0.0 ||
      phase.final_stream_sync_seconds != 0.0;
  if (has_stream_ordered_fields) {
    return phase.north_south_communication_enqueue_seconds +
           phase.transpose_copy_enqueue_seconds +
           phase.east_west_communication_enqueue_seconds +
           phase.final_stream_sync_seconds;
  }
  if (has_generic_fields) {
    return phase.input_device_copy_seconds +
           phase.north_south_communication_seconds +
           phase.north_south_sync_seconds + phase.transpose_copy_seconds +
           phase.transpose_sync_seconds +
           phase.east_west_communication_seconds +
           phase.east_west_sync_seconds;
  }
  return phase.input_device_copy_seconds + phase.north_south_mpi_seconds +
         phase.north_south_sync_seconds +
         phase.transpose_device_copy_seconds +
         phase.transpose_copy_sync_seconds + phase.east_west_mpi_seconds +
         phase.east_west_sync_seconds;
}

class Backend {
public:
  virtual ~Backend() = default;

  virtual std::string name() const = 0;
  virtual std::string algorithm() const = 0;
  virtual TopologyInfo topology() const = 0;
  virtual BackendMetadata metadata() const { return {}; }

  virtual int rank() const = 0;
  virtual int size() const = 0;
  virtual bool is_root() const = 0;

  virtual void setup(std::size_t halo_words) = 0;
  virtual void validate_current_halo() {}
  virtual void exchange() = 0;
  virtual void barrier() = 0;
  virtual double now() const = 0;
  virtual double max_time(double local_seconds) = 0;
  virtual std::vector<IterationTimingReduction> max_time_ranks(
      const std::vector<double>& local_seconds) {
    std::vector<IterationTimingReduction> reductions;
    reductions.reserve(local_seconds.size());
    for (const double seconds : local_seconds) {
      reductions.push_back({max_time(seconds), rank()});
    }
    return reductions;
  }
  virtual std::vector<StalledRankTimingRecord> gather_stalled_rank_timings(
      std::size_t halo_words, std::size_t sample_index,
      std::size_t sample_count, int iterations_in_sample,
      double stall_threshold_us, const std::string& backend_schema,
      const std::vector<StalledIterationLocalRecord>& local_records) {
    std::vector<StalledRankTimingRecord> records;
    const auto topo = topology();
    const auto meta = metadata();
    RankMetadata rank_meta;
    for (const auto& candidate : meta.ranks) {
      if (candidate.world_rank == topo.world_rank) {
        rank_meta = candidate;
        break;
      }
    }
    if (rank_meta.world_rank == 0 && topo.world_rank != 0) {
      rank_meta.world_rank = topo.world_rank;
      rank_meta.cart_rank = topo.cart_rank;
      rank_meta.row = topo.row;
      rank_meta.col = topo.col;
    }
    for (const auto& local : local_records) {
      StalledRankTimingRecord record;
      record.backend = name();
      record.rccl_sync_mode = meta.rccl_sync_mode;
      record.schema = backend_schema;
      record.world_size = topo.world_size;
      record.halo_words = halo_words;
      record.sample_index = sample_index;
      record.sample_count = sample_count;
      record.iteration_index = local.iteration_index;
      record.iterations_in_sample = iterations_in_sample;
      record.stall_threshold_us = stall_threshold_us;
      record.world_rank = topo.world_rank;
      record.local_rank = rank_meta.local_rank;
      record.hostname = rank_meta.hostname;
      record.selected_hip_device = rank_meta.hip_device_index;
      record.cart_rank = topo.cart_rank;
      record.cart_row = topo.row;
      record.cart_col = topo.col;
      record.local_total_iteration_seconds =
          local.local_total_iteration_seconds;
      record.global_max_iteration_seconds =
          local.global_max_iteration_seconds;
      record.global_max_iteration_rank = local.global_max_iteration_rank;
      record.local_phase = local.local_phase;
      records.push_back(record);
    }
    return records;
  }
  virtual void run_development_validation(const std::vector<std::size_t>&) {
    throw std::runtime_error(
        "development validation is not supported by backend " + name());
  }

  virtual bool supports_phase_timing() const { return false; }
  virtual void set_phase_timing_enabled(bool enabled) {
    if (enabled) {
      throw std::runtime_error("phase timing is not supported by backend " +
                               name());
    }
  }
  virtual bool phase_timing_enabled() const { return false; }
  virtual void reset_phase_timing() {}
  virtual PhaseTimingResult phase_timing_result(int /*iterations*/,
                                                double /*total_seconds*/) {
    return {};
  }
  virtual void begin_iteration_phase_timing() {}
  virtual PhaseTimingResult end_iteration_phase_timing(
      double /*total_seconds*/) {
    return {};
  }
};

} // namespace ghalo
