#include "ghalo/output.hpp"

#include "ghalo/version.hpp"

#include <fstream>
#include <iomanip>
#include <ostream>
#include <stdexcept>
#include <string>

namespace ghalo {
namespace {

void require_stream(const std::ofstream& stream, const std::string& path) {
  if (!stream) {
    throw std::runtime_error("failed to open output file: " + path);
  }
}

void write_json_string(std::ostream& out, const std::string& value) {
  out << '"';
  for (const unsigned char c : value) {
    switch (c) {
    case '\\':
      out << "\\\\";
      break;
    case '"':
      out << "\\\"";
      break;
    case '\n':
      out << "\\n";
      break;
    case '\r':
      out << "\\r";
      break;
    case '\t':
      out << "\\t";
      break;
    case '\b':
      out << "\\b";
      break;
    case '\f':
      out << "\\f";
      break;
    default:
      if (c < 0x20) {
        const auto flags = out.flags();
        const auto fill = out.fill();
        out << "\\u" << std::hex << std::uppercase << std::setw(4)
            << std::setfill('0') << static_cast<int>(c);
        out.flags(flags);
        out.fill(fill);
      } else {
        out << static_cast<char>(c);
      }
      break;
    }
  }
  out << '"';
}

bool has_phase_timing(const std::vector<BenchmarkResult>& results) {
  for (const auto& result : results) {
    if (result.phase_timing.has_value()) {
      return true;
    }
  }
  return false;
}

bool has_multiple_samples(const std::vector<BenchmarkResult>& results) {
  for (const auto& result : results) {
    if (result.sample_count > 1) {
      return true;
    }
  }
  return false;
}

bool has_iteration_timing(const std::vector<BenchmarkResult>& results) {
  for (const auto& result : results) {
    if (result.iteration_timing_enabled) {
      return true;
    }
  }
  return false;
}

std::size_t total_iteration_records_emitted(
    const std::vector<BenchmarkResult>& results) {
  std::size_t total = 0;
  for (const auto& result : results) {
    total += result.iteration_records_emitted;
  }
  return total;
}

std::size_t total_iteration_stall_count(
    const std::vector<BenchmarkResult>& results) {
  std::size_t total = 0;
  for (const auto& result : results) {
    total += result.iteration_stall_count;
  }
  return total;
}

std::size_t total_iteration_phase_records_emitted(
    const std::vector<BenchmarkResult>& results) {
  std::size_t total = 0;
  for (const auto& result : results) {
    total += result.iteration_phase_records_emitted;
  }
  return total;
}

bool has_iteration_phase_timing(const std::vector<BenchmarkResult>& results) {
  for (const auto& result : results) {
    if (result.iteration_phase_timing_enabled) {
      return true;
    }
  }
  return false;
}

bool has_input_device_copy_phase(const std::vector<BenchmarkResult>& results) {
  for (const auto& result : results) {
    if (result.phase_timing.has_value() &&
        result.phase_timing->input_device_copy_seconds != 0.0) {
      return true;
    }
  }
  return false;
}

bool has_generic_phase_fields(const PhaseTimingResult& phase) {
  return phase.north_south_communication_seconds != 0.0 ||
         phase.transpose_copy_seconds != 0.0 ||
         phase.transpose_sync_seconds != 0.0 ||
         phase.east_west_communication_seconds != 0.0 ||
         phase.total_exchange_seconds != 0.0;
}

bool has_stream_ordered_phase_fields(const PhaseTimingResult& phase) {
  return phase.north_south_communication_enqueue_seconds != 0.0 ||
         phase.transpose_copy_enqueue_seconds != 0.0 ||
         phase.east_west_communication_enqueue_seconds != 0.0 ||
         phase.final_stream_sync_seconds != 0.0;
}

double phase_north_south_communication(const PhaseTimingResult& phase) {
  return has_generic_phase_fields(phase)
             ? phase.north_south_communication_seconds
             : phase.north_south_mpi_seconds;
}

double phase_transpose_copy(const PhaseTimingResult& phase) {
  return has_generic_phase_fields(phase) ? phase.transpose_copy_seconds
                                         : phase.transpose_device_copy_seconds;
}

double phase_transpose_sync(const PhaseTimingResult& phase) {
  return has_generic_phase_fields(phase) ? phase.transpose_sync_seconds
                                         : phase.transpose_copy_sync_seconds;
}

double phase_east_west_communication(const PhaseTimingResult& phase) {
  return has_generic_phase_fields(phase)
             ? phase.east_west_communication_seconds
             : phase.east_west_mpi_seconds;
}

double phase_total_exchange(const PhaseTimingResult& phase,
                            const BenchmarkResult& result) {
  return phase.total_exchange_seconds != 0.0 ? phase.total_exchange_seconds
                                             : result.max_average_seconds;
}

double phase_total_minus_sum(const PhaseTimingResult& phase) {
  return has_generic_phase_fields(phase) ||
                 has_stream_ordered_phase_fields(phase)
             ? phase.total_minus_sum_of_phase_maxima_seconds
             : phase.unattributed_seconds;
}

std::string phase_timing_label(const BenchmarkResult& result) {
  if (!result.metadata.rccl_version.empty() ||
      result.algorithm.find("rccl") != std::string::npos ||
      result.backend.find("RCCL") != std::string::npos) {
    return "RCCL phase timing";
  }
  return "MPI-HIP phase timing";
}

} // namespace

void write_console(std::ostream& out,
                   const std::vector<BenchmarkResult>& results) {
  if (results.empty()) {
    out << "gHALO: no benchmark results\n";
    return;
  }

  const auto& topo = results.front().topology;
  const auto& metadata = results.front().metadata;
  out << "gHALO " << version << "\n";
  out << "Backend: " << results.front().backend << "\n";
  out << "Algorithm: " << results.front().algorithm << "\n";
  out << "Memory: " << metadata.memory_location << "\n";
  if (!metadata.hip_runtime_version.empty()) {
    out << "HIP runtime: " << metadata.hip_runtime_version << "\n";
  }
  if (!metadata.rccl_version.empty()) {
    out << "RCCL: " << metadata.rccl_version << "\n";
  }
  if (!metadata.rccl_stage.empty()) {
    out << "RCCL stage: " << metadata.rccl_stage << "\n";
  }
  if (!metadata.rccl_sync_mode.empty()) {
    out << "RCCL sync mode: " << metadata.rccl_sync_mode << "\n";
  }
  if (has_multiple_samples(results)) {
    out << "Samples per halo: " << results.front().sample_count << "\n";
  }
  out << "Ranks: " << topo.world_size << " as a " << topo.rows << " x "
      << topo.cols << " periodic Cartesian grid\n\n";

  if (!metadata.ranks.empty()) {
    out << "Rank mapping summary:\n";
    if (metadata.ranks.size() <= 16) {
      for (const auto& rank : metadata.ranks) {
        out << "  rank " << rank.world_rank << " local " << rank.local_rank
            << " host " << rank.hostname;
        if (rank.hip_device_index >= 0) {
          out << " hip_device " << rank.hip_device_index << " ("
              << rank.hip_device_name << ")";
        }
        if (!rank.rocr_visible_devices.empty()) {
          out << " ROCR_VISIBLE_DEVICES=" << rank.rocr_visible_devices;
        }
        out << " cart=(" << rank.row << "," << rank.col << ")\n";
      }
    } else {
      out << "  " << metadata.ranks.size()
          << " ranks; full rank mapping is available in JSON output\n";
    }
    out << "\n";
  }

  const bool multi_sample = has_multiple_samples(results);
  out << std::setw(8) << "N";
  if (multi_sample) {
    out << std::setw(10) << "sample";
  }
  out << std::setw(14) << "iters" << std::setw(18)
      << "max avg seconds" << std::setw(14) << "bytes/rank" << "\n";
  out << std::string(multi_sample ? 64 : 54, '-') << "\n";

  out << std::scientific << std::setprecision(6);
  for (const auto& result : results) {
    out << std::setw(8) << result.halo_words;
    if (multi_sample) {
      out << std::setw(10) << result.sample_index;
    }
    out << std::setw(14) << result.iterations << std::setw(18)
        << result.max_average_seconds << std::setw(14)
        << result.total_exchange_bytes_per_rank << "\n";
  }

  if (has_phase_timing(results)) {
    const bool include_input_copy = has_input_device_copy_phase(results);
    const bool stream_ordered_phases =
        results.front().metadata.rccl_sync_mode == "stream-ordered" ||
        (results.front().phase_timing.has_value() &&
         has_stream_ordered_phase_fields(*results.front().phase_timing));
    out << "\n" << phase_timing_label(results.front())
        << " (microseconds, max average per rank):\n";
    out << std::setw(8) << "N";
    if (multi_sample) {
      out << std::setw(10) << "sample";
    }
    if (stream_ordered_phases) {
      out << std::setw(18) << "ns_enqueue_us" << std::setw(22)
          << "transpose_enqueue_us" << std::setw(18) << "ew_enqueue_us"
          << std::setw(18) << "final_sync_us" << std::setw(16)
          << "phase_sum_us" << std::setw(14) << "total_us" << std::setw(22)
          << "total_minus_sum_us" << "\n";
      out << std::string(multi_sample ? 146 : 136, '-') << "\n";
    } else {
      if (include_input_copy) {
        out << std::setw(16) << "input_copy_us";
      }
      out << std::setw(14) << "ns_comm_us" << std::setw(14) << "ns_sync_us"
          << std::setw(20) << "transpose_copy_us" << std::setw(18)
          << "transpose_sync_us" << std::setw(14) << "ew_comm_us"
          << std::setw(14) << "ew_sync_us" << std::setw(16)
          << "phase_sum_us" << std::setw(14) << "total_us" << std::setw(22)
          << "total_minus_sum_us" << "\n";
      out << std::string((include_input_copy ? 156 : 140) +
                         (multi_sample ? 10 : 0),
                         '-')
          << "\n";
    }
    for (const auto& result : results) {
      if (!result.phase_timing.has_value()) {
        continue;
      }
      const auto& phase = *result.phase_timing;
      constexpr double us = 1.0e6;
      out << std::setw(8) << result.halo_words;
      if (multi_sample) {
        out << std::setw(10) << result.sample_index;
      }
      if (stream_ordered_phases) {
        out << std::setw(18)
            << phase.north_south_communication_enqueue_seconds * us
            << std::setw(22) << phase.transpose_copy_enqueue_seconds * us
            << std::setw(18)
            << phase.east_west_communication_enqueue_seconds * us
            << std::setw(18) << phase.final_stream_sync_seconds * us
            << std::setw(16) << phase.phase_sum_seconds * us << std::setw(14)
            << phase_total_exchange(phase, result) * us << std::setw(22)
            << phase_total_minus_sum(phase) * us << "\n";
        continue;
      }
      if (include_input_copy) {
        out << std::setw(16) << phase.input_device_copy_seconds * us;
      }
      out << std::setw(14) << phase_north_south_communication(phase) * us
          << std::setw(14)
          << phase.north_south_sync_seconds * us << std::setw(20)
          << phase_transpose_copy(phase) * us << std::setw(18)
          << phase_transpose_sync(phase) * us << std::setw(14)
          << phase_east_west_communication(phase) * us << std::setw(14)
          << phase.east_west_sync_seconds * us << std::setw(16)
          << phase.phase_sum_seconds * us << std::setw(14)
          << phase_total_exchange(phase, result) * us << std::setw(22)
          << phase_total_minus_sum(phase) * us << "\n";
    }
  }

  if (has_iteration_timing(results)) {
    out << "\nIteration timing diagnostics:\n";
    out << "  records emitted: " << total_iteration_records_emitted(results)
        << "\n";
    out << "  stall count: " << total_iteration_stall_count(results) << "\n";
    out << "  threshold_us: " << results.front().iteration_stall_threshold_us
        << " (0 emits every measured iteration)\n";
    if (has_multiple_samples(results)) {
      out << std::setw(8) << "N" << std::setw(10) << "sample"
          << std::setw(12) << "observed" << std::setw(12) << "emitted"
          << std::setw(10) << "stalls" << "\n";
      out << std::string(52, '-') << "\n";
      for (const auto& result : results) {
        out << std::setw(8) << result.halo_words << std::setw(10)
            << result.sample_index << std::setw(12)
            << result.iteration_total_observed << std::setw(12)
            << result.iteration_records_emitted << std::setw(10)
            << result.iteration_stall_count << "\n";
      }
    }
  }

  if (has_iteration_phase_timing(results)) {
    out << "\nIteration phase timing diagnostics:\n";
    out << "  iteration phase timing: enabled\n";
    out << "  phase records emitted: "
        << total_iteration_phase_records_emitted(results) << "\n";
    out << "  schema: " << results.front().iteration_phase_backend_schema
        << "\n";
  }
}

void write_csv(const std::string& path,
               const std::vector<BenchmarkResult>& results) {
  std::ofstream out(path);
  require_stream(out, path);
  const bool include_phase_timing = has_phase_timing(results);

  out << "version,backend,algorithm,world_size,rows,cols,halo_words,"
         "word_bytes,n_message_bytes,two_n_message_bytes,"
         "total_exchange_bytes_per_rank,iterations,max_total_seconds,"
         "max_average_seconds,sample_index,sample_count,root_world_rank,"
         "root_cart_rank,root_row,"
         "root_col,root_north,root_south,root_east,root_west,"
         "memory_location,mpi_library_version,hip_runtime_version,"
         "rocm_version,rccl_sync_mode,synchronization_model,validation_enabled,"
         "validation_passed";
  if (include_phase_timing) {
    out << ",phase_input_device_copy_seconds,"
           "phase_north_south_mpi_seconds,"
           "phase_north_south_sync_seconds,"
           "phase_transpose_device_copy_seconds,"
           "phase_transpose_copy_sync_seconds,"
           "phase_east_west_mpi_seconds,"
           "phase_east_west_sync_seconds,"
           "phase_north_south_communication_seconds,"
           "phase_transpose_copy_seconds,"
           "phase_transpose_sync_seconds,"
           "phase_east_west_communication_seconds,"
           "phase_north_south_communication_enqueue_seconds,"
           "phase_transpose_copy_enqueue_seconds,"
           "phase_east_west_communication_enqueue_seconds,"
           "phase_final_stream_sync_seconds,"
           "phase_sum_seconds,"
           "phase_unattributed_seconds,"
           "phase_total_exchange_seconds,"
           "phase_total_minus_sum_of_phase_maxima_seconds";
  }
  out << "\n";

  out << std::scientific << std::setprecision(12);
  for (const auto& result : results) {
    out << version << ',' << result.backend << ',' << result.algorithm << ','
        << result.topology.world_size << ',' << result.topology.rows << ','
        << result.topology.cols << ',' << result.halo_words << ','
        << result.word_bytes << ',' << result.n_message_bytes << ','
        << result.two_n_message_bytes << ','
        << result.total_exchange_bytes_per_rank << ',' << result.iterations
        << ',' << result.max_total_seconds << ','
        << result.max_average_seconds << ',' << result.sample_index << ','
        << result.sample_count << ',' << result.topology.world_rank << ','
        << result.topology.cart_rank << ',' << result.topology.row << ','
        << result.topology.col << ',' << result.topology.north << ','
        << result.topology.south << ',' << result.topology.east << ','
        << result.topology.west << ',' << result.metadata.memory_location
        << ',';
    write_json_string(out, result.metadata.mpi_library_version);
    out << ',';
    write_json_string(out, result.metadata.hip_runtime_version);
    out << ',';
    write_json_string(out, result.metadata.rocm_version);
    out << ',';
    write_json_string(out, result.metadata.rccl_sync_mode);
    out << ',';
    write_json_string(out, result.metadata.synchronization_model);
    out << ',' << (result.metadata.validation_enabled ? "true" : "false")
        << ',' << (result.metadata.validation_passed ? "true" : "false");
    if (include_phase_timing) {
      if (result.phase_timing.has_value()) {
        const auto& phase = *result.phase_timing;
        out << ',' << phase.input_device_copy_seconds << ','
            << phase.north_south_mpi_seconds << ','
            << phase.north_south_sync_seconds << ','
            << phase.transpose_device_copy_seconds << ','
            << phase.transpose_copy_sync_seconds << ','
            << phase.east_west_mpi_seconds << ','
            << phase.east_west_sync_seconds << ','
            << phase.north_south_communication_seconds << ','
            << phase.transpose_copy_seconds << ','
            << phase.transpose_sync_seconds << ','
            << phase.east_west_communication_seconds << ','
            << phase.north_south_communication_enqueue_seconds << ','
            << phase.transpose_copy_enqueue_seconds << ','
            << phase.east_west_communication_enqueue_seconds << ','
            << phase.final_stream_sync_seconds << ','
            << phase.phase_sum_seconds << ',' << phase.unattributed_seconds
            << ',' << phase_total_exchange(phase, result) << ','
            << phase_total_minus_sum(phase);
      } else {
        out << ",,,,,,,,,,,,,,,,,,,";
      }
    }
    out << '\n';
  }
}

void write_json(const std::string& path,
                const std::vector<BenchmarkResult>& results) {
  std::ofstream out(path);
  require_stream(out, path);

  out << std::scientific << std::setprecision(12);
  out << "{\n";
  out << "  \"version\": ";
  write_json_string(out, version);
  out << ",\n";
  out << "  \"results\": [\n";

  for (std::size_t i = 0; i < results.size(); ++i) {
    const auto& r = results[i];
    out << "    {\n";
    out << "      \"backend\": ";
    write_json_string(out, r.backend);
    out << ",\n";
    out << "      \"algorithm\": ";
    write_json_string(out, r.algorithm);
    out << ",\n";
    out << "      \"halo_words\": " << r.halo_words << ",\n";
    out << "      \"word_bytes\": " << r.word_bytes << ",\n";
    out << "      \"n_message_bytes\": " << r.n_message_bytes << ",\n";
    out << "      \"two_n_message_bytes\": " << r.two_n_message_bytes
        << ",\n";
    out << "      \"total_exchange_bytes_per_rank\": "
        << r.total_exchange_bytes_per_rank << ",\n";
    out << "      \"iterations\": " << r.iterations << ",\n";
    out << "      \"sample_index\": " << r.sample_index << ",\n";
    out << "      \"sample_count\": " << r.sample_count << ",\n";
    out << "      \"max_total_seconds\": " << r.max_total_seconds << ",\n";
    out << "      \"max_average_seconds\": " << r.max_average_seconds
        << ",\n";
    out << "      \"metadata\": {\n";
    out << "        \"memory_location\": ";
    write_json_string(out, r.metadata.memory_location);
    out << ",\n";
    out << "        \"mpi_library_version\": ";
    write_json_string(out, r.metadata.mpi_library_version);
    out << ",\n";
    out << "        \"hip_runtime_version\": ";
    write_json_string(out, r.metadata.hip_runtime_version);
    out << ",\n";
    out << "        \"rccl_version\": ";
    write_json_string(out, r.metadata.rccl_version);
    out << ",\n";
    out << "        \"rocm_version\": ";
    write_json_string(out, r.metadata.rocm_version);
    out << ",\n";
    out << "        \"rccl_stage\": ";
    write_json_string(out, r.metadata.rccl_stage);
    out << ",\n";
    out << "        \"rccl_sync_mode\": ";
    write_json_string(out, r.metadata.rccl_sync_mode);
    out << ",\n";
    out << "        \"rccl_plugin_root\": ";
    write_json_string(out, r.metadata.rccl_plugin_root);
    out << ",\n";
    out << "        \"synchronization_model\": ";
    write_json_string(out, r.metadata.synchronization_model);
    out << ",\n";
    out << "        \"transport_provider\": ";
    write_json_string(out, r.metadata.transport_provider);
    out << ",\n";
    out << "        \"device_map\": ";
    write_json_string(out, r.metadata.device_map);
    out << ",\n";
    out << "        \"gpu_aware_mpi_requested\": "
        << (r.metadata.gpu_aware_mpi_requested ? "true" : "false") << ",\n";
    out << "        \"validation_enabled\": "
        << (r.metadata.validation_enabled ? "true" : "false") << ",\n";
    out << "        \"validation_passed\": "
        << (r.metadata.validation_passed ? "true" : "false") << ",\n";
    out << "        \"iteration_timing_enabled\": "
        << (r.iteration_timing_enabled ? "true" : "false") << ",\n";
    out << "        \"iteration_stall_threshold_us\": "
        << r.iteration_stall_threshold_us << ",\n";
    out << "        \"iteration_total_observed\": "
        << r.iteration_total_observed << ",\n";
    out << "        \"iteration_records_emitted\": "
        << r.iteration_records_emitted << ",\n";
    out << "        \"iteration_stall_count\": "
        << r.iteration_stall_count << ",\n";
    out << "        \"iteration_phase_timing_enabled\": "
        << (r.iteration_phase_timing_enabled ? "true" : "false") << ",\n";
    out << "        \"iteration_phase_records_emitted\": "
        << r.iteration_phase_records_emitted << ",\n";
    out << "        \"iteration_phase_backend_schema\": ";
    write_json_string(out, r.iteration_phase_backend_schema);
    out << ",\n";
    out << "        \"iteration_phase_stall_threshold_us\": "
        << r.iteration_phase_stall_threshold_us << ",\n";
    if (r.metadata.phase_timing_enabled) {
      out << "        \"phase_timing_metadata\": {\n";
      out << "          \"enabled\": true,\n";
      out << "          \"timing_source\": ";
      write_json_string(out, r.metadata.phase_timing_source);
      out << ",\n";
      out << "          \"aggregation\": ";
      write_json_string(out, r.metadata.phase_timing_aggregation);
      out << ",\n";
      out << "          \"synchronization_model\": ";
      write_json_string(out, r.metadata.synchronization_model);
      out << ",\n";
      out << "          \"active_phases\": ";
      write_json_string(out, r.metadata.phase_timing_active_phases);
      out << "\n";
      out << "        },\n";
    }
    out << "        \"ranks\": [\n";
    for (std::size_t j = 0; j < r.metadata.ranks.size(); ++j) {
      const auto& rank = r.metadata.ranks[j];
      out << "          {\n";
      out << "            \"global_rank\": " << rank.world_rank << ",\n";
      out << "            \"local_rank\": " << rank.local_rank << ",\n";
      out << "            \"hostname\": ";
      write_json_string(out, rank.hostname);
      out << ",\n";
      out << "            \"hip_device_index\": " << rank.hip_device_index
          << ",\n";
      out << "            \"hip_device_name\": ";
      write_json_string(out, rank.hip_device_name);
      out << ",\n";
      out << "            \"rocr_visible_devices\": ";
      write_json_string(out, rank.rocr_visible_devices);
      out << ",\n";
      out << "            \"cart_rank\": " << rank.cart_rank << ",\n";
      out << "            \"cartesian_coordinates\": [" << rank.row << ", "
          << rank.col << "]\n";
      out << "          }" << (j + 1 == r.metadata.ranks.size() ? "" : ",")
          << "\n";
    }
    out << "        ]\n";
    out << "      },\n";
    out << "      \"topology\": {\n";
    out << "        \"world_size\": " << r.topology.world_size << ",\n";
    out << "        \"world_rank\": " << r.topology.world_rank << ",\n";
    out << "        \"cart_rank\": " << r.topology.cart_rank << ",\n";
    out << "        \"rows\": " << r.topology.rows << ",\n";
    out << "        \"cols\": " << r.topology.cols << ",\n";
    out << "        \"row\": " << r.topology.row << ",\n";
    out << "        \"col\": " << r.topology.col << ",\n";
    out << "        \"north\": " << r.topology.north << ",\n";
    out << "        \"south\": " << r.topology.south << ",\n";
    out << "        \"east\": " << r.topology.east << ",\n";
    out << "        \"west\": " << r.topology.west << "\n";
    out << "      }" << (r.phase_timing.has_value() ? "," : "") << "\n";
    if (r.phase_timing.has_value()) {
      const auto& phase = *r.phase_timing;
      out << "      \"phase_timing\": {\n";
      out << "        \"input_device_copy_seconds\": "
          << phase.input_device_copy_seconds << ",\n";
      out << "        \"north_south_mpi_seconds\": "
          << phase.north_south_mpi_seconds << ",\n";
      out << "        \"north_south_communication_seconds\": "
          << phase.north_south_communication_seconds << ",\n";
      out << "        \"north_south_sync_seconds\": "
          << phase.north_south_sync_seconds << ",\n";
      out << "        \"transpose_device_copy_seconds\": "
          << phase.transpose_device_copy_seconds << ",\n";
      out << "        \"transpose_copy_seconds\": "
          << phase.transpose_copy_seconds << ",\n";
      out << "        \"transpose_copy_sync_seconds\": "
          << phase.transpose_copy_sync_seconds << ",\n";
      out << "        \"transpose_sync_seconds\": "
          << phase.transpose_sync_seconds << ",\n";
      out << "        \"east_west_mpi_seconds\": "
          << phase.east_west_mpi_seconds << ",\n";
      out << "        \"east_west_communication_seconds\": "
          << phase.east_west_communication_seconds << ",\n";
      out << "        \"east_west_sync_seconds\": "
          << phase.east_west_sync_seconds << ",\n";
      out << "        \"north_south_communication_enqueue_seconds\": "
          << phase.north_south_communication_enqueue_seconds << ",\n";
      out << "        \"transpose_copy_enqueue_seconds\": "
          << phase.transpose_copy_enqueue_seconds << ",\n";
      out << "        \"east_west_communication_enqueue_seconds\": "
          << phase.east_west_communication_enqueue_seconds << ",\n";
      out << "        \"final_stream_sync_seconds\": "
          << phase.final_stream_sync_seconds << ",\n";
      out << "        \"phase_sum_seconds\": " << phase.phase_sum_seconds
          << ",\n";
      out << "        \"unattributed_seconds\": "
          << phase.unattributed_seconds << ",\n";
      out << "        \"total_exchange_seconds\": "
          << phase_total_exchange(phase, r) << ",\n";
      out << "        \"total_minus_sum_of_phase_maxima_seconds\": "
          << phase_total_minus_sum(phase) << "\n";
      out << "      }\n";
    }
    out << "    }" << (i + 1 == results.size() ? "" : ",") << "\n";
  }

  out << "  ]\n";
  out << "}\n";
}

void write_iteration_times_csv(const std::string& path,
                               const std::vector<BenchmarkResult>& results) {
  std::ofstream out(path);
  require_stream(out, path);
  out << "halo_words,sample_index,iteration_index,"
         "global_max_iteration_seconds,max_rank,backend,rccl_sync_mode,"
         "world_size\n";
  out << std::scientific << std::setprecision(12);
  for (const auto& result : results) {
    for (const auto& record : result.iteration_times) {
      out << record.halo_words << ',' << record.sample_index << ','
          << record.iteration_index << ','
          << record.global_max_iteration_seconds << ',' << record.max_rank
          << ',' << record.backend << ',';
      write_json_string(out, record.rccl_sync_mode);
      out << ',' << record.world_size << '\n';
    }
  }
}

void write_iteration_phase_times_csv(
    const std::string& path, const std::vector<BenchmarkResult>& results) {
  std::ofstream out(path);
  require_stream(out, path);
  out << "version,backend,rccl_sync_mode,world_size,halo_words,"
         "sample_index,sample_count,iteration_index,iterations_in_sample,"
         "stall_threshold_us,total_iteration_seconds,total_iteration_max_rank,"
         "backend_schema,phase_name,phase_seconds,phase_max_rank\n";
  out << std::scientific << std::setprecision(12);
  for (const auto& result : results) {
    for (const auto& record : result.iteration_phase_times) {
      out << version << ',';
      write_json_string(out, record.backend);
      out << ',';
      write_json_string(out, record.rccl_sync_mode);
      out << ',' << record.world_size << ',' << record.halo_words << ','
          << record.sample_index << ',' << record.sample_count << ','
          << record.iteration_index << ',' << record.iterations_in_sample
          << ',' << record.stall_threshold_us << ','
          << record.total_iteration_seconds << ','
          << record.total_iteration_max_rank << ',';
      write_json_string(out, record.backend_schema);
      out << ',';
      write_json_string(out, record.phase_name);
      out << ',' << record.phase_seconds << ',' << record.phase_max_rank
          << '\n';
    }
  }
}

} // namespace ghalo
