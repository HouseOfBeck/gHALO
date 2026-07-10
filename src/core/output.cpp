#include "ghalo/output.hpp"

#include "ghalo/version.hpp"

#include <fstream>
#include <iomanip>
#include <ostream>
#include <stdexcept>

namespace ghalo {
namespace {

void require_stream(const std::ofstream& stream, const std::string& path) {
  if (!stream) {
    throw std::runtime_error("failed to open output file: " + path);
  }
}

void write_json_string(std::ostream& out, const std::string& value) {
  out << '"';
  for (const char c : value) {
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
    default:
      out << c;
      break;
    }
  }
  out << '"';
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

  out << std::setw(8) << "N" << std::setw(14) << "iters" << std::setw(18)
      << "max avg seconds" << std::setw(14) << "bytes/rank" << "\n";
  out << std::string(54, '-') << "\n";

  out << std::scientific << std::setprecision(6);
  for (const auto& result : results) {
    out << std::setw(8) << result.halo_words << std::setw(14)
        << result.iterations << std::setw(18)
        << result.max_average_seconds << std::setw(14)
        << result.total_exchange_bytes_per_rank << "\n";
  }
}

void write_csv(const std::string& path,
               const std::vector<BenchmarkResult>& results) {
  std::ofstream out(path);
  require_stream(out, path);

  out << "version,backend,algorithm,world_size,rows,cols,halo_words,"
         "word_bytes,n_message_bytes,two_n_message_bytes,"
         "total_exchange_bytes_per_rank,iterations,max_total_seconds,"
         "max_average_seconds,root_world_rank,root_cart_rank,root_row,"
         "root_col,root_north,root_south,root_east,root_west,"
         "memory_location,mpi_library_version,hip_runtime_version,"
         "validation_enabled,validation_passed\n";

  out << std::scientific << std::setprecision(12);
  for (const auto& result : results) {
    out << version << ',' << result.backend << ',' << result.algorithm << ','
        << result.topology.world_size << ',' << result.topology.rows << ','
        << result.topology.cols << ',' << result.halo_words << ','
        << result.word_bytes << ',' << result.n_message_bytes << ','
        << result.two_n_message_bytes << ','
        << result.total_exchange_bytes_per_rank << ',' << result.iterations
        << ',' << result.max_total_seconds << ','
        << result.max_average_seconds << ',' << result.topology.world_rank
        << ',' << result.topology.cart_rank << ',' << result.topology.row
        << ',' << result.topology.col << ',' << result.topology.north << ','
        << result.topology.south << ',' << result.topology.east << ','
        << result.topology.west << ',' << result.metadata.memory_location
        << ',';
    write_json_string(out, result.metadata.mpi_library_version);
    out << ',';
    write_json_string(out, result.metadata.hip_runtime_version);
    out << ',' << (result.metadata.validation_enabled ? "true" : "false")
        << ',' << (result.metadata.validation_passed ? "true" : "false")
        << '\n';
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
    out << "        \"device_map\": ";
    write_json_string(out, r.metadata.device_map);
    out << ",\n";
    out << "        \"gpu_aware_mpi_requested\": "
        << (r.metadata.gpu_aware_mpi_requested ? "true" : "false") << ",\n";
    out << "        \"validation_enabled\": "
        << (r.metadata.validation_enabled ? "true" : "false") << ",\n";
    out << "        \"validation_passed\": "
        << (r.metadata.validation_passed ? "true" : "false") << ",\n";
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
    out << "      }\n";
    out << "    }" << (i + 1 == results.size() ? "" : ",") << "\n";
  }

  out << "  ]\n";
  out << "}\n";
}

} // namespace ghalo
