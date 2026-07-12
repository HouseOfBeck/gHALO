#include "ghalo/benchmark.hpp"
#include "ghalo/cli.hpp"
#include "ghalo/exchange.hpp"
#include "ghalo/output.hpp"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>
#include <chrono>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <stdexcept>
#include <system_error>

namespace {

class MockBackend final : public ghalo::Backend {
public:
  std::string name() const override { return "MockBackend"; }
  std::string algorithm() const override { return "mock"; }
  ghalo::TopologyInfo topology() const override { return topology_; }

  int rank() const override { return 0; }
  int size() const override { return 1; }
  bool is_root() const override { return true; }

  void setup(std::size_t halo_words) override { halo_words_ = halo_words; }

  void exchange() override {
    assert(halo_words_ != 0);
    now_ += 0.01;
    ++exchanges_;
  }

  void barrier() override {}

  double now() const override { return now_; }

  double max_time(double local_seconds) override { return local_seconds; }

  int exchanges() const { return exchanges_; }

private:
  ghalo::TopologyInfo topology_{1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0};
  std::size_t halo_words_ = 0;
  double now_ = 0.0;
  int exchanges_ = 0;
};

struct ConceptualTopology {
  int rank;
  int rows;
  int cols;
  int row;
  int col;
  int north;
  int south;
  int east;
  int west;
};

ConceptualTopology topology_for(int rows, int cols, int rank) {
  const int row = rank / cols;
  const int col = rank % cols;
  return {
      rank,
      rows,
      cols,
      row,
      col,
      ((row + 1) % rows) * cols + col,
      ((row + rows - 1) % rows) * cols + col,
      row * cols + ((col + 1) % cols),
      row * cols + ((col + cols - 1) % cols),
  };
}

void check_self_copy_decisions(const ConceptualTopology& topology,
                               bool ns_self, bool ew_self) {
  assert(ghalo::use_local_copy_for_exchange_segment(
             topology.rank, topology.south, topology.north) == ns_self);
  assert(ghalo::use_local_copy_for_exchange_segment(
             topology.rank, topology.north, topology.south) == ns_self);
  assert(ghalo::use_local_copy_for_exchange_segment(
             topology.rank, topology.west, topology.east) == ew_self);
  assert(ghalo::use_local_copy_for_exchange_segment(
             topology.rank, topology.east, topology.west) == ew_self);
}

void check_north_south_plan(const ConceptualTopology& topology) {
  assert(topology.south == topology_for(topology.rows, topology.cols,
                                        topology.rank)
                              .south);
  assert(topology.north == topology_for(topology.rows, topology.cols,
                                        topology.rank)
                              .north);
}

float conceptual_pattern_value(int source_rank, int segment, int halo_words,
                               int index) {
  return static_cast<float>(source_rank * 100000 + segment * 10000 +
                            halo_words * 10 + (index % 10));
}

void test_halo_buffer_layout_helpers() {
  assert(ghalo::halo_n_offset() == 0);
  assert(ghalo::halo_two_n_offset(2) == 2);
  assert(ghalo::halo_buffer_words(2) == 6);
  assert(ghalo::halo_two_n_offset(1024) == 1024);
  assert(ghalo::halo_buffer_words(1024) == 3072);
}

void test_self_copy_topologies() {
  check_self_copy_decisions(topology_for(1, 1, 0), true, true);

  check_self_copy_decisions(topology_for(1, 2, 0), true, false);
  check_self_copy_decisions(topology_for(1, 2, 1), true, false);

  for (int rank = 0; rank < 4; ++rank) {
    check_self_copy_decisions(topology_for(2, 2, rank), false, false);
  }
}

void test_full_halo_corner_expectations() {
  constexpr int halo_words = 8;
  const int expected_corner_rank_by_rank[] = {3, 2, 1, 0};
  for (int rank = 0; rank < 4; ++rank) {
    const auto topology = topology_for(2, 2, rank);
    const int north_east =
        topology_for(topology.rows, topology.cols, topology.north).east;
    const int south_west =
        topology_for(topology.rows, topology.cols, topology.south).west;

    assert(north_east == expected_corner_rank_by_rank[rank]);
    assert(south_west == expected_corner_rank_by_rank[rank]);
    assert(conceptual_pattern_value(north_east, 1, halo_words, 0) ==
           static_cast<float>(north_east * 100000 + 10000 +
                              halo_words * 10));
    const float expected_south_west =
        static_cast<float>(south_west * 100000 + 20000 + halo_words * 10 +
                           (halo_words % 10));
    assert(conceptual_pattern_value(south_west, 2, halo_words, halo_words) ==
           expected_south_west);
  }
}

void test_bytes_per_rank_consistency() {
  constexpr std::size_t halo_words = 1024;
  constexpr std::size_t word_bytes = sizeof(float);
  const std::size_t n_message_bytes = halo_words * word_bytes;
  const std::size_t two_n_message_bytes = 2 * halo_words * word_bytes;
  const std::size_t total_exchange_bytes_per_rank =
      6 * halo_words * word_bytes;

  assert(n_message_bytes == 4096);
  assert(two_n_message_bytes == 8192);
  assert(total_exchange_bytes_per_rank == 24576);
}

void test_north_south_stage_b_conceptual_plans() {
  check_north_south_plan(topology_for(1, 1, 0));

  for (int rank = 0; rank < 2; ++rank) {
    const auto topology = topology_for(1, 2, rank);
    assert(topology.north == rank);
    assert(topology.south == rank);
    check_north_south_plan(topology);
  }

  for (int rank = 0; rank < 2; ++rank) {
    const auto topology = topology_for(2, 1, rank);
    assert(topology.north == 1 - rank);
    assert(topology.south == 1 - rank);
    check_north_south_plan(topology);
  }

  for (int rank = 0; rank < 4; ++rank) {
    const auto topology = topology_for(2, 2, rank);
    assert(topology.north == topology.south);
    check_north_south_plan(topology);
  }
}

class ScopedTempDirectory {
public:
  ScopedTempDirectory() {
    const auto parent = std::filesystem::temp_directory_path();
    std::filesystem::create_directories(parent);
    const auto stamp = std::chrono::steady_clock::now()
                           .time_since_epoch()
                           .count();
    for (int attempt = 0; attempt != 100; ++attempt) {
      const auto candidate =
          parent / ("ghalo_core_smoke_" +
                    std::to_string(stamp + static_cast<long long>(attempt)));
      if (std::filesystem::create_directory(candidate)) {
        path_ = candidate;
        return;
      }
    }
    throw std::runtime_error("failed to create unique core smoke temp directory");
  }

  ~ScopedTempDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  ScopedTempDirectory(const ScopedTempDirectory&) = delete;
  ScopedTempDirectory& operator=(const ScopedTempDirectory&) = delete;

  std::filesystem::path file(const std::string& name) const {
    return path_ / name;
  }

private:
  std::filesystem::path path_;
};

std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path);
  assert(input);
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

ghalo::BenchmarkResult sample_result() {
  ghalo::BenchmarkResult result;
  result.backend = "MockBackend";
  result.algorithm = "mock";
  result.halo_words = 2;
  result.word_bytes = sizeof(float);
  result.n_message_bytes = 8;
  result.two_n_message_bytes = 16;
  result.total_exchange_bytes_per_rank = 48;
  result.iterations = 5;
  result.max_total_seconds = 0.05;
  result.max_average_seconds = 0.01;
  result.topology = {1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0};
  result.metadata.memory_location = "host";
  return result;
}

void test_phase_sum_calculation() {
  ghalo::PhaseTimingResult phase;
  phase.input_device_copy_seconds = 1.0;
  phase.north_south_mpi_seconds = 2.0;
  phase.north_south_sync_seconds = 3.0;
  phase.transpose_device_copy_seconds = 4.0;
  phase.transpose_copy_sync_seconds = 5.0;
  phase.east_west_mpi_seconds = 6.0;
  phase.east_west_sync_seconds = 7.0;
  assert(ghalo::phase_timing_sum(phase) == 28.0);

  ghalo::PhaseTimingResult generic_phase;
  generic_phase.north_south_communication_seconds = 1.0;
  generic_phase.north_south_sync_seconds = 2.0;
  generic_phase.transpose_copy_seconds = 3.0;
  generic_phase.transpose_sync_seconds = 4.0;
  generic_phase.east_west_communication_seconds = 5.0;
  generic_phase.east_west_sync_seconds = 6.0;
  assert(ghalo::phase_timing_sum(generic_phase) == 21.0);
}

void test_unsupported_phase_timing() {
  MockBackend backend;
  bool threw = false;
  try {
    backend.set_phase_timing_enabled(true);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  assert(threw);
  assert(!backend.phase_timing_enabled());
}

void test_cli_phase_timing_parse() {
  const char* argv_storage[] = {"ghalo", "--backend", "mpi-hip",
                                "--phase-timing", "--target-seconds", "0.1"};
  auto* argv = const_cast<char**>(argv_storage);
  const auto options = ghalo::parse_cli_options(6, argv);
  assert(options.backend == "mpi-hip");
  assert(options.phase_timing);
  assert(options.target_seconds == 0.1);

  const char* rccl_argv_storage[] = {"ghalo", "--backend", "rccl"};
  auto* rccl_argv = const_cast<char**>(rccl_argv_storage);
  const auto rccl_options = ghalo::parse_cli_options(3, rccl_argv);
  assert(rccl_options.backend == "rccl");
  assert(!rccl_options.rccl_stage_b);
  assert(rccl_options.rccl_sync_mode == "conservative");

  const char* rccl_stream_argv_storage[] = {
      "ghalo", "--backend", "rccl", "--rccl-sync-mode", "stream-ordered"};
  auto* rccl_stream_argv = const_cast<char**>(rccl_stream_argv_storage);
  const auto rccl_stream_options =
      ghalo::parse_cli_options(5, rccl_stream_argv);
  assert(rccl_stream_options.backend == "rccl");
  assert(rccl_stream_options.rccl_sync_mode == "stream-ordered");

  const char* rccl_stage_argv_storage[] = {"ghalo", "--backend", "rccl",
                                           "--rccl-stage-b"};
  auto* rccl_stage_argv = const_cast<char**>(rccl_stage_argv_storage);
  const auto rccl_stage_options =
      ghalo::parse_cli_options(4, rccl_stage_argv);
  assert(rccl_stage_options.backend == "rccl");
  assert(rccl_stage_options.rccl_stage_b);

  const char* bad_sync_argv_storage[] = {
      "ghalo", "--backend", "rccl", "--rccl-sync-mode", "fast"};
  auto* bad_sync_argv = const_cast<char**>(bad_sync_argv_storage);
  bool bad_sync_threw = false;
  try {
    (void)ghalo::parse_cli_options(5, bad_sync_argv);
  } catch (const std::invalid_argument&) {
    bad_sync_threw = true;
  }
  assert(bad_sync_threw);

  const char* bad_argv_storage[] = {"ghalo", "--phase-timing", "--csv"};
  auto* bad_argv = const_cast<char**>(bad_argv_storage);
  bool threw = false;
  try {
    (void)ghalo::parse_cli_options(3, bad_argv);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  assert(threw);
}

void test_development_validation_refusal() {
  MockBackend backend;
  bool threw = false;
  try {
    backend.run_development_validation({2});
  } catch (const std::runtime_error& error) {
    threw = true;
    assert(std::string(error.what()).find("development validation") !=
           std::string::npos);
  }
  assert(threw);
}

void test_output_without_phase_timing_is_unchanged() {
  const std::vector<ghalo::BenchmarkResult> results{sample_result()};
  ScopedTempDirectory temp;

  std::ostringstream console;
  ghalo::write_console(console, results);
  assert(console.str().find("phase timing") == std::string::npos);

  const auto csv_path = temp.file("ghalo_core_smoke_no_phase.csv");
  ghalo::write_csv(csv_path.string(), results);
  const std::string csv = read_file(csv_path);
  assert(csv.find("phase_input_device_copy_seconds") == std::string::npos);

  const auto json_path = temp.file("ghalo_core_smoke_no_phase.json");
  ghalo::write_json(json_path.string(), results);
  const std::string json = read_file(json_path);
  assert(json.find("\"phase_timing\"") == std::string::npos);
}

void test_output_with_phase_timing() {
  auto result = sample_result();
  ghalo::PhaseTimingResult phase;
  phase.input_device_copy_seconds = 0.001;
  phase.north_south_mpi_seconds = 0.002;
  phase.north_south_sync_seconds = 0.003;
  phase.transpose_device_copy_seconds = 0.004;
  phase.transpose_copy_sync_seconds = 0.005;
  phase.east_west_mpi_seconds = 0.006;
  phase.east_west_sync_seconds = 0.0;
  phase.phase_sum_seconds = ghalo::phase_timing_sum(phase);
  phase.unattributed_seconds =
      result.max_average_seconds - phase.phase_sum_seconds;
  result.phase_timing = phase;
  result.metadata.phase_timing_enabled = true;
  result.metadata.phase_timing_source = "MPI_Wtime";
  result.metadata.phase_timing_aggregation =
      "maximum local average across ranks";

  const std::vector<ghalo::BenchmarkResult> results{result};
  ScopedTempDirectory temp;

  std::ostringstream console;
  ghalo::write_console(console, results);
  assert(console.str().find("MPI-HIP phase timing") != std::string::npos);
  assert(console.str().find("input_copy_us") != std::string::npos);
  assert(console.str().find("total_minus_sum_us") != std::string::npos);

  const auto csv_path = temp.file("ghalo_core_smoke_phase.csv");
  ghalo::write_csv(csv_path.string(), results);
  const std::string csv = read_file(csv_path);
  assert(csv.find("phase_input_device_copy_seconds") != std::string::npos);
  assert(csv.find("phase_unattributed_seconds") != std::string::npos);

  const auto json_path = temp.file("ghalo_core_smoke_phase.json");
  ghalo::write_json(json_path.string(), results);
  const std::string json = read_file(json_path);
  assert(json.find("\"phase_timing\"") != std::string::npos);
  assert(json.find("\"phase_timing_metadata\"") != std::string::npos);
  assert(json.find("\"timing_source\": \"MPI_Wtime\"") != std::string::npos);
  assert(json.find("\"phase_sum_seconds\"") != std::string::npos);
}

void test_output_with_rccl_phase_timing() {
  auto result = sample_result();
  result.backend = "RCCLBackend";
  result.algorithm = "rccl";
  result.metadata.rccl_version = "20400";
  result.metadata.rccl_sync_mode = "conservative";
  result.metadata.synchronization_model = "three_stream_synchronizations";
  ghalo::PhaseTimingResult phase;
  phase.north_south_communication_seconds = 0.002;
  phase.north_south_sync_seconds = 0.001;
  phase.transpose_copy_seconds = 0.003;
  phase.transpose_sync_seconds = 0.001;
  phase.east_west_communication_seconds = 0.004;
  phase.east_west_sync_seconds = 0.002;
  phase.phase_sum_seconds = ghalo::phase_timing_sum(phase);
  phase.total_exchange_seconds = 0.01;
  phase.total_minus_sum_of_phase_maxima_seconds =
      phase.total_exchange_seconds - phase.phase_sum_seconds;
  phase.unattributed_seconds = phase.total_minus_sum_of_phase_maxima_seconds;
  result.phase_timing = phase;
  result.metadata.phase_timing_enabled = true;
  result.metadata.phase_timing_source = "MPI_Wtime";
  result.metadata.phase_timing_aggregation =
      "maximum local average across ranks";

  const std::vector<ghalo::BenchmarkResult> results{result};
  ScopedTempDirectory temp;

  std::ostringstream console;
  ghalo::write_console(console, results);
  assert(console.str().find("RCCL phase timing") != std::string::npos);
  assert(console.str().find("ns_comm_us") != std::string::npos);
  assert(console.str().find("input_copy_us") == std::string::npos);

  const auto csv_path = temp.file("ghalo_core_smoke_rccl_phase.csv");
  ghalo::write_csv(csv_path.string(), results);
  const std::string csv = read_file(csv_path);
  assert(csv.find("phase_north_south_communication_seconds") !=
         std::string::npos);
  assert(csv.find("phase_total_minus_sum_of_phase_maxima_seconds") !=
         std::string::npos);

  const auto json_path = temp.file("ghalo_core_smoke_rccl_phase.json");
  ghalo::write_json(json_path.string(), results);
  const std::string json = read_file(json_path);
  assert(json.find("\"north_south_communication_seconds\"") !=
         std::string::npos);
  assert(json.find("\"total_minus_sum_of_phase_maxima_seconds\"") !=
         std::string::npos);
  assert(json.find("\"synchronization_model\": "
                   "\"three_stream_synchronizations\"") != std::string::npos);
}

void test_output_with_rccl_stream_ordered_phase_timing() {
  auto result = sample_result();
  result.backend = "RCCLBackend";
  result.algorithm = "rccl";
  result.metadata.rccl_version = "20400";
  result.metadata.rccl_sync_mode = "stream-ordered";
  result.metadata.synchronization_model = "one_final_stream_sync";
  result.metadata.phase_timing_active_phases =
      "north_south_communication_enqueue,transpose_copy_enqueue,"
      "east_west_communication_enqueue,final_stream_sync";
  ghalo::PhaseTimingResult phase;
  phase.north_south_communication_enqueue_seconds = 0.0002;
  phase.transpose_copy_enqueue_seconds = 0.0003;
  phase.east_west_communication_enqueue_seconds = 0.0004;
  phase.final_stream_sync_seconds = 0.009;
  phase.phase_sum_seconds = ghalo::phase_timing_sum(phase);
  phase.total_exchange_seconds = 0.01;
  phase.total_minus_sum_of_phase_maxima_seconds =
      phase.total_exchange_seconds - phase.phase_sum_seconds;
  phase.unattributed_seconds = phase.total_minus_sum_of_phase_maxima_seconds;
  result.phase_timing = phase;
  result.metadata.phase_timing_enabled = true;
  result.metadata.phase_timing_source = "MPI_Wtime";
  result.metadata.phase_timing_aggregation =
      "maximum local average across ranks";

  const std::vector<ghalo::BenchmarkResult> results{result};
  ScopedTempDirectory temp;

  std::ostringstream console;
  ghalo::write_console(console, results);
  assert(console.str().find("RCCL sync mode: stream-ordered") !=
         std::string::npos);
  assert(console.str().find("ns_enqueue_us") != std::string::npos);
  assert(console.str().find("ns_sync_us") == std::string::npos);

  const auto json_path = temp.file("ghalo_core_smoke_rccl_stream_phase.json");
  ghalo::write_json(json_path.string(), results);
  const std::string json = read_file(json_path);
  assert(json.find("\"rccl_sync_mode\": \"stream-ordered\"") !=
         std::string::npos);
  assert(json.find("\"final_stream_sync_seconds\"") != std::string::npos);
  assert(json.find("\"active_phases\": "
                   "\"north_south_communication_enqueue") !=
         std::string::npos);
}

} // namespace

int main() {
  test_halo_buffer_layout_helpers();
  test_self_copy_topologies();
  test_north_south_stage_b_conceptual_plans();
  test_full_halo_corner_expectations();
  test_bytes_per_rank_consistency();
  test_phase_sum_calculation();
  test_unsupported_phase_timing();
  test_cli_phase_timing_parse();
  test_development_validation_refusal();
  test_output_without_phase_timing_is_unchanged();
  test_output_with_phase_timing();
  test_output_with_rccl_phase_timing();
  test_output_with_rccl_stream_ordered_phase_timing();

  MockBackend backend;
  ghalo::BenchmarkConfig config;
  config.halo_lengths = {2};
  config.target_seconds = 0.05;
  config.calibration_iterations = 5;

  const auto results = ghalo::run_benchmark(backend, config);

  assert(results.size() == 1);
  assert(results.front().backend == "MockBackend");
  assert(results.front().halo_words == 2);
  assert(results.front().iterations >= config.calibration_iterations);
  assert(results.front().max_average_seconds > 0.0);
  assert(backend.exchanges() >= 11);

  return 0;
}
