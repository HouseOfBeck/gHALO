#include "ghalo/benchmark.hpp"
#include "ghalo/cli.hpp"
#include "ghalo/exchange.hpp"
#include "ghalo/output.hpp"

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

void test_self_copy_topologies() {
  check_self_copy_decisions(topology_for(1, 1, 0), true, true);

  check_self_copy_decisions(topology_for(1, 2, 0), true, false);
  check_self_copy_decisions(topology_for(1, 2, 1), true, false);

  for (int rank = 0; rank < 4; ++rank) {
    check_self_copy_decisions(topology_for(2, 2, rank), false, false);
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
  assert(console.str().find("unattributed_us") != std::string::npos);

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

} // namespace

int main() {
  test_self_copy_topologies();
  test_phase_sum_calculation();
  test_unsupported_phase_timing();
  test_cli_phase_timing_parse();
  test_output_without_phase_timing_is_unchanged();
  test_output_with_phase_timing();

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
