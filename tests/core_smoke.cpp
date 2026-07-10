#include "ghalo/benchmark.hpp"

#include <cassert>
#include <cstddef>
#include <string>

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

} // namespace

int main() {
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
