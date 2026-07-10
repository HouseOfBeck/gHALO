#pragma once

#include <cstddef>
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
  std::string memory_location = "host";
  std::string device_map;
  bool gpu_aware_mpi_requested = false;
  bool validation_enabled = false;
  bool validation_passed = false;
  std::vector<RankMetadata> ranks;
};

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
  virtual void exchange() = 0;
  virtual void barrier() = 0;
  virtual double now() const = 0;
  virtual double max_time(double local_seconds) = 0;
};

} // namespace ghalo
