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

class Backend {
public:
  virtual ~Backend() = default;

  virtual std::string name() const = 0;
  virtual std::string algorithm() const = 0;
  virtual TopologyInfo topology() const = 0;

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
