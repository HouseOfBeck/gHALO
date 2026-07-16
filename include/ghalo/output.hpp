#pragma once

#include "ghalo/benchmark.hpp"

#include <iosfwd>
#include <string>
#include <vector>

namespace ghalo {

void write_console(std::ostream& out,
                   const std::vector<BenchmarkResult>& results);

void write_csv(const std::string& path,
               const std::vector<BenchmarkResult>& results);

void write_json(const std::string& path,
                const std::vector<BenchmarkResult>& results);

void write_iteration_times_csv(const std::string& path,
                               const std::vector<BenchmarkResult>& results);

void write_iteration_phase_times_csv(
    const std::string& path, const std::vector<BenchmarkResult>& results);

void write_stalled_rank_times_csv(
    const std::string& path, const std::vector<BenchmarkResult>& results);

} // namespace ghalo
