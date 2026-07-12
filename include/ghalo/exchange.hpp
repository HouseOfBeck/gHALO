#pragma once

#include <cstddef>

namespace ghalo {

inline bool use_local_copy_for_exchange_segment(int local_rank,
                                                int destination_rank,
                                                int source_rank) {
  return destination_rank == local_rank && source_rank == local_rank;
}

inline std::size_t halo_buffer_words(std::size_t halo_words) {
  return 3 * halo_words;
}

inline std::size_t halo_n_offset() {
  return 0;
}

inline std::size_t halo_two_n_offset(std::size_t halo_words) {
  return halo_words;
}

} // namespace ghalo
