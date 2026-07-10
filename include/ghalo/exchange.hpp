#pragma once

namespace ghalo {

inline bool use_local_copy_for_exchange_segment(int local_rank,
                                                int destination_rank,
                                                int source_rank) {
  return destination_rank == local_rank && source_rank == local_rank;
}

} // namespace ghalo
