#include "linear_engine.hpp"
#include <cmath>

namespace cryptowords {

LinearEngine::LinearEngine(const std::vector<std::string>& wordlist,
                             size_t word_count,
                             const std::vector<PositionConstraint>& constraints)
    : wordlist_(wordlist), word_count_(word_count), constraints_(constraints) {}

SearchInterval LinearEngine::generate_interval(uint64_t worker_id, uint64_t total_workers) const {
    uint64_t total_combinations = 1;
    for (size_t i = 0; i < word_count_; ++i) total_combinations *= wordlist_.size();

    uint64_t chunk_size = total_combinations / total_workers;
    uint64_t start = worker_id * chunk_size;
    uint64_t end = (worker_id == total_workers - 1) ? total_combinations : start + chunk_size;

    return {start, end, {}, constraints_};
}

bool LinearEngine::advance_interval(SearchInterval& interval) const {
    interval.start += 1;
    return interval.start < interval.end;
}

bool LinearEngine::is_contaminated(const SearchInterval& interval, size_t position, const std::string& removed_word) const {
    for (const auto& c : interval.active_constraints) {
        if (c.position == position && c.is_fixed && c.fixed_word == removed_word) return true;
    }
    return false;
}

} // namespace cryptowords
