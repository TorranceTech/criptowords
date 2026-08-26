#include "checkpoint_manager.hpp"
#include <fstream>

namespace cryptowords {

CheckpointManager::CheckpointManager(const std::string& checkpoint_path)
    : path_(checkpoint_path) {}

void CheckpointManager::save_interval(const SearchInterval& interval) {
    std::ofstream out(path_, std::ios::app);
    out << interval.start << " " << interval.end << "\n";
}

std::vector<SearchInterval> CheckpointManager::load_intervals() const {
    std::vector<SearchInterval> result;
    std::ifstream in(path_);
    uint64_t s, e;
    while (in >> s >> e) {
        result.push_back({s, e, {}, {}});
    }
    return result;
}

std::vector<SearchInterval> CheckpointManager::filter_valid_intervals(
    const std::vector<SearchInterval>& intervals,
    size_t position,
    const std::string& removed_word,
    const std::vector<PositionConstraint>& current_constraints) const {
    std::vector<SearchInterval> valid;
    for (const auto& interval : intervals) {
        bool contaminated = false;
        for (const auto& constraint : interval.active_constraints) {
            if (constraint.position == position && !constraint.is_fixed) {
                for (const auto& w : constraint.allowed_words) {
                    if (w == removed_word) {
                        contaminated = true;
                        break;
                    }
                }
            }
        }
        if (!contaminated) valid.push_back(interval);
    }
    return valid;
}

} // namespace cryptowords
