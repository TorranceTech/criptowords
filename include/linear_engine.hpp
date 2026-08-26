#pragma once
#include "../include/bip39.hpp"
#include <vector>
#include <string>
#include <functional>

namespace cryptowords {

// Motor de busca linear determinística com derivação incremental
class LinearEngine {
public:
    LinearEngine(const std::vector<std::string>& wordlist,
                 size_t word_count,
                 const std::vector<PositionConstraint>& constraints);

    // Gera intervalo linear [start, end) para um worker
    SearchInterval generate_interval(uint64_t worker_id, uint64_t total_workers) const;

    // Progride para próximo intervalo
    bool advance_interval(SearchInterval& interval) const;

    // Verifica se uma palavra removida contamina o intervalo
    bool is_contaminated(const SearchInterval& interval, size_t position, const std::string& removed_word) const;

private:
    std::vector<std::string> wordlist_;
    size_t word_count_;
    std::vector<PositionConstraint> constraints_;
};

} // namespace cryptowords
