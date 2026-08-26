#pragma once
#include "../include/bip39.hpp"
#include <vector>
#include <string>

namespace cryptowords {

// Filtra o espaço de busca com base nas restrições
class PositionFilter {
public:
    static std::vector<std::string> filter_wordlist_for_position(
        const std::vector<std::string>& full_wordlist,
        const std::vector<PositionConstraint>& constraints,
        size_t position);

    static bool is_word_valid_for_interval(
        const std::string& word,
        size_t position,
        const SearchInterval& interval);
};

} // namespace cryptowords
