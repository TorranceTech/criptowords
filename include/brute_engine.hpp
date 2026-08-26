#pragma once
#include <span>
#include <string>
#include <string_view>
#include <vector>

class BruteEngine {
public:
    static void run(
        std::span<const std::string> wordlist,
        std::span<const std::string> base_words,
        std::span<const size_t> unknown_positions,
        std::span<const std::vector<std::string>> candidates,
        std::string_view target,
        int num_threads,
        int pbkdf2_rounds,
        bool use_gpu
    );
};
