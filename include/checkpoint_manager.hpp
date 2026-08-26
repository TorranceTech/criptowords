#pragma once
#include "linear_engine.hpp"
#include <vector>
#include <fstream>
#include <string>

namespace cryptowords {

// Gerencia checkpoints compactos de intervalos
class CheckpointManager {
public:
    CheckpointManager(const std::string& checkpoint_path);

    // Salva intervalos concluídos
    void save_interval(const SearchInterval& interval);

    // Lê todos os checkpoints
    std::vector<SearchInterval> load_intervals() const;

    // Analisa se um intervalo pode ser reaproveitado após remoção de palavra
    std::vector<SearchInterval> filter_valid_intervals(
        const std::vector<SearchInterval>& intervals,
        size_t position,
        const std::string& removed_word,
        const std::vector<PositionConstraint>& current_constraints) const;

private:
    std::string path_;
};

} // namespace cryptowords
