#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cstring>
#include "../include/cli_parser.hpp"
#include "../include/brute_engine.hpp"
#include "../include/bip39.hpp"

// Função auxiliar para concatenar as palavras com espaçamento BIP39
static inline void build_mnemonic_fast(const std::vector<std::string>& words, char* buf, size_t& len) {
    len = 0;
    for (size_t i = 0; i < words.size(); ++i) {
        std::memcpy(buf + len, words[i].data(), words[i].size());
        len += words[i].size();
        if (i < words.size() - 1) {
            buf[len] = ' ';
            len++;
        }
    }
}

int main_cli(int argc, char* argv[]) {
    // 1. Parsing da Linha de Comando
    AppConfig cfg = CLIParser::parse(argc, argv);
    if (cfg.wants_help || cfg.has_errors) {
        CLIParser::print_help();
        return cfg.has_errors ? 1 : 0;
    }

    // 2. Carregar Wordlist
    std::vector<std::string> wl;
    {
        std::ifstream f(cfg.wordlist_path);
        if (!f.is_open()) {
            std::cerr << "Erro: Arquivo de wordlist não encontrado: " << cfg.wordlist_path << std::endl;
            return 1;
        }
        std::string line;
        while (std::getline(f, line) && !line.empty()) wl.push_back(line);
    }
    if (wl.empty()) return 1;

    // 3. Processar Mnemonic
    std::vector<std::string> mn;
    if (!cfg.mnemonic_arg.empty()) {
        size_t start = 0;
        while (start < cfg.mnemonic_arg.size()) {
            size_t space_pos = cfg.mnemonic_arg.find(' ', start);
            std::string_view w = std::string_view(cfg.mnemonic_arg).substr(start, space_pos - start);
            start = (space_pos == std::string_view::npos) ? cfg.mnemonic_arg.size() : space_pos + 1;
            if (!w.empty()) mn.emplace_back(std::string(w));
        }
    }
    if (mn.empty()) return 1;

    // 4. Aplicar Regras de Fix/Allow
    for (auto& [p, ws] : cfg.fix_map) {
        if (p < mn.size() && !ws.empty()) mn[p] = ws[0];
    }
    for (auto& [p, ws] : cfg.allow_map) {
        if (p < mn.size() && mn[p] != "?") {
            bool ok = std::ranges::any_of(ws, [&](const std::string& w){ return w == mn[p]; });
            if (!ok) mn[p] = "?";
        }
    }

    // 5. Geração Direta (Sem Target) -> Foi o que faltou!
    if (cfg.target.empty()) {
        char buf[300];
        size_t blen = 0;
        build_mnemonic_fast(mn, buf, blen);
        std::cout << "BTC: " << cryptowords::Bip39Deriver::derive_btc_address_from_string(buf, blen) << std::endl;
        std::cout << "ETH: " << cryptowords::Bip39Deriver::derive_eth_address(mn) << std::endl;
        return 0;
    }

    // 6. Mapear Posições Desconhecidas (?)
    std::vector<size_t> up;
    for (size_t i = 0; i < mn.size(); ++i) {
        if (mn[i] == "?") up.push_back(i);
    }

    // 7. Validação Instantânea se o Mnemonic já estiver completo
    if (up.empty()) {
        char buf[300];
        size_t blen = 0;
        build_mnemonic_fast(mn, buf, blen);
        bool eth = cfg.target.size() >= 2 && cfg.target[0] == '0'
                   && (cfg.target[1] == 'x' || cfg.target[1] == 'X');
        std::string addr = eth
            ? cryptowords::Bip39Deriver::derive_eth_address_from_string(buf, blen)
            : cryptowords::Bip39Deriver::derive_btc_address_from_string(buf, blen);
        std::cout << (eth ? "ETH: " : "BTC: ") << addr << std::endl;

        // Compara pelo endereco derivado (case-insensitive para ETH).
        bool ok = (addr.size() == cfg.target.size());
        if (ok) {
            for (size_t i = 0; i < addr.size() && ok; ++i) {
                char a = addr[i], b = cfg.target[i];
                if (eth) { if (a >= 'A' && a <= 'Z') a += 32; if (b >= 'A' && b <= 'Z') b += 32; }
                if (a != b) ok = false;
            }
        }
        std::cout << (ok ? "MATCH!" : "NO MATCH") << std::endl;
        return 0;
    }

    // 8. Montar Dicionários Candidatos
    std::vector<std::vector<std::string>> cands;
    for (size_t p : up) {
        auto it = cfg.allow_map.find(p);
        cands.push_back(it != cfg.allow_map.end() ? it->second : wl);
    }

    std::cout << "=== CritoWords - Brute Force ===\n\n";
    std::cout << "Conhecidas: " << mn.size() - up.size() << "/" << mn.size() << " | Descobrir: ";
    for (size_t p : up) std::cout << p << " ";
    std::cout << "\nTarget: " << cfg.target << "\n\n";

    // 9. Delegar ao Motor
    BruteEngine::run(wl, mn, up, cands, cfg.target, cfg.num_threads, cfg.pbkdf2_rounds, cfg.use_gpu);

    return 0;
}
