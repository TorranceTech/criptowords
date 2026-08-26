#include "../include/cli_parser.hpp"
#include <iostream>
#include <charconv>
#include <thread>

template <typename T>
static bool parse_int(std::string_view sv, T& out) {
    auto [ptr, ec] = std::from_chars(sv.data(), sv.data() + sv.size(), out);
    return ec == std::errc();
}

static std::map<size_t, std::vector<std::string>> parse_constraint_map(std::string_view input) {
    std::map<size_t, std::vector<std::string>> result;
    size_t start = 0;
    while (start < input.size()) {
        size_t comma_pos = input.find(',', start);
        std::string_view token = input.substr(start, comma_pos - start);
        start = (comma_pos == std::string_view::npos) ? input.size() : comma_pos + 1;

        size_t colon_pos = token.find(':');
        if (colon_pos == std::string_view::npos) continue;

        std::string_view pos_str = token.substr(0, colon_pos);
        size_t pos = 0;
        if (!parse_int(pos_str, pos)) continue;

        std::string_view words_part = token.substr(colon_pos + 1);
        size_t w_start = 0;
        while (w_start < words_part.size()) {
            size_t pipe_pos = words_part.find('|', w_start);
            std::string_view w = words_part.substr(w_start, pipe_pos - w_start);
            w_start = (pipe_pos == std::string_view::npos) ? words_part.size() : pipe_pos + 1;

            while (!w.empty() && w.front() == ' ') w.remove_prefix(1);
            while (!w.empty() && w.back() == ' ') w.remove_suffix(1);
            if (!w.empty()) result[pos].emplace_back(w);
        }
    }
    return result;
}

void CLIParser::print_help() {
    std::cout << "=== CritoWords - Brute Force Mnemonic ===\n\n"
              << "Uso: criptowords --mnemonic \"palavra1 ? ...\" --hash <endereco> [opcoes]\n\n"
              << "Opcoes:\n"
              << "  --mnemonic \"palavras...\"       Palavras (use ? para posicoes desconhecidas)\n"
              << "  --fix \"pos:palavra,...\"         Fixa palavras em posicoes especificas\n"
              << "  --allow \"pos:pal1|pal2,...\"    Palavras permitidas por posicao (use | para separar)\n"
              << "  --lang <idioma>                Idioma da wordlist (padrao: english)\n"
              << "  --wordlist <arquivo>           Arquivo com wordlist customizado\n"
              << "  --words <n>                    Quantidade de palavras (padrao: 12)\n"
              << "  --hash <endereco>              Endereco BTC/ETH para validar\n"
              << "  --threads <n>                  Numero de threads (padrao: todas as CPUs)\n"
              << "  --rounds <n>                   Iteracoes PBKDF2 (padrao: 2048, BIP39)\n"
              << "  --gpu                          Usa GPU via OpenCL (se disponivel)\n"
              << "  --help                         Mostra esta ajuda\n" << std::endl;
}

AppConfig CLIParser::parse(int argc, char* argv[]) {
    AppConfig cfg;
    if (argc < 2) {
        cfg.wants_help = true;
        return cfg;
    }

    for (int i = 1; i < argc; ++i) {
        std::string_view a = argv[i];
        if (a == "--help" || a == "-h") { cfg.wants_help = true; return cfg; }
        else if (a == "--wordlist" && i + 1 < argc) cfg.wordlist_path = argv[++i];
        else if (a == "--words" && i + 1 < argc) parse_int(argv[++i], cfg.words);
        else if (a == "--hash" && i + 1 < argc) cfg.target = argv[++i];
        else if (a == "--mnemonic" && i + 1 < argc) cfg.mnemonic_arg = argv[++i];
        else if (a == "--threads" && i + 1 < argc) parse_int(argv[++i], cfg.num_threads);
        else if (a == "--rounds" && i + 1 < argc) parse_int(argv[++i], cfg.pbkdf2_rounds);
        else if (a == "--gpu") cfg.use_gpu = true;
        else if (a == "--lang" && i + 1 < argc) cfg.lang = argv[++i];
        else if (a == "--fix" && i + 1 < argc) {
            auto parsed = parse_constraint_map(argv[++i]);
            for (const auto& [k, v] : parsed) cfg.fix_map[k] = v;
        }
        else if (a == "--allow" && i + 1 < argc) {
            auto parsed = parse_constraint_map(argv[++i]);
            for (const auto& [k, v] : parsed) {
                auto it = cfg.allow_map.find(k);
                if (it != cfg.allow_map.end()) {
                    it->second.insert(it->second.end(), v.begin(), v.end());
                } else {
                    cfg.allow_map[k] = v;
                }
            }
        }
    }

    if (!cfg.lang.empty()) {
        cfg.wordlist_path = "../wordlist/" + cfg.lang + ".txt";
    }
    if (cfg.num_threads <= 0) {
        cfg.num_threads = std::thread::hardware_concurrency();
        if (cfg.num_threads <= 0) cfg.num_threads = 4;
    }
    return cfg;
}
