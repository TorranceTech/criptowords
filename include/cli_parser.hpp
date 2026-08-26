#pragma once
#include <string>
#include <string_view>
#include <vector>
#include <map>

struct AppConfig {
    std::string wordlist_path = "../wordlist/english.txt";
    std::string target = "";
    std::string mnemonic_arg = "";
    std::string lang = "";
    size_t words = 12;
    int pbkdf2_rounds = 2048;
    int num_threads = 0;
    bool use_gpu = false;
    std::map<size_t, std::vector<std::string>> fix_map;
    std::map<size_t, std::vector<std::string>> allow_map;
    bool wants_help = false;
    bool has_errors = false;
};

class CLIParser {
public:
    static AppConfig parse(int argc, char* argv[]);
    static void print_help();
};
