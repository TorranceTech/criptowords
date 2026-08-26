#include "../include/brute_engine.hpp"
#include "../include/bip39.hpp"
#include "../include/gpu_engine.hpp"
#include "../include/Arbitrary/UInt.hpp"
#include <iostream>
#include <atomic>
#include <thread>
#include <mutex>
#include <cstring>
#include <iomanip>
#include <secp256k1.h>
#include <openssl/hmac.h>

static std::mutex cout_mutex;

static inline void build_mnemonic_fast(std::span<const std::string> words, char* buf, size_t& len) {
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

[[nodiscard]] static inline bool advance_odometer(std::span<size_t> idx, std::span<const std::vector<std::string>> candidates) {
    for (int i = static_cast<int>(idx.size()) - 1; i >= 0; --i) {
        idx[i]++;
        if (idx[i] < candidates[i].size()) return false;
        idx[i] = 0;
    }
    return true;
}

void BruteEngine::run(
    std::span<const std::string> wordlist,
    std::span<const std::string> base_words,
    std::span<const size_t> unknown_positions,
    std::span<const std::vector<std::string>> candidates,
    std::string_view target,
    int num_threads,
    int pbkdf2_rounds,
    bool use_gpu)
{
    std::atomic<bool> found(false);
    std::atomic<uint64_t> total_attempts(0);

    UInt<255> total_combinations = 1;
    for (const auto& c : candidates) total_combinations = total_combinations * c.size();

    UInt<255> per_thread = (total_combinations + num_threads - 1) / num_threads;

    std::cout << "Configuracao:\n"
              << "  - Wordlist: " << wordlist.size() << " | Descobrir: " << unknown_positions.size() << " pos\n"
              << "  - Combinacoes: " << total_combinations << " | Threads: " << num_threads << std::endl;

    cryptowords::TargetBytes target_bytes;
    target_bytes.init(std::string(target));

    auto start_time = std::chrono::high_resolution_clock::now();

    // --- GPU PATH ---
    if (use_gpu) {
        gpu::Engine gpu_engine;
        if (gpu_engine.init("pbkdf2_hmac512.cl")) {
            std::cout << "  - GPU: " << gpu_engine.device_name() << " (OpenCL)\n";

            const int GPU_BATCH = 256;
            std::vector<std::string> batch_mnemonics;
            std::vector<std::vector<std::string>> batch_words;
            std::vector<size_t> idx(unknown_positions.size(), 0);
            UInt<255> processed = 0;

            secp256k1_context* ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);

            while (!found.load(std::memory_order_relaxed) && processed < total_combinations) {
                batch_mnemonics.clear();
                batch_words.clear();

                for (int b = 0; b < GPU_BATCH && processed + b < total_combinations; b++) {
                    std::vector<std::string> current_words(base_words.begin(), base_words.end());
                    for (size_t vi = 0; vi < unknown_positions.size(); vi++) {
                        current_words[unknown_positions[vi]] = candidates[vi][idx[vi]];
                    }

                    char buf[300];
                    size_t blen = 0;
                    build_mnemonic_fast(current_words, buf, blen);
                    batch_mnemonics.emplace_back(buf, blen);
                    batch_words.push_back(std::move(current_words));

                    (void)advance_odometer(idx, candidates);
                }

                std::vector<std::vector<uint8_t>> seeds;
                if (!gpu_engine.pbkdf2_batch(batch_mnemonics, pbkdf2_rounds, seeds)) {
                    std::cerr << "  - GPU batch falhou, retornando processamento para a CPU...\n";
                    use_gpu = false;
                    break;
                }

                for (size_t i = 0; i < seeds.size() && !found.load(std::memory_order_relaxed); i++) {
                    uint8_t master_node[64];
                    unsigned int md_len = 64;
                    HMAC(EVP_sha512(), "Bitcoin seed", 12, seeds[i].data(), 64, master_node, &md_len);

                    secp256k1_pubkey pubkey;
                    if (!secp256k1_ec_pubkey_create(ctx, &pubkey, master_node)) continue;

                    uint8_t pub_serialized[33];
                    size_t pub_len = 33;
                    secp256k1_ec_pubkey_serialize(ctx, pub_serialized, &pub_len, &pubkey, SECP256K1_EC_COMPRESSED);

                    uint8_t hash_buf[32], ripemd_buf[20], payload[25], checksum[32];
                    char base58_buf[64];

                    crypto::SHA256::hash(pub_serialized, pub_len, hash_buf);
                    crypto::RIPEMD160::hash(hash_buf, 32, ripemd_buf);

                    payload[0] = 0x00;
                    std::memcpy(payload + 1, ripemd_buf, 20);
                    crypto::SHA256::hash(payload, 21, checksum);
                    crypto::SHA256::hash(checksum, 32, checksum);
                    std::memcpy(payload + 21, checksum, 4);

                    size_t b58_len = cryptowords::Bip39Deriver::base58_encode_raw(payload, 25, base58_buf);

                    if (b58_len == target_bytes.len && std::memcmp(base58_buf, target_bytes.data, target_bytes.len) == 0) {
                        found.store(true, std::memory_order_relaxed);
                        std::lock_guard<std::mutex> lk(cout_mutex);
                        std::cout << "\n  MATCH FOUND (GPU)!\n  Mnemonic: ";
                        for (const auto& w : batch_words[i]) std::cout << w << " ";
                        std::cout << std::endl;
                    }
                    ++processed;
                    total_attempts.fetch_add(1, std::memory_order_relaxed);
                }
            }

            secp256k1_context_destroy(ctx);

            if (use_gpu) {
                auto end = std::chrono::high_resolution_clock::now();
                double secs = std::chrono::duration<double>(end - start_time).count();
                std::cout << "\nBusca concluida (GPU). Total: " << total_attempts.load()
                          << " | Tempo: " << std::fixed << std::setprecision(2) << secs << "s"
                          << " | Velocidade: " << std::setprecision(0) << (total_attempts.load() / secs) << " tentativas/s\n";
                if (!found.load()) std::cout << "Nenhum match encontrado.\n";
                return;
            }
        } else {
            std::cout << "  - GPU indisponivel (Fall-back para CPU)\n";
        }
    }

    // --- CPU PATH ---
    std::vector<std::jthread> workers;
    workers.reserve(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        workers.emplace_back([&, t]() {
            std::vector<std::string> current_words(base_words.begin(), base_words.end());
            std::vector<size_t> idx(unknown_positions.size(), 0);

            UInt<255> start_pos = per_thread * t;
            UInt<255> tmp_pos = start_pos;

            for (int i = (int)idx.size() - 1; i >= 0; --i) {
                idx[i] = (tmp_pos % candidates[i].size()).bits[0];
                tmp_pos = tmp_pos / candidates[i].size();
            }

            char mnemonic_buf[300];
            size_t mnemonic_len = 0;
            UInt<255> local_attempts = 0;
            UInt<255> local_limit = per_thread;

            if (total_combinations - start_pos < local_limit) {
                local_limit = total_combinations - start_pos;
            }

            while (!found.load(std::memory_order_relaxed) && local_attempts < local_limit) {
                for (size_t vi = 0; vi < unknown_positions.size(); ++vi) {
                    current_words[unknown_positions[vi]] = candidates[vi][idx[vi]];
                }

                build_mnemonic_fast(current_words, mnemonic_buf, mnemonic_len);

                if (cryptowords::Bip39Deriver::derive_and_compare(mnemonic_buf, mnemonic_len, target_bytes.data, target_bytes.len, pbkdf2_rounds)) {
                    found.store(true, std::memory_order_relaxed);
                    std::string btc = cryptowords::Bip39Deriver::derive_btc_address_from_string(mnemonic_buf, mnemonic_len);

                    std::lock_guard<std::mutex> lk(cout_mutex);
                    std::cout << "\n  MATCH FOUND by thread " << t << "!\n  Mnemonic: ";
                    for (const auto& w : current_words) std::cout << w << " ";
                    std::cout << "\n  BTC: " << btc << std::endl;
                    return;
                }

                local_attempts += 1;
                total_attempts.fetch_add(1, std::memory_order_relaxed);

                if (advance_odometer(idx, candidates)) break;
            }
        });
    }

    workers.clear();

    auto end = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(end - start_time).count();
    std::cout << "\nBusca concluida. Total: " << total_attempts.load()
              << " | Tempo: " << std::fixed << std::setprecision(2) << secs << "s"
              << " | Velocidade: " << std::setprecision(0) << (total_attempts.load() / secs) << " tentativas/s\n";
    if (!found.load()) std::cout << "Nenhum match encontrado.\n";
}
