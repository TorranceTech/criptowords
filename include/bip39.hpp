#pragma once
#include <string>
#include <vector>
#include <optional>
#include <cstdint>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <secp256k1.h>
#include <openssl/hmac.h>
#include "crypto_impl.hpp"
#include "keccak256.hpp"
#include "Arbitrary/UInt.hpp"

// Define helper para predição de branch se não existir no ambiente global
#ifndef BUILTIN_EXPECT
#define BUILTIN_EXPECT(x, y) (__builtin_expect(!!(x), y))
#endif

namespace cryptowords {

static constexpr size_t MAX_MNEMONIC_LEN = 300;
static constexpr size_t BIP39_WORDLIST_SIZE = 2048;

struct PositionConstraint {
    size_t position;
    std::vector<std::string> allowed_words;
    bool is_fixed = false;
    std::string fixed_word;
};

struct SearchInterval {
    uint64_t start;
    uint64_t end;
    std::vector<std::string> prefix;
    std::vector<PositionConstraint> active_constraints;
};

class Bip39Deriver {
public:
    // ============================================================
    // BIP32 / BIP44 — derivacao de chave filha
    // ============================================================
    // CKDpriv: deriva (k,c) -> filho no indice dado (hardened se bit 31 setado).
    // Atualiza k e c no lugar. Retorna false se a chave resultante for invalida.
    static bool ckd_priv(uint8_t k[32], uint8_t c[32], uint32_t index, secp256k1_context* ctx) {
        uint8_t data[37];
        if (index & 0x80000000u) {
            // Hardened: 0x00 || ser256(k_par) || ser32(i)
            data[0] = 0x00;
            std::memcpy(data + 1, k, 32);
        } else {
            // Normal: serP(point(k_par)) || ser32(i)  (pubkey comprimida, 33 bytes)
            secp256k1_pubkey pub;
            if (!secp256k1_ec_pubkey_create(ctx, &pub, k)) return false;
            size_t l = 33;
            secp256k1_ec_pubkey_serialize(ctx, data, &l, &pub, SECP256K1_EC_COMPRESSED);
        }
        data[33] = static_cast<uint8_t>(index >> 24);
        data[34] = static_cast<uint8_t>(index >> 16);
        data[35] = static_cast<uint8_t>(index >> 8);
        data[36] = static_cast<uint8_t>(index);

        uint8_t I[64];
        unsigned int md_len = 64;
        HMAC(EVP_sha512(), c, 32, data, 37, I, &md_len);

        std::memcpy(c, I + 32, 32);            // chain code filho = IR
        // k_filho = (IL + k_par) mod n  — tweak_add valida (0 se resultado invalido)
        if (!secp256k1_ec_seckey_tweak_add(ctx, k, I)) return false;
        return true;
    }

    // Deriva a chave privada em m/44'/0'/0'/0/0 a partir da seed BIP39 (64 bytes).
    static bool bip44_seed_to_privkey(const uint8_t seed[64], uint8_t out_priv[32], secp256k1_context* ctx) {
        uint8_t I[64];
        unsigned int md_len = 64;
        HMAC(EVP_sha512(), "Bitcoin seed", 12, seed, 64, I, &md_len);
        uint8_t k[32], c[32];
        std::memcpy(k, I, 32);
        std::memcpy(c, I + 32, 32);
        static constexpr uint32_t H = 0x80000000u;
        const uint32_t path[5] = { 44u | H, 0u | H, 0u | H, 0u, 0u };  // m/44'/0'/0'/0/0
        for (int i = 0; i < 5; ++i) {
            if (!ckd_priv(k, c, path[i], ctx)) return false;
        }
        std::memcpy(out_priv, k, 32);
        return true;
    }

    // Chave privada -> endereco P2PKH legacy ("1..."). Retorna comprimento base58 (0 se falhar).
    static size_t privkey_to_p2pkh(const uint8_t priv[32], char* out_buf, secp256k1_context* ctx) {
        secp256k1_pubkey pub;
        if (!secp256k1_ec_pubkey_create(ctx, &pub, priv)) return 0;
        uint8_t ser[33]; size_t l = 33;
        secp256k1_ec_pubkey_serialize(ctx, ser, &l, &pub, SECP256K1_EC_COMPRESSED);

        uint8_t sha[32], rip[20], payload[25], cks[32];
        crypto::SHA256::hash(ser, l, sha);
        crypto::RIPEMD160::hash(sha, 32, rip);
        payload[0] = 0x00;
        std::memcpy(payload + 1, rip, 20);
        crypto::SHA256::hash(payload, 21, cks);
        crypto::SHA256::hash(cks, 32, cks);
        std::memcpy(payload + 21, cks, 4);
        return base58_encode_raw(payload, 25, out_buf);
    }

    // seed BIP39 -> endereco BIP44 m/44'/0'/0'/0/0. Retorna comprimento (0 se falhar).
    static size_t seed_to_p2pkh_bip44(const uint8_t seed[64], char* out_buf, secp256k1_context* ctx) {
        uint8_t priv[32];
        if (!bip44_seed_to_privkey(seed, priv, ctx)) return 0;
        return privkey_to_p2pkh(priv, out_buf, ctx);
    }

    // ============================================================
    // Ethereum: BIP44 m/44'/60'/0'/0/0 + Keccak-256
    // ============================================================
    // Deriva a chave privada em m/44'/60'/0'/0/0 (coin type 60 = ETH) da seed.
    static bool eth_seed_to_privkey(const uint8_t seed[64], uint8_t out_priv[32], secp256k1_context* ctx) {
        uint8_t I[64];
        unsigned int md_len = 64;
        HMAC(EVP_sha512(), "Bitcoin seed", 12, seed, 64, I, &md_len);
        uint8_t k[32], c[32];
        std::memcpy(k, I, 32);
        std::memcpy(c, I + 32, 32);
        static constexpr uint32_t H = 0x80000000u;
        const uint32_t path[5] = { 44u | H, 60u | H, 0u | H, 0u, 0u };  // m/44'/60'/0'/0/0
        for (int i = 0; i < 5; ++i) {
            if (!ckd_priv(k, c, path[i], ctx)) return false;
        }
        std::memcpy(out_priv, k, 32);
        return true;
    }

    // seed BIP39 -> endereco Ethereum "0x" + 40 hex minusculo. Retorna 42 (0 se falhar).
    static size_t seed_to_eth_address(const uint8_t seed[64], char* out_buf, secp256k1_context* ctx) {
        uint8_t priv[32];
        if (!eth_seed_to_privkey(seed, priv, ctx)) return 0;

        secp256k1_pubkey pub;
        if (!secp256k1_ec_pubkey_create(ctx, &pub, priv)) return 0;
        // Pubkey DESCOMPRIMIDA: 0x04 || X(32) || Y(32); Keccak usa os 64 bytes (sem o 0x04).
        uint8_t ser[65]; size_t l = 65;
        secp256k1_ec_pubkey_serialize(ctx, ser, &l, &pub, SECP256K1_EC_UNCOMPRESSED);

        uint8_t h[32];
        crypto::Keccak256::hash(ser + 1, 64, h);   // ultimos 20 bytes = endereco

        static const char hx[] = "0123456789abcdef";
        out_buf[0] = '0'; out_buf[1] = 'x';
        for (int i = 0; i < 20; ++i) {
            out_buf[2 + i * 2]     = hx[(h[12 + i] >> 4) & 0x0f];
            out_buf[2 + i * 2 + 1] = hx[h[12 + i] & 0x0f];
        }
        out_buf[42] = '\0';
        return 42;
    }

    // Detecta se o alvo e um endereco Ethereum (comeca com 0x / 0X).
    static bool target_is_eth(const uint8_t* t, size_t len) {
        return len >= 2 && t[0] == '0' && (t[1] == 'x' || t[1] == 'X');
    }

    // Comparacao de hex ignorando maiusc./minusc. (enderecos ETH usam checksum EIP-55).
    static bool ci_hex_equal(const char* a, const uint8_t* b, size_t len) {
        auto lc = [](unsigned char ch) -> unsigned char {
            return (ch >= 'A' && ch <= 'Z') ? ch + 32 : ch;
        };
        for (size_t i = 0; i < len; ++i)
            if (lc(static_cast<unsigned char>(a[i])) != lc(b[i])) return false;
        return true;
    }

    // seed -> compara com o alvo (BTC 1... ou ETH 0x..., detectado automaticamente).
    static bool seed_matches_target(const uint8_t seed[64], const uint8_t* target,
                                    size_t target_len, secp256k1_context* ctx) {
        char buf[64];
        if (target_is_eth(target, target_len)) {
            size_t l = seed_to_eth_address(seed, buf, ctx);
            return l == target_len && ci_hex_equal(buf, target, l);
        }
        size_t l = seed_to_p2pkh_bip44(seed, buf, ctx);
        return l == target_len && std::memcmp(buf, target, l) == 0;
    }

    // ============================================================
    // FASTEST PATH: derive + compare sem alocar string
    // ============================================================
    static bool derive_and_compare(const char* mnemonic_str, size_t mnemonic_len,
                                   const uint8_t* target_bytes, size_t target_len,
                                   int pbkdf2_rounds = 2048) {
        thread_local uint8_t seed[64];
        thread_local char base58_buf[64];

        // Cacheia o contexto SECP256k1 localmente por thread para máxima performance
        thread_local secp256k1_context* ctx = nullptr;
        if (BUILTIN_EXPECT(!ctx, 0)) {
            ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
        }

        static const uint8_t salt[] = "mnemonic";

        // 1. PBKDF2 -> seed
        crypto::pbkdf2_hmac_sha512(mnemonic_str, mnemonic_len,
                                   salt, 8,
                                   pbkdf2_rounds, seed, 64);

        // 2. seed -> compara com o alvo (BTC m/44'/0'/0'/0/0 ou ETH m/44'/60'/0'/0/0,
        //    detectado automaticamente pelo prefixo 0x).
        (void)base58_buf;
        return seed_matches_target(seed, target_bytes, target_len, ctx);
    }

    // Derive BTC address from pre-built mnemonic string
    static std::string derive_btc_address_from_string(const char* mnemonic_str, size_t mnemonic_len) {
        thread_local uint8_t seed[64];
        thread_local char result[64];

        thread_local secp256k1_context* ctx = nullptr;
        if (BUILTIN_EXPECT(!ctx, 0)) {
            ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
        }

        static const uint8_t salt[] = "mnemonic";

        crypto::pbkdf2_hmac_sha512(mnemonic_str, mnemonic_len, salt, 8, 2048, seed, 64);

        // seed -> endereco BIP44 (m/44'/0'/0'/0/0) P2PKH legacy
        size_t len = seed_to_p2pkh_bip44(seed, result, ctx);
        return std::string(result, len);
    }

    // Interface original (para testes)
    static std::string derive_btc_address(const std::vector<std::string>& mnemonic_words) {
        thread_local char buf[MAX_MNEMONIC_LEN];
        size_t len = 0;
        for (size_t i = 0; i < mnemonic_words.size(); ++i) {
            const auto& w = mnemonic_words[i];
            std::memcpy(buf + len, w.data(), w.size());
            len += w.size();
            // Correção: Adicionar espaço obrigatorio do BIP39!
            if (i < mnemonic_words.size() - 1) {
                buf[len++] = ' ';
            }
        }
        return derive_btc_address_from_string(buf, len);
    }

    // Deriva o endereco Ethereum (m/44'/60'/0'/0/0 + Keccak-256) de uma string mnemonic.
    static std::string derive_eth_address_from_string(const char* mnemonic_str, size_t mnemonic_len) {
        thread_local uint8_t seed[64];
        thread_local char result[64];
        thread_local secp256k1_context* ctx = nullptr;
        if (BUILTIN_EXPECT(!ctx, 0)) {
            ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
        }
        static const uint8_t salt[] = "mnemonic";
        crypto::pbkdf2_hmac_sha512(mnemonic_str, mnemonic_len, salt, 8, 2048, seed, 64);
        size_t len = seed_to_eth_address(seed, result, ctx);
        return std::string(result, len);
    }

    static std::string derive_eth_address(const std::vector<std::string>& mnemonic_words) {
        thread_local char buf[MAX_MNEMONIC_LEN];
        size_t len = 0;
        for (size_t i = 0; i < mnemonic_words.size(); ++i) {
            const auto& w = mnemonic_words[i];
            std::memcpy(buf + len, w.data(), w.size());
            len += w.size();
            if (i < mnemonic_words.size() - 1) buf[len++] = ' ';
        }
        return derive_eth_address_from_string(buf, len);
    }

public:
    // Tabela oficial do Bitcoin
    constexpr static const char* BASE58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";

    // Codificador hiper-otimizado usando Arbitrary UInt<4>
    static inline size_t base58_encode_raw(const uint8_t* payload, size_t len, char* out_buf) {
        size_t leading_zeros = 0;
        while (leading_zeros < len && payload[leading_zeros] == 0x00) {
            leading_zeros++;
        }

        UInt<4> num = 0;
        num.bits.fill(0);

        for (size_t i = 0; i < len; ++i) {
            num *= 256;
            num += static_cast<uint64_t>(payload[i]);
        }

        char temp_buf[64];
        size_t temp_len = 0;

        while (!num.eqz()) {
            unsigned __int128 rem = 0;
            for (int i = 3; i >= 0; --i) {
                unsigned __int128 cur = static_cast<unsigned __int128>(num.bits[i]) | (rem << 64);
                num.bits[i] = static_cast<uint64_t>(cur / 58);
                rem = cur % 58;
            }
            temp_buf[temp_len++] = BASE58_ALPHABET[static_cast<size_t>(rem)];
        }

        for (size_t i = 0; i < leading_zeros; ++i) {
            temp_buf[temp_len++] = BASE58_ALPHABET[0];
        }

        for (size_t i = 0; i < temp_len; ++i) {
            out_buf[i] = temp_buf[temp_len - 1 - i];
        }
        out_buf[temp_len] = '\0';

        return temp_len;
    }
};

class HashValidator {
public:
    static bool validate_target(const std::string& derived_hash, const std::string& target) {
        if (derived_hash.size() != target.size()) return false;
        return derived_hash == target;
    }
};

struct TargetBytes {
    uint8_t data[80];
    size_t len;

    TargetBytes() : len(0) {}

    bool init(const std::string& target) {
        if (target.size() > sizeof(data)) return false;
        len = target.size();
        std::memcpy(data, target.c_str(), len);
        return true;
    }
};

} // namespace cryptowords
