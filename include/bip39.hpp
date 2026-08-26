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
    // FASTEST PATH: derive + compare sem alocar string
    // ============================================================
    static bool derive_and_compare(const char* mnemonic_str, size_t mnemonic_len,
                                   const uint8_t* target_bytes, size_t target_len,
                                   int pbkdf2_rounds = 2048) {
        thread_local uint8_t seed[64];
        thread_local uint8_t master_node[64];
        thread_local uint8_t pub_serialized[33];
        thread_local uint8_t hash_buf[32];
        thread_local uint8_t ripemd_buf[20];
        thread_local uint8_t payload[25];
        thread_local uint8_t checksum[32];
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

        // 2. HMAC-SHA512 -> Master Node (BIP32)
        unsigned int md_len = 64;
        HMAC(EVP_sha512(), "Bitcoin seed", 12, seed, 64, master_node, &md_len);

        // 3. SECP256k1 -> Public Key
        secp256k1_pubkey pubkey;
        if (!secp256k1_ec_pubkey_create(ctx, &pubkey, master_node)) {
            return false;
        }
        size_t pub_len = 33;
        secp256k1_ec_pubkey_serialize(ctx, pub_serialized, &pub_len, &pubkey, SECP256K1_EC_COMPRESSED);

        // 4. SHA256(pubkey) + RIPEMD160
        crypto::SHA256::hash(pub_serialized, pub_len, hash_buf);
        crypto::RIPEMD160::hash(hash_buf, 32, ripemd_buf);

        // 5. Payload: version(0x00) + RIPEMD160(20)
        payload[0] = 0x00;
        std::memcpy(payload + 1, ripemd_buf, 20);

        // 6. Double SHA256 for checksum
        crypto::SHA256::hash(payload, 21, checksum);
        crypto::SHA256::hash(checksum, 32, checksum);

        // 7. Append 4-byte checksum
        std::memcpy(payload + 21, checksum, 4);

        // 8. Base58 encode ultra-rápido (Arbitrary UInt<4>)
        size_t b58_len = base58_encode_raw(payload, 25, base58_buf);

        if (b58_len != target_len) return false;
        return std::memcmp(base58_buf, target_bytes, target_len) == 0;
    }

    // Derive BTC address from pre-built mnemonic string
    static std::string derive_btc_address_from_string(const char* mnemonic_str, size_t mnemonic_len) {
        thread_local uint8_t seed[64];
        thread_local uint8_t master_node[64];
        thread_local uint8_t pub_serialized[33];
        thread_local uint8_t hash_buf[32];
        thread_local uint8_t ripemd_buf[20];
        thread_local uint8_t payload[25];
        thread_local uint8_t checksum[32];
        thread_local char result[64];

        thread_local secp256k1_context* ctx = nullptr;
        if (BUILTIN_EXPECT(!ctx, 0)) {
            ctx = secp256k1_context_create(SECP256K1_CONTEXT_SIGN | SECP256K1_CONTEXT_VERIFY);
        }

        static const uint8_t salt[] = "mnemonic";

        crypto::pbkdf2_hmac_sha512(mnemonic_str, mnemonic_len, salt, 8, 2048, seed, 64);

        unsigned int md_len = 64;
        HMAC(EVP_sha512(), "Bitcoin seed", 12, seed, 64, master_node, &md_len);

        secp256k1_pubkey pubkey;
        if (!secp256k1_ec_pubkey_create(ctx, &pubkey, master_node)) return "";

        size_t pub_len = 33;
        secp256k1_ec_pubkey_serialize(ctx, pub_serialized, &pub_len, &pubkey, SECP256K1_EC_COMPRESSED);

        crypto::SHA256::hash(pub_serialized, pub_len, hash_buf);
        crypto::RIPEMD160::hash(hash_buf, 32, ripemd_buf);

        payload[0] = 0x00;
        std::memcpy(payload + 1, ripemd_buf, 20);

        crypto::SHA256::hash(payload, 21, checksum);
        crypto::SHA256::hash(checksum, 32, checksum);

        std::memcpy(payload + 21, checksum, 4);

        size_t len = base58_encode_raw(payload, 25, result);
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

    static std::string derive_eth_address(const std::vector<std::string>& mnemonic_words) {
        // NOTA: Ethereum usa a curva secp256k1 mas exige o hash Keccak-256 (não SHA256)
        // da chave publica descompactada. Caso seu crypto_impl não possua Keccak256,
        // este método não gerará endereços ETH válidos.
        thread_local uint8_t seed[64];
        thread_local uint8_t hash_buf[32];
        thread_local char buf[MAX_MNEMONIC_LEN];
        size_t len = 0;

        for (size_t i = 0; i < mnemonic_words.size(); ++i) {
            const auto& w = mnemonic_words[i];
            std::memcpy(buf + len, w.data(), w.size());
            len += w.size();
            if (i < mnemonic_words.size() - 1) {
                buf[len++] = ' '; // Correção: Espaçamento
            }
        }

        static const uint8_t salt[] = "mnemonic";
        crypto::pbkdf2_hmac_sha512(buf, len, salt, 8, 2048, seed, 64);

        // Cuidado: Isto é tecnicamente invalido para Ethereum (ETH usa Keccak-256 na PubKey)
        // Mantido apenas para não quebrar a assinatura original da sua biblioteca
        crypto::SHA256::hash(seed, 32, hash_buf);

        char hex[43];
        hex[0] = '0'; hex[1] = 'x';
        static const char hextable[] = "0123456789abcdef";
        for (int i = 0; i < 20; ++i) {
            hex[2 + i * 2]     = hextable[(hash_buf[i] >> 4) & 0x0f];
            hex[2 + i * 2 + 1] = hextable[hash_buf[i] & 0x0f];
        }
        return std::string(hex, 42);
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
