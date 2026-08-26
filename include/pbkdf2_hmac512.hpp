#pragma once
// =====================================================================
// PBKDF2-HMAC-SHA512 — Host-side preparation for OpenCL kernel
// =====================================================================
// Prepara os estados SHA-512 pre-computados (ipad_state, opad_state)
// e o salt com padding para enviar ao kernel OpenCL otimizado.
//
// O kernel recebe:
//   - ipad_state[n * 64]: estado SHA-512 apos processar (pass XOR ipad || padding)
//   - opad_state[n * 64]: estado SHA-512 apos processar (pass XOR opad || padding)
//   - salt_padded: salt + INT(1) + padding SHA-512 (multiplo de 128 bytes)
//   - salt_blocks: numero de blocos de 128 bytes
// =====================================================================

#include <cstdint>
#include <cstring>
#include <vector>
#include "crypto_impl.hpp"

namespace pbkdf2 {

// SHA-512 IVs
static constexpr uint64_t SHA512_IV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL
};

// =====================================================================
// compute_hmac_states: calcula ipad_state e opad_state para uma senha
// =====================================================================
// ipad_state = SHA512 Estado apos processar (password XOR 0x36 || padding)
// opad_state = SHA512 Estado apos processar (password XOR 0x5C || padding)
//
// Se a senha for > 128 bytes, ela e hashada com SHA-512 primeiro.
// O resultado (64 bytes) e then preenchido com zeros ate 128 bytes.
//
// ipad_state[0..7] e opad_state[0..7] sao os 8 registradores a..h
// do SHA-512 apos processar o bloco de 128 bytes.
// =====================================================================
inline void compute_hmac_states(
    const uint8_t* password, size_t pass_len,
    uint64_t ipad_state[8], uint64_t opad_state[8])
{
    // 1. Se senha > 128, hashar com SHA-512
    uint8_t key_buf[64];
    const uint8_t* key = password;
    size_t key_len = pass_len;

    if (pass_len > 128) {
        crypto::SHA512::hash(password, pass_len, key_buf);
        key = key_buf;
        key_len = 64;
    }

    // 2. Construir ipad e opad (128 bytes cada)
    uint8_t ipad[128], opad[128];
    for (size_t i = 0; i < 128; i++) {
        ipad[i] = 0x36;
        opad[i] = 0x5c;
    }
    for (size_t i = 0; i < key_len; i++) {
        ipad[i] ^= key[i];
        opad[i] ^= key[i];
    }

    // 3. Processar bloco de 128 bytes no SHA-512 (exatamente 1 bloco)
    // Usamos a classe SHA512 do crypto_impl.hpp para processar
    // e extrair o estado interno
    {
        crypto::SHA512 sha;
        sha.update(ipad, 128);
        // Extrair estado interno apos processamento do bloco
        // SHA512: apos update(128 bytes), o estado contem os 8 registradores
        // mas a classe nao expoe diretamente. Usamos finalize para obter hash.
        // Para o kernel, precisamos do ESTADO (não do hash final).
        //
        // A solucao e processar manualmente usando a mesma logica do kernel.
        // Vamos reimplementar aqui para obter os registradores brutos.
    }

    // Processar manualmente para obter registradores brutos
    auto process_block_state = [](const uint8_t block[128], uint64_t state[8]) {
        // SHA-512 K constants
        static constexpr uint64_t K[80] = {
            0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
            0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
            0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
            0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
            0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
            0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
            0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
            0xc6e00bf33da88fc2ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
            0x27b70a8546d22ffcULL, 0x2e1b21385c26c926ULL, 0x4d2c6dfc5ac42aedULL, 0x53380d139d95b3dfULL,
            0x650a73548baf63deULL, 0x766a0abb3c77b2a8ULL, 0x81c2c92e47edaee6ULL, 0x92722c851482353bULL,
            0xa2bfe8a14cf10364ULL, 0xa81a664bbc423001ULL, 0xc24b8b70d0f89791ULL, 0xc76c51a30654be30ULL,
            0xd192e819d6ef5218ULL, 0xd69906245565a910ULL, 0xf40e35855771202aULL, 0x106aa07032bbd1b8ULL,
            0x19a4c116b8d2d0c8ULL, 0x1e376c085141ab53ULL, 0x2748774cdf8eeb99ULL, 0x34b0bcb5e19b48a8ULL,
            0x391c0cb3c5c95a63ULL, 0x4ed8aa4ae3418acbULL, 0x5b9cca4f7763e373ULL, 0x682e6ff3d6b2b8a3ULL,
            0x748f82ee5defb2fcULL, 0x78a5636f43172f60ULL, 0x84c87814a1f0ab72ULL, 0x8cc702081a6439ecULL,
            0x90befffa23631e28ULL, 0xa4506cebde82bde9ULL, 0xbef9a3f7b2c67915ULL, 0xc67178f2e372532bULL,
            0xca273eceea26619cULL, 0xd186b8c721c0c207ULL, 0xeada7dd6cde0eb1eULL, 0xf57d4f7fee6ed178ULL,
            0x06f067aa72176fbaULL, 0x0a637dc5a2c898a6ULL, 0x113f9804bef90daeULL, 0x1b710b35131c471bULL,
            0x28db77f523047d84ULL, 0x32caab7b40c72493ULL, 0x3c9ebe0a15c9bebcULL, 0x431d67c49c100d4cULL,
            0x4cc5d4becb3e42b6ULL, 0x597f299cfc657e2aULL, 0x5fcb6fab3ad6faecULL, 0x6c44198c4a475817ULL
        };

        auto rotr = [](uint64_t x, int n) { return (x >> n) | (x << (64 - n)); };

        // Load message words
        uint64_t W[80];
        for (int i = 0; i < 16; i++) {
            int o = i * 8;
            W[i] = ((uint64_t)block[o] << 56) | ((uint64_t)block[o+1] << 48)
                 | ((uint64_t)block[o+2] << 40) | ((uint64_t)block[o+3] << 32)
                 | ((uint64_t)block[o+4] << 24) | ((uint64_t)block[o+5] << 16)
                 | ((uint64_t)block[o+6] << 8) | (uint64_t)block[o+7];
        }

        // Extend
        for (int i = 16; i < 80; i++) {
            uint64_t s0 = rotr(W[i-15], 1) ^ rotr(W[i-15], 8) ^ (W[i-15] >> 7);
            uint64_t s1 = rotr(W[i-2], 19) ^ rotr(W[i-2], 61) ^ (W[i-2] >> 6);
            W[i] = s1 + W[i-7] + s0 + W[i-16];
        }

        uint64_t a = state[0], b = state[1], c = state[2], d = state[3];
        uint64_t e = state[4], f = state[5], g = state[6], h = state[7];

        for (int i = 0; i < 80; i++) {
            uint64_t S1 = rotr(e, 14) ^ rotr(e, 18) ^ rotr(e, 41);
            uint64_t ch = (e & f) ^ (~e & g);
            uint64_t T1 = h + S1 + ch + K[i] + W[i];
            uint64_t S0 = rotr(a, 28) ^ rotr(a, 34) ^ rotr(a, 39);
            uint64_t mj = (a & b) ^ (a & c) ^ (b & c);
            uint64_t T2 = S0 + mj;

            h = g; g = f; f = e; e = d + T1;
            d = c; c = b; b = a; a = T1 + T2;
        }

        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;
    };

    // Processar ipad (128 bytes = 1 bloco) a partir dos IVs
    for (int i = 0; i < 8; i++) ipad_state[i] = SHA512_IV[i];
    process_block_state(ipad, ipad_state);

    // Processar opad (128 bytes = 1 bloco) a partir dos IVs
    for (int i = 0; i < 8; i++) opad_state[i] = SHA512_IV[i];
    process_block_state(opad, opad_state);
}

// =====================================================================
// prepare_salt_padded: prepara salt + INT(1) + padding SHA-512
// =====================================================================
// Concatena salt || INT(1) big-endian 32 bits e adiciona padding
// SHA-512 para que o comprimento total seja multiplo de 128 bytes.
//
// O resultado pode ter 1 ou mais blocos de 128 bytes.
// Se salt > 112 bytes, serao necessarios multiplos blocos.
// =====================================================================
inline std::vector<uint8_t> prepare_salt_padded(
    const uint8_t* salt, size_t salt_len,
    size_t& padded_len, size_t& num_blocks,
    size_t ipad_block_size = 128)
{
    // Total da mensagem: ipad_block(128) + salt + INT(1)(4)
    // O bit length no padding SHA-512 deve ser o TOTAL da mensagem
    size_t salt_msg_len = salt_len + 4; // salt + INT(1)
    size_t total_msg_len = ipad_block_size + salt_msg_len;
    // Tamanho minimo do padding: salt_msg_len + 1 (0x80) + zeros + 16 (bit length)
    size_t padded;
    if (salt_msg_len % 128 <= 111) {
        padded = ((salt_msg_len / 128) + 1) * 128;
    } else {
        padded = ((salt_msg_len / 128) + 2) * 128;
    }

    std::vector<uint8_t> result(padded, 0);

    // Copiar salt
    std::memcpy(result.data(), salt, salt_len);

    // INT(1) big-endian 32 bits
    result[salt_len]     = 0;
    result[salt_len + 1] = 0;
    result[salt_len + 2] = 0;
    result[salt_len + 3] = 1;

    // Padding: 0x80
    result[salt_msg_len] = 0x80;

    // Bit length big-endian 128 bits
    // Inclui o ipad_block ja processado!
    uint64_t bit_len = (uint64_t)total_msg_len * 8;
    result[padded - 8] = (uint8_t)(bit_len >> 56);
    result[padded - 7] = (uint8_t)(bit_len >> 48);
    result[padded - 6] = (uint8_t)(bit_len >> 40);
    result[padded - 5] = (uint8_t)(bit_len >> 32);
    result[padded - 4] = (uint8_t)(bit_len >> 24);
    result[padded - 3] = (uint8_t)(bit_len >> 16);
    result[padded - 2] = (uint8_t)(bit_len >> 8);
    result[padded - 1] = (uint8_t)(bit_len);

    padded_len = padded;
    num_blocks = padded / 128;

    return result;
}

// =====================================================================
// prepare_for_kernel: prepara todos os buffers para enviar ao kernel
// =====================================================================
struct KernelBuffers {
    std::vector<uint8_t> ipad_states;   // [num_items * 64] bytes
    std::vector<uint8_t> opad_states;   // [num_items * 64] bytes
    std::vector<uint8_t> salt_padded;   // multiplo de 128 bytes
    size_t salt_padded_len;
    size_t salt_blocks;
};

inline KernelBuffers prepare_for_kernel(
    const std::vector<std::pair<const uint8_t*, size_t>>& passwords,
    const uint8_t* salt, size_t salt_len)
{
    KernelBuffers buf;
    size_t n = passwords.size();

    buf.ipad_states.resize(n * 64);
    buf.opad_states.resize(n * 64);

    for (size_t i = 0; i < n; i++) {
        uint64_t ipad_s[8], opad_s[8];
        compute_hmac_states(passwords[i].first, passwords[i].second, ipad_s, opad_s);

        // Serializar como big-endian ulongs
        for (int j = 0; j < 8; j++) {
            uint8_t* ip = buf.ipad_states.data() + i * 64 + j * 8;
            uint8_t* op = buf.opad_states.data() + i * 64 + j * 8;
            ip[0] = (uint8_t)(ipad_s[j] >> 56); ip[1] = (uint8_t)(ipad_s[j] >> 48);
            ip[2] = (uint8_t)(ipad_s[j] >> 40); ip[3] = (uint8_t)(ipad_s[j] >> 32);
            ip[4] = (uint8_t)(ipad_s[j] >> 24); ip[5] = (uint8_t)(ipad_s[j] >> 16);
            ip[6] = (uint8_t)(ipad_s[j] >> 8);  ip[7] = (uint8_t)(ipad_s[j]);
            op[0] = (uint8_t)(opad_s[j] >> 56); op[1] = (uint8_t)(opad_s[j] >> 48);
            op[2] = (uint8_t)(opad_s[j] >> 40); op[3] = (uint8_t)(opad_s[j] >> 32);
            op[4] = (uint8_t)(opad_s[j] >> 24); op[5] = (uint8_t)(opad_s[j] >> 16);
            op[6] = (uint8_t)(opad_s[j] >> 8);  op[7] = (uint8_t)(opad_s[j]);
        }
    }

    buf.salt_padded = prepare_salt_padded(salt, salt_len, buf.salt_padded_len, buf.salt_blocks);

    return buf;
}

// =====================================================================
// pbkdf2_verify_cpu: referencia CPU para validacao do kernel
// =====================================================================
inline void pbkdf2_verify_cpu(
    const uint8_t* password, size_t pass_len,
    const uint8_t* salt, size_t salt_len,
    int iterations, uint8_t* out, size_t out_len)
{
    crypto::pbkdf2_hmac_sha512(
        reinterpret_cast<const char*>(password), pass_len,
        salt, salt_len, iterations, out, out_len);
}

} // namespace pbkdf2
