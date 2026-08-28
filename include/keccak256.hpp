#pragma once
#include <cstdint>
#include <cstddef>
#include <cstring>

// Keccak-256 (variante original usada pelo Ethereum — padding 0x01, NAO o
// SHA3-256 do NIST que usa 0x06). Implementacao compacta do Keccak-f[1600].
// Observacao: assume maquina little-endian (x86), como o resto do projeto.
namespace crypto {

class Keccak256 {
public:
    static void hash(const uint8_t* in, size_t len, uint8_t out[32]) {
        uint64_t st[25];
        std::memset(st, 0, sizeof(st));
        const size_t rate = 136;            // 1088 bits / 8 (taxa para saida de 256 bits)
        auto* sb = reinterpret_cast<uint8_t*>(st);

        // Absorcao dos blocos completos
        while (len >= rate) {
            for (size_t j = 0; j < rate; ++j) sb[j] ^= in[j];
            keccakf(st);
            in += rate;
            len -= rate;
        }

        // Ultimo bloco com padding pad10*1 + dominio 0x01 do Keccak
        uint8_t block[136];
        std::memset(block, 0, sizeof(block));
        std::memcpy(block, in, len);
        block[len] ^= 0x01;
        block[rate - 1] ^= 0x80;
        for (size_t j = 0; j < rate; ++j) sb[j] ^= block[j];
        keccakf(st);

        std::memcpy(out, sb, 32);           // squeeze de 32 bytes
    }

private:
    static inline uint64_t rotl64(uint64_t x, int y) {
        return (x << y) | (x >> (64 - y));
    }

    static void keccakf(uint64_t st[25]) {
        static const uint64_t RC[24] = {
            0x0000000000000001ULL, 0x0000000000008082ULL, 0x800000000000808aULL, 0x8000000080008000ULL,
            0x000000000000808bULL, 0x0000000080000001ULL, 0x8000000080008081ULL, 0x8000000000008009ULL,
            0x000000000000008aULL, 0x0000000000000088ULL, 0x0000000080008009ULL, 0x000000008000000aULL,
            0x000000008000808bULL, 0x800000000000008bULL, 0x8000000000008089ULL, 0x8000000000008003ULL,
            0x8000000000008002ULL, 0x8000000000000080ULL, 0x000000000000800aULL, 0x800000008000000aULL,
            0x8000000080008081ULL, 0x8000000000008080ULL, 0x0000000080000001ULL, 0x8000000080008008ULL
        };
        static const int ROTC[24] = {
            1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44
        };
        static const int PILN[24] = {
            10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1
        };

        for (int round = 0; round < 24; ++round) {
            uint64_t bc[5];
            // Theta
            for (int i = 0; i < 5; ++i)
                bc[i] = st[i] ^ st[i + 5] ^ st[i + 10] ^ st[i + 15] ^ st[i + 20];
            for (int i = 0; i < 5; ++i) {
                uint64_t t = bc[(i + 4) % 5] ^ rotl64(bc[(i + 1) % 5], 1);
                for (int j = 0; j < 25; j += 5) st[j + i] ^= t;
            }
            // Rho + Pi
            uint64_t t = st[1];
            for (int i = 0; i < 24; ++i) {
                int j = PILN[i];
                uint64_t tmp = st[j];
                st[j] = rotl64(t, ROTC[i]);
                t = tmp;
            }
            // Chi
            for (int j = 0; j < 25; j += 5) {
                for (int i = 0; i < 5; ++i) bc[i] = st[j + i];
                for (int i = 0; i < 5; ++i)
                    st[j + i] ^= (~bc[(i + 1) % 5]) & bc[(i + 2) % 5];
            }
            // Iota
            st[0] ^= RC[round];
        }
    }
};

} // namespace crypto
