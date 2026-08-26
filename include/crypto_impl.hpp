#pragma once
#include <cstdint>
#include <cstring>
#include <array>
#include <vector>
#include <openssl/evp.h>

// SHA-NI intrinsics (Intel SHA Extensions)
#ifdef __SHA__
#include <immintrin.h>
#define HAS_SHA_NI 0  // SHA-NI disabled: using generic path for correctness
#else
#define HAS_SHA_NI 0
#endif

namespace crypto {

// ============================================================
// SHA-256 — SHA-NI quando disponível, fallback genérico
// ============================================================
class SHA256 {
public:
    SHA256() { reset(); }

    void reset() {
        h_[0] = 0x6a09e667; h_[1] = 0xbb67ae85;
        h_[2] = 0x3c6ef372; h_[3] = 0xa54ff53a;
        h_[4] = 0x510e527f; h_[5] = 0x9b05688c;
        h_[6] = 0x1f83d9ab; h_[7] = 0x5be0cd19;
        total_len_ = 0;
        buf_len_ = 0;
    }

    void update(const void* data, size_t len) {
        auto p = static_cast<const uint8_t*>(data);
        total_len_ += len;

        if (buf_len_ > 0) {
            size_t to_copy = std::min(len, 64 - buf_len_);
            std::memcpy(buf_ + buf_len_, p, to_copy);
            buf_len_ += to_copy;
            p += to_copy;
            len -= to_copy;
            if (buf_len_ == 64) {
                process_block(buf_);
                buf_len_ = 0;
            }
        }

        while (len >= 64) {
            process_block(p);
            p += 64;
            len -= 64;
        }

        if (len > 0) {
            std::memcpy(buf_, p, len);
            buf_len_ = len;
        }
    }

    void finalize(uint8_t out[32]) {
        uint64_t bit_len = total_len_ * 8;
        buf_[buf_len_++] = 0x80;
        if (buf_len_ > 56) {
            std::memset(buf_ + buf_len_, 0, 64 - buf_len_);
            process_block(buf_);
            buf_len_ = 0;
        }
        std::memset(buf_ + buf_len_, 0, 56 - buf_len_);
        for (int i = 7; i >= 0; --i) {
            buf_[56 + (7 - i)] = static_cast<uint8_t>(bit_len >> (i * 8));
        }
        process_block(buf_);

        for (int i = 0; i < 8; ++i) {
            out[i * 4]     = static_cast<uint8_t>(h_[i] >> 24);
            out[i * 4 + 1] = static_cast<uint8_t>(h_[i] >> 16);
            out[i * 4 + 2] = static_cast<uint8_t>(h_[i] >> 8);
            out[i * 4 + 3] = static_cast<uint8_t>(h_[i]);
        }
    }

    static void hash(const void* data, size_t len, uint8_t out[32]) {
        SHA256 ctx;
        ctx.update(data, len);
        ctx.finalize(out);
    }

private:
    uint32_t h_[8];
    uint8_t buf_[64];
    size_t buf_len_;
    uint64_t total_len_;

    static inline uint32_t rotr(uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }
    static inline uint32_t Ch(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (~x & z); }
    static inline uint32_t Maj(uint32_t x, uint32_t y, uint32_t z) { return (x & y) ^ (x & z) ^ (y & z); }
    static inline uint32_t Sigma0(uint32_t x) { return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22); }
    static inline uint32_t Sigma1(uint32_t x) { return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25); }
    static inline uint32_t sigma0(uint32_t x) { return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3); }
    static inline uint32_t sigma1(uint32_t x) { return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10); }

    static constexpr uint32_t K[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
        0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
        0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
        0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
        0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    };

#if HAS_SHA_NI
    __attribute__((hot, optimize("O3,unroll-loops")))
    void process_block(const uint8_t block[64]) {
        __m128i STATE0, STATE1, MSG;
        __m128i MSG0, MSG1, MSG2, MSG3;
        __m128i TMP;
        const __m128i MASK=_mm_set_epi64x(0x0000000000000000ULL,0xFFFFFFFFFFFFFFFFULL);
        const __m128i SHUF_MASK=_mm_set_epi8(28,29,30,31,24,25,26,27,20,21,22,23,16,17,18,19);

        STATE0 = _mm_loadu_si128((__m128i*)&h_[0]);
        STATE1 = _mm_loadu_si128((__m128i*)&h_[4]);
        STATE0 = _mm_shuffle_epi8(STATE0, SHUF_MASK);
        STATE1 = _mm_shuffle_epi8(STATE1, SHUF_MASK);

        MSG = _mm_loadu_si128((__m128i*)(block+0));
        MSG0 = _mm_shuffle_epi8(MSG, SHUF_MASK);
        MSG = _mm_loadu_si128((__m128i*)(block+16));
        MSG1 = _mm_shuffle_epi8(MSG, SHUF_MASK);
        MSG = _mm_loadu_si128((__m128i*)(block+32));
        MSG2 = _mm_shuffle_epi8(MSG, SHUF_MASK);
        MSG = _mm_loadu_si128((__m128i*)(block+48));
        MSG3 = _mm_shuffle_epi8(MSG, SHUF_MASK);

        TMP = _mm_add_epi32(MSG0, _mm_set_epi64x(0xE9B5DBA58189DBBCLL, 0xB5C0FBCFEC4D3B2FLL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, TMP);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(TMP, 0x0E));

        MSG0 = _mm_sha256msg1_epu32(MSG0, MSG1);
        TMP = _mm_alignr_epi8(MSG3, MSG2, 8);
        MSG0 = _mm_add_epi32(MSG0, TMP);
        MSG0 = _mm_sha256msg2_epu32(MSG0, MSG3);

        TMP = _mm_add_epi32(MSG1, _mm_set_epi64x(0xC67178F2BEF9A3F7ULL, 0xE49B69C19EF14AD2ULL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, TMP);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(TMP, 0x0E));

        MSG1 = _mm_sha256msg1_epu32(MSG1, MSG2);
        TMP = _mm_alignr_epi8(MSG0, MSG3, 8);
        MSG1 = _mm_add_epi32(MSG1, TMP);
        MSG1 = _mm_sha256msg2_epu32(MSG1, MSG0);

        TMP = _mm_add_epi32(MSG2, _mm_set_epi64x(0x510E527FADE682D1ULL, 0x3956C25BF348B538ULL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, TMP);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(TMP, 0x0E));

        MSG2 = _mm_sha256msg1_epu32(MSG2, MSG3);
        TMP = _mm_alignr_epi8(MSG1, MSG0, 8);
        MSG2 = _mm_add_epi32(MSG2, TMP);
        MSG2 = _mm_sha256msg2_epu32(MSG2, MSG1);

        TMP = _mm_add_epi32(MSG3, _mm_set_epi64x(0x9B05688C2B3E6C1FULL, 0x59F111F1B605D019ULL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, TMP);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(TMP, 0x0E));

        TMP = _mm_add_epi32(MSG0, _mm_set_epi64x(0xC19BF174CF692694ULL, 0x7137449123EF65CDULL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, TMP);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(TMP, 0x0E));

        MSG3 = _mm_sha256msg1_epu32(MSG3, MSG0);
        TMP = _mm_alignr_epi8(MSG2, MSG1, 8);
        MSG3 = _mm_add_epi32(MSG3, TMP);
        MSG3 = _mm_sha256msg2_epu32(MSG3, MSG2);

        TMP = _mm_add_epi32(MSG1, _mm_set_epi64x(0x550C7DC3D5FFB4E2ULL, 0xD807AA98A3030242ULL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, TMP);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(TMP, 0x0E));

        MSG0 = _mm_sha256msg1_epu32(MSG0, MSG1);
        TMP = _mm_alignr_epi8(MSG3, MSG2, 8);
        MSG0 = _mm_add_epi32(MSG0, TMP);
        MSG0 = _mm_sha256msg2_epu32(MSG0, MSG3);

        TMP = _mm_add_epi32(MSG2, _mm_set_epi64x(0x243185BEE4E4B28CULL, 0x12835B0145706FBEULL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, TMP);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(TMP, 0x0E));

        MSG1 = _mm_sha256msg1_epu32(MSG1, MSG2);
        TMP = _mm_alignr_epi8(MSG0, MSG3, 8);
        MSG1 = _mm_add_epi32(MSG1, TMP);
        MSG1 = _mm_sha256msg2_epu32(MSG1, MSG0);

        TMP = _mm_add_epi32(MSG3, _mm_set_epi64x(0x9BDC06A725C71235ULL, 0x80DEB1FE3B1696B1ULL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, TMP);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(TMP, 0x0E));

        MSG2 = _mm_sha256msg1_epu32(MSG2, MSG3);
        TMP = _mm_alignr_epi8(MSG1, MSG0, 8);
        MSG2 = _mm_add_epi32(MSG2, TMP);
        MSG2 = _mm_sha256msg2_epu32(MSG2, MSG1);

        TMP = _mm_add_epi32(MSG0, _mm_set_epi64x(0xE9B5DBA58189DBBCLL, 0x72BE5D74F27B896FLL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, TMP);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(TMP, 0x0E));

        MSG3 = _mm_sha256msg1_epu32(MSG3, MSG0);
        TMP = _mm_alignr_epi8(MSG2, MSG1, 8);
        MSG3 = _mm_add_epi32(MSG3, TMP);
        MSG3 = _mm_sha256msg2_epu32(MSG3, MSG2);

        TMP = _mm_add_epi32(MSG1, _mm_set_epi64x(0xB5C0FBCFEC4D3B2FULL, 0x428A2F98D728AE22ULL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, TMP);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(TMP, 0x0E));

        MSG0 = _mm_sha256msg1_epu32(MSG0, MSG1);
        TMP = _mm_alignr_epi8(MSG3, MSG2, 8);
        MSG0 = _mm_add_epi32(MSG0, TMP);
        MSG0 = _mm_sha256msg2_epu32(MSG0, MSG3);

        TMP = _mm_add_epi32(MSG2, _mm_set_epi64x(0x0FC19DC68B8CD5B5ULL, 0xEFBE4786384F25E3ULL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, TMP);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(TMP, 0x0E));

        MSG1 = _mm_sha256msg1_epu32(MSG1, MSG2);
        TMP = _mm_alignr_epi8(MSG0, MSG3, 8);
        MSG1 = _mm_add_epi32(MSG1, TMP);
        MSG1 = _mm_sha256msg2_epu32(MSG1, MSG0);

        TMP = _mm_add_epi32(MSG3, _mm_set_epi64x(0x2DE92C6F592B0275ULL, 0x240CA1CC77AC9C65ULL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, TMP);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(TMP, 0x0E));

        MSG2 = _mm_sha256msg1_epu32(MSG2, MSG3);
        TMP = _mm_alignr_epi8(MSG1, MSG0, 8);
        MSG2 = _mm_add_epi32(MSG2, TMP);
        MSG2 = _mm_sha256msg2_epu32(MSG2, MSG1);

        TMP = _mm_add_epi32(MSG0, _mm_set_epi64x(0x5CB0A9DCBD41FBD4ULL, 0x4A7484AA6EA6E483ULL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, TMP);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(TMP, 0x0E));

        MSG3 = _mm_sha256msg1_epu32(MSG3, MSG0);
        TMP = _mm_alignr_epi8(MSG2, MSG1, 8);
        MSG3 = _mm_add_epi32(MSG3, TMP);
        MSG3 = _mm_sha256msg2_epu32(MSG3, MSG2);

        TMP = _mm_add_epi32(MSG1, _mm_set_epi64x(0xBF597FC7BEEF0EE4ULL, 0x983E5152EE66DFABULL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, TMP);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(TMP, 0x0E));

        TMP = _mm_add_epi32(MSG2, _mm_set_epi64x(0xC6E00BF33DA88FC7ULL, 0xB00327C898FB213FULL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, TMP);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(TMP, 0x0E));

        TMP = _mm_add_epi32(MSG3, _mm_set_epi64x(0x142929670A0E6E70ULL, 0x06CA6351E003826FLL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, TMP);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(TMP, 0x0E));

        TMP = _mm_add_epi32(MSG0, _mm_set_epi64x(0x2E1B21385C26C926ULL, 0x27B70A8546D22FFCLL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, TMP);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(TMP, 0x0E));

        TMP = _mm_add_epi32(MSG1, _mm_set_epi64x(0x53380D139D95B3DFULL, 0x4D2C6DFC5AC42AEDULL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, TMP);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(TMP, 0x0E));

        TMP = _mm_add_epi32(MSG2, _mm_set_epi64x(0x81C2C92E47EDAEE6ULL, 0x766A0ABB3C77B2A8ULL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, TMP);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(TMP, 0x0E));

        TMP = _mm_add_epi32(MSG3, _mm_set_epi64x(0xA2BFE8A14CF10364ULL, 0x650A73548BAF63DEULL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, TMP);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(TMP, 0x0E));

        TMP = _mm_add_epi32(MSG0, _mm_set_epi64x(0xC24B8B70D0F89791ULL, 0xA81A664BBC423001ULL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, TMP);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(TMP, 0x0E));

        MSG = _mm_add_epi32(MSG0, _mm_set_epi64x(0xD192E819D6EF5218ULL, 0xC76C51A30654BE30ULL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(MSG, 0x0E));

        MSG = _mm_add_epi32(MSG1, _mm_set_epi64x(0x106AA07032BBD1B8ULL, 0xF40E35855771202ALL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(MSG, 0x0E));

        MSG = _mm_add_epi32(MSG2, _mm_set_epi64x(0x2748774CDF8EEB99ULL, 0x19A4C116B8D2D0C8ULL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(MSG, 0x0E));

        MSG = _mm_add_epi32(MSG3, _mm_set_epi64x(0x391C0CB3C5C95A63ULL, 0x1E376C085141AB53ULL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(MSG, 0x0E));

        MSG = _mm_add_epi32(MSG0, _mm_set_epi64x(0x4ED8AA4AE3418ACBULL, 0x34B0BCB5E19B48A8ULL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(MSG, 0x0E));

        MSG = _mm_add_epi32(MSG1, _mm_set_epi64x(0x682E6FF3D6B2B8A3ULL, 0x5B9CCA4F7763E373ULL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(MSG, 0x0E));

        MSG = _mm_add_epi32(MSG2, _mm_set_epi64x(0x84C87814A1F0AB72ULL, 0x78A5636F43172F60ULL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(MSG, 0x0E));

        MSG = _mm_add_epi32(MSG3, _mm_set_epi64x(0x90BEFFFA23631E28ULL, 0x8CC702081A6439ECLL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(MSG, 0x0E));

        MSG = _mm_add_epi32(MSG0, _mm_set_epi64x(0xBEF9A3F7B2C67915ULL, 0xA4506CEBDE82BDE9ULL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(MSG, 0x0E));

        MSG = _mm_add_epi32(MSG1, _mm_set_epi64x(0xCA273ECEEA26619CULL, 0xC67178F2E372532BULL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(MSG, 0x0E));

        MSG = _mm_add_epi32(MSG2, _mm_set_epi64x(0xF57D4F7FEE6ED178ULL, 0xD186B8C721C0C207ULL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(MSG, 0x0E));

        MSG = _mm_add_epi32(MSG3, _mm_set_epi64x(0x0A637DC5A2C898A6ULL, 0x06F067AA72176FBALL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(MSG, 0x0E));

        MSG = _mm_add_epi32(MSG0, _mm_set_epi64x(0x1B710B35131C471BULL, 0x113F9804BEF90DAEULL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(MSG, 0x0E));

        MSG = _mm_add_epi32(MSG1, _mm_set_epi64x(0x32CAAB7B40C72493ULL, 0x28DB77F523047D84ULL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(MSG, 0x0E));

        MSG = _mm_add_epi32(MSG2, _mm_set_epi64x(0x431D67C49C100D4CULL, 0x3C9EBE0A15C9BEBCLL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(MSG, 0x0E));

        MSG = _mm_add_epi32(MSG3, _mm_set_epi64x(0x597F299CFC657E2AULL, 0x4CC5D4BECB3E42B6ULL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(MSG, 0x0E));

        MSG = _mm_add_epi32(MSG0, _mm_set_epi64x(0x6C44198C4A475817ULL, 0x5FCBFAB3AD6FAECELL));
        STATE1 = _mm_sha256rnds2_epu32(STATE1, STATE0, MSG);
        STATE0 = _mm_sha256rnds2_epu32(STATE0, STATE1, _mm_shuffle_epi32(MSG, 0x0E));

        // Add back to state
        STATE0 = _mm_shuffle_epi8(STATE0, SHUF_MASK);
        STATE1 = _mm_shuffle_epi8(STATE1, SHUF_MASK);

        __m128i TMPA = _mm_loadu_si128((__m128i*)&h_[0]);
        __m128i TMPB = _mm_loadu_si128((__m128i*)&h_[4]);
        STATE0 = _mm_add_epi32(STATE0, TMPA);
        STATE1 = _mm_add_epi32(STATE1, TMPB);

        _mm_storeu_si128((__m128i*)&h_[0], STATE0);
        _mm_storeu_si128((__m128i*)&h_[4], STATE1);
    }
#else
    __attribute__((hot, optimize("O3,unroll-loops")))
    void process_block(const uint8_t block[64]) {
        uint32_t W[64];
        for (int i = 0; i < 16; ++i) {
            W[i] = (uint32_t(block[i*4]) << 24) | (uint32_t(block[i*4+1]) << 16)
                  | (uint32_t(block[i*4+2]) << 8) | block[i*4+3];
        }
        for (int i = 16; i < 64; ++i) {
            W[i] = sigma1(W[i-2]) + W[i-7] + sigma0(W[i-15]) + W[i-16];
        }

        uint32_t a = h_[0], b = h_[1], c = h_[2], d = h_[3];
        uint32_t e = h_[4], f = h_[5], g = h_[6], h = h_[7];

        for (int i = 0; i < 64; ++i) {
            uint32_t T1 = h + Sigma1(e) + Ch(e, f, g) + K[i] + W[i];
            uint32_t T2 = Sigma0(a) + Maj(a, b, c);
            h = g; g = f; f = e; e = d + T1;
            d = c; c = b; b = a; a = T1 + T2;
        }

        h_[0] += a; h_[1] += b; h_[2] += c; h_[3] += d;
        h_[4] += e; h_[5] += f; h_[6] += g; h_[7] += h;
    }
#endif
};

// ============================================================
// SHA-512 — reference implementation (FIPS 180-4)
// ============================================================
class SHA512 {
public:
    SHA512() { reset(); }

    void reset() {
        h_[0] = 0x6a09e667f3bcc908ULL; h_[1] = 0xbb67ae8584caa73bULL;
        h_[2] = 0x3c6ef372fe94f82bULL; h_[3] = 0xa54ff53a5f1d36f1ULL;
        h_[4] = 0x510e527fade682d1ULL; h_[5] = 0x9b05688c2b3e6c1fULL;
        h_[6] = 0x1f83d9abfb41bd6bULL; h_[7] = 0x5be0cd19137e2179ULL;
        total_len_ = 0;
        buf_len_ = 0;
    }

    void update(const void* data, size_t len) {
        auto p = static_cast<const uint8_t*>(data);
        total_len_ += len;
        if (buf_len_ > 0) {
            size_t to_copy = std::min(len, (size_t)128 - buf_len_);
            std::memcpy(buf_ + buf_len_, p, to_copy);
            buf_len_ += to_copy;
            p += to_copy;
            len -= to_copy;
            if (buf_len_ == 128) { process_block(buf_); buf_len_ = 0; }
        }
        while (len >= 128) { process_block(p); p += 128; len -= 128; }
        if (len > 0) { std::memcpy(buf_, p, len); buf_len_ = len; }
    }

    void finalize(uint8_t out[64]) {
        uint64_t bit_len = total_len_ * 8;
        buf_[buf_len_++] = 0x80;
        if (buf_len_ > 112) {
            std::memset(buf_ + buf_len_, 0, (size_t)128 - buf_len_);
            process_block(buf_);
            buf_len_ = 0;
        }
        std::memset(buf_ + buf_len_, 0, (size_t)112 - buf_len_);
        // 128-bit big-endian length at bytes 112-127
        // High 64 bits (bytes 112-119) = 0 for messages < 2^64 bits
        std::memset(buf_ + 112, 0, 8);
        // Low 64 bits (bytes 120-127) = bit_len big-endian
        buf_[120] = (uint8_t)(bit_len >> 56);
        buf_[121] = (uint8_t)(bit_len >> 48);
        buf_[122] = (uint8_t)(bit_len >> 40);
        buf_[123] = (uint8_t)(bit_len >> 32);
        buf_[124] = (uint8_t)(bit_len >> 24);
        buf_[125] = (uint8_t)(bit_len >> 16);
        buf_[126] = (uint8_t)(bit_len >> 8);
        buf_[127] = (uint8_t)(bit_len);
        process_block(buf_);
        for (int i = 0; i < 8; ++i) {
            out[i*8]   = (uint8_t)(h_[i] >> 56);
            out[i*8+1] = (uint8_t)(h_[i] >> 48);
            out[i*8+2] = (uint8_t)(h_[i] >> 40);
            out[i*8+3] = (uint8_t)(h_[i] >> 32);
            out[i*8+4] = (uint8_t)(h_[i] >> 24);
            out[i*8+5] = (uint8_t)(h_[i] >> 16);
            out[i*8+6] = (uint8_t)(h_[i] >> 8);
            out[i*8+7] = (uint8_t)(h_[i]);
        }
    }

    static void hash(const void* data, size_t len, uint8_t out[64]) {
        SHA512 ctx;
        ctx.update(data, len);
        ctx.finalize(out);
    }

    // Clone for HMAC pre-computation
    void clone_from(const SHA512& src) {
        std::memcpy(h_, src.h_, sizeof(h_));
        std::memcpy(buf_, src.buf_, sizeof(buf_));
        buf_len_ = src.buf_len_;
        total_len_ = src.total_len_;
    }

private:
    uint64_t h_[8];
    uint8_t buf_[128];
    size_t buf_len_;
    uint64_t total_len_;

    static constexpr uint64_t K[80] = {
        0x428a2f98d728ae22ULL, 0x7137449123ef65cdULL, 0xb5c0fbcfec4d3b2fULL, 0xe9b5dba58189dbbcULL,
        0x3956c25bf348b538ULL, 0x59f111f1b605d019ULL, 0x923f82a4af194f9bULL, 0xab1c5ed5da6d8118ULL,
        0xd807aa98a3030242ULL, 0x12835b0145706fbeULL, 0x243185be4ee4b28cULL, 0x550c7dc3d5ffb4e2ULL,
        0x72be5d74f27b896fULL, 0x80deb1fe3b1696b1ULL, 0x9bdc06a725c71235ULL, 0xc19bf174cf692694ULL,
        0xe49b69c19ef14ad2ULL, 0xefbe4786384f25e3ULL, 0x0fc19dc68b8cd5b5ULL, 0x240ca1cc77ac9c65ULL,
        0x2de92c6f592b0275ULL, 0x4a7484aa6ea6e483ULL, 0x5cb0a9dcbd41fbd4ULL, 0x76f988da831153b5ULL,
        0x983e5152ee66dfabULL, 0xa831c66d2db43210ULL, 0xb00327c898fb213fULL, 0xbf597fc7beef0ee4ULL,
        0xc6e00bf33da88fc7ULL, 0xd5a79147930aa725ULL, 0x06ca6351e003826fULL, 0x142929670a0e6e70ULL,
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

    static inline uint64_t rotr(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }
    static inline uint64_t Ch(uint64_t x, uint64_t y, uint64_t z) { return (x & y) ^ (~x & z); }
    static inline uint64_t Maj(uint64_t x, uint64_t y, uint64_t z) { return (x & y) ^ (x & z) ^ (y & z); }
    static inline uint64_t Sigma0(uint64_t x) { return rotr(x, 28) ^ rotr(x, 34) ^ rotr(x, 39); }
    static inline uint64_t Sigma1(uint64_t x) { return rotr(x, 14) ^ rotr(x, 18) ^ rotr(x, 41); }
    static inline uint64_t sigma0(uint64_t x) { return rotr(x, 1) ^ rotr(x, 8) ^ (x >> 7); }
    static inline uint64_t sigma1(uint64_t x) { return rotr(x, 19) ^ rotr(x, 61) ^ (x >> 6); }

    __attribute__((hot, optimize("O3")))
    void process_block(const uint8_t block[128]) {
        uint64_t W[80];
        for (int i = 0; i < 16; ++i) {
            int o = i * 8;
            W[i] = ((uint64_t)block[o] << 56) | ((uint64_t)block[o+1] << 48)
                 | ((uint64_t)block[o+2] << 40) | ((uint64_t)block[o+3] << 32)
                 | ((uint64_t)block[o+4] << 24) | ((uint64_t)block[o+5] << 16)
                 | ((uint64_t)block[o+6] << 8) | (uint64_t)block[o+7];
        }
        for (int i = 16; i < 80; ++i)
            W[i] = sigma1(W[i-2]) + W[i-7] + sigma0(W[i-15]) + W[i-16];
        uint64_t a=h_[0],b=h_[1],c=h_[2],d=h_[3],e=h_[4],f=h_[5],g=h_[6],hh=h_[7];
        for (int i = 0; i < 80; ++i) {
            uint64_t T1 = hh + Sigma1(e) + Ch(e,f,g) + K[i] + W[i];
            uint64_t T2 = Sigma0(a) + Maj(a,b,c);
            hh=g; g=f; f=e; e=d+T1; d=c; c=b; b=a; a=T1+T2;
        }
        h_[0]+=a; h_[1]+=b; h_[2]+=c; h_[3]+=d;
        h_[4]+=e; h_[5]+=f; h_[6]+=g; h_[7]+=hh;
    }
};

// ============================================================
// HMAC-SHA512
// ============================================================
class HMAC_SHA512 {
public:
    void init(const uint8_t* key, size_t key_len) {
        uint8_t k_hash[64];
        const uint8_t* k_use; size_t k_use_len;
        if (key_len > 128) { SHA512::hash(key, key_len, k_hash); k_use=k_hash; k_use_len=64; }
        else { k_use=key; k_use_len=key_len; }
        std::memset(ipad_, 0x36, 128); std::memset(opad_, 0x5c, 128);
        for (size_t i=0;i<k_use_len;++i) { ipad_[i]^=k_use[i]; opad_[i]^=k_use[i]; }
        inner_.update(ipad_, 128);
    }
    void update(const void* data, size_t len) { inner_.update(data, len); }
    void finalize(uint8_t out[64]) {
        uint8_t inner_hash[64]; inner_.finalize(inner_hash);
        SHA512 outer; outer.update(opad_, 128); outer.update(inner_hash, 64); outer.finalize(out);
    }
private:
    uint8_t ipad_[128], opad_[128]; SHA512 inner_;
};

// ============================================================
// PBKDF2-HMAC-SHA512 — optimized with pre-computed HMAC states
// ============================================================
__attribute__((hot, optimize("O3")))
inline void pbkdf2_hmac_sha512(
    const char* password, size_t password_len,
    const uint8_t* salt, size_t salt_len,
    int iterations, uint8_t* out, size_t out_len)
{
    const EVP_MD* md = EVP_sha512();
    uint8_t key_buf[128];
    const uint8_t* key = reinterpret_cast<const uint8_t*>(password);
    size_t key_len = password_len;
    if (password_len > 128) {
        unsigned int md_len = 0;
        EVP_Digest(password, password_len, key_buf, &md_len, md, nullptr);
        key = key_buf; key_len = 64;
    }
    uint8_t ipad_block[128], opad_block[128];
    std::memset(ipad_block, 0x36, 128);
    std::memset(opad_block, 0x5c, 128);
    for (size_t i = 0; i < key_len; i++) {
        ipad_block[i] ^= key[i];
        opad_block[i] ^= key[i];
    }
    EVP_MD_CTX* ipad_ctx = EVP_MD_CTX_new();
    EVP_MD_CTX* opad_ctx = EVP_MD_CTX_new();
    EVP_DigestInit_ex(ipad_ctx, md, nullptr);
    EVP_DigestUpdate(ipad_ctx, ipad_block, 128);
    EVP_DigestInit_ex(opad_ctx, md, nullptr);
    EVP_DigestUpdate(opad_ctx, opad_block, 128);
    uint8_t U[64], T[64];
    { EVP_MD_CTX* inner = EVP_MD_CTX_new();
      EVP_MD_CTX_copy_ex(inner, ipad_ctx); EVP_DigestUpdate(inner, salt, salt_len);
      uint8_t be4[4]={0,0,0,1}; EVP_DigestUpdate(inner, be4, 4);
      unsigned int md_len=0; EVP_DigestFinal_ex(inner, U, &md_len);
      EVP_MD_CTX_copy_ex(inner, opad_ctx); EVP_DigestUpdate(inner, U, 64);
      EVP_DigestFinal_ex(inner, U, &md_len); EVP_MD_CTX_free(inner); }
    std::memcpy(T, U, 64);
    { EVP_MD_CTX* inner = EVP_MD_CTX_new();
      for (int iter=1; iter<iterations; ++iter) {
          EVP_MD_CTX_copy_ex(inner, ipad_ctx); EVP_DigestUpdate(inner, U, 64);
          unsigned int md_len=0; EVP_DigestFinal_ex(inner, U, &md_len);
          EVP_MD_CTX_copy_ex(inner, opad_ctx); EVP_DigestUpdate(inner, U, 64);
          EVP_DigestFinal_ex(inner, U, &md_len);
          for (int j=0;j<64;++j) T[j]^=U[j];
      } EVP_MD_CTX_free(inner); }
    EVP_MD_CTX_free(ipad_ctx); EVP_MD_CTX_free(opad_ctx);
    std::memcpy(out, T, std::min((size_t)64, out_len));
}

// ============================================================
// RIPEMD-160
// ============================================================
class RIPEMD160 {
public:
    RIPEMD160() { reset(); }

    void reset() {
        h_[0] = 0x67452301; h_[1] = 0xefcdab89;
        h_[2] = 0x98badcfe; h_[3] = 0x10325476;
        h_[4] = 0xc3d2e1f0;
        total_len_ = 0;
        buf_len_ = 0;
    }

    void update(const void* data, size_t len) {
        auto p = static_cast<const uint8_t*>(data);
        total_len_ += len;

        if (buf_len_ > 0) {
            size_t to_copy = std::min(len, 64 - buf_len_);
            std::memcpy(buf_ + buf_len_, p, to_copy);
            buf_len_ += to_copy;
            p += to_copy;
            len -= to_copy;
            if (buf_len_ == 64) {
                process_block(buf_);
                buf_len_ = 0;
            }
        }

        while (len >= 64) {
            process_block(p);
            p += 64;
            len -= 64;
        }

        if (len > 0) {
            std::memcpy(buf_, p, len);
            buf_len_ = len;
        }
    }

    void finalize(uint8_t out[20]) {
        uint64_t bit_len = total_len_ * 8;
        buf_[buf_len_++] = 0x80;
        if (buf_len_ > 56) {
            std::memset(buf_ + buf_len_, 0, 64 - buf_len_);
            process_block(buf_);
            buf_len_ = 0;
        }
        std::memset(buf_ + buf_len_, 0, 56 - buf_len_);
        for (int i = 0; i < 8; ++i) {
            buf_[56 + i] = static_cast<uint8_t>(bit_len >> (i * 8));
        }
        process_block(buf_);

        for (int i = 0; i < 5; ++i) {
            out[i * 4]     = static_cast<uint8_t>(h_[i]);
            out[i * 4 + 1] = static_cast<uint8_t>(h_[i] >> 8);
            out[i * 4 + 2] = static_cast<uint8_t>(h_[i] >> 16);
            out[i * 4 + 3] = static_cast<uint8_t>(h_[i] >> 24);
        }
    }

    static void hash(const void* data, size_t len, uint8_t out[20]) {
        RIPEMD160 ctx;
        ctx.update(data, len);
        ctx.finalize(out);
    }

private:
    uint32_t h_[5];
    uint8_t buf_[64];
    size_t buf_len_;
    uint64_t total_len_;

    static inline uint32_t rotl(uint32_t x, int n) { return (x << n) | (x >> (32 - n)); }

    static constexpr uint32_t KL[5] = { 0x00000000, 0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xa953fd4e };
    static constexpr uint32_t KR[5] = { 0x50a28be6, 0x5c4dd124, 0x6d703ef3, 0x7a6d76e9, 0x00000000 };

    static constexpr int RL[80] = {
        0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
        7,4,13,1,10,6,15,3,12,0,9,5,2,14,11,8,
        3,10,14,4,9,15,8,1,2,7,0,6,13,11,5,12,
        1,9,11,10,0,8,12,4,13,3,7,15,14,5,6,2,
        4,0,5,9,7,12,2,10,14,1,3,8,11,6,15,13
    };
    static constexpr int RR_[80] = {
        5,14,7,0,9,2,11,4,13,6,15,8,1,10,3,12,
        6,11,3,7,0,13,5,10,14,15,8,12,4,9,1,2,
        15,5,1,3,7,14,6,9,11,8,12,2,10,0,4,13,
        8,6,4,1,3,11,15,0,5,12,2,13,9,7,10,14,
        12,15,10,4,1,5,8,7,6,2,13,14,0,3,9,11
    };

    static constexpr int SL[80] = {
        11,14,15,12,5,8,7,9,11,13,14,15,6,7,9,8,
        7,6,8,13,11,9,7,15,7,12,15,9,11,7,13,12,
        11,13,6,7,14,9,13,15,14,8,13,6,5,12,7,5,
        11,12,14,15,14,15,9,8,9,14,5,6,8,6,5,12,
        9,15,5,11,6,8,13,12,5,12,13,14,11,8,5,6
    };
    static constexpr int SR_[80] = {
        8,9,9,11,13,15,15,5,7,7,8,11,14,14,12,6,
        9,13,15,7,12,8,9,11,7,7,12,7,6,15,13,11,
        9,7,15,11,8,6,6,14,12,13,5,14,13,13,7,5,
        15,5,8,11,14,14,6,14,6,9,12,9,12,5,15,8,
        8,5,12,9,12,5,14,6,8,13,6,5,15,13,11,11
    };

    static inline uint32_t F1(uint32_t x, uint32_t y, uint32_t z) { return x ^ y ^ z; }
    static inline uint32_t F2(uint32_t x, uint32_t y, uint32_t z) { return (x & y) | (~x & z); }
    static inline uint32_t F3(uint32_t x, uint32_t y, uint32_t z) { return (x | ~y) ^ z; }
    static inline uint32_t F4(uint32_t x, uint32_t y, uint32_t z) { return (x & z) | (y & ~z); }
    static inline uint32_t F5(uint32_t x, uint32_t y, uint32_t z) { return x ^ (y | ~z); }

    using RoundFn = uint32_t(*)(uint32_t, uint32_t, uint32_t);
    static constexpr RoundFn FL[5] = { F1, F2, F3, F4, F5 };
    static constexpr RoundFn FR[5] = { F5, F4, F3, F2, F1 };

    void process_block(const uint8_t block[64]) {
        uint32_t X[16];
        for (int i = 0; i < 16; ++i) {
            X[i] = (uint32_t(block[i*4])) | (uint32_t(block[i*4+1]) << 8)
                 | (uint32_t(block[i*4+2]) << 16) | (uint32_t(block[i*4+3]) << 24);
        }

        uint32_t al = h_[0], bl = h_[1], cl = h_[2], dl = h_[3], el = h_[4];
        uint32_t ar = h_[0], br = h_[1], cr = h_[2], dr = h_[3], er = h_[4];

        for (int j = 0; j < 80; ++j) {
            int round = j / 16;
            uint32_t t = al + FL[round](bl, cl, dl) + X[RL[j]] + KL[round];
            t = rotl(t, SL[j]) + el;
            al = el; el = dl; dl = rotl(cl, 10); cl = bl; bl = t;

            t = ar + FR[round](br, cr, dr) + X[RR_[j]] + KR[round];
            t = rotl(t, SR_[j]) + er;
            ar = er; er = dr; dr = rotl(cr, 10); cr = br; br = t;
        }

        uint32_t t = h_[1] + cl + dr;
        h_[1] = h_[2] + dl + er;
        h_[2] = h_[3] + el + ar;
        h_[3] = h_[4] + al + br;
        h_[4] = h_[0] + bl + cr;
        h_[0] = t;
    }
};
} // namespace crypto
