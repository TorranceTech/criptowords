#pragma once
#include <cstdint>
#include <cstring>
#include <algorithm>

namespace crypto {
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
        std::memset(buf_ + 112, 0, 8);
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

    static inline uint64_t rotr(uint64_t x, int n) { return (x >> n) | (x << (64 - n)); }
    static inline uint64_t Ch(uint64_t x, uint64_t y, uint64_t z) { return (x & y) ^ (~x & z); }
    static inline uint64_t Maj(uint64_t x, uint64_t y, uint64_t z) { return (x & y) ^ (x & z) ^ (y & z); }
    static inline uint64_t Sigma0(uint64_t x) { return rotr(x, 28) ^ rotr(x, 34) ^ rotr(x, 39); }
    static inline uint64_t Sigma1(uint64_t x) { return rotr(x, 14) ^ rotr(x, 18) ^ rotr(x, 41); }
    static inline uint64_t sigma0(uint64_t x) { return rotr(x, 1) ^ rotr(x, 8) ^ (x >> 7); }
    static inline uint64_t sigma1(uint64_t x) { return rotr(x, 19) ^ rotr(x, 61) ^ (x >> 6); }

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
} // namespace crypto
