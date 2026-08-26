#pragma once
#include <cstdint>
#include <cstring>
#include <immintrin.h>
#include "crypto_impl.hpp"

namespace crypto {

// ============================================================
// SHA-512 Multi-Buffer: 4 blocos independentes simultaneos
// ============================================================
class SHA512_MB {
public:
    struct State { __m256i h[8]; };

    static constexpr uint64_t K[80] = {
        0x428a2f98d728ae22ULL,0x7137449123ef65cdULL,0xb5c0fbcfec4d3b2fULL,0xe9b5dba58189dbbcULL,
        0x3956c25bf348b538ULL,0x59f111f1b605d019ULL,0x923f82a4af194f9bULL,0xab1c5ed5da6d8118ULL,
        0xd807aa98a3030242ULL,0x12835b0145706fbeULL,0x243185be4ee4b28cULL,0x550c7dc3d5ffb4e2ULL,
        0x72be5d74f27b896fULL,0x80deb1fe3b1696b1ULL,0x9bdc06a725c71235ULL,0xc19bf174cf692694ULL,
        0xe49b69c19ef14ad2ULL,0xefbe4786384f25e3ULL,0x0fc19dc68b8cd5b5ULL,0x240ca1cc77ac9c65ULL,
        0x2de92c6f592b0275ULL,0x4a7484aa6ea6e483ULL,0x5cb0a9dcbd41fbd4ULL,0x76f988da831153b5ULL,
        0x983e5152ee66dfabULL,0xa831c66d2db43210ULL,0xb00327c898fb213fULL,0xbf597fc7beef0ee4ULL,
        0xc6e00bf33da88fc7ULL,0xd5a79147930aa725ULL,0x06ca6351e003826fULL,0x142929670a0e6e70ULL,
        0x27b70a8546d22ffcULL,0x2e1b21385c26c926ULL,0x4d2c6dfc5ac42aedULL,0x53380d139d95b3dfULL,
        0x650a73548baf63deULL,0x766a0abb3c77b2a8ULL,0x81c2c92e47edaee6ULL,0x92722c851482353bULL,
        0xa2bfe8a14cf10364ULL,0xa81a664bbc423001ULL,0xc24b8b70d0f89791ULL,0xc76c51a30654be30ULL,
        0xd192e819d6ef5218ULL,0xd69906245565a910ULL,0xf40e35855771202aULL,0x106aa07032bbd1b8ULL,
        0x19a4c116b8d2d0c8ULL,0x1e376c085141ab53ULL,0x2748774cdf8eeb99ULL,0x34b0bcb5e19b48a8ULL,
        0x391c0cb3c5c95a63ULL,0x4ed8aa4ae3418acbULL,0x5b9cca4f7763e373ULL,0x682e6ff3d6b2b8a3ULL,
        0x748f82ee5defb2fcULL,0x78a5636f43172f60ULL,0x84c87814a1f0ab72ULL,0x8cc702081a6439ecULL,
        0x90befffa23631e28ULL,0xa4506cebde82bde9ULL,0xbef9a3f7b2c67915ULL,0xc67178f2e372532bULL,
        0xca273eceea26619cULL,0xd186b8c721c0c207ULL,0xeada7dd6cde0eb1eULL,0xf57d4f7fee6ed178ULL,
        0x06f067aa72176fbaULL,0x0a637dc5a2c898a6ULL,0x113f9804bef90daeULL,0x1b710b35131c471bULL,
        0x28db77f523047d84ULL,0x32caab7b40c72493ULL,0x3c9ebe0a15c9bebcULL,0x431d67c49c100d4cULL,
        0x4cc5d4becb3e42b6ULL,0x597f299cfc657e2aULL,0x5fcb6fab3ad6faecULL,0x6c44198c4a475817ULL
    };

    static inline uint64_t be64(const uint8_t* p) {
        return ((uint64_t)p[0]<<56)|((uint64_t)p[1]<<48)|((uint64_t)p[2]<<40)|((uint64_t)p[3]<<32)|
               ((uint64_t)p[4]<<24)|((uint64_t)p[5]<<16)|((uint64_t)p[6]<<8)|p[7];
    }
    static inline __m256i S0(__m256i x){return _mm256_xor_si256(_mm256_xor_si256(_mm256_srli_epi64(x,28),_mm256_slli_epi64(x,36)),_mm256_xor_si256(_mm256_srli_epi64(x,34),_mm256_xor_si256(_mm256_slli_epi64(x,30),_mm256_xor_si256(_mm256_srli_epi64(x,39),_mm256_slli_epi64(x,25)))));}
    static inline __m256i S1(__m256i x){return _mm256_xor_si256(_mm256_xor_si256(_mm256_srli_epi64(x,14),_mm256_slli_epi64(x,50)),_mm256_xor_si256(_mm256_srli_epi64(x,18),_mm256_xor_si256(_mm256_slli_epi64(x,46),_mm256_xor_si256(_mm256_srli_epi64(x,41),_mm256_slli_epi64(x,23)))));}
    static inline __m256i s0(__m256i x){
        __m256i a=_mm256_xor_si256(_mm256_srli_epi64(x,1),_mm256_slli_epi64(x,63));
        __m256i b=_mm256_xor_si256(_mm256_srli_epi64(x,8),_mm256_slli_epi64(x,56));
        __m256i c=_mm256_srli_epi64(x,7);
        return _mm256_xor_si256(_mm256_xor_si256(a,b),c);
    }
    static inline __m256i s1(__m256i x){
        __m256i a=_mm256_xor_si256(_mm256_srli_epi64(x,19),_mm256_slli_epi64(x,45));
        __m256i b=_mm256_xor_si256(_mm256_srli_epi64(x,61),_mm256_slli_epi64(x,3));
        __m256i c=_mm256_srli_epi64(x,6);
        return _mm256_xor_si256(_mm256_xor_si256(a,b),c);
    }
    static inline __m256i Ch(__m256i e,__m256i f,__m256i g){return _mm256_xor_si256(_mm256_and_si256(e,f),_mm256_andnot_si256(e,g));}
    static inline __m256i Maj(__m256i a,__m256i b,__m256i c){return _mm256_xor_si256(_mm256_and_si256(a,b),_mm256_xor_si256(_mm256_and_si256(a,c),_mm256_and_si256(b,c)));}

    static void process_block(const uint8_t m0[128],const uint8_t m1[128],
                              const uint8_t m2[128],const uint8_t m3[128],State& s) {
        __m256i W[80];
        for(int i=0;i<16;++i)
            W[i]=_mm256_set_epi64x(be64(m3+i*8),be64(m2+i*8),be64(m1+i*8),be64(m0+i*8));
        for(int i=16;i<80;++i)
            W[i]=_mm256_add_epi64(_mm256_add_epi64(s1(W[i-2]),W[i-7]),_mm256_add_epi64(s0(W[i-15]),W[i-16]));
        __m256i a=s.h[0],b=s.h[1],c=s.h[2],d=s.h[3],e=s.h[4],f=s.h[5],g=s.h[6],h=s.h[7];
        for(int i=0;i<80;++i){
            __m256i k=_mm256_set1_epi64x(K[i]);
            __m256i t1=_mm256_add_epi64(h,_mm256_add_epi64(S1(e),_mm256_add_epi64(Ch(e,f,g),_mm256_add_epi64(k,W[i]))));
            __m256i t2=_mm256_add_epi64(S0(a),Maj(a,b,c));
            h=g;g=f;f=e;e=_mm256_add_epi64(d,t1);d=c;c=b;b=a;a=_mm256_add_epi64(t1,t2);
        }
        s.h[0]=_mm256_add_epi64(s.h[0],a);s.h[1]=_mm256_add_epi64(s.h[1],b);
        s.h[2]=_mm256_add_epi64(s.h[2],c);s.h[3]=_mm256_add_epi64(s.h[3],d);
        s.h[4]=_mm256_add_epi64(s.h[4],e);s.h[5]=_mm256_add_epi64(s.h[5],f);
        s.h[6]=_mm256_add_epi64(s.h[6],g);s.h[7]=_mm256_add_epi64(s.h[7],h);
    }

    static void finalize_single(const State& s,int st,uint8_t out[64]) {
        alignas(32) uint64_t tmp[4];
        for(int i=0;i<8;++i) {
            _mm256_store_si256(reinterpret_cast<__m256i*>(tmp), s.h[i]);
            uint64_t v=tmp[st];
            out[i*8]=v>>56;out[i*8+1]=v>>48;out[i*8+2]=v>>40;out[i*8+3]=v>>32;
            out[i*8+4]=v>>24;out[i*8+5]=v>>16;out[i*8+6]=v>>8;out[i*8+7]=v;
        }
    }

    static void init_state(State& s) {
        s.h[0]=_mm256_set1_epi64x(0x6a09e667f3bcc908ULL);s.h[1]=_mm256_set1_epi64x(0xbb67ae8584caa73bULL);
        s.h[2]=_mm256_set1_epi64x(0x3c6ef372fe94f82bULL);s.h[3]=_mm256_set1_epi64x(0xa54ff53a5f1d36f1ULL);
        s.h[4]=_mm256_set1_epi64x(0x510e527fade682d1ULL);s.h[5]=_mm256_set1_epi64x(0x9b05688c2b3e6c1fULL);
        s.h[6]=_mm256_set1_epi64x(0x1f83d9abfb41bd6bULL);s.h[7]=_mm256_set1_epi64x(0x5be0cd19137e2179ULL);
    }

    // Carregar estado SHA-512 serial em State MB (todos os 4 streams = mesmo valor)
    static void set_state_from_serial(State& s, const uint64_t h[8]) {
        for(int i=0;i<8;++i) s.h[i]=_mm256_set1_epi64x(h[i]);
    }
};

// ============================================================
// PBKDF2-HMAC-SHA512 4-way
// 4 candidatos simultaneos, usando batched SHA-512 MB para
// o inner hash de cada HMAC
// ============================================================
inline void pbkdf2_hmac_sha512_4way(
    const char* pw0, size_t pl0, const char* pw1, size_t pl1,
    const char* pw2, size_t pl2, const char* pw3, size_t pl3,
    const uint8_t* salt, size_t salt_len,
    int iterations,
    uint8_t out0[64], uint8_t out1[64], uint8_t out2[64], uint8_t out3[64])
{
    // Pre-computar HMAC key states para cada stream
    struct HmacPrecomp {
        uint8_t ipad[128];
        uint8_t opad[128];
    };

    auto make_hmac_key = [](const uint8_t* key, size_t klen, HmacPrecomp& hk) {
        uint8_t kh[64];
        const uint8_t* ku; size_t ku_len;
        if(klen>128){SHA512::hash(key,klen,kh);ku=kh;ku_len=64;}
        else{ku=key;ku_len=klen;}
        memset(hk.ipad,0x36,128);memset(hk.opad,0x5c,128);
        for(size_t i=0;i<ku_len;++i){hk.ipad[i]^=ku[i];hk.opad[i]^=ku[i];}
    };

    HmacPrecomp hk[4];
    make_hmac_key(reinterpret_cast<const uint8_t*>(pw0),pl0,hk[0]);
    make_hmac_key(reinterpret_cast<const uint8_t*>(pw1),pl1,hk[1]);
    make_hmac_key(reinterpret_cast<const uint8_t*>(pw2),pl2,hk[2]);
    make_hmac_key(reinterpret_cast<const uint8_t*>(pw3),pl3,hk[3]);

    // Salt + block_be = message for U1
    uint8_t msg[132];
    memcpy(msg,salt,salt_len);
    msg[salt_len]=0;msg[salt_len+1]=0;msg[salt_len+2]=0;msg[salt_len+3]=1;

    uint8_t U[4][64],T[4][64];

    // PBKDF2: U1 = HMAC(password, salt||block), Ti = U1 ^ U2 ^ ... ^ Ui
    // HMAC(k,m) = H(opad || H(ipad || m))
    // We can batch the 4 inner hashes using SHA-512 MB

    // For U1: inner = SHA512(ipad_i || msg) for each stream i
    // ipad is 128 bytes, msg is salt_len+4 bytes
    // Total: 128 + 132 = 260 bytes = 2 blocks (256 + padding)

    // Block 1: ipad[0..127] (same structure but different per stream)
    // Block 2: msg[0..131] + padding

    // Process Block 1 for 4 streams simultaneously
    SHA512_MB::State inner_mb;
    SHA512_MB::init_state(inner_mb);

    // Block 1: each stream has different ipad
    SHA512_MB::process_block(hk[0].ipad, hk[1].ipad, hk[2].ipad, hk[3].ipad, inner_mb);

    // Block 2: msg + padding (same for all streams)
    uint8_t block2[128];
    memset(block2,0,128);
    memcpy(block2,msg,salt_len+4);
    // SHA-512 padding: append 1 bit, then zeros, then 64-bit big-endian bit length
    block2[salt_len+4]=0x80;
    uint64_t bit_len=(128+salt_len+4)*8;
    block2[120]=bit_len>>56;block2[121]=bit_len>>48;block2[122]=bit_len>>40;block2[123]=bit_len>>32;
    block2[124]=bit_len>>24;block2[125]=bit_len>>16;block2[126]=bit_len>>8;block2[127]=bit_len;

    // Same block2 for all 4 streams
    SHA512_MB::process_block(block2, block2, block2, block2, inner_mb);

    // Extract4 inner hashes
    for(int s=0;s<4;++s) SHA512_MB::finalize_single(inner_mb,s,U[s]);

    // Outer hash: SHA512(opad || inner_hash)
    SHA512_MB::State outer_mb;
    SHA512_MB::init_state(outer_mb);
    SHA512_MB::process_block(hk[0].opad, hk[1].opad, hk[2].opad, hk[3].opad, outer_mb);

    // Block2 for outer: inner_hash(64) + padding
    uint8_t oblock2[128];
    memset(oblock2,0,128);
    memcpy(oblock2,U[0],64);  // BUG: should be per-stream
    // Need per-stream block2 for outer hash since U values differ
    // This means we can't use MB for outer hash directly

    // Fall back to serial for outer hash
    for(int s=0;s<4;++s) {
        SHA512 o;
        o.update(hk[s].opad,128);
        o.update(U[s],64);
        o.finalize(U[s]);
    }

    for(int s=0;s<4;++s) memcpy(T[s],U[s],64);

    // U2..U2048
    for(int iter=1;iter<iterations;++iter) {
        // Inner hash: SHA512(ipad_i || U_i) — U_i differs per stream
        // So we can't batch inner either (different messages)
        // But we CAN batch if we use MB with different message blocks
        // Each stream has: ipad(128) + U(64) + padding = 2 blocks
        // Block1: ipad (different per stream) — batchable!
        // Block2: U(64) + padding (different per stream) — batchable!

        SHA512_MB::State imb;
        SHA512_MB::init_state(imb);
        SHA512_MB::process_block(hk[0].ipad, hk[1].ipad, hk[2].ipad, hk[3].ipad, imb);

        uint8_t ub2[4][128];
        for(int s=0;s<4;++s) {
            memset(ub2[s],0,128);
            memcpy(ub2[s],U[s],64);
            ub2[s][64]=0x80;
            uint64_t bl=(128+64)*8;
            ub2[s][120]=bl>>56;ub2[s][121]=bl>>48;ub2[s][122]=bl>>40;ub2[s][123]=bl>>32;
            ub2[s][124]=bl>>24;ub2[s][125]=bl>>16;ub2[s][126]=bl>>8;ub2[s][127]=bl;
        }
        SHA512_MB::process_block(ub2[0],ub2[1],ub2[2],ub2[3],imb);
        for(int s=0;s<4;++s) SHA512_MB::finalize_single(imb,s,U[s]);

        // Outer: serial (different opad states per stream)
        for(int s=0;s<4;++s) {
            SHA512 o;
            o.update(hk[s].opad,128);
            o.update(U[s],64);
            o.finalize(U[s]);
        }

        for(int s=0;s<4;++s) for(int j=0;j<64;++j) T[s][j]^=U[s][j];
    }

    memcpy(out0,T[0],64);memcpy(out1,T[1],64);
    memcpy(out2,T[2],64);memcpy(out3,T[3],64);
}

} // namespace crypto
