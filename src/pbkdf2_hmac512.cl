// PBKDF2-HMAC-SHA512 - OpenCL kernel with pre-computed ipad/opad states
// Compatible with PoCL (CPU) and rusticl (AMD GPU)

#define ROTR64(x, n) (((x) >> (n)) | ((x) << (64 - (n))))

__constant ulong K[80] = {
    0x428a2f98d728ae22UL,0x7137449123ef65cdUL,0xb5c0fbcfec4d3b2fUL,0xe9b5dba58189dbbcUL,
    0x3956c25bf348b538UL,0x59f111f1b605d019UL,0x923f82a4af194f9bUL,0xab1c5ed5da6d8118UL,
    0xd807aa98a3030242UL,0x12835b0145706fbeUL,0x243185be4ee4b28cUL,0x550c7dc3d5ffb4e2UL,
    0x72be5d74f27b896fUL,0x80deb1fe3b1696b1UL,0x9bdc06a725c71235UL,0xc19bf174cf692694UL,
    0xe49b69c19ef14ad2UL,0xefbe4786384f25e3UL,0x0fc19dc68b8cd5b5UL,0x240ca1cc77ac9c65UL,
    0x2de92c6f592b0275UL,0x4a7484aa6ea6e483UL,0x5cb0a9dcbd41fbd4UL,0x76f988da831153b5UL,
    0x983e5152ee66dfabUL,0xa831c66d2db43210UL,0xb00327c898fb213fUL,0xbf597fc7beef0ee4UL,
    0xc6e00bf33da88fc2UL,0xd5a79147930aa725UL,0x06ca6351e003826fUL,0x142929670a0e6e70UL,
    0x27b70a8546d22ffcUL,0x2e1b21385c26c926UL,0x4d2c6dfc5ac42aedUL,0x53380d139d95b3dfUL,
    0x650a73548baf63deUL,0x766a0abb3c77b2a8UL,0x81c2c92e47edaee6UL,0x92722c851482353bUL,
    0xa2bfe8a14cf10364UL,0xa81a664bbc423001UL,0xc24b8b70d0f89791UL,0xc76c51a30654be30UL,
    0xd192e819d6ef5218UL,0xd69906245565a910UL,0xf40e35855771202aUL,0x106aa07032bbd1b8UL,
    0x19a4c116b8d2d0c8UL,0x1e376c085141ab53UL,0x2748774cdf8eeb99UL,0x34b0bcb5e19b48a8UL,
    0x391c0cb3c5c95a63UL,0x4ed8aa4ae3418acbUL,0x5b9cca4f7763e373UL,0x682e6ff3d6b2b8a3UL,
    0x748f82ee5defb2fcUL,0x78a5636f43172f60UL,0x84c87814a1f0ab72UL,0x8cc702081a6439ecUL,
    0x90befffa23631e28UL,0xa4506cebde82bde9UL,0xbef9a3f7b2c67915UL,0xc67178f2e372532bUL,
    0xca273eceea26619cUL,0xd186b8c721c0c207UL,0xeada7dd6cde0eb1eUL,0xf57d4f7fee6ed178UL,
    0x06f067aa72176fbaUL,0x0a637dc5a2c898a6UL,0x113f9804bef90daeUL,0x1b710b35131c471bUL,
    0x28db77f523047d84UL,0x32caab7b40c72493UL,0x3c9ebe0a15c9bebcUL,0x431d67c49c100d4cUL,
    0x4cc5d4becb3e42b6UL,0x597f299cfc657e2aUL,0x5fcb6fab3ad6faecUL,0x6c44198c4a475817UL
};

inline void sha512_compress(ulong *st, const ulong *M) {
    ulong W[80];
    for (int i = 0; i < 16; i++) W[i] = M[i];
    for (int i = 16; i < 80; i++)
        W[i] = (ROTR64(W[i-2],19)^ROTR64(W[i-2],61)^(W[i-2]>>6)) + W[i-7]
              + (ROTR64(W[i-15],1)^ROTR64(W[i-15],8)^(W[i-15]>>7)) + W[i-16];
    ulong a=st[0],b=st[1],c=st[2],d=st[3],e=st[4],f=st[5],g=st[6],h=st[7];
    for (int i = 0; i < 80; i++) {
        ulong Sigma1_e = ROTR64(e,14) ^ ROTR64(e,18) ^ ROTR64(e,41);
        ulong Ch_efg = (e & f) ^ ((~e) & g);
        ulong T1 = h + Sigma1_e + Ch_efg + K[i] + W[i];
        ulong T2 = (ROTR64(a,28)^ROTR64(a,34)^ROTR64(a,39)) + ((a&b)^(a&c)^(b&c));
        h=g; g=f; f=e; e=d+T1; d=c; c=b; b=a; a=T1+T2;
    }
    st[0]+=a; st[1]+=b; st[2]+=c; st[3]+=d;
    st[4]+=e; st[5]+=f; st[6]+=g; st[7]+=h;
}

// Load 16 ulongs from a __private uchar[128] buffer (inline, no address space issues)
inline void load16_private(ulong *M, const uchar *src) {
    for (int i = 0; i < 16; i++) {
        int o = i * 8;
        M[i] = ((ulong)src[o]<<56)|((ulong)src[o+1]<<48)|((ulong)src[o+2]<<40)|((ulong)src[o+3]<<32)|
               ((ulong)src[o+4]<<24)|((ulong)src[o+5]<<16)|((ulong)src[o+6]<<8)|(ulong)src[o+7];
    }
}

// Store 8 ulongs to a __private uchar[64] buffer
inline void store64(ulong *s, uchar *out) {
    for (int i = 0; i < 8; i++) {
        out[i*8]=(uchar)(s[i]>>56); out[i*8+1]=(uchar)(s[i]>>48);
        out[i*8+2]=(uchar)(s[i]>>40); out[i*8+3]=(uchar)(s[i]>>32);
        out[i*8+4]=(uchar)(s[i]>>24); out[i*8+5]=(uchar)(s[i]>>16);
        out[i*8+6]=(uchar)(s[i]>>8); out[i*8+7]=(uchar)s[i];
    }
}

// Load 16 ulongs from a __global uchar* buffer (with offset)
inline void load16_global(ulong *M, __global const uchar *src) {
    for (int i = 0; i < 16; i++) {
        int o = i * 8;
        M[i] = ((ulong)src[o]<<56)|((ulong)src[o+1]<<48)|((ulong)src[o+2]<<40)|((ulong)src[o+3]<<32)|
               ((ulong)src[o+4]<<24)|((ulong)src[o+5]<<16)|((ulong)src[o+6]<<8)|(ulong)src[o+7];
    }
}

// Load ipad/opad state from __global buffer (64 bytes per work-item, big-endian)
inline void load_state(ulong *st, __global const uchar *buf, uint gid) {
    for (int i = 0; i < 8; i++) {
        int off = gid * 64 + i * 8;
        st[i] = ((ulong)buf[off]<<56)|((ulong)buf[off+1]<<48)|((ulong)buf[off+2]<<40)|((ulong)buf[off+3]<<32)|
                ((ulong)buf[off+4]<<24)|((ulong)buf[off+5]<<16)|((ulong)buf[off+6]<<8)|(ulong)buf[off+7];
    }
}

__kernel void pbkdf2_sha512(
    __global const uchar *ipad_state,
    __global const uchar *opad_state,
    __global const uchar *salt_padded,
    uint salt_padded_len,
    uint salt_blocks,
    uint iterations,
    uint dk_len_bytes,
    __global uchar *output)
{
    uint gid = get_global_id(0);

    ulong ipad[8], opad[8];
    load_state(ipad, ipad_state, gid);
    load_state(opad, opad_state, gid);

    // U1: inner hash = HMAC inner pass on salt
    ulong inner[8];
    for (int i=0;i<8;i++) inner[i]=ipad[i];
    ulong M[16];
    for (uint b=0; b<salt_blocks; b++) {
        load16_global(M, salt_padded + b*128);
        sha512_compress(inner, M);
    }
    uchar inner_hash[64];
    store64(inner, inner_hash);

    // Outer hash on inner_hash (64 bytes)
    uchar block[128];
    for (int i=0;i<64;i++) block[i]=inner_hash[i];
    block[64]=0x80;
    for (int i=65;i<128;i++) block[i]=0;
    // Bit length: (128 + 64) * 8 = 1536 = 0x0600 (opad_block + inner_hash)
    block[126]=0x06; block[127]=0x00;

    ulong outer[8];
    for (int i=0;i<8;i++) outer[i]=opad[i];
    load16_private(M, block);
    sha512_compress(outer, M);
    uchar U[64];
    store64(outer, U);

    uchar T[64];
    for (int i=0;i<64;i++) T[i]=U[i];

    // Subsequent iterations: U_i = HMAC(password, U_{i-1})
    for (uint iter=1; iter<iterations; iter++) {
        // Inner: ipad + U_{i-1}
        for (int i=0;i<8;i++) inner[i]=ipad[i];
        for (int i=0;i<64;i++) block[i]=U[i];
        block[64]=0x80;
        for (int i=65;i<128;i++) block[i]=0;
        block[126]=0x06; block[127]=0x00;
        load16_private(M, block);
        sha512_compress(inner, M);
        store64(inner, inner_hash);

        // Outer: opad + inner_hash
        for (int i=0;i<8;i++) outer[i]=opad[i];
        for (int i=0;i<64;i++) block[i]=inner_hash[i];
        block[64]=0x80;
        for (int i=65;i<128;i++) block[i]=0;
        block[126]=0x06; block[127]=0x00;
        load16_private(M, block);
        sha512_compress(outer, M);
        store64(outer, U);

        for (int i=0;i<64;i++) T[i]^=U[i];
    }

    // Write output
    __global uchar *out_ptr = output + gid * dk_len_bytes;
    for (uint i=0; i<dk_len_bytes && i<64; i++) out_ptr[i]=T[i];
}
