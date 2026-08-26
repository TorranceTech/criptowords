#pragma once

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <immintrin.h>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <sstream>
#include <iomanip>
#include <ios>
#include <charconv>

// Core modules
#include "Division.hpp"
#include "Multiplication.hpp"

// Type Aliases
using namespace std;
using u8 = unsigned char;
using u16 = unsigned short;
using u64 = unsigned long long;
using u128 = unsigned __int128;

#define BUILTIN_EXPECT(x, y) (__builtin_expect(!!(x), y))
#define FORCE_INLINE inline __attribute__((always_inline))

template <u8 N>
class alignas(64) UInt {
    static_assert(N > 0, "Limbs must be positive");

    private:
        static FORCE_INLINE constexpr uint64_t rand(u64& state) noexcept {
            state += 0x9e3779b97f4a7c15ULL;
            uint64_t result = state;
            result = (result ^ (result >> 30)) * 0xbf58476d1ce4e5b9ULL;
            result = (result ^ (result >> 27)) * 0x94d049bb133111ebULL;
            return result ^ (result >> 31);
        }

        [[nodiscard]] FORCE_INLINE u8 num_limbs() const noexcept {
            for (int i = N - 1; i >= 0; --i) {
                if (bits[i] != 0)
                    return i + 1;
            }
            return 0;
        }

    public:
        array<u64, N> bits = {}; // Inicializa tudo em 0 por padrão

        constexpr UInt() noexcept = default;
        constexpr UInt(const UInt &o) noexcept = default;
        constexpr UInt(UInt &&o) noexcept = default;

        constexpr UInt(u64 value) noexcept : bits{value} {}

        constexpr explicit UInt(string_view sv) {
            if (sv.empty()) return;
            if (sv.front() == '-') throw invalid_argument("Negative values not supported in UInt");
            if (sv.front() == '+') sv.remove_prefix(1);

            if (sv.starts_with("0x") || sv.starts_with("0X")) {
                sv.remove_prefix(2);
                if (sv.empty()) return;
                if (sv.size() > (N * 16)) throw out_of_range("Hex string too long");

                size_t len = sv.size();
                for (u8 i = 0; i < N && len > 0; ++i) {
                    size_t chunk_sz = min<size_t>(16, len);
                    size_t start = len - chunk_sz;
                    u64 chunk_val = 0;

                    auto res = from_chars(sv.data() + start, sv.data() + start + chunk_sz, chunk_val, 16);
                    if (res.ec != errc{}) throw invalid_argument("Invalid hex character");

                    bits[i] = chunk_val;
                    len -= chunk_sz;
                }
            } else {
                // C++26 permite literais numéricos com ' e é 100% constexpr para from_chars
                constexpr u64 CHUNK_POW = 1'000'000'000'000'000'000ull;
                constexpr int CHUNK_SIZE = 18;

                size_t remaining = sv.size() % CHUNK_SIZE;
                if (remaining == 0 && !sv.empty()) remaining = CHUNK_SIZE;

                u64 chunk_val = 0;
                auto res = from_chars(sv.data(), sv.data() + remaining, chunk_val, 10);
                if (res.ec != errc{}) throw invalid_argument("Invalid decimal character");
                bits[0] = chunk_val;

                for (size_t i = remaining; i < sv.size(); i += CHUNK_SIZE) {
                    from_chars(sv.data() + i, sv.data() + i + CHUNK_SIZE, chunk_val, 10);
                    *this *= CHUNK_POW;
                    *this += chunk_val;
                }
            }
        }

        template <std::convertible_to<u64>... Args>
        requires (sizeof...(Args) > 0 && sizeof...(Args) <= N)
        constexpr UInt(Args... args) noexcept : bits{static_cast<u64>(args)...} {}

        constexpr UInt &operator=(const UInt &o) noexcept = default;
        constexpr UInt &operator=(UInt &&o) noexcept = default;

        constexpr FORCE_INLINE UInt &operator=(u64 o) noexcept {
            bits[0] = o;
            if constexpr (N > 1) {
                std::fill(bits.begin() + 1, bits.end(), 0ull);
            }
            return *this;
        }

        // comparators
        [[nodiscard]] constexpr bool operator==(const UInt &other) const noexcept = default;

        [[nodiscard]] FORCE_INLINE constexpr auto operator<=>(const UInt &other) const noexcept {
            return std::lexicographical_compare_three_way(
                bits.rbegin(), bits.rend(),
                other.bits.rbegin(), other.bits.rend()
            );
        }

        [[nodiscard]] FORCE_INLINE constexpr bool operator==(const u64 o) const noexcept {
            if (bits[0] != o) return false;
            if constexpr (N > 1) {
                return std::ranges::all_of(bits.begin() + 1, bits.end(), [](u64 v) { return v == 0; });
            }
            return true;
        }

        [[nodiscard]] FORCE_INLINE constexpr auto operator<=>(const u64 o) const noexcept {
            if constexpr (N > 1) {
                auto it = std::ranges::find_if(bits.rbegin(), bits.rend() - 1, [](u64 v) { return v != 0; });
                if (it != bits.rend() - 1) return std::strong_ordering::greater;
            }
            return bits[0] <=> o;
        }

        FORCE_INLINE constexpr UInt &operator++() noexcept { // melhor forma. testado!
            u8 i = 0;
            do {if (++bits[i] == 0) [[unlikely]] ++i;
                else return *this;
            }while (i < N);
            return *this;
        }
        FORCE_INLINE constexpr UInt &operator--() noexcept { // melhor forma. testado!
            u8 i = 0;
            do {if (--bits[i] == -1ull) [[unlikely]] ++i;
                else return *this;
            }while (i < N);
            return *this;
        }
        FORCE_INLINE constexpr UInt operator++(int) noexcept {// melhor forma. testado!
            UInt t = *this;
            ++*this;
            return t;
        }
        FORCE_INLINE constexpr UInt operator--(int) noexcept {// melhor forma. testado!
            UInt t = *this;
            --*this;
            return t;
        }

        [[nodiscard]] FORCE_INLINE constexpr u16 lzc() const noexcept {
            for (int i = N - 1; i >= 0; --i) {
                if (bits[i] != 0) {
                    return _lzcnt_u64(bits[i]) + (N - 1 - i) * 64;
                }
            }
            return N * 64; // Retorna o total de bits caso o número seja totalmente zero
        }

        constexpr FORCE_INLINE UInt &operator+=(const UInt &other) noexcept { // melhor forma. testado!
            if consteval {
                u64 carry = 0;
                for (u8 i = 0; i < N; ++i) {
                    u128 sum = static_cast<u128>(bits[i]) + other.bits[i] + carry;
                    bits[i] = static_cast<u64>(sum);
                    carry = sum >> 64;
                }
            } else {
                asm volatile(R"(
                    .set offset, 0
                    movq offset(%[src]), %%rax
                    addq %%rax, offset(%[dst])
                    .set offset, offset+8
                    .rept %c[count]
                        movq offset(%[src]), %%rax
                        adcq %%rax, offset(%[dst])
                        .set offset, offset+8
                    .endr
                )"
                    : "+m"(bits)
                    : [dst]   "r"(bits.data()),
                    [src]   "r"(other.bits.data()),
                    [count] "n"(N - 1),
                    "m"(other.bits)
                    : "rax", "cc"
                );
            }
            return *this;
        }

        constexpr FORCE_INLINE UInt &operator+=(u64 val) noexcept { // melhor forma. testado!
            if ((bits[0] += val) < val)[[unlikely]]{
                for (u8 i = 1; i < N; ++i) {
                    if (++bits[i] != 0) [[likely]] return *this;
                }
            }
            return *this;
        }

        constexpr FORCE_INLINE UInt &operator-=(const UInt &other) noexcept { // melhor forma. testado!
            if consteval {
                u64 carry = 0;
                for (u8 i = 0; i < N; ++i) {
                    u128 sub = static_cast<u128>(bits[i]) - other.bits[i] - carry;
                    bits[i] = static_cast<u64>(sub);
                    // borrow: em u128 a subtração com empréstimo envolve (2^128 - algo),
                    // então `sub >> 64` vira 2^64-1 (não 1) — precisa mascarar.
                    carry = (sub >> 64) & 1;
                }
            } else {
                asm volatile(R"(
                    .set offset, 0
                    movq offset(%[src]), %%rax
                    subq %%rax, offset(%[dst])
                    .set offset, offset+8
                    .rept %c[count]
                        movq offset(%[src]), %%rax
                        sbbq %%rax, offset(%[dst])
                        .set offset, offset+8
                    .endr
                )"
                    : "+m"(bits)
                    : [dst]   "r"(bits.data()),
                    [src]   "r"(other.bits.data()),
                    [count] "n"(N - 1),
                    "m"(other.bits)
                    : "rax", "cc"
                );
            }
            return *this;
        }

        constexpr FORCE_INLINE UInt &operator-=(const u64 val) noexcept {
            u64 old = bits[0];
            bits[0] -= val;
            if (bits[0] > old) [[unlikely]] {
                for (u8 i = 1; i < N; ++i) {
                    if (--bits[i] != -1ull) [[likely]] return *this;
                }
            }
            return *this;
        }

        // Runtime dispatch — non-constexpr to prevent GCC's if consteval
        // miscompilation (GCC 15 bug: if consteval treated as always-true
        // in constexpr FORCE_INLINE functions).
        FORCE_INLINE UInt &mul_runtime(const UInt &other) noexcept {
            if constexpr (N == 1) {
                bits[0] *= other.bits[0];
                return *this;
            }
            else if constexpr (N <= 8) {
                u64 p_res[N];
                if constexpr (N == 1) {
                    bits[0] *= other.bits[0];
                } else if constexpr (N == 2) {
                    u64 a0 = bits[0], a1 = bits[1];
                    u64 b0 = other.bits[0], b1 = other.bits[1];
                    u64 lo_ab, hi_ab;
                    lo_ab = _mulx_u64(a0, b0, &hi_ab);
                    u64 t1 = a0 * b1;
                    u64 t2 = a1 * b0;
                    bits[0] = lo_ab;
                    bits[1] = t1 + hi_ab + t2;
                } else if constexpr (N == 3) {
                    u64 a0 = bits[0], a1 = bits[1], a2 = bits[2];
                    u64 b0 = other.bits[0], b1 = other.bits[1], b2 = other.bits[2];
                    u64 r0, r1, r2;
                    __asm__ volatile (R"(
                        movq   %[b0], %%rdx
                        mulxq  %[a0], %[r0], %[r1]
                        xorl   %k[r2], %k[r2]
                        movq   %[b2], %%r10
                        imulq  %[a0], %%r10
                        movq   %[b1], %%rdx
                        mulxq  %[a0], %%r11, %%rcx
                        addq   %%r11, %[r1]
                        adcq   %%rcx, %[r2]
                        movq   %[a2], %%rcx
                        movq   %[a1], %%rdx
                        mulxq  %[b0], %%r11, %%r9
                        imulq  %[b0], %%rcx
                        addq   %%r11, %[r1]
                        adcq   %%r9, %[r2]
                        imulq  %[b1], %%rdx
                        addq   %%rdx, %%r10
                        addq   %%rcx, %%r10
                        addq   %%r10, %[r2]
                    )"

                        : [r0] "=&r" (r0), [r1] "=&r" (r1), [r2] "=&r" (r2)
                        : [a0] "rm" (a0), [a1] "rm" (a1), [a2] "rm" (a2),
                        [b0] "rm" (b0), [b1] "rm" (b1), [b2] "rm" (b2)
                        : "rdx", "r9", "r10", "r11", "rcx", "cc"
                    );
                    bits[0] = r0;
                    bits[1] = r1;
                    bits[2] = r2;
                } else if constexpr (N == 8) {
                    if (this == &other) {
                        Multiplication::square_schoolbook_truncated_fixed<8>(p_res, &bits[0]);
                    } else {
                        Multiplication::mul_split_truncated_fixed_8(p_res, &bits[0], &other.bits[0]);
                    }
                    copy_n(p_res, N, &bits[0]);
                } else if constexpr (N == 4) {
#define MUL4_ASM_CORE \
    "xorl   %%r9d, %%r9d\n\t" \
    "xorl   %%eax, %%eax\n\t" \
    "xorl   %%r10d, %%r10d\n\t" \
    "xorl   %%r8d, %%r8d\n\t" \
    "movq   %[a0], %%rdx\n\t" \
    "mulxq  %[b0], %%rcx, %%rsi\n\t" \
    "addq   %%rcx, %%r9\n\t" \
    "adcq   %%rsi, %%rax\n\t" \
    "mulxq  %[b1], %%rcx, %%rsi\n\t" \
    "addq   %%rcx, %%rax\n\t" \
    "adcq   %%rsi, %%r10\n\t" \
    "adcq   $0, %%r8\n\t" \
    "mulxq  %[b2], %%rcx, %%rsi\n\t" \
    "addq   %%rcx, %%r10\n\t" \
    "adcq   %%rsi, %%r8\n\t" \
    "imulq  %[b3], %%rdx\n\t" \
    "addq   %%rdx, %%r8\n\t" \
    "movq   %[a1], %%rdx\n\t" \
    "mulxq  %[b0], %%rcx, %%rsi\n\t" \
    "addq   %%rcx, %%rax\n\t" \
    "adcq   %%rsi, %%r10\n\t" \
    "adcq   $0, %%r8\n\t" \
    "mulxq  %[b1], %%rcx, %%rsi\n\t" \
    "addq   %%rcx, %%r10\n\t" \
    "adcq   %%rsi, %%r8\n\t" \
    "imulq  %[b2], %%rdx\n\t" \
    "addq   %%rdx, %%r8\n\t" \
    "movq   %[a2], %%rdx\n\t" \
    "mulxq  %[b0], %%rcx, %%rsi\n\t" \
    "addq   %%rcx, %%r10\n\t" \
    "adcq   %%rsi, %%r8\n\t" \
    "imulq  %[b1], %%rdx\n\t" \
    "addq   %%rdx, %%r8\n\t" \
    "movq   %[a3], %%rdx\n\t" \
    "imulq  %[b0], %%rdx\n\t" \
    "addq   %%rdx, %%r8\n\t"

                    __asm__ volatile (
                        MUL4_ASM_CORE
                        "movq   %%r9, %[a0]\n\t"
                        "movq   %%rax, %[a1]\n\t"
                        "movq   %%r10, %[a2]\n\t"
                        "movq   %%r8, %[a3]\n\t"
                        : [a0] "+m" (bits[0]), [a1] "+m" (bits[1]),
                        [a2] "+m" (bits[2]), [a3] "+m" (bits[3])
                        : [b0] "m" (other.bits[0]), [b1] "m" (other.bits[1]),
                        [b2] "m" (other.bits[2]), [b3] "m" (other.bits[3])
                        : "rax", "rcx", "rdx", "rsi", "r8", "r9", "r10", "cc"
                    );
                } else if (this == &other) {
                    Multiplication::square_schoolbook_truncated_fixed<N>(p_res, &bits[0]);
                    copy_n(p_res, N, &bits[0]);
                } else {
                    Multiplication::mul_schoolbook_truncated_fixed<N>(p_res, &bits[0], &other.bits[0]);
                    copy_n(p_res, N, &bits[0]);
                }

            } else if constexpr (N <= 16) {
                u64 buf[N];
                if (this == &other) {
                    Multiplication::square_schoolbook_truncated_fixed<N>(buf, &bits[0]);
                } else {
                    Multiplication::mul_schoolbook_truncated_fixed<N>(buf, &bits[0],
                                            &other.bits[0]);
                }
                copy_n(buf, N, &bits[0]);
            } else {
                alignas(64) u64 tmp[8 * N + 1000];
                alignas(64) u64 buf[N];
                if (this == &other) {
                    Multiplication::square_truncated_fixed<N>(buf, &bits[0], tmp);
                } else {
                    Multiplication::mul_truncated_fixed<N>(buf, &bits[0],
                                            &other.bits[0], tmp);
                }
                copy_n(buf, N, &bits[0]);
            }
            return *this;
        }

        // Consteval-only schoolbook (public method for compile-time multiplication)
        constexpr UInt &mul_consteval(const UInt &other) noexcept {
            UInt<N> self_copy = *this;
            this->bits.fill(0);
            for (u8 i = 0; i < N; ++i) {
                u64 y = self_copy.bits[i];
                if (y == 0) continue;
                u128 carry = 0;
                for (u8 j = 0; j < N - i; ++j) {
                    u128 temp = (u128)other.bits[j] * y + this->bits[i + j] + carry;
                    this->bits[i + j] = (u64)temp;
                    carry = temp >> 64;
                }
            }
            return *this;
        }

        FORCE_INLINE UInt &operator*=(const UInt &other) noexcept {
            return mul_runtime(other);
        }

        constexpr FORCE_INLINE UInt &operator*=(u64 val) noexcept {
            u64 c = 0;
            for (u8 i = 0; i < N; ++i) {
                u128 temp = (u128)bits[i] * val + c;
                bits[i] = (u64)temp;
                c = temp >> 64;
            }
            return *this;
        }

        FORCE_INLINE UInt &operator/=(u64 val) {
            if (val == 0) throw domain_error("Division by zero");
            u128 rem = 0;
            for (int i = N - 1; i >= 0; --i) {
                u128 cur = bits[i] | (rem << 64);
                bits[i] = (u64)(cur / val);
                rem = cur % val;
            }
            return *this;
        }

        FORCE_INLINE UInt &operator%=(u64 val) {
            if (val == 0) throw domain_error("Division by zero");
            u128 rem = 0;
            for (int i = N - 1; i >= 0; --i) {
                u128 cur = bits[i] | (rem << 64);
                rem = cur % val;
            }
            fill(bits.begin(), bits.end(), 0);
            bits[0] = (u64)rem;
            return *this;
        }

        constexpr FORCE_INLINE UInt &operator%=(const UInt &other) {
            if (other == 0) [[unlikely]] throw domain_error("Division by zero");
            if (*this < other) return *this;
            else [[likely]] {
                UInt temp = other;
                temp <<= (other.lzc() - this->lzc());
                do {
                    if (temp > *this)[[unlikely]] temp>>=1;
                    *this -= temp;
                    temp >>= (this->lzc() - temp.lzc());
                }while (*this >= other);
            }
            return *this;
        }

        [[nodiscard]] FORCE_INLINE constexpr static UInt<N> random(const u64 seed = 0) noexcept;
        [[nodiscard]] static constexpr pair<UInt<N>, UInt<N>> divmod(UInt<N> u, UInt<N> v);

        FORCE_INLINE UInt<N> &operator/=(const UInt<N> &other) {
            if (other == 0) [[unlikely]] throw domain_error("Division by zero");
            if (other > *this) [[unlikely]] { this->bits.fill(0); return *this; }
            if (other == 1) [[unlikely]] return *this;
            if (other == *this) [[unlikely]] return *this = 1;

            if constexpr (N == 2) {
                u64 u0 = bits[0], u1 = bits[1];
                u64 v0 = other.bits[0], v1 = other.bits[1];

                if (v1 == 0 && v0 == 0) [[unlikely]]
                    throw domain_error("Division by zero");

                if (v1 == 0) [[unlikely]] {
                    u64 q1, r1;
                    __asm__("divq %[v]" : "=a"(q1), "=d"(r1) : "a"(u1), "d"(0), [v]"rm"(v0) : "cc");
                    u64 q0;
                    __asm__("divq %[v]" : "=a"(q0), "=d"(r1) : "a"(u0), "d"(r1), [v]"rm"(v0) : "cc");
                    bits[0] = q0; bits[1] = q1;
                    return *this;
                }

                if (u1 < v1 || (u1 == v1 && u0 < v0)) [[unlikely]] {
                    bits[0] = 0; bits[1] = 0;
                    return *this;
                }

                int shift = __builtin_clzll(v1);

                if (shift == 0) [[likely]] {
                    unsigned char b = 0;
                    b = _subborrow_u64(b, u0, v0, &u0);
                    b = _subborrow_u64(b, u1, v1, &u1);
                    bits[0] = 1;
                    bits[1] = 0;
                    return *this;
                }

                u64 v1n = (v1 << shift) | (v0 >> (64 - shift));
                u64 v0n = v0 << shift;
                u64 u2n = u1 >> (64 - shift);
                u64 u1n = (u1 << shift) | (u0 >> (64 - shift));
                u64 u0n = u0 << shift;

                u64 q0;
                u128 r_hat;
                {
                    u64 rl;
                    __asm__("divq %[v1n]"
                        : "=a"(q0), "=d"(rl)
                        : "a"(u1n), "d"(u2n), [v1n]"rm"(v1n)
                        : "cc");
                    r_hat = rl;
                }

                u128 p0 = (u128)q0 * v0n;

                if (p0 > ((r_hat << 64) | u0n)) {
                    q0--;
                    r_hat += v1n;
                    p0 = (u128)q0 * v0n;
                    if (r_hat < v1n && p0 > ((r_hat << 64) | u0n)) {
                        q0--;
                        p0 = (u128)q0 * v0n;
                    }
                }

                {
                    u128 p1 = (u128)q0 * v1n + (p0 >> 64);
                    unsigned char b = 0;
                    b = _subborrow_u64(b, u0n, (u64)p0, &u0n);
                    b = _subborrow_u64(b, u1n, (u64)p1, &u1n);
                    b = _subborrow_u64(b, u2n, p1 >> 64, &u2n);

                    if (b) [[unlikely]] {
                        q0--;
                        unsigned char c = 0;
                        c = _addcarry_u64(c, u0n, v0n, &u0n);
                        c = _addcarry_u64(c, u1n, v1n, &u1n);
                    }
                }

                bits[0] = q0;
                bits[1] = 0;
                return *this;
            }
            *this = divmod(*this, other).first;
            return *this;
        }

        [[nodiscard]] constexpr FORCE_INLINE UInt<N> operator%(const UInt<N> &other) const {
            return divmod(*this, other).second;
        }
        FORCE_INLINE constexpr UInt &operator&=(const UInt &other) noexcept {
            for (u8 i = 0; i < N; ++i)
                bits[i] &= other.bits[i];
            return *this;
        }
        FORCE_INLINE constexpr UInt &operator|=(const UInt &other) noexcept {
            for (u8 i = 0; i < N; ++i)
                bits[i] |= other.bits[i];
            return *this;
        }
        FORCE_INLINE constexpr UInt &operator^=(const UInt &other) noexcept {
            for (u8 i = 0; i < N; ++i)
                bits[i] ^= other.bits[i];
            return *this;
        }
        FORCE_INLINE constexpr UInt &operator<<=(u16 n) noexcept;
        FORCE_INLINE constexpr UInt &operator>>=(u16 n) noexcept;

        [[nodiscard]] string to_string() const;
        [[nodiscard]] string to_hex_string() const;

        //eqz : equals zero?
        [[nodiscard]] FORCE_INLINE constexpr bool eqz() const noexcept {
            for (u64 limb : bits)
                if (limb != 0)
                    return false;
            return true;
        }
        [[nodiscard]] FORCE_INLINE constexpr bool bt(const u16 index) const noexcept {
            if (BUILTIN_EXPECT(index >= N * 64, 0))
                return false;
            return (bits[index / 64] >> (index % 64)) & 1;
        }
        //bit test and set
        FORCE_INLINE constexpr void bts(const u16 index) noexcept {
            if (BUILTIN_EXPECT(index < N * 64, 1))
                bits[index / 64] |= (1ull << (index % 64));
        }

        // Montgomery context — pré-computa o inverso do módulo (p_inv) e a
        // constante R² mod p uma única vez para reuso em múltiplas reduções.
        struct MontgomeryCtx {
            u64 p_inv = 1;
            UInt r2{}; // R² mod p, R = 2^(64*N)
        };

        // Converte um número (já reduzido: < p) para forma de Montgomery: x·R mod p.
        FORCE_INLINE void to_mont(UInt &x, const UInt &p,
                                  const MontgomeryCtx &ctx) noexcept {
            // mont_mul(x, R²) = x·R²·R⁻¹ = x·R (mod p)
            mul_mod_mont(x, ctx.r2, p, ctx);
        }

        // Converte da forma de Montgomery para a forma natural: x·R⁻¹ mod p.
        FORCE_INLINE void from_mont(UInt &x, const UInt &p,
                                    const MontgomeryCtx &ctx) noexcept {
            // mont_mul(x, 1) = x·1·R⁻¹ = x·R⁻¹ (mod p)
            UInt one = 1;
            mul_mod_mont(x, one, p, ctx);
        }

        // x = (x * y * 2^-k) mod p via Montgomery (p ímpar de N limbs).
        // Ambas as entradas E a saída estão em forma de Montgomery.
        // ctx deve ter sido preenchido por montgomery_precompute(p).
        FORCE_INLINE void mul_mod_mont(UInt &x, const UInt &y, const UInt &p,
                                       const MontgomeryCtx &ctx) noexcept {
            alignas(64) u64 t[2 * N + 2];   // acumulador: produto x·y é SOMADO a partir de 0
            alignas(64) u64 xbuf[N];        // operando x (não pode aliasing t)
            alignas(64) u64 ybuf[N];        // operando y (não pode aliasing t)
            for (u8 i = 0; i < 2 * N + 2; ++i) t[i] = 0;
            for (u8 i = 0; i < N; ++i) { xbuf[i] = x.bits[i]; ybuf[i] = y.bits[i]; }
            Multiplication::mul_montgomery_cios<N>(t, xbuf, ybuf, p.bits.data(),
                                                   ctx.p_inv, 64 * N);
            for (u8 i = 0; i < N; ++i) x.bits[i] = t[i];
        }

        // Pré-computa o contexto Montgomery para o módulo p (p ímpar).
        // Custo: UMA divisão longa (R² = 2^(128N) mod p) — amortizada sobre
        // centenas de mul_mod no invMod, então desprezível.
        FORCE_INLINE MontgomeryCtx montgomery_precompute(const UInt &p) noexcept {
            MontgomeryCtx ctx;
            ctx.p_inv = Multiplication::montgomery_p_inv(p.bits[0]);
            // R² mod p = 2^(128N) mod p (R = 2^(64N)). Dividimos o número de
            // 2N+1 limbs (big[2N] = 1) por p e guardamos o resto em r2.
            alignas(64) u64 big[2 * N + 1];
            for (u8 i = 0; i < 2 * N + 1; ++i) big[i] = 0;
            big[2 * N] = 1;
            int n_v = p.num_limbs();
            if (n_v == 1) {
                u64 rem = 0;
                for (int i = 2 * N; i >= 0; --i) {
                    u128 cur = ((u128)rem << 64) | big[i];
                    rem = (u64)(cur % p.bits[0]);
                }
                ctx.r2.bits[0] = rem;
            } else {
                u64 qq[2 * N + 4];
                u64 rr[N];
                Division::div_knuth_impl<2 * N + 4>(qq, rr, big, 2 * N + 1, p.bits.data(), n_v);
                for (u8 i = 0; i < N; ++i) ctx.r2.bits[i] = rr[i];
            }
            return ctx;
        }

        // seta this no proprio inverso modular: this⁻¹ (mod p).
        // assume que p é primo e que *this não é múltiplo de p.
        // Pequeno teorema de Fermat: a⁻¹ ≡ a^(p-2) (mod p).
        //
        // Para p ímpar usa Montgomery (redução por multiplicação, sem divisão
        // no loop — apenas uma divisão na pré-computação de R² mod p). Para p
        // par caímos no caminho genérico (divmod em UInt<2N>).
        FORCE_INLINE UInt &invMod(const UInt &p) noexcept{
            UInt base = *this % p;
            UInt result = 1;
            UInt exp = p;
            --exp; // p - 1
            --exp; // p - 2

            // Montgomery exige p ímpar E full-width (n_v == N): a subtração
            // condicional reduz < 2^(64N), e o resto da divisão longa de R²
            // tem exatamente N limbs. Caso contrário, caminho genérico.
            const bool mont = (p.bits[0] & 1) != 0 && p.bits[N - 1] != 0;
            MontgomeryCtx ctx;
            if (mont) {
                ctx = montgomery_precompute(p);
                // trabalha o loop inteiro em forma Montgomery: base e result
                // precisam estar em forma Montgomery (x·R mod p)
                to_mont(base, p, ctx);
                to_mont(result, p, ctx);
            }

            while (!exp.eqz()) {
                if (exp.bt(0)) {
                    if (mont) mul_mod_mont(result, base, p, ctx);
                    else      mul_mod(result, base, p);
                }
                exp >>= 1;
                if (!exp.eqz()) {
                    if (mont) mul_mod_mont(base, base, p, ctx);
                    else      mul_mod(base, base, p);
                }
            }
            if (mont) from_mont(result, p, ctx);
            return *this = result;
        }

        // x = (x * y) mod p, sem truncar o produto (correto para p multi-limb).
        // Caminho genérico (produto exato em 2N limbs + divisão): mais rápido que
        // Montgomery para N ≤ ~96 (a divisão por limb é barata; o precompute de
        // Montgomery por chamada não compensa). O invMod usa Montgomery por cima
        // (contexto pré-computado UMA vez, amortizado no loop de exponenciação).
        FORCE_INLINE void mul_mod(UInt &x, const UInt &y, const UInt &p) noexcept {
            if constexpr (2 * N <= 255) {
                // Intermediário largo: produto exato em 2N limbs, reduzido por p
                UInt<2 * N> acc{};
                UInt<2 * N> wy{};
                UInt<2 * N> wp{};
                for (u8 i = 0; i < N; ++i) {
                    acc.bits[i] = x.bits[i];
                    wy.bits[i] = y.bits[i];
                    wp.bits[i] = p.bits[i];
                }
                acc.mul_consteval(wy);
                auto [q, r] = UInt<2 * N>::divmod(acc, wp);
                (void)q;
                x.bits.fill(0);
                for (u8 i = 0; i < N; ++i) x.bits[i] = r.bits[i];
            } else {
                // N > 127: UInt<2N> não cabe em u8; multiplica em buffer cru
                u64 prod[2 * N];
                Multiplication::mul_schoolbook_full(prod, x.bits.data(), y.bits.data(), N);
                int n_v = p.num_limbs();
                if (n_v == 1) {
                    u64 rem = 0;
                    for (int i = 2 * N - 1; i >= 0; --i) {
                        u128 cur = ((u128)rem << 64) | prod[i];
                        rem = (u64)(cur % p.bits[0]);
                    }
                    x.bits.fill(0);
                    x.bits[0] = rem;
                } else {
                    u64 qq[2 * N + 2];
                    u64 rr[N];
                    Division::div_knuth_impl<2 * N + 4>(qq, rr, prod, 2 * N, p.bits.data(), n_v);
                    x.bits.fill(0);
                    for (int i = 0; i < n_v; ++i) x.bits[i] = rr[i];
                }
            }
        }
    };

    template <u8 N>
    FORCE_INLINE constexpr UInt<N> &UInt<N>::operator<<=(const u16 n) noexcept {
        if (n == 0)[[likely]] return *this;
        else if (n >= N * 64){
            bits.fill(0);
            return *this;
        }
        const u16 block_shift = n >> 6;
        const u16 bit_shift = n & 63;
        if (block_shift > 0) {
            for (int i = N - 1; i >= (int)block_shift; --i)
                bits[i] = bits[i - block_shift];
            fill(bits.begin(), bits.begin() + block_shift, 0ull);
        }
        if (bit_shift > 0) {
            u64 carry = 0;
            for (u8 i = 0; i < N; ++i) {
                u64 next_carry = bits[i] >> (64 - bit_shift);
                bits[i] = (bits[i] << bit_shift) | carry;
                carry = next_carry;
            }
        }
        return *this;
    }

    template <u8 N>
    FORCE_INLINE constexpr UInt<N> &UInt<N>::operator>>=(u16 n) noexcept {
        if (BUILTIN_EXPECT(n == 0, 1))
            return *this;
        if (n >= N * 64) {
            bits.fill(0);
            return *this;
        }
        const u16 block_shift = n / 64;
        const u16 bit_shift = n % 64;
        if (block_shift > 0) {
            for (u8 i = 0; i < N - block_shift; ++i)
                bits[i] = bits[i + block_shift];
            fill(bits.begin() + N - block_shift, bits.end(), 0ull);
        }
        if (bit_shift > 0) {
            u64 carry = 0;
            for (int i = N - 1; i >= 0; --i) {
                u64 next_carry = bits[i] << (64 - bit_shift);
                bits[i] = (bits[i] >> bit_shift) | carry;
                carry = next_carry;
            }
        }
        return *this;
    }
    template <u8 N>
    constexpr pair<UInt<N>, UInt<N>> UInt<N>::divmod(UInt<N> u, UInt<N> v) {
        if (v.eqz())
            throw domain_error("Division by zero");

        if consteval {
            if (u < v) return {UInt<N>(0), u};
            UInt<N> q(0);
            UInt<N> r(0);
            for (int i = N * 64 - 1; i >= 0; --i) {
                r <<= 1;
                if ((u.bits[i / 64] >> (i % 64)) & 1) {
                    r.bits[0] |= 1;
                }
                if (r >= v) {
                    r -= v;
                    q.bits[i / 64] |= (1ULL << (i % 64));
                }
            }
            return {q, r};
        }

        if (u < v)
            return {UInt<N>(0), u};
        if (u == v)
            return {UInt<N>(1), 0};

        u8 n_v = v.num_limbs();
        u8 n_u = u.num_limbs(); // We know u >= v, so n_u >= n_v

        // Optimization for small quotients (common in random distribution)
        // ATENÇÃO: se o quociente for maior que ~12, saímos do bloco com 'u'
        // já reduzido e o quociente acumulado em q_acc precisa ser somado ao
        // quociente do caminho geral (senão o resultado fica errado).
        UInt<N> q_acc{};
        if constexpr (N == 2 || N == 3 || N == 5 || N == 8) {
            if (n_u == n_v) {
                // q is likely small. Try finding it by subtraction.
                u -= v;
                q_acc.bits[0] = 1;

                // Check if done
                if (u < v) return {q_acc, u};

                // Try one more time (q=2)
                u -= v;
                q_acc.bits[0]++;
                if (u < v) return {q_acc, u};

                for (int k = 0; k < 10; ++k) {
                    u -= v;
                    q_acc.bits[0]++;
                    if (u < v) return {q_acc, u};
                    if (u.num_limbs() < n_v) return {q_acc, u}; // Optimization
                }
            }
        }

        u8 n = n_v;
        if (n == 1) {
            u64 rem = 0;
            UInt<N> quo{};
            u64 d = v.bits[0];
            for (int i = u.num_limbs() - 1; i >= 0; --i) {
                u128 temp = ((u128)rem << 64) | u.bits[i];
                quo.bits[i] = temp / d;
                rem = temp % d;
            }
            quo += q_acc;
            return {quo, UInt<N>(rem)};
        }

        UInt<N> q{}, r{};

        if constexpr (N <= 16) {
            Division::DivisionFixed<N>::div(q.bits.data(), r.bits.data(), u.bits.data(), v.bits.data());
        } else if constexpr (N <= 64) {
            Division::div_knuth_impl<N>(q.bits.data(), r.bits.data(), u.bits.data(),
                                    u.num_limbs(), v.bits.data(), v.num_limbs());
        } else {
            int u_len = u.num_limbs();
            int v_len = v.num_limbs();
            if (u_len - v_len < 32) {
                Division::div_knuth_impl<N>(q.bits.data(), r.bits.data(), u.bits.data(),
                                        u_len, v.bits.data(), v_len);
            } else {
                int n_limbs = v.num_limbs();
                int shift_limbs = N - n_limbs;
                int shift_bits = __builtin_clzll(v.bits[n_limbs - 1]);

                u64 u_norm[2 * N];
                u64 v_norm[N];
                u64 tmp[20 * N + 1000];

                if (shift_limbs > 0 || shift_bits > 0) {
                    for(int i = N - 1; i >= shift_limbs; --i)
                        v_norm[i] = v.bits[i - shift_limbs];
                    fill_n(v_norm, shift_limbs, 0);

                    if (shift_bits > 0) {
                        u64 carry = 0;
                        for(int i = 0; i < N; ++i) {
                            u64 val = v_norm[i];
                            v_norm[i] = (val << shift_bits) | carry;
                            carry = val >> (64 - shift_bits);
                        }
                    }
                } else {
                    copy_n(v.bits.data(), N, v_norm);
                }

                fill_n(u_norm, 2 * N, 0);

                for(int i = 0; i < N; ++i) {
                    u_norm[i + shift_limbs] = u.bits[i];
                }

                if (shift_bits > 0) {
                    u64 carry = 0;
                    for(int i = 0; i < 2 * N; ++i) {
                        u64 val = u_norm[i];
                        u_norm[i] = (val << shift_bits) | carry;
                        carry = val >> (64 - shift_bits);
                    }
                }

                Division::div_recursive(q.bits.data(), r.bits.data(), u_norm,
                                        v_norm, N, tmp);

                if (shift_bits > 0) {
                    u64 carry = 0;
                    for(int i = N - 1; i >= 0; --i) {
                        u64 val = r.bits[i];
                        r.bits[i] = (val >> shift_bits) | carry;
                        carry = (val << (64 - shift_bits));
                    }
                }
                if (shift_limbs > 0) {
                    for(int i = 0; i < N - shift_limbs; ++i)
                        r.bits[i] = r.bits[i + shift_limbs];
                    fill_n(r.bits.data() + N - shift_limbs, shift_limbs, 0);
                }
            }
        }

        q += q_acc;
        return {q, r};
    }

    template<u8 N>
    constexpr UInt<N> UInt<N>::random(const u64 seed) noexcept {
        UInt<N> number;
        u64 state = seed;

        if (state == 0) {
            if consteval {
                state = 0xCAFEBABE + N;
            } else {
                state = reinterpret_cast<uintptr_t>(&number) ^ 0x9e3779b97f4a7c15ULL;
            }
        }

        for (u64 &block : number.bits) {
            block = rand(state);
        }
        return number;
    }

    template <u8 N> string UInt<N>::to_string() const {
        if (this->eqz())
            return "0";
        UInt<N> temp = *this;
        constexpr u64 CHUNK_POW = 1000000000000000000ull;
        constexpr int CHUNK_SIZE = 18;

        u64 chunks[300];
        int chunk_count = 0;

        while (temp != 0) {
            auto [quotient, remainder_uint] = divmod(temp, UInt<N>(CHUNK_POW));
            chunks[chunk_count++] = remainder_uint.bits[0];
            temp = move(quotient);
        }
        ostringstream oss;
        oss << chunks[chunk_count - 1];
        for (int i = chunk_count - 2; i >= 0; --i) {
            oss << setw(CHUNK_SIZE) << setfill('0') << chunks[i];
        }
        return oss.str();
    }

    template <u8 N>
    string UInt<N>::to_hex_string() const {
        if (this->eqz())
            return "0x0";
        int msb_idx = N - 1;
        while (msb_idx > 0 && bits[msb_idx] == 0)
            --msb_idx;
        ostringstream oss;
        oss << "0x" << hex << nouppercase;
        oss << bits[msb_idx];
        for (int i = msb_idx - 1; i >= 0; --i)
            oss << setw(16) << setfill('0') << bits[i];
        return oss.str();
    }

    template <u8 N>
    [[nodiscard]] FORCE_INLINE constexpr UInt<N> operator+(UInt<N> lhs,
                                                    const UInt<N> &rhs) noexcept {
        return lhs += rhs;
    }
    template <u8 N>
    [[nodiscard]] FORCE_INLINE constexpr UInt<N> operator-(UInt<N> lhs,
                                                    const UInt<N> &rhs) noexcept {
        return lhs -= rhs;
    }
    template <u8 N>
    [[nodiscard]] FORCE_INLINE UInt<N> operator*(UInt<N> lhs,
                                            const UInt<N> &rhs) noexcept {
        if constexpr (N == 4) {
            u64 r0, r1, r2, r3;
            __asm__ volatile (
                MUL4_ASM_CORE
                "movq   %%r9, %[r0]\n\t"
                "movq   %%rax, %[r1]\n\t"
                "movq   %%r10, %[r2]\n\t"
                "movq   %%r8, %[r3]\n\t"
                : [r0] "=&r" (r0), [r1] "=&r" (r1), [r2] "=&r" (r2), [r3] "=&r" (r3)
                : [a0] "m" (lhs.bits[0]), [a1] "m" (lhs.bits[1]),
                [a2] "m" (lhs.bits[2]), [a3] "m" (lhs.bits[3]),
                [b0] "m" (rhs.bits[0]), [b1] "m" (rhs.bits[1]),
                [b2] "m" (rhs.bits[2]), [b3] "m" (rhs.bits[3])
                : "rax", "rcx", "rdx", "rsi", "r8", "r9", "r10", "cc"
            );
            lhs.bits[0] = r0;
            lhs.bits[1] = r1;
            lhs.bits[2] = r2;
            lhs.bits[3] = r3;
            return lhs;
        }
#undef MUL4_ASM_CORE
        return lhs *= rhs;
    }
    template <u8 N>
    [[nodiscard]] FORCE_INLINE UInt<N> operator/(UInt<N> lhs, const UInt<N> &rhs) {
        return lhs /= rhs;
    }
    template <u8 N>
    [[nodiscard]] FORCE_INLINE UInt<N> operator%(UInt<N> lhs, const UInt<N> &rhs) {
        return lhs %= rhs;
    }

    template <u8 N>
    [[nodiscard]] FORCE_INLINE constexpr UInt<N> operator+(UInt<N> lhs, u64 rhs) noexcept {
        return lhs += rhs;
    }
    template <u8 N>
    [[nodiscard]] FORCE_INLINE constexpr UInt<N> operator+(u64 lhs, UInt<N> rhs) noexcept {
        return rhs += lhs;
    }

    template <u8 N>
    [[nodiscard]] FORCE_INLINE constexpr UInt<N> operator-(UInt<N> lhs, u64 rhs) noexcept {
        return lhs -= rhs;
    }
    template <u8 N>
    [[nodiscard]] FORCE_INLINE constexpr UInt<N> operator-(u64 lhs, const UInt<N>& rhs) noexcept {
        return UInt<N>(lhs) -= rhs;
    }

    template <u8 N>
    [[nodiscard]] FORCE_INLINE UInt<N> operator*(UInt<N> lhs, u64 rhs) noexcept {
        return lhs *= rhs;
    }
    template <u8 N>
    [[nodiscard]] FORCE_INLINE UInt<N> operator*(u64 lhs, UInt<N> rhs) noexcept {
        return rhs *= lhs;
    }

    template <u8 N>
    [[nodiscard]] FORCE_INLINE UInt<N> operator/(UInt<N> lhs, u64 rhs) {
        return lhs /= rhs;
    }

    template <u8 N>
    [[nodiscard]] FORCE_INLINE UInt<N> operator%(UInt<N> lhs, u64 rhs) {
        return lhs %= rhs;
    }

    template <u8 N>
    [[nodiscard]] FORCE_INLINE constexpr UInt<N> operator<<(UInt<N> lhs,
                                                    u16 rhs) noexcept {
        return lhs <<= rhs;
    }
    template <u8 N>
    [[nodiscard]] FORCE_INLINE constexpr UInt<N> operator>>(UInt<N> lhs,
                                                    u16 rhs) noexcept {
        return lhs >>= rhs;
    }

    template <u8 N>
    [[nodiscard]] FORCE_INLINE constexpr UInt<N> operator&(UInt<N> lhs, const UInt<N> &rhs) noexcept {
        return lhs &= rhs;
    }
    template <u8 N>
    [[nodiscard]] FORCE_INLINE constexpr UInt<N> operator|(UInt<N> lhs, const UInt<N> &rhs) noexcept {
        return lhs |= rhs;
    }
    template <u8 N>
    [[nodiscard]] FORCE_INLINE constexpr UInt<N> operator^(UInt<N> lhs, const UInt<N> &rhs) noexcept {
        return lhs ^= rhs;
    }
    template <u8 N>
    [[nodiscard]] FORCE_INLINE constexpr UInt<N> operator~(UInt<N> val) noexcept {
        for (auto &limb : val.bits) limb = ~limb;
        return val;
    }

    template <u8 N>
    inline std::ostream& operator<<(std::ostream& os, const UInt<N>& val) {
        return os << val.to_string();
    }
