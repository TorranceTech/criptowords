#pragma once

#include "Multiplication.hpp"
#include <stdexcept>
#include <algorithm>
#include <cstdint>
#include <immintrin.h>
#include <vector>
#include <cstring> // For memset

namespace Division {

    using u64 = unsigned long long;
    using u128 = unsigned __int128;

    #ifndef FORCE_INLINE
    #define FORCE_INLINE inline __attribute__((always_inline))
    #endif
    #define BUILTIN_EXPECT(x, y) (__builtin_expect(!!(x), y))

    // Helper: Compare two N-block numbers
    // Returns 1 if a > b, -1 if a < b, 0 if a == b
    inline int cmp_n(const u64 *a, const u64 *b, size_t n) {
        for (size_t i = n; i-- > 0;) {
            if (a[i] > b[i])
                return 1;
            if (a[i] < b[i])
                return -1;
        }
        return 0;
    }

    // Helper: Add N-block numbers: res = a + b. Returns carry
    inline u64 add_n(u64 *res, const u64 *a, const u64 *b, size_t n) {
        unsigned char carry = 0;
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            carry = _addcarry_u64(carry, a[i], b[i], &res[i]);
            carry = _addcarry_u64(carry, a[i+1], b[i+1], &res[i+1]);
            carry = _addcarry_u64(carry, a[i+2], b[i+2], &res[i+2]);
            carry = _addcarry_u64(carry, a[i+3], b[i+3], &res[i+3]);
        }
        for (; i < n; ++i) {
            carry = _addcarry_u64(carry, a[i], b[i], &res[i]);
        }
        return carry;
    }

    // Helper: Subtract N-block numbers: res = a - b
    // Returns borrow
    inline u64 sub_n(u64 *res, const u64 *a, const u64 *b, size_t n) {
        unsigned char borrow = 0;
        size_t i = 0;
        for (; i + 4 <= n; i += 4) {
            borrow = _subborrow_u64(borrow, a[i], b[i], &res[i]);
            borrow = _subborrow_u64(borrow, a[i+1], b[i+1], &res[i+1]);
            borrow = _subborrow_u64(borrow, a[i+2], b[i+2], &res[i+2]);
            borrow = _subborrow_u64(borrow, a[i+3], b[i+3], &res[i+3]);
        }
        for (; i < n; ++i) {
            borrow = _subborrow_u64(borrow, a[i], b[i], &res[i]);
        }
        return borrow;
    }

    // Helper: Shift Left N-block number by bits (0-63)
    // res = a << shift
    // Returns carry out
    inline u64 shl_n(u64 *res, const u64 *a, size_t n, int shift) {
        if (shift == 0) {
            if (res != a)
                std::copy_n(a, n, res);
            return 0;
        }
        u64 carry = 0;
        for (size_t i = 0; i < n; ++i) {
            u64 next_carry = a[i] >> (64 - shift);
            res[i] = (a[i] << shift) | carry;
            carry = next_carry;
        }
        return carry;
    }

    // Helper: Shift Right N-block number by bits (0-63)
    // res = a >> shift
    // Returns carry out (bits shifted out)
    inline u64 shr_n(u64 *res, const u64 *a, size_t n, int shift) {
        if (shift == 0) {
            if (res != a)
                std::copy_n(a, n, res);
            return 0;
        }
        u64 carry = 0;
        for (size_t i = n; i-- > 0;) {
            u64 next_carry = a[i] << (64 - shift);
            res[i] = (a[i] >> shift) | carry;
            carry = next_carry;
        }
        return carry;
    }


    // Forward declaration
    void div_recursive(u64 *q, u64 *r, const u64 *a, const u64 *b, size_t n,
                       u64 *tmp);

    // Knuth's Algorithm D for M-limb / N-limb division (M >= N)
    // Computes q = u / v, r = u % v
    // u: dividend (size u_len)
    // v: divisor (size v_len)
    template <size_t MaxBlocks = 600> // MaxBlocks is the total capacity for internal buffers
    inline void div_knuth_impl(u64 *q, u64 *r, const u64 *u, int u_len,
                               const u64 *v, int v_len) {
        // D1: Normalize
        // Assumes v_len >= 2. Single-limb divisor is handled by UInt's fast path.
        // v_len can be 1 for div_recursive base case but div_knuth_impl should not be called with v_len < 2 for D3 step (v_next, u_norm[j+bn-2]).
        // However, MaxBlocks will always be >= N which is >= 1.
        if (v_len == 0) {
            throw std::domain_error("Division by zero in div_knuth_impl");
        }
      
        if (u_len < v_len) {
            std::fill_n(q, u_len - v_len + 1, 0); // Quotient is 0
            std::copy_n(u, u_len, r); // Remainder is u
            std::fill_n(r + u_len, v_len - u_len, 0); // Pad remainder with zeros if r smaller than v
            return;
        }


        const int shift = __builtin_clzll(v[v_len - 1]);
        const int an = u_len;
        const int bn = v_len;
      
        // Stack buffers
        u64 u_norm[MaxBlocks + 1];
        u64 v_norm_storage[MaxBlocks];
        const u64* v_ptr;

        // Normalize divisor v
        if (shift > 0) {
            shl_n(v_norm_storage, v, bn, shift);
            v_ptr = v_norm_storage;
        } else {
            // Optimization: Avoid copy if no shift
            v_ptr = v;
        }

        // Normalize dividend u
        // u_norm needs to be an + 1 size to handle potential carry
        if (shift > 0) {
            u_norm[an] = shl_n(u_norm, u, an, shift);
        } else {
            std::copy_n(u, an, u_norm);
            u_norm[an] = 0;
        }

        const u64 v_high = v_ptr[bn - 1];
        // v_next is v_{n-2}
        const u64 v_next = (bn > 1) ? v_ptr[bn - 2] : 0; 

        // D2-D7: Main loop - compute quotient digits
        // Loop from j = an - bn down to 0
        for (int j = an - bn; j >= 0; --j) {
            // D3: Calculate trial quotient digit
            u64 u_high = u_norm[j + bn];
            u64 u_mid = u_norm[j + bn - 1];
            u64 q_hat;
            u128 current_dividend_top_two_limbs = ((u128)u_high << 64) | u_mid;

            // Initial estimate for q_hat
            // q_hat = floor((U_{j+n}*B + U_{j+n-1}) / V_{n-1})
            if (u_high == v_high) [[unlikely]] {
                q_hat = ~0ULL; // Set to B-1 (max u64)
            } else {
                q_hat = current_dividend_top_two_limbs / v_high;
            }
          
            // D3: Refine q_hat.
            // We need to check: (q_hat * V_n-2) > ((current_remainder_for_q_hat << 64) | U_j+n-2)
            // where current_remainder_for_q_hat = current_dividend_top_two_limbs - (u128)q_hat * v_high;
            //
            // Knuth D3: the check is a valid over-estimate indicator ONLY while
            // r_hat < v_high. Once r_hat >= v_high, q_hat is within 1 of the true
            // digit and the D4/D6 correction handles it. Without this bound the
            // loop can spin forever (e.g. v_next == v_high == b-1 with q_hat
            // exactly 1 above the true digit: the lhs/rhs gap stays constant).
            u128 r_hat_val_for_check = current_dividend_top_two_limbs - (u128)q_hat * v_high;
            // Knuth D.3 condition for adjustment: `q_hat * v_{n-2} > (r_hat * B + u_{j+n-2})`
            u128 lhs_compare = (u128)q_hat * v_next; // v_next is V_n-2
            // u_norm[j + bn - 2] needs bn >= 2. Handled by explicit N=1 fast path in UInt.
            u128 rhs_compare = (r_hat_val_for_check << 64) | u_norm[j + bn - 2];

            if (lhs_compare > rhs_compare) [[unlikely]] {
                q_hat--; // q_hat was too large, decrement
                r_hat_val_for_check += v_high;
                // Re-test only while r_hat is still a valid remainder (< v_high).
                if (r_hat_val_for_check < v_high) {
                    lhs_compare = (u128)q_hat * v_next;
                    rhs_compare = (r_hat_val_for_check << 64) | u_norm[j + bn - 2];
                    if (lhs_compare > rhs_compare) [[unlikely]] {
                        q_hat--;
                    }
                }
            }
          
            // D4: Multiply and subtract
            u64 mult_carry = 0;
            unsigned char sub_borrow = 0;
          
            size_t i = 0;
            for (; i + 4 <= (size_t)bn; i += 4) {
                u64 lo;
              
                // Unrolled multiply and add carry
                u128 prod0 = (u128)q_hat * v_ptr[i] + mult_carry;
                lo = (u64)prod0; mult_carry = prod0 >> 64;

                u128 prod1 = (u128)q_hat * v_ptr[i+1] + mult_carry;
                u64 lo1 = (u64)prod1; mult_carry = prod1 >> 64;
              
                u128 prod2 = (u128)q_hat * v_ptr[i+2] + mult_carry;
                u64 lo2 = (u64)prod2; mult_carry = prod2 >> 64;
              
                u128 prod3 = (u128)q_hat * v_ptr[i+3] + mult_carry;
                u64 lo3 = (u64)prod3; mult_carry = prod3 >> 64;
              
                // Subtract from u_norm
                sub_borrow = _subborrow_u64(sub_borrow, u_norm[j + i], lo, &u_norm[j + i]);
                sub_borrow = _subborrow_u64(sub_borrow, u_norm[j + i+1], lo1, &u_norm[j + i+1]);
                sub_borrow = _subborrow_u64(sub_borrow, u_norm[j + i+2], lo2, &u_norm[j + i+2]);
                sub_borrow = _subborrow_u64(sub_borrow, u_norm[j + i+3], lo3, &u_norm[j + i+3]);
            }
          
            for (; i < (size_t)bn; ++i) {
                u128 prod = (u128)q_hat * v_ptr[i] + mult_carry;
                u64 lo = (u64)prod;
                mult_carry = prod >> 64;
                sub_borrow = _subborrow_u64(sub_borrow, u_norm[j + i], lo, &u_norm[j + i]);
            }
          
            // Handle last borrow/carry from multiplication and subtraction
            sub_borrow = _subborrow_u64(sub_borrow, u_norm[j + bn], mult_carry, &u_norm[j + bn]);

            // D6: Add back if negative
            if (sub_borrow) {
                q_hat--;
                unsigned char add_carry = 0;
                for (size_t k = 0; k < (size_t)bn; ++k) {
                    add_carry = _addcarry_u64(add_carry, u_norm[j + k], v_ptr[k], &u_norm[j + k]);
                }
                u_norm[j + bn] += add_carry;
            }

            q[j] = q_hat;
        }

        // D8: Unnormalize remainder
        if (shift > 0) {
            shr_n(r, u_norm, bn, shift);
        } else {
            std::copy_n(u_norm, bn, r);
        }
    }

    // Optimized Fixed-Size Knuth Division
    template <size_t N>
    struct DivisionFixed {
        // Fixed size add/sub/mul/shl helpers to ensure unrolling
        static FORCE_INLINE u64 add_n(u64 *res, const u64 *a, const u64 *b) {
            unsigned char carry = 0;
            // Compiler will unroll this loop for small N
            for (size_t i = 0; i < N; ++i) {
                carry = _addcarry_u64(carry, a[i], b[i], &res[i]);
            }
            return carry;
        }

        static FORCE_INLINE u64 sub_n(u64 *res, const u64 *a, const u64 *b) {
            unsigned char borrow = 0;
            for (size_t i = 0; i < N; ++i) {
                borrow = _subborrow_u64(borrow, a[i], b[i], &res[i]);
            }
            return borrow;
        }

        static FORCE_INLINE void shl_n(u64 *res, const u64 *a, int shift, u64 &carry_out) {
            if (shift == 0) {
                for(size_t i=0; i<N; ++i) res[i] = a[i];
                carry_out = 0;
                return;
            }
            u64 c = 0;
            for (size_t i = 0; i < N; ++i) {
                u64 val = a[i];
                u64 next_c = val >> (64 - shift);
                res[i] = (val << shift) | c;
                c = next_c;
            }
            carry_out = c;
        }

        static FORCE_INLINE void shr_n(u64 *res, const u64 *a, int shift) {
            if (shift == 0) {
                for(size_t i=0; i<N; ++i) res[i] = a[i];
                return;
            }
            u64 c = 0;
            for (size_t i = N; i-- > 0;) {
                u64 val = a[i];
                u64 next_c = val << (64 - shift);
                res[i] = (val >> shift) | c;
                c = next_c;
            }
        }



        static void div(u64 *q, u64 *r, const u64 *u, const u64 *v) {
            // Effective size check
            int n_limbs = 0;
            for (int i = N - 1; i >= 0; --i) {
                if (v[i] != 0) {
                    n_limbs = i + 1;
                    break;
                }
            }
            
            if (n_limbs == 0) throw std::domain_error("Division by zero");

            // Fallback for partial size
            if (n_limbs < (int)N) {
                 div_knuth_impl<N>(q, r, u, N, v, n_limbs);
                 return;
            }

            int shift = __builtin_clzll(v[N - 1]);
            
            u64 v_norm_storage[N];
            const u64* v_ptr;
            u64 u_norm[N + 2]; 

            // Normalize v (Avoid copy if shift=0)
            if (shift == 0) {
                v_ptr = v; 
            } else {
                u64 c = 0;
                #pragma GCC unroll 8
                for(size_t i=0; i<N; ++i) {
                    u64 val = v[i];
                    v_norm_storage[i] = (val << shift) | c;
                    c = val >> (64 - shift);
                }
                v_ptr = v_norm_storage;
            }

            // Normalize u
            if (shift == 0) {
                 #pragma GCC unroll 8
                 for(size_t i=0; i<N; ++i) u_norm[i] = u[i];
                 u_norm[N] = 0;
                 u_norm[N+1] = 0;
            } else {
                 u64 c = 0;
                 #pragma GCC unroll 8
                 for(size_t i=0; i<N; ++i) {
                     u64 val = u[i];
                     u_norm[i] = (val << shift) | c;
                     c = val >> (64 - shift);
                 }
                 u_norm[N] = c;
                 u_norm[N+1] = 0;
            }

            const u64 v_high = v_ptr[N - 1];
            const u64 v_next = (N > 1) ? v_ptr[N - 2] : 0; 
            
            // j=1, j=0.
            for (int j = 1; j >= 0; --j) {
                u64 u_high = u_norm[j + N];
                u64 u_mid = u_norm[j + N - 1];
                
                if (u_high == 0 && u_mid < v_high) {
                    q[j] = 0;
                    continue;
                }
                
                u64 q_hat;
                if (u_high == v_high) {
                    q_hat = ~0ULL;
                } else {
                    u128 num = ((u128)u_high << 64) | u_mid;
                    q_hat = num / v_high;
                }
                
                u128 r_hat = ((u128)u_high << 64) | u_mid;
                r_hat -= (u128)q_hat * v_high;
                
                if ((u128)q_hat * v_next > ((r_hat << 64) | u_norm[j + N - 2])) {
                    q_hat--;
                    r_hat += v_high;
                    if (r_hat < v_high && ((u128)q_hat * v_next > ((r_hat << 64) | u_norm[j + N - 2]))) {
                         q_hat--;
                    }
                }
                
                u64 mul_carry = 0;
                unsigned char sub_borrow = 0;
                #pragma GCC unroll 8
                for (size_t i = 0; i < N; ++i) {
                    u128 prod = (u128)q_hat * v_ptr[i] + mul_carry;
                    mul_carry = prod >> 64;
                    sub_borrow = _subborrow_u64(sub_borrow, u_norm[j+i], (u64)prod, &u_norm[j+i]);
                }
                sub_borrow = _subborrow_u64(sub_borrow, u_norm[j+N], mul_carry, &u_norm[j+N]);
                
                if (sub_borrow) {
                    q_hat--;
                    add_n(&u_norm[j], &u_norm[j], v_ptr);
                }
                q[j] = q_hat;
            }
            
            // Unnormalize Remainder
            if (shift == 0) {
                #pragma GCC unroll 8
                for(size_t i=0; i<N; ++i) r[i] = u_norm[i];
            } else {
                #pragma GCC unroll 8
                for (int i=N-1; i>=0; --i) {
                    u64 val = u_norm[i];
                    u64 val_high = u_norm[i+1];
                    r[i] = (val >> shift) | (val_high << (64 - shift));
                }
            }
        }
    };
    inline void div_recursive(u64 *q, u64 *r, const u64 *a, const u64 *b, size_t n,
                              u64 *tmp) {
        div_knuth_impl<600>(q, r, a, 2 * n, b, n);
    }

}
