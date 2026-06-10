// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan/detail/mix.hpp
// ----------------------------------------------------------------------------
// The FPSan "payload algebra": the bijection phi between floating-point bit
// patterns and integers mod 2^w, and the integer operations that stand in for
// floating-point ops. This is the width-generic construction reverse-engineered
// from Triton (see the independent reference in tests/fpsan_generic.hpp and the
// Triton sources referenced there); here it is constexpr and host/device so it
// can run on a GPU with zero runtime setup.
//
// All arithmetic is performed in std::uint64_t and masked back to the format
// width: for w <= 32 the products are exact, and for w == 64 unsigned
// wraparound modulo 2^64 followed by masking yields the correct value modulo
// 2^w. This sidesteps both overflow and the integer-promotion pitfalls of
// narrow types.
// ----------------------------------------------------------------------------
#ifndef FPSAN_DETAIL_MIX_HPP
#define FPSAN_DETAIL_MIX_HPP

#include "fpsan/detail/config.hpp"
#include "fpsan/detail/traits.hpp"

#include <cstdint>

namespace fpsan
{
    namespace detail
    {

        using u64 = std::uint64_t;

        // Modular inverse of an odd 64-bit number (Hensel/Newton lifting).
        FPSAN_HOST_DEVICE constexpr u64 inv_odd_u64(u64 a)
        {
            // a must be odd.
            u64 x = 2 - a;
            for(unsigned correct_bits = 2; correct_bits < 64; correct_bits *= 2)
                x *= 2 - a * x;
            return x;
        }

        FPSAN_HOST_DEVICE constexpr unsigned countr_zero_u64(u64 x)
        {
            unsigned n = 0;
            while(((x >> n) & 1u) == 0u)
                ++n;
            return n;
        }

        // The per-format mixing configuration. Pure function of fp_traits<T>; intended
        // to be held as a `static constexpr` so it costs nothing at run time.
        struct MixConfig
        {
            unsigned bit_width;
            unsigned shift; // xorshift amount == mantissa width
            u64      sign_mask; // top bit
            u64      mag_mask; // low (bit_width - 1) bits
            u64      full_mask; // all bit_width bits == ring modulus minus one
            u64      mul_a, mul_a_inv;
            u64      mul_b_pos, mul_b_pos_inv;
            u64      mul_b_neg, mul_b_neg_inv;
        };

        template <class T>
        FPSAN_HOST_DEVICE constexpr MixConfig make_mix_config()
        {
            using tr = fp_traits<T>;
            MixConfig c{};
            c.bit_width = tr::bit_width;
            c.sign_mask = u64{1} << (tr::bit_width - 1);
            c.mag_mask  = c.sign_mask - 1;
            c.full_mask = c.sign_mask | c.mag_mask; // all bit_width bits (avoids <<64)

            // IEEE bit pattern of +1.0 is exp=bias, mantissa=0.
            const u64 one_bits = u64(static_cast<unsigned>(tr::bias)) << tr::mantissa_bits;
            // Triton's usual formats have odd bias, so this equals mantissa_bits.
            // AMD FNUZ fp8 uses an even bias; using the actual trailing-zero count
            // keeps the +1.0 mixed value odd and therefore invertible.
            c.shift = countr_zero_u64(one_bits);

            c.mul_a       = 922291u & c.mag_mask;
            u64 one_mixed = (one_bits * c.mul_a) & c.mag_mask;
            one_mixed ^= one_mixed >> c.shift; // odd by construction

            c.mul_b_pos = inv_odd_u64(one_mixed) & c.mag_mask;
            // mag_mask == -1 (mod 2^(bit_width-1)), so mul_b_neg == -mul_b_pos: this is
            // what makes phi(-x) == -phi(x).
            c.mul_b_neg = (c.mul_b_pos * c.mag_mask) & c.mag_mask;

            c.mul_a_inv     = inv_odd_u64(c.mul_a) & c.mag_mask;
            c.mul_b_pos_inv = inv_odd_u64(c.mul_b_pos) & c.mag_mask;
            c.mul_b_neg_inv = inv_odd_u64(c.mul_b_neg) & c.mag_mask;
            return c;
        }

        // phi: float bit pattern -> integer payload (Triton's mixFloatToInt).
        FPSAN_HOST_DEVICE constexpr u64 mix_bits_to_payload(const MixConfig& c, u64 u)
        {
            const u64 sign_flip = (u & c.sign_mask) ? c.sign_mask : u64{0};
            const u64 x         = u ^ sign_flip; // magnitude, sign cleared
            const u64 y         = (x * c.mul_a) & c.mag_mask;
            const u64 z         = y ^ (y >> c.shift);
            const u64 mul_b     = (u & c.sign_mask) ? c.mul_b_neg : c.mul_b_pos;
            const u64 w         = (z * mul_b) & c.mag_mask;
            return w ^ sign_flip;
        }

        // phi^{-1}: integer payload -> float bit pattern (Triton's unmixIntToFloat).
        FPSAN_HOST_DEVICE constexpr u64 unmix_payload_to_bits(const MixConfig& c, u64 v)
        {
            const u64 sign_flip = (v & c.sign_mask) ? c.sign_mask : u64{0};
            const u64 w         = v ^ sign_flip;
            const u64 mul_b_inv = (v & c.sign_mask) ? c.mul_b_neg_inv : c.mul_b_pos_inv;
            const u64 z         = (w * mul_b_inv) & c.mag_mask;
            u64       y         = z; // invert the single xorshift-right by `shift`
            for(unsigned s = c.shift; s < c.bit_width; s *= 2)
                y ^= y >> s;
            const u64 x = (y * c.mul_a_inv) & c.mag_mask;
            return x ^ sign_flip;
        }

        // Payload reciprocal for division (Triton's fpsanIntInv): true modular inverse
        // for odd payloads, a parity-preserving involution for even ones.
        FPSAN_HOST_DEVICE constexpr u64 payload_inv(const MixConfig& c, u64 u)
        {
            const u64 a = u | 1u; // force odd
            u64       x = (2u - a) & c.full_mask;
            for(unsigned correct_bits = 2; correct_bits < c.bit_width; correct_bits *= 2)
                x = (x * ((2u - a * x) & c.full_mask)) & c.full_mask;
            const u64 even_part       = x & (c.full_mask ^ u64{1});
            const u64 original_parity = u & 1u;
            return even_part | original_parity;
        }

        // Ring operations on payloads: ordinary integer arithmetic modulo 2^w.
        FPSAN_HOST_DEVICE constexpr u64 payload_add(const MixConfig& c, u64 a, u64 b)
        {
            return (a + b) & c.full_mask;
        }
        FPSAN_HOST_DEVICE constexpr u64 payload_mul(const MixConfig& c, u64 a, u64 b)
        {
            return (a * b) & c.full_mask;
        }
        FPSAN_HOST_DEVICE constexpr u64 payload_div(const MixConfig& c, u64 a, u64 b)
        {
            return payload_mul(c, a, payload_inv(c, b));
        }

        // ---------------------------------------------------------------------------
        //  Generic bits engine: the same operations over a `Bits` type that is either a
        //  scalar unsigned integer (u16/u32/u64) or a Clang vector of such. The scalar
        //  primitives above are reused per lane. Lane-wise native arithmetic plus the
        //  width mask is exact at every lane width, so + - * neg vectorize directly;
        //  the phi mix / division reuse the (cold-path) scalar code per lane.
        // ---------------------------------------------------------------------------

        template <class Bits>
        FPSAN_HOST_DEVICE constexpr Bits bits_embed(const MixConfig& c, Bits raw)
        {
            if constexpr(!is_clang_vector_v<Bits>)
            {
                return static_cast<Bits>(mix_bits_to_payload(c, static_cast<u64>(raw)));
            }
            else
            {
                using L = bits_lane_t<Bits>;
                Bits out{};
                for(unsigned i = 0; i < sizeof(Bits) / sizeof(L); ++i)
                    out[i] = static_cast<L>(mix_bits_to_payload(c, static_cast<u64>(raw[i])));
                return out;
            }
        }
        template <class Bits>
        FPSAN_HOST_DEVICE constexpr Bits bits_unembed(const MixConfig& c, Bits payload)
        {
            if constexpr(!is_clang_vector_v<Bits>)
            {
                return static_cast<Bits>(unmix_payload_to_bits(c, static_cast<u64>(payload)));
            }
            else
            {
                using L = bits_lane_t<Bits>;
                Bits out{};
                for(unsigned i = 0; i < sizeof(Bits) / sizeof(L); ++i)
                    out[i] = static_cast<L>(unmix_payload_to_bits(c, static_cast<u64>(payload[i])));
                return out;
            }
        }

        template <class Bits>
        FPSAN_HOST_DEVICE constexpr Bits ring_add(const MixConfig& c, Bits a, Bits b)
        {
            return static_cast<Bits>((a + b) & static_cast<bits_lane_t<Bits>>(c.full_mask));
        }
        template <class Bits>
        FPSAN_HOST_DEVICE constexpr Bits ring_sub(const MixConfig& c, Bits a, Bits b)
        {
            return static_cast<Bits>((a - b) & static_cast<bits_lane_t<Bits>>(c.full_mask));
        }
        template <class Bits>
        FPSAN_HOST_DEVICE constexpr Bits ring_mul(const MixConfig& c, Bits a, Bits b)
        {
            return static_cast<Bits>((a * b) & static_cast<bits_lane_t<Bits>>(c.full_mask));
        }
        template <class Bits>
        FPSAN_HOST_DEVICE constexpr Bits ring_neg(const MixConfig& c, Bits a)
        {
            using L = bits_lane_t<Bits>;
            return static_cast<Bits>((static_cast<L>(0) - a) & static_cast<L>(c.full_mask));
        }
        template <class Bits>
        FPSAN_HOST_DEVICE constexpr Bits ring_div(const MixConfig& c, Bits a, Bits b)
        {
            if constexpr(!is_clang_vector_v<Bits>)
            {
                return static_cast<Bits>(payload_div(c, static_cast<u64>(a), static_cast<u64>(b)));
            }
            else
            {
                using L = bits_lane_t<Bits>;
                Bits out{};
                for(unsigned i = 0; i < sizeof(Bits) / sizeof(L); ++i)
                    out[i] = static_cast<L>(
                        payload_div(c, static_cast<u64>(a[i]), static_cast<u64>(b[i])));
                return out;
            }
        }

    } // namespace detail
} // namespace fpsan

#endif // FPSAN_DETAIL_MIX_HPP
