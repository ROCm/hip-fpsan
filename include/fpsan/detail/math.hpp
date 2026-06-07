// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan/detail/math.hpp
// ----------------------------------------------------------------------------
// Payload-level math, transcribed from Triton's FpSanitizer.cpp. Three flavors,
// matching Triton exactly:
//
//   * Algebraic   exp2/exp (modular exponentiation by a fixed generator) and
//                 cos/sin (a doubling recurrence). These preserve real
//                 identities: exp2(x+y)=exp2(x)*exp2(y), the angle-addition
//                 formulas, etc.
//   * Tagged      log/log2/sqrt/rsqrt/erf/floor/ceil/precise_sqrt:
//   deterministic
//                 op-distinguishing scrambles (NOT real math). Their only
//                 guarantee is that equal inputs give equal outputs and
//                 different ops give different outputs.
//   * Modular     fma (a*b+c), frem (signed remainder), min/max (signed order).
//
// All constants and the bit-for-bit procedure are taken from Triton; widths
// other than 32 reuse the same constants truncated to the format width (the
// 32-bit constants' low bits are what matter, e.g. the exp generator stays
// == 5 (mod 8)).
// ----------------------------------------------------------------------------
#ifndef FPSAN_DETAIL_MATH_HPP
#define FPSAN_DETAIL_MATH_HPP

#include "fpsan/detail/config.hpp"
#include "fpsan/detail/mix.hpp"

#include <cstdint>

namespace fpsan
{
    namespace detail
    {

        // Triton's UnaryOpId enum values; the exact numbers feed the op tag hash in
        // payload_tagged_unary, so they must match Triton bit-for-bit. Kept as a
        // complete 1:1 mirror of Triton's enum: Exp/Exp2/Cos/Sin are realized
        // algebraically here (payload_exp/exp2/cos_sin) and never reach
        // payload_tagged_unary, but listing them documents the full enum and pins
        // the numbering of the tagged ops that do.
        enum class UnaryOpId : std::uint64_t
        {
            Exp         = 0, // algebraic (payload_exp), not tagged
            Log         = 1,
            Exp2        = 2, // algebraic (payload_exp2), not tagged
            Log2        = 3,
            Cos         = 4, // algebraic (payload_cos_sin), not tagged
            Sin         = 5, // algebraic (payload_cos_sin), not tagged
            Sqrt        = 6,
            Rsqrt       = 7,
            Erf         = 8,
            Floor       = 9,
            Ceil        = 10,
            PreciseSqrt = 11,
            Rcp         = 12,
            Fract       = 13,
            Tanh        = 14,
        };

        FPSAN_HOST_DEVICE constexpr u64 murmur64_mixer(u64 h)
        {
            h ^= h >> 33;
            h *= 0xff51afd7ed558ccdull;
            h ^= h >> 33;
            h *= 0xc4ceb9fe1a85ec53ull;
            h ^= h >> 33;
            return h;
        }

        // ---- algebraic: exp2 / exp -------------------------------------------------
        // exp2(x): C^payload (mod 2^w), MSB-first square-and-multiply. C == 0xA343836D
        // truncated to width; note C == 5 (mod 8) at every width.
        FPSAN_HOST_DEVICE constexpr u64 payload_exp2(const MixConfig& c, u64 x)
        {
            const u64 C = 0xA343836Dull & c.full_mask;
            u64       y = 1;
            for(int i = static_cast<int>(c.bit_width) - 1; i >= 0; --i)
            {
                y                = (y * y) & c.full_mask;
                const u64 factor = (x & (u64{1} << i)) ? C : u64{1};
                y                = (y * factor) & c.full_mask;
            }
            return y;
        }
        // exp(x) == exp2(x / ln2): scale the payload by 0x236EE9BF, then exp2.
        FPSAN_HOST_DEVICE constexpr u64 payload_exp(const MixConfig& c, u64 x)
        {
            const u64 rcp_log2 = 0x236EE9BFull & c.full_mask;
            return payload_exp2(c, (x * rcp_log2) & c.full_mask);
        }

        // ---- algebraic: cos / sin (doubling recurrence) ----------------------------
        struct CosSin
        {
            u64 cos;
            u64 sin;
        };
        FPSAN_HOST_DEVICE constexpr CosSin payload_cos_sin(const MixConfig& cfg, u64 x)
        {
            const u64 m    = cfg.full_mask;
            const u64 rcp5 = inv_odd_u64(5) & m;
            const u64 a    = (u64{0} - ((u64{3} * rcp5) & m)) & m;
            const u64 b    = (u64{4} * rcp5) & m;
            u64       c = 1, s = 0;
            for(unsigned bit = 0; bit < cfg.bit_width; ++bit)
            {
                const u64      cc        = (c * c) & m;
                const u64      ss        = (s * s) & m;
                const u64      c_double  = (cc - ss) & m;
                const u64      cs        = (c * s) & m;
                const u64      s_double  = (u64{2} * cs) & m;
                const u64      c_inc     = (((a * c_double) & m) - ((b * s_double) & m)) & m;
                const u64      s_inc     = (((a * s_double) & m) + ((b * c_double) & m)) & m;
                const unsigned bit_index = (cfg.bit_width - 1) - bit;
                const bool     is_zero   = (x & (u64{1} << bit_index)) == 0;
                c                        = is_zero ? c_double : c_inc;
                s                        = is_zero ? s_double : s_inc;
            }
            return {c, s};
        }

        // ---- tagged op-distinguishing scramble -------------------------------------
        FPSAN_HOST_DEVICE constexpr u64
            payload_tagged_unary(const MixConfig& c, u64 p, UnaryOpId op)
        {
            const u64 mult     = 314159u & c.full_mask;
            const u64 h        = murmur64_mixer(static_cast<u64>(op)) & c.full_mask;
            const u64 mixed_in = (p * mult) & c.full_mask;
            const u64 tagged   = mixed_in ^ h;
            return (tagged * mult) & c.full_mask;
        }

        // ---- modular: fma, frem, min/max -------------------------------------------
        FPSAN_HOST_DEVICE constexpr u64 payload_fma(const MixConfig& c, u64 a, u64 b, u64 cc)
        {
            return payload_add(c, payload_mul(c, a, b), cc);
        }

        // Sign-extend a w-bit payload to a signed 64-bit value.
        FPSAN_HOST_DEVICE constexpr std::int64_t payload_sext(const MixConfig& c, u64 v)
        {
            v &= c.full_mask;
            if(v & c.sign_mask)
                v |= ~c.full_mask; // set the high bits
            return static_cast<std::int64_t>(v);
        }
        // frem: signed remainder by (den | 1) (Triton forces the denominator odd).
        FPSAN_HOST_DEVICE constexpr u64 payload_srem(const MixConfig& c, u64 num, u64 den)
        {
            const std::int64_t n = payload_sext(c, num);
            const std::int64_t d = payload_sext(c, den | 1u);
            return static_cast<u64>(n % d) & c.full_mask;
        }
        FPSAN_HOST_DEVICE constexpr u64 payload_min(const MixConfig& c, u64 a, u64 b)
        {
            return payload_sext(c, a) < payload_sext(c, b) ? a : b;
        }
        FPSAN_HOST_DEVICE constexpr u64 payload_max(const MixConfig& c, u64 a, u64 b)
        {
            return payload_sext(c, a) < payload_sext(c, b) ? b : a;
        }

    } // namespace detail
} // namespace fpsan

#endif // FPSAN_DETAIL_MATH_HPP
