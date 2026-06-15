// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan/detail/fp8.hpp
// ----------------------------------------------------------------------------
// OCP FP8 scalar types (fp8_e4m3, fp8_e5m2) and the generic <-> f32 conversion
// used by Semantics::Native casts and by the Float-mode oracles in tests.
//
// The conversion routines (narrow_to_f32, f32_to_narrow) are a clean C++ port
// of iree/runtime/src/iree/base/internal/math.h (Apache-2.0): a single generic
// implementation parameterized by (exp_bits, mant_bits, have_inf, have_nan,
// bias_tweak, nan_as_neg_zero), reused by every narrow FP format we need.
//
// In FPSan mode the cast machinery does NOT call these -- fpsan::cast<float>
// merely sign-resizes the 8-bit payload to 32 bits (the Triton ext/trunc
// model), purely integer-level. The conversion is only needed for Float-mode
// parity with the hardware builtin (which uses the hardware fp8 unit).
// ----------------------------------------------------------------------------
#ifndef FPSAN_DETAIL_FP8_HPP
#define FPSAN_DETAIL_FP8_HPP

#include "fpsan/detail/config.hpp"
#include "fpsan/detail/traits.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace fpsan
{
    namespace detail
    {

        // Description of a less-than-32-bit floating-point format.
        struct FpFormat
        {
            int  exp_bits;
            int  mantissa_bits;
            bool have_infinity;
            bool have_nan;
            int  bias_tweak; // adjusts the standard bias = 2^(exp_bits-1)-1
            bool nan_as_neg_zero; // F8E*FNUZ-style: -0 encoding is reused for NaN
        };

        // Generic conversion narrow-FP-bits -> f32. Bits beyond the format's footprint
        // in `src` are ignored.
        FPSAN_HOST_DEVICE inline float narrow_to_f32(std::uint32_t src, const FpFormat& f)
        {
            const int           sign_shift    = f.exp_bits + f.mantissa_bits;
            const int           exp_shift     = f.mantissa_bits;
            const std::uint32_t sign_mask     = 1u << sign_shift;
            const std::uint32_t mantissa_mask = (1u << exp_shift) - 1u;
            const std::uint32_t exp_mask      = (1u << sign_shift) - (1u << exp_shift);
            const int           exp_bias      = f.bias_tweak + (1 << (f.exp_bits - 1)) - 1;
            const float         sgn           = (src & sign_mask) ? -1.f : 1.f;
            const std::uint32_t src_exp       = src & exp_mask;
            const std::uint32_t src_mantissa  = src & mantissa_mask;
            if(src_exp == exp_mask)
            {
                if(f.have_infinity)
                {
                    if(f.have_nan && src_mantissa)
                        return std::numeric_limits<float>::quiet_NaN();
                    return sgn * std::numeric_limits<float>::infinity();
                }
                if(f.have_nan)
                {
                    if(src_mantissa == mantissa_mask && !f.nan_as_neg_zero)
                        return std::numeric_limits<float>::quiet_NaN();
                }
            }
            else if(f.have_nan && f.nan_as_neg_zero && src == sign_mask)
            {
                return std::numeric_limits<float>::quiet_NaN();
            }
            else if(src_exp == 0)
            {
                return sgn
                       * std::ldexp(static_cast<float>(src_mantissa),
                                    1 - exp_bias - f.mantissa_bits);
            }
            return sgn
                   * std::ldexp(static_cast<float>(src_mantissa + (1u << f.mantissa_bits)),
                                (static_cast<int>(src_exp) >> exp_shift) - exp_bias
                                    - f.mantissa_bits);
        }

        // Round-to-nearest-even bias. Adds the right amount so that a subsequent right
        // shift by `shift_amount` performs RNE.
        FPSAN_HOST_DEVICE inline std::uint32_t bias_rne(std::uint32_t input, int shift_amount)
        {
            const std::uint32_t even = 1u << shift_amount;
            const std::uint32_t odd  = even >> 1;
            const std::uint32_t bias = (input & even) ? odd : (odd - 1u);
            return input + bias;
        }

        // Generic conversion f32 -> narrow-FP-bits with RNE. Returns the narrow bits in
        // the low (1 + exp_bits + mant_bits) bits; upper bits are zero.
        FPSAN_HOST_DEVICE inline std::uint32_t f32_to_narrow(float value, const FpFormat& f)
        {
            const int           dst_sign_shift    = f.exp_bits + f.mantissa_bits;
            const int           dst_exp_shift     = f.mantissa_bits;
            const std::uint32_t dst_exp_mask      = (1u << dst_sign_shift) - (1u << dst_exp_shift);
            const std::uint32_t dst_mantissa_mask = (1u << dst_exp_shift) - 1u;
            const int           dst_exp_bias      = f.bias_tweak + (1 << (f.exp_bits - 1)) - 1;
            // f32 constants.
            constexpr int           f32_exp_bits      = 8;
            constexpr int           f32_mantissa_bits = 23;
            constexpr int           f32_sign_shift    = 31;
            constexpr std::uint32_t f32_sign_mask     = 1u << 31;
            constexpr std::uint32_t f32_mantissa_mask = (1u << 23) - 1u;
            constexpr std::uint32_t f32_exp_mask      = (1u << 31) - (1u << 23);
            constexpr int           f32_exp_bias      = 127;
            std::uint32_t           u32;
            std::memcpy(&u32, &value, sizeof value);
            const std::uint32_t f32_sign     = u32 & f32_sign_mask;
            const std::uint32_t dst_sign     = f32_sign >> (f32_sign_shift - dst_sign_shift);
            const std::uint32_t f32_exp      = u32 & f32_exp_mask;
            const std::uint32_t f32_mantissa = u32 & f32_mantissa_mask;
            std::uint32_t       dst_exp      = 0;
            std::uint32_t       dst_mantissa = 0;
            bool                gen_nan      = false;
            bool                gen_inf      = false;
            if(f32_exp >= f32_exp_mask)
            {
                dst_exp = dst_exp_mask;
                if(f32_mantissa)
                    gen_nan = true;
                else
                    gen_inf = true;
            }
            else if(f32_exp == 0)
            {
                if(f.exp_bits == f32_exp_bits)
                {
                    const int sh = f32_mantissa_bits - f.mantissa_bits;
                    dst_mantissa = bias_rne(f32_mantissa, sh) >> sh;
                }
            }
            else
            {
                int       arith_exp = static_cast<int>(f32_exp >> f32_mantissa_bits) - f32_exp_bias;
                const int max_arith_exp
                    = ((1 << f.exp_bits) - 1 - static_cast<int>(f.have_infinity)) - dst_exp_bias;
                if(arith_exp > max_arith_exp)
                {
                    gen_inf = true;
                }
                else if(arith_exp + dst_exp_bias <= 0)
                {
                    dst_exp                 = 0;
                    const int dst_arith_exp = 1 - dst_exp_bias;
                    const int sh = f32_mantissa_bits - f.mantissa_bits - arith_exp + dst_arith_exp;
                    if(sh < 0 || sh > f32_mantissa_bits)
                    {
                        dst_mantissa = 0;
                    }
                    else
                    {
                        const std::uint32_t eff = (1u << f32_mantissa_bits) + f32_mantissa;
                        dst_mantissa            = bias_rne(eff, sh) >> sh;
                    }
                }
                else
                {
                    const int     sh     = f32_mantissa_bits - f.mantissa_bits;
                    std::uint32_t biased = bias_rne(f32_mantissa, sh);
                    if(biased > f32_mantissa_mask)
                    {
                        biased = 0;
                        ++arith_exp;
                    }
                    dst_exp = static_cast<std::uint32_t>(arith_exp + dst_exp_bias) << dst_exp_shift;
                    dst_mantissa = biased >> sh;
                    if(!f.have_infinity && dst_exp > dst_exp_mask)
                        gen_nan = true;
                }
            }
            if(gen_inf)
            {
                if(f.have_infinity)
                    return dst_sign | dst_exp_mask;
                if(f.have_nan)
                    gen_nan = true;
                else
                    return dst_sign | dst_exp_mask | dst_mantissa_mask;
            }
            if(gen_nan)
            {
                if(!f.have_nan)
                    return 0;
                if(f.nan_as_neg_zero)
                    return 1u << dst_sign_shift;
                return dst_sign | dst_exp_mask | dst_mantissa_mask;
            }
            if(f.nan_as_neg_zero && dst_exp == 0 && dst_mantissa == 0)
                return 0;
            return dst_sign | dst_exp | dst_mantissa;
        }

        // OCP FP8 format constants. The bit layout these specify is the *hardware*
        // (gfx12 RDNA4, gfx950 CDNA4) representation; FPSan adds its phi bijection on
        // top of these bits.
        inline constexpr FpFormat kFp8E4M3 = {/*exp_bits=*/4,
                                              /*mantissa_bits=*/3,
                                              /*have_infinity=*/false,
                                              /*have_nan=*/true,
                                              /*bias_tweak=*/0,
                                              /*nan_as_neg_zero=*/false};
        inline constexpr FpFormat kFp8E5M2 = {/*exp_bits=*/5,
                                              /*mantissa_bits=*/2,
                                              /*have_infinity=*/true,
                                              /*have_nan=*/true,
                                              /*bias_tweak=*/0,
                                              /*nan_as_neg_zero=*/false};
        // CDNA3's "FP8"/"BF8" formats are AMD FNUZ encodings: the exponent
        // bias is one greater than OCP E4M3/E5M2, there is no infinity, and
        // negative zero (0x80) encodes NaN. See the MI300 ISA, BF8 and FP8
        // Formats and Conversions.
        inline constexpr FpFormat kAmdFp8E4M3 = {/*exp_bits=*/4,
                                                 /*mantissa_bits=*/3,
                                                 /*have_infinity=*/false,
                                                 /*have_nan=*/true,
                                                 /*bias_tweak=*/1,
                                                 /*nan_as_neg_zero=*/true};
        inline constexpr FpFormat kAmdFp8E5M2 = {/*exp_bits=*/5,
                                                 /*mantissa_bits=*/2,
                                                 /*have_infinity=*/false,
                                                 /*have_nan=*/true,
                                                 /*bias_tweak=*/1,
                                                 /*nan_as_neg_zero=*/true};
        // gfx1250 "E5M3" scale format: an
        // 8-bit UNSIGNED magnitude format -- 5 exponent bits (bias 15), 3 mantissa
        // bits, NO sign bit, no infinity, the single all-ones byte 0xFF is NaN,
        // 0x00 is zero, denormals supported. Modeled here as exp_bits=5,
        // mantissa_bits=3: the generic routines put the (never-set) sign bit at bit
        // 8, outside the byte, so an 8-bit input/output is treated as non-negative.
        // Verified values: 0x08->2^-14 (min normal),
        // 0xFE->114688 (max normal), 0x07->0.875*2^-14 (max denorm), 0x01->2^-17
        // (min denorm), 0xFF->NaN. Used by cvt_{f32_fp8,pk_fp8_f32,sr_fp8_f32}_e5m3.
        inline constexpr FpFormat kFp8E5M3 = {/*exp_bits=*/5,
                                              /*mantissa_bits=*/3,
                                              /*have_infinity=*/false,
                                              /*have_nan=*/true,
                                              /*bias_tweak=*/0,
                                              /*nan_as_neg_zero=*/false};

        // OCP MX sub-byte formats (gfx950 cvt_scalef32_* operands). None has inf or
        // NaN: overflow saturates to max-finite and f32 NaN/Inf maps to 0. The generic
        // narrow_to_f32 / f32_to_narrow above already realize exactly that from
        // have_infinity=false, have_nan=false (the gen_inf path returns all-ones
        // exp+mant = max finite; the gen_nan path returns 0). Biases: E2M1/E2M3 -> 1,
        // E3M2 -> 3. Representable magnitudes: fp4 {0,.5,1,1.5,2,3,4,6} (max 6),
        // fp6 e2m3 max 7.5, bf6 e3m2 max 28.
        inline constexpr FpFormat kFp4E2M1 = {/*exp_bits=*/2,
                                              /*mantissa_bits=*/1,
                                              /*have_infinity=*/false,
                                              /*have_nan=*/false,
                                              /*bias_tweak=*/0,
                                              /*nan_as_neg_zero=*/false};
        inline constexpr FpFormat kFp6E2M3 = {/*exp_bits=*/2,
                                              /*mantissa_bits=*/3,
                                              /*have_infinity=*/false,
                                              /*have_nan=*/false,
                                              /*bias_tweak=*/0,
                                              /*nan_as_neg_zero=*/false};
        inline constexpr FpFormat kBf6E3M2 = {/*exp_bits=*/3,
                                              /*mantissa_bits=*/2,
                                              /*have_infinity=*/false,
                                              /*have_nan=*/false,
                                              /*bias_tweak=*/0,
                                              /*nan_as_neg_zero=*/false};

    } // namespace detail

    // OCP FP8 scalar types. Trivially-copyable 1-byte structs. Designed to be the
    // element type of `Value<...>` (via the existing fp_traits specializations) and
    // of a per-lane fragment struct (declared in amdgcn_matrix.hpp when needed).
    struct fp8_e4m3
    {
        std::uint8_t bits;
        FPSAN_HOST_DEVICE constexpr fp8_e4m3()
            : bits(0)
        {
        }
        FPSAN_HOST_DEVICE constexpr explicit fp8_e4m3(std::uint8_t b)
            : bits(b)
        {
        }
        FPSAN_HOST_DEVICE explicit fp8_e4m3(float v)
            : bits(static_cast<std::uint8_t>(detail::f32_to_narrow(v, detail::kFp8E4M3)))
        {
        }
        FPSAN_HOST_DEVICE operator float() const
        {
            return detail::narrow_to_f32(bits, detail::kFp8E4M3);
        }
        // Value::cmp_t derivation needs operator< on the float_type. Order by the
        // represented float (the only natural meaning, used in tests / future fmin).
        FPSAN_HOST_DEVICE friend bool operator<(fp8_e4m3 a, fp8_e4m3 b)
        {
            return static_cast<float>(a) < static_cast<float>(b);
        }
        FPSAN_HOST_DEVICE friend bool operator==(fp8_e4m3 a, fp8_e4m3 b)
        {
            return a.bits == b.bits;
        }
        FPSAN_HOST_DEVICE friend bool operator!=(fp8_e4m3 a, fp8_e4m3 b)
        {
            return !(a == b);
        }
    };

    struct fp8_e5m2
    {
        std::uint8_t bits;
        FPSAN_HOST_DEVICE constexpr fp8_e5m2()
            : bits(0)
        {
        }
        FPSAN_HOST_DEVICE constexpr explicit fp8_e5m2(std::uint8_t b)
            : bits(b)
        {
        }
        FPSAN_HOST_DEVICE explicit fp8_e5m2(float v)
            : bits(static_cast<std::uint8_t>(detail::f32_to_narrow(v, detail::kFp8E5M2)))
        {
        }
        FPSAN_HOST_DEVICE operator float() const
        {
            return detail::narrow_to_f32(bits, detail::kFp8E5M2);
        }
        FPSAN_HOST_DEVICE friend bool operator<(fp8_e5m2 a, fp8_e5m2 b)
        {
            return static_cast<float>(a) < static_cast<float>(b);
        }
        FPSAN_HOST_DEVICE friend bool operator==(fp8_e5m2 a, fp8_e5m2 b)
        {
            return a.bits == b.bits;
        }
        FPSAN_HOST_DEVICE friend bool operator!=(fp8_e5m2 a, fp8_e5m2 b)
        {
            return !(a == b);
        }
    };

#define FPSAN_DEFINE_FP8_SCALAR(NAME, FORMAT)                                   \
    struct NAME                                                                 \
    {                                                                           \
        std::uint8_t bits;                                                      \
        FPSAN_HOST_DEVICE constexpr NAME()                                      \
            : bits(0)                                                           \
        {                                                                       \
        }                                                                       \
        FPSAN_HOST_DEVICE constexpr explicit NAME(std::uint8_t b)               \
            : bits(b)                                                           \
        {                                                                       \
        }                                                                       \
        FPSAN_HOST_DEVICE explicit NAME(float v)                                \
            : bits(static_cast<std::uint8_t>(detail::f32_to_narrow(v, FORMAT))) \
        {                                                                       \
        }                                                                       \
        FPSAN_HOST_DEVICE operator float() const                                \
        {                                                                       \
            return detail::narrow_to_f32(bits, FORMAT);                         \
        }                                                                       \
        FPSAN_HOST_DEVICE friend bool operator<(NAME a, NAME b)                 \
        {                                                                       \
            return static_cast<float>(a) < static_cast<float>(b);               \
        }                                                                       \
        FPSAN_HOST_DEVICE friend bool operator==(NAME a, NAME b)                \
        {                                                                       \
            return a.bits == b.bits;                                            \
        }                                                                       \
        FPSAN_HOST_DEVICE friend bool operator!=(NAME a, NAME b)                \
        {                                                                       \
            return !(a == b);                                                   \
        }                                                                       \
    };

    FPSAN_DEFINE_FP8_SCALAR(amd_fp8_e4m3, detail::kAmdFp8E4M3)
    FPSAN_DEFINE_FP8_SCALAR(amd_fp8_e5m2, detail::kAmdFp8E5M2)

#undef FPSAN_DEFINE_FP8_SCALAR

    namespace detail
    {
        FPSAN_DEFINE_FP_TRAITS(::fpsan::fp8_e4m3, 3, 4, 7);
        FPSAN_DEFINE_FP_TRAITS(::fpsan::fp8_e5m2, 2, 5, 15);
        FPSAN_DEFINE_FP_TRAITS(::fpsan::amd_fp8_e4m3, 3, 4, 8);
        FPSAN_DEFINE_FP_TRAITS(::fpsan::amd_fp8_e5m2, 2, 5, 16);
        // Note: there is intentionally NO fp_traits for the gfx1250 "E5M3" scale
        // format. E5M3 is an 8-bit UNSIGNED magnitude format (5 exp bits bias 15,
        // 3 mantissa bits, no sign bit), so
        // 1 + exp + mant = 9 != 8 and it cannot be a signed-fp Value element type.
        // Its decode/encode is reachable via narrow_to_f32 / f32_to_narrow with
        // kFp8E5M3 below (sign bit lands outside the byte -> effectively unsigned);
        // its FPSan payload model is a plain width-8 resize (see amdgcn_cvt.hpp).
// Defined in detail/traits.hpp; fp8 holds the last expansions, so retire the
// generator here rather than leak it into translation units that pull fpsan.hpp.
#undef FPSAN_DEFINE_FP_TRAITS
    } // namespace detail

} // namespace fpsan

#endif // FPSAN_DETAIL_FP8_HPP
