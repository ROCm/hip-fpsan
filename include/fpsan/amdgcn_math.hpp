// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan/amdgcn_math.hpp
// ----------------------------------------------------------------------------
// FPSan wrappers for AMDGPU math intrinsics (rcp, rsq, sqrt, sin, cos, log,
// exp2, fract, tanh, fmed3, ...). Opt-in (not pulled by <fpsan/fpsan.hpp>).
//
// Float mode forwards to the matching __builtin_amdgcn_*. FPSan mode dispatches
// to the corresponding fpsan:: tagged op in math.hpp -- deterministic, op-
// distinct, payload-domain math that captures "I called this transcendental"
// without trying to actually evaluate it in the payload ring.
//
// Note: customers porting code that calls __builtin_amdgcn_sinh / cosh might
// expect hyperbolic functions, but these AMD builtins are actually f16-typed
// sin/cos. Our wrappers preserve that (sinh -> fpsan::sin, cosh -> fpsan::cos).
// For real hyperbolic tanh, use the dedicated __builtin_amdgcn_tanh{f,h}.
// ----------------------------------------------------------------------------
#ifndef FPSAN_AMDGCN_MATH_HPP
#define FPSAN_AMDGCN_MATH_HPP

#include "fpsan/amdgcn_matrix.hpp" // v4* fp8 native aliases (dot4 fp8)
#include "fpsan/cast.hpp"
#include "fpsan/detail/native_vec.hpp"
#include "fpsan/math.hpp"
#include "fpsan/value.hpp"

#if !defined(__HIP__) && !defined(__CUDACC__)
#error "fpsan/amdgcn_math.hpp is GPU-only; compile as HIP (or CUDA)."
#endif

#if defined(__GFX11__) || defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) \
    || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
#define FPSAN_AMDGCN_MATH_DEVICE_IS_GFX11 1
#else
#define FPSAN_AMDGCN_MATH_DEVICE_IS_GFX11 0
#endif

#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
#define FPSAN_AMDGCN_MATH_DEVICE_IS_GFX94X 1
#else
#define FPSAN_AMDGCN_MATH_DEVICE_IS_GFX94X 0
#endif

namespace fpsan
{

// Define one unary FPSan wrapper for an AMD intrinsic. Float mode calls the
// builtin; FPSan mode delegates to the named fpsan:: tagged op.
#define FPSAN_DEFINE_AMDGCN_UNARY(name, type, FPSAN_OP, BUILTIN)                      \
    template <Semantics S = Semantics::Native, Conversions C = Conversions::Explicit> \
    FPSAN_DEVICE Value<type, S, C> name(Value<type, S, C> v)                          \
    {                                                                                 \
        if constexpr(S == Semantics::Native)                                          \
            return Value<type, S, C>(BUILTIN(v.to_float()));                          \
        else                                                                          \
            return fpsan::FPSAN_OP(v);                                                \
    }

#define FPSAN_DEFINE_AMDGCN_TERNARY(name, type, FPSAN_OP, BUILTIN)                       \
    template <Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>    \
    FPSAN_DEVICE Value<type, S, C> name(                                                 \
        Value<type, S, C> a, Value<type, S, C> b, Value<type, S, C> c)                   \
    {                                                                                    \
        if constexpr(S == Semantics::Native)                                             \
            return Value<type, S, C>(BUILTIN(a.to_float(), b.to_float(), c.to_float())); \
        else                                                                             \
            return fpsan::FPSAN_OP(a, b, c);                                             \
    }

    // ---- reciprocal ----
    FPSAN_DEFINE_AMDGCN_UNARY(amdgcn_rcpf, float, rcp, __builtin_amdgcn_rcpf)
    FPSAN_DEFINE_AMDGCN_UNARY(amdgcn_rcp, double, rcp, __builtin_amdgcn_rcp)
    FPSAN_DEFINE_AMDGCN_UNARY(amdgcn_rcph, _Float16, rcp, __builtin_amdgcn_rcph)

    // ---- sqrt ----
    FPSAN_DEFINE_AMDGCN_UNARY(amdgcn_sqrtf, float, sqrt, __builtin_amdgcn_sqrtf)
    FPSAN_DEFINE_AMDGCN_UNARY(amdgcn_sqrt, double, sqrt, __builtin_amdgcn_sqrt)
    FPSAN_DEFINE_AMDGCN_UNARY(amdgcn_sqrth, _Float16, sqrt, __builtin_amdgcn_sqrth)

    // ---- rsq (reciprocal sqrt = our fpsan::rsqrt) ----
    FPSAN_DEFINE_AMDGCN_UNARY(amdgcn_rsqf, float, rsqrt, __builtin_amdgcn_rsqf)
    FPSAN_DEFINE_AMDGCN_UNARY(amdgcn_rsq, double, rsqrt, __builtin_amdgcn_rsq)
    FPSAN_DEFINE_AMDGCN_UNARY(amdgcn_rsqh, _Float16, rsqrt, __builtin_amdgcn_rsqh)

    // rsq_clamp: clamp variant; we model it as plain rsqrt in FPSan (no
    // distinct payload semantics in the existing tagged set).
    FPSAN_DEFINE_AMDGCN_UNARY(amdgcn_rsq_clampf, float, rsqrt, __builtin_amdgcn_rsq_clampf)
    FPSAN_DEFINE_AMDGCN_UNARY(amdgcn_rsq_clamp, double, rsqrt, __builtin_amdgcn_rsq_clamp)

    // ---- sin / cos (note: sinh/cosh in AMD's naming are f16 sin/cos, not the
    // hyperbolic functions) ----
    FPSAN_DEFINE_AMDGCN_UNARY(amdgcn_sinf, float, sin, __builtin_amdgcn_sinf)
    FPSAN_DEFINE_AMDGCN_UNARY(amdgcn_cosf, float, cos, __builtin_amdgcn_cosf)
    FPSAN_DEFINE_AMDGCN_UNARY(amdgcn_sinh, _Float16, sin, __builtin_amdgcn_sinh)
    FPSAN_DEFINE_AMDGCN_UNARY(amdgcn_cosh, _Float16, cos, __builtin_amdgcn_cosh)

    // ---- log / log_clamp / exp2 ----
    FPSAN_DEFINE_AMDGCN_UNARY(amdgcn_logf, float, log, __builtin_amdgcn_logf)
// `__has_builtin` reports log_clampf on gfx1100 and gfx942, but direct lowering
// probes reject it as unsupported on those subtargets.
#if !defined(__HIP_DEVICE_COMPILE__)                                              \
    || (!FPSAN_AMDGCN_MATH_DEVICE_IS_GFX11 && !FPSAN_AMDGCN_MATH_DEVICE_IS_GFX94X \
        && __has_builtin(__builtin_amdgcn_log_clampf))
    FPSAN_DEFINE_AMDGCN_UNARY(amdgcn_log_clampf, float, log, __builtin_amdgcn_log_clampf)
#endif
    FPSAN_DEFINE_AMDGCN_UNARY(amdgcn_exp2f, float, exp2, __builtin_amdgcn_exp2f)

    // ---- fract ----
    FPSAN_DEFINE_AMDGCN_UNARY(amdgcn_fractf, float, fract, __builtin_amdgcn_fractf)
    FPSAN_DEFINE_AMDGCN_UNARY(amdgcn_fract, double, fract, __builtin_amdgcn_fract)
    FPSAN_DEFINE_AMDGCN_UNARY(amdgcn_fracth, _Float16, fract, __builtin_amdgcn_fracth)

    // ---- tanh (real hyperbolic) ----
// tanh needs the target's `tanh-insts` feature. RDNA3/gfx11 lacks that feature
// in the audited LLVM/HIP surface, so hide the wrappers there.
#if !defined(__HIP_DEVICE_COMPILE__) \
    || (!FPSAN_AMDGCN_MATH_DEVICE_IS_GFX11 && __has_builtin(__builtin_amdgcn_tanhf))
    FPSAN_DEFINE_AMDGCN_UNARY(amdgcn_tanhf, float, tanh, __builtin_amdgcn_tanhf)
#endif
#if !defined(__HIP_DEVICE_COMPILE__) \
    || (!FPSAN_AMDGCN_MATH_DEVICE_IS_GFX11 && __has_builtin(__builtin_amdgcn_tanhh))
    FPSAN_DEFINE_AMDGCN_UNARY(amdgcn_tanhh, _Float16, tanh, __builtin_amdgcn_tanhh)
#endif

    // ---- bf16 transcendentals (gfx1250 "bf16-trans-insts") ----
    // gfx1250 adds native bf16 rcp / rsq / sqrt / sin / cos / exp2 / log / tanh
    // (each is __bf16(__bf16)). These features are exclusive to gfx1250: RDNA4
    // and CDNA have no bf16 transcendental unit. Float mode calls the builtin;
    // FPSan mode routes to the SAME tagged op as the f16/f32 variants -- in the
    // FPSan ring it is the op identity (sin vs cos vs rsqrt ...), not the element
    // type, that determines the payload, so the bf16 wrapper is just another
    // entry point to the existing tagged op. One representative __has_builtin
    // gate covers the whole block (all eight share bf16-trans-insts).
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_rcp_bf16)
    FPSAN_DEFINE_AMDGCN_UNARY(amdgcn_rcp_bf16, __bf16, rcp, __builtin_amdgcn_rcp_bf16)
    FPSAN_DEFINE_AMDGCN_UNARY(amdgcn_rsq_bf16, __bf16, rsqrt, __builtin_amdgcn_rsq_bf16)
    FPSAN_DEFINE_AMDGCN_UNARY(amdgcn_sqrt_bf16, __bf16, sqrt, __builtin_amdgcn_sqrt_bf16)
    FPSAN_DEFINE_AMDGCN_UNARY(amdgcn_sin_bf16, __bf16, sin, __builtin_amdgcn_sin_bf16)
    FPSAN_DEFINE_AMDGCN_UNARY(amdgcn_cos_bf16, __bf16, cos, __builtin_amdgcn_cos_bf16)
    FPSAN_DEFINE_AMDGCN_UNARY(amdgcn_exp2_bf16, __bf16, exp2, __builtin_amdgcn_exp2_bf16)
    FPSAN_DEFINE_AMDGCN_UNARY(amdgcn_log_bf16, __bf16, log, __builtin_amdgcn_log_bf16)
    FPSAN_DEFINE_AMDGCN_UNARY(amdgcn_tanh_bf16, __bf16, tanh, __builtin_amdgcn_tanh_bf16)
#endif

    // ---- fmed3 (median of three) ----
    FPSAN_DEFINE_AMDGCN_TERNARY(amdgcn_fmed3f, float, fmed3, __builtin_amdgcn_fmed3f)
    FPSAN_DEFINE_AMDGCN_TERNARY(amdgcn_fmed3h, _Float16, fmed3, __builtin_amdgcn_fmed3h)

#undef FPSAN_DEFINE_AMDGCN_UNARY
#undef FPSAN_DEFINE_AMDGCN_TERNARY

    // =============================================================================
    // fdot2: 2-element dot product of half-precision (f16 / bf16) vectors,
    // accumulated into f32 or f16. The mini-MMA. In FPSan mode, the result is
    // just the explicit ring expression `acc + cast<acc>(a[0])*cast<acc>(b[0]) +
    // cast<acc>(a[1])*cast<acc>(b[1])` -- which is exact because the FPSan ring
    // is integer add/mul mod 2^w on the accumulator type's payload.
    // =============================================================================

    // v2h_native / v2bf_native / v2i16_native come from
    // fpsan/detail/native_vec.hpp.

// fdot2 family: each builtin needs a specific dotN-insts target feature
// (dot7-insts / dot9-insts / etc.).  __has_builtin gates each variant so
// the header still compiles on archs that don't expose the dot family
// (e.g. gfx950, which uses MFMA for these patterns instead).

// fdot2: a, b are v2h; acc is f32. `Clamp` is the builtin's clamp flag.
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_fdot2)
    template <bool        Clamp = false,
              Semantics   S     = Semantics::Native,
              Conversions C     = Conversions::Explicit>
    FPSAN_DEVICE Value<float, S, C>
        amdgcn_fdot2(Value<v2h_native, S, C> a, Value<v2h_native, S, C> b, Value<float, S, C> c)
    {
        if constexpr(S == Semantics::Native)
        {
            return Value<float, S, C>(
                __builtin_amdgcn_fdot2(a.to_float(), b.to_float(), c.to_float(), Clamp));
        }
        else
        {
            auto acc = c;
            acc      = acc + fpsan::cast<float>(a.get(0)) * fpsan::cast<float>(b.get(0));
            acc      = acc + fpsan::cast<float>(a.get(1)) * fpsan::cast<float>(b.get(1));
            return acc;
        }
    }
#endif

// fdot2_f16_f16: a, b are v2h; acc is _Float16.
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_fdot2_f16_f16)
    template <Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>
    FPSAN_DEVICE Value<_Float16, S, C> amdgcn_fdot2_f16_f16(Value<v2h_native, S, C> a,
                                                            Value<v2h_native, S, C> b,
                                                            Value<_Float16, S, C>   c)
    {
        if constexpr(S == Semantics::Native)
        {
            return Value<_Float16, S, C>(
                __builtin_amdgcn_fdot2_f16_f16(a.to_float(), b.to_float(), c.to_float()));
        }
        else
        {
            auto acc = c;
            acc      = acc + a.get(0) * b.get(0);
            acc      = acc + a.get(1) * b.get(1);
            return acc;
        }
    }
#endif

// fdot2_bf16_bf16: a, b are v2bf; acc is __bf16.  Clang exposes the
// BF16-valued builtin through short/v2short ABI types, so Float mode bit-casts
// at the boundary and the public wrapper stays typed as __bf16.
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_fdot2_bf16_bf16)
    template <Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>
    FPSAN_DEVICE Value<__bf16, S, C> amdgcn_fdot2_bf16_bf16(Value<v2bf_native, S, C> a,
                                                            Value<v2bf_native, S, C> b,
                                                            Value<__bf16, S, C>      c)
    {
        if constexpr(S == Semantics::Native)
        {
            const v2i16_native ai = __builtin_bit_cast(v2i16_native, a.to_float());
            const v2i16_native bi = __builtin_bit_cast(v2i16_native, b.to_float());
            const short        ci = __builtin_bit_cast(short, c.to_float());
            const short        di = __builtin_amdgcn_fdot2_bf16_bf16(ai, bi, ci);
            return Value<__bf16, S, C>(__builtin_bit_cast(__bf16, di));
        }
        else
        {
            auto acc = c;
            acc      = acc + a.get(0) * b.get(0);
            acc      = acc + a.get(1) * b.get(1);
            return acc;
        }
    }
#endif

// fdot2_f32_bf16: a, b are v2bf (passed to the builtin as v2i16 -- the
// builtin's signature predates Clang's __bf16 type); acc is f32.
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_fdot2_f32_bf16)
    template <bool        Clamp = false,
              Semantics   S     = Semantics::Native,
              Conversions C     = Conversions::Explicit>
    FPSAN_DEVICE Value<float, S, C> amdgcn_fdot2_f32_bf16(Value<v2bf_native, S, C> a,
                                                          Value<v2bf_native, S, C> b,
                                                          Value<float, S, C>       c)
    {
        if constexpr(S == Semantics::Native)
        {
            v2i16_native ai = __builtin_bit_cast(v2i16_native, a.to_float());
            v2i16_native bi = __builtin_bit_cast(v2i16_native, b.to_float());
            return Value<float, S, C>(__builtin_amdgcn_fdot2_f32_bf16(ai, bi, c.to_float(), Clamp));
        }
        else
        {
            auto acc = c;
            acc      = acc + fpsan::cast<float>(a.get(0)) * fpsan::cast<float>(b.get(0));
            acc      = acc + fpsan::cast<float>(a.get(1)) * fpsan::cast<float>(b.get(1));
            return acc;
        }
    }
#endif

// =============================================================================
// dot4: 4-element dot product of packed 8-bit FP vectors (fp8 e4m3 / bf8 e5m2)
// accumulated into f32. The 8-bit micro-MMA: same shape as fdot2 but twice the
// K and a per-lane fp8/bf8 byte-vector instead of a v2 half. In
// FPSan mode the result is the explicit ring expression
//   acc + sum_{k=0..3} cast<f32>(a[k]) * cast<f32>(b[k]),
// which is exact in the payload ring (integer add/mul mod 2^32 on the
// accumulator's f32 payload). CDNA3/gfx94x uses AMD FNUZ FP8/BF8 encodings;
// RDNA4 uses OCP encodings. Gated by __has_builtin so the header still compiles
// on archs that don't have the dot11-insts feature (e.g. gfx950).
// =============================================================================
#define FPSAN_DEFINE_AMDGCN_DOT4_FP8(NAME, AVec, BVec, BUILTIN)                          \
    template <Semantics S = Semantics::Native, Conversions C = Conversions::Explicit>    \
    FPSAN_DEVICE Value<float, S, C> NAME(                                                \
        Value<AVec, S, C> a, Value<BVec, S, C> b, Value<float, S, C> c)                  \
    {                                                                                    \
        if constexpr(S == Semantics::Native)                                             \
        {                                                                                \
            const unsigned ai = __builtin_bit_cast(unsigned, a.to_float());              \
            const unsigned bi = __builtin_bit_cast(unsigned, b.to_float());              \
            return Value<float, S, C>(BUILTIN(ai, bi, c.to_float()));                    \
        }                                                                                \
        else                                                                             \
        {                                                                                \
            auto acc = c;                                                                \
            for(int k = 0; k < 4; ++k)                                                   \
                acc = acc + fpsan::cast<float>(a.get(k)) * fpsan::cast<float>(b.get(k)); \
            return acc;                                                                  \
        }                                                                                \
    }

#if !defined(__HIP_DEVICE_COMPILE__)                                             \
    || (__has_builtin(__builtin_amdgcn_dot4_f32_fp8_fp8) && !defined(__gfx940__) \
        && !defined(__gfx941__) && !defined(__gfx942__))
    FPSAN_DEFINE_AMDGCN_DOT4_FP8(amdgcn_dot4_f32_fp8_fp8,
                                 v4e4m3_native,
                                 v4e4m3_native,
                                 __builtin_amdgcn_dot4_f32_fp8_fp8)
    FPSAN_DEFINE_AMDGCN_DOT4_FP8(amdgcn_dot4_f32_fp8_bf8,
                                 v4e4m3_native,
                                 v4e5m2_native,
                                 __builtin_amdgcn_dot4_f32_fp8_bf8)
    FPSAN_DEFINE_AMDGCN_DOT4_FP8(amdgcn_dot4_f32_bf8_fp8,
                                 v4e5m2_native,
                                 v4e4m3_native,
                                 __builtin_amdgcn_dot4_f32_bf8_fp8)
    FPSAN_DEFINE_AMDGCN_DOT4_FP8(amdgcn_dot4_f32_bf8_bf8,
                                 v4e5m2_native,
                                 v4e5m2_native,
                                 __builtin_amdgcn_dot4_f32_bf8_bf8)
#endif

#if !defined(__HIP_DEVICE_COMPILE__)                     \
    || (__has_builtin(__builtin_amdgcn_dot4_f32_fp8_fp8) \
        && (defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)))
    FPSAN_DEFINE_AMDGCN_DOT4_FP8(amdgcn_dot4_f32_fp8_fp8,
                                 v4amd_e4m3_native,
                                 v4amd_e4m3_native,
                                 __builtin_amdgcn_dot4_f32_fp8_fp8)
    FPSAN_DEFINE_AMDGCN_DOT4_FP8(amdgcn_dot4_f32_fp8_bf8,
                                 v4amd_e4m3_native,
                                 v4amd_e5m2_native,
                                 __builtin_amdgcn_dot4_f32_fp8_bf8)
    FPSAN_DEFINE_AMDGCN_DOT4_FP8(amdgcn_dot4_f32_bf8_fp8,
                                 v4amd_e5m2_native,
                                 v4amd_e4m3_native,
                                 __builtin_amdgcn_dot4_f32_bf8_fp8)
    FPSAN_DEFINE_AMDGCN_DOT4_FP8(amdgcn_dot4_f32_bf8_bf8,
                                 v4amd_e5m2_native,
                                 v4amd_e5m2_native,
                                 __builtin_amdgcn_dot4_f32_bf8_bf8)
#endif
#undef FPSAN_DEFINE_AMDGCN_DOT4_FP8

// =============================================================================
// ldexp: v * 2^n (n is an integer exponent, not a Value). Float mode forwards
// to the builtin. FPSan mode models it as a payload-ring multiply by the
// constant 2^n -- the SAME "scale by a power of two" operation the MX scaled
// cvt wrappers use (`v * scale`). This is the natural and consistent FPSan
// semantics for an exact power-of-two rescale, so a downstream FPSan
// implementation has an unambiguous reference to match.
// =============================================================================
#define FPSAN_DEFINE_AMDGCN_LDEXP(NAME, FT, BUILTIN)                                  \
    template <Semantics S = Semantics::Native, Conversions C = Conversions::Explicit> \
    FPSAN_DEVICE Value<FT, S, C> NAME(Value<FT, S, C> v, int n)                       \
    {                                                                                 \
        if constexpr(S == Semantics::Native)                                          \
            return Value<FT, S, C>(BUILTIN(v.to_float(), n));                         \
        else                                                                          \
            return v * Value<FT, S, C>(static_cast<FT>(__builtin_ldexp(1.0, n)));     \
    }
    FPSAN_DEFINE_AMDGCN_LDEXP(amdgcn_ldexpf, float, __builtin_amdgcn_ldexpf)
    FPSAN_DEFINE_AMDGCN_LDEXP(amdgcn_ldexp, double, __builtin_amdgcn_ldexp)
    FPSAN_DEFINE_AMDGCN_LDEXP(amdgcn_ldexph, _Float16, __builtin_amdgcn_ldexph)
#undef FPSAN_DEFINE_AMDGCN_LDEXP

    // Deliberately LEFT OUT -- no faithful FPSan (payload-ring) model exists, so
    // shipping a wrapper would risk a silent wrong answer in an authoritative
    // suite. The FPSan payload is a *symbolic* integer (op-distinguishing scramble
    // / modular arithmetic), NOT the operand's IEEE bit pattern. The following all
    // require inspecting or producing actual float bit fields and/or have data-
    // dependent special-case behavior the symbolic ring cannot represent:
    //
    //   amdgcn_frexp_mant{,f,h}  -- extracts the IEEE mantissa field (a bit query
    //                               on the real float; undefined on a symbolic
    //                               payload).
    //   amdgcn_frexp_exp{,f,h}   -- extracts the IEEE exponent as an int (same).
    //   amdgcn_div_scale{,f}     -- IEEE division step: returns a possibly-2^64-
    //                               rescaled operand AND a VCC flag chosen by the
    //                               operands' exponent ranges (a paired int+bool
    //                               output with bit-pattern-dependent control
    //                               flow).
    //   amdgcn_div_fmas{,f}      -- fma whose result is multiplied by 2^(+/-64)
    //                               depending on a carry-in bit (VCC) -- the scale
    //                               is selected by hidden state, not the values.
    //   amdgcn_div_fixup{,f,h}   -- patches NaN/Inf/zero/sign special cases of a
    //                               division using all three operands' classes.
    //   amdgcn_trig_preop{,f}    -- emits a 53-bit segment of 2/pi for trig
    //                               argument reduction, indexed by the operand's
    //                               exponent (a table lookup on the bit pattern).
    //   amdgcn_cube{id,sc,tc,ma} -- cubemap face/coord selection (graphics): out
    //                               of FPSan's numeric scope.
    //
    // The shared rationale: an FPSan payload is a symbolic integer, not the
    // operand's IEEE bit pattern, so any op that reads/writes float bit fields
    // or branches on exponent/class has no faithful payload-ring image.

} // namespace fpsan

#undef FPSAN_AMDGCN_MATH_DEVICE_IS_GFX94X
#undef FPSAN_AMDGCN_MATH_DEVICE_IS_GFX11

#endif // FPSAN_AMDGCN_MATH_HPP
