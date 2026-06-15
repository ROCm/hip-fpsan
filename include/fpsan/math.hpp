// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan/math.hpp
// ----------------------------------------------------------------------------
// Standard-library-style math on Value, found by argument-dependent lookup
// (write `using std::exp; ... exp(x);` in generic code, or `fpsan::exp(x)`).
//
//   semantics == Semantics::Native : the real std:: function (drop-in).
//   semantics == Semantics::Triton : Tritons FPSan handler (see detail/math).
//
// In FPSan mode exp/exp2/sin/cos carry real algebraic identities, while
// log/log2/sqrt/precise_sqrt/rsqrt/erf/floor/ceil (and rcp/fract/tanh) are
// deterministic op-distinguishing scrambles (not real math) -- exactly as
// Triton's sanitizer treats them.
// ----------------------------------------------------------------------------
#ifndef FPSAN_MATH_HPP
#define FPSAN_MATH_HPP

#include "fpsan/detail/config.hpp"
#include "fpsan/detail/math.hpp"
#include "fpsan/value.hpp"

#include <cmath>
#include <type_traits>

namespace fpsan
{
    namespace detail
    {
        // Precision used for the native (Semantics::Native) math path. _Float16 /
        // __bf16 promote to float, matching how the standard library handles them.
        template <class FT>
        using compute_t = std::conditional_t<std::is_same_v<FT, double>, double, float>;
    } // namespace detail

// p is the payload of x; build the result wrapper from a computed payload.
#define FPSAN_FROM_PAYLOAD(F, expr) F::from_fpsan_payload(static_cast<typename F::bits_type>(expr))

// ---- algebraic unary: exp, exp2, sin, cos ----------------------------------
#define FPSAN_DEFINE_ALGEBRAIC_UNARY(NAME, PAYLOAD_FN, STD_FN)                                   \
    template <class FT, Semantics S, Conversions C>                                              \
    FPSAN_HOST_DEVICE Value<FT, S, C> NAME(Value<FT, S, C> x)                                    \
    {                                                                                            \
        using F = Value<FT, S, C>;                                                               \
        if constexpr(F::semantics == Semantics::Triton)                                          \
            return FPSAN_FROM_PAYLOAD(F, detail::PAYLOAD_FN(F::config, x.fpsan_payload()));      \
        else                                                                                     \
            return F(static_cast<FT>(STD_FN(static_cast<detail::compute_t<FT>>(x.to_float())))); \
    }
    FPSAN_DEFINE_ALGEBRAIC_UNARY(exp, payload_exp, std::exp)
    FPSAN_DEFINE_ALGEBRAIC_UNARY(exp2, payload_exp2, std::exp2)
#undef FPSAN_DEFINE_ALGEBRAIC_UNARY

    // cos / sin share one payload computation.
    template <class FT, Semantics S, Conversions C>
    FPSAN_HOST_DEVICE Value<FT, S, C> cos(Value<FT, S, C> x)
    {
        using F = Value<FT, S, C>;
        if constexpr(F::semantics == Semantics::Triton)
            return FPSAN_FROM_PAYLOAD(F, detail::payload_cos_sin(F::config, x.fpsan_payload()).cos);
        else
            return F(static_cast<FT>(std::cos(static_cast<detail::compute_t<FT>>(x.to_float()))));
    }
    template <class FT, Semantics S, Conversions C>
    FPSAN_HOST_DEVICE Value<FT, S, C> sin(Value<FT, S, C> x)
    {
        using F = Value<FT, S, C>;
        if constexpr(F::semantics == Semantics::Triton)
            return FPSAN_FROM_PAYLOAD(F, detail::payload_cos_sin(F::config, x.fpsan_payload()).sin);
        else
            return F(static_cast<FT>(std::sin(static_cast<detail::compute_t<FT>>(x.to_float()))));
    }

// ---- tagged unary: log, log2, sqrt, rsqrt, erf, floor, ceil ----------------
#define FPSAN_DEFINE_TAGGED_UNARY(NAME, OPID, NATIVE)                                              \
    template <class FT, Semantics S, Conversions C>                                                \
    FPSAN_HOST_DEVICE Value<FT, S, C> NAME(Value<FT, S, C> x)                                      \
    {                                                                                              \
        using F = Value<FT, S, C>;                                                                 \
        if constexpr(F::semantics == Semantics::Triton)                                            \
            return FPSAN_FROM_PAYLOAD(F,                                                           \
                                      detail::payload_tagged_unary(                                \
                                          F::config, x.fpsan_payload(), detail::UnaryOpId::OPID)); \
        else                                                                                       \
        {                                                                                          \
            const detail::compute_t<FT> v = static_cast<detail::compute_t<FT>>(x.to_float());      \
            return F(static_cast<FT>(NATIVE));                                                     \
        }                                                                                          \
    }
    FPSAN_DEFINE_TAGGED_UNARY(log, Log, std::log(v))
    FPSAN_DEFINE_TAGGED_UNARY(log2, Log2, std::log2(v))
    FPSAN_DEFINE_TAGGED_UNARY(sqrt, Sqrt, std::sqrt(v))
    // precise_sqrt mirrors Triton's IEEE-correct sqrt: a distinct FPSan tag from
    // `sqrt`, but the same correctly-rounded std::sqrt in Float mode.
    FPSAN_DEFINE_TAGGED_UNARY(precise_sqrt, PreciseSqrt, std::sqrt(v))
    FPSAN_DEFINE_TAGGED_UNARY(rsqrt, Rsqrt, detail::compute_t<FT>(1) / std::sqrt(v))
    FPSAN_DEFINE_TAGGED_UNARY(erf, Erf, std::erf(v))
    FPSAN_DEFINE_TAGGED_UNARY(floor, Floor, std::floor(v))
    FPSAN_DEFINE_TAGGED_UNARY(ceil, Ceil, std::ceil(v))
    FPSAN_DEFINE_TAGGED_UNARY(rcp, Rcp, detail::compute_t<FT>(1) / v)
    FPSAN_DEFINE_TAGGED_UNARY(fract, Fract, v - std::floor(v))
    FPSAN_DEFINE_TAGGED_UNARY(tanh, Tanh, std::tanh(v))
#undef FPSAN_DEFINE_TAGGED_UNARY

    // ---- modular binary / ternary: fma, fmod, fmin/fmax, min/max ---------------
    template <class FT, Semantics S, Conversions C>
    FPSAN_HOST_DEVICE Value<FT, S, C> fma(Value<FT, S, C> a, Value<FT, S, C> b, Value<FT, S, C> c)
    {
        using F = Value<FT, S, C>;
        if constexpr(F::semantics == Semantics::Triton)
            return FPSAN_FROM_PAYLOAD(
                F,
                detail::payload_fma(
                    F::config, a.fpsan_payload(), b.fpsan_payload(), c.fpsan_payload()));
        else
            return F(static_cast<FT>(std::fma(static_cast<detail::compute_t<FT>>(a.to_float()),
                                              static_cast<detail::compute_t<FT>>(b.to_float()),
                                              static_cast<detail::compute_t<FT>>(c.to_float()))));
    }
    template <class FT, Semantics S, Conversions C>
    FPSAN_HOST_DEVICE Value<FT, S, C> fmod(Value<FT, S, C> a, Value<FT, S, C> b)
    {
        using F = Value<FT, S, C>;
        if constexpr(F::semantics == Semantics::Triton)
            return FPSAN_FROM_PAYLOAD(
                F, detail::payload_srem(F::config, a.fpsan_payload(), b.fpsan_payload()));
        else
            return F(static_cast<FT>(std::fmod(static_cast<detail::compute_t<FT>>(a.to_float()),
                                               static_cast<detail::compute_t<FT>>(b.to_float()))));
    }

#define FPSAN_DEFINE_MINMAX(NAME, PAYLOAD_FN, STD_FN)                                            \
    template <class FT, Semantics S, Conversions C>                                              \
    FPSAN_HOST_DEVICE Value<FT, S, C> NAME(Value<FT, S, C> a, Value<FT, S, C> b)                 \
    {                                                                                            \
        using F = Value<FT, S, C>;                                                               \
        if constexpr(F::semantics == Semantics::Triton)                                          \
            return FPSAN_FROM_PAYLOAD(                                                           \
                F, detail::PAYLOAD_FN(F::config, a.fpsan_payload(), b.fpsan_payload()));         \
        else                                                                                     \
            return F(static_cast<FT>(STD_FN(static_cast<detail::compute_t<FT>>(a.to_float()),    \
                                            static_cast<detail::compute_t<FT>>(b.to_float())))); \
    }
    FPSAN_DEFINE_MINMAX(fmin, payload_min, std::fmin)
    FPSAN_DEFINE_MINMAX(fmax, payload_max, std::fmax)
    FPSAN_DEFINE_MINMAX(min, payload_min, std::fmin)
    FPSAN_DEFINE_MINMAX(max, payload_max, std::fmax)
#undef FPSAN_DEFINE_MINMAX

    // ---- fmed3: median of 3. In FPSan mode this resolves to signed-int median
    // on the payload via three min/max calls (each already exact). In Float mode
    // it's the float median expressed the same way. ---------------------------
    template <class FT, Semantics S, Conversions C>
    FPSAN_HOST_DEVICE Value<FT, S, C> fmed3(Value<FT, S, C> a, Value<FT, S, C> b, Value<FT, S, C> c)
    {
        return fpsan::max(fpsan::min(fpsan::max(a, b), c), fpsan::min(a, b));
    }

#undef FPSAN_FROM_PAYLOAD

} // namespace fpsan

#endif // FPSAN_MATH_HPP
