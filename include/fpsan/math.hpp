// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan/math.hpp
// ----------------------------------------------------------------------------
// Standard-library-style math on Value, found by argument-dependent lookup
// (write `using std::exp; ... exp(x);` in generic code, or `fpsan::exp(x)`).
//
//   semantics == Semantics::Native : the real std:: function (drop-in).
//   semantics == Semantics::Triton : Triton FPSan handler (see detail/math).
//   algebraic semantics           : finite-ring handler or deterministic tag
//                                  (see detail/algebraic.hpp).
//
// A tagged operation is deterministic and operation-distinguishing, but it does
// not claim an identity such as log(x*y)==log(x)+log(y). The exact set of
// honored identities depends on the selected Semantics.
// ----------------------------------------------------------------------------
#ifndef FPSAN_MATH_HPP
#define FPSAN_MATH_HPP

#include "fpsan/detail/config.hpp"
#include "fpsan/detail/math.hpp"
#include "fpsan/value.hpp"

#include <cmath>
#include <type_traits>

namespace fpsan {
namespace detail {
// Precision used for the native (Semantics::Native) math path. _Float16 /
// __bf16 promote to float, matching how the standard library handles them.
template <class FT> using compute_t = std::conditional_t<std::is_same_v<FT, double>, double, float>;
} // namespace detail

// p is the payload of x; build the result wrapper from a computed payload.
#define FPSAN_FROM_PAYLOAD(F, expr) F::from_fpsan_payload(static_cast<typename F::bits_type>(expr))

template <class FT, Semantics S, Conversions C>
FPSAN_HOST_DEVICE Value<FT, S, C> abs(Value<FT, S, C> x) {
  using F = Value<FT, S, C>;
  if constexpr (F::is_vector) {
    F out{};
    for (unsigned i = 0; i < F::lanes; ++i)
      out.set(i, fpsan::abs(x.get(i)));
    return out;
  } else if constexpr (F::is_algebraic && detail::alg_has_qr_order(F::alg_cfg()))
    return FPSAN_FROM_PAYLOAD(F, detail::alg_qr_abs(F::alg_cfg(), x.fpsan_payload()));
  else if constexpr (F::is_fpsan)
    return x < F(FT(0)) ? -x : x;
  else
    return F(static_cast<FT>(std::fabs(static_cast<detail::compute_t<FT>>(x.to_float()))));
}

template <class FT, Semantics S, Conversions C>
FPSAN_HOST_DEVICE Value<FT, S, C> fabs(Value<FT, S, C> x) {
  return fpsan::abs(x);
}

// ---- algebraic unary: exp, exp2, sin, cos ----------------------------------
#define FPSAN_DEFINE_ALGEBRAIC_UNARY(NAME, PAYLOAD_FN, ALG_EXPR, STD_FN)                           \
  template <class FT, Semantics S, Conversions C>                                                  \
  FPSAN_HOST_DEVICE Value<FT, S, C> NAME(Value<FT, S, C> x) {                                      \
    using F = Value<FT, S, C>;                                                                     \
    if constexpr (F::semantics == Semantics::Triton)                                               \
      return FPSAN_FROM_PAYLOAD(F, detail::PAYLOAD_FN(F::config, x.fpsan_payload()));              \
    else if constexpr (F::is_algebraic)                                                            \
      return FPSAN_FROM_PAYLOAD(F, ALG_EXPR);                                                      \
    else                                                                                           \
      return F(static_cast<FT>(STD_FN(static_cast<detail::compute_t<FT>>(x.to_float()))));         \
  }
// exp and exp2 both get genuine homomorphisms g^(v mod d) on the order-d
// channel (SophieGermainRing and PythagoreanRing families):
// exp(a+b)==exp(a)*exp(b) and likewise for exp2, which uses a fixed base change
// (see alg_exp2). Field-family semantics fall back to tags.
FPSAN_DEFINE_ALGEBRAIC_UNARY(exp, payload_exp, detail::alg_exp(F::alg_cfg(), x.fpsan_payload()),
                             std::exp)
FPSAN_DEFINE_ALGEBRAIC_UNARY(exp2, payload_exp2, detail::alg_exp2(F::alg_cfg(), x.fpsan_payload()),
                             std::exp2)
#undef FPSAN_DEFINE_ALGEBRAIC_UNARY

// cos / sin share one payload computation.
template <class FT, Semantics S, Conversions C>
FPSAN_HOST_DEVICE Value<FT, S, C> cos(Value<FT, S, C> x) {
  using F = Value<FT, S, C>;
  if constexpr (F::semantics == Semantics::Triton)
    return FPSAN_FROM_PAYLOAD(F, detail::payload_cos_sin(F::config, x.fpsan_payload()).cos);
  else if constexpr (F::is_algebraic)
    return FPSAN_FROM_PAYLOAD(F, detail::alg_cos(F::alg_cfg(), x.fpsan_payload()));
  else
    return F(static_cast<FT>(std::cos(static_cast<detail::compute_t<FT>>(x.to_float()))));
}
template <class FT, Semantics S, Conversions C>
FPSAN_HOST_DEVICE Value<FT, S, C> sin(Value<FT, S, C> x) {
  using F = Value<FT, S, C>;
  if constexpr (F::semantics == Semantics::Triton)
    return FPSAN_FROM_PAYLOAD(F, detail::payload_cos_sin(F::config, x.fpsan_payload()).sin);
  else if constexpr (F::is_algebraic)
    return FPSAN_FROM_PAYLOAD(F, detail::alg_sin(F::alg_cfg(), x.fpsan_payload()));
  else
    return F(static_cast<FT>(std::sin(static_cast<detail::compute_t<FT>>(x.to_float()))));
}

// ---- tagged unary: log, log2, sqrt, rsqrt, erf, floor, ceil ----------------
#define FPSAN_DEFINE_TAGGED_UNARY(NAME, OPID, NATIVE)                                              \
  template <class FT, Semantics S, Conversions C>                                                  \
  FPSAN_HOST_DEVICE Value<FT, S, C> NAME(Value<FT, S, C> x) {                                      \
    using F = Value<FT, S, C>;                                                                     \
    if constexpr (F::semantics == Semantics::Triton)                                               \
      return FPSAN_FROM_PAYLOAD(                                                                   \
          F, detail::payload_tagged_unary(F::config, x.fpsan_payload(), detail::UnaryOpId::OPID)); \
    else if constexpr (F::is_algebraic)                                                            \
      return FPSAN_FROM_PAYLOAD(                                                                   \
          F, detail::alg_tagged(F::alg_cfg(), x.fpsan_payload(),                                   \
                                static_cast<detail::u64>(detail::UnaryOpId::OPID) + 0x100u));      \
    else {                                                                                         \
      const detail::compute_t<FT> v = static_cast<detail::compute_t<FT>>(x.to_float());            \
      return F(static_cast<FT>(NATIVE));                                                           \
    }                                                                                              \
  }
FPSAN_DEFINE_TAGGED_UNARY(erf, Erf, std::erf(v))
FPSAN_DEFINE_TAGGED_UNARY(floor, Floor, std::floor(v))
FPSAN_DEFINE_TAGGED_UNARY(ceil, Ceil, std::ceil(v))
FPSAN_DEFINE_TAGGED_UNARY(rcp, Rcp, detail::compute_t<FT>(1) / v)
FPSAN_DEFINE_TAGGED_UNARY(fract, Fract, v - std::floor(v))
FPSAN_DEFINE_TAGGED_UNARY(tanh, Tanh, std::tanh(v))
#undef FPSAN_DEFINE_TAGGED_UNARY

// sqrt / precise_sqrt / rsqrt / cbrt are ALGEBRAIC roots in the faithful
// algebraic variants: multiplicative power maps, so sqrt(x*y)==sqrt(x)*sqrt(y),
// cbrt(x*y)==cbrt(x)*cbrt(y), and rsqrt==1/sqrt hold exactly. Triton
// keeps them as tags, and FieldFast deliberately follows that cheaper
// tag path. cbrt is a perfect cube root when 3 is coprime to the relevant
// group exponent, and a tag otherwise.
#define FPSAN_DEFINE_ALGEBRAIC_ROOT(NAME, OPID, ALG_FN, NATIVE)                                    \
  template <class FT, Semantics S, Conversions C>                                                  \
  FPSAN_HOST_DEVICE Value<FT, S, C> NAME(Value<FT, S, C> x) {                                      \
    using F = Value<FT, S, C>;                                                                     \
    if constexpr (F::semantics == Semantics::Triton)                                               \
      return FPSAN_FROM_PAYLOAD(                                                                   \
          F, detail::payload_tagged_unary(F::config, x.fpsan_payload(), detail::UnaryOpId::OPID)); \
    else if constexpr (detail::has_fast_field_ops(F::semantics))                                   \
      return FPSAN_FROM_PAYLOAD(                                                                   \
          F, detail::alg_tagged(F::alg_cfg(), x.fpsan_payload(),                                   \
                                static_cast<detail::u64>(detail::UnaryOpId::OPID) + 0x100u));      \
    else if constexpr (F::is_algebraic)                                                            \
      return FPSAN_FROM_PAYLOAD(F, detail::ALG_FN(F::alg_cfg(), x.fpsan_payload()));               \
    else {                                                                                         \
      const detail::compute_t<FT> v = static_cast<detail::compute_t<FT>>(x.to_float());            \
      return F(static_cast<FT>(NATIVE));                                                           \
    }                                                                                              \
  }
FPSAN_DEFINE_ALGEBRAIC_ROOT(sqrt, Sqrt, alg_sqrt, std::sqrt(v))
// precise_sqrt: same algebraic value as sqrt, but a distinct FPSan tag (mirrors
// Triton's correctly-rounded sqrt) and the same std::sqrt in Native mode.
FPSAN_DEFINE_ALGEBRAIC_ROOT(precise_sqrt, PreciseSqrt, alg_sqrt, std::sqrt(v))
FPSAN_DEFINE_ALGEBRAIC_ROOT(rsqrt, Rsqrt, alg_rsqrt, detail::compute_t<FT>(1) / std::sqrt(v))
#undef FPSAN_DEFINE_ALGEBRAIC_ROOT

// cbrt: Triton lowers it to a libdevice extern, so the Triton path tags it by
// symbol name (the generic extern fallback). The faithful algebraic variants
// realize it as a power map (perfect cube root where has_cbrt), while FieldFast
// uses a cheap deterministic tag.
template <class FT, Semantics S, Conversions C>
FPSAN_HOST_DEVICE Value<FT, S, C> cbrt(Value<FT, S, C> x) {
  using F = Value<FT, S, C>;
  if constexpr (F::semantics == Semantics::Triton)
    return FPSAN_FROM_PAYLOAD(F, detail::payload_extern_tagged(F::config,
                                                               detail::stable_string_hash("cbrt"),
                                                               x.fpsan_payload()));
  else if constexpr (detail::has_fast_field_ops(F::semantics))
    return FPSAN_FROM_PAYLOAD(
        F, detail::alg_tagged(F::alg_cfg(), x.fpsan_payload(), 0x63627274ull /*"cbrt"*/));
  else if constexpr (F::is_algebraic)
    return FPSAN_FROM_PAYLOAD(F, detail::alg_cbrt(F::alg_cfg(), x.fpsan_payload()));
  else
    return F(static_cast<FT>(std::cbrt(static_cast<detail::compute_t<FT>>(x.to_float()))));
}

// log is special: the composite algebraic families honor
// log(x*y)=log(x)+log(y) via the discrete log on the order-d channel (the dual
// of exp's g^(v mod d)); Triton and the Field-family semantics keep it as a tag.
template <class FT, Semantics S, Conversions C>
FPSAN_HOST_DEVICE Value<FT, S, C> log(Value<FT, S, C> x) {
  using F = Value<FT, S, C>;
  if constexpr (F::semantics == Semantics::Triton)
    return FPSAN_FROM_PAYLOAD(
        F, detail::payload_tagged_unary(F::config, x.fpsan_payload(), detail::UnaryOpId::Log));
  else if constexpr (F::is_algebraic)
    return FPSAN_FROM_PAYLOAD(F, detail::alg_log(F::alg_cfg(), x.fpsan_payload()));
  else {
    const detail::compute_t<FT> v = static_cast<detail::compute_t<FT>>(x.to_float());
    return F(static_cast<FT>(std::log(v)));
  }
}

// log2 mirrors log: the composite algebraic families honor
// log2(x*y)=log2(x)+log2(y) as the exact inverse of exp2 on the order-d channel;
// Triton and the Field-family semantics keep it as a tag.
template <class FT, Semantics S, Conversions C>
FPSAN_HOST_DEVICE Value<FT, S, C> log2(Value<FT, S, C> x) {
  using F = Value<FT, S, C>;
  if constexpr (F::semantics == Semantics::Triton)
    return FPSAN_FROM_PAYLOAD(
        F, detail::payload_tagged_unary(F::config, x.fpsan_payload(), detail::UnaryOpId::Log2));
  else if constexpr (F::is_algebraic)
    return FPSAN_FROM_PAYLOAD(F, detail::alg_log2(F::alg_cfg(), x.fpsan_payload()));
  else {
    const detail::compute_t<FT> v = static_cast<detail::compute_t<FT>>(x.to_float());
    return F(static_cast<FT>(std::log2(v)));
  }
}

// exp10 / log10: base-10 members of the exp_b/log_b family. The composite
// algebraic families honor them as genuine homomorphisms on the order-d channel
// (a fixed base change, like exp2/log2). Triton has no exp10/log10 op (it
// lowers them to libdevice externs), so the Triton path tags them by symbol
// name. Note: std::exp10 is non-standard, so the Native path uses pow(10, v);
// std::log10 is standard.
template <class FT, Semantics S, Conversions C>
FPSAN_HOST_DEVICE Value<FT, S, C> exp10(Value<FT, S, C> x) {
  using F = Value<FT, S, C>;
  if constexpr (F::semantics == Semantics::Triton)
    return FPSAN_FROM_PAYLOAD(F, detail::payload_extern_tagged(F::config,
                                                               detail::stable_string_hash("exp10"),
                                                               x.fpsan_payload()));
  else if constexpr (F::is_algebraic)
    return FPSAN_FROM_PAYLOAD(F, detail::alg_exp10(F::alg_cfg(), x.fpsan_payload()));
  else {
    const detail::compute_t<FT> v = static_cast<detail::compute_t<FT>>(x.to_float());
    return F(static_cast<FT>(std::pow(detail::compute_t<FT>(10), v)));
  }
}
template <class FT, Semantics S, Conversions C>
FPSAN_HOST_DEVICE Value<FT, S, C> log10(Value<FT, S, C> x) {
  using F = Value<FT, S, C>;
  if constexpr (F::semantics == Semantics::Triton)
    return FPSAN_FROM_PAYLOAD(F, detail::payload_extern_tagged(F::config,
                                                               detail::stable_string_hash("log10"),
                                                               x.fpsan_payload()));
  else if constexpr (F::is_algebraic)
    return FPSAN_FROM_PAYLOAD(F, detail::alg_log10(F::alg_cfg(), x.fpsan_payload()));
  else {
    const detail::compute_t<FT> v = static_cast<detail::compute_t<FT>>(x.to_float());
    return F(static_cast<FT>(std::log10(v)));
  }
}

// ---- modular binary / ternary: fma, fmod, fmin/fmax, min/max ---------------
template <class FT, Semantics S, Conversions C>
FPSAN_HOST_DEVICE Value<FT, S, C> fma(Value<FT, S, C> a, Value<FT, S, C> b, Value<FT, S, C> c) {
  using F = Value<FT, S, C>;
  if constexpr (F::semantics == Semantics::Triton)
    return FPSAN_FROM_PAYLOAD(
        F, detail::payload_fma(F::config, a.fpsan_payload(), b.fpsan_payload(), c.fpsan_payload()));
  else if constexpr (F::is_algebraic)
    return a * b + c; // exact and value-faithful: the algebra's own + and *
  else
    return F(static_cast<FT>(std::fma(static_cast<detail::compute_t<FT>>(a.to_float()),
                                      static_cast<detail::compute_t<FT>>(b.to_float()),
                                      static_cast<detail::compute_t<FT>>(c.to_float()))));
}
template <class FT, Semantics S, Conversions C>
FPSAN_HOST_DEVICE Value<FT, S, C> fmod(Value<FT, S, C> a, Value<FT, S, C> b) {
  using F = Value<FT, S, C>;
  if constexpr (F::semantics == Semantics::Triton)
    return FPSAN_FROM_PAYLOAD(
        F, detail::payload_srem(F::config, a.fpsan_payload(), b.fpsan_payload()));
  else if constexpr (F::is_algebraic)
    return FPSAN_FROM_PAYLOAD(F, detail::alg_tagged2(F::alg_cfg(), a.fpsan_payload(),
                                                     b.fpsan_payload(), 0x666D6F64ull /*"fmod"*/));
  else
    return F(static_cast<FT>(std::fmod(static_cast<detail::compute_t<FT>>(a.to_float()),
                                       static_cast<detail::compute_t<FT>>(b.to_float()))));
}

#define FPSAN_DEFINE_MINMAX(NAME, PAYLOAD_FN, ALGEBRAIC_FIELD_FN, STD_FN)                          \
  template <class FT, Semantics S, Conversions C>                                                  \
  FPSAN_HOST_DEVICE Value<FT, S, C> NAME(Value<FT, S, C> a, Value<FT, S, C> b) {                   \
    using F = Value<FT, S, C>;                                                                     \
    /* Algebraic values use the qr-positive prototype order. Other payload */                      \
    /* modes keep Triton's signed-payload min/max contract. */                                     \
    if constexpr (F::is_vector) {                                                                  \
      F out{};                                                                                     \
      for (unsigned i = 0; i < F::lanes; ++i)                                                      \
        out.set(i, fpsan::NAME(a.get(i), b.get(i)));                                               \
      return out;                                                                                  \
    } else if constexpr (F::is_algebraic && detail::alg_has_qr_order(F::alg_cfg())) {              \
      return FPSAN_FROM_PAYLOAD(                                                                   \
          F, detail::ALGEBRAIC_FIELD_FN(F::alg_cfg(), a.fpsan_payload(), b.fpsan_payload()));      \
    } else if constexpr (F::is_fpsan)                                                              \
      return FPSAN_FROM_PAYLOAD(                                                                   \
          F, detail::PAYLOAD_FN(F::config, a.fpsan_payload(), b.fpsan_payload()));                 \
    else                                                                                           \
      return F(static_cast<FT>(STD_FN(static_cast<detail::compute_t<FT>>(a.to_float()),            \
                                      static_cast<detail::compute_t<FT>>(b.to_float()))));         \
  }
FPSAN_DEFINE_MINMAX(fmin, payload_min, alg_qr_min, std::fmin)
FPSAN_DEFINE_MINMAX(fmax, payload_max, alg_qr_max, std::fmax)
FPSAN_DEFINE_MINMAX(min, payload_min, alg_qr_min, std::fmin)
FPSAN_DEFINE_MINMAX(max, payload_max, alg_qr_max, std::fmax)
#undef FPSAN_DEFINE_MINMAX

// ---- fmed3: median of 3 via three min/max calls. In Native mode it's the
// float median expressed the same way. ------------------------------------
template <class FT, Semantics S, Conversions C>
FPSAN_HOST_DEVICE Value<FT, S, C> fmed3(Value<FT, S, C> a, Value<FT, S, C> b, Value<FT, S, C> c) {
  return fpsan::max(fpsan::min(fpsan::max(a, b), c), fpsan::min(a, b));
}

// ---- generic extern fallback: tag any unmodeled call by a symbol name ------
// For an op with no honored identity (a new intrinsic, an arbitrary libdevice
// symbol) this gives a deterministic, symbol-distinct, argument-order-sensitive
// fingerprint -- the only structure an opaque function can carry. The scheme is
// bit-for-bit Triton's extern tagging (see detail::payload_extern_tagged), so a
// symbol tagged here matches what Triton's sanitizer emits for it. There is no
// Native-mode behavior (an opaque symbol has no native implementation), so the
// native wrapper must call the real function -- e.g. the FPSAN_DEFINE_AMDGCN_*_EXTERN
// macros do exactly that. Pass detail::stable_string_hash("symbol") as the key.
template <class FT, Semantics S, Conversions C, class... Rest>
FPSAN_HOST_DEVICE Value<FT, S, C> extern_tagged(detail::u64 name_hash, Value<FT, S, C> first,
                                                Rest... rest) {
  using F = Value<FT, S, C>;
  static_assert(F::is_fpsan, "extern_tagged is defined only in a payload mode (an opaque "
                             "symbol has no native implementation); call the real function "
                             "in Native mode");
  static_assert((std::is_same_v<Rest, Value<FT, S, C>> && ...),
                "all operands of extern_tagged must be the same Value type");
  if constexpr (F::is_vector) {
    F out{};
    for (unsigned l = 0; l < F::lanes; ++l)
      out.set(l, extern_tagged(name_hash, first.get(l), rest.get(l)...));
    return out;
  } else if constexpr (F::is_algebraic)
    return FPSAN_FROM_PAYLOAD(F, detail::alg_extern_tagged1(F::alg_cfg(), name_hash,
                                                            first.fpsan_payload(),
                                                            rest.fpsan_payload()...));
  else // Triton
    return FPSAN_FROM_PAYLOAD(F, detail::payload_extern_tagged(F::config, name_hash,
                                                               first.fpsan_payload(),
                                                               rest.fpsan_payload()...));
}

#undef FPSAN_FROM_PAYLOAD

} // namespace fpsan

#endif // FPSAN_MATH_HPP
