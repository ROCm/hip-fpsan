// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan/detail/subbyte_widen.hpp
// ----------------------------------------------------------------------------
// Deterministic conversion of sub-byte FPSan payload fields.
//
// FP4/FP6 are not scalar Value element types, so the packed gfx intrinsics decode
// raw Width-bit fields and then enter the ordinary FPSan payload model. Triton
// keeps its signed-resize payload convention. Algebraic semantics first treat the
// field as a finite sub-byte residue, then use the same cast policy as ordinary
// Value casts, except that fp6/bf6 is not part of the FieldWithMulCasts tower.
// Do not use these helpers for public FP8/BF8 Value types; those go through
// fpsan::cast.
// ----------------------------------------------------------------------------
#ifndef FPSAN_DETAIL_SUBBYTE_WIDEN_HPP
#define FPSAN_DETAIL_SUBBYTE_WIDEN_HPP

#include "fpsan/value.hpp"

#include <cstdint>
#include <type_traits>

namespace fpsan {
namespace detail {
template <int Width>
FPSAN_HOST_DEVICE constexpr std::int32_t sign_extend_subbyte(std::uint32_t field) {
  static_assert(Width > 0 && Width < 32, "sub-byte width must fit in uint32_t");
  const std::uint32_t raw = field & ((std::uint32_t{1} << Width) - 1u);
  return static_cast<std::int32_t>(raw << (32 - Width)) >> (32 - Width);
}

template <int Width, Semantics S> FPSAN_HOST_DEVICE constexpr AlgConfig finite_subbyte_alg_cfg() {
  static_assert(Width <= 8, "only finite payload fields up to byte width are supported here");
  const auto m = alg_modulus(alg_variant_of(S), Width);
  AlgConfig c{};
  c.n = m.n;
  c.inv2 = (m.n + 1) / 2;
  c.g = m.g;
  c.cast_g = m.g;
  c.d = m.d;
  c.inf_code = m.n;
  c.nan_code = m.n + 1;
  c.two_moduli = m.two_moduli;
  c.has_inf_nan = true;
  c.bit_width = Width;
  return c;
}

template <int Width, Semantics S>
FPSAN_HOST_DEVICE constexpr std::uint64_t finite_subbyte_source_payload(std::uint32_t field) {
  static_assert(is_algebraic_semantics(S), "algebraic semantics required");
  const auto c = finite_subbyte_alg_cfg<Width, S>();
  const std::int32_t e = sign_extend_subbyte<Width>(field);
  if (e >= 0)
    return static_cast<std::uint64_t>(e) % c.n;
  const auto mag = static_cast<std::uint64_t>(-static_cast<std::int64_t>(e)) % c.n;
  return mag == 0 ? 0 : c.n - mag;
}

template <int Width> FPSAN_HOST_DEVICE constexpr std::uint32_t subbyte_positive_max_code() {
  return (std::uint32_t{1} << (Width - 1)) - 1u;
}

template <int Width, Semantics S>
FPSAN_HOST_DEVICE constexpr std::uint32_t canonical_subbyte_code_from_residue(std::uint64_t p) {
  static_assert(is_algebraic_semantics(S), "algebraic semantics required");
  const auto c = finite_subbyte_alg_cfg<Width, S>();
  const auto positive = static_cast<std::uint64_t>(subbyte_positive_max_code<Width>());
  const auto mask = (std::uint32_t{1} << Width) - 1u;
  if (p <= positive)
    return static_cast<std::uint32_t>(p);
  const auto signed_code = static_cast<std::int64_t>(p) - static_cast<std::int64_t>(c.n);
  return static_cast<std::uint32_t>(signed_code) & mask;
}

// Widen a packed Width-bit payload field into f32 FPSan payload space. Triton
// uses signed resize. Algebraic modes embed the signed field as a finite
// residue and then follow the same sub-byte cast policy used by
// subbyte_widen_to.
// Public FP8/BF8 Value formats must use fpsan::cast so algebraic same-width
// format identity is preserved.
template <int Width, Semantics S, Conversions C>
FPSAN_HOST_DEVICE Value<float, S, C> packed_field_widen(std::uint32_t field) {
  if constexpr (S == Semantics::Triton) {
    const std::int32_t e = sign_extend_subbyte<Width>(field);
    return Value<float, S, C>::from_fpsan_payload(static_cast<std::uint32_t>(e));
  } else if constexpr (is_algebraic_semantics(S)) {
    using Out = Value<float, S, C>;
    using ToBits = typename Out::bits_type;
    const auto from = finite_subbyte_alg_cfg<Width, S>();
    const auto to = Out::alg_cfg();
    const auto p = finite_subbyte_source_payload<Width, S>(field);
    if constexpr (Width == 4 && has_multiplicative_field_casts(S))
      return Out::from_fpsan_payload(static_cast<ToBits>(alg_cast1(from, to, p)));
    else
      return Out::from_fpsan_payload(static_cast<ToBits>(p % to.n));
  } else {
    static_assert(always_false<std::integral_constant<int, Width>>, "FPSan semantics required");
  }
}

template <class DstFT, int Width, Semantics S, Conversions C>
FPSAN_HOST_DEVICE Value<DstFT, S, C> subbyte_widen_to(std::uint32_t field) {
  static_assert(Width < 8, "public FP8/BF8 payloads must be widened through fpsan::cast");
  using Out = Value<DstFT, S, C>;
  if constexpr (S == Semantics::Triton) {
    const std::int32_t e = sign_extend_subbyte<Width>(field);
    return Out::from_fpsan_payload(static_cast<typename Out::bits_type>(e));
  } else if constexpr (is_algebraic_semantics(S)) {
    using ToBits = typename Out::bits_type;
    const auto from = finite_subbyte_alg_cfg<Width, S>();
    const auto to = Out::alg_cfg();
    const auto p = finite_subbyte_source_payload<Width, S>(field);
    if constexpr (Width == 4 && has_multiplicative_field_casts(S))
      return Out::from_fpsan_payload(static_cast<ToBits>(alg_cast1(from, to, p)));
    else
      return Out::from_fpsan_payload(static_cast<ToBits>(p % to.n));
  } else {
    static_assert(always_false<DstFT>, "FPSan semantics required");
  }
}

template <int Width, class SrcFT, Semantics S, Conversions C>
FPSAN_HOST_DEVICE std::uint32_t subbyte_narrow_code(Value<SrcFT, S, C> v) {
  static_assert(Width < 8, "public FP8/BF8 payloads must be narrowed through fpsan::cast");
  const auto mask = (std::uint32_t{1} << Width) - 1u;
  if constexpr (S == Semantics::Triton) {
    return static_cast<std::uint32_t>(v.fpsan_payload()) & mask;
  } else if constexpr (is_algebraic_semantics(S)) {
    using In = Value<SrcFT, S, C>;
    const auto from = In::alg_cfg();
    const auto to = finite_subbyte_alg_cfg<Width, S>();
    const auto p = v.fpsan_payload();
    if (p == from.nan_code)
      return 0u;
    if (p == from.inf_code)
      return subbyte_positive_max_code<Width>();
    const auto q = [&]() {
      if constexpr (Width == 4 && has_multiplicative_field_casts(S))
        return alg_cast1(from, to, p);
      else
        return static_cast<std::uint64_t>(p) % to.n;
    }();
    return canonical_subbyte_code_from_residue<Width, S>(q);
  } else {
    static_assert(always_false<SrcFT>, "FPSan semantics required");
  }
}

// Element bit width of an f8f6f4 format immediate: fp8/bf8 (0,1) -> 8,
// fp6/bf6 (2,3) -> 6, fp4 (4) -> 4. Shared by the gfx950 MFMA and gfx1250
// WMMA f8f6f4 paths (both decode the same packed sub-byte fragments).
FPSAN_HOST_DEVICE constexpr int f8f6f4_width(int code) {
  return (code <= 1) ? 8 : (code <= 3) ? 6 : 4;
}

// OCP-MX E8M0 block-scale -> float. E8M0 stores only an exponent (bias
// 127): byte b -> 2^(b-127); 0xFF is NaN. Built by bit construction so it
// is exact (no libm) and host/device safe. Shared by the gfx950 MFMA and
// gfx1250 WMMA block-scaled paths.
FPSAN_HOST_DEVICE inline float e8m0_to_float(unsigned byte) {
  if (byte == 0xFFu)
    return __builtin_nanf("");
  if (byte == 0u)
    return __builtin_bit_cast(float, std::uint32_t(0x00400000u)); // 2^-127
  return __builtin_bit_cast(float, static_cast<std::uint32_t>(byte) << 23);
}
} // namespace detail
} // namespace fpsan

#endif // FPSAN_DETAIL_SUBBYTE_WIDEN_HPP
