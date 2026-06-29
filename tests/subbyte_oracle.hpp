// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/subbyte_oracle.hpp
//
// Shared test oracle for fp4/fp6/bf6 FPSan conversions. These packed hardware
// formats have no IEEE Inf/NaN encodings, so every raw Width-bit widen source is
// treated as a finite value. Algebraic FPSan then canonicalizes finite values
// through the same cast policy used by ordinary Value casts.
#ifndef FPSAN_TESTS_SUBBYTE_ORACLE_HPP
#define FPSAN_TESTS_SUBBYTE_ORACLE_HPP

#include "fpsan/fpsan.hpp"

#include <cstdint>
#include <type_traits>

namespace fpsan_test {
template <int Width> std::int32_t sign_extend_subbyte(std::uint32_t field) {
  const std::uint32_t raw = field & ((std::uint32_t{1} << Width) - 1u);
  return static_cast<std::int32_t>(raw << (32 - Width)) >> (32 - Width);
}

template <int Width, fpsan::Semantics S> fpsan::detail::AlgConfig finite_subbyte_alg_cfg() {
  const auto m = fpsan::detail::alg_modulus(fpsan::detail::alg_variant_of(S), Width);
  fpsan::detail::AlgConfig c{};
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

template <int Width, fpsan::Semantics S>
std::uint64_t finite_subbyte_source_payload(std::uint32_t field) {
  const auto c = finite_subbyte_alg_cfg<Width, S>();
  const std::int32_t e = sign_extend_subbyte<Width>(field);
  if (e >= 0)
    return static_cast<std::uint64_t>(e) % c.n;
  const auto mag = static_cast<std::uint64_t>(-static_cast<std::int64_t>(e)) % c.n;
  return mag == 0 ? 0 : c.n - mag;
}

template <int Width, fpsan::Semantics S>
std::uint32_t canonical_subbyte_code_from_residue(std::uint64_t p) {
  const auto c = finite_subbyte_alg_cfg<Width, S>();
  const auto residue = p;
  const auto positive = (std::uint64_t{1} << (Width - 1)) - 1u;
  const auto mask = (std::uint32_t{1} << Width) - 1u;
  if (residue <= positive)
    return static_cast<std::uint32_t>(residue);
  const auto signed_code = static_cast<std::int64_t>(residue) - static_cast<std::int64_t>(c.n);
  return static_cast<std::uint32_t>(signed_code) & mask;
}

template <int Width> constexpr std::uint32_t subbyte_positive_max_code() {
  return (std::uint32_t{1} << (Width - 1)) - 1u;
}

template <class DstFT, int Width, fpsan::Semantics S, fpsan::Conversions C>
typename fpsan::Value<DstFT, S, C>::bits_type canonical_subbyte_widen_payload(std::uint32_t field) {
  using Out = fpsan::Value<DstFT, S, C>;
  using ToBits = typename Out::bits_type;
  if constexpr (S == fpsan::Semantics::Triton) {
    const auto e = sign_extend_subbyte<Width>(field);
    const auto r = static_cast<std::make_signed_t<ToBits>>(e);
    return static_cast<ToBits>(r);
  } else if constexpr (fpsan::detail::is_algebraic_semantics(S)) {
    const auto from = finite_subbyte_alg_cfg<Width, S>();
    const auto to = Out::alg_cfg();
    const auto p = finite_subbyte_source_payload<Width, S>(field);
    if constexpr (Width == 4 && fpsan::detail::has_multiplicative_field_casts(S))
      return static_cast<ToBits>(fpsan::detail::alg_cast1(from, to, p));
    else
      return static_cast<ToBits>(p % to.n);
  } else {
    static_assert(fpsan::detail::always_false<DstFT>, "FPSan semantics required");
  }
}

template <class DstFT, int Width, fpsan::Semantics S, fpsan::Conversions C>
fpsan::Value<DstFT, S, C> canonical_subbyte_widen_to(std::uint32_t field) {
  using Out = fpsan::Value<DstFT, S, C>;
  return Out::from_fpsan_payload(canonical_subbyte_widen_payload<DstFT, Width, S, C>(field));
}

template <int Width, fpsan::Semantics S, fpsan::Conversions C>
fpsan::Value<float, S, C> canonical_subbyte_widen(std::uint32_t field) {
  return canonical_subbyte_widen_to<float, Width, S, C>(field);
}

template <int Width, class SrcFT, fpsan::Semantics S, fpsan::Conversions C>
std::uint32_t canonical_subbyte_narrow_code(fpsan::Value<SrcFT, S, C> v) {
  const auto mask = (std::uint32_t{1} << Width) - 1u;
  if constexpr (S == fpsan::Semantics::Triton) {
    return static_cast<std::uint32_t>(v.fpsan_payload()) & mask;
  } else if constexpr (fpsan::detail::is_algebraic_semantics(S)) {
    using In = fpsan::Value<SrcFT, S, C>;
    const auto from = In::alg_cfg();
    const auto to = finite_subbyte_alg_cfg<Width, S>();
    const auto p = v.fpsan_payload();
    // Native finite-only subbyte narrowing maps source NaN to +0 and
    // the unsigned algebraic pole to max-finite. Keep non-finite rules
    // separate from finite casts.
    if (p == from.nan_code)
      return 0u;
    if (p == from.inf_code)
      return subbyte_positive_max_code<Width>();
    const auto q = [&]() {
      if constexpr (Width == 4 && fpsan::detail::has_multiplicative_field_casts(S))
        return fpsan::detail::alg_cast1(from, to, p);
      else
        return static_cast<std::uint64_t>(p) % to.n;
    }();
    return canonical_subbyte_code_from_residue<Width, S>(q);
  } else {
    static_assert(fpsan::detail::always_false<SrcFT>, "FPSan semantics required");
  }
}
} // namespace fpsan_test

#endif // FPSAN_TESTS_SUBBYTE_ORACLE_HPP
