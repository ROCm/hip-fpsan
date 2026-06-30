// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/fpsan_semantics.hpp
// ----------------------------------------------------------------------------
// The single source of truth for "which FPSan-family semantics every device
// self-consistency test exercises". A self-consistency test asserts that the
// device payload equals the host recomputation in the SAME semantics, so it
// generalizes to every algebraic model; the test bodies are written generic over
// Semantics and driven by the loops below.
//
// Most tests use the default loop, which skips the "2" reroll variants because
// they share the same implementation paths and differ only in collision sets.
// Core algebraic tests use the all-variants loop when the reroll distinction is
// itself under test. To add, remove, or rename a Semantics value, edit these
// lists in one place.
//
// Native is excluded on purpose: it is the bit-exact-vs-hardware oracle (the
// reference the fpsan payloads are checked against), driven separately by each
// test, not a self-consistency variant.
// ----------------------------------------------------------------------------
#ifndef FPSAN_TESTS_FPSAN_SEMANTICS_HPP
#define FPSAN_TESTS_FPSAN_SEMANTICS_HPP

#include "fpsan/value.hpp"

#include <type_traits>

namespace fpsan_test {
// Invoke f(std::integral_constant<Semantics, S>{}) for each default
// FPSan-family semantics. Use a generic lambda and read the value as
// decltype(sem)::value:
//
//   for_each_fpsan_semantics([](auto sem) {
//       run_my_self_consistency_test<Traits, decltype(sem)::value>();
//   });
template <class F> void for_each_fpsan_semantics(F &&f) {
  using S = fpsan::Semantics;
  f(std::integral_constant<S, S::Triton>{});
  f(std::integral_constant<S, S::Field>{});
  f(std::integral_constant<S, S::FieldFast>{});
  f(std::integral_constant<S, S::FieldWithMulCasts>{});
  f(std::integral_constant<S, S::SophieGermainRing>{});
  f(std::integral_constant<S, S::PythagoreanRing>{});
}

// The exhaustive variant loop keeps the 2-suffixed reroll variants. Use it
// for tests whose purpose is the algebraic semantics themselves, not merely
// verifying that a generic wrapper is semantics-agnostic.
template <class F> void for_each_fpsan_semantics_all_variants(F &&f) {
  using S = fpsan::Semantics;
  f(std::integral_constant<S, S::Triton>{});
  f(std::integral_constant<S, S::Field>{});
  f(std::integral_constant<S, S::Field2>{});
  f(std::integral_constant<S, S::FieldFast>{});
  f(std::integral_constant<S, S::FieldFast2>{});
  f(std::integral_constant<S, S::FieldWithMulCasts>{});
  f(std::integral_constant<S, S::FieldWithMulCasts2>{});
  f(std::integral_constant<S, S::SophieGermainRing>{});
  f(std::integral_constant<S, S::SophieGermainRing2>{});
  f(std::integral_constant<S, S::PythagoreanRing>{});
  f(std::integral_constant<S, S::PythagoreanRing2>{});
}

// ---- per-(element type, semantics) capability predicates -------------------
// Which exact mathematical contract a flavor honors, so a testcase can loop
// over all flavors and gate its capability-specific assertions on these rather
// than being restricted to a hand-picked subset. The contracts are
// width-dependent (the algebraic exp/trig channels fall back to a plain field
// below 8 bits), so each predicate is keyed on the element type FT too.

// Value-faithful: rational-function identities (2+2==4, x/x==1) hold because
// the embedding is a ring homomorphism and division uses the field/ring inverse.
// FieldFast keeps the same embedding and +/* laws, but deliberately tags
// expensive division/roots.
template <class FT, fpsan::Semantics S> constexpr bool flavor_is_value_faithful() {
  static_cast<void>(sizeof(FT));
  return fpsan::detail::is_algebraic_semantics(S) && !fpsan::detail::has_fast_field_ops(S);
}

// exp/exp2/log/log2 are genuine homomorphisms (exp(a+b)==exp(a)*exp(b)). Triton
// constructs them as such in the free ring; the two-moduli algebraic variants
// (Sophie Germain / Pythagorean) carry a real exp channel; the Field variants
// reduce exp to a tag.
template <class FT, fpsan::Semantics S> constexpr bool flavor_has_exp_homomorphism() {
  if constexpr (S == fpsan::Semantics::Triton)
    return true;
  else if constexpr (fpsan::detail::is_algebraic_semantics(S))
    return fpsan::Value<FT, S, fpsan::Conversions::Explicit>::alg_cfg().two_moduli;
  else
    return false; // Native
}

// sin/cos satisfy the angle-addition laws. Triton constructs them; the
// Pythagorean variants carry a real trig channel; others reduce them to tags.
template <class FT, fpsan::Semantics S> constexpr bool flavor_has_sin_cos() {
  if constexpr (S == fpsan::Semantics::Triton)
    return true;
  else if constexpr (fpsan::detail::is_algebraic_semantics(S))
    return fpsan::Value<FT, S, fpsan::Conversions::Explicit>::alg_cfg().has_sin_cos;
  else
    return false;
}

// Width casts are value-faithful enough for these tests (cast(x*y)==cast(x)*cast(y)
// where meaningful, and 0/1/-1 map across widths). Triton resizes the scramble
// invertibly; FieldWithMulCasts carries a genuine multiplicative cast
// homomorphism; the cheap-cast Field variants and the composite two-moduli rings
// do not, so their casts are best-effort.
template <class FT, fpsan::Semantics S> constexpr bool flavor_has_cast_homomorphism() {
  if constexpr (S == fpsan::Semantics::Triton)
    return true;
  else if constexpr (fpsan::detail::has_multiplicative_field_casts(S))
    return true;
  else
    return false;
}

// cbrt is an exact multiplicative cube root (cbrt(x)^3==x on the cube residues).
// Only the algebraic variants whose group order is coprime to 3 (Field / Sophie
// Germain, not Pythagorean) and that opt into algebraic roots honor it; Triton
// and FieldFast make it a tag.
template <class FT, fpsan::Semantics S> constexpr bool flavor_has_cbrt() {
  if constexpr (fpsan::detail::is_algebraic_semantics(S) && !fpsan::detail::has_fast_field_ops(S))
    return fpsan::Value<FT, S, fpsan::Conversions::Explicit>::alg_cfg().has_cbrt;
  else
    return false;
}

} // namespace fpsan_test

// Convenience for tests whose body is a single Semantics-templated function
// fn<S>(args...).
//   TEST(Foo, BarFpsan) { FPSAN_RUN_FPSAN_SEMANTICS(test_bar, 17); }
#define FPSAN_RUN_FPSAN_SEMANTICS(fn, ...)                                                         \
  ::fpsan_test::for_each_fpsan_semantics([&](auto sem) { fn<decltype(sem)::value>(__VA_ARGS__); })

#define FPSAN_RUN_ALL_VARIANTS(fn, ...)                                                            \
  ::fpsan_test::for_each_fpsan_semantics_all_variants(                                             \
      [&](auto sem) { fn<decltype(sem)::value>(__VA_ARGS__); })

#endif // FPSAN_TESTS_FPSAN_SEMANTICS_HPP
