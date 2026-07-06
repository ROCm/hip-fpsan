// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// End-to-end test of the algebraic Semantics wired into Value<>.
//   c++ -std=c++17 -I include tests/algebraic_value_test.cpp -o /tmp/algv && /tmp/algv
// ----------------------------------------------------------------------------
#include "fpsan/cast.hpp"
#include "fpsan/detail/fp8.hpp"
#include "fpsan/math.hpp"
#include "fpsan/value.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>

using namespace fpsan;
template <Semantics S> using F = Value<float, S, Conversions::Explicit>;
template <Semantics S> using D = Value<double, S, Conversions::Explicit>;
using v4f = float __attribute__((ext_vector_type(4)));
template <Semantics S> using VF = Value<v4f, S, Conversions::Explicit>;
#if FPSAN_HAS_FLOAT16
template <Semantics S> using H = Value<_Float16, S, Conversions::Explicit>;
#endif
#if FPSAN_HAS_BF16
template <Semantics S> using Bf = Value<__bf16, S, Conversions::Explicit>;
#endif

static long pass = 0, fail = 0;
static void check(bool ok, const char *msg) {
  if (ok)
    ++pass;
  else {
    ++fail;
    std::printf("  FAIL: %s\n", msg);
  }
}

// Generic multiply-accumulate over an arbitrary index order -- the exact shape
// the MMA dataflows use (acc = acc + a*b). Written once, instantiated for any
// Semantics: this is the orthogonality the intrinsic layer relies on.
template <class V> static V mac(const float *a, const float *b, int n, const int *order) {
  V acc{0.0f};
  for (int i = 0; i < n; ++i)
    acc = acc + V{a[order[i]]} * V{b[order[i]]};
  return acc;
}

static void checkf(bool ok, const char *tag, const char *what) {
  char buf[192];
  std::snprintf(buf, sizeof buf, "%s: %s", tag, what);
  check(ok, buf);
}

template <class V> static bool mask_lane(typename V::cmp_t mask, unsigned i) {
  return mask[i] != 0;
}

template <class V> static bool vector_mask_all(typename V::cmp_t mask, bool expected) {
  for (unsigned i = 0; i < V::lanes; ++i)
    if (mask_lane<V>(mask, i) != expected)
      return false;
  return true;
}

template <class V> static bool vector_values_equal(V a, V b) {
  for (unsigned i = 0; i < V::lanes; ++i)
    if (!(a.get(i) == b.get(i)))
      return false;
  return true;
}

template <class FP8, Semantics S> static void check_fp8_specials_exhaustive(const char *tag) {
  using V = Value<FP8, S, Conversions::Explicit>;
  const auto cfg = V::alg_cfg();
  for (unsigned b = 0; b < 256; ++b) {
    const FP8 x{static_cast<std::uint8_t>(b)};
    const float f = static_cast<float>(x);
    const auto p = V{x}.fpsan_payload();
    char msg[192];
    std::snprintf(msg, sizeof msg, "%s byte 0x%02x", tag, b);
    if (std::isnan(f))
      check(p == cfg.nan_code, msg);
    else if (std::isinf(f))
      check(p == cfg.inf_code, msg);
    else
      check(p < cfg.n, msg);
  }
}

#define CHECK_FP8_SPECIALS_ALL(FP8, NAME)                                                          \
  do {                                                                                             \
    check_fp8_specials_exhaustive<FP8, Semantics::Field>(NAME " Field");                           \
    check_fp8_specials_exhaustive<FP8, Semantics::Field2>(NAME " Field2");                         \
    check_fp8_specials_exhaustive<FP8, Semantics::FieldFast>(NAME " FieldFast");                   \
    check_fp8_specials_exhaustive<FP8, Semantics::FieldFast2>(NAME " FieldFast2");                 \
    check_fp8_specials_exhaustive<FP8, Semantics::FieldWithMulCasts>(NAME " FieldWithMulCasts");   \
    check_fp8_specials_exhaustive<FP8, Semantics::FieldWithMulCasts2>(NAME " FieldWithMulCasts2"); \
    check_fp8_specials_exhaustive<FP8, Semantics::SophieGermainRing>(NAME " SophieGermainRing");   \
    check_fp8_specials_exhaustive<FP8, Semantics::SophieGermainRing2>(NAME " SophieGermainRing2"); \
    check_fp8_specials_exhaustive<FP8, Semantics::PythagoreanRing>(NAME " PythagoreanRing");       \
    check_fp8_specials_exhaustive<FP8, Semantics::PythagoreanRing2>(NAME " PythagoreanRing2");     \
  } while (false)

// ===========================================================================
// Scorecard batteries: one function per scorecard SECTION, run for EVERY
// applicable variant so no row can silently regress. (See algebraic-fpsan.md.)
// ===========================================================================

// "ring axioms and exact value relations" -- hold in every algebraic model
// (the leaf encoding is itself a ring homomorphism, so identities of the exact
// values hold of the fingerprints). Use small dyadic rationals -- all units.
template <class V> static void battery_ring(const char *tag) {
  checkf(V{0.0f} + V{2.5f} == V{2.5f}, tag, "0+x == x");
  checkf(V{1.0f} * V{2.5f} == V{2.5f}, tag, "1*x == x");
  checkf(V{2.5f} - V{2.5f} == V{0.0f}, tag, "x-x == 0");
  checkf(V{2.0f} + V{2.0f} == V{4.0f}, tag, "constant fold 2+2 == 4");
  checkf(V{2.0f} * V{3.0f} == V{6.0f}, tag, "constant fold 2*3 == 6");
  checkf(V{0.5f} + V{0.5f} == V{1.0f}, tag, "constant fold 0.5+0.5 == 1");
  checkf(V{1.5f} + V{1.5f} == V{2.0f} * V{1.5f}, tag, "symbolic x+x == 2x");
  {
    V a{2.5f};
    checkf((a + V{1.0f}) * (a - V{1.0f}) == a * a - V{1.0f}, tag, "symbolic (x+1)(x-1)==x^2-1");
  }
  {
    V a{2.0f}, b{3.0f};
    checkf((a + b) * (a + b) == a * a + V{2.0f} * a * b + b * b, tag,
           "symbolic (a+b)^2 == a^2+2ab+b^2");
  }
  checkf(V{1.1f} + V{2.2f} == V{2.2f} + V{1.1f}, tag, "commutativity");
  checkf((V{1.1f} + V{2.2f}) + V{3.3f} == V{1.1f} + (V{2.2f} + V{3.3f}), tag, "associativity");
  {
    V a{2.5f}, b{1.25f}, c{0.5f};
    checkf(a * (b + c) == a * b + a * c, tag, "distributivity");
  }
}

// "division & field structure". For a field, division is total; for the composite
// rings it holds on units -- the small integers below are units (huge primes).
template <class V> static void battery_division(const char *tag) {
  long xx = 0, abb = 0, n = 0;
  for (int i = 1; i <= 50; ++i)
    for (int j = 1; j <= 7; ++j, ++n) {
      V a{static_cast<float>(i)}, b{static_cast<float>(j)};
      xx += (a / a == V{1.0f});
      abb += ((a / b) * b == a);
    }
  checkf(xx == n, tag, "x/x == 1 (on units)");
  checkf(abb == n, tag, "(a/b)*b == a (b a unit)");
}

// "infinity & NaN" -- the projective extension, identical across all variants.
template <class V> static void battery_infnan(const char *tag) {
  const V inf = V{1.0f} / V{0.0f};
  const V nan = V{0.0f} / V{0.0f};
  checkf(inf == V{2.0f} / V{0.0f}, tag, "1/0 is a single unsigned inf");
  checkf(V{1.0f} / inf == V{0.0f}, tag, "1/inf == 0");
  checkf(V{3.0f} + inf == inf, tag, "x + inf == inf");
  checkf(V{3.0f} * inf == inf, tag, "x * inf == inf");
  checkf(inf + inf == nan, tag, "inf +- inf -> NaN");
  checkf(inf - inf == nan, tag, "inf - inf -> NaN");
  checkf(V{0.0f} * inf == nan, tag, "0 * inf -> NaN");
  checkf(inf / inf == nan, tag, "inf / inf -> NaN");
  checkf(nan == nan, tag, "NaN deterministic (compares equal to itself)");
  checkf(nan + V{1.0f} == nan, tag, "NaN absorbing under +");
  checkf(nan * V{2.0f} == nan, tag, "NaN absorbing under *");
}

// "algebraic functions: roots". sqrt/rsqrt are multiplicative power maps in every
// variant; cbrt is a perfect cube root where 3 is coprime to the group order
// (Field/SophieGermain) and a tag otherwise (Pythagorean).
template <class V> static void battery_roots(const char *tag, bool has_cbrt) {
  checkf(V::alg_cfg().has_cbrt == has_cbrt, tag, "has_cbrt matches the variant");
  long ms = 0, rinv = 0, rcons = 0, sq = 0, n = 0;
  float xs[] = {1.f,  2.f,  3.f,  5.f,  7.f, 11.f,  13.f,  17.f,
                19.f, 23.f, 0.5f, 1.5f, 6.f, 0.25f, 0.75f, 10.f};
  for (float u : xs) {
    V a{u};
    rinv += (rsqrt(a) * sqrt(a) == V{1.0f});
    rcons += (rsqrt(a) == V{1.0f} / sqrt(a));
    for (float v : xs) {
      V b{v};
      ms += (sqrt(a * b) == sqrt(a) * sqrt(b)); // multiplicative: universal
      sq += (sqrt(a) * sqrt(a) == a);           // round-trip: square residues only
      ++n;
    }
  }
  checkf(ms == n, tag, "sqrt(x*y) == sqrt(x)*sqrt(y) (multiplicative)");
  // sqrt(x)^2 == x holds on the square residues only -- a proper, nonempty subset
  // (1 is always a square, not every value is). Field ~1/2, SG ~1/4, Pyth ~1/8.
  checkf(sq > 0 && sq < n, tag, "sqrt(x)^2 == x on the square residues (not all)");
  checkf(rinv == static_cast<long>(sizeof xs / sizeof *xs), tag, "rsqrt(x)*sqrt(x) == 1");
  checkf(rcons == static_cast<long>(sizeof xs / sizeof *xs), tag, "rsqrt == 1/sqrt");
  if (has_cbrt) {
    long mc = 0, c3 = 0;
    for (float u : xs)
      for (float v : xs) {
        V a{u}, b{v};
        mc += (cbrt(a * b) == cbrt(a) * cbrt(b));
        c3 += (cbrt(a) * cbrt(a) * cbrt(a) == a);
      }
    checkf(mc == n, tag, "cbrt(x*y) == cbrt(x)*cbrt(y)");
    checkf(c3 == n, tag, "cbrt(x)^3 == x (perfect)");
  } else {
    checkf(cbrt(V{7.0f}) == cbrt(V{7.0f}), tag, "cbrt deterministic (tag)");
    checkf(cbrt(V{2.0f} * V{3.0f}) != cbrt(V{2.0f}) * cbrt(V{3.0f}), tag,
           "cbrt NOT multiplicative (tag)");
  }
}

// "transcendental functions". two_moduli (SophieGermain/Pythagorean): exp/exp2/
// exp10 and their log inverses are genuine homomorphisms; otherwise (Field) every
// one is a tag (no homomorphism).
template <class V> static void battery_transcendental(const char *tag, bool two_moduli) {
  float as[] = {0.5f, 1.0f, 1.5f, 2.0f, -1.0f, 0.25f, 3.0f};
  float ms[] = {1.0f, 2.0f, 3.0f, 5.0f, 0.5f, 1.5f, 7.0f};
  long e = 0, e2 = 0, e10 = 0, l = 0, l2 = 0, l10 = 0, invc = 0, na = 0, nm = 0;
  for (float u : as)
    for (float v : as) {
      V a{u}, b{v};
      e += (exp(a + b) == exp(a) * exp(b));
      e2 += (exp2(a + b) == exp2(a) * exp2(b));
      e10 += (exp10(a + b) == exp10(a) * exp10(b));
      ++na;
    }
  for (float u : ms)
    for (float v : ms) {
      V a{u}, b{v};
      l += (log(a * b) == log(a) + log(b));
      l2 += (log2(a * b) == log2(a) + log2(b));
      l10 += (log10(a * b) == log10(a) + log10(b));
      invc += (exp(log(exp(a))) == exp(a));
      ++nm;
    }
  if (two_moduli) {
    checkf(e == na, tag, "exp(a+b) == exp(a)*exp(b)");
    checkf(e2 == na, tag, "exp2(a+b) == exp2(a)*exp2(b)");
    checkf(e10 == na, tag, "exp10(a+b) == exp10(a)*exp10(b)");
    checkf(l == nm, tag, "log(x*y) == log(x)+log(y)");
    checkf(l2 == nm, tag, "log2(x*y) == log2(x)+log2(y)");
    checkf(l10 == nm, tag, "log10(x*y) == log10(x)+log10(y)");
    checkf(invc == nm, tag, "exp(log(exp v)) == exp v");
    checkf(exp2(V{2.0f}) != exp(V{2.0f}) && exp10(V{2.0f}) != exp(V{2.0f}) &&
               exp10(V{2.0f}) != exp2(V{2.0f}),
           tag, "exp/exp2/exp10 are distinct bases");
  } else {
    // Field (and any non-two-moduli): every exp/log base is a tag.
    checkf(e < na && e2 < na && e10 < na, tag, "exp/exp2/exp10 NOT homomorphisms (tag)");
    checkf(l < nm && l2 < nm && l10 < nm, tag, "log/log2/log10 NOT homomorphisms (tag)");
  }
}

// "sin/cos". Pythagorean semantics carry a genuine angle-addition channel and
// cos^2+sin^2==1; otherwise sin/cos are tags (angle-addition fails).
template <class V> static void battery_sin_cos(const char *tag, bool has_sin_cos) {
  float xs[] = {0.5f, 1.0f, 1.5f, 2.0f, 3.0f, -1.0f, 0.25f};
  long ok = 0, n = 0;
  for (float u : xs)
    for (float v : xs) {
      V a{u}, b{v};
      bool c1 = (cos(a + b) == cos(a) * cos(b) - sin(a) * sin(b));
      bool c2 = (sin(a + b) == sin(a) * cos(b) + cos(a) * sin(b));
      ok += (c1 && c2);
      ++n;
    }
  if (has_sin_cos) {
    checkf(cos(V{0.0f}) == V{1.0f} && sin(V{0.0f}) == V{0.0f}, tag, "cos(0)==1, sin(0)==0");
    checkf(ok == n, tag, "sin/cos angle-addition");
    checkf(cos(V{1.3f}) * cos(V{1.3f}) + sin(V{1.3f}) * sin(V{1.3f}) == V{1.0f}, tag,
           "cos^2 + sin^2 == 1");
  } else {
    checkf(ok < n, tag, "sin/cos are tags (no angle-addition)");
  }
}

// "miscellaneous": min/max/comparisons do NOT follow the IEEE numeric order
// (scorecard ❌, shared by all incl. Triton), but they are deterministic,
// commutative, and reassociation-invariant -- the sanitizer property that must
// not regress.
template <class V> static void battery_misc(const char *tag) {
  checkf(min(V{1.0f}, V{2.0f}) == min(V{2.0f}, V{1.0f}), tag, "min commutes");
  checkf(max(V{1.0f}, V{2.0f}) == max(V{2.0f}, V{1.0f}), tag, "max commutes");
  checkf(max(min(V{3.0f}, V{1.0f}), V{2.0f}) == max(V{2.0f}, min(V{1.0f}, V{3.0f})), tag,
         "min/max reassociation-invariant");
}

template <class V> static typename V::bits_type first_non_qr_payload() {
  using Bits = typename V::bits_type;
  const auto cfg = V::alg_cfg();
  for (detail::u64 r = 2; r < cfg.n; ++r) {
    const Bits p = static_cast<Bits>(r);
    if (!detail::alg_is_nonzero_qr(cfg, p))
      return p;
  }
  return Bits{0};
}

template <class V> static void battery_qr_order(const char *tag) {
  static_assert(!V::is_vector, "scalar algebraic qr-order semantics required");
  static_assert(detail::alg_has_qr_order(V::alg_cfg()), "algebraic qr-order semantics required");
  using Bits = typename V::bits_type;
  const auto cfg = V::alg_cfg();
  const Bits qr_payload = Bits{1};
  const Bits non_qr_payload = first_non_qr_payload<V>();
  const V zero = V::from_fpsan_payload(Bits{0});
  const V qr = V::from_fpsan_payload(qr_payload);
  const V non_qr = V::from_fpsan_payload(non_qr_payload);
  const V inf = V::from_fpsan_payload(static_cast<Bits>(cfg.inf_code));
  const V nan = V::from_fpsan_payload(static_cast<Bits>(cfg.nan_code));

  checkf(non_qr_payload != Bits{0}, tag, "found a finite non-qr residue");
  checkf(detail::alg_is_nonzero_qr(cfg, qr.fpsan_payload()), tag, "1 is qr-positive");
  checkf(!detail::alg_is_nonzero_qr(cfg, zero.fpsan_payload()), tag, "0 is not qr-positive");
  checkf(!detail::alg_is_nonzero_qr(cfg, non_qr.fpsan_payload()), tag, "test residue is non-qr");
  checkf(detail::alg_is_nonzero_qr(cfg, (-non_qr).fpsan_payload()), tag,
         "negative of non-qr is qr-positive");

  checkf(abs(qr) == qr, tag, "abs keeps qr-positive residues");
  checkf(abs(-qr) == qr, tag, "abs(-qr) canonicalizes to qr");
  checkf(abs(non_qr) == -non_qr, tag, "abs(non-qr) negates to qr");
  checkf(abs(non_qr) == abs(-non_qr), tag, "abs is sign-canonical on nonzero residues");
  checkf(fabs(non_qr) == abs(non_qr), tag, "fabs delegates to abs");
  checkf(abs(zero) == zero, tag, "abs(0) == 0");
  checkf(abs(inf) == inf, tag, "abs(Inf) == Inf");
  checkf(abs(nan) == nan, tag, "abs(NaN) == NaN");

  checkf(non_qr < qr, tag, "qr order: non-qr < qr");
  checkf(zero < qr, tag, "qr order: zero < qr");
  checkf(!(qr < non_qr), tag, "qr order: qr is not < non-qr");
  checkf(!(zero < non_qr) && !(non_qr < zero), tag,
         "qr order: zero and non-qr are not strictly ordered");
  checkf(non_qr <= qr, tag, "qr order: non-qr <= qr");
  checkf(zero <= non_qr && non_qr <= zero, tag, "qr preorder: zero and non-qr <= each other");
  checkf(!(qr <= non_qr), tag, "qr order: qr is not <= non-qr");
  checkf(!(qr <= zero), tag, "qr order: qr is not <= zero");
  checkf(qr >= non_qr, tag, "qr order: qr >= non-qr");
  checkf(!(non_qr >= qr), tag, "qr order: non-qr is not >= qr");
  checkf(min(qr, non_qr) == non_qr && min(non_qr, qr) == non_qr, tag,
         "min selects non-qr before qr");
  checkf(min(qr, zero) == zero && min(zero, qr) == zero, tag, "min selects zero before qr");
  checkf(max(qr, non_qr) == qr && max(non_qr, qr) == qr, tag, "max selects qr before non-qr");
  checkf(max(qr, zero) == qr && max(zero, qr) == qr, tag, "max selects qr before zero");
}

template <class V>
static void battery_qr_small_primes(const char *tag, bool expect_two_positive,
                                    bool expect_three_positive) {
  using Elem = typename V::element_type;
  const auto cfg = V::alg_cfg();
  const V two{Elem{2.0f}};
  const V three{Elem{3.0f}};
  checkf(detail::alg_has_qr_order(cfg), tag, "small primes: selected factor supports qr-order");
  checkf(detail::alg_is_nonzero_qr(cfg, two.fpsan_payload()) == expect_two_positive, tag,
         expect_two_positive ? "2 is qr-positive" : "2 is not qr-positive");
  checkf(detail::alg_is_nonzero_qr(cfg, three.fpsan_payload()) == expect_three_positive, tag,
         expect_three_positive ? "3 is qr-positive" : "3 is not qr-positive");
}

template <class V> static void battery_qr_order_vector(const char *tag) {
  static_assert(V::is_vector, "vector algebraic qr-order semantics required");
  static_assert(V::lanes >= 4, "test expects at least four lanes");
  static_assert(detail::alg_has_qr_order(V::alg_cfg()), "algebraic qr-order semantics required");
  using Bits = typename V::bits_type;
  using LaneBits = detail::bits_lane_t<Bits>;
  using Scalar = Value<typename V::element_type, V::semantics, V::conversions>;
  const LaneBits qr_payload = LaneBits{1};
  const LaneBits non_qr_payload = static_cast<LaneBits>(first_non_qr_payload<Scalar>());
  Bits qr_bits{}, non_qr_bits{}, zero_bits{};
  for (unsigned i = 0; i < V::lanes; ++i) {
    qr_bits[i] = qr_payload;
    non_qr_bits[i] = non_qr_payload;
  }
  const V zero = V::from_fpsan_payload(zero_bits);
  const V qr = V::from_fpsan_payload(qr_bits);
  const V non_qr = V::from_fpsan_payload(non_qr_bits);

  checkf(vector_mask_all<V>(non_qr < qr, true), tag, "vector qr order: non-qr < qr");
  checkf(vector_mask_all<V>(zero < qr, true), tag, "vector qr order: zero < qr");
  checkf(vector_mask_all<V>(qr < non_qr, false), tag, "vector qr order: qr is not < non-qr");
  checkf(vector_mask_all<V>(qr <= non_qr, false), tag, "vector qr order: qr is not <= non-qr");
  checkf(vector_mask_all<V>(zero <= non_qr, true), tag, "vector qr preorder: zero <= non-qr");
  checkf(vector_values_equal(abs(non_qr), -non_qr), tag, "vector abs canonicalizes non-qr lanes");
  checkf(vector_values_equal(min(qr, non_qr), non_qr) &&
             vector_values_equal(min(non_qr, qr), non_qr),
         tag, "vector min selects non-qr before qr");
  checkf(vector_values_equal(max(qr, non_qr), qr) && vector_values_equal(max(non_qr, qr), qr), tag,
         "vector max selects qr before non-qr");

  Bits left{}, right{};
  left[0] = non_qr_payload;
  right[0] = qr_payload;
  left[1] = qr_payload;
  right[1] = non_qr_payload;
  left[2] = LaneBits{0};
  right[2] = qr_payload;
  left[3] = LaneBits{0};
  right[3] = non_qr_payload;
  for (unsigned i = 4; i < V::lanes; ++i) {
    left[i] = non_qr_payload;
    right[i] = qr_payload;
  }
  const auto mixed = V::from_fpsan_payload(left) < V::from_fpsan_payload(right);
  checkf(mask_lane<V>(mixed, 0) && !mask_lane<V>(mixed, 1) && mask_lane<V>(mixed, 2) &&
             !mask_lane<V>(mixed, 3),
         tag, "vector qr order: mixed lanes compare independently");
}

template <class V> static void battery_fast_field(const char *tag) {
  battery_misc<V>(tag);
  battery_qr_order<V>(tag);
  battery_ring<V>(tag);
  battery_infnan<V>(tag);
  battery_transcendental<V>(tag, false);
  battery_sin_cos<V>(tag, false);

  V a{7.0f}, b{3.0f};
  checkf(V{0.0f} / b == V{0.0f}, tag, "fast division preserves 0/x == 0");
  checkf(a / V{1.0f} == a, tag, "fast division preserves x/1 == x");
  checkf(a / b == a / b, tag, "fast division deterministic");
  checkf(a / a != V{1.0f}, tag, "fast division is tagged, not x/x == 1");
  checkf((a / b) * b != a, tag, "fast division is not a field inverse");
  checkf(sqrt(a * b) != sqrt(a) * sqrt(b), tag, "fast sqrt is tagged, not multiplicative");
  checkf(cbrt(a) * cbrt(a) * cbrt(a) != a, tag, "fast cbrt is tagged, not a cube root");
  checkf(rsqrt(a) * sqrt(a) != V{1.0f}, tag, "fast rsqrt is tagged, not reciprocal sqrt");
}

template <class V> static void battery_supported_narrow_type(const char *tag) {
  using FT = typename V::float_type;
  auto v = [](float x) { return V{static_cast<FT>(x)}; };

  checkf(v(0.0f) + v(2.0f) == v(2.0f), tag, "narrow type: 0+x == x");
  checkf(v(2.0f) + v(2.0f) == v(4.0f), tag, "narrow type: 2+2 == 4");
  checkf(v(3.0f) * v(2.0f) == v(6.0f), tag, "narrow type: 3*2 == 6");
  checkf((v(1.0f) + v(2.0f)) + v(3.0f) == v(1.0f) + (v(2.0f) + v(3.0f)), tag,
         "narrow type: reassociation");
  checkf(v(1.0f) / v(0.0f) == v(2.0f) / v(0.0f), tag, "narrow type: x/0 -> Inf");
  if constexpr (detail::has_fast_field_ops(V::semantics))
    checkf(v(7.0f) / v(7.0f) != v(1.0f), tag, "narrow type: fast division is tagged");
  else
    checkf(v(7.0f) / v(7.0f) == v(1.0f), tag, "narrow type: faithful x/x == 1");
  static_assert(detail::alg_has_qr_order(V::alg_cfg()),
                "checked-in algebraic configs must support qr-order");
  battery_qr_order<V>(tag);
}

// Run the whole scorecard over one variant (T = a Value<...> type).
template <class Fld, class SG, class Py>
static void run_scorecard(const char *fld_tag, const char *sg_tag, const char *py_tag) {
  battery_misc<Fld>(fld_tag);
  battery_misc<SG>(sg_tag);
  battery_misc<Py>(py_tag);
  battery_qr_order<Fld>(fld_tag);
  battery_qr_order<SG>(sg_tag);
  battery_qr_order<Py>(py_tag);
  battery_qr_small_primes<SG>(sg_tag, true, true);
  battery_qr_small_primes<Py>(py_tag, true, false);
  battery_ring<Fld>(fld_tag);
  battery_ring<SG>(sg_tag);
  battery_ring<Py>(py_tag);
  battery_division<Fld>(fld_tag);
  battery_division<SG>(sg_tag);
  battery_division<Py>(py_tag);
  battery_infnan<Fld>(fld_tag);
  battery_infnan<SG>(sg_tag);
  battery_infnan<Py>(py_tag);
  battery_roots<Fld>(fld_tag, true);
  battery_roots<SG>(sg_tag, true);
  battery_roots<Py>(py_tag, false);
  battery_transcendental<Fld>(fld_tag, false);
  battery_transcendental<SG>(sg_tag, true);
  battery_transcendental<Py>(py_tag, true);
  battery_sin_cos<Fld>(fld_tag, false);
  battery_sin_cos<SG>(sg_tag, false);
  battery_sin_cos<Py>(py_tag, true);
}

static void check_field_cast_tower_constants(detail::AlgVariant v, const char *tag) {
  constexpr unsigned widths[] = {4, 8, 16, 32, 64};
  detail::u64 orders[5]{};
  for (unsigned i = 0; i < 5; ++i) {
    const detail::u64 p = detail::alg_field_prime(v, widths[i]);
    const detail::u64 g = detail::alg_field_root(v, widths[i]);
    orders[i] = p - 1;

    char msg[192];
    std::snprintf(msg, sizeof msg, "p_%u is an 11 mod 12 Field prime", widths[i]);
    checkf(p != 0 && p % 12 == 11, tag, msg);

    std::snprintf(msg, sizeof msg, "g_%u^3 is primitive if g_%u is primitive", widths[i],
                  widths[i]);
    checkf(g != 0 && detail::alg_gcd(3, orders[i]) == 1, tag, msg);
  }

  for (unsigned i = 0; i < 5; ++i)
    for (unsigned j = i + 1; j < 5; ++j) {
      char msg[192];
      std::snprintf(msg, sizeof msg, "p_%u-1 divides p_%u-1", widths[i], widths[j]);
      checkf(orders[j] % orders[i] == 0, tag, msg);

      const detail::u64 cofactor = orders[j] / orders[i];
      std::snprintf(msg, sizeof msg, "(p_%u-1)/(p_%u-1) is coprime to p_%u-1", widths[j], widths[i],
                    widths[i]);
      checkf(detail::alg_gcd(cofactor, orders[i]) == 1, tag, msg);
    }
}

template <Semantics S> static void check_composite_dlog_fast_path(const char *tag) {
  using V = F<S>;
  const auto cfg = V::alg_cfg();
  checkf(cfg.two_moduli, tag, "composite dlog test requires a two-moduli variant");
  const detail::u64 p = cfg.n / cfg.d;
  const detail::u64 gp = cfg.g % p;
  const detail::u64 cases[] = {0,         1,         2,         3,         5,        cfg.d / 7,
                               cfg.d / 3, cfg.d / 2, cfg.d - 3, cfg.d - 2, cfg.d - 1};
  for (detail::u64 k : cases) {
    const detail::u64 r = detail::alg_powmod(cfg.g, k, cfg.n);
    const detail::u64 target = detail::alg_powmod(r % p, cfg.d + 1, p);
    const detail::u64 exact = detail::alg_dlog_scan(target, gp, cfg.d, p);
    const detail::u64 got = detail::alg_dlog1(cfg, r);
    char msg[192];
    std::snprintf(msg, sizeof msg, "dlog fast path matches exact scan for k=%llu",
                  static_cast<unsigned long long>(k));
    checkf(got == exact && exact == k, tag, msg);
    checkf(
        log(V::from_fpsan_payload(static_cast<typename V::bits_type>(r))) ==
            V::from_fpsan_payload(static_cast<typename V::bits_type>((cfg.n / cfg.d) * k % cfg.n)),
        tag, "log(g^k) lands in the additive order-d subgroup");
  }
}

int main() {
  using Alg = F<Semantics::Field>;
  using Scr = F<Semantics::Triton>; // Triton model

  // ---- systematic scorecard coverage: every section x every variant --------
  // (float width). The "2" twins run the same batteries to lock in that an
  // independent prime gives the same algebra. Double is covered below.
  run_scorecard<F<Semantics::Field>, F<Semantics::SophieGermainRing>,
                F<Semantics::PythagoreanRing>>("field", "SophieGermain", "Pythagorean");
  run_scorecard<F<Semantics::Field2>, F<Semantics::SophieGermainRing2>,
                F<Semantics::PythagoreanRing2>>("field2", "SophieGermain2", "Pythagorean2");
  battery_fast_field<F<Semantics::FieldFast>>("fieldFast");
  battery_fast_field<F<Semantics::FieldFast2>>("fieldFast2");
  {
    // Triton contrast: the value-faithful rows FAIL (the encoding
    // is a non-homomorphic scramble), while the pure ring identities still hold.
    checkf(Scr{2.0f} + Scr{2.0f} != Scr{4.0f}, "triton", "2+2 != 4 (no value fidelity)");
    checkf(Scr{2.0f} * Scr{3.0f} != Scr{6.0f}, "triton", "2*3 != 6");
    checkf(Scr{1.5f} + Scr{1.5f} != Scr{2.0f} * Scr{1.5f}, "triton", "x+x != 2x");
    Scr a{1.1f}, b{2.2f}, c{0.5f};
    checkf(a * (b + c) == a * b + a * c, "triton", "distributivity still holds (ring identity)");
    checkf((a + b) + c == a + (b + c), "triton", "associativity still holds");
  }
  // "collisions re-rollable": a fresh prime (the "2" twin) gives independent
  // blind spots -- the same value maps to a different residue under each.
  checkf(F<Semantics::Field>{0.5f}.fpsan_payload() != F<Semantics::Field2>{0.5f}.fpsan_payload(),
         "re-rollable", "field vs field2 distinct moduli");
  check_field_cast_tower_constants(detail::AlgVariant::Field1, "field cast tower");
  check_field_cast_tower_constants(detail::AlgVariant::Field2, "field2 cast tower");
  check_composite_dlog_fast_path<Semantics::SophieGermainRing>("SophieGermain dlog");
  check_composite_dlog_fast_path<Semantics::SophieGermainRing2>("SophieGermain2 dlog");
  check_composite_dlog_fast_path<Semantics::PythagoreanRing>("Pythagorean dlog");
  check_composite_dlog_fast_path<Semantics::PythagoreanRing2>("Pythagorean2 dlog");
  checkf(F<Semantics::Field>::alg_cfg().n == F<Semantics::FieldWithMulCasts>::alg_cfg().n,
         "field mirror", "Field and FieldWithMulCasts share the same prime");
  checkf(F<Semantics::Field>::alg_cfg().n == F<Semantics::FieldFast>::alg_cfg().n, "field mirror",
         "Field and FieldFast share the same prime");
  checkf(F<Semantics::Field2>::alg_cfg().n == F<Semantics::FieldWithMulCasts2>::alg_cfg().n,
         "field mirror", "Field2 and FieldWithMulCasts2 share the same prime");
  checkf(F<Semantics::Field2>::alg_cfg().n == F<Semantics::FieldFast2>::alg_cfg().n, "field mirror",
         "Field2 and FieldFast2 share the same prime");
  checkf(F<Semantics::Field>{0.5f}.fpsan_payload() ==
             F<Semantics::FieldWithMulCasts>{0.5f}.fpsan_payload(),
         "field mirror", "Field and FieldWithMulCasts share finite fingerprints");
  checkf(F<Semantics::Field>{0.5f}.fpsan_payload() == F<Semantics::FieldFast>{0.5f}.fpsan_payload(),
         "field mirror", "Field and FieldFast share finite fingerprints");
  checkf(F<Semantics::Field2>{0.5f}.fpsan_payload() ==
             F<Semantics::FieldWithMulCasts2>{0.5f}.fpsan_payload(),
         "field mirror", "Field2 and FieldWithMulCasts2 share finite fingerprints");
  checkf(F<Semantics::Field2>{0.5f}.fpsan_payload() ==
             F<Semantics::FieldFast2>{0.5f}.fpsan_payload(),
         "field mirror", "Field2 and FieldFast2 share finite fingerprints");
  battery_qr_order<F<Semantics::FieldWithMulCasts>>("fieldWithMulCasts");
  battery_qr_order<F<Semantics::FieldWithMulCasts2>>("fieldWithMulCasts2");
  battery_qr_order<Value<fp8_e4m3, Semantics::Field, Conversions::Explicit>>("fp8 e4m3 field");
  battery_qr_order<Value<fp8_e5m2, Semantics::Field, Conversions::Explicit>>("fp8 e5m2 field");
  battery_qr_order<Value<fp8_e4m3, Semantics::FieldFast, Conversions::Explicit>>(
      "fp8 e4m3 fieldFast");
  battery_qr_order<Value<fp8_e5m2, Semantics::FieldWithMulCasts, Conversions::Explicit>>(
      "fp8 e5m2 fieldWithMulCasts");
  battery_qr_order<Value<fp8_e4m3, Semantics::SophieGermainRing, Conversions::Explicit>>(
      "fp8 e4m3 SophieGermain");
  battery_qr_order<Value<fp8_e5m2, Semantics::SophieGermainRing, Conversions::Explicit>>(
      "fp8 e5m2 SophieGermain");
  battery_qr_order<Value<fp8_e4m3, Semantics::SophieGermainRing2, Conversions::Explicit>>(
      "fp8 e4m3 SophieGermain2");
  battery_qr_order<Value<fp8_e5m2, Semantics::SophieGermainRing2, Conversions::Explicit>>(
      "fp8 e5m2 SophieGermain2");
  battery_qr_order<Value<fp8_e4m3, Semantics::PythagoreanRing, Conversions::Explicit>>(
      "fp8 e4m3 Pythagorean");
  battery_qr_order<Value<fp8_e5m2, Semantics::PythagoreanRing, Conversions::Explicit>>(
      "fp8 e5m2 Pythagorean");
  battery_qr_order<Value<fp8_e4m3, Semantics::PythagoreanRing2, Conversions::Explicit>>(
      "fp8 e4m3 Pythagorean2");
  battery_qr_order<Value<fp8_e5m2, Semantics::PythagoreanRing2, Conversions::Explicit>>(
      "fp8 e5m2 Pythagorean2");
  battery_qr_small_primes<Value<fp8_e4m3, Semantics::SophieGermainRing, Conversions::Explicit>>(
      "fp8 e4m3 SophieGermain", true, true);
  battery_qr_small_primes<Value<fp8_e4m3, Semantics::SophieGermainRing2, Conversions::Explicit>>(
      "fp8 e4m3 SophieGermain2", true, true);
  battery_qr_small_primes<Value<fp8_e4m3, Semantics::PythagoreanRing, Conversions::Explicit>>(
      "fp8 e4m3 Pythagorean", true, false);
  battery_qr_small_primes<Value<fp8_e4m3, Semantics::PythagoreanRing2, Conversions::Explicit>>(
      "fp8 e4m3 Pythagorean2", true, false);
  battery_qr_order_vector<VF<Semantics::Field>>("vector field");
  battery_qr_order_vector<VF<Semantics::Field2>>("vector field2");
  battery_qr_order_vector<VF<Semantics::FieldFast>>("vector fieldFast");
  battery_qr_order_vector<VF<Semantics::FieldWithMulCasts>>("vector fieldWithMulCasts");
  checkf(F<Semantics::SophieGermainRing>{0.5f}.fpsan_payload() !=
             F<Semantics::SophieGermainRing2>{0.5f}.fpsan_payload(),
         "re-rollable", "SophieGermain vs twin distinct moduli");
  checkf(F<Semantics::PythagoreanRing>{0.5f}.fpsan_payload() !=
             F<Semantics::PythagoreanRing2>{0.5f}.fpsan_payload(),
         "re-rollable", "Pythagorean vs twin distinct moduli");

  // ---- algebraic: rational identities hold within a width ----
  check((Alg{2.0f} + Alg{2.0f}) == Alg{4.0f}, "alg: 2+2 == 4");
  check((Alg{3.0f} * Alg{3.0f}) == Alg{9.0f}, "alg: 3*3 == 9");
  check((Alg{1.5f} + Alg{1.5f}) == (Alg{2.0f} * Alg{1.5f}), "alg: x+x == 2x");
  {
    Alg a{2.5f}, b{1.25f}, c{0.5f};
    check(a * (b + c) == a * b + a * c, "alg: distributivity");
  }
  // field: x/x == 1 for every nonzero value
  {
    long ok = 0, n = 0;
    for (int i = 1; i <= 300; ++i, ++n)
      ok += ((Alg{static_cast<float>(i)} / Alg{static_cast<float>(i)}) == Alg{1.0f});
    check(ok == n, "alg: x/x == 1 (field)");
  }
  // commutativity / associativity
  check(Alg{1.1f} + Alg{2.2f} == Alg{2.2f} + Alg{1.1f}, "alg: a+b == b+a");
  check((Alg{1.1f} + Alg{2.2f}) + Alg{3.3f} == Alg{1.1f} + (Alg{2.2f} + Alg{3.3f}),
        "alg: associativity");

  // ---- contrast: Triton (FPSan) does NOT see value coincidences ----
  check((Scr{2.0f} + Scr{2.0f}) != Scr{4.0f}, "fpsan Triton: 2+2 != 4");

  // ---- a second prime variant is independent but obeys the same identities -
  using Alg2 = F<Semantics::Field2>;
  check((Alg2{2.0f} + Alg2{2.0f}) == Alg2{4.0f}, "alg2: 2+2 == 4");

  // ---- SophieGermainRing honors exp(a+b) == exp(a)*exp(b) ----
  using SG = F<Semantics::SophieGermainRing>;
  check(exp(SG{0.0f}) == SG{1.0f}, "exp: exp(0) == 1");
  {
    long ok = 0, n = 0;
    float xs[] = {0.5f, 1.0f, 1.5f, 2.0f, -1.0f, 0.25f, 3.0f};
    for (float u : xs)
      for (float v : xs) {
        SG a{u}, b{v};
        ok += (exp(a + b) == exp(a) * exp(b));
        ++n;
      }
    check(ok == n, "exp: exp(a+b) == exp(a)*exp(b) (SophieGermainRing)");
  }
  // the Field variant has no exp homomorphism (exp is a tag there)
  check(exp(Alg{1.25f} + Alg{2.5f}) != exp(Alg{1.25f}) * exp(Alg{2.5f}),
        "field: exp is NOT a homomorphism (tag)");

  // SophieGermainRing2 is independent: its own exp homomorphism, distinct modulus.
  using SG2 = F<Semantics::SophieGermainRing2>;
  check(exp(SG2{1.5f} + SG2{2.5f}) == exp(SG2{1.5f}) * exp(SG2{2.5f}),
        "SophieGermainRing2: exp(a+b)==exp(a)*exp(b)");
  // 0.5 -> (n+1)/2 differs between the two moduli (small integers wouldn't).
  check(SG{0.5f}.fpsan_payload() != SG2{0.5f}.fpsan_payload(),
        "SophieGermain variants use distinct moduli (different residue for 0.5)");

  // ---- log: SophieGermainRing honors log(x*y) == log(x) + log(y) ----
  check(log(SG{1.0f}) == SG{0.0f}, "log: log(1) == 0");
  {
    long ok = 0, n = 0;
    float xs[] = {1.0f, 2.0f, 3.0f, 5.0f, 0.5f, 1.5f, 7.0f};
    for (float u : xs)
      for (float v : xs) {
        SG a{u}, b{v};
        ok += (log(a * b) == log(a) + log(b));
        ++n;
      }
    check(ok == n, "log: log(x*y) == log(x)+log(y) (SophieGermainRing dlog)");
  }
  // log inverts exp on the exp-image: exp(log(exp v)) == exp v
  check(exp(log(exp(SG{1.5f}))) == exp(SG{1.5f}), "log: exp(log(exp v)) == exp v");
  // Field variant: log is a tag, not a homomorphism
  check(log(Alg{2.0f} * Alg{3.0f}) != log(Alg{2.0f}) + log(Alg{3.0f}),
        "field: log is NOT a homomorphism (tag)");

  // ---- PythagoreanRing: genuine sin/cos angle-addition (order-d rotation)
  using Py = F<Semantics::PythagoreanRing>;
  check(cos(Py{0.0f}) == Py{1.0f}, "pythagorean: cos(0) == 1");
  check(sin(Py{0.0f}) == Py{0.0f}, "pythagorean: sin(0) == 0");
  {
    long ok = 0, n = 0;
    float xs[] = {0.5f, 1.0f, 1.5f, 2.0f, 3.0f, -1.0f, 0.25f};
    for (float u : xs)
      for (float v : xs) {
        Py a{u}, b{v};
        bool c1 = (cos(a + b) == cos(a) * cos(b) - sin(a) * sin(b));
        bool c2 = (sin(a + b) == sin(a) * cos(b) + cos(a) * sin(b));
        ok += (c1 && c2);
        ++n;
      }
    check(ok == n, "pythagorean: angle-addition for sin & cos");
  }
  check(cos(Py{1.3f}) * cos(Py{1.3f}) + sin(Py{1.3f}) * sin(Py{1.3f}) == Py{1.0f},
        "pythagorean: cos^2 + sin^2 == 1");
  // PythagoreanRing also carries exp + log (p=4d+1 keeps the d-channel).
  check(exp(Py{1.0f} + Py{2.0f}) == exp(Py{1.0f}) * exp(Py{2.0f}),
        "pythagorean: exp homomorphism still holds");
  check(log(Py{2.0f} * Py{3.0f}) == log(Py{2.0f}) + log(Py{3.0f}),
        "pythagorean: log homomorphism still holds");
  // SophieGermainRing has no angle-addition channel (sin/cos are tags there).
  check(cos(SG{0.5f} + SG{1.0f}) != cos(SG{0.5f}) * cos(SG{1.0f}) - sin(SG{0.5f}) * sin(SG{1.0f}),
        "SophieGermainRing: sin/cos are tagged (no angle-addition)");

  // ---- exp2 / log2: a second homomorphism pair on the same order-d channel ----
  check(exp2(SG{0.0f}) == SG{1.0f}, "exp2: exp2(0) == 1");
  check(log2(SG{1.0f}) == SG{0.0f}, "log2: log2(1) == 0");
  {
    long ok = 0, n = 0;
    float xs[] = {0.5f, 1.0f, 1.5f, 2.0f, -1.0f, 0.25f, 3.0f};
    for (float u : xs)
      for (float v : xs) {
        SG a{u}, b{v};
        ok += (exp2(a + b) == exp2(a) * exp2(b));
        ++n;
      }
    check(ok == n, "exp2: exp2(a+b) == exp2(a)*exp2(b) (SophieGermainRing)");
  }
  {
    long ok = 0, n = 0;
    float xs[] = {1.0f, 2.0f, 3.0f, 5.0f, 0.5f, 1.5f, 7.0f};
    for (float u : xs)
      for (float v : xs) {
        SG a{u}, b{v};
        ok += (log2(a * b) == log2(a) + log2(b));
        ++n;
      }
    check(ok == n, "log2: log2(x*y) == log2(x)+log2(y) (SophieGermainRing dlog)");
  }
  check(exp2(log2(exp2(SG{1.5f}))) == exp2(SG{1.5f}), "log2: exp2(log2(exp2 v)) == exp2 v");
  // exp2 uses a distinct base, so it is NOT the same fingerprint as exp
  check(exp2(SG{2.0f}) != exp(SG{2.0f}), "exp2 != exp (distinct base change)");
  // exp2/log2 carry over to PythagoreanRing (also a two-moduli channel).
  check(exp2(Py{1.0f} + Py{2.0f}) == exp2(Py{1.0f}) * exp2(Py{2.0f}),
        "pythagorean: exp2 homomorphism still holds");
  check(log2(Py{2.0f} * Py{3.0f}) == log2(Py{2.0f}) + log2(Py{3.0f}),
        "pythagorean: log2 homomorphism still holds");
  // Field variant: exp2/log2 are tags, not homomorphisms
  check(exp2(Alg{1.25f} + Alg{2.5f}) != exp2(Alg{1.25f}) * exp2(Alg{2.5f}),
        "field: exp2 is NOT a homomorphism (tag)");
  check(log2(Alg{2.0f} * Alg{3.0f}) != log2(Alg{2.0f}) + log2(Alg{3.0f}),
        "field: log2 is NOT a homomorphism (tag)");

  // ---- exp10 / log10: the base-10 members of the same family ----
  check(exp10(SG{0.0f}) == SG{1.0f}, "exp10: exp10(0) == 1");
  check(log10(SG{1.0f}) == SG{0.0f}, "log10: log10(1) == 0");
  {
    long e10 = 0, l10 = 0, inv = 0, n = 0;
    float xs[] = {0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 5.0f, 7.0f};
    for (float u : xs)
      for (float v : xs) {
        SG a{u}, b{v};
        e10 += (exp10(a + b) == exp10(a) * exp10(b));
        l10 += (log10(a * b) == log10(a) + log10(b));
        inv += (exp10(log10(exp10(a))) == exp10(a));
        ++n;
      }
    check(e10 == n, "exp10: exp10(a+b) == exp10(a)*exp10(b) (SophieGermainRing)");
    check(l10 == n, "log10: log10(x*y) == log10(x)+log10(y)");
    check(inv == n, "log10: exp10(log10(exp10 v)) == exp10 v");
  }
  // the three bases are distinct fingerprints (e, 2, 10)
  check(exp10(SG{2.0f}) != exp(SG{2.0f}) && exp10(SG{2.0f}) != exp2(SG{2.0f}),
        "exp10 != exp and != exp2 (distinct base)");
  // carries to PythagoreanRing; tags in the Field variant
  check(exp10(Py{1.0f} + Py{2.0f}) == exp10(Py{1.0f}) * exp10(Py{2.0f}),
        "pythagorean: exp10 homomorphism holds");
  check(exp10(Alg{1.25f} + Alg{2.5f}) != exp10(Alg{1.25f}) * exp10(Alg{2.5f}),
        "field: exp10 is NOT a homomorphism (tag)");

  // ---- sqrt / cbrt / rsqrt: multiplicative algebraic roots ----
  check(Alg::alg_cfg().has_cbrt, "field: cbrt is a perfect cube root (has_cbrt)");
  {
    long ms = 0, mc = 0, c3 = 0, inset = 0, half = 0, n = 0;
    float xs[] = {1.f, 2.f, 3.f, 5.f, 7.f, 0.5f, 1.5f, 6.f, 9.f, 0.25f, 11.f, 13.f};
    for (float u : xs)
      for (float v : xs) {
        Alg a{u}, b{v};
        ms += (sqrt(a * b) == sqrt(a) * sqrt(b)); // sqrt multiplicative
        mc += (cbrt(a * b) == cbrt(a) * cbrt(b)); // cbrt multiplicative
        c3 += (cbrt(a) * cbrt(a) * cbrt(a) == a); // cbrt perfect: cbrt(x)^3==x
        Alg s2 = sqrt(a) * sqrt(a);
        inset += ((s2 == a) || (s2 == -a)); // sqrt(x)^2 == +/- x always
        half += (s2 == a);                  // == x on the squares (~half)
        ++n;
      }
    check(ms == n, "field sqrt: sqrt(x*y) == sqrt(x)*sqrt(y) (multiplicative)");
    check(mc == n, "field cbrt: cbrt(x*y) == cbrt(x)*cbrt(y) (multiplicative)");
    check(c3 == n, "field cbrt: cbrt(x)^3 == x (perfect)");
    check(inset == n, "field sqrt: sqrt(x)^2 in {x, -x} always");
    check(half > 0 && half < n,
          "field sqrt: round-trip sqrt(x)^2==x on ~half (quadratic residues)");
  }
  // rsqrt is exactly 1/sqrt
  {
    long ok = 0, cons = 0, n = 0;
    float xs[] = {1.f, 2.f, 3.f, 5.f, 0.5f, 4.f, 9.f, 7.f};
    for (float u : xs) {
      Alg a{u};
      ok += (rsqrt(a) * sqrt(a) == Alg{1.0f});
      cons += (rsqrt(a) == Alg{1.0f} / sqrt(a));
      ++n;
    }
    check(ok == n, "field rsqrt: rsqrt(x)*sqrt(x) == 1");
    check(cons == n, "field rsqrt == 1/sqrt");
  }
  // SophieGermainRing: sqrt multiplicative, cbrt a perfect cube root.
  check(SG::alg_cfg().has_cbrt, "SophieGermainRing: cbrt is a perfect cube root (has_cbrt)");
  {
    long ms = 0, c3 = 0, n = 0;
    float xs[] = {1.f, 2.f, 3.f, 5.f, 7.f, 0.5f, 1.5f, 6.f};
    for (float u : xs)
      for (float v : xs) {
        SG a{u}, b{v};
        ms += (sqrt(a * b) == sqrt(a) * sqrt(b));
        c3 += (cbrt(a) * cbrt(a) * cbrt(a) == a);
        ++n;
      }
    check(ms == n, "SophieGermainRing sqrt: multiplicative");
    check(c3 == n, "SophieGermainRing cbrt: cbrt(x)^3 == x (perfect)");
  }
  // PythagoreanRing: sqrt multiplicative; cbrt has no cube root here.
  check(!Py::alg_cfg().has_cbrt, "pythagorean: cbrt is a tag (3 divides the group order)");
  {
    long ms = 0, n = 0;
    float xs[] = {1.f, 2.f, 3.f, 5.f, 0.5f, 1.5f};
    for (float u : xs)
      for (float v : xs) {
        Py a{u}, b{v};
        ms += (sqrt(a * b) == sqrt(a) * sqrt(b));
        ++n;
      }
    check(ms == n, "pythagorean sqrt: multiplicative");
  }
  // contrast: the Triton's sqrt is a tag (not multiplicative)
  check(sqrt(Scr{2.0f} * Scr{3.0f}) != sqrt(Scr{2.0f}) * sqrt(Scr{3.0f}),
        "fpsan Triton: sqrt is a tag (not multiplicative)");

  // ---- FieldWithMulCasts casts form a commutative diagram of homomorphisms ----
  // The fp4|fp8|fp16|fp32|fp64 primes are a coprime tower, so every widening and
  // narrowing cast is multiplicative, they compose, and narrow(widen(x)) == x.
  // The fp32<->fp64 edge dlogs over p_32-1 ~ 2^32 via Pohlig-Hellman.
  {
    using F8 = Value<fp8_e4m3, Semantics::FieldWithMulCasts, Conversions::Explicit>;
    using F16 = Value<_Float16, Semantics::FieldWithMulCasts, Conversions::Explicit>;
    using F32 = Value<float, Semantics::FieldWithMulCasts, Conversions::Explicit>;
    using F64 = Value<double, Semantics::FieldWithMulCasts, Conversions::Explicit>;
    long w16 = 0, w32 = 0, nA = 0, rt = 0, w64 = 0, n64 = 0, rt64 = 0, n = 0;
    float xs[] = {1.f, 2.f, 3.f, 0.5f, 4.f, 6.f, 1.5f, 0.25f};
    for (float u : xs)
      for (float v : xs) {
        F8 a{static_cast<fp8_e4m3>(u)}, b{static_cast<fp8_e4m3>(v)};
        F32 c{u}, d{v};
        F64 e{static_cast<double>(u)}, f{static_cast<double>(v)};
        // widening is multiplicative (fp8 -> fp16, fp8 -> fp32)
        w16 += (cast<_Float16>(a * b) == cast<_Float16>(a) * cast<_Float16>(b));
        w32 += (cast<float>(a * b) == cast<float>(a) * cast<float>(b));
        // narrowing is multiplicative (fp32 -> fp8)
        nA += (cast<fp8_e4m3>(c * d) == cast<fp8_e4m3>(c) * cast<fp8_e4m3>(d));
        // up-then-down round trip recovers the original (narrow . widen == id)
        rt += (cast<fp8_e4m3>(cast<float>(a)) == a);
        // fp64 joins the tower: fp32<->fp64 multiplicative (Pohlig-Hellman),
        // and the fp32->fp64->fp32 round trip is the identity.
        w64 += (cast<double>(c * d) == cast<double>(c) * cast<double>(d));
        n64 += (cast<float>(e * f) == cast<float>(e) * cast<float>(f));
        rt64 += (cast<float>(cast<double>(c)) == c);
        ++n;
      }
    check(w16 == n, "field-with-mul-casts fp8->fp16: multiplicative "
                    "(cast(x*y)==cast(x)*cast(y))");
    check(w32 == n, "field-with-mul-casts fp8->fp32: multiplicative");
    check(nA == n, "field-with-mul-casts fp32->fp8: narrowing is multiplicative");
    check(rt == n, "field-with-mul-casts: narrow(widen(x)) == x "
                   "(round-trip identity)");
    check(w64 == n, "field-with-mul-casts fp32->fp64: multiplicative "
                    "(Pohlig-Hellman dlog)");
    check(n64 == n, "field-with-mul-casts fp64->fp32: narrowing is "
                    "multiplicative (Pohlig-Hellman)");
    check(rt64 == n, "field-with-mul-casts: narrow(widen(x)) == x across fp32<->fp64");
    // commutative diagram: widen direct == widen via an intermediate width
    {
      long ok = 0, ok64 = 0, m = 0;
      for (float u : xs) {
        F8 a{static_cast<fp8_e4m3>(u)};
        ok += (cast<float>(a) == cast<float>(cast<_Float16>(a))); // fp8->fp32 == fp8->fp16->fp32
        // fp8->fp64 direct == fp8->fp32->fp64 (fp64 composes into the diagram)
        ok64 += (cast<double>(a) == cast<double>(cast<float>(a)));
        ++m;
      }
      check(ok == m, "field-with-mul-casts: fp8->fp32 == fp8->fp16->fp32 "
                     "(widening composes)");
      check(ok64 == m, "field-with-mul-casts: fp8->fp64 == fp8->fp32->fp64 "
                       "(fp64 composes)");
    }
    check(cast<_Float16>(F8{static_cast<fp8_e4m3>(1.0f)}) == F16{static_cast<_Float16>(1.0f)},
          "field-with-mul-casts fp8->fp16: cast(1) == 1");
    check(cast<double>(F8{static_cast<fp8_e4m3>(1.0f)}) == F64{1.0},
          "field-with-mul-casts fp8->fp64: cast(1) == 1");
    check(cast<float>(F32{1.25f}) == F32{1.25f},
          "field-with-mul-casts: same-format same-width is identity");
    {
      using E5 = Value<fp8_e5m2, Semantics::FieldWithMulCasts, Conversions::Explicit>;
      F8 a{static_cast<fp8_e4m3>(1.5f)};
      const E5 direct = cast<fp8_e5m2>(a);
      const E5 via32 = cast<fp8_e5m2>(cast<float>(a));
      check(direct == via32,
            "field-with-mul-casts: same-width fp8 format cast composes through fp32");
      check(cast<fp8_e4m3>(direct) == a,
            "field-with-mul-casts: same-width fp8 format cast round-trips");
      check(direct.fpsan_payload() != static_cast<typename E5::bits_type>(a.fpsan_payload()),
            "field-with-mul-casts: same-width fp8 format cast is not identity");
    }
#if FPSAN_HAS_FLOAT16 && FPSAN_HAS_BF16
    {
      using BF16 = Value<__bf16, Semantics::FieldWithMulCasts, Conversions::Explicit>;
      F16 h{static_cast<_Float16>(1.1f)};
      const BF16 direct = cast<__bf16>(h);
      const BF16 via32 = cast<__bf16>(cast<float>(h));
      check(direct == via32,
            "field-with-mul-casts: f16->bf16 same-width cast composes through fp32");
      check(cast<_Float16>(direct) == h,
            "field-with-mul-casts: f16<->bf16 same-width cast round-trips");
      check(direct.fpsan_payload() != static_cast<typename BF16::bits_type>(h.fpsan_payload()),
            "field-with-mul-casts: f16->bf16 same-width cast is not identity");
    }
#endif
  }

  // ---- Inf / NaN reach the payload, via 1/0 ----
  check(((Alg{1.0f} / Alg{0.0f}) / (Alg{1.0f} / Alg{0.0f})) ==
            (Alg{1.0f} / Alg{0.0f}) / (Alg{1.0f} / Alg{0.0f}),
        "alg: Inf compares equal to itself (deterministic)");
  // FP8 formats have non-IEEE non-finite layouts. E4M3 has NaN but no Inf;
  // AMD FNUZ formats use negative zero as NaN and have no Inf. The algebraic
  // embed path must classify those raw source encodings before reducing finite
  // values modulo p.
  {
    CHECK_FP8_SPECIALS_ALL(fp8_e4m3, "fp8 e4m3");
    CHECK_FP8_SPECIALS_ALL(fp8_e5m2, "fp8 e5m2");
    CHECK_FP8_SPECIALS_ALL(amd_fp8_e4m3, "amd fp8 e4m3 fnuz");
    CHECK_FP8_SPECIALS_ALL(amd_fp8_e5m2, "amd fp8 e5m2 fnuz");

    using E4 = Value<fp8_e4m3, Semantics::Field, Conversions::Explicit>;
    using E5 = Value<fp8_e5m2, Semantics::Field, Conversions::Explicit>;
    using A4 = Value<amd_fp8_e4m3, Semantics::Field, Conversions::Explicit>;
    using A5 = Value<amd_fp8_e5m2, Semantics::Field, Conversions::Explicit>;
    auto e4cf = E4::alg_cfg();
    auto e5cf = E5::alg_cfg();
    auto a4cf = A4::alg_cfg();
    auto a5cf = A5::alg_cfg();
    check(E4{fp8_e4m3{std::uint8_t{0x78}}}.fpsan_payload() != e4cf.inf_code,
          "fp8 e4m3: all-ones exponent, zero mantissa is finite (no Inf)");
    check(E4{fp8_e4m3{std::uint8_t{0x7e}}}.fpsan_payload() != e4cf.nan_code,
          "fp8 e4m3: all-ones exponent is finite except max mantissa");
    check(E4{fp8_e4m3{std::uint8_t{0x7f}}}.fpsan_payload() == e4cf.nan_code,
          "fp8 e4m3: max positive code is NaN");
    check(E4{fp8_e4m3{std::uint8_t{0xff}}}.fpsan_payload() == e4cf.nan_code,
          "fp8 e4m3: max negative code is NaN");
    check(E5{fp8_e5m2{std::uint8_t{0x7c}}}.fpsan_payload() == e5cf.inf_code,
          "fp8 e5m2: all-ones exponent, zero mantissa is Inf");
    check(E5{fp8_e5m2{std::uint8_t{0x7d}}}.fpsan_payload() == e5cf.nan_code,
          "fp8 e5m2: all-ones exponent, nonzero mantissa is NaN");
    check(A4{amd_fp8_e4m3{std::uint8_t{0x80}}}.fpsan_payload() == a4cf.nan_code,
          "amd fp8 e4m3 fnuz: negative zero encodes NaN");
    check(A4{amd_fp8_e4m3{std::uint8_t{0x78}}}.fpsan_payload() != a4cf.inf_code,
          "amd fp8 e4m3 fnuz: no infinity encoding");
    check(A5{amd_fp8_e5m2{std::uint8_t{0x80}}}.fpsan_payload() == a5cf.nan_code,
          "amd fp8 e5m2 fnuz: negative zero encodes NaN");
  }

  // ---- orthogonality: the SAME generic dataflow runs for every Semantics, and
  //      is reassociation-invariant in a payload mode (the sanitizer property) --
  {
    float A[4] = {0.5f, 1.25f, -0.75f, 2.0f};
    float B[4] = {2.0f, 0.5f, 4.0f, 1.5f};
    int o1[4] = {0, 1, 2, 3};
    int o2[4] = {3, 1, 0, 2}; // a different accumulation order
    // compiles & runs identically for Native, Triton, and the algebraic variants:
    (void)mac<F<Semantics::Native>>(A, B, 4, o1);
    (void)mac<F<Semantics::Triton>>(A, B, 4, o1);
    check(mac<Alg>(A, B, 4, o1) == mac<Alg>(A, B, 4, o2),
          "alg: matmul is reassociation-invariant (sanitizer property)");
    check(mac<SG>(A, B, 4, o1) == mac<SG>(A, B, 4, o2),
          "SophieGermainRing: matmul is reassociation-invariant");
  }

  // ---- faithful fma: a*b+c exactly (algebraic model) ----
  check(fma(Alg{2.0f}, Alg{3.0f}, Alg{1.0f}) == Alg{7.0f}, "alg: fma(2,3,1) == 7");
  check(fma(Alg{1.5f}, Alg{2.0f}, Alg{0.5f}) == Alg{3.5f}, "alg: fma value-faithful");

  // ---- min/max: deterministic, commutative, reassociation-invariant ----
  check(min(Alg{1.0f}, Alg{2.0f}) == min(Alg{2.0f}, Alg{1.0f}), "alg: min commutes");
  check(max(min(Alg{3.0f}, Alg{1.0f}), Alg{2.0f}) == max(Alg{2.0f}, min(Alg{1.0f}, Alg{3.0f})),
        "alg: min/max reassoc-invariant");

  // ---- cast: same-width is identity; cross-width is deterministic ----
  check(cast<float>(Alg{1.25f}) == Alg{1.25f}, "alg: same-width cast is identity");
  {
    using H = Value<_Float16, Semantics::Field, Conversions::Explicit>;
    H h1 = cast<_Float16>(Alg{1.5f});
    H h2 = cast<_Float16>(Alg{1.5f});
    check(h1 == h2, "alg: cross-width cast is deterministic");
  }

  // ---- 64-bit element types (double): the same scorecard, every section -----
  // Runs the full battery set at n ~ 2^64, exercising the 128-bit modular
  // multiply, the overflow-safe modular add, the overflow-free cbrt exponent,
  // and -- for log/log2/log10 on the d ~ 2^31 channel -- the Pollard-rho dlog.
  run_scorecard<D<Semantics::Field>, D<Semantics::SophieGermainRing>,
                D<Semantics::PythagoreanRing>>("dbl field", "dbl SophieGermain", "dbl Pythagorean");
  run_scorecard<D<Semantics::Field2>, D<Semantics::SophieGermainRing2>,
                D<Semantics::PythagoreanRing2>>("dbl field2", "dbl SophieGermain2",
                                                "dbl Pythagorean2");
  battery_fast_field<D<Semantics::FieldFast>>("dbl fieldFast");
  battery_fast_field<D<Semantics::FieldFast2>>("dbl fieldFast2");
  battery_qr_order<D<Semantics::FieldWithMulCasts>>("dbl fieldWithMulCasts");
  battery_qr_order<D<Semantics::FieldWithMulCasts2>>("dbl fieldWithMulCasts2");
  check(D<Semantics::Field>{0.5}.fpsan_payload() != D<Semantics::Field2>{0.5}.fpsan_payload(),
        "dbl field/field2 use distinct moduli");
#if FPSAN_HAS_FLOAT16
  battery_supported_narrow_type<H<Semantics::Field>>("f16 field");
  battery_supported_narrow_type<H<Semantics::Field2>>("f16 field2");
  battery_supported_narrow_type<H<Semantics::FieldFast>>("f16 fieldFast");
  battery_supported_narrow_type<H<Semantics::FieldFast2>>("f16 fieldFast2");
  battery_supported_narrow_type<H<Semantics::FieldWithMulCasts>>("f16 fieldWithMulCasts");
  battery_supported_narrow_type<H<Semantics::FieldWithMulCasts2>>("f16 fieldWithMulCasts2");
  battery_supported_narrow_type<H<Semantics::SophieGermainRing>>("f16 SophieGermain");
  battery_supported_narrow_type<H<Semantics::SophieGermainRing2>>("f16 SophieGermain2");
  battery_supported_narrow_type<H<Semantics::PythagoreanRing>>("f16 Pythagorean");
  battery_supported_narrow_type<H<Semantics::PythagoreanRing2>>("f16 Pythagorean2");
  battery_qr_small_primes<H<Semantics::SophieGermainRing>>("f16 SophieGermain", true, true);
  battery_qr_small_primes<H<Semantics::SophieGermainRing2>>("f16 SophieGermain2", true, true);
  battery_qr_small_primes<H<Semantics::PythagoreanRing>>("f16 Pythagorean", true, false);
  battery_qr_small_primes<H<Semantics::PythagoreanRing2>>("f16 Pythagorean2", true, false);
#endif
#if FPSAN_HAS_BF16
  battery_supported_narrow_type<Bf<Semantics::Field>>("bf16 field");
  battery_supported_narrow_type<Bf<Semantics::Field2>>("bf16 field2");
  battery_supported_narrow_type<Bf<Semantics::FieldFast>>("bf16 fieldFast");
  battery_supported_narrow_type<Bf<Semantics::FieldFast2>>("bf16 fieldFast2");
  battery_supported_narrow_type<Bf<Semantics::FieldWithMulCasts>>("bf16 fieldWithMulCasts");
  battery_supported_narrow_type<Bf<Semantics::FieldWithMulCasts2>>("bf16 fieldWithMulCasts2");
  battery_supported_narrow_type<Bf<Semantics::SophieGermainRing>>("bf16 SophieGermain");
  battery_supported_narrow_type<Bf<Semantics::SophieGermainRing2>>("bf16 SophieGermain2");
  battery_supported_narrow_type<Bf<Semantics::PythagoreanRing>>("bf16 Pythagorean");
  battery_supported_narrow_type<Bf<Semantics::PythagoreanRing2>>("bf16 Pythagorean2");
  battery_qr_small_primes<Bf<Semantics::SophieGermainRing>>("bf16 SophieGermain", true, true);
  battery_qr_small_primes<Bf<Semantics::SophieGermainRing2>>("bf16 SophieGermain2", true, true);
  battery_qr_small_primes<Bf<Semantics::PythagoreanRing>>("bf16 Pythagorean", true, false);
  battery_qr_small_primes<Bf<Semantics::PythagoreanRing2>>("bf16 Pythagorean2", true, false);
#endif

  std::printf("passed %ld, failed %ld\n", pass, fail);
  return fail == 0 ? 0 : 1;
}
