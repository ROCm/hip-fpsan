// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/core_test.cpp
//
// Core Value tests: embedding/round-trip, the fixed points, drop-in
// Native-mode parity with native arithmetic, FPSan-mode ring laws,
// mixed-POD operators, and cross-checks against the ground-truth reference
// fpsan_generic.hpp (which itself mirrors Triton). Exhaustive 16-bit checks
// live in their own TESTs so ctest -j can parallelize.
#include "fpsan/fpsan.hpp"

#include "fpsan_generic.hpp" // ground truth, namespace fpsan_generic
#include "test_random.hpp"
#include "test_utils.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

namespace {

// Sample values exercised across the float types. The first few are special
// anchors the property tests care about (zero, the multiplicative identity and
// its negation, a sign pair, a large magnitude); the rest are deterministic
// pseudo-random quarters (see test_random.hpp) for general coverage. Kept at 16
// entries because RingLaws iterates samples^3. Each value is whatever the cast
// to T yields, so it is exactly representable and round trips / ring laws stay
// exact.
template <class T> std::vector<T> samples() {
  std::vector<T> s = {T(0), T(1), T(-1), T(100), T(-100)};
  std::mt19937 rng = fpsan_test::make_rng();
  while (s.size() < 16)
    s.push_back(fpsan_test::pick_quarter<T>(rng, -400, 400)); // -100 .. 100
  return s;
}

} // namespace

// ===========================================================================
// Typed tests over the full (float_type x semantics x conversions)
// matrix. Each TypeParam bundles the three parameters.
// ===========================================================================
template <class FT, Semantics S, Conversions C> struct Cfg {
  using ftype = FT;
  static constexpr Semantics sem = S;
  static constexpr Conversions conv = C;
};

template <class C> class CoreTyped : public ::testing::Test {};
TYPED_TEST_SUITE_P(CoreTyped);

TYPED_TEST_P(CoreTyped, DefaultIsZero) {
  using FT = typename TypeParam::ftype;
  using F = Value<FT, TypeParam::sem, TypeParam::conv>;
  F z{};
  EXPECT_EQ(z.to_storage_bits(), typename F::bits_type(0));
}

TYPED_TEST_P(CoreTyped, RoundTripExact) {
  using FT = typename TypeParam::ftype;
  using F = Value<FT, TypeParam::sem, TypeParam::conv>;
  // phi is a bit-level bijection only for the bijective models (Native is the
  // identity; Triton is an invertible bit scramble), so the exact
  // construct/convert round trip is asserted there. The algebraic models embed
  // by value and deliberately do not bit-preserve arbitrary inputs; their
  // round-trip / quantization behavior is covered by algebraic_value_test.
  if constexpr (TypeParam::sem == Semantics::Native || TypeParam::sem == Semantics::Triton) {
    for (FT x : samples<FT>()) {
      F a(x);
      EXPECT_EQ(bits_of(a.to_float()), bits_of(x));
    }
  }
}

TYPED_TEST_P(CoreTyped, FixedPoints) {
  using FT = typename TypeParam::ftype;
  using F = Value<FT, TypeParam::sem, TypeParam::conv>;
  using B = typename F::bits_type;
  // Payload-algebra flavors only (Triton + algebraic); Native has no payload.
  if constexpr (F::is_fpsan) {
    // 0 and 1 are universal fixed points: any homomorphic encoding (and
    // Triton's free-ring scramble) sends the additive identity to 0 and the
    // multiplicative identity to 1.
    EXPECT_EQ(F(FT(0)).fpsan_payload(), B(0));
    EXPECT_EQ(F(FT(1)).fpsan_payload(), B(1));
    // -1 is the additive inverse of 1 in every flavor, with a flavor-specific
    // stored residue: Triton's free ring Z/2^w stores the all-ones two's
    // complement; an algebraic Z/nZ stores n-1.
    EXPECT_TRUE(F(FT(-1)) == -F(FT(1)));
    if constexpr (TypeParam::sem == Semantics::Triton)
      EXPECT_EQ(F(FT(-1)).fpsan_payload(), static_cast<B>(~B(0)));
    else
      EXPECT_EQ(F(FT(-1)).fpsan_payload(), static_cast<B>(F::alg_cfg().n - 1));
  }
}

TYPED_TEST_P(CoreTyped, AdditiveAndMultiplicativeIdentities) {
  using FT = typename TypeParam::ftype;
  using F = Value<FT, TypeParam::sem, TypeParam::conv>;
  F zero(FT(0)), one(FT(1));
  for (FT x : samples<FT>()) {
    F a(x);
    EXPECT_TRUE(a + zero == a);
    EXPECT_TRUE(a * one == a);
    EXPECT_TRUE(a - a == zero);
    EXPECT_TRUE(a + (-a) == zero);
  }
}

TYPED_TEST_P(CoreTyped, RingLaws) {
  using FT = typename TypeParam::ftype;
  using F = Value<FT, TypeParam::sem, TypeParam::conv>;
  auto s = samples<FT>();
  for (FT xa : s)
    for (FT xb : s)
      for (FT xc : s) {
        F a(xa), b(xb), c(xc);
        // In FPSan mode these are exact ring laws. In Native mode they are
        // the native float laws, which can round; so only assert the exact
        // ones in FPSan mode.
        if constexpr (TypeParam::sem != Semantics::Native) {
          EXPECT_TRUE((a + b) + c == a + (b + c));
          EXPECT_TRUE((a * b) * c == a * (b * c));
          EXPECT_TRUE(a * (b + c) == (a * b) + (a * c));
          EXPECT_TRUE(a + b == b + a);
          EXPECT_TRUE(a * b == b * a);
        } else {
          // commutativity holds even with rounding
          EXPECT_TRUE(a + b == b + a);
          EXPECT_TRUE(a * b == b * a);
        }
      }
}

TYPED_TEST_P(CoreTyped, ModeFalseMatchesNative) {
  using FT = typename TypeParam::ftype;
  using F = Value<FT, TypeParam::sem, TypeParam::conv>;
  if constexpr (TypeParam::sem == Semantics::Native) {
    auto s = samples<FT>();
    for (FT x : s)
      for (FT y : s) {
        F u(x), v(y);
        EXPECT_EQ(bits_of((u + v).to_float()), bits_of(static_cast<FT>(x + y)));
        EXPECT_EQ(bits_of((u - v).to_float()), bits_of(static_cast<FT>(x - y)));
        EXPECT_EQ(bits_of((u * v).to_float()), bits_of(static_cast<FT>(x * y)));
        if (bits_of(y) != bits_of(FT(0)))
          EXPECT_EQ(bits_of((u / v).to_float()), bits_of(static_cast<FT>(x / y)));
        EXPECT_EQ(u < v, x < y);
        EXPECT_EQ(u == v, x == y);
      }
  }
}

REGISTER_TYPED_TEST_SUITE_P(CoreTyped, DefaultIsZero, RoundTripExact, FixedPoints,
                            AdditiveAndMultiplicativeIdentities, RingLaws, ModeFalseMatchesNative);

// Build the type list: all (mode x explicit) combinations for each available
// float type.
// Native + Triton carry both conversion modes; the algebraic value
// models are orthogonal to Conversions, so one representative (Explicit) each.
#define FPSAN_CFGS(FT)                                                                             \
  Cfg<FT, Semantics::Native, Conversions::Implicit>,                                               \
      Cfg<FT, Semantics::Native, Conversions::Explicit>,                                           \
      Cfg<FT, Semantics::Triton, Conversions::Implicit>,                                           \
      Cfg<FT, Semantics::Triton, Conversions::Explicit>,                                           \
      Cfg<FT, Semantics::Field, Conversions::Explicit>,                                            \
      Cfg<FT, Semantics::Field2, Conversions::Explicit>,                                           \
      Cfg<FT, Semantics::FieldFast, Conversions::Explicit>,                                        \
      Cfg<FT, Semantics::FieldFast2, Conversions::Explicit>,                                       \
      Cfg<FT, Semantics::FieldWithMulCasts, Conversions::Explicit>,                                \
      Cfg<FT, Semantics::FieldWithMulCasts2, Conversions::Explicit>,                               \
      Cfg<FT, Semantics::SophieGermainRing, Conversions::Explicit>,                                \
      Cfg<FT, Semantics::SophieGermainRing2, Conversions::Explicit>,                               \
      Cfg<FT, Semantics::PythagoreanRing, Conversions::Explicit>,                                  \
      Cfg<FT, Semantics::PythagoreanRing2, Conversions::Explicit>

using CoreConfigs = ::testing::Types<FPSAN_CFGS(float), FPSAN_CFGS(double)
#if FPSAN_HAS_FLOAT16
                                                            ,
                                     FPSAN_CFGS(_Float16)
#endif
#if FPSAN_HAS_BF16
                                         ,
                                     FPSAN_CFGS(__bf16)
#endif
                                     >;
INSTANTIATE_TYPED_TEST_SUITE_P(All, CoreTyped, CoreConfigs);

// ===========================================================================
// Ground-truth cross-checks: the library's payloads must equal fpsan_generic's
// (and therefore Triton's) for every enumerable bit pattern.
// ===========================================================================
namespace {
template <class FT> void cross_check_all_bits(const fpsan_generic::FPFormat &fmt) {
  using F = Value<FT, fpsan::Semantics::Triton, fpsan::Conversions::Explicit>;
  using B = typename F::bits_type;
  const uint64_t n = uint64_t{1} << (sizeof(FT) * 8);
  for (uint64_t b = 0; b < n; ++b) {
    B bb = static_cast<B>(b);
    FT v;
    std::memcpy(&v, &bb, sizeof v);
    const B observed = static_cast<B>(bits_of(v));
    B lib = F(v).fpsan_payload();
    B gen = static_cast<B>(
        fpsan_generic::FPSanFloat::embed(fmt, static_cast<uint32_t>(observed)).payload());
    ASSERT_EQ(lib, gen) << "bit pattern 0x" << std::hex << b << " observed as 0x"
                        << static_cast<uint64_t>(observed);
  }
}
} // namespace

#if FPSAN_HAS_FLOAT16
TEST(GroundTruth, Float16AllBits) { cross_check_all_bits<_Float16>({"half", 16, 5, 10, 15, true}); }
#endif
#if FPSAN_HAS_BF16
TEST(GroundTruth, BF16AllBits) { cross_check_all_bits<__bf16>({"bf16", 16, 8, 7, 127, true}); }
#endif

TEST(GroundTruth, Float32Samples) {
  const auto &fmt = fpsan_generic::formats::F32;
  for (float x : samples<float>()) {
    uint32_t lib =
        Value<float, fpsan::Semantics::Triton, fpsan::Conversions::Explicit>(x).fpsan_payload();
    uint32_t gen =
        fpsan_generic::FPSanFloat::embed(fmt, static_cast<uint32_t>(bits_of(x))).payload();
    EXPECT_EQ(lib, gen) << "x=" << x;
  }
}

// ===========================================================================
// Exhaustive 16-bit: phi is a bijection and phi(-x) == -phi(x).
// ===========================================================================
#if FPSAN_HAS_FLOAT16
TEST(Exhaustive, Float16Bijection) {
  using F = Value<_Float16, fpsan::Semantics::Triton, fpsan::Conversions::Explicit>;
  std::vector<int> hit(1 << 16, 0);
  for (uint32_t b = 0; b < (1u << 16); ++b) {
    uint16_t bb = static_cast<uint16_t>(b);
    _Float16 v;
    std::memcpy(&v, &bb, sizeof v);
    uint16_t p = F(v).fpsan_payload();
    ++hit[p];
    if ((bb & 0x7FFF) != 0) {
      uint16_t negb = bb ^ 0x8000;
      _Float16 nv;
      std::memcpy(&nv, &negb, sizeof nv);
      EXPECT_EQ(F(nv).fpsan_payload(), static_cast<uint16_t>(-p)) << "x bits 0x" << std::hex << bb;
    }
  }
  for (int h : hit)
    EXPECT_EQ(h, 1);
}
#endif

// ===========================================================================
// Mixed Value <op> POD (Conversions::Implicit).
// ===========================================================================
TEST(MixedPod, ImplicitScalarOps) {
  using Ff = Value<float, fpsan::Semantics::Native, fpsan::Conversions::Implicit>;
  Ff u(1.0f);
  EXPECT_EQ(bits_of((u + 2.0f).to_float()), bits_of(3.0f));
  EXPECT_EQ(bits_of((2.0f + u).to_float()), bits_of(3.0f));
  EXPECT_EQ(bits_of((u * 4).to_float()), bits_of(4.0f)); // int scalar
  EXPECT_TRUE(u < 2.0f);
  EXPECT_TRUE(2.0f > u);

  using Ft = Value<float, fpsan::Semantics::Triton, fpsan::Conversions::Implicit>;
  Ft p(2.0f), three(3.0f);
  // mixed op must embed the scalar and combine payloads: same as homogeneous.
  EXPECT_TRUE(p + 3.0f == p + three);
  EXPECT_TRUE(p * 3.0f == p * three);
}
