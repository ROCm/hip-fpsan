// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/math_test.cpp
//
// FPSan math: abs, exp2/exp cross-checked bit-for-bit against the ground-truth
// reference; algebraic identities for exp/exp2/cos/sin; determinism and
// op-distinctness for the tagged functions; and native (mode=false) parity with
// std::.
#include "fpsan/fpsan.hpp"

#include "fpsan_generic.hpp"
#include "fpsan_semantics.hpp"
#include "test_random.hpp"
#include "test_utils.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

namespace {
// Test values: zero (for the explicit identity checks), then deterministic
// pseudo-random quarters (see test_random.hpp). Range kept modest so exp/exp2
// do not overflow and so native libm parity is exercised on well-behaved
// inputs; quarters are exact, so the FPSan ring identities stay exact.
std::vector<float> xs() {
  std::vector<float> s = {0.f, 1.f, -1.f};
  std::mt19937 rng = fpsan_test::make_rng();
  while (s.size() < 14)
    s.push_back(fpsan_test::pick_quarter<float>(rng, -36, 36)); // -9 .. 9
  return s;
}
} // namespace

// ---- exp2 / exp payloads match the ground-truth reference exactly ----------
// The ground-truth reference fpsan_generic models Triton specifically, so the
// exp/exp2 bit-for-bit cross-checks against it are Triton-only by construction.
// The exp homomorphism *law* is exercised on every exp-capable flavor by
// Math.ExpHomomorphism below, and the algebraic exp/log channels by
// algebraic_value_test.
TEST(Math, Exp2ExpMatchGroundTruthFloat) {
  using F = Value<float, fpsan::Semantics::Triton, fpsan::Conversions::Explicit>;
  const auto &fmt = fpsan_generic::formats::F32;
  for (float x : xs()) {
    auto gen = fpsan_generic::FPSanFloat::embed(fmt, static_cast<uint32_t>(bits_of(x)));
    EXPECT_EQ(fpsan::exp2(F(x)).fpsan_payload(), gen.exp2().payload()) << x;
    EXPECT_EQ(fpsan::exp(F(x)).fpsan_payload(), gen.exp().payload()) << x;
  }
}

TEST(Math, Exp2MatchesGroundTruthFloat16Exhaustive) {
#if FPSAN_HAS_FLOAT16
  using F = Value<_Float16, fpsan::Semantics::Triton, fpsan::Conversions::Explicit>;
  fpsan_generic::FPFormat fmt{"half", 16, 5, 10, 15, true};
  for (uint32_t b = 0; b < (1u << 16); ++b) {
    uint16_t bb = static_cast<uint16_t>(b);
    _Float16 v;
    std::memcpy(&v, &bb, sizeof v);
    auto gen = fpsan_generic::FPSanFloat::embed(fmt, b);
    ASSERT_EQ(fpsan::exp2(F(v)).fpsan_payload(), static_cast<uint16_t>(gen.exp2().payload()))
        << "bits 0x" << std::hex << b;
  }
#else
  GTEST_SKIP();
#endif
}

// ---- algebraic identities (exact in FPSan mode) ----------------------------
// exp(a+b) == exp(a)*exp(b) holds wherever exp is a genuine homomorphism: Triton
// (constructed) and the two-moduli algebraic variants (Sophie Germain /
// Pythagorean). The Field variants reduce exp to a tag, so the law is
// asserted only on the exp-capable flavors.
TEST(Math, ExpHomomorphism) {
  fpsan_test::for_each_fpsan_semantics_all_variants([](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    if constexpr (fpsan_test::flavor_has_exp_homomorphism<float, S>()) {
      using F = Value<float, S, Conversions::Explicit>;
      using B = typename F::bits_type;
      for (float a : xs())
        for (float b : xs()) {
          F A(a), B2(b);
          EXPECT_TRUE(fpsan::exp(A + B2) == fpsan::exp(A) * fpsan::exp(B2)) << "S=" << int(S);
          EXPECT_TRUE(fpsan::exp2(A + B2) == fpsan::exp2(A) * fpsan::exp2(B2)) << "S=" << int(S);
        }
      EXPECT_EQ(fpsan::exp(F(0.f)).fpsan_payload(), B(1)) << "S=" << int(S);
      EXPECT_EQ(fpsan::exp2(F(0.f)).fpsan_payload(), B(1)) << "S=" << int(S);
    }
  });
}

// cos/sin angle-addition holds wherever trig is genuine: Triton (constructed) and
// the Pythagorean variants. Elsewhere they are tags, so the law is
// asserted only on the trig-capable flavors.
TEST(Math, TrigAngleAddition) {
  fpsan_test::for_each_fpsan_semantics_all_variants([](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    if constexpr (fpsan_test::flavor_has_sin_cos<float, S>()) {
      using F = Value<float, S, Conversions::Explicit>;
      using B = typename F::bits_type;
      for (float a : xs())
        for (float b : xs()) {
          F A(a), B2(b);
          F cosA = fpsan::cos(A), sinA = fpsan::sin(A);
          F cosB = fpsan::cos(B2), sinB = fpsan::sin(B2);
          EXPECT_TRUE(fpsan::cos(A + B2) == cosA * cosB - sinA * sinB) << "S=" << int(S);
          EXPECT_TRUE(fpsan::sin(A + B2) == sinA * cosB + cosA * sinB) << "S=" << int(S);
        }
      EXPECT_EQ(fpsan::cos(F(0.f)).fpsan_payload(), B(1)) << "S=" << int(S); // cos 0 = 1
      EXPECT_EQ(fpsan::sin(F(0.f)).fpsan_payload(), B(0)) << "S=" << int(S); // sin 0 = 0
    }
  });
}

// ---- tagged ops: deterministic, op-distinct, not real math -----------------
// Determinism (same input -> same payload) and op-distinctness (different ops ->
// different payloads, catching UnaryOpId collisions) hold in every fpsan flavor:
// where an op is tagged, the operation id distinguishes it, and where it is a
// structured algebraic map the maps still differ. Driven over all flavors.
TEST(Math, TaggedDeterministicAndDistinct) {
  // Determinism (same input -> same payload) holds in every fpsan flavor.
  fpsan_test::for_each_fpsan_semantics_all_variants([](auto sem) {
    using F = Value<float, decltype(sem)::value, Conversions::Explicit>;
    for (float x : xs()) {
      F a(x), b(x);
      EXPECT_TRUE(fpsan::log(a) == fpsan::log(b)); // deterministic
    }
  });
  // Op-distinctness (each UnaryOpId yields a distinct payload, catching tag-id
  // collisions) is a Triton tagged-scramble property. The algebraic models give these
  // ops genuine structured maps that legitimately coincide -- precise_sqrt IS
  // sqrt, and sqrt(1) == rsqrt(1) == 1 -- so distinctness is a hard Triton-only
  // contract here.
  using F = Value<float, Semantics::Triton, Conversions::Explicit>;
  for (float x : xs()) {
    F a(x);
    if (x != 0.f) {
      EXPECT_TRUE(fpsan::log(a) != fpsan::sqrt(a));
      EXPECT_TRUE(fpsan::sqrt(a) != fpsan::rsqrt(a));
      // precise_sqrt is a distinct tag from sqrt/rsqrt (catches a
      // Sqrt/PreciseSqrt UnaryOpId collision).
      EXPECT_TRUE(fpsan::precise_sqrt(a) != fpsan::sqrt(a));
      EXPECT_TRUE(fpsan::precise_sqrt(a) != fpsan::rsqrt(a));
      EXPECT_TRUE(fpsan::floor(a) != fpsan::ceil(a));
      EXPECT_TRUE(fpsan::erf(a) != fpsan::log2(a));
      // New tagged ops (Rcp/Fract/Tanh) must be distinct from each other and
      // from the existing tagged set. This catches UnaryOpId collisions.
      EXPECT_TRUE(fpsan::rcp(a) != fpsan::sqrt(a));
      EXPECT_TRUE(fpsan::rcp(a) != fpsan::log(a));
      EXPECT_TRUE(fpsan::rcp(a) != fpsan::sin(a));
      EXPECT_TRUE(fpsan::rcp(a) != fpsan::cos(a));
      EXPECT_TRUE(fpsan::fract(a) != fpsan::rcp(a));
      EXPECT_TRUE(fpsan::tanh(a) != fpsan::rcp(a));
      EXPECT_TRUE(fpsan::tanh(a) != fpsan::fract(a));
    }
  }
}

// ---- native (mode=false) parity with std:: ---------------------------------
// Native-only by definition: this pins the Native passthrough to libm bit-for-bit.
// The sanitizing flavors deliberately do not reproduce hardware transcendentals.
TEST(Math, NativeParity) {
  using F = Value<float, fpsan::Semantics::Native, fpsan::Conversions::Explicit>;
  for (float x : xs()) {
    F a(x);
    EXPECT_EQ(bits_of(static_cast<float>(fpsan::exp(a))), bits_of(std::exp(x)));
    EXPECT_EQ(bits_of(static_cast<float>(fpsan::exp2(a))), bits_of(std::exp2(x)));
    EXPECT_EQ(bits_of(static_cast<float>(fpsan::sin(a))), bits_of(std::sin(x)));
    EXPECT_EQ(bits_of(static_cast<float>(fpsan::cos(a))), bits_of(std::cos(x)));
    EXPECT_EQ(bits_of(static_cast<float>(fpsan::abs(a))), bits_of(std::fabs(x)));
    EXPECT_EQ(bits_of(static_cast<float>(fpsan::fabs(a))), bits_of(std::fabs(x)));
    EXPECT_EQ(bits_of(static_cast<float>(fpsan::floor(a))), bits_of(std::floor(x)));
    EXPECT_EQ(bits_of(static_cast<float>(fpsan::ceil(a))), bits_of(std::ceil(x)));
    if (x > 0.f) {
      EXPECT_EQ(bits_of(static_cast<float>(fpsan::log(a))), bits_of(std::log(x)));
      EXPECT_EQ(bits_of(static_cast<float>(fpsan::sqrt(a))), bits_of(std::sqrt(x)));
      EXPECT_EQ(bits_of(static_cast<float>(fpsan::precise_sqrt(a))), bits_of(std::sqrt(x)));
    }
  }
}

// ---- modular: fma / fmod / min / max ---------------------------------------
TEST(Math, AbsFpsanBasicProperties) {
  fpsan_test::for_each_fpsan_semantics_all_variants([](auto sem) {
    using F = Value<float, decltype(sem)::value, Conversions::Explicit>;
    for (float x : xs()) {
      F a(x);
      EXPECT_TRUE(fpsan::abs(a) == a || fpsan::abs(a) == -a);
      EXPECT_TRUE(fpsan::fabs(a) == fpsan::abs(a));
      EXPECT_TRUE(fpsan::abs(fpsan::abs(a)) == fpsan::abs(a));
    }
  });
}

// fma(A,B,C) == A*B + C is an exact payload-ring identity in every fpsan flavor.
TEST(Math, FmaMatchesMulAdd) {
  fpsan_test::for_each_fpsan_semantics_all_variants([](auto sem) {
    using F = Value<float, decltype(sem)::value, Conversions::Explicit>;
    for (float a : xs())
      for (float b : xs())
        for (float c : xs()) {
          F A(a), B(b), C(c);
          EXPECT_TRUE(fpsan::fma(A, B, C) == A * B + C); // exact in payload ring
        }
  });
}

TEST(Math, MinMaxNativeParity) {
  using F = Value<float, fpsan::Semantics::Native, fpsan::Conversions::Explicit>;
  for (float a : xs())
    for (float b : xs()) {
      EXPECT_EQ(bits_of(static_cast<float>(fpsan::fmin(F(a), F(b)))), bits_of(std::fmin(a, b)));
      EXPECT_EQ(bits_of(static_cast<float>(fpsan::fmax(F(a), F(b)))), bits_of(std::fmax(a, b)));
    }
}

// min/max are idempotent and commutative in every fpsan flavor.
TEST(Math, MinMaxFpsanIdempotentCommutative) {
  fpsan_test::for_each_fpsan_semantics_all_variants([](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using F = Value<float, S, Conversions::Explicit>;
    for (float a : xs()) {
      F A(a);
      EXPECT_TRUE(fpsan::min(A, A) == A) << "S=" << int(S);
      EXPECT_TRUE(fpsan::max(A, A) == A) << "S=" << int(S);
      for (float b : xs()) {
        F B(b);
        EXPECT_TRUE(fpsan::min(A, B) == fpsan::min(B, A)) << "S=" << int(S);
        EXPECT_TRUE(fpsan::max(A, B) == fpsan::max(B, A)) << "S=" << int(S);
      }
    }
  });
}
