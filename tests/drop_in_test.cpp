// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/drop_in_test.cpp
//
// Demonstrates that Value<T, fpsan::Semantics::Native,
// fpsan::Conversions::Implicit> is a bit-exact drop-in for T: a generic numeric
// kernel, instantiated on the wrapper, produces identical bits to the same
// kernel on the raw float. Also covers numeric_limits and io.
#include "fpsan/fpsan.hpp"
#include "fpsan/io.hpp"

#include "fpsan_semantics.hpp"
#include "test_random.hpp"
#include "test_utils.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <vector>

using fpsan::Value;

namespace {
// A generic numeric kernel written against an abstract scalar type: Horner
// polynomial evaluation followed by a normalized accumulation. Exercises
// + - * / and comparisons, plus scalar literals.
template <class Scalar> Scalar kernel(const std::vector<double> &input) {
  Scalar acc(0);
  Scalar c0(2), c1(-3), c2(0.5);
  for (double d : input) {
    Scalar x(static_cast<float>(d));
    Scalar poly = ((c2 * x + c1) * x + c0); // Horner
    if (poly < Scalar(0))
      poly = poly * Scalar(-1);
    acc += poly / (x * x + Scalar(1));
  }
  return acc;
}

// Deterministic pseudo-random inputs (see test_random.hpp). Native-mode Value is
// a bit-exact passthrough to native ops, so the wrapper-vs-raw comparison holds
// for any finite values; quarters give broad, exactly-representable coverage.
std::vector<double> make_input(int n) {
  std::vector<double> v(n);
  std::mt19937 rng = fpsan_test::make_rng();
  for (double &d : v)
    d = fpsan_test::pick_quarter<double>(rng, -200, 200); // -50 .. 50
  return v;
}
} // namespace

// The bit-exact-vs-raw drop-in property is Native-only by definition: only the
// Native passthrough reproduces hardware float bit-for-bit (the sanitizing
// flavors deliberately diverge), and the kernel itself uses ordering and
// division that the payload algebras do not provide. So these two stay Native.
TEST(DropIn, KernelMatchesRawFloatBitExactly) {
  std::vector<double> input = make_input(32);
  float raw = kernel<float>(input);
  Value<float, fpsan::Semantics::Native, fpsan::Conversions::Implicit> wrapped =
      kernel<Value<float, fpsan::Semantics::Native, fpsan::Conversions::Implicit>>(input);
  EXPECT_EQ(bits_of(static_cast<float>(wrapped)), bits_of(raw));
}

TEST(DropIn, DoubleKernelMatchesRawDoubleBitExactly) {
  std::vector<double> input = make_input(24);
  double raw = kernel<double>(input);
  Value<double, fpsan::Semantics::Native, fpsan::Conversions::Implicit> wrapped =
      kernel<Value<double, fpsan::Semantics::Native, fpsan::Conversions::Implicit>>(input);
  EXPECT_EQ(bits_of(static_cast<double>(wrapped)), bits_of(raw));
}

namespace {
// numeric_limits forwards the underlying float type's traits in every flavor;
// only is_iec559 distinguishes Native (a true IEEE float) from the sanitizing
// flavors. The limit *values* are reproduced bit-exactly only by the
// bit-faithful flavors (Native identity, Triton invertible scramble); the
// algebraic encodings re-quantize by value, so their max()/epsilon()/lowest()
// need not unembed to the float extremes.
template <fpsan::Semantics S> void check_numeric_limits() {
  using W = Value<float, S, fpsan::Conversions::Explicit>;
  using NL = std::numeric_limits<W>;
  using FL = std::numeric_limits<float>;
  EXPECT_TRUE(NL::is_specialized) << "S=" << int(S);
  EXPECT_TRUE(NL::is_signed) << "S=" << int(S);
  EXPECT_FALSE(NL::is_integer) << "S=" << int(S);
  EXPECT_EQ(NL::digits, FL::digits) << "S=" << int(S);
  EXPECT_EQ(NL::is_iec559, S == fpsan::Semantics::Native) << "S=" << int(S);
  if constexpr (S == fpsan::Semantics::Native || S == fpsan::Semantics::Triton) {
    EXPECT_EQ(bits_of(static_cast<float>(NL::max())), bits_of(FL::max()));
    EXPECT_EQ(bits_of(static_cast<float>(NL::epsilon())), bits_of(FL::epsilon()));
    EXPECT_EQ(bits_of(static_cast<float>(NL::lowest())), bits_of(FL::lowest()));
  }
}
} // namespace

TEST(DropIn, NumericLimitsForward) {
  check_numeric_limits<fpsan::Semantics::Native>();
  fpsan_test::for_each_fpsan_semantics(
      [](auto sem) { check_numeric_limits<decltype(sem)::value>(); });
}

TEST(DropIn, StreamOutput) {
  // Output leads with the exact Semantics enum name; Native then shows the value.
  std::ostringstream a;
  a << Value<float, fpsan::Semantics::Native, fpsan::Conversions::Implicit>(2.5f);
  EXPECT_EQ(a.str(), "Native(2.5)");

  // Triton streams its payload as raw hex bits; the payload of 1.0 is 1.
  std::ostringstream t;
  t << Value<float, fpsan::Semantics::Triton, fpsan::Conversions::Explicit>(1.0f);
  EXPECT_NE(t.str().find("payload=0x1"), std::string::npos) << t.str();

  // Algebraic flavors stream the residue as "x mod n"; phi(1)==1 in all of them.
  fpsan_test::for_each_fpsan_semantics([](auto sem) {
    constexpr fpsan::Semantics S = decltype(sem)::value;
    std::ostringstream b;
    b << Value<float, S, fpsan::Conversions::Explicit>(1.0f);
    if constexpr (fpsan::detail::is_algebraic_semantics(S)) {
      EXPECT_NE(b.str().find("payload=1 mod "), std::string::npos)
          << "S=" << int(S) << " " << b.str();
    }
    EXPECT_FALSE(b.str().empty()) << "S=" << int(S);
  });
}

TEST(DropIn, ExplicitModeUsableWithExplicitSyntax) {
  // Explicit-mode construction, explicit cast, and homogeneous ops are usable
  // (compile and run, deterministically) in every flavor.
  fpsan_test::for_each_fpsan_semantics([](auto sem) {
    constexpr fpsan::Semantics S = decltype(sem)::value;
    using E = Value<float, S, fpsan::Conversions::Explicit>;
    E x(1.0f);         // direct-init OK even though explicit
    E y = x + E(2.0f); // homogeneous op with explicit construction
    if constexpr (!fpsan::detail::is_algebraic_semantics(S))
      (void)static_cast<float>(y);                   // explicit cast OK (Native/Triton only)
    EXPECT_TRUE(y == x + E(2.0f)) << "S=" << int(S); // deterministic
  });
  // Native is the value-faithful drop-in: the result reads back as 3.0 via
  // static_cast<float>. Triton's arithmetic is not value-faithful, and the
  // algebraic to_float() is not a value-unembed (the value lives in the payload,
  // checked in core_test / algebraic_value_test), so the float read-back is a
  // Native-only property here.
  using E = Value<float, fpsan::Semantics::Native, fpsan::Conversions::Explicit>;
  E x(1.0f);
  EXPECT_EQ(bits_of(static_cast<float>(x)), bits_of(1.0f));
  E y = x + E(2.0f);
  EXPECT_EQ(bits_of(static_cast<float>(y)), bits_of(3.0f));
}

TEST(DropIn, CopyAssignValueSemantics) {
  fpsan_test::for_each_fpsan_semantics([](auto sem) {
    constexpr fpsan::Semantics S = decltype(sem)::value;
    using V = Value<float, S, fpsan::Conversions::Explicit>;
    V a(3.0f);
    V b = a; // copy
    EXPECT_TRUE(a == b) << "S=" << int(S);
    V c(9.0f);
    c = a; // assign
    EXPECT_TRUE(c == a) << "S=" << int(S);
  });
}
