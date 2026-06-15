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

#include "test_random.hpp"
#include "test_utils.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <vector>

using fpsan::Value;

namespace
{
    // A generic numeric kernel written against an abstract scalar type: Horner
    // polynomial evaluation followed by a normalized accumulation. Exercises
    // + - * / and comparisons, plus scalar literals.
    template <class Scalar>
    Scalar kernel(const std::vector<double>& input)
    {
        Scalar acc(0);
        Scalar c0(2), c1(-3), c2(0.5);
        for(double d : input)
        {
            Scalar x(static_cast<float>(d));
            Scalar poly = ((c2 * x + c1) * x + c0); // Horner
            if(poly < Scalar(0))
                poly = poly * Scalar(-1);
            acc += poly / (x * x + Scalar(1));
        }
        return acc;
    }

    // Deterministic pseudo-random inputs (see test_random.hpp). Float-mode Value is
    // a bit-exact passthrough to native ops, so the wrapper-vs-raw comparison holds
    // for any finite values; quarters give broad, exactly-representable coverage.
    std::vector<double> make_input(int n)
    {
        std::vector<double> v(n);
        std::mt19937        rng = fpsan_test::make_rng();
        for(double& d : v)
            d = fpsan_test::pick_quarter<double>(rng, -200, 200); // -50 .. 50
        return v;
    }
} // namespace

TEST(DropIn, KernelMatchesRawFloatBitExactly)
{
    std::vector<double>                                                  input = make_input(32);
    float                                                                raw = kernel<float>(input);
    Value<float, fpsan::Semantics::Native, fpsan::Conversions::Implicit> wrapped
        = kernel<Value<float, fpsan::Semantics::Native, fpsan::Conversions::Implicit>>(input);
    EXPECT_EQ(bits_of(static_cast<float>(wrapped)), bits_of(raw));
}

TEST(DropIn, DoubleKernelMatchesRawDoubleBitExactly)
{
    std::vector<double> input = make_input(24);
    double              raw   = kernel<double>(input);
    Value<double, fpsan::Semantics::Native, fpsan::Conversions::Implicit> wrapped
        = kernel<Value<double, fpsan::Semantics::Native, fpsan::Conversions::Implicit>>(input);
    EXPECT_EQ(bits_of(static_cast<double>(wrapped)), bits_of(raw));
}

TEST(DropIn, NumericLimitsForward)
{
    using W  = Value<float, fpsan::Semantics::Native, fpsan::Conversions::Implicit>;
    using NL = std::numeric_limits<W>;
    EXPECT_TRUE(NL::is_specialized);
    EXPECT_TRUE(NL::is_signed);
    EXPECT_FALSE(NL::is_integer);
    EXPECT_EQ(NL::digits, std::numeric_limits<float>::digits);
    EXPECT_EQ(bits_of(static_cast<float>(NL::max())), bits_of(std::numeric_limits<float>::max()));
    EXPECT_EQ(bits_of(static_cast<float>(NL::epsilon())),
              bits_of(std::numeric_limits<float>::epsilon()));
    EXPECT_EQ(bits_of(static_cast<float>(NL::lowest())),
              bits_of(std::numeric_limits<float>::lowest()));
    // FPSan-mode wrapper is not iec559; mode=false wrapper of float is.
    EXPECT_TRUE(NL::is_iec559);
    EXPECT_FALSE(
        (std::numeric_limits<
            Value<float, fpsan::Semantics::Triton, fpsan::Conversions::Implicit>>::is_iec559));
}

TEST(DropIn, StreamOutput)
{
    std::ostringstream a;
    a << Value<float, fpsan::Semantics::Native, fpsan::Conversions::Implicit>(2.5f);
    EXPECT_EQ(a.str(), "2.5");

    std::ostringstream b;
    b << Value<float, fpsan::Semantics::Triton, fpsan::Conversions::Implicit>(
        1.0f); // payload of 1.0 is 1
    EXPECT_NE(b.str().find("payload=1"), std::string::npos);
}

TEST(DropIn, ExplicitModeUsableWithExplicitSyntax)
{
    using E = Value<float, fpsan::Semantics::Native, fpsan::Conversions::Explicit>;
    E x(1.0f); // direct-init OK even though explicit
    EXPECT_EQ(bits_of(static_cast<float>(x)), bits_of(1.0f)); // explicit cast OK
    E y = x + E(2.0f); // homogeneous op with explicit construction
    EXPECT_EQ(bits_of(static_cast<float>(y)), bits_of(3.0f));
}

TEST(DropIn, CopyAssignValueSemantics)
{
    Value<float, fpsan::Semantics::Triton, fpsan::Conversions::Implicit> a(3.0f);
    Value<float, fpsan::Semantics::Triton, fpsan::Conversions::Implicit> b = a; // copy
    EXPECT_TRUE(a == b);
    Value<float, fpsan::Semantics::Triton, fpsan::Conversions::Implicit> c(9.0f);
    c = a; // assign
    EXPECT_TRUE(c == a);
}
