// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/math_test.cpp
//
// FPSan math: exp2/exp cross-checked bit-for-bit against the ground-truth
// reference; algebraic identities for exp/exp2/cos/sin; determinism and
// op-distinctness for the tagged functions; and native (mode=false) parity with
// std::.
#include "fpsan/fpsan.hpp"

#include "fpsan_generic.hpp"
#include "test_random.hpp"
#include "test_utils.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using fpsan::Value;

namespace
{
    // Test values: zero (for the explicit identity checks), then deterministic
    // pseudo-random quarters (see test_random.hpp). Range kept modest so exp/exp2
    // do not overflow and so native libm parity is exercised on well-behaved
    // inputs; quarters are exact, so the FPSan ring identities stay exact.
    std::vector<float> xs()
    {
        std::vector<float> s   = {0.f, 1.f, -1.f};
        std::mt19937       rng = fpsan_test::make_rng();
        while(s.size() < 14)
            s.push_back(fpsan_test::pick_quarter<float>(rng, -36, 36)); // -9 .. 9
        return s;
    }
} // namespace

// ---- exp2 / exp payloads match the ground-truth reference exactly ----------
TEST(Math, Exp2ExpMatchGroundTruthFloat)
{
    using F         = Value<float, fpsan::Semantics::FPSan, fpsan::Conversions::Explicit>;
    const auto& fmt = fpsan_generic::formats::F32;
    for(float x : xs())
    {
        auto gen = fpsan_generic::FPSanFloat::embed(fmt, (uint32_t)bits_of(x));
        EXPECT_EQ(fpsan::exp2(F(x)).fpsan_payload(), gen.exp2().payload()) << x;
        EXPECT_EQ(fpsan::exp(F(x)).fpsan_payload(), gen.exp().payload()) << x;
    }
}

TEST(Math, Exp2MatchesGroundTruthFloat16Exhaustive)
{
#if FPSAN_HAS_FLOAT16
    using F = Value<_Float16, fpsan::Semantics::FPSan, fpsan::Conversions::Explicit>;
    fpsan_generic::FPFormat fmt{"half", 16, 5, 10, 15, true};
    for(uint32_t b = 0; b < (1u << 16); ++b)
    {
        uint16_t bb = (uint16_t)b;
        _Float16 v;
        std::memcpy(&v, &bb, sizeof v);
        auto gen = fpsan_generic::FPSanFloat::embed(fmt, b);
        ASSERT_EQ(fpsan::exp2(F(v)).fpsan_payload(), (uint16_t)gen.exp2().payload())
            << "bits 0x" << std::hex << b;
    }
#else
    GTEST_SKIP();
#endif
}

// ---- algebraic identities (exact in FPSan mode) ----------------------------
TEST(Math, ExpHomomorphism)
{
    using F = Value<float, fpsan::Semantics::FPSan, fpsan::Conversions::Explicit>;
    for(float a : xs())
        for(float b : xs())
        {
            F A(a), B(b);
            EXPECT_TRUE(fpsan::exp(A + B) == fpsan::exp(A) * fpsan::exp(B));
            EXPECT_TRUE(fpsan::exp2(A + B) == fpsan::exp2(A) * fpsan::exp2(B));
        }
    EXPECT_EQ(fpsan::exp(F(0.f)).fpsan_payload(), 1u);
    EXPECT_EQ(fpsan::exp2(F(0.f)).fpsan_payload(), 1u);
}

TEST(Math, TrigAngleAddition)
{
    using F = Value<float, fpsan::Semantics::FPSan, fpsan::Conversions::Explicit>;
    for(float a : xs())
        for(float b : xs())
        {
            F A(a), B(b);
            F cosA = fpsan::cos(A), sinA = fpsan::sin(A);
            F cosB = fpsan::cos(B), sinB = fpsan::sin(B);
            EXPECT_TRUE(fpsan::cos(A + B) == cosA * cosB - sinA * sinB);
            EXPECT_TRUE(fpsan::sin(A + B) == sinA * cosB + cosA * sinB);
        }
    EXPECT_EQ(fpsan::cos(F(0.f)).fpsan_payload(), 1u); // cos 0 = 1
    EXPECT_EQ(fpsan::sin(F(0.f)).fpsan_payload(), 0u); // sin 0 = 0
}

// ---- tagged ops: deterministic, op-distinct, not real math -----------------
TEST(Math, TaggedDeterministicAndDistinct)
{
    using F = Value<float, fpsan::Semantics::FPSan, fpsan::Conversions::Explicit>;
    for(float x : xs())
    {
        F a(x), b(x);
        EXPECT_TRUE(fpsan::log(a) == fpsan::log(b)); // deterministic
        // distinct ops generally produce distinct payloads for the same input
        if(x != 0.f)
        {
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
TEST(Math, NativeParity)
{
    using F = Value<float, fpsan::Semantics::Float, fpsan::Conversions::Explicit>;
    for(float x : xs())
    {
        F a(x);
        EXPECT_EQ(bits_of((float)fpsan::exp(a)), bits_of(std::exp(x)));
        EXPECT_EQ(bits_of((float)fpsan::exp2(a)), bits_of(std::exp2(x)));
        EXPECT_EQ(bits_of((float)fpsan::sin(a)), bits_of(std::sin(x)));
        EXPECT_EQ(bits_of((float)fpsan::cos(a)), bits_of(std::cos(x)));
        EXPECT_EQ(bits_of((float)fpsan::floor(a)), bits_of(std::floor(x)));
        EXPECT_EQ(bits_of((float)fpsan::ceil(a)), bits_of(std::ceil(x)));
        if(x > 0.f)
        {
            EXPECT_EQ(bits_of((float)fpsan::log(a)), bits_of(std::log(x)));
            EXPECT_EQ(bits_of((float)fpsan::sqrt(a)), bits_of(std::sqrt(x)));
            EXPECT_EQ(bits_of((float)fpsan::precise_sqrt(a)), bits_of(std::sqrt(x)));
        }
    }
}

// ---- modular: fma / fmod / min / max ---------------------------------------
TEST(Math, FmaMatchesMulAdd)
{
    using F = Value<float, fpsan::Semantics::FPSan, fpsan::Conversions::Explicit>;
    for(float a : xs())
        for(float b : xs())
            for(float c : xs())
            {
                F A(a), B(b), C(c);
                EXPECT_TRUE(fpsan::fma(A, B, C) == A * B + C); // exact in payload ring
            }
}

TEST(Math, MinMaxNativeParity)
{
    using F = Value<float, fpsan::Semantics::Float, fpsan::Conversions::Explicit>;
    for(float a : xs())
        for(float b : xs())
        {
            EXPECT_EQ(bits_of((float)fpsan::fmin(F(a), F(b))), bits_of(std::fmin(a, b)));
            EXPECT_EQ(bits_of((float)fpsan::fmax(F(a), F(b))), bits_of(std::fmax(a, b)));
        }
}

TEST(Math, MinMaxFpsanIdempotentCommutative)
{
    using F = Value<float, fpsan::Semantics::FPSan, fpsan::Conversions::Explicit>;
    for(float a : xs())
    {
        F A(a);
        EXPECT_TRUE(fpsan::min(A, A) == A);
        EXPECT_TRUE(fpsan::max(A, A) == A);
        for(float b : xs())
        {
            F B(b);
            EXPECT_TRUE(fpsan::min(A, B) == fpsan::min(B, A));
            EXPECT_TRUE(fpsan::max(A, B) == fpsan::max(B, A));
        }
    }
}
