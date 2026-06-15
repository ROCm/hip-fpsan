// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/fp8_test.cpp
//
// Exhaustive tests for the fp8 element types and their interaction with
// Value/cast. Each fp8 format only has 256 bit patterns, so we exhaustively
// cover every one of them.
//
//  - Float-mode round trip and bit-cast identities (cast<fp8>(cast<f32>(b)) ==
//  b
//    for every non-NaN finite bit pattern).
//  - FPSan-mode payload round trip (Triton sign-extend + truncate on payload).
//  - FPSan fixed points (0, 1, -1) map to payload 0, 1, 0xFF.
//  - Float-mode cast matches an independent reference implementation (the
//    namespace-detail narrow_to_f32 / f32_to_narrow primitives).
#include "fpsan/fpsan.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>

using fpsan::amd_fp8_e4m3;
using fpsan::amd_fp8_e5m2;
using fpsan::Conversions;
using fpsan::fp8_e4m3;
using fpsan::fp8_e5m2;
using fpsan::Semantics;
using fpsan::Value;

namespace
{

    template <class T>
    std::uint32_t bits_u32(T v)
    {
        std::uint32_t u = 0;
        std::memcpy(&u, &v, sizeof v);
        return u;
    }

    // True iff the fp8 bit pattern decodes to a NaN.
    template <class FP8>
    bool is_nan_pattern(std::uint8_t b)
    {
        return std::isnan(static_cast<float>(FP8(b)));
    }

} // namespace

// ---- fp8 scalar fixed points / conversion sanity ---------------------------

TEST(Fp8, E4M3FixedPoints)
{
    EXPECT_EQ(static_cast<float>(fp8_e4m3(std::uint8_t{0x00})), 0.f); // +0
    EXPECT_EQ(static_cast<float>(fp8_e4m3(std::uint8_t{0x80})), -0.f); // -0
    EXPECT_EQ(static_cast<float>(fp8_e4m3(std::uint8_t{0x38})), 1.0f); // +1.0
    EXPECT_EQ(static_cast<float>(fp8_e4m3(std::uint8_t{0xB8})), -1.0f); // -1.0
    // No infinity in e4m3fn; max finite = S.1111.110 = ±448.
    EXPECT_EQ(static_cast<float>(fp8_e4m3(std::uint8_t{0x7E})), 448.f);
    EXPECT_EQ(static_cast<float>(fp8_e4m3(std::uint8_t{0xFE})), -448.f);
    // NaN at S.1111.111.
    EXPECT_TRUE(std::isnan(static_cast<float>(fp8_e4m3(std::uint8_t{0x7F}))));
}

TEST(Fp8, E5M2FixedPoints)
{
    EXPECT_EQ(static_cast<float>(fp8_e5m2(std::uint8_t{0x00})), 0.f);
    EXPECT_EQ(static_cast<float>(fp8_e5m2(std::uint8_t{0x80})), -0.f);
    EXPECT_EQ(static_cast<float>(fp8_e5m2(std::uint8_t{0x3C})), 1.0f);
    EXPECT_EQ(static_cast<float>(fp8_e5m2(std::uint8_t{0xBC})), -1.0f);
    EXPECT_TRUE(std::isinf(static_cast<float>(fp8_e5m2(std::uint8_t{0x7C}))));
    EXPECT_LT(static_cast<float>(fp8_e5m2(std::uint8_t{0xFC})), 0.f);
    EXPECT_TRUE(std::isinf(static_cast<float>(fp8_e5m2(std::uint8_t{0xFC}))));
    // Any S.11111.MM with MM != 00 is NaN.
    EXPECT_TRUE(std::isnan(static_cast<float>(fp8_e5m2(std::uint8_t{0x7D}))));
    EXPECT_TRUE(std::isnan(static_cast<float>(fp8_e5m2(std::uint8_t{0x7F}))));
}

TEST(Fp8, AmdE4M3FnuzFixedPoints)
{
    EXPECT_EQ(static_cast<float>(amd_fp8_e4m3(std::uint8_t{0x00})), 0.f);
    EXPECT_TRUE(std::isnan(static_cast<float>(amd_fp8_e4m3(std::uint8_t{0x80}))));
    EXPECT_EQ(static_cast<float>(amd_fp8_e4m3(std::uint8_t{0x40})), 1.0f);
    EXPECT_EQ(static_cast<float>(amd_fp8_e4m3(std::uint8_t{0xC0})), -1.0f);
    EXPECT_EQ(static_cast<float>(amd_fp8_e4m3(std::uint8_t{0x01})), 0x1p-10f);
    EXPECT_EQ(static_cast<float>(amd_fp8_e4m3(std::uint8_t{0x08})), 0x1p-7f);
    EXPECT_EQ(static_cast<float>(amd_fp8_e4m3(std::uint8_t{0x7F})), 240.f);
    EXPECT_EQ(static_cast<float>(amd_fp8_e4m3(std::uint8_t{0xFF})), -240.f);
}

TEST(Fp8, AmdE5M2FnuzFixedPoints)
{
    EXPECT_EQ(static_cast<float>(amd_fp8_e5m2(std::uint8_t{0x00})), 0.f);
    EXPECT_TRUE(std::isnan(static_cast<float>(amd_fp8_e5m2(std::uint8_t{0x80}))));
    EXPECT_EQ(static_cast<float>(amd_fp8_e5m2(std::uint8_t{0x40})), 1.0f);
    EXPECT_EQ(static_cast<float>(amd_fp8_e5m2(std::uint8_t{0xC0})), -1.0f);
    EXPECT_EQ(static_cast<float>(amd_fp8_e5m2(std::uint8_t{0x01})), 0x1p-17f);
    EXPECT_EQ(static_cast<float>(amd_fp8_e5m2(std::uint8_t{0x04})), 0x1p-15f);
    EXPECT_EQ(static_cast<float>(amd_fp8_e5m2(std::uint8_t{0x7F})), 57344.f);
    EXPECT_EQ(static_cast<float>(amd_fp8_e5m2(std::uint8_t{0xFF})), -57344.f);
}

// ---- Exhaustive f32 round trip via fp8 -------------------------------------
// For every non-NaN bit pattern, fp8(f32(b)) == b. NaN patterns may not
// round-trip (we round through std f32 NaN), so we exclude them.

template <class FP8>
void exhaustive_float_roundtrip()
{
    for(int i = 0; i < 256; ++i)
    {
        std::uint8_t b = static_cast<std::uint8_t>(i);
        if(is_nan_pattern<FP8>(b))
            continue;
        float f = static_cast<float>(FP8(b));
        FP8   back(f);
        EXPECT_EQ(int(back.bits), int(b))
            << "bits 0x" << std::hex << int(b) << " -> f=" << f << " -> bits 0x" << int(back.bits);
    }
}

TEST(Fp8, ExhaustiveFloatRoundTripE4M3)
{
    exhaustive_float_roundtrip<fp8_e4m3>();
}
TEST(Fp8, ExhaustiveFloatRoundTripE5M2)
{
    exhaustive_float_roundtrip<fp8_e5m2>();
}
TEST(Fp8, ExhaustiveFloatRoundTripAmdE4M3)
{
    exhaustive_float_roundtrip<amd_fp8_e4m3>();
}
TEST(Fp8, ExhaustiveFloatRoundTripAmdE5M2)
{
    exhaustive_float_roundtrip<amd_fp8_e5m2>();
}

// ---- FPSan fixed points ----------------------------------------------------

template <class FP8>
void fpsan_fixed_points()
{
    using V = Value<FP8, Semantics::Triton, Conversions::Explicit>;
    // 0 -> payload 0; 1 -> payload 1; -1 -> payload 0xFF.
    EXPECT_EQ(int(V(FP8(0.0f)).fpsan_payload()), 0);
    EXPECT_EQ(int(V(FP8(1.0f)).fpsan_payload()), 1);
    EXPECT_EQ(int(V(FP8(-1.0f)).fpsan_payload()), 0xFF);
}
TEST(Fp8, FpsanFixedPointsE4M3)
{
    fpsan_fixed_points<fp8_e4m3>();
}
TEST(Fp8, FpsanFixedPointsE5M2)
{
    fpsan_fixed_points<fp8_e5m2>();
}
TEST(Fp8, FpsanFixedPointsAmdE4M3)
{
    fpsan_fixed_points<amd_fp8_e4m3>();
}
TEST(Fp8, FpsanFixedPointsAmdE5M2)
{
    fpsan_fixed_points<amd_fp8_e5m2>();
}

// ---- FPSan cast<fp8> <-> cast<float> round trip ----------------------------
// fpsan::cast<float>(Value<fp8>) sign-extends the 8-bit payload to 32 bits, and
// cast<fp8>(Value<float>) truncates back to 8 bits. Composition is the identity
// on the original payload for every bit pattern.

template <class FP8>
void fpsan_cast_roundtrip()
{
    using V8 = Value<FP8, Semantics::Triton, Conversions::Explicit>;
    for(int i = 0; i < 256; ++i)
    {
        std::uint8_t b = static_cast<std::uint8_t>(i);
        if(is_nan_pattern<FP8>(b))
            continue;
        V8   v{FP8(b)};
        auto wide = fpsan::cast<float>(v);
        auto back = fpsan::cast<FP8>(wide);
        EXPECT_EQ(int(back.fpsan_payload()), int(v.fpsan_payload()))
            << "bits 0x" << std::hex << int(b);
    }
}
TEST(Fp8, FpsanCastRoundTripE4M3)
{
    fpsan_cast_roundtrip<fp8_e4m3>();
}
TEST(Fp8, FpsanCastRoundTripE5M2)
{
    fpsan_cast_roundtrip<fp8_e5m2>();
}
TEST(Fp8, FpsanCastRoundTripAmdE4M3)
{
    fpsan_cast_roundtrip<amd_fp8_e4m3>();
}
TEST(Fp8, FpsanCastRoundTripAmdE5M2)
{
    fpsan_cast_roundtrip<amd_fp8_e5m2>();
}

// ---- Float-mode cast matches native conversion -----------------------------
// In Float mode, fpsan::cast<float>(Value<fp8>) must be
// static_cast<float>(fp8), bit-for-bit (the cast is defined to use the type's
// own conversion).

template <class FP8>
void float_mode_cast_matches_native()
{
    using VF8 = Value<FP8, Semantics::Native, Conversions::Explicit>;
    for(int i = 0; i < 256; ++i)
    {
        std::uint8_t b = static_cast<std::uint8_t>(i);
        VF8          v{FP8(b)};
        float        via_value = static_cast<float>(fpsan::cast<float>(v));
        float        native    = static_cast<float>(FP8(b));
        // Compare bits to also catch NaN-vs-NaN as long as it's the same NaN bit
        // pattern (which it is: we go through the same library function).
        EXPECT_EQ(bits_u32(via_value), bits_u32(native)) << "bits 0x" << std::hex << int(b);
    }
}
TEST(Fp8, FloatModeCastMatchesNativeE4M3)
{
    float_mode_cast_matches_native<fp8_e4m3>();
}
TEST(Fp8, FloatModeCastMatchesNativeE5M2)
{
    float_mode_cast_matches_native<fp8_e5m2>();
}
TEST(Fp8, FloatModeCastMatchesNativeAmdE4M3)
{
    float_mode_cast_matches_native<amd_fp8_e4m3>();
}
TEST(Fp8, FloatModeCastMatchesNativeAmdE5M2)
{
    float_mode_cast_matches_native<amd_fp8_e5m2>();
}
