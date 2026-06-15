// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/cast_test.cpp
//
// fpsan::cast<ToFT> between scalar Values. Pure C++/Clang.
#include "fpsan/fpsan.hpp"

#include "test_utils.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

using fpsan::cast;
using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

#if FPSAN_HAS_FLOAT16
// FPSan f16<->f32 cast matches Triton: embed -> signed-resize payload ->
// unembed.
TEST(Cast, FpsanF16ToF32FixedPoints)
{
    using H = Value<_Float16, Semantics::Triton, Conversions::Explicit>;
    // 0, 1, -1 are shared fixed points (payload 0, 1, all-ones) at every width,
    // so the sign-resize maps them across precisions exactly.
    EXPECT_EQ(cast<float>(H(_Float16(0))).fpsan_payload(), 0u);
    EXPECT_EQ(cast<float>(H(_Float16(1))).fpsan_payload(), 1u);
    EXPECT_EQ(cast<float>(H(_Float16(-1))).fpsan_payload(), 0xFFFFFFFFu);
}

TEST(Cast, FpsanF32ToF16ToF32RoundTrips)
{
    using H = Value<_Float16, Semantics::Triton, Conversions::Explicit>;
    // truncate(sign_extend(p)) == p, so f16 -> f32 -> f16 recovers the payload.
    for(int i = 0; i < (1 << 16); ++i)
    {
        _Float16 v;
        uint16_t b = (uint16_t)i;
        std::memcpy(&v, &b, sizeof v);
        H    h(v);
        auto back = cast<_Float16>(cast<float>(h));
        ASSERT_EQ(back.fpsan_payload(), h.fpsan_payload()) << "bits 0x" << std::hex << i;
    }
}

TEST(Cast, FloatModeIsNativeConversion)
{
    using H = Value<_Float16, Semantics::Native, Conversions::Explicit>;
    using F = Value<float, Semantics::Native, Conversions::Explicit>;
    for(float x : {0.0f, 1.5f, -2.25f, 100.0f, 0.1f})
    {
        F f(x);
        EXPECT_EQ(bits_of(static_cast<_Float16>(cast<_Float16>(f))),
                  bits_of(static_cast<_Float16>(x)));
        H h(static_cast<_Float16>(x));
        EXPECT_EQ(bits_of(static_cast<float>(cast<float>(h))),
                  bits_of(static_cast<float>(static_cast<_Float16>(x))));
    }
}
#endif
