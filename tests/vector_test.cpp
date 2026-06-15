// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/vector_test.cpp
//
// Value over Clang vector element types. The key checks tie the vector path to
// the already-ground-truthed scalar path: a vector Value must embed and compute
// lane-for-lane identically to the corresponding scalar Values. Pure C++/Clang
// (no GPU).
#include "fpsan/fpsan.hpp"

#include "test_utils.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

template <class T, int N>
using vec = T __attribute__((ext_vector_type(N)));
using f4  = vec<float, 4>;
using d2  = vec<double, 2>;

// ---- a vector Value embeds lane-for-lane like scalar Values ----------------
TEST(Vector, EmbedMatchesScalarPerLane)
{
    using Vf = Value<f4, Semantics::Triton, Conversions::Explicit>;
    using Sf = Value<float, Semantics::Triton, Conversions::Explicit>;
    f4   v   = {1.5f, -2.0f, 0.0f, 3.14159f};
    Vf   a(v);
    auto pay = a.fpsan_payload(); // uint32 x4
    for(int i = 0; i < 4; ++i)
        EXPECT_EQ(pay[i], Sf(v[i]).fpsan_payload()) << "lane " << i;
    // round trip recovers the vector exactly
    f4 back = static_cast<f4>(a);
    for(int i = 0; i < 4; ++i)
        EXPECT_EQ(bits_of(back[i]), bits_of(v[i])) << "lane " << i;
}

// ---- ring arithmetic is lane-wise identical to scalar Values ---------------
TEST(Vector, FpsanArithmeticMatchesScalarPerLane)
{
    using Vf = Value<f4, Semantics::Triton, Conversions::Explicit>;
    using Sf = Value<float, Semantics::Triton, Conversions::Explicit>;
    f4   x = {1.f, 2.f, -3.f, 0.5f}, y = {4.f, -5.f, 6.f, 7.f};
    Vf   a(x), b(y);
    auto add = (a + b).fpsan_payload();
    auto sub = (a - b).fpsan_payload();
    auto mul = (a * b).fpsan_payload();
    auto neg = (-a).fpsan_payload();
    for(int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(add[i], (Sf(x[i]) + Sf(y[i])).fpsan_payload());
        EXPECT_EQ(sub[i], (Sf(x[i]) - Sf(y[i])).fpsan_payload());
        EXPECT_EQ(mul[i], (Sf(x[i]) * Sf(y[i])).fpsan_payload());
        EXPECT_EQ(neg[i], (-Sf(x[i])).fpsan_payload());
    }
}

// ---- FPSan ring laws hold lane-wise (exact) --------------------------------
TEST(Vector, FpsanRingLawsExact)
{
    using Vf = Value<f4, Semantics::Triton, Conversions::Explicit>;
    f4   xa = {1e8f, 1.f, -2.f, 3.f}, xb = {-1e8f, 2.f, 4.f, -5.f}, xc = {1.f, 3.f, -6.f, 7.f};
    Vf   a(xa), b(xb), c(xc);
    auto all_true = [](auto mask) {
        for(int i = 0; i < 4; ++i)
            if(!mask[i])
                return false;
        return true;
    };
    EXPECT_TRUE(all_true((a + b) + c == a + (b + c))); // associativity, exact
    EXPECT_TRUE(all_true((a * b) * c == a * (b * c)));
    EXPECT_TRUE(all_true(a * (b + c) == a * b + a * c)); // distributivity
}

// ---- mode=false vector Value is bit-exact native vector arithmetic ---------
TEST(Vector, ModeFalseMatchesNative)
{
    using Vn = Value<f4, Semantics::Native, Conversions::Explicit>;
    f4 x = {1.1f, 2.2f, -3.3f, 4.4f}, y = {0.5f, -0.25f, 8.f, 0.1f};
    f4 add = static_cast<f4>(Vn(x) + Vn(y));
    f4 mul = static_cast<f4>(Vn(x) * Vn(y));
    f4 nx = x + y, ny = x * y;
    for(int i = 0; i < 4; ++i)
    {
        EXPECT_EQ(bits_of(add[i]), bits_of(nx[i]));
        EXPECT_EQ(bits_of(mul[i]), bits_of(ny[i]));
    }
}

// ---- comparisons yield a per-lane mask (both modes) ------------------------
TEST(Vector, ComparisonMasks)
{
    using Vn = Value<f4, Semantics::Native, Conversions::Explicit>;
    f4   x = {1.f, 5.f, 3.f, 4.f}, y = {2.f, 2.f, 3.f, 1.f};
    auto lt = (Vn(x) < Vn(y)); // native float order
    auto eq = (Vn(x) == Vn(y));
    for(int i = 0; i < 4; ++i)
    {
        EXPECT_EQ((bool)lt[i], x[i] < y[i]);
        EXPECT_EQ((bool)eq[i], x[i] == y[i]);
    }
    // FPSan mode: == is exact payload equality, lane-wise.
    using Vf = Value<f4, Semantics::Triton, Conversions::Explicit>;
    auto feq = (Vf(x) == Vf(x));
    for(int i = 0; i < 4; ++i)
        EXPECT_TRUE((bool)feq[i]);
}

// ---- lane accessors get(i)/set(i) ------------------------------------------
TEST(Vector, GetSetLanes)
{
    using Vf = Value<f4, Semantics::Triton, Conversions::Explicit>;
    using Sf = Value<float, Semantics::Triton, Conversions::Explicit>;
    Vf a{};
    for(int i = 0; i < 4; ++i)
        a.set(i, Sf(float(i) + 0.5f));
    for(int i = 0; i < 4; ++i)
        EXPECT_TRUE(a.get(i) == Sf(float(i) + 0.5f));
}

// ---- double2 (64-bit lanes) sanity -----------------------------------------
TEST(Vector, Double2EmbedMatchesScalar)
{
    using Vd = Value<d2, Semantics::Triton, Conversions::Explicit>;
    using Sd = Value<double, Semantics::Triton, Conversions::Explicit>;
    d2   v   = {3.25, -7.5};
    auto pay = Vd(v).fpsan_payload(); // uint64 x2
    for(int i = 0; i < 2; ++i)
        EXPECT_EQ(pay[i], Sd(v[i]).fpsan_payload());
}

#if FPSAN_HAS_FLOAT16
using h8 = vec<_Float16, 8>;
TEST(Vector, Half8EmbedMatchesScalar)
{
    using Vh = Value<h8, Semantics::Triton, Conversions::Explicit>;
    using Sh = Value<_Float16, Semantics::Triton, Conversions::Explicit>;
    h8 v;
    for(int i = 0; i < 8; ++i)
        v[i] = static_cast<_Float16>(i - 3) * static_cast<_Float16>(0.5f);
    auto pay = Vh(v).fpsan_payload(); // uint16 x8
    for(int i = 0; i < 8; ++i)
        EXPECT_EQ(pay[i], Sh(v[i]).fpsan_payload()) << "lane " << i;
}
#endif
