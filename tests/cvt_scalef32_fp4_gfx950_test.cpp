// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/cvt_scalef32_fp4_gfx950_test.cpp
//
// gfx950 tests for the fpsan:: fp4 (e2m1) scaled-conversion WRAPPERS in
// fpsan/amdgcn_cvt.hpp (cvt_scalef32_pk_{f32,f16,bf16}_fp4 unpack;
// cvt_scalef32_pk_fp4_{f32,f16,bf16} + sr_pk_fp4_f32 pack). The sibling file
// cvt_scalef32_mx_gfx950_test.cpp exercises the raw builtins; this one drives
// the Value-typed wrappers in BOTH modes:
//
//   Float : wrapper == hardware, checked against the INDEPENDENT host OCP
//           reference (detail::narrow_to_f32 / f32_to_narrow) -- not the
//           builtin -- so any implementation that diverges from the host reference fails here.
//   FPSan : wrapper == an independent payload-domain reference. The packed
//           register holds 4-bit fp4 PAYLOADS per nibble; widen sign-extends
//           4->32, narrow keeps the low 4 bits, scale multiplies (unpack) /
//           divides (pack) in the f32 payload ring. The reference recomputes
//           that with explicit bit ops + public Value arithmetic, independent
//           of the wrapper's helpers.
#include "fpsan/amdgcn_cvt.hpp"
#include "fpsan/fpsan.hpp"

#include "hip_test_utils.hpp"
#include "test_random.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;
using fpsan::detail::f32_to_narrow;
using fpsan::detail::kFp4E2M1;
using fpsan::detail::narrow_to_f32;

static constexpr Conversions kCC = Conversions::Explicit;
using VF                         = Value<float, Semantics::Triton, kCC>;

namespace
{
    // Sign-extend the low 4 bits of n to 32 bits.
    std::int32_t sext4(std::uint32_t n)
    {
        return static_cast<std::int32_t>(n << 28) >> 28;
    }
} // namespace

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_scalef32_pk_f32_fp4)

// ============================ Float mode ====================================

// Unpack every fp4 code through the WRAPPER (Sel=0), scale=1, vs OCP host ref.
__global__ void k_f32_unpack(const unsigned* packed, float* out)
{
    int  l = threadIdx.x;
    auto r = fpsan::amdgcn_cvt_scalef32_pk_f32_fp4<0, Semantics::Native, kCC>(
        packed[l], Value<float, Semantics::Native, kCC>{1.0f});
    out[2 * l]     = r.get(0).to_float();
    out[2 * l + 1] = r.get(1).to_float();
}

TEST(CvtScalef32Fp4Wrap, FloatUnpackAllCodes)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    std::vector<unsigned> in(16);
    for(int c = 0; c < 16; ++c)
        in[c] = unsigned(c) | (unsigned(c) << 4);
    unsigned* dIn = to_dev(in);
    float*    dO;
    HIP_CHECK(hipMalloc(&dO, 32 * sizeof(float)));
    k_f32_unpack<<<1, 16>>>(dIn, dO);
    HIP_CHECK(hipDeviceSynchronize());
    auto got = from_dev(dO, 32);
    for(int c = 0; c < 16; ++c)
    {
        float ref = narrow_to_f32(unsigned(c), kFp4E2M1);
        EXPECT_EQ(got[2 * c], ref) << "code " << c << " elem0";
        EXPECT_EQ(got[2 * c + 1], ref) << "code " << c << " elem1";
    }
    (void)hipFree(dIn);
    (void)hipFree(dO);
}

// Pack pairs of representable fp4 values through the wrapper (Sel=0, scale=1)
// vs OCP host encode; also confirm Sel placement + `old` preservation.
template <int Sel>
__global__ void k_f32_pack(unsigned old, float a, float b, unsigned* out)
{
    *out = fpsan::amdgcn_cvt_scalef32_pk_fp4_f32<Sel, Semantics::Native, kCC>(
        old,
        Value<float, Semantics::Native, kCC>{a},
        Value<float, Semantics::Native, kCC>{b},
        Value<float, Semantics::Native, kCC>{1.0f});
}

TEST(CvtScalef32Fp4Wrap, FloatPackExactBitsAndSel)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    unsigned* dO;
    HIP_CHECK(hipMalloc(&dO, sizeof(unsigned)));
    const unsigned old = 0xA5A5A5A5u;
    for(int ca = 0; ca < 8; ++ca)
        for(int cb = 0; cb < 8; ++cb)
        {
            float    a = narrow_to_f32(unsigned(ca), kFp4E2M1);
            float    b = narrow_to_f32(unsigned(cb), kFp4E2M1);
            unsigned pair
                = (f32_to_narrow(a, kFp4E2M1) & 0xF) | ((f32_to_narrow(b, kFp4E2M1) & 0xF) << 4);
            k_f32_pack<0><<<1, 1>>>(old, a, b, dO);
            HIP_CHECK(hipDeviceSynchronize());
            unsigned got    = from_dev(dO, 1)[0];
            unsigned expect = (old & 0xFFFFFF00u) | pair;
            EXPECT_EQ(got, expect) << "a=" << ca << " b=" << cb;
        }
    // Sel places the pair in byte Sel, preserving the rest.
    float    a = narrow_to_f32(0x6u, kFp4E2M1), b = narrow_to_f32(0x7u, kFp4E2M1);
    unsigned pair = 0x6u | (0x7u << 4);
    auto     run  = [&](auto sel_tag, int sel) {
        k_f32_pack<decltype(sel_tag)::value><<<1, 1>>>(old, a, b, dO);
        HIP_CHECK(hipDeviceSynchronize());
        unsigned got  = from_dev(dO, 1)[0];
        unsigned mask = 0xFFu << (8 * sel);
        EXPECT_EQ(got, (old & ~mask) | (pair << (8 * sel))) << "sel " << sel;
    };
    run(std::integral_constant<int, 0>{}, 0);
    run(std::integral_constant<int, 1>{}, 1);
    run(std::integral_constant<int, 2>{}, 2);
    run(std::integral_constant<int, 3>{}, 3);
    (void)hipFree(dO);
}

// Scale direction through the wrapper: unpack multiplies, pack divides.
__global__ void k_f32_scale(
    unsigned packed, float scale, float* outUnpack, float a, float pscale, unsigned* outPack)
{
    auto r = fpsan::amdgcn_cvt_scalef32_pk_f32_fp4<0, Semantics::Native, kCC>(
        packed, Value<float, Semantics::Native, kCC>{scale});
    outUnpack[0] = r.get(0).to_float();
    *outPack     = fpsan::amdgcn_cvt_scalef32_pk_fp4_f32<0, Semantics::Native, kCC>(
        0u,
        Value<float, Semantics::Native, kCC>{a},
        Value<float, Semantics::Native, kCC>{0.0f},
        Value<float, Semantics::Native, kCC>{pscale});
}

TEST(CvtScalef32Fp4Wrap, FloatScaleDirection)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    float*    dU;
    unsigned* dP;
    HIP_CHECK(hipMalloc(&dU, sizeof(float)));
    HIP_CHECK(hipMalloc(&dP, sizeof(unsigned)));
    // packed nibble0 = code 2 (=1.0); unpack scale=4 -> 4.0.
    k_f32_scale<<<1, 1>>>(0x2u, 4.0f, dU, /*a=*/8.0f, /*pscale=*/2.0f, dP);
    HIP_CHECK(hipDeviceSynchronize());
    EXPECT_EQ(from_dev(dU, 1)[0], 4.0f); // 1.0 * 4
    // pack 8.0 / 2 = 4.0 -> code 6.
    EXPECT_EQ(from_dev(dP, 1)[0] & 0xF, unsigned(0x6));
    (void)hipFree(dU);
    (void)hipFree(dP);
}

// ============================ FPSan mode ====================================

// Unpack plumbing: packed holds fp4 payloads; widen sign-extends 4->32, then
// *scale. Reference is explicit sext + public Value multiply.
__global__ void k_fpsan_unpack(unsigned packed, float scale, unsigned* out)
{
    auto r = fpsan::amdgcn_cvt_scalef32_pk_f32_fp4<1, Semantics::Triton, kCC>(packed, VF{scale});
    out[0] = r.get(0).fpsan_payload();
    out[1] = r.get(1).fpsan_payload();
}

TEST(CvtScalef32Fp4Wrap, FpsanUnpackSelAndWiden)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    // Put nibble payloads 0x3 and 0xF (=-1 after sext) in byte Sel=1.
    unsigned    packed = (0x3u << 8) | (0xFu << 12);
    const float scale  = 2.0f;
    unsigned*   dO;
    HIP_CHECK(hipMalloc(&dO, 2 * sizeof(unsigned)));
    k_fpsan_unpack<<<1, 1>>>(packed, scale, dO);
    HIP_CHECK(hipDeviceSynchronize());
    auto     got  = from_dev(dO, 2);
    unsigned ref0 = (VF::from_fpsan_payload(static_cast<std::uint32_t>(sext4(0x3))) * VF{scale})
                        .fpsan_payload();
    unsigned ref1 = (VF::from_fpsan_payload(static_cast<std::uint32_t>(sext4(0xF))) * VF{scale})
                        .fpsan_payload();
    EXPECT_EQ(got[0], ref0);
    EXPECT_EQ(got[1], ref1);
    (void)hipFree(dO);
}

// Pack plumbing + divide-by-scale: nibble = low4(payload(a/scale)), placed at
// Sel, `old` preserved. Reference uses public Value division.
__global__ void k_fpsan_pack(unsigned old, float a, float b, float scale, unsigned* out)
{
    *out = fpsan::amdgcn_cvt_scalef32_pk_fp4_f32<2, Semantics::Triton, kCC>(
        old, VF{a}, VF{b}, VF{scale});
}

TEST(CvtScalef32Fp4Wrap, FpsanPackSelDivideAndPreserve)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    const unsigned old = 0xDEADBEEFu;
    std::mt19937   rng = fpsan_test::make_rng();
    unsigned*      dO;
    HIP_CHECK(hipMalloc(&dO, sizeof(unsigned)));
    for(int t = 0; t < 32; ++t)
    {
        float       a     = fpsan_test::pick_int_valued<float>(rng, -6, 6);
        float       b     = fpsan_test::pick_int_valued<float>(rng, -6, 6);
        const float scale = 2.0f;
        k_fpsan_pack<<<1, 1>>>(old, a, b, scale, dO);
        HIP_CHECK(hipDeviceSynchronize());
        unsigned got  = from_dev(dO, 1)[0];
        unsigned na   = (VF{a} / VF{scale}).fpsan_payload() & 0xF;
        unsigned nb   = (VF{b} / VF{scale}).fpsan_payload() & 0xF;
        unsigned pair = na | (nb << 4);
        unsigned mask = 0xFFu << (8 * 2);
        EXPECT_EQ(got, (old & ~mask) | (pair << (8 * 2))) << "a=" << a << " b=" << b;
    }
    (void)hipFree(dO);
}

// sr_pk_fp4_f32: exact (seed-invariant) packing matches the deterministic pack.
__global__ void k_fpsan_srpack(float a, float b, unsigned seed, unsigned* out)
{
    *out = fpsan::amdgcn_cvt_scalef32_sr_pk_fp4_f32<0, Semantics::Triton, kCC>(
        0u, VF{a}, VF{b}, seed, VF{1.0f});
}

TEST(CvtScalef32Fp4Wrap, FpsanSrPackMatchesDeterministic)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    unsigned* dO;
    HIP_CHECK(hipMalloc(&dO, sizeof(unsigned)));
    for(int ca = 0; ca < 8; ++ca)
        for(int cb = 0; cb < 8; ++cb)
        {
            float a = narrow_to_f32(unsigned(ca), kFp4E2M1);
            float b = narrow_to_f32(unsigned(cb), kFp4E2M1);
            k_fpsan_srpack<<<1, 1>>>(a, b, 0xdeadbeefu, dO);
            HIP_CHECK(hipDeviceSynchronize());
            unsigned got = from_dev(dO, 1)[0] & 0xFF;
            unsigned na  = VF{a}.fpsan_payload() & 0xF;
            unsigned nb  = VF{b}.fpsan_payload() & 0xF;
            EXPECT_EQ(got, na | (nb << 4)) << "a=" << ca << " b=" << cb;
        }
    (void)hipFree(dO);
}

// f16/bf16 unpack/pack: Float mode vs OCP host ref (round-trip through the f32
// reference), confirming the wrappers compile and are hardware-correct.
__global__ void k_f16_unpack(unsigned packed, float* out)
{
    auto r = fpsan::amdgcn_cvt_scalef32_pk_f16_fp4<0, Semantics::Native, kCC>(
        packed, Value<float, Semantics::Native, kCC>{1.0f});
    out[0] = static_cast<float>(static_cast<_Float16>(r.get(0)));
    out[1] = static_cast<float>(static_cast<_Float16>(r.get(1)));
}

TEST(CvtScalef32Fp4Wrap, FloatF16UnpackAllCodes)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    float* dO;
    HIP_CHECK(hipMalloc(&dO, 2 * sizeof(float)));
    for(int c = 0; c < 16; ++c)
    {
        unsigned packed = unsigned(c) | (unsigned(c) << 4);
        k_f16_unpack<<<1, 1>>>(packed, dO);
        HIP_CHECK(hipDeviceSynchronize());
        auto  got = from_dev(dO, 2);
        float ref = narrow_to_f32(unsigned(c), kFp4E2M1); // exact in f16
        EXPECT_EQ(got[0], ref) << "f16 code " << c;
        EXPECT_EQ(got[1], ref) << "f16 code " << c;
    }
    (void)hipFree(dO);
}

#endif // has builtin
