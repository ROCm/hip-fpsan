// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/cvt_scalef32_mx_gfx950_test.cpp
//
// gfx950 conformance tests for the MX sub-byte scaled conversions
// (cvt_scalef32_* with fp4 e2m1 / fp6 e2m3 / bf6 e3m2 operands).
//
// Methodology mirrors the rest of the gfx950 suite: every expected value is a
// host-side reference computed by fpsan's generic narrow<->f32 conversion
// (detail::narrow_to_f32 / f32_to_narrow, the IREE-derived OCP math), which is
// INDEPENDENT of the device builtin. So any implementation that diverges from
// the host reference shows up as a failure here.
//
// We exercise the raw __builtin_amdgcn_cvt_scalef32_* directly (in fpsan's
// Float mode these wrappers are exactly the builtin) because the FPSan
// payload-ring dataflow for sub-byte packed fragments is a separate, larger
// build-out; for validating the conversion *instructions* the builtin-vs-OCP
// comparison is the relevant property.
//
// Coverage: fp4/fp6/bf6 widen (decode) over EVERY bit pattern; narrow (encode)
// exact-bit packing incl. the 192-bit 6-DWORD layout; stochastic-rounding pack;
// and the fp4 sr_pk reversed-operand path the PR specifically calls out.
#include "fpsan/detail/fp8.hpp"

#include "hip_test_utils.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

namespace
{

    using fpsan::detail::f32_to_narrow;
    using fpsan::detail::kBf6E3M2;
    using fpsan::detail::kFp4E2M1;
    using fpsan::detail::kFp6E2M3;
    using fpsan::detail::narrow_to_f32;

    // ---- host packing helpers for the canonical contiguous little-endian layouts.
    // fp4: 8 nibbles per u32. fp6/bf6: 6-bit fields packed LE across DWORDs.
    unsigned pack_fp4_8(const std::uint8_t codes[8])
    {
        unsigned out = 0;
        for(int i = 0; i < 8; ++i)
            out |= unsigned(codes[i] & 0xF) << (4 * i);
        return out;
    }
    void pack6(const std::uint8_t* codes, int n, unsigned* out, int words)
    {
        for(int i = 0; i < words; ++i)
            out[i] = 0;
        for(int i = 0; i < n; ++i)
            for(int b = 0; b < 6; ++b)
                if((codes[i] >> b) & 1)
                {
                    int p = i * 6 + b;
                    out[p / 32] |= 1u << (p % 32);
                }
    }
    // 2xpk16 packs two 16-element groups INTERLEAVED at 6-bit granularity: read as
    // a contiguous 6-bit stream, field 2k is lo[k] and field 2k+1 is hi[k]
    // (verified against real gfx950). lo/hi are the two v16 inputs.
    void pack6_2xpk16(const std::uint8_t* lo, const std::uint8_t* hi, unsigned* out)
    {
        std::uint8_t inter[32];
        for(int k = 0; k < 16; ++k)
        {
            inter[2 * k]     = lo[k];
            inter[2 * k + 1] = hi[k];
        }
        pack6(inter, 32, out, 6);
    }
    using v2f  = float __attribute__((ext_vector_type(2)));
    using v16f = float __attribute__((ext_vector_type(16)));
    using v32f = float __attribute__((ext_vector_type(32)));
    using v6u  = unsigned __attribute__((ext_vector_type(6)));

} // namespace

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_scalef32_pk32_f32_fp6)

// =====================  FP4 e2m1  ============================================

// Widen: pk_f32_fp4 extracts nibble-pair `opsel` of a u32 -> 2 f32 * scale.
__global__ void k_fp4_widen(const unsigned* packed, float* out, float scale)
{
    int l          = threadIdx.x;
    v2f r          = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(packed[l], scale, 0);
    out[2 * l]     = r[0];
    out[2 * l + 1] = r[1];
}

TEST(CvtScalef32Fp4, WidenAllCodes)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    // 16 lanes; lane c carries fp4 code c in nibbles 0 and 1.
    std::vector<unsigned> in(16);
    for(int c = 0; c < 16; ++c)
        in[c] = unsigned(c) | (unsigned(c) << 4);
    unsigned* dIn = to_dev(in);
    float*    dO;
    HIP_CHECK(hipMalloc(&dO, 32 * sizeof(float)));
    k_fp4_widen<<<1, 16>>>(dIn, dO, 1.0f);
    HIP_CHECK(hipDeviceSynchronize());
    auto got = from_dev(dO, 32);
    for(int c = 0; c < 16; ++c)
    {
        float ref = narrow_to_f32(unsigned(c), kFp4E2M1);
        EXPECT_EQ(got[2 * c], ref) << "fp4 code " << c << " elem0";
        EXPECT_EQ(got[2 * c + 1], ref) << "fp4 code " << c << " elem1";
    }
    (void)hipFree(dIn);
    (void)hipFree(dO);
}

// opsel must select the right nibble pair (0..3 -> nibbles {0,1}..{6,7}).
__global__ void k_fp4_opsel(unsigned packed, float* out)
{
    out[0] = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(packed, 1.0f, 0)[0];
    out[1] = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(packed, 1.0f, 1)[0];
    out[2] = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(packed, 1.0f, 2)[0];
    out[3] = __builtin_amdgcn_cvt_scalef32_pk_f32_fp4(packed, 1.0f, 3)[0];
}

TEST(CvtScalef32Fp4, WidenOpselSelectsPair)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    std::uint8_t codes[8] = {0x1, 0x2, 0x3, 0x4, 0x5, 0x6, 0x7, 0x0};
    unsigned     packed   = pack_fp4_8(codes);
    float*       dO;
    HIP_CHECK(hipMalloc(&dO, 4 * sizeof(float)));
    k_fp4_opsel<<<1, 1>>>(packed, dO);
    HIP_CHECK(hipDeviceSynchronize());
    auto got = from_dev(dO, 4);
    for(int p = 0; p < 4; ++p)
        EXPECT_EQ(got[p], narrow_to_f32(codes[2 * p], kFp4E2M1)) << "opsel " << p;
    (void)hipFree(dO);
}

// Narrow: pk_fp4_f32 packs 2 f32 -> the nibble pair selected by opsel of a u32
// (read-modify-write of `old`), with RNE.
template <int Opsel>
__global__ void k_fp4_narrow(unsigned old, float a, float b, unsigned* out, float scale)
{
    *out = __builtin_amdgcn_cvt_scalef32_pk_fp4_f32(old, a, b, scale, Opsel);
}

TEST(CvtScalef32Fp4, NarrowExactBits)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    unsigned* dO;
    HIP_CHECK(hipMalloc(&dO, sizeof(unsigned)));
    // Every ordered pair of the 8 positive fp4 values into nibble pair 0.
    for(int ca = 0; ca < 8; ++ca)
        for(int cb = 0; cb < 8; ++cb)
        {
            float a = narrow_to_f32(unsigned(ca), kFp4E2M1);
            float b = narrow_to_f32(unsigned(cb), kFp4E2M1);
            k_fp4_narrow<0><<<1, 1>>>(0u, a, b, dO, 1.0f);
            HIP_CHECK(hipDeviceSynchronize());
            unsigned got = from_dev(dO, 1)[0];
            unsigned expect
                = (f32_to_narrow(a, kFp4E2M1) & 0xF) | ((f32_to_narrow(b, kFp4E2M1) & 0xF) << 4);
            EXPECT_EQ(got & 0xFF, expect) << "a=" << ca << " b=" << cb;
        }
    (void)hipFree(dO);
}

TEST(CvtScalef32Fp4, NarrowOpselPlacesPairAndPreservesOld)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    unsigned* dO;
    HIP_CHECK(hipMalloc(&dO, sizeof(unsigned)));
    const unsigned old   = 0xABCDEF12u;
    float          a     = narrow_to_f32(0x6u, kFp4E2M1); // code 6 -> 4.0
    float          b     = narrow_to_f32(0x7u, kFp4E2M1); // code 7 -> 6.0
    const unsigned pair  = 0x6u | (0x7u << 4);
    auto           check = [&](int opsel, unsigned got) {
        unsigned mask   = 0xFFu << (8 * opsel);
        unsigned expect = (old & ~mask) | (pair << (8 * opsel));
        EXPECT_EQ(got, expect) << "opsel " << opsel;
    };
    k_fp4_narrow<0><<<1, 1>>>(old, a, b, dO, 1.0f);
    HIP_CHECK(hipDeviceSynchronize());
    check(0, from_dev(dO, 1)[0]);
    k_fp4_narrow<1><<<1, 1>>>(old, a, b, dO, 1.0f);
    HIP_CHECK(hipDeviceSynchronize());
    check(1, from_dev(dO, 1)[0]);
    k_fp4_narrow<2><<<1, 1>>>(old, a, b, dO, 1.0f);
    HIP_CHECK(hipDeviceSynchronize());
    check(2, from_dev(dO, 1)[0]);
    k_fp4_narrow<3><<<1, 1>>>(old, a, b, dO, 1.0f);
    HIP_CHECK(hipDeviceSynchronize());
    check(3, from_dev(dO, 1)[0]);
    (void)hipFree(dO);
}

// sr_pk_fp4_f32: the PR flags a reversed-operand quirk (src0=random,src1=data).
// Round-trip exact values through it: a swapped data/seed corrupts the codes.
__global__ void k_fp4_sr_pk(float a, float b, unsigned* out, unsigned seed)
{
    v2f v;
    v[0] = a;
    v[1] = b;
    *out = __builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f32(0u, v, seed, 1.0f, 0);
}

TEST(CvtScalef32Fp4, SrPkReversedOperandExact)
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
            k_fp4_sr_pk<<<1, 1>>>(a, b, dO, 0xdeadbeefu);
            HIP_CHECK(hipDeviceSynchronize());
            unsigned got = from_dev(dO, 1)[0];
            unsigned expect
                = (f32_to_narrow(a, kFp4E2M1) & 0xF) | ((f32_to_narrow(b, kFp4E2M1) & 0xF) << 4);
            EXPECT_EQ(got & 0xFF, expect) << "a=" << ca << " b=" << cb;
        }
    (void)hipFree(dO);
}

// =====================  FP6 e2m3 / BF6 e3m2  =================================

__global__ void k_fp6_widen(const unsigned* in6, float* out, float scale)
{
    v6u src;
    for(int i = 0; i < 6; ++i)
        src[i] = in6[i];
    v32f r = __builtin_amdgcn_cvt_scalef32_pk32_f32_fp6(src, scale);
    for(int i = 0; i < 32; ++i)
        out[i] = r[i];
}
__global__ void k_bf6_widen(const unsigned* in6, float* out, float scale)
{
    v6u src;
    for(int i = 0; i < 6; ++i)
        src[i] = in6[i];
    v32f r = __builtin_amdgcn_cvt_scalef32_pk32_f32_bf6(src, scale);
    for(int i = 0; i < 32; ++i)
        out[i] = r[i];
}

// Widen all 64 codes (two packs of 32) against the host OCP decode. This is the
// 192-bit 6-DWORD unpack path the PR calls out.
template <class Kern>
void widen_all_codes(Kern kern, const fpsan::detail::FpFormat& fmt, const char* name)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    float* dO;
    HIP_CHECK(hipMalloc(&dO, 32 * sizeof(float)));
    for(int base = 0; base < 64; base += 32)
    {
        std::uint8_t codes[32];
        for(int i = 0; i < 32; ++i)
            codes[i] = std::uint8_t(base + i);
        unsigned packed[6];
        pack6(codes, 32, packed, 6);
        std::vector<unsigned> hp(packed, packed + 6);
        unsigned*             dIn = to_dev(hp);
        kern<<<1, 1>>>(dIn, dO, 1.0f);
        HIP_CHECK(hipDeviceSynchronize());
        auto got = from_dev(dO, 32);
        for(int i = 0; i < 32; ++i)
            EXPECT_EQ(got[i], narrow_to_f32(unsigned(codes[i]), fmt))
                << name << " code " << int(codes[i]);
        (void)hipFree(dIn);
    }
    (void)hipFree(dO);
}

TEST(CvtScalef32Fp6, WidenAllCodes)
{
    widen_all_codes(k_fp6_widen, kFp6E2M3, "fp6");
}
TEST(CvtScalef32Bf6, WidenAllCodes)
{
    widen_all_codes(k_bf6_widen, kBf6E3M2, "bf6");
}

// Narrow f32 -> fp6 packed (2xpk16_fp6_f32): two 16-lane groups -> the 192-bit
// LE pack. Check the packed bits directly against the host packing of the
// host-encoded codes.
__global__ void k_fp6_narrow(const float* in, unsigned* out, float scale)
{
    v16f lo, hi;
    for(int i = 0; i < 16; ++i)
    {
        lo[i] = in[i];
        hi[i] = in[16 + i];
    }
    v6u r = __builtin_amdgcn_cvt_scalef32_2xpk16_fp6_f32(lo, hi, scale);
    for(int i = 0; i < 6; ++i)
        out[i] = r[i];
}

TEST(CvtScalef32Fp6, NarrowExactBits)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    // The 32 positive fp6 values -> expect codes 0..31, host-packed LE.
    std::vector<float> in(32);
    for(int c = 0; c < 32; ++c)
        in[c] = narrow_to_f32(unsigned(c), kFp6E2M3);
    float*    dIn = to_dev(in);
    unsigned* dO;
    HIP_CHECK(hipMalloc(&dO, 6 * sizeof(unsigned)));
    k_fp6_narrow<<<1, 1>>>(dIn, dO, 1.0f);
    HIP_CHECK(hipDeviceSynchronize());
    auto         got = from_dev(dO, 6);
    std::uint8_t lo[16], hi[16];
    for(int k = 0; k < 16; ++k)
    {
        lo[k] = f32_to_narrow(in[k], kFp6E2M3) & 0x3F;
        hi[k] = f32_to_narrow(in[16 + k], kFp6E2M3) & 0x3F;
    }
    unsigned expect[6];
    pack6_2xpk16(lo, hi, expect);
    for(int i = 0; i < 6; ++i)
        EXPECT_EQ(got[i], expect[i]) << "fp6 narrow word " << i;
    (void)hipFree(dIn);
    (void)hipFree(dO);
}

// bf6 narrow uses the gfx950 2xpk16 form (two 16-lane groups -> 192 bits).
__global__ void k_bf6_narrow(const float* in, unsigned* out, float scale)
{
    v16f lo, hi;
    for(int i = 0; i < 16; ++i)
    {
        lo[i] = in[i];
        hi[i] = in[16 + i];
    }
    v6u r = __builtin_amdgcn_cvt_scalef32_2xpk16_bf6_f32(lo, hi, scale);
    for(int i = 0; i < 6; ++i)
        out[i] = r[i];
}

TEST(CvtScalef32Bf6, NarrowExactBits)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    std::vector<float> in(32);
    for(int c = 0; c < 32; ++c)
        in[c] = narrow_to_f32(unsigned(c), kBf6E3M2);
    float*    dIn = to_dev(in);
    unsigned* dO;
    HIP_CHECK(hipMalloc(&dO, 6 * sizeof(unsigned)));
    k_bf6_narrow<<<1, 1>>>(dIn, dO, 1.0f);
    HIP_CHECK(hipDeviceSynchronize());
    auto         got = from_dev(dO, 6);
    std::uint8_t lo[16], hi[16];
    for(int k = 0; k < 16; ++k)
    {
        lo[k] = f32_to_narrow(in[k], kBf6E3M2) & 0x3F;
        hi[k] = f32_to_narrow(in[16 + k], kBf6E3M2) & 0x3F;
    }
    unsigned expect[6];
    pack6_2xpk16(lo, hi, expect);
    for(int i = 0; i < 6; ++i)
        EXPECT_EQ(got[i], expect[i]) << "bf6 narrow word " << i;
    (void)hipFree(dIn);
    (void)hipFree(dO);
}

// Stochastic-rounding fp6 narrow (sr_pk32_fp6_f32): exact inputs are
// seed-invariant. Validates the SR path + the 192-bit pack together.
__global__ void k_fp6_sr_narrow(const float* in, unsigned* out, unsigned seed)
{
    v32f v;
    for(int i = 0; i < 32; ++i)
        v[i] = in[i];
    v6u r = __builtin_amdgcn_cvt_scalef32_sr_pk32_fp6_f32(v, seed, 1.0f);
    for(int i = 0; i < 6; ++i)
        out[i] = r[i];
}

TEST(CvtScalef32Fp6, SrNarrowExactIsSeedInvariant)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    std::vector<float> in(32);
    for(int c = 0; c < 32; ++c)
        in[c] = narrow_to_f32(unsigned(c), kFp6E2M3);
    float*    dIn = to_dev(in);
    unsigned* dO;
    HIP_CHECK(hipMalloc(&dO, 6 * sizeof(unsigned)));
    std::uint8_t codes[32];
    for(int c = 0; c < 32; ++c)
        codes[c] = f32_to_narrow(in[c], kFp6E2M3) & 0x3F;
    unsigned expect[6];
    pack6(codes, 32, expect, 6);
    for(unsigned seed : {0u, 0xffffffffu, 0x12345678u})
    {
        k_fp6_sr_narrow<<<1, 1>>>(dIn, dO, seed);
        HIP_CHECK(hipDeviceSynchronize());
        auto got = from_dev(dO, 6);
        for(int i = 0; i < 6; ++i)
            EXPECT_EQ(got[i], expect[i]) << "seed " << std::hex << seed << " w " << i;
    }
    (void)hipFree(dIn);
    (void)hipFree(dO);
}

// Round-trip narrow->widen identity. sr_pk32 narrow and pk32 widen share the
// contiguous 192-bit layout, so they are exact inverses for representable
// values (SR never rounds an exact input). Catches any narrow/widen layout or
// value disagreement directly, with no host packing assumption.
TEST(CvtScalef32Fp6, RoundTripNarrowWiden)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    std::vector<float> in(32);
    for(int c = 0; c < 32; ++c)
        in[c] = narrow_to_f32(unsigned(c), kFp6E2M3);
    float*    dIn = to_dev(in);
    unsigned* dPacked;
    float*    dO;
    HIP_CHECK(hipMalloc(&dPacked, 6 * sizeof(unsigned)));
    HIP_CHECK(hipMalloc(&dO, 32 * sizeof(float)));
    k_fp6_sr_narrow<<<1, 1>>>(dIn, dPacked, 0u);
    HIP_CHECK(hipDeviceSynchronize());
    k_fp6_widen<<<1, 1>>>(dPacked, dO, 1.0f);
    HIP_CHECK(hipDeviceSynchronize());
    auto got = from_dev(dO, 32);
    for(int c = 0; c < 32; ++c)
        EXPECT_EQ(got[c], in[c]) << "fp6 round-trip " << c;
    (void)hipFree(dIn);
    (void)hipFree(dPacked);
    (void)hipFree(dO);
}

// Scale multiplier: widen with scale=4 yields value*4 (block-scale model).
TEST(CvtScalef32Fp6, WidenScaleMultiplies)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    std::uint8_t codes[32];
    for(int i = 0; i < 32; ++i)
        codes[i] = std::uint8_t(i);
    unsigned packed[6];
    pack6(codes, 32, packed, 6);
    std::vector<unsigned> hp(packed, packed + 6);
    unsigned*             dIn = to_dev(hp);
    float*                dO;
    HIP_CHECK(hipMalloc(&dO, 32 * sizeof(float)));
    k_fp6_widen<<<1, 1>>>(dIn, dO, 4.0f);
    HIP_CHECK(hipDeviceSynchronize());
    auto got = from_dev(dO, 32);
    for(int c = 0; c < 32; ++c)
        EXPECT_EQ(got[c], narrow_to_f32(unsigned(c), kFp6E2M3) * 4.0f) << "code " << c;
    (void)hipFree(dIn);
    (void)hipFree(dO);
}

#endif // has builtins
