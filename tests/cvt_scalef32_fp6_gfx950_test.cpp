// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/cvt_scalef32_fp6_gfx950_test.cpp
//
// gfx950 tests for the fpsan:: fp6 (e2m3) / bf6 (e3m2) scaled-conversion
// WRAPPERS in fpsan/amdgcn_cvt.hpp (pk32_* unpack; 2xpk16_*_f32 + pk32_*_f16/
// bf16 + sr_pk32_*_f32 pack). Float vs the INDEPENDENT host OCP reference;
// FPSan vs an independent payload reference that re-packs/unpacks the 192-bit
// 6-DWORD stream with host bit ops (cross-checking the device extract6/insert6
// against a separate implementation).
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
using fpsan::detail::kBf6E3M2;
using fpsan::detail::kFp6E2M3;
using fpsan::detail::narrow_to_f32;

static constexpr Conversions kCC = Conversions::Explicit;
using VF                         = Value<float, Semantics::Triton, kCC>;
using v6u                        = unsigned __attribute__((ext_vector_type(6)));

namespace
{
    // Independent host bitstream helpers (separate impl from the device side).
    void host_pack6(const std::uint8_t* codes, int n, unsigned* out)
    {
        for(int i = 0; i < 6; ++i)
            out[i] = 0;
        for(int i = 0; i < n; ++i)
            for(int b = 0; b < 6; ++b)
                if((codes[i] >> b) & 1)
                {
                    int p = i * 6 + b;
                    out[p / 32] |= 1u << (p % 32);
                }
    }
    std::uint8_t host_extract6(const unsigned* w, int i)
    {
        std::uint8_t v = 0;
        for(int b = 0; b < 6; ++b)
        {
            int p = i * 6 + b;
            if((w[p / 32] >> (p % 32)) & 1)
                v |= 1u << b;
        }
        return v;
    }
    std::int32_t sext6(std::uint32_t f)
    {
        return static_cast<std::int32_t>(f << 26) >> 26;
    }
} // namespace

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_scalef32_pk32_f32_fp6)

// ============================ Float mode ====================================

__global__ void k_f32_unpack_fp6(const unsigned* in6, float* out, float scale)
{
    v6u s;
    for(int i = 0; i < 6; ++i)
        s[i] = in6[i];
    auto r = fpsan::amdgcn_cvt_scalef32_pk32_f32_fp6<Semantics::Native, kCC>(
        s, Value<float, Semantics::Native, kCC>{scale});
    for(int i = 0; i < 32; ++i)
        out[i] = r.get(i).to_float();
}
__global__ void k_f32_unpack_bf6(const unsigned* in6, float* out, float scale)
{
    v6u s;
    for(int i = 0; i < 6; ++i)
        s[i] = in6[i];
    auto r = fpsan::amdgcn_cvt_scalef32_pk32_f32_bf6<Semantics::Native, kCC>(
        s, Value<float, Semantics::Native, kCC>{scale});
    for(int i = 0; i < 32; ++i)
        out[i] = r.get(i).to_float();
}

template <class Kern>
void float_unpack_all(Kern kern, const fpsan::detail::FpFormat& fmt, const char* name)
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
        host_pack6(codes, 32, packed);
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

TEST(CvtScalef32Fp6Wrap, FloatUnpackFp6AllCodes)
{
    float_unpack_all(k_f32_unpack_fp6, kFp6E2M3, "fp6");
}
TEST(CvtScalef32Fp6Wrap, FloatUnpackBf6AllCodes)
{
    float_unpack_all(k_f32_unpack_bf6, kBf6E3M2, "bf6");
}

__global__ void k_f32_pack_fp6(const float* in, unsigned* out, float scale)
{
    fpsan::v16f_native_cvt lo, hi;
    for(int i = 0; i < 16; ++i)
    {
        lo[i] = in[i];
        hi[i] = in[16 + i];
    }
    v6u r = fpsan::amdgcn_cvt_scalef32_2xpk16_fp6_f32<Semantics::Native, kCC>(
        Value<fpsan::v16f_native_cvt, Semantics::Native, kCC>{lo},
        Value<fpsan::v16f_native_cvt, Semantics::Native, kCC>{hi},
        Value<float, Semantics::Native, kCC>{scale});
    for(int i = 0; i < 6; ++i)
        out[i] = r[i];
}

TEST(CvtScalef32Fp6Wrap, FloatPack2xpk16ExactBits)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    std::vector<float> in(32);
    for(int c = 0; c < 32; ++c)
        in[c] = narrow_to_f32(unsigned(c), kFp6E2M3);
    float*    dIn = to_dev(in);
    unsigned* dO;
    HIP_CHECK(hipMalloc(&dO, 6 * sizeof(unsigned)));
    k_f32_pack_fp6<<<1, 1>>>(dIn, dO, 1.0f);
    HIP_CHECK(hipDeviceSynchronize());
    auto got = from_dev(dO, 6);
    // Expected: interleaved codes, field 2k=lo[k], 2k+1=hi[k].
    std::uint8_t inter[32];
    for(int k = 0; k < 16; ++k)
    {
        inter[2 * k]     = f32_to_narrow(in[k], kFp6E2M3) & 0x3F;
        inter[2 * k + 1] = f32_to_narrow(in[16 + k], kFp6E2M3) & 0x3F;
    }
    unsigned expect[6];
    host_pack6(inter, 32, expect);
    for(int i = 0; i < 6; ++i)
        EXPECT_EQ(got[i], expect[i]) << "word " << i;
    (void)hipFree(dIn);
    (void)hipFree(dO);
}

// ============================ FPSan mode ====================================

// Unpack: packed v6u32 holds 6-bit payloads (host-packed independently); wrapper
// widens (sext 6->32) and multiplies by scale.
__global__ void k_fpsan_unpack(const unsigned* in6, unsigned* out, float scale)
{
    v6u s;
    for(int i = 0; i < 6; ++i)
        s[i] = in6[i];
    auto r = fpsan::amdgcn_cvt_scalef32_pk32_f32_fp6<Semantics::Triton, kCC>(s, VF{scale});
    for(int i = 0; i < 32; ++i)
        out[i] = r.get(i).fpsan_payload();
}

TEST(CvtScalef32Fp6Wrap, FpsanUnpackWidenAndScale)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    std::uint8_t payloads[32];
    std::mt19937 rng = fpsan_test::make_rng();
    for(int i = 0; i < 32; ++i)
        payloads[i] = std::uint8_t(rng() & 0x3F);
    unsigned packed[6];
    host_pack6(payloads, 32, packed);
    std::vector<unsigned> hp(packed, packed + 6);
    unsigned*             dIn = to_dev(hp);
    unsigned*             dO;
    HIP_CHECK(hipMalloc(&dO, 32 * sizeof(unsigned)));
    const float scale = 2.0f;
    k_fpsan_unpack<<<1, 1>>>(dIn, dO, scale);
    HIP_CHECK(hipDeviceSynchronize());
    auto got = from_dev(dO, 32);
    for(int i = 0; i < 32; ++i)
    {
        unsigned ref
            = (VF::from_fpsan_payload(static_cast<std::uint32_t>(sext6(payloads[i]))) * VF{scale})
                  .fpsan_payload();
        EXPECT_EQ(got[i], ref) << "elem " << i;
    }
    (void)hipFree(dIn);
    (void)hipFree(dO);
}

// Pack (2xpk16): wrapper divides by scale, narrows to 6 bits, interleaves.
// Decode the result with the independent host_extract6 and compare to the
// public Value division payload truncated to 6 bits.
__global__ void k_fpsan_pack(const float* in, unsigned* out, float scale)
{
    fpsan::v16f_native_cvt lo, hi;
    for(int i = 0; i < 16; ++i)
    {
        lo[i] = in[i];
        hi[i] = in[16 + i];
    }
    v6u r = fpsan::amdgcn_cvt_scalef32_2xpk16_fp6_f32<Semantics::Triton, kCC>(
        Value<fpsan::v16f_native_cvt, Semantics::Triton, kCC>{lo},
        Value<fpsan::v16f_native_cvt, Semantics::Triton, kCC>{hi},
        VF{scale});
    for(int i = 0; i < 6; ++i)
        out[i] = r[i];
}

TEST(CvtScalef32Fp6Wrap, FpsanPackDivideInterleave)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    std::vector<float> in(32);
    std::mt19937       rng = fpsan_test::make_rng();
    for(auto& x : in)
        x = fpsan_test::pick_int_valued<float>(rng, -7, 7);
    const float scale = 2.0f;
    float*      dIn   = to_dev(in);
    unsigned*   dO;
    HIP_CHECK(hipMalloc(&dO, 6 * sizeof(unsigned)));
    k_fpsan_pack<<<1, 1>>>(dIn, dO, scale);
    HIP_CHECK(hipDeviceSynchronize());
    auto got = from_dev(dO, 6);
    for(int k = 0; k < 16; ++k)
    {
        unsigned refLo = (VF{in[k]} / VF{scale}).fpsan_payload() & 0x3F;
        unsigned refHi = (VF{in[16 + k]} / VF{scale}).fpsan_payload() & 0x3F;
        EXPECT_EQ(host_extract6(got.data(), 2 * k), refLo) << "lo " << k;
        EXPECT_EQ(host_extract6(got.data(), 2 * k + 1), refHi) << "hi " << k;
    }
    (void)hipFree(dIn);
    (void)hipFree(dO);
}

#endif // has builtin
