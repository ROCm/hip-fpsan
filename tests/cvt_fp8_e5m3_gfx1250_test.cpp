// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/cvt_fp8_e5m3_gfx1250_test.cpp
//
// GPU tests for the gfx1250 E5M3 fp8 conversion wrappers in
// fpsan/amdgcn_cvt.hpp (cvt_f32_fp8_e5m3, cvt_pk_fp8_f32_e5m3,
// cvt_sr_fp8_f32_e5m3). Registered by CMake only for the gfx1250 tier.
//
// E5M3 is gfx1250's UNSIGNED 8-bit scale format (5 exp bits bias 15, 3 mantissa
// bits, no sign, no inf, NaN == 0xFF, zero == 0x00, denormals). It is not in
// LLVM APFloat, so the authoritative reference is the host codec
// detail::kFp8E5M3 (a clean realization of that format) PLUS its explicit
// special-value table, asserted directly here. Float-mode references are
// host-computed and INDEPENDENT of the device builtin, so any implementation
// that diverges from them fails. FPSan-mode references use the deterministic
// width-8 payload resize.
#include "fpsan/amdgcn_cvt.hpp"
#include "fpsan/fpsan.hpp"

#include "hip_test_utils.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;
using fpsan::detail::f32_to_narrow;
using fpsan::detail::kFp8E5M3;
using fpsan::detail::narrow_to_f32;

static constexpr Conversions kCC = Conversions::Explicit;
using VF                         = Value<float, Semantics::Triton, kCC>;

namespace
{
    std::int32_t sext8(std::uint32_t b)
    {
        return static_cast<std::int32_t>(b << 24) >> 24;
    }
} // namespace

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_f32_fp8_e5m3)

// ===========================================================================
// 0. Host codec matches the gfx1250 E5M3 special-value table exactly.
// This locks the host reference itself to the known E5M3 special values before it
// is used to judge the device.
// ===========================================================================
TEST(CvtFp8E5M3, HostCodecMatchesGuideTable)
{
    EXPECT_EQ(narrow_to_f32(0x00, kFp8E5M3), 0.0f); // zero
    EXPECT_EQ(narrow_to_f32(0x08, kFp8E5M3), std::ldexp(1.0f, -14)); // min normal 2^-14
    EXPECT_EQ(narrow_to_f32(0xFE, kFp8E5M3), 114688.0f); // max normal
    EXPECT_EQ(narrow_to_f32(0x07, kFp8E5M3), 0.875f * std::ldexp(1.0f, -14)); // max denorm
    EXPECT_EQ(narrow_to_f32(0x01, kFp8E5M3), std::ldexp(1.0f, -17)); // min denorm
    EXPECT_TRUE(std::isnan(narrow_to_f32(0xFF, kFp8E5M3))); // NaN
    // No infinities: the byte just below NaN is the finite max normal.
    EXPECT_TRUE(std::isfinite(narrow_to_f32(0xFE, kFp8E5M3)));
}

// ===========================================================================
// 1. Decode: cvt_f32_fp8_e5m3 over ALL 256 byte patterns at byte0, vs host.
// ===========================================================================
__global__ void k_e5m3_decode_all(const int* packed, float* out)
{
    int l  = threadIdx.x;
    out[l] = fpsan::amdgcn_cvt_f32_fp8_e5m3<0, Semantics::Native, kCC>(packed[l]).to_float();
}

TEST(CvtFp8E5M3, DecodeAllBytes)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    std::vector<int> packed(256);
    for(int b = 0; b < 256; ++b)
        packed[b] = b;
    int*   dIn = to_dev(packed);
    float* dO;
    HIP_CHECK(hipMalloc(&dO, 256 * sizeof(float)));
    k_e5m3_decode_all<<<1, 256>>>(dIn, dO);
    HIP_CHECK(hipDeviceSynchronize());
    auto got = from_dev(dO, 256);
    for(int b = 0; b < 256; ++b)
    {
        float ref = narrow_to_f32(static_cast<std::uint32_t>(b), kFp8E5M3);
        if(std::isnan(ref))
            EXPECT_TRUE(std::isnan(got[b])) << "byte 0x" << std::hex << b;
        else
            EXPECT_EQ(got[b], ref) << "byte 0x" << std::hex << b;
    }
    (void)hipFree(dIn);
    (void)hipFree(dO);
}

// Byte-select decode: place a finite E5M3 byte at byte ByteIdx, decode<ByteIdx>.
template <int ByteIdx>
__global__ void k_e5m3_decode_bytesel(const int* packed, float* out)
{
    int l  = threadIdx.x;
    out[l] = fpsan::amdgcn_cvt_f32_fp8_e5m3<ByteIdx, Semantics::Native, kCC>(packed[l]).to_float();
}

template <int ByteIdx>
void run_decode_bytesel()
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    std::vector<int> packed(255);
    for(int b = 0; b < 255; ++b) // 0x00..0xFE finite
        packed[b] = static_cast<int>(static_cast<std::uint32_t>(b) << (ByteIdx * 8));
    int*   dIn = to_dev(packed);
    float* dO;
    HIP_CHECK(hipMalloc(&dO, 255 * sizeof(float)));
    k_e5m3_decode_bytesel<ByteIdx><<<1, 255>>>(dIn, dO);
    HIP_CHECK(hipDeviceSynchronize());
    auto got = from_dev(dO, 255);
    for(int b = 0; b < 255; ++b)
        EXPECT_EQ(got[b], narrow_to_f32(static_cast<std::uint32_t>(b), kFp8E5M3))
            << "byteIdx " << ByteIdx << " byte 0x" << std::hex << b;
    (void)hipFree(dIn);
    (void)hipFree(dO);
}

TEST(CvtFp8E5M3, DecodeByteSel)
{
    run_decode_bytesel<1>();
    run_decode_bytesel<2>();
    run_decode_bytesel<3>();
}

// ===========================================================================
// 2. Pack: cvt_pk_fp8_f32_e5m3 -- every finite byte pair round-trips through
// host decode -> pack -> expected bytes, with `old` preserved and DstLo half
// selection. This authoritatively checks the encoder + byte placement.
// ===========================================================================
template <bool DstLo>
__global__ void k_e5m3_pack(int old, float a, float b, int* out)
{
    *out = fpsan::amdgcn_cvt_pk_fp8_f32_e5m3<DstLo, Semantics::Native, kCC>(
        Value<float, Semantics::Native, kCC>{a}, Value<float, Semantics::Native, kCC>{b}, old);
}

template <bool DstLo>
void run_pack()
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    const int old = static_cast<int>(0xA5A5A5A5u);
    int*      dO;
    HIP_CHECK(hipMalloc(&dO, sizeof(int)));
    // A representative spread of finite E5M3 bytes (denormals, normals, max).
    const int bytes[] = {0x00, 0x01, 0x07, 0x08, 0x40, 0x78, 0x80, 0xC0, 0xFE};
    for(int ca : bytes)
        for(int cb : bytes)
        {
            float a = narrow_to_f32(static_cast<std::uint32_t>(ca), kFp8E5M3);
            float b = narrow_to_f32(static_cast<std::uint32_t>(cb), kFp8E5M3);
            k_e5m3_pack<DstLo><<<1, 1>>>(old, a, b, dO);
            HIP_CHECK(hipDeviceSynchronize());
            std::uint32_t got = static_cast<std::uint32_t>(from_dev(dO, 1)[0]);
            std::uint32_t ea  = f32_to_narrow(a, kFp8E5M3) & 0xFFu;
            std::uint32_t eb  = f32_to_narrow(b, kFp8E5M3) & 0xFFu;
            std::uint32_t pair, want;
            if(DstLo)
            {
                pair = ea | (eb << 8);
                want = (static_cast<std::uint32_t>(old) & 0xFFFF0000u) | pair;
            }
            else
            {
                pair = (ea << 16) | (eb << 24);
                want = (static_cast<std::uint32_t>(old) & 0x0000FFFFu) | pair;
            }
            EXPECT_EQ(got, want) << "DstLo=" << DstLo << " ca=0x" << std::hex << ca << " cb=0x"
                                 << cb;
        }
    (void)hipFree(dO);
}

TEST(CvtFp8E5M3, PackLowHalf)
{
    run_pack<true>();
}
TEST(CvtFp8E5M3, PackHighHalf)
{
    run_pack<false>();
}

// Pack -> decode round trip is the identity for representable values.
__global__ void k_e5m3_roundtrip(const int* bytes, float* out)
{
    int   l  = threadIdx.x;
    float v  = fpsan::amdgcn_cvt_f32_fp8_e5m3<0, Semantics::Native, kCC>(bytes[l]).to_float();
    int   pk = fpsan::amdgcn_cvt_pk_fp8_f32_e5m3<true, Semantics::Native, kCC>(
        Value<float, Semantics::Native, kCC>{v}, Value<float, Semantics::Native, kCC>{0.0f}, 0);
    out[l] = fpsan::amdgcn_cvt_f32_fp8_e5m3<0, Semantics::Native, kCC>(pk).to_float();
}

TEST(CvtFp8E5M3, PackDecodeRoundTrip)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    std::vector<int> bytes(255);
    for(int b = 0; b < 255; ++b)
        bytes[b] = b;
    int*   dIn = to_dev(bytes);
    float* dO;
    HIP_CHECK(hipMalloc(&dO, 255 * sizeof(float)));
    k_e5m3_roundtrip<<<1, 255>>>(dIn, dO);
    HIP_CHECK(hipDeviceSynchronize());
    auto got = from_dev(dO, 255);
    for(int b = 0; b < 255; ++b)
        EXPECT_EQ(got[b], narrow_to_f32(static_cast<std::uint32_t>(b), kFp8E5M3))
            << "round-trip byte 0x" << std::hex << b;
    (void)hipFree(dIn);
    (void)hipFree(dO);
}

// ===========================================================================
// 3. SR pack: at exactly-representable inputs stochastic rounding is exact, so
// the byte equals the host encode regardless of seed.
// ===========================================================================
template <int ByteIdx>
__global__ void k_e5m3_sr(int old, float v, unsigned seed, int* out)
{
    *out = fpsan::amdgcn_cvt_sr_fp8_f32_e5m3<ByteIdx, Semantics::Native, kCC>(
        Value<float, Semantics::Native, kCC>{v}, old, seed);
}

TEST(CvtFp8E5M3, SrPackExact)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    const int old = static_cast<int>(0xDEADBEEFu);
    int*      dO;
    HIP_CHECK(hipMalloc(&dO, sizeof(int)));
    for(int b = 0; b < 255; ++b)
    {
        float v = narrow_to_f32(static_cast<std::uint32_t>(b), kFp8E5M3);
        k_e5m3_sr<2><<<1, 1>>>(old, v, 0x1234u + b, dO);
        HIP_CHECK(hipDeviceSynchronize());
        std::uint32_t got  = static_cast<std::uint32_t>(from_dev(dO, 1)[0]);
        std::uint32_t enc  = f32_to_narrow(v, kFp8E5M3) & 0xFFu;
        std::uint32_t want = (static_cast<std::uint32_t>(old) & ~(0xFFu << 16)) | (enc << 16);
        EXPECT_EQ(got, want) << "byte 0x" << std::hex << b;
    }
    (void)hipFree(dO);
}

// ===========================================================================
// 4. FPSan mode: deterministic width-8 payload resize (decode = sext8, pack =
// low 8 bits), independent of the device builtin.
// ===========================================================================
__global__ void k_e5m3_fpsan_decode(const int* packed, unsigned* out)
{
    int l  = threadIdx.x;
    out[l] = fpsan::amdgcn_cvt_f32_fp8_e5m3<0, Semantics::Triton, kCC>(packed[l]).fpsan_payload();
}

TEST(CvtFp8E5M3, FpsanDecodeWiden8)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    std::vector<int> packed(256);
    for(int b = 0; b < 256; ++b)
        packed[b] = b;
    int*      dIn = to_dev(packed);
    unsigned* dO;
    HIP_CHECK(hipMalloc(&dO, 256 * sizeof(unsigned)));
    k_e5m3_fpsan_decode<<<1, 256>>>(dIn, dO);
    HIP_CHECK(hipDeviceSynchronize());
    auto got = from_dev(dO, 256);
    for(int b = 0; b < 256; ++b)
        EXPECT_EQ(got[b], static_cast<unsigned>(sext8(static_cast<std::uint32_t>(b))))
            << "byte 0x" << std::hex << b;
    (void)hipFree(dIn);
    (void)hipFree(dO);
}

__global__ void k_e5m3_fpsan_pack(int old, float a, float b, int* out)
{
    *out = fpsan::amdgcn_cvt_pk_fp8_f32_e5m3<true, Semantics::Triton, kCC>(VF{a}, VF{b}, old);
}

TEST(CvtFp8E5M3, FpsanPackLow8)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    const int    old = static_cast<int>(0x12345678u);
    std::mt19937 rng = std::mt19937(0xc0ffeeu);
    int*         dO;
    HIP_CHECK(hipMalloc(&dO, sizeof(int)));
    std::uniform_int_distribution<int> dist(-200, 200);
    for(int t = 0; t < 64; ++t)
    {
        float a = static_cast<float>(dist(rng));
        float b = static_cast<float>(dist(rng));
        k_e5m3_fpsan_pack<<<1, 1>>>(old, a, b, dO);
        HIP_CHECK(hipDeviceSynchronize());
        std::uint32_t got  = static_cast<std::uint32_t>(from_dev(dO, 1)[0]);
        std::uint32_t ea   = VF{a}.fpsan_payload() & 0xFFu;
        std::uint32_t eb   = VF{b}.fpsan_payload() & 0xFFu;
        std::uint32_t want = (static_cast<std::uint32_t>(old) & 0xFFFF0000u) | ea | (eb << 8);
        EXPECT_EQ(got, want) << "a=" << a << " b=" << b;
    }
    (void)hipFree(dO);
}

#endif // has builtin
