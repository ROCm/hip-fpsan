// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/cvt_sr_pk_f16_gfx1250_test.cpp
//
// GPU tests for the gfx1250 stochastic-rounding pack wrappers
// cvt_sr_pk_f16_f32 / cvt_sr_pk_bf16_f32 in fpsan/amdgcn_cvt.hpp. Each packs two
// f32 into a v2{f16,bf16} with SR from a u32 seed (lane0<-a, lane1<-b).
//
// Authoritative checks (host-referenced, independent of the device builtin):
//   Float, exact inputs : at f32 values exactly representable in the narrow
//                         type, SR is exact, so both lanes equal the host
//                         narrowing.
//   Float, SR property  : for an f32 exactly between two representable
//                         neighbors, every result must be ONE of those two
//                         neighbors (bracketed), and across many seeds BOTH must
//                         occur (the rounding is actually stochastic).
//   FPSan               : each lane equals fpsan::cast<{_Float16,__bf16}> of the
//                         input (deterministic payload truncate; seed opaque).
#include "fpsan/amdgcn_cvt.hpp"
#include "fpsan/fpsan.hpp"

#include "hip_test_utils.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <set>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

static constexpr Conversions kCC = Conversions::Explicit;
using FVF                        = Value<float, Semantics::Native, kCC>;
using VF                         = Value<float, Semantics::Triton, kCC>;

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_sr_pk_f16_f32)

// ----------------------------- f16 -----------------------------------------
__global__ void k_f16_exact(const float* a, const float* b, float* out)
{
    int  l = threadIdx.x;
    auto r = fpsan::amdgcn_cvt_sr_pk_f16_f32<Semantics::Native, kCC>(
        FVF{a[l]}, FVF{b[l]}, 0xABCDu + l);
    out[2 * l]     = static_cast<float>(static_cast<_Float16>(r.get(0)));
    out[2 * l + 1] = static_cast<float>(static_cast<_Float16>(r.get(1)));
}

TEST(CvtSrPkF16, FloatExactInputs)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    // Exactly representable in f16 (and f32).
    std::vector<float> a  = {0.0f, 1.0f, -2.0f, 0.5f, 3.0f, -0.25f, 16.0f, -1.5f};
    std::vector<float> b  = {1.0f, -1.0f, 2.0f, -0.5f, 4.0f, 0.125f, -8.0f, 6.0f};
    const int          n  = static_cast<int>(a.size());
    float *            dA = to_dev(a), *dB = to_dev(b), *dO;
    HIP_CHECK(hipMalloc(&dO, 2 * n * sizeof(float)));
    k_f16_exact<<<1, n>>>(dA, dB, dO);
    HIP_CHECK(hipDeviceSynchronize());
    auto got = from_dev(dO, 2 * n);
    for(int i = 0; i < n; ++i)
    {
        EXPECT_EQ(got[2 * i], static_cast<float>(static_cast<_Float16>(a[i]))) << "lane0 i=" << i;
        EXPECT_EQ(got[2 * i + 1], static_cast<float>(static_cast<_Float16>(b[i])))
            << "lane1 i=" << i;
    }
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dO);
}

// One f32 strictly between two f16 neighbors; each thread uses a distinct seed.
__global__ void k_f16_sr(float x, const unsigned* seeds, float* out)
{
    int  l = threadIdx.x;
    auto r = fpsan::amdgcn_cvt_sr_pk_f16_f32<Semantics::Native, kCC>(FVF{x}, FVF{x}, seeds[l]);
    out[l] = static_cast<float>(static_cast<_Float16>(r.get(0)));
}

TEST(CvtSrPkF16, FloatStochasticBracket)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    // f16 step near 1.0 is 2^-10; midpoint rounds to 1.0 or 1.0 + 2^-10.
    const float           lo = 1.0f;
    const float           hi = 1.0f + std::ldexp(1.0f, -10);
    const float           x  = 0.5f * (lo + hi);
    const int             N  = 256;
    std::vector<unsigned> seeds(N);
    for(int i = 0; i < N; ++i)
        seeds[i] = 0x10000u * (i + 1) + 0x9E3779B9u * i; // spread across the u32 range
    unsigned* dS = to_dev(seeds);
    float*    dO;
    HIP_CHECK(hipMalloc(&dO, N * sizeof(float)));
    k_f16_sr<<<1, N>>>(x, dS, dO);
    HIP_CHECK(hipDeviceSynchronize());
    auto            got = from_dev(dO, N);
    std::set<float> seen;
    for(int i = 0; i < N; ++i)
    {
        EXPECT_TRUE(got[i] == lo || got[i] == hi) << "seed#" << i << " gave " << got[i];
        seen.insert(got[i]);
    }
    EXPECT_TRUE(seen.count(lo) && seen.count(hi))
        << "rounding was not stochastic (got " << seen.size() << " distinct values)";
    (void)hipFree(dS);
    (void)hipFree(dO);
}

__global__ void k_f16_fpsan(const float* a, const float* b, unsigned* out)
{
    int  l = threadIdx.x;
    auto r = fpsan::amdgcn_cvt_sr_pk_f16_f32<Semantics::Triton, kCC>(VF{a[l]}, VF{b[l]}, 0x55u + l);
    out[2 * l]     = r.get(0).fpsan_payload();
    out[2 * l + 1] = r.get(1).fpsan_payload();
}

TEST(CvtSrPkF16, FpsanPayloadTruncate)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    std::vector<float> a  = {1.0f, -2.5f, 3.0f, -0.5f};
    std::vector<float> b  = {2.0f, 0.25f, -4.0f, 7.0f};
    const int          n  = static_cast<int>(a.size());
    float *            dA = to_dev(a), *dB = to_dev(b);
    unsigned*          dO;
    HIP_CHECK(hipMalloc(&dO, 2 * n * sizeof(unsigned)));
    k_f16_fpsan<<<1, n>>>(dA, dB, dO);
    HIP_CHECK(hipDeviceSynchronize());
    auto got = from_dev(dO, 2 * n);
    for(int i = 0; i < n; ++i)
    {
        EXPECT_EQ(got[2 * i], fpsan::cast<_Float16>(VF{a[i]}).fpsan_payload()) << "lane0 i=" << i;
        EXPECT_EQ(got[2 * i + 1], fpsan::cast<_Float16>(VF{b[i]}).fpsan_payload())
            << "lane1 i=" << i;
    }
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dO);
}

#endif // cvt_sr_pk_f16_f32

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_sr_pk_bf16_f32)

// ----------------------------- bf16 ----------------------------------------
__global__ void k_bf16_exact(const float* a, const float* b, float* out)
{
    int  l = threadIdx.x;
    auto r = fpsan::amdgcn_cvt_sr_pk_bf16_f32<Semantics::Native, kCC>(
        FVF{a[l]}, FVF{b[l]}, 0xABCDu + l);
    out[2 * l]     = static_cast<float>(static_cast<__bf16>(r.get(0)));
    out[2 * l + 1] = static_cast<float>(static_cast<__bf16>(r.get(1)));
}

TEST(CvtSrPkBf16, FloatExactInputs)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    std::vector<float> a  = {0.0f, 1.0f, -2.0f, 0.5f, 3.0f, -0.25f, 16.0f, -1.5f};
    std::vector<float> b  = {1.0f, -1.0f, 2.0f, -0.5f, 4.0f, 0.125f, -8.0f, 6.0f};
    const int          n  = static_cast<int>(a.size());
    float *            dA = to_dev(a), *dB = to_dev(b), *dO;
    HIP_CHECK(hipMalloc(&dO, 2 * n * sizeof(float)));
    k_bf16_exact<<<1, n>>>(dA, dB, dO);
    HIP_CHECK(hipDeviceSynchronize());
    auto got = from_dev(dO, 2 * n);
    for(int i = 0; i < n; ++i)
    {
        EXPECT_EQ(got[2 * i], static_cast<float>(static_cast<__bf16>(a[i]))) << "lane0 i=" << i;
        EXPECT_EQ(got[2 * i + 1], static_cast<float>(static_cast<__bf16>(b[i]))) << "lane1 i=" << i;
    }
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dO);
}

__global__ void k_bf16_sr(float x, const unsigned* seeds, float* out)
{
    int  l = threadIdx.x;
    auto r = fpsan::amdgcn_cvt_sr_pk_bf16_f32<Semantics::Native, kCC>(FVF{x}, FVF{x}, seeds[l]);
    out[l] = static_cast<float>(static_cast<__bf16>(r.get(0)));
}

TEST(CvtSrPkBf16, FloatStochasticBracket)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    // bf16 step near 1.0 is 2^-7; midpoint rounds to 1.0 or 1.0 + 2^-7.
    const float           lo = 1.0f;
    const float           hi = 1.0f + std::ldexp(1.0f, -7);
    const float           x  = 0.5f * (lo + hi);
    const int             N  = 256;
    std::vector<unsigned> seeds(N);
    for(int i = 0; i < N; ++i)
        seeds[i] = 0x10000u * (i + 1) + 0x9E3779B9u * i;
    unsigned* dS = to_dev(seeds);
    float*    dO;
    HIP_CHECK(hipMalloc(&dO, N * sizeof(float)));
    k_bf16_sr<<<1, N>>>(x, dS, dO);
    HIP_CHECK(hipDeviceSynchronize());
    auto            got = from_dev(dO, N);
    std::set<float> seen;
    for(int i = 0; i < N; ++i)
    {
        EXPECT_TRUE(got[i] == lo || got[i] == hi) << "seed#" << i << " gave " << got[i];
        seen.insert(got[i]);
    }
    EXPECT_TRUE(seen.count(lo) && seen.count(hi)) << "rounding was not stochastic";
    (void)hipFree(dS);
    (void)hipFree(dO);
}

__global__ void k_bf16_fpsan(const float* a, const float* b, unsigned* out)
{
    int  l = threadIdx.x;
    auto r
        = fpsan::amdgcn_cvt_sr_pk_bf16_f32<Semantics::Triton, kCC>(VF{a[l]}, VF{b[l]}, 0x55u + l);
    out[2 * l]     = r.get(0).fpsan_payload();
    out[2 * l + 1] = r.get(1).fpsan_payload();
}

TEST(CvtSrPkBf16, FpsanPayloadTruncate)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    std::vector<float> a  = {1.0f, -2.5f, 3.0f, -0.5f};
    std::vector<float> b  = {2.0f, 0.25f, -4.0f, 7.0f};
    const int          n  = static_cast<int>(a.size());
    float *            dA = to_dev(a), *dB = to_dev(b);
    unsigned*          dO;
    HIP_CHECK(hipMalloc(&dO, 2 * n * sizeof(unsigned)));
    k_bf16_fpsan<<<1, n>>>(dA, dB, dO);
    HIP_CHECK(hipDeviceSynchronize());
    auto got = from_dev(dO, 2 * n);
    for(int i = 0; i < n; ++i)
    {
        EXPECT_EQ(got[2 * i], fpsan::cast<__bf16>(VF{a[i]}).fpsan_payload()) << "lane0 i=" << i;
        EXPECT_EQ(got[2 * i + 1], fpsan::cast<__bf16>(VF{b[i]}).fpsan_payload()) << "lane1 i=" << i;
    }
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dO);
}

#endif // cvt_sr_pk_bf16_f32
