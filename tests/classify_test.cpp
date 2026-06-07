// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/classify_test.cpp
//
// GPU tests for amdgcn_classify.hpp. Float and FPSan modes both classify the
// represented float, so the per-lane bool / ballot mask must be identical.
#include "fpsan/amdgcn_classify.hpp"
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

static constexpr Conversions kCC = Conversions::Explicit;

// Mask 0x3FF = all categories ON; classf returns true for any normal value.
__global__ void k_classf_pair(const float* in, char* bf, char* bp)
{
    int                                 i = threadIdx.x;
    Value<float, Semantics::Float, kCC> vf{in[i]};
    Value<float, Semantics::FPSan, kCC> vp{in[i]};
    bf[i] = fpsan::amdgcn_classf<Semantics::Float, kCC>(vf, 0x3FF) ? 1 : 0;
    bp[i] = fpsan::amdgcn_classf<Semantics::FPSan, kCC>(vp, 0x3FF) ? 1 : 0;
}

TEST(Classify, ClassfFloatAndFpsanAgree)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    std::vector<float> in(32);
    std::mt19937       rng = fpsan_test::make_rng();
    for(auto& x : in)
        x = fpsan_test::pick_quarter<float>(rng, -20, 20); // -5 .. 5
    float* dIn;
    HIP_CHECK(hipMalloc(&dIn, 32 * sizeof(float)));
    HIP_CHECK(hipMemcpy(dIn, in.data(), 32 * sizeof(float), hipMemcpyHostToDevice));
    char *dBf, *dBp;
    HIP_CHECK(hipMalloc(&dBf, 32));
    HIP_CHECK(hipMalloc(&dBp, 32));
    k_classf_pair<<<1, 32>>>(dIn, dBf, dBp);
    HIP_CHECK(hipDeviceSynchronize());
    char bf[32], bp[32];
    HIP_CHECK(hipMemcpy(bf, dBf, 32, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(bp, dBp, 32, hipMemcpyDeviceToHost));
    for(int i = 0; i < 32; ++i)
    {
        // Float and FPSan must agree with each other AND with an independent host
        // reference: classf(v, 0x3FF) returns true for any value matching any of
        // the 10 IEEE categories, which is true for every finite/inf/nan -- so
        // for finite inputs (which our quarters are) it should be true.
        EXPECT_EQ(int(bf[i]), int(bp[i])) << "lane " << i;
        EXPECT_EQ(int(bf[i]), 1) << "lane " << i << " host ref: any finite";
    }
    (void)hipFree(dIn);
    (void)hipFree(dBf);
    (void)hipFree(dBp);
}

__global__ void k_fcmpf_pair(const float* a, const float* b, std::uint64_t* mf, std::uint64_t* mp)
{
    int                                 i = threadIdx.x;
    Value<float, Semantics::Float, kCC> af{a[i]}, bf{b[i]};
    Value<float, Semantics::FPSan, kCC> ap{a[i]}, bp{b[i]};
    // Predicate 1 = OEQ (ordered equal); see LLVM fcmp predicates.
    if(i == 0)
    {
        *mf = fpsan::amdgcn_fcmpf<1, Semantics::Float, kCC>(af, bf);
        *mp = fpsan::amdgcn_fcmpf<1, Semantics::FPSan, kCC>(ap, bp);
    }
}

TEST(Classify, FcmpfFloatAndFpsanAgree)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    std::vector<float> a(32), b(32);
    // Half identical, half different so the mask is nontrivial.
    for(int i = 0; i < 32; ++i)
    {
        a[i] = static_cast<float>(i % 4);
        b[i] = static_cast<float>(i % 5);
    }
    float *dA, *dB;
    HIP_CHECK(hipMalloc(&dA, 32 * sizeof(float)));
    HIP_CHECK(hipMalloc(&dB, 32 * sizeof(float)));
    HIP_CHECK(hipMemcpy(dA, a.data(), 32 * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, b.data(), 32 * sizeof(float), hipMemcpyHostToDevice));
    std::uint64_t *dMf, *dMp;
    HIP_CHECK(hipMalloc(&dMf, sizeof(std::uint64_t)));
    HIP_CHECK(hipMalloc(&dMp, sizeof(std::uint64_t)));
    k_fcmpf_pair<<<1, 32>>>(dA, dB, dMf, dMp);
    HIP_CHECK(hipDeviceSynchronize());
    std::uint64_t mf = 0, mp = 0;
    HIP_CHECK(hipMemcpy(&mf, dMf, sizeof mf, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&mp, dMp, sizeof mp, hipMemcpyDeviceToHost));
    EXPECT_EQ(mf, mp);
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dMf);
    (void)hipFree(dMp);
}
