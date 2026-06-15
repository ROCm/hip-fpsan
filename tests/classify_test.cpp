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
#ifndef FPSAN_TEST_FORCE_WAVE_SIZE
#define FPSAN_TEST_FORCE_WAVE_SIZE 32
#endif
static_assert(FPSAN_TEST_FORCE_WAVE_SIZE == 32 || FPSAN_TEST_FORCE_WAVE_SIZE == 64,
              "classify tests support one wave32 or wave64 wave");
static constexpr int LANES = FPSAN_TEST_FORCE_WAVE_SIZE;

// Mask 0x3FF = all categories ON; classf returns true for any normal value.
__global__ void k_classf_pair(const float* in, char* bf, char* bp)
{
    int                                  i = threadIdx.x;
    Value<float, Semantics::Native, kCC> vf{in[i]};
    Value<float, Semantics::Triton, kCC> vp{in[i]};
    bf[i] = fpsan::amdgcn_classf<Semantics::Native, kCC>(vf, 0x3FF) ? 1 : 0;
    bp[i] = fpsan::amdgcn_classf<Semantics::Triton, kCC>(vp, 0x3FF) ? 1 : 0;
}

TEST(Classify, ClassfFloatAndFpsanAgree)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    std::vector<float> in(LANES);
    std::mt19937       rng = fpsan_test::make_rng();
    for(auto& x : in)
        x = fpsan_test::pick_quarter<float>(rng, -20, 20); // -5 .. 5
    float* dIn;
    HIP_CHECK(hipMalloc(&dIn, LANES * sizeof(float)));
    HIP_CHECK(hipMemcpy(dIn, in.data(), LANES * sizeof(float), hipMemcpyHostToDevice));
    char *dBf, *dBp;
    HIP_CHECK(hipMalloc(&dBf, LANES));
    HIP_CHECK(hipMalloc(&dBp, LANES));
    k_classf_pair<<<1, LANES>>>(dIn, dBf, dBp);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<char> bf(LANES), bp(LANES);
    HIP_CHECK(hipMemcpy(bf.data(), dBf, LANES, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(bp.data(), dBp, LANES, hipMemcpyDeviceToHost));
    for(int i = 0; i < LANES; ++i)
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
    int                                  i = threadIdx.x;
    Value<float, Semantics::Native, kCC> af{a[i]}, bf{b[i]};
    Value<float, Semantics::Triton, kCC> ap{a[i]}, bp{b[i]};
    // Predicate 1 = OEQ (ordered equal); see LLVM fcmp predicates.
    std::uint64_t f = fpsan::amdgcn_fcmpf<1, Semantics::Native, kCC>(af, bf);
    std::uint64_t p = fpsan::amdgcn_fcmpf<1, Semantics::Triton, kCC>(ap, bp);
    if(i == 0)
    {
        *mf = f;
        *mp = p;
    }
}

TEST(Classify, FcmpfFloatAndFpsanAgree)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    std::vector<float> a(LANES), b(LANES);
    // Half identical, half different so the mask is nontrivial.
    std::uint64_t expected = 0;
    for(int i = 0; i < LANES; ++i)
    {
        a[i] = static_cast<float>(i % 4);
        b[i] = static_cast<float>(i % 5);
        if(a[i] == b[i])
            expected |= (1ull << i);
    }
    float *dA, *dB;
    HIP_CHECK(hipMalloc(&dA, LANES * sizeof(float)));
    HIP_CHECK(hipMalloc(&dB, LANES * sizeof(float)));
    HIP_CHECK(hipMemcpy(dA, a.data(), LANES * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, b.data(), LANES * sizeof(float), hipMemcpyHostToDevice));
    std::uint64_t *dMf, *dMp;
    HIP_CHECK(hipMalloc(&dMf, sizeof(std::uint64_t)));
    HIP_CHECK(hipMalloc(&dMp, sizeof(std::uint64_t)));
    k_fcmpf_pair<<<1, LANES>>>(dA, dB, dMf, dMp);
    HIP_CHECK(hipDeviceSynchronize());
    std::uint64_t mf = 0, mp = 0;
    HIP_CHECK(hipMemcpy(&mf, dMf, sizeof mf, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(&mp, dMp, sizeof mp, hipMemcpyDeviceToHost));
    EXPECT_EQ(mf, mp);
    EXPECT_EQ(mf, expected);
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dMf);
    (void)hipFree(dMp);
}
