// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/hip_device_test.cpp
//
// Runs Value arithmetic on the GPU and checks that device results match
// the host bit-for-bit (the payload algebra is constexpr host/device, so they
// must agree exactly), plus that FPSan associativity holds on device. Only
// built/registered when FPSAN_ENABLE_HIP is ON.
#include "fpsan/fpsan.hpp"

#include "hip_test_utils.hpp"
#include "test_random.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using F = fpsan::Value<float, fpsan::Semantics::Triton, fpsan::Conversions::Explicit>;

__global__ void fpsan_kernel(const float*   a,
                             const float*   b,
                             const float*   c,
                             std::uint32_t* embed_payload,
                             std::uint32_t* assoc_ok,
                             std::uint32_t* exp_hom_ok,
                             int            n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;
    F A(a[i]), B(b[i]), C(c[i]);
    embed_payload[i] = A.fpsan_payload();
    assoc_ok[i]      = ((A + B) + C == A + (B + C)) ? 1u : 0u;
    exp_hom_ok[i]    = (fpsan::exp(A + B) == fpsan::exp(A) * fpsan::exp(B)) ? 1u : 0u;
}

TEST(HipDevice, MatchesHostAndPreservesIdentities)
{
    int        ndev = 0;
    hipError_t e    = hipGetDeviceCount(&ndev);
    if(e != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device available";

    // Deterministic pseudo-random inputs (see test_random.hpp). The
    // device-vs-host and identity checks hold for any finite values; quarters
    // give broad, exactly-representable coverage.
    const int          n = 64;
    std::vector<float> a(n), b(n), c(n);
    std::mt19937       rng = fpsan_test::make_rng();
    for(int i = 0; i < n; ++i)
    {
        a[i] = fpsan_test::pick_quarter<float>(rng, -36, 36); // -9 .. 9
        b[i] = fpsan_test::pick_quarter<float>(rng, -36, 36);
        c[i] = fpsan_test::pick_quarter<float>(rng, -36, 36);
    }

    float *        da, *db, *dc;
    std::uint32_t *dembed, *dassoc, *dexp;
    HIP_CHECK(hipMalloc(&da, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&db, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&dc, n * sizeof(float)));
    HIP_CHECK(hipMalloc(&dembed, n * sizeof(std::uint32_t)));
    HIP_CHECK(hipMalloc(&dassoc, n * sizeof(std::uint32_t)));
    HIP_CHECK(hipMalloc(&dexp, n * sizeof(std::uint32_t)));
    HIP_CHECK(hipMemcpy(da, a.data(), n * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(db, b.data(), n * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dc, c.data(), n * sizeof(float), hipMemcpyHostToDevice));

    fpsan_kernel<<<dim3((n + 63) / 64), dim3(64)>>>(da, db, dc, dembed, dassoc, dexp, n);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<std::uint32_t> embed(n), assoc(n), exph(n);
    HIP_CHECK(hipMemcpy(embed.data(), dembed, n * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(assoc.data(), dassoc, n * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(exph.data(), dexp, n * sizeof(std::uint32_t), hipMemcpyDeviceToHost));

    for(int i = 0; i < n; ++i)
    {
        EXPECT_EQ(embed[i], F(a[i]).fpsan_payload()) << "device embed != host @" << i;
        EXPECT_EQ(assoc[i], 1u) << "associativity failed on device @" << i;
        EXPECT_EQ(exph[i], 1u) << "exp homomorphism failed on device @" << i;
    }

    (void)hipFree(da);
    (void)hipFree(db);
    (void)hipFree(dc);
    (void)hipFree(dembed);
    (void)hipFree(dassoc);
    (void)hipFree(dexp);
}
