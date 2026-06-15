// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/wmma_16x16x4_gfx1250_test.cpp
//
// Deep tests for the gfx1250 16x16x4 WMMA FPSan wrapper (K=4, f32 operands).
// Same two-property structure as the other gfx1250 WMMA suites:
//
//  (1) Layout + dataflow vs real hardware. The software MMA run with real-float
//      arithmetic (Semantics::Native dataflow) must match the real builtin
//      bit-for-bit on EXACT-integer inputs. This validates the K=4 fragment
//      layout (Wmma16x16x4Layout, v2f per lane) + cross-lane gather against the
//      hardware ABI.
//
//  (2) Payload algebra. The shipped FPSan path must match an independent scalar
//      FPSan reference matmul computed on the host, payload for payload.
//
// Requires gfx1250 hardware; built only under the gfx1250 test tier.
#include "fpsan/amdgcn_matrix.hpp"
#include "fpsan/fpsan.hpp"

#include "hip_test_utils.hpp"
#include "test_random.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::v2f_native;
using fpsan::v8f_native;
using fpsan::Value;

static constexpr int         M = 16, N = 16, K = 4;
static constexpr Conversions kCC = Conversions::Explicit;

// Inverse of Wmma16x16x4Layout: given lane and v2 fragment slot idx, the logical
// K-index this slot holds. Forward map: index = k&1; lane = row + 16*((k>>1)&1).
__device__ inline int frag_k_inv4(int lane, int idx)
{
    int b0 = idx & 1; // index == k&1
    int b1 = (lane >> 4) & 1; // lane half == (k>>1)&1
    return b0 | (b1 << 1);
}

template <Semantics S>
__device__ void load_frags(const float*               A,
                           const float*               B,
                           const float*               C,
                           int                        lane,
                           Value<v2f_native, S, kCC>& a,
                           Value<v2f_native, S, kCC>& b,
                           Value<v8f_native, S, kCC>& c)
{
    v2f_native an, bn;
    v8f_native cn;
    for(int idx = 0; idx < 2; ++idx)
    {
        int k   = frag_k_inv4(lane, idx);
        an[idx] = A[(lane & 15) * K + k]; // A[m=lane&15][k]
        bn[idx] = B[k * N + (lane & 15)]; // B[k][n=lane&15]
    }
    for(int e = 0; e < 8; ++e)
        cn[e] = C[(e + 8 * (lane >> 4)) * N + (lane & 15)];
    a = Value<v2f_native, S, kCC>(an);
    b = Value<v2f_native, S, kCC>(bn);
    c = Value<v8f_native, S, kCC>(cn);
}

__global__ void k_builtin(const float* A, const float* B, const float* C, float* D)
{
    int                                       lane = threadIdx.x;
    Value<v2f_native, Semantics::Native, kCC> a;
    Value<v2f_native, Semantics::Native, kCC> b;
    Value<v8f_native, Semantics::Native, kCC> c;
    load_frags<Semantics::Native>(A, B, C, lane, a, b, c);
    auto d = fpsan::amdgcn_wmma_f32_16x16x4_f32<Semantics::Native, kCC>(a, b, c);
    for(int e = 0; e < 8; ++e)
        D[(e + 8 * (lane >> 4)) * N + (lane & 15)] = d.get(e).to_float();
}

__global__ void k_float_dataflow(const float* A, const float* B, const float* C, float* D)
{
    int                                       lane = threadIdx.x;
    Value<v2f_native, Semantics::Native, kCC> a;
    Value<v2f_native, Semantics::Native, kCC> b;
    Value<v8f_native, Semantics::Native, kCC> c;
    load_frags<Semantics::Native>(A, B, C, lane, a, b, c);
    auto d = fpsan::detail::wmma_16x16x4_dataflow(a, b, c);
    for(int e = 0; e < 8; ++e)
        D[(e + 8 * (lane >> 4)) * N + (lane & 15)] = d.get(e).to_float();
}

__global__ void k_fpsan(const float* A, const float* B, const float* C, std::uint32_t* Dpay)
{
    int                                       lane = threadIdx.x;
    Value<v2f_native, Semantics::Triton, kCC> a;
    Value<v2f_native, Semantics::Triton, kCC> b;
    Value<v8f_native, Semantics::Triton, kCC> c;
    load_frags<Semantics::Triton>(A, B, C, lane, a, b, c);
    auto d = fpsan::amdgcn_wmma_f32_16x16x4_f32<Semantics::Triton, kCC>(a, b, c);
    for(int e = 0; e < 8; ++e)
        Dpay[(e + 8 * (lane >> 4)) * N + (lane & 15)] = d.get(e).fpsan_payload();
}

namespace
{
    struct Mats
    {
        std::vector<float> A, B, C;
    };
    Mats make_inputs()
    {
        Mats m;
        m.A.resize(M * K);
        m.B.resize(K * N);
        m.C.resize(M * N);
        std::mt19937 rng = fpsan_test::make_rng();
        for(auto& x : m.A)
            x = fpsan_test::pick_int_valued<float>(rng, -4, 4);
        for(auto& x : m.B)
            x = fpsan_test::pick_int_valued<float>(rng, -4, 4);
        for(auto& x : m.C)
            x = fpsan_test::pick_int_valued<float>(rng, -8, 8);
        return m;
    }
} // namespace

TEST(WmmaF32F32_4, LayoutMatchesHardware)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    Mats   m  = make_inputs();
    float* dA = to_dev(m.A);
    float* dB = to_dev(m.B);
    float* dC = to_dev(m.C);
    float *dHw, *dOurs;
    HIP_CHECK(hipMalloc(&dHw, M * N * sizeof(float)));
    HIP_CHECK(hipMalloc(&dOurs, M * N * sizeof(float)));
    k_builtin<<<1, 32>>>(dA, dB, dC, dHw);
    k_float_dataflow<<<1, 32>>>(dA, dB, dC, dOurs);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> hw(M * N), ours(M * N);
    HIP_CHECK(hipMemcpy(hw.data(), dHw, M * N * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(ours.data(), dOurs, M * N * sizeof(float), hipMemcpyDeviceToHost));
    for(int i = 0; i < M * N; ++i)
        EXPECT_EQ(bits_of(hw[i]), bits_of(ours[i])) << "mismatch at " << (i / N) << "," << (i % N);
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dHw);
    (void)hipFree(dOurs);
}

TEST(WmmaF32F32_4, FpsanMatchesScalarReference)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    Mats m = make_inputs();

    using VF = Value<float, Semantics::Triton, kCC>;
    std::vector<std::uint32_t> ref(M * N);
    for(int mm = 0; mm < M; ++mm)
        for(int nn = 0; nn < N; ++nn)
        {
            VF acc(m.C[mm * N + nn]);
            for(int k = 0; k < K; ++k)
                acc = acc + VF(m.A[mm * K + k]) * VF(m.B[k * N + nn]);
            ref[mm * N + nn] = acc.fpsan_payload();
        }

    float*         dA = to_dev(m.A);
    float*         dB = to_dev(m.B);
    float*         dC = to_dev(m.C);
    std::uint32_t* dD;
    HIP_CHECK(hipMalloc(&dD, M * N * sizeof(std::uint32_t)));
    k_fpsan<<<1, 32>>>(dA, dB, dC, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(M * N);
    HIP_CHECK(hipMemcpy(got.data(), dD, M * N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for(int i = 0; i < M * N; ++i)
        EXPECT_EQ(got[i], ref[i]) << "payload mismatch at " << (i / N) << "," << (i % N);
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
}
