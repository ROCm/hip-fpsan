// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/wmma_gfx12_test.cpp
//
// Deep tests for the gfx12 (RDNA4 / gfx1250) 16x16x16 WMMA FPSan wrappers. Two
// independent properties, run for every variant:
//
//  (1) Layout + dataflow vs the real hardware. Our reverse-engineered software
//      MMA, run with *real float* arithmetic (Semantics::Native dataflow), must
//      match the real builtin bit-for-bit on EXACT-integer inputs (so FP
//      rounding / accumulation order can't mask a layout error). This validates
//      the fragment layout maps + cross-lane gather against the hardware.
//
//  (2) Payload algebra. The shipped FPSan path must match an independent scalar
//      FPSan reference matmul (computed with the library on the host), payload
//      for payload. The maps cancel here (used consistently on both ends), so
//      this isolates the reduction/gather math -- the layout is covered by (1).
//
// The two share the same maps, so together they certify each FPSan WMMA. A
// templated harness (Harness<Traits>) drives both tests; each intrinsic is
// described by a small Traits struct (10 lines). Adding a variant therefore
// adds ~15 lines, not ~150.
//
// Requires a gfx1200/gfx1201 GPU (built only under FPSAN_ENABLE_HIP).
#include "fpsan/amdgcn_matrix.hpp"
#include "fpsan/fpsan.hpp"

#include "hip_test_utils.hpp"
#include "test_random.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::v4bf_native;
using fpsan::v4e4m3_native;
using fpsan::v4e5m2_native;
using fpsan::v4f_native;
using fpsan::v4h_native;
using fpsan::v8bf_native;
using fpsan::v8f_native;
using fpsan::v8h_native;
using fpsan::Value;

static constexpr int         M = 16, N = 16, K = 16;
static constexpr Conversions kCC = Conversions::Explicit;

#if defined(FPSAN_TEST_GFX12_WAVE64)
static constexpr int kWaveSize = 64;
#else
static constexpr int kWaveSize = 32;
#endif

// Inverse of Wmma16x16x16Layout's k(lane, idx) for the A/B fragments: given the
// lane and the fragment slot idx, which logical K-index does it hold.
__device__ inline int frag_k_inv(int lane, int idx)
{
#if defined(FPSAN_TEST_GFX12_WAVE64)
    int reg = idx >> 1, half = idx & 1;
    return half | ((reg & 1) << 1) | (((lane >> 4) & 1) << 2) | (((lane >> 5) & 1) << 3);
#else
    int reg = idx >> 1, half = idx & 1;
    return half | ((reg & 1) << 1) | ((lane >> 4) << 2) | ((reg >> 1) << 3);
#endif
}

__host__ __device__ inline int cd_m_from_lane(int lane, int e)
{
#if defined(FPSAN_TEST_GFX12_WAVE64)
    return e + 8 * ((lane >> 4) & 1) + 4 * (lane >> 5);
#else
    return e + 8 * (lane >> 4);
#endif
}

// ---------------------------------------------------------------------------
// Templated harness shared by every 16x16x16 WMMA variant. A Traits struct
// declares the three native fragment vector types (AVec/BVec/CVec) and a
// type+semantics-generic call() that dispatches to the right fpsan:: wrapper:
//
//   struct WmmaF32F16Traits {
//     using AVec = v8h_native;
//     using BVec = v8h_native;
//     using CVec = v8f_native;
//     static constexpr int a_lo=-3, a_hi=3, b_lo=-2, b_hi=2, c_lo=-4, c_hi=4;
//     template <Semantics S, Conversions C>
//     __device__ static Value<CVec,S,C> call(...) {
//       return fpsan::amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
//     }
//   };
// ---------------------------------------------------------------------------
template <class Traits>
struct Harness
{
    using AVec                   = typename Traits::AVec;
    using BVec                   = typename Traits::BVec;
    using CVec                   = typename Traits::CVec;
    using AElem                  = fpsan::detail::vector_element_t<AVec>;
    using BElem                  = fpsan::detail::vector_element_t<BVec>;
    using CElem                  = fpsan::detail::vector_element_t<CVec>;
    using CBits                  = typename fpsan::detail::fp_traits<CElem>::bits_type;
    static constexpr int a_lanes = sizeof(AVec) / sizeof(AElem);
    static constexpr int b_lanes = sizeof(BVec) / sizeof(BElem);
    static constexpr int c_lanes = sizeof(CVec) / sizeof(CElem);
};

// Pack this lane's A/B/C fragments from row-major logical matrices.
template <class Traits, Semantics S>
__device__ void load_frags(const typename Harness<Traits>::AElem*         A,
                           const typename Harness<Traits>::BElem*         B,
                           const typename Harness<Traits>::CElem*         C,
                           int                                            lane,
                           Value<typename Harness<Traits>::AVec, S, kCC>& a,
                           Value<typename Harness<Traits>::BVec, S, kCC>& b,
                           Value<typename Harness<Traits>::CVec, S, kCC>& c)
{
    typename Harness<Traits>::AVec an;
    typename Harness<Traits>::BVec bn;
    typename Harness<Traits>::CVec cn;
    for(int idx = 0; idx < Harness<Traits>::a_lanes; ++idx)
    {
        int k   = frag_k_inv(lane, idx);
        an[idx] = A[(lane & 15) * K + k]; // A[m=lane&15][k]
        bn[idx] = B[k * N + (lane & 15)]; // B[k][n=lane&15]
    }
    for(int e = 0; e < Harness<Traits>::c_lanes; ++e)
    {
        int m = cd_m_from_lane(lane, e), n = lane & 15;
        cn[e] = C[m * N + n];
    }
    a = Value<typename Harness<Traits>::AVec, S, kCC>(an);
    b = Value<typename Harness<Traits>::BVec, S, kCC>(bn);
    c = Value<typename Harness<Traits>::CVec, S, kCC>(cn);
}

// Float-mode kernel: invokes Traits::call (which goes to the real builtin in
// Float mode) and writes D as element-type values to a row-major host buffer.
template <class Traits>
__global__ void k_builtin(const typename Harness<Traits>::AElem* A,
                          const typename Harness<Traits>::BElem* B,
                          const typename Harness<Traits>::CElem* C,
                          typename Harness<Traits>::CElem*       D)
{
    int                                                           lane = threadIdx.x;
    Value<typename Harness<Traits>::AVec, Semantics::Native, kCC> a;
    Value<typename Harness<Traits>::BVec, Semantics::Native, kCC> b;
    Value<typename Harness<Traits>::CVec, Semantics::Native, kCC> c;
    load_frags<Traits, Semantics::Native>(A, B, C, lane, a, b, c);
    auto d = Traits::template call<Semantics::Native, kCC>(a, b, c);
    for(int e = 0; e < Harness<Traits>::c_lanes; ++e)
        D[cd_m_from_lane(lane, e) * N + (lane & 15)] = d.get(e).to_float();
}

// Float-mode kernel: invokes our software dataflow (NOT the builtin), used as
// the oracle vs the builtin in the layout test.
template <class Traits>
__global__ void k_float_dataflow(const typename Harness<Traits>::AElem* A,
                                 const typename Harness<Traits>::BElem* B,
                                 const typename Harness<Traits>::CElem* C,
                                 typename Harness<Traits>::CElem*       D)
{
    int                                                           lane = threadIdx.x;
    Value<typename Harness<Traits>::AVec, Semantics::Native, kCC> a;
    Value<typename Harness<Traits>::BVec, Semantics::Native, kCC> b;
    Value<typename Harness<Traits>::CVec, Semantics::Native, kCC> c;
    load_frags<Traits, Semantics::Native>(A, B, C, lane, a, b, c);
    auto d = fpsan::detail::wmma_16x16x16_dataflow(a, b, c);
    for(int e = 0; e < Harness<Traits>::c_lanes; ++e)
        D[cd_m_from_lane(lane, e) * N + (lane & 15)] = d.get(e).to_float();
}

// FPSan-mode kernel: writes the per-element FPSan payload to a row-major buffer
// of the matching unsigned integer type.
template <class Traits>
__global__ void k_fpsan(const typename Harness<Traits>::AElem* A,
                        const typename Harness<Traits>::BElem* B,
                        const typename Harness<Traits>::CElem* C,
                        typename Harness<Traits>::CBits*       Dpay)
{
    int                                                           lane = threadIdx.x;
    Value<typename Harness<Traits>::AVec, Semantics::Triton, kCC> a;
    Value<typename Harness<Traits>::BVec, Semantics::Triton, kCC> b;
    Value<typename Harness<Traits>::CVec, Semantics::Triton, kCC> c;
    load_frags<Traits, Semantics::Triton>(A, B, C, lane, a, b, c);
    auto d = Traits::template call<Semantics::Triton, kCC>(a, b, c);
    for(int e = 0; e < Harness<Traits>::c_lanes; ++e)
        Dpay[cd_m_from_lane(lane, e) * N + (lane & 15)] = d.get(e).fpsan_payload();
}

namespace
{

    // Deterministic pseudo-random inputs (see test_random.hpp). Small integer
    // values keep products and length-16 sums exact in every supported output
    // type (f16/bf16/f32), so the hardware-vs-dataflow comparison stays bit-exact.
    template <class Traits>
    struct Mats
    {
        std::vector<typename Harness<Traits>::AElem> A;
        std::vector<typename Harness<Traits>::BElem> B;
        std::vector<typename Harness<Traits>::CElem> C;
    };
    template <class Traits>
    Mats<Traits> make_inputs()
    {
        using AE = typename Harness<Traits>::AElem;
        using BE = typename Harness<Traits>::BElem;
        using CE = typename Harness<Traits>::CElem;
        Mats<Traits> m;
        m.A.resize(M * K);
        m.B.resize(K * N);
        m.C.resize(M * N);
        std::mt19937 rng = fpsan_test::make_rng();
        for(auto& x : m.A)
            x = fpsan_test::pick_int_valued<AE>(rng, Traits::a_lo, Traits::a_hi);
        for(auto& x : m.B)
            x = fpsan_test::pick_int_valued<BE>(rng, Traits::b_lo, Traits::b_hi);
        for(auto& x : m.C)
            x = fpsan_test::pick_int_valued<CE>(rng, Traits::c_lo, Traits::c_hi);
        return m;
    }
} // namespace

// (1) Layout test: real hardware vs our software dataflow, both Float.
template <class Traits>
void run_layout_matches_hardware()
{
    using CE = typename Harness<Traits>::CElem;
    using AE = typename Harness<Traits>::AElem;
    using BE = typename Harness<Traits>::BElem;
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    Mats<Traits> m = make_inputs<Traits>();

    std::vector<CE> ref(M * N);
    for(int mm = 0; mm < M; ++mm)
        for(int nn = 0; nn < N; ++nn)
        {
            float acc = static_cast<float>(m.C[mm * N + nn]);
            for(int k = 0; k < K; ++k)
                acc += static_cast<float>(m.A[mm * K + k]) * static_cast<float>(m.B[k * N + nn]);
            ref[mm * N + nn] = static_cast<CE>(acc);
        }

    AE* dA = to_dev(m.A);
    BE* dB = to_dev(m.B);
    CE* dC = to_dev(m.C);
    CE *dHw, *dOurs;
    HIP_CHECK(hipMalloc(&dHw, M * N * sizeof(CE)));
    HIP_CHECK(hipMalloc(&dOurs, M * N * sizeof(CE)));

    k_builtin<Traits><<<1, kWaveSize>>>(dA, dB, dC, dHw);
    k_float_dataflow<Traits><<<1, kWaveSize>>>(dA, dB, dC, dOurs);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<CE> hw(M * N), ours(M * N);
    HIP_CHECK(hipMemcpy(hw.data(), dHw, M * N * sizeof(CE), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(ours.data(), dOurs, M * N * sizeof(CE), hipMemcpyDeviceToHost));
    for(int i = 0; i < M * N; ++i)
    {
        EXPECT_EQ(bits_of(hw[i]), bits_of(ref[i]))
            << "hardware mismatch at " << (i / N) << "," << (i % N);
        EXPECT_EQ(bits_of(ours[i]), bits_of(ref[i]))
            << "dataflow mismatch at " << (i / N) << "," << (i % N);
    }
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dHw);
    (void)hipFree(dOurs);
}

// (2) Payload test: shipped FPSan path vs host scalar FPSan reference.
template <class Traits>
void run_fpsan_matches_scalar_reference()
{
    using AE    = typename Harness<Traits>::AElem;
    using BE    = typename Harness<Traits>::BElem;
    using CE    = typename Harness<Traits>::CElem;
    using CBits = typename Harness<Traits>::CBits;
    int ndev    = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    Mats<Traits> m = make_inputs<Traits>();

    // Host scalar FPSan reference using the same dataflow:
    //   D[m][n] = C[m][n] + sum_k cast<CE>(A[m][k]) * cast<CE>(B[k][n]).
    using VA = Value<AE, Semantics::Triton, kCC>;
    using VB = Value<BE, Semantics::Triton, kCC>;
    using VC = Value<CE, Semantics::Triton, kCC>;
    std::vector<CBits> ref(M * N);
    for(int mm = 0; mm < M; ++mm)
        for(int nn = 0; nn < N; ++nn)
        {
            VC acc(m.C[mm * N + nn]);
            for(int k = 0; k < K; ++k)
                acc = acc
                      + fpsan::cast<CE>(VA(m.A[mm * K + k])) * fpsan::cast<CE>(VB(m.B[k * N + nn]));
            ref[mm * N + nn] = acc.fpsan_payload();
        }

    AE*    dA = to_dev(m.A);
    BE*    dB = to_dev(m.B);
    CE*    dC = to_dev(m.C);
    CBits* dD;
    HIP_CHECK(hipMalloc(&dD, M * N * sizeof(CBits)));
    k_fpsan<Traits><<<1, kWaveSize>>>(dA, dB, dC, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<CBits> got(M * N);
    HIP_CHECK(hipMemcpy(got.data(), dD, M * N * sizeof(CBits), hipMemcpyDeviceToHost));
    for(int i = 0; i < M * N; ++i)
        EXPECT_EQ(got[i], ref[i]) << "payload mismatch at " << (i / N) << "," << (i % N);
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
}

// ---------------------------------------------------------------------------
// Per-intrinsic Traits + TESTs. Each entry is ~15 lines and gets BOTH a
// LayoutMatchesHardware and a FpsanMatchesScalarReference test for free.
// ---------------------------------------------------------------------------

struct WmmaF32F16
{
    using AVec                = std::conditional_t<kWaveSize == 64, v4h_native, v8h_native>;
    using BVec                = std::conditional_t<kWaveSize == 64, v4h_native, v8h_native>;
    using CVec                = std::conditional_t<kWaveSize == 64, v4f_native, v8f_native>;
    static constexpr int a_lo = -3, a_hi = 3;
    static constexpr int b_lo = -2, b_hi = 2;
    static constexpr int c_lo = -4, c_hi = 4;
    template <Semantics S, Conversions C>
    __device__ static Value<CVec, S, C>
        call(Value<AVec, S, C> a, Value<BVec, S, C> b, Value<CVec, S, C> c)
    {
#if defined(FPSAN_TEST_GFX12_WAVE64)
        return fpsan::amdgcn_wmma_f32_16x16x16_f16_w64(a, b, c);
#else
        return fpsan::amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#endif
    }
};

struct WmmaF16F16
{
    using AVec = std::conditional_t<kWaveSize == 64, v4h_native, v8h_native>;
    using BVec = std::conditional_t<kWaveSize == 64, v4h_native, v8h_native>;
    using CVec = std::conditional_t<kWaveSize == 64, v4h_native, v8h_native>;
    // f16 stays exact for sums up to a few thousand; keep ranges modest.
    static constexpr int a_lo = -3, a_hi = 3;
    static constexpr int b_lo = -2, b_hi = 2;
    static constexpr int c_lo = -4, c_hi = 4;
    template <Semantics S, Conversions C>
    __device__ static Value<CVec, S, C>
        call(Value<AVec, S, C> a, Value<BVec, S, C> b, Value<CVec, S, C> c)
    {
#if defined(FPSAN_TEST_GFX12_WAVE64)
        return fpsan::amdgcn_wmma_f16_16x16x16_f16_w64(a, b, c);
#else
        return fpsan::amdgcn_wmma_f16_16x16x16_f16_w32(a, b, c);
#endif
    }
};

struct WmmaF32BF16
{
    using AVec = std::conditional_t<kWaveSize == 64, v4bf_native, v8bf_native>;
    using BVec = std::conditional_t<kWaveSize == 64, v4bf_native, v8bf_native>;
    using CVec = std::conditional_t<kWaveSize == 64, v4f_native, v8f_native>;
    // bf16 has only 7 mantissa bits; small ints fit exactly.
    static constexpr int a_lo = -3, a_hi = 3;
    static constexpr int b_lo = -2, b_hi = 2;
    static constexpr int c_lo = -4, c_hi = 4;
    template <Semantics S, Conversions C>
    __device__ static Value<CVec, S, C>
        call(Value<AVec, S, C> a, Value<BVec, S, C> b, Value<CVec, S, C> c)
    {
#if defined(FPSAN_TEST_GFX12_WAVE64)
        return fpsan::amdgcn_wmma_f32_16x16x16_bf16_w64(a, b, c);
#else
        return fpsan::amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
#endif
    }
};

struct WmmaBF16BF16
{
    using AVec = std::conditional_t<kWaveSize == 64, v4bf_native, v8bf_native>;
    using BVec = std::conditional_t<kWaveSize == 64, v4bf_native, v8bf_native>;
    using CVec = std::conditional_t<kWaveSize == 64, v4bf_native, v8bf_native>;
    // bf16 accumulation: keep magnitudes small so length-16 sums stay exact.
    static constexpr int a_lo = -3, a_hi = 3;
    static constexpr int b_lo = -2, b_hi = 2;
    static constexpr int c_lo = -4, c_hi = 4;
    template <Semantics S, Conversions C>
    __device__ static Value<CVec, S, C>
        call(Value<AVec, S, C> a, Value<BVec, S, C> b, Value<CVec, S, C> c)
    {
#if defined(FPSAN_TEST_GFX12_WAVE64)
        return fpsan::amdgcn_wmma_bf16_16x16x16_bf16_w64(a, b, c);
#else
        return fpsan::amdgcn_wmma_bf16_16x16x16_bf16_w32(a, b, c);
#endif
    }
};

// ---- fp8 variants (AMD's naming: 'fp8' = OCP E4M3FN, 'bf8' = OCP E5M2) ------
// Per-lane fragment is 8 fp8s = 8 bytes = the v2i32 the builtin actually wants.

using fpsan::v8e4m3_native;
using fpsan::v8e5m2_native;

struct WmmaF32Fp8Fp8
{
    using AVec = std::conditional_t<kWaveSize == 64, v4e4m3_native, v8e4m3_native>;
    using BVec = std::conditional_t<kWaveSize == 64, v4e4m3_native, v8e4m3_native>;
    using CVec = std::conditional_t<kWaveSize == 64, v4f_native, v8f_native>;
    // E4M3FN represents -3..3 exactly; sums of 16 products of |v|<=3*2 = 6 fit
    // exactly in f32, so the hardware-vs-dataflow comparison stays bit-exact.
    static constexpr int a_lo = -3, a_hi = 3;
    static constexpr int b_lo = -2, b_hi = 2;
    static constexpr int c_lo = -4, c_hi = 4;
    template <Semantics S, Conversions C>
    __device__ static Value<CVec, S, C>
        call(Value<AVec, S, C> a, Value<BVec, S, C> b, Value<CVec, S, C> c)
    {
#if defined(FPSAN_TEST_GFX12_WAVE64)
        return fpsan::amdgcn_wmma_f32_16x16x16_fp8_fp8_w64(a, b, c);
#else
        return fpsan::amdgcn_wmma_f32_16x16x16_fp8_fp8_w32(a, b, c);
#endif
    }
};

struct WmmaF32Fp8Bf8
{
    using AVec                = std::conditional_t<kWaveSize == 64, v4e4m3_native, v8e4m3_native>;
    using BVec                = std::conditional_t<kWaveSize == 64, v4e5m2_native, v8e5m2_native>;
    using CVec                = std::conditional_t<kWaveSize == 64, v4f_native, v8f_native>;
    static constexpr int a_lo = -3, a_hi = 3;
    static constexpr int b_lo = -2, b_hi = 2;
    static constexpr int c_lo = -4, c_hi = 4;
    template <Semantics S, Conversions C>
    __device__ static Value<CVec, S, C>
        call(Value<AVec, S, C> a, Value<BVec, S, C> b, Value<CVec, S, C> c)
    {
#if defined(FPSAN_TEST_GFX12_WAVE64)
        return fpsan::amdgcn_wmma_f32_16x16x16_fp8_bf8_w64(a, b, c);
#else
        return fpsan::amdgcn_wmma_f32_16x16x16_fp8_bf8_w32(a, b, c);
#endif
    }
};

struct WmmaF32Bf8Fp8
{
    using AVec                = std::conditional_t<kWaveSize == 64, v4e5m2_native, v8e5m2_native>;
    using BVec                = std::conditional_t<kWaveSize == 64, v4e4m3_native, v8e4m3_native>;
    using CVec                = std::conditional_t<kWaveSize == 64, v4f_native, v8f_native>;
    static constexpr int a_lo = -3, a_hi = 3;
    static constexpr int b_lo = -2, b_hi = 2;
    static constexpr int c_lo = -4, c_hi = 4;
    template <Semantics S, Conversions C>
    __device__ static Value<CVec, S, C>
        call(Value<AVec, S, C> a, Value<BVec, S, C> b, Value<CVec, S, C> c)
    {
#if defined(FPSAN_TEST_GFX12_WAVE64)
        return fpsan::amdgcn_wmma_f32_16x16x16_bf8_fp8_w64(a, b, c);
#else
        return fpsan::amdgcn_wmma_f32_16x16x16_bf8_fp8_w32(a, b, c);
#endif
    }
};

struct WmmaF32Bf8Bf8
{
    using AVec                = std::conditional_t<kWaveSize == 64, v4e5m2_native, v8e5m2_native>;
    using BVec                = std::conditional_t<kWaveSize == 64, v4e5m2_native, v8e5m2_native>;
    using CVec                = std::conditional_t<kWaveSize == 64, v4f_native, v8f_native>;
    static constexpr int a_lo = -3, a_hi = 3;
    static constexpr int b_lo = -2, b_hi = 2;
    static constexpr int c_lo = -4, c_hi = 4;
    template <Semantics S, Conversions C>
    __device__ static Value<CVec, S, C>
        call(Value<AVec, S, C> a, Value<BVec, S, C> b, Value<CVec, S, C> c)
    {
#if defined(FPSAN_TEST_GFX12_WAVE64)
        return fpsan::amdgcn_wmma_f32_16x16x16_bf8_bf8_w64(a, b, c);
#else
        return fpsan::amdgcn_wmma_f32_16x16x16_bf8_bf8_w32(a, b, c);
#endif
    }
};

#define FPSAN_WMMA_GFX12_CONCAT2(A, B) A##B
#define FPSAN_WMMA_GFX12_CONCAT(A, B) FPSAN_WMMA_GFX12_CONCAT2(A, B)
#if defined(FPSAN_TEST_GFX12_WAVE64)
#define FPSAN_WMMA_GFX12_SUITE(NAME) FPSAN_WMMA_GFX12_CONCAT(NAME, W64)
#else
#define FPSAN_WMMA_GFX12_SUITE(NAME) FPSAN_WMMA_GFX12_CONCAT(NAME, W32)
#endif

#define FPSAN_WMMA_GFX12_TESTS(NAME)                                \
    TEST(FPSAN_WMMA_GFX12_SUITE(NAME), LayoutMatchesHardware)       \
    {                                                               \
        run_layout_matches_hardware<NAME>();                        \
    }                                                               \
    TEST(FPSAN_WMMA_GFX12_SUITE(NAME), FpsanMatchesScalarReference) \
    {                                                               \
        run_fpsan_matches_scalar_reference<NAME>();                 \
    }

FPSAN_WMMA_GFX12_TESTS(WmmaF32F16)
FPSAN_WMMA_GFX12_TESTS(WmmaF16F16)
FPSAN_WMMA_GFX12_TESTS(WmmaF32BF16)
FPSAN_WMMA_GFX12_TESTS(WmmaBF16BF16)
FPSAN_WMMA_GFX12_TESTS(WmmaF32Fp8Fp8)
FPSAN_WMMA_GFX12_TESTS(WmmaF32Fp8Bf8)
FPSAN_WMMA_GFX12_TESTS(WmmaF32Bf8Fp8)
FPSAN_WMMA_GFX12_TESTS(WmmaF32Bf8Bf8)

#undef FPSAN_WMMA_GFX12_TESTS
#undef FPSAN_WMMA_GFX12_SUITE
#undef FPSAN_WMMA_GFX12_CONCAT
#undef FPSAN_WMMA_GFX12_CONCAT2
