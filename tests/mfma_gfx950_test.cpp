// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/mfma_gfx950_test.cpp
//
// GPU tests for the gfx950 (CDNA4 / MI350) MFMA wrappers in
// fpsan/amdgcn_mfma.hpp. Same two-property structure as wmma_gfx12_test.cpp,
// but for the wave64 CDNA MFMA fragment layout (input_loc / output_loc,
// validated against real silicon -- these tests are the gfx950 source of truth,
// not any simulator):
//
//  (1) LayoutMatchesHardware (Float mode): we load the A/B/C fragments into the
//      per-lane registers using the ported input_loc / output_loc maps, call
//      the real __builtin_amdgcn_mfma_*, read D back through output_loc, and
//      compare against a host integer reference D = C + A*B.  Inputs are exact
//      small integers, so the accumulation is exact and the comparison is
//      bit-for-bit.  Because the builtin embodies the *true* hardware layout,
//      this passes only if both input_loc and output_loc match silicon.
//
//  (2) FpsanMatchesScalarReference (FPSan mode): the shipped FPSan software
//      dataflow (detail::mfma_software) must match an independent host scalar
//      FPSan reference matmul computed in the payload ring, payload for
//      payload. The maps cancel (used on both ends), so this isolates the
//      gather + payload arithmetic; the layout itself is pinned by (1).
//
// A templated Harness<Traits> drives both; each shape is a ~12-line Traits.
//
// Requires real MI350 (gfx950) hardware; built only under FPSAN_ENABLE_HIP with
// gfx950 in CMAKE_HIP_ARCHITECTURES.
#include "fpsan/amdgcn_mfma.hpp"
#include "fpsan/amdgcn_smfmac.hpp"
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
using fpsan::Value;

static constexpr Conversions kCC  = Conversions::Explicit;
static constexpr int         WAVE = 64;

template <class T, class = void>
struct MfmaTraitCBSZ : std::integral_constant<int, 0>
{
};
template <class T>
struct MfmaTraitCBSZ<T, std::void_t<decltype(T::CBSZ)>> : std::integral_constant<int, T::CBSZ>
{
};
template <class T, class = void>
struct MfmaTraitABID : std::integral_constant<int, 0>
{
};
template <class T>
struct MfmaTraitABID<T, std::void_t<decltype(T::ABID)>> : std::integral_constant<int, T::ABID>
{
};
template <class T, class = void>
struct MfmaTraitBLGP : std::integral_constant<int, 0>
{
};
template <class T>
struct MfmaTraitBLGP<T, std::void_t<decltype(T::BLGP)>> : std::integral_constant<int, T::BLGP>
{
};

static int host_mfma_a_src_lane(int lane, int cbsz, int abid)
{
    if(cbsz == 0)
        return lane;
    const int block = WAVE / (1 << cbsz);
    return (lane % block) + block * abid;
}

static int host_mfma_b_src_lane(int lane, int blgp)
{
    switch(blgp)
    {
    case 0:
        return lane;
    case 1:
        return lane % 32;
    case 2:
        return (lane % 32) + 32;
    case 3:
        return (lane + 16) % WAVE;
    case 4:
        return lane % 16;
    case 5:
        return (lane % 16) + 16;
    case 6:
        return (lane % 16) + 32;
    case 7:
        return (lane % 16) + 48;
    default:
        return lane;
    }
}

// ---------------------------------------------------------------------------
// Harness shared by every dense MFMA shape whose A/B operands are packed
// per-lane vector fragments (v8 of f16/bf16/fp8).  A Traits declares the
// native fragment types, the shape (M,N,K) and operand width InBits, value
// ranges, and a semantics-generic call() dispatching to the fpsan:: wrapper.
// ---------------------------------------------------------------------------
template <class Traits>
struct Harness
{
    using AVec             = typename Traits::AVec;
    using BVec             = typename Traits::BVec;
    using CVec             = typename Traits::CVec;
    using AElem            = fpsan::detail::vector_element_t<AVec>;
    using BElem            = fpsan::detail::vector_element_t<BVec>;
    using CElem            = fpsan::detail::vector_element_t<CVec>;
    using CBits            = typename fpsan::detail::fp_traits<CElem>::bits_type;
    static constexpr int M = Traits::M, N = Traits::N, K = Traits::K;
    static constexpr int InBits    = Traits::InBits;
    static constexpr int per_dword = 32 / InBits;
    static constexpr int CBSZ      = MfmaTraitCBSZ<Traits>::value;
    static constexpr int ABID      = MfmaTraitABID<Traits>::value;
    static constexpr int BLGP      = MfmaTraitBLGP<Traits>::value;
};

// Pack this lane's A/B/C fragments from row-major logical matrices, using the
// hardware fragment layout (input_loc for A/B, output_loc_32 for C). Shared by
// the Float and FPSan kernels; the Value ctor embeds for FPSan automatically.
template <class Traits, Semantics S>
__device__ void load_frags(const typename Harness<Traits>::AElem*         A,
                           const typename Harness<Traits>::BElem*         B,
                           const typename Harness<Traits>::CElem*         C,
                           int                                            lane,
                           Value<typename Harness<Traits>::AVec, S, kCC>& a,
                           Value<typename Harness<Traits>::BVec, S, kCC>& b,
                           Value<typename Harness<Traits>::CVec, S, kCC>& c)
{
    using H = Harness<Traits>;
    typename H::AVec an{};
    typename H::BVec bn{};
    typename H::CVec cn{};
    // A[i][k]: dim = M, outer index = i (row).
    for(int i = 0; i < H::M; ++i)
        for(int k = 0; k < H::K; ++k)
        {
            auto loc = fpsan::detail::input_loc(H::M, H::K, 1, i, k, 0, H::InBits);
            if(loc.lane == lane)
                an[H::per_dword * loc.reg + loc.sub] = A[i * H::K + k];
        }
    // B[k][j]: dim = N, outer index = j (column).
    for(int j = 0; j < H::N; ++j)
        for(int k = 0; k < H::K; ++k)
        {
            auto loc = fpsan::detail::input_loc(H::N, H::K, 1, j, k, 0, H::InBits);
            if(loc.lane == lane)
                bn[H::per_dword * loc.reg + loc.sub] = B[k * H::N + j];
        }
    // C[i][j]: output layout.
    for(int i = 0; i < H::M; ++i)
        for(int j = 0; j < H::N; ++j)
        {
            auto loc = fpsan::detail::output_loc_32(H::M, H::N, i, j, 0);
            if(loc.lane == lane)
                cn[loc.reg] = C[i * H::N + j];
        }
    a = Value<typename H::AVec, S, kCC>(an);
    b = Value<typename H::BVec, S, kCC>(bn);
    c = Value<typename H::CVec, S, kCC>(cn);
}

template <class Traits>
typename Harness<Traits>::AElem
    host_mfma_a_value(const std::vector<typename Harness<Traits>::AElem>& A, int i, int k)
{
    using H            = Harness<Traits>;
    auto      want     = fpsan::detail::input_loc(H::M, H::K, 1, i, k, 0, H::InBits);
    const int src_lane = host_mfma_a_src_lane(want.lane, H::CBSZ, H::ABID);
    for(int ii = 0; ii < H::M; ++ii)
        for(int kk = 0; kk < H::K; ++kk)
        {
            auto loc = fpsan::detail::input_loc(H::M, H::K, 1, ii, kk, 0, H::InBits);
            if(loc.lane == src_lane && loc.reg == want.reg && loc.sub == want.sub)
                return A[ii * H::K + kk];
        }
    return typename H::AElem{};
}

template <class Traits>
typename Harness<Traits>::BElem
    host_mfma_b_value(const std::vector<typename Harness<Traits>::BElem>& B, int k, int j)
{
    using H            = Harness<Traits>;
    auto      want     = fpsan::detail::input_loc(H::N, H::K, 1, j, k, 0, H::InBits);
    const int src_lane = host_mfma_b_src_lane(want.lane, H::BLGP);
    for(int jj = 0; jj < H::N; ++jj)
        for(int kk = 0; kk < H::K; ++kk)
        {
            auto loc = fpsan::detail::input_loc(H::N, H::K, 1, jj, kk, 0, H::InBits);
            if(loc.lane == src_lane && loc.reg == want.reg && loc.sub == want.sub)
                return B[kk * H::N + jj];
        }
    return typename H::BElem{};
}

// Float-mode kernel: calls the real builtin (Float mode) and writes D as
// element-type values to a row-major host buffer through output_loc_32.
template <class Traits>
__global__ void k_builtin(const typename Harness<Traits>::AElem* A,
                          const typename Harness<Traits>::BElem* B,
                          const typename Harness<Traits>::CElem* C,
                          typename Harness<Traits>::CElem*       D)
{
    using H                                              = Harness<Traits>;
    int                                             lane = threadIdx.x;
    Value<typename H::AVec, Semantics::Native, kCC> a;
    Value<typename H::BVec, Semantics::Native, kCC> b;
    Value<typename H::CVec, Semantics::Native, kCC> c;
    load_frags<Traits, Semantics::Native>(A, B, C, lane, a, b, c);
    auto d = Traits::template call<Semantics::Native, kCC>(a, b, c);
    for(int i = 0; i < H::M; ++i)
        for(int j = 0; j < H::N; ++j)
        {
            auto loc = fpsan::detail::output_loc_32(H::M, H::N, i, j, 0);
            if(loc.lane == lane)
                D[i * H::N + j] = d.get(loc.reg).to_float();
        }
}

// FPSan-mode kernel: writes per-element FPSan payloads through output_loc_32.
template <class Traits>
__global__ void k_fpsan(const typename Harness<Traits>::AElem* A,
                        const typename Harness<Traits>::BElem* B,
                        const typename Harness<Traits>::CElem* C,
                        typename Harness<Traits>::CBits*       Dpay)
{
    using H                                              = Harness<Traits>;
    int                                             lane = threadIdx.x;
    Value<typename H::AVec, Semantics::Triton, kCC> a;
    Value<typename H::BVec, Semantics::Triton, kCC> b;
    Value<typename H::CVec, Semantics::Triton, kCC> c;
    load_frags<Traits, Semantics::Triton>(A, B, C, lane, a, b, c);
    auto d = Traits::template call<Semantics::Triton, kCC>(a, b, c);
    for(int i = 0; i < H::M; ++i)
        for(int j = 0; j < H::N; ++j)
        {
            auto loc = fpsan::detail::output_loc_32(H::M, H::N, i, j, 0);
            if(loc.lane == lane)
                Dpay[i * H::N + j] = d.get(loc.reg).fpsan_payload();
        }
}

namespace
{

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
        using H = Harness<Traits>;
        Mats<Traits> m;
        m.A.resize(H::M * H::K);
        m.B.resize(H::K * H::N);
        m.C.resize(H::M * H::N);
        std::mt19937 rng = fpsan_test::make_rng();
        for(auto& x : m.A)
            x = fpsan_test::pick_int_valued<typename H::AElem>(rng, Traits::a_lo, Traits::a_hi);
        for(auto& x : m.B)
            x = fpsan_test::pick_int_valued<typename H::BElem>(rng, Traits::b_lo, Traits::b_hi);
        for(auto& x : m.C)
            x = fpsan_test::pick_int_valued<typename H::CElem>(rng, Traits::c_lo, Traits::c_hi);
        return m;
    }
} // namespace

// (1) Layout test: real builtin (Float) vs host integer reference.
template <class Traits>
void run_layout_matches_hardware()
{
    using H  = Harness<Traits>;
    using AE = typename H::AElem;
    using BE = typename H::BElem;
    using CE = typename H::CElem;
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    Mats<Traits> m = make_inputs<Traits>();

    // Host integer reference: exact because inputs are small integers.
    std::vector<CE> ref(H::M * H::N);
    for(int i = 0; i < H::M; ++i)
        for(int j = 0; j < H::N; ++j)
        {
            double acc = static_cast<double>(static_cast<float>(m.C[i * H::N + j]));
            for(int k = 0; k < H::K; ++k)
                acc += static_cast<double>(static_cast<float>(host_mfma_a_value<Traits>(m.A, i, k)))
                       * static_cast<double>(
                           static_cast<float>(host_mfma_b_value<Traits>(m.B, k, j)));
            ref[i * H::N + j] = static_cast<CE>(static_cast<float>(acc));
        }

    AE* dA = to_dev(m.A);
    BE* dB = to_dev(m.B);
    CE* dC = to_dev(m.C);
    CE* dD;
    HIP_CHECK(hipMalloc(&dD, H::M * H::N * sizeof(CE)));
    k_builtin<Traits><<<1, WAVE>>>(dA, dB, dC, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<CE> got(H::M * H::N);
    HIP_CHECK(hipMemcpy(got.data(), dD, H::M * H::N * sizeof(CE), hipMemcpyDeviceToHost));
    for(int i = 0; i < H::M * H::N; ++i)
        EXPECT_EQ(bits_of(got[i]), bits_of(ref[i]))
            << "layout mismatch at " << (i / H::N) << "," << (i % H::N);
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
}

// (2) Payload test: shipped FPSan dataflow vs host scalar FPSan reference.
template <class Traits>
void run_fpsan_matches_scalar_reference()
{
    using H     = Harness<Traits>;
    using AE    = typename H::AElem;
    using BE    = typename H::BElem;
    using CE    = typename H::CElem;
    using CBits = typename H::CBits;
    int ndev    = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    Mats<Traits> m = make_inputs<Traits>();

    using VA = Value<AE, Semantics::Triton, kCC>;
    using VB = Value<BE, Semantics::Triton, kCC>;
    using VC = Value<CE, Semantics::Triton, kCC>;
    std::vector<CBits> ref(H::M * H::N);
    for(int i = 0; i < H::M; ++i)
        for(int j = 0; j < H::N; ++j)
        {
            VC acc(m.C[i * H::N + j]);
            for(int k = 0; k < H::K; ++k)
                acc = acc
                      + fpsan::cast<CE>(VA(host_mfma_a_value<Traits>(m.A, i, k)))
                            * fpsan::cast<CE>(VB(host_mfma_b_value<Traits>(m.B, k, j)));
            ref[i * H::N + j] = acc.fpsan_payload();
        }

    AE*    dA = to_dev(m.A);
    BE*    dB = to_dev(m.B);
    CE*    dC = to_dev(m.C);
    CBits* dD;
    HIP_CHECK(hipMalloc(&dD, H::M * H::N * sizeof(CBits)));
    k_fpsan<Traits><<<1, WAVE>>>(dA, dB, dC, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<CBits> got(H::M * H::N);
    HIP_CHECK(hipMemcpy(got.data(), dD, H::M * H::N * sizeof(CBits), hipMemcpyDeviceToHost));
    for(int i = 0; i < H::M * H::N; ++i)
        EXPECT_EQ(got[i], ref[i]) << "payload mismatch at " << (i / H::N) << "," << (i % H::N);
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
}

// ---------------------------------------------------------------------------
// Per-shape Traits + TESTs.  Each gets both a LayoutMatchesHardware and a
// FpsanMatchesScalarReference test.
// ---------------------------------------------------------------------------
using fpsan::v16f_native;
using fpsan::v4f_native;
using fpsan::v8bf_native;
using fpsan::v8e4m3_native;
using fpsan::v8e5m2_native;
using fpsan::v8h_native;

#define MFMA_TESTS(Name)                            \
    TEST(Name, LayoutMatchesHardware)               \
    {                                               \
        run_layout_matches_hardware<Name>();        \
    }                                               \
    TEST(Name, FpsanMatchesScalarReference)         \
    {                                               \
        run_fpsan_matches_scalar_reference<Name>(); \
    }

// ---- F16 / BF16 inputs, F32 accumulator -----------------------------------
struct MfmaF32_16x16x32_F16
{
    using AVec             = v8h_native;
    using BVec             = v8h_native;
    using CVec             = v4f_native;
    static constexpr int M = 16, N = 16, K = 32, InBits = 16;
    static constexpr int a_lo = -3, a_hi = 3, b_lo = -2, b_hi = 2, c_lo = -4, c_hi = 4;
    template <Semantics S, Conversions C>
    __device__ static Value<CVec, S, C>
        call(Value<AVec, S, C> a, Value<BVec, S, C> b, Value<CVec, S, C> c)
    {
        return fpsan::amdgcn_mfma_f32_16x16x32_f16<0, 0, 0, S, C>(a, b, c);
    }
};
MFMA_TESTS(MfmaF32_16x16x32_F16)

struct MfmaF32_16x16x32_BF16
{
    using AVec             = v8bf_native;
    using BVec             = v8bf_native;
    using CVec             = v4f_native;
    static constexpr int M = 16, N = 16, K = 32, InBits = 16;
    static constexpr int a_lo = -3, a_hi = 3, b_lo = -2, b_hi = 2, c_lo = -4, c_hi = 4;
    template <Semantics S, Conversions C>
    __device__ static Value<CVec, S, C>
        call(Value<AVec, S, C> a, Value<BVec, S, C> b, Value<CVec, S, C> c)
    {
        return fpsan::amdgcn_mfma_f32_16x16x32_bf16<0, 0, 0, S, C>(a, b, c);
    }
};
MFMA_TESTS(MfmaF32_16x16x32_BF16)

struct MfmaF32_32x32x16_F16
{
    using AVec             = v8h_native;
    using BVec             = v8h_native;
    using CVec             = v16f_native;
    static constexpr int M = 32, N = 32, K = 16, InBits = 16;
    static constexpr int a_lo = -3, a_hi = 3, b_lo = -2, b_hi = 2, c_lo = -4, c_hi = 4;
    template <Semantics S, Conversions C>
    __device__ static Value<CVec, S, C>
        call(Value<AVec, S, C> a, Value<BVec, S, C> b, Value<CVec, S, C> c)
    {
        return fpsan::amdgcn_mfma_f32_32x32x16_f16<0, 0, 0, S, C>(a, b, c);
    }
};
MFMA_TESTS(MfmaF32_32x32x16_F16)

struct MfmaF32_32x32x16_BF16
{
    using AVec             = v8bf_native;
    using BVec             = v8bf_native;
    using CVec             = v16f_native;
    static constexpr int M = 32, N = 32, K = 16, InBits = 16;
    static constexpr int a_lo = -3, a_hi = 3, b_lo = -2, b_hi = 2, c_lo = -4, c_hi = 4;
    template <Semantics S, Conversions C>
    __device__ static Value<CVec, S, C>
        call(Value<AVec, S, C> a, Value<BVec, S, C> b, Value<CVec, S, C> c)
    {
        return fpsan::amdgcn_mfma_f32_32x32x16_bf16<0, 0, 0, S, C>(a, b, c);
    }
};
MFMA_TESTS(MfmaF32_32x32x16_BF16)

// ---- FP8 / BF8 inputs, F32 accumulator ------------------------------------
#define MFMA_FP8_TRAITS(Name, AV, BV, CV, M_, N_, K_, WRAP)                                 \
    struct Name                                                                             \
    {                                                                                       \
        using AVec             = AV;                                                        \
        using BVec             = BV;                                                        \
        using CVec             = CV;                                                        \
        static constexpr int M = M_, N = N_, K = K_, InBits = 8;                            \
        static constexpr int a_lo = -3, a_hi = 3, b_lo = -2, b_hi = 2, c_lo = -4, c_hi = 4; \
        template <Semantics S, Conversions C>                                               \
        __device__ static Value<CVec, S, C>                                                 \
            call(Value<AVec, S, C> a, Value<BVec, S, C> b, Value<CVec, S, C> c)             \
        {                                                                                   \
            return fpsan::WRAP<0, 0, 0, S, C>(a, b, c);                                     \
        }                                                                                   \
    };                                                                                      \
    MFMA_TESTS(Name)

MFMA_FP8_TRAITS(MfmaF32_16x16x32_FP8_FP8,
                v8e4m3_native,
                v8e4m3_native,
                v4f_native,
                16,
                16,
                32,
                amdgcn_mfma_f32_16x16x32_fp8_fp8)
MFMA_FP8_TRAITS(MfmaF32_16x16x32_FP8_BF8,
                v8e4m3_native,
                v8e5m2_native,
                v4f_native,
                16,
                16,
                32,
                amdgcn_mfma_f32_16x16x32_fp8_bf8)
MFMA_FP8_TRAITS(MfmaF32_16x16x32_BF8_FP8,
                v8e5m2_native,
                v8e4m3_native,
                v4f_native,
                16,
                16,
                32,
                amdgcn_mfma_f32_16x16x32_bf8_fp8)
MFMA_FP8_TRAITS(MfmaF32_16x16x32_BF8_BF8,
                v8e5m2_native,
                v8e5m2_native,
                v4f_native,
                16,
                16,
                32,
                amdgcn_mfma_f32_16x16x32_bf8_bf8)
MFMA_FP8_TRAITS(MfmaF32_32x32x16_FP8_FP8,
                v8e4m3_native,
                v8e4m3_native,
                v16f_native,
                32,
                32,
                16,
                amdgcn_mfma_f32_32x32x16_fp8_fp8)
MFMA_FP8_TRAITS(MfmaF32_32x32x16_FP8_BF8,
                v8e4m3_native,
                v8e5m2_native,
                v16f_native,
                32,
                32,
                16,
                amdgcn_mfma_f32_32x32x16_fp8_bf8)
MFMA_FP8_TRAITS(MfmaF32_32x32x16_BF8_FP8,
                v8e5m2_native,
                v8e4m3_native,
                v16f_native,
                32,
                32,
                16,
                amdgcn_mfma_f32_32x32x16_bf8_fp8)
MFMA_FP8_TRAITS(MfmaF32_32x32x16_BF8_BF8,
                v8e5m2_native,
                v8e5m2_native,
                v16f_native,
                32,
                32,
                16,
                amdgcn_mfma_f32_32x32x16_bf8_bf8)

// Broadcast/permutation modifiers are covered on the legal multi-block legacy
// MFMA shapes below. The dense gfx950 16x16x32/32x32x16 shapes have B=1, so
// non-zero CBSZ would exceed log2(blocks) and is not a useful conformance case.
// xF32 is intentionally absent here: LLVM's gfx950 source-of-truth marks
// v_mfma_f32_*_xf32 unsupported on gfx950. Those wrappers are covered by the
// CDNA3 MFMA tests where the xf32-insts feature exists.

// ---------------------------------------------------------------------------
// F64 16x16x4 MFMA. A and B are *scalar* doubles per lane (not vector
// fragments), C/D are a v4d (4 doubles per lane). Bespoke kernels because the
// operand shape differs from the packed-vector dense shapes above.
//
//   A[i][k]: input_loc(16,4,1,i,k,0,64) -> lane = k*16 + i  (one elem/lane)
//   B[k][j]: input_loc(16,4,1,j,k,0,64) -> lane = k*16 + j
//   D/C[i][j]: output_loc_64(16,16,i,j,0) -> (reg, lane); f64 accumulators
//              occupy VGPR pairs so the v4d slot index is reg/2 (4 elems/lane)
// ---------------------------------------------------------------------------
using fpsan::v4d_native;

static constexpr int F64_M = 16, F64_N = 16, F64_K = 4;

template <Semantics S>
__device__ void load_f64_frags(const double*              A,
                               const double*              B,
                               const double*              C,
                               int                        lane,
                               Value<double, S, kCC>&     a,
                               Value<double, S, kCC>&     b,
                               Value<v4d_native, S, kCC>& c)
{
    double an = 0, bn = 0;
    for(int i = 0; i < F64_M; ++i)
        for(int k = 0; k < F64_K; ++k)
        {
            auto loc = fpsan::detail::input_loc(F64_M, F64_K, 1, i, k, 0, 64);
            if(loc.lane == lane)
                an = A[i * F64_K + k];
        }
    for(int j = 0; j < F64_N; ++j)
        for(int k = 0; k < F64_K; ++k)
        {
            auto loc = fpsan::detail::input_loc(F64_N, F64_K, 1, j, k, 0, 64);
            if(loc.lane == lane)
                bn = B[k * F64_N + j];
        }
    v4d_native cn{};
    for(int i = 0; i < F64_M; ++i)
        for(int j = 0; j < F64_N; ++j)
        {
            auto loc = fpsan::detail::output_loc_64(F64_M, F64_N, i, j, 0);
            if(loc.lane == lane)
                cn[loc.reg / 2] = C[i * F64_N + j];
        }
    a = Value<double, S, kCC>(an);
    b = Value<double, S, kCC>(bn);
    c = Value<v4d_native, S, kCC>(cn);
}

template <Semantics S, class Out, int CBSZ = 0, int ABID = 0, int NEG = 0>
__global__ void k_mfma_f64_16x16x4(const double* A, const double* B, const double* C, Out* D)
{
    int                       lane = threadIdx.x;
    Value<double, S, kCC>     a, b;
    Value<v4d_native, S, kCC> c;
    load_f64_frags<S>(A, B, C, lane, a, b, c);
    auto d = fpsan::amdgcn_mfma_f64_16x16x4f64<CBSZ, ABID, NEG, S, kCC>(a, b, c);
    for(int i = 0; i < F64_M; ++i)
        for(int j = 0; j < F64_N; ++j)
        {
            auto loc = fpsan::detail::output_loc_64(F64_M, F64_N, i, j, 0);
            if(loc.lane == lane)
            {
                if constexpr(S == Semantics::Native)
                    D[i * F64_N + j] = d.get(loc.reg / 2).to_float();
                else
                    D[i * F64_N + j] = d.get(loc.reg / 2).fpsan_payload();
            }
        }
}

namespace
{
    struct F64Mats
    {
        std::vector<double> A, B, C;
    };
    F64Mats make_f64_inputs()
    {
        F64Mats m;
        m.A.resize(F64_M * F64_K);
        m.B.resize(F64_K * F64_N);
        m.C.resize(F64_M * F64_N);
        std::mt19937 rng = fpsan_test::make_rng();
        for(auto& x : m.A)
            x = fpsan_test::pick_int_valued<double>(rng, -3, 3);
        for(auto& x : m.B)
            x = fpsan_test::pick_int_valued<double>(rng, -2, 2);
        for(auto& x : m.C)
            x = fpsan_test::pick_int_valued<double>(rng, -4, 4);
        return m;
    }

    template <class T>
    T host_neg_if(T v, bool neg)
    {
        return neg ? -v : v;
    }
} // namespace

TEST(MfmaF64_16x16x4, LayoutMatchesHardware)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    F64Mats             m = make_f64_inputs();
    std::vector<double> ref(F64_M * F64_N);
    for(int i = 0; i < F64_M; ++i)
        for(int j = 0; j < F64_N; ++j)
        {
            double acc = m.C[i * F64_N + j];
            for(int k = 0; k < F64_K; ++k)
                acc += m.A[i * F64_K + k] * m.B[k * F64_N + j];
            ref[i * F64_N + j] = acc;
        }
    double *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C), *dD;
    HIP_CHECK(hipMalloc(&dD, F64_M * F64_N * sizeof(double)));
    k_mfma_f64_16x16x4<Semantics::Native, double><<<1, WAVE>>>(dA, dB, dC, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<double> got(F64_M * F64_N);
    HIP_CHECK(hipMemcpy(got.data(), dD, F64_M * F64_N * sizeof(double), hipMemcpyDeviceToHost));
    for(int i = 0; i < F64_M * F64_N; ++i)
        EXPECT_EQ(bits_of(got[i]), bits_of(ref[i]))
            << "layout mismatch at " << (i / F64_N) << "," << (i % F64_N);
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
}

TEST(MfmaF64_16x16x4, FpsanMatchesScalarReference)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    F64Mats m = make_f64_inputs();
    using VD  = Value<double, Semantics::Triton, kCC>;
    std::vector<std::uint64_t> ref(F64_M * F64_N);
    for(int i = 0; i < F64_M; ++i)
        for(int j = 0; j < F64_N; ++j)
        {
            VD acc(m.C[i * F64_N + j]);
            for(int k = 0; k < F64_K; ++k)
                acc = acc + VD(m.A[i * F64_K + k]) * VD(m.B[k * F64_N + j]);
            ref[i * F64_N + j] = acc.fpsan_payload();
        }
    double *       dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C);
    std::uint64_t* dD;
    HIP_CHECK(hipMalloc(&dD, F64_M * F64_N * sizeof(std::uint64_t)));
    k_mfma_f64_16x16x4<Semantics::Triton, std::uint64_t><<<1, WAVE>>>(dA, dB, dC, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint64_t> got(F64_M * F64_N);
    HIP_CHECK(
        hipMemcpy(got.data(), dD, F64_M * F64_N * sizeof(std::uint64_t), hipMemcpyDeviceToHost));
    for(int i = 0; i < F64_M * F64_N; ++i)
        EXPECT_EQ(got[i], ref[i]) << "payload mismatch at " << (i / F64_N) << "," << (i % F64_N);
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
}

TEST(MfmaF64_16x16x4_NEG5, LayoutMatchesHardware)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    constexpr int       NEG = 5; // f64 MFMA third immediate is NEG: negate A and C.
    F64Mats             m   = make_f64_inputs();
    std::vector<double> ref(F64_M * F64_N);
    for(int i = 0; i < F64_M; ++i)
        for(int j = 0; j < F64_N; ++j)
        {
            double acc = host_neg_if(m.C[i * F64_N + j], (NEG & 4) != 0);
            for(int k = 0; k < F64_K; ++k)
                acc += host_neg_if(m.A[i * F64_K + k], (NEG & 1) != 0)
                       * host_neg_if(m.B[k * F64_N + j], (NEG & 2) != 0);
            ref[i * F64_N + j] = acc;
        }
    double *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C), *dD;
    HIP_CHECK(hipMalloc(&dD, F64_M * F64_N * sizeof(double)));
    k_mfma_f64_16x16x4<Semantics::Native, double, 0, 0, NEG><<<1, WAVE>>>(dA, dB, dC, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<double> got(F64_M * F64_N);
    HIP_CHECK(hipMemcpy(got.data(), dD, F64_M * F64_N * sizeof(double), hipMemcpyDeviceToHost));
    for(int i = 0; i < F64_M * F64_N; ++i)
        EXPECT_EQ(bits_of(got[i]), bits_of(ref[i]))
            << "NEG mismatch at " << (i / F64_N) << "," << (i % F64_N);
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
}

TEST(MfmaF64_16x16x4_NEG5, FpsanMatchesScalarReference)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    constexpr int NEG = 5;
    F64Mats       m   = make_f64_inputs();
    using VD          = Value<double, Semantics::Triton, kCC>;
    std::vector<std::uint64_t> ref(F64_M * F64_N);
    for(int i = 0; i < F64_M; ++i)
        for(int j = 0; j < F64_N; ++j)
        {
            VD acc = host_neg_if(VD(m.C[i * F64_N + j]), (NEG & 4) != 0);
            for(int k = 0; k < F64_K; ++k)
                acc = acc
                      + host_neg_if(VD(m.A[i * F64_K + k]), (NEG & 1) != 0)
                            * host_neg_if(VD(m.B[k * F64_N + j]), (NEG & 2) != 0);
            ref[i * F64_N + j] = acc.fpsan_payload();
        }
    double *       dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C);
    std::uint64_t* dD;
    HIP_CHECK(hipMalloc(&dD, F64_M * F64_N * sizeof(std::uint64_t)));
    k_mfma_f64_16x16x4<Semantics::Triton, std::uint64_t, 0, 0, NEG><<<1, WAVE>>>(dA, dB, dC, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint64_t> got(F64_M * F64_N);
    HIP_CHECK(
        hipMemcpy(got.data(), dD, F64_M * F64_N * sizeof(std::uint64_t), hipMemcpyDeviceToHost));
    for(int i = 0; i < F64_M * F64_N; ++i)
        EXPECT_EQ(got[i], ref[i]) << "payload mismatch at " << (i / F64_N) << "," << (i % F64_N);
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
}

// ---------------------------------------------------------------------------
// Scaled f8f6f4 MFMA (16x16x128 / 32x32x64). The Float path forwards to the
// real builtin with the corrected 9-arg signature (A/B as v8i32, no ABID); the
// FPSan path runs the OCP-MX scaled dataflow (see amdgcn_mfma.hpp). This is a
// minimal Float smoke launch that asserts the builtin executes cleanly on
// MI350; the exhaustive Float-vs-hardware and FPSan-vs-scalar-reference checks
// live in the ScaledMfma16x16x128_* / ScaledMfma32x32x64_* suites further down.
// ---------------------------------------------------------------------------
using fpsan::v32e4m3_native;

template <Semantics S>
__global__ void k_scale_16x16x128(const fpsan::v32e4m3_native* A,
                                  const fpsan::v32e4m3_native* B,
                                  const v4f_native*            C,
                                  v4f_native*                  D,
                                  int                          sa,
                                  int                          sb)
{
    Value<fpsan::v32e4m3_native, S, kCC> a{A[0]}, b{B[0]};
    Value<v4f_native, S, kCC>            c{C[0]};
    auto d = fpsan::amdgcn_mfma_scale_f32_16x16x128_f8f6f4<0, 0, 0, 0, 0, S, kCC>(a, b, c, sa, sb);
    if constexpr(S == Semantics::Native)
        D[0] = static_cast<v4f_native>(d);
}

template <Semantics S>
__global__ void k_scale_32x32x64(const fpsan::v32e4m3_native* A,
                                 const fpsan::v32e4m3_native* B,
                                 const v16f_native*           C,
                                 v16f_native*                 D,
                                 int                          sa,
                                 int                          sb)
{
    Value<fpsan::v32e4m3_native, S, kCC> a{A[0]}, b{B[0]};
    Value<v16f_native, S, kCC>           c{C[0]};
    auto d = fpsan::amdgcn_mfma_scale_f32_32x32x64_f8f6f4<0, 0, 0, 0, 0, S, kCC>(a, b, c, sa, sb);
    if constexpr(S == Semantics::Native)
        D[0] = static_cast<v16f_native>(d);
}

TEST(ScaledMfma, F8F6F4_16x16x128_FloatLaunches)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    fpsan::v32e4m3_native *dA, *dB;
    v4f_native *           dC, *dD;
    HIP_CHECK(hipMalloc(&dA, sizeof(*dA)));
    HIP_CHECK(hipMalloc(&dB, sizeof(*dB)));
    HIP_CHECK(hipMalloc(&dC, sizeof(*dC)));
    HIP_CHECK(hipMalloc(&dD, sizeof(*dD)));
    HIP_CHECK(hipMemset(dA, 0, sizeof(*dA)));
    HIP_CHECK(hipMemset(dB, 0, sizeof(*dB)));
    HIP_CHECK(hipMemset(dC, 0, sizeof(*dC)));
    k_scale_16x16x128<Semantics::Native><<<1, WAVE>>>(dA, dB, dC, dD, 0x7F, 0x7F);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipGetLastError());
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
}

TEST(ScaledMfma, F8F6F4_32x32x64_FloatLaunches)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    fpsan::v32e4m3_native *dA, *dB;
    v16f_native *          dC, *dD;
    HIP_CHECK(hipMalloc(&dA, sizeof(*dA)));
    HIP_CHECK(hipMalloc(&dB, sizeof(*dB)));
    HIP_CHECK(hipMalloc(&dC, sizeof(*dC)));
    HIP_CHECK(hipMalloc(&dD, sizeof(*dD)));
    HIP_CHECK(hipMemset(dA, 0, sizeof(*dA)));
    HIP_CHECK(hipMemset(dB, 0, sizeof(*dB)));
    HIP_CHECK(hipMemset(dC, 0, sizeof(*dC)));
    k_scale_32x32x64<Semantics::Native><<<1, WAVE>>>(dA, dB, dC, dD, 0x7F, 0x7F);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipGetLastError());
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
}

// ---------------------------------------------------------------------------
// Sparse MFMA (SMFMAC), f16/bf16. The Float path forwards to the builtin
// (6-arg signature: A, B, C, idx, CBSZ, ABID -- verified correct on gfx950);
// the FPSan path runs the sparse 2:4 software dataflow (see amdgcn_smfmac.hpp).
//
// This particular case is a Float sanity check that holds for any idx: with
// A == 0 every selected product is 0, so D == C. It exercises the Float builtin
// end-to-end and the C/D output layout. The exhaustive FPSan-vs-scalar-reference
// checks live in the SmfmacF16_*/SmfmacFp8_* suites further down.
// ---------------------------------------------------------------------------
using fpsan::v16bf_native;
using fpsan::v16h_native;

template <Semantics S, class AVec, class BVec, class CVec, int M_, int N_, class Wrap>
__global__ void k_smfmac_zeroA(const CVec* Cin, CVec* Dout, int idx, Wrap wrap)
{
    int                 lane = threadIdx.x;
    Value<AVec, S, kCC> a{AVec{}}; // A = 0
    Value<BVec, S, kCC> b{BVec{}}; // B = 0
    Value<CVec, S, kCC> c{Cin[lane]}; // per-lane C fragment, verbatim
    auto                d = wrap(a, b, c, idx);
    if constexpr(S == Semantics::Native)
        Dout[lane] = static_cast<CVec>(d);
}

namespace
{
    // Wrap each SMFMAC opcode in a functor so a single kernel template covers all.
    struct Smf_16x16x64_f16
    {
        template <Semantics S, class C>
        __device__ auto operator()(Value<fpsan::v8h_native, S, kCC>  a,
                                   Value<fpsan::v16h_native, S, kCC> b,
                                   C                                 c,
                                   int                               idx) const
        {
            return fpsan::amdgcn_smfmac_f32_16x16x64_f16<0, 0, S, kCC>(a, b, c, idx);
        }
    };
    struct Smf_32x32x32_f16
    {
        template <Semantics S, class C>
        __device__ auto operator()(Value<fpsan::v8h_native, S, kCC>  a,
                                   Value<fpsan::v16h_native, S, kCC> b,
                                   C                                 c,
                                   int                               idx) const
        {
            return fpsan::amdgcn_smfmac_f32_32x32x32_f16<0, 0, S, kCC>(a, b, c, idx);
        }
    };
} // namespace

template <class Wrap, class CVec, int LANEREGS>
void run_smfmac_zeroA()
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    // CVec is the per-lane accumulator fragment (v4f / v16f). Fill C with a
    // distinct value per (lane, reg) and require D == C.
    std::vector<CVec> hC(WAVE), hD(WAVE);
    for(int l = 0; l < WAVE; ++l)
        for(int r = 0; r < LANEREGS; ++r)
            hC[l][r] = static_cast<float>((l * LANEREGS + r) % 17 - 8);
    CVec *dC, *dD;
    HIP_CHECK(hipMalloc(&dC, WAVE * sizeof(CVec)));
    HIP_CHECK(hipMalloc(&dD, WAVE * sizeof(CVec)));
    HIP_CHECK(hipMemcpy(dC, hC.data(), WAVE * sizeof(CVec), hipMemcpyHostToDevice));
    k_smfmac_zeroA<Semantics::Native, fpsan::v8h_native, fpsan::v16h_native, CVec, 0, 0, Wrap>
        <<<1, WAVE>>>(dC, dD, 0, Wrap{});
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpy(hD.data(), dD, WAVE * sizeof(CVec), hipMemcpyDeviceToHost));
    for(int l = 0; l < WAVE; ++l)
        for(int r = 0; r < LANEREGS; ++r)
            EXPECT_EQ(bits_of(hD[l][r]), bits_of(hC[l][r])) << "lane " << l << " reg " << r;
    (void)hipFree(dC);
    (void)hipFree(dD);
}

TEST(Smfmac, F32_16x16x64_F16_ZeroAGivesC)
{
    run_smfmac_zeroA<Smf_16x16x64_f16, v4f_native, 4>();
}
TEST(Smfmac, F32_32x32x32_F16_ZeroAGivesC)
{
    run_smfmac_zeroA<Smf_32x32x32_f16, v16f_native, 16>();
}

// ---------------------------------------------------------------------------
// F64 4x4x4 MFMA (V_MFMA_F64_4X4X4_4B_F64): 4 independent 4x4 blocks, each a
// K=4 contraction, scalar A/B/C/D per lane. Layout reverse-engineered on real
// MI350 (the dense input_loc/output_loc do NOT apply -- K=4 < 16 lanes/block):
//   output lane L holds D[i][j] of block b: i=L/16, b=(L%16)/4, j=L%4
//   A[i][k] of block b at lane 16k+4b+i; B[k][j] of block b at lane 16k+4b+j
//   D[L] = C[L] + sum_{k=0..3} A[16k+4b+i] * B[16k+4b+j]
// ---------------------------------------------------------------------------
template <Semantics S, class Out, int CBSZ = 0, int ABID = 0, int NEG = 0>
__global__ void k_mfma_f64_4x4x4(const double* A, const double* B, const double* C, Out* D)
{
    int                   lane = threadIdx.x;
    Value<double, S, kCC> a{A[lane]}, b{B[lane]}, c{C[lane]};
    auto                  d = fpsan::amdgcn_mfma_f64_4x4x4f64<CBSZ, ABID, NEG, S, kCC>(a, b, c);
    if constexpr(S == Semantics::Native)
        D[lane] = d.to_float();
    else
        D[lane] = d.fpsan_payload();
}

namespace
{
    std::vector<double> make_f64_4x4x4_vec(int seed)
    {
        std::vector<double> v(WAVE);
        std::mt19937        rng = fpsan_test::make_rng();
        for(int s = 0; s < seed; ++s)
            (void)rng();
        for(auto& x : v)
            x = fpsan_test::pick_int_valued<double>(rng, -4, 4);
        return v;
    }
} // namespace

TEST(MfmaF64_4x4x4, LayoutMatchesHardware)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    auto A = make_f64_4x4x4_vec(0), B = make_f64_4x4x4_vec(1), C = make_f64_4x4x4_vec(2);
    std::vector<double> ref(WAVE);
    for(int L = 0; L < WAVE; ++L)
    {
        int    i = L / 16, b = (L % 16) / 4, j = L % 4;
        double acc = C[L];
        for(int k = 0; k < 4; ++k)
            acc += A[16 * k + 4 * b + i] * B[16 * k + 4 * b + j];
        ref[L] = acc;
    }
    double *dA = to_dev(A), *dB = to_dev(B), *dC = to_dev(C), *dD;
    HIP_CHECK(hipMalloc(&dD, WAVE * sizeof(double)));
    k_mfma_f64_4x4x4<Semantics::Native, double><<<1, WAVE>>>(dA, dB, dC, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<double> got(WAVE);
    HIP_CHECK(hipMemcpy(got.data(), dD, WAVE * sizeof(double), hipMemcpyDeviceToHost));
    for(int L = 0; L < WAVE; ++L)
        EXPECT_EQ(bits_of(got[L]), bits_of(ref[L])) << "lane " << L;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
}

TEST(MfmaF64_4x4x4, FpsanMatchesScalarReference)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    auto A = make_f64_4x4x4_vec(0), B = make_f64_4x4x4_vec(1), C = make_f64_4x4x4_vec(2);
    using VD = Value<double, Semantics::Triton, kCC>;
    std::vector<std::uint64_t> ref(WAVE);
    for(int L = 0; L < WAVE; ++L)
    {
        int i = L / 16, b = (L % 16) / 4, j = L % 4;
        VD  acc(C[L]);
        for(int k = 0; k < 4; ++k)
            acc = acc + VD(A[16 * k + 4 * b + i]) * VD(B[16 * k + 4 * b + j]);
        ref[L] = acc.fpsan_payload();
    }
    double *       dA = to_dev(A), *dB = to_dev(B), *dC = to_dev(C);
    std::uint64_t* dD;
    HIP_CHECK(hipMalloc(&dD, WAVE * sizeof(std::uint64_t)));
    k_mfma_f64_4x4x4<Semantics::Triton, std::uint64_t><<<1, WAVE>>>(dA, dB, dC, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint64_t> got(WAVE);
    HIP_CHECK(hipMemcpy(got.data(), dD, WAVE * sizeof(std::uint64_t), hipMemcpyDeviceToHost));
    for(int L = 0; L < WAVE; ++L)
        EXPECT_EQ(got[L], ref[L]) << "lane " << L;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
}

TEST(MfmaF64_4x4x4_NEG5, LayoutMatchesHardware)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    constexpr int NEG = 5; // f64 MFMA third immediate is NEG: negate A and C.
    auto          A = make_f64_4x4x4_vec(0), B = make_f64_4x4x4_vec(1), C = make_f64_4x4x4_vec(2);
    std::vector<double> ref(WAVE);
    for(int L = 0; L < WAVE; ++L)
    {
        int    i = L / 16, b = (L % 16) / 4, j = L % 4;
        double acc = host_neg_if(C[L], (NEG & 4) != 0);
        for(int k = 0; k < 4; ++k)
        {
            acc += host_neg_if(A[16 * k + 4 * b + i], (NEG & 1) != 0)
                   * host_neg_if(B[16 * k + 4 * b + j], (NEG & 2) != 0);
        }
        ref[L] = acc;
    }
    double *dA = to_dev(A), *dB = to_dev(B), *dC = to_dev(C), *dD;
    HIP_CHECK(hipMalloc(&dD, WAVE * sizeof(double)));
    k_mfma_f64_4x4x4<Semantics::Native, double, 0, 0, NEG><<<1, WAVE>>>(dA, dB, dC, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<double> got(WAVE);
    HIP_CHECK(hipMemcpy(got.data(), dD, WAVE * sizeof(double), hipMemcpyDeviceToHost));
    for(int L = 0; L < WAVE; ++L)
        EXPECT_EQ(bits_of(got[L]), bits_of(ref[L])) << "lane " << L;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
}

TEST(MfmaF64_4x4x4_NEG5, FpsanMatchesScalarReference)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    constexpr int NEG = 5;
    auto          A = make_f64_4x4x4_vec(0), B = make_f64_4x4x4_vec(1), C = make_f64_4x4x4_vec(2);
    using VD = Value<double, Semantics::Triton, kCC>;
    std::vector<std::uint64_t> ref(WAVE);
    for(int L = 0; L < WAVE; ++L)
    {
        int i = L / 16, b = (L % 16) / 4, j = L % 4;
        VD  acc = host_neg_if(VD(C[L]), (NEG & 4) != 0);
        for(int k = 0; k < 4; ++k)
        {
            acc = acc
                  + host_neg_if(VD(A[16 * k + 4 * b + i]), (NEG & 1) != 0)
                        * host_neg_if(VD(B[16 * k + 4 * b + j]), (NEG & 2) != 0);
        }
        ref[L] = acc.fpsan_payload();
    }
    double *       dA = to_dev(A), *dB = to_dev(B), *dC = to_dev(C);
    std::uint64_t* dD;
    HIP_CHECK(hipMalloc(&dD, WAVE * sizeof(std::uint64_t)));
    k_mfma_f64_4x4x4<Semantics::Triton, std::uint64_t, 0, 0, NEG><<<1, WAVE>>>(dA, dB, dC, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint64_t> got(WAVE);
    HIP_CHECK(hipMemcpy(got.data(), dD, WAVE * sizeof(std::uint64_t), hipMemcpyDeviceToHost));
    for(int L = 0; L < WAVE; ++L)
        EXPECT_EQ(got[L], ref[L]) << "lane " << L;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
}

// ---------------------------------------------------------------------------
// Legacy gfx9 f32-input MFMA shapes (16x16x4, 16x16x1, 32x32x2, 32x32x1,
// 4x4x1). SCALAR f32 A/B per lane, `B` independent MxN blocks. Layouts use
// input_loc / output_loc_32 (silicon-verified, full-block random match). Both
// the Float builtin and the FPSan dataflow are pinned here.
// ---------------------------------------------------------------------------
template <class Traits>
__global__ void k_legf32(const float* A, const float* B, const float* C, float* D)
{
    using T    = Traits;
    int   lane = threadIdx.x;
    float an = 0, bn = 0;
    for(int b = 0; b < T::Bk; ++b)
        for(int i = 0; i < T::M; ++i)
            for(int k = 0; k < T::K; ++k)
            {
                auto l = fpsan::detail::input_loc(T::M, T::K, T::Bk, i, k, b, 32);
                if(l.lane == lane)
                    an = A[(b * T::M + i) * T::K + k];
            }
    for(int b = 0; b < T::Bk; ++b)
        for(int j = 0; j < T::N; ++j)
            for(int k = 0; k < T::K; ++k)
            {
                auto l = fpsan::detail::input_loc(T::N, T::K, T::Bk, j, k, b, 32);
                if(l.lane == lane)
                    bn = B[(b * T::K + k) * T::N + j];
            }
    typename T::CVec cn{};
    for(int b = 0; b < T::Bk; ++b)
        for(int i = 0; i < T::M; ++i)
            for(int j = 0; j < T::N; ++j)
            {
                auto o = fpsan::detail::output_loc_32(T::M, T::N, i, j, b);
                if(o.lane == lane)
                    cn[o.reg] = C[(b * T::M + i) * T::N + j];
            }
    Value<float, Semantics::Native, kCC>            a{an}, b2{bn};
    Value<typename T::CVec, Semantics::Native, kCC> c{cn};
    auto d = T::template call<Semantics::Native, kCC>(a, b2, c);
    for(int b = 0; b < T::Bk; ++b)
        for(int i = 0; i < T::M; ++i)
            for(int j = 0; j < T::N; ++j)
            {
                auto o = fpsan::detail::output_loc_32(T::M, T::N, i, j, b);
                if(o.lane == lane)
                    D[(b * T::M + i) * T::N + j] = d.get(o.reg).to_float();
            }
}

template <class Traits>
__global__ void k_legf32_p(const float* A, const float* B, const float* C, std::uint32_t* D)
{
    using T    = Traits;
    int   lane = threadIdx.x;
    float an = 0, bn = 0;
    for(int b = 0; b < T::Bk; ++b)
        for(int i = 0; i < T::M; ++i)
            for(int k = 0; k < T::K; ++k)
            {
                auto l = fpsan::detail::input_loc(T::M, T::K, T::Bk, i, k, b, 32);
                if(l.lane == lane)
                    an = A[(b * T::M + i) * T::K + k];
            }
    for(int b = 0; b < T::Bk; ++b)
        for(int j = 0; j < T::N; ++j)
            for(int k = 0; k < T::K; ++k)
            {
                auto l = fpsan::detail::input_loc(T::N, T::K, T::Bk, j, k, b, 32);
                if(l.lane == lane)
                    bn = B[(b * T::K + k) * T::N + j];
            }
    typename T::CVec cn{};
    for(int b = 0; b < T::Bk; ++b)
        for(int i = 0; i < T::M; ++i)
            for(int j = 0; j < T::N; ++j)
            {
                auto o = fpsan::detail::output_loc_32(T::M, T::N, i, j, b);
                if(o.lane == lane)
                    cn[o.reg] = C[(b * T::M + i) * T::N + j];
            }
    Value<float, Semantics::Triton, kCC>            a{an}, b2{bn};
    Value<typename T::CVec, Semantics::Triton, kCC> c{cn};
    auto d = T::template call<Semantics::Triton, kCC>(a, b2, c);
    for(int b = 0; b < T::Bk; ++b)
        for(int i = 0; i < T::M; ++i)
            for(int j = 0; j < T::N; ++j)
            {
                auto o = fpsan::detail::output_loc_32(T::M, T::N, i, j, b);
                if(o.lane == lane)
                    D[(b * T::M + i) * T::N + j] = d.get(o.reg).fpsan_payload();
            }
}

namespace
{
    template <class Traits>
    struct LegF32Data
    {
        std::vector<float> A, B, C;
    };
    template <class T>
    LegF32Data<T> make_legf32()
    {
        LegF32Data<T> d;
        d.A.resize(T::Bk * T::M * T::K);
        d.B.resize(T::Bk * T::K * T::N);
        d.C.resize(T::Bk * T::M * T::N);
        std::mt19937 rng = fpsan_test::make_rng();
        for(auto& x : d.A)
            x = fpsan_test::pick_int(rng, -3, 3);
        for(auto& x : d.B)
            x = fpsan_test::pick_int(rng, -2, 2);
        for(auto& x : d.C)
            x = fpsan_test::pick_int(rng, -3, 3);
        return d;
    }
} // namespace

template <class T>
float host_legf32_a_value(const std::vector<float>& A, int b, int i, int k)
{
    auto      want = fpsan::detail::input_loc(T::M, T::K, T::Bk, i, k, b, 32);
    const int src_lane
        = host_mfma_a_src_lane(want.lane, MfmaTraitCBSZ<T>::value, MfmaTraitABID<T>::value);
    for(int bb = 0; bb < T::Bk; ++bb)
        for(int ii = 0; ii < T::M; ++ii)
            for(int kk = 0; kk < T::K; ++kk)
            {
                auto loc = fpsan::detail::input_loc(T::M, T::K, T::Bk, ii, kk, bb, 32);
                if(loc.lane == src_lane && loc.reg == want.reg && loc.sub == want.sub)
                    return A[(bb * T::M + ii) * T::K + kk];
            }
    return 0.0f;
}

template <class T>
float host_legf32_b_value(const std::vector<float>& B, int b, int k, int j)
{
    auto      want     = fpsan::detail::input_loc(T::N, T::K, T::Bk, j, k, b, 32);
    const int src_lane = host_mfma_b_src_lane(want.lane, MfmaTraitBLGP<T>::value);
    for(int bb = 0; bb < T::Bk; ++bb)
        for(int jj = 0; jj < T::N; ++jj)
            for(int kk = 0; kk < T::K; ++kk)
            {
                auto loc = fpsan::detail::input_loc(T::N, T::K, T::Bk, jj, kk, bb, 32);
                if(loc.lane == src_lane && loc.reg == want.reg && loc.sub == want.sub)
                    return B[(bb * T::K + kk) * T::N + jj];
            }
    return 0.0f;
}

template <class T>
void run_legf32_layout()
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    LegF32Data<T>      m = make_legf32<T>();
    std::vector<float> ref(T::Bk * T::M * T::N);
    for(int b = 0; b < T::Bk; ++b)
        for(int i = 0; i < T::M; ++i)
            for(int j = 0; j < T::N; ++j)
            {
                double acc = m.C[(b * T::M + i) * T::N + j];
                for(int k = 0; k < T::K; ++k)
                    acc += (double)host_legf32_a_value<T>(m.A, b, i, k)
                           * (double)host_legf32_b_value<T>(m.B, b, k, j);
                ref[(b * T::M + i) * T::N + j] = (float)acc;
            }
    float *   dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C), *dD;
    const int total = T::Bk * T::M * T::N;
    HIP_CHECK(hipMalloc(&dD, total * sizeof(float)));
    k_legf32<T><<<1, WAVE>>>(dA, dB, dC, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> got(total);
    HIP_CHECK(hipMemcpy(got.data(), dD, total * sizeof(float), hipMemcpyDeviceToHost));
    for(int t = 0; t < total; ++t)
        EXPECT_EQ(bits_of(got[t]), bits_of(ref[t])) << "elem " << t;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
}

template <class T>
void run_legf32_fpsan()
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    LegF32Data<T> m = make_legf32<T>();
    using VF        = Value<float, Semantics::Triton, kCC>;
    std::vector<std::uint32_t> ref(T::Bk * T::M * T::N);
    for(int b = 0; b < T::Bk; ++b)
        for(int i = 0; i < T::M; ++i)
            for(int j = 0; j < T::N; ++j)
            {
                VF acc(m.C[(b * T::M + i) * T::N + j]);
                for(int k = 0; k < T::K; ++k)
                    acc = acc
                          + VF(host_legf32_a_value<T>(m.A, b, i, k))
                                * VF(host_legf32_b_value<T>(m.B, b, k, j));
                ref[(b * T::M + i) * T::N + j] = acc.fpsan_payload();
            }
    float *        dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C);
    std::uint32_t* dD;
    const int      total = T::Bk * T::M * T::N;
    HIP_CHECK(hipMalloc(&dD, total * sizeof(std::uint32_t)));
    k_legf32_p<T><<<1, WAVE>>>(dA, dB, dC, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(total);
    HIP_CHECK(hipMemcpy(got.data(), dD, total * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for(int t = 0; t < total; ++t)
        EXPECT_EQ(got[t], ref[t]) << "elem " << t;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
}

#define LEGF32_TRAITS(Name, M_, N_, K_, B_, CV, WRAP)                             \
    struct Name                                                                   \
    {                                                                             \
        using CVec             = CV;                                              \
        static constexpr int M = M_, N = N_, K = K_, Bk = B_;                     \
        static constexpr int CBSZ = 0, ABID = 0, BLGP = 0;                        \
        template <Semantics S, Conversions C>                                     \
        __device__ static Value<CVec, S, C>                                       \
            call(Value<float, S, C> a, Value<float, S, C> b, Value<CVec, S, C> c) \
        {                                                                         \
            return fpsan::WRAP<CBSZ, ABID, BLGP, S, C>(a, b, c);                  \
        }                                                                         \
    };                                                                            \
    TEST(Name, LayoutMatchesHardware)                                             \
    {                                                                             \
        run_legf32_layout<Name>();                                                \
    }                                                                             \
    TEST(Name, FpsanMatchesScalarReference)                                       \
    {                                                                             \
        run_legf32_fpsan<Name>();                                                 \
    }

LEGF32_TRAITS(LegacyMfmaF32_16x16x4, 16, 16, 4, 1, v4f_native, amdgcn_mfma_f32_16x16x4f32)
LEGF32_TRAITS(LegacyMfmaF32_16x16x1, 16, 16, 1, 4, v16f_native, amdgcn_mfma_f32_16x16x1f32)
LEGF32_TRAITS(LegacyMfmaF32_32x32x2, 32, 32, 2, 1, v16f_native, amdgcn_mfma_f32_32x32x2f32)
LEGF32_TRAITS(LegacyMfmaF32_32x32x1, 32, 32, 1, 2, fpsan::v32f_native, amdgcn_mfma_f32_32x32x1f32)
LEGF32_TRAITS(LegacyMfmaF32_4x4x1, 4, 4, 1, 16, v4f_native, amdgcn_mfma_f32_4x4x1f32)

struct LegacyMfmaF32_16x16x1_CBSZ2_ABID3_BLGP5
{
    using CVec             = v16f_native;
    static constexpr int M = 16, N = 16, K = 1, Bk = 4;
    static constexpr int CBSZ = 2, ABID = 3, BLGP = 5;
    template <Semantics S, Conversions C>
    __device__ static Value<CVec, S, C>
        call(Value<float, S, C> a, Value<float, S, C> b, Value<CVec, S, C> c)
    {
        return fpsan::amdgcn_mfma_f32_16x16x1f32<CBSZ, ABID, BLGP, S, C>(a, b, c);
    }
};
TEST(LegacyMfmaF32_16x16x1_CBSZ2_ABID3_BLGP5, LayoutMatchesHardware)
{
    run_legf32_layout<LegacyMfmaF32_16x16x1_CBSZ2_ABID3_BLGP5>();
}
TEST(LegacyMfmaF32_16x16x1_CBSZ2_ABID3_BLGP5, FpsanMatchesScalarReference)
{
    run_legf32_fpsan<LegacyMfmaF32_16x16x1_CBSZ2_ABID3_BLGP5>();
}

struct LegacyMfmaF32_32x32x1_CBSZ1_ABID1_BLGP1
{
    using CVec             = fpsan::v32f_native;
    static constexpr int M = 32, N = 32, K = 1, Bk = 2;
    static constexpr int CBSZ = 1, ABID = 1, BLGP = 1;
    template <Semantics S, Conversions C>
    __device__ static Value<CVec, S, C>
        call(Value<float, S, C> a, Value<float, S, C> b, Value<CVec, S, C> c)
    {
        return fpsan::amdgcn_mfma_f32_32x32x1f32<CBSZ, ABID, BLGP, S, C>(a, b, c);
    }
};
TEST(LegacyMfmaF32_32x32x1_CBSZ1_ABID1_BLGP1, LayoutMatchesHardware)
{
    run_legf32_layout<LegacyMfmaF32_32x32x1_CBSZ1_ABID1_BLGP1>();
}
TEST(LegacyMfmaF32_32x32x1_CBSZ1_ABID1_BLGP1, FpsanMatchesScalarReference)
{
    run_legf32_fpsan<LegacyMfmaF32_32x32x1_CBSZ1_ABID1_BLGP1>();
}

// ---------------------------------------------------------------------------
// Legacy gfx9 f16 / bf16_1k MFMA shapes. A,B per-lane fragment is 4 elements
// of f16 (or bf16). Layouts use input_loc (bits=16, element = 2*reg+sub) /
// output_loc_32 with `B` independent MxN blocks (silicon-verified, full-block
// random match). Both the Float builtin and the FPSan dataflow are pinned here.
// The non-1k bf16 shapes are gfx908-only (cannot select on gfx950) and are
// intentionally omitted.
// ---------------------------------------------------------------------------
template <class Traits>
__global__ void k_legf16(const float* A, const float* B, const float* C, float* D)
{
    using T               = Traits;
    using AE              = typename T::AElem;
    int              lane = threadIdx.x;
    typename T::AVec an{}, bn{};
    for(int b = 0; b < T::Bk; ++b)
        for(int i = 0; i < T::M; ++i)
            for(int k = 0; k < T::K; ++k)
            {
                auto l = fpsan::detail::input_loc(T::M, T::K, T::Bk, i, k, b, 16);
                if(l.lane == lane)
                    an[2 * l.reg + l.sub] = AE(A[(b * T::M + i) * T::K + k]);
            }
    for(int b = 0; b < T::Bk; ++b)
        for(int j = 0; j < T::N; ++j)
            for(int k = 0; k < T::K; ++k)
            {
                auto l = fpsan::detail::input_loc(T::N, T::K, T::Bk, j, k, b, 16);
                if(l.lane == lane)
                    bn[2 * l.reg + l.sub] = AE(B[(b * T::K + k) * T::N + j]);
            }
    typename T::CVec cn{};
    for(int b = 0; b < T::Bk; ++b)
        for(int i = 0; i < T::M; ++i)
            for(int j = 0; j < T::N; ++j)
            {
                auto o = fpsan::detail::output_loc_32(T::M, T::N, i, j, b);
                if(o.lane == lane)
                    cn[o.reg] = C[(b * T::M + i) * T::N + j];
            }
    Value<typename T::AVec, Semantics::Native, kCC> a{an}, b2{bn};
    Value<typename T::CVec, Semantics::Native, kCC> c{cn};
    auto d = T::template call<Semantics::Native, kCC>(a, b2, c);
    for(int b = 0; b < T::Bk; ++b)
        for(int i = 0; i < T::M; ++i)
            for(int j = 0; j < T::N; ++j)
            {
                auto o = fpsan::detail::output_loc_32(T::M, T::N, i, j, b);
                if(o.lane == lane)
                    D[(b * T::M + i) * T::N + j] = d.get(o.reg).to_float();
            }
}

template <class Traits>
__global__ void k_legf16_p(const float* A, const float* B, const float* C, std::uint32_t* D)
{
    using T               = Traits;
    using AE              = typename T::AElem;
    int              lane = threadIdx.x;
    typename T::AVec an{}, bn{};
    for(int b = 0; b < T::Bk; ++b)
        for(int i = 0; i < T::M; ++i)
            for(int k = 0; k < T::K; ++k)
            {
                auto l = fpsan::detail::input_loc(T::M, T::K, T::Bk, i, k, b, 16);
                if(l.lane == lane)
                    an[2 * l.reg + l.sub] = AE(A[(b * T::M + i) * T::K + k]);
            }
    for(int b = 0; b < T::Bk; ++b)
        for(int j = 0; j < T::N; ++j)
            for(int k = 0; k < T::K; ++k)
            {
                auto l = fpsan::detail::input_loc(T::N, T::K, T::Bk, j, k, b, 16);
                if(l.lane == lane)
                    bn[2 * l.reg + l.sub] = AE(B[(b * T::K + k) * T::N + j]);
            }
    typename T::CVec cn{};
    for(int b = 0; b < T::Bk; ++b)
        for(int i = 0; i < T::M; ++i)
            for(int j = 0; j < T::N; ++j)
            {
                auto o = fpsan::detail::output_loc_32(T::M, T::N, i, j, b);
                if(o.lane == lane)
                    cn[o.reg] = C[(b * T::M + i) * T::N + j];
            }
    Value<typename T::AVec, Semantics::Triton, kCC> a{an}, b2{bn};
    Value<typename T::CVec, Semantics::Triton, kCC> c{cn};
    auto d = T::template call<Semantics::Triton, kCC>(a, b2, c);
    for(int b = 0; b < T::Bk; ++b)
        for(int i = 0; i < T::M; ++i)
            for(int j = 0; j < T::N; ++j)
            {
                auto o = fpsan::detail::output_loc_32(T::M, T::N, i, j, b);
                if(o.lane == lane)
                    D[(b * T::M + i) * T::N + j] = d.get(o.reg).fpsan_payload();
            }
}

template <class T>
float host_legf16_a_value(const std::vector<float>& A, int b, int i, int k)
{
    auto      want = fpsan::detail::input_loc(T::M, T::K, T::Bk, i, k, b, 16);
    const int src_lane
        = host_mfma_a_src_lane(want.lane, MfmaTraitCBSZ<T>::value, MfmaTraitABID<T>::value);
    for(int bb = 0; bb < T::Bk; ++bb)
        for(int ii = 0; ii < T::M; ++ii)
            for(int kk = 0; kk < T::K; ++kk)
            {
                auto loc = fpsan::detail::input_loc(T::M, T::K, T::Bk, ii, kk, bb, 16);
                if(loc.lane == src_lane && loc.reg == want.reg && loc.sub == want.sub)
                    return A[(bb * T::M + ii) * T::K + kk];
            }
    return 0.0f;
}

template <class T>
float host_legf16_b_value(const std::vector<float>& B, int b, int k, int j)
{
    auto      want     = fpsan::detail::input_loc(T::N, T::K, T::Bk, j, k, b, 16);
    const int src_lane = host_mfma_b_src_lane(want.lane, MfmaTraitBLGP<T>::value);
    for(int bb = 0; bb < T::Bk; ++bb)
        for(int jj = 0; jj < T::N; ++jj)
            for(int kk = 0; kk < T::K; ++kk)
            {
                auto loc = fpsan::detail::input_loc(T::N, T::K, T::Bk, jj, kk, bb, 16);
                if(loc.lane == src_lane && loc.reg == want.reg && loc.sub == want.sub)
                    return B[(bb * T::K + kk) * T::N + jj];
            }
    return 0.0f;
}

template <class T>
void run_legf16_layout()
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    LegF32Data<T>      m = make_legf32<T>();
    std::vector<float> ref(T::Bk * T::M * T::N);
    for(int b = 0; b < T::Bk; ++b)
        for(int i = 0; i < T::M; ++i)
            for(int j = 0; j < T::N; ++j)
            {
                double acc = m.C[(b * T::M + i) * T::N + j];
                for(int k = 0; k < T::K; ++k)
                    acc += (double)host_legf16_a_value<T>(m.A, b, i, k)
                           * (double)host_legf16_b_value<T>(m.B, b, k, j);
                ref[(b * T::M + i) * T::N + j] = (float)acc;
            }
    float *   dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C), *dD;
    const int total = T::Bk * T::M * T::N;
    HIP_CHECK(hipMalloc(&dD, total * sizeof(float)));
    k_legf16<T><<<1, WAVE>>>(dA, dB, dC, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> got(total);
    HIP_CHECK(hipMemcpy(got.data(), dD, total * sizeof(float), hipMemcpyDeviceToHost));
    for(int t = 0; t < total; ++t)
        EXPECT_EQ(bits_of(got[t]), bits_of(ref[t])) << "elem " << t;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
}

template <class T>
void run_legf16_fpsan()
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    LegF32Data<T> m = make_legf32<T>();
    using AE        = typename T::AElem;
    using VF        = Value<float, Semantics::Triton, kCC>;
    std::vector<std::uint32_t> ref(T::Bk * T::M * T::N);
    for(int b = 0; b < T::Bk; ++b)
        for(int i = 0; i < T::M; ++i)
            for(int j = 0; j < T::N; ++j)
            {
                VF acc(m.C[(b * T::M + i) * T::N + j]);
                for(int k = 0; k < T::K; ++k)
                    acc = acc
                          + fpsan::cast<float>(Value<AE, Semantics::Triton, kCC>(
                                AE(host_legf16_a_value<T>(m.A, b, i, k))))
                                * fpsan::cast<float>(Value<AE, Semantics::Triton, kCC>(
                                    AE(host_legf16_b_value<T>(m.B, b, k, j))));
                ref[(b * T::M + i) * T::N + j] = acc.fpsan_payload();
            }
    float *        dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C);
    std::uint32_t* dD;
    const int      total = T::Bk * T::M * T::N;
    HIP_CHECK(hipMalloc(&dD, total * sizeof(std::uint32_t)));
    k_legf16_p<T><<<1, WAVE>>>(dA, dB, dC, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(total);
    HIP_CHECK(hipMemcpy(got.data(), dD, total * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for(int t = 0; t < total; ++t)
        EXPECT_EQ(got[t], ref[t]) << "elem " << t;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
}

#define LEGF16_TRAITS(Name, M_, N_, K_, B_, AV, AE, CV, WRAP)                   \
    struct Name                                                                 \
    {                                                                           \
        using AVec             = AV;                                            \
        using AElem            = AE;                                            \
        using CVec             = CV;                                            \
        static constexpr int M = M_, N = N_, K = K_, Bk = B_;                   \
        static constexpr int CBSZ = 0, ABID = 0, BLGP = 0;                      \
        template <Semantics S, Conversions C>                                   \
        __device__ static Value<CVec, S, C>                                     \
            call(Value<AVec, S, C> a, Value<AVec, S, C> b, Value<CVec, S, C> c) \
        {                                                                       \
            return fpsan::WRAP<CBSZ, ABID, BLGP, S, C>(a, b, c);                \
        }                                                                       \
    };                                                                          \
    TEST(Name, LayoutMatchesHardware)                                           \
    {                                                                           \
        run_legf16_layout<Name>();                                              \
    }                                                                           \
    TEST(Name, FpsanMatchesScalarReference)                                     \
    {                                                                           \
        run_legf16_fpsan<Name>();                                               \
    }

LEGF16_TRAITS(LegacyMfmaF16_16x16x16,
              16,
              16,
              16,
              1,
              fpsan::v4h_native,
              _Float16,
              v4f_native,
              amdgcn_mfma_f32_16x16x16f16)
LEGF16_TRAITS(LegacyMfmaF16_16x16x4,
              16,
              16,
              4,
              4,
              fpsan::v4h_native,
              _Float16,
              v16f_native,
              amdgcn_mfma_f32_16x16x4f16)
LEGF16_TRAITS(LegacyMfmaF16_32x32x8,
              32,
              32,
              8,
              1,
              fpsan::v4h_native,
              _Float16,
              v16f_native,
              amdgcn_mfma_f32_32x32x8f16)
LEGF16_TRAITS(LegacyMfmaF16_32x32x4,
              32,
              32,
              4,
              2,
              fpsan::v4h_native,
              _Float16,
              fpsan::v32f_native,
              amdgcn_mfma_f32_32x32x4f16)
LEGF16_TRAITS(LegacyMfmaF16_4x4x4,
              4,
              4,
              4,
              16,
              fpsan::v4h_native,
              _Float16,
              v4f_native,
              amdgcn_mfma_f32_4x4x4f16)
LEGF16_TRAITS(LegacyMfmaBF16_16x16x16_1k,
              16,
              16,
              16,
              1,
              fpsan::v4bf_native,
              __bf16,
              v4f_native,
              amdgcn_mfma_f32_16x16x16bf16_1k)
LEGF16_TRAITS(LegacyMfmaBF16_16x16x4_1k,
              16,
              16,
              4,
              4,
              fpsan::v4bf_native,
              __bf16,
              v16f_native,
              amdgcn_mfma_f32_16x16x4bf16_1k)
LEGF16_TRAITS(LegacyMfmaBF16_32x32x8_1k,
              32,
              32,
              8,
              1,
              fpsan::v4bf_native,
              __bf16,
              v16f_native,
              amdgcn_mfma_f32_32x32x8bf16_1k)
LEGF16_TRAITS(LegacyMfmaBF16_32x32x4_1k,
              32,
              32,
              4,
              2,
              fpsan::v4bf_native,
              __bf16,
              fpsan::v32f_native,
              amdgcn_mfma_f32_32x32x4bf16_1k)
LEGF16_TRAITS(LegacyMfmaBF16_4x4x4_1k,
              4,
              4,
              4,
              16,
              fpsan::v4bf_native,
              __bf16,
              v4f_native,
              amdgcn_mfma_f32_4x4x4bf16_1k)

struct LegacyMfmaF16_16x16x4_CBSZ2_ABID3_BLGP5
{
    using AVec             = fpsan::v4h_native;
    using AElem            = _Float16;
    using CVec             = v16f_native;
    static constexpr int M = 16, N = 16, K = 4, Bk = 4;
    static constexpr int CBSZ = 2, ABID = 3, BLGP = 5;
    template <Semantics S, Conversions C>
    __device__ static Value<CVec, S, C>
        call(Value<AVec, S, C> a, Value<AVec, S, C> b, Value<CVec, S, C> c)
    {
        return fpsan::amdgcn_mfma_f32_16x16x4f16<CBSZ, ABID, BLGP, S, C>(a, b, c);
    }
};
TEST(LegacyMfmaF16_16x16x4_CBSZ2_ABID3_BLGP5, LayoutMatchesHardware)
{
    run_legf16_layout<LegacyMfmaF16_16x16x4_CBSZ2_ABID3_BLGP5>();
}
TEST(LegacyMfmaF16_16x16x4_CBSZ2_ABID3_BLGP5, FpsanMatchesScalarReference)
{
    run_legf16_fpsan<LegacyMfmaF16_16x16x4_CBSZ2_ABID3_BLGP5>();
}

struct LegacyMfmaBF16_32x32x4_1k_CBSZ1_ABID1_BLGP1
{
    using AVec             = fpsan::v4bf_native;
    using AElem            = __bf16;
    using CVec             = fpsan::v32f_native;
    static constexpr int M = 32, N = 32, K = 4, Bk = 2;
    static constexpr int CBSZ = 1, ABID = 1, BLGP = 1;
    template <Semantics S, Conversions C>
    __device__ static Value<CVec, S, C>
        call(Value<AVec, S, C> a, Value<AVec, S, C> b, Value<CVec, S, C> c)
    {
        return fpsan::amdgcn_mfma_f32_32x32x4bf16_1k<CBSZ, ABID, BLGP, S, C>(a, b, c);
    }
};
TEST(LegacyMfmaBF16_32x32x4_1k_CBSZ1_ABID1_BLGP1, LayoutMatchesHardware)
{
    run_legf16_layout<LegacyMfmaBF16_32x32x4_1k_CBSZ1_ABID1_BLGP1>();
}
TEST(LegacyMfmaBF16_32x32x4_1k_CBSZ1_ABID1_BLGP1, FpsanMatchesScalarReference)
{
    run_legf16_fpsan<LegacyMfmaBF16_32x32x4_1k_CBSZ1_ABID1_BLGP1>();
}

// ---------------------------------------------------------------------------
// Scaled f8f6f4 MFMA 16x16x128 (E4M3), FPSan dataflow. Layout + scale are the
// silicon-verified model in amdgcn_mfma.hpp:
//   A[i][k] @ lane 16*((k%64)/16)+i, byte (k/64)*16+(k%16); B symmetric in j.
//   D[i][j] @ output_loc_32: reg=i%4, lane=(i/4)*16+j.
//   scale = E8M0 2^(byte-127), applied per row/col after the dot product.
// Both scale operands are exercised with non-trivial E8M0 exponents; the FPSan
// dataflow and the real builtin must agree bit-for-bit.
// ---------------------------------------------------------------------------
static constexpr int SK = 128, SM = 16, SN = 16;

// Element-generic scaled-MFMA kernel. AElem/BElem select the 8-bit operand
// format (fpsan::fp8_e4m3 or fpsan::fp8_e5m2); CBSZ/BLGP are the matching
// format immediates (E4M3=0, E5M2=1).
template <Semantics S, class Out, class AElem, class BElem, int CBSZ, int BLGP>
__global__ void k_scale16(const float* A, const float* B, const float* C, Out* D, int sa, int sb)
{
    using AFrag = fpsan::detail::v32_fragment<AElem>;
    using BFrag = fpsan::detail::v32_fragment<BElem>;
    int   lane  = threadIdx.x;
    AFrag an{};
    BFrag bn{};
    for(int half = 0; half < 2; ++half)
        for(int kk = 0; kk < 16; ++kk)
        {
            int g = lane / 16, idx = lane % 16;
            int k = half * 64 + g * 16 + kk, byte = half * 16 + kk;
            an[byte] = AElem(A[idx * SK + k]);
            bn[byte] = BElem(B[k * SN + idx]);
        }
    fpsan::v4f_native cn{};
    for(int reg = 0; reg < 4; ++reg)
    {
        int i = 4 * (lane / 16) + reg, j = lane % 16;
        cn[reg] = C[i * SN + j];
    }
    Value<AFrag, S, kCC>             a{an};
    Value<BFrag, S, kCC>             b{bn};
    Value<fpsan::v4f_native, S, kCC> c{cn};
    auto d = fpsan::amdgcn_mfma_scale_f32_16x16x128_f8f6f4<CBSZ, 0, BLGP, 0, 0, S, kCC>(
        a, b, c, sa, sb);
    for(int reg = 0; reg < 4; ++reg)
    {
        int i = 4 * (lane / 16) + reg, j = lane % 16;
        if constexpr(S == Semantics::Native)
            D[i * SN + j] = d.get(reg).to_float();
        else
            D[i * SN + j] = d.get(reg).fpsan_payload();
    }
}

namespace
{
    struct ScaleMats
    {
        std::vector<float> A, B, C;
    };
    ScaleMats make_scale_mats()
    {
        ScaleMats m;
        m.A.resize(SM * SK);
        m.B.resize(SK * SN);
        m.C.resize(SM * SN);
        std::mt19937 rng = fpsan_test::make_rng();
        for(auto& x : m.A)
            x = fpsan_test::pick_int_valued<float>(rng, -3, 3);
        for(auto& x : m.B)
            x = fpsan_test::pick_int_valued<float>(rng, -2, 2);
        for(auto& x : m.C)
            x = fpsan_test::pick_int_valued<float>(rng, -4, 4);
        return m;
    }
} // namespace

// Float dataflow (== builtin) with both scales nonzero. Small-int inputs are
// exact in both e4m3 and e5m2, so the reference matmul is format-independent.
template <class AElem, class BElem, int CBSZ, int BLGP>
void run_scale16_layout()
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    ScaleMats   m  = make_scale_mats();
    const int   sa = 129, sb = 126; // 2^2 and 2^-1
    const float fa = fpsan::detail::e8m0_to_float(129), fb = fpsan::detail::e8m0_to_float(126);
    std::vector<float> ref(SM * SN);
    for(int i = 0; i < SM; ++i)
        for(int j = 0; j < SN; ++j)
        {
            double dot = 0;
            for(int k = 0; k < SK; ++k)
                dot += (double)m.A[i * SK + k] * (double)m.B[k * SN + j];
            ref[i * SN + j] = (float)((double)m.C[i * SN + j] + dot * (double)fa * (double)fb);
        }
    float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C), *dD;
    HIP_CHECK(hipMalloc(&dD, SM * SN * sizeof(float)));
    k_scale16<Semantics::Native, float, AElem, BElem, CBSZ, BLGP>
        <<<1, WAVE>>>(dA, dB, dC, dD, sa, sb);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> got(SM * SN);
    HIP_CHECK(hipMemcpy(got.data(), dD, SM * SN * sizeof(float), hipMemcpyDeviceToHost));
    for(int i = 0; i < SM * SN; ++i)
        EXPECT_EQ(bits_of(got[i]), bits_of(ref[i])) << "at " << i;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
}

template <class AElem, class BElem, int CBSZ, int BLGP>
void run_scale16_fpsan()
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    ScaleMats m  = make_scale_mats();
    const int sa = 128, sb = 130; // both scales nonzero
    using VF = Value<float, Semantics::Triton, kCC>;
    using VA = Value<AElem, Semantics::Triton, kCC>;
    using VB = Value<BElem, Semantics::Triton, kCC>;
    const VF vsa(fpsan::detail::e8m0_to_float(128)), vsb(fpsan::detail::e8m0_to_float(130));
    std::vector<std::uint32_t> ref(SM * SN);
    for(int i = 0; i < SM; ++i)
        for(int j = 0; j < SN; ++j)
        {
            VF dot(0.0f);
            // Mirror the dataflow: embed as fp8, then cast to float in the ring.
            for(int k = 0; k < SK; ++k)
                dot = dot
                      + fpsan::cast<float>(VA(AElem(m.A[i * SK + k])))
                            * fpsan::cast<float>(VB(BElem(m.B[k * SN + j])));
            ref[i * SN + j] = (VF(m.C[i * SN + j]) + dot * vsa * vsb).fpsan_payload();
        }
    float *        dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C);
    std::uint32_t* dD;
    HIP_CHECK(hipMalloc(&dD, SM * SN * sizeof(std::uint32_t)));
    k_scale16<Semantics::Triton, std::uint32_t, AElem, BElem, CBSZ, BLGP>
        <<<1, WAVE>>>(dA, dB, dC, dD, sa, sb);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(SM * SN);
    HIP_CHECK(hipMemcpy(got.data(), dD, SM * SN * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for(int i = 0; i < SM * SN; ++i)
        EXPECT_EQ(got[i], ref[i]) << "at " << i;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
}

using fpsan::fp8_e4m3;
using fpsan::fp8_e5m2;

TEST(ScaledMfma16x16x128_E4M3, LayoutMatchesHardware)
{
    run_scale16_layout<fp8_e4m3, fp8_e4m3, 0, 0>();
}
TEST(ScaledMfma16x16x128_E4M3, FpsanMatchesScalarReference)
{
    run_scale16_fpsan<fp8_e4m3, fp8_e4m3, 0, 0>();
}
TEST(ScaledMfma16x16x128_E5M2, LayoutMatchesHardware)
{
    run_scale16_layout<fp8_e5m2, fp8_e5m2, 1, 1>();
}
TEST(ScaledMfma16x16x128_E5M2, FpsanMatchesScalarReference)
{
    run_scale16_fpsan<fp8_e5m2, fp8_e5m2, 1, 1>();
}
TEST(ScaledMfma16x16x128_Mixed, LayoutMatchesHardware)
{
    run_scale16_layout<fp8_e4m3, fp8_e5m2, 0, 1>(); // A=E4M3, B=E5M2
}
TEST(ScaledMfma16x16x128_Mixed, FpsanMatchesScalarReference)
{
    run_scale16_fpsan<fp8_e4m3, fp8_e5m2, 0, 1>();
}

// ---------------------------------------------------------------------------
// Scaled f8f6f4 MFMA 32x32x64. Layout (ISA 7.1.5):
//   A[m][k] @ lane 16*(2*((k%32)/16) + m/16) + (m%16), byte (k/32)*16+(k%16);
//   B symmetric in n. Output via output_loc_32(32,32,...), 16 regs/lane.
// ---------------------------------------------------------------------------
static constexpr int S2M = 32, S2N = 32, S2K = 64;

template <Semantics S, class Out, class AElem, class BElem, int CBSZ, int BLGP>
__global__ void k_scale32(const float* A, const float* B, const float* C, Out* D, int sa, int sb)
{
    using AFrag = fpsan::detail::v32_fragment<AElem>;
    using BFrag = fpsan::detail::v32_fragment<BElem>;
    int   lane  = threadIdx.x;
    int   mn    = (lane % 16) + 16 * ((lane / 16) & 1);
    AFrag an{};
    BFrag bn{};
    for(int b = 0; b < 32; ++b)
    {
        int k = (b / 16) * 32 + ((lane / 16) / 2) * 16 + (b % 16);
        an[b] = AElem(A[mn * S2K + k]);
        bn[b] = BElem(B[k * S2N + mn]);
    }
    fpsan::v16f_native cn{};
    for(int i = 0; i < S2M; ++i)
        for(int j = 0; j < S2N; ++j)
        {
            auto loc = fpsan::detail::output_loc_32(S2M, S2N, i, j, 0);
            if(loc.lane == lane)
                cn[loc.reg] = C[i * S2N + j];
        }
    Value<AFrag, S, kCC>              a{an};
    Value<BFrag, S, kCC>              b{bn};
    Value<fpsan::v16f_native, S, kCC> c{cn};
    auto d = fpsan::amdgcn_mfma_scale_f32_32x32x64_f8f6f4<CBSZ, 0, BLGP, 0, 0, S, kCC>(
        a, b, c, sa, sb);
    for(int i = 0; i < S2M; ++i)
        for(int j = 0; j < S2N; ++j)
        {
            auto loc = fpsan::detail::output_loc_32(S2M, S2N, i, j, 0);
            if(loc.lane == lane)
            {
                if constexpr(S == Semantics::Native)
                    D[i * S2N + j] = d.get(loc.reg).to_float();
                else
                    D[i * S2N + j] = d.get(loc.reg).fpsan_payload();
            }
        }
}

namespace
{
    ScaleMats make_scale_mats32()
    {
        ScaleMats m;
        m.A.resize(S2M * S2K);
        m.B.resize(S2K * S2N);
        m.C.resize(S2M * S2N);
        std::mt19937 rng = fpsan_test::make_rng();
        for(auto& x : m.A)
            x = fpsan_test::pick_int_valued<float>(rng, -3, 3);
        for(auto& x : m.B)
            x = fpsan_test::pick_int_valued<float>(rng, -2, 2);
        for(auto& x : m.C)
            x = fpsan_test::pick_int_valued<float>(rng, -4, 4);
        return m;
    }
} // namespace

template <class AElem, class BElem, int CBSZ, int BLGP>
void run_scale32_layout()
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    ScaleMats   m  = make_scale_mats32();
    const int   sa = 129, sb = 126;
    const float fa = fpsan::detail::e8m0_to_float(129), fb = fpsan::detail::e8m0_to_float(126);
    std::vector<float> ref(S2M * S2N);
    for(int i = 0; i < S2M; ++i)
        for(int j = 0; j < S2N; ++j)
        {
            double dot = 0;
            for(int k = 0; k < S2K; ++k)
                dot += (double)m.A[i * S2K + k] * (double)m.B[k * S2N + j];
            ref[i * S2N + j] = (float)((double)m.C[i * S2N + j] + dot * (double)fa * (double)fb);
        }
    float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C), *dD;
    HIP_CHECK(hipMalloc(&dD, S2M * S2N * sizeof(float)));
    k_scale32<Semantics::Native, float, AElem, BElem, CBSZ, BLGP>
        <<<1, WAVE>>>(dA, dB, dC, dD, sa, sb);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> got(S2M * S2N);
    HIP_CHECK(hipMemcpy(got.data(), dD, S2M * S2N * sizeof(float), hipMemcpyDeviceToHost));
    for(int i = 0; i < S2M * S2N; ++i)
        EXPECT_EQ(bits_of(got[i]), bits_of(ref[i])) << "at " << i;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
}

template <class AElem, class BElem, int CBSZ, int BLGP>
void run_scale32_fpsan()
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    ScaleMats m  = make_scale_mats32();
    const int sa = 128, sb = 130;
    using VF = Value<float, Semantics::Triton, kCC>;
    using VA = Value<AElem, Semantics::Triton, kCC>;
    using VB = Value<BElem, Semantics::Triton, kCC>;
    const VF vsa(fpsan::detail::e8m0_to_float(128)), vsb(fpsan::detail::e8m0_to_float(130));
    std::vector<std::uint32_t> ref(S2M * S2N);
    for(int i = 0; i < S2M; ++i)
        for(int j = 0; j < S2N; ++j)
        {
            VF dot(0.0f);
            for(int k = 0; k < S2K; ++k)
                dot = dot
                      + fpsan::cast<float>(VA(AElem(m.A[i * S2K + k])))
                            * fpsan::cast<float>(VB(BElem(m.B[k * S2N + j])));
            ref[i * S2N + j] = (VF(m.C[i * S2N + j]) + dot * vsa * vsb).fpsan_payload();
        }
    float *        dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C);
    std::uint32_t* dD;
    HIP_CHECK(hipMalloc(&dD, S2M * S2N * sizeof(std::uint32_t)));
    k_scale32<Semantics::Triton, std::uint32_t, AElem, BElem, CBSZ, BLGP>
        <<<1, WAVE>>>(dA, dB, dC, dD, sa, sb);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(S2M * S2N);
    HIP_CHECK(hipMemcpy(got.data(), dD, S2M * S2N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for(int i = 0; i < S2M * S2N; ++i)
        EXPECT_EQ(got[i], ref[i]) << "at " << i;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
}

TEST(ScaledMfma32x32x64_E4M3, LayoutMatchesHardware)
{
    run_scale32_layout<fp8_e4m3, fp8_e4m3, 0, 0>();
}
TEST(ScaledMfma32x32x64_E4M3, FpsanMatchesScalarReference)
{
    run_scale32_fpsan<fp8_e4m3, fp8_e4m3, 0, 0>();
}
TEST(ScaledMfma32x32x64_E5M2, LayoutMatchesHardware)
{
    run_scale32_layout<fp8_e5m2, fp8_e5m2, 1, 1>();
}
TEST(ScaledMfma32x32x64_E5M2, FpsanMatchesScalarReference)
{
    run_scale32_fpsan<fp8_e5m2, fp8_e5m2, 1, 1>();
}
TEST(ScaledMfma32x32x64_Mixed, LayoutMatchesHardware)
{
    run_scale32_layout<fp8_e5m2, fp8_e4m3, 1, 0>(); // A=E5M2, B=E4M3
}
TEST(ScaledMfma32x32x64_Mixed, FpsanMatchesScalarReference)
{
    run_scale32_fpsan<fp8_e5m2, fp8_e4m3, 1, 0>();
}

// ---------------------------------------------------------------------------
// Sub-byte (fp6 / bf6 / fp4) scaled f8f6f4 MFMA. CBSZ/BLGP select the format:
// 2=E2M3 (fp6), 3=E3M2 (bf6), 4=E2M1 (fp4). Per-lane operands are the raw
// packed v8i32 (Width-bit field s at bits Width*s..). A and B must share a bit
// width. Layout reverse-engineered + silicon-verified on MI350 and pinned by
// amdgcn_mfma.hpp.
//
//   * LayoutMatchesHardware (Float): pack the exact fp6/fp4 hardware codes,
//     run the builtin via the wrapper, compare to a numeric reference. Inputs
//     are small exact integers, so the accumulation is exact.
//   * FpsanMatchesScalarReference (FPSan): pack arbitrary Width-bit payloads,
//     run the FPSan dataflow, compare to a host reference that signed-resizes
//     the same payloads (subbyte widen) and accumulates in the payload ring --
//     isolating the gather + extraction layout (which the Float test pins).
// ---------------------------------------------------------------------------
namespace
{
    // fp6 (E2M3, code 2), bf6 (E3M2, code 3), fp4 (E2M1, code 4) decode.
    int sub_width(int code)
    {
        return code <= 3 ? 6 : 4;
    }
    float sub_decode(int code, unsigned c)
    {
        if(code == 2)
        { // E2M3
            int   s = (c >> 5) & 1, e = (c >> 3) & 3, m = c & 7;
            float v = e == 0 ? m / 8.0f : (1 + m / 8.0f) * std::ldexp(1.0f, e - 1);
            return s ? -v : v;
        }
        if(code == 3)
        { // E3M2
            int   s = (c >> 5) & 1, e = (c >> 2) & 7, m = c & 3;
            float v = e == 0 ? (m / 4.0f) * std::ldexp(1.0f, 1 - 3)
                             : (1 + m / 4.0f) * std::ldexp(1.0f, e - 3);
            return s ? -v : v;
        }
        int   s = (c >> 3) & 1, e = (c >> 1) & 3, m = c & 1; // E2M1
        float v = e == 0 ? m * 0.5f : (1 + 0.5f * m) * std::ldexp(1.0f, e - 1);
        return s ? -v : v;
    }
    unsigned sub_encode(int code, int val)
    {
        int n = 1 << sub_width(code);
        for(int c = 0; c < n; ++c)
            if(sub_decode(code, (unsigned)c) == (float)val)
                return (unsigned)c;
        return 0u;
    }
    void set_field(fpsan::v8i32_native& reg, int bitoff, unsigned val, int nb)
    {
        for(int b = 0; b < nb; ++b)
        {
            int      bo = bitoff + b, w = bo / 32, sh = bo % 32;
            unsigned u = (unsigned)reg[w];
            u          = (u & ~(1u << sh)) | (((val >> b) & 1u) << sh);
            reg[w]     = (int)u;
        }
    }
} // namespace

template <Semantics S, int CBSZ, int BLGP, class Out>
__global__ void k_scale_sub16(const fpsan::v8i32_native* A,
                              const fpsan::v8i32_native* B,
                              const float*               C,
                              Out*                       D,
                              int                        sa,
                              int                        sb)
{
    int                 lane = threadIdx.x;
    fpsan::v8i32_native an = A[lane], bn = B[lane];
    fpsan::v4f_native   cn{};
    for(int reg = 0; reg < 4; ++reg)
    {
        int i = 4 * (lane / 16) + reg, j = lane % 16;
        cn[reg] = C[i * SN + j];
    }
    Value<fpsan::v4f_native, S, kCC> c{cn};
    auto d = fpsan::amdgcn_mfma_scale_f32_16x16x128_f8f6f4_sub<CBSZ, BLGP, 0, 0, S, kCC>(
        an, bn, c, sa, sb);
    for(int reg = 0; reg < 4; ++reg)
    {
        int i = 4 * (lane / 16) + reg, j = lane % 16;
        if constexpr(S == Semantics::Native)
            D[i * SN + j] = d.get(reg).to_float();
        else
            D[i * SN + j] = d.get(reg).fpsan_payload();
    }
}

template <int CBSZ, int BLGP>
void run_scale_sub16_layout()
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    const int        wa = sub_width(CBSZ), wb = sub_width(BLGP);
    std::mt19937     rng = fpsan_test::make_rng();
    const int        sa = 129, sb = 126;
    const float      fa = fpsan::detail::e8m0_to_float(129), fb = fpsan::detail::e8m0_to_float(126);
    std::vector<int> Acode(SM * SK), Bcode(SK * SN);
    for(auto& x : Acode)
        x = (int)sub_encode(CBSZ, fpsan_test::pick_int(rng, -2, 3));
    for(auto& x : Bcode)
        x = (int)sub_encode(BLGP, fpsan_test::pick_int(rng, -2, 3));
    std::vector<float> Cm(SM * SN);
    for(auto& x : Cm)
        x = fpsan_test::pick_int_valued<float>(rng, -8, 8);
    std::vector<fpsan::v8i32_native> A(WAVE, fpsan::v8i32_native{}), B(WAVE, fpsan::v8i32_native{});
    for(int i = 0; i < SM; ++i)
        for(int k = 0; k < SK; ++k)
        {
            int lane = 16 * ((k % 64) / 16) + i, s = (k / 64) * 16 + (k % 16);
            set_field(A[lane], wa * s, (unsigned)Acode[i * SK + k], wa);
        }
    for(int k = 0; k < SK; ++k)
        for(int j = 0; j < SN; ++j)
        {
            int lane = 16 * ((k % 64) / 16) + j, s = (k / 64) * 16 + (k % 16);
            set_field(B[lane], wb * s, (unsigned)Bcode[k * SN + j], wb);
        }
    std::vector<float> ref(SM * SN);
    for(int i = 0; i < SM; ++i)
        for(int j = 0; j < SN; ++j)
        {
            double dot = 0;
            for(int k = 0; k < SK; ++k)
                dot += (double)sub_decode(CBSZ, (unsigned)Acode[i * SK + k])
                       * (double)sub_decode(BLGP, (unsigned)Bcode[k * SN + j]);
            ref[i * SN + j] = (float)((double)Cm[i * SN + j] + dot * (double)fa * (double)fb);
        }
    fpsan::v8i32_native *dA, *dB;
    float *              dC, *dD;
    HIP_CHECK(hipMalloc(&dA, WAVE * sizeof(fpsan::v8i32_native)));
    HIP_CHECK(hipMalloc(&dB, WAVE * sizeof(fpsan::v8i32_native)));
    HIP_CHECK(hipMalloc(&dC, SM * SN * sizeof(float)));
    HIP_CHECK(hipMalloc(&dD, SM * SN * sizeof(float)));
    HIP_CHECK(hipMemcpy(dA, A.data(), WAVE * sizeof(fpsan::v8i32_native), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, B.data(), WAVE * sizeof(fpsan::v8i32_native), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dC, Cm.data(), SM * SN * sizeof(float), hipMemcpyHostToDevice));
    k_scale_sub16<Semantics::Native, CBSZ, BLGP, float><<<1, WAVE>>>(dA, dB, dC, dD, sa, sb);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> got(SM * SN);
    HIP_CHECK(hipMemcpy(got.data(), dD, SM * SN * sizeof(float), hipMemcpyDeviceToHost));
    for(int i = 0; i < SM * SN; ++i)
        EXPECT_EQ(bits_of(got[i]), bits_of(ref[i])) << "at " << i;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
}

template <int CBSZ, int BLGP>
void run_scale_sub16_fpsan()
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    const int    wa = sub_width(CBSZ), wb = sub_width(BLGP);
    std::mt19937 rng = fpsan_test::make_rng();
    const int    sa = 128, sb = 130;
    using VF = Value<float, Semantics::Triton, kCC>;
    const VF vsa(fpsan::detail::e8m0_to_float(128)), vsb(fpsan::detail::e8m0_to_float(130));
    // Arbitrary Width-bit payloads (the realistic FPSan content of sub-byte data;
    // not necessarily valid format codes -- the dataflow only signed-resizes).
    std::vector<int> Ap(SM * SK), Bp(SK * SN);
    for(auto& x : Ap)
        x = (int)(rng() & ((1u << wa) - 1u));
    for(auto& x : Bp)
        x = (int)(rng() & ((1u << wb) - 1u));
    std::vector<float> Cm(SM * SN);
    for(auto& x : Cm)
        x = fpsan_test::pick_int_valued<float>(rng, -8, 8);
    std::vector<fpsan::v8i32_native> A(WAVE, fpsan::v8i32_native{}), B(WAVE, fpsan::v8i32_native{});
    for(int i = 0; i < SM; ++i)
        for(int k = 0; k < SK; ++k)
        {
            int lane = 16 * ((k % 64) / 16) + i, s = (k / 64) * 16 + (k % 16);
            set_field(A[lane], wa * s, (unsigned)Ap[i * SK + k], wa);
        }
    for(int k = 0; k < SK; ++k)
        for(int j = 0; j < SN; ++j)
        {
            int lane = 16 * ((k % 64) / 16) + j, s = (k / 64) * 16 + (k % 16);
            set_field(B[lane], wb * s, (unsigned)Bp[k * SN + j], wb);
        }
    auto widen = [&](int code, unsigned f) {
        int          w = sub_width(code);
        std::int32_t e = (std::int32_t)(f << (32 - w)) >> (32 - w);
        return VF::from_fpsan_payload((std::uint32_t)e);
    };
    std::vector<std::uint32_t> ref(SM * SN);
    for(int i = 0; i < SM; ++i)
        for(int j = 0; j < SN; ++j)
        {
            VF dot(0.0f);
            for(int k = 0; k < SK; ++k)
                dot = dot
                      + widen(CBSZ, (unsigned)Ap[i * SK + k])
                            * widen(BLGP, (unsigned)Bp[k * SN + j]);
            ref[i * SN + j] = (VF(Cm[i * SN + j]) + dot * vsa * vsb).fpsan_payload();
        }
    fpsan::v8i32_native *dA, *dB;
    float*               dC;
    std::uint32_t*       dD;
    HIP_CHECK(hipMalloc(&dA, WAVE * sizeof(fpsan::v8i32_native)));
    HIP_CHECK(hipMalloc(&dB, WAVE * sizeof(fpsan::v8i32_native)));
    HIP_CHECK(hipMalloc(&dC, SM * SN * sizeof(float)));
    HIP_CHECK(hipMalloc(&dD, SM * SN * sizeof(std::uint32_t)));
    HIP_CHECK(hipMemcpy(dA, A.data(), WAVE * sizeof(fpsan::v8i32_native), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, B.data(), WAVE * sizeof(fpsan::v8i32_native), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dC, Cm.data(), SM * SN * sizeof(float), hipMemcpyHostToDevice));
    k_scale_sub16<Semantics::Triton, CBSZ, BLGP, std::uint32_t>
        <<<1, WAVE>>>(dA, dB, dC, dD, sa, sb);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(SM * SN);
    HIP_CHECK(hipMemcpy(got.data(), dD, SM * SN * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for(int i = 0; i < SM * SN; ++i)
        EXPECT_EQ(got[i], ref[i]) << "at " << i;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
}

TEST(ScaledMfma16x16x128_FP6, LayoutMatchesHardware)
{
    run_scale_sub16_layout<2, 2>();
}
TEST(ScaledMfma16x16x128_FP6, FpsanMatchesScalarReference)
{
    run_scale_sub16_fpsan<2, 2>();
}
TEST(ScaledMfma16x16x128_BF6, LayoutMatchesHardware)
{
    run_scale_sub16_layout<3, 3>();
}
TEST(ScaledMfma16x16x128_BF6, FpsanMatchesScalarReference)
{
    run_scale_sub16_fpsan<3, 3>();
}
TEST(ScaledMfma16x16x128_FP6BF6, LayoutMatchesHardware)
{
    run_scale_sub16_layout<2, 3>();
}
TEST(ScaledMfma16x16x128_FP6BF6, FpsanMatchesScalarReference)
{
    run_scale_sub16_fpsan<2, 3>();
}
TEST(ScaledMfma16x16x128_FP4, LayoutMatchesHardware)
{
    run_scale_sub16_layout<4, 4>();
}
TEST(ScaledMfma16x16x128_FP4, FpsanMatchesScalarReference)
{
    run_scale_sub16_fpsan<4, 4>();
}

template <Semantics S, int CBSZ, int BLGP, class Out>
__global__ void k_scale_sub32(const fpsan::v8i32_native* A,
                              const fpsan::v8i32_native* B,
                              const float*               C,
                              Out*                       D,
                              int                        sa,
                              int                        sb)
{
    int                 lane = threadIdx.x;
    fpsan::v8i32_native an = A[lane], bn = B[lane];
    fpsan::v16f_native  cn{};
    for(int i = 0; i < S2M; ++i)
        for(int j = 0; j < S2N; ++j)
        {
            auto loc = fpsan::detail::output_loc_32(S2M, S2N, i, j, 0);
            if(loc.lane == lane)
                cn[loc.reg] = C[i * S2N + j];
        }
    Value<fpsan::v16f_native, S, kCC> c{cn};
    auto d = fpsan::amdgcn_mfma_scale_f32_32x32x64_f8f6f4_sub<CBSZ, BLGP, 0, 0, S, kCC>(
        an, bn, c, sa, sb);
    for(int i = 0; i < S2M; ++i)
        for(int j = 0; j < S2N; ++j)
        {
            auto loc = fpsan::detail::output_loc_32(S2M, S2N, i, j, 0);
            if(loc.lane == lane)
            {
                if constexpr(S == Semantics::Native)
                    D[i * S2N + j] = d.get(loc.reg).to_float();
                else
                    D[i * S2N + j] = d.get(loc.reg).fpsan_payload();
            }
        }
}

template <int CBSZ, int BLGP>
void pack_sub32(std::vector<fpsan::v8i32_native>& A,
                std::vector<fpsan::v8i32_native>& B,
                const std::vector<int>&           Af,
                const std::vector<int>&           Bf)
{
    const int wa = sub_width(CBSZ), wb = sub_width(BLGP);
    for(int m = 0; m < S2M; ++m)
        for(int k = 0; k < S2K; ++k)
        {
            int lane = 16 * (2 * ((k % 32) / 16) + m / 16) + (m % 16), s = (k / 32) * 16 + (k % 16);
            set_field(A[lane], wa * s, (unsigned)Af[m * S2K + k], wa);
        }
    for(int k = 0; k < S2K; ++k)
        for(int n = 0; n < S2N; ++n)
        {
            int lane = 16 * (2 * ((k % 32) / 16) + n / 16) + (n % 16), s = (k / 32) * 16 + (k % 16);
            set_field(B[lane], wb * s, (unsigned)Bf[k * S2N + n], wb);
        }
}

template <int CBSZ, int BLGP>
void run_scale_sub32_layout()
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    std::mt19937     rng = fpsan_test::make_rng();
    const int        sa = 129, sb = 126;
    const float      fa = fpsan::detail::e8m0_to_float(129), fb = fpsan::detail::e8m0_to_float(126);
    std::vector<int> Acode(S2M * S2K), Bcode(S2K * S2N);
    for(auto& x : Acode)
        x = (int)sub_encode(CBSZ, fpsan_test::pick_int(rng, -2, 3));
    for(auto& x : Bcode)
        x = (int)sub_encode(BLGP, fpsan_test::pick_int(rng, -2, 3));
    std::vector<float> Cm(S2M * S2N);
    for(auto& x : Cm)
        x = fpsan_test::pick_int_valued<float>(rng, -8, 8);
    std::vector<fpsan::v8i32_native> A(WAVE, fpsan::v8i32_native{}), B(WAVE, fpsan::v8i32_native{});
    pack_sub32<CBSZ, BLGP>(A, B, Acode, Bcode);
    std::vector<float> ref(S2M * S2N);
    for(int i = 0; i < S2M; ++i)
        for(int j = 0; j < S2N; ++j)
        {
            double dot = 0;
            for(int k = 0; k < S2K; ++k)
                dot += (double)sub_decode(CBSZ, (unsigned)Acode[i * S2K + k])
                       * (double)sub_decode(BLGP, (unsigned)Bcode[k * S2N + j]);
            ref[i * S2N + j] = (float)((double)Cm[i * S2N + j] + dot * (double)fa * (double)fb);
        }
    fpsan::v8i32_native *dA, *dB;
    float *              dC, *dD;
    HIP_CHECK(hipMalloc(&dA, WAVE * sizeof(fpsan::v8i32_native)));
    HIP_CHECK(hipMalloc(&dB, WAVE * sizeof(fpsan::v8i32_native)));
    HIP_CHECK(hipMalloc(&dC, S2M * S2N * sizeof(float)));
    HIP_CHECK(hipMalloc(&dD, S2M * S2N * sizeof(float)));
    HIP_CHECK(hipMemcpy(dA, A.data(), WAVE * sizeof(fpsan::v8i32_native), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, B.data(), WAVE * sizeof(fpsan::v8i32_native), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dC, Cm.data(), S2M * S2N * sizeof(float), hipMemcpyHostToDevice));
    k_scale_sub32<Semantics::Native, CBSZ, BLGP, float><<<1, WAVE>>>(dA, dB, dC, dD, sa, sb);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> got(S2M * S2N);
    HIP_CHECK(hipMemcpy(got.data(), dD, S2M * S2N * sizeof(float), hipMemcpyDeviceToHost));
    for(int i = 0; i < S2M * S2N; ++i)
        EXPECT_EQ(bits_of(got[i]), bits_of(ref[i])) << "at " << i;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
}

template <int CBSZ, int BLGP>
void run_scale_sub32_fpsan()
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    const int    wa = sub_width(CBSZ), wb = sub_width(BLGP);
    std::mt19937 rng = fpsan_test::make_rng();
    const int    sa = 128, sb = 130;
    using VF = Value<float, Semantics::Triton, kCC>;
    const VF         vsa(fpsan::detail::e8m0_to_float(128)), vsb(fpsan::detail::e8m0_to_float(130));
    std::vector<int> Ap(S2M * S2K), Bp(S2K * S2N);
    for(auto& x : Ap)
        x = (int)(rng() & ((1u << wa) - 1u));
    for(auto& x : Bp)
        x = (int)(rng() & ((1u << wb) - 1u));
    std::vector<float> Cm(S2M * S2N);
    for(auto& x : Cm)
        x = fpsan_test::pick_int_valued<float>(rng, -8, 8);
    std::vector<fpsan::v8i32_native> A(WAVE, fpsan::v8i32_native{}), B(WAVE, fpsan::v8i32_native{});
    pack_sub32<CBSZ, BLGP>(A, B, Ap, Bp);
    auto widen = [&](int code, unsigned f) {
        int          w = sub_width(code);
        std::int32_t e = (std::int32_t)(f << (32 - w)) >> (32 - w);
        return VF::from_fpsan_payload((std::uint32_t)e);
    };
    std::vector<std::uint32_t> ref(S2M * S2N);
    for(int i = 0; i < S2M; ++i)
        for(int j = 0; j < S2N; ++j)
        {
            VF dot(0.0f);
            for(int k = 0; k < S2K; ++k)
                dot = dot
                      + widen(CBSZ, (unsigned)Ap[i * S2K + k])
                            * widen(BLGP, (unsigned)Bp[k * S2N + j]);
            ref[i * S2N + j] = (VF(Cm[i * S2N + j]) + dot * vsa * vsb).fpsan_payload();
        }
    fpsan::v8i32_native *dA, *dB;
    float*               dC;
    std::uint32_t*       dD;
    HIP_CHECK(hipMalloc(&dA, WAVE * sizeof(fpsan::v8i32_native)));
    HIP_CHECK(hipMalloc(&dB, WAVE * sizeof(fpsan::v8i32_native)));
    HIP_CHECK(hipMalloc(&dC, S2M * S2N * sizeof(float)));
    HIP_CHECK(hipMalloc(&dD, S2M * S2N * sizeof(std::uint32_t)));
    HIP_CHECK(hipMemcpy(dA, A.data(), WAVE * sizeof(fpsan::v8i32_native), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, B.data(), WAVE * sizeof(fpsan::v8i32_native), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dC, Cm.data(), S2M * S2N * sizeof(float), hipMemcpyHostToDevice));
    k_scale_sub32<Semantics::Triton, CBSZ, BLGP, std::uint32_t>
        <<<1, WAVE>>>(dA, dB, dC, dD, sa, sb);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(S2M * S2N);
    HIP_CHECK(hipMemcpy(got.data(), dD, S2M * S2N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for(int i = 0; i < S2M * S2N; ++i)
        EXPECT_EQ(got[i], ref[i]) << "at " << i;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
}

TEST(ScaledMfma32x32x64_FP6, LayoutMatchesHardware)
{
    run_scale_sub32_layout<2, 2>();
}
TEST(ScaledMfma32x32x64_FP6, FpsanMatchesScalarReference)
{
    run_scale_sub32_fpsan<2, 2>();
}
TEST(ScaledMfma32x32x64_BF6, LayoutMatchesHardware)
{
    run_scale_sub32_layout<3, 3>();
}
TEST(ScaledMfma32x32x64_BF6, FpsanMatchesScalarReference)
{
    run_scale_sub32_fpsan<3, 3>();
}
TEST(ScaledMfma32x32x64_FP4, LayoutMatchesHardware)
{
    run_scale_sub32_layout<4, 4>();
}
TEST(ScaledMfma32x32x64_FP4, FpsanMatchesScalarReference)
{
    run_scale_sub32_fpsan<4, 4>();
}

// ---------------------------------------------------------------------------
// sub x sub DIFFERENT-width (fp6 x fp4) -- now supported (all sub formats share
// one physical k order, so equal field indices pair the same k). Reuses the
// sub harnesses, which already track wa/wb separately. Silicon-verified for
// mixed-width sub x sub cases at both 16x16x128 and 32x32x64 shapes.
// ---------------------------------------------------------------------------
TEST(ScaledMfma16x16x128_FP6FP4, LayoutMatchesHardware)
{
    run_scale_sub16_layout<2, 4>();
}
TEST(ScaledMfma16x16x128_FP6FP4, FpsanMatchesScalarReference)
{
    run_scale_sub16_fpsan<2, 4>();
}
TEST(ScaledMfma16x16x128_FP4FP6, LayoutMatchesHardware)
{
    run_scale_sub16_layout<4, 2>();
}
TEST(ScaledMfma16x16x128_FP4FP6, FpsanMatchesScalarReference)
{
    run_scale_sub16_fpsan<4, 2>();
}
TEST(ScaledMfma16x16x128_BF6FP4, LayoutMatchesHardware)
{
    run_scale_sub16_layout<3, 4>();
}
TEST(ScaledMfma16x16x128_BF6FP4, FpsanMatchesScalarReference)
{
    run_scale_sub16_fpsan<3, 4>();
}
TEST(ScaledMfma32x32x64_FP6FP4, LayoutMatchesHardware)
{
    run_scale_sub32_layout<2, 4>();
}
TEST(ScaledMfma32x32x64_FP6FP4, FpsanMatchesScalarReference)
{
    run_scale_sub32_fpsan<2, 4>();
}
TEST(ScaledMfma32x32x64_FP4FP6, LayoutMatchesHardware)
{
    run_scale_sub32_layout<4, 2>();
}
TEST(ScaledMfma32x32x64_FP4FP6, FpsanMatchesScalarReference)
{
    run_scale_sub32_fpsan<4, 2>();
}

// ===========================================================================
// Per-K-block (NON-UNIFORM) scale. Each 32-element K-block carries its own
// E8M0 scale, read from lane (stride*kb + row) of the per-lane scale operand
// (stride = 16 at 16x16x128, 32 at 32x32x64). Silicon-RE'd + verified with
// full-block-random coverage.
// These tests give DIFFERENT scales to each K-block (the uniform tests above
// are the special case where every lane carries the same scale operand).
// ===========================================================================
namespace
{
    // Random E8M0 exponent bytes near 127 (so 2^(b-127) stays exact-ish, keeping
    // the small-int matmul exact).
    std::vector<int> rand_exps(std::mt19937& rng, int n)
    {
        std::vector<int> e(n);
        for(auto& x : e)
            x = 124 + (int)(rng() % 7); // 2^-3 .. 2^+3
        return e;
    }
} // namespace

// ---- fp8 16x16x128 per-block scale ----------------------------------------
template <Semantics S, class Out, class AElem, class BElem, int CBSZ, int BLGP>
__global__ void k_scale16_pb(
    const float* A, const float* B, const float* C, Out* D, const int* SA, const int* SB)
{
    using AFrag = fpsan::detail::v32_fragment<AElem>;
    using BFrag = fpsan::detail::v32_fragment<BElem>;
    int   lane  = threadIdx.x;
    AFrag an{};
    BFrag bn{};
    for(int half = 0; half < 2; ++half)
        for(int kk = 0; kk < 16; ++kk)
        {
            int g = lane / 16, idx = lane % 16;
            int k = half * 64 + g * 16 + kk, byte = half * 16 + kk;
            an[byte] = AElem(A[idx * SK + k]);
            bn[byte] = BElem(B[k * SN + idx]);
        }
    fpsan::v4f_native cn{};
    for(int reg = 0; reg < 4; ++reg)
    {
        int i = 4 * (lane / 16) + reg, j = lane % 16;
        cn[reg] = C[i * SN + j];
    }
    Value<AFrag, S, kCC>             a{an};
    Value<BFrag, S, kCC>             b{bn};
    Value<fpsan::v4f_native, S, kCC> c{cn};
    auto d = fpsan::amdgcn_mfma_scale_f32_16x16x128_f8f6f4<CBSZ, 0, BLGP, 0, 0, S, kCC>(
        a, b, c, SA[lane], SB[lane]);
    for(int reg = 0; reg < 4; ++reg)
    {
        int i = 4 * (lane / 16) + reg, j = lane % 16;
        if constexpr(S == Semantics::Native)
            D[i * SN + j] = d.get(reg).to_float();
        else
            D[i * SN + j] = d.get(reg).fpsan_payload();
    }
}

template <class AElem, class BElem, int CBSZ, int BLGP>
void run_scale16_perblock()
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    const int        NB  = SK / 32; // 4 K-blocks
    ScaleMats        m   = make_scale_mats();
    std::mt19937     rng = fpsan_test::make_rng();
    std::vector<int> eA = rand_exps(rng, SM * NB), eB = rand_exps(rng, SN * NB);
    // Scale operands: byte0 at lane 16*kb+row carries the (row,block) exponent.
    std::vector<int> SA(WAVE, 0), SB(WAVE, 0);
    for(int kb = 0; kb < NB; ++kb)
        for(int i = 0; i < SM; ++i)
            SA[16 * kb + i] = eA[i * NB + kb];
    for(int kb = 0; kb < NB; ++kb)
        for(int j = 0; j < SN; ++j)
            SB[16 * kb + j] = eB[j * NB + kb];
    auto   e8 = [](int b) { return fpsan::detail::e8m0_to_float((unsigned)b); };
    float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C);
    int *  dSA = to_dev(SA), *dSB = to_dev(SB);
    // ---- Float vs host reference (pins the hardware per-block scale model) ----
    {
        std::vector<float> ref(SM * SN);
        for(int i = 0; i < SM; ++i)
            for(int j = 0; j < SN; ++j)
            {
                double acc = m.C[i * SN + j];
                for(int kb = 0; kb < NB; ++kb)
                {
                    double blk = 0;
                    for(int k = 32 * kb; k < 32 * kb + 32; ++k)
                        blk += (double)m.A[i * SK + k] * (double)m.B[k * SN + j];
                    acc += blk * (double)e8(eA[i * NB + kb]) * (double)e8(eB[j * NB + kb]);
                }
                ref[i * SN + j] = (float)acc;
            }
        float* dD;
        HIP_CHECK(hipMalloc(&dD, SM * SN * sizeof(float)));
        k_scale16_pb<Semantics::Native, float, AElem, BElem, CBSZ, BLGP>
            <<<1, WAVE>>>(dA, dB, dC, dD, dSA, dSB);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<float> got(SM * SN);
        HIP_CHECK(hipMemcpy(got.data(), dD, SM * SN * sizeof(float), hipMemcpyDeviceToHost));
        for(int i = 0; i < SM * SN; ++i)
            EXPECT_EQ(bits_of(got[i]), bits_of(ref[i])) << "Float at " << i;
        (void)hipFree(dD);
    }
    // ---- FPSan vs payload-ring reference --------------------------------------
    {
        using VF = Value<float, Semantics::Triton, kCC>;
        using VA = Value<AElem, Semantics::Triton, kCC>;
        using VB = Value<BElem, Semantics::Triton, kCC>;
        std::vector<std::uint32_t> ref(SM * SN);
        for(int i = 0; i < SM; ++i)
            for(int j = 0; j < SN; ++j)
            {
                VF acc(m.C[i * SN + j]);
                for(int kb = 0; kb < NB; ++kb)
                {
                    VF blk(0.0f);
                    for(int k = 32 * kb; k < 32 * kb + 32; ++k)
                        blk = blk
                              + fpsan::cast<float>(VA(AElem(m.A[i * SK + k])))
                                    * fpsan::cast<float>(VB(BElem(m.B[k * SN + j])));
                    acc = acc + blk * VF(e8(eA[i * NB + kb])) * VF(e8(eB[j * NB + kb]));
                }
                ref[i * SN + j] = acc.fpsan_payload();
            }
        std::uint32_t* dD;
        HIP_CHECK(hipMalloc(&dD, SM * SN * sizeof(std::uint32_t)));
        k_scale16_pb<Semantics::Triton, std::uint32_t, AElem, BElem, CBSZ, BLGP>
            <<<1, WAVE>>>(dA, dB, dC, dD, dSA, dSB);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<std::uint32_t> got(SM * SN);
        HIP_CHECK(
            hipMemcpy(got.data(), dD, SM * SN * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
        for(int i = 0; i < SM * SN; ++i)
            EXPECT_EQ(got[i], ref[i]) << "FPSan at " << i;
        (void)hipFree(dD);
    }
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dSA);
    (void)hipFree(dSB);
}

TEST(ScaledMfma16x16x128_PerBlockScale, E4M3)
{
    run_scale16_perblock<fp8_e4m3, fp8_e4m3, 0, 0>();
}
TEST(ScaledMfma16x16x128_PerBlockScale, E5M2)
{
    run_scale16_perblock<fp8_e5m2, fp8_e5m2, 1, 1>();
}

// ---- fp8 32x32x64 per-block scale -----------------------------------------
template <Semantics S, class Out, class AElem, class BElem, int CBSZ, int BLGP>
__global__ void k_scale32_pb(
    const float* A, const float* B, const float* C, Out* D, const int* SA, const int* SB)
{
    using AFrag = fpsan::detail::v32_fragment<AElem>;
    using BFrag = fpsan::detail::v32_fragment<BElem>;
    int   lane  = threadIdx.x;
    int   mn    = (lane % 16) + 16 * ((lane / 16) & 1);
    AFrag an{};
    BFrag bn{};
    for(int b = 0; b < 32; ++b)
    {
        int k = (b / 16) * 32 + ((lane / 16) / 2) * 16 + (b % 16);
        an[b] = AElem(A[mn * S2K + k]);
        bn[b] = BElem(B[k * S2N + mn]);
    }
    fpsan::v16f_native cn{};
    for(int i = 0; i < S2M; ++i)
        for(int j = 0; j < S2N; ++j)
        {
            auto loc = fpsan::detail::output_loc_32(S2M, S2N, i, j, 0);
            if(loc.lane == lane)
                cn[loc.reg] = C[i * S2N + j];
        }
    Value<AFrag, S, kCC>              a{an};
    Value<BFrag, S, kCC>              b{bn};
    Value<fpsan::v16f_native, S, kCC> c{cn};
    auto d = fpsan::amdgcn_mfma_scale_f32_32x32x64_f8f6f4<CBSZ, 0, BLGP, 0, 0, S, kCC>(
        a, b, c, SA[lane], SB[lane]);
    for(int i = 0; i < S2M; ++i)
        for(int j = 0; j < S2N; ++j)
        {
            auto loc = fpsan::detail::output_loc_32(S2M, S2N, i, j, 0);
            if(loc.lane == lane)
            {
                if constexpr(S == Semantics::Native)
                    D[i * S2N + j] = d.get(loc.reg).to_float();
                else
                    D[i * S2N + j] = d.get(loc.reg).fpsan_payload();
            }
        }
}

template <class AElem, class BElem, int CBSZ, int BLGP>
void run_scale32_perblock()
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    const int        NB  = S2K / 32; // 2 K-blocks
    ScaleMats        m   = make_scale_mats32();
    std::mt19937     rng = fpsan_test::make_rng();
    std::vector<int> eA = rand_exps(rng, S2M * NB), eB = rand_exps(rng, S2N * NB);
    std::vector<int> SA(WAVE, 0), SB(WAVE, 0);
    for(int kb = 0; kb < NB; ++kb)
        for(int mrow = 0; mrow < S2M; ++mrow)
            SA[32 * kb + mrow] = eA[mrow * NB + kb];
    for(int kb = 0; kb < NB; ++kb)
        for(int n = 0; n < S2N; ++n)
            SB[32 * kb + n] = eB[n * NB + kb];
    auto   e8 = [](int b) { return fpsan::detail::e8m0_to_float((unsigned)b); };
    float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C);
    int *  dSA = to_dev(SA), *dSB = to_dev(SB);
    {
        std::vector<float> ref(S2M * S2N);
        for(int i = 0; i < S2M; ++i)
            for(int j = 0; j < S2N; ++j)
            {
                double acc = m.C[i * S2N + j];
                for(int kb = 0; kb < NB; ++kb)
                {
                    double blk = 0;
                    for(int k = 32 * kb; k < 32 * kb + 32; ++k)
                        blk += (double)m.A[i * S2K + k] * (double)m.B[k * S2N + j];
                    acc += blk * (double)e8(eA[i * NB + kb]) * (double)e8(eB[j * NB + kb]);
                }
                ref[i * S2N + j] = (float)acc;
            }
        float* dD;
        HIP_CHECK(hipMalloc(&dD, S2M * S2N * sizeof(float)));
        k_scale32_pb<Semantics::Native, float, AElem, BElem, CBSZ, BLGP>
            <<<1, WAVE>>>(dA, dB, dC, dD, dSA, dSB);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<float> got(S2M * S2N);
        HIP_CHECK(hipMemcpy(got.data(), dD, S2M * S2N * sizeof(float), hipMemcpyDeviceToHost));
        for(int i = 0; i < S2M * S2N; ++i)
            EXPECT_EQ(bits_of(got[i]), bits_of(ref[i])) << "Float at " << i;
        (void)hipFree(dD);
    }
    {
        using VF = Value<float, Semantics::Triton, kCC>;
        using VA = Value<AElem, Semantics::Triton, kCC>;
        using VB = Value<BElem, Semantics::Triton, kCC>;
        std::vector<std::uint32_t> ref(S2M * S2N);
        for(int i = 0; i < S2M; ++i)
            for(int j = 0; j < S2N; ++j)
            {
                VF acc(m.C[i * S2N + j]);
                for(int kb = 0; kb < NB; ++kb)
                {
                    VF blk(0.0f);
                    for(int k = 32 * kb; k < 32 * kb + 32; ++k)
                        blk = blk
                              + fpsan::cast<float>(VA(AElem(m.A[i * S2K + k])))
                                    * fpsan::cast<float>(VB(BElem(m.B[k * S2N + j])));
                    acc = acc + blk * VF(e8(eA[i * NB + kb])) * VF(e8(eB[j * NB + kb]));
                }
                ref[i * S2N + j] = acc.fpsan_payload();
            }
        std::uint32_t* dD;
        HIP_CHECK(hipMalloc(&dD, S2M * S2N * sizeof(std::uint32_t)));
        k_scale32_pb<Semantics::Triton, std::uint32_t, AElem, BElem, CBSZ, BLGP>
            <<<1, WAVE>>>(dA, dB, dC, dD, dSA, dSB);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<std::uint32_t> got(S2M * S2N);
        HIP_CHECK(
            hipMemcpy(got.data(), dD, S2M * S2N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
        for(int i = 0; i < S2M * S2N; ++i)
            EXPECT_EQ(got[i], ref[i]) << "FPSan at " << i;
        (void)hipFree(dD);
    }
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dSA);
    (void)hipFree(dSB);
}

TEST(ScaledMfma32x32x64_PerBlockScale, E4M3)
{
    run_scale32_perblock<fp8_e4m3, fp8_e4m3, 0, 0>();
}

// ---- sub fp4 16x16x128 per-block scale (confirms the sub path reads the same
// per-block scale lanes) ----------------------------------------------------
template <Semantics S, int CBSZ, int BLGP, class Out>
__global__ void k_scale_sub16_pb(const fpsan::v8i32_native* A,
                                 const fpsan::v8i32_native* B,
                                 const float*               C,
                                 Out*                       D,
                                 const int*                 SA,
                                 const int*                 SB)
{
    int                 lane = threadIdx.x;
    fpsan::v8i32_native an = A[lane], bn = B[lane];
    fpsan::v4f_native   cn{};
    for(int reg = 0; reg < 4; ++reg)
    {
        int i = 4 * (lane / 16) + reg, j = lane % 16;
        cn[reg] = C[i * SN + j];
    }
    Value<fpsan::v4f_native, S, kCC> c{cn};
    auto d = fpsan::amdgcn_mfma_scale_f32_16x16x128_f8f6f4_sub<CBSZ, BLGP, 0, 0, S, kCC>(
        an, bn, c, SA[lane], SB[lane]);
    for(int reg = 0; reg < 4; ++reg)
    {
        int i = 4 * (lane / 16) + reg, j = lane % 16;
        if constexpr(S == Semantics::Native)
            D[i * SN + j] = d.get(reg).to_float();
        else
            D[i * SN + j] = d.get(reg).fpsan_payload();
    }
}

template <int CBSZ, int BLGP>
void run_scale_sub16_perblock_fpsan()
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    const int        wa = sub_width(CBSZ), wb = sub_width(BLGP), NB = SK / 32;
    std::mt19937     rng = fpsan_test::make_rng();
    std::vector<int> Ap(SM * SK), Bp(SK * SN);
    for(auto& x : Ap)
        x = (int)(rng() & ((1u << wa) - 1u));
    for(auto& x : Bp)
        x = (int)(rng() & ((1u << wb) - 1u));
    std::vector<float> Cm(SM * SN);
    for(auto& x : Cm)
        x = fpsan_test::pick_int_valued<float>(rng, -8, 8);
    std::vector<int> eA = rand_exps(rng, SM * NB), eB = rand_exps(rng, SN * NB);
    std::vector<int> SA(WAVE, 0), SB(WAVE, 0);
    for(int kb = 0; kb < NB; ++kb)
        for(int i = 0; i < SM; ++i)
            SA[16 * kb + i] = eA[i * NB + kb];
    for(int kb = 0; kb < NB; ++kb)
        for(int j = 0; j < SN; ++j)
            SB[16 * kb + j] = eB[j * NB + kb];
    std::vector<fpsan::v8i32_native> A(WAVE, fpsan::v8i32_native{}), B(WAVE, fpsan::v8i32_native{});
    for(int i = 0; i < SM; ++i)
        for(int k = 0; k < SK; ++k)
        {
            int lane = 16 * ((k % 64) / 16) + i, s = (k / 64) * 16 + (k % 16);
            set_field(A[lane], wa * s, (unsigned)Ap[i * SK + k], wa);
        }
    for(int k = 0; k < SK; ++k)
        for(int j = 0; j < SN; ++j)
        {
            int lane = 16 * ((k % 64) / 16) + j, s = (k / 64) * 16 + (k % 16);
            set_field(B[lane], wb * s, (unsigned)Bp[k * SN + j], wb);
        }
    using VF   = Value<float, Semantics::Triton, kCC>;
    auto widen = [&](int code, unsigned f) {
        int          w = sub_width(code);
        std::int32_t e = (std::int32_t)(f << (32 - w)) >> (32 - w);
        return VF::from_fpsan_payload((std::uint32_t)e);
    };
    auto                       e8 = [](int b) { return fpsan::detail::e8m0_to_float((unsigned)b); };
    std::vector<std::uint32_t> ref(SM * SN);
    for(int i = 0; i < SM; ++i)
        for(int j = 0; j < SN; ++j)
        {
            VF acc(Cm[i * SN + j]);
            for(int kb = 0; kb < NB; ++kb)
            {
                VF blk(0.0f);
                for(int k = 32 * kb; k < 32 * kb + 32; ++k)
                    blk = blk
                          + widen(CBSZ, (unsigned)Ap[i * SK + k])
                                * widen(BLGP, (unsigned)Bp[k * SN + j]);
                acc = acc + blk * VF(e8(eA[i * NB + kb])) * VF(e8(eB[j * NB + kb]));
            }
            ref[i * SN + j] = acc.fpsan_payload();
        }
    fpsan::v8i32_native *dA, *dB;
    float*               dC;
    std::uint32_t*       dD;
    HIP_CHECK(hipMalloc(&dA, WAVE * sizeof(fpsan::v8i32_native)));
    HIP_CHECK(hipMalloc(&dB, WAVE * sizeof(fpsan::v8i32_native)));
    HIP_CHECK(hipMalloc(&dC, SM * SN * sizeof(float)));
    HIP_CHECK(hipMalloc(&dD, SM * SN * sizeof(std::uint32_t)));
    HIP_CHECK(hipMemcpy(dA, A.data(), WAVE * sizeof(fpsan::v8i32_native), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, B.data(), WAVE * sizeof(fpsan::v8i32_native), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dC, Cm.data(), SM * SN * sizeof(float), hipMemcpyHostToDevice));
    int *dSA = to_dev(SA), *dSB = to_dev(SB);
    k_scale_sub16_pb<Semantics::Triton, CBSZ, BLGP, std::uint32_t>
        <<<1, WAVE>>>(dA, dB, dC, dD, dSA, dSB);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(SM * SN);
    HIP_CHECK(hipMemcpy(got.data(), dD, SM * SN * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for(int i = 0; i < SM * SN; ++i)
        EXPECT_EQ(got[i], ref[i]) << "at " << i;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
    (void)hipFree(dSA);
    (void)hipFree(dSB);
}

TEST(ScaledMfma16x16x128_PerBlockScale, FP4Sub)
{
    run_scale_sub16_perblock_fpsan<4, 4>();
}

// ===========================================================================
// Mixed-WIDTH 8-bit x sub-byte (fp8 x fp4 / fp6, and the mirror). The sub
// operand follows the silicon-RE'd "mix model" that pairs it with fp8
// (different physical k order than fp8). Uniform scale here isolates the mixed
// LAYOUT (per-block scale is covered above). Proven with full-block-random
// coverage.
// ===========================================================================
namespace
{
    // 16x16x128 sub mix-model physical slot for logical k and output index.
    void sub_mix_loc16(int k, int idx, int& lane, int& slot)
    {
        int q0 = 2 * ((k >> 6) & 1) + ((k >> 5) & 1);
        lane   = 16 * q0 + idx;
        slot   = 16 * ((k >> 4) & 1) + (k & 15);
    }
    // 32x32x64 sub mix-model physical slot.
    void sub_mix_loc32(int k, int idx, int& lane, int& slot)
    {
        lane = 16 * (2 * (k / 32) + idx / 16) + (idx % 16);
        slot = 16 * ((k % 32) / 16) + (k % 16);
    }
} // namespace

// A = fp8 (built from float matrix), B = sub-byte (raw v8i32, mix-packed).
template <Semantics S, class Out, class AElem, int CBSZ, int BLGP>
__global__ void k_scale16_mix_a8(
    const float* A, const fpsan::v8i32_native* B, const float* C, Out* D, int sa, int sb)
{
    using AFrag = fpsan::detail::v32_fragment<AElem>;
    int   lane  = threadIdx.x;
    AFrag an{};
    for(int half = 0; half < 2; ++half)
        for(int kk = 0; kk < 16; ++kk)
        {
            int g = lane / 16, idx = lane % 16;
            int k = half * 64 + g * 16 + kk, byte = half * 16 + kk;
            an[byte] = AElem(A[idx * SK + k]);
        }
    fpsan::v8i32_native bn = B[lane];
    fpsan::v4f_native   cn{};
    for(int reg = 0; reg < 4; ++reg)
    {
        int i = 4 * (lane / 16) + reg, j = lane % 16;
        cn[reg] = C[i * SN + j];
    }
    Value<AFrag, S, kCC>             a{an};
    Value<fpsan::v4f_native, S, kCC> c{cn};
    auto                             d
        = fpsan::amdgcn_mfma_scale_f32_16x16x128_f8f6f4_mixed_a8<CBSZ, BLGP, 0, 0, S, kCC, AFrag>(
            a, bn, c, sa, sb);
    for(int reg = 0; reg < 4; ++reg)
    {
        int i = 4 * (lane / 16) + reg, j = lane % 16;
        if constexpr(S == Semantics::Native)
            D[i * SN + j] = d.get(reg).to_float();
        else
            D[i * SN + j] = d.get(reg).fpsan_payload();
    }
}

// A = sub-byte (raw v8i32, mix-packed), B = fp8.
template <Semantics S, class Out, class BElem, int CBSZ, int BLGP>
__global__ void k_scale16_mix_b8(
    const fpsan::v8i32_native* A, const float* B, const float* C, Out* D, int sa, int sb)
{
    using BFrag              = fpsan::detail::v32_fragment<BElem>;
    int                 lane = threadIdx.x;
    fpsan::v8i32_native an   = A[lane];
    BFrag               bn{};
    for(int half = 0; half < 2; ++half)
        for(int kk = 0; kk < 16; ++kk)
        {
            int g = lane / 16, idx = lane % 16;
            int k = half * 64 + g * 16 + kk, byte = half * 16 + kk;
            bn[byte] = BElem(B[k * SN + idx]);
        }
    fpsan::v4f_native cn{};
    for(int reg = 0; reg < 4; ++reg)
    {
        int i = 4 * (lane / 16) + reg, j = lane % 16;
        cn[reg] = C[i * SN + j];
    }
    Value<BFrag, S, kCC>             b{bn};
    Value<fpsan::v4f_native, S, kCC> c{cn};
    auto                             d
        = fpsan::amdgcn_mfma_scale_f32_16x16x128_f8f6f4_mixed_b8<CBSZ, BLGP, 0, 0, S, kCC, BFrag>(
            an, b, c, sa, sb);
    for(int reg = 0; reg < 4; ++reg)
    {
        int i = 4 * (lane / 16) + reg, j = lane % 16;
        if constexpr(S == Semantics::Native)
            D[i * SN + j] = d.get(reg).to_float();
        else
            D[i * SN + j] = d.get(reg).fpsan_payload();
    }
}

// AIsSub=false: A fp8, B sub. AIsSub=true: A sub, B fp8.
template <bool AIsSub, class Fp8Elem, int CBSZ, int BLGP>
void run_scale16_mixed()
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    const int    subfmt = AIsSub ? CBSZ : BLGP, w = sub_width(subfmt);
    std::mt19937 rng = fpsan_test::make_rng();
    const int    sa = 129, sb = 126;
    const float  fa = fpsan::detail::e8m0_to_float(129), fb = fpsan::detail::e8m0_to_float(126);
    // fp8 matrix (small ints) + sub code matrix (small ints, exact in the format).
    std::vector<float> Afp8, Bfp8, Cm(SM * SN);
    std::vector<int>   subcode(SM * SK); // [i][k] if A sub, else [k][j] reused
    for(auto& x : Cm)
        x = fpsan_test::pick_int_valued<float>(rng, -6, 6);
    // We always lay out A as [i][k] and B as [k][j] logically.
    std::vector<float> fp8mat;
    std::vector<int>   submat;
    fp8mat.resize(AIsSub ? SK * SN : SM * SK);
    submat.resize(AIsSub ? SM * SK : SK * SN);
    for(auto& x : fp8mat)
        x = fpsan_test::pick_int_valued<float>(rng, -3, 3);
    for(auto& x : submat)
        x = (int)sub_encode(subfmt, fpsan_test::pick_int(rng, -2, 3));
    // Pack the sub operand in mix-model.
    std::vector<fpsan::v8i32_native> sub(WAVE, fpsan::v8i32_native{});
    if(AIsSub)
    {
        for(int i = 0; i < SM; ++i)
            for(int k = 0; k < SK; ++k)
            {
                int lane, slot;
                sub_mix_loc16(k, i, lane, slot);
                set_field(sub[lane], w * slot, (unsigned)submat[i * SK + k], w);
            }
    }
    else
    {
        for(int k = 0; k < SK; ++k)
            for(int j = 0; j < SN; ++j)
            {
                int lane, slot;
                sub_mix_loc16(k, j, lane, slot);
                set_field(sub[lane], w * slot, (unsigned)submat[k * SN + j], w);
            }
    }
    // Host reference (small ints exact): D = C + (sum_k A*B) * fa * fb.
    auto Aval = [&](int i, int k) {
        return AIsSub ? (double)sub_decode(subfmt, (unsigned)submat[i * SK + k])
                      : (double)fp8mat[i * SK + k];
    };
    auto Bval = [&](int k, int j) {
        return AIsSub ? (double)fp8mat[k * SN + j]
                      : (double)sub_decode(subfmt, (unsigned)submat[k * SN + j]);
    };
    std::vector<float> ref(SM * SN);
    for(int i = 0; i < SM; ++i)
        for(int j = 0; j < SN; ++j)
        {
            double dot = 0;
            for(int k = 0; k < SK; ++k)
                dot += Aval(i, k) * Bval(k, j);
            ref[i * SN + j] = (float)((double)Cm[i * SN + j] + dot * fa * fb);
        }
    float *              dF = to_dev(fp8mat), *dC = to_dev(Cm), *dD;
    fpsan::v8i32_native* dSub;
    HIP_CHECK(hipMalloc(&dSub, WAVE * sizeof(fpsan::v8i32_native)));
    HIP_CHECK(
        hipMemcpy(dSub, sub.data(), WAVE * sizeof(fpsan::v8i32_native), hipMemcpyHostToDevice));
    HIP_CHECK(hipMalloc(&dD, SM * SN * sizeof(float)));
    if constexpr(AIsSub)
        k_scale16_mix_b8<Semantics::Native, float, Fp8Elem, CBSZ, BLGP>
            <<<1, WAVE>>>(dSub, dF, dC, dD, sa, sb);
    else
        k_scale16_mix_a8<Semantics::Native, float, Fp8Elem, CBSZ, BLGP>
            <<<1, WAVE>>>(dF, dSub, dC, dD, sa, sb);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> got(SM * SN);
    HIP_CHECK(hipMemcpy(got.data(), dD, SM * SN * sizeof(float), hipMemcpyDeviceToHost));
    for(int i = 0; i < SM * SN; ++i)
        EXPECT_EQ(bits_of(got[i]), bits_of(ref[i])) << "at " << i;
    (void)hipFree(dF);
    (void)hipFree(dC);
    (void)hipFree(dD);
    (void)hipFree(dSub);
}

TEST(ScaledMfma16x16x128_Mixed8xSub, FP8xFP4_Layout)
{
    run_scale16_mixed<false, fp8_e4m3, 0, 4>();
}
TEST(ScaledMfma16x16x128_Mixed8xSub, FP8xFP6_Layout)
{
    run_scale16_mixed<false, fp8_e4m3, 0, 2>();
}
TEST(ScaledMfma16x16x128_Mixed8xSub, BF8xFP4_Layout)
{
    run_scale16_mixed<false, fp8_e5m2, 1, 4>();
}
TEST(ScaledMfma16x16x128_Mixed8xSub, FP4xFP8_Layout)
{
    run_scale16_mixed<true, fp8_e4m3, 4, 0>();
}
TEST(ScaledMfma16x16x128_Mixed8xSub, FP6xFP8_Layout)
{
    run_scale16_mixed<true, fp8_e4m3, 2, 0>();
}

// FPSan-mode mixed 8 x sub: fp8 operand carries real fp8 values (cast widening);
// sub operand carries arbitrary Width-bit payloads (signed-resize widening).
template <bool AIsSub, class Fp8Elem, int CBSZ, int BLGP>
void run_scale16_mixed_fpsan()
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    const int    subfmt = AIsSub ? CBSZ : BLGP, w = sub_width(subfmt);
    std::mt19937 rng = fpsan_test::make_rng();
    const int    sa = 128, sb = 130;
    using VF   = Value<float, Semantics::Triton, kCC>;
    using VFp8 = Value<Fp8Elem, Semantics::Triton, kCC>;
    const VF vsa(fpsan::detail::e8m0_to_float(128)), vsb(fpsan::detail::e8m0_to_float(130));
    std::vector<float> fp8mat(AIsSub ? SK * SN : SM * SK), Cm(SM * SN);
    std::vector<int>   subpay(AIsSub ? SM * SK : SK * SN);
    for(auto& x : fp8mat)
        x = fpsan_test::pick_int_valued<float>(rng, -3, 3);
    for(auto& x : subpay)
        x = (int)(rng() & ((1u << w) - 1u));
    for(auto& x : Cm)
        x = fpsan_test::pick_int_valued<float>(rng, -6, 6);
    std::vector<fpsan::v8i32_native> sub(WAVE, fpsan::v8i32_native{});
    if(AIsSub)
        for(int i = 0; i < SM; ++i)
            for(int k = 0; k < SK; ++k)
            {
                int lane, slot;
                sub_mix_loc16(k, i, lane, slot);
                set_field(sub[lane], w * slot, (unsigned)subpay[i * SK + k], w);
            }
    else
        for(int k = 0; k < SK; ++k)
            for(int j = 0; j < SN; ++j)
            {
                int lane, slot;
                sub_mix_loc16(k, j, lane, slot);
                set_field(sub[lane], w * slot, (unsigned)subpay[k * SN + j], w);
            }
    auto widen = [&](unsigned f) {
        std::int32_t e = (std::int32_t)(f << (32 - w)) >> (32 - w);
        return VF::from_fpsan_payload((std::uint32_t)e);
    };
    auto Av = [&](int i, int k) {
        return AIsSub ? widen((unsigned)subpay[i * SK + k])
                      : fpsan::cast<float>(VFp8(Fp8Elem(fp8mat[i * SK + k])));
    };
    auto Bv = [&](int k, int j) {
        return AIsSub ? fpsan::cast<float>(VFp8(Fp8Elem(fp8mat[k * SN + j])))
                      : widen((unsigned)subpay[k * SN + j]);
    };
    std::vector<std::uint32_t> ref(SM * SN);
    for(int i = 0; i < SM; ++i)
        for(int j = 0; j < SN; ++j)
        {
            VF dot(0.0f);
            for(int k = 0; k < SK; ++k)
                dot = dot + Av(i, k) * Bv(k, j);
            ref[i * SN + j] = (VF(Cm[i * SN + j]) + dot * vsa * vsb).fpsan_payload();
        }
    float *              dF = to_dev(fp8mat), *dC = to_dev(Cm);
    std::uint32_t*       dD;
    fpsan::v8i32_native* dSub;
    HIP_CHECK(hipMalloc(&dSub, WAVE * sizeof(fpsan::v8i32_native)));
    HIP_CHECK(
        hipMemcpy(dSub, sub.data(), WAVE * sizeof(fpsan::v8i32_native), hipMemcpyHostToDevice));
    HIP_CHECK(hipMalloc(&dD, SM * SN * sizeof(std::uint32_t)));
    if constexpr(AIsSub)
        k_scale16_mix_b8<Semantics::Triton, std::uint32_t, Fp8Elem, CBSZ, BLGP>
            <<<1, WAVE>>>(dSub, dF, dC, dD, sa, sb);
    else
        k_scale16_mix_a8<Semantics::Triton, std::uint32_t, Fp8Elem, CBSZ, BLGP>
            <<<1, WAVE>>>(dF, dSub, dC, dD, sa, sb);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(SM * SN);
    HIP_CHECK(hipMemcpy(got.data(), dD, SM * SN * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for(int i = 0; i < SM * SN; ++i)
        EXPECT_EQ(got[i], ref[i]) << "at " << i;
    (void)hipFree(dF);
    (void)hipFree(dC);
    (void)hipFree(dD);
    (void)hipFree(dSub);
}

TEST(ScaledMfma16x16x128_Mixed8xSub, FP8xFP4_Fpsan)
{
    run_scale16_mixed_fpsan<false, fp8_e4m3, 0, 4>();
}
TEST(ScaledMfma16x16x128_Mixed8xSub, FP4xFP8_Fpsan)
{
    run_scale16_mixed_fpsan<true, fp8_e4m3, 4, 0>();
}
TEST(ScaledMfma16x16x128_Mixed8xSub, FP8xFP6_Fpsan)
{
    run_scale16_mixed_fpsan<false, fp8_e4m3, 0, 2>();
}
TEST(ScaledMfma16x16x128_Mixed8xSub, BF8xFP4_Fpsan)
{
    run_scale16_mixed_fpsan<false, fp8_e5m2, 1, 4>();
}

// ---- 32x32x64 mixed 8 x sub (Float layout) --------------------------------
template <Semantics S, class Out, class AElem, int CBSZ, int BLGP>
__global__ void k_scale32_mix_a8(
    const float* A, const fpsan::v8i32_native* B, const float* C, Out* D, int sa, int sb)
{
    using AFrag = fpsan::detail::v32_fragment<AElem>;
    int   lane  = threadIdx.x;
    int   mn    = (lane % 16) + 16 * ((lane / 16) & 1);
    AFrag an{};
    for(int b = 0; b < 32; ++b)
    {
        int k = (b / 16) * 32 + ((lane / 16) / 2) * 16 + (b % 16);
        an[b] = AElem(A[mn * S2K + k]);
    }
    fpsan::v8i32_native bn = B[lane];
    fpsan::v16f_native  cn{};
    for(int i = 0; i < S2M; ++i)
        for(int j = 0; j < S2N; ++j)
        {
            auto loc = fpsan::detail::output_loc_32(S2M, S2N, i, j, 0);
            if(loc.lane == lane)
                cn[loc.reg] = C[i * S2N + j];
        }
    Value<AFrag, S, kCC>              a{an};
    Value<fpsan::v16f_native, S, kCC> c{cn};
    auto d = fpsan::amdgcn_mfma_scale_f32_32x32x64_f8f6f4_mixed_a8<CBSZ, BLGP, 0, 0, S, kCC, AFrag>(
        a, bn, c, sa, sb);
    for(int i = 0; i < S2M; ++i)
        for(int j = 0; j < S2N; ++j)
        {
            auto loc = fpsan::detail::output_loc_32(S2M, S2N, i, j, 0);
            if(loc.lane == lane)
            {
                if constexpr(S == Semantics::Native)
                    D[i * S2N + j] = d.get(loc.reg).to_float();
                else
                    D[i * S2N + j] = d.get(loc.reg).fpsan_payload();
            }
        }
}

template <Semantics S, class Out, class BElem, int CBSZ, int BLGP>
__global__ void k_scale32_mix_b8(
    const fpsan::v8i32_native* A, const float* B, const float* C, Out* D, int sa, int sb)
{
    using BFrag = fpsan::detail::v32_fragment<BElem>;
    int   lane  = threadIdx.x;
    int   mn    = (lane % 16) + 16 * ((lane / 16) & 1);
    BFrag bn{};
    for(int b = 0; b < 32; ++b)
    {
        int k = (b / 16) * 32 + ((lane / 16) / 2) * 16 + (b % 16);
        bn[b] = BElem(B[k * S2N + mn]);
    }
    fpsan::v8i32_native an = A[lane];
    fpsan::v16f_native  cn{};
    for(int i = 0; i < S2M; ++i)
        for(int j = 0; j < S2N; ++j)
        {
            auto loc = fpsan::detail::output_loc_32(S2M, S2N, i, j, 0);
            if(loc.lane == lane)
                cn[loc.reg] = C[i * S2N + j];
        }
    Value<BFrag, S, kCC>              b{bn};
    Value<fpsan::v16f_native, S, kCC> c{cn};
    auto d = fpsan::amdgcn_mfma_scale_f32_32x32x64_f8f6f4_mixed_b8<CBSZ, BLGP, 0, 0, S, kCC, BFrag>(
        an, b, c, sa, sb);
    for(int i = 0; i < S2M; ++i)
        for(int j = 0; j < S2N; ++j)
        {
            auto loc = fpsan::detail::output_loc_32(S2M, S2N, i, j, 0);
            if(loc.lane == lane)
            {
                if constexpr(S == Semantics::Native)
                    D[i * S2N + j] = d.get(loc.reg).to_float();
                else
                    D[i * S2N + j] = d.get(loc.reg).fpsan_payload();
            }
        }
}

template <bool AIsSub, class Fp8Elem, int CBSZ, int BLGP>
void run_scale32_mixed()
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    const int    subfmt = AIsSub ? CBSZ : BLGP, w = sub_width(subfmt);
    std::mt19937 rng = fpsan_test::make_rng();
    const int    sa = 129, sb = 126;
    const float  fa = fpsan::detail::e8m0_to_float(129), fb = fpsan::detail::e8m0_to_float(126);
    std::vector<float> fp8mat(AIsSub ? S2K * S2N : S2M * S2K), Cm(S2M * S2N);
    std::vector<int>   submat(AIsSub ? S2M * S2K : S2K * S2N);
    for(auto& x : fp8mat)
        x = fpsan_test::pick_int_valued<float>(rng, -3, 3);
    for(auto& x : submat)
        x = (int)sub_encode(subfmt, fpsan_test::pick_int(rng, -2, 3));
    for(auto& x : Cm)
        x = fpsan_test::pick_int_valued<float>(rng, -6, 6);
    std::vector<fpsan::v8i32_native> sub(WAVE, fpsan::v8i32_native{});
    if(AIsSub)
        for(int i = 0; i < S2M; ++i)
            for(int k = 0; k < S2K; ++k)
            {
                int lane, slot;
                sub_mix_loc32(k, i, lane, slot);
                set_field(sub[lane], w * slot, (unsigned)submat[i * S2K + k], w);
            }
    else
        for(int k = 0; k < S2K; ++k)
            for(int j = 0; j < S2N; ++j)
            {
                int lane, slot;
                sub_mix_loc32(k, j, lane, slot);
                set_field(sub[lane], w * slot, (unsigned)submat[k * S2N + j], w);
            }
    auto Aval = [&](int i, int k) {
        return AIsSub ? (double)sub_decode(subfmt, (unsigned)submat[i * S2K + k])
                      : (double)fp8mat[i * S2K + k];
    };
    auto Bval = [&](int k, int j) {
        return AIsSub ? (double)fp8mat[k * S2N + j]
                      : (double)sub_decode(subfmt, (unsigned)submat[k * S2N + j]);
    };
    std::vector<float> ref(S2M * S2N);
    for(int i = 0; i < S2M; ++i)
        for(int j = 0; j < S2N; ++j)
        {
            double dot = 0;
            for(int k = 0; k < S2K; ++k)
                dot += Aval(i, k) * Bval(k, j);
            ref[i * S2N + j] = (float)((double)Cm[i * S2N + j] + dot * fa * fb);
        }
    float *              dF = to_dev(fp8mat), *dC = to_dev(Cm), *dD;
    fpsan::v8i32_native* dSub;
    HIP_CHECK(hipMalloc(&dSub, WAVE * sizeof(fpsan::v8i32_native)));
    HIP_CHECK(
        hipMemcpy(dSub, sub.data(), WAVE * sizeof(fpsan::v8i32_native), hipMemcpyHostToDevice));
    HIP_CHECK(hipMalloc(&dD, S2M * S2N * sizeof(float)));
    if constexpr(AIsSub)
        k_scale32_mix_b8<Semantics::Native, float, Fp8Elem, CBSZ, BLGP>
            <<<1, WAVE>>>(dSub, dF, dC, dD, sa, sb);
    else
        k_scale32_mix_a8<Semantics::Native, float, Fp8Elem, CBSZ, BLGP>
            <<<1, WAVE>>>(dF, dSub, dC, dD, sa, sb);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> got(S2M * S2N);
    HIP_CHECK(hipMemcpy(got.data(), dD, S2M * S2N * sizeof(float), hipMemcpyDeviceToHost));
    for(int i = 0; i < S2M * S2N; ++i)
        EXPECT_EQ(bits_of(got[i]), bits_of(ref[i])) << "at " << i;
    (void)hipFree(dF);
    (void)hipFree(dC);
    (void)hipFree(dD);
    (void)hipFree(dSub);
}

template <bool AIsSub, class Fp8Elem, int CBSZ, int BLGP>
void run_scale32_mixed_fpsan()
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    const int    subfmt = AIsSub ? CBSZ : BLGP, w = sub_width(subfmt);
    std::mt19937 rng = fpsan_test::make_rng();
    const int    sa = 128, sb = 130;
    using VF   = Value<float, Semantics::Triton, kCC>;
    using VFp8 = Value<Fp8Elem, Semantics::Triton, kCC>;
    const VF vsa(fpsan::detail::e8m0_to_float(128)), vsb(fpsan::detail::e8m0_to_float(130));
    std::vector<float> fp8mat(AIsSub ? S2K * S2N : S2M * S2K), Cm(S2M * S2N);
    std::vector<int>   subpay(AIsSub ? S2M * S2K : S2K * S2N);
    for(auto& x : fp8mat)
        x = fpsan_test::pick_int_valued<float>(rng, -3, 3);
    for(auto& x : subpay)
        x = (int)(rng() & ((1u << w) - 1u));
    for(auto& x : Cm)
        x = fpsan_test::pick_int_valued<float>(rng, -6, 6);
    std::vector<fpsan::v8i32_native> sub(WAVE, fpsan::v8i32_native{});
    if(AIsSub)
        for(int i = 0; i < S2M; ++i)
            for(int k = 0; k < S2K; ++k)
            {
                int lane, slot;
                sub_mix_loc32(k, i, lane, slot);
                set_field(sub[lane], w * slot, (unsigned)subpay[i * S2K + k], w);
            }
    else
        for(int k = 0; k < S2K; ++k)
            for(int j = 0; j < S2N; ++j)
            {
                int lane, slot;
                sub_mix_loc32(k, j, lane, slot);
                set_field(sub[lane], w * slot, (unsigned)subpay[k * S2N + j], w);
            }
    auto widen = [&](unsigned f) {
        std::int32_t e = (std::int32_t)(f << (32 - w)) >> (32 - w);
        return VF::from_fpsan_payload((std::uint32_t)e);
    };
    auto Av = [&](int i, int k) {
        return AIsSub ? widen((unsigned)subpay[i * S2K + k])
                      : fpsan::cast<float>(VFp8(Fp8Elem(fp8mat[i * S2K + k])));
    };
    auto Bv = [&](int k, int j) {
        return AIsSub ? fpsan::cast<float>(VFp8(Fp8Elem(fp8mat[k * S2N + j])))
                      : widen((unsigned)subpay[k * S2N + j]);
    };
    std::vector<std::uint32_t> ref(S2M * S2N);
    for(int i = 0; i < S2M; ++i)
        for(int j = 0; j < S2N; ++j)
        {
            VF dot(0.0f);
            for(int k = 0; k < S2K; ++k)
                dot = dot + Av(i, k) * Bv(k, j);
            ref[i * S2N + j] = (VF(Cm[i * S2N + j]) + dot * vsa * vsb).fpsan_payload();
        }
    float *              dF = to_dev(fp8mat), *dC = to_dev(Cm);
    std::uint32_t*       dD;
    fpsan::v8i32_native* dSub;
    HIP_CHECK(hipMalloc(&dSub, WAVE * sizeof(fpsan::v8i32_native)));
    HIP_CHECK(
        hipMemcpy(dSub, sub.data(), WAVE * sizeof(fpsan::v8i32_native), hipMemcpyHostToDevice));
    HIP_CHECK(hipMalloc(&dD, S2M * S2N * sizeof(std::uint32_t)));
    if constexpr(AIsSub)
        k_scale32_mix_b8<Semantics::Triton, std::uint32_t, Fp8Elem, CBSZ, BLGP>
            <<<1, WAVE>>>(dSub, dF, dC, dD, sa, sb);
    else
        k_scale32_mix_a8<Semantics::Triton, std::uint32_t, Fp8Elem, CBSZ, BLGP>
            <<<1, WAVE>>>(dF, dSub, dC, dD, sa, sb);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(S2M * S2N);
    HIP_CHECK(hipMemcpy(got.data(), dD, S2M * S2N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for(int i = 0; i < S2M * S2N; ++i)
        EXPECT_EQ(got[i], ref[i]) << "at " << i;
    (void)hipFree(dF);
    (void)hipFree(dC);
    (void)hipFree(dD);
    (void)hipFree(dSub);
}

TEST(ScaledMfma32x32x64_Mixed8xSub, FP8xFP4_Layout)
{
    run_scale32_mixed<false, fp8_e4m3, 0, 4>();
}
TEST(ScaledMfma32x32x64_Mixed8xSub, FP8xFP6_Layout)
{
    run_scale32_mixed<false, fp8_e4m3, 0, 2>();
}
TEST(ScaledMfma32x32x64_Mixed8xSub, FP4xFP8_Layout)
{
    run_scale32_mixed<true, fp8_e4m3, 4, 0>();
}
TEST(ScaledMfma32x32x64_Mixed8xSub, FP8xFP4_Fpsan)
{
    run_scale32_mixed_fpsan<false, fp8_e4m3, 0, 4>();
}
TEST(ScaledMfma32x32x64_Mixed8xSub, FP4xFP8_Fpsan)
{
    run_scale32_mixed_fpsan<true, fp8_e4m3, 4, 0>();
}
TEST(ScaledMfma32x32x64_Mixed8xSub, FP8xFP6_Fpsan)
{
    run_scale32_mixed_fpsan<false, fp8_e4m3, 0, 2>();
}

// ---------------------------------------------------------------------------
// Sparse MFMA 16x16x64 f16 (V_SMFMAC_F32_16X16X64_F16): real golden test of
// the reverse-engineered 2:4 sparse dataflow. Layouts (see amdgcn_smfmac.hpp):
//   A_comp[i][c] @ lane (c/8)*16+i, half c%8; B[k][j] @ lane j+16*((k%32)/8),
//   vreg (k%8)/2+(k>=32?4:0), half k%2; D via output_loc_32; index per lane =
//   4 nibbles for its 4 groups, nibble = idx0 | (idx1<<2).
// ---------------------------------------------------------------------------
static constexpr int QM = 16, QN = 16, QK = 64, QC = 32; // QC = compressed K

template <int CBSZ, int ABID, Semantics S, class Out>
__global__ void
    k_smf64(const float* Acomp, const float* B, const float* C, const int* idxbuf, Out* D)
{
    using v8h  = fpsan::v8h_native;
    using v16h = fpsan::v16h_native;
    int lane = threadIdx.x, g = lane / 16, nlane = lane % 16;
    v8h an{};
    for(int h = 0; h < 8; ++h)
        an[h] = (_Float16)Acomp[nlane * QC + (g * 8 + h)];
    v16h bn{};
    for(int vr = 0; vr < 8; ++vr)
        for(int h = 0; h < 2; ++h)
        {
            int k          = (vr < 4) ? (8 * g + 2 * vr + h) : (32 + 8 * g + 2 * (vr - 4) + h);
            bn[2 * vr + h] = (_Float16)B[k * QN + nlane];
        }
    fpsan::v4f_native cn{};
    for(int reg = 0; reg < 4; ++reg)
    {
        int i = 4 * g + reg, j = nlane;
        cn[reg] = C[i * QN + j];
    }
    Value<v8h, S, kCC>               a{an};
    Value<v16h, S, kCC>              b{bn};
    Value<fpsan::v4f_native, S, kCC> c{cn};
    auto d = fpsan::amdgcn_smfmac_f32_16x16x64_f16<CBSZ, ABID, S, kCC>(a, b, c, idxbuf[lane]);
    for(int reg = 0; reg < 4; ++reg)
    {
        int i = 4 * g + reg, j = nlane;
        if constexpr(S == Semantics::Native)
            D[i * QN + j] = d.get(reg).to_float();
        else
            D[i * QN + j] = d.get(reg).fpsan_payload();
    }
}

namespace
{
    struct SmfData
    {
        std::vector<float> A, B, C; // A: 16x32 compressed, B: 64x16, C: 16x16
        std::vector<int>   idxbuf; // per-lane (64)
        std::vector<int>   p0, p1; // sel[i*16+q] live K offsets, i in 0..15 q in 0..15
    };
    SmfData make_smf_data()
    {
        SmfData d;
        d.A.resize(QM * QC);
        d.B.resize(QK * QN);
        d.C.resize(QM * QN);
        d.idxbuf.assign(64, 0);
        d.p0.resize(QM * 16);
        d.p1.resize(QM * 16);
        std::mt19937 rng = fpsan_test::make_rng();
        for(auto& x : d.A)
            x = fpsan_test::pick_int_valued<float>(rng, -3, 3);
        for(auto& x : d.B)
            x = fpsan_test::pick_int_valued<float>(rng, -2, 2);
        for(auto& x : d.C)
            x = fpsan_test::pick_int_valued<float>(rng, -4, 4);
        // Random valid 2:4 selectors (p0 < p1) per (row i, group q).
        for(int i = 0; i < QM; ++i)
            for(int q = 0; q < 16; ++q)
            {
                int a0 = fpsan_test::pick_int(rng, 0, 3), a1 = fpsan_test::pick_int(rng, 0, 3);
                while(a1 == a0)
                    a1 = fpsan_test::pick_int(rng, 0, 3);
                if(a0 > a1)
                {
                    int t = a0;
                    a0    = a1;
                    a1    = t;
                }
                d.p0[i * 16 + q] = a0;
                d.p1[i * 16 + q] = a1;
            }
        // Encode into per-lane index: lane L=(q/4)*16+i holds nibble q%4 for row i.
        for(int L = 0; L < 64; ++L)
        {
            int i = L % 16, baseq = (L / 16) * 4, v = 0;
            for(int n = 0; n < 4; ++n)
            {
                int q = baseq + n;
                v |= (d.p0[i * 16 + q] | (d.p1[i * 16 + q] << 2)) << (4 * n);
            }
            d.idxbuf[L] = v;
        }
        return d;
    }
} // namespace

TEST(SmfmacF16_16x16x64, LayoutMatchesHardware)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    SmfData            m = make_smf_data();
    std::vector<float> ref(QM * QN);
    for(int i = 0; i < QM; ++i)
        for(int j = 0; j < QN; ++j)
        {
            double acc = m.C[i * QN + j];
            for(int q = 0; q < 16; ++q)
            {
                acc += (double)m.A[i * QC + 2 * q]
                       * (double)m.B[(4 * q + m.p0[i * 16 + q]) * QN + j];
                acc += (double)m.A[i * QC + 2 * q + 1]
                       * (double)m.B[(4 * q + m.p1[i * 16 + q]) * QN + j];
            }
            ref[i * QN + j] = (float)acc;
        }
    float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C), *dD;
    int*   dI = to_dev(m.idxbuf);
    HIP_CHECK(hipMalloc(&dD, QM * QN * sizeof(float)));
    k_smf64<0, 0, Semantics::Native, float><<<1, WAVE>>>(dA, dB, dC, dI, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> got(QM * QN);
    HIP_CHECK(hipMemcpy(got.data(), dD, QM * QN * sizeof(float), hipMemcpyDeviceToHost));
    for(int t = 0; t < QM * QN; ++t)
        EXPECT_EQ(bits_of(got[t]), bits_of(ref[t])) << "at " << t;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
    (void)hipFree(dI);
}

TEST(SmfmacF16_16x16x64, FpsanMatchesScalarReference)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    SmfData m = make_smf_data();
    using VF  = Value<float, Semantics::Triton, kCC>;
    using VH  = Value<_Float16, Semantics::Triton, kCC>;
    std::vector<std::uint32_t> ref(QM * QN);
    for(int i = 0; i < QM; ++i)
        for(int j = 0; j < QN; ++j)
        {
            VF acc(m.C[i * QN + j]);
            for(int q = 0; q < 16; ++q)
            {
                acc = acc
                      + fpsan::cast<float>(VH((_Float16)m.A[i * QC + 2 * q]))
                            * fpsan::cast<float>(
                                VH((_Float16)m.B[(4 * q + m.p0[i * 16 + q]) * QN + j]));
                acc = acc
                      + fpsan::cast<float>(VH((_Float16)m.A[i * QC + 2 * q + 1]))
                            * fpsan::cast<float>(
                                VH((_Float16)m.B[(4 * q + m.p1[i * 16 + q]) * QN + j]));
            }
            ref[i * QN + j] = acc.fpsan_payload();
        }
    float *        dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C);
    int*           dI = to_dev(m.idxbuf);
    std::uint32_t* dD;
    HIP_CHECK(hipMalloc(&dD, QM * QN * sizeof(std::uint32_t)));
    k_smf64<0, 0, Semantics::Triton, std::uint32_t><<<1, WAVE>>>(dA, dB, dC, dI, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(QM * QN);
    HIP_CHECK(hipMemcpy(got.data(), dD, QM * QN * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for(int t = 0; t < QM * QN; ++t)
        EXPECT_EQ(got[t], ref[t]) << "at " << t;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
    (void)hipFree(dI);
}

TEST(SmfmacF16_16x16x64_Modifiers, CBSZNonzeroUsesFirstIndexSetLayout)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    SmfData            m = make_smf_data();
    std::vector<float> ref(QM * QN);
    for(int i = 0; i < QM; ++i)
        for(int j = 0; j < QN; ++j)
        {
            double acc = m.C[i * QN + j];
            for(int q = 0; q < 16; ++q)
            {
                acc += (double)m.A[i * QC + 2 * q]
                       * (double)m.B[(4 * q + m.p0[i * 16 + q]) * QN + j];
                acc += (double)m.A[i * QC + 2 * q + 1]
                       * (double)m.B[(4 * q + m.p1[i * 16 + q]) * QN + j];
            }
            ref[i * QN + j] = (float)acc;
        }
    float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C), *dD;
    int*   dI = to_dev(m.idxbuf);
    HIP_CHECK(hipMalloc(&dD, QM * QN * sizeof(float)));
    k_smf64<1, 3, Semantics::Native, float><<<1, WAVE>>>(dA, dB, dC, dI, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> got(QM * QN);
    HIP_CHECK(hipMemcpy(got.data(), dD, QM * QN * sizeof(float), hipMemcpyDeviceToHost));
    for(int t = 0; t < QM * QN; ++t)
        EXPECT_EQ(bits_of(got[t]), bits_of(ref[t])) << "at " << t;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
    (void)hipFree(dI);
}

TEST(SmfmacF16_16x16x64_Modifiers, CBSZNonzeroUsesFirstIndexSetFpsan)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    SmfData m = make_smf_data();
    using VF  = Value<float, Semantics::Triton, kCC>;
    using VH  = Value<_Float16, Semantics::Triton, kCC>;
    std::vector<std::uint32_t> ref(QM * QN);
    for(int i = 0; i < QM; ++i)
        for(int j = 0; j < QN; ++j)
        {
            VF acc(m.C[i * QN + j]);
            for(int q = 0; q < 16; ++q)
            {
                acc = acc
                      + fpsan::cast<float>(VH((_Float16)m.A[i * QC + 2 * q]))
                            * fpsan::cast<float>(
                                VH((_Float16)m.B[(4 * q + m.p0[i * 16 + q]) * QN + j]));
                acc = acc
                      + fpsan::cast<float>(VH((_Float16)m.A[i * QC + 2 * q + 1]))
                            * fpsan::cast<float>(
                                VH((_Float16)m.B[(4 * q + m.p1[i * 16 + q]) * QN + j]));
            }
            ref[i * QN + j] = acc.fpsan_payload();
        }
    float *        dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C);
    int*           dI = to_dev(m.idxbuf);
    std::uint32_t* dD;
    HIP_CHECK(hipMalloc(&dD, QM * QN * sizeof(std::uint32_t)));
    k_smf64<1, 3, Semantics::Triton, std::uint32_t><<<1, WAVE>>>(dA, dB, dC, dI, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(QM * QN);
    HIP_CHECK(hipMemcpy(got.data(), dD, QM * QN * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for(int t = 0; t < QM * QN; ++t)
        EXPECT_EQ(got[t], ref[t]) << "at " << t;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
    (void)hipFree(dI);
}

// ---------------------------------------------------------------------------
// Sparse MFMA 32x32x32 f16 (V_SMFMAC_F32_32x32x32_F16). Layouts verified on
// MI350 (see amdgcn_smfmac.hpp smfmac_software_32x32x32). K=32 logical (16
// compressed), 8 groups.
// ---------------------------------------------------------------------------
static constexpr int TM = 32, TN = 32, TK = 32, TC = 16; // TC = compressed K

template <Semantics S, class Out>
__global__ void
    k_smf32(const float* Acomp, const float* B, const float* C, const int* idxbuf, Out* D)
{
    using v8h  = fpsan::v8h_native;
    using v16h = fpsan::v16h_native;
    int lane   = threadIdx.x;
    v8h an{};
    for(int h = 0; h < 8; ++h)
        an[h] = (_Float16)Acomp[(lane % 32) * TC + ((lane / 32) * 8 + h)];
    v16h bn{};
    int  jcol = (lane % 16) + 16 * ((lane / 16) % 2), kgrp = (lane / 16) / 2;
    for(int e = 0; e < 16; ++e)
    {
        int k = 16 * (e / 8) + 8 * kgrp + 2 * ((e / 2) % 4) + (e % 2);
        bn[e] = (_Float16)B[k * TN + jcol];
    }
    fpsan::v16f_native cn{};
    for(int i = 0; i < TM; ++i)
        for(int j = 0; j < TN; ++j)
        {
            auto loc = fpsan::detail::output_loc_32(TM, TN, i, j, 0);
            if(loc.lane == lane)
                cn[loc.reg] = C[i * TN + j];
        }
    Value<v8h, S, kCC>                a{an};
    Value<v16h, S, kCC>               b{bn};
    Value<fpsan::v16f_native, S, kCC> c{cn};
    auto d = fpsan::amdgcn_smfmac_f32_32x32x32_f16<0, 0, S, kCC>(a, b, c, idxbuf[lane]);
    for(int i = 0; i < TM; ++i)
        for(int j = 0; j < TN; ++j)
        {
            auto loc = fpsan::detail::output_loc_32(TM, TN, i, j, 0);
            if(loc.lane == lane)
            {
                if constexpr(S == Semantics::Native)
                    D[i * TN + j] = d.get(loc.reg).to_float();
                else
                    D[i * TN + j] = d.get(loc.reg).fpsan_payload();
            }
        }
}

namespace
{
    struct Smf32Data
    {
        std::vector<float> A, B, C; // A: 32x16, B: 32x32, C: 32x32
        std::vector<int>   idxbuf; // 64
        std::vector<int>   p0, p1; // [32*8]
    };
    Smf32Data make_smf32_data()
    {
        Smf32Data d;
        d.A.resize(TM * TC);
        d.B.resize(TK * TN);
        d.C.resize(TM * TN);
        d.idxbuf.assign(64, 0);
        d.p0.resize(TM * 8);
        d.p1.resize(TM * 8);
        std::mt19937 rng = fpsan_test::make_rng();
        for(auto& x : d.A)
            x = fpsan_test::pick_int_valued<float>(rng, -3, 3);
        for(auto& x : d.B)
            x = fpsan_test::pick_int_valued<float>(rng, -2, 2);
        for(auto& x : d.C)
            x = fpsan_test::pick_int_valued<float>(rng, -4, 4);
        for(int i = 0; i < TM; ++i)
            for(int q = 0; q < 8; ++q)
            {
                int a0 = fpsan_test::pick_int(rng, 0, 3), a1 = fpsan_test::pick_int(rng, 0, 3);
                while(a1 == a0)
                    a1 = fpsan_test::pick_int(rng, 0, 3);
                if(a0 > a1)
                {
                    int t = a0;
                    a0    = a1;
                    a1    = t;
                }
                d.p0[i * 8 + q] = a0;
                d.p1[i * 8 + q] = a1;
            }
        for(int L = 0; L < 64; ++L)
        {
            int i = L % 32, baseq = (L / 32) * 4, v = 0;
            for(int n = 0; n < 4; ++n)
            {
                int q = baseq + n;
                v |= (d.p0[i * 8 + q] | (d.p1[i * 8 + q] << 2)) << (4 * n);
            }
            d.idxbuf[L] = v;
        }
        return d;
    }
} // namespace

TEST(SmfmacF16_32x32x32, LayoutMatchesHardware)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    Smf32Data          m = make_smf32_data();
    std::vector<float> ref(TM * TN);
    for(int i = 0; i < TM; ++i)
        for(int j = 0; j < TN; ++j)
        {
            double acc = m.C[i * TN + j];
            for(int q = 0; q < 8; ++q)
            {
                acc += (double)m.A[i * TC + 2 * q]
                       * (double)m.B[(4 * q + m.p0[i * 8 + q]) * TN + j];
                acc += (double)m.A[i * TC + 2 * q + 1]
                       * (double)m.B[(4 * q + m.p1[i * 8 + q]) * TN + j];
            }
            ref[i * TN + j] = (float)acc;
        }
    float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C), *dD;
    int*   dI = to_dev(m.idxbuf);
    HIP_CHECK(hipMalloc(&dD, TM * TN * sizeof(float)));
    k_smf32<Semantics::Native, float><<<1, WAVE>>>(dA, dB, dC, dI, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> got(TM * TN);
    HIP_CHECK(hipMemcpy(got.data(), dD, TM * TN * sizeof(float), hipMemcpyDeviceToHost));
    for(int t = 0; t < TM * TN; ++t)
        EXPECT_EQ(bits_of(got[t]), bits_of(ref[t])) << "at " << t;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
    (void)hipFree(dI);
}

TEST(SmfmacF16_32x32x32, FpsanMatchesScalarReference)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    Smf32Data m = make_smf32_data();
    using VF    = Value<float, Semantics::Triton, kCC>;
    using VH    = Value<_Float16, Semantics::Triton, kCC>;
    std::vector<std::uint32_t> ref(TM * TN);
    for(int i = 0; i < TM; ++i)
        for(int j = 0; j < TN; ++j)
        {
            VF acc(m.C[i * TN + j]);
            for(int q = 0; q < 8; ++q)
            {
                acc = acc
                      + fpsan::cast<float>(VH((_Float16)m.A[i * TC + 2 * q]))
                            * fpsan::cast<float>(
                                VH((_Float16)m.B[(4 * q + m.p0[i * 8 + q]) * TN + j]));
                acc = acc
                      + fpsan::cast<float>(VH((_Float16)m.A[i * TC + 2 * q + 1]))
                            * fpsan::cast<float>(
                                VH((_Float16)m.B[(4 * q + m.p1[i * 8 + q]) * TN + j]));
            }
            ref[i * TN + j] = acc.fpsan_payload();
        }
    float *        dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C);
    int*           dI = to_dev(m.idxbuf);
    std::uint32_t* dD;
    HIP_CHECK(hipMalloc(&dD, TM * TN * sizeof(std::uint32_t)));
    k_smf32<Semantics::Triton, std::uint32_t><<<1, WAVE>>>(dA, dB, dC, dI, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(TM * TN);
    HIP_CHECK(hipMemcpy(got.data(), dD, TM * TN * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for(int t = 0; t < TM * TN; ++t)
        EXPECT_EQ(got[t], ref[t]) << "at " << t;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
    (void)hipFree(dI);
}

// ---------------------------------------------------------------------------
// bf16 SMFMAC (16x16x64 / 32x32x32). Same reverse-engineered dataflow as the
// f16 case (smfmac_software_* is element-type-agnostic) -- these instances pin
// that the bf16 wrappers, which share that dataflow, are also correct. Inputs
// are exact small integers, so __bf16 casts are lossless and the host double
// reference is exact (identical reference math to the f16 tests).
// ---------------------------------------------------------------------------
template <Semantics S, class Out>
__global__ void
    k_smf64_bf16(const float* Acomp, const float* B, const float* C, const int* idxbuf, Out* D)
{
    using v8bf  = fpsan::v8bf_native;
    using v16bf = fpsan::v16bf_native;
    int  lane = threadIdx.x, g = lane / 16, nlane = lane % 16;
    v8bf an{};
    for(int h = 0; h < 8; ++h)
        an[h] = (__bf16)Acomp[nlane * QC + (g * 8 + h)];
    v16bf bn{};
    for(int vr = 0; vr < 8; ++vr)
        for(int h = 0; h < 2; ++h)
        {
            int k          = (vr < 4) ? (8 * g + 2 * vr + h) : (32 + 8 * g + 2 * (vr - 4) + h);
            bn[2 * vr + h] = (__bf16)B[k * QN + nlane];
        }
    fpsan::v4f_native cn{};
    for(int reg = 0; reg < 4; ++reg)
        cn[reg] = C[(4 * g + reg) * QN + nlane];
    Value<v8bf, S, kCC>              a{an};
    Value<v16bf, S, kCC>             b{bn};
    Value<fpsan::v4f_native, S, kCC> c{cn};
    auto d = fpsan::amdgcn_smfmac_f32_16x16x64_bf16<0, 0, S, kCC>(a, b, c, idxbuf[lane]);
    for(int reg = 0; reg < 4; ++reg)
    {
        int i = 4 * g + reg, j = nlane;
        if constexpr(S == Semantics::Native)
            D[i * QN + j] = d.get(reg).to_float();
        else
            D[i * QN + j] = d.get(reg).fpsan_payload();
    }
}

TEST(SmfmacBf16_16x16x64, LayoutMatchesHardware)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    SmfData            m = make_smf_data();
    std::vector<float> ref(QM * QN);
    for(int i = 0; i < QM; ++i)
        for(int j = 0; j < QN; ++j)
        {
            double acc = m.C[i * QN + j];
            for(int q = 0; q < 16; ++q)
            {
                acc += (double)m.A[i * QC + 2 * q]
                       * (double)m.B[(4 * q + m.p0[i * 16 + q]) * QN + j];
                acc += (double)m.A[i * QC + 2 * q + 1]
                       * (double)m.B[(4 * q + m.p1[i * 16 + q]) * QN + j];
            }
            ref[i * QN + j] = (float)acc;
        }
    float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C), *dD;
    int*   dI = to_dev(m.idxbuf);
    HIP_CHECK(hipMalloc(&dD, QM * QN * sizeof(float)));
    k_smf64_bf16<Semantics::Native, float><<<1, WAVE>>>(dA, dB, dC, dI, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> got(QM * QN);
    HIP_CHECK(hipMemcpy(got.data(), dD, QM * QN * sizeof(float), hipMemcpyDeviceToHost));
    for(int t = 0; t < QM * QN; ++t)
        EXPECT_EQ(bits_of(got[t]), bits_of(ref[t])) << "at " << t;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
    (void)hipFree(dI);
}

TEST(SmfmacBf16_16x16x64, FpsanMatchesScalarReference)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    SmfData m = make_smf_data();
    using VF  = Value<float, Semantics::Triton, kCC>;
    using VB  = Value<__bf16, Semantics::Triton, kCC>;
    std::vector<std::uint32_t> ref(QM * QN);
    for(int i = 0; i < QM; ++i)
        for(int j = 0; j < QN; ++j)
        {
            VF acc(m.C[i * QN + j]);
            for(int q = 0; q < 16; ++q)
            {
                acc = acc
                      + fpsan::cast<float>(VB((__bf16)m.A[i * QC + 2 * q]))
                            * fpsan::cast<float>(
                                VB((__bf16)m.B[(4 * q + m.p0[i * 16 + q]) * QN + j]));
                acc = acc
                      + fpsan::cast<float>(VB((__bf16)m.A[i * QC + 2 * q + 1]))
                            * fpsan::cast<float>(
                                VB((__bf16)m.B[(4 * q + m.p1[i * 16 + q]) * QN + j]));
            }
            ref[i * QN + j] = acc.fpsan_payload();
        }
    float *        dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C);
    int*           dI = to_dev(m.idxbuf);
    std::uint32_t* dD;
    HIP_CHECK(hipMalloc(&dD, QM * QN * sizeof(std::uint32_t)));
    k_smf64_bf16<Semantics::Triton, std::uint32_t><<<1, WAVE>>>(dA, dB, dC, dI, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(QM * QN);
    HIP_CHECK(hipMemcpy(got.data(), dD, QM * QN * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for(int t = 0; t < QM * QN; ++t)
        EXPECT_EQ(got[t], ref[t]) << "at " << t;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
    (void)hipFree(dI);
}

template <Semantics S, class Out>
__global__ void
    k_smf32_bf16(const float* Acomp, const float* B, const float* C, const int* idxbuf, Out* D)
{
    using v8bf  = fpsan::v8bf_native;
    using v16bf = fpsan::v16bf_native;
    int  lane   = threadIdx.x;
    v8bf an{};
    for(int h = 0; h < 8; ++h)
        an[h] = (__bf16)Acomp[(lane % 32) * TC + ((lane / 32) * 8 + h)];
    v16bf bn{};
    int   jcol = (lane % 16) + 16 * ((lane / 16) % 2), kgrp = (lane / 16) / 2;
    for(int e = 0; e < 16; ++e)
    {
        int k = 16 * (e / 8) + 8 * kgrp + 2 * ((e / 2) % 4) + (e % 2);
        bn[e] = (__bf16)B[k * TN + jcol];
    }
    fpsan::v16f_native cn{};
    for(int i = 0; i < TM; ++i)
        for(int j = 0; j < TN; ++j)
        {
            auto loc = fpsan::detail::output_loc_32(TM, TN, i, j, 0);
            if(loc.lane == lane)
                cn[loc.reg] = C[i * TN + j];
        }
    Value<v8bf, S, kCC>               a{an};
    Value<v16bf, S, kCC>              b{bn};
    Value<fpsan::v16f_native, S, kCC> c{cn};
    auto d = fpsan::amdgcn_smfmac_f32_32x32x32_bf16<0, 0, S, kCC>(a, b, c, idxbuf[lane]);
    for(int i = 0; i < TM; ++i)
        for(int j = 0; j < TN; ++j)
        {
            auto loc = fpsan::detail::output_loc_32(TM, TN, i, j, 0);
            if(loc.lane == lane)
            {
                if constexpr(S == Semantics::Native)
                    D[i * TN + j] = d.get(loc.reg).to_float();
                else
                    D[i * TN + j] = d.get(loc.reg).fpsan_payload();
            }
        }
}

TEST(SmfmacBf16_32x32x32, LayoutMatchesHardware)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    Smf32Data          m = make_smf32_data();
    std::vector<float> ref(TM * TN);
    for(int i = 0; i < TM; ++i)
        for(int j = 0; j < TN; ++j)
        {
            double acc = m.C[i * TN + j];
            for(int q = 0; q < 8; ++q)
            {
                acc += (double)m.A[i * TC + 2 * q]
                       * (double)m.B[(4 * q + m.p0[i * 8 + q]) * TN + j];
                acc += (double)m.A[i * TC + 2 * q + 1]
                       * (double)m.B[(4 * q + m.p1[i * 8 + q]) * TN + j];
            }
            ref[i * TN + j] = (float)acc;
        }
    float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C), *dD;
    int*   dI = to_dev(m.idxbuf);
    HIP_CHECK(hipMalloc(&dD, TM * TN * sizeof(float)));
    k_smf32_bf16<Semantics::Native, float><<<1, WAVE>>>(dA, dB, dC, dI, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> got(TM * TN);
    HIP_CHECK(hipMemcpy(got.data(), dD, TM * TN * sizeof(float), hipMemcpyDeviceToHost));
    for(int t = 0; t < TM * TN; ++t)
        EXPECT_EQ(bits_of(got[t]), bits_of(ref[t])) << "at " << t;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
    (void)hipFree(dI);
}

TEST(SmfmacBf16_32x32x32, FpsanMatchesScalarReference)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    Smf32Data m = make_smf32_data();
    using VF    = Value<float, Semantics::Triton, kCC>;
    using VB    = Value<__bf16, Semantics::Triton, kCC>;
    std::vector<std::uint32_t> ref(TM * TN);
    for(int i = 0; i < TM; ++i)
        for(int j = 0; j < TN; ++j)
        {
            VF acc(m.C[i * TN + j]);
            for(int q = 0; q < 8; ++q)
            {
                acc = acc
                      + fpsan::cast<float>(VB((__bf16)m.A[i * TC + 2 * q]))
                            * fpsan::cast<float>(
                                VB((__bf16)m.B[(4 * q + m.p0[i * 8 + q]) * TN + j]));
                acc = acc
                      + fpsan::cast<float>(VB((__bf16)m.A[i * TC + 2 * q + 1]))
                            * fpsan::cast<float>(
                                VB((__bf16)m.B[(4 * q + m.p1[i * 8 + q]) * TN + j]));
            }
            ref[i * TN + j] = acc.fpsan_payload();
        }
    float *        dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C);
    int*           dI = to_dev(m.idxbuf);
    std::uint32_t* dD;
    HIP_CHECK(hipMalloc(&dD, TM * TN * sizeof(std::uint32_t)));
    k_smf32_bf16<Semantics::Triton, std::uint32_t><<<1, WAVE>>>(dA, dB, dC, dI, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(TM * TN);
    HIP_CHECK(hipMemcpy(got.data(), dD, TM * TN * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for(int t = 0; t < TM * TN; ++t)
        EXPECT_EQ(got[t], ref[t]) << "at " << t;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
    (void)hipFree(dI);
}

// ===========================================================================
// CDNA3 mai-inst SMFMAC shapes (also present on gfx950): 16x16x32 and 32x32x16,
// f16/bf16. Layouts reverse-engineered + silicon-verified (single-hot probe +
// full-random) -- see amdgcn_smfmac.hpp smfmac_software_16x16x32 / _32x32x16.
// Both shapes: A v4 half, B v8 half, index packs 2 groups/lane.
// ===========================================================================
static constexpr int HM = 16, HN = 16, HK = 32, HC = 16; // HC = compressed K
static constexpr int UM = 32, UN = 32, UK = 16, UC = 8; // 32x32x16

template <class E, Semantics S, class Out>
__global__ void k_smf1632(const float* A, const float* B, const float* C, const int* idx, Out* D)
{
    using v4e = E __attribute__((ext_vector_type(4)));
    using v8e = E __attribute__((ext_vector_type(8)));
    int lane = threadIdx.x, g = lane / 16, nl = lane % 16;
    v4e an{};
    for(int h = 0; h < 4; ++h)
        an[h] = (E)A[nl * HC + (g * 4 + h)];
    v8e bn{};
    for(int e = 0; e < 8; ++e)
        bn[e] = (E)B[(g * 8 + e) * HN + nl];
    fpsan::v4f_native cn{};
    for(int reg = 0; reg < 4; ++reg)
        cn[reg] = C[(4 * g + reg) * HN + nl];
    Value<v4e, S, kCC>               a{an};
    Value<v8e, S, kCC>               b{bn};
    Value<fpsan::v4f_native, S, kCC> c{cn};
    auto                             d = [&] {
        if constexpr(std::is_same_v<E, _Float16>)
            return fpsan::amdgcn_smfmac_f32_16x16x32_f16<0, 0, S, kCC>(a, b, c, idx[lane]);
        else
            return fpsan::amdgcn_smfmac_f32_16x16x32_bf16<0, 0, S, kCC>(a, b, c, idx[lane]);
    }();
    for(int reg = 0; reg < 4; ++reg)
    {
        int i = 4 * g + reg, j = nl;
        if constexpr(S == Semantics::Native)
            D[i * HN + j] = d.get(reg).to_float();
        else
            D[i * HN + j] = d.get(reg).fpsan_payload();
    }
}

template <class E, Semantics S, class Out>
__global__ void k_smf3216(const float* A, const float* B, const float* C, const int* idx, Out* D)
{
    using v4e = E __attribute__((ext_vector_type(4)));
    using v8e = E __attribute__((ext_vector_type(8)));
    int lane  = threadIdx.x;
    v4e an{};
    for(int h = 0; h < 4; ++h)
        an[h] = (E)A[(lane % 32) * UC + ((lane / 32) * 4 + h)];
    v8e bn{};
    int jcol = (lane % 16) + 16 * ((lane / 16) % 2), kgrp = (lane / 16) / 2;
    for(int e = 0; e < 8; ++e)
        bn[e] = (E)B[(8 * kgrp + e) * UN + jcol];
    fpsan::v16f_native cn{};
    for(int i = 0; i < UM; ++i)
        for(int j = 0; j < UN; ++j)
        {
            auto loc = fpsan::detail::output_loc_32(UM, UN, i, j, 0);
            if(loc.lane == lane)
                cn[loc.reg] = C[i * UN + j];
        }
    Value<v4e, S, kCC>                a{an};
    Value<v8e, S, kCC>                b{bn};
    Value<fpsan::v16f_native, S, kCC> c{cn};
    auto                              d = [&] {
        if constexpr(std::is_same_v<E, _Float16>)
            return fpsan::amdgcn_smfmac_f32_32x32x16_f16<0, 0, S, kCC>(a, b, c, idx[lane]);
        else
            return fpsan::amdgcn_smfmac_f32_32x32x16_bf16<0, 0, S, kCC>(a, b, c, idx[lane]);
    }();
    for(int i = 0; i < UM; ++i)
        for(int j = 0; j < UN; ++j)
        {
            auto loc = fpsan::detail::output_loc_32(UM, UN, i, j, 0);
            if(loc.lane == lane)
            {
                if constexpr(S == Semantics::Native)
                    D[i * UN + j] = d.get(loc.reg).to_float();
                else
                    D[i * UN + j] = d.get(loc.reg).fpsan_payload();
            }
        }
}

namespace
{
    struct SmfCdna3Data
    {
        std::vector<float> A, B, C;
        std::vector<int>   idxbuf, p0, p1; // p0/p1 indexed [row*G + q]
    };
    // M/N/K with G groups (=K/4), compressed columns C=2*G, index packs 2 grp/lane.
    SmfCdna3Data make_cdna3(int Mm, int Nn, int Kk)
    {
        const int    G = Kk / 4, Cc = 2 * G;
        SmfCdna3Data d;
        d.A.resize(Mm * Cc);
        d.B.resize(Kk * Nn);
        d.C.resize(Mm * Nn);
        d.idxbuf.assign(64, 0);
        d.p0.resize(Mm * G);
        d.p1.resize(Mm * G);
        std::mt19937 rng = fpsan_test::make_rng();
        for(auto& x : d.A)
            x = fpsan_test::pick_int_valued<float>(rng, -3, 3);
        for(auto& x : d.B)
            x = fpsan_test::pick_int_valued<float>(rng, -2, 2);
        for(auto& x : d.C)
            x = fpsan_test::pick_int_valued<float>(rng, -4, 4);
        for(int i = 0; i < Mm; ++i)
            for(int q = 0; q < G; ++q)
            {
                int a0 = fpsan_test::pick_int(rng, 0, 3), a1 = fpsan_test::pick_int(rng, 0, 3);
                while(a1 == a0)
                    a1 = fpsan_test::pick_int(rng, 0, 3);
                if(a0 > a1)
                    std::swap(a0, a1);
                d.p0[i * G + q] = a0;
                d.p1[i * G + q] = a1;
            }
        // index: lane (q/2)*M+i carries (row i, group q) at nibble q%2.
        for(int i = 0; i < Mm; ++i)
            for(int q = 0; q < G; ++q)
            {
                int L = (q / 2) * Mm + i;
                d.idxbuf[L] |= (d.p0[i * G + q] | (d.p1[i * G + q] << 2)) << (4 * (q % 2));
            }
        return d;
    }
} // namespace

template <class E>
static void cdna3_layout_test(int Mm, int Nn, int Kk)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    const int          G = Kk / 4, Cc = 2 * G;
    SmfCdna3Data       m = make_cdna3(Mm, Nn, Kk);
    std::vector<float> ref(Mm * Nn);
    for(int i = 0; i < Mm; ++i)
        for(int j = 0; j < Nn; ++j)
        {
            double acc = m.C[i * Nn + j];
            for(int q = 0; q < G; ++q)
            {
                acc += (double)m.A[i * Cc + 2 * q]
                       * (double)m.B[(4 * q + m.p0[i * G + q]) * Nn + j];
                acc += (double)m.A[i * Cc + 2 * q + 1]
                       * (double)m.B[(4 * q + m.p1[i * G + q]) * Nn + j];
            }
            ref[i * Nn + j] = (float)acc;
        }
    float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C), *dD;
    int*   dI = to_dev(m.idxbuf);
    HIP_CHECK(hipMalloc(&dD, Mm * Nn * sizeof(float)));
    if(Mm == 16)
        k_smf1632<E, Semantics::Native, float><<<1, WAVE>>>(dA, dB, dC, dI, dD);
    else
        k_smf3216<E, Semantics::Native, float><<<1, WAVE>>>(dA, dB, dC, dI, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> got(Mm * Nn);
    HIP_CHECK(hipMemcpy(got.data(), dD, Mm * Nn * sizeof(float), hipMemcpyDeviceToHost));
    for(int t = 0; t < Mm * Nn; ++t)
        EXPECT_EQ(bits_of(got[t]), bits_of(ref[t])) << "at " << t;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
    (void)hipFree(dI);
}

template <class E>
static void cdna3_fpsan_test(int Mm, int Nn, int Kk)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    const int    G = Kk / 4, Cc = 2 * G;
    SmfCdna3Data m = make_cdna3(Mm, Nn, Kk);
    using VF       = Value<float, Semantics::Triton, kCC>;
    using VE       = Value<E, Semantics::Triton, kCC>;
    std::vector<std::uint32_t> ref(Mm * Nn);
    for(int i = 0; i < Mm; ++i)
        for(int j = 0; j < Nn; ++j)
        {
            VF acc(m.C[i * Nn + j]);
            for(int q = 0; q < G; ++q)
            {
                acc = acc
                      + fpsan::cast<float>(VE((E)m.A[i * Cc + 2 * q]))
                            * fpsan::cast<float>(VE((E)m.B[(4 * q + m.p0[i * G + q]) * Nn + j]));
                acc = acc
                      + fpsan::cast<float>(VE((E)m.A[i * Cc + 2 * q + 1]))
                            * fpsan::cast<float>(VE((E)m.B[(4 * q + m.p1[i * G + q]) * Nn + j]));
            }
            ref[i * Nn + j] = acc.fpsan_payload();
        }
    float *        dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C);
    int*           dI = to_dev(m.idxbuf);
    std::uint32_t* dD;
    HIP_CHECK(hipMalloc(&dD, Mm * Nn * sizeof(std::uint32_t)));
    if(Mm == 16)
        k_smf1632<E, Semantics::Triton, std::uint32_t><<<1, WAVE>>>(dA, dB, dC, dI, dD);
    else
        k_smf3216<E, Semantics::Triton, std::uint32_t><<<1, WAVE>>>(dA, dB, dC, dI, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(Mm * Nn);
    HIP_CHECK(hipMemcpy(got.data(), dD, Mm * Nn * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for(int t = 0; t < Mm * Nn; ++t)
        EXPECT_EQ(got[t], ref[t]) << "at " << t;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
    (void)hipFree(dI);
}

TEST(SmfmacF16_16x16x32, LayoutMatchesHardware)
{
    cdna3_layout_test<_Float16>(HM, HN, HK);
}
TEST(SmfmacF16_16x16x32, FpsanMatchesScalarReference)
{
    cdna3_fpsan_test<_Float16>(HM, HN, HK);
}
TEST(SmfmacBf16_16x16x32, LayoutMatchesHardware)
{
    cdna3_layout_test<__bf16>(HM, HN, HK);
}
TEST(SmfmacBf16_16x16x32, FpsanMatchesScalarReference)
{
    cdna3_fpsan_test<__bf16>(HM, HN, HK);
}
TEST(SmfmacF16_32x32x16, LayoutMatchesHardware)
{
    cdna3_layout_test<_Float16>(UM, UN, UK);
}
TEST(SmfmacF16_32x32x16, FpsanMatchesScalarReference)
{
    cdna3_fpsan_test<_Float16>(UM, UN, UK);
}
TEST(SmfmacBf16_32x32x16, LayoutMatchesHardware)
{
    cdna3_layout_test<__bf16>(UM, UN, UK);
}
TEST(SmfmacBf16_32x32x16, FpsanMatchesScalarReference)
{
    cdna3_fpsan_test<__bf16>(UM, UN, UK);
}

// ===========================================================================
// CDNA3 fp8-insts SMFMAC shapes: 16x16x64 and 32x32x32, fp8/bf8 x fp8/bf8.
// Layouts reverse-engineered + multi-seed full-random verified on MI350 (see
// amdgcn_smfmac.hpp smfmac_software_*_fp8). A = v8 fp8, B = v16 fp8. fp8 packs
// 4 bytes/dword, so A-col map = (Cc/2)*(b/4)+4*g+(b%4) and the index packs the
// low/high K half into the nibble's high bit -- different from the f16 shapes.
// ===========================================================================
static constexpr int FW = 16, FK = 64, FCc = 32; // 16x16x64: groups=16
static constexpr int FX = 32, FXK = 32, FXC = 16; // 32x32x32: groups=8

template <class AE, class BE, Semantics S, class Out>
__global__ void
    k_smf_fp8_1664(const float* A, const float* B, const float* C, const int* idx, Out* D)
{
    using AVec = fpsan::detail::v8_fragment<AE>;
    using BVec = fpsan::detail::v16_fragment<BE>;
    int  lane = threadIdx.x, g = lane / 16, nl = lane % 16;
    AVec an{};
    for(int b = 0; b < 8; ++b)
    {
        int c = 16 * (b / 4) + 4 * g + (b % 4);
        an[b] = AE(A[nl * FCc + c]);
    }
    BVec bn{};
    for(int e = 0; e < 16; ++e)
    {
        int k = 32 * (e / 8) + 8 * g + (e % 8);
        bn[e] = BE(B[k * FW + nl]);
    }
    fpsan::v4f_native cn{};
    for(int reg = 0; reg < 4; ++reg)
        cn[reg] = C[(4 * g + reg) * FW + nl];
    Value<AVec, S, kCC>              a{an};
    Value<BVec, S, kCC>              b{bn};
    Value<fpsan::v4f_native, S, kCC> c{cn};
    auto                             d = [&] {
        if constexpr(std::is_same_v<AE, fpsan::fp8_e4m3> && std::is_same_v<BE, fpsan::fp8_e4m3>)
            return fpsan::amdgcn_smfmac_f32_16x16x64_fp8_fp8<0, 0, S, kCC>(a, b, c, idx[lane]);
        else if constexpr(std::is_same_v<AE, fpsan::fp8_e4m3>)
            return fpsan::amdgcn_smfmac_f32_16x16x64_fp8_bf8<0, 0, S, kCC>(a, b, c, idx[lane]);
        else if constexpr(std::is_same_v<BE, fpsan::fp8_e4m3>)
            return fpsan::amdgcn_smfmac_f32_16x16x64_bf8_fp8<0, 0, S, kCC>(a, b, c, idx[lane]);
        else
            return fpsan::amdgcn_smfmac_f32_16x16x64_bf8_bf8<0, 0, S, kCC>(a, b, c, idx[lane]);
    }();
    for(int reg = 0; reg < 4; ++reg)
    {
        int i = 4 * g + reg, j = nl;
        if constexpr(S == Semantics::Native)
            D[i * FW + j] = d.get(reg).to_float();
        else
            D[i * FW + j] = d.get(reg).fpsan_payload();
    }
}

template <class AE, class BE, Semantics S, class Out>
__global__ void
    k_smf_fp8_3232(const float* A, const float* B, const float* C, const int* idx, Out* D)
{
    using AVec = fpsan::detail::v8_fragment<AE>;
    using BVec = fpsan::detail::v16_fragment<BE>;
    int  lane = threadIdx.x, g = lane / 32, nl = lane % 32;
    AVec an{};
    for(int b = 0; b < 8; ++b)
    {
        int c = 8 * (b / 4) + 4 * g + (b % 4);
        an[b] = AE(A[nl * FXC + c]);
    }
    BVec bn{};
    int  jcol = (lane % 16) + 16 * ((lane / 16) % 2), kgrp = (lane / 16) / 2;
    for(int e = 0; e < 16; ++e)
    {
        int k = 16 * (e / 8) + 8 * kgrp + 2 * ((e / 2) % 4) + (e % 2);
        bn[e] = BE(B[k * FX + jcol]);
    }
    fpsan::v16f_native cn{};
    for(int i = 0; i < FX; ++i)
        for(int j = 0; j < FX; ++j)
        {
            auto loc = fpsan::detail::output_loc_32(FX, FX, i, j, 0);
            if(loc.lane == lane)
                cn[loc.reg] = C[i * FX + j];
        }
    Value<AVec, S, kCC>               a{an};
    Value<BVec, S, kCC>               b{bn};
    Value<fpsan::v16f_native, S, kCC> c{cn};
    auto                              d = [&] {
        if constexpr(std::is_same_v<AE, fpsan::fp8_e4m3> && std::is_same_v<BE, fpsan::fp8_e4m3>)
            return fpsan::amdgcn_smfmac_f32_32x32x32_fp8_fp8<0, 0, S, kCC>(a, b, c, idx[lane]);
        else if constexpr(std::is_same_v<AE, fpsan::fp8_e4m3>)
            return fpsan::amdgcn_smfmac_f32_32x32x32_fp8_bf8<0, 0, S, kCC>(a, b, c, idx[lane]);
        else if constexpr(std::is_same_v<BE, fpsan::fp8_e4m3>)
            return fpsan::amdgcn_smfmac_f32_32x32x32_bf8_fp8<0, 0, S, kCC>(a, b, c, idx[lane]);
        else
            return fpsan::amdgcn_smfmac_f32_32x32x32_bf8_bf8<0, 0, S, kCC>(a, b, c, idx[lane]);
    }();
    for(int i = 0; i < FX; ++i)
        for(int j = 0; j < FX; ++j)
        {
            auto loc = fpsan::detail::output_loc_32(FX, FX, i, j, 0);
            if(loc.lane == lane)
            {
                if constexpr(S == Semantics::Native)
                    D[i * FX + j] = d.get(loc.reg).to_float();
                else
                    D[i * FX + j] = d.get(loc.reg).fpsan_payload();
            }
        }
}

namespace
{
    // Shared data gen for the fp8 SMFMAC shapes (Mm=Nn, K, groups G=K/4).
    SmfCdna3Data make_fp8(int Mm, int Kk)
    {
        const int    G = Kk / 4, Cc = 2 * G, Nn = Mm;
        SmfCdna3Data d;
        d.A.resize(Mm * Cc);
        d.B.resize(Kk * Nn);
        d.C.resize(Mm * Nn);
        d.idxbuf.assign(64, 0);
        d.p0.resize(Mm * G);
        d.p1.resize(Mm * G);
        std::mt19937 rng = fpsan_test::make_rng();
        for(auto& x : d.A)
            x = fpsan_test::pick_int(rng, -3, 3);
        for(auto& x : d.B)
            x = fpsan_test::pick_int(rng, -2, 2);
        for(auto& x : d.C)
            x = fpsan_test::pick_int(rng, -3, 3);
        for(int i = 0; i < Mm; ++i)
            for(int q = 0; q < G; ++q)
            {
                int a0 = fpsan_test::pick_int(rng, 0, 3), a1 = fpsan_test::pick_int(rng, 0, 3);
                while(a1 == a0)
                    a1 = fpsan_test::pick_int(rng, 0, 3);
                if(a0 > a1)
                    std::swap(a0, a1);
                d.p0[i * G + q] = a0;
                d.p1[i * G + q] = a1;
            }
        const int half = G / 2;
        for(int i = 0; i < Mm; ++i)
            for(int q = 0; q < G; ++q)
            {
                int L = ((q % half) / 2) * Mm + i, nb = 2 * (q / half) + (q % 2);
                d.idxbuf[L] |= (d.p0[i * G + q] | (d.p1[i * G + q] << 2)) << (4 * nb);
            }
        return d;
    }
} // namespace

template <class AE, class BE>
static void fp8_smf_test(int Mm, int Kk)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    const int    G = Kk / 4, Cc = 2 * G, Nn = Mm;
    SmfCdna3Data m = make_fp8(Mm, Kk);
    // Float-mode layout reference (host integer matmul, exact small ints).
    std::vector<float> ref(Mm * Nn);
    // FPSan reference (payload ring).
    using VF = Value<float, Semantics::Triton, kCC>;
    using VA = Value<AE, Semantics::Triton, kCC>;
    using VB = Value<BE, Semantics::Triton, kCC>;
    std::vector<std::uint32_t> refp(Mm * Nn);
    for(int i = 0; i < Mm; ++i)
        for(int j = 0; j < Nn; ++j)
        {
            double acc = m.C[i * Nn + j];
            VF     accp(m.C[i * Nn + j]);
            for(int q = 0; q < G; ++q)
            {
                int k0 = 4 * q + m.p0[i * G + q], k1 = 4 * q + m.p1[i * G + q];
                acc += (double)(float)AE(m.A[i * Cc + 2 * q]) * (double)(float)BE(m.B[k0 * Nn + j]);
                acc += (double)(float)AE(m.A[i * Cc + 2 * q + 1])
                       * (double)(float)BE(m.B[k1 * Nn + j]);
                accp = accp
                       + fpsan::cast<float>(VA(AE(m.A[i * Cc + 2 * q])))
                             * fpsan::cast<float>(VB(BE(m.B[k0 * Nn + j])));
                accp = accp
                       + fpsan::cast<float>(VA(AE(m.A[i * Cc + 2 * q + 1])))
                             * fpsan::cast<float>(VB(BE(m.B[k1 * Nn + j])));
            }
            ref[i * Nn + j]  = (float)acc;
            refp[i * Nn + j] = accp.fpsan_payload();
        }
    float *        dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C), *dDf;
    int*           dI = to_dev(m.idxbuf);
    std::uint32_t* dDp;
    HIP_CHECK(hipMalloc(&dDf, Mm * Nn * sizeof(float)));
    HIP_CHECK(hipMalloc(&dDp, Mm * Nn * sizeof(std::uint32_t)));
    if(Mm == 16)
    {
        k_smf_fp8_1664<AE, BE, Semantics::Native, float><<<1, WAVE>>>(dA, dB, dC, dI, dDf);
        HIP_CHECK(hipDeviceSynchronize());
        k_smf_fp8_1664<AE, BE, Semantics::Triton, std::uint32_t><<<1, WAVE>>>(dA, dB, dC, dI, dDp);
    }
    else
    {
        k_smf_fp8_3232<AE, BE, Semantics::Native, float><<<1, WAVE>>>(dA, dB, dC, dI, dDf);
        HIP_CHECK(hipDeviceSynchronize());
        k_smf_fp8_3232<AE, BE, Semantics::Triton, std::uint32_t><<<1, WAVE>>>(dA, dB, dC, dI, dDp);
    }
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float>         gotf(Mm * Nn);
    std::vector<std::uint32_t> gotp(Mm * Nn);
    HIP_CHECK(hipMemcpy(gotf.data(), dDf, Mm * Nn * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(gotp.data(), dDp, Mm * Nn * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for(int t = 0; t < Mm * Nn; ++t)
    {
        EXPECT_EQ(bits_of(gotf[t]), bits_of(ref[t])) << "float at " << t;
        EXPECT_EQ(gotp[t], refp[t]) << "fpsan at " << t;
    }
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dDf);
    (void)hipFree(dDp);
    (void)hipFree(dI);
}

TEST(SmfmacFp8_16x16x64, FP8_FP8)
{
    fp8_smf_test<fpsan::fp8_e4m3, fpsan::fp8_e4m3>(FW, FK);
}
TEST(SmfmacFp8_16x16x64, FP8_BF8)
{
    fp8_smf_test<fpsan::fp8_e4m3, fpsan::fp8_e5m2>(FW, FK);
}
TEST(SmfmacFp8_16x16x64, BF8_FP8)
{
    fp8_smf_test<fpsan::fp8_e5m2, fpsan::fp8_e4m3>(FW, FK);
}
TEST(SmfmacFp8_16x16x64, BF8_BF8)
{
    fp8_smf_test<fpsan::fp8_e5m2, fpsan::fp8_e5m2>(FW, FK);
}
TEST(SmfmacFp8_32x32x32, FP8_FP8)
{
    fp8_smf_test<fpsan::fp8_e4m3, fpsan::fp8_e4m3>(FX, FXK);
}
TEST(SmfmacFp8_32x32x32, FP8_BF8)
{
    fp8_smf_test<fpsan::fp8_e4m3, fpsan::fp8_e5m2>(FX, FXK);
}
TEST(SmfmacFp8_32x32x32, BF8_FP8)
{
    fp8_smf_test<fpsan::fp8_e5m2, fpsan::fp8_e4m3>(FX, FXK);
}
TEST(SmfmacFp8_32x32x32, BF8_BF8)
{
    fp8_smf_test<fpsan::fp8_e5m2, fpsan::fp8_e5m2>(FX, FXK);
}

// ===========================================================================
// gfx950 fp8-insts SMFMAC shapes: 16x16x128 and 32x32x64, fp8/bf8 x fp8/bf8.
// Layouts reverse-engineered + multi-seed full-random verified on MI350 (see
// amdgcn_smfmac.hpp smfmac_software_16x16x128_fp8 / _32x32x64_fp8). A = v16 fp8,
// B = v32 fp8. These double the K of the CDNA3 fp8 shapes above.
// ===========================================================================
static constexpr int GW = 16, GK = 128, GCc = 64; // 16x16x128: groups=32
static constexpr int GX = 32, GXK = 64, GXC = 32; // 32x32x64:  groups=16

template <class AE, class BE, Semantics S, class Out>
__global__ void
    k_smf_fp8big_16128(const float* A, const float* B, const float* C, const int* idx, Out* D)
{
    using AVec = fpsan::detail::v16_fragment<AE>;
    using BVec = fpsan::detail::v32_fragment<BE>;
    int  lane = threadIdx.x, g = lane / 16, nl = lane % 16;
    AVec an{};
    for(int b = 0; b < 16; ++b)
    {
        int hb = b / 4, lb = b % 4;
        int c = lb + 16 * (hb % 2) + 4 * (hb / 2) + 32 * (g / 2) + 8 * (g % 2);
        an[b] = AE(A[nl * GCc + c]);
    }
    BVec bn{};
    for(int e = 0; e < 32; ++e)
    {
        int k = 32 * (e / 8) + 8 * g + (e % 8);
        bn[e] = BE(B[k * GW + nl]);
    }
    fpsan::v4f_native cn{};
    for(int reg = 0; reg < 4; ++reg)
        cn[reg] = C[(4 * g + reg) * GW + nl];
    Value<AVec, S, kCC>              a{an};
    Value<BVec, S, kCC>              b{bn};
    Value<fpsan::v4f_native, S, kCC> c{cn};
    auto                             d = [&] {
        if constexpr(std::is_same_v<AE, fpsan::fp8_e4m3> && std::is_same_v<BE, fpsan::fp8_e4m3>)
            return fpsan::amdgcn_smfmac_f32_16x16x128_fp8_fp8<0, 0, S, kCC>(a, b, c, idx[lane]);
        else if constexpr(std::is_same_v<AE, fpsan::fp8_e4m3>)
            return fpsan::amdgcn_smfmac_f32_16x16x128_fp8_bf8<0, 0, S, kCC>(a, b, c, idx[lane]);
        else if constexpr(std::is_same_v<BE, fpsan::fp8_e4m3>)
            return fpsan::amdgcn_smfmac_f32_16x16x128_bf8_fp8<0, 0, S, kCC>(a, b, c, idx[lane]);
        else
            return fpsan::amdgcn_smfmac_f32_16x16x128_bf8_bf8<0, 0, S, kCC>(a, b, c, idx[lane]);
    }();
    for(int reg = 0; reg < 4; ++reg)
    {
        int i = 4 * g + reg, j = nl;
        if constexpr(S == Semantics::Native)
            D[i * GW + j] = d.get(reg).to_float();
        else
            D[i * GW + j] = d.get(reg).fpsan_payload();
    }
}

template <class AE, class BE, Semantics S, class Out>
__global__ void
    k_smf_fp8big_3264(const float* A, const float* B, const float* C, const int* idx, Out* D)
{
    using AVec = fpsan::detail::v16_fragment<AE>;
    using BVec = fpsan::detail::v32_fragment<BE>;
    int  lane = threadIdx.x, g = lane / 32, nl = lane % 32, kgrp = lane / 32;
    AVec an{};
    for(int b = 0; b < 16; ++b)
    {
        int hb = b / 4, lb = b % 4;
        int c = lb + 8 * (hb % 2) + 4 * (hb / 2) + 16 * g;
        an[b] = AE(A[nl * GXC + c]);
    }
    BVec bn{};
    for(int e = 0; e < 32; ++e)
    {
        int k = 16 * (e / 8) + 8 * kgrp + (e % 8);
        bn[e] = BE(B[k * GX + nl]);
    }
    fpsan::v16f_native cn{};
    for(int i = 0; i < GX; ++i)
        for(int j = 0; j < GX; ++j)
        {
            auto loc = fpsan::detail::output_loc_32(GX, GX, i, j, 0);
            if(loc.lane == lane)
                cn[loc.reg] = C[i * GX + j];
        }
    Value<AVec, S, kCC>               a{an};
    Value<BVec, S, kCC>               b{bn};
    Value<fpsan::v16f_native, S, kCC> c{cn};
    auto                              d = [&] {
        if constexpr(std::is_same_v<AE, fpsan::fp8_e4m3> && std::is_same_v<BE, fpsan::fp8_e4m3>)
            return fpsan::amdgcn_smfmac_f32_32x32x64_fp8_fp8<0, 0, S, kCC>(a, b, c, idx[lane]);
        else if constexpr(std::is_same_v<AE, fpsan::fp8_e4m3>)
            return fpsan::amdgcn_smfmac_f32_32x32x64_fp8_bf8<0, 0, S, kCC>(a, b, c, idx[lane]);
        else if constexpr(std::is_same_v<BE, fpsan::fp8_e4m3>)
            return fpsan::amdgcn_smfmac_f32_32x32x64_bf8_fp8<0, 0, S, kCC>(a, b, c, idx[lane]);
        else
            return fpsan::amdgcn_smfmac_f32_32x32x64_bf8_bf8<0, 0, S, kCC>(a, b, c, idx[lane]);
    }();
    for(int i = 0; i < GX; ++i)
        for(int j = 0; j < GX; ++j)
        {
            auto loc = fpsan::detail::output_loc_32(GX, GX, i, j, 0);
            if(loc.lane == lane)
            {
                if constexpr(S == Semantics::Native)
                    D[i * GX + j] = d.get(loc.reg).to_float();
                else
                    D[i * GX + j] = d.get(loc.reg).fpsan_payload();
            }
        }
}

namespace
{
    // Data + index gen for the gfx950 fp8 SMFMAC shapes (Mm in {16,32}).
    SmfCdna3Data make_fp8big(int Mm, int Kk)
    {
        const int    G = Kk / 4, Cc = 2 * G, Nn = Mm;
        SmfCdna3Data d;
        d.A.resize(Mm * Cc);
        d.B.resize(Kk * Nn);
        d.C.resize(Mm * Nn);
        d.idxbuf.assign(64, 0);
        d.p0.resize(Mm * G);
        d.p1.resize(Mm * G);
        std::mt19937 rng = fpsan_test::make_rng();
        for(auto& x : d.A)
            x = fpsan_test::pick_int(rng, -3, 3);
        for(auto& x : d.B)
            x = fpsan_test::pick_int(rng, -2, 2);
        for(auto& x : d.C)
            x = fpsan_test::pick_int(rng, -3, 3);
        for(int i = 0; i < Mm; ++i)
            for(int q = 0; q < G; ++q)
            {
                int a0 = fpsan_test::pick_int(rng, 0, 3), a1 = fpsan_test::pick_int(rng, 0, 3);
                while(a1 == a0)
                    a1 = fpsan_test::pick_int(rng, 0, 3);
                d.p0[i * G + q] = a0;
                d.p1[i * G + q] = a1;
            }
        for(int i = 0; i < Mm; ++i)
            for(int q = 0; q < G; ++q)
            {
                int L, nb;
                if(Mm == 16)
                { // 16x16x128
                    L  = 16 * (2 * (q / 16) + ((q / 4) % 2)) + i;
                    nb = 2 * ((q / 8) % 2) + 4 * ((q % 4) / 2) + ((q % 4) % 2);
                }
                else
                { // 32x32x64
                    L  = 32 * (q / 8) + i;
                    nb = (q % 2) + 2 * ((q / 4) % 2) + 4 * ((q / 2) % 2);
                }
                d.idxbuf[L] |= (d.p0[i * G + q] | (d.p1[i * G + q] << 2)) << (4 * nb);
            }
        return d;
    }
} // namespace

template <class AE, class BE>
static void fp8big_smf_test(int Mm, int Kk)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    const int          G = Kk / 4, Cc = 2 * G, Nn = Mm;
    SmfCdna3Data       m = make_fp8big(Mm, Kk);
    std::vector<float> ref(Mm * Nn);
    using VF = Value<float, Semantics::Triton, kCC>;
    using VA = Value<AE, Semantics::Triton, kCC>;
    using VB = Value<BE, Semantics::Triton, kCC>;
    std::vector<std::uint32_t> refp(Mm * Nn);
    for(int i = 0; i < Mm; ++i)
        for(int j = 0; j < Nn; ++j)
        {
            double acc = m.C[i * Nn + j];
            VF     accp(m.C[i * Nn + j]);
            for(int q = 0; q < G; ++q)
            {
                int k0 = 4 * q + m.p0[i * G + q], k1 = 4 * q + m.p1[i * G + q];
                acc += (double)(float)AE(m.A[i * Cc + 2 * q]) * (double)(float)BE(m.B[k0 * Nn + j]);
                acc += (double)(float)AE(m.A[i * Cc + 2 * q + 1])
                       * (double)(float)BE(m.B[k1 * Nn + j]);
                accp = accp
                       + fpsan::cast<float>(VA(AE(m.A[i * Cc + 2 * q])))
                             * fpsan::cast<float>(VB(BE(m.B[k0 * Nn + j])));
                accp = accp
                       + fpsan::cast<float>(VA(AE(m.A[i * Cc + 2 * q + 1])))
                             * fpsan::cast<float>(VB(BE(m.B[k1 * Nn + j])));
            }
            ref[i * Nn + j]  = (float)acc;
            refp[i * Nn + j] = accp.fpsan_payload();
        }
    float *        dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C), *dDf;
    int*           dI = to_dev(m.idxbuf);
    std::uint32_t* dDp;
    HIP_CHECK(hipMalloc(&dDf, Mm * Nn * sizeof(float)));
    HIP_CHECK(hipMalloc(&dDp, Mm * Nn * sizeof(std::uint32_t)));
    if(Mm == 16)
    {
        k_smf_fp8big_16128<AE, BE, Semantics::Native, float><<<1, WAVE>>>(dA, dB, dC, dI, dDf);
        HIP_CHECK(hipDeviceSynchronize());
        k_smf_fp8big_16128<AE, BE, Semantics::Triton, std::uint32_t>
            <<<1, WAVE>>>(dA, dB, dC, dI, dDp);
    }
    else
    {
        k_smf_fp8big_3264<AE, BE, Semantics::Native, float><<<1, WAVE>>>(dA, dB, dC, dI, dDf);
        HIP_CHECK(hipDeviceSynchronize());
        k_smf_fp8big_3264<AE, BE, Semantics::Triton, std::uint32_t>
            <<<1, WAVE>>>(dA, dB, dC, dI, dDp);
    }
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float>         gotf(Mm * Nn);
    std::vector<std::uint32_t> gotp(Mm * Nn);
    HIP_CHECK(hipMemcpy(gotf.data(), dDf, Mm * Nn * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(gotp.data(), dDp, Mm * Nn * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for(int t = 0; t < Mm * Nn; ++t)
    {
        EXPECT_EQ(bits_of(gotf[t]), bits_of(ref[t])) << "float at " << t;
        EXPECT_EQ(gotp[t], refp[t]) << "fpsan at " << t;
    }
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dDf);
    (void)hipFree(dDp);
    (void)hipFree(dI);
}

TEST(SmfmacFp8_16x16x128, FP8_FP8)
{
    fp8big_smf_test<fpsan::fp8_e4m3, fpsan::fp8_e4m3>(GW, GK);
}
TEST(SmfmacFp8_16x16x128, FP8_BF8)
{
    fp8big_smf_test<fpsan::fp8_e4m3, fpsan::fp8_e5m2>(GW, GK);
}
TEST(SmfmacFp8_16x16x128, BF8_FP8)
{
    fp8big_smf_test<fpsan::fp8_e5m2, fpsan::fp8_e4m3>(GW, GK);
}
TEST(SmfmacFp8_16x16x128, BF8_BF8)
{
    fp8big_smf_test<fpsan::fp8_e5m2, fpsan::fp8_e5m2>(GW, GK);
}
TEST(SmfmacFp8_32x32x64, FP8_FP8)
{
    fp8big_smf_test<fpsan::fp8_e4m3, fpsan::fp8_e4m3>(GX, GXK);
}
TEST(SmfmacFp8_32x32x64, FP8_BF8)
{
    fp8big_smf_test<fpsan::fp8_e4m3, fpsan::fp8_e5m2>(GX, GXK);
}
TEST(SmfmacFp8_32x32x64, BF8_FP8)
{
    fp8big_smf_test<fpsan::fp8_e5m2, fpsan::fp8_e4m3>(GX, GXK);
}
TEST(SmfmacFp8_32x32x64, BF8_BF8)
{
    fp8big_smf_test<fpsan::fp8_e5m2, fpsan::fp8_e5m2>(GX, GXK);
}
