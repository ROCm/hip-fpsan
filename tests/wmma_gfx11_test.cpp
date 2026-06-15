// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/wmma_gfx11_test.cpp
//
// Deep tests for the gfx11 (RDNA3) 16x16x16 WMMA FPSan wrappers. gfx11 shares
// wrapper names with gfx12 for the 16-bit floating variants, but uses 256-bit
// replicated A/B operands and a different C/D row mapping. Keep this test
// separate from wmma_gfx12_test.cpp so the two layouts cannot accidentally
// converge.
// The source is built twice for gfx11: default wave32, and with
// FPSAN_TEST_GFX11_WAVE64 plus -mwavefrontsize64 for the wave64-only builtins.
//
// The two tested properties mirror the gfx12 WMMA harness:
//   (1) Semantics::Native software dataflow matches the real hardware builtin
//       within the small observed gfx11 WMMA numeric tolerance.
//   (2) Semantics::Triton matches an independent scalar host FPSan reference
//       exactly, payload-for-payload.

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
using fpsan::v16bf_wmma_native;
using fpsan::v16h_wmma_native;
using fpsan::v4f_native;
using fpsan::v8bf_native;
using fpsan::v8f_native;
using fpsan::v8h_native;
using fpsan::Value;

static constexpr int         M = 16, N = 16, K = 16;
static constexpr Conversions kCC = Conversions::Explicit;

#if defined(FPSAN_TEST_GFX11_WAVE64)
static constexpr int kWaveSize = 64;
#else
static constexpr int kWaveSize = 32;
#endif

namespace
{
    bool current_device_is_gfx11()
    {
        int ndev = 0;
        if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
            return false;
        int dev = 0;
        if(hipGetDevice(&dev) != hipSuccess)
            return false;
        hipDeviceProp_t prop{};
        if(hipGetDeviceProperties(&prop, dev) != hipSuccess)
            return false;
        return std::strncmp(prop.gcnArchName, "gfx11", 5) == 0;
    }
} // namespace

// ---------------------------------------------------------------------------
// Templated harness shared by every gfx11 16x16x16 floating WMMA variant.
// ---------------------------------------------------------------------------
template <class Traits>
struct Harness
{
    using AVec                       = typename Traits::AVec;
    using BVec                       = typename Traits::BVec;
    using CVec                       = typename Traits::CVec;
    using AElem                      = fpsan::detail::vector_element_t<AVec>;
    using BElem                      = fpsan::detail::vector_element_t<BVec>;
    using CElem                      = fpsan::detail::vector_element_t<CVec>;
    using CBits                      = typename fpsan::detail::fp_traits<CElem>::bits_type;
    static constexpr int a_lanes     = sizeof(AVec) / sizeof(AElem);
    static constexpr int b_lanes     = sizeof(BVec) / sizeof(BElem);
    static constexpr int c_lanes     = sizeof(CVec) / sizeof(CElem);
    static constexpr int output_regs = Traits::has_opsel ? c_lanes / 2 : c_lanes;
};

template <class Traits>
__device__ inline int gfx11_out_n(int lane)
{
    return lane & 15;
}

template <class Traits>
__device__ inline int gfx11_out_m(int lane, int reg)
{
    return (Traits::wave_size / 16) * reg + (lane >> 4);
}

template <class Traits>
__device__ inline int gfx11_out_index(int reg)
{
    if constexpr(Traits::has_opsel)
        return 2 * reg + (Traits::opsel ? 1 : 0);
    else
        return reg;
}

namespace
{
    template <class T>
    FPSAN_HOST_DEVICE T exact_int(int v)
    {
        return static_cast<T>(static_cast<float>(v));
    }

    template <class T>
    FPSAN_HOST_DEVICE T preserve_seed(int lane, int idx)
    {
        return exact_int<T>(1 + ((lane * 17 + idx * 5) % 127));
    }
} // namespace

// Pack this lane's A/B/C fragments from row-major logical matrices. For gfx11,
// A[m][k] and B[k][n] use vector index k, and row/column values are replicated
// across every lane whose index is congruent modulo 16.
template <class Traits, Semantics S>
__device__ void load_frags(const typename Harness<Traits>::AElem*         A,
                           const typename Harness<Traits>::BElem*         B,
                           const typename Harness<Traits>::CElem*         C,
                           int                                            lane,
                           Value<typename Harness<Traits>::AVec, S, kCC>& a,
                           Value<typename Harness<Traits>::BVec, S, kCC>& b,
                           Value<typename Harness<Traits>::CVec, S, kCC>& c)
{
    typename Harness<Traits>::AVec an{};
    typename Harness<Traits>::BVec bn{};
    typename Harness<Traits>::CVec cn{};
    const int                      outer = lane & 15;
    for(int k = 0; k < K; ++k)
    {
        an[k] = A[outer * K + k];
        bn[k] = B[k * N + outer];
    }
    for(int r = 0; r < Harness<Traits>::output_regs; ++r)
    {
        const int idx = gfx11_out_index<Traits>(r);
        const int m   = gfx11_out_m<Traits>(lane, r);
        const int n   = gfx11_out_n<Traits>(lane);
        cn[idx]       = C[m * N + n];
    }
    a = Value<typename Harness<Traits>::AVec, S, kCC>(an);
    b = Value<typename Harness<Traits>::BVec, S, kCC>(bn);
    c = Value<typename Harness<Traits>::CVec, S, kCC>(cn);
}

template <class Traits, Semantics S, class Out>
__device__ void store_result(Value<typename Harness<Traits>::CVec, S, kCC> d, Out* D)
{
    const int lane = threadIdx.x;
    for(int r = 0; r < Harness<Traits>::output_regs; ++r)
    {
        const int idx = gfx11_out_index<Traits>(r);
        const int m   = gfx11_out_m<Traits>(lane, r);
        const int n   = gfx11_out_n<Traits>(lane);
        if constexpr(S == Semantics::Native)
            D[m * N + n] = d.get(idx).to_float();
        else
            D[m * N + n] = d.get(idx).fpsan_payload();
    }
}

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
    store_result<Traits, Semantics::Native>(d, D);
}

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
    store_result<Traits, Semantics::Triton>(d, Dpay);
}

template <class Traits>
__global__ void k_tied_preserve_float(typename Harness<Traits>::CElem* out)
{
    using H  = Harness<Traits>;
    using AE = typename H::AElem;
    using BE = typename H::BElem;
    using CE = typename H::CElem;
    typename H::AVec an{};
    typename H::BVec bn{};
    typename H::CVec cn{};
    const int        lane = threadIdx.x;
    for(int i = 0; i < H::a_lanes; ++i)
        an[i] = exact_int<AE>(1 + (i % 3));
    for(int i = 0; i < H::b_lanes; ++i)
        bn[i] = exact_int<BE>(1 + ((i + 1) % 3));
    for(int i = 0; i < H::c_lanes; ++i)
        cn[i] = preserve_seed<CE>(lane, i);

    Value<typename H::AVec, Semantics::Native, kCC> a(an);
    Value<typename H::BVec, Semantics::Native, kCC> b(bn);
    Value<typename H::CVec, Semantics::Native, kCC> c(cn);
    auto d = Traits::template call<Semantics::Native, kCC>(a, b, c);
    for(int i = 0; i < H::c_lanes; ++i)
        out[lane * H::c_lanes + i] = d.get(i).to_float();
}

template <class Traits>
__global__ void k_tied_preserve_fpsan(typename Harness<Traits>::CBits* out)
{
    using H  = Harness<Traits>;
    using AE = typename H::AElem;
    using BE = typename H::BElem;
    using CE = typename H::CElem;
    typename H::AVec an{};
    typename H::BVec bn{};
    typename H::CVec cn{};
    const int        lane = threadIdx.x;
    for(int i = 0; i < H::a_lanes; ++i)
        an[i] = exact_int<AE>(1 + (i % 3));
    for(int i = 0; i < H::b_lanes; ++i)
        bn[i] = exact_int<BE>(1 + ((i + 1) % 3));
    for(int i = 0; i < H::c_lanes; ++i)
        cn[i] = preserve_seed<CE>(lane, i);

    Value<typename H::AVec, Semantics::Triton, kCC> a(an);
    Value<typename H::BVec, Semantics::Triton, kCC> b(bn);
    Value<typename H::CVec, Semantics::Triton, kCC> c(cn);
    auto d = Traits::template call<Semantics::Triton, kCC>(a, b, c);
    for(int i = 0; i < H::c_lanes; ++i)
        out[lane * H::c_lanes + i] = d.get(i).fpsan_payload();
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

    template <class T>
    float layout_tolerance()
    {
        if constexpr(std::is_same_v<T, float>)
            return 1.0e-2f;
        else if constexpr(std::is_same_v<T, __bf16>)
            return 7.5e-1f;
        else
            return 5.0e-2f;
    }
} // namespace

template <class Traits>
void run_layout_matches_hardware()
{
    if(!current_device_is_gfx11())
        GTEST_SKIP() << "current HIP device is not gfx11";
    using CE       = typename Harness<Traits>::CElem;
    using AE       = typename Harness<Traits>::AElem;
    using BE       = typename Harness<Traits>::BElem;
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
    CE* dD;
    HIP_CHECK(hipMalloc(&dD, M * N * sizeof(CE)));
    k_builtin<Traits><<<1, Traits::wave_size>>>(dA, dB, dC, dD);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<CE> got(M * N);
    HIP_CHECK(hipMemcpy(got.data(), dD, M * N * sizeof(CE), hipMemcpyDeviceToHost));
    for(int i = 0; i < M * N; ++i)
        EXPECT_NEAR(static_cast<float>(got[i]), static_cast<float>(ref[i]), layout_tolerance<CE>())
            << "layout mismatch at " << (i / N) << "," << (i % N) << " got_bits=" << bits_of(got[i])
            << " ref_bits=" << bits_of(ref[i]);
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
}

template <class Traits>
void run_fpsan_matches_scalar_reference()
{
    if(!current_device_is_gfx11())
        GTEST_SKIP() << "current HIP device is not gfx11";
    using AE       = typename Harness<Traits>::AElem;
    using BE       = typename Harness<Traits>::BElem;
    using CE       = typename Harness<Traits>::CElem;
    using CBits    = typename Harness<Traits>::CBits;
    Mats<Traits> m = make_inputs<Traits>();

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
    k_fpsan<Traits><<<1, Traits::wave_size>>>(dA, dB, dC, dD);
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

template <class Traits>
void run_tied_preserves_unselected()
{
    static_assert(Traits::has_opsel, "preservation is only meaningful for OPSEL WMMA forms");
    if(!current_device_is_gfx11())
        GTEST_SKIP() << "current HIP device is not gfx11";
    using H                 = Harness<Traits>;
    using CE                = typename H::CElem;
    using CBits             = typename H::CBits;
    const std::size_t count = static_cast<std::size_t>(Traits::wave_size) * H::c_lanes;

    CE* dFloat;
    HIP_CHECK(hipMalloc(&dFloat, count * sizeof(CE)));
    k_tied_preserve_float<Traits><<<1, Traits::wave_size>>>(dFloat);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<CE> got_float(count);
    HIP_CHECK(hipMemcpy(got_float.data(), dFloat, count * sizeof(CE), hipMemcpyDeviceToHost));
    (void)hipFree(dFloat);

    CBits* dFpsan;
    HIP_CHECK(hipMalloc(&dFpsan, count * sizeof(CBits)));
    k_tied_preserve_fpsan<Traits><<<1, Traits::wave_size>>>(dFpsan);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<CBits> got_fpsan(count);
    HIP_CHECK(hipMemcpy(got_fpsan.data(), dFpsan, count * sizeof(CBits), hipMemcpyDeviceToHost));
    (void)hipFree(dFpsan);

    for(int lane = 0; lane < Traits::wave_size; ++lane)
        for(int r = 0; r < H::output_regs; ++r)
        {
            const int  idx  = 2 * r + (Traits::opsel ? 0 : 1);
            const CE   seed = preserve_seed<CE>(lane, idx);
            const auto flat = static_cast<std::size_t>(lane) * H::c_lanes + idx;
            EXPECT_EQ(bits_of(got_float[flat]), bits_of(seed))
                << "Float preserve mismatch lane=" << lane << " idx=" << idx;
            using VF = Value<CE, Semantics::Triton, kCC>;
            EXPECT_EQ(got_fpsan[flat], VF(seed).fpsan_payload())
                << "FPSan preserve mismatch lane=" << lane << " idx=" << idx;
        }
}

struct WmmaGfx11F32F16
{
    using AVec                      = v16h_wmma_native;
    using BVec                      = v16h_wmma_native;
    using CVec                      = std::conditional_t<kWaveSize == 64, v4f_native, v8f_native>;
    static constexpr bool has_opsel = false;
    static constexpr bool opsel     = false;
    static constexpr int  wave_size = kWaveSize;
    static constexpr int  a_lo = -3, a_hi = 3;
    static constexpr int  b_lo = -2, b_hi = 2;
    static constexpr int  c_lo = -4, c_hi = 4;
    template <Semantics S, Conversions Conv>
    __device__ static Value<CVec, S, Conv>
        call(Value<AVec, S, Conv> a, Value<BVec, S, Conv> b, Value<CVec, S, Conv> c)
    {
#if defined(FPSAN_TEST_GFX11_WAVE64)
        return fpsan::amdgcn_wmma_f32_16x16x16_f16_w64<S, Conv>(a, b, c);
#else
        return fpsan::amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#endif
    }
};

struct WmmaGfx11F32BF16
{
    using AVec                      = v16bf_wmma_native;
    using BVec                      = v16bf_wmma_native;
    using CVec                      = std::conditional_t<kWaveSize == 64, v4f_native, v8f_native>;
    static constexpr bool has_opsel = false;
    static constexpr bool opsel     = false;
    static constexpr int  wave_size = kWaveSize;
    static constexpr int  a_lo = -3, a_hi = 3;
    static constexpr int  b_lo = -2, b_hi = 2;
    static constexpr int  c_lo = -4, c_hi = 4;
    template <Semantics S, Conversions Conv>
    __device__ static Value<CVec, S, Conv>
        call(Value<AVec, S, Conv> a, Value<BVec, S, Conv> b, Value<CVec, S, Conv> c)
    {
#if defined(FPSAN_TEST_GFX11_WAVE64)
        return fpsan::amdgcn_wmma_f32_16x16x16_bf16_w64<S, Conv>(a, b, c);
#else
        return fpsan::amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
#endif
    }
};

template <bool Opsel, bool Tied = false>
struct WmmaGfx11F16F16
{
    using AVec = v16h_wmma_native;
    using BVec = v16h_wmma_native;
    using CVec = std::conditional_t<kWaveSize == 64, v8h_native, v16h_wmma_native>;
    static constexpr bool has_opsel = true;
    static constexpr bool opsel     = Opsel;
    static constexpr int  wave_size = kWaveSize;
    static constexpr int  a_lo = -3, a_hi = 3;
    static constexpr int  b_lo = -2, b_hi = 2;
    static constexpr int  c_lo = -4, c_hi = 4;
    template <Semantics S, Conversions Conv>
    __device__ static Value<CVec, S, Conv>
        call(Value<AVec, S, Conv> a, Value<BVec, S, Conv> b, Value<CVec, S, Conv> c)
    {
#if defined(FPSAN_TEST_GFX11_WAVE64)
        if constexpr(Tied)
            return fpsan::amdgcn_wmma_f16_16x16x16_f16_tied_w64<Opsel, S, Conv>(a, b, c);
        else
            return fpsan::amdgcn_wmma_f16_16x16x16_f16_w64<Opsel, S, Conv>(a, b, c);
#else
        if constexpr(Tied)
            return fpsan::amdgcn_wmma_f16_16x16x16_f16_tied_w32<Opsel, S, Conv>(a, b, c);
        else
            return fpsan::amdgcn_wmma_f16_16x16x16_f16_w32<Opsel, S, Conv>(a, b, c);
#endif
    }
};

template <bool Opsel, bool Tied = false>
struct WmmaGfx11BF16BF16
{
    using AVec = v16bf_wmma_native;
    using BVec = v16bf_wmma_native;
    using CVec = std::conditional_t<kWaveSize == 64, v8bf_native, v16bf_wmma_native>;
    static constexpr bool has_opsel = true;
    static constexpr bool opsel     = Opsel;
    static constexpr int  wave_size = kWaveSize;
    static constexpr int  a_lo = -3, a_hi = 3;
    static constexpr int  b_lo = -2, b_hi = 2;
    static constexpr int  c_lo = -4, c_hi = 4;
    template <Semantics S, Conversions Conv>
    __device__ static Value<CVec, S, Conv>
        call(Value<AVec, S, Conv> a, Value<BVec, S, Conv> b, Value<CVec, S, Conv> c)
    {
#if defined(FPSAN_TEST_GFX11_WAVE64)
        if constexpr(Tied)
            return fpsan::amdgcn_wmma_bf16_16x16x16_bf16_tied_w64<Opsel, S, Conv>(a, b, c);
        else
            return fpsan::amdgcn_wmma_bf16_16x16x16_bf16_w64<Opsel, S, Conv>(a, b, c);
#else
        if constexpr(Tied)
            return fpsan::amdgcn_wmma_bf16_16x16x16_bf16_tied_w32<Opsel, S, Conv>(a, b, c);
        else
            return fpsan::amdgcn_wmma_bf16_16x16x16_bf16_w32<Opsel, S, Conv>(a, b, c);
#endif
    }
};

using WmmaGfx11F16F16Tied0   = WmmaGfx11F16F16<false, true>;
using WmmaGfx11F16F16Tied1   = WmmaGfx11F16F16<true, true>;
using WmmaGfx11BF16BF16Tied0 = WmmaGfx11BF16BF16<false, true>;
using WmmaGfx11BF16BF16Tied1 = WmmaGfx11BF16BF16<true, true>;

#define FPSAN_WMMA_GFX11_CONCAT2(A, B) A##B
#define FPSAN_WMMA_GFX11_CONCAT(A, B) FPSAN_WMMA_GFX11_CONCAT2(A, B)
#if defined(FPSAN_TEST_GFX11_WAVE64)
#define FPSAN_WMMA_GFX11_SUITE(NAME) FPSAN_WMMA_GFX11_CONCAT(NAME, W64)
#else
#define FPSAN_WMMA_GFX11_SUITE(NAME) FPSAN_WMMA_GFX11_CONCAT(NAME, W32)
#endif

#define FPSAN_WMMA_GFX11_TESTS(NAME, TRAITS)                        \
    TEST(FPSAN_WMMA_GFX11_SUITE(NAME), LayoutMatchesHardware)       \
    {                                                               \
        run_layout_matches_hardware<TRAITS>();                      \
    }                                                               \
    TEST(FPSAN_WMMA_GFX11_SUITE(NAME), FpsanMatchesScalarReference) \
    {                                                               \
        run_fpsan_matches_scalar_reference<TRAITS>();               \
    }

#define FPSAN_WMMA_GFX11_TIED_PRESERVE_TEST(NAME, TRAITS)       \
    TEST(FPSAN_WMMA_GFX11_SUITE(NAME), PreservesUnselectedHalf) \
    {                                                           \
        run_tied_preserves_unselected<TRAITS>();                \
    }

FPSAN_WMMA_GFX11_TESTS(WmmaGfx11F32F16, WmmaGfx11F32F16)
FPSAN_WMMA_GFX11_TESTS(WmmaGfx11F32BF16, WmmaGfx11F32BF16)
FPSAN_WMMA_GFX11_TESTS(WmmaGfx11F16F16Opsel0, WmmaGfx11F16F16<false>)
FPSAN_WMMA_GFX11_TESTS(WmmaGfx11F16F16Opsel1, WmmaGfx11F16F16<true>)
FPSAN_WMMA_GFX11_TESTS(WmmaGfx11BF16BF16Opsel0, WmmaGfx11BF16BF16<false>)
FPSAN_WMMA_GFX11_TESTS(WmmaGfx11BF16BF16Opsel1, WmmaGfx11BF16BF16<true>)
FPSAN_WMMA_GFX11_TESTS(WmmaGfx11F16F16TiedOpsel0, WmmaGfx11F16F16Tied0)
FPSAN_WMMA_GFX11_TESTS(WmmaGfx11F16F16TiedOpsel1, WmmaGfx11F16F16Tied1)
FPSAN_WMMA_GFX11_TESTS(WmmaGfx11BF16BF16TiedOpsel0, WmmaGfx11BF16BF16Tied0)
FPSAN_WMMA_GFX11_TESTS(WmmaGfx11BF16BF16TiedOpsel1, WmmaGfx11BF16BF16Tied1)
FPSAN_WMMA_GFX11_TIED_PRESERVE_TEST(WmmaGfx11F16F16TiedOpsel0, WmmaGfx11F16F16Tied0)
FPSAN_WMMA_GFX11_TIED_PRESERVE_TEST(WmmaGfx11F16F16TiedOpsel1, WmmaGfx11F16F16Tied1)
FPSAN_WMMA_GFX11_TIED_PRESERVE_TEST(WmmaGfx11BF16BF16TiedOpsel0, WmmaGfx11BF16BF16Tied0)
FPSAN_WMMA_GFX11_TIED_PRESERVE_TEST(WmmaGfx11BF16BF16TiedOpsel1, WmmaGfx11BF16BF16Tied1)

#undef FPSAN_WMMA_GFX11_TIED_PRESERVE_TEST
#undef FPSAN_WMMA_GFX11_TESTS
#undef FPSAN_WMMA_GFX11_SUITE
#undef FPSAN_WMMA_GFX11_CONCAT
#undef FPSAN_WMMA_GFX11_CONCAT2
