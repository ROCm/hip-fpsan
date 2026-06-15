// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/wave_test.cpp
//
// GPU tests for the wave-cooperative reduce wrappers in fpsan/amdgcn_wave.hpp.
// The harness is wave-size agnostic: it queries the device warp size at run
// time unless FPSAN_TEST_FORCE_WAVE_SIZE selects a compile-time wave size,
// launches exactly one full wave, and reduces the host references over that
// same lane count. This is what makes the suite authoritative on gfx950 and on
// explicit gfx11 wave64 FPSan targets -- a wave32-only reference would silently
// pass while the wrapper folded only half the wave.
//
// Per (op, type) we certify:
//   - FpsanMatchesHostButterfly: FPSan-mode payload equals a host scalar
//     butterfly that mirrors the wrapper's XOR tree stage-for-stage over the
//     full wave. Certifies the payload algebra AND the cross-lane shuffle
//     (32- and 64-bit scalars, 32- and 64-lane waves) together.
//   - FpsanMatchesSeqFold (associative ops only): FPSan-mode payload equals an
//     INDEPENDENT host reference that folds the lane payloads left-to-right --
//     a different reduction order than the device butterfly. Passing both pins
//     down order-independence empirically, not just by construction.
//   - FloatMatchesHostButterfly (associative ops only): Float-mode wrapper
//     bit-exactly matches a host butterfly with float arithmetic on exact-int
//     inputs (the hardware reduce and our tree agree on these values).
//   - FpsanStrategyInvariant: FPSan output is bit-identical across strategies.
#include "fpsan/amdgcn_wave.hpp"
#include "fpsan/fpsan.hpp"

#include "hip_test_utils.hpp"
#include "test_random.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

static constexpr Conversions kCC = Conversions::Explicit;
#ifndef FPSAN_TEST_ENABLE_WAVE_F64
#define FPSAN_TEST_ENABLE_WAVE_F64 0
#endif
#ifndef FPSAN_TEST_FORCE_WAVE_SIZE
#define FPSAN_TEST_FORCE_WAVE_SIZE 0
#endif
#ifndef FPSAN_TEST_SKIP_FLOAT_WAVE_REDUCE
#define FPSAN_TEST_SKIP_FLOAT_WAVE_REDUCE 0
#endif
static_assert(FPSAN_TEST_FORCE_WAVE_SIZE == 0 || FPSAN_TEST_FORCE_WAVE_SIZE == 32
                  || FPSAN_TEST_FORCE_WAVE_SIZE == 64,
              "wave tests support default device wave size, wave32, or wave64");
// Upper bound on lanes in one wave (wave64). Host reference scratch is sized
// to this; the active count is the runtime warp size.
static constexpr int kMaxLanes = 64;

// ---------------------------------------------------------------------------
// Per-(op, type) traits. Only the wrapper name and the combine expression
// vary. COMBINE_EXPR names the two operands `a` and `b`.
// ---------------------------------------------------------------------------
#define WAVE_TRAITS(name, FT_, builtin_name, COMBINE_EXPR, fsub_ok)                                  \
    struct name                                                                                      \
    {                                                                                                \
        using FT                                 = FT_;                                              \
        using Bits                               = typename fpsan::detail::fp_traits<FT>::bits_type; \
        static constexpr bool sequential_is_safe = (fsub_ok);                                        \
        template <int Strategy, Semantics S>                                                         \
        __device__ static Value<FT, S, kCC> call(Value<FT, S, kCC> v)                                \
        {                                                                                            \
            return fpsan::builtin_name<Strategy>(v);                                                 \
        }                                                                                            \
        template <Semantics S>                                                                       \
        static Value<FT, S, kCC> combine(Value<FT, S, kCC> a, Value<FT, S, kCC> b)                   \
        {                                                                                            \
            return (COMBINE_EXPR);                                                                   \
        }                                                                                            \
        template <Semantics S>                                                                       \
        static Value<FT, S, kCC> host_butterfly(const Value<FT, S, kCC>* lane_vals, int n)           \
        {                                                                                            \
            Value<FT, S, kCC> r[kMaxLanes], nx[kMaxLanes];                                           \
            for(int i = 0; i < n; ++i)                                                               \
                r[i] = lane_vals[i];                                                                 \
            for(int off = 1; off < n; off <<= 1)                                                     \
            {                                                                                        \
                for(int i = 0; i < n; ++i)                                                           \
                    nx[i] = combine<S>(r[i], r[i ^ off]);                                            \
                for(int i = 0; i < n; ++i)                                                           \
                    r[i] = nx[i];                                                                    \
            }                                                                                        \
            return r[0];                                                                             \
        }                                                                                            \
        template <Semantics S>                                                                       \
        static Value<FT, S, kCC> host_seq_fold(const Value<FT, S, kCC>* lane_vals, int n)            \
        {                                                                                            \
            Value<FT, S, kCC> r = lane_vals[0];                                                      \
            for(int i = 1; i < n; ++i)                                                               \
                r = combine<S>(r, lane_vals[i]);                                                     \
            return r;                                                                                \
        }                                                                                            \
    };

// `sequential_is_safe`: whether float-arith over the reduce is associative +
// commutative enough that hardware reduce, host butterfly, and host sequential
// fold all land on the same bits for exact-int inputs. fsub is not
// associative, so its Float comparison and its seq-fold comparison are skipped;
// the FPSan butterfly reference (same tree shape) still certifies it.
WAVE_TRAITS(WaveFaddF32, float, amdgcn_wave_reduce_fadd_f32, a + b, true)
WAVE_TRAITS(WaveFsubF32, float, amdgcn_wave_reduce_fsub_f32, a - b, false)
WAVE_TRAITS(WaveFminF32, float, amdgcn_wave_reduce_fmin_f32, fpsan::min(a, b), true)
WAVE_TRAITS(WaveFmaxF32, float, amdgcn_wave_reduce_fmax_f32, fpsan::max(a, b), true)
#if FPSAN_TEST_ENABLE_WAVE_F64
WAVE_TRAITS(WaveFaddF64, double, amdgcn_wave_reduce_fadd_f64, a + b, true)
WAVE_TRAITS(WaveFsubF64, double, amdgcn_wave_reduce_fsub_f64, a - b, false)
WAVE_TRAITS(WaveFminF64, double, amdgcn_wave_reduce_fmin_f64, fpsan::min(a, b), true)
WAVE_TRAITS(WaveFmaxF64, double, amdgcn_wave_reduce_fmax_f64, fpsan::max(a, b), true)
#endif

// ---------------------------------------------------------------------------
// Kernels. One full wave; lane reads in[lane], lane 0 writes the result.
// ---------------------------------------------------------------------------
template <class T, int Strategy>
__global__ void k_wave_float(const typename T::FT* in, typename T::FT* out)
{
    const int                                     lane = threadIdx.x;
    Value<typename T::FT, Semantics::Native, kCC> v{in[lane]};
    auto r = T::template call<Strategy, Semantics::Native>(v);
    if(lane == 0)
        *out = static_cast<typename T::FT>(r);
}

template <class T, int Strategy>
__global__ void k_wave_fpsan(const typename T::FT* in, typename T::Bits* out)
{
    const int                                     lane = threadIdx.x;
    Value<typename T::FT, Semantics::Triton, kCC> v{in[lane]};
    auto r = T::template call<Strategy, Semantics::Triton>(v);
    if(lane == 0)
        *out = r.fpsan_payload();
}

// ---------------------------------------------------------------------------
// Inputs + helpers.
// ---------------------------------------------------------------------------
namespace
{

    // Runtime warp size of device 0 (32 or 64), or 0 if no device.
    int device_wave_size()
    {
#if FPSAN_TEST_FORCE_WAVE_SIZE != 0
        return FPSAN_TEST_FORCE_WAVE_SIZE;
#else
        int ndev = 0;
        if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
            return 0;
        hipDeviceProp_t p{};
        if(hipGetDeviceProperties(&p, 0) != hipSuccess)
            return 0;
        return p.warpSize;
#endif
    }

    template <class FT>
    std::vector<FT> make_lane_inputs(int n)
    {
        // Small exact integers so sums/mins/maxes stay exact in f32 and the
        // Float-mode comparison against the host reference is bit-exact.
        std::vector<FT> v(n);
        std::mt19937    rng = fpsan_test::make_rng();
        for(auto& x : v)
            x = fpsan_test::pick_int_valued<FT>(rng, -7, 7);
        return v;
    }

} // namespace

// ---------------------------------------------------------------------------
// Test functions (parameterized on Trait T).
// ---------------------------------------------------------------------------
template <class T>
void run_float_matches_host_butterfly()
{
    if constexpr(FPSAN_TEST_SKIP_FLOAT_WAVE_REDUCE)
    {
        GTEST_SKIP() << "Float wave_reduce is skipped for this target";
    }
    else if constexpr(!T::sequential_is_safe)
    {
        GTEST_SKIP() << "non-associative op; float result depends on hw tree shape";
    }
    else
    {
        const int W = device_wave_size();
        if(W == 0)
            GTEST_SKIP() << "no HIP device";
        using FT = typename T::FT;
        auto in  = make_lane_inputs<FT>(W);
        using F  = Value<FT, Semantics::Native, kCC>;
        std::vector<F> lane_vals(W);
        for(int i = 0; i < W; ++i)
            lane_vals[i] = F{in[i]};
        F ref = T::template host_butterfly<Semantics::Native>(lane_vals.data(), W);

        FT *dIn = to_dev(in), *dOut;
        HIP_CHECK(hipMalloc(&dOut, sizeof(FT)));
        k_wave_float<T, 0><<<1, W>>>(dIn, dOut);
        HIP_CHECK(hipDeviceSynchronize());
        FT got = FT(0);
        HIP_CHECK(hipMemcpy(&got, dOut, sizeof(FT), hipMemcpyDeviceToHost));
        EXPECT_EQ(bits_of(got), bits_of(static_cast<FT>(ref)));
        (void)hipFree(dIn);
        (void)hipFree(dOut);
    }
}

template <class T>
void run_fpsan_matches_host_butterfly()
{
    const int W = device_wave_size();
    if(W == 0)
        GTEST_SKIP() << "no HIP device";
    using FT   = typename T::FT;
    using Bits = typename T::Bits;
    auto in    = make_lane_inputs<FT>(W);
    using F    = Value<FT, Semantics::Triton, kCC>;
    std::vector<F> lane_vals(W);
    for(int i = 0; i < W; ++i)
        lane_vals[i] = F{in[i]};
    F ref = T::template host_butterfly<Semantics::Triton>(lane_vals.data(), W);

    FT*   dIn = to_dev(in);
    Bits* dOut;
    HIP_CHECK(hipMalloc(&dOut, sizeof(Bits)));
    k_wave_fpsan<T, 0><<<1, W>>>(dIn, dOut);
    HIP_CHECK(hipDeviceSynchronize());
    Bits got = Bits(0);
    HIP_CHECK(hipMemcpy(&got, dOut, sizeof(Bits), hipMemcpyDeviceToHost));
    EXPECT_EQ(got, ref.fpsan_payload());
    (void)hipFree(dIn);
    (void)hipFree(dOut);
}

// Independent oracle: a left-to-right sequential fold of the lane payloads,
// which is a *different* reduction order than the device butterfly. For the
// associative + commutative payload ops (fadd, fmin, fmax) the two must agree;
// this is a stronger statement than "matches a same-shaped butterfly".
template <class T>
void run_fpsan_matches_seq_fold()
{
    if constexpr(!T::sequential_is_safe)
    {
        GTEST_SKIP() << "non-associative op; sequential order may differ from hw";
    }
    else
    {
        const int W = device_wave_size();
        if(W == 0)
            GTEST_SKIP() << "no HIP device";
        using FT   = typename T::FT;
        using Bits = typename T::Bits;
        auto in    = make_lane_inputs<FT>(W);
        using F    = Value<FT, Semantics::Triton, kCC>;
        std::vector<F> lane_vals(W);
        for(int i = 0; i < W; ++i)
            lane_vals[i] = F{in[i]};
        F ref = T::template host_seq_fold<Semantics::Triton>(lane_vals.data(), W);

        FT*   dIn = to_dev(in);
        Bits* dOut;
        HIP_CHECK(hipMalloc(&dOut, sizeof(Bits)));
        k_wave_fpsan<T, 0><<<1, W>>>(dIn, dOut);
        HIP_CHECK(hipDeviceSynchronize());
        Bits got = Bits(0);
        HIP_CHECK(hipMemcpy(&got, dOut, sizeof(Bits), hipMemcpyDeviceToHost));
        EXPECT_EQ(got, ref.fpsan_payload());
        (void)hipFree(dIn);
        (void)hipFree(dOut);
    }
}

template <class T>
void run_fpsan_strategy_invariant()
{
    const int W = device_wave_size();
    if(W == 0)
        GTEST_SKIP() << "no HIP device";
    using FT   = typename T::FT;
    using Bits = typename T::Bits;
    auto  in   = make_lane_inputs<FT>(W);
    FT*   dIn  = to_dev(in);
    Bits* dOut;
    HIP_CHECK(hipMalloc(&dOut, sizeof(Bits)));
    k_wave_fpsan<T, 0><<<1, W>>>(dIn, dOut);
    HIP_CHECK(hipDeviceSynchronize());
    Bits got0 = Bits(0);
    HIP_CHECK(hipMemcpy(&got0, dOut, sizeof(Bits), hipMemcpyDeviceToHost));
    k_wave_fpsan<T, 1><<<1, W>>>(dIn, dOut);
    HIP_CHECK(hipDeviceSynchronize());
    Bits got1 = Bits(0);
    HIP_CHECK(hipMemcpy(&got1, dOut, sizeof(Bits), hipMemcpyDeviceToHost));
    EXPECT_EQ(got0, got1);
    (void)hipFree(dIn);
    (void)hipFree(dOut);
}

// ---------------------------------------------------------------------------
// Per-Trait TEST instantiations.
// ---------------------------------------------------------------------------
#define WAVE_TESTS(Trait)                          \
    TEST(Trait, FloatMatchesHostButterfly)         \
    {                                              \
        run_float_matches_host_butterfly<Trait>(); \
    }                                              \
    TEST(Trait, FpsanMatchesHostButterfly)         \
    {                                              \
        run_fpsan_matches_host_butterfly<Trait>(); \
    }                                              \
    TEST(Trait, FpsanMatchesSeqFold)               \
    {                                              \
        run_fpsan_matches_seq_fold<Trait>();       \
    }                                              \
    TEST(Trait, FpsanStrategyInvariant)            \
    {                                              \
        run_fpsan_strategy_invariant<Trait>();     \
    }

WAVE_TESTS(WaveFaddF32)
WAVE_TESTS(WaveFsubF32)
WAVE_TESTS(WaveFminF32)
WAVE_TESTS(WaveFmaxF32)
#if FPSAN_TEST_ENABLE_WAVE_F64
WAVE_TESTS(WaveFaddF64)
WAVE_TESTS(WaveFsubF64)
WAVE_TESTS(WaveFminF64)
WAVE_TESTS(WaveFmaxF64)
#endif
