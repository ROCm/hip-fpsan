// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/xlane_test.cpp
//
// GPU tests for the cross-lane data movers in fpsan/amdgcn_wave.hpp. Every
// wrapper is pure bit movement -- the storage bits of one lane's Value end
// up at some other lane unchanged -- so the property under test is that
// after the move, each lane's payload (FPSan mode) and float bits (Float
// mode) equal the source lane's, with the source lane chosen by the
// builtin's documented semantics. Float-mode and FPSan-mode share the same
// bit-mover, so they should agree bit-for-bit on the lane mapping.
#include "fpsan/amdgcn_wave.hpp"
#include "fpsan/fpsan.hpp"

#include "hip_test_utils.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

static constexpr Conversions kCC   = Conversions::Explicit;
static constexpr int         LANES = 32;

// Distinct per-lane f32 input: integer lane*7 + 1, signed -- exact in f32, but
// large enough that the bit pattern at each lane is unique.
static __device__ inline float lane_input_float(int lane)
{
    return static_cast<float>(lane * 7 + 1) - 100.f;
}

// ---- readlane ---------------------------------------------------------------
template <Semantics S, class Out>
__global__ void k_readlane(Out* out, int from)
{
    const int            lane = threadIdx.x;
    Value<float, S, kCC> v{lane_input_float(lane)};
    auto                 r = fpsan::amdgcn_readlane(v, from);
    if constexpr(S == Semantics::Float)
        out[lane] = static_cast<float>(r);
    else
        out[lane] = r.fpsan_payload();
}

template <Semantics S>
void test_readlane(int from)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    using Out = std::conditional_t<S == Semantics::Float, float, std::uint32_t>;
    Out* d_out;
    HIP_CHECK(hipMalloc(&d_out, LANES * sizeof(Out)));
    k_readlane<S><<<1, LANES>>>(d_out, from);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<Out> got(LANES);
    HIP_CHECK(hipMemcpy(got.data(), d_out, LANES * sizeof(Out), hipMemcpyDeviceToHost));
    // Every lane must see the SAME value: source lane's bits.
    using V = Value<float, S, kCC>;
    // lane_input_float is __device__ -- re-derive on host.
    const float src = static_cast<float>(from * 7 + 1) - 100.f;
    V           src_v{src};
    Out         expected;
    if constexpr(S == Semantics::Float)
        expected = static_cast<float>(src_v);
    else
        expected = src_v.fpsan_payload();
    for(int i = 0; i < LANES; ++i)
        EXPECT_EQ(got[i], expected) << "lane " << i;
    (void)hipFree(d_out);
}

TEST(Xlane, ReadlaneFloat0)
{
    test_readlane<Semantics::Float>(0);
}
TEST(Xlane, ReadlaneFloat17)
{
    test_readlane<Semantics::Float>(17);
}
TEST(Xlane, ReadlaneFpsan0)
{
    test_readlane<Semantics::FPSan>(0);
}
TEST(Xlane, ReadlaneFpsan17)
{
    test_readlane<Semantics::FPSan>(17);
}

// ---- readfirstlane (= readlane(0) when lane 0 is active) --------------------
template <Semantics S, class Out>
__global__ void k_readfirstlane(Out* out)
{
    const int            lane = threadIdx.x;
    Value<float, S, kCC> v{lane_input_float(lane)};
    auto                 r = fpsan::amdgcn_readfirstlane(v);
    if constexpr(S == Semantics::Float)
        out[lane] = static_cast<float>(r);
    else
        out[lane] = r.fpsan_payload();
}

template <Semantics S>
void test_readfirstlane()
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    using Out = std::conditional_t<S == Semantics::Float, float, std::uint32_t>;
    Out* d_out;
    HIP_CHECK(hipMalloc(&d_out, LANES * sizeof(Out)));
    k_readfirstlane<S><<<1, LANES>>>(d_out);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<Out> got(LANES);
    HIP_CHECK(hipMemcpy(got.data(), d_out, LANES * sizeof(Out), hipMemcpyDeviceToHost));
    using V         = Value<float, S, kCC>;
    const float src = static_cast<float>(0 * 7 + 1) - 100.f;
    V           src_v{src};
    Out         expected;
    if constexpr(S == Semantics::Float)
        expected = static_cast<float>(src_v);
    else
        expected = src_v.fpsan_payload();
    for(int i = 0; i < LANES; ++i)
        EXPECT_EQ(got[i], expected) << "lane " << i;
    (void)hipFree(d_out);
}

TEST(Xlane, ReadfirstlaneFloat)
{
    test_readfirstlane<Semantics::Float>();
}
TEST(Xlane, ReadfirstlaneFpsan)
{
    test_readfirstlane<Semantics::FPSan>();
}

// ---- ds_bpermute (gather: result[lane] = src[addr[lane]/4]) -----------------
// Each lane writes a chosen src_lane*4 to addr; ds_bpermute returns the value
// from the lane addressed by addr/4. We test the common pattern lane^XOR by
// running a butterfly stage.
template <Semantics S, class Out>
__global__ void k_ds_bpermute_xor(Out* out, int off)
{
    const int            lane = threadIdx.x;
    Value<float, S, kCC> v{lane_input_float(lane)};
    // Each lane requests the value from (lane ^ off).
    auto r = fpsan::amdgcn_ds_bpermute((lane ^ off) * 4, v);
    if constexpr(S == Semantics::Float)
        out[lane] = static_cast<float>(r);
    else
        out[lane] = r.fpsan_payload();
}

template <Semantics S>
void test_ds_bpermute_xor(int off)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    using Out = std::conditional_t<S == Semantics::Float, float, std::uint32_t>;
    Out* d_out;
    HIP_CHECK(hipMalloc(&d_out, LANES * sizeof(Out)));
    k_ds_bpermute_xor<S><<<1, LANES>>>(d_out, off);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<Out> got(LANES);
    HIP_CHECK(hipMemcpy(got.data(), d_out, LANES * sizeof(Out), hipMemcpyDeviceToHost));
    using V = Value<float, S, kCC>;
    for(int i = 0; i < LANES; ++i)
    {
        const float src = static_cast<float>((i ^ off) * 7 + 1) - 100.f;
        V           src_v{src};
        Out         expected;
        if constexpr(S == Semantics::Float)
            expected = static_cast<float>(src_v);
        else
            expected = src_v.fpsan_payload();
        EXPECT_EQ(got[i], expected) << "lane " << i << " xor " << off;
    }
    (void)hipFree(d_out);
}

TEST(Xlane, DsBpermuteXorFloat1)
{
    test_ds_bpermute_xor<Semantics::Float>(1);
}
TEST(Xlane, DsBpermuteXorFloat16)
{
    test_ds_bpermute_xor<Semantics::Float>(16);
}
TEST(Xlane, DsBpermuteXorFpsan1)
{
    test_ds_bpermute_xor<Semantics::FPSan>(1);
}
TEST(Xlane, DsBpermuteXorFpsan16)
{
    test_ds_bpermute_xor<Semantics::FPSan>(16);
}

// ---- ds_permute (scatter: result[addr[lane]/4] = src[lane]) -----------------
// Inverse semantics from bpermute: each lane WRITES to the lane indicated by
// addr/4. Using addr = (lane ^ off) * 4, the value at lane = lane was written
// by lane (lane ^ off), so result[lane] = src[lane ^ off] -- same observed
// mapping as ds_bpermute under symmetric XOR.
template <Semantics S, class Out>
__global__ void k_ds_permute_xor(Out* out, int off)
{
    const int            lane = threadIdx.x;
    Value<float, S, kCC> v{lane_input_float(lane)};
    auto                 r = fpsan::amdgcn_ds_permute((lane ^ off) * 4, v);
    if constexpr(S == Semantics::Float)
        out[lane] = static_cast<float>(r);
    else
        out[lane] = r.fpsan_payload();
}

template <Semantics S>
void test_ds_permute_xor(int off)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    using Out = std::conditional_t<S == Semantics::Float, float, std::uint32_t>;
    Out* d_out;
    HIP_CHECK(hipMalloc(&d_out, LANES * sizeof(Out)));
    k_ds_permute_xor<S><<<1, LANES>>>(d_out, off);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<Out> got(LANES);
    HIP_CHECK(hipMemcpy(got.data(), d_out, LANES * sizeof(Out), hipMemcpyDeviceToHost));
    using V = Value<float, S, kCC>;
    for(int i = 0; i < LANES; ++i)
    {
        // Lane i was written by lane (i ^ off).
        const float src = static_cast<float>((i ^ off) * 7 + 1) - 100.f;
        V           src_v{src};
        Out         expected;
        if constexpr(S == Semantics::Float)
            expected = static_cast<float>(src_v);
        else
            expected = src_v.fpsan_payload();
        EXPECT_EQ(got[i], expected) << "lane " << i << " xor " << off;
    }
    (void)hipFree(d_out);
}

TEST(Xlane, DsPermuteXorFloat1)
{
    test_ds_permute_xor<Semantics::Float>(1);
}
TEST(Xlane, DsPermuteXorFloat16)
{
    test_ds_permute_xor<Semantics::Float>(16);
}
TEST(Xlane, DsPermuteXorFpsan1)
{
    test_ds_permute_xor<Semantics::FPSan>(1);
}
TEST(Xlane, DsPermuteXorFpsan16)
{
    test_ds_permute_xor<Semantics::FPSan>(16);
}

// ---- ds_swizzle (cross-mode consistency) ------------------------------------
// ds_swizzle encodings are intricate and hardware-revision-specific; rather
// than pin down a specific permutation, we verify the load-bearing FPSan
// invariant: Float-mode and FPSan-mode wrappers move the bits the SAME way.
// That is, for any swizzle pattern, the lane mapping in Float mode must match
// the lane mapping in FPSan mode (both wrappers route through the same
// detail::bit_move helper, but this test catches any divergence).
template <Semantics S, class Out>
__global__ void k_ds_swizzle(Out* out, int pattern_select)
{
    const int            lane = threadIdx.x;
    Value<float, S, kCC> v{lane_input_float(lane)};
    // Use template-parameter dispatch on a small set of patterns; the runtime
    // arg picks which one.
    decltype(v) r;
    if(pattern_select == 0)
        r = fpsan::amdgcn_ds_swizzle<0x041F>(v); // some permutation
    else
        r = fpsan::amdgcn_ds_swizzle<0x8000>(v); // BroadcastMode lane 0
    if constexpr(S == Semantics::Float)
        out[lane] = static_cast<float>(r);
    else
        out[lane] = r.fpsan_payload();
}

TEST(Xlane, DsSwizzleFloatVsFpsanLaneMapping)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    for(int sel = 0; sel < 2; ++sel)
    {
        float*         d_f;
        std::uint32_t* d_p;
        HIP_CHECK(hipMalloc(&d_f, LANES * sizeof(float)));
        HIP_CHECK(hipMalloc(&d_p, LANES * sizeof(std::uint32_t)));
        k_ds_swizzle<Semantics::Float><<<1, LANES>>>(d_f, sel);
        k_ds_swizzle<Semantics::FPSan><<<1, LANES>>>(d_p, sel);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<float>         got_f(LANES);
        std::vector<std::uint32_t> got_p(LANES);
        HIP_CHECK(hipMemcpy(got_f.data(), d_f, LANES * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(
            hipMemcpy(got_p.data(), d_p, LANES * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
        using VF = Value<float, Semantics::Float, kCC>;
        using VP = Value<float, Semantics::FPSan, kCC>;
        // For each output lane, reverse-engineer which source lane the Float
        // wrapper picked, then verify the FPSan wrapper picked the SAME lane.
        for(int i = 0; i < LANES; ++i)
        {
            int src_lane = -1;
            for(int j = 0; j < LANES; ++j)
            {
                const float src = static_cast<float>(j * 7 + 1) - 100.f;
                if(static_cast<float>(VF{src}) == got_f[i])
                {
                    src_lane = j;
                    break;
                }
            }
            ASSERT_NE(src_lane, -1)
                << "Float output at lane " << i << " sel=" << sel << " matches no source lane";
            const float src = static_cast<float>(src_lane * 7 + 1) - 100.f;
            EXPECT_EQ(got_p[i], VP{src}.fpsan_payload())
                << "FPSan lane mapping differs from Float at lane " << i << " sel=" << sel;
        }
        (void)hipFree(d_f);
        (void)hipFree(d_p);
    }
}

// ---- mov_dpp (QUAD_PERM identity = 0xE4: lane k <- lane k) ------------------
// QUAD_PERM is universal across gfx generations; the encoding 0xE4 selects
// (0,1,2,3) which is the identity within each quad. Together with row_mask =
// bank_mask = 0xF (all rows/banks active), every lane should observe its own
// input value bit-for-bit -- a clean smoke test that the wrapper compiles and
// the bit-mover round-trips through the DPP unit.
template <Semantics S, class Out>
__global__ void k_mov_dpp_identity(Out* out)
{
    const int            lane = threadIdx.x;
    Value<float, S, kCC> v{lane_input_float(lane)};
    auto                 r = fpsan::amdgcn_mov_dpp<0xE4, 0xF, 0xF, false>(v);
    if constexpr(S == Semantics::Float)
        out[lane] = static_cast<float>(r);
    else
        out[lane] = r.fpsan_payload();
}

template <Semantics S>
void test_mov_dpp_identity()
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    using Out = std::conditional_t<S == Semantics::Float, float, std::uint32_t>;
    Out* d_out;
    HIP_CHECK(hipMalloc(&d_out, LANES * sizeof(Out)));
    k_mov_dpp_identity<S><<<1, LANES>>>(d_out);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<Out> got(LANES);
    HIP_CHECK(hipMemcpy(got.data(), d_out, LANES * sizeof(Out), hipMemcpyDeviceToHost));
    using V = Value<float, S, kCC>;
    for(int i = 0; i < LANES; ++i)
    {
        const float src = static_cast<float>(i * 7 + 1) - 100.f;
        V           src_v{src};
        Out         expected;
        if constexpr(S == Semantics::Float)
            expected = static_cast<float>(src_v);
        else
            expected = src_v.fpsan_payload();
        EXPECT_EQ(got[i], expected) << "lane " << i;
    }
    (void)hipFree(d_out);
}

TEST(Xlane, MovDppIdentityFloat)
{
    test_mov_dpp_identity<Semantics::Float>();
}
TEST(Xlane, MovDppIdentityFpsan)
{
    test_mov_dpp_identity<Semantics::FPSan>();
}

// ---- mov_dpp8 (identity selector 0x76543210 = lane i reads lane i) ----------
template <Semantics S, class Out>
__global__ void k_mov_dpp8_identity(Out* out)
{
    const int            lane = threadIdx.x;
    Value<float, S, kCC> v{lane_input_float(lane)};
    // Identity 8-lane selector: each 3-bit field i holds value i, packed ->
    // 0xFAC688 (NOT 0x76543210 — fields are 3 bits wide for 8 lanes/row, not 4).
    auto r = fpsan::amdgcn_mov_dpp8<0xFAC688u>(v);
    if constexpr(S == Semantics::Float)
        out[lane] = static_cast<float>(r);
    else
        out[lane] = r.fpsan_payload();
}

template <Semantics S>
void test_mov_dpp8_identity()
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    using Out = std::conditional_t<S == Semantics::Float, float, std::uint32_t>;
    Out* d_out;
    HIP_CHECK(hipMalloc(&d_out, LANES * sizeof(Out)));
    k_mov_dpp8_identity<S><<<1, LANES>>>(d_out);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<Out> got(LANES);
    HIP_CHECK(hipMemcpy(got.data(), d_out, LANES * sizeof(Out), hipMemcpyDeviceToHost));
    using V = Value<float, S, kCC>;
    for(int i = 0; i < LANES; ++i)
    {
        const float src = static_cast<float>(i * 7 + 1) - 100.f;
        V           src_v{src};
        Out         expected;
        if constexpr(S == Semantics::Float)
            expected = static_cast<float>(src_v);
        else
            expected = src_v.fpsan_payload();
        EXPECT_EQ(got[i], expected) << "lane " << i;
    }
    (void)hipFree(d_out);
}

TEST(Xlane, MovDpp8IdentityFloat)
{
    test_mov_dpp8_identity<Semantics::Float>();
}
TEST(Xlane, MovDpp8IdentityFpsan)
{
    test_mov_dpp8_identity<Semantics::FPSan>();
}

// ---- permlane64 (wave32: half-swap is unobservable; check no-crash) --------
__global__ void k_permlane64_smoke(const float* in, float* out)
{
    const int                           lane = threadIdx.x;
    Value<float, Semantics::Float, kCC> v{in[lane]};
    out[lane] = static_cast<float>(fpsan::amdgcn_permlane64(v));
}

TEST(Xlane, Permlane64Smoke)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    std::vector<float> in(LANES);
    for(int i = 0; i < LANES; ++i)
        in[i] = static_cast<float>(i);
    float *dIn, *dOut;
    HIP_CHECK(hipMalloc(&dIn, LANES * sizeof(float)));
    HIP_CHECK(hipMalloc(&dOut, LANES * sizeof(float)));
    HIP_CHECK(hipMemcpy(dIn, in.data(), LANES * sizeof(float), hipMemcpyHostToDevice));
    k_permlane64_smoke<<<1, LANES>>>(dIn, dOut);
    HIP_CHECK(hipDeviceSynchronize());
    // No crash + the bit-mover round-trips: wave32 lane i reads its own value
    // back (the upper-half partner doesn't exist).
    std::vector<float> got(LANES);
    HIP_CHECK(hipMemcpy(got.data(), dOut, LANES * sizeof(float), hipMemcpyDeviceToHost));
    for(int i = 0; i < LANES; ++i)
        EXPECT_EQ(got[i], in[i]) << "lane " << i;
    (void)hipFree(dIn);
    (void)hipFree(dOut);
}

// ---- ballot (wave32: bit i of result set iff lane i passed true) -----------
__global__ void k_ballot(const int* pred, std::uint32_t* mask)
{
    const int     lane = threadIdx.x;
    std::uint32_t m    = fpsan::amdgcn_ballot_w32(pred[lane] != 0);
    if(lane == 0)
        *mask = m;
}

TEST(Xlane, BallotW32)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    std::vector<int> pred(LANES);
    // Set even lanes -> true, odd -> false. Expected mask = 0x55555555.
    for(int i = 0; i < LANES; ++i)
        pred[i] = (i % 2 == 0) ? 1 : 0;
    int*           dPred;
    std::uint32_t* dMask;
    HIP_CHECK(hipMalloc(&dPred, LANES * sizeof(int)));
    HIP_CHECK(hipMalloc(&dMask, sizeof(std::uint32_t)));
    HIP_CHECK(hipMemcpy(dPred, pred.data(), LANES * sizeof(int), hipMemcpyHostToDevice));
    k_ballot<<<1, LANES>>>(dPred, dMask);
    HIP_CHECK(hipDeviceSynchronize());
    std::uint32_t mask = 0;
    HIP_CHECK(hipMemcpy(&mask, dMask, sizeof mask, hipMemcpyDeviceToHost));
    EXPECT_EQ(mask, 0x55555555u);
    (void)hipFree(dPred);
    (void)hipFree(dMask);
}
