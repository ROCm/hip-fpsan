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
// builtin's semantics. Float-mode and FPSan-mode share the same
// bit-mover, so they should agree bit-for-bit on the lane mapping.
#include "fpsan/amdgcn_wave.hpp"
#include "fpsan/fpsan.hpp"

#include "hip_test_utils.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

static constexpr Conversions kCC = Conversions::Explicit;
#ifndef FPSAN_TEST_FORCE_WAVE_SIZE
#define FPSAN_TEST_FORCE_WAVE_SIZE 32
#endif
static_assert(FPSAN_TEST_FORCE_WAVE_SIZE == 32 || FPSAN_TEST_FORCE_WAVE_SIZE == 64,
              "xlane tests support one wave32 or wave64 wave");
static constexpr int LANES = FPSAN_TEST_FORCE_WAVE_SIZE;

// Distinct per-lane f32 input: integer lane*7 + 1, signed -- exact in f32, but
// large enough that the bit pattern at each lane is unique.
static __device__ inline float lane_input_float(int lane)
{
    return static_cast<float>(lane * 7 + 1) - 100.f;
}

#if FPSAN_TEST_FORCE_WAVE_SIZE == 64
static bool device_is_rdna4()
{
    int dev = 0;
    if(hipGetDevice(&dev) != hipSuccess)
        return false;
    hipDeviceProp_t prop{};
    if(hipGetDeviceProperties(&prop, dev) != hipSuccess)
        return false;
    return std::strncmp(prop.gcnArchName, "gfx120", 6) == 0;
}

static bool ds_permute_uses_full_wave64()
{
    return device_is_rdna4() || device_is_gfx950();
}
#endif

// ---- readlane ---------------------------------------------------------------
template <Semantics S, class Out>
__global__ void k_readlane(Out* out, int from)
{
    const int            lane = threadIdx.x;
    Value<float, S, kCC> v{lane_input_float(lane)};
    auto                 r = fpsan::amdgcn_readlane(v, from);
    if constexpr(S == Semantics::Native)
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
    using Out = std::conditional_t<S == Semantics::Native, float, std::uint32_t>;
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
    if constexpr(S == Semantics::Native)
        expected = static_cast<float>(src_v);
    else
        expected = src_v.fpsan_payload();
    for(int i = 0; i < LANES; ++i)
        EXPECT_EQ(got[i], expected) << "lane " << i;
    (void)hipFree(d_out);
}

TEST(Xlane, ReadlaneFloat0)
{
    test_readlane<Semantics::Native>(0);
}
TEST(Xlane, ReadlaneFloat17)
{
    test_readlane<Semantics::Native>(17);
}
TEST(Xlane, ReadlaneFpsan0)
{
    test_readlane<Semantics::Triton>(0);
}
TEST(Xlane, ReadlaneFpsan17)
{
    test_readlane<Semantics::Triton>(17);
}
#if FPSAN_TEST_FORCE_WAVE_SIZE == 64
TEST(Xlane, ReadlaneFloat48)
{
    test_readlane<Semantics::Native>(48);
}
TEST(Xlane, ReadlaneFpsan48)
{
    test_readlane<Semantics::Triton>(48);
}
#endif

// ---- readfirstlane (= readlane(0) when lane 0 is active) --------------------
template <Semantics S, class Out>
__global__ void k_readfirstlane(Out* out)
{
    const int            lane = threadIdx.x;
    Value<float, S, kCC> v{lane_input_float(lane)};
    auto                 r = fpsan::amdgcn_readfirstlane(v);
    if constexpr(S == Semantics::Native)
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
    using Out = std::conditional_t<S == Semantics::Native, float, std::uint32_t>;
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
    if constexpr(S == Semantics::Native)
        expected = static_cast<float>(src_v);
    else
        expected = src_v.fpsan_payload();
    for(int i = 0; i < LANES; ++i)
        EXPECT_EQ(got[i], expected) << "lane " << i;
    (void)hipFree(d_out);
}

TEST(Xlane, ReadfirstlaneFloat)
{
    test_readfirstlane<Semantics::Native>();
}
TEST(Xlane, ReadfirstlaneFpsan)
{
    test_readfirstlane<Semantics::Triton>();
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
    if constexpr(S == Semantics::Native)
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
    using Out = std::conditional_t<S == Semantics::Native, float, std::uint32_t>;
    Out* d_out;
    HIP_CHECK(hipMalloc(&d_out, LANES * sizeof(Out)));
    k_ds_bpermute_xor<S><<<1, LANES>>>(d_out, off);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<Out> got(LANES);
    HIP_CHECK(hipMemcpy(got.data(), d_out, LANES * sizeof(Out), hipMemcpyDeviceToHost));
    using V = Value<float, S, kCC>;
#if FPSAN_TEST_FORCE_WAVE_SIZE == 64
    const bool fullwave64 = ds_permute_uses_full_wave64();
#endif
    for(int i = 0; i < LANES; ++i)
    {
        int src_lane = i ^ off;
#if FPSAN_TEST_FORCE_WAVE_SIZE == 64
        if(!fullwave64)
            src_lane = (i & ~31) | (src_lane & 31);
#endif
        const float src = static_cast<float>(src_lane * 7 + 1) - 100.f;
        V           src_v{src};
        Out         expected;
        if constexpr(S == Semantics::Native)
            expected = static_cast<float>(src_v);
        else
            expected = src_v.fpsan_payload();
        EXPECT_EQ(got[i], expected) << "lane " << i << " xor " << off;
    }
    (void)hipFree(d_out);
}

TEST(Xlane, DsBpermuteXorFloat1)
{
    test_ds_bpermute_xor<Semantics::Native>(1);
}
TEST(Xlane, DsBpermuteXorFloat16)
{
    test_ds_bpermute_xor<Semantics::Native>(16);
}
TEST(Xlane, DsBpermuteXorFpsan1)
{
    test_ds_bpermute_xor<Semantics::Triton>(1);
}
TEST(Xlane, DsBpermuteXorFpsan16)
{
    test_ds_bpermute_xor<Semantics::Triton>(16);
}
#if FPSAN_TEST_FORCE_WAVE_SIZE == 64
// off=31 covers the largest same-half XOR in W64 mode.
TEST(Xlane, DsBpermuteXorFloat31)
{
    test_ds_bpermute_xor<Semantics::Native>(31);
}
TEST(Xlane, DsBpermuteXorFpsan31)
{
    test_ds_bpermute_xor<Semantics::Triton>(31);
}
// off=32 is the cross-half boundary: gfx11 expects modulo-32 lane selection,
// while RDNA4/gfx950 expect a full-wave exchange.
TEST(Xlane, DsBpermuteXorFloat32Boundary)
{
    test_ds_bpermute_xor<Semantics::Native>(32);
}
TEST(Xlane, DsBpermuteXorFpsan32Boundary)
{
    test_ds_bpermute_xor<Semantics::Triton>(32);
}
#endif

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
    if constexpr(S == Semantics::Native)
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
    using Out = std::conditional_t<S == Semantics::Native, float, std::uint32_t>;
    Out* d_out;
    HIP_CHECK(hipMalloc(&d_out, LANES * sizeof(Out)));
    k_ds_permute_xor<S><<<1, LANES>>>(d_out, off);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<Out> got(LANES);
    HIP_CHECK(hipMemcpy(got.data(), d_out, LANES * sizeof(Out), hipMemcpyDeviceToHost));
    using V = Value<float, S, kCC>;
#if FPSAN_TEST_FORCE_WAVE_SIZE == 64
    const bool fullwave64 = ds_permute_uses_full_wave64();
#endif
    for(int i = 0; i < LANES; ++i)
    {
        int src_lane = i ^ off;
#if FPSAN_TEST_FORCE_WAVE_SIZE == 64
        if(!fullwave64)
            src_lane = (i & ~31) | (src_lane & 31);
#endif
        // Lane i was written by the selected source lane.
        const float src = static_cast<float>(src_lane * 7 + 1) - 100.f;
        V           src_v{src};
        Out         expected;
        if constexpr(S == Semantics::Native)
            expected = static_cast<float>(src_v);
        else
            expected = src_v.fpsan_payload();
        EXPECT_EQ(got[i], expected) << "lane " << i << " xor " << off;
    }
    (void)hipFree(d_out);
}

TEST(Xlane, DsPermuteXorFloat1)
{
    test_ds_permute_xor<Semantics::Native>(1);
}
TEST(Xlane, DsPermuteXorFloat16)
{
    test_ds_permute_xor<Semantics::Native>(16);
}
TEST(Xlane, DsPermuteXorFpsan1)
{
    test_ds_permute_xor<Semantics::Triton>(1);
}
TEST(Xlane, DsPermuteXorFpsan16)
{
    test_ds_permute_xor<Semantics::Triton>(16);
}
#if FPSAN_TEST_FORCE_WAVE_SIZE == 64
// off=31 covers the largest same-half XOR in W64 mode.
TEST(Xlane, DsPermuteXorFloat31)
{
    test_ds_permute_xor<Semantics::Native>(31);
}
TEST(Xlane, DsPermuteXorFpsan31)
{
    test_ds_permute_xor<Semantics::Triton>(31);
}
// off=32 is the cross-half boundary: gfx11 expects modulo-32 lane selection,
// while RDNA4/gfx950 expect a full-wave exchange.
TEST(Xlane, DsPermuteXorFloat32Boundary)
{
    test_ds_permute_xor<Semantics::Native>(32);
}
TEST(Xlane, DsPermuteXorFpsan32Boundary)
{
    test_ds_permute_xor<Semantics::Triton>(32);
}
#endif

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
    if constexpr(S == Semantics::Native)
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
        k_ds_swizzle<Semantics::Native><<<1, LANES>>>(d_f, sel);
        k_ds_swizzle<Semantics::Triton><<<1, LANES>>>(d_p, sel);
        HIP_CHECK(hipDeviceSynchronize());
        std::vector<float>         got_f(LANES);
        std::vector<std::uint32_t> got_p(LANES);
        HIP_CHECK(hipMemcpy(got_f.data(), d_f, LANES * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(
            hipMemcpy(got_p.data(), d_p, LANES * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
        using VF = Value<float, Semantics::Native, kCC>;
        using VP = Value<float, Semantics::Triton, kCC>;
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

// ---- ds_swizzle (independent host oracle of the lane map) --------------------
// Beyond the Float==FPSan cross-mode check above, decode the ds_swizzle pattern
// to the exact source lane and assert BOTH modes match it -- so the wrapper is
// checked against the known hardware lane map, not just against itself.
// Encoding (DS_SWIZZLE_B32 offset, silicon-grounded below):
//   bit15=1  -> QuadPerm: within each group of 4, dst = quad_base + sel where
//               sel = (pat >> (2*(lane&3))) & 3.
//   bit15=0  -> Bitmask (within groups of 32): l' = ((l & and) | or) ^ xor,
//               and=pat[4:0], or=pat[9:5], xor=pat[14:10].
static int ds_swizzle_src(int lane, unsigned pat)
{
    if(pat & 0x8000u)
    {
        const int q   = lane & ~3;
        const int sel = (pat >> (2 * (lane & 3))) & 3;
        return q + sel;
    }
    const int andm = pat & 0x1F, orm = (pat >> 5) & 0x1F, xorm = (pat >> 10) & 0x1F;
    const int base = lane & ~31, l = lane & 31;
    return base + (((l & andm) | orm) ^ xorm);
}

template <unsigned PAT, Semantics S, class Out>
__global__ void k_ds_swizzle_pat(Out* out)
{
    const int            lane = threadIdx.x;
    Value<float, S, kCC> v{lane_input_float(lane)};
    auto                 r = fpsan::amdgcn_ds_swizzle<PAT>(v);
    if constexpr(S == Semantics::Native)
        out[lane] = static_cast<float>(r);
    else
        out[lane] = r.fpsan_payload();
}

template <unsigned PAT>
static void check_ds_swizzle_oracle(const char* tag)
{
    using FOut = float;
    using POut = std::uint32_t;
    FOut* d_f;
    POut* d_p;
    HIP_CHECK(hipMalloc(&d_f, LANES * sizeof(FOut)));
    HIP_CHECK(hipMalloc(&d_p, LANES * sizeof(POut)));
    k_ds_swizzle_pat<PAT, Semantics::Native><<<1, LANES>>>(d_f);
    k_ds_swizzle_pat<PAT, Semantics::Triton><<<1, LANES>>>(d_p);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<FOut> gf(LANES);
    std::vector<POut> gp(LANES);
    HIP_CHECK(hipMemcpy(gf.data(), d_f, LANES * sizeof(FOut), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(gp.data(), d_p, LANES * sizeof(POut), hipMemcpyDeviceToHost));
    using VF = Value<float, Semantics::Native, kCC>;
    using VP = Value<float, Semantics::Triton, kCC>;
    for(int i = 0; i < LANES; ++i)
    {
        const int   s   = ds_swizzle_src(i, PAT);
        const float src = static_cast<float>(s * 7 + 1) - 100.f;
        EXPECT_EQ(gf[i], static_cast<float>(VF{src}))
            << tag << " Float lane " << i << " expected src lane " << s;
        EXPECT_EQ(gp[i], VP{src}.fpsan_payload())
            << tag << " FPSan lane " << i << " expected src lane " << s;
    }
    (void)hipFree(d_f);
    (void)hipFree(d_p);
}

TEST(Xlane, DsSwizzleHostOracle)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    check_ds_swizzle_oracle<0x001F>("identity"); // bitmask: and=0x1F -> l'=l
    check_ds_swizzle_oracle<0x041F>("xor1"); // bitmask: xor=1 -> swap pairs
    check_ds_swizzle_oracle<0x081F>("xor2"); // bitmask: xor=2
    check_ds_swizzle_oracle<0x101F>("xor4"); // bitmask: xor=4
    check_ds_swizzle_oracle<0x8000>("quad_bcast0"); // quad: all pick quad-lane 0
    check_ds_swizzle_oracle<0x80B1>("quad_swap"); // quad perm (1,0,3,2)
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
    if constexpr(S == Semantics::Native)
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
    using Out = std::conditional_t<S == Semantics::Native, float, std::uint32_t>;
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
        if constexpr(S == Semantics::Native)
            expected = static_cast<float>(src_v);
        else
            expected = src_v.fpsan_payload();
        EXPECT_EQ(got[i], expected) << "lane " << i;
    }
    (void)hipFree(d_out);
}

TEST(Xlane, MovDppIdentityFloat)
{
    test_mov_dpp_identity<Semantics::Native>();
}
TEST(Xlane, MovDppIdentityFpsan)
{
    test_mov_dpp_identity<Semantics::Triton>();
}

// ---- update_dpp (quad swap plus row-mask blend with old value) ---------------
static int dpp_quad_src(int lane, unsigned ctrl)
{
    const int base = lane & ~3;
    return base + int((ctrl >> (2 * (lane & 3))) & 3u);
}

template <Semantics S, class Out>
__global__ void k_update_dpp_quad_swap_row_mask(Out* out)
{
    const int            lane = threadIdx.x;
    Value<float, S, kCC> old{lane_input_float(lane) + 1000.f};
    Value<float, S, kCC> v{lane_input_float(lane)};
    auto                 r = fpsan::amdgcn_update_dpp<0xB1, 0x5, 0xF, false>(old, v);
    if constexpr(S == Semantics::Native)
        out[lane] = static_cast<float>(r);
    else
        out[lane] = r.fpsan_payload();
}

template <Semantics S>
void test_update_dpp_quad_swap_row_mask()
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    using Out = std::conditional_t<S == Semantics::Native, float, std::uint32_t>;
    Out* d_out;
    HIP_CHECK(hipMalloc(&d_out, LANES * sizeof(Out)));
    k_update_dpp_quad_swap_row_mask<S><<<1, LANES>>>(d_out);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<Out> got(LANES);
    HIP_CHECK(hipMemcpy(got.data(), d_out, LANES * sizeof(Out), hipMemcpyDeviceToHost));
    using V = Value<float, S, kCC>;
    for(int i = 0; i < LANES; ++i)
    {
        const int  row            = (i >> 4) & 3;
        const bool row_is_enabled = (0x5 & (1 << row)) != 0;
        const int  src_lane       = row_is_enabled ? dpp_quad_src(i, 0xB1) : i;
        float      src            = static_cast<float>(src_lane * 7 + 1) - 100.f;
        if(!row_is_enabled)
            src += 1000.f;
        V   src_v{src};
        Out expected;
        if constexpr(S == Semantics::Native)
            expected = static_cast<float>(src_v);
        else
            expected = src_v.fpsan_payload();
        EXPECT_EQ(got[i], expected) << "lane " << i;
    }
    (void)hipFree(d_out);
}

TEST(Xlane, UpdateDppQuadSwapRowMaskFloat)
{
    test_update_dpp_quad_swap_row_mask<Semantics::Native>();
}
TEST(Xlane, UpdateDppQuadSwapRowMaskFpsan)
{
    test_update_dpp_quad_swap_row_mask<Semantics::Triton>();
}

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_mov_dpp8)
// ---- mov_dpp8 (identity selector 0x76543210 = lane i reads lane i) ----------
template <Semantics S, class Out>
__global__ void k_mov_dpp8_identity(Out* out)
{
    const int            lane = threadIdx.x;
    Value<float, S, kCC> v{lane_input_float(lane)};
    // Identity 8-lane selector: each 3-bit field i holds value i, packed ->
    // 0xFAC688 (NOT 0x76543210 — fields are 3 bits wide for 8 lanes/row, not 4).
    auto r = fpsan::amdgcn_mov_dpp8<0xFAC688u>(v);
    if constexpr(S == Semantics::Native)
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
    using Out = std::conditional_t<S == Semantics::Native, float, std::uint32_t>;
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
        if constexpr(S == Semantics::Native)
            expected = static_cast<float>(src_v);
        else
            expected = src_v.fpsan_payload();
        EXPECT_EQ(got[i], expected) << "lane " << i;
    }
    (void)hipFree(d_out);
}

TEST(Xlane, MovDpp8IdentityFloat)
{
    if(device_is_gfx950())
        GTEST_SKIP() << "mov_dpp8 is not a gfx950 op";
    test_mov_dpp8_identity<Semantics::Native>();
}
TEST(Xlane, MovDpp8IdentityFpsan)
{
    if(device_is_gfx950())
        GTEST_SKIP() << "mov_dpp8 is not a gfx950 op";
    test_mov_dpp8_identity<Semantics::Triton>();
}

#endif // __has_builtin(__builtin_amdgcn_mov_dpp8)

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_permlane64)
// ---- permlane64 (wave64 swaps low/high halves; wave32 is identity) ----------
template <Semantics S, class Out>
__global__ void k_permlane64(const float* in, Out* out)
{
    const int            lane = threadIdx.x;
    Value<float, S, kCC> v{in[lane]};
    auto                 r = fpsan::amdgcn_permlane64(v);
    if constexpr(S == Semantics::Native)
        out[lane] = static_cast<float>(r);
    else
        out[lane] = r.fpsan_payload();
}

template <Semantics S>
void test_permlane64()
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    if(device_is_gfx950())
        GTEST_SKIP() << "permlane64 is not a gfx950 op";
    using Out = std::conditional_t<S == Semantics::Native, float, std::uint32_t>;
    std::vector<float> in(LANES);
    for(int i = 0; i < LANES; ++i)
        in[i] = static_cast<float>(i);
    float* dIn;
    Out*   dOut;
    HIP_CHECK(hipMalloc(&dIn, LANES * sizeof(float)));
    HIP_CHECK(hipMalloc(&dOut, LANES * sizeof(Out)));
    HIP_CHECK(hipMemcpy(dIn, in.data(), LANES * sizeof(float), hipMemcpyHostToDevice));
    k_permlane64<S><<<1, LANES>>>(dIn, dOut);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<Out> got(LANES);
    HIP_CHECK(hipMemcpy(got.data(), dOut, LANES * sizeof(Out), hipMemcpyDeviceToHost));
    if constexpr(LANES == 32)
    {
        if(device_is_gfx1250())
        {
            SUCCEED() << "permlane64 wrapper executed without crashing on gfx1250 wave32";
            (void)hipFree(dIn);
            (void)hipFree(dOut);
            return;
        }
    }
    using V = Value<float, S, kCC>;
    for(int i = 0; i < LANES; ++i)
    {
        const int src = (LANES == 64) ? (i ^ 32) : i;
        V         src_v{in[src]};
        Out       expected;
        if constexpr(S == Semantics::Native)
            expected = static_cast<float>(src_v);
        else
            expected = src_v.fpsan_payload();
        EXPECT_EQ(got[i], expected) << "lane " << i;
    }
    (void)hipFree(dIn);
    (void)hipFree(dOut);
}

TEST(Xlane, Permlane64Float)
{
    test_permlane64<Semantics::Native>();
}
TEST(Xlane, Permlane64Fpsan)
{
    test_permlane64<Semantics::Triton>();
}
#endif // __has_builtin(__builtin_amdgcn_permlane64)

// ---- gfx1250 permlane bcast/down/up/xor (cross-mode bit-identity) -----------
// Selector semantics are intricate; we certify the load-bearing invariant that
// the Float and FPSan wrappers route bits identically. With input encoding the
// lane id, the value that lands at output lane i (whatever the permutation,
// including broadcast/fill) is the same source in both modes, so the FPSan
// payload at lane i must equal the payload of the Float float at lane i.
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_permlane_bcast)
enum class PermOp
{
    Bcast,
    Down,
    Up,
    Xor
};

template <PermOp Op, Semantics S, class Out>
__global__ void k_permlane_family(Out* out, int sel0, int sel1)
{
    const int            lane = threadIdx.x;
    Value<float, S, kCC> v{lane_input_float(lane)};
    Value<float, S, kCC> r;
    if constexpr(Op == PermOp::Bcast)
        r = fpsan::amdgcn_permlane_bcast(v, sel0, sel1);
    else if constexpr(Op == PermOp::Down)
        r = fpsan::amdgcn_permlane_down(v, sel0, sel1);
    else if constexpr(Op == PermOp::Up)
        r = fpsan::amdgcn_permlane_up(v, sel0, sel1);
    else
        r = fpsan::amdgcn_permlane_xor(v, sel0, sel1);
    if constexpr(S == Semantics::Native)
        out[lane] = static_cast<float>(r);
    else
        out[lane] = r.fpsan_payload();
}

template <PermOp Op>
void test_permlane_family(int sel0, int sel1)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    if(!device_is_gfx1250())
        GTEST_SKIP() << "permlane_{bcast,down,up,xor} are gfx1250-only ops";
    float*         d_f;
    std::uint32_t* d_p;
    HIP_CHECK(hipMalloc(&d_f, LANES * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_p, LANES * sizeof(std::uint32_t)));
    k_permlane_family<Op, Semantics::Native><<<1, LANES>>>(d_f, sel0, sel1);
    k_permlane_family<Op, Semantics::Triton><<<1, LANES>>>(d_p, sel0, sel1);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float>         got_f(LANES);
    std::vector<std::uint32_t> got_p(LANES);
    HIP_CHECK(hipMemcpy(got_f.data(), d_f, LANES * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(got_p.data(), d_p, LANES * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    using VP = Value<float, Semantics::Triton, kCC>;
    for(int i = 0; i < LANES; ++i)
        EXPECT_EQ(got_p[i], VP{got_f[i]}.fpsan_payload()) << "lane " << i;
    (void)hipFree(d_f);
    (void)hipFree(d_p);
}

// gfx1250 v_permlane_{bcast,down,up,xor}_b32 lane crossbar: the Float and FPSan
// wrappers are pure pass-through bit moves (identical structure to the validated
// readlane / ds_bpermute / ds_swizzle / mov_dpp movers above), so with the input
// encoding the lane id, the value landing at output lane i must be the same
// source in both modes -- the load-bearing cross-mode bit-identity invariant.
// NOTE: the second wrapper argument (sel1) is S2 = the *lane group width*, which
// the ISA requires to be a nonzero power of two -- the hardware computes
// numLaneGroups = waveSize / laneGroupWidth, so width 0 is a divide-by-zero.
// Use the full wave32 width (32). sel0 is the op-specific selector: broadcast
// index (bcast), shift delta (down/up), or xor mask (xor).
TEST(Xlane, PermlaneBcastCrossMode)
{
    test_permlane_family<PermOp::Bcast>(0, 32);
}
TEST(Xlane, PermlaneDownCrossMode)
{
    test_permlane_family<PermOp::Down>(0x1, 32);
}
TEST(Xlane, PermlaneUpCrossMode)
{
    test_permlane_family<PermOp::Up>(0x1, 32);
}
TEST(Xlane, PermlaneXorCrossMode)
{
    test_permlane_family<PermOp::Xor>(0x1, 32);
}
#endif // __has_builtin(__builtin_amdgcn_permlane_bcast)

// ---- ds_bpermute_fi (gather; same observed mapping as ds_bpermute) ----------
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_ds_bpermute_fi_b32)
template <Semantics S, class Out>
__global__ void k_ds_bpermute_fi_xor(Out* out, int off)
{
    const int            lane = threadIdx.x;
    Value<float, S, kCC> v{lane_input_float(lane)};
    auto                 r = fpsan::amdgcn_ds_bpermute_fi((lane ^ off) * 4, v);
    if constexpr(S == Semantics::Native)
        out[lane] = static_cast<float>(r);
    else
        out[lane] = r.fpsan_payload();
}

template <Semantics S>
void test_ds_bpermute_fi_xor(int off)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    if(!device_is_gfx12())
        GTEST_SKIP() << "ds_bpermute_fi_b32 requires gfx12-insts";
    using Out = std::conditional_t<S == Semantics::Native, float, std::uint32_t>;
    Out* d_out;
    HIP_CHECK(hipMalloc(&d_out, LANES * sizeof(Out)));
    k_ds_bpermute_fi_xor<S><<<1, LANES>>>(d_out, off);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<Out> got(LANES);
    HIP_CHECK(hipMemcpy(got.data(), d_out, LANES * sizeof(Out), hipMemcpyDeviceToHost));
    using V = Value<float, S, kCC>;
    for(int i = 0; i < LANES; ++i)
    {
        const float src = static_cast<float>((i ^ off) * 7 + 1) - 100.f;
        V           src_v{src};
        Out         expected;
        if constexpr(S == Semantics::Native)
            expected = static_cast<float>(src_v);
        else
            expected = src_v.fpsan_payload();
        EXPECT_EQ(got[i], expected) << "lane " << i << " xor " << off;
    }
    (void)hipFree(d_out);
}

TEST(Xlane, DsBpermuteFiXorFloat1)
{
    test_ds_bpermute_fi_xor<Semantics::Native>(1);
}
TEST(Xlane, DsBpermuteFiXorFpsan1)
{
    test_ds_bpermute_fi_xor<Semantics::Triton>(1);
}
#if FPSAN_TEST_FORCE_WAVE_SIZE == 64
TEST(Xlane, DsBpermuteFiXorFloat32)
{
    test_ds_bpermute_fi_xor<Semantics::Native>(32);
}
TEST(Xlane, DsBpermuteFiXorFpsan32)
{
    test_ds_bpermute_fi_xor<Semantics::Triton>(32);
}
#endif
#endif // __has_builtin(__builtin_amdgcn_ds_bpermute_fi_b32)

// ---- permlane16 / permlanex16 (independent host oracle of the lane map) ------
// sel0 packs the source-lane nibble for dst lanes 0..7, sel1 for 8..15 (each a
// 4-bit index within the 16-group). Silicon-grounded:
//   permlane16 [lane] = (lane & ~15)        + idx(lane & 15)   (own 16-group)
//   permlanex16[lane] = ((lane & ~15) ^ 16) + idx(lane & 15)   (other 16-group)
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_permlane16)
static int pl16_idx(int j, unsigned sel0, unsigned sel1)
{
    return j < 8 ? int((sel0 >> (4 * j)) & 0xF) : int((sel1 >> (4 * (j - 8))) & 0xF);
}

template <Semantics S, bool CROSS, class Out>
__global__ void k_permlane16(Out* out, unsigned sel0, unsigned sel1)
{
    const int            lane = threadIdx.x;
    Value<float, S, kCC> src{lane_input_float(lane)};
    Value<float, S, kCC> old{-999.f};
    auto r = CROSS ? fpsan::amdgcn_permlanex16<false, false>(old, src, (int)sel0, (int)sel1)
                   : fpsan::amdgcn_permlane16<false, false>(old, src, (int)sel0, (int)sel1);
    if constexpr(S == Semantics::Native)
        out[lane] = static_cast<float>(r);
    else
        out[lane] = r.fpsan_payload();
}

template <bool CROSS>
static void check_permlane16(unsigned sel0, unsigned sel1, const char* tag)
{
    float*         d_f;
    std::uint32_t* d_p;
    HIP_CHECK(hipMalloc(&d_f, LANES * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_p, LANES * sizeof(std::uint32_t)));
    k_permlane16<Semantics::Native, CROSS><<<1, LANES>>>(d_f, sel0, sel1);
    k_permlane16<Semantics::Triton, CROSS><<<1, LANES>>>(d_p, sel0, sel1);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float>         gf(LANES);
    std::vector<std::uint32_t> gp(LANES);
    HIP_CHECK(hipMemcpy(gf.data(), d_f, LANES * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(gp.data(), d_p, LANES * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    using VF = Value<float, Semantics::Native, kCC>;
    using VP = Value<float, Semantics::Triton, kCC>;
    for(int i = 0; i < LANES; ++i)
    {
        const int   base = CROSS ? ((i & ~15) ^ 16) : (i & ~15);
        const int   s    = base + pl16_idx(i & 15, sel0, sel1);
        const float src  = static_cast<float>(s * 7 + 1) - 100.f;
        EXPECT_EQ(gf[i], static_cast<float>(VF{src})) << tag << " Float lane " << i << " src " << s;
        EXPECT_EQ(gp[i], VP{src}.fpsan_payload()) << tag << " FPSan lane " << i << " src " << s;
    }
    (void)hipFree(d_f);
    (void)hipFree(d_p);
}

TEST(Xlane, Permlane16HostOracle)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    if(device_is_gfx950())
        GTEST_SKIP() << "permlane16/permlanex16 are not gfx950 ops";
    unsigned id0 = 0, id1 = 0, sw0 = 0, sw1 = 0;
    for(int i = 0; i < 8; ++i)
    {
        id0 |= unsigned(i) << (4 * i); // identity: idx j -> j
        id1 |= unsigned(i + 8) << (4 * i);
        sw0 |= unsigned(i ^ 1) << (4 * i); // swap pairs: idx j -> j^1
        sw1 |= unsigned((i + 8) ^ 1) << (4 * i);
    }
    check_permlane16<false>(id0, id1, "pl16_identity");
    check_permlane16<false>(sw0, sw1, "pl16_swap");
    check_permlane16<true>(id0, id1, "plx16_identity");
    check_permlane16<true>(sw0, sw1, "plx16_swap");
}
#endif // __has_builtin(__builtin_amdgcn_permlane16)

// ---- permlane16_swap (cross-mode bit-identity on both swapped operands) -----
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_permlane16_swap)
template <Semantics S, class Out>
__global__ void k_permlane16_swap(Out* outx, Out* outy)
{
    const int            lane = threadIdx.x;
    Value<float, S, kCC> x{lane_input_float(lane)};
    Value<float, S, kCC> y{lane_input_float(lane) + 1000.f};
    fpsan::amdgcn_permlane16_swap<false, false>(x, y);
    if constexpr(S == Semantics::Native)
    {
        outx[lane] = static_cast<float>(x);
        outy[lane] = static_cast<float>(y);
    }
    else
    {
        outx[lane] = x.fpsan_payload();
        outy[lane] = y.fpsan_payload();
    }
}

TEST(Xlane, Permlane16SwapCrossMode)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    if(!device_is_gfx1250() && !device_is_gfx950())
        GTEST_SKIP() << "permlane16_swap is only tested on gfx1250/gfx950";
    float *        dfx, *dfy;
    std::uint32_t *dpx, *dpy;
    HIP_CHECK(hipMalloc(&dfx, LANES * sizeof(float)));
    HIP_CHECK(hipMalloc(&dfy, LANES * sizeof(float)));
    HIP_CHECK(hipMalloc(&dpx, LANES * sizeof(std::uint32_t)));
    HIP_CHECK(hipMalloc(&dpy, LANES * sizeof(std::uint32_t)));
    k_permlane16_swap<Semantics::Native><<<1, LANES>>>(dfx, dfy);
    k_permlane16_swap<Semantics::Triton><<<1, LANES>>>(dpx, dpy);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float>         fx(LANES), fy(LANES);
    std::vector<std::uint32_t> px(LANES), py(LANES);
    HIP_CHECK(hipMemcpy(fx.data(), dfx, LANES * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(fy.data(), dfy, LANES * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(px.data(), dpx, LANES * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(py.data(), dpy, LANES * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    using VP = Value<float, Semantics::Triton, kCC>;
    for(int i = 0; i < LANES; ++i)
    {
        EXPECT_EQ(px[i], VP{fx[i]}.fpsan_payload()) << "x lane " << i;
        EXPECT_EQ(py[i], VP{fy[i]}.fpsan_payload()) << "y lane " << i;
    }
    (void)hipFree(dfx);
    (void)hipFree(dfy);
    (void)hipFree(dpx);
    (void)hipFree(dpy);
}
#endif // __has_builtin(__builtin_amdgcn_permlane16_swap)

// ---- permlane32_swap (cross-mode bit-identity on both swapped operands) -----
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_permlane32_swap)
template <Semantics S, class Out>
__global__ void k_permlane32_swap(Out* outx, Out* outy)
{
    const int            lane = threadIdx.x;
    Value<float, S, kCC> x{lane_input_float(lane)};
    Value<float, S, kCC> y{lane_input_float(lane) + 2000.f};
    fpsan::amdgcn_permlane32_swap<false, false>(x, y);
    if constexpr(S == Semantics::Native)
    {
        outx[lane] = static_cast<float>(x);
        outy[lane] = static_cast<float>(y);
    }
    else
    {
        outx[lane] = x.fpsan_payload();
        outy[lane] = y.fpsan_payload();
    }
}

TEST(Xlane, Permlane32SwapCrossMode)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    if(!device_is_gfx950())
        GTEST_SKIP() << "permlane32_swap is only tested on gfx950";
    float *        dfx, *dfy;
    std::uint32_t *dpx, *dpy;
    HIP_CHECK(hipMalloc(&dfx, LANES * sizeof(float)));
    HIP_CHECK(hipMalloc(&dfy, LANES * sizeof(float)));
    HIP_CHECK(hipMalloc(&dpx, LANES * sizeof(std::uint32_t)));
    HIP_CHECK(hipMalloc(&dpy, LANES * sizeof(std::uint32_t)));
    k_permlane32_swap<Semantics::Native><<<1, LANES>>>(dfx, dfy);
    k_permlane32_swap<Semantics::Triton><<<1, LANES>>>(dpx, dpy);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float>         fx(LANES), fy(LANES);
    std::vector<std::uint32_t> px(LANES), py(LANES);
    HIP_CHECK(hipMemcpy(fx.data(), dfx, LANES * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(fy.data(), dfy, LANES * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(px.data(), dpx, LANES * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(py.data(), dpy, LANES * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    using VP = Value<float, Semantics::Triton, kCC>;
    for(int i = 0; i < LANES; ++i)
    {
        EXPECT_EQ(px[i], VP{fx[i]}.fpsan_payload()) << "x lane " << i;
        EXPECT_EQ(py[i], VP{fy[i]}.fpsan_payload()) << "y lane " << i;
    }
    (void)hipFree(dfx);
    (void)hipFree(dfy);
    (void)hipFree(dpx);
    (void)hipFree(dpy);
}
#endif // __has_builtin(__builtin_amdgcn_permlane32_swap)

// ---- permlane_idx_gen (integer index generator; mode-independent smoke) -----
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_permlane_idx_gen)
__global__ void k_permlane_idx_gen(std::uint32_t* out, int sel)
{
    const int lane = threadIdx.x;
    out[lane]      = fpsan::amdgcn_permlane_idx_gen(lane, sel);
}

TEST(Xlane, PermlaneIdxGenDeterministic)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    if(!device_is_gfx1250())
        GTEST_SKIP() << "permlane_idx_gen is a gfx1250-only op";
    std::uint32_t* d0;
    std::uint32_t* d1;
    HIP_CHECK(hipMalloc(&d0, LANES * sizeof(std::uint32_t)));
    HIP_CHECK(hipMalloc(&d1, LANES * sizeof(std::uint32_t)));
    k_permlane_idx_gen<<<1, LANES>>>(d0, 0);
    k_permlane_idx_gen<<<1, LANES>>>(d1, 0);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> a(LANES), b(LANES);
    HIP_CHECK(hipMemcpy(a.data(), d0, LANES * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(b.data(), d1, LANES * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for(int i = 0; i < LANES; ++i)
        EXPECT_EQ(a[i], b[i]) << "lane " << i;
    (void)hipFree(d0);
    (void)hipFree(d1);
}
#endif // __has_builtin(__builtin_amdgcn_permlane_idx_gen)

// ---- ballot (wave32: bit i of result set iff lane i passed true) -----------
#if FPSAN_TEST_FORCE_WAVE_SIZE == 64
__global__ void k_ballot(const int* pred, std::uint64_t* mask)
{
    const int     lane = threadIdx.x;
    std::uint64_t m    = fpsan::amdgcn_ballot_w64(pred[lane] != 0);
    if(lane == 0)
        *mask = m;
}

TEST(Xlane, BallotW64)
{
    int ndev = 0;
    if(hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
        GTEST_SKIP() << "no HIP device";
    std::vector<int> pred(LANES);
    for(int i = 0; i < LANES; ++i)
        pred[i] = (i % 2 == 0) ? 1 : 0;
    int*           dPred;
    std::uint64_t* dMask;
    HIP_CHECK(hipMalloc(&dPred, LANES * sizeof(int)));
    HIP_CHECK(hipMalloc(&dMask, sizeof(std::uint64_t)));
    HIP_CHECK(hipMemcpy(dPred, pred.data(), LANES * sizeof(int), hipMemcpyHostToDevice));
    k_ballot<<<1, LANES>>>(dPred, dMask);
    HIP_CHECK(hipDeviceSynchronize());
    std::uint64_t mask = 0;
    HIP_CHECK(hipMemcpy(&mask, dMask, sizeof mask, hipMemcpyDeviceToHost));
    EXPECT_EQ(mask, 0x5555555555555555ull);
    (void)hipFree(dPred);
    (void)hipFree(dMask);
}
#else
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
#endif
