// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/global_load_tr_test.cpp
//
// GPU tests for the gfx12 (RDNA4) matrix-transposed global-load wrappers in
// fpsan/amdgcn_global_load.hpp (global_load_tr_b128_{f16,bf16}).
//
// These wrappers are thin, bit-faithful shims over the hardware transpose, so
// the test pins exactly that contract without needing to know the transpose
// pattern itself:
//
//   * MatchesBuiltin: the Float-mode wrapper returns the same bits as a direct
//     __builtin_amdgcn_global_load_tr_b128_v8{f16,bf16} call over identically-
//     staged global memory.
//   * FpsanMovesSameBits: the FPSan-mode wrapper, given memory staged with the
//     same bit patterns (as payloads), returns those same bits transposed --
//     i.e. Float and FPSan move bits identically (the whole point: a transpose
//     observes no values, only bits).
//
// Requires real gfx1201 (or later gfx12) hardware; built only under
// FPSAN_ENABLE_HIP with a gfx12 architecture.
#include "fpsan/amdgcn_global_load.hpp"
#include "fpsan/fpsan.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

#define HIP_CHECK(e)                                        \
    do                                                      \
    {                                                       \
        hipError_t e_ = (e);                                \
        ASSERT_EQ(e_, hipSuccess) << hipGetErrorString(e_); \
    } while(0)

static constexpr Conversions kCC  = Conversions::Explicit;
static constexpr int         WAVE = 32; // gfx12 default wavefront size

// Deterministic distinct per-(lane,slot) 16-bit pattern. Avoids 0 and stays in
// a range that is a normal f16 bit pattern (doesn't matter for bit movement,
// but keeps the staged values unsurprising).
__device__ __host__ inline std::uint16_t pat16(int lane, int slot)
{
    return static_cast<std::uint16_t>(0x3000 + lane * 8 + slot);
}

// ---- raw builtin reference for f16 -----------------------------------------
__global__ void k_loadtr_raw_f16(const std::uint16_t* gmem, std::uint16_t* out)
{
    using v8fp16 = __fp16 __attribute__((ext_vector_type(8)));
    int lane     = threadIdx.x;
    (void)gmem;
    v8fp16 r{};
#ifdef __HIP_DEVICE_COMPILE__
    r = __builtin_amdgcn_global_load_tr_b128_v8f16(
        (v8fp16 __attribute__((address_space(1)))*)(&gmem[lane * 8]));
#endif
    union
    {
        v8fp16        v;
        std::uint16_t u[8];
    } u;
    u.v = r;
    for(int s = 0; s < 8; ++s)
        out[lane * 8 + s] = u.u[s];
}

// ---- fpsan wrapper for f16, both semantics ---------------------------------
template <Semantics S>
__global__ void k_loadtr_wrap_f16(const std::uint16_t* gmem, std::uint16_t* out)
{
    using V   = Value<_Float16, S, kCC>;
    int  lane = threadIdx.x;
    auto r    = fpsan::amdgcn_global_load_tr_b128_f16<S, kCC>(
        reinterpret_cast<const V*>(&gmem[lane * 8]));
    for(int s = 0; s < 8; ++s)
        out[lane * 8 + s] = r.get(s).to_storage_bits();
}

// ---- raw builtin reference for bf16 ----------------------------------------
__global__ void k_loadtr_raw_bf16(const std::uint16_t* gmem, std::uint16_t* out)
{
    using v8bf = __bf16 __attribute__((ext_vector_type(8)));
    int lane   = threadIdx.x;
    (void)gmem;
    v8bf r{};
#ifdef __HIP_DEVICE_COMPILE__
    r = __builtin_amdgcn_global_load_tr_b128_v8bf16(
        (v8bf __attribute__((address_space(1)))*)(&gmem[lane * 8]));
#endif
    union
    {
        v8bf          v;
        std::uint16_t u[8];
    } u;
    u.v = r;
    for(int s = 0; s < 8; ++s)
        out[lane * 8 + s] = u.u[s];
}

// ---- fpsan wrapper for bf16, both semantics --------------------------------
template <Semantics S>
__global__ void k_loadtr_wrap_bf16(const std::uint16_t* gmem, std::uint16_t* out)
{
    using V   = Value<__bf16, S, kCC>;
    int  lane = threadIdx.x;
    auto r    = fpsan::amdgcn_global_load_tr_b128_bf16<S, kCC>(
        reinterpret_cast<const V*>(&gmem[lane * 8]));
    for(int s = 0; s < 8; ++s)
        out[lane * 8 + s] = r.get(s).to_storage_bits();
}

namespace
{
    // Stage a wave-worth of distinct 16-bit patterns in global memory, then run
    // `k(gmem, out)`. Returns out[].
    std::vector<std::uint16_t> run(void (*k)(const std::uint16_t*, std::uint16_t*))
    {
        const int      N     = WAVE * 8;
        std::uint16_t* d_in  = nullptr;
        std::uint16_t* d_out = nullptr;
        (void)hipMalloc(&d_in, N * sizeof(std::uint16_t));
        (void)hipMalloc(&d_out, N * sizeof(std::uint16_t));
        std::vector<std::uint16_t> h_in(N);
        for(int lane = 0; lane < WAVE; ++lane)
            for(int s = 0; s < 8; ++s)
                h_in[lane * 8 + s] = pat16(lane, s);
        (void)hipMemcpy(d_in, h_in.data(), N * sizeof(std::uint16_t), hipMemcpyHostToDevice);
        k<<<1, WAVE>>>(d_in, d_out);
        (void)hipDeviceSynchronize();
        std::vector<std::uint16_t> h_out(N);
        (void)hipMemcpy(h_out.data(), d_out, N * sizeof(std::uint16_t), hipMemcpyDeviceToHost);
        (void)hipFree(d_in);
        (void)hipFree(d_out);
        return h_out;
    }
    bool have_device()
    {
        int n = 0;
        return hipGetDeviceCount(&n) == hipSuccess && n > 0;
    }
} // namespace

TEST(GlobalLoadTr, F16_MatchesBuiltinAndFpsanMovesSameBits)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    auto raw = run(k_loadtr_raw_f16);
    auto flt = run(k_loadtr_wrap_f16<Semantics::Float>);
    auto fps = run(k_loadtr_wrap_f16<Semantics::FPSan>);
    for(size_t i = 0; i < raw.size(); ++i)
    {
        EXPECT_EQ(flt[i], raw[i]) << "Float wrapper != builtin at " << i;
        EXPECT_EQ(fps[i], raw[i]) << "FPSan wrapper != builtin at " << i;
    }
}

TEST(GlobalLoadTr, BF16_MatchesBuiltinAndFpsanMovesSameBits)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    auto raw = run(k_loadtr_raw_bf16);
    auto flt = run(k_loadtr_wrap_bf16<Semantics::Float>);
    auto fps = run(k_loadtr_wrap_bf16<Semantics::FPSan>);
    for(size_t i = 0; i < raw.size(); ++i)
    {
        EXPECT_EQ(flt[i], raw[i]) << "Float wrapper != builtin at " << i;
        EXPECT_EQ(fps[i], raw[i]) << "FPSan wrapper != builtin at " << i;
    }
}
