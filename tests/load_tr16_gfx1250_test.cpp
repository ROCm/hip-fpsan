// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/load_tr16_gfx1250_test.cpp
//
// GPU tests for the gfx1250 matrix-transposed load wrappers:
//   * fpsan/amdgcn_global_load.hpp : global_load_tr16_b128_{f16,bf16}
//   * fpsan/amdgcn_ds.hpp          : ds_load_tr16_b128_{f16,bf16}
//
// Both read 128 bits (8 x 16-bit) per lane with the WMMA cooperative-transpose
// lane mapping. They are thin, bit-faithful shims over the hardware transpose,
// so each test pins exactly that contract without knowing the permutation:
//   * MatchesBuiltin   : Float-mode wrapper bits == direct builtin bits over
//                        identically staged memory.
//   * FpsanMovesSameBits: FPSan-mode wrapper (memory staged with the same bit
//                        patterns as payloads) returns those same bits
//                        transposed -- Float and FPSan move bits identically
//                        (a transpose observes no values, only bits).
//
// Requires real gfx1250 hardware; built only under the gfx1250 tier.
#include "fpsan/amdgcn_ds.hpp"
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
static constexpr int         WAVE = 32; // gfx1250 wave32 transpose loads

__device__ __host__ inline std::uint16_t pat16(int lane, int slot)
{
    return static_cast<std::uint16_t>(0x3000 + lane * 8 + slot);
}

// ===================== global_load_tr16_b128 ================================
__global__ void k_g_raw_f16(const std::uint16_t* gmem, std::uint16_t* out)
{
    using v8fp16 = __fp16 __attribute__((ext_vector_type(8)));
    int    lane  = threadIdx.x;
    v8fp16 r{};
    (void)gmem;
#ifdef __HIP_DEVICE_COMPILE__
    r = __builtin_amdgcn_global_load_tr16_b128_v8f16(
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

template <Semantics S>
__global__ void k_g_wrap_f16(const std::uint16_t* gmem, std::uint16_t* out)
{
    using V   = Value<_Float16, S, kCC>;
    int  lane = threadIdx.x;
    auto r    = fpsan::amdgcn_global_load_tr16_b128_f16<S, kCC>(
        reinterpret_cast<const V*>(&gmem[lane * 8]));
    for(int s = 0; s < 8; ++s)
        out[lane * 8 + s] = r.get(s).to_storage_bits();
}

__global__ void k_g_raw_bf16(const std::uint16_t* gmem, std::uint16_t* out)
{
    using v8bf = __bf16 __attribute__((ext_vector_type(8)));
    int  lane  = threadIdx.x;
    v8bf r{};
    (void)gmem;
#ifdef __HIP_DEVICE_COMPILE__
    r = __builtin_amdgcn_global_load_tr16_b128_v8bf16(
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

template <Semantics S>
__global__ void k_g_wrap_bf16(const std::uint16_t* gmem, std::uint16_t* out)
{
    using V   = Value<__bf16, S, kCC>;
    int  lane = threadIdx.x;
    auto r    = fpsan::amdgcn_global_load_tr16_b128_bf16<S, kCC>(
        reinterpret_cast<const V*>(&gmem[lane * 8]));
    for(int s = 0; s < 8; ++s)
        out[lane * 8 + s] = r.get(s).to_storage_bits();
}

// ===================== ds_load_tr16_b128 (LDS) =============================
__global__ void k_d_raw_f16(const std::uint16_t* gmem, std::uint16_t* out)
{
    using v8fp16 = __fp16 __attribute__((ext_vector_type(8)));
    __shared__ std::uint16_t lds[WAVE * 8];
    int                      lane = threadIdx.x;
    for(int s = 0; s < 8; ++s)
        lds[lane * 8 + s] = gmem[lane * 8 + s];
    __syncthreads();
    v8fp16 r{};
#ifdef __HIP_DEVICE_COMPILE__
    r = __builtin_amdgcn_ds_load_tr16_b128_v8f16(
        (v8fp16 __attribute__((address_space(3)))*)(&lds[lane * 8]));
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

template <Semantics S>
__global__ void k_d_wrap_f16(const std::uint16_t* gmem, std::uint16_t* out)
{
    using V = Value<_Float16, S, kCC>;
    __shared__ std::uint16_t lds[WAVE * 8];
    int                      lane = threadIdx.x;
    for(int s = 0; s < 8; ++s)
        lds[lane * 8 + s] = gmem[lane * 8 + s];
    __syncthreads();
    auto r
        = fpsan::amdgcn_ds_load_tr16_b128_f16<S, kCC>(reinterpret_cast<const V*>(&lds[lane * 8]));
    for(int s = 0; s < 8; ++s)
        out[lane * 8 + s] = r.get(s).to_storage_bits();
}

__global__ void k_d_raw_bf16(const std::uint16_t* gmem, std::uint16_t* out)
{
    using v8bf = __bf16 __attribute__((ext_vector_type(8)));
    __shared__ std::uint16_t lds[WAVE * 8];
    int                      lane = threadIdx.x;
    for(int s = 0; s < 8; ++s)
        lds[lane * 8 + s] = gmem[lane * 8 + s];
    __syncthreads();
    v8bf r{};
#ifdef __HIP_DEVICE_COMPILE__
    r = __builtin_amdgcn_ds_load_tr16_b128_v8bf16(
        (v8bf __attribute__((address_space(3)))*)(&lds[lane * 8]));
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

template <Semantics S>
__global__ void k_d_wrap_bf16(const std::uint16_t* gmem, std::uint16_t* out)
{
    using V = Value<__bf16, S, kCC>;
    __shared__ std::uint16_t lds[WAVE * 8];
    int                      lane = threadIdx.x;
    for(int s = 0; s < 8; ++s)
        lds[lane * 8 + s] = gmem[lane * 8 + s];
    __syncthreads();
    auto r
        = fpsan::amdgcn_ds_load_tr16_b128_bf16<S, kCC>(reinterpret_cast<const V*>(&lds[lane * 8]));
    for(int s = 0; s < 8; ++s)
        out[lane * 8 + s] = r.get(s).to_storage_bits();
}

namespace
{
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

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_global_load_tr16_b128_v8f16)
TEST(GlobalLoadTr16Gfx1250, F16_MatchesBuiltinAndFpsanMovesSameBits)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    auto raw = run(k_g_raw_f16);
    auto flt = run(k_g_wrap_f16<Semantics::Native>);
    auto fps = run(k_g_wrap_f16<Semantics::Triton>);
    for(size_t i = 0; i < raw.size(); ++i)
    {
        EXPECT_EQ(flt[i], raw[i]) << "Float wrapper != builtin at " << i;
        EXPECT_EQ(fps[i], raw[i]) << "FPSan wrapper != builtin at " << i;
    }
}

TEST(GlobalLoadTr16Gfx1250, BF16_MatchesBuiltinAndFpsanMovesSameBits)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    auto raw = run(k_g_raw_bf16);
    auto flt = run(k_g_wrap_bf16<Semantics::Native>);
    auto fps = run(k_g_wrap_bf16<Semantics::Triton>);
    for(size_t i = 0; i < raw.size(); ++i)
    {
        EXPECT_EQ(flt[i], raw[i]) << "Float wrapper != builtin at " << i;
        EXPECT_EQ(fps[i], raw[i]) << "FPSan wrapper != builtin at " << i;
    }
}
#endif

// ===================== gfx1250 tr8 / tr4 / tr6 (fp8 / fp4 / fp6) ============
// Same bit-faithful contract as the tr16 tests above, for the byte- and
// sub-byte-granular gfx1250 transpose loads. Self-staging kernels (no input
// buffer) for the LDS forms; the global tr8 form takes a staged global buffer.

// ---- ds_load_tr8_b64 (fp8): raw builtin vs wrapper ------------------------
__global__ void k_d_raw_tr8(std::uint8_t* out)
{
    using v2i32 = int __attribute__((ext_vector_type(2)));
    __shared__ std::uint8_t lds[WAVE * 8];
    int                     lane = threadIdx.x;
    for(int s = 0; s < 8; ++s)
        lds[lane * 8 + s] = static_cast<std::uint8_t>(0x10 + lane + s);
    __syncthreads();
    v2i32 r{};
#ifdef __HIP_DEVICE_COMPILE__
    r = __builtin_amdgcn_ds_load_tr8_b64_v2i32(
        (v2i32 __attribute__((address_space(3)))*)(&lds[lane * 8]));
#endif
    __builtin_memcpy(out + lane * 8, &r, 8);
}

template <Semantics S>
__global__ void k_d_wrap_tr8(std::uint8_t* out)
{
    using V = Value<fpsan::fp8_e4m3, S, kCC>;
    __shared__ std::uint8_t lds[WAVE * 8];
    int                     lane = threadIdx.x;
    for(int s = 0; s < 8; ++s)
        lds[lane * 8 + s] = static_cast<std::uint8_t>(0x10 + lane + s);
    __syncthreads();
    auto r = fpsan::amdgcn_ds_load_tr8_b64_fp8<S, kCC>(reinterpret_cast<const V*>(&lds[lane * 8]));
    for(int s = 0; s < 8; ++s)
        out[lane * 8 + s] = r.get(s).to_storage_bits();
}

// ---- global_load_tr8_b64 (fp8): raw builtin vs wrapper --------------------
__global__ void k_g_raw_tr8(const std::uint8_t* gmem, std::uint8_t* out)
{
    using v2i32 = int __attribute__((ext_vector_type(2)));
    int   lane  = threadIdx.x;
    v2i32 r{};
    (void)gmem;
#ifdef __HIP_DEVICE_COMPILE__
    r = __builtin_amdgcn_global_load_tr8_b64_v2i32(
        (v2i32 __attribute__((address_space(1)))*)(&gmem[lane * 8]));
#endif
    __builtin_memcpy(out + lane * 8, &r, 8);
}

template <Semantics S>
__global__ void k_g_wrap_tr8(const std::uint8_t* gmem, std::uint8_t* out)
{
    using V   = Value<fpsan::fp8_e4m3, S, kCC>;
    int  lane = threadIdx.x;
    auto r    = fpsan::amdgcn_global_load_tr8_b64_fp8<S, kCC>(
        reinterpret_cast<const V*>(&gmem[lane * 8]));
    for(int s = 0; s < 8; ++s)
        out[lane * 8 + s] = r.get(s).to_storage_bits();
}

// ---- ds_load_tr4_b64 (fp4 packed dwords): raw vs wrapper ------------------
__global__ void k_d_raw_tr4(std::uint32_t* out)
{
    using v2i32 = int __attribute__((ext_vector_type(2)));
    __shared__ std::uint32_t lds[WAVE * 2];
    int                      lane = threadIdx.x;
    lds[lane * 2 + 0]             = 0x12345670u + lane;
    lds[lane * 2 + 1]             = 0x89abcdefu - lane;
    __syncthreads();
    v2i32 r{};
#ifdef __HIP_DEVICE_COMPILE__
    r = __builtin_amdgcn_ds_load_tr4_b64_v2i32(
        (v2i32 __attribute__((address_space(3)))*)(&lds[lane * 2]));
#endif
    out[lane * 2 + 0] = static_cast<std::uint32_t>(r[0]);
    out[lane * 2 + 1] = static_cast<std::uint32_t>(r[1]);
}

template <Semantics S>
__global__ void k_d_wrap_tr4(std::uint32_t* out)
{
    __shared__ std::uint32_t lds[WAVE * 2];
    int                      lane = threadIdx.x;
    lds[lane * 2 + 0]             = 0x12345670u + lane;
    lds[lane * 2 + 1]             = 0x89abcdefu - lane;
    __syncthreads();
    auto r            = fpsan::amdgcn_ds_load_tr4_b64<S, kCC>(&lds[lane * 2]);
    out[lane * 2 + 0] = r[0];
    out[lane * 2 + 1] = r[1];
}

// ---- ds_load_tr6_b96 (fp6/bf6 packed dwords): raw vs wrapper --------------
__global__ void k_d_raw_tr6(std::uint32_t* out)
{
    using v3i32 = int __attribute__((ext_vector_type(3)));
    __shared__ std::uint32_t lds[WAVE * 3];
    int                      lane = threadIdx.x;
    lds[lane * 3 + 0]             = 0x0f1e2d3cu + lane;
    lds[lane * 3 + 1]             = 0x4b5a6978u - lane;
    lds[lane * 3 + 2]             = 0x8c7d6e5fu + lane * 3;
    __syncthreads();
    v3i32 r{};
#ifdef __HIP_DEVICE_COMPILE__
    r = __builtin_amdgcn_ds_load_tr6_b96_v3i32(
        (v3i32 __attribute__((address_space(3)))*)(&lds[lane * 3]));
#endif
    out[lane * 3 + 0] = static_cast<std::uint32_t>(r[0]);
    out[lane * 3 + 1] = static_cast<std::uint32_t>(r[1]);
    out[lane * 3 + 2] = static_cast<std::uint32_t>(r[2]);
}

template <Semantics S>
__global__ void k_d_wrap_tr6(std::uint32_t* out)
{
    __shared__ std::uint32_t lds[WAVE * 3];
    int                      lane = threadIdx.x;
    lds[lane * 3 + 0]             = 0x0f1e2d3cu + lane;
    lds[lane * 3 + 1]             = 0x4b5a6978u - lane;
    lds[lane * 3 + 2]             = 0x8c7d6e5fu + lane * 3;
    __syncthreads();
    auto r            = fpsan::amdgcn_ds_load_tr6_b96<S, kCC>(&lds[lane * 3]);
    out[lane * 3 + 0] = r[0];
    out[lane * 3 + 1] = r[1];
    out[lane * 3 + 2] = r[2];
}

namespace
{
    template <class T>
    std::vector<T> run_self(void (*k)(T*), int n)
    {
        T* d = nullptr;
        (void)hipMalloc(&d, n * sizeof(T));
        k<<<1, WAVE>>>(d);
        (void)hipDeviceSynchronize();
        std::vector<T> h(n);
        (void)hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost);
        (void)hipFree(d);
        return h;
    }
    std::vector<std::uint8_t> run_g8(void (*k)(const std::uint8_t*, std::uint8_t*))
    {
        const int     N     = WAVE * 8;
        std::uint8_t* d_in  = nullptr;
        std::uint8_t* d_out = nullptr;
        (void)hipMalloc(&d_in, N);
        (void)hipMalloc(&d_out, N);
        std::vector<std::uint8_t> h_in(N);
        for(int lane = 0; lane < WAVE; ++lane)
            for(int s = 0; s < 8; ++s)
                h_in[lane * 8 + s] = static_cast<std::uint8_t>(0x10 + lane + s);
        (void)hipMemcpy(d_in, h_in.data(), N, hipMemcpyHostToDevice);
        k<<<1, WAVE>>>(d_in, d_out);
        (void)hipDeviceSynchronize();
        std::vector<std::uint8_t> h_out(N);
        (void)hipMemcpy(h_out.data(), d_out, N, hipMemcpyDeviceToHost);
        (void)hipFree(d_in);
        (void)hipFree(d_out);
        return h_out;
    }
} // namespace

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_ds_load_tr8_b64_v2i32)
TEST(DsLoadTr8Gfx1250, FP8_MatchesBuiltinAndFpsanMovesSameBits)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    const int N   = WAVE * 8;
    auto      raw = run_self<std::uint8_t>(k_d_raw_tr8, N);
    auto      flt = run_self<std::uint8_t>(k_d_wrap_tr8<Semantics::Native>, N);
    auto      fps = run_self<std::uint8_t>(k_d_wrap_tr8<Semantics::Triton>, N);
    for(int i = 0; i < N; ++i)
    {
        EXPECT_EQ(flt[i], raw[i]) << "Float wrapper != builtin at " << i;
        EXPECT_EQ(fps[i], raw[i]) << "FPSan wrapper != builtin at " << i;
    }
}
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_global_load_tr8_b64_v2i32)
TEST(GlobalLoadTr8Gfx1250, FP8_MatchesBuiltinAndFpsanMovesSameBits)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    const int N   = WAVE * 8;
    auto      raw = run_g8(k_g_raw_tr8);
    auto      flt = run_g8(k_g_wrap_tr8<Semantics::Native>);
    auto      fps = run_g8(k_g_wrap_tr8<Semantics::Triton>);
    for(int i = 0; i < N; ++i)
    {
        EXPECT_EQ(flt[i], raw[i]) << "Float wrapper != builtin at " << i;
        EXPECT_EQ(fps[i], raw[i]) << "FPSan wrapper != builtin at " << i;
    }
}
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_ds_load_tr4_b64_v2i32)
TEST(DsLoadTr4Gfx1250, FP4_MatchesBuiltinAndFpsanMovesSameBits)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    const int N   = WAVE * 2;
    auto      raw = run_self<std::uint32_t>(k_d_raw_tr4, N);
    auto      flt = run_self<std::uint32_t>(k_d_wrap_tr4<Semantics::Native>, N);
    auto      fps = run_self<std::uint32_t>(k_d_wrap_tr4<Semantics::Triton>, N);
    for(int i = 0; i < N; ++i)
    {
        EXPECT_EQ(flt[i], raw[i]) << "Float wrapper != builtin at " << i;
        EXPECT_EQ(fps[i], raw[i]) << "FPSan wrapper != builtin at " << i;
    }
}
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_ds_load_tr6_b96_v3i32)
TEST(DsLoadTr6Gfx1250, FP6_MatchesBuiltinAndFpsanMovesSameBits)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    const int N   = WAVE * 3;
    auto      raw = run_self<std::uint32_t>(k_d_raw_tr6, N);
    auto      flt = run_self<std::uint32_t>(k_d_wrap_tr6<Semantics::Native>, N);
    auto      fps = run_self<std::uint32_t>(k_d_wrap_tr6<Semantics::Triton>, N);
    for(int i = 0; i < N; ++i)
    {
        EXPECT_EQ(flt[i], raw[i]) << "Float wrapper != builtin at " << i;
        EXPECT_EQ(fps[i], raw[i]) << "FPSan wrapper != builtin at " << i;
    }
}
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_ds_load_tr16_b128_v8f16)
TEST(DsLoadTr16Gfx1250, F16_MatchesBuiltinAndFpsanMovesSameBits)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    auto raw = run(k_d_raw_f16);
    auto flt = run(k_d_wrap_f16<Semantics::Native>);
    auto fps = run(k_d_wrap_f16<Semantics::Triton>);
    for(size_t i = 0; i < raw.size(); ++i)
    {
        EXPECT_EQ(flt[i], raw[i]) << "Float wrapper != builtin at " << i;
        EXPECT_EQ(fps[i], raw[i]) << "FPSan wrapper != builtin at " << i;
    }
}

TEST(DsLoadTr16Gfx1250, BF16_MatchesBuiltinAndFpsanMovesSameBits)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    auto raw = run(k_d_raw_bf16);
    auto flt = run(k_d_wrap_bf16<Semantics::Native>);
    auto fps = run(k_d_wrap_bf16<Semantics::Triton>);
    for(size_t i = 0; i < raw.size(); ++i)
    {
        EXPECT_EQ(flt[i], raw[i]) << "Float wrapper != builtin at " << i;
        EXPECT_EQ(fps[i], raw[i]) << "FPSan wrapper != builtin at " << i;
    }
}
#endif
