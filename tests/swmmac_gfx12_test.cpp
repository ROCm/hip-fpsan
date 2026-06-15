// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/swmmac_gfx12_test.cpp
//
// Silicon tests for the gfx12 (RDNA4) sparse WMMA wrappers in
// fpsan/amdgcn_swmmac_gfx12.hpp. Three layers:
//
//   * *FloatMatchesBuiltin: the wrapper in Float mode is a bit-exact
//     pass-through to the underlying __builtin_amdgcn_swmmac_*.
//   * *LayoutMatchesHardware: the software dataflow in Float mode produces
//     the same D fragment as the hardware builtin, byte-for-byte. This is
//     the load-bearing layout proof -- if A/B/D/idx-lane mapping or the
//     sparse-K selection were off by one, this fails.
//   * *FpsanMatchesScalarReference: the dataflow in FPSan mode (payload ring)
//     matches a host-side scalar reference doing the equivalent sparse MAC
//     with the same FPSan scalar Value type. Verifies that the per-lane
//     payload algebra is plumbed correctly through the wave-cooperative
//     dataflow.
//
// Requires real gfx1201 hardware; built only under FPSAN_ENABLE_HIP with a
// gfx12 architecture.
#include "fpsan/amdgcn_swmmac_gfx12.hpp"
#include "fpsan/fpsan.hpp"
#include "test_random.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <random>
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

static constexpr Conversions kCC = Conversions::Explicit;

// ---- f16 SWMMAC: side-by-side raw vs wrapper, bit-exact comparison ---------
__global__ void k_swmmac_f16_pair(const std::uint16_t* a_in,
                                  const std::uint16_t* b_in,
                                  const float*         c_in,
                                  std::uint16_t        idx,
                                  float*               raw_out,
                                  float*               wrap_out)
{
    using v8h  = _Float16 __attribute__((ext_vector_type(8)));
    using v16h = _Float16 __attribute__((ext_vector_type(16)));
    using v8f  = float __attribute__((ext_vector_type(8)));
    int lane   = threadIdx.x;

    // Reconstruct lane-private A/B/C from staged bit patterns.
    v8h  a;
    v16h b;
    v8f  c;
    for(int s = 0; s < 8; ++s)
    {
        std::uint16_t u = a_in[lane * 8 + s];
        _Float16      v;
        __builtin_memcpy(&v, &u, sizeof v);
        a[s] = v;
    }
    for(int s = 0; s < 16; ++s)
    {
        std::uint16_t u = b_in[lane * 16 + s];
        _Float16      v;
        __builtin_memcpy(&v, &u, sizeof v);
        b[s] = v;
    }
    for(int s = 0; s < 8; ++s)
        c[s] = c_in[lane * 8 + s];

    // --- direct builtin ---
    v8f raw{};
#ifdef __HIP_DEVICE_COMPILE__
    raw = __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(a, b, c, static_cast<short>(idx));
#endif
    for(int s = 0; s < 8; ++s)
        raw_out[lane * 8 + s] = raw[s];

    // --- via fpsan wrapper (Float mode) ---
    Value<fpsan::v8h_native, Semantics::Native, kCC>         av{a};
    Value<fpsan::v16h_swmmac_native, Semantics::Native, kCC> bv{b};
    Value<fpsan::v8f_native, Semantics::Native, kCC>         cv{c};
    auto dv = fpsan::amdgcn_swmmac_f32_16x16x32_f16_w32<Semantics::Native, kCC>(av, bv, cv, idx);
    fpsan::v8f_native d = static_cast<fpsan::v8f_native>(dv);
    for(int s = 0; s < 8; ++s)
        wrap_out[lane * 8 + s] = d[s];
}

// ---- bf16 SWMMAC ----------------------------------------------------------
__global__ void k_swmmac_bf16_pair(const std::uint16_t* a_in,
                                   const std::uint16_t* b_in,
                                   const float*         c_in,
                                   std::uint16_t        idx,
                                   float*               raw_out,
                                   float*               wrap_out)
{
    using v8bf  = __bf16 __attribute__((ext_vector_type(8)));
    using v16bf = __bf16 __attribute__((ext_vector_type(16)));
    using v8f   = float __attribute__((ext_vector_type(8)));
    int   lane  = threadIdx.x;
    v8bf  a;
    v16bf b;
    v8f   c;
    for(int s = 0; s < 8; ++s)
    {
        std::uint16_t u = a_in[lane * 8 + s];
        __bf16        v;
        __builtin_memcpy(&v, &u, sizeof v);
        a[s] = v;
    }
    for(int s = 0; s < 16; ++s)
    {
        std::uint16_t u = b_in[lane * 16 + s];
        __bf16        v;
        __builtin_memcpy(&v, &u, sizeof v);
        b[s] = v;
    }
    for(int s = 0; s < 8; ++s)
        c[s] = c_in[lane * 8 + s];
    v8f raw{};
#ifdef __HIP_DEVICE_COMPILE__
    raw = __builtin_amdgcn_swmmac_f32_16x16x32_bf16_w32(a, b, c, static_cast<short>(idx));
#endif
    for(int s = 0; s < 8; ++s)
        raw_out[lane * 8 + s] = raw[s];
    Value<fpsan::v8bf_native, Semantics::Native, kCC>         av{a};
    Value<fpsan::v16bf_swmmac_native, Semantics::Native, kCC> bv{b};
    Value<fpsan::v8f_native, Semantics::Native, kCC>          cv{c};
    auto dv = fpsan::amdgcn_swmmac_f32_16x16x32_bf16_w32<Semantics::Native, kCC>(av, bv, cv, idx);
    fpsan::v8f_native d = static_cast<fpsan::v8f_native>(dv);
    for(int s = 0; s < 8; ++s)
        wrap_out[lane * 8 + s] = d[s];
}

// ---- f16 -> f16 SWMMAC ----------------------------------------------------
__global__ void k_swmmac_f16h_pair(const std::uint16_t* a_in,
                                   const std::uint16_t* b_in,
                                   const std::uint16_t* c_in,
                                   std::uint16_t        idx,
                                   std::uint16_t*       raw_out,
                                   std::uint16_t*       wrap_out)
{
    using v8h  = _Float16 __attribute__((ext_vector_type(8)));
    using v16h = _Float16 __attribute__((ext_vector_type(16)));
    int  lane  = threadIdx.x;
    v8h  a, c;
    v16h b;
    auto load_h = [](std::uint16_t u) {
        _Float16 v;
        __builtin_memcpy(&v, &u, sizeof v);
        return v;
    };
    for(int s = 0; s < 8; ++s)
        a[s] = load_h(a_in[lane * 8 + s]);
    for(int s = 0; s < 16; ++s)
        b[s] = load_h(b_in[lane * 16 + s]);
    for(int s = 0; s < 8; ++s)
        c[s] = load_h(c_in[lane * 8 + s]);
    v8h raw{};
#ifdef __HIP_DEVICE_COMPILE__
    raw = __builtin_amdgcn_swmmac_f16_16x16x32_f16_w32(a, b, c, static_cast<short>(idx));
#endif
    auto store_h = [](_Float16 v) {
        std::uint16_t u;
        __builtin_memcpy(&u, &v, sizeof u);
        return u;
    };
    for(int s = 0; s < 8; ++s)
        raw_out[lane * 8 + s] = store_h(raw[s]);
    Value<fpsan::v8h_native, Semantics::Native, kCC>         av{a}, cv{c};
    Value<fpsan::v16h_swmmac_native, Semantics::Native, kCC> bv{b};
    auto dv = fpsan::amdgcn_swmmac_f16_16x16x32_f16_w32<Semantics::Native, kCC>(av, bv, cv, idx);
    fpsan::v8h_native d = static_cast<fpsan::v8h_native>(dv);
    for(int s = 0; s < 8; ++s)
        wrap_out[lane * 8 + s] = store_h(d[s]);
}

// ---- FP8/BF8 SWMMAC kernel template -- 8 fp8 bytes / lane for A, 16 for B,
// v8f for C/D. Pre-/post-byte serialization via memcpy keeps everything
// portable.
template <int VARIANT> // 0=fp8/fp8, 1=fp8/bf8, 2=bf8/fp8, 3=bf8/bf8
__global__ void k_swmmac_fp8_pair(const std::uint8_t* a_in,
                                  const std::uint8_t* b_in,
                                  const float*        c_in,
                                  std::uint16_t       idx,
                                  float*              raw_out,
                                  float*              wrap_out)
{
    using v2i = int __attribute__((ext_vector_type(2)));
    using v4i = int __attribute__((ext_vector_type(4)));
    using v8f = float __attribute__((ext_vector_type(8)));
    int lane  = threadIdx.x;
    v2i a;
    v4i b;
    v8f c;
    __builtin_memcpy(&a, a_in + lane * 8, 8);
    __builtin_memcpy(&b, b_in + lane * 16, 16);
    for(int s = 0; s < 8; ++s)
        c[s] = c_in[lane * 8 + s];
    v8f raw{};
#ifdef __HIP_DEVICE_COMPILE__
    if constexpr(VARIANT == 0)
        raw = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a, b, c, static_cast<short>(idx));
    else if constexpr(VARIANT == 1)
        raw = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w32(a, b, c, static_cast<short>(idx));
    else if constexpr(VARIANT == 2)
        raw = __builtin_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w32(a, b, c, static_cast<short>(idx));
    else
        raw = __builtin_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w32(a, b, c, static_cast<short>(idx));
#endif
    for(int s = 0; s < 8; ++s)
        raw_out[lane * 8 + s] = raw[s];

    Value<fpsan::v8f_native, Semantics::Native, kCC> cv{c};
    fpsan::v8f_native                                d{};
    if constexpr(VARIANT == 0)
    {
        Value<fpsan::v8e4m3_native, Semantics::Native, kCC> av{
            __builtin_bit_cast(fpsan::v8e4m3_native, a)};
        Value<fpsan::v16e4m3_swmmac_native, Semantics::Native, kCC> bv{
            __builtin_bit_cast(fpsan::v16e4m3_swmmac_native, b)};
        d = static_cast<fpsan::v8f_native>(
            fpsan::amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32<Semantics::Native, kCC>(av, bv, cv, idx));
    }
    else if constexpr(VARIANT == 1)
    {
        Value<fpsan::v8e4m3_native, Semantics::Native, kCC> av{
            __builtin_bit_cast(fpsan::v8e4m3_native, a)};
        Value<fpsan::v16e5m2_swmmac_native, Semantics::Native, kCC> bv{
            __builtin_bit_cast(fpsan::v16e5m2_swmmac_native, b)};
        d = static_cast<fpsan::v8f_native>(
            fpsan::amdgcn_swmmac_f32_16x16x32_fp8_bf8_w32<Semantics::Native, kCC>(av, bv, cv, idx));
    }
    else if constexpr(VARIANT == 2)
    {
        Value<fpsan::v8e5m2_native, Semantics::Native, kCC> av{
            __builtin_bit_cast(fpsan::v8e5m2_native, a)};
        Value<fpsan::v16e4m3_swmmac_native, Semantics::Native, kCC> bv{
            __builtin_bit_cast(fpsan::v16e4m3_swmmac_native, b)};
        d = static_cast<fpsan::v8f_native>(
            fpsan::amdgcn_swmmac_f32_16x16x32_bf8_fp8_w32<Semantics::Native, kCC>(av, bv, cv, idx));
    }
    else
    {
        Value<fpsan::v8e5m2_native, Semantics::Native, kCC> av{
            __builtin_bit_cast(fpsan::v8e5m2_native, a)};
        Value<fpsan::v16e5m2_swmmac_native, Semantics::Native, kCC> bv{
            __builtin_bit_cast(fpsan::v16e5m2_swmmac_native, b)};
        d = static_cast<fpsan::v8f_native>(
            fpsan::amdgcn_swmmac_f32_16x16x32_bf8_bf8_w32<Semantics::Native, kCC>(av, bv, cv, idx));
    }
    for(int s = 0; s < 8; ++s)
        wrap_out[lane * 8 + s] = d[s];
}

namespace
{
    constexpr int WAVE = 32;
    bool          have_device()
    {
        int n = 0;
        return hipGetDeviceCount(&n) == hipSuccess && n > 0;
    }
} // namespace

TEST(SwmmacGfx12, F32_F16_FloatMatchesBuiltin)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";

    // Generate small-range half-precision A/B and f32 C so the accumulated
    // products stay well within f32 precision -- but bit-exact equality is
    // what we're after either way.
    std::mt19937                          rng(0xa5'12'5c'7a);
    std::uniform_int_distribution<int>    di_h(0x3000, 0x4000); // ~ [0.5, 2.0)
    std::uniform_real_distribution<float> df(-1.f, 1.f);
    std::vector<std::uint16_t>            ha(WAVE * 8), hb(WAVE * 16);
    std::vector<float>                    hc(WAVE * 8);
    for(auto& x : ha)
        x = static_cast<std::uint16_t>(di_h(rng));
    for(auto& x : hb)
        x = static_cast<std::uint16_t>(di_h(rng));
    for(auto& x : hc)
        x = df(rng);

    std::uint16_t* dA    = nullptr;
    std::uint16_t* dB    = nullptr;
    float*         dC    = nullptr;
    float*         dRaw  = nullptr;
    float*         dWrap = nullptr;
    HIP_CHECK(hipMalloc(&dA, ha.size() * sizeof(std::uint16_t)));
    HIP_CHECK(hipMalloc(&dB, hb.size() * sizeof(std::uint16_t)));
    HIP_CHECK(hipMalloc(&dC, hc.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dRaw, WAVE * 8 * sizeof(float)));
    HIP_CHECK(hipMalloc(&dWrap, WAVE * 8 * sizeof(float)));
    HIP_CHECK(hipMemcpy(dA, ha.data(), ha.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hb.data(), hb.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dC, hc.data(), hc.size() * sizeof(float), hipMemcpyHostToDevice));

    // Random 16-bit index per launch.
    std::uniform_int_distribution<int> di16(0, 0xFFFF);
    std::uint16_t                      idx = static_cast<std::uint16_t>(di16(rng));

    k_swmmac_f16_pair<<<1, WAVE>>>(dA, dB, dC, idx, dRaw, dWrap);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<float> raw(WAVE * 8), wrap(WAVE * 8);
    HIP_CHECK(hipMemcpy(raw.data(), dRaw, raw.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(wrap.data(), dWrap, wrap.size() * sizeof(float), hipMemcpyDeviceToHost));
    for(size_t i = 0; i < raw.size(); ++i)
    {
        std::uint32_t br, bw;
        std::memcpy(&br, &raw[i], 4);
        std::memcpy(&bw, &wrap[i], 4);
        EXPECT_EQ(bw, br) << "lane " << (i / 8) << " slot " << (i % 8);
    }
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dRaw);
    (void)hipFree(dWrap);
}

TEST(SwmmacGfx12, F32_BF16_FloatMatchesBuiltin)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    std::mt19937                          rng(0xb5'13'5c'7b);
    std::uniform_int_distribution<int>    di(0x3F00, 0x4000); // ~ [0.5, 2.0) in bf16
    std::uniform_real_distribution<float> df(-1.f, 1.f);
    std::vector<std::uint16_t>            ha(WAVE * 8), hb(WAVE * 16);
    std::vector<float>                    hc(WAVE * 8);
    for(auto& x : ha)
        x = static_cast<std::uint16_t>(di(rng));
    for(auto& x : hb)
        x = static_cast<std::uint16_t>(di(rng));
    for(auto& x : hc)
        x = df(rng);
    std::uint16_t* dA    = nullptr;
    std::uint16_t* dB    = nullptr;
    float*         dC    = nullptr;
    float*         dRaw  = nullptr;
    float*         dWrap = nullptr;
    HIP_CHECK(hipMalloc(&dA, ha.size() * sizeof(std::uint16_t)));
    HIP_CHECK(hipMalloc(&dB, hb.size() * sizeof(std::uint16_t)));
    HIP_CHECK(hipMalloc(&dC, hc.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dRaw, WAVE * 8 * sizeof(float)));
    HIP_CHECK(hipMalloc(&dWrap, WAVE * 8 * sizeof(float)));
    HIP_CHECK(hipMemcpy(dA, ha.data(), ha.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hb.data(), hb.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dC, hc.data(), hc.size() * sizeof(float), hipMemcpyHostToDevice));
    std::uniform_int_distribution<int> di16(0, 0xFFFF);
    std::uint16_t                      idx = static_cast<std::uint16_t>(di16(rng));
    k_swmmac_bf16_pair<<<1, WAVE>>>(dA, dB, dC, idx, dRaw, dWrap);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> raw(WAVE * 8), wrap(WAVE * 8);
    HIP_CHECK(hipMemcpy(raw.data(), dRaw, raw.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(wrap.data(), dWrap, wrap.size() * sizeof(float), hipMemcpyDeviceToHost));
    for(size_t i = 0; i < raw.size(); ++i)
    {
        std::uint32_t br, bw;
        std::memcpy(&br, &raw[i], 4);
        std::memcpy(&bw, &wrap[i], 4);
        EXPECT_EQ(bw, br) << "lane " << (i / 8) << " slot " << (i % 8);
    }
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dRaw);
    (void)hipFree(dWrap);
}

TEST(SwmmacGfx12, F16_F16_FloatMatchesBuiltin)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    std::mt19937                       rng(0xc5'14'5c'7c);
    std::uniform_int_distribution<int> di(0x3000, 0x4000);
    std::vector<std::uint16_t>         ha(WAVE * 8), hb(WAVE * 16), hc(WAVE * 8);
    for(auto& x : ha)
        x = static_cast<std::uint16_t>(di(rng));
    for(auto& x : hb)
        x = static_cast<std::uint16_t>(di(rng));
    for(auto& x : hc)
        x = static_cast<std::uint16_t>(di(rng));
    std::uint16_t* dA    = nullptr;
    std::uint16_t* dB    = nullptr;
    std::uint16_t* dC    = nullptr;
    std::uint16_t* dRaw  = nullptr;
    std::uint16_t* dWrap = nullptr;
    HIP_CHECK(hipMalloc(&dA, ha.size() * sizeof(std::uint16_t)));
    HIP_CHECK(hipMalloc(&dB, hb.size() * sizeof(std::uint16_t)));
    HIP_CHECK(hipMalloc(&dC, hc.size() * sizeof(std::uint16_t)));
    HIP_CHECK(hipMalloc(&dRaw, WAVE * 8 * sizeof(std::uint16_t)));
    HIP_CHECK(hipMalloc(&dWrap, WAVE * 8 * sizeof(std::uint16_t)));
    HIP_CHECK(hipMemcpy(dA, ha.data(), ha.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hb.data(), hb.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dC, hc.data(), hc.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice));
    std::uniform_int_distribution<int> di16(0, 0xFFFF);
    std::uint16_t                      idx = static_cast<std::uint16_t>(di16(rng));
    k_swmmac_f16h_pair<<<1, WAVE>>>(dA, dB, dC, idx, dRaw, dWrap);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint16_t> raw(WAVE * 8), wrap(WAVE * 8);
    HIP_CHECK(
        hipMemcpy(raw.data(), dRaw, raw.size() * sizeof(std::uint16_t), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(wrap.data(), dWrap, wrap.size() * sizeof(std::uint16_t), hipMemcpyDeviceToHost));
    for(size_t i = 0; i < raw.size(); ++i)
        EXPECT_EQ(wrap[i], raw[i]) << "lane " << (i / 8) << " slot " << (i % 8);
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dRaw);
    (void)hipFree(dWrap);
}

template <int VARIANT>
static void run_fp8_swmmac_test(std::uint32_t seed)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    std::mt19937                          rng(seed);
    std::uniform_int_distribution<int>    di_byte(0, 0x7F); // positive fp8 bytes
    std::uniform_real_distribution<float> df(-1.f, 1.f);
    std::vector<std::uint8_t>             ha(WAVE * 8), hb(WAVE * 16);
    std::vector<float>                    hc(WAVE * 8);
    for(auto& x : ha)
        x = static_cast<std::uint8_t>(di_byte(rng));
    for(auto& x : hb)
        x = static_cast<std::uint8_t>(di_byte(rng));
    for(auto& x : hc)
        x = df(rng);
    std::uint8_t* dA    = nullptr;
    std::uint8_t* dB    = nullptr;
    float*        dC    = nullptr;
    float*        dRaw  = nullptr;
    float*        dWrap = nullptr;
    HIP_CHECK(hipMalloc(&dA, ha.size()));
    HIP_CHECK(hipMalloc(&dB, hb.size()));
    HIP_CHECK(hipMalloc(&dC, hc.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dRaw, WAVE * 8 * sizeof(float)));
    HIP_CHECK(hipMalloc(&dWrap, WAVE * 8 * sizeof(float)));
    HIP_CHECK(hipMemcpy(dA, ha.data(), ha.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hb.data(), hb.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dC, hc.data(), hc.size() * sizeof(float), hipMemcpyHostToDevice));
    std::uniform_int_distribution<int> di16(0, 0xFFFF);
    std::uint16_t                      idx = static_cast<std::uint16_t>(di16(rng));
    k_swmmac_fp8_pair<VARIANT><<<1, WAVE>>>(dA, dB, dC, idx, dRaw, dWrap);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> raw(WAVE * 8), wrap(WAVE * 8);
    HIP_CHECK(hipMemcpy(raw.data(), dRaw, raw.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(wrap.data(), dWrap, wrap.size() * sizeof(float), hipMemcpyDeviceToHost));
    for(size_t i = 0; i < raw.size(); ++i)
    {
        std::uint32_t br, bw;
        std::memcpy(&br, &raw[i], 4);
        std::memcpy(&bw, &wrap[i], 4);
        EXPECT_EQ(bw, br) << "lane " << (i / 8) << " slot " << (i % 8);
    }
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dRaw);
    (void)hipFree(dWrap);
}

TEST(SwmmacGfx12, F32_FP8_FP8_FloatMatchesBuiltin)
{
    run_fp8_swmmac_test<0>(0xfe'00);
}
TEST(SwmmacGfx12, F32_FP8_BF8_FloatMatchesBuiltin)
{
    run_fp8_swmmac_test<1>(0xfe'01);
}
TEST(SwmmacGfx12, F32_BF8_FP8_FloatMatchesBuiltin)
{
    run_fp8_swmmac_test<2>(0xfe'02);
}
TEST(SwmmacGfx12, F32_BF8_BF8_FloatMatchesBuiltin)
{
    run_fp8_swmmac_test<3>(0xfe'03);
}

// ============================================================================
// Layout / FPSan dataflow tests for f16, bf16, f16-out (shared layout).
// ============================================================================
//
// Common scaffolding: stage A in dense (M, K) form keeping only the 2 live K
// per group nonzero (per a per-(i,g) selector pair p0<p1), assemble per-lane
// A_comp fragments from the live values using the verified compression layout,
// assemble per-lane B fragments using the verified B layout, run both the
// hardware builtin and the wrapper (Float/FPSan), and compare to a host scalar
// reference computed directly from the sparse selectors.
namespace
{
    constexpr int kSwM = 16, kSwN = 16, kSwK = 32;

    struct SwData
    {
        std::vector<float>         A; // M x K dense, zero outside live positions
        std::vector<float>         B; // K x N
        std::vector<float>         C; // M x N
        std::vector<std::uint16_t> idx; // per-lane sparse index (WAVE)
        std::vector<int>           p0; // M*8: live K offset 0 per (row, group)
        std::vector<int>           p1; // M*8: live K offset 1 per (row, group)
    };

    // Build deterministic 2:4 sparse inputs. The encoding of per-lane idx
    // depends on which lane holds the compression nibble for (row i, group g)
    // -- different between f16 and fp8. encode_idx is provided by the caller.
    template <class EncodeIdxFn>
    SwData make_sw_data(std::uint32_t seed, EncodeIdxFn encode_idx)
    {
        SwData d;
        d.A.assign(kSwM * kSwK, 0.f);
        d.B.resize(kSwK * kSwN);
        d.C.resize(kSwM * kSwN);
        d.idx.assign(WAVE, 0);
        d.p0.resize(kSwM * 8);
        d.p1.resize(kSwM * 8);
        std::mt19937 rng = fpsan_test::make_rng();
        rng.discard(seed); // cheap per-test diversity, still deterministic
        for(auto& x : d.B)
            x = fpsan_test::pick_int_valued<float>(rng, -2, 2);
        for(auto& x : d.C)
            x = fpsan_test::pick_int_valued<float>(rng, -4, 4);
        for(int i = 0; i < kSwM; ++i)
            for(int g = 0; g < 8; ++g)
            {
                int a0 = fpsan_test::pick_int(rng, 0, 3);
                int a1 = fpsan_test::pick_int(rng, 0, 3);
                while(a1 == a0)
                    a1 = fpsan_test::pick_int(rng, 0, 3);
                if(a0 > a1)
                {
                    int t = a0;
                    a0    = a1;
                    a1    = t;
                }
                d.p0[i * 8 + g]            = a0;
                d.p1[i * 8 + g]            = a1;
                d.A[i * kSwK + 4 * g + a0] = fpsan_test::pick_int_valued<float>(rng, -3, 3);
                d.A[i * kSwK + 4 * g + a1] = fpsan_test::pick_int_valued<float>(rng, -3, 3);
            }
        for(int L = 0; L < WAVE; ++L)
            d.idx[L] = encode_idx(L, d.p0, d.p1);
        return d;
    }

    // Encode per-lane idx for the f16/bf16/f16-out shared layout.
    // Compression for (row i, group g) lives at lane i + 16*((g/2)%2), nibble
    // (g%2) within byte (g/4) -> bit_off = 4*(g%2) + 8*(g/4).
    //   * Lane L < 16 holds (i=L, g in {0,1,4,5}).
    //   * Lane L >= 16 holds (i=L-16, g in {2,3,6,7}).
    inline std::uint16_t encode_idx_h(int L, const std::vector<int>& p0, const std::vector<int>& p1)
    {
        const int     i    = L & 15;
        const int     high = L >> 4;
        std::uint16_t v    = 0;
        for(int g = 0; g < 8; ++g)
        {
            const int side = (g >> 1) & 1;
            if(side != high)
                continue;
            const int bit_off = 4 * (g & 1) + 8 * (g >> 2);
            const int field   = (p0[i * 8 + g] & 3) | ((p1[i * 8 + g] & 3) << 2);
            v |= static_cast<std::uint16_t>(field << bit_off);
        }
        return v;
    }

    // Encode per-lane idx for the fp8/bf8 layout.
    // Compression for (row i, group g) lives at lane i + 16*(g/4), nibble
    // (g%4) -> bit_off = 4*(g%4).
    inline std::uint16_t
        encode_idx_fp8(int L, const std::vector<int>& p0, const std::vector<int>& p1)
    {
        const int     i    = L & 15;
        const int     high = L >> 4;
        std::uint16_t v    = 0;
        for(int g = 0; g < 8; ++g)
        {
            const int side = (g >> 2) & 1;
            if(side != high)
                continue;
            const int bit_off = 4 * (g & 3);
            const int field   = (p0[i * 8 + g] & 3) | ((p1[i * 8 + g] & 3) << 2);
            v |= static_cast<std::uint16_t>(field << bit_off);
        }
        return v;
    }

    template <class T>
    T* to_dev(const std::vector<T>& h)
    {
        T* d = nullptr;
        (void)hipMalloc(&d, h.size() * sizeof(T));
        (void)hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice);
        return d;
    }

    template <class T>
    std::uint64_t bits_of(T v)
    {
        std::uint64_t u = 0;
        std::memcpy(&u, &v, sizeof v);
        return u;
    }
} // namespace

// ---- shared f16 / bf16 / f16-out kernels ----------------------------------
// Stage per-lane fragments from dense A. The (lane, slot) -> (i, k) mapping is
// the layout we are verifying; building the fragment from A_dense means a
// layout bug shows up as a hardware vs dataflow mismatch.
//
// Lane L holds row i = L%16. The 8 v8 slots map to dense K via:
//   side = L/16  (0 or 1)
//   For each g in 0..7 with ((g/2)%2) == side:
//     a_gpr  = 2*(g/4) + (g%2)
//     slot 0 = 2*a_gpr     -> A_dense[i][4g + p0(i,g)]
//     slot 1 = 2*a_gpr + 1 -> A_dense[i][4g + p1(i,g)]
template <Semantics S, class AScalar, class BScalar, class CScalar, class Out, class WrapFn>
__device__ inline void swmmac_h_dataflow(const float*         Adense,
                                         const float*         B,
                                         const float*         C,
                                         const int*           p0,
                                         const int*           p1,
                                         const std::uint16_t* idx,
                                         Out*                 D,
                                         WrapFn               fn)
{
    using AV       = AScalar __attribute__((ext_vector_type(8)));
    using BV       = BScalar __attribute__((ext_vector_type(16)));
    using CV       = CScalar __attribute__((ext_vector_type(8)));
    const int lane = threadIdx.x;
    const int i    = lane & 15;
    const int side = lane >> 4;

    AV an{};
    for(int g = 0; g < 8; ++g)
    {
        if(((g >> 1) & 1) != side)
            continue;
        const int a_gpr = 2 * (g >> 2) + (g & 1);
        const int slot0 = 2 * a_gpr;
        an[slot0]       = static_cast<AScalar>(Adense[i * kSwK + 4 * g + p0[i * 8 + g]]);
        an[slot0 + 1]   = static_cast<AScalar>(Adense[i * kSwK + 4 * g + p1[i * 8 + g]]);
    }
    BV bn{};
    // Lane L holds (j = L%16, side = L/16). For each k with ((k/8)%2) == side:
    //   slot = 8*(k/16) + 2*((k/2)%4) + (k%2)
    const int j = lane & 15;
    for(int k = 0; k < kSwK; ++k)
    {
        if(((k >> 3) & 1) != side)
            continue;
        const int slot = 8 * (k / 16) + 2 * ((k / 2) % 4) + (k % 2);
        bn[slot]       = static_cast<BScalar>(B[k * kSwN + j]);
    }
    CV cn{};
    for(int reg = 0; reg < 8; ++reg)
    {
        const int m = reg + 8 * side;
        cn[reg]     = static_cast<CScalar>(C[m * kSwN + j]);
    }
    Value<AV, S, kCC> av(an);
    Value<BV, S, kCC> bv(bn);
    Value<CV, S, kCC> cv(cn);
    auto              dv = fn(av, bv, cv, idx[lane]);
    for(int reg = 0; reg < 8; ++reg)
    {
        const int m = reg + 8 * side;
        if constexpr(S == Semantics::Native)
            D[m * kSwN + j] = static_cast<Out>(dv.get(reg).to_float());
        else
            D[m * kSwN + j] = static_cast<Out>(dv.get(reg).fpsan_payload());
    }
}

// Compute the sparse host-side reference D[m][n] = C + A_dense * B (the live
// values are already the only nonzeros in A_dense).
template <class CScalar>
std::vector<CScalar> sw_reference_h(const SwData& d)
{
    std::vector<CScalar> ref(kSwM * kSwN);
    for(int m = 0; m < kSwM; ++m)
        for(int n = 0; n < kSwN; ++n)
        {
            float acc = d.C[m * kSwN + n];
            for(int g = 0; g < 8; ++g)
            {
                acc += d.A[m * kSwK + 4 * g + d.p0[m * 8 + g]]
                       * d.B[(4 * g + d.p0[m * 8 + g]) * kSwN + n];
                acc += d.A[m * kSwK + 4 * g + d.p1[m * 8 + g]]
                       * d.B[(4 * g + d.p1[m * 8 + g]) * kSwN + n];
            }
            ref[m * kSwN + n] = static_cast<CScalar>(acc);
        }
    return ref;
}

// FPSan-domain reference using scalar Value arithmetic. Uses the same scalar
// types the kernel uses (AScalar/BScalar/CScalar = element types).
template <class AScalar, class BScalar, class CScalar>
std::vector<std::uint64_t> sw_reference_fpsan_h(const SwData& d)
{
    using VA = Value<AScalar, Semantics::Triton, kCC>;
    using VB = Value<BScalar, Semantics::Triton, kCC>;
    using VC = Value<CScalar, Semantics::Triton, kCC>;
    std::vector<std::uint64_t> ref(kSwM * kSwN);
    for(int m = 0; m < kSwM; ++m)
        for(int n = 0; n < kSwN; ++n)
        {
            VC acc(static_cast<CScalar>(d.C[m * kSwN + n]));
            for(int g = 0; g < 8; ++g)
            {
                const int k0 = 4 * g + d.p0[m * 8 + g];
                const int k1 = 4 * g + d.p1[m * 8 + g];
                acc          = acc
                      + fpsan::cast<CScalar>(VA(static_cast<AScalar>(d.A[m * kSwK + k0])))
                            * fpsan::cast<CScalar>(VB(static_cast<BScalar>(d.B[k0 * kSwN + n])));
                acc = acc
                      + fpsan::cast<CScalar>(VA(static_cast<AScalar>(d.A[m * kSwK + k1])))
                            * fpsan::cast<CScalar>(VB(static_cast<BScalar>(d.B[k1 * kSwN + n])));
            }
            ref[m * kSwN + n] = static_cast<std::uint64_t>(acc.fpsan_payload());
        }
    return ref;
}

// ---- f16 layout + payload ---------------------------------------------------
__global__ void k_swmmac_f16_float(const float*         A,
                                   const float*         B,
                                   const float*         C,
                                   const int*           p0,
                                   const int*           p1,
                                   const std::uint16_t* idx,
                                   float*               D)
{
    auto fn = [](auto av, auto bv, auto cv, std::uint16_t i) {
        return fpsan::amdgcn_swmmac_f32_16x16x32_f16_w32<Semantics::Native, kCC>(av, bv, cv, i);
    };
    swmmac_h_dataflow<Semantics::Native, _Float16, _Float16, float, float>(
        A, B, C, p0, p1, idx, D, fn);
}
__global__ void k_swmmac_f16_fpsan(const float*         A,
                                   const float*         B,
                                   const float*         C,
                                   const int*           p0,
                                   const int*           p1,
                                   const std::uint16_t* idx,
                                   std::uint32_t*       D)
{
    auto fn = [](auto av, auto bv, auto cv, std::uint16_t i) {
        return fpsan::amdgcn_swmmac_f32_16x16x32_f16_w32<Semantics::Triton, kCC>(av, bv, cv, i);
    };
    swmmac_h_dataflow<Semantics::Triton, _Float16, _Float16, float, std::uint32_t>(
        A, B, C, p0, p1, idx, D, fn);
}

TEST(SwmmacGfx12, F32_F16_LayoutMatchesHardware)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    SwData             d   = make_sw_data(0x100, encode_idx_h);
    std::vector<float> ref = sw_reference_h<float>(d);
    float *            dA = to_dev(d.A), *dB = to_dev(d.B), *dC = to_dev(d.C);
    int *              dp0 = to_dev(d.p0), *dp1 = to_dev(d.p1);
    std::uint16_t*     dI = to_dev(d.idx);
    float*             dD = nullptr;
    HIP_CHECK(hipMalloc(&dD, kSwM * kSwN * sizeof(float)));
    k_swmmac_f16_float<<<1, WAVE>>>(dA, dB, dC, dp0, dp1, dI, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> got(kSwM * kSwN);
    HIP_CHECK(hipMemcpy(got.data(), dD, kSwM * kSwN * sizeof(float), hipMemcpyDeviceToHost));
    for(int t = 0; t < kSwM * kSwN; ++t)
        EXPECT_EQ(bits_of(got[t]), bits_of(ref[t])) << "at " << t;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dp0);
    (void)hipFree(dp1);
    (void)hipFree(dI);
    (void)hipFree(dD);
}

TEST(SwmmacGfx12, F32_F16_FpsanMatchesScalarReference)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    SwData                     d   = make_sw_data(0x100, encode_idx_h);
    std::vector<std::uint64_t> ref = sw_reference_fpsan_h<_Float16, _Float16, float>(d);
    float *                    dA = to_dev(d.A), *dB = to_dev(d.B), *dC = to_dev(d.C);
    int *                      dp0 = to_dev(d.p0), *dp1 = to_dev(d.p1);
    std::uint16_t*             dI = to_dev(d.idx);
    std::uint32_t*             dD = nullptr;
    HIP_CHECK(hipMalloc(&dD, kSwM * kSwN * sizeof(std::uint32_t)));
    k_swmmac_f16_fpsan<<<1, WAVE>>>(dA, dB, dC, dp0, dp1, dI, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(kSwM * kSwN);
    HIP_CHECK(
        hipMemcpy(got.data(), dD, kSwM * kSwN * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for(int t = 0; t < kSwM * kSwN; ++t)
        EXPECT_EQ(static_cast<std::uint64_t>(got[t]), ref[t]) << "at " << t;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dp0);
    (void)hipFree(dp1);
    (void)hipFree(dI);
    (void)hipFree(dD);
}

// ---- bf16 layout + payload --------------------------------------------------
__global__ void k_swmmac_bf16_float(const float*         A,
                                    const float*         B,
                                    const float*         C,
                                    const int*           p0,
                                    const int*           p1,
                                    const std::uint16_t* idx,
                                    float*               D)
{
    auto fn = [](auto av, auto bv, auto cv, std::uint16_t i) {
        return fpsan::amdgcn_swmmac_f32_16x16x32_bf16_w32<Semantics::Native, kCC>(av, bv, cv, i);
    };
    swmmac_h_dataflow<Semantics::Native, __bf16, __bf16, float, float>(A, B, C, p0, p1, idx, D, fn);
}
__global__ void k_swmmac_bf16_fpsan(const float*         A,
                                    const float*         B,
                                    const float*         C,
                                    const int*           p0,
                                    const int*           p1,
                                    const std::uint16_t* idx,
                                    std::uint32_t*       D)
{
    auto fn = [](auto av, auto bv, auto cv, std::uint16_t i) {
        return fpsan::amdgcn_swmmac_f32_16x16x32_bf16_w32<Semantics::Triton, kCC>(av, bv, cv, i);
    };
    swmmac_h_dataflow<Semantics::Triton, __bf16, __bf16, float, std::uint32_t>(
        A, B, C, p0, p1, idx, D, fn);
}

TEST(SwmmacGfx12, F32_BF16_LayoutMatchesHardware)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    SwData             d   = make_sw_data(0x200, encode_idx_h);
    std::vector<float> ref = sw_reference_h<float>(d);
    float *            dA = to_dev(d.A), *dB = to_dev(d.B), *dC = to_dev(d.C);
    int *              dp0 = to_dev(d.p0), *dp1 = to_dev(d.p1);
    std::uint16_t*     dI = to_dev(d.idx);
    float*             dD = nullptr;
    HIP_CHECK(hipMalloc(&dD, kSwM * kSwN * sizeof(float)));
    k_swmmac_bf16_float<<<1, WAVE>>>(dA, dB, dC, dp0, dp1, dI, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> got(kSwM * kSwN);
    HIP_CHECK(hipMemcpy(got.data(), dD, kSwM * kSwN * sizeof(float), hipMemcpyDeviceToHost));
    for(int t = 0; t < kSwM * kSwN; ++t)
        EXPECT_EQ(bits_of(got[t]), bits_of(ref[t])) << "at " << t;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dp0);
    (void)hipFree(dp1);
    (void)hipFree(dI);
    (void)hipFree(dD);
}

TEST(SwmmacGfx12, F32_BF16_FpsanMatchesScalarReference)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    SwData                     d   = make_sw_data(0x200, encode_idx_h);
    std::vector<std::uint64_t> ref = sw_reference_fpsan_h<__bf16, __bf16, float>(d);
    float *                    dA = to_dev(d.A), *dB = to_dev(d.B), *dC = to_dev(d.C);
    int *                      dp0 = to_dev(d.p0), *dp1 = to_dev(d.p1);
    std::uint16_t*             dI = to_dev(d.idx);
    std::uint32_t*             dD = nullptr;
    HIP_CHECK(hipMalloc(&dD, kSwM * kSwN * sizeof(std::uint32_t)));
    k_swmmac_bf16_fpsan<<<1, WAVE>>>(dA, dB, dC, dp0, dp1, dI, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(kSwM * kSwN);
    HIP_CHECK(
        hipMemcpy(got.data(), dD, kSwM * kSwN * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for(int t = 0; t < kSwM * kSwN; ++t)
        EXPECT_EQ(static_cast<std::uint64_t>(got[t]), ref[t]) << "at " << t;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dp0);
    (void)hipFree(dp1);
    (void)hipFree(dI);
    (void)hipFree(dD);
}

// ---- f16 -> f16 layout + payload --------------------------------------------
__global__ void k_swmmac_f16h_float(const float*         A,
                                    const float*         B,
                                    const float*         C,
                                    const int*           p0,
                                    const int*           p1,
                                    const std::uint16_t* idx,
                                    _Float16*            D)
{
    auto fn = [](auto av, auto bv, auto cv, std::uint16_t i) {
        return fpsan::amdgcn_swmmac_f16_16x16x32_f16_w32<Semantics::Native, kCC>(av, bv, cv, i);
    };
    swmmac_h_dataflow<Semantics::Native, _Float16, _Float16, _Float16, _Float16>(
        A, B, C, p0, p1, idx, D, fn);
}
__global__ void k_swmmac_f16h_fpsan(const float*         A,
                                    const float*         B,
                                    const float*         C,
                                    const int*           p0,
                                    const int*           p1,
                                    const std::uint16_t* idx,
                                    std::uint16_t*       D)
{
    auto fn = [](auto av, auto bv, auto cv, std::uint16_t i) {
        return fpsan::amdgcn_swmmac_f16_16x16x32_f16_w32<Semantics::Triton, kCC>(av, bv, cv, i);
    };
    swmmac_h_dataflow<Semantics::Triton, _Float16, _Float16, _Float16, std::uint16_t>(
        A, B, C, p0, p1, idx, D, fn);
}

TEST(SwmmacGfx12, F16_F16_LayoutMatchesHardware)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    SwData                d   = make_sw_data(0x300, encode_idx_h);
    std::vector<_Float16> ref = sw_reference_h<_Float16>(d);
    float *               dA = to_dev(d.A), *dB = to_dev(d.B), *dC = to_dev(d.C);
    int *                 dp0 = to_dev(d.p0), *dp1 = to_dev(d.p1);
    std::uint16_t*        dI = to_dev(d.idx);
    _Float16*             dD = nullptr;
    HIP_CHECK(hipMalloc(&dD, kSwM * kSwN * sizeof(_Float16)));
    k_swmmac_f16h_float<<<1, WAVE>>>(dA, dB, dC, dp0, dp1, dI, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<_Float16> got(kSwM * kSwN);
    HIP_CHECK(hipMemcpy(got.data(), dD, kSwM * kSwN * sizeof(_Float16), hipMemcpyDeviceToHost));
    for(int t = 0; t < kSwM * kSwN; ++t)
        EXPECT_EQ(bits_of(got[t]), bits_of(ref[t])) << "at " << t;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dp0);
    (void)hipFree(dp1);
    (void)hipFree(dI);
    (void)hipFree(dD);
}

TEST(SwmmacGfx12, F16_F16_FpsanMatchesScalarReference)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    SwData                     d   = make_sw_data(0x300, encode_idx_h);
    std::vector<std::uint64_t> ref = sw_reference_fpsan_h<_Float16, _Float16, _Float16>(d);
    float *                    dA = to_dev(d.A), *dB = to_dev(d.B), *dC = to_dev(d.C);
    int *                      dp0 = to_dev(d.p0), *dp1 = to_dev(d.p1);
    std::uint16_t*             dI = to_dev(d.idx);
    std::uint16_t*             dD = nullptr;
    HIP_CHECK(hipMalloc(&dD, kSwM * kSwN * sizeof(std::uint16_t)));
    k_swmmac_f16h_fpsan<<<1, WAVE>>>(dA, dB, dC, dp0, dp1, dI, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint16_t> got(kSwM * kSwN);
    HIP_CHECK(
        hipMemcpy(got.data(), dD, kSwM * kSwN * sizeof(std::uint16_t), hipMemcpyDeviceToHost));
    for(int t = 0; t < kSwM * kSwN; ++t)
        EXPECT_EQ(static_cast<std::uint64_t>(got[t]), ref[t]) << "at " << t;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dp0);
    (void)hipFree(dp1);
    (void)hipFree(dI);
    (void)hipFree(dD);
}

// ============================================================================
// Layout / FPSan dataflow tests for fp8/bf8 (different per-lane layout).
// ============================================================================
//
// Lane L holds (i = L%16, side = L/16). A v8 packs 8 groups linearly,
// 2 bytes per group. For each g in 0..7 with (g/4) == side:
//   byte_off = 2*(g%4)
//   byte_off + 0 -> A_dense[i][4g + p0]
//   byte_off + 1 -> A_dense[i][4g + p1]
// B v16 holds k = 16*side + (byte_off) linearly: byte = k%16, lane = j + 16*(k/16).
//
// fp8/bf8 path: stage bytes directly, build per-lane fragments using
// AScalar/BScalar element types (fpsan::fp8_e4m3 / fp8_e5m2).
template <Semantics S, class AScalar, class BScalar, class Out, class WrapFn>
__global__ void k_swmmac_fp8_kernel(const float*         Adense,
                                    const float*         B,
                                    const float*         C,
                                    const int*           p0,
                                    const int*           p1,
                                    const std::uint16_t* idx,
                                    Out*                 D,
                                    WrapFn               fn)
{
    using AFrag    = fpsan::detail::v8_fragment<AScalar>;
    using BFrag    = fpsan::detail::v16_byte_fragment<BScalar>;
    using v8f      = fpsan::v8f_native;
    const int lane = threadIdx.x;
    const int i    = lane & 15;
    const int j    = lane & 15;
    const int side = lane >> 4;

    AFrag an{};
    for(int g = 0; g < 8; ++g)
    {
        if(((g >> 2) & 1) != side)
            continue;
        const int b0 = 2 * (g & 3);
        an[b0]       = static_cast<AScalar>(Adense[i * kSwK + 4 * g + p0[i * 8 + g]]);
        an[b0 + 1]   = static_cast<AScalar>(Adense[i * kSwK + 4 * g + p1[i * 8 + g]]);
    }
    BFrag bn{};
    for(int k = 0; k < kSwK; ++k)
    {
        if((k >> 4) != side)
            continue;
        const int byte = k & 15;
        bn[byte]       = static_cast<BScalar>(B[k * kSwN + j]);
    }
    v8f cn{};
    for(int reg = 0; reg < 8; ++reg)
    {
        const int m = reg + 8 * side;
        cn[reg]     = C[m * kSwN + j];
    }
    Value<AFrag, S, kCC> av(an);
    Value<BFrag, S, kCC> bv(bn);
    Value<v8f, S, kCC>   cv(cn);
    auto                 dv = fn(av, bv, cv, idx[lane]);
    for(int reg = 0; reg < 8; ++reg)
    {
        const int m = reg + 8 * side;
        if constexpr(S == Semantics::Native)
            D[m * kSwN + j] = static_cast<Out>(dv.get(reg).to_float());
        else
            D[m * kSwN + j] = static_cast<Out>(dv.get(reg).fpsan_payload());
    }
}

template <class AScalar, class BScalar>
std::vector<float> sw_reference_fp8(const SwData& d)
{
    std::vector<float> ref(kSwM * kSwN);
    for(int m = 0; m < kSwM; ++m)
        for(int n = 0; n < kSwN; ++n)
        {
            float acc = d.C[m * kSwN + n];
            for(int g = 0; g < 8; ++g)
            {
                const int   k0  = 4 * g + d.p0[m * 8 + g];
                const int   k1  = 4 * g + d.p1[m * 8 + g];
                const float av0 = static_cast<float>(static_cast<AScalar>(d.A[m * kSwK + k0]));
                const float av1 = static_cast<float>(static_cast<AScalar>(d.A[m * kSwK + k1]));
                const float bv0 = static_cast<float>(static_cast<BScalar>(d.B[k0 * kSwN + n]));
                const float bv1 = static_cast<float>(static_cast<BScalar>(d.B[k1 * kSwN + n]));
                acc += av0 * bv0;
                acc += av1 * bv1;
            }
            ref[m * kSwN + n] = acc;
        }
    return ref;
}

template <class AScalar, class BScalar>
std::vector<std::uint32_t> sw_reference_fp8_fpsan(const SwData& d)
{
    using VA = Value<AScalar, Semantics::Triton, kCC>;
    using VB = Value<BScalar, Semantics::Triton, kCC>;
    using VF = Value<float, Semantics::Triton, kCC>;
    std::vector<std::uint32_t> ref(kSwM * kSwN);
    for(int m = 0; m < kSwM; ++m)
        for(int n = 0; n < kSwN; ++n)
        {
            VF acc(d.C[m * kSwN + n]);
            for(int g = 0; g < 8; ++g)
            {
                const int k0 = 4 * g + d.p0[m * 8 + g];
                const int k1 = 4 * g + d.p1[m * 8 + g];
                acc          = acc
                      + fpsan::cast<float>(VA(static_cast<AScalar>(d.A[m * kSwK + k0])))
                            * fpsan::cast<float>(VB(static_cast<BScalar>(d.B[k0 * kSwN + n])));
                acc = acc
                      + fpsan::cast<float>(VA(static_cast<AScalar>(d.A[m * kSwK + k1])))
                            * fpsan::cast<float>(VB(static_cast<BScalar>(d.B[k1 * kSwN + n])));
            }
            ref[m * kSwN + n] = acc.fpsan_payload();
        }
    return ref;
}

// Generic fp8/bf8 layout + fpsan test runner.
template <class AScalar, class BScalar, class WrapFloatFn, class WrapFpsanFn>
void run_fp8_layout_and_fpsan(std::uint32_t seed, WrapFloatFn wf, WrapFpsanFn wp)
{
    if(!have_device())
        GTEST_SKIP() << "no HIP device";
    SwData             d   = make_sw_data(seed, encode_idx_fp8);
    std::vector<float> ref = sw_reference_fp8<AScalar, BScalar>(d);
    float *            dA = to_dev(d.A), *dB = to_dev(d.B), *dC = to_dev(d.C);
    int *              dp0 = to_dev(d.p0), *dp1 = to_dev(d.p1);
    std::uint16_t*     dI  = to_dev(d.idx);
    float*             dDf = nullptr;
    HIP_CHECK(hipMalloc(&dDf, kSwM * kSwN * sizeof(float)));
    k_swmmac_fp8_kernel<Semantics::Native, AScalar, BScalar, float>
        <<<1, WAVE>>>(dA, dB, dC, dp0, dp1, dI, dDf, wf);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> got(kSwM * kSwN);
    HIP_CHECK(hipMemcpy(got.data(), dDf, kSwM * kSwN * sizeof(float), hipMemcpyDeviceToHost));
    for(int t = 0; t < kSwM * kSwN; ++t)
        EXPECT_EQ(bits_of(got[t]), bits_of(ref[t])) << "Layout at " << t;
    (void)hipFree(dDf);

    std::vector<std::uint32_t> ref_p = sw_reference_fp8_fpsan<AScalar, BScalar>(d);
    std::uint32_t*             dDp   = nullptr;
    HIP_CHECK(hipMalloc(&dDp, kSwM * kSwN * sizeof(std::uint32_t)));
    k_swmmac_fp8_kernel<Semantics::Triton, AScalar, BScalar, std::uint32_t>
        <<<1, WAVE>>>(dA, dB, dC, dp0, dp1, dI, dDp, wp);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got_p(kSwM * kSwN);
    HIP_CHECK(
        hipMemcpy(got_p.data(), dDp, kSwM * kSwN * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for(int t = 0; t < kSwM * kSwN; ++t)
        EXPECT_EQ(got_p[t], ref_p[t]) << "FPSan at " << t;
    (void)hipFree(dDp);

    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dp0);
    (void)hipFree(dp1);
    (void)hipFree(dI);
}

TEST(SwmmacGfx12, F32_FP8_FP8_LayoutAndFpsan)
{
    auto wf = [](auto av, auto bv, auto cv, std::uint16_t i) {
        return fpsan::amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32<Semantics::Native, kCC>(av, bv, cv, i);
    };
    auto wp = [](auto av, auto bv, auto cv, std::uint16_t i) {
        return fpsan::amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32<Semantics::Triton, kCC>(av, bv, cv, i);
    };
    run_fp8_layout_and_fpsan<fpsan::fp8_e4m3, fpsan::fp8_e4m3>(0x400, wf, wp);
}
TEST(SwmmacGfx12, F32_FP8_BF8_LayoutAndFpsan)
{
    auto wf = [](auto av, auto bv, auto cv, std::uint16_t i) {
        return fpsan::amdgcn_swmmac_f32_16x16x32_fp8_bf8_w32<Semantics::Native, kCC>(av, bv, cv, i);
    };
    auto wp = [](auto av, auto bv, auto cv, std::uint16_t i) {
        return fpsan::amdgcn_swmmac_f32_16x16x32_fp8_bf8_w32<Semantics::Triton, kCC>(av, bv, cv, i);
    };
    run_fp8_layout_and_fpsan<fpsan::fp8_e4m3, fpsan::fp8_e5m2>(0x500, wf, wp);
}
TEST(SwmmacGfx12, F32_BF8_FP8_LayoutAndFpsan)
{
    auto wf = [](auto av, auto bv, auto cv, std::uint16_t i) {
        return fpsan::amdgcn_swmmac_f32_16x16x32_bf8_fp8_w32<Semantics::Native, kCC>(av, bv, cv, i);
    };
    auto wp = [](auto av, auto bv, auto cv, std::uint16_t i) {
        return fpsan::amdgcn_swmmac_f32_16x16x32_bf8_fp8_w32<Semantics::Triton, kCC>(av, bv, cv, i);
    };
    run_fp8_layout_and_fpsan<fpsan::fp8_e5m2, fpsan::fp8_e4m3>(0x600, wf, wp);
}
TEST(SwmmacGfx12, F32_BF8_BF8_LayoutAndFpsan)
{
    auto wf = [](auto av, auto bv, auto cv, std::uint16_t i) {
        return fpsan::amdgcn_swmmac_f32_16x16x32_bf8_bf8_w32<Semantics::Native, kCC>(av, bv, cv, i);
    };
    auto wp = [](auto av, auto bv, auto cv, std::uint16_t i) {
        return fpsan::amdgcn_swmmac_f32_16x16x32_bf8_bf8_w32<Semantics::Triton, kCC>(av, bv, cv, i);
    };
    run_fp8_layout_and_fpsan<fpsan::fp8_e5m2, fpsan::fp8_e5m2>(0x700, wf, wp);
}
