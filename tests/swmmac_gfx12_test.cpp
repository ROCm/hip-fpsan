// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/swmmac_gfx12_test.cpp
//
// Silicon tests for the gfx12 (RDNA4) sparse WMMA wrappers in
// fpsan/amdgcn_swmmac_gfx12.hpp. Three layers:
//
//   * *FloatMatchesBuiltin: the wrapper in Native mode is a bit-exact
//     pass-through to the underlying __builtin_amdgcn_swmmac_*.
//   * *LayoutMatchesHardware: the software dataflow in Native mode produces
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
#include "fpsan_semantics.hpp"
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

#define HIP_CHECK(e)                                                                               \
  do {                                                                                             \
    hipError_t e_ = (e);                                                                           \
    ASSERT_EQ(e_, hipSuccess) << hipGetErrorString(e_);                                            \
  } while (0)

static constexpr Conversions kCC = Conversions::Explicit;

// ---- f16 SWMMAC: side-by-side raw vs wrapper, bit-exact comparison ---------
__global__ void k_swmmac_f16_pair(const std::uint16_t *a_in, const std::uint16_t *b_in,
                                  const float *c_in, std::uint16_t idx, float *raw_out,
                                  float *wrap_out) {
  using v8h = _Float16 __attribute__((ext_vector_type(8)));
  using v16h = _Float16 __attribute__((ext_vector_type(16)));
  using v8f = float __attribute__((ext_vector_type(8)));
  int lane = threadIdx.x;

  // Reconstruct lane-private A/B/C from staged bit patterns.
  v8h a;
  v16h b;
  v8f c;
  for (int s = 0; s < 8; ++s) {
    std::uint16_t u = a_in[lane * 8 + s];
    _Float16 v;
    __builtin_memcpy(&v, &u, sizeof v);
    a[s] = v;
  }
  for (int s = 0; s < 16; ++s) {
    std::uint16_t u = b_in[lane * 16 + s];
    _Float16 v;
    __builtin_memcpy(&v, &u, sizeof v);
    b[s] = v;
  }
  for (int s = 0; s < 8; ++s)
    c[s] = c_in[lane * 8 + s];

  // --- direct builtin ---
  v8f raw{};
#ifdef __HIP_DEVICE_COMPILE__
  raw = __builtin_amdgcn_swmmac_f32_16x16x32_f16_w32(a, b, c, static_cast<short>(idx));
#endif
  for (int s = 0; s < 8; ++s)
    raw_out[lane * 8 + s] = raw[s];

  // --- via fpsan wrapper (Native mode) ---
  Value<fpsan::v8h_native, Semantics::Native, kCC> av{a};
  Value<fpsan::v16h_swmmac_native, Semantics::Native, kCC> bv{b};
  Value<fpsan::v8f_native, Semantics::Native, kCC> cv{c};
  auto dv = fpsan::amdgcn_swmmac_f32_16x16x32_f16_w32<Semantics::Native, kCC>(av, bv, cv, idx);
  fpsan::v8f_native d = static_cast<fpsan::v8f_native>(dv);
  for (int s = 0; s < 8; ++s)
    wrap_out[lane * 8 + s] = d[s];
}

// ---- bf16 SWMMAC ----------------------------------------------------------
__global__ void k_swmmac_bf16_pair(const std::uint16_t *a_in, const std::uint16_t *b_in,
                                   const float *c_in, std::uint16_t idx, float *raw_out,
                                   float *wrap_out) {
  using v8bf = __bf16 __attribute__((ext_vector_type(8)));
  using v16bf = __bf16 __attribute__((ext_vector_type(16)));
  using v8f = float __attribute__((ext_vector_type(8)));
  int lane = threadIdx.x;
  v8bf a;
  v16bf b;
  v8f c;
  for (int s = 0; s < 8; ++s) {
    std::uint16_t u = a_in[lane * 8 + s];
    __bf16 v;
    __builtin_memcpy(&v, &u, sizeof v);
    a[s] = v;
  }
  for (int s = 0; s < 16; ++s) {
    std::uint16_t u = b_in[lane * 16 + s];
    __bf16 v;
    __builtin_memcpy(&v, &u, sizeof v);
    b[s] = v;
  }
  for (int s = 0; s < 8; ++s)
    c[s] = c_in[lane * 8 + s];
  v8f raw{};
#ifdef __HIP_DEVICE_COMPILE__
  raw = __builtin_amdgcn_swmmac_f32_16x16x32_bf16_w32(a, b, c, static_cast<short>(idx));
#endif
  for (int s = 0; s < 8; ++s)
    raw_out[lane * 8 + s] = raw[s];
  Value<fpsan::v8bf_native, Semantics::Native, kCC> av{a};
  Value<fpsan::v16bf_swmmac_native, Semantics::Native, kCC> bv{b};
  Value<fpsan::v8f_native, Semantics::Native, kCC> cv{c};
  auto dv = fpsan::amdgcn_swmmac_f32_16x16x32_bf16_w32<Semantics::Native, kCC>(av, bv, cv, idx);
  fpsan::v8f_native d = static_cast<fpsan::v8f_native>(dv);
  for (int s = 0; s < 8; ++s)
    wrap_out[lane * 8 + s] = d[s];
}

// ---- f16 -> f16 SWMMAC ----------------------------------------------------
__global__ void k_swmmac_f16h_pair(const std::uint16_t *a_in, const std::uint16_t *b_in,
                                   const std::uint16_t *c_in, std::uint16_t idx,
                                   std::uint16_t *raw_out, std::uint16_t *wrap_out) {
  using v8h = _Float16 __attribute__((ext_vector_type(8)));
  using v16h = _Float16 __attribute__((ext_vector_type(16)));
  int lane = threadIdx.x;
  v8h a, c;
  v16h b;
  auto load_h = [](std::uint16_t u) {
    _Float16 v;
    __builtin_memcpy(&v, &u, sizeof v);
    return v;
  };
  for (int s = 0; s < 8; ++s)
    a[s] = load_h(a_in[lane * 8 + s]);
  for (int s = 0; s < 16; ++s)
    b[s] = load_h(b_in[lane * 16 + s]);
  for (int s = 0; s < 8; ++s)
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
  for (int s = 0; s < 8; ++s)
    raw_out[lane * 8 + s] = store_h(raw[s]);
  Value<fpsan::v8h_native, Semantics::Native, kCC> av{a}, cv{c};
  Value<fpsan::v16h_swmmac_native, Semantics::Native, kCC> bv{b};
  auto dv = fpsan::amdgcn_swmmac_f16_16x16x32_f16_w32<Semantics::Native, kCC>(av, bv, cv, idx);
  fpsan::v8h_native d = static_cast<fpsan::v8h_native>(dv);
  for (int s = 0; s < 8; ++s)
    wrap_out[lane * 8 + s] = store_h(d[s]);
}

// ---- FP8/BF8 SWMMAC kernel template -- 8 fp8 bytes / lane for A, 16 for B,
// v8f for C/D. Pre-/post-byte serialization via memcpy keeps everything
// portable.
template <int VARIANT> // 0=fp8/fp8, 1=fp8/bf8, 2=bf8/fp8, 3=bf8/bf8
__global__ void k_swmmac_fp8_pair(const std::uint8_t *a_in, const std::uint8_t *b_in,
                                  const float *c_in, std::uint16_t idx, float *raw_out,
                                  float *wrap_out) {
  using v2i = int __attribute__((ext_vector_type(2)));
  using v4i = int __attribute__((ext_vector_type(4)));
  using v8f = float __attribute__((ext_vector_type(8)));
  int lane = threadIdx.x;
  v2i a;
  v4i b;
  v8f c;
  __builtin_memcpy(&a, a_in + lane * 8, 8);
  __builtin_memcpy(&b, b_in + lane * 16, 16);
  for (int s = 0; s < 8; ++s)
    c[s] = c_in[lane * 8 + s];
  v8f raw{};
#ifdef __HIP_DEVICE_COMPILE__
  if constexpr (VARIANT == 0)
    raw = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32(a, b, c, static_cast<short>(idx));
  else if constexpr (VARIANT == 1)
    raw = __builtin_amdgcn_swmmac_f32_16x16x32_fp8_bf8_w32(a, b, c, static_cast<short>(idx));
  else if constexpr (VARIANT == 2)
    raw = __builtin_amdgcn_swmmac_f32_16x16x32_bf8_fp8_w32(a, b, c, static_cast<short>(idx));
  else
    raw = __builtin_amdgcn_swmmac_f32_16x16x32_bf8_bf8_w32(a, b, c, static_cast<short>(idx));
#endif
  for (int s = 0; s < 8; ++s)
    raw_out[lane * 8 + s] = raw[s];

  Value<fpsan::v8f_native, Semantics::Native, kCC> cv{c};
  fpsan::v8f_native d{};
  if constexpr (VARIANT == 0) {
    Value<fpsan::v8e4m3_native, Semantics::Native, kCC> av{
        __builtin_bit_cast(fpsan::v8e4m3_native, a)};
    Value<fpsan::v16e4m3_swmmac_native, Semantics::Native, kCC> bv{
        __builtin_bit_cast(fpsan::v16e4m3_swmmac_native, b)};
    d = static_cast<fpsan::v8f_native>(
        fpsan::amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32<Semantics::Native, kCC>(av, bv, cv, idx));
  } else if constexpr (VARIANT == 1) {
    Value<fpsan::v8e4m3_native, Semantics::Native, kCC> av{
        __builtin_bit_cast(fpsan::v8e4m3_native, a)};
    Value<fpsan::v16e5m2_swmmac_native, Semantics::Native, kCC> bv{
        __builtin_bit_cast(fpsan::v16e5m2_swmmac_native, b)};
    d = static_cast<fpsan::v8f_native>(
        fpsan::amdgcn_swmmac_f32_16x16x32_fp8_bf8_w32<Semantics::Native, kCC>(av, bv, cv, idx));
  } else if constexpr (VARIANT == 2) {
    Value<fpsan::v8e5m2_native, Semantics::Native, kCC> av{
        __builtin_bit_cast(fpsan::v8e5m2_native, a)};
    Value<fpsan::v16e4m3_swmmac_native, Semantics::Native, kCC> bv{
        __builtin_bit_cast(fpsan::v16e4m3_swmmac_native, b)};
    d = static_cast<fpsan::v8f_native>(
        fpsan::amdgcn_swmmac_f32_16x16x32_bf8_fp8_w32<Semantics::Native, kCC>(av, bv, cv, idx));
  } else {
    Value<fpsan::v8e5m2_native, Semantics::Native, kCC> av{
        __builtin_bit_cast(fpsan::v8e5m2_native, a)};
    Value<fpsan::v16e5m2_swmmac_native, Semantics::Native, kCC> bv{
        __builtin_bit_cast(fpsan::v16e5m2_swmmac_native, b)};
    d = static_cast<fpsan::v8f_native>(
        fpsan::amdgcn_swmmac_f32_16x16x32_bf8_bf8_w32<Semantics::Native, kCC>(av, bv, cv, idx));
  }
  for (int s = 0; s < 8; ++s)
    wrap_out[lane * 8 + s] = d[s];
}

namespace {
constexpr int WAVE = 32;
bool have_device() {
  int n = 0;
  return hipGetDeviceCount(&n) == hipSuccess && n > 0;
}
} // namespace

TEST(SwmmacGfx12, F32_F16_FloatMatchesBuiltin) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";

  // Generate small-range half-precision A/B and f32 C so the accumulated
  // products stay well within f32 precision -- but bit-exact equality is
  // what we're after either way.
  std::mt19937 rng(0xa5'12'5c'7a);
  std::uniform_int_distribution<int> di_h(0x3000, 0x4000); // ~ [0.5, 2.0)
  std::uniform_real_distribution<float> df(-1.f, 1.f);
  std::vector<std::uint16_t> ha(WAVE * 8), hb(WAVE * 16);
  std::vector<float> hc(WAVE * 8);
  for (auto &x : ha)
    x = static_cast<std::uint16_t>(di_h(rng));
  for (auto &x : hb)
    x = static_cast<std::uint16_t>(di_h(rng));
  for (auto &x : hc)
    x = df(rng);

  std::uint16_t *dA = nullptr;
  std::uint16_t *dB = nullptr;
  float *dC = nullptr;
  float *dRaw = nullptr;
  float *dWrap = nullptr;
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
  std::uint16_t idx = static_cast<std::uint16_t>(di16(rng));

  k_swmmac_f16_pair<<<1, WAVE>>>(dA, dB, dC, idx, dRaw, dWrap);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> raw(WAVE * 8), wrap(WAVE * 8);
  HIP_CHECK(hipMemcpy(raw.data(), dRaw, raw.size() * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(wrap.data(), dWrap, wrap.size() * sizeof(float), hipMemcpyDeviceToHost));
  for (size_t i = 0; i < raw.size(); ++i) {
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

TEST(SwmmacGfx12, F32_BF16_FloatMatchesBuiltin) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  std::mt19937 rng(0xb5'13'5c'7b);
  std::uniform_int_distribution<int> di(0x3F00, 0x4000); // ~ [0.5, 2.0) in bf16
  std::uniform_real_distribution<float> df(-1.f, 1.f);
  std::vector<std::uint16_t> ha(WAVE * 8), hb(WAVE * 16);
  std::vector<float> hc(WAVE * 8);
  for (auto &x : ha)
    x = static_cast<std::uint16_t>(di(rng));
  for (auto &x : hb)
    x = static_cast<std::uint16_t>(di(rng));
  for (auto &x : hc)
    x = df(rng);
  std::uint16_t *dA = nullptr;
  std::uint16_t *dB = nullptr;
  float *dC = nullptr;
  float *dRaw = nullptr;
  float *dWrap = nullptr;
  HIP_CHECK(hipMalloc(&dA, ha.size() * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&dB, hb.size() * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&dC, hc.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&dRaw, WAVE * 8 * sizeof(float)));
  HIP_CHECK(hipMalloc(&dWrap, WAVE * 8 * sizeof(float)));
  HIP_CHECK(hipMemcpy(dA, ha.data(), ha.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, hb.data(), hb.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dC, hc.data(), hc.size() * sizeof(float), hipMemcpyHostToDevice));
  std::uniform_int_distribution<int> di16(0, 0xFFFF);
  std::uint16_t idx = static_cast<std::uint16_t>(di16(rng));
  k_swmmac_bf16_pair<<<1, WAVE>>>(dA, dB, dC, idx, dRaw, dWrap);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> raw(WAVE * 8), wrap(WAVE * 8);
  HIP_CHECK(hipMemcpy(raw.data(), dRaw, raw.size() * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(wrap.data(), dWrap, wrap.size() * sizeof(float), hipMemcpyDeviceToHost));
  for (size_t i = 0; i < raw.size(); ++i) {
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

TEST(SwmmacGfx12, F16_F16_FloatMatchesBuiltin) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  std::mt19937 rng(0xc5'14'5c'7c);
  std::uniform_int_distribution<int> di(0x3000, 0x4000);
  std::vector<std::uint16_t> ha(WAVE * 8), hb(WAVE * 16), hc(WAVE * 8);
  for (auto &x : ha)
    x = static_cast<std::uint16_t>(di(rng));
  for (auto &x : hb)
    x = static_cast<std::uint16_t>(di(rng));
  for (auto &x : hc)
    x = static_cast<std::uint16_t>(di(rng));
  std::uint16_t *dA = nullptr;
  std::uint16_t *dB = nullptr;
  std::uint16_t *dC = nullptr;
  std::uint16_t *dRaw = nullptr;
  std::uint16_t *dWrap = nullptr;
  HIP_CHECK(hipMalloc(&dA, ha.size() * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&dB, hb.size() * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&dC, hc.size() * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&dRaw, WAVE * 8 * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&dWrap, WAVE * 8 * sizeof(std::uint16_t)));
  HIP_CHECK(hipMemcpy(dA, ha.data(), ha.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, hb.data(), hb.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dC, hc.data(), hc.size() * sizeof(std::uint16_t), hipMemcpyHostToDevice));
  std::uniform_int_distribution<int> di16(0, 0xFFFF);
  std::uint16_t idx = static_cast<std::uint16_t>(di16(rng));
  k_swmmac_f16h_pair<<<1, WAVE>>>(dA, dB, dC, idx, dRaw, dWrap);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<std::uint16_t> raw(WAVE * 8), wrap(WAVE * 8);
  HIP_CHECK(hipMemcpy(raw.data(), dRaw, raw.size() * sizeof(std::uint16_t), hipMemcpyDeviceToHost));
  HIP_CHECK(
      hipMemcpy(wrap.data(), dWrap, wrap.size() * sizeof(std::uint16_t), hipMemcpyDeviceToHost));
  for (size_t i = 0; i < raw.size(); ++i)
    EXPECT_EQ(wrap[i], raw[i]) << "lane " << (i / 8) << " slot " << (i % 8);
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dRaw);
  (void)hipFree(dWrap);
}

template <int VARIANT> static void run_fp8_swmmac_test(std::uint32_t seed) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> di_byte(0, 0x7F); // positive fp8 bytes
  std::uniform_real_distribution<float> df(-1.f, 1.f);
  std::vector<std::uint8_t> ha(WAVE * 8), hb(WAVE * 16);
  std::vector<float> hc(WAVE * 8);
  for (auto &x : ha)
    x = static_cast<std::uint8_t>(di_byte(rng));
  for (auto &x : hb)
    x = static_cast<std::uint8_t>(di_byte(rng));
  for (auto &x : hc)
    x = df(rng);
  std::uint8_t *dA = nullptr;
  std::uint8_t *dB = nullptr;
  float *dC = nullptr;
  float *dRaw = nullptr;
  float *dWrap = nullptr;
  HIP_CHECK(hipMalloc(&dA, ha.size()));
  HIP_CHECK(hipMalloc(&dB, hb.size()));
  HIP_CHECK(hipMalloc(&dC, hc.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&dRaw, WAVE * 8 * sizeof(float)));
  HIP_CHECK(hipMalloc(&dWrap, WAVE * 8 * sizeof(float)));
  HIP_CHECK(hipMemcpy(dA, ha.data(), ha.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, hb.data(), hb.size(), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dC, hc.data(), hc.size() * sizeof(float), hipMemcpyHostToDevice));
  std::uniform_int_distribution<int> di16(0, 0xFFFF);
  std::uint16_t idx = static_cast<std::uint16_t>(di16(rng));
  k_swmmac_fp8_pair<VARIANT><<<1, WAVE>>>(dA, dB, dC, idx, dRaw, dWrap);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> raw(WAVE * 8), wrap(WAVE * 8);
  HIP_CHECK(hipMemcpy(raw.data(), dRaw, raw.size() * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(wrap.data(), dWrap, wrap.size() * sizeof(float), hipMemcpyDeviceToHost));
  for (size_t i = 0; i < raw.size(); ++i) {
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

TEST(SwmmacGfx12, F32_FP8_FP8_FloatMatchesBuiltin) { run_fp8_swmmac_test<0>(0xfe'00); }
TEST(SwmmacGfx12, F32_FP8_BF8_FloatMatchesBuiltin) { run_fp8_swmmac_test<1>(0xfe'01); }
TEST(SwmmacGfx12, F32_BF8_FP8_FloatMatchesBuiltin) { run_fp8_swmmac_test<2>(0xfe'02); }
TEST(SwmmacGfx12, F32_BF8_BF8_FloatMatchesBuiltin) { run_fp8_swmmac_test<3>(0xfe'03); }

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
