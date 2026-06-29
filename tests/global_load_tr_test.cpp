// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/global_load_tr_test.cpp
//
// GPU tests for the gfx12 (RDNA4) matrix-transposed global-load wrappers in
// fpsan/amdgcn_global_load.hpp.
//
// These wrappers are thin, bit-faithful shims over the hardware transpose, so
// the test pins exactly that contract without needing to know the transpose
// pattern itself:
//
//   * MatchesBuiltin: the Native-mode wrapper returns the same bits as a direct
//     __builtin_amdgcn_global_load_tr_* call over identically staged global
//     memory.
//   * FpsanMovesSameBits: the FPSan-mode wrapper, given memory staged with the
//     same bit patterns (as payloads), returns those same bits transposed --
//     i.e. Float and FPSan move bits identically (the whole point: a transpose
//     observes no values, only bits).
//
// Requires real gfx1201 (or later gfx12) hardware; built only under
// FPSAN_ENABLE_HIP with a gfx12 architecture.
#include "fpsan/amdgcn_global_load.hpp"
#include "fpsan/fpsan.hpp"

#include "fpsan_semantics.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
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

#ifndef FPSAN_TEST_GLOBAL_LOAD_TR_W64
#define FPSAN_TEST_GLOBAL_LOAD_TR_W64 0
#endif

static constexpr int WAVE = FPSAN_TEST_GLOBAL_LOAD_TR_W64 ? 64 : 32;
static constexpr int SLOTS16 = FPSAN_TEST_GLOBAL_LOAD_TR_W64 ? 4 : 8;
static constexpr int SLOTS8 = 4;

// Deterministic distinct per-(lane,slot) 16-bit pattern. Avoids 0 and stays in
// a range that is a normal f16 bit pattern (doesn't matter for bit movement,
// but keeps the staged values unsurprising).
__device__ __host__ inline std::uint16_t pat16(int lane, int slot) {
  return static_cast<std::uint16_t>(0x3000 + lane * SLOTS16 + slot);
}

__device__ __host__ inline std::uint8_t pat8(int lane, int slot) {
  return static_cast<std::uint8_t>(lane * SLOTS8 + slot);
}

// ---- raw builtin reference for f16 -----------------------------------------
__global__ void k_loadtr_raw_f16(const std::uint16_t *gmem, std::uint16_t *out) {
#if FPSAN_TEST_GLOBAL_LOAD_TR_W64
  using vfp16 = __fp16 __attribute__((ext_vector_type(4)));
#else
  using v8fp16 = __fp16 __attribute__((ext_vector_type(8)));
#endif
  int lane = threadIdx.x;
  (void)gmem;
#if FPSAN_TEST_GLOBAL_LOAD_TR_W64
  vfp16 r{};
#else
  v8fp16 r{};
#endif
#ifdef __HIP_DEVICE_COMPILE__
#if FPSAN_TEST_GLOBAL_LOAD_TR_W64
  r = __builtin_amdgcn_global_load_tr_b128_v4f16(
      (vfp16 __attribute__((address_space(1))) *)(&gmem[lane * SLOTS16]));
#else
  r = __builtin_amdgcn_global_load_tr_b128_v8f16(
      (v8fp16 __attribute__((address_space(1))) *)(&gmem[lane * SLOTS16]));
#endif
#endif
  union {
#if FPSAN_TEST_GLOBAL_LOAD_TR_W64
    vfp16 v;
#else
    v8fp16 v;
#endif
    std::uint16_t u[SLOTS16];
  } u;
  u.v = r;
  for (int s = 0; s < SLOTS16; ++s)
    out[lane * SLOTS16 + s] = u.u[s];
}

// ---- fpsan wrapper for f16, both semantics ---------------------------------
template <Semantics S>
__global__ void k_loadtr_wrap_f16(const std::uint16_t *gmem, std::uint16_t *out) {
  using V = Value<_Float16, S, kCC>;
  int lane = threadIdx.x;
#if FPSAN_TEST_GLOBAL_LOAD_TR_W64
  auto r = fpsan::amdgcn_global_load_tr_b128_f16_w64<S, kCC>(
      reinterpret_cast<const V *>(&gmem[lane * SLOTS16]));
#else
  auto r = fpsan::amdgcn_global_load_tr_b128_f16<S, kCC>(
      reinterpret_cast<const V *>(&gmem[lane * SLOTS16]));
#endif
  for (int s = 0; s < SLOTS16; ++s)
    out[lane * SLOTS16 + s] = r.get(s).to_storage_bits();
}

// ---- raw builtin reference for bf16 ----------------------------------------
__global__ void k_loadtr_raw_bf16(const std::uint16_t *gmem, std::uint16_t *out) {
#if FPSAN_TEST_GLOBAL_LOAD_TR_W64
  using vbf = __bf16 __attribute__((ext_vector_type(4)));
#else
  using v8bf = __bf16 __attribute__((ext_vector_type(8)));
#endif
  int lane = threadIdx.x;
  (void)gmem;
#if FPSAN_TEST_GLOBAL_LOAD_TR_W64
  vbf r{};
#else
  v8bf r{};
#endif
#ifdef __HIP_DEVICE_COMPILE__
#if FPSAN_TEST_GLOBAL_LOAD_TR_W64
  r = __builtin_amdgcn_global_load_tr_b128_v4bf16(
      (vbf __attribute__((address_space(1))) *)(&gmem[lane * SLOTS16]));
#else
  r = __builtin_amdgcn_global_load_tr_b128_v8bf16(
      (v8bf __attribute__((address_space(1))) *)(&gmem[lane * SLOTS16]));
#endif
#endif
  union {
#if FPSAN_TEST_GLOBAL_LOAD_TR_W64
    vbf v;
#else
    v8bf v;
#endif
    std::uint16_t u[SLOTS16];
  } u;
  u.v = r;
  for (int s = 0; s < SLOTS16; ++s)
    out[lane * SLOTS16 + s] = u.u[s];
}

// ---- fpsan wrapper for bf16, both semantics --------------------------------
template <Semantics S>
__global__ void k_loadtr_wrap_bf16(const std::uint16_t *gmem, std::uint16_t *out) {
  using V = Value<__bf16, S, kCC>;
  int lane = threadIdx.x;
#if FPSAN_TEST_GLOBAL_LOAD_TR_W64
  auto r = fpsan::amdgcn_global_load_tr_b128_bf16_w64<S, kCC>(
      reinterpret_cast<const V *>(&gmem[lane * SLOTS16]));
#else
  auto r = fpsan::amdgcn_global_load_tr_b128_bf16<S, kCC>(
      reinterpret_cast<const V *>(&gmem[lane * SLOTS16]));
#endif
  for (int s = 0; s < SLOTS16; ++s)
    out[lane * SLOTS16 + s] = r.get(s).to_storage_bits();
}

#if FPSAN_TEST_GLOBAL_LOAD_TR_W64
// ---- raw builtin reference for 8-bit global_load_tr_b64 --------------------
__global__ void k_loadtr_raw_b64(const std::uint8_t *gmem, std::uint8_t *out) {
  int lane = threadIdx.x;
  (void)gmem;
  int r{};
#ifdef __HIP_DEVICE_COMPILE__
  r = __builtin_amdgcn_global_load_tr_b64_i32(
      (int __attribute__((address_space(1))) *)(&gmem[lane * SLOTS8]));
#endif
  union {
    int v;
    std::uint8_t u[SLOTS8];
  } u;
  u.v = r;
  for (int s = 0; s < SLOTS8; ++s)
    out[lane * SLOTS8 + s] = u.u[s];
}

// ---- fpsan wrapper for fp8/bf8 global_load_tr_b64, both semantics ----------
template <Semantics S>
__global__ void k_loadtr_wrap_fp8(const std::uint8_t *gmem, std::uint8_t *out) {
  using V = Value<fpsan::fp8_e4m3, S, kCC>;
  int lane = threadIdx.x;
  auto r = fpsan::amdgcn_global_load_tr_b64_fp8_w64<S, kCC>(
      reinterpret_cast<const V *>(&gmem[lane * SLOTS8]));
  for (int s = 0; s < SLOTS8; ++s)
    out[lane * SLOTS8 + s] = r.get(s).to_storage_bits();
}

template <Semantics S>
__global__ void k_loadtr_wrap_bf8(const std::uint8_t *gmem, std::uint8_t *out) {
  using V = Value<fpsan::fp8_e5m2, S, kCC>;
  int lane = threadIdx.x;
  auto r = fpsan::amdgcn_global_load_tr_b64_bf8_w64<S, kCC>(
      reinterpret_cast<const V *>(&gmem[lane * SLOTS8]));
  for (int s = 0; s < SLOTS8; ++s)
    out[lane * SLOTS8 + s] = r.get(s).to_storage_bits();
}
#endif

namespace {
// Stage a wave-worth of distinct 16-bit patterns in global memory, then run
// `k(gmem, out)`. Returns out[].
std::vector<std::uint16_t> run16(void (*k)(const std::uint16_t *, std::uint16_t *)) {
  const int N = WAVE * SLOTS16;
  std::uint16_t *d_in = nullptr;
  std::uint16_t *d_out = nullptr;
  (void)hipMalloc(&d_in, N * sizeof(std::uint16_t));
  (void)hipMalloc(&d_out, N * sizeof(std::uint16_t));
  std::vector<std::uint16_t> h_in(N);
  for (int lane = 0; lane < WAVE; ++lane)
    for (int s = 0; s < SLOTS16; ++s)
      h_in[lane * SLOTS16 + s] = pat16(lane, s);
  (void)hipMemcpy(d_in, h_in.data(), N * sizeof(std::uint16_t), hipMemcpyHostToDevice);
  k<<<1, WAVE>>>(d_in, d_out);
  (void)hipDeviceSynchronize();
  std::vector<std::uint16_t> h_out(N);
  (void)hipMemcpy(h_out.data(), d_out, N * sizeof(std::uint16_t), hipMemcpyDeviceToHost);
  (void)hipFree(d_in);
  (void)hipFree(d_out);
  return h_out;
}

#if FPSAN_TEST_GLOBAL_LOAD_TR_W64
std::vector<std::uint8_t> run8(void (*k)(const std::uint8_t *, std::uint8_t *)) {
  const int N = WAVE * SLOTS8;
  std::uint8_t *d_in = nullptr;
  std::uint8_t *d_out = nullptr;
  (void)hipMalloc(&d_in, N * sizeof(std::uint8_t));
  (void)hipMalloc(&d_out, N * sizeof(std::uint8_t));
  std::vector<std::uint8_t> h_in(N);
  for (int lane = 0; lane < WAVE; ++lane)
    for (int s = 0; s < SLOTS8; ++s)
      h_in[lane * SLOTS8 + s] = pat8(lane, s);
  (void)hipMemcpy(d_in, h_in.data(), N * sizeof(std::uint8_t), hipMemcpyHostToDevice);
  k<<<1, WAVE>>>(d_in, d_out);
  (void)hipDeviceSynchronize();
  std::vector<std::uint8_t> h_out(N);
  (void)hipMemcpy(h_out.data(), d_out, N * sizeof(std::uint8_t), hipMemcpyDeviceToHost);
  (void)hipFree(d_in);
  (void)hipFree(d_out);
  return h_out;
}
#endif

bool have_device() {
  int n = 0;
  return hipGetDeviceCount(&n) == hipSuccess && n > 0;
}
} // namespace

TEST(GlobalLoadTr, F16_MatchesBuiltinAndFpsanMovesSameBits) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  auto raw = run16(k_loadtr_raw_f16);
  auto flt = run16(k_loadtr_wrap_f16<Semantics::Native>);
  for (size_t i = 0; i < raw.size(); ++i)
    EXPECT_EQ(flt[i], raw[i]) << "Float wrapper != builtin at " << i;
  // Every FPSan-family semantics moves the same bits as the builtin.
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    auto fps = run16(k_loadtr_wrap_f16<decltype(sem)::value>);
    for (size_t i = 0; i < raw.size(); ++i)
      EXPECT_EQ(fps[i], raw[i]) << "FPSan wrapper != builtin at " << i;
  });
}

TEST(GlobalLoadTr, BF16_MatchesBuiltinAndFpsanMovesSameBits) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  auto raw = run16(k_loadtr_raw_bf16);
  auto flt = run16(k_loadtr_wrap_bf16<Semantics::Native>);
  for (size_t i = 0; i < raw.size(); ++i)
    EXPECT_EQ(flt[i], raw[i]) << "Float wrapper != builtin at " << i;
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    auto fps = run16(k_loadtr_wrap_bf16<decltype(sem)::value>);
    for (size_t i = 0; i < raw.size(); ++i)
      EXPECT_EQ(fps[i], raw[i]) << "FPSan wrapper != builtin at " << i;
  });
}

#if FPSAN_TEST_GLOBAL_LOAD_TR_W64
TEST(GlobalLoadTr, FP8W64_MatchesBuiltinAndFpsanMovesSameBits) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  auto raw = run8(k_loadtr_raw_b64);
  auto flt = run8(k_loadtr_wrap_fp8<Semantics::Native>);
  for (size_t i = 0; i < raw.size(); ++i)
    EXPECT_EQ(flt[i], raw[i]) << "Float wrapper != builtin at " << i;
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    auto fps = run8(k_loadtr_wrap_fp8<decltype(sem)::value>);
    for (size_t i = 0; i < raw.size(); ++i)
      EXPECT_EQ(fps[i], raw[i]) << "FPSan wrapper != builtin at " << i;
  });
}

TEST(GlobalLoadTr, BF8W64_MatchesBuiltinAndFpsanMovesSameBits) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  auto raw = run8(k_loadtr_raw_b64);
  auto flt = run8(k_loadtr_wrap_bf8<Semantics::Native>);
  for (size_t i = 0; i < raw.size(); ++i)
    EXPECT_EQ(flt[i], raw[i]) << "Float wrapper != builtin at " << i;
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    auto fps = run8(k_loadtr_wrap_bf8<decltype(sem)::value>);
    for (size_t i = 0; i < raw.size(); ++i)
      EXPECT_EQ(fps[i], raw[i]) << "FPSan wrapper != builtin at " << i;
  });
}
#endif
