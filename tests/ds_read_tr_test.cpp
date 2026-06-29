// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/ds_read_tr_test.cpp
//
// GPU tests for the gfx950 matrix-transposed LDS-read wrappers in
// fpsan/amdgcn_ds.hpp (ds_read_tr16_b64, ds_read_tr8_b64).
//
// These wrappers are thin, bit-faithful shims over the hardware transpose, so
// the test pins exactly that contract without needing to know the transpose
// pattern itself:
//
//   * MatchesBuiltin: the Native-mode wrapper returns the same bits as a direct
//     __builtin_amdgcn_ds_read_tr*_* call over identically-staged LDS.
//   * FpsanMovesSameBits: the FPSan-mode wrapper, given LDS staged with the
//     same bit patterns (as payloads), returns those same bits transposed --
//     i.e. Float and FPSan move bits identically (the whole point: a transpose
//     observes no values, only bits).
//
// Requires real MI350 (gfx950); built only under FPSAN_ENABLE_HIP with gfx950.
#include "fpsan/amdgcn_ds.hpp"
#include "fpsan/fpsan.hpp"

#include "fpsan_semantics.hpp"

#include "hip_test_utils.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

static constexpr Conversions kCC = Conversions::Explicit;
static constexpr int WAVE = 64;

// Deterministic distinct per-(lane,slot) 16-bit pattern. Avoids 0 and stays in
// a range that is a normal f16 bit pattern (doesn't matter for bit movement,
// but keeps the staged values unsurprising).
__device__ __host__ inline std::uint16_t pat16(int lane, int slot) {
  return static_cast<std::uint16_t>(0x3000 + lane * 4 + slot);
}

// ---- tr16: raw builtin reference ------------------------------------------
__global__ void k_tr16_raw(std::uint16_t *out) {
  using v4i16 = short __attribute__((ext_vector_type(4)));
  __shared__ v4i16 lds[WAVE];
  int lane = threadIdx.x;
  v4i16 v;
  for (int s = 0; s < 4; ++s)
    v[s] = static_cast<short>(pat16(lane, s));
  lds[lane] = v;
  __syncthreads();
  v4i16 r{};
#ifdef __HIP_DEVICE_COMPILE__
  r = __builtin_amdgcn_ds_read_tr16_b64_v4i16(&lds[lane]);
#endif
  for (int s = 0; s < 4; ++s)
    out[lane * 4 + s] = static_cast<std::uint16_t>(r[s]);
}

// ---- tr16: fpsan wrapper (f16), both semantics -----------------------------
template <Semantics S> __global__ void k_tr16_wrap(std::uint16_t *out) {
  // Stage raw bits in LDS (Value has a non-trivial ctor, disallowed for
  // __shared__) and reinterpret to Value* -- Value is a bit-for-bit storage
  // wrapper, so the staged bits ARE the lane's payload (FPSan) / float bits.
  using V = Value<_Float16, S, kCC>;
  __shared__ std::uint16_t lds[WAVE * 4];
  int lane = threadIdx.x;
  for (int s = 0; s < 4; ++s)
    lds[lane * 4 + s] = pat16(lane, s);
  __syncthreads();
  auto r = fpsan::amdgcn_ds_read_tr16_b64_f16<S, kCC>(reinterpret_cast<const V *>(&lds[lane * 4]));
  for (int s = 0; s < 4; ++s)
    out[lane * 4 + s] = r.get(s).to_storage_bits();
}

// ---- tr8: raw builtin reference -------------------------------------------
__global__ void k_tr8_raw(std::uint8_t *out) {
  using v2i32 = int __attribute__((ext_vector_type(2)));
  __shared__ v2i32 lds[WAVE];
  int lane = threadIdx.x;
  std::uint8_t bytes[8];
  for (int s = 0; s < 8; ++s)
    bytes[s] = static_cast<std::uint8_t>(0x10 + lane + s);
  v2i32 v;
  __builtin_memcpy(&v, bytes, 8);
  lds[lane] = v;
  __syncthreads();
  v2i32 r{};
#ifdef __HIP_DEVICE_COMPILE__
  r = __builtin_amdgcn_ds_read_tr8_b64_v2i32(&lds[lane]);
#endif
  __builtin_memcpy(out + lane * 8, &r, 8);
}

// ---- tr8: fpsan wrapper (fp8), both semantics ------------------------------
template <Semantics S> __global__ void k_tr8_wrap(std::uint8_t *out) {
  using V = Value<fpsan::fp8_e4m3, S, kCC>;
  __shared__ std::uint8_t lds[WAVE * 8];
  int lane = threadIdx.x;
  for (int s = 0; s < 8; ++s)
    lds[lane * 8 + s] = static_cast<std::uint8_t>(0x10 + lane + s);
  __syncthreads();
  auto r = fpsan::amdgcn_ds_read_tr8_b64_fp8<S, kCC>(reinterpret_cast<const V *>(&lds[lane * 8]));
  for (int s = 0; s < 8; ++s)
    out[lane * 8 + s] = r.get(s).to_storage_bits();
}

// ---- tr4: raw builtin reference -------------------------------------------
// 64 bits per lane = 16 fp4 nibbles, staged here as 2 dwords. The transpose is
// 4-bit-element granular; we only check bit-faithfulness, not the pattern.
__global__ void k_tr4_raw(std::uint32_t *out) {
  using v2i32 = int __attribute__((ext_vector_type(2)));
  __shared__ v2i32 lds[WAVE];
  int lane = threadIdx.x;
  v2i32 v;
  v[0] = static_cast<int>(0x12345670u + lane);
  v[1] = static_cast<int>(0x89abcdefu - lane);
  lds[lane] = v;
  __syncthreads();
  v2i32 r{};
#ifdef __HIP_DEVICE_COMPILE__
  r = __builtin_amdgcn_ds_read_tr4_b64_v2i32(&lds[lane]);
#endif
  out[lane * 2 + 0] = static_cast<std::uint32_t>(r[0]);
  out[lane * 2 + 1] = static_cast<std::uint32_t>(r[1]);
}

// ---- tr4: fpsan wrapper, both semantics ------------------------------------
template <Semantics S> __global__ void k_tr4_wrap(std::uint32_t *out) {
  __shared__ std::uint32_t lds[WAVE * 2];
  int lane = threadIdx.x;
  lds[lane * 2 + 0] = 0x12345670u + lane;
  lds[lane * 2 + 1] = 0x89abcdefu - lane;
  __syncthreads();
  auto r = fpsan::amdgcn_ds_read_tr4_b64<S, kCC>(&lds[lane * 2]);
  out[lane * 2 + 0] = r[0];
  out[lane * 2 + 1] = r[1];
}

// ---- tr6: raw builtin reference -------------------------------------------
// 96 bits per lane = 16 fp6 codes. Stage PACKED (12-byte stride) -- identical
// to the wrapper's layout -- so this pins the wrapper==builtin contract rather
// than an incidental LDS-stride difference (a v3i32[] array would 16-byte-pad
// each lane, a layout the packed fp6 matrix tile never uses).
__global__ void k_tr6_raw(std::uint32_t *out) {
  using v3i32 = int __attribute__((ext_vector_type(3)));
  __shared__ std::uint32_t lds[WAVE * 3];
  int lane = threadIdx.x;
  lds[lane * 3 + 0] = 0x0f1e2d3cu + lane;
  lds[lane * 3 + 1] = 0x4b5a6978u - lane;
  lds[lane * 3 + 2] = 0x8c7d6e5fu + lane * 3;
  __syncthreads();
  v3i32 r{};
#ifdef __HIP_DEVICE_COMPILE__
  r = __builtin_amdgcn_ds_read_tr6_b96_v3i32(
      (v3i32 __attribute__((address_space(3))) *)(&lds[lane * 3]));
#endif
  out[lane * 3 + 0] = static_cast<std::uint32_t>(r[0]);
  out[lane * 3 + 1] = static_cast<std::uint32_t>(r[1]);
  out[lane * 3 + 2] = static_cast<std::uint32_t>(r[2]);
}

// ---- tr6: fpsan wrapper, both semantics ------------------------------------
template <Semantics S> __global__ void k_tr6_wrap(std::uint32_t *out) {
  __shared__ std::uint32_t lds[WAVE * 3];
  int lane = threadIdx.x;
  lds[lane * 3 + 0] = 0x0f1e2d3cu + lane;
  lds[lane * 3 + 1] = 0x4b5a6978u - lane;
  lds[lane * 3 + 2] = 0x8c7d6e5fu + lane * 3;
  __syncthreads();
  auto r = fpsan::amdgcn_ds_read_tr6_b96<S, kCC>(&lds[lane * 3]);
  out[lane * 3 + 0] = r[0];
  out[lane * 3 + 1] = r[1];
  out[lane * 3 + 2] = r[2];
}

namespace {
template <class T> std::vector<T> run(void (*k)(T *), int n) {
  T *d = nullptr;
  (void)hipMalloc(&d, n * sizeof(T));
  k<<<1, WAVE>>>(d);
  (void)hipDeviceSynchronize();
  std::vector<T> h(n);
  (void)hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost);
  (void)hipFree(d);
  return h;
}
} // namespace

TEST(DsReadTr16, MatchesBuiltinAndFpsanMovesSameBits) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  const int N = WAVE * 4;
  auto raw = run<std::uint16_t>(k_tr16_raw, N);
  auto flt = run<std::uint16_t>(k_tr16_wrap<Semantics::Native>, N);
  for (int i = 0; i < N; ++i)
    EXPECT_EQ(flt[i], raw[i]) << "Float wrapper != builtin at " << i;
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    auto fps = run<std::uint16_t>(k_tr16_wrap<S>, N);
    for (int i = 0; i < N; ++i)
      EXPECT_EQ(fps[i], raw[i]) << "FPSan wrapper != builtin at " << i;
  });
}

TEST(DsReadTr8, MatchesBuiltinAndFpsanMovesSameBits) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  const int N = WAVE * 8;
  auto raw = run<std::uint8_t>(k_tr8_raw, N);
  auto flt = run<std::uint8_t>(k_tr8_wrap<Semantics::Native>, N);
  for (int i = 0; i < N; ++i)
    EXPECT_EQ(flt[i], raw[i]) << "Float wrapper != builtin at " << i;
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    auto fps = run<std::uint8_t>(k_tr8_wrap<S>, N);
    for (int i = 0; i < N; ++i)
      EXPECT_EQ(fps[i], raw[i]) << "FPSan wrapper != builtin at " << i;
  });
}

TEST(DsReadTr4, MatchesBuiltinAndFpsanMovesSameBits) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  const int N = WAVE * 2;
  auto raw = run<std::uint32_t>(k_tr4_raw, N);
  auto flt = run<std::uint32_t>(k_tr4_wrap<Semantics::Native>, N);
  for (int i = 0; i < N; ++i)
    EXPECT_EQ(flt[i], raw[i]) << "Float wrapper != builtin at " << i;
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    auto fps = run<std::uint32_t>(k_tr4_wrap<S>, N);
    for (int i = 0; i < N; ++i)
      EXPECT_EQ(fps[i], raw[i]) << "FPSan wrapper != builtin at " << i;
  });
}

TEST(DsReadTr6, MatchesBuiltinAndFpsanMovesSameBits) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  const int N = WAVE * 3;
  auto raw = run<std::uint32_t>(k_tr6_raw, N);
  auto flt = run<std::uint32_t>(k_tr6_wrap<Semantics::Native>, N);
  for (int i = 0; i < N; ++i)
    EXPECT_EQ(flt[i], raw[i]) << "Float wrapper != builtin at " << i;
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    auto fps = run<std::uint32_t>(k_tr6_wrap<S>, N);
    for (int i = 0; i < N; ++i)
      EXPECT_EQ(fps[i], raw[i]) << "FPSan wrapper != builtin at " << i;
  });
}
