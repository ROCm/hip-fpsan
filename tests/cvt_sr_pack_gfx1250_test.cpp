// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/cvt_sr_pack_gfx1250_test.cpp
//
// Regression + correctness test for the stochastic-rounding fp8 PACK wrappers
// cvt_sr_{fp8,bf8}_{f32,f16} in fpsan/amdgcn_cvt.hpp, specifically their
// `old`-register preservation and byte_sel placement.
//
// These wrappers exist for the whole fp8-conversion family; their prior test
// (cvt_gfx950_test) only ever passed old == 0 and read back the single written
// byte, which cannot observe the builtin argument order. The clang builtin
// signature is (val, SEED, OLD, byte_sel) -- see
// clang/test/CodeGenOpenCL/builtins-amdgcn-fp8.cl and ...-gfx1250.cl -- so a
// wrapper that forwarded (val, old, seed, sel) would silently swap seed/old and
// corrupt the standard idiom of packing 4 fp8 bytes into one register via 4
// successive sr calls (each feeding the previous result in as `old`). This test
// locks the correct behavior:
//   * 4 chained sr calls (byte_sel 0..3) must build a register whose 4 bytes
//     decode back to the 4 inputs (Native mode, host-referenced).
//   * a single sr call must preserve the other 3 bytes of a nonzero `old`.
// All references are host-computed and independent of the device builtin.
#include "fpsan/amdgcn_cvt.hpp"
#include "fpsan/fpsan.hpp"

#include "hip_test_utils.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

static constexpr Conversions kCC = Conversions::Explicit;
using FVF = Value<float, Semantics::Native, kCC>;

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_sr_fp8_f32)

// Chain 4 sr_fp8_f32 calls (byte_sel 0..3) starting from old=0, then return the
// packed register. Correct old-forwarding accumulates all 4 bytes.
__global__ void k_chain_fp8_f32(float v0, float v1, float v2, float v3, int *out) {
  int p = 0;
  p = fpsan::amdgcn_cvt_sr_fp8_f32<0, Semantics::Native, kCC>(FVF{v0}, p, 0x111u);
  p = fpsan::amdgcn_cvt_sr_fp8_f32<1, Semantics::Native, kCC>(FVF{v1}, p, 0x222u);
  p = fpsan::amdgcn_cvt_sr_fp8_f32<2, Semantics::Native, kCC>(FVF{v2}, p, 0x333u);
  p = fpsan::amdgcn_cvt_sr_fp8_f32<3, Semantics::Native, kCC>(FVF{v3}, p, 0x444u);
  *out = p;
}
__global__ void k_chain_bf8_f32(float v0, float v1, float v2, float v3, int *out) {
  int p = 0;
  p = fpsan::amdgcn_cvt_sr_bf8_f32<0, Semantics::Native, kCC>(FVF{v0}, p, 0x111u);
  p = fpsan::amdgcn_cvt_sr_bf8_f32<1, Semantics::Native, kCC>(FVF{v1}, p, 0x222u);
  p = fpsan::amdgcn_cvt_sr_bf8_f32<2, Semantics::Native, kCC>(FVF{v2}, p, 0x333u);
  p = fpsan::amdgcn_cvt_sr_bf8_f32<3, Semantics::Native, kCC>(FVF{v3}, p, 0x444u);
  *out = p;
}

template <class FP8> void run_chain(void (*k)(float, float, float, float, int *)) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  // Four distinct values exactly representable in both fp8 formats.
  const float v[4] = {1.0f, -2.0f, 0.5f, 3.0f};
  int *dO;
  HIP_CHECK(hipMalloc(&dO, sizeof(int)));
  k<<<1, 1>>>(v[0], v[1], v[2], v[3], dO);
  HIP_CHECK(hipDeviceSynchronize());
  std::uint32_t got = static_cast<std::uint32_t>(from_dev(dO, 1)[0]);
  for (int i = 0; i < 4; ++i) {
    std::uint32_t want = FP8(v[i]).bits;
    std::uint32_t have = (got >> (8 * i)) & 0xFFu;
    EXPECT_EQ(have, want) << "byte " << i << " (val " << v[i] << ")";
  }
  (void)hipFree(dO);
}

TEST(CvtSrPack, ChainFp8F32) { run_chain<fpsan::fp8_e4m3>(k_chain_fp8_f32); }
TEST(CvtSrPack, ChainBf8F32) { run_chain<fpsan::fp8_e5m2>(k_chain_bf8_f32); }

// Single sr call must preserve the other 3 bytes of a nonzero `old`.
__global__ void k_preserve_fp8_f32(int old, float v, unsigned seed, int *out) {
  *out = fpsan::amdgcn_cvt_sr_fp8_f32<2, Semantics::Native, kCC>(FVF{v}, old, seed);
}

TEST(CvtSrPack, PreservesOldFp8F32) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  const int old = static_cast<int>(0xAABBCCDDu);
  int *dO;
  HIP_CHECK(hipMalloc(&dO, sizeof(int)));
  const float v = 3.0f; // exact in fp8
  k_preserve_fp8_f32<<<1, 1>>>(old, v, 0x9999u, dO);
  HIP_CHECK(hipDeviceSynchronize());
  std::uint32_t got = static_cast<std::uint32_t>(from_dev(dO, 1)[0]);
  std::uint32_t enc = fpsan::fp8_e4m3(v).bits;
  std::uint32_t want = (static_cast<std::uint32_t>(old) & ~(0xFFu << 16)) | (enc << 16);
  EXPECT_EQ(got, want);
  (void)hipFree(dO);
}

#endif // cvt_sr_fp8_f32

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_sr_fp8_f16)

__global__ void k_chain_fp8_f16(_Float16 v0, _Float16 v1, _Float16 v2, _Float16 v3, int *out) {
  using H = Value<_Float16, Semantics::Native, kCC>;
  int p = 0;
  p = fpsan::amdgcn_cvt_sr_fp8_f16<0, Semantics::Native, kCC>(H{v0}, p, 0x111u);
  p = fpsan::amdgcn_cvt_sr_fp8_f16<1, Semantics::Native, kCC>(H{v1}, p, 0x222u);
  p = fpsan::amdgcn_cvt_sr_fp8_f16<2, Semantics::Native, kCC>(H{v2}, p, 0x333u);
  p = fpsan::amdgcn_cvt_sr_fp8_f16<3, Semantics::Native, kCC>(H{v3}, p, 0x444u);
  *out = p;
}
__global__ void k_chain_bf8_f16(_Float16 v0, _Float16 v1, _Float16 v2, _Float16 v3, int *out) {
  using H = Value<_Float16, Semantics::Native, kCC>;
  int p = 0;
  p = fpsan::amdgcn_cvt_sr_bf8_f16<0, Semantics::Native, kCC>(H{v0}, p, 0x111u);
  p = fpsan::amdgcn_cvt_sr_bf8_f16<1, Semantics::Native, kCC>(H{v1}, p, 0x222u);
  p = fpsan::amdgcn_cvt_sr_bf8_f16<2, Semantics::Native, kCC>(H{v2}, p, 0x333u);
  p = fpsan::amdgcn_cvt_sr_bf8_f16<3, Semantics::Native, kCC>(H{v3}, p, 0x444u);
  *out = p;
}

template <class FP8> void run_chain_f16(void (*k)(_Float16, _Float16, _Float16, _Float16, int *)) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  const _Float16 v[4] = {static_cast<_Float16>(1.0f), static_cast<_Float16>(-2.0f),
                         static_cast<_Float16>(0.5f), static_cast<_Float16>(3.0f)};
  int *dO;
  HIP_CHECK(hipMalloc(&dO, sizeof(int)));
  k<<<1, 1>>>(v[0], v[1], v[2], v[3], dO);
  HIP_CHECK(hipDeviceSynchronize());
  std::uint32_t got = static_cast<std::uint32_t>(from_dev(dO, 1)[0]);
  for (int i = 0; i < 4; ++i) {
    std::uint32_t want = FP8(static_cast<float>(v[i])).bits;
    std::uint32_t have = (got >> (8 * i)) & 0xFFu;
    EXPECT_EQ(have, want) << "byte " << i;
  }
  (void)hipFree(dO);
}

TEST(CvtSrPack, ChainFp8F16) { run_chain_f16<fpsan::fp8_e4m3>(k_chain_fp8_f16); }
TEST(CvtSrPack, ChainBf8F16) { run_chain_f16<fpsan::fp8_e5m2>(k_chain_bf8_f16); }

#endif // cvt_sr_fp8_f16
