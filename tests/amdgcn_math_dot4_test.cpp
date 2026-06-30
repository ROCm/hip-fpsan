// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/amdgcn_math_test.cpp
//
// GPU tests for the AMD math intrinsic wrappers in fpsan/amdgcn_math.hpp. The
// scalar math cases (rcp/rsq/sqrt/sin/cos/log/exp2/fract) are shared across the
// gfx12 family (RDNA4 and gfx1250), gfx94x, and gfx950. Two properties per
// scalar wrapper:
//   - Native mode forwards to the builtin (the wrapper produces the same bits
//     as __builtin_amdgcn_{rcp,rsq,...}f directly).
//   - FPSan-family semantics match fpsan::{rcp,rsqrt,...} payload-for-payload
//     in the same semantics.
//
// The dot-product cases (fdot2 / dot4 fp8) are gated by compile definitions
// (FPSAN_TEST_ENABLE_FDOT2 / FPSAN_TEST_ENABLE_GFX12_DOT_MATH) to the families
// whose builtins lower: the dot9-12 families exist on RDNA4 and CDNA3 but not
// on gfx1250, so they are excluded from the gfx1250 build.
#include "fpsan/amdgcn_math.hpp"
#include "fpsan/fpsan.hpp"

#include "fpsan_semantics.hpp"

#include "hip_test_utils.hpp"
#include "test_random.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

[[maybe_unused]] static constexpr Conversions kCC = Conversions::Explicit;
static constexpr int kScalarN = 32;

#ifndef FPSAN_TEST_ENABLE_FDOT2
#define FPSAN_TEST_ENABLE_FDOT2 0
#endif

#ifndef FPSAN_TEST_ENABLE_FDOT2_EXTENDED
#define FPSAN_TEST_ENABLE_FDOT2_EXTENDED 0
#endif

#ifndef FPSAN_TEST_ENABLE_GFX12_DOT_MATH
#define FPSAN_TEST_ENABLE_GFX12_DOT_MATH 0
#endif

#if defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx1202__) || defined(__gfx1250__)
#define FPSAN_TEST_DEVICE_IS_GFX12 1
#else
#define FPSAN_TEST_DEVICE_IS_GFX12 0
#endif

namespace {
template <class FT> std::vector<FT> make_positive_inputs() {
  std::vector<FT> v(kScalarN);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : v)
    x = fpsan_test::pick_quarter<FT>(rng, 4, 36);
  return v;
}

template <class FT> std::vector<FT> make_signed_inputs() {
  std::vector<FT> v(kScalarN);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : v)
    x = fpsan_test::pick_quarter<FT>(rng, -16, 16);
  return v;
}

std::string current_arch() {
  int device = 0;
  if (hipGetDevice(&device) != hipSuccess)
    return {};
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, device) != hipSuccess)
    return {};
  return prop.gcnArchName;
}

bool starts_with(const std::string &value, const char *prefix) {
  return value.rfind(prefix, 0) == 0;
}

[[maybe_unused]] bool current_arch_is_gfx12() { return starts_with(current_arch(), "gfx12"); }

[[maybe_unused]] bool current_arch_supports_fdot2() {
  const auto arch = current_arch();
  return starts_with(arch, "gfx11") || starts_with(arch, "gfx12") || starts_with(arch, "gfx940") ||
         starts_with(arch, "gfx941") || starts_with(arch, "gfx942");
}

[[maybe_unused]] bool current_arch_supports_fdot2_extended() {
  const auto arch = current_arch();
  return starts_with(arch, "gfx11") || starts_with(arch, "gfx12");
}

template <class T> std::uint32_t bits_u32(T v) {
  std::uint32_t u = 0;
  std::memcpy(&u, &v, sizeof v);
  return u;
}

} // namespace

// dot4 fp8 / bf8 family (gfx12 dot11-insts): 4-element 8-bit dot product.
// Each A and B is a v4 of fp8 (e4m3) or bf8 (e5m2) packed as 32 bits per lane.
// Native-mode: wrapper bit-exact equals direct builtin.  FPSan-mode: wrapper
// payload equals explicit ring expression `acc + sum cast<f32>(a[k]) *
// cast<f32>(b[k])`.
// ============================================================================
#if FPSAN_TEST_ENABLE_GFX12_DOT_MATH &&                                                            \
    (!defined(__HIP_DEVICE_COMPILE__) || FPSAN_TEST_DEVICE_IS_GFX12)
using v4e4 = fpsan::v4e4m3_native;
using v4e5 = fpsan::v4e5m2_native;

#define DOT4_PAIR_KERNEL(NAME, AV, BV, BUILTIN)                                                    \
  template <Semantics S>                                                                           \
  __global__ void k_##NAME##_pair(const unsigned *a, const unsigned *b, const float *c,            \
                                  float *direct, float *wrapper, std::uint32_t *pay_direct,        \
                                  std::uint32_t *pay_wrapper) {                                    \
    int i = threadIdx.x;                                                                           \
    unsigned ai = a[i], bi = b[i];                                                                 \
    float ci = c[i];                                                                               \
    direct[i] = BUILTIN(ai, bi, ci);                                                               \
    AV av = __builtin_bit_cast(AV, ai);                                                            \
    BV bv = __builtin_bit_cast(BV, bi);                                                            \
    Value<AV, Semantics::Native, kCC> avF{av};                                                     \
    Value<BV, Semantics::Native, kCC> bvF{bv};                                                     \
    Value<float, Semantics::Native, kCC> cF{ci};                                                   \
    wrapper[i] = static_cast<float>(fpsan::NAME<Semantics::Native, kCC>(avF, bvF, cF));            \
    Value<AV, S, kCC> avP{av};                                                                     \
    Value<BV, S, kCC> bvP{bv};                                                                     \
    Value<float, S, kCC> cP{ci};                                                                   \
    auto expanded = cP;                                                                            \
    for (int k = 0; k < 4; ++k)                                                                    \
      expanded = expanded + fpsan::cast<float>(avP.get(k)) * fpsan::cast<float>(bvP.get(k));       \
    pay_direct[i] = expanded.fpsan_payload();                                                      \
    pay_wrapper[i] = fpsan::NAME<S, kCC>(avP, bvP, cP).fpsan_payload();                            \
  }

DOT4_PAIR_KERNEL(amdgcn_dot4_f32_fp8_fp8, v4e4, v4e4, __builtin_amdgcn_dot4_f32_fp8_fp8)
DOT4_PAIR_KERNEL(amdgcn_dot4_f32_fp8_bf8, v4e4, v4e5, __builtin_amdgcn_dot4_f32_fp8_bf8)
DOT4_PAIR_KERNEL(amdgcn_dot4_f32_bf8_fp8, v4e5, v4e4, __builtin_amdgcn_dot4_f32_bf8_fp8)
DOT4_PAIR_KERNEL(amdgcn_dot4_f32_bf8_bf8, v4e5, v4e5, __builtin_amdgcn_dot4_f32_bf8_bf8)
#undef DOT4_PAIR_KERNEL

namespace {
constexpr int kDot4N = 32;

std::vector<unsigned> make_packed_u32() {
  std::vector<unsigned> v(kDot4N);
  std::mt19937 rng = fpsan_test::make_rng();
  std::uniform_int_distribution<std::uint32_t> dist;
  for (auto &x : v)
    x = dist(rng);
  return v;
}

std::vector<float> make_dot4_acc() {
  std::vector<float> v(kDot4N);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : v)
    x = fpsan_test::pick_quarter<float>(rng, -4, 4);
  return v;
}
} // namespace

#define DOT4_TEST(NAME)                                                                            \
  TEST(AmdgcnMath, NAME##_FloatAndFpsan) {                                                         \
    if (!have_device())                                                                            \
      GTEST_SKIP() << "no HIP device";                                                             \
    if (!current_arch_is_gfx12())                                                                  \
      GTEST_SKIP() << "current device is not gfx12";                                               \
    auto a = make_packed_u32();                                                                    \
    auto b = make_packed_u32();                                                                    \
    auto c = make_dot4_acc();                                                                      \
    unsigned *dA = to_dev(a), *dB = to_dev(b);                                                     \
    float *dC = to_dev(c);                                                                         \
    float *dDir, *dWrap;                                                                           \
    std::uint32_t *dPdir, *dPwrap;                                                                 \
    HIP_CHECK(hipMalloc(&dDir, kDot4N * sizeof(float)));                                           \
    HIP_CHECK(hipMalloc(&dWrap, kDot4N * sizeof(float)));                                          \
    HIP_CHECK(hipMalloc(&dPdir, kDot4N * sizeof(std::uint32_t)));                                  \
    HIP_CHECK(hipMalloc(&dPwrap, kDot4N * sizeof(std::uint32_t)));                                 \
    fpsan_test::for_each_fpsan_semantics([&](auto sem) {                                           \
      k_##NAME##_pair<decltype(sem)::value>                                                        \
          <<<1, kDot4N>>>(dA, dB, dC, dDir, dWrap, dPdir, dPwrap);                                 \
      HIP_CHECK(hipDeviceSynchronize());                                                           \
      std::vector<float> dir(kDot4N), wrap(kDot4N);                                                \
      std::vector<std::uint32_t> pdir(kDot4N), pwrap(kDot4N);                                      \
      HIP_CHECK(hipMemcpy(dir.data(), dDir, kDot4N * sizeof(float), hipMemcpyDeviceToHost));       \
      HIP_CHECK(hipMemcpy(wrap.data(), dWrap, kDot4N * sizeof(float), hipMemcpyDeviceToHost));     \
      HIP_CHECK(                                                                                   \
          hipMemcpy(pdir.data(), dPdir, kDot4N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));   \
      HIP_CHECK(                                                                                   \
          hipMemcpy(pwrap.data(), dPwrap, kDot4N * sizeof(std::uint32_t), hipMemcpyDeviceToHost)); \
      for (int i = 0; i < kDot4N; ++i) {                                                           \
        EXPECT_EQ(bits_u32(wrap[i]), bits_u32(dir[i])) << "Float lane " << i;                      \
        EXPECT_EQ(pwrap[i], pdir[i]) << "FPSan lane " << i;                                        \
      }                                                                                            \
    });                                                                                            \
    (void)hipFree(dA);                                                                             \
    (void)hipFree(dB);                                                                             \
    (void)hipFree(dC);                                                                             \
    (void)hipFree(dDir);                                                                           \
    (void)hipFree(dWrap);                                                                          \
    (void)hipFree(dPdir);                                                                          \
    (void)hipFree(dPwrap);                                                                         \
  }

DOT4_TEST(amdgcn_dot4_f32_fp8_fp8)
DOT4_TEST(amdgcn_dot4_f32_fp8_bf8)
DOT4_TEST(amdgcn_dot4_f32_bf8_fp8)
DOT4_TEST(amdgcn_dot4_f32_bf8_bf8)
#undef DOT4_TEST
#endif
