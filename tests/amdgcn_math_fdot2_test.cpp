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

static constexpr Conversions kCC = Conversions::Explicit;
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

// fdot2 family.  Two properties per wrapper:
//   - Native mode bit-exact matches the underlying builtin.
//   - FPSan-family semantics match the expanded `acc + cast(a0)*cast(b0) +
//     cast(a1)*cast(b1)` expression (payload-for-payload). We compute the
//     expanded expression in the same kernel using FPSan operations, so the test
//     exercises the wrapper's contract rather than re-deriving the payload
//     arithmetic host-side.
// ============================================================================

using v2h = _Float16 __attribute__((ext_vector_type(2)));
using v2bf = __bf16 __attribute__((ext_vector_type(2)));
using v2i16 = short __attribute__((ext_vector_type(2)));

// ---- fdot2: v2h x v2h -> f32 -----------------------------------------------
#if FPSAN_TEST_ENABLE_FDOT2 &&                                                                     \
    (!defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_fdot2))
template <Semantics S>
__global__ void k_fdot2_pair(const v2h *a, const v2h *b, const float *c, float *direct,
                             float *wrapper, std::uint32_t *pay_direct,
                             std::uint32_t *pay_wrapper) {
  int i = threadIdx.x;
  v2h ai = a[i], bi = b[i];
  float ci = c[i];
  direct[i] = __builtin_amdgcn_fdot2(ai, bi, ci, false);
  Value<v2h, Semantics::Native, kCC> va{ai}, vb{bi};
  Value<float, Semantics::Native, kCC> vc{ci};
  wrapper[i] = static_cast<float>(fpsan::amdgcn_fdot2<false, Semantics::Native, kCC>(va, vb, vc));
  Value<v2h, S, kCC> vap{ai}, vbp{bi};
  Value<float, S, kCC> vcp{ci};
  auto expanded = vcp + fpsan::cast<float>(vap.get(0)) * fpsan::cast<float>(vbp.get(0)) +
                  fpsan::cast<float>(vap.get(1)) * fpsan::cast<float>(vbp.get(1));
  pay_direct[i] = expanded.fpsan_payload();
  pay_wrapper[i] = fpsan::amdgcn_fdot2<false, S, kCC>(vap, vbp, vcp).fpsan_payload();
}
#endif

// ---- fdot2_f16_f16: v2h x v2h -> f16 ---------------------------------------
#if FPSAN_TEST_ENABLE_FDOT2_EXTENDED &&                                                            \
    (!defined(__HIP_DEVICE_COMPILE__) || (__has_builtin(__builtin_amdgcn_fdot2_f16_f16) &&         \
                                          __has_builtin(__builtin_amdgcn_fdot2_bf16_bf16) &&       \
                                          __has_builtin(__builtin_amdgcn_fdot2_f32_bf16)))
template <Semantics S>
__global__ void k_fdot2_f16_f16_pair(const v2h *a, const v2h *b, const _Float16 *c,
                                     _Float16 *direct, _Float16 *wrapper, std::uint16_t *pay_direct,
                                     std::uint16_t *pay_wrapper) {
  int i = threadIdx.x;
  v2h ai = a[i], bi = b[i];
  _Float16 ci = c[i];
  direct[i] = __builtin_amdgcn_fdot2_f16_f16(ai, bi, ci);
  Value<v2h, Semantics::Native, kCC> va{ai}, vb{bi};
  Value<_Float16, Semantics::Native, kCC> vc{ci};
  wrapper[i] =
      static_cast<_Float16>(fpsan::amdgcn_fdot2_f16_f16<Semantics::Native, kCC>(va, vb, vc));
  Value<v2h, S, kCC> vap{ai}, vbp{bi};
  Value<_Float16, S, kCC> vcp{ci};
  auto expanded = vcp + vap.get(0) * vbp.get(0) + vap.get(1) * vbp.get(1);
  pay_direct[i] = static_cast<std::uint16_t>(expanded.fpsan_payload());
  pay_wrapper[i] = static_cast<std::uint16_t>(
      fpsan::amdgcn_fdot2_f16_f16<S, kCC>(vap, vbp, vcp).fpsan_payload());
}

// ---- fdot2_bf16_bf16: v2bf x v2bf -> bf16 ----------------------------------
template <Semantics S>
__global__ void k_fdot2_bf16_bf16_pair(const v2bf *a, const v2bf *b, const __bf16 *c,
                                       __bf16 *direct, __bf16 *wrapper, std::uint16_t *pay_direct,
                                       std::uint16_t *pay_wrapper) {
  int i = threadIdx.x;
  v2bf ai = a[i], bi = b[i];
  __bf16 ci = c[i];
  v2i16 a_i = __builtin_bit_cast(v2i16, ai);
  v2i16 b_i = __builtin_bit_cast(v2i16, bi);
  short c_i = __builtin_bit_cast(short, ci);
  short d_i = __builtin_amdgcn_fdot2_bf16_bf16(a_i, b_i, c_i);
  direct[i] = __builtin_bit_cast(__bf16, d_i);
  Value<v2bf, Semantics::Native, kCC> va{ai}, vb{bi};
  Value<__bf16, Semantics::Native, kCC> vc{ci};
  wrapper[i] =
      static_cast<__bf16>(fpsan::amdgcn_fdot2_bf16_bf16<Semantics::Native, kCC>(va, vb, vc));
  Value<v2bf, S, kCC> vap{ai}, vbp{bi};
  Value<__bf16, S, kCC> vcp{ci};
  auto expanded = vcp + vap.get(0) * vbp.get(0) + vap.get(1) * vbp.get(1);
  pay_direct[i] = static_cast<std::uint16_t>(expanded.fpsan_payload());
  pay_wrapper[i] = static_cast<std::uint16_t>(
      fpsan::amdgcn_fdot2_bf16_bf16<S, kCC>(vap, vbp, vcp).fpsan_payload());
}

// ---- fdot2_f32_bf16: v2bf x v2bf -> f32 ------------------------------------
template <Semantics S>
__global__ void k_fdot2_f32_bf16_pair(const v2bf *a, const v2bf *b, const float *c, float *direct,
                                      float *wrapper, std::uint32_t *pay_direct,
                                      std::uint32_t *pay_wrapper) {
  int i = threadIdx.x;
  v2bf ai = a[i], bi = b[i];
  float ci = c[i];
  v2i16 a_i = __builtin_bit_cast(v2i16, ai);
  v2i16 b_i = __builtin_bit_cast(v2i16, bi);
  direct[i] = __builtin_amdgcn_fdot2_f32_bf16(a_i, b_i, ci, false);
  Value<v2bf, Semantics::Native, kCC> va{ai}, vb{bi};
  Value<float, Semantics::Native, kCC> vc{ci};
  wrapper[i] =
      static_cast<float>(fpsan::amdgcn_fdot2_f32_bf16<false, Semantics::Native, kCC>(va, vb, vc));
  Value<v2bf, S, kCC> vap{ai}, vbp{bi};
  Value<float, S, kCC> vcp{ci};
  auto expanded = vcp + fpsan::cast<float>(vap.get(0)) * fpsan::cast<float>(vbp.get(0)) +
                  fpsan::cast<float>(vap.get(1)) * fpsan::cast<float>(vbp.get(1));
  pay_direct[i] = expanded.fpsan_payload();
  pay_wrapper[i] = fpsan::amdgcn_fdot2_f32_bf16<false, S, kCC>(vap, vbp, vcp).fpsan_payload();
}
#endif

namespace {
constexpr int kFDot2N = 32;

[[maybe_unused]] std::vector<v2h> make_v2h() {
  std::vector<v2h> v(kFDot2N);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : v) {
    float a = fpsan_test::pick_quarter<float>(rng, -8, 8);
    float b = fpsan_test::pick_quarter<float>(rng, -8, 8);
    x = v2h{static_cast<_Float16>(a), static_cast<_Float16>(b)};
  }
  return v;
}

[[maybe_unused]] std::vector<v2bf> make_v2bf() {
  std::vector<v2bf> v(kFDot2N);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : v) {
    float a = fpsan_test::pick_quarter<float>(rng, -8, 8);
    float b = fpsan_test::pick_quarter<float>(rng, -8, 8);
    x = v2bf{static_cast<__bf16>(a), static_cast<__bf16>(b)};
  }
  return v;
}

[[maybe_unused]] std::vector<float> make_acc_f32() {
  std::vector<float> v(kFDot2N);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : v)
    x = fpsan_test::pick_quarter<float>(rng, -4, 4);
  return v;
}

[[maybe_unused]] std::vector<_Float16> make_acc_f16() {
  std::vector<_Float16> v(kFDot2N);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : v)
    x = static_cast<_Float16>(fpsan_test::pick_quarter<float>(rng, -4, 4));
  return v;
}
} // namespace

#if FPSAN_TEST_ENABLE_FDOT2 &&                                                                     \
    (!defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_fdot2))
TEST(AmdgcnMath, fdot2_FloatAndFpsan) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  if (!current_arch_supports_fdot2())
    GTEST_SKIP() << "current device does not support fdot2";
  auto a = make_v2h(), b = make_v2h();
  auto c = make_acc_f32();
  v2h *dA = to_dev(a), *dB = to_dev(b);
  float *dC = to_dev(c);
  float *dDir, *dWrap;
  std::uint32_t *dPdir, *dPwrap;
  HIP_CHECK(hipMalloc(&dDir, kFDot2N * sizeof(float)));
  HIP_CHECK(hipMalloc(&dWrap, kFDot2N * sizeof(float)));
  HIP_CHECK(hipMalloc(&dPdir, kFDot2N * sizeof(std::uint32_t)));
  HIP_CHECK(hipMalloc(&dPwrap, kFDot2N * sizeof(std::uint32_t)));
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    k_fdot2_pair<decltype(sem)::value><<<1, kFDot2N>>>(dA, dB, dC, dDir, dWrap, dPdir, dPwrap);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> dir(kFDot2N), wrap(kFDot2N);
    std::vector<std::uint32_t> pdir(kFDot2N), pwrap(kFDot2N);
    HIP_CHECK(hipMemcpy(dir.data(), dDir, kFDot2N * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(wrap.data(), dWrap, kFDot2N * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(pdir.data(), dPdir, kFDot2N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(pwrap.data(), dPwrap, kFDot2N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for (int i = 0; i < kFDot2N; ++i) {
      EXPECT_EQ(bits_u32(wrap[i]), bits_u32(dir[i])) << "Float lane " << i;
      EXPECT_EQ(pwrap[i], pdir[i]) << "FPSan lane " << i;
    }
  });
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dDir);
  (void)hipFree(dWrap);
  (void)hipFree(dPdir);
  (void)hipFree(dPwrap);
}
#endif

#if FPSAN_TEST_ENABLE_FDOT2_EXTENDED &&                                                            \
    (!defined(__HIP_DEVICE_COMPILE__) || (__has_builtin(__builtin_amdgcn_fdot2_f16_f16) &&         \
                                          __has_builtin(__builtin_amdgcn_fdot2_bf16_bf16) &&       \
                                          __has_builtin(__builtin_amdgcn_fdot2_f32_bf16)))
TEST(AmdgcnMath, fdot2_f16_f16_FloatAndFpsan) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  if (!current_arch_supports_fdot2_extended())
    GTEST_SKIP() << "current device does not support extended fdot2";
  auto a = make_v2h(), b = make_v2h();
  auto c = make_acc_f16();
  v2h *dA = to_dev(a), *dB = to_dev(b);
  _Float16 *dC = to_dev(c);
  _Float16 *dDir, *dWrap;
  std::uint16_t *dPdir, *dPwrap;
  HIP_CHECK(hipMalloc(&dDir, kFDot2N * sizeof(_Float16)));
  HIP_CHECK(hipMalloc(&dWrap, kFDot2N * sizeof(_Float16)));
  HIP_CHECK(hipMalloc(&dPdir, kFDot2N * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&dPwrap, kFDot2N * sizeof(std::uint16_t)));
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    k_fdot2_f16_f16_pair<decltype(sem)::value>
        <<<1, kFDot2N>>>(dA, dB, dC, dDir, dWrap, dPdir, dPwrap);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<_Float16> dir(kFDot2N), wrap(kFDot2N);
    std::vector<std::uint16_t> pdir(kFDot2N), pwrap(kFDot2N);
    HIP_CHECK(hipMemcpy(dir.data(), dDir, kFDot2N * sizeof(_Float16), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(wrap.data(), dWrap, kFDot2N * sizeof(_Float16), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(pdir.data(), dPdir, kFDot2N * sizeof(std::uint16_t), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(pwrap.data(), dPwrap, kFDot2N * sizeof(std::uint16_t), hipMemcpyDeviceToHost));
    for (int i = 0; i < kFDot2N; ++i) {
      std::uint16_t bw = 0, bd = 0;
      std::memcpy(&bw, &wrap[i], sizeof bw);
      std::memcpy(&bd, &dir[i], sizeof bd);
      EXPECT_EQ(bw, bd) << "Float lane " << i;
      EXPECT_EQ(pwrap[i], pdir[i]) << "FPSan lane " << i;
    }
  });
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dDir);
  (void)hipFree(dWrap);
  (void)hipFree(dPdir);
  (void)hipFree(dPwrap);
}

TEST(AmdgcnMath, fdot2_bf16_bf16_FloatAndFpsan) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  if (!current_arch_supports_fdot2_extended())
    GTEST_SKIP() << "current device does not support extended fdot2";
  auto a = make_v2bf(), b = make_v2bf();
  auto cf = make_acc_f32();
  std::vector<__bf16> c(kFDot2N);
  for (int i = 0; i < kFDot2N; ++i)
    c[i] = static_cast<__bf16>(cf[i]);
  v2bf *dA = to_dev(a), *dB = to_dev(b);
  __bf16 *dC = to_dev(c);
  __bf16 *dDir, *dWrap;
  std::uint16_t *dPdir, *dPwrap;
  HIP_CHECK(hipMalloc(&dDir, kFDot2N * sizeof(__bf16)));
  HIP_CHECK(hipMalloc(&dWrap, kFDot2N * sizeof(__bf16)));
  HIP_CHECK(hipMalloc(&dPdir, kFDot2N * sizeof(std::uint16_t)));
  HIP_CHECK(hipMalloc(&dPwrap, kFDot2N * sizeof(std::uint16_t)));
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    k_fdot2_bf16_bf16_pair<decltype(sem)::value>
        <<<1, kFDot2N>>>(dA, dB, dC, dDir, dWrap, dPdir, dPwrap);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<__bf16> dir(kFDot2N), wrap(kFDot2N);
    std::vector<std::uint16_t> pdir(kFDot2N), pwrap(kFDot2N);
    HIP_CHECK(hipMemcpy(dir.data(), dDir, kFDot2N * sizeof(__bf16), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(wrap.data(), dWrap, kFDot2N * sizeof(__bf16), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(pdir.data(), dPdir, kFDot2N * sizeof(std::uint16_t), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(pwrap.data(), dPwrap, kFDot2N * sizeof(std::uint16_t), hipMemcpyDeviceToHost));
    for (int i = 0; i < kFDot2N; ++i) {
      std::uint16_t bw = 0, bd = 0;
      std::memcpy(&bw, &wrap[i], sizeof bw);
      std::memcpy(&bd, &dir[i], sizeof bd);
      EXPECT_EQ(bw, bd) << "Float lane " << i;
      EXPECT_EQ(pwrap[i], pdir[i]) << "FPSan lane " << i;
    }
  });
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dDir);
  (void)hipFree(dWrap);
  (void)hipFree(dPdir);
  (void)hipFree(dPwrap);
}

TEST(AmdgcnMath, fdot2_f32_bf16_FloatAndFpsan) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  if (!current_arch_supports_fdot2_extended())
    GTEST_SKIP() << "current device does not support extended fdot2";
  auto a = make_v2bf(), b = make_v2bf();
  auto c = make_acc_f32();
  v2bf *dA = to_dev(a), *dB = to_dev(b);
  float *dC = to_dev(c);
  float *dDir, *dWrap;
  std::uint32_t *dPdir, *dPwrap;
  HIP_CHECK(hipMalloc(&dDir, kFDot2N * sizeof(float)));
  HIP_CHECK(hipMalloc(&dWrap, kFDot2N * sizeof(float)));
  HIP_CHECK(hipMalloc(&dPdir, kFDot2N * sizeof(std::uint32_t)));
  HIP_CHECK(hipMalloc(&dPwrap, kFDot2N * sizeof(std::uint32_t)));
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    k_fdot2_f32_bf16_pair<decltype(sem)::value>
        <<<1, kFDot2N>>>(dA, dB, dC, dDir, dWrap, dPdir, dPwrap);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> dir(kFDot2N), wrap(kFDot2N);
    std::vector<std::uint32_t> pdir(kFDot2N), pwrap(kFDot2N);
    HIP_CHECK(hipMemcpy(dir.data(), dDir, kFDot2N * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(wrap.data(), dWrap, kFDot2N * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(pdir.data(), dPdir, kFDot2N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(
        hipMemcpy(pwrap.data(), dPwrap, kFDot2N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for (int i = 0; i < kFDot2N; ++i) {
      EXPECT_EQ(bits_u32(wrap[i]), bits_u32(dir[i])) << "Float lane " << i;
      EXPECT_EQ(pwrap[i], pdir[i]) << "FPSan lane " << i;
    }
  });
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dDir);
  (void)hipFree(dWrap);
  (void)hipFree(dPdir);
  (void)hipFree(dPwrap);
}
#endif

// ============================================================================
