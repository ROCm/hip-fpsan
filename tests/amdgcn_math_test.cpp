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

#define AMDGCN_MATH_UNARY_KERNEL(NAME, FT, BUILTIN, FPSAN_OP)                                      \
  template <Semantics S>                                                                           \
  __global__ void k_##NAME(const FT *in, FT *direct, FT *wrapper,                                  \
                           typename Value<FT, Semantics::Triton, kCC>::bits_type *pay_direct,      \
                           typename Value<FT, Semantics::Triton, kCC>::bits_type *pay_wrapper) {   \
    const int i = threadIdx.x;                                                                     \
    const FT x = in[i];                                                                            \
    direct[i] = BUILTIN(x);                                                                        \
    Value<FT, Semantics::Native, kCC> vf{x};                                                       \
    wrapper[i] = fpsan::NAME<Semantics::Native, kCC>(vf).to_float();                               \
    Value<FT, S, kCC> vp{x};                                                                       \
    pay_direct[i] = fpsan::FPSAN_OP(vp).fpsan_payload();                                           \
    pay_wrapper[i] = fpsan::NAME<S, kCC>(vp).fpsan_payload();                                      \
  }

AMDGCN_MATH_UNARY_KERNEL(amdgcn_rcpf, float, __builtin_amdgcn_rcpf, rcp)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_sqrtf, float, __builtin_amdgcn_sqrtf, sqrt)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_rsqf, float, __builtin_amdgcn_rsqf, rsqrt)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_rsq_clampf, float, __builtin_amdgcn_rsq_clampf, rsqrt)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_sinf, float, __builtin_amdgcn_sinf, sin)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_cosf, float, __builtin_amdgcn_cosf, cos)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_logf, float, __builtin_amdgcn_logf, log)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_exp2f, float, __builtin_amdgcn_exp2f, exp2)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_fractf, float, __builtin_amdgcn_fractf, fract)

AMDGCN_MATH_UNARY_KERNEL(amdgcn_rcp, double, __builtin_amdgcn_rcp, rcp)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_sqrt, double, __builtin_amdgcn_sqrt, sqrt)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_rsq, double, __builtin_amdgcn_rsq, rsqrt)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_rsq_clamp, double, __builtin_amdgcn_rsq_clamp, rsqrt)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_fract, double, __builtin_amdgcn_fract, fract)

AMDGCN_MATH_UNARY_KERNEL(amdgcn_rcph, _Float16, __builtin_amdgcn_rcph, rcp)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_sqrth, _Float16, __builtin_amdgcn_sqrth, sqrt)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_rsqh, _Float16, __builtin_amdgcn_rsqh, rsqrt)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_sinh, _Float16, __builtin_amdgcn_sinh, sin)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_cosh, _Float16, __builtin_amdgcn_cosh, cos)
AMDGCN_MATH_UNARY_KERNEL(amdgcn_fracth, _Float16, __builtin_amdgcn_fracth, fract)
#undef AMDGCN_MATH_UNARY_KERNEL

#define AMDGCN_MATH_FMED3_KERNEL(NAME, FT, BUILTIN)                                                \
  template <Semantics S>                                                                           \
  __global__ void k_##NAME(const FT *a, const FT *b, const FT *c, FT *direct, FT *wrapper,         \
                           typename Value<FT, Semantics::Triton, kCC>::bits_type *pay_direct,      \
                           typename Value<FT, Semantics::Triton, kCC>::bits_type *pay_wrapper) {   \
    const int i = threadIdx.x;                                                                     \
    direct[i] = BUILTIN(a[i], b[i], c[i]);                                                         \
    Value<FT, Semantics::Native, kCC> af{a[i]}, bf{b[i]}, cf{c[i]};                                \
    wrapper[i] = fpsan::NAME<Semantics::Native, kCC>(af, bf, cf).to_float();                       \
    Value<FT, S, kCC> ap{a[i]}, bp{b[i]}, cp{c[i]};                                                \
    pay_direct[i] = fpsan::fmed3(ap, bp, cp).fpsan_payload();                                      \
    pay_wrapper[i] = fpsan::NAME<S, kCC>(ap, bp, cp).fpsan_payload();                              \
  }

AMDGCN_MATH_FMED3_KERNEL(amdgcn_fmed3f, float, __builtin_amdgcn_fmed3f)
AMDGCN_MATH_FMED3_KERNEL(amdgcn_fmed3h, _Float16, __builtin_amdgcn_fmed3h)
#undef AMDGCN_MATH_FMED3_KERNEL

#define AMDGCN_MATH_UNARY_TEST(NAME, FT, INPUTS)                                                   \
  TEST(AmdgcnMath, NAME) {                                                                         \
    if (!have_device())                                                                            \
      GTEST_SKIP() << "no HIP device";                                                             \
    using Bits = typename Value<FT, Semantics::Triton, kCC>::bits_type;                            \
    auto inputs = INPUTS<FT>();                                                                    \
    FT *dIn = to_dev(inputs);                                                                      \
    FT *dDirect, *dWrapper;                                                                        \
    Bits *dPayDirect, *dPayWrapper;                                                                \
    HIP_CHECK(hipMalloc(&dDirect, kScalarN * sizeof(FT)));                                         \
    HIP_CHECK(hipMalloc(&dWrapper, kScalarN * sizeof(FT)));                                        \
    HIP_CHECK(hipMalloc(&dPayDirect, kScalarN * sizeof(Bits)));                                    \
    HIP_CHECK(hipMalloc(&dPayWrapper, kScalarN * sizeof(Bits)));                                   \
    fpsan_test::for_each_fpsan_semantics([&](auto sem) {                                           \
      k_##NAME<decltype(sem)::value>                                                               \
          <<<1, kScalarN>>>(dIn, dDirect, dWrapper, dPayDirect, dPayWrapper);                      \
      HIP_CHECK(hipDeviceSynchronize());                                                           \
      std::vector<FT> direct(kScalarN), wrapper(kScalarN);                                         \
      std::vector<Bits> pay_direct(kScalarN), pay_wrapper(kScalarN);                               \
      HIP_CHECK(hipMemcpy(direct.data(), dDirect, kScalarN * sizeof(FT), hipMemcpyDeviceToHost));  \
      HIP_CHECK(                                                                                   \
          hipMemcpy(wrapper.data(), dWrapper, kScalarN * sizeof(FT), hipMemcpyDeviceToHost));      \
      HIP_CHECK(hipMemcpy(pay_direct.data(), dPayDirect, kScalarN * sizeof(Bits),                  \
                          hipMemcpyDeviceToHost));                                                 \
      HIP_CHECK(hipMemcpy(pay_wrapper.data(), dPayWrapper, kScalarN * sizeof(Bits),                \
                          hipMemcpyDeviceToHost));                                                 \
      for (int i = 0; i < kScalarN; ++i) {                                                         \
        EXPECT_EQ(bits_of(wrapper[i]), bits_of(direct[i])) << "Float lane " << i;                  \
        EXPECT_EQ(pay_wrapper[i], pay_direct[i]) << "FPSan lane " << i;                            \
      }                                                                                            \
    });                                                                                            \
    (void)hipFree(dIn);                                                                            \
    (void)hipFree(dDirect);                                                                        \
    (void)hipFree(dWrapper);                                                                       \
    (void)hipFree(dPayDirect);                                                                     \
    (void)hipFree(dPayWrapper);                                                                    \
  }

AMDGCN_MATH_UNARY_TEST(amdgcn_rcpf, float, make_positive_inputs)
AMDGCN_MATH_UNARY_TEST(amdgcn_sqrtf, float, make_positive_inputs)
AMDGCN_MATH_UNARY_TEST(amdgcn_rsqf, float, make_positive_inputs)
AMDGCN_MATH_UNARY_TEST(amdgcn_rsq_clampf, float, make_positive_inputs)
AMDGCN_MATH_UNARY_TEST(amdgcn_sinf, float, make_signed_inputs)
AMDGCN_MATH_UNARY_TEST(amdgcn_cosf, float, make_signed_inputs)
AMDGCN_MATH_UNARY_TEST(amdgcn_logf, float, make_positive_inputs)
AMDGCN_MATH_UNARY_TEST(amdgcn_exp2f, float, make_signed_inputs)
AMDGCN_MATH_UNARY_TEST(amdgcn_fractf, float, make_signed_inputs)

#undef AMDGCN_MATH_UNARY_TEST

#define AMDGCN_MATH_FMED3_TEST(NAME, FT)                                                           \
  TEST(AmdgcnMath, NAME) {                                                                         \
    if (!have_device())                                                                            \
      GTEST_SKIP() << "no HIP device";                                                             \
    using Bits = typename Value<FT, Semantics::Triton, kCC>::bits_type;                            \
    auto a = make_signed_inputs<FT>();                                                             \
    auto b = make_signed_inputs<FT>();                                                             \
    auto c = make_signed_inputs<FT>();                                                             \
    FT *dA = to_dev(a), *dB = to_dev(b), *dC = to_dev(c);                                          \
    FT *dDirect, *dWrapper;                                                                        \
    Bits *dPayDirect, *dPayWrapper;                                                                \
    HIP_CHECK(hipMalloc(&dDirect, kScalarN * sizeof(FT)));                                         \
    HIP_CHECK(hipMalloc(&dWrapper, kScalarN * sizeof(FT)));                                        \
    HIP_CHECK(hipMalloc(&dPayDirect, kScalarN * sizeof(Bits)));                                    \
    HIP_CHECK(hipMalloc(&dPayWrapper, kScalarN * sizeof(Bits)));                                   \
    fpsan_test::for_each_fpsan_semantics([&](auto sem) {                                           \
      k_##NAME<decltype(sem)::value>                                                               \
          <<<1, kScalarN>>>(dA, dB, dC, dDirect, dWrapper, dPayDirect, dPayWrapper);               \
      HIP_CHECK(hipDeviceSynchronize());                                                           \
      std::vector<FT> direct(kScalarN), wrapper(kScalarN);                                         \
      std::vector<Bits> pay_direct(kScalarN), pay_wrapper(kScalarN);                               \
      HIP_CHECK(hipMemcpy(direct.data(), dDirect, kScalarN * sizeof(FT), hipMemcpyDeviceToHost));  \
      HIP_CHECK(                                                                                   \
          hipMemcpy(wrapper.data(), dWrapper, kScalarN * sizeof(FT), hipMemcpyDeviceToHost));      \
      HIP_CHECK(hipMemcpy(pay_direct.data(), dPayDirect, kScalarN * sizeof(Bits),                  \
                          hipMemcpyDeviceToHost));                                                 \
      HIP_CHECK(hipMemcpy(pay_wrapper.data(), dPayWrapper, kScalarN * sizeof(Bits),                \
                          hipMemcpyDeviceToHost));                                                 \
      for (int i = 0; i < kScalarN; ++i) {                                                         \
        EXPECT_EQ(bits_of(wrapper[i]), bits_of(direct[i])) << "Float lane " << i;                  \
        EXPECT_EQ(pay_wrapper[i], pay_direct[i]) << "FPSan lane " << i;                            \
      }                                                                                            \
    });                                                                                            \
    (void)hipFree(dA);                                                                             \
    (void)hipFree(dB);                                                                             \
    (void)hipFree(dC);                                                                             \
    (void)hipFree(dDirect);                                                                        \
    (void)hipFree(dWrapper);                                                                       \
    (void)hipFree(dPayDirect);                                                                     \
    (void)hipFree(dPayWrapper);                                                                    \
  }

#undef AMDGCN_MATH_FMED3_TEST

// tanhf/tanhh need the tanh-insts feature and log_clampf is backend-deferred
// on the audited targets, so this shared file intentionally leaves them out.

// ============================================================================
