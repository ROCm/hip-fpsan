// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/amdgcn_bf16_trans_test.cpp
//
// GPU tests for the gfx1250 bf16 transcendental wrappers in
// fpsan/amdgcn_math.hpp: rcp/rsq/sqrt/sin/cos/exp2/log/tanh, each __bf16(__bf16)
// and gated by the gfx1250-exclusive "bf16-trans-insts" feature. Registered by
// CMake only for the gfx1250 tier.
//
// Test policy note (authoritative-correctness mandate): these are HARDWARE
// TRANSCENDENTAL APPROXIMATIONS. There is no host-computable "known-good" value
// for a hardware rcp/rsq/sin/... approximation (the exact result is defined by
// the unit's table+interpolation, not by std::sin), and we have no real silicon
// to sample. So the two assertions we CAN make authoritatively are:
//   - Native mode: the wrapper is a bit-exact pass-through to the builtin (this
//     proves our wrapper plumbing; the builtin's accuracy is the hardware's
//     contract, not ours to redefine).
//   - FPSan-family semantics: the wrapper's payload equals the matching fpsan::
//     operation, payload-for-payload, in that same semantics.
// We deliberately do NOT assert a numeric tolerance here, to avoid encoding a
// non-authoritative ULP bound.
#include "fpsan/amdgcn_math.hpp"
#include "fpsan/fpsan.hpp"

#include "fpsan_semantics.hpp"
#include "hip_test_utils.hpp"
#include "test_random.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

static constexpr Conversions kCC = Conversions::Explicit;

namespace {
std::uint16_t bits_u16(__bf16 v) {
  std::uint16_t u = 0;
  std::memcpy(&u, &v, sizeof u);
  return u;
}

// Positive finite inputs (1.0 .. 9.0) so the restricted-domain ops
// (log/sqrt/rsq) and their FPSan tagged-op references are well-defined.
std::vector<__bf16> make_inputs() {
  std::vector<__bf16> v(32);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : v)
    x = static_cast<__bf16>(fpsan_test::pick_quarter<float>(rng, 4, 36));
  return v;
}
} // namespace

#define BF16_TRANS_KERNEL(name, FPSAN_OP)                                                          \
  template <Semantics S>                                                                           \
  __global__ void k_##name##_pair(const __bf16 *in, __bf16 *direct, __bf16 *via_wrapper,           \
                                  std::uint16_t *pay_direct, std::uint16_t *pay_wrapper) {         \
    int i = threadIdx.x;                                                                           \
    __bf16 x = in[i];                                                                              \
    direct[i] = __builtin_##name(x);                                                               \
    Value<__bf16, Semantics::Native, kCC> vf{x};                                                   \
    via_wrapper[i] = static_cast<__bf16>(fpsan::name<Semantics::Native, kCC>(vf));                 \
    Value<__bf16, S, kCC> vp{x};                                                                   \
    pay_direct[i] = static_cast<std::uint16_t>(fpsan::FPSAN_OP(vp).fpsan_payload());               \
    pay_wrapper[i] = static_cast<std::uint16_t>(fpsan::name<S, kCC>(vp).fpsan_payload());          \
  }

BF16_TRANS_KERNEL(amdgcn_rcp_bf16, rcp)
BF16_TRANS_KERNEL(amdgcn_rsq_bf16, rsqrt)
BF16_TRANS_KERNEL(amdgcn_sqrt_bf16, sqrt)
BF16_TRANS_KERNEL(amdgcn_sin_bf16, sin)
BF16_TRANS_KERNEL(amdgcn_cos_bf16, cos)
BF16_TRANS_KERNEL(amdgcn_exp2_bf16, exp2)
BF16_TRANS_KERNEL(amdgcn_log_bf16, log)
BF16_TRANS_KERNEL(amdgcn_tanh_bf16, tanh)

#define BF16_TRANS_TEST(name)                                                                      \
  TEST(AmdgcnBf16Trans, name##_FloatForwardsAndFpsanRoutes) {                                      \
    int ndev = 0;                                                                                  \
    if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)                                       \
      GTEST_SKIP() << "no HIP device";                                                             \
    auto in = make_inputs();                                                                       \
    const int N = static_cast<int>(in.size());                                                     \
    __bf16 *dIn = to_dev(in);                                                                      \
    __bf16 *dDirect, *dWrap;                                                                       \
    std::uint16_t *dPdir, *dPwrap;                                                                 \
    HIP_CHECK(hipMalloc(&dDirect, N * sizeof(__bf16)));                                            \
    HIP_CHECK(hipMalloc(&dWrap, N * sizeof(__bf16)));                                              \
    HIP_CHECK(hipMalloc(&dPdir, N * sizeof(std::uint16_t)));                                       \
    HIP_CHECK(hipMalloc(&dPwrap, N * sizeof(std::uint16_t)));                                      \
    k_##name##_pair<Semantics::Triton><<<1, N>>>(dIn, dDirect, dWrap, dPdir, dPwrap);              \
    HIP_CHECK(hipDeviceSynchronize());                                                             \
    std::vector<__bf16> direct(N), wrap(N);                                                        \
    std::vector<std::uint16_t> pdir(N), pwrap(N);                                                  \
    HIP_CHECK(hipMemcpy(direct.data(), dDirect, N * sizeof(__bf16), hipMemcpyDeviceToHost));       \
    HIP_CHECK(hipMemcpy(wrap.data(), dWrap, N * sizeof(__bf16), hipMemcpyDeviceToHost));           \
    HIP_CHECK(hipMemcpy(pdir.data(), dPdir, N * sizeof(std::uint16_t), hipMemcpyDeviceToHost));    \
    HIP_CHECK(hipMemcpy(pwrap.data(), dPwrap, N * sizeof(std::uint16_t), hipMemcpyDeviceToHost));  \
    for (int i = 0; i < N; ++i) {                                                                  \
      EXPECT_EQ(bits_u16(wrap[i]), bits_u16(direct[i])) << "Float lane " << i;                     \
      EXPECT_EQ(pwrap[i], pdir[i]) << "FPSan lane " << i;                                          \
    }                                                                                              \
    fpsan_test::for_each_fpsan_semantics([&](auto sem) {                                           \
      constexpr Semantics S = decltype(sem)::value;                                                \
      k_##name##_pair<S><<<1, N>>>(dIn, dDirect, dWrap, dPdir, dPwrap);                            \
      HIP_CHECK(hipDeviceSynchronize());                                                           \
      HIP_CHECK(hipMemcpy(pdir.data(), dPdir, N * sizeof(std::uint16_t), hipMemcpyDeviceToHost));  \
      HIP_CHECK(                                                                                   \
          hipMemcpy(pwrap.data(), dPwrap, N * sizeof(std::uint16_t), hipMemcpyDeviceToHost));      \
      for (int i = 0; i < N; ++i)                                                                  \
        EXPECT_EQ(pwrap[i], pdir[i]) << fpsan::semantics_name(S) << " lane " << i;                 \
    });                                                                                            \
    (void)hipFree(dIn);                                                                            \
    (void)hipFree(dDirect);                                                                        \
    (void)hipFree(dWrap);                                                                          \
    (void)hipFree(dPdir);                                                                          \
    (void)hipFree(dPwrap);                                                                         \
  }

BF16_TRANS_TEST(amdgcn_rcp_bf16)
BF16_TRANS_TEST(amdgcn_rsq_bf16)
BF16_TRANS_TEST(amdgcn_sqrt_bf16)
BF16_TRANS_TEST(amdgcn_sin_bf16)
BF16_TRANS_TEST(amdgcn_cos_bf16)
BF16_TRANS_TEST(amdgcn_exp2_bf16)
BF16_TRANS_TEST(amdgcn_log_bf16)
BF16_TRANS_TEST(amdgcn_tanh_bf16)
