// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/amdgcn_ldexp_test.cpp
//
// GPU tests for the amdgcn_ldexp{f,,h} wrappers in fpsan/amdgcn_math.hpp.
// ldexp(v, n) = v * 2^n. The builtins exist and lower on both RDNA (gfx12) and
// CDNA4 (gfx950), so this test is arch-agnostic.
//
//   * Native mode: bit-exact vs the host std::ldexp (independent reference).
//   * FPSan mode: the wrapper must equal the payload-ring model --
//     a multiply by the constant 2^n -- i.e. (v * Value(2^n)). The reference is
//     computed independently on the host with the same payload ring.
#include "fpsan/amdgcn_math.hpp"
#include "fpsan/fpsan.hpp"

#include "fpsan_semantics.hpp"

#include "hip_test_utils.hpp"
#include "test_random.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

static constexpr Conversions kCC = Conversions::Explicit;

namespace {
// The exponents under test (small enough to keep f16 in-range, exact rescale).
constexpr int kExps[] = {-6, -3, -1, 0, 1, 2, 5};
constexpr int kNExp = static_cast<int>(sizeof(kExps) / sizeof(kExps[0]));
} // namespace

template <Semantics S, class Out> __global__ void k_ldexpf(const float *in, Out *out, int n) {
  int l = threadIdx.x;
  Value<float, S, kCC> v{in[l]};
  auto r = fpsan::amdgcn_ldexpf<S, kCC>(v, n);
  if constexpr (S == Semantics::Native)
    out[l] = r.to_float();
  else
    out[l] = r.fpsan_payload();
}

TEST(AmdgcnLdexp, F32) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  const int N = 64;
  std::vector<float> in(N);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : in)
    x = fpsan_test::pick_int_valued<float>(rng, -8, 8); // exact mantissa
  float *dIn = to_dev(in);
  for (int e = 0; e < kNExp; ++e) {
    const int n = kExps[e];
    // Float: bit-exact vs host std::ldexp.
    {
      float *dO;
      HIP_CHECK(hipMalloc(&dO, N * sizeof(float)));
      k_ldexpf<Semantics::Native, float><<<1, N>>>(dIn, dO, n);
      HIP_CHECK(hipDeviceSynchronize());
      std::vector<float> got(N);
      HIP_CHECK(hipMemcpy(got.data(), dO, N * sizeof(float), hipMemcpyDeviceToHost));
      for (int i = 0; i < N; ++i)
        EXPECT_EQ(got[i], std::ldexp(in[i], n)) << "n=" << n << " i=" << i;
      (void)hipFree(dO);
    }
    // FPSan: payload == (v * 2^n) in the ring, for every algebraic model.
    fpsan_test::for_each_fpsan_semantics([&](auto sem) {
      constexpr Semantics S = decltype(sem)::value;
      using VF = Value<float, S, kCC>;
      std::vector<std::uint32_t> ref(N);
      for (int i = 0; i < N; ++i)
        ref[i] = (VF(in[i]) * VF(std::ldexp(1.0f, n))).fpsan_payload();
      std::uint32_t *dO;
      HIP_CHECK(hipMalloc(&dO, N * sizeof(std::uint32_t)));
      k_ldexpf<S, std::uint32_t><<<1, N>>>(dIn, dO, n);
      HIP_CHECK(hipDeviceSynchronize());
      std::vector<std::uint32_t> got(N);
      HIP_CHECK(hipMemcpy(got.data(), dO, N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
      for (int i = 0; i < N; ++i)
        EXPECT_EQ(got[i], ref[i]) << "FPSan n=" << n << " i=" << i;
      (void)hipFree(dO);
    });
  }
  (void)hipFree(dIn);
}

template <Semantics S, class Out> __global__ void k_ldexp(const double *in, Out *out, int n) {
  int l = threadIdx.x;
  Value<double, S, kCC> v{in[l]};
  auto r = fpsan::amdgcn_ldexp<S, kCC>(v, n);
  if constexpr (S == Semantics::Native)
    out[l] = r.to_float();
  else
    out[l] = r.fpsan_payload();
}

TEST(AmdgcnLdexp, F64) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  const int N = 64;
  std::vector<double> in(N);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : in)
    x = fpsan_test::pick_int_valued<double>(rng, -8, 8);
  double *dIn = to_dev(in);
  for (int e = 0; e < kNExp; ++e) {
    const int n = kExps[e];
    {
      double *dO;
      HIP_CHECK(hipMalloc(&dO, N * sizeof(double)));
      k_ldexp<Semantics::Native, double><<<1, N>>>(dIn, dO, n);
      HIP_CHECK(hipDeviceSynchronize());
      std::vector<double> got(N);
      HIP_CHECK(hipMemcpy(got.data(), dO, N * sizeof(double), hipMemcpyDeviceToHost));
      for (int i = 0; i < N; ++i)
        EXPECT_EQ(got[i], std::ldexp(in[i], n)) << "n=" << n << " i=" << i;
      (void)hipFree(dO);
    }
    fpsan_test::for_each_fpsan_semantics([&](auto sem) {
      constexpr Semantics S = decltype(sem)::value;
      using VD = Value<double, S, kCC>;
      std::vector<std::uint64_t> ref(N);
      for (int i = 0; i < N; ++i)
        ref[i] = (VD(in[i]) * VD(std::ldexp(1.0, n))).fpsan_payload();
      std::uint64_t *dO;
      HIP_CHECK(hipMalloc(&dO, N * sizeof(std::uint64_t)));
      k_ldexp<S, std::uint64_t><<<1, N>>>(dIn, dO, n);
      HIP_CHECK(hipDeviceSynchronize());
      std::vector<std::uint64_t> got(N);
      HIP_CHECK(hipMemcpy(got.data(), dO, N * sizeof(std::uint64_t), hipMemcpyDeviceToHost));
      for (int i = 0; i < N; ++i)
        EXPECT_EQ(got[i], ref[i]) << "FPSan n=" << n << " i=" << i;
      (void)hipFree(dO);
    });
  }
  (void)hipFree(dIn);
}

template <Semantics S, class Out> __global__ void k_ldexph(const float *in, Out *out, int n) {
  int l = threadIdx.x;
  Value<_Float16, S, kCC> v{static_cast<_Float16>(in[l])};
  auto r = fpsan::amdgcn_ldexph<S, kCC>(v, n);
  if constexpr (S == Semantics::Native)
    out[l] = static_cast<float>(r.to_float());
  else
    out[l] = r.fpsan_payload();
}

TEST(AmdgcnLdexp, F16) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  const int N = 64;
  std::vector<float> in(N);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : in)
    x = fpsan_test::pick_int_valued<float>(rng, -8, 8);
  float *dIn = to_dev(in);
  for (int e = 0; e < kNExp; ++e) {
    const int n = kExps[e];
    {
      float *dO;
      HIP_CHECK(hipMalloc(&dO, N * sizeof(float)));
      k_ldexph<Semantics::Native, float><<<1, N>>>(dIn, dO, n);
      HIP_CHECK(hipDeviceSynchronize());
      std::vector<float> got(N);
      HIP_CHECK(hipMemcpy(got.data(), dO, N * sizeof(float), hipMemcpyDeviceToHost));
      for (int i = 0; i < N; ++i) {
        _Float16 ref =
            static_cast<_Float16>(std::ldexp(static_cast<float>(static_cast<_Float16>(in[i])), n));
        EXPECT_EQ(got[i], static_cast<float>(ref)) << "n=" << n << " i=" << i;
      }
      (void)hipFree(dO);
    }
    fpsan_test::for_each_fpsan_semantics([&](auto sem) {
      constexpr Semantics S = decltype(sem)::value;
      using VH = Value<_Float16, S, kCC>;
      std::vector<std::uint32_t> ref(N);
      for (int i = 0; i < N; ++i)
        ref[i] = (VH(static_cast<_Float16>(in[i])) * VH(static_cast<_Float16>(std::ldexp(1.0f, n))))
                     .fpsan_payload();
      std::uint32_t *dO;
      HIP_CHECK(hipMalloc(&dO, N * sizeof(std::uint32_t)));
      k_ldexph<S, std::uint32_t><<<1, N>>>(dIn, dO, n);
      HIP_CHECK(hipDeviceSynchronize());
      std::vector<std::uint32_t> got(N);
      HIP_CHECK(hipMemcpy(got.data(), dO, N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
      for (int i = 0; i < N; ++i)
        EXPECT_EQ(got[i], ref[i]) << "FPSan n=" << n << " i=" << i;
      (void)hipFree(dO);
    });
  }
  (void)hipFree(dIn);
}
