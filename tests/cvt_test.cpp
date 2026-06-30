// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/cvt_test.cpp
//
// GPU tests for fpsan/amdgcn_cvt.hpp.
#include "fpsan/amdgcn_cvt.hpp"
#include "fpsan/fpsan.hpp"

#include "fpsan_semantics.hpp"
#include "hip_test_utils.hpp"
#include "test_random.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

static constexpr Conversions kCC = Conversions::Explicit;

#ifndef FPSAN_TEST_ENABLE_FP8_CVT
#define FPSAN_TEST_ENABLE_FP8_CVT 0
#endif
#ifndef FPSAN_TEST_ENABLE_PKRTZ_CVT
#define FPSAN_TEST_ENABLE_PKRTZ_CVT 1
#endif

#if FPSAN_TEST_ENABLE_PKRTZ_CVT
__global__ void k_cvt_pkrtz_float(const float *a, const float *b, std::uint32_t *out) {
  int i = threadIdx.x;
  Value<float, Semantics::Native, kCC> av{a[i]}, bv{b[i]};
  auto r = fpsan::amdgcn_cvt_pkrtz<Semantics::Native, kCC>(av, bv);
  auto vec = static_cast<fpsan::v2h_native>(r);
  out[i] = __builtin_bit_cast(std::uint32_t, vec);
}

template <Semantics S>
__global__ void k_cvt_pkrtz_fpsan(const float *a, const float *b, std::uint32_t *out) {
  int i = threadIdx.x;
  Value<float, S, kCC> av{a[i]}, bv{b[i]};
  auto r = fpsan::amdgcn_cvt_pkrtz<S, kCC>(av, bv);
  auto pay = r.fpsan_payload(); // v2u16
  out[i] = __builtin_bit_cast(std::uint32_t, pay);
}

TEST(Cvt, PkrtzFloatMatchesBuiltin) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  std::vector<float> a(32), b(32);
  std::mt19937 rng = fpsan_test::make_rng();
  for (int i = 0; i < 32; ++i) {
    a[i] = fpsan_test::pick_quarter<float>(rng, -40, 40);
    b[i] = fpsan_test::pick_quarter<float>(rng, -40, 40);
  }
  float *dA, *dB;
  std::uint32_t *dOut;
  HIP_CHECK(hipMalloc(&dA, 32 * sizeof(float)));
  HIP_CHECK(hipMalloc(&dB, 32 * sizeof(float)));
  HIP_CHECK(hipMalloc(&dOut, 32 * sizeof(std::uint32_t)));
  HIP_CHECK(hipMemcpy(dA, a.data(), 32 * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, b.data(), 32 * sizeof(float), hipMemcpyHostToDevice));
  k_cvt_pkrtz_float<<<1, 32>>>(dA, dB, dOut);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<std::uint32_t> got(32);
  HIP_CHECK(hipMemcpy(got.data(), dOut, 32 * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
  // Native-mode result should equal pack(static_cast<_Float16>(a),
  // static_cast<_Float16>(b)) which is what cvt_pkrtz does up to RTZ rounding;
  // since our inputs are quarters in [-10,10], they are representable in
  // _Float16 exactly, so RTZ collapses to identity and the result IS the
  // pack of f16 casts.
  for (int i = 0; i < 32; ++i) {
    _Float16 ah = static_cast<_Float16>(a[i]);
    _Float16 bh = static_cast<_Float16>(b[i]);
    std::uint16_t ab, bb;
    std::memcpy(&ab, &ah, sizeof ab);
    std::memcpy(&bb, &bh, sizeof bb);
    std::uint32_t expected =
        static_cast<std::uint32_t>(ab) | (static_cast<std::uint32_t>(bb) << 16);
    EXPECT_EQ(got[i], expected) << "lane " << i;
  }
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dOut);
}

TEST(Cvt, PkrtzFpsanMatchesPerLaneCast) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  std::vector<float> a(32), b(32);
  std::mt19937 rng = fpsan_test::make_rng();
  for (int i = 0; i < 32; ++i) {
    a[i] = fpsan_test::pick_quarter<float>(rng, -40, 40);
    b[i] = fpsan_test::pick_quarter<float>(rng, -40, 40);
  }
  float *dA, *dB;
  std::uint32_t *dOut;
  HIP_CHECK(hipMalloc(&dA, 32 * sizeof(float)));
  HIP_CHECK(hipMalloc(&dB, 32 * sizeof(float)));
  HIP_CHECK(hipMalloc(&dOut, 32 * sizeof(std::uint32_t)));
  HIP_CHECK(hipMemcpy(dA, a.data(), 32 * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, b.data(), 32 * sizeof(float), hipMemcpyHostToDevice));
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    k_cvt_pkrtz_fpsan<S><<<1, 32>>>(dA, dB, dOut);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(32);
    HIP_CHECK(hipMemcpy(got.data(), dOut, 32 * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    using F = Value<float, S, kCC>;
    using H = Value<_Float16, S, kCC>;
    for (int i = 0; i < 32; ++i) {
      H ah = fpsan::cast<_Float16>(F{a[i]});
      H bh = fpsan::cast<_Float16>(F{b[i]});
      std::uint32_t expected = static_cast<std::uint32_t>(ah.fpsan_payload()) |
                               (static_cast<std::uint32_t>(bh.fpsan_payload()) << 16);
      EXPECT_EQ(got[i], expected) << "lane " << i;
    }
  });
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dOut);
}
#endif // FPSAN_TEST_ENABLE_PKRTZ_CVT

// ============================================================================
