// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/classify_test.cpp
//
// GPU tests for amdgcn_classify.hpp. Float and FPSan modes both classify the
// represented float, so the per-lane bool / ballot mask must match the host
// IEEE-category oracle.
#include "fpsan/amdgcn_classify.hpp"
#include "fpsan/fpsan.hpp"

#include "fpsan_semantics.hpp"
#include "hip_test_utils.hpp"
#include "test_random.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

static constexpr Conversions kCC = Conversions::Explicit;
#ifndef FPSAN_TEST_FORCE_WAVE_SIZE
#define FPSAN_TEST_FORCE_WAVE_SIZE 32
#endif
static_assert(FPSAN_TEST_FORCE_WAVE_SIZE == 32 || FPSAN_TEST_FORCE_WAVE_SIZE == 64,
              "classify tests support one wave32 or wave64 wave");
static constexpr int LANES = FPSAN_TEST_FORCE_WAVE_SIZE;

template <Semantics S> __device__ bool class_value(Value<float, S, kCC> v, int mask) {
  return fpsan::amdgcn_classf<S, kCC>(v, mask);
}
template <Semantics S> __device__ bool class_value(Value<double, S, kCC> v, int mask) {
  return fpsan::amdgcn_class<S, kCC>(v, mask);
}
#if FPSAN_HAS_FLOAT16
template <Semantics S> __device__ bool class_value(Value<_Float16, S, kCC> v, int mask) {
  return fpsan::amdgcn_classh<S, kCC>(v, mask);
}
#endif

template <class FT, Semantics S>
__global__ void k_class_pair_masks(const FT *in, const int *masks, char *bf, char *bp, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  Value<FT, Semantics::Native, kCC> vf{in[i]};
  Value<FT, S, kCC> vp{in[i]};
  bf[i] = class_value<Semantics::Native>(vf, masks[i]) ? 1 : 0;
  bp[i] = class_value<S>(vp, masks[i]) ? 1 : 0;
}

// Mask 0x3FF = all categories ON; classf returns true for any normal value.
template <Semantics S> __global__ void k_classf_pair(const float *in, char *bf, char *bp) {
  int i = threadIdx.x;
  Value<float, Semantics::Native, kCC> vf{in[i]};
  Value<float, S, kCC> vp{in[i]};
  bf[i] = fpsan::amdgcn_classf<Semantics::Native, kCC>(vf, 0x3FF) ? 1 : 0;
  bp[i] = fpsan::amdgcn_classf<S, kCC>(vp, 0x3FF) ? 1 : 0;
}

TEST(Classify, ClassfFloatAndFpsanAgree) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  std::vector<float> in(LANES);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : in)
    x = fpsan_test::pick_quarter<float>(rng, -20, 20); // -5 .. 5
  float *dIn;
  HIP_CHECK(hipMalloc(&dIn, LANES * sizeof(float)));
  HIP_CHECK(hipMemcpy(dIn, in.data(), LANES * sizeof(float), hipMemcpyHostToDevice));
  char *dBf, *dBp;
  HIP_CHECK(hipMalloc(&dBf, LANES));
  HIP_CHECK(hipMalloc(&dBp, LANES));
  // classf classifies the *represented float* (it calls to_float() internally),
  // so it is defined only on the flavors that carry a float: Native (the oracle)
  // and the bit-faithful Triton scramble. The algebraic payloads are residues in
  // Z/nZ with no float to classify -- a hard limitation, so they are skipped.
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    if constexpr (!fpsan::detail::is_algebraic_semantics(S)) {
      k_classf_pair<S><<<1, LANES>>>(dIn, dBf, dBp);
      HIP_CHECK(hipDeviceSynchronize());
      std::vector<char> bf(LANES), bp(LANES);
      HIP_CHECK(hipMemcpy(bf.data(), dBf, LANES, hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(bp.data(), dBp, LANES, hipMemcpyDeviceToHost));
      for (int i = 0; i < LANES; ++i) {
        // Float and FPSan must agree with each other AND with an independent host
        // reference: classf(v, 0x3FF) returns true for any value matching any of
        // the 10 IEEE categories, which is true for every finite/inf/nan -- so
        // for finite inputs (which our quarters are) it should be true.
        EXPECT_EQ(int(bf[i]), int(bp[i])) << "lane " << i;
        EXPECT_EQ(int(bf[i]), 1) << "lane " << i << " host ref: any finite";
      }
    }
  });
  (void)hipFree(dIn);
  (void)hipFree(dBf);
  (void)hipFree(dBp);
}

template <class FT> std::vector<FT> classify_samples();

template <class FT, class Bits> FT class_sample_from_bits(Bits bits) {
  static_assert(sizeof(FT) == sizeof(Bits), "class sample bit width mismatch");
  return __builtin_bit_cast(FT, bits);
}

template <> std::vector<float> classify_samples<float>() {
  return {0.0f,
          -0.0f,
          1.0f,
          -1.0f,
          std::numeric_limits<float>::denorm_min(),
          -std::numeric_limits<float>::denorm_min(),
          std::numeric_limits<float>::infinity(),
          -std::numeric_limits<float>::infinity(),
          class_sample_from_bits<float>(std::uint32_t{0x7F800001u}),
          std::numeric_limits<float>::quiet_NaN()};
}

template <> std::vector<double> classify_samples<double>() {
  return {0.0,
          -0.0,
          1.0,
          -1.0,
          std::numeric_limits<double>::denorm_min(),
          -std::numeric_limits<double>::denorm_min(),
          std::numeric_limits<double>::infinity(),
          -std::numeric_limits<double>::infinity(),
          class_sample_from_bits<double>(std::uint64_t{0x7FF0000000000001ull}),
          std::numeric_limits<double>::quiet_NaN()};
}

#if FPSAN_HAS_FLOAT16
template <> std::vector<_Float16> classify_samples<_Float16>() {
  return {static_cast<_Float16>(0.0f),
          static_cast<_Float16>(-0.0f),
          static_cast<_Float16>(1.0f),
          static_cast<_Float16>(-1.0f),
          static_cast<_Float16>(5.9604644775390625e-8f),
          static_cast<_Float16>(-5.9604644775390625e-8f),
          static_cast<_Float16>(std::numeric_limits<float>::infinity()),
          static_cast<_Float16>(-std::numeric_limits<float>::infinity()),
          class_sample_from_bits<_Float16>(std::uint16_t{0x7C01u}),
          static_cast<_Float16>(std::numeric_limits<float>::quiet_NaN())};
}
#endif

template <class FT> int host_class_bit(FT x) {
  const std::uint64_t bits = bits_of(x);
  std::uint64_t sign = 0;
  std::uint64_t exp = 0;
  std::uint64_t frac = 0;
  std::uint64_t exp_all = 0;
  std::uint64_t quiet = 0;

  if constexpr (sizeof(FT) == 2) {
    sign = bits & 0x8000u;
    exp = bits & 0x7C00u;
    frac = bits & 0x03FFu;
    exp_all = 0x7C00u;
    quiet = 0x0200u;
  } else if constexpr (sizeof(FT) == 4) {
    sign = bits & 0x80000000ull;
    exp = bits & 0x7F800000ull;
    frac = bits & 0x007FFFFFull;
    exp_all = 0x7F800000ull;
    quiet = 0x00400000ull;
  } else {
    static_assert(sizeof(FT) == 8, "unsupported class test type");
    sign = bits & 0x8000000000000000ull;
    exp = bits & 0x7FF0000000000000ull;
    frac = bits & 0x000FFFFFFFFFFFFFull;
    exp_all = 0x7FF0000000000000ull;
    quiet = 0x0008000000000000ull;
  }

  if (exp == exp_all) {
    if (frac == 0)
      return sign ? 2 : 9;         // -inf, +inf
    return (frac & quiet) ? 1 : 0; // qNaN, sNaN
  }
  if (exp == 0) {
    if (frac == 0)
      return sign ? 5 : 6; // -0, +0
    return sign ? 4 : 7;   // negative subnormal, positive subnormal
  }
  return sign ? 3 : 8; // negative normal, positive normal
}

template <class FT> bool host_class_matches(FT x, int mask) {
  return (mask & (1 << host_class_bit(x))) != 0;
}

template <class FT> void run_class_mask_parity(const char *label) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";

  std::vector<FT> samples = classify_samples<FT>();
  std::vector<FT> in;
  std::vector<int> masks;
  in.reserve(samples.size() * 10);
  masks.reserve(samples.size() * 10);
  for (FT x : samples)
    for (int bit = 0; bit < 10; ++bit) {
      in.push_back(x);
      masks.push_back(1 << bit);
    }

  const int n = static_cast<int>(in.size());
  FT *dIn;
  int *dMasks;
  char *dBf, *dBp;
  HIP_CHECK(hipMalloc(&dIn, n * sizeof(FT)));
  HIP_CHECK(hipMalloc(&dMasks, n * sizeof(int)));
  HIP_CHECK(hipMalloc(&dBf, n));
  HIP_CHECK(hipMalloc(&dBp, n));
  HIP_CHECK(hipMemcpy(dIn, in.data(), n * sizeof(FT), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dMasks, masks.data(), n * sizeof(int), hipMemcpyHostToDevice));

  k_class_pair_masks<FT, Semantics::Triton>
      <<<dim3((n + 63) / 64), dim3(64)>>>(dIn, dMasks, dBf, dBp, n);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<char> bf(n), bp(n);
  HIP_CHECK(hipMemcpy(bf.data(), dBf, n, hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(bp.data(), dBp, n, hipMemcpyDeviceToHost));

  int expected_true_count = 0;
  int expected_bit_hits[10]{};
  for (int i = 0; i < n; ++i) {
    const int expected = host_class_matches(in[i], masks[i]) ? 1 : 0;
    EXPECT_EQ(int(bf[i]), expected)
        << label << " Native sample " << (i / 10) << " mask bit " << (i % 10);
    EXPECT_EQ(int(bp[i]), expected)
        << label << " Triton sample " << (i / 10) << " mask bit " << (i % 10);
    expected_true_count += expected;
    expected_bit_hits[i % 10] += expected;
  }
  for (int bit = 0; bit < 10; ++bit)
    EXPECT_GT(expected_bit_hits[bit], 0) << label << " class bit " << bit;
  EXPECT_GT(expected_true_count, 0) << label;
  EXPECT_LT(expected_true_count, n) << label;

  (void)hipFree(dIn);
  (void)hipFree(dMasks);
  (void)hipFree(dBf);
  (void)hipFree(dBp);
}

TEST(Classify, ClassMasksMatchHostExpectedForAllCategories) {
  run_class_mask_parity<float>("f32");
  run_class_mask_parity<double>("f64");
#if FPSAN_HAS_FLOAT16
  run_class_mask_parity<_Float16>("f16");
#endif
}

template <Semantics S>
__global__ void k_fcmpf_pair(const float *a, const float *b, std::uint64_t *mf, std::uint64_t *mp) {
  int i = threadIdx.x;
  Value<float, Semantics::Native, kCC> af{a[i]}, bf{b[i]};
  Value<float, S, kCC> ap{a[i]}, bp{b[i]};
  // Predicate 1 = OEQ (ordered equal); see LLVM fcmp predicates.
  std::uint64_t f = fpsan::amdgcn_fcmpf<1, Semantics::Native, kCC>(af, bf);
  std::uint64_t p = fpsan::amdgcn_fcmpf<1, S, kCC>(ap, bp);
  if (i == 0) {
    *mf = f;
    *mp = p;
  }
}

TEST(Classify, FcmpfFloatAndFpsanAgree) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  std::vector<float> a(LANES), b(LANES);
  // Half identical, half different so the mask is nontrivial.
  std::uint64_t expected = 0;
  for (int i = 0; i < LANES; ++i) {
    a[i] = static_cast<float>(i % 4);
    b[i] = static_cast<float>(i % 5);
    if (a[i] == b[i])
      expected |= (1ull << i);
  }
  float *dA, *dB;
  HIP_CHECK(hipMalloc(&dA, LANES * sizeof(float)));
  HIP_CHECK(hipMalloc(&dB, LANES * sizeof(float)));
  HIP_CHECK(hipMemcpy(dA, a.data(), LANES * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, b.data(), LANES * sizeof(float), hipMemcpyHostToDevice));
  std::uint64_t *dMf, *dMp;
  HIP_CHECK(hipMalloc(&dMf, sizeof(std::uint64_t)));
  HIP_CHECK(hipMalloc(&dMp, sizeof(std::uint64_t)));
  // fcmpf compares the *represented floats* (via to_float), so like classf it is
  // defined only on the float-carrying flavors (Native, Triton); the algebraic
  // residues have no ordered float to compare -- a hard limitation, skipped.
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    if constexpr (!fpsan::detail::is_algebraic_semantics(S)) {
      k_fcmpf_pair<S><<<1, LANES>>>(dA, dB, dMf, dMp);
      HIP_CHECK(hipDeviceSynchronize());
      std::uint64_t mf = 0, mp = 0;
      HIP_CHECK(hipMemcpy(&mf, dMf, sizeof mf, hipMemcpyDeviceToHost));
      HIP_CHECK(hipMemcpy(&mp, dMp, sizeof mp, hipMemcpyDeviceToHost));
      EXPECT_EQ(mf, mp);
      EXPECT_EQ(mf, expected);
    }
  });
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dMf);
  (void)hipFree(dMp);
}
