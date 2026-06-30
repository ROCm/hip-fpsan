// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mfma_cdna3_parts.hpp"

// ---------------------------------------------------------------------------
// F64 16x16x4 and 4x4x4 MFMA shapes.
// ---------------------------------------------------------------------------
using fpsan::v4d_native;

static constexpr int F64_M = 16, F64_N = 16, F64_K = 4;

struct F64Data {
  std::vector<double> A, B, C;
};

static F64Data make_f64_16_inputs() {
  F64Data d;
  d.A.resize(F64_M * F64_K);
  d.B.resize(F64_K * F64_N);
  d.C.resize(F64_M * F64_N);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : d.A)
    x = fpsan_test::pick_int_valued<double>(rng, -3, 3);
  for (auto &x : d.B)
    x = fpsan_test::pick_int_valued<double>(rng, -2, 2);
  for (auto &x : d.C)
    x = fpsan_test::pick_int_valued<double>(rng, -4, 4);
  return d;
}

template <class T> T host_neg_if(T v, bool neg) { return neg ? -v : v; }

template <Semantics S, class Out, int NEG = 0>
__global__ void k_f64_16x16x4(const double *A, const double *B, const double *C, Out *D) {
  int lane = threadIdx.x;
  double an = 0.0, bn = 0.0;
  for (int i = 0; i < F64_M; ++i)
    for (int k = 0; k < F64_K; ++k) {
      auto loc = fpsan::detail::input_loc(F64_M, F64_K, 1, i, k, 0, 64);
      if (loc.lane == lane)
        an = A[i * F64_K + k];
    }
  for (int j = 0; j < F64_N; ++j)
    for (int k = 0; k < F64_K; ++k) {
      auto loc = fpsan::detail::input_loc(F64_N, F64_K, 1, j, k, 0, 64);
      if (loc.lane == lane)
        bn = B[k * F64_N + j];
    }
  v4d_native cn{};
  for (int i = 0; i < F64_M; ++i)
    for (int j = 0; j < F64_N; ++j) {
      auto loc = fpsan::detail::output_loc_64(F64_M, F64_N, i, j, 0);
      if (loc.lane == lane)
        cn[loc.reg / 2] = C[i * F64_N + j];
    }
  Value<double, S, kCC> a{an}, b{bn};
  Value<v4d_native, S, kCC> c{cn};
  auto d = fpsan::amdgcn_mfma_f64_16x16x4f64<0, 0, NEG, S, kCC>(a, b, c);
  for (int i = 0; i < F64_M; ++i)
    for (int j = 0; j < F64_N; ++j) {
      auto loc = fpsan::detail::output_loc_64(F64_M, F64_N, i, j, 0);
      if (loc.lane == lane) {
        if constexpr (S == Semantics::Native)
          D[i * F64_N + j] = d.get(loc.reg / 2).to_float();
        else
          D[i * F64_N + j] = d.get(loc.reg / 2).fpsan_payload();
      }
    }
}

TEST(MfmaF64_16x16x4, LayoutMatchesHardware) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  auto d = make_f64_16_inputs();
  std::vector<double> ref(F64_M * F64_N);
  for (int i = 0; i < F64_M; ++i)
    for (int j = 0; j < F64_N; ++j) {
      double acc = d.C[i * F64_N + j];
      for (int k = 0; k < F64_K; ++k)
        acc += d.A[i * F64_K + k] * d.B[k * F64_N + j];
      ref[i * F64_N + j] = acc;
    }
  double *dA = to_dev(d.A), *dB = to_dev(d.B), *dC = to_dev(d.C), *dD;
  HIP_CHECK(hipMalloc(&dD, ref.size() * sizeof(double)));
  k_f64_16x16x4<Semantics::Native, double><<<1, WAVE>>>(dA, dB, dC, dD);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<double> got(ref.size());
  HIP_CHECK(hipMemcpy(got.data(), dD, got.size() * sizeof(double), hipMemcpyDeviceToHost));
  for (std::size_t i = 0; i < got.size(); ++i)
    EXPECT_EQ(bits_of(got[i]), bits_of(ref[i])) << "elem " << i;
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dD);
}

TEST(MfmaF64_16x16x4, FpsanMatchesScalarReference) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  auto d = make_f64_16_inputs();
  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VD = Value<double, S, kCC>;
    std::vector<std::uint64_t> ref(F64_M * F64_N);
    for (int i = 0; i < F64_M; ++i)
      for (int j = 0; j < F64_N; ++j) {
        VD acc(d.C[i * F64_N + j]);
        for (int k = 0; k < F64_K; ++k)
          acc = acc + VD(d.A[i * F64_K + k]) * VD(d.B[k * F64_N + j]);
        ref[i * F64_N + j] = acc.fpsan_payload();
      }
    double *dA = to_dev(d.A), *dB = to_dev(d.B), *dC = to_dev(d.C);
    std::uint64_t *dD;
    HIP_CHECK(hipMalloc(&dD, ref.size() * sizeof(std::uint64_t)));
    k_f64_16x16x4<S, std::uint64_t><<<1, WAVE>>>(dA, dB, dC, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint64_t> got(ref.size());
    HIP_CHECK(hipMemcpy(got.data(), dD, got.size() * sizeof(std::uint64_t), hipMemcpyDeviceToHost));
    for (std::size_t i = 0; i < got.size(); ++i)
      EXPECT_EQ(got[i], ref[i]) << "elem " << i;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
  });
}

TEST(MfmaF64_16x16x4_NEG5, LayoutMatchesHardware) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  constexpr int NEG = 5;
  auto d = make_f64_16_inputs();
  std::vector<double> ref(F64_M * F64_N);
  for (int i = 0; i < F64_M; ++i)
    for (int j = 0; j < F64_N; ++j) {
      double acc = host_neg_if(d.C[i * F64_N + j], (NEG & 4) != 0);
      for (int k = 0; k < F64_K; ++k)
        acc += host_neg_if(d.A[i * F64_K + k], (NEG & 1) != 0) *
               host_neg_if(d.B[k * F64_N + j], (NEG & 2) != 0);
      ref[i * F64_N + j] = acc;
    }
  double *dA = to_dev(d.A), *dB = to_dev(d.B), *dC = to_dev(d.C), *dD;
  HIP_CHECK(hipMalloc(&dD, ref.size() * sizeof(double)));
  k_f64_16x16x4<Semantics::Native, double, NEG><<<1, WAVE>>>(dA, dB, dC, dD);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<double> got(ref.size());
  HIP_CHECK(hipMemcpy(got.data(), dD, got.size() * sizeof(double), hipMemcpyDeviceToHost));
  for (std::size_t i = 0; i < got.size(); ++i)
    EXPECT_EQ(bits_of(got[i]), bits_of(ref[i])) << "elem " << i;
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dD);
}

TEST(MfmaF64_16x16x4_NEG5, FpsanMatchesScalarReference) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  constexpr int NEG = 5;
  auto d = make_f64_16_inputs();
  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VD = Value<double, S, kCC>;
    std::vector<std::uint64_t> ref(F64_M * F64_N);
    for (int i = 0; i < F64_M; ++i)
      for (int j = 0; j < F64_N; ++j) {
        VD acc = host_neg_if(VD(d.C[i * F64_N + j]), (NEG & 4) != 0);
        for (int k = 0; k < F64_K; ++k)
          acc = acc + host_neg_if(VD(d.A[i * F64_K + k]), (NEG & 1) != 0) *
                          host_neg_if(VD(d.B[k * F64_N + j]), (NEG & 2) != 0);
        ref[i * F64_N + j] = acc.fpsan_payload();
      }
    double *dA = to_dev(d.A), *dB = to_dev(d.B), *dC = to_dev(d.C);
    std::uint64_t *dD;
    HIP_CHECK(hipMalloc(&dD, ref.size() * sizeof(std::uint64_t)));
    k_f64_16x16x4<S, std::uint64_t, NEG><<<1, WAVE>>>(dA, dB, dC, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint64_t> got(ref.size());
    HIP_CHECK(hipMemcpy(got.data(), dD, got.size() * sizeof(std::uint64_t), hipMemcpyDeviceToHost));
    for (std::size_t i = 0; i < got.size(); ++i)
      EXPECT_EQ(got[i], ref[i]) << "elem " << i;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
  });
}
