// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mfma_cdna3_parts.hpp"

struct F64Data {
  std::vector<double> A, B, C;
};

template <class T> T host_neg_if(T v, bool neg) { return neg ? -v : v; }

static constexpr int F64S_M = 4, F64S_N = 4, F64S_K = 4, F64S_B = 4;

static F64Data make_f64_4_inputs() {
  F64Data d;
  d.A.resize(F64S_B * F64S_M * F64S_K);
  d.B.resize(F64S_B * F64S_K * F64S_N);
  d.C.resize(F64S_B * F64S_M * F64S_N);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : d.A)
    x = fpsan_test::pick_int_valued<double>(rng, -3, 3);
  for (auto &x : d.B)
    x = fpsan_test::pick_int_valued<double>(rng, -2, 2);
  for (auto &x : d.C)
    x = fpsan_test::pick_int_valued<double>(rng, -4, 4);
  return d;
}

template <Semantics S, class Out, int NEG = 0>
__global__ void k_f64_4x4x4(const double *A, const double *B, const double *C, Out *D) {
  const int lane = threadIdx.x;
  const int in_k = lane / 16;
  const int blk = (lane % 16) / 4;
  const int idx = lane % 4;
  const double an = A[(blk * F64S_M + idx) * F64S_K + in_k];
  const double bn = B[(blk * F64S_K + in_k) * F64S_N + idx];
  const int out_i = lane / 16;
  const int out_j = lane % 4;
  Value<double, S, kCC> a{an}, b{bn}, c{C[(blk * F64S_M + out_i) * F64S_N + out_j]};
  auto d = fpsan::amdgcn_mfma_f64_4x4x4f64<0, 0, NEG, S, kCC>(a, b, c);
  if constexpr (S == Semantics::Native)
    D[(blk * F64S_M + out_i) * F64S_N + out_j] = d.to_float();
  else
    D[(blk * F64S_M + out_i) * F64S_N + out_j] = d.fpsan_payload();
}

TEST(MfmaF64_4x4x4, LayoutMatchesHardware) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  auto d = make_f64_4_inputs();
  std::vector<double> ref(F64S_B * F64S_M * F64S_N);
  for (int blk = 0; blk < F64S_B; ++blk)
    for (int i = 0; i < F64S_M; ++i)
      for (int j = 0; j < F64S_N; ++j) {
        double acc = d.C[(blk * F64S_M + i) * F64S_N + j];
        for (int k = 0; k < F64S_K; ++k)
          acc += d.A[(blk * F64S_M + i) * F64S_K + k] * d.B[(blk * F64S_K + k) * F64S_N + j];
        ref[(blk * F64S_M + i) * F64S_N + j] = acc;
      }
  double *dA = to_dev(d.A), *dB = to_dev(d.B), *dC = to_dev(d.C), *dD;
  HIP_CHECK(hipMalloc(&dD, ref.size() * sizeof(double)));
  k_f64_4x4x4<Semantics::Native, double><<<1, WAVE>>>(dA, dB, dC, dD);
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

TEST(MfmaF64_4x4x4, FpsanMatchesScalarReference) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  auto d = make_f64_4_inputs();
  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VD = Value<double, S, kCC>;
    std::vector<std::uint64_t> ref(F64S_B * F64S_M * F64S_N);
    for (int blk = 0; blk < F64S_B; ++blk)
      for (int i = 0; i < F64S_M; ++i)
        for (int j = 0; j < F64S_N; ++j) {
          VD acc(d.C[(blk * F64S_M + i) * F64S_N + j]);
          for (int k = 0; k < F64S_K; ++k)
            acc = acc + VD(d.A[(blk * F64S_M + i) * F64S_K + k]) *
                            VD(d.B[(blk * F64S_K + k) * F64S_N + j]);
          ref[(blk * F64S_M + i) * F64S_N + j] = acc.fpsan_payload();
        }
    double *dA = to_dev(d.A), *dB = to_dev(d.B), *dC = to_dev(d.C);
    std::uint64_t *dD;
    HIP_CHECK(hipMalloc(&dD, ref.size() * sizeof(std::uint64_t)));
    k_f64_4x4x4<S, std::uint64_t><<<1, WAVE>>>(dA, dB, dC, dD);
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

TEST(MfmaF64_4x4x4_NEG5, LayoutMatchesHardware) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  constexpr int NEG = 5;
  auto d = make_f64_4_inputs();
  std::vector<double> ref(F64S_B * F64S_M * F64S_N);
  for (int blk = 0; blk < F64S_B; ++blk)
    for (int i = 0; i < F64S_M; ++i)
      for (int j = 0; j < F64S_N; ++j) {
        double acc = host_neg_if(d.C[(blk * F64S_M + i) * F64S_N + j], (NEG & 4) != 0);
        for (int k = 0; k < F64S_K; ++k)
          acc += host_neg_if(d.A[(blk * F64S_M + i) * F64S_K + k], (NEG & 1) != 0) *
                 host_neg_if(d.B[(blk * F64S_K + k) * F64S_N + j], (NEG & 2) != 0);
        ref[(blk * F64S_M + i) * F64S_N + j] = acc;
      }
  double *dA = to_dev(d.A), *dB = to_dev(d.B), *dC = to_dev(d.C), *dD;
  HIP_CHECK(hipMalloc(&dD, ref.size() * sizeof(double)));
  k_f64_4x4x4<Semantics::Native, double, NEG><<<1, WAVE>>>(dA, dB, dC, dD);
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

TEST(MfmaF64_4x4x4_NEG5, FpsanMatchesScalarReference) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  constexpr int NEG = 5;
  auto d = make_f64_4_inputs();
  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VD = Value<double, S, kCC>;
    std::vector<std::uint64_t> ref(F64S_B * F64S_M * F64S_N);
    for (int blk = 0; blk < F64S_B; ++blk)
      for (int i = 0; i < F64S_M; ++i)
        for (int j = 0; j < F64S_N; ++j) {
          VD acc = host_neg_if(VD(d.C[(blk * F64S_M + i) * F64S_N + j]), (NEG & 4) != 0);
          for (int k = 0; k < F64S_K; ++k)
            acc = acc + host_neg_if(VD(d.A[(blk * F64S_M + i) * F64S_K + k]), (NEG & 1) != 0) *
                            host_neg_if(VD(d.B[(blk * F64S_K + k) * F64S_N + j]), (NEG & 2) != 0);
          ref[(blk * F64S_M + i) * F64S_N + j] = acc.fpsan_payload();
        }
    double *dA = to_dev(d.A), *dB = to_dev(d.B), *dC = to_dev(d.C);
    std::uint64_t *dD;
    HIP_CHECK(hipMalloc(&dD, ref.size() * sizeof(std::uint64_t)));
    k_f64_4x4x4<S, std::uint64_t, NEG><<<1, WAVE>>>(dA, dB, dC, dD);
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
