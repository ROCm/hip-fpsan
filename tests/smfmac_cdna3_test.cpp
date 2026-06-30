// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/smfmac_cdna3_test.cpp
//
// GPU tests for the CDNA3 / gfx94x SMFMAC wrappers. Native mode is checked
// against the real hardware result for a host sparse-matmul reference. FPSan
// mode is checked against an independent scalar payload-ring reference.
#include "fpsan/amdgcn_smfmac.hpp"
#include "fpsan/fpsan.hpp"

#include "fpsan_semantics.hpp"

#include "hip_test_utils.hpp"
#include "test_random.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <type_traits>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

static constexpr Conversions kCC = Conversions::Explicit;
static constexpr int WAVE = 64;

namespace {
struct SparseData {
  std::vector<float> A, B, C;
  std::vector<int> idxbuf, p0, p1;
};

SparseData make_sparse_data(int m, int n, int k) {
  const int groups = k / 4;
  const int ccols = 2 * groups;
  SparseData d;
  d.A.resize(m * ccols);
  d.B.resize(k * n);
  d.C.resize(m * n);
  d.idxbuf.assign(WAVE, 0);
  d.p0.resize(m * groups);
  d.p1.resize(m * groups);

  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : d.A)
    x = static_cast<float>(fpsan_test::pick_int(rng, -3, 3));
  for (auto &x : d.B)
    x = static_cast<float>(fpsan_test::pick_int(rng, -2, 2));
  for (auto &x : d.C)
    x = static_cast<float>(fpsan_test::pick_int(rng, -4, 4));

  for (int i = 0; i < m; ++i)
    for (int q = 0; q < groups; ++q) {
      int a0 = fpsan_test::pick_int(rng, 0, 3);
      int a1 = fpsan_test::pick_int(rng, 0, 3);
      while (a1 == a0)
        a1 = fpsan_test::pick_int(rng, 0, 3);
      if (a0 > a1)
        std::swap(a0, a1);
      d.p0[i * groups + q] = a0;
      d.p1[i * groups + q] = a1;
    }

  // CDNA3 f16/bf16 SMFMAC index: lane (q/2)*M+i carries row i, group q
  // at nibble q%2.
  for (int i = 0; i < m; ++i)
    for (int q = 0; q < groups; ++q) {
      const int lane = (q / 2) * m + i;
      d.idxbuf[lane] |= (d.p0[i * groups + q] | (d.p1[i * groups + q] << 2)) << (4 * (q % 2));
    }
  return d;
}

SparseData make_sparse_fp8_data(int m, int k) {
  const int groups = k / 4;
  const int ccols = 2 * groups;
  SparseData d;
  d.A.resize(m * ccols);
  d.B.resize(k * m);
  d.C.resize(m * m);
  d.idxbuf.assign(WAVE, 0);
  d.p0.resize(m * groups);
  d.p1.resize(m * groups);

  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : d.A)
    x = static_cast<float>(fpsan_test::pick_int(rng, -3, 3));
  for (auto &x : d.B)
    x = static_cast<float>(fpsan_test::pick_int(rng, -2, 2));
  for (auto &x : d.C)
    x = static_cast<float>(fpsan_test::pick_int(rng, -3, 3));

  for (int i = 0; i < m; ++i)
    for (int q = 0; q < groups; ++q) {
      int a0 = fpsan_test::pick_int(rng, 0, 3);
      int a1 = fpsan_test::pick_int(rng, 0, 3);
      while (a1 == a0)
        a1 = fpsan_test::pick_int(rng, 0, 3);
      if (a0 > a1)
        std::swap(a0, a1);
      d.p0[i * groups + q] = a0;
      d.p1[i * groups + q] = a1;
    }

  // CDNA3 fp8 SMFMAC index packs low/high K halves into separate nibble
  // pairs. This matches the gfx94x fp8-insts layout in amdgcn_smfmac.hpp.
  const int half = groups / 2;
  for (int i = 0; i < m; ++i)
    for (int q = 0; q < groups; ++q) {
      const int lane = ((q % half) / 2) * m + i;
      const int nib = 2 * (q / half) + (q % 2);
      d.idxbuf[lane] |= (d.p0[i * groups + q] | (d.p1[i * groups + q] << 2)) << (4 * nib);
    }
  return d;
}
} // namespace

template <class E, Semantics S, class Out>
__global__ void k_smf_16x16x32(const float *A, const float *B, const float *C, const int *idx,
                               Out *D) {
  using v4e = E __attribute__((ext_vector_type(4)));
  using v8e = E __attribute__((ext_vector_type(8)));
  constexpr int N = 16, K = 32, Cc = K / 2;

  const int lane = threadIdx.x;
  const int g = lane / 16;
  const int j = lane % 16;

  v4e an{};
  for (int h = 0; h < 4; ++h)
    an[h] = static_cast<E>(A[j * Cc + g * 4 + h]);

  v8e bn{};
  for (int e = 0; e < 8; ++e)
    bn[e] = static_cast<E>(B[(g * 8 + e) * N + j]);

  fpsan::v4f_native cn{};
  for (int reg = 0; reg < 4; ++reg)
    cn[reg] = C[(4 * g + reg) * N + j];

  Value<v4e, S, kCC> a{an};
  Value<v8e, S, kCC> b{bn};
  Value<fpsan::v4f_native, S, kCC> c{cn};
  auto d = [&] {
    if constexpr (std::is_same_v<E, _Float16>)
      return fpsan::amdgcn_smfmac_f32_16x16x32_f16<0, 0, S, kCC>(a, b, c, idx[lane]);
    else
      return fpsan::amdgcn_smfmac_f32_16x16x32_bf16<0, 0, S, kCC>(a, b, c, idx[lane]);
  }();

  for (int reg = 0; reg < 4; ++reg) {
    const int i = 4 * g + reg;
    if constexpr (S == Semantics::Native)
      D[i * N + j] = d.get(reg).to_float();
    else
      D[i * N + j] = d.get(reg).fpsan_payload();
  }
}

template <class E, Semantics S, class Out>
__global__ void k_smf_32x32x16(const float *A, const float *B, const float *C, const int *idx,
                               Out *D) {
  using v4e = E __attribute__((ext_vector_type(4)));
  using v8e = E __attribute__((ext_vector_type(8)));
  constexpr int M = 32, N = 32, K = 16, Cc = K / 2;

  const int lane = threadIdx.x;
  v4e an{};
  for (int h = 0; h < 4; ++h)
    an[h] = static_cast<E>(A[(lane % 32) * Cc + (lane / 32) * 4 + h]);

  const int j = (lane % 16) + 16 * ((lane / 16) % 2);
  const int kgrp = (lane / 16) / 2;
  v8e bn{};
  for (int e = 0; e < 8; ++e)
    bn[e] = static_cast<E>(B[(8 * kgrp + e) * N + j]);

  fpsan::v16f_native cn{};
  for (int i = 0; i < M; ++i)
    for (int jj = 0; jj < N; ++jj) {
      auto loc = fpsan::detail::output_loc_32(M, N, i, jj, 0);
      if (loc.lane == lane)
        cn[loc.reg] = C[i * N + jj];
    }

  Value<v4e, S, kCC> a{an};
  Value<v8e, S, kCC> b{bn};
  Value<fpsan::v16f_native, S, kCC> c{cn};
  auto d = [&] {
    if constexpr (std::is_same_v<E, _Float16>)
      return fpsan::amdgcn_smfmac_f32_32x32x16_f16<0, 0, S, kCC>(a, b, c, idx[lane]);
    else
      return fpsan::amdgcn_smfmac_f32_32x32x16_bf16<0, 0, S, kCC>(a, b, c, idx[lane]);
  }();

  for (int i = 0; i < M; ++i)
    for (int jj = 0; jj < N; ++jj) {
      auto loc = fpsan::detail::output_loc_32(M, N, i, jj, 0);
      if (loc.lane == lane) {
        if constexpr (S == Semantics::Native)
          D[i * N + jj] = d.get(loc.reg).to_float();
        else
          D[i * N + jj] = d.get(loc.reg).fpsan_payload();
      }
    }
}

template <class E> static void run_smf_layout(int m, int n, int k) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";

  const int groups = k / 4;
  const int ccols = 2 * groups;
  SparseData data = make_sparse_data(m, n, k);
  std::vector<float> ref(m * n);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) {
      double acc = data.C[i * n + j];
      for (int q = 0; q < groups; ++q) {
        acc += static_cast<double>(static_cast<float>(static_cast<E>(data.A[i * ccols + 2 * q]))) *
               static_cast<double>(static_cast<float>(
                   static_cast<E>(data.B[(4 * q + data.p0[i * groups + q]) * n + j])));
        acc +=
            static_cast<double>(static_cast<float>(static_cast<E>(data.A[i * ccols + 2 * q + 1]))) *
            static_cast<double>(static_cast<float>(
                static_cast<E>(data.B[(4 * q + data.p1[i * groups + q]) * n + j])));
      }
      ref[i * n + j] = static_cast<float>(acc);
    }

  float *dA = to_dev(data.A);
  float *dB = to_dev(data.B);
  float *dC = to_dev(data.C);
  int *dI = to_dev(data.idxbuf);
  float *dD;
  HIP_CHECK(hipMalloc(&dD, ref.size() * sizeof(float)));
  if (m == 16)
    k_smf_16x16x32<E, Semantics::Native, float><<<1, WAVE>>>(dA, dB, dC, dI, dD);
  else
    k_smf_32x32x16<E, Semantics::Native, float><<<1, WAVE>>>(dA, dB, dC, dI, dD);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> got(ref.size());
  HIP_CHECK(hipMemcpy(got.data(), dD, got.size() * sizeof(float), hipMemcpyDeviceToHost));
  for (std::size_t i = 0; i < got.size(); ++i)
    EXPECT_EQ(bits_of(got[i]), bits_of(ref[i])) << "elem " << i;

  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dI);
  (void)hipFree(dD);
}

template <class E> static void run_smf_fpsan(int m, int n, int k) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";

  const int groups = k / 4;
  const int ccols = 2 * groups;
  SparseData data = make_sparse_data(m, n, k);

  float *dA = to_dev(data.A);
  float *dB = to_dev(data.B);
  float *dC = to_dev(data.C);
  int *dI = to_dev(data.idxbuf);

  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VF = Value<float, S, kCC>;
    using VE = Value<E, S, kCC>;
    std::vector<std::uint32_t> ref(m * n);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) {
        VF acc(data.C[i * n + j]);
        for (int q = 0; q < groups; ++q) {
          acc = acc + fpsan::cast<float>(VE(static_cast<E>(data.A[i * ccols + 2 * q]))) *
                          fpsan::cast<float>(VE(
                              static_cast<E>(data.B[(4 * q + data.p0[i * groups + q]) * n + j])));
          acc = acc + fpsan::cast<float>(VE(static_cast<E>(data.A[i * ccols + 2 * q + 1]))) *
                          fpsan::cast<float>(VE(
                              static_cast<E>(data.B[(4 * q + data.p1[i * groups + q]) * n + j])));
        }
        ref[i * n + j] = acc.fpsan_payload();
      }

    std::uint32_t *dD;
    HIP_CHECK(hipMalloc(&dD, ref.size() * sizeof(std::uint32_t)));
    if (m == 16)
      k_smf_16x16x32<E, S, std::uint32_t><<<1, WAVE>>>(dA, dB, dC, dI, dD);
    else
      k_smf_32x32x16<E, S, std::uint32_t><<<1, WAVE>>>(dA, dB, dC, dI, dD);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<std::uint32_t> got(ref.size());
    HIP_CHECK(hipMemcpy(got.data(), dD, got.size() * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for (std::size_t i = 0; i < got.size(); ++i)
      EXPECT_EQ(got[i], ref[i]) << "elem " << i;

    (void)hipFree(dD);
  });

  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dI);
}

template <class AE, class BE, Semantics S, class Out>
__global__ void k_smf_fp8_16x16x64(const float *A, const float *B, const float *C, const int *idx,
                                   Out *D) {
  using AVec = fpsan::detail::v8_fragment<AE>;
  using BVec = fpsan::detail::v16_fragment<BE>;
  constexpr int N = 16, K = 64, Cc = K / 2;

  const int lane = threadIdx.x;
  const int g = lane / 16;
  const int j = lane % 16;

  AVec an{};
  for (int b = 0; b < 8; ++b) {
    const int c = 16 * (b / 4) + 4 * g + (b % 4);
    an[b] = AE(A[j * Cc + c]);
  }

  BVec bn{};
  for (int e = 0; e < 16; ++e) {
    const int k = 32 * (e / 8) + 8 * g + (e % 8);
    bn[e] = BE(B[k * N + j]);
  }

  fpsan::v4f_native cn{};
  for (int reg = 0; reg < 4; ++reg)
    cn[reg] = C[(4 * g + reg) * N + j];

  Value<AVec, S, kCC> a{an};
  Value<BVec, S, kCC> b{bn};
  Value<fpsan::v4f_native, S, kCC> c{cn};
  auto d = [&] {
    if constexpr (std::is_same_v<AE, fpsan::amd_fp8_e4m3> &&
                  std::is_same_v<BE, fpsan::amd_fp8_e4m3>)
      return fpsan::amdgcn_smfmac_f32_16x16x64_fp8_fp8<0, 0, S, kCC>(a, b, c, idx[lane]);
    else if constexpr (std::is_same_v<AE, fpsan::amd_fp8_e4m3>)
      return fpsan::amdgcn_smfmac_f32_16x16x64_fp8_bf8<0, 0, S, kCC>(a, b, c, idx[lane]);
    else if constexpr (std::is_same_v<BE, fpsan::amd_fp8_e4m3>)
      return fpsan::amdgcn_smfmac_f32_16x16x64_bf8_fp8<0, 0, S, kCC>(a, b, c, idx[lane]);
    else
      return fpsan::amdgcn_smfmac_f32_16x16x64_bf8_bf8<0, 0, S, kCC>(a, b, c, idx[lane]);
  }();

  for (int reg = 0; reg < 4; ++reg) {
    const int i = 4 * g + reg;
    if constexpr (S == Semantics::Native)
      D[i * N + j] = d.get(reg).to_float();
    else
      D[i * N + j] = d.get(reg).fpsan_payload();
  }
}

template <class AE, class BE, Semantics S, class Out>
__global__ void k_smf_fp8_32x32x32(const float *A, const float *B, const float *C, const int *idx,
                                   Out *D) {
  using AVec = fpsan::detail::v8_fragment<AE>;
  using BVec = fpsan::detail::v16_fragment<BE>;
  constexpr int M = 32, N = 32, K = 32, Cc = K / 2;

  const int lane = threadIdx.x;
  const int g = lane / 32;
  const int j = lane % 32;

  AVec an{};
  for (int b = 0; b < 8; ++b) {
    const int c = 8 * (b / 4) + 4 * g + (b % 4);
    an[b] = AE(A[j * Cc + c]);
  }

  const int jcol = (lane % 16) + 16 * ((lane / 16) % 2);
  const int kgrp = (lane / 16) / 2;
  BVec bn{};
  for (int e = 0; e < 16; ++e) {
    const int k = 16 * (e / 8) + 8 * kgrp + 2 * ((e / 2) % 4) + (e % 2);
    bn[e] = BE(B[k * N + jcol]);
  }

  fpsan::v16f_native cn{};
  for (int i = 0; i < M; ++i)
    for (int jj = 0; jj < N; ++jj) {
      auto loc = fpsan::detail::output_loc_32(M, N, i, jj, 0);
      if (loc.lane == lane)
        cn[loc.reg] = C[i * N + jj];
    }

  Value<AVec, S, kCC> a{an};
  Value<BVec, S, kCC> b{bn};
  Value<fpsan::v16f_native, S, kCC> c{cn};
  auto d = [&] {
    if constexpr (std::is_same_v<AE, fpsan::amd_fp8_e4m3> &&
                  std::is_same_v<BE, fpsan::amd_fp8_e4m3>)
      return fpsan::amdgcn_smfmac_f32_32x32x32_fp8_fp8<0, 0, S, kCC>(a, b, c, idx[lane]);
    else if constexpr (std::is_same_v<AE, fpsan::amd_fp8_e4m3>)
      return fpsan::amdgcn_smfmac_f32_32x32x32_fp8_bf8<0, 0, S, kCC>(a, b, c, idx[lane]);
    else if constexpr (std::is_same_v<BE, fpsan::amd_fp8_e4m3>)
      return fpsan::amdgcn_smfmac_f32_32x32x32_bf8_fp8<0, 0, S, kCC>(a, b, c, idx[lane]);
    else
      return fpsan::amdgcn_smfmac_f32_32x32x32_bf8_bf8<0, 0, S, kCC>(a, b, c, idx[lane]);
  }();

  for (int i = 0; i < M; ++i)
    for (int jj = 0; jj < N; ++jj) {
      auto loc = fpsan::detail::output_loc_32(M, N, i, jj, 0);
      if (loc.lane == lane) {
        if constexpr (S == Semantics::Native)
          D[i * N + jj] = d.get(loc.reg).to_float();
        else
          D[i * N + jj] = d.get(loc.reg).fpsan_payload();
      }
    }
}

template <class AE, class BE> static void run_smf_fp8(int m, int k) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";

  const int n = m;
  const int groups = k / 4;
  const int ccols = 2 * groups;
  SparseData data = make_sparse_fp8_data(m, k);
  std::vector<float> ref(m * n);

  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) {
      double acc = data.C[i * n + j];
      for (int q = 0; q < groups; ++q) {
        const int k0 = 4 * q + data.p0[i * groups + q];
        const int k1 = 4 * q + data.p1[i * groups + q];
        const AE a0(data.A[i * ccols + 2 * q]);
        const AE a1(data.A[i * ccols + 2 * q + 1]);
        const BE b0(data.B[k0 * n + j]);
        const BE b1(data.B[k1 * n + j]);
        acc += static_cast<double>(static_cast<float>(a0)) *
               static_cast<double>(static_cast<float>(b0));
        acc += static_cast<double>(static_cast<float>(a1)) *
               static_cast<double>(static_cast<float>(b1));
      }
      ref[i * n + j] = static_cast<float>(acc);
    }

  float *dA = to_dev(data.A);
  float *dB = to_dev(data.B);
  float *dC = to_dev(data.C);
  int *dI = to_dev(data.idxbuf);
  float *dDf;
  HIP_CHECK(hipMalloc(&dDf, ref.size() * sizeof(float)));

  if (m == 16)
    k_smf_fp8_16x16x64<AE, BE, Semantics::Native, float><<<1, WAVE>>>(dA, dB, dC, dI, dDf);
  else
    k_smf_fp8_32x32x32<AE, BE, Semantics::Native, float><<<1, WAVE>>>(dA, dB, dC, dI, dDf);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> got(ref.size());
  HIP_CHECK(hipMemcpy(got.data(), dDf, got.size() * sizeof(float), hipMemcpyDeviceToHost));
  for (std::size_t i = 0; i < got.size(); ++i)
    EXPECT_EQ(bits_of(got[i]), bits_of(ref[i])) << "float elem " << i;

  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VF = Value<float, S, kCC>;
    using VA = Value<AE, S, kCC>;
    using VB = Value<BE, S, kCC>;

    std::vector<std::uint32_t> refp(m * n);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) {
        VF accp(data.C[i * n + j]);
        for (int q = 0; q < groups; ++q) {
          const int k0 = 4 * q + data.p0[i * groups + q];
          const int k1 = 4 * q + data.p1[i * groups + q];
          const AE a0(data.A[i * ccols + 2 * q]);
          const AE a1(data.A[i * ccols + 2 * q + 1]);
          const BE b0(data.B[k0 * n + j]);
          const BE b1(data.B[k1 * n + j]);
          accp = accp + fpsan::cast<float>(VA(a0)) * fpsan::cast<float>(VB(b0));
          accp = accp + fpsan::cast<float>(VA(a1)) * fpsan::cast<float>(VB(b1));
        }
        refp[i * n + j] = accp.fpsan_payload();
      }

    std::uint32_t *dDp;
    HIP_CHECK(hipMalloc(&dDp, refp.size() * sizeof(std::uint32_t)));
    if (m == 16)
      k_smf_fp8_16x16x64<AE, BE, S, std::uint32_t><<<1, WAVE>>>(dA, dB, dC, dI, dDp);
    else
      k_smf_fp8_32x32x32<AE, BE, S, std::uint32_t><<<1, WAVE>>>(dA, dB, dC, dI, dDp);
    HIP_CHECK(hipDeviceSynchronize());

    std::vector<std::uint32_t> gotp(refp.size());
    HIP_CHECK(
        hipMemcpy(gotp.data(), dDp, gotp.size() * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for (std::size_t i = 0; i < gotp.size(); ++i)
      EXPECT_EQ(gotp[i], refp[i]) << "fpsan elem " << i;

    (void)hipFree(dDp);
  });

  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dI);
  (void)hipFree(dDf);
}

TEST(SmfmacF16_16x16x32, LayoutMatchesHardware) { run_smf_layout<_Float16>(16, 16, 32); }

TEST(SmfmacF16_16x16x32, FpsanMatchesScalarReference) { run_smf_fpsan<_Float16>(16, 16, 32); }

TEST(SmfmacBf16_16x16x32, LayoutMatchesHardware) { run_smf_layout<__bf16>(16, 16, 32); }

TEST(SmfmacBf16_16x16x32, FpsanMatchesScalarReference) { run_smf_fpsan<__bf16>(16, 16, 32); }

TEST(SmfmacF16_32x32x16, LayoutMatchesHardware) { run_smf_layout<_Float16>(32, 32, 16); }

TEST(SmfmacF16_32x32x16, FpsanMatchesScalarReference) { run_smf_fpsan<_Float16>(32, 32, 16); }

TEST(SmfmacBf16_32x32x16, LayoutMatchesHardware) { run_smf_layout<__bf16>(32, 32, 16); }

TEST(SmfmacBf16_32x32x16, FpsanMatchesScalarReference) { run_smf_fpsan<__bf16>(32, 32, 16); }

TEST(SmfmacFp8_16x16x64, FP8_FP8) { run_smf_fp8<fpsan::amd_fp8_e4m3, fpsan::amd_fp8_e4m3>(16, 64); }

TEST(SmfmacFp8_16x16x64, FP8_BF8) { run_smf_fp8<fpsan::amd_fp8_e4m3, fpsan::amd_fp8_e5m2>(16, 64); }

TEST(SmfmacFp8_16x16x64, BF8_FP8) { run_smf_fp8<fpsan::amd_fp8_e5m2, fpsan::amd_fp8_e4m3>(16, 64); }

TEST(SmfmacFp8_16x16x64, BF8_BF8) { run_smf_fp8<fpsan::amd_fp8_e5m2, fpsan::amd_fp8_e5m2>(16, 64); }

TEST(SmfmacFp8_32x32x32, FP8_FP8) { run_smf_fp8<fpsan::amd_fp8_e4m3, fpsan::amd_fp8_e4m3>(32, 32); }

TEST(SmfmacFp8_32x32x32, FP8_BF8) { run_smf_fp8<fpsan::amd_fp8_e4m3, fpsan::amd_fp8_e5m2>(32, 32); }

TEST(SmfmacFp8_32x32x32, BF8_FP8) { run_smf_fp8<fpsan::amd_fp8_e5m2, fpsan::amd_fp8_e4m3>(32, 32); }

TEST(SmfmacFp8_32x32x32, BF8_BF8) { run_smf_fp8<fpsan::amd_fp8_e5m2, fpsan::amd_fp8_e5m2>(32, 32); }
