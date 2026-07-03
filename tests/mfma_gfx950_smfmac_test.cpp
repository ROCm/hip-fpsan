// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mfma_gfx950_parts.hpp"

// ---------------------------------------------------------------------------
// Sparse MFMA (SMFMAC), f16/bf16. The Float path forwards to the builtin
// (6-arg signature: A, B, C, idx, CBSZ, ABID -- verified correct on gfx950);
// the FPSan path runs the sparse 2:4 software dataflow (see amdgcn_smfmac.hpp).
//
// This particular case is a Float sanity check that holds for any idx: with
// A == 0 every selected product is 0, so D == C. It exercises the Float builtin
// end-to-end and the C/D output layout. The exhaustive FPSan-vs-scalar-reference
// checks live in the SmfmacF16_*/SmfmacFp8_* suites further down.
// ---------------------------------------------------------------------------
using fpsan::v16bf_native;
using fpsan::v16h_native;

template <Semantics S, class AVec, class BVec, class CVec, int M_, int N_, class Wrap>
__global__ void k_smfmac_zeroA(const CVec *Cin, CVec *Dout, int idx, Wrap wrap) {
  int lane = threadIdx.x;
  Value<AVec, S, kCC> a{AVec{}};    // A = 0
  Value<BVec, S, kCC> b{BVec{}};    // B = 0
  Value<CVec, S, kCC> c{Cin[lane]}; // per-lane C fragment, verbatim
  auto d = wrap(a, b, c, idx);
  if constexpr (S == Semantics::Native)
    Dout[lane] = static_cast<CVec>(d);
}

namespace {
// Wrap each SMFMAC opcode in a functor so a single kernel template covers all.
struct Smf_16x16x64_f16 {
  template <Semantics S, class C>
  __device__ auto operator()(Value<fpsan::v8h_native, S, kCC> a,
                             Value<fpsan::v16h_native, S, kCC> b, C c, int idx) const {
    return fpsan::amdgcn_smfmac_f32_16x16x64_f16<0, 0, S, kCC>(a, b, c, idx);
  }
};
struct Smf_32x32x32_f16 {
  template <Semantics S, class C>
  __device__ auto operator()(Value<fpsan::v8h_native, S, kCC> a,
                             Value<fpsan::v16h_native, S, kCC> b, C c, int idx) const {
    return fpsan::amdgcn_smfmac_f32_32x32x32_f16<0, 0, S, kCC>(a, b, c, idx);
  }
};
} // namespace

template <class Wrap, class CVec, int LANEREGS> void run_smfmac_zeroA() {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  // CVec is the per-lane accumulator fragment (v4f / v16f). Fill C with a
  // distinct value per (lane, reg) and require D == C.
  std::vector<CVec> hC(WAVE), hD(WAVE);
  for (int l = 0; l < WAVE; ++l)
    for (int r = 0; r < LANEREGS; ++r)
      hC[l][r] = static_cast<float>((l * LANEREGS + r) % 17 - 8);
  CVec *dC, *dD;
  HIP_CHECK(hipMalloc(&dC, WAVE * sizeof(CVec)));
  HIP_CHECK(hipMalloc(&dD, WAVE * sizeof(CVec)));
  HIP_CHECK(hipMemcpy(dC, hC.data(), WAVE * sizeof(CVec), hipMemcpyHostToDevice));
  k_smfmac_zeroA<Semantics::Native, fpsan::v8h_native, fpsan::v16h_native, CVec, 0, 0, Wrap>
      <<<1, WAVE>>>(dC, dD, 0, Wrap{});
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(hD.data(), dD, WAVE * sizeof(CVec), hipMemcpyDeviceToHost));
  for (int l = 0; l < WAVE; ++l)
    for (int r = 0; r < LANEREGS; ++r)
      EXPECT_EQ(bits_of(hD[l][r]), bits_of(hC[l][r])) << "lane " << l << " reg " << r;
  (void)hipFree(dC);
  (void)hipFree(dD);
}

TEST(Smfmac, F32_16x16x64_F16_ZeroAGivesC) { run_smfmac_zeroA<Smf_16x16x64_f16, v4f_native, 4>(); }
TEST(Smfmac, F32_32x32x32_F16_ZeroAGivesC) {
  run_smfmac_zeroA<Smf_32x32x32_f16, v16f_native, 16>();
}

// ---------------------------------------------------------------------------
// Sparse MFMA 16x16x64 f16 (V_SMFMAC_F32_16X16X64_F16): real golden test of
// the reverse-engineered 2:4 sparse dataflow. Layouts (see amdgcn_smfmac.hpp):
//   A_comp[i][c] @ lane (c/8)*16+i, half c%8; B[k][j] @ lane j+16*((k%32)/8),
//   vreg (k%8)/2+(k>=32?4:0), half k%2; D via output_loc_32; index per lane =
//   4 nibbles for its 4 groups, nibble = idx0 | (idx1<<2).
// ---------------------------------------------------------------------------
static constexpr int QM = 16, QN = 16, QK = 64, QC = 32; // QC = compressed K

template <int CBSZ, int ABID, Semantics S, class Out>
__global__ void k_smf64(const float *Acomp, const float *B, const float *C, const int *idxbuf,
                        Out *D) {
  using v8h = fpsan::v8h_native;
  using v16h = fpsan::v16h_native;
  int lane = threadIdx.x, g = lane / 16, nlane = lane % 16;
  v8h an{};
  for (int h = 0; h < 8; ++h)
    an[h] = static_cast<_Float16>(Acomp[nlane * QC + (g * 8 + h)]);
  v16h bn{};
  for (int vr = 0; vr < 8; ++vr)
    for (int h = 0; h < 2; ++h) {
      int k = (vr < 4) ? (8 * g + 2 * vr + h) : (32 + 8 * g + 2 * (vr - 4) + h);
      bn[2 * vr + h] = static_cast<_Float16>(B[k * QN + nlane]);
    }
  fpsan::v4f_native cn{};
  for (int reg = 0; reg < 4; ++reg) {
    int i = 4 * g + reg, j = nlane;
    cn[reg] = C[i * QN + j];
  }
  Value<v8h, S, kCC> a{an};
  Value<v16h, S, kCC> b{bn};
  Value<fpsan::v4f_native, S, kCC> c{cn};
  auto d = fpsan::amdgcn_smfmac_f32_16x16x64_f16<CBSZ, ABID, S, kCC>(a, b, c, idxbuf[lane]);
  for (int reg = 0; reg < 4; ++reg) {
    int i = 4 * g + reg, j = nlane;
    if constexpr (S == Semantics::Native)
      D[i * QN + j] = d.get(reg).to_float();
    else
      D[i * QN + j] = d.get(reg).fpsan_payload();
  }
}

namespace {
struct SmfData {
  std::vector<float> A, B, C; // A: 16x32 compressed, B: 64x16, C: 16x16
  std::vector<int> idxbuf;    // per-lane (64)
  std::vector<int> p0, p1;    // sel[i*16+q] live K offsets, i in 0..15 q in 0..15
};
SmfData make_smf_data() {
  SmfData d;
  d.A.resize(QM * QC);
  d.B.resize(QK * QN);
  d.C.resize(QM * QN);
  d.idxbuf.assign(64, 0);
  d.p0.resize(QM * 16);
  d.p1.resize(QM * 16);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : d.A)
    x = fpsan_test::pick_int_valued<float>(rng, -3, 3);
  for (auto &x : d.B)
    x = fpsan_test::pick_int_valued<float>(rng, -2, 2);
  for (auto &x : d.C)
    x = fpsan_test::pick_int_valued<float>(rng, -4, 4);
  // Random valid 2:4 selectors (p0 < p1) per (row i, group q).
  for (int i = 0; i < QM; ++i)
    for (int q = 0; q < 16; ++q) {
      int a0 = fpsan_test::pick_int(rng, 0, 3), a1 = fpsan_test::pick_int(rng, 0, 3);
      while (a1 == a0)
        a1 = fpsan_test::pick_int(rng, 0, 3);
      if (a0 > a1) {
        int t = a0;
        a0 = a1;
        a1 = t;
      }
      d.p0[i * 16 + q] = a0;
      d.p1[i * 16 + q] = a1;
    }
  // Encode into per-lane index: lane L=(q/4)*16+i holds nibble q%4 for row i.
  for (int L = 0; L < 64; ++L) {
    int i = L % 16, baseq = (L / 16) * 4, v = 0;
    for (int n = 0; n < 4; ++n) {
      int q = baseq + n;
      v |= (d.p0[i * 16 + q] | (d.p1[i * 16 + q] << 2)) << (4 * n);
    }
    d.idxbuf[L] = v;
  }
  return d;
}
} // namespace

template <int CBSZ, int ABID> void run_smf64_f16_fpsan();

TEST(SmfmacF16_16x16x64, LayoutMatchesHardware) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  SmfData m = make_smf_data();
  std::vector<float> ref(QM * QN);
  for (int i = 0; i < QM; ++i)
    for (int j = 0; j < QN; ++j) {
      double acc = m.C[i * QN + j];
      for (int q = 0; q < 16; ++q) {
        acc += static_cast<double>(m.A[i * QC + 2 * q]) *
               static_cast<double>(m.B[(4 * q + m.p0[i * 16 + q]) * QN + j]);
        acc += static_cast<double>(m.A[i * QC + 2 * q + 1]) *
               static_cast<double>(m.B[(4 * q + m.p1[i * 16 + q]) * QN + j]);
      }
      ref[i * QN + j] = static_cast<float>(acc);
    }
  float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C), *dD;
  int *dI = to_dev(m.idxbuf);
  HIP_CHECK(hipMalloc(&dD, QM * QN * sizeof(float)));
  k_smf64<0, 0, Semantics::Native, float><<<1, WAVE>>>(dA, dB, dC, dI, dD);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> got(QM * QN);
  HIP_CHECK(hipMemcpy(got.data(), dD, QM * QN * sizeof(float), hipMemcpyDeviceToHost));
  for (int t = 0; t < QM * QN; ++t)
    EXPECT_EQ(bits_of(got[t]), bits_of(ref[t])) << "at " << t;
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dD);
  (void)hipFree(dI);
}

TEST(SmfmacF16_16x16x64, FpsanMatchesScalarReference) { run_smf64_f16_fpsan<0, 0>(); }

template <int CBSZ, int ABID> void run_smf64_f16_fpsan() {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  SmfData m = make_smf_data();
  float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C);
  int *dI = to_dev(m.idxbuf);
  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VF = Value<float, S, kCC>;
    using VH = Value<_Float16, S, kCC>;
    std::vector<std::uint32_t> ref(QM * QN);
    for (int i = 0; i < QM; ++i)
      for (int j = 0; j < QN; ++j) {
        VF acc(m.C[i * QN + j]);
        for (int q = 0; q < 16; ++q) {
          acc = acc + fpsan::cast<float>(VH(static_cast<_Float16>(m.A[i * QC + 2 * q]))) *
                          fpsan::cast<float>(
                              VH(static_cast<_Float16>(m.B[(4 * q + m.p0[i * 16 + q]) * QN + j])));
          acc = acc + fpsan::cast<float>(VH(static_cast<_Float16>(m.A[i * QC + 2 * q + 1]))) *
                          fpsan::cast<float>(
                              VH(static_cast<_Float16>(m.B[(4 * q + m.p1[i * 16 + q]) * QN + j])));
        }
        ref[i * QN + j] = acc.fpsan_payload();
      }
    std::uint32_t *dD;
    HIP_CHECK(hipMalloc(&dD, QM * QN * sizeof(std::uint32_t)));
    k_smf64<CBSZ, ABID, S, std::uint32_t><<<1, WAVE>>>(dA, dB, dC, dI, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(QM * QN);
    HIP_CHECK(hipMemcpy(got.data(), dD, QM * QN * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for (int t = 0; t < QM * QN; ++t)
      EXPECT_EQ(got[t], ref[t]) << "at " << t;
    (void)hipFree(dD);
  });
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dI);
}

TEST(SmfmacF16_16x16x64_Modifiers, CBSZNonzeroUsesFirstIndexSetLayout) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  SmfData m = make_smf_data();
  std::vector<float> ref(QM * QN);
  for (int i = 0; i < QM; ++i)
    for (int j = 0; j < QN; ++j) {
      double acc = m.C[i * QN + j];
      for (int q = 0; q < 16; ++q) {
        acc += static_cast<double>(m.A[i * QC + 2 * q]) *
               static_cast<double>(m.B[(4 * q + m.p0[i * 16 + q]) * QN + j]);
        acc += static_cast<double>(m.A[i * QC + 2 * q + 1]) *
               static_cast<double>(m.B[(4 * q + m.p1[i * 16 + q]) * QN + j]);
      }
      ref[i * QN + j] = static_cast<float>(acc);
    }
  float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C), *dD;
  int *dI = to_dev(m.idxbuf);
  HIP_CHECK(hipMalloc(&dD, QM * QN * sizeof(float)));
  k_smf64<1, 3, Semantics::Native, float><<<1, WAVE>>>(dA, dB, dC, dI, dD);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> got(QM * QN);
  HIP_CHECK(hipMemcpy(got.data(), dD, QM * QN * sizeof(float), hipMemcpyDeviceToHost));
  for (int t = 0; t < QM * QN; ++t)
    EXPECT_EQ(bits_of(got[t]), bits_of(ref[t])) << "at " << t;
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dD);
  (void)hipFree(dI);
}

TEST(SmfmacF16_16x16x64_Modifiers, CBSZNonzeroUsesFirstIndexSetFpsan) {
  run_smf64_f16_fpsan<1, 3>();
}

// ---------------------------------------------------------------------------
// Sparse MFMA 32x32x32 f16 (V_SMFMAC_F32_32x32x32_F16). Layouts verified on
// MI350 (see amdgcn_smfmac.hpp smfmac_software_32x32x32). K=32 logical (16
// compressed), 8 groups.
// ---------------------------------------------------------------------------
static constexpr int TM = 32, TN = 32, TK = 32, TC = 16; // TC = compressed K

template <Semantics S, class Out>
__global__ void k_smf32(const float *Acomp, const float *B, const float *C, const int *idxbuf,
                        Out *D) {
  using v8h = fpsan::v8h_native;
  using v16h = fpsan::v16h_native;
  int lane = threadIdx.x;
  v8h an{};
  for (int h = 0; h < 8; ++h)
    an[h] = static_cast<_Float16>(Acomp[(lane % 32) * TC + ((lane / 32) * 8 + h)]);
  v16h bn{};
  int jcol = (lane % 16) + 16 * ((lane / 16) % 2), kgrp = (lane / 16) / 2;
  for (int e = 0; e < 16; ++e) {
    int k = 16 * (e / 8) + 8 * kgrp + 2 * ((e / 2) % 4) + (e % 2);
    bn[e] = static_cast<_Float16>(B[k * TN + jcol]);
  }
  fpsan::v16f_native cn{};
  for (int i = 0; i < TM; ++i)
    for (int j = 0; j < TN; ++j) {
      auto loc = fpsan::detail::output_loc_32(TM, TN, i, j, 0);
      if (loc.lane == lane)
        cn[loc.reg] = C[i * TN + j];
    }
  Value<v8h, S, kCC> a{an};
  Value<v16h, S, kCC> b{bn};
  Value<fpsan::v16f_native, S, kCC> c{cn};
  auto d = fpsan::amdgcn_smfmac_f32_32x32x32_f16<0, 0, S, kCC>(a, b, c, idxbuf[lane]);
  for (int i = 0; i < TM; ++i)
    for (int j = 0; j < TN; ++j) {
      auto loc = fpsan::detail::output_loc_32(TM, TN, i, j, 0);
      if (loc.lane == lane) {
        if constexpr (S == Semantics::Native)
          D[i * TN + j] = d.get(loc.reg).to_float();
        else
          D[i * TN + j] = d.get(loc.reg).fpsan_payload();
      }
    }
}

namespace {
struct Smf32Data {
  std::vector<float> A, B, C; // A: 32x16, B: 32x32, C: 32x32
  std::vector<int> idxbuf;    // 64
  std::vector<int> p0, p1;    // [32*8]
};
Smf32Data make_smf32_data() {
  Smf32Data d;
  d.A.resize(TM * TC);
  d.B.resize(TK * TN);
  d.C.resize(TM * TN);
  d.idxbuf.assign(64, 0);
  d.p0.resize(TM * 8);
  d.p1.resize(TM * 8);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : d.A)
    x = fpsan_test::pick_int_valued<float>(rng, -3, 3);
  for (auto &x : d.B)
    x = fpsan_test::pick_int_valued<float>(rng, -2, 2);
  for (auto &x : d.C)
    x = fpsan_test::pick_int_valued<float>(rng, -4, 4);
  for (int i = 0; i < TM; ++i)
    for (int q = 0; q < 8; ++q) {
      int a0 = fpsan_test::pick_int(rng, 0, 3), a1 = fpsan_test::pick_int(rng, 0, 3);
      while (a1 == a0)
        a1 = fpsan_test::pick_int(rng, 0, 3);
      if (a0 > a1) {
        int t = a0;
        a0 = a1;
        a1 = t;
      }
      d.p0[i * 8 + q] = a0;
      d.p1[i * 8 + q] = a1;
    }
  for (int L = 0; L < 64; ++L) {
    int i = L % 32, baseq = (L / 32) * 4, v = 0;
    for (int n = 0; n < 4; ++n) {
      int q = baseq + n;
      v |= (d.p0[i * 8 + q] | (d.p1[i * 8 + q] << 2)) << (4 * n);
    }
    d.idxbuf[L] = v;
  }
  return d;
}
} // namespace

TEST(SmfmacF16_32x32x32, LayoutMatchesHardware) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  Smf32Data m = make_smf32_data();
  std::vector<float> ref(TM * TN);
  for (int i = 0; i < TM; ++i)
    for (int j = 0; j < TN; ++j) {
      double acc = m.C[i * TN + j];
      for (int q = 0; q < 8; ++q) {
        acc += static_cast<double>(m.A[i * TC + 2 * q]) *
               static_cast<double>(m.B[(4 * q + m.p0[i * 8 + q]) * TN + j]);
        acc += static_cast<double>(m.A[i * TC + 2 * q + 1]) *
               static_cast<double>(m.B[(4 * q + m.p1[i * 8 + q]) * TN + j]);
      }
      ref[i * TN + j] = static_cast<float>(acc);
    }
  float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C), *dD;
  int *dI = to_dev(m.idxbuf);
  HIP_CHECK(hipMalloc(&dD, TM * TN * sizeof(float)));
  k_smf32<Semantics::Native, float><<<1, WAVE>>>(dA, dB, dC, dI, dD);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> got(TM * TN);
  HIP_CHECK(hipMemcpy(got.data(), dD, TM * TN * sizeof(float), hipMemcpyDeviceToHost));
  for (int t = 0; t < TM * TN; ++t)
    EXPECT_EQ(bits_of(got[t]), bits_of(ref[t])) << "at " << t;
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dD);
  (void)hipFree(dI);
}

TEST(SmfmacF16_32x32x32, FpsanMatchesScalarReference) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  Smf32Data m = make_smf32_data();
  float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C);
  int *dI = to_dev(m.idxbuf);
  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VF = Value<float, S, kCC>;
    using VH = Value<_Float16, S, kCC>;
    std::vector<std::uint32_t> ref(TM * TN);
    for (int i = 0; i < TM; ++i)
      for (int j = 0; j < TN; ++j) {
        VF acc(m.C[i * TN + j]);
        for (int q = 0; q < 8; ++q) {
          acc = acc + fpsan::cast<float>(VH(static_cast<_Float16>(m.A[i * TC + 2 * q]))) *
                          fpsan::cast<float>(
                              VH(static_cast<_Float16>(m.B[(4 * q + m.p0[i * 8 + q]) * TN + j])));
          acc = acc + fpsan::cast<float>(VH(static_cast<_Float16>(m.A[i * TC + 2 * q + 1]))) *
                          fpsan::cast<float>(
                              VH(static_cast<_Float16>(m.B[(4 * q + m.p1[i * 8 + q]) * TN + j])));
        }
        ref[i * TN + j] = acc.fpsan_payload();
      }
    std::uint32_t *dD;
    HIP_CHECK(hipMalloc(&dD, TM * TN * sizeof(std::uint32_t)));
    k_smf32<S, std::uint32_t><<<1, WAVE>>>(dA, dB, dC, dI, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(TM * TN);
    HIP_CHECK(hipMemcpy(got.data(), dD, TM * TN * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for (int t = 0; t < TM * TN; ++t)
      EXPECT_EQ(got[t], ref[t]) << "at " << t;
    (void)hipFree(dD);
  });
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dI);
}

// ---------------------------------------------------------------------------
// bf16 SMFMAC (16x16x64 / 32x32x32). Same reverse-engineered dataflow as the
// f16 case (smfmac_software_* is element-type-agnostic) -- these instances pin
// that the bf16 wrappers, which share that dataflow, are also correct. Inputs
// are exact small integers, so __bf16 casts are lossless and the host double
// reference is exact (identical reference math to the f16 tests).
// ---------------------------------------------------------------------------
template <Semantics S, class Out>
__global__ void k_smf64_bf16(const float *Acomp, const float *B, const float *C, const int *idxbuf,
                             Out *D) {
  using v8bf = fpsan::v8bf_native;
  using v16bf = fpsan::v16bf_native;
  int lane = threadIdx.x, g = lane / 16, nlane = lane % 16;
  v8bf an{};
  for (int h = 0; h < 8; ++h)
    an[h] = static_cast<__bf16>(Acomp[nlane * QC + (g * 8 + h)]);
  v16bf bn{};
  for (int vr = 0; vr < 8; ++vr)
    for (int h = 0; h < 2; ++h) {
      int k = (vr < 4) ? (8 * g + 2 * vr + h) : (32 + 8 * g + 2 * (vr - 4) + h);
      bn[2 * vr + h] = static_cast<__bf16>(B[k * QN + nlane]);
    }
  fpsan::v4f_native cn{};
  for (int reg = 0; reg < 4; ++reg)
    cn[reg] = C[(4 * g + reg) * QN + nlane];
  Value<v8bf, S, kCC> a{an};
  Value<v16bf, S, kCC> b{bn};
  Value<fpsan::v4f_native, S, kCC> c{cn};
  auto d = fpsan::amdgcn_smfmac_f32_16x16x64_bf16<0, 0, S, kCC>(a, b, c, idxbuf[lane]);
  for (int reg = 0; reg < 4; ++reg) {
    int i = 4 * g + reg, j = nlane;
    if constexpr (S == Semantics::Native)
      D[i * QN + j] = d.get(reg).to_float();
    else
      D[i * QN + j] = d.get(reg).fpsan_payload();
  }
}

TEST(SmfmacBf16_16x16x64, LayoutMatchesHardware) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  SmfData m = make_smf_data();
  std::vector<float> ref(QM * QN);
  for (int i = 0; i < QM; ++i)
    for (int j = 0; j < QN; ++j) {
      double acc = m.C[i * QN + j];
      for (int q = 0; q < 16; ++q) {
        acc += static_cast<double>(m.A[i * QC + 2 * q]) *
               static_cast<double>(m.B[(4 * q + m.p0[i * 16 + q]) * QN + j]);
        acc += static_cast<double>(m.A[i * QC + 2 * q + 1]) *
               static_cast<double>(m.B[(4 * q + m.p1[i * 16 + q]) * QN + j]);
      }
      ref[i * QN + j] = static_cast<float>(acc);
    }
  float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C), *dD;
  int *dI = to_dev(m.idxbuf);
  HIP_CHECK(hipMalloc(&dD, QM * QN * sizeof(float)));
  k_smf64_bf16<Semantics::Native, float><<<1, WAVE>>>(dA, dB, dC, dI, dD);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> got(QM * QN);
  HIP_CHECK(hipMemcpy(got.data(), dD, QM * QN * sizeof(float), hipMemcpyDeviceToHost));
  for (int t = 0; t < QM * QN; ++t)
    EXPECT_EQ(bits_of(got[t]), bits_of(ref[t])) << "at " << t;
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dD);
  (void)hipFree(dI);
}

TEST(SmfmacBf16_16x16x64, FpsanMatchesScalarReference) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  SmfData m = make_smf_data();
  float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C);
  int *dI = to_dev(m.idxbuf);
  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VF = Value<float, S, kCC>;
    using VB = Value<__bf16, S, kCC>;
    std::vector<std::uint32_t> ref(QM * QN);
    for (int i = 0; i < QM; ++i)
      for (int j = 0; j < QN; ++j) {
        VF acc(m.C[i * QN + j]);
        for (int q = 0; q < 16; ++q) {
          acc = acc + fpsan::cast<float>(VB(static_cast<__bf16>(m.A[i * QC + 2 * q]))) *
                          fpsan::cast<float>(
                              VB(static_cast<__bf16>(m.B[(4 * q + m.p0[i * 16 + q]) * QN + j])));
          acc = acc + fpsan::cast<float>(VB(static_cast<__bf16>(m.A[i * QC + 2 * q + 1]))) *
                          fpsan::cast<float>(
                              VB(static_cast<__bf16>(m.B[(4 * q + m.p1[i * 16 + q]) * QN + j])));
        }
        ref[i * QN + j] = acc.fpsan_payload();
      }
    std::uint32_t *dD;
    HIP_CHECK(hipMalloc(&dD, QM * QN * sizeof(std::uint32_t)));
    k_smf64_bf16<S, std::uint32_t><<<1, WAVE>>>(dA, dB, dC, dI, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(QM * QN);
    HIP_CHECK(hipMemcpy(got.data(), dD, QM * QN * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for (int t = 0; t < QM * QN; ++t)
      EXPECT_EQ(got[t], ref[t]) << "at " << t;
    (void)hipFree(dD);
  });
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dI);
}

template <Semantics S, class Out>
__global__ void k_smf32_bf16(const float *Acomp, const float *B, const float *C, const int *idxbuf,
                             Out *D) {
  using v8bf = fpsan::v8bf_native;
  using v16bf = fpsan::v16bf_native;
  int lane = threadIdx.x;
  v8bf an{};
  for (int h = 0; h < 8; ++h)
    an[h] = static_cast<__bf16>(Acomp[(lane % 32) * TC + ((lane / 32) * 8 + h)]);
  v16bf bn{};
  int jcol = (lane % 16) + 16 * ((lane / 16) % 2), kgrp = (lane / 16) / 2;
  for (int e = 0; e < 16; ++e) {
    int k = 16 * (e / 8) + 8 * kgrp + 2 * ((e / 2) % 4) + (e % 2);
    bn[e] = static_cast<__bf16>(B[k * TN + jcol]);
  }
  fpsan::v16f_native cn{};
  for (int i = 0; i < TM; ++i)
    for (int j = 0; j < TN; ++j) {
      auto loc = fpsan::detail::output_loc_32(TM, TN, i, j, 0);
      if (loc.lane == lane)
        cn[loc.reg] = C[i * TN + j];
    }
  Value<v8bf, S, kCC> a{an};
  Value<v16bf, S, kCC> b{bn};
  Value<fpsan::v16f_native, S, kCC> c{cn};
  auto d = fpsan::amdgcn_smfmac_f32_32x32x32_bf16<0, 0, S, kCC>(a, b, c, idxbuf[lane]);
  for (int i = 0; i < TM; ++i)
    for (int j = 0; j < TN; ++j) {
      auto loc = fpsan::detail::output_loc_32(TM, TN, i, j, 0);
      if (loc.lane == lane) {
        if constexpr (S == Semantics::Native)
          D[i * TN + j] = d.get(loc.reg).to_float();
        else
          D[i * TN + j] = d.get(loc.reg).fpsan_payload();
      }
    }
}

TEST(SmfmacBf16_32x32x32, LayoutMatchesHardware) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  Smf32Data m = make_smf32_data();
  std::vector<float> ref(TM * TN);
  for (int i = 0; i < TM; ++i)
    for (int j = 0; j < TN; ++j) {
      double acc = m.C[i * TN + j];
      for (int q = 0; q < 8; ++q) {
        acc += static_cast<double>(m.A[i * TC + 2 * q]) *
               static_cast<double>(m.B[(4 * q + m.p0[i * 8 + q]) * TN + j]);
        acc += static_cast<double>(m.A[i * TC + 2 * q + 1]) *
               static_cast<double>(m.B[(4 * q + m.p1[i * 8 + q]) * TN + j]);
      }
      ref[i * TN + j] = static_cast<float>(acc);
    }
  float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C), *dD;
  int *dI = to_dev(m.idxbuf);
  HIP_CHECK(hipMalloc(&dD, TM * TN * sizeof(float)));
  k_smf32_bf16<Semantics::Native, float><<<1, WAVE>>>(dA, dB, dC, dI, dD);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> got(TM * TN);
  HIP_CHECK(hipMemcpy(got.data(), dD, TM * TN * sizeof(float), hipMemcpyDeviceToHost));
  for (int t = 0; t < TM * TN; ++t)
    EXPECT_EQ(bits_of(got[t]), bits_of(ref[t])) << "at " << t;
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dD);
  (void)hipFree(dI);
}

TEST(SmfmacBf16_32x32x32, FpsanMatchesScalarReference) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  Smf32Data m = make_smf32_data();
  float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C);
  int *dI = to_dev(m.idxbuf);
  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VF = Value<float, S, kCC>;
    using VB = Value<__bf16, S, kCC>;
    std::vector<std::uint32_t> ref(TM * TN);
    for (int i = 0; i < TM; ++i)
      for (int j = 0; j < TN; ++j) {
        VF acc(m.C[i * TN + j]);
        for (int q = 0; q < 8; ++q) {
          acc = acc + fpsan::cast<float>(VB(static_cast<__bf16>(m.A[i * TC + 2 * q]))) *
                          fpsan::cast<float>(
                              VB(static_cast<__bf16>(m.B[(4 * q + m.p0[i * 8 + q]) * TN + j])));
          acc = acc + fpsan::cast<float>(VB(static_cast<__bf16>(m.A[i * TC + 2 * q + 1]))) *
                          fpsan::cast<float>(
                              VB(static_cast<__bf16>(m.B[(4 * q + m.p1[i * 8 + q]) * TN + j])));
        }
        ref[i * TN + j] = acc.fpsan_payload();
      }
    std::uint32_t *dD;
    HIP_CHECK(hipMalloc(&dD, TM * TN * sizeof(std::uint32_t)));
    k_smf32_bf16<S, std::uint32_t><<<1, WAVE>>>(dA, dB, dC, dI, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(TM * TN);
    HIP_CHECK(hipMemcpy(got.data(), dD, TM * TN * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for (int t = 0; t < TM * TN; ++t)
      EXPECT_EQ(got[t], ref[t]) << "at " << t;
    (void)hipFree(dD);
  });
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dI);
}

// ===========================================================================
// CDNA3 mai-inst SMFMAC shapes (also present on gfx950): 16x16x32 and 32x32x16,
// f16/bf16. Layouts reverse-engineered + silicon-verified (single-hot probe +
// full-random) -- see amdgcn_smfmac.hpp smfmac_software_16x16x32 / _32x32x16.
// Both shapes: A v4 half, B v8 half, index packs 2 groups/lane.
// ===========================================================================
static constexpr int HM = 16, HN = 16, HK = 32, HC = 16; // HC = compressed K
static constexpr int UM = 32, UN = 32, UK = 16, UC = 8;  // 32x32x16

template <class E, Semantics S, class Out>
__global__ void k_smf1632(const float *A, const float *B, const float *C, const int *idx, Out *D) {
  using v4e = E __attribute__((ext_vector_type(4)));
  using v8e = E __attribute__((ext_vector_type(8)));
  int lane = threadIdx.x, g = lane / 16, nl = lane % 16;
  v4e an{};
  for (int h = 0; h < 4; ++h)
    an[h] = static_cast<E>(A[nl * HC + (g * 4 + h)]);
  v8e bn{};
  for (int e = 0; e < 8; ++e)
    bn[e] = static_cast<E>(B[(g * 8 + e) * HN + nl]);
  fpsan::v4f_native cn{};
  for (int reg = 0; reg < 4; ++reg)
    cn[reg] = C[(4 * g + reg) * HN + nl];
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
    int i = 4 * g + reg, j = nl;
    if constexpr (S == Semantics::Native)
      D[i * HN + j] = d.get(reg).to_float();
    else
      D[i * HN + j] = d.get(reg).fpsan_payload();
  }
}

template <class E, Semantics S, class Out>
__global__ void k_smf3216(const float *A, const float *B, const float *C, const int *idx, Out *D) {
  using v4e = E __attribute__((ext_vector_type(4)));
  using v8e = E __attribute__((ext_vector_type(8)));
  int lane = threadIdx.x;
  v4e an{};
  for (int h = 0; h < 4; ++h)
    an[h] = static_cast<E>(A[(lane % 32) * UC + ((lane / 32) * 4 + h)]);
  v8e bn{};
  int jcol = (lane % 16) + 16 * ((lane / 16) % 2), kgrp = (lane / 16) / 2;
  for (int e = 0; e < 8; ++e)
    bn[e] = static_cast<E>(B[(8 * kgrp + e) * UN + jcol]);
  fpsan::v16f_native cn{};
  for (int i = 0; i < UM; ++i)
    for (int j = 0; j < UN; ++j) {
      auto loc = fpsan::detail::output_loc_32(UM, UN, i, j, 0);
      if (loc.lane == lane)
        cn[loc.reg] = C[i * UN + j];
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
  for (int i = 0; i < UM; ++i)
    for (int j = 0; j < UN; ++j) {
      auto loc = fpsan::detail::output_loc_32(UM, UN, i, j, 0);
      if (loc.lane == lane) {
        if constexpr (S == Semantics::Native)
          D[i * UN + j] = d.get(loc.reg).to_float();
        else
          D[i * UN + j] = d.get(loc.reg).fpsan_payload();
      }
    }
}

namespace {
struct SmfCdna3Data {
  std::vector<float> A, B, C;
  std::vector<int> idxbuf, p0, p1; // p0/p1 indexed [row*G + q]
};
// M/N/K with G groups (=K/4), compressed columns C=2*G, index packs 2 grp/lane.
SmfCdna3Data make_cdna3(int Mm, int Nn, int Kk) {
  const int G = Kk / 4, Cc = 2 * G;
  SmfCdna3Data d;
  d.A.resize(Mm * Cc);
  d.B.resize(Kk * Nn);
  d.C.resize(Mm * Nn);
  d.idxbuf.assign(64, 0);
  d.p0.resize(Mm * G);
  d.p1.resize(Mm * G);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : d.A)
    x = fpsan_test::pick_int_valued<float>(rng, -3, 3);
  for (auto &x : d.B)
    x = fpsan_test::pick_int_valued<float>(rng, -2, 2);
  for (auto &x : d.C)
    x = fpsan_test::pick_int_valued<float>(rng, -4, 4);
  for (int i = 0; i < Mm; ++i)
    for (int q = 0; q < G; ++q) {
      int a0 = fpsan_test::pick_int(rng, 0, 3), a1 = fpsan_test::pick_int(rng, 0, 3);
      while (a1 == a0)
        a1 = fpsan_test::pick_int(rng, 0, 3);
      if (a0 > a1)
        std::swap(a0, a1);
      d.p0[i * G + q] = a0;
      d.p1[i * G + q] = a1;
    }
  // index: lane (q/2)*M+i carries (row i, group q) at nibble q%2.
  for (int i = 0; i < Mm; ++i)
    for (int q = 0; q < G; ++q) {
      int L = (q / 2) * Mm + i;
      d.idxbuf[L] |= (d.p0[i * G + q] | (d.p1[i * G + q] << 2)) << (4 * (q % 2));
    }
  return d;
}
} // namespace

template <class E> static void cdna3_layout_test(int Mm, int Nn, int Kk) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  const int G = Kk / 4, Cc = 2 * G;
  SmfCdna3Data m = make_cdna3(Mm, Nn, Kk);
  std::vector<float> ref(Mm * Nn);
  for (int i = 0; i < Mm; ++i)
    for (int j = 0; j < Nn; ++j) {
      double acc = m.C[i * Nn + j];
      for (int q = 0; q < G; ++q) {
        acc += static_cast<double>(m.A[i * Cc + 2 * q]) *
               static_cast<double>(m.B[(4 * q + m.p0[i * G + q]) * Nn + j]);
        acc += static_cast<double>(m.A[i * Cc + 2 * q + 1]) *
               static_cast<double>(m.B[(4 * q + m.p1[i * G + q]) * Nn + j]);
      }
      ref[i * Nn + j] = static_cast<float>(acc);
    }
  float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C), *dD;
  int *dI = to_dev(m.idxbuf);
  HIP_CHECK(hipMalloc(&dD, Mm * Nn * sizeof(float)));
  if (Mm == 16)
    k_smf1632<E, Semantics::Native, float><<<1, WAVE>>>(dA, dB, dC, dI, dD);
  else
    k_smf3216<E, Semantics::Native, float><<<1, WAVE>>>(dA, dB, dC, dI, dD);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> got(Mm * Nn);
  HIP_CHECK(hipMemcpy(got.data(), dD, Mm * Nn * sizeof(float), hipMemcpyDeviceToHost));
  for (int t = 0; t < Mm * Nn; ++t)
    EXPECT_EQ(bits_of(got[t]), bits_of(ref[t])) << "at " << t;
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dD);
  (void)hipFree(dI);
}

template <class E> static void cdna3_fpsan_test(int Mm, int Nn, int Kk) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  const int G = Kk / 4, Cc = 2 * G;
  SmfCdna3Data m = make_cdna3(Mm, Nn, Kk);
  float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C);
  int *dI = to_dev(m.idxbuf);
  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VF = Value<float, S, kCC>;
    using VE = Value<E, S, kCC>;
    std::vector<std::uint32_t> ref(Mm * Nn);
    for (int i = 0; i < Mm; ++i)
      for (int j = 0; j < Nn; ++j) {
        VF acc(m.C[i * Nn + j]);
        for (int q = 0; q < G; ++q) {
          acc = acc +
                fpsan::cast<float>(VE(static_cast<E>(m.A[i * Cc + 2 * q]))) *
                    fpsan::cast<float>(VE(static_cast<E>(m.B[(4 * q + m.p0[i * G + q]) * Nn + j])));
          acc = acc +
                fpsan::cast<float>(VE(static_cast<E>(m.A[i * Cc + 2 * q + 1]))) *
                    fpsan::cast<float>(VE(static_cast<E>(m.B[(4 * q + m.p1[i * G + q]) * Nn + j])));
        }
        ref[i * Nn + j] = acc.fpsan_payload();
      }
    std::uint32_t *dD;
    HIP_CHECK(hipMalloc(&dD, Mm * Nn * sizeof(std::uint32_t)));
    if (Mm == 16)
      k_smf1632<E, S, std::uint32_t><<<1, WAVE>>>(dA, dB, dC, dI, dD);
    else
      k_smf3216<E, S, std::uint32_t><<<1, WAVE>>>(dA, dB, dC, dI, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(Mm * Nn);
    HIP_CHECK(hipMemcpy(got.data(), dD, Mm * Nn * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for (int t = 0; t < Mm * Nn; ++t)
      EXPECT_EQ(got[t], ref[t]) << "at " << t;
    (void)hipFree(dD);
  });
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dI);
}

TEST(SmfmacF16_16x16x32, LayoutMatchesHardware) { cdna3_layout_test<_Float16>(HM, HN, HK); }
TEST(SmfmacF16_16x16x32, FpsanMatchesScalarReference) { cdna3_fpsan_test<_Float16>(HM, HN, HK); }
TEST(SmfmacBf16_16x16x32, LayoutMatchesHardware) { cdna3_layout_test<__bf16>(HM, HN, HK); }
TEST(SmfmacBf16_16x16x32, FpsanMatchesScalarReference) { cdna3_fpsan_test<__bf16>(HM, HN, HK); }
TEST(SmfmacF16_32x32x16, LayoutMatchesHardware) { cdna3_layout_test<_Float16>(UM, UN, UK); }
TEST(SmfmacF16_32x32x16, FpsanMatchesScalarReference) { cdna3_fpsan_test<_Float16>(UM, UN, UK); }
TEST(SmfmacBf16_32x32x16, LayoutMatchesHardware) { cdna3_layout_test<__bf16>(UM, UN, UK); }
TEST(SmfmacBf16_32x32x16, FpsanMatchesScalarReference) { cdna3_fpsan_test<__bf16>(UM, UN, UK); }

// ===========================================================================
// CDNA3 fp8-insts SMFMAC shapes: 16x16x64 and 32x32x32, fp8/bf8 x fp8/bf8.
// Layouts reverse-engineered + multi-seed full-random verified on MI350 (see
// amdgcn_smfmac.hpp smfmac_software_*_fp8). A = v8 fp8, B = v16 fp8. fp8 packs
// 4 bytes/dword, so A-col map = (Cc/2)*(b/4)+4*g+(b%4) and the index packs the
// low/high K half into the nibble's high bit -- different from the f16 shapes.
// ===========================================================================
static constexpr int FW = 16, FK = 64, FCc = 32;  // 16x16x64: groups=16
static constexpr int FX = 32, FXK = 32, FXC = 16; // 32x32x32: groups=8

template <class AE, class BE, Semantics S, class Out>
__global__ void k_smf_fp8_1664(const float *A, const float *B, const float *C, const int *idx,
                               Out *D) {
  using AVec = fpsan::detail::v8_fragment<AE>;
  using BVec = fpsan::detail::v16_fragment<BE>;
  int lane = threadIdx.x, g = lane / 16, nl = lane % 16;
  AVec an{};
  for (int b = 0; b < 8; ++b) {
    int c = 16 * (b / 4) + 4 * g + (b % 4);
    an[b] = AE(A[nl * FCc + c]);
  }
  BVec bn{};
  for (int e = 0; e < 16; ++e) {
    int k = 32 * (e / 8) + 8 * g + (e % 8);
    bn[e] = BE(B[k * FW + nl]);
  }
  fpsan::v4f_native cn{};
  for (int reg = 0; reg < 4; ++reg)
    cn[reg] = C[(4 * g + reg) * FW + nl];
  Value<AVec, S, kCC> a{an};
  Value<BVec, S, kCC> b{bn};
  Value<fpsan::v4f_native, S, kCC> c{cn};
  auto d = [&] {
    if constexpr (std::is_same_v<AE, fpsan::fp8_e4m3> && std::is_same_v<BE, fpsan::fp8_e4m3>)
      return fpsan::amdgcn_smfmac_f32_16x16x64_fp8_fp8<0, 0, S, kCC>(a, b, c, idx[lane]);
    else if constexpr (std::is_same_v<AE, fpsan::fp8_e4m3>)
      return fpsan::amdgcn_smfmac_f32_16x16x64_fp8_bf8<0, 0, S, kCC>(a, b, c, idx[lane]);
    else if constexpr (std::is_same_v<BE, fpsan::fp8_e4m3>)
      return fpsan::amdgcn_smfmac_f32_16x16x64_bf8_fp8<0, 0, S, kCC>(a, b, c, idx[lane]);
    else
      return fpsan::amdgcn_smfmac_f32_16x16x64_bf8_bf8<0, 0, S, kCC>(a, b, c, idx[lane]);
  }();
  for (int reg = 0; reg < 4; ++reg) {
    int i = 4 * g + reg, j = nl;
    if constexpr (S == Semantics::Native)
      D[i * FW + j] = d.get(reg).to_float();
    else
      D[i * FW + j] = d.get(reg).fpsan_payload();
  }
}

template <class AE, class BE, Semantics S, class Out>
__global__ void k_smf_fp8_3232(const float *A, const float *B, const float *C, const int *idx,
                               Out *D) {
  using AVec = fpsan::detail::v8_fragment<AE>;
  using BVec = fpsan::detail::v16_fragment<BE>;
  int lane = threadIdx.x, g = lane / 32, nl = lane % 32;
  AVec an{};
  for (int b = 0; b < 8; ++b) {
    int c = 8 * (b / 4) + 4 * g + (b % 4);
    an[b] = AE(A[nl * FXC + c]);
  }
  BVec bn{};
  int jcol = (lane % 16) + 16 * ((lane / 16) % 2), kgrp = (lane / 16) / 2;
  for (int e = 0; e < 16; ++e) {
    int k = 16 * (e / 8) + 8 * kgrp + 2 * ((e / 2) % 4) + (e % 2);
    bn[e] = BE(B[k * FX + jcol]);
  }
  fpsan::v16f_native cn{};
  for (int i = 0; i < FX; ++i)
    for (int j = 0; j < FX; ++j) {
      auto loc = fpsan::detail::output_loc_32(FX, FX, i, j, 0);
      if (loc.lane == lane)
        cn[loc.reg] = C[i * FX + j];
    }
  Value<AVec, S, kCC> a{an};
  Value<BVec, S, kCC> b{bn};
  Value<fpsan::v16f_native, S, kCC> c{cn};
  auto d = [&] {
    if constexpr (std::is_same_v<AE, fpsan::fp8_e4m3> && std::is_same_v<BE, fpsan::fp8_e4m3>)
      return fpsan::amdgcn_smfmac_f32_32x32x32_fp8_fp8<0, 0, S, kCC>(a, b, c, idx[lane]);
    else if constexpr (std::is_same_v<AE, fpsan::fp8_e4m3>)
      return fpsan::amdgcn_smfmac_f32_32x32x32_fp8_bf8<0, 0, S, kCC>(a, b, c, idx[lane]);
    else if constexpr (std::is_same_v<BE, fpsan::fp8_e4m3>)
      return fpsan::amdgcn_smfmac_f32_32x32x32_bf8_fp8<0, 0, S, kCC>(a, b, c, idx[lane]);
    else
      return fpsan::amdgcn_smfmac_f32_32x32x32_bf8_bf8<0, 0, S, kCC>(a, b, c, idx[lane]);
  }();
  for (int i = 0; i < FX; ++i)
    for (int j = 0; j < FX; ++j) {
      auto loc = fpsan::detail::output_loc_32(FX, FX, i, j, 0);
      if (loc.lane == lane) {
        if constexpr (S == Semantics::Native)
          D[i * FX + j] = d.get(loc.reg).to_float();
        else
          D[i * FX + j] = d.get(loc.reg).fpsan_payload();
      }
    }
}

namespace {
// Shared data gen for the fp8 SMFMAC shapes (Mm=Nn, K, groups G=K/4).
SmfCdna3Data make_fp8(int Mm, int Kk) {
  const int G = Kk / 4, Cc = 2 * G, Nn = Mm;
  SmfCdna3Data d;
  d.A.resize(Mm * Cc);
  d.B.resize(Kk * Nn);
  d.C.resize(Mm * Nn);
  d.idxbuf.assign(64, 0);
  d.p0.resize(Mm * G);
  d.p1.resize(Mm * G);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : d.A)
    x = fpsan_test::pick_int(rng, -3, 3);
  for (auto &x : d.B)
    x = fpsan_test::pick_int(rng, -2, 2);
  for (auto &x : d.C)
    x = fpsan_test::pick_int(rng, -3, 3);
  for (int i = 0; i < Mm; ++i)
    for (int q = 0; q < G; ++q) {
      int a0 = fpsan_test::pick_int(rng, 0, 3), a1 = fpsan_test::pick_int(rng, 0, 3);
      while (a1 == a0)
        a1 = fpsan_test::pick_int(rng, 0, 3);
      if (a0 > a1)
        std::swap(a0, a1);
      d.p0[i * G + q] = a0;
      d.p1[i * G + q] = a1;
    }
  const int half = G / 2;
  for (int i = 0; i < Mm; ++i)
    for (int q = 0; q < G; ++q) {
      int L = ((q % half) / 2) * Mm + i, nb = 2 * (q / half) + (q % 2);
      d.idxbuf[L] |= (d.p0[i * G + q] | (d.p1[i * G + q] << 2)) << (4 * nb);
    }
  return d;
}
} // namespace

template <class AE, class BE> static void fp8_smf_test(int Mm, int Kk) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  const int G = Kk / 4, Cc = 2 * G, Nn = Mm;
  SmfCdna3Data m = make_fp8(Mm, Kk);
  // Native-mode layout reference (host integer matmul, exact small ints).
  std::vector<float> ref(Mm * Nn);
  for (int i = 0; i < Mm; ++i)
    for (int j = 0; j < Nn; ++j) {
      double acc = m.C[i * Nn + j];
      for (int q = 0; q < G; ++q) {
        int k0 = 4 * q + m.p0[i * G + q], k1 = 4 * q + m.p1[i * G + q];
        acc += static_cast<double>(static_cast<float>(AE(m.A[i * Cc + 2 * q]))) *
               static_cast<double>(static_cast<float>(BE(m.B[k0 * Nn + j])));
        acc += static_cast<double>(static_cast<float>(AE(m.A[i * Cc + 2 * q + 1]))) *
               static_cast<double>(static_cast<float>(BE(m.B[k1 * Nn + j])));
      }
      ref[i * Nn + j] = static_cast<float>(acc);
    }
  float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C), *dDf;
  int *dI = to_dev(m.idxbuf);
  HIP_CHECK(hipMalloc(&dDf, Mm * Nn * sizeof(float)));
  if (Mm == 16)
    k_smf_fp8_1664<AE, BE, Semantics::Native, float><<<1, WAVE>>>(dA, dB, dC, dI, dDf);
  else
    k_smf_fp8_3232<AE, BE, Semantics::Native, float><<<1, WAVE>>>(dA, dB, dC, dI, dDf);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> gotf(Mm * Nn);
  HIP_CHECK(hipMemcpy(gotf.data(), dDf, Mm * Nn * sizeof(float), hipMemcpyDeviceToHost));
  for (int t = 0; t < Mm * Nn; ++t)
    EXPECT_EQ(bits_of(gotf[t]), bits_of(ref[t])) << "float at " << t;
  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VF = Value<float, S, kCC>;
    using VA = Value<AE, S, kCC>;
    using VB = Value<BE, S, kCC>;
    std::vector<std::uint32_t> refp(Mm * Nn);
    for (int i = 0; i < Mm; ++i)
      for (int j = 0; j < Nn; ++j) {
        VF accp(m.C[i * Nn + j]);
        for (int q = 0; q < G; ++q) {
          int k0 = 4 * q + m.p0[i * G + q], k1 = 4 * q + m.p1[i * G + q];
          accp = accp + fpsan::cast<float>(VA(AE(m.A[i * Cc + 2 * q]))) *
                            fpsan::cast<float>(VB(BE(m.B[k0 * Nn + j])));
          accp = accp + fpsan::cast<float>(VA(AE(m.A[i * Cc + 2 * q + 1]))) *
                            fpsan::cast<float>(VB(BE(m.B[k1 * Nn + j])));
        }
        refp[i * Nn + j] = accp.fpsan_payload();
      }
    std::uint32_t *dDp;
    HIP_CHECK(hipMalloc(&dDp, Mm * Nn * sizeof(std::uint32_t)));
    if (Mm == 16)
      k_smf_fp8_1664<AE, BE, S, std::uint32_t><<<1, WAVE>>>(dA, dB, dC, dI, dDp);
    else
      k_smf_fp8_3232<AE, BE, S, std::uint32_t><<<1, WAVE>>>(dA, dB, dC, dI, dDp);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> gotp(Mm * Nn);
    HIP_CHECK(hipMemcpy(gotp.data(), dDp, Mm * Nn * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for (int t = 0; t < Mm * Nn; ++t)
      EXPECT_EQ(gotp[t], refp[t]) << "fpsan at " << t;
    (void)hipFree(dDp);
  });
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dDf);
  (void)hipFree(dI);
}

TEST(SmfmacFp8_16x16x64, FP8_FP8) { fp8_smf_test<fpsan::fp8_e4m3, fpsan::fp8_e4m3>(FW, FK); }
TEST(SmfmacFp8_16x16x64, FP8_BF8) { fp8_smf_test<fpsan::fp8_e4m3, fpsan::fp8_e5m2>(FW, FK); }
TEST(SmfmacFp8_16x16x64, BF8_FP8) { fp8_smf_test<fpsan::fp8_e5m2, fpsan::fp8_e4m3>(FW, FK); }
TEST(SmfmacFp8_16x16x64, BF8_BF8) { fp8_smf_test<fpsan::fp8_e5m2, fpsan::fp8_e5m2>(FW, FK); }
TEST(SmfmacFp8_32x32x32, FP8_FP8) { fp8_smf_test<fpsan::fp8_e4m3, fpsan::fp8_e4m3>(FX, FXK); }
TEST(SmfmacFp8_32x32x32, FP8_BF8) { fp8_smf_test<fpsan::fp8_e4m3, fpsan::fp8_e5m2>(FX, FXK); }
TEST(SmfmacFp8_32x32x32, BF8_FP8) { fp8_smf_test<fpsan::fp8_e5m2, fpsan::fp8_e4m3>(FX, FXK); }
TEST(SmfmacFp8_32x32x32, BF8_BF8) { fp8_smf_test<fpsan::fp8_e5m2, fpsan::fp8_e5m2>(FX, FXK); }

// ===========================================================================
// gfx950 fp8-insts SMFMAC shapes: 16x16x128 and 32x32x64, fp8/bf8 x fp8/bf8.
// Layouts reverse-engineered + multi-seed full-random verified on MI350 (see
// amdgcn_smfmac.hpp smfmac_software_16x16x128_fp8 / _32x32x64_fp8). A = v16 fp8,
// B = v32 fp8. These double the K of the CDNA3 fp8 shapes above.
// ===========================================================================
static constexpr int GW = 16, GK = 128, GCc = 64; // 16x16x128: groups=32
static constexpr int GX = 32, GXK = 64, GXC = 32; // 32x32x64:  groups=16

template <class AE, class BE, Semantics S, class Out>
__global__ void k_smf_fp8big_16128(const float *A, const float *B, const float *C, const int *idx,
                                   Out *D) {
  using AVec = fpsan::detail::v16_fragment<AE>;
  using BVec = fpsan::detail::v32_fragment<BE>;
  int lane = threadIdx.x, g = lane / 16, nl = lane % 16;
  AVec an{};
  for (int b = 0; b < 16; ++b) {
    int hb = b / 4, lb = b % 4;
    int c = lb + 16 * (hb % 2) + 4 * (hb / 2) + 32 * (g / 2) + 8 * (g % 2);
    an[b] = AE(A[nl * GCc + c]);
  }
  BVec bn{};
  for (int e = 0; e < 32; ++e) {
    int k = 32 * (e / 8) + 8 * g + (e % 8);
    bn[e] = BE(B[k * GW + nl]);
  }
  fpsan::v4f_native cn{};
  for (int reg = 0; reg < 4; ++reg)
    cn[reg] = C[(4 * g + reg) * GW + nl];
  Value<AVec, S, kCC> a{an};
  Value<BVec, S, kCC> b{bn};
  Value<fpsan::v4f_native, S, kCC> c{cn};
  auto d = [&] {
    if constexpr (std::is_same_v<AE, fpsan::fp8_e4m3> && std::is_same_v<BE, fpsan::fp8_e4m3>)
      return fpsan::amdgcn_smfmac_f32_16x16x128_fp8_fp8<0, 0, S, kCC>(a, b, c, idx[lane]);
    else if constexpr (std::is_same_v<AE, fpsan::fp8_e4m3>)
      return fpsan::amdgcn_smfmac_f32_16x16x128_fp8_bf8<0, 0, S, kCC>(a, b, c, idx[lane]);
    else if constexpr (std::is_same_v<BE, fpsan::fp8_e4m3>)
      return fpsan::amdgcn_smfmac_f32_16x16x128_bf8_fp8<0, 0, S, kCC>(a, b, c, idx[lane]);
    else
      return fpsan::amdgcn_smfmac_f32_16x16x128_bf8_bf8<0, 0, S, kCC>(a, b, c, idx[lane]);
  }();
  for (int reg = 0; reg < 4; ++reg) {
    int i = 4 * g + reg, j = nl;
    if constexpr (S == Semantics::Native)
      D[i * GW + j] = d.get(reg).to_float();
    else
      D[i * GW + j] = d.get(reg).fpsan_payload();
  }
}

template <class AE, class BE, Semantics S, class Out>
__global__ void k_smf_fp8big_3264(const float *A, const float *B, const float *C, const int *idx,
                                  Out *D) {
  using AVec = fpsan::detail::v16_fragment<AE>;
  using BVec = fpsan::detail::v32_fragment<BE>;
  int lane = threadIdx.x, g = lane / 32, nl = lane % 32, kgrp = lane / 32;
  AVec an{};
  for (int b = 0; b < 16; ++b) {
    int hb = b / 4, lb = b % 4;
    int c = lb + 8 * (hb % 2) + 4 * (hb / 2) + 16 * g;
    an[b] = AE(A[nl * GXC + c]);
  }
  BVec bn{};
  for (int e = 0; e < 32; ++e) {
    int k = 16 * (e / 8) + 8 * kgrp + (e % 8);
    bn[e] = BE(B[k * GX + nl]);
  }
  fpsan::v16f_native cn{};
  for (int i = 0; i < GX; ++i)
    for (int j = 0; j < GX; ++j) {
      auto loc = fpsan::detail::output_loc_32(GX, GX, i, j, 0);
      if (loc.lane == lane)
        cn[loc.reg] = C[i * GX + j];
    }
  Value<AVec, S, kCC> a{an};
  Value<BVec, S, kCC> b{bn};
  Value<fpsan::v16f_native, S, kCC> c{cn};
  auto d = [&] {
    if constexpr (std::is_same_v<AE, fpsan::fp8_e4m3> && std::is_same_v<BE, fpsan::fp8_e4m3>)
      return fpsan::amdgcn_smfmac_f32_32x32x64_fp8_fp8<0, 0, S, kCC>(a, b, c, idx[lane]);
    else if constexpr (std::is_same_v<AE, fpsan::fp8_e4m3>)
      return fpsan::amdgcn_smfmac_f32_32x32x64_fp8_bf8<0, 0, S, kCC>(a, b, c, idx[lane]);
    else if constexpr (std::is_same_v<BE, fpsan::fp8_e4m3>)
      return fpsan::amdgcn_smfmac_f32_32x32x64_bf8_fp8<0, 0, S, kCC>(a, b, c, idx[lane]);
    else
      return fpsan::amdgcn_smfmac_f32_32x32x64_bf8_bf8<0, 0, S, kCC>(a, b, c, idx[lane]);
  }();
  for (int i = 0; i < GX; ++i)
    for (int j = 0; j < GX; ++j) {
      auto loc = fpsan::detail::output_loc_32(GX, GX, i, j, 0);
      if (loc.lane == lane) {
        if constexpr (S == Semantics::Native)
          D[i * GX + j] = d.get(loc.reg).to_float();
        else
          D[i * GX + j] = d.get(loc.reg).fpsan_payload();
      }
    }
}

namespace {
// Data + index gen for the gfx950 fp8 SMFMAC shapes (Mm in {16,32}).
SmfCdna3Data make_fp8big(int Mm, int Kk) {
  const int G = Kk / 4, Cc = 2 * G, Nn = Mm;
  SmfCdna3Data d;
  d.A.resize(Mm * Cc);
  d.B.resize(Kk * Nn);
  d.C.resize(Mm * Nn);
  d.idxbuf.assign(64, 0);
  d.p0.resize(Mm * G);
  d.p1.resize(Mm * G);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : d.A)
    x = fpsan_test::pick_int(rng, -3, 3);
  for (auto &x : d.B)
    x = fpsan_test::pick_int(rng, -2, 2);
  for (auto &x : d.C)
    x = fpsan_test::pick_int(rng, -3, 3);
  for (int i = 0; i < Mm; ++i)
    for (int q = 0; q < G; ++q) {
      int a0 = fpsan_test::pick_int(rng, 0, 3), a1 = fpsan_test::pick_int(rng, 0, 3);
      while (a1 == a0)
        a1 = fpsan_test::pick_int(rng, 0, 3);
      d.p0[i * G + q] = a0;
      d.p1[i * G + q] = a1;
    }
  for (int i = 0; i < Mm; ++i)
    for (int q = 0; q < G; ++q) {
      int L, nb;
      if (Mm == 16) { // 16x16x128
        L = 16 * (2 * (q / 16) + ((q / 4) % 2)) + i;
        nb = 2 * ((q / 8) % 2) + 4 * ((q % 4) / 2) + ((q % 4) % 2);
      } else { // 32x32x64
        L = 32 * (q / 8) + i;
        nb = (q % 2) + 2 * ((q / 4) % 2) + 4 * ((q / 2) % 2);
      }
      d.idxbuf[L] |= (d.p0[i * G + q] | (d.p1[i * G + q] << 2)) << (4 * nb);
    }
  return d;
}
} // namespace

template <class AE, class BE> static void fp8big_smf_test(int Mm, int Kk) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  const int G = Kk / 4, Cc = 2 * G, Nn = Mm;
  SmfCdna3Data m = make_fp8big(Mm, Kk);
  std::vector<float> ref(Mm * Nn);
  for (int i = 0; i < Mm; ++i)
    for (int j = 0; j < Nn; ++j) {
      double acc = m.C[i * Nn + j];
      for (int q = 0; q < G; ++q) {
        int k0 = 4 * q + m.p0[i * G + q], k1 = 4 * q + m.p1[i * G + q];
        acc += static_cast<double>(static_cast<float>(AE(m.A[i * Cc + 2 * q]))) *
               static_cast<double>(static_cast<float>(BE(m.B[k0 * Nn + j])));
        acc += static_cast<double>(static_cast<float>(AE(m.A[i * Cc + 2 * q + 1]))) *
               static_cast<double>(static_cast<float>(BE(m.B[k1 * Nn + j])));
      }
      ref[i * Nn + j] = static_cast<float>(acc);
    }
  float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C), *dDf;
  int *dI = to_dev(m.idxbuf);
  HIP_CHECK(hipMalloc(&dDf, Mm * Nn * sizeof(float)));
  if (Mm == 16)
    k_smf_fp8big_16128<AE, BE, Semantics::Native, float><<<1, WAVE>>>(dA, dB, dC, dI, dDf);
  else
    k_smf_fp8big_3264<AE, BE, Semantics::Native, float><<<1, WAVE>>>(dA, dB, dC, dI, dDf);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> gotf(Mm * Nn);
  HIP_CHECK(hipMemcpy(gotf.data(), dDf, Mm * Nn * sizeof(float), hipMemcpyDeviceToHost));
  for (int t = 0; t < Mm * Nn; ++t)
    EXPECT_EQ(bits_of(gotf[t]), bits_of(ref[t])) << "float at " << t;
  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VF = Value<float, S, kCC>;
    using VA = Value<AE, S, kCC>;
    using VB = Value<BE, S, kCC>;
    std::vector<std::uint32_t> refp(Mm * Nn);
    for (int i = 0; i < Mm; ++i)
      for (int j = 0; j < Nn; ++j) {
        VF accp(m.C[i * Nn + j]);
        for (int q = 0; q < G; ++q) {
          int k0 = 4 * q + m.p0[i * G + q], k1 = 4 * q + m.p1[i * G + q];
          accp = accp + fpsan::cast<float>(VA(AE(m.A[i * Cc + 2 * q]))) *
                            fpsan::cast<float>(VB(BE(m.B[k0 * Nn + j])));
          accp = accp + fpsan::cast<float>(VA(AE(m.A[i * Cc + 2 * q + 1]))) *
                            fpsan::cast<float>(VB(BE(m.B[k1 * Nn + j])));
        }
        refp[i * Nn + j] = accp.fpsan_payload();
      }
    std::uint32_t *dDp;
    HIP_CHECK(hipMalloc(&dDp, Mm * Nn * sizeof(std::uint32_t)));
    if (Mm == 16)
      k_smf_fp8big_16128<AE, BE, S, std::uint32_t><<<1, WAVE>>>(dA, dB, dC, dI, dDp);
    else
      k_smf_fp8big_3264<AE, BE, S, std::uint32_t><<<1, WAVE>>>(dA, dB, dC, dI, dDp);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> gotp(Mm * Nn);
    HIP_CHECK(hipMemcpy(gotp.data(), dDp, Mm * Nn * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for (int t = 0; t < Mm * Nn; ++t)
      EXPECT_EQ(gotp[t], refp[t]) << "fpsan at " << t;
    (void)hipFree(dDp);
  });
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dDf);
  (void)hipFree(dI);
}

TEST(SmfmacFp8_16x16x128, FP8_FP8) { fp8big_smf_test<fpsan::fp8_e4m3, fpsan::fp8_e4m3>(GW, GK); }
TEST(SmfmacFp8_16x16x128, FP8_BF8) { fp8big_smf_test<fpsan::fp8_e4m3, fpsan::fp8_e5m2>(GW, GK); }
TEST(SmfmacFp8_16x16x128, BF8_FP8) { fp8big_smf_test<fpsan::fp8_e5m2, fpsan::fp8_e4m3>(GW, GK); }
TEST(SmfmacFp8_16x16x128, BF8_BF8) { fp8big_smf_test<fpsan::fp8_e5m2, fpsan::fp8_e5m2>(GW, GK); }
TEST(SmfmacFp8_32x32x64, FP8_FP8) { fp8big_smf_test<fpsan::fp8_e4m3, fpsan::fp8_e4m3>(GX, GXK); }
TEST(SmfmacFp8_32x32x64, FP8_BF8) { fp8big_smf_test<fpsan::fp8_e4m3, fpsan::fp8_e5m2>(GX, GXK); }
TEST(SmfmacFp8_32x32x64, BF8_FP8) { fp8big_smf_test<fpsan::fp8_e5m2, fpsan::fp8_e4m3>(GX, GXK); }
TEST(SmfmacFp8_32x32x64, BF8_BF8) { fp8big_smf_test<fpsan::fp8_e5m2, fpsan::fp8_e5m2>(GX, GXK); }
