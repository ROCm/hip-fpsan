// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mfma_gfx950_parts.hpp"

#define MFMA_TESTS(Name)                                                                           \
  TEST(Name, LayoutMatchesHardware) { run_layout_matches_hardware<Name>(); }                       \
  TEST(Name, FpsanMatchesScalarReference) { run_fpsan_matches_scalar_reference_all<Name>(); }

// ---- F16 / BF16 inputs, F32 accumulator -----------------------------------
struct MfmaF32_16x16x32_F16 {
  using AVec = v8h_native;
  using BVec = v8h_native;
  using CVec = v4f_native;
  static constexpr int M = 16, N = 16, K = 32, InBits = 16;
  static constexpr int a_lo = -3, a_hi = 3, b_lo = -2, b_hi = 2, c_lo = -4, c_hi = 4;
  template <Semantics S, Conversions C>
  __device__ static Value<CVec, S, C> call(Value<AVec, S, C> a, Value<BVec, S, C> b,
                                           Value<CVec, S, C> c) {
    return fpsan::amdgcn_mfma_f32_16x16x32_f16<0, 0, 0, S, C>(a, b, c);
  }
};
MFMA_TESTS(MfmaF32_16x16x32_F16)

struct MfmaF32_16x16x32_BF16 {
  using AVec = v8bf_native;
  using BVec = v8bf_native;
  using CVec = v4f_native;
  static constexpr int M = 16, N = 16, K = 32, InBits = 16;
  static constexpr int a_lo = -3, a_hi = 3, b_lo = -2, b_hi = 2, c_lo = -4, c_hi = 4;
  template <Semantics S, Conversions C>
  __device__ static Value<CVec, S, C> call(Value<AVec, S, C> a, Value<BVec, S, C> b,
                                           Value<CVec, S, C> c) {
    return fpsan::amdgcn_mfma_f32_16x16x32_bf16<0, 0, 0, S, C>(a, b, c);
  }
};
MFMA_TESTS(MfmaF32_16x16x32_BF16)

struct MfmaF32_32x32x16_F16 {
  using AVec = v8h_native;
  using BVec = v8h_native;
  using CVec = v16f_native;
  static constexpr int M = 32, N = 32, K = 16, InBits = 16;
  static constexpr int a_lo = -3, a_hi = 3, b_lo = -2, b_hi = 2, c_lo = -4, c_hi = 4;
  template <Semantics S, Conversions C>
  __device__ static Value<CVec, S, C> call(Value<AVec, S, C> a, Value<BVec, S, C> b,
                                           Value<CVec, S, C> c) {
    return fpsan::amdgcn_mfma_f32_32x32x16_f16<0, 0, 0, S, C>(a, b, c);
  }
};
MFMA_TESTS(MfmaF32_32x32x16_F16)

struct MfmaF32_32x32x16_BF16 {
  using AVec = v8bf_native;
  using BVec = v8bf_native;
  using CVec = v16f_native;
  static constexpr int M = 32, N = 32, K = 16, InBits = 16;
  static constexpr int a_lo = -3, a_hi = 3, b_lo = -2, b_hi = 2, c_lo = -4, c_hi = 4;
  template <Semantics S, Conversions C>
  __device__ static Value<CVec, S, C> call(Value<AVec, S, C> a, Value<BVec, S, C> b,
                                           Value<CVec, S, C> c) {
    return fpsan::amdgcn_mfma_f32_32x32x16_bf16<0, 0, 0, S, C>(a, b, c);
  }
};
MFMA_TESTS(MfmaF32_32x32x16_BF16)

// ---- FP8 / BF8 inputs, F32 accumulator ------------------------------------
#define MFMA_FP8_TRAITS(Name, AV, BV, CV, M_, N_, K_, WRAP)                                        \
  struct Name {                                                                                    \
    using AVec = AV;                                                                               \
    using BVec = BV;                                                                               \
    using CVec = CV;                                                                               \
    static constexpr int M = M_, N = N_, K = K_, InBits = 8;                                       \
    static constexpr int a_lo = -3, a_hi = 3, b_lo = -2, b_hi = 2, c_lo = -4, c_hi = 4;            \
    template <Semantics S, Conversions C>                                                          \
    __device__ static Value<CVec, S, C> call(Value<AVec, S, C> a, Value<BVec, S, C> b,             \
                                             Value<CVec, S, C> c) {                                \
      return fpsan::WRAP<0, 0, 0, S, C>(a, b, c);                                                  \
    }                                                                                              \
  };                                                                                               \
  MFMA_TESTS(Name)

MFMA_FP8_TRAITS(MfmaF32_16x16x32_FP8_FP8, v8e4m3_native, v8e4m3_native, v4f_native, 16, 16, 32,
                amdgcn_mfma_f32_16x16x32_fp8_fp8)
MFMA_FP8_TRAITS(MfmaF32_16x16x32_FP8_BF8, v8e4m3_native, v8e5m2_native, v4f_native, 16, 16, 32,
                amdgcn_mfma_f32_16x16x32_fp8_bf8)
MFMA_FP8_TRAITS(MfmaF32_16x16x32_BF8_FP8, v8e5m2_native, v8e4m3_native, v4f_native, 16, 16, 32,
                amdgcn_mfma_f32_16x16x32_bf8_fp8)
MFMA_FP8_TRAITS(MfmaF32_16x16x32_BF8_BF8, v8e5m2_native, v8e5m2_native, v4f_native, 16, 16, 32,
                amdgcn_mfma_f32_16x16x32_bf8_bf8)
MFMA_FP8_TRAITS(MfmaF32_32x32x16_FP8_FP8, v8e4m3_native, v8e4m3_native, v16f_native, 32, 32, 16,
                amdgcn_mfma_f32_32x32x16_fp8_fp8)
MFMA_FP8_TRAITS(MfmaF32_32x32x16_FP8_BF8, v8e4m3_native, v8e5m2_native, v16f_native, 32, 32, 16,
                amdgcn_mfma_f32_32x32x16_fp8_bf8)
MFMA_FP8_TRAITS(MfmaF32_32x32x16_BF8_FP8, v8e5m2_native, v8e4m3_native, v16f_native, 32, 32, 16,
                amdgcn_mfma_f32_32x32x16_bf8_fp8)
MFMA_FP8_TRAITS(MfmaF32_32x32x16_BF8_BF8, v8e5m2_native, v8e5m2_native, v16f_native, 32, 32, 16,
                amdgcn_mfma_f32_32x32x16_bf8_bf8)

// Broadcast/permutation modifiers are covered on the legal multi-block legacy
// MFMA shapes below. The dense gfx950 16x16x32/32x32x16 shapes have B=1, so
// non-zero CBSZ would exceed log2(blocks) and is not a useful conformance case.
// xF32 is intentionally absent here: LLVM's gfx950 source-of-truth marks
// v_mfma_f32_*_xf32 unsupported on gfx950. Those wrappers are covered by the
// CDNA3 MFMA tests where the xf32-insts feature exists.

// ---------------------------------------------------------------------------
// F64 16x16x4 MFMA. A and B are *scalar* doubles per lane (not vector
// fragments), C/D are a v4d (4 doubles per lane). Bespoke kernels because the
// operand shape differs from the packed-vector dense shapes above.
//
//   A[i][k]: input_loc(16,4,1,i,k,0,64) -> lane = k*16 + i  (one elem/lane)
//   B[k][j]: input_loc(16,4,1,j,k,0,64) -> lane = k*16 + j
//   D/C[i][j]: output_loc_64(16,16,i,j,0) -> (reg, lane); f64 accumulators
//              occupy VGPR pairs so the v4d slot index is reg/2 (4 elems/lane)
// ---------------------------------------------------------------------------
using fpsan::v4d_native;

static constexpr int F64_M = 16, F64_N = 16, F64_K = 4;

template <Semantics S>
__device__ void load_f64_frags(const double *A, const double *B, const double *C, int lane,
                               Value<double, S, kCC> &a, Value<double, S, kCC> &b,
                               Value<v4d_native, S, kCC> &c) {
  double an = 0, bn = 0;
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
  a = Value<double, S, kCC>(an);
  b = Value<double, S, kCC>(bn);
  c = Value<v4d_native, S, kCC>(cn);
}

template <Semantics S, class Out, int CBSZ = 0, int ABID = 0, int NEG = 0>
__global__ void k_mfma_f64_16x16x4(const double *A, const double *B, const double *C, Out *D) {
  int lane = threadIdx.x;
  Value<double, S, kCC> a, b;
  Value<v4d_native, S, kCC> c;
  load_f64_frags<S>(A, B, C, lane, a, b, c);
  auto d = fpsan::amdgcn_mfma_f64_16x16x4f64<CBSZ, ABID, NEG, S, kCC>(a, b, c);
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

namespace {
struct F64Mats {
  std::vector<double> A, B, C;
};
F64Mats make_f64_inputs() {
  F64Mats m;
  m.A.resize(F64_M * F64_K);
  m.B.resize(F64_K * F64_N);
  m.C.resize(F64_M * F64_N);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : m.A)
    x = fpsan_test::pick_int_valued<double>(rng, -3, 3);
  for (auto &x : m.B)
    x = fpsan_test::pick_int_valued<double>(rng, -2, 2);
  for (auto &x : m.C)
    x = fpsan_test::pick_int_valued<double>(rng, -4, 4);
  return m;
}

template <class T> T host_neg_if(T v, bool neg) { return neg ? -v : v; }
} // namespace

TEST(MfmaF64_16x16x4, LayoutMatchesHardware) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  F64Mats m = make_f64_inputs();
  std::vector<double> ref(F64_M * F64_N);
  for (int i = 0; i < F64_M; ++i)
    for (int j = 0; j < F64_N; ++j) {
      double acc = m.C[i * F64_N + j];
      for (int k = 0; k < F64_K; ++k)
        acc += m.A[i * F64_K + k] * m.B[k * F64_N + j];
      ref[i * F64_N + j] = acc;
    }
  double *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C), *dD;
  HIP_CHECK(hipMalloc(&dD, F64_M * F64_N * sizeof(double)));
  k_mfma_f64_16x16x4<Semantics::Native, double><<<1, WAVE>>>(dA, dB, dC, dD);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<double> got(F64_M * F64_N);
  HIP_CHECK(hipMemcpy(got.data(), dD, F64_M * F64_N * sizeof(double), hipMemcpyDeviceToHost));
  for (int i = 0; i < F64_M * F64_N; ++i)
    EXPECT_EQ(bits_of(got[i]), bits_of(ref[i]))
        << "layout mismatch at " << (i / F64_N) << "," << (i % F64_N);
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dD);
}

TEST(MfmaF64_16x16x4, FpsanMatchesScalarReference) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  F64Mats m = make_f64_inputs();
  double *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C);
  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VD = Value<double, S, kCC>;
    std::vector<std::uint64_t> ref(F64_M * F64_N);
    for (int i = 0; i < F64_M; ++i)
      for (int j = 0; j < F64_N; ++j) {
        VD acc(m.C[i * F64_N + j]);
        for (int k = 0; k < F64_K; ++k)
          acc = acc + VD(m.A[i * F64_K + k]) * VD(m.B[k * F64_N + j]);
        ref[i * F64_N + j] = acc.fpsan_payload();
      }
    std::uint64_t *dD;
    HIP_CHECK(hipMalloc(&dD, F64_M * F64_N * sizeof(std::uint64_t)));
    k_mfma_f64_16x16x4<S, std::uint64_t><<<1, WAVE>>>(dA, dB, dC, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint64_t> got(F64_M * F64_N);
    HIP_CHECK(
        hipMemcpy(got.data(), dD, F64_M * F64_N * sizeof(std::uint64_t), hipMemcpyDeviceToHost));
    for (int i = 0; i < F64_M * F64_N; ++i)
      EXPECT_EQ(got[i], ref[i]) << "payload mismatch at " << (i / F64_N) << "," << (i % F64_N);
    (void)hipFree(dD);
  });
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
}

TEST(MfmaF64_16x16x4_NEG5, LayoutMatchesHardware) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  constexpr int NEG = 5; // f64 MFMA third immediate is NEG: negate A and C.
  F64Mats m = make_f64_inputs();
  std::vector<double> ref(F64_M * F64_N);
  for (int i = 0; i < F64_M; ++i)
    for (int j = 0; j < F64_N; ++j) {
      double acc = host_neg_if(m.C[i * F64_N + j], (NEG & 4) != 0);
      for (int k = 0; k < F64_K; ++k)
        acc += host_neg_if(m.A[i * F64_K + k], (NEG & 1) != 0) *
               host_neg_if(m.B[k * F64_N + j], (NEG & 2) != 0);
      ref[i * F64_N + j] = acc;
    }
  double *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C), *dD;
  HIP_CHECK(hipMalloc(&dD, F64_M * F64_N * sizeof(double)));
  k_mfma_f64_16x16x4<Semantics::Native, double, 0, 0, NEG><<<1, WAVE>>>(dA, dB, dC, dD);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<double> got(F64_M * F64_N);
  HIP_CHECK(hipMemcpy(got.data(), dD, F64_M * F64_N * sizeof(double), hipMemcpyDeviceToHost));
  for (int i = 0; i < F64_M * F64_N; ++i)
    EXPECT_EQ(bits_of(got[i]), bits_of(ref[i]))
        << "NEG mismatch at " << (i / F64_N) << "," << (i % F64_N);
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dD);
}

TEST(MfmaF64_16x16x4_NEG5, FpsanMatchesScalarReference) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  constexpr int NEG = 5;
  F64Mats m = make_f64_inputs();
  double *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C);
  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VD = Value<double, S, kCC>;
    std::vector<std::uint64_t> ref(F64_M * F64_N);
    for (int i = 0; i < F64_M; ++i)
      for (int j = 0; j < F64_N; ++j) {
        VD acc = host_neg_if(VD(m.C[i * F64_N + j]), (NEG & 4) != 0);
        for (int k = 0; k < F64_K; ++k)
          acc = acc + host_neg_if(VD(m.A[i * F64_K + k]), (NEG & 1) != 0) *
                          host_neg_if(VD(m.B[k * F64_N + j]), (NEG & 2) != 0);
        ref[i * F64_N + j] = acc.fpsan_payload();
      }
    std::uint64_t *dD;
    HIP_CHECK(hipMalloc(&dD, F64_M * F64_N * sizeof(std::uint64_t)));
    k_mfma_f64_16x16x4<S, std::uint64_t, 0, 0, NEG><<<1, WAVE>>>(dA, dB, dC, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint64_t> got(F64_M * F64_N);
    HIP_CHECK(
        hipMemcpy(got.data(), dD, F64_M * F64_N * sizeof(std::uint64_t), hipMemcpyDeviceToHost));
    for (int i = 0; i < F64_M * F64_N; ++i)
      EXPECT_EQ(got[i], ref[i]) << "payload mismatch at " << (i / F64_N) << "," << (i % F64_N);
    (void)hipFree(dD);
  });
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
}

// ---------------------------------------------------------------------------
// F64 4x4x4 MFMA (V_MFMA_F64_4X4X4_4B_F64): 4 independent 4x4 blocks, each a
// K=4 contraction, scalar A/B/C/D per lane. Layout reverse-engineered on real
// MI350 (the dense input_loc/output_loc do NOT apply -- K=4 < 16 lanes/block):
//   output lane L holds D[i][j] of block b: i=L/16, b=(L%16)/4, j=L%4
//   A[i][k] of block b at lane 16k+4b+i; B[k][j] of block b at lane 16k+4b+j
//   D[L] = C[L] + sum_{k=0..3} A[16k+4b+i] * B[16k+4b+j]
// ---------------------------------------------------------------------------
template <Semantics S, class Out, int CBSZ = 0, int ABID = 0, int NEG = 0>
__global__ void k_mfma_f64_4x4x4(const double *A, const double *B, const double *C, Out *D) {
  int lane = threadIdx.x;
  Value<double, S, kCC> a{A[lane]}, b{B[lane]}, c{C[lane]};
  auto d = fpsan::amdgcn_mfma_f64_4x4x4f64<CBSZ, ABID, NEG, S, kCC>(a, b, c);
  if constexpr (S == Semantics::Native)
    D[lane] = d.to_float();
  else
    D[lane] = d.fpsan_payload();
}

namespace {
std::vector<double> make_f64_4x4x4_vec(int seed) {
  std::vector<double> v(WAVE);
  std::mt19937 rng = fpsan_test::make_rng();
  for (int s = 0; s < seed; ++s)
    (void)rng();
  for (auto &x : v)
    x = fpsan_test::pick_int_valued<double>(rng, -4, 4);
  return v;
}
} // namespace

TEST(MfmaF64_4x4x4, LayoutMatchesHardware) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  auto A = make_f64_4x4x4_vec(0), B = make_f64_4x4x4_vec(1), C = make_f64_4x4x4_vec(2);
  std::vector<double> ref(WAVE);
  for (int L = 0; L < WAVE; ++L) {
    int i = L / 16, b = (L % 16) / 4, j = L % 4;
    double acc = C[L];
    for (int k = 0; k < 4; ++k)
      acc += A[16 * k + 4 * b + i] * B[16 * k + 4 * b + j];
    ref[L] = acc;
  }
  double *dA = to_dev(A), *dB = to_dev(B), *dC = to_dev(C), *dD;
  HIP_CHECK(hipMalloc(&dD, WAVE * sizeof(double)));
  k_mfma_f64_4x4x4<Semantics::Native, double><<<1, WAVE>>>(dA, dB, dC, dD);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<double> got(WAVE);
  HIP_CHECK(hipMemcpy(got.data(), dD, WAVE * sizeof(double), hipMemcpyDeviceToHost));
  for (int L = 0; L < WAVE; ++L)
    EXPECT_EQ(bits_of(got[L]), bits_of(ref[L])) << "lane " << L;
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dD);
}

TEST(MfmaF64_4x4x4, FpsanMatchesScalarReference) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  auto A = make_f64_4x4x4_vec(0), B = make_f64_4x4x4_vec(1), C = make_f64_4x4x4_vec(2);
  double *dA = to_dev(A), *dB = to_dev(B), *dC = to_dev(C);
  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VD = Value<double, S, kCC>;
    std::vector<std::uint64_t> ref(WAVE);
    for (int L = 0; L < WAVE; ++L) {
      int i = L / 16, b = (L % 16) / 4, j = L % 4;
      VD acc(C[L]);
      for (int k = 0; k < 4; ++k)
        acc = acc + VD(A[16 * k + 4 * b + i]) * VD(B[16 * k + 4 * b + j]);
      ref[L] = acc.fpsan_payload();
    }
    std::uint64_t *dD;
    HIP_CHECK(hipMalloc(&dD, WAVE * sizeof(std::uint64_t)));
    k_mfma_f64_4x4x4<S, std::uint64_t><<<1, WAVE>>>(dA, dB, dC, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint64_t> got(WAVE);
    HIP_CHECK(hipMemcpy(got.data(), dD, WAVE * sizeof(std::uint64_t), hipMemcpyDeviceToHost));
    for (int L = 0; L < WAVE; ++L)
      EXPECT_EQ(got[L], ref[L]) << "lane " << L;
    (void)hipFree(dD);
  });
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
}

TEST(MfmaF64_4x4x4_NEG5, LayoutMatchesHardware) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  constexpr int NEG = 5; // f64 MFMA third immediate is NEG: negate A and C.
  auto A = make_f64_4x4x4_vec(0), B = make_f64_4x4x4_vec(1), C = make_f64_4x4x4_vec(2);
  std::vector<double> ref(WAVE);
  for (int L = 0; L < WAVE; ++L) {
    int i = L / 16, b = (L % 16) / 4, j = L % 4;
    double acc = host_neg_if(C[L], (NEG & 4) != 0);
    for (int k = 0; k < 4; ++k) {
      acc += host_neg_if(A[16 * k + 4 * b + i], (NEG & 1) != 0) *
             host_neg_if(B[16 * k + 4 * b + j], (NEG & 2) != 0);
    }
    ref[L] = acc;
  }
  double *dA = to_dev(A), *dB = to_dev(B), *dC = to_dev(C), *dD;
  HIP_CHECK(hipMalloc(&dD, WAVE * sizeof(double)));
  k_mfma_f64_4x4x4<Semantics::Native, double, 0, 0, NEG><<<1, WAVE>>>(dA, dB, dC, dD);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<double> got(WAVE);
  HIP_CHECK(hipMemcpy(got.data(), dD, WAVE * sizeof(double), hipMemcpyDeviceToHost));
  for (int L = 0; L < WAVE; ++L)
    EXPECT_EQ(bits_of(got[L]), bits_of(ref[L])) << "lane " << L;
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dD);
}

TEST(MfmaF64_4x4x4_NEG5, FpsanMatchesScalarReference) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  constexpr int NEG = 5;
  auto A = make_f64_4x4x4_vec(0), B = make_f64_4x4x4_vec(1), C = make_f64_4x4x4_vec(2);
  double *dA = to_dev(A), *dB = to_dev(B), *dC = to_dev(C);
  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VD = Value<double, S, kCC>;
    std::vector<std::uint64_t> ref(WAVE);
    for (int L = 0; L < WAVE; ++L) {
      int i = L / 16, b = (L % 16) / 4, j = L % 4;
      VD acc = host_neg_if(VD(C[L]), (NEG & 4) != 0);
      for (int k = 0; k < 4; ++k) {
        acc = acc + host_neg_if(VD(A[16 * k + 4 * b + i]), (NEG & 1) != 0) *
                        host_neg_if(VD(B[16 * k + 4 * b + j]), (NEG & 2) != 0);
      }
      ref[L] = acc.fpsan_payload();
    }
    std::uint64_t *dD;
    HIP_CHECK(hipMalloc(&dD, WAVE * sizeof(std::uint64_t)));
    k_mfma_f64_4x4x4<S, std::uint64_t, 0, 0, NEG><<<1, WAVE>>>(dA, dB, dC, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint64_t> got(WAVE);
    HIP_CHECK(hipMemcpy(got.data(), dD, WAVE * sizeof(std::uint64_t), hipMemcpyDeviceToHost));
    for (int L = 0; L < WAVE; ++L)
      EXPECT_EQ(got[L], ref[L]) << "lane " << L;
    (void)hipFree(dD);
  });
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
}

// ---------------------------------------------------------------------------
// Legacy gfx9 f32-input MFMA shapes (16x16x4, 16x16x1, 32x32x2, 32x32x1,
// 4x4x1). SCALAR f32 A/B per lane, `B` independent MxN blocks. Layouts use
// input_loc / output_loc_32 (silicon-verified, full-block random match). Both
// the Float builtin and the FPSan dataflow are pinned here.
// ---------------------------------------------------------------------------
template <class Traits>
__global__ void k_legf32(const float *A, const float *B, const float *C, float *D) {
  using T = Traits;
  int lane = threadIdx.x;
  float an = 0, bn = 0;
  for (int b = 0; b < T::Bk; ++b)
    for (int i = 0; i < T::M; ++i)
      for (int k = 0; k < T::K; ++k) {
        auto l = fpsan::detail::input_loc(T::M, T::K, T::Bk, i, k, b, 32);
        if (l.lane == lane)
          an = A[(b * T::M + i) * T::K + k];
      }
  for (int b = 0; b < T::Bk; ++b)
    for (int j = 0; j < T::N; ++j)
      for (int k = 0; k < T::K; ++k) {
        auto l = fpsan::detail::input_loc(T::N, T::K, T::Bk, j, k, b, 32);
        if (l.lane == lane)
          bn = B[(b * T::K + k) * T::N + j];
      }
  typename T::CVec cn{};
  for (int b = 0; b < T::Bk; ++b)
    for (int i = 0; i < T::M; ++i)
      for (int j = 0; j < T::N; ++j) {
        auto o = fpsan::detail::output_loc_32(T::M, T::N, i, j, b);
        if (o.lane == lane)
          cn[o.reg] = C[(b * T::M + i) * T::N + j];
      }
  Value<float, Semantics::Native, kCC> a{an}, b2{bn};
  Value<typename T::CVec, Semantics::Native, kCC> c{cn};
  auto d = T::template call<Semantics::Native, kCC>(a, b2, c);
  for (int b = 0; b < T::Bk; ++b)
    for (int i = 0; i < T::M; ++i)
      for (int j = 0; j < T::N; ++j) {
        auto o = fpsan::detail::output_loc_32(T::M, T::N, i, j, b);
        if (o.lane == lane)
          D[(b * T::M + i) * T::N + j] = d.get(o.reg).to_float();
      }
}

template <class Traits, Semantics S>
__global__ void k_legf32_p(const float *A, const float *B, const float *C, std::uint32_t *D) {
  using T = Traits;
  int lane = threadIdx.x;
  float an = 0, bn = 0;
  for (int b = 0; b < T::Bk; ++b)
    for (int i = 0; i < T::M; ++i)
      for (int k = 0; k < T::K; ++k) {
        auto l = fpsan::detail::input_loc(T::M, T::K, T::Bk, i, k, b, 32);
        if (l.lane == lane)
          an = A[(b * T::M + i) * T::K + k];
      }
  for (int b = 0; b < T::Bk; ++b)
    for (int j = 0; j < T::N; ++j)
      for (int k = 0; k < T::K; ++k) {
        auto l = fpsan::detail::input_loc(T::N, T::K, T::Bk, j, k, b, 32);
        if (l.lane == lane)
          bn = B[(b * T::K + k) * T::N + j];
      }
  typename T::CVec cn{};
  for (int b = 0; b < T::Bk; ++b)
    for (int i = 0; i < T::M; ++i)
      for (int j = 0; j < T::N; ++j) {
        auto o = fpsan::detail::output_loc_32(T::M, T::N, i, j, b);
        if (o.lane == lane)
          cn[o.reg] = C[(b * T::M + i) * T::N + j];
      }
  Value<float, S, kCC> a{an}, b2{bn};
  Value<typename T::CVec, S, kCC> c{cn};
  auto d = T::template call<S, kCC>(a, b2, c);
  for (int b = 0; b < T::Bk; ++b)
    for (int i = 0; i < T::M; ++i)
      for (int j = 0; j < T::N; ++j) {
        auto o = fpsan::detail::output_loc_32(T::M, T::N, i, j, b);
        if (o.lane == lane)
          D[(b * T::M + i) * T::N + j] = d.get(o.reg).fpsan_payload();
      }
}

namespace {
template <class Traits> struct LegF32Data {
  std::vector<float> A, B, C;
};
template <class T> LegF32Data<T> make_legf32() {
  LegF32Data<T> d;
  d.A.resize(T::Bk * T::M * T::K);
  d.B.resize(T::Bk * T::K * T::N);
  d.C.resize(T::Bk * T::M * T::N);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : d.A)
    x = fpsan_test::pick_int(rng, -3, 3);
  for (auto &x : d.B)
    x = fpsan_test::pick_int(rng, -2, 2);
  for (auto &x : d.C)
    x = fpsan_test::pick_int(rng, -3, 3);
  return d;
}
} // namespace

template <class T> float host_legf32_a_value(const std::vector<float> &A, int b, int i, int k) {
  auto want = fpsan::detail::input_loc(T::M, T::K, T::Bk, i, k, b, 32);
  const int src_lane =
      host_mfma_a_src_lane(want.lane, MfmaTraitCBSZ<T>::value, MfmaTraitABID<T>::value);
  for (int bb = 0; bb < T::Bk; ++bb)
    for (int ii = 0; ii < T::M; ++ii)
      for (int kk = 0; kk < T::K; ++kk) {
        auto loc = fpsan::detail::input_loc(T::M, T::K, T::Bk, ii, kk, bb, 32);
        if (loc.lane == src_lane && loc.reg == want.reg && loc.sub == want.sub)
          return A[(bb * T::M + ii) * T::K + kk];
      }
  return 0.0f;
}

template <class T> float host_legf32_b_value(const std::vector<float> &B, int b, int k, int j) {
  auto want = fpsan::detail::input_loc(T::N, T::K, T::Bk, j, k, b, 32);
  const int src_lane = host_mfma_b_src_lane(want.lane, MfmaTraitBLGP<T>::value);
  for (int bb = 0; bb < T::Bk; ++bb)
    for (int jj = 0; jj < T::N; ++jj)
      for (int kk = 0; kk < T::K; ++kk) {
        auto loc = fpsan::detail::input_loc(T::N, T::K, T::Bk, jj, kk, bb, 32);
        if (loc.lane == src_lane && loc.reg == want.reg && loc.sub == want.sub)
          return B[(bb * T::K + kk) * T::N + jj];
      }
  return 0.0f;
}

template <class T> void run_legf32_layout() {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  LegF32Data<T> m = make_legf32<T>();
  std::vector<float> ref(T::Bk * T::M * T::N);
  for (int b = 0; b < T::Bk; ++b)
    for (int i = 0; i < T::M; ++i)
      for (int j = 0; j < T::N; ++j) {
        double acc = m.C[(b * T::M + i) * T::N + j];
        for (int k = 0; k < T::K; ++k)
          acc += static_cast<double>(host_legf32_a_value<T>(m.A, b, i, k)) *
                 static_cast<double>(host_legf32_b_value<T>(m.B, b, k, j));
        ref[(b * T::M + i) * T::N + j] = static_cast<float>(acc);
      }
  float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C), *dD;
  const int total = T::Bk * T::M * T::N;
  HIP_CHECK(hipMalloc(&dD, total * sizeof(float)));
  k_legf32<T><<<1, WAVE>>>(dA, dB, dC, dD);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> got(total);
  HIP_CHECK(hipMemcpy(got.data(), dD, total * sizeof(float), hipMemcpyDeviceToHost));
  for (int t = 0; t < total; ++t)
    EXPECT_EQ(bits_of(got[t]), bits_of(ref[t])) << "elem " << t;
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dD);
}

template <class T> void run_legf32_fpsan() {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  LegF32Data<T> m = make_legf32<T>();
  float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C);
  const int total = T::Bk * T::M * T::N;
  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VF = Value<float, S, kCC>;
    std::vector<std::uint32_t> ref(total);
    for (int b = 0; b < T::Bk; ++b)
      for (int i = 0; i < T::M; ++i)
        for (int j = 0; j < T::N; ++j) {
          VF acc(m.C[(b * T::M + i) * T::N + j]);
          for (int k = 0; k < T::K; ++k)
            acc = acc + VF(host_legf32_a_value<T>(m.A, b, i, k)) *
                            VF(host_legf32_b_value<T>(m.B, b, k, j));
          ref[(b * T::M + i) * T::N + j] = acc.fpsan_payload();
        }
    std::uint32_t *dD;
    HIP_CHECK(hipMalloc(&dD, total * sizeof(std::uint32_t)));
    k_legf32_p<T, S><<<1, WAVE>>>(dA, dB, dC, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(total);
    HIP_CHECK(hipMemcpy(got.data(), dD, total * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for (int t = 0; t < total; ++t)
      EXPECT_EQ(got[t], ref[t]) << "elem " << t;
    (void)hipFree(dD);
  });
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
}

#define LEGF32_TRAITS(Name, M_, N_, K_, B_, CV, WRAP)                                              \
  struct Name {                                                                                    \
    using CVec = CV;                                                                               \
    static constexpr int M = M_, N = N_, K = K_, Bk = B_;                                          \
    static constexpr int CBSZ = 0, ABID = 0, BLGP = 0;                                             \
    template <Semantics S, Conversions C>                                                          \
    __device__ static Value<CVec, S, C> call(Value<float, S, C> a, Value<float, S, C> b,           \
                                             Value<CVec, S, C> c) {                                \
      return fpsan::WRAP<CBSZ, ABID, BLGP, S, C>(a, b, c);                                         \
    }                                                                                              \
  };                                                                                               \
  TEST(Name, LayoutMatchesHardware) { run_legf32_layout<Name>(); }                                 \
  TEST(Name, FpsanMatchesScalarReference) { run_legf32_fpsan<Name>(); }

LEGF32_TRAITS(LegacyMfmaF32_16x16x4, 16, 16, 4, 1, v4f_native, amdgcn_mfma_f32_16x16x4f32)
LEGF32_TRAITS(LegacyMfmaF32_16x16x1, 16, 16, 1, 4, v16f_native, amdgcn_mfma_f32_16x16x1f32)
LEGF32_TRAITS(LegacyMfmaF32_32x32x2, 32, 32, 2, 1, v16f_native, amdgcn_mfma_f32_32x32x2f32)
LEGF32_TRAITS(LegacyMfmaF32_32x32x1, 32, 32, 1, 2, fpsan::v32f_native, amdgcn_mfma_f32_32x32x1f32)
LEGF32_TRAITS(LegacyMfmaF32_4x4x1, 4, 4, 1, 16, v4f_native, amdgcn_mfma_f32_4x4x1f32)

struct LegacyMfmaF32_16x16x1_CBSZ2_ABID3_BLGP5 {
  using CVec = v16f_native;
  static constexpr int M = 16, N = 16, K = 1, Bk = 4;
  static constexpr int CBSZ = 2, ABID = 3, BLGP = 5;
  template <Semantics S, Conversions C>
  __device__ static Value<CVec, S, C> call(Value<float, S, C> a, Value<float, S, C> b,
                                           Value<CVec, S, C> c) {
    return fpsan::amdgcn_mfma_f32_16x16x1f32<CBSZ, ABID, BLGP, S, C>(a, b, c);
  }
};
TEST(LegacyMfmaF32_16x16x1_CBSZ2_ABID3_BLGP5, LayoutMatchesHardware) {
  run_legf32_layout<LegacyMfmaF32_16x16x1_CBSZ2_ABID3_BLGP5>();
}
TEST(LegacyMfmaF32_16x16x1_CBSZ2_ABID3_BLGP5, FpsanMatchesScalarReference) {
  run_legf32_fpsan<LegacyMfmaF32_16x16x1_CBSZ2_ABID3_BLGP5>();
}

struct LegacyMfmaF32_32x32x1_CBSZ1_ABID1_BLGP1 {
  using CVec = fpsan::v32f_native;
  static constexpr int M = 32, N = 32, K = 1, Bk = 2;
  static constexpr int CBSZ = 1, ABID = 1, BLGP = 1;
  template <Semantics S, Conversions C>
  __device__ static Value<CVec, S, C> call(Value<float, S, C> a, Value<float, S, C> b,
                                           Value<CVec, S, C> c) {
    return fpsan::amdgcn_mfma_f32_32x32x1f32<CBSZ, ABID, BLGP, S, C>(a, b, c);
  }
};
TEST(LegacyMfmaF32_32x32x1_CBSZ1_ABID1_BLGP1, LayoutMatchesHardware) {
  run_legf32_layout<LegacyMfmaF32_32x32x1_CBSZ1_ABID1_BLGP1>();
}
TEST(LegacyMfmaF32_32x32x1_CBSZ1_ABID1_BLGP1, FpsanMatchesScalarReference) {
  run_legf32_fpsan<LegacyMfmaF32_32x32x1_CBSZ1_ABID1_BLGP1>();
}

// ---------------------------------------------------------------------------
// Legacy gfx9 f16 / bf16_1k MFMA shapes. A,B per-lane fragment is 4 elements
// of f16 (or bf16). Layouts use input_loc (bits=16, element = 2*reg+sub) /
// output_loc_32 with `B` independent MxN blocks (silicon-verified, full-block
// random match). Both the Float builtin and the FPSan dataflow are pinned here.
// The non-1k bf16 shapes are gfx908-only (cannot select on gfx950) and are
// intentionally omitted.
// ---------------------------------------------------------------------------
template <class Traits>
__global__ void k_legf16(const float *A, const float *B, const float *C, float *D) {
  using T = Traits;
  using AE = typename T::AElem;
  int lane = threadIdx.x;
  typename T::AVec an{}, bn{};
  for (int b = 0; b < T::Bk; ++b)
    for (int i = 0; i < T::M; ++i)
      for (int k = 0; k < T::K; ++k) {
        auto l = fpsan::detail::input_loc(T::M, T::K, T::Bk, i, k, b, 16);
        if (l.lane == lane)
          an[2 * l.reg + l.sub] = AE(A[(b * T::M + i) * T::K + k]);
      }
  for (int b = 0; b < T::Bk; ++b)
    for (int j = 0; j < T::N; ++j)
      for (int k = 0; k < T::K; ++k) {
        auto l = fpsan::detail::input_loc(T::N, T::K, T::Bk, j, k, b, 16);
        if (l.lane == lane)
          bn[2 * l.reg + l.sub] = AE(B[(b * T::K + k) * T::N + j]);
      }
  typename T::CVec cn{};
  for (int b = 0; b < T::Bk; ++b)
    for (int i = 0; i < T::M; ++i)
      for (int j = 0; j < T::N; ++j) {
        auto o = fpsan::detail::output_loc_32(T::M, T::N, i, j, b);
        if (o.lane == lane)
          cn[o.reg] = C[(b * T::M + i) * T::N + j];
      }
  Value<typename T::AVec, Semantics::Native, kCC> a{an}, b2{bn};
  Value<typename T::CVec, Semantics::Native, kCC> c{cn};
  auto d = T::template call<Semantics::Native, kCC>(a, b2, c);
  for (int b = 0; b < T::Bk; ++b)
    for (int i = 0; i < T::M; ++i)
      for (int j = 0; j < T::N; ++j) {
        auto o = fpsan::detail::output_loc_32(T::M, T::N, i, j, b);
        if (o.lane == lane)
          D[(b * T::M + i) * T::N + j] = d.get(o.reg).to_float();
      }
}

template <class Traits, Semantics S>
__global__ void k_legf16_p(const float *A, const float *B, const float *C, std::uint32_t *D) {
  using T = Traits;
  using AE = typename T::AElem;
  int lane = threadIdx.x;
  typename T::AVec an{}, bn{};
  for (int b = 0; b < T::Bk; ++b)
    for (int i = 0; i < T::M; ++i)
      for (int k = 0; k < T::K; ++k) {
        auto l = fpsan::detail::input_loc(T::M, T::K, T::Bk, i, k, b, 16);
        if (l.lane == lane)
          an[2 * l.reg + l.sub] = AE(A[(b * T::M + i) * T::K + k]);
      }
  for (int b = 0; b < T::Bk; ++b)
    for (int j = 0; j < T::N; ++j)
      for (int k = 0; k < T::K; ++k) {
        auto l = fpsan::detail::input_loc(T::N, T::K, T::Bk, j, k, b, 16);
        if (l.lane == lane)
          bn[2 * l.reg + l.sub] = AE(B[(b * T::K + k) * T::N + j]);
      }
  typename T::CVec cn{};
  for (int b = 0; b < T::Bk; ++b)
    for (int i = 0; i < T::M; ++i)
      for (int j = 0; j < T::N; ++j) {
        auto o = fpsan::detail::output_loc_32(T::M, T::N, i, j, b);
        if (o.lane == lane)
          cn[o.reg] = C[(b * T::M + i) * T::N + j];
      }
  Value<typename T::AVec, S, kCC> a{an}, b2{bn};
  Value<typename T::CVec, S, kCC> c{cn};
  auto d = T::template call<S, kCC>(a, b2, c);
  for (int b = 0; b < T::Bk; ++b)
    for (int i = 0; i < T::M; ++i)
      for (int j = 0; j < T::N; ++j) {
        auto o = fpsan::detail::output_loc_32(T::M, T::N, i, j, b);
        if (o.lane == lane)
          D[(b * T::M + i) * T::N + j] = d.get(o.reg).fpsan_payload();
      }
}

template <class T> float host_legf16_a_value(const std::vector<float> &A, int b, int i, int k) {
  auto want = fpsan::detail::input_loc(T::M, T::K, T::Bk, i, k, b, 16);
  const int src_lane =
      host_mfma_a_src_lane(want.lane, MfmaTraitCBSZ<T>::value, MfmaTraitABID<T>::value);
  for (int bb = 0; bb < T::Bk; ++bb)
    for (int ii = 0; ii < T::M; ++ii)
      for (int kk = 0; kk < T::K; ++kk) {
        auto loc = fpsan::detail::input_loc(T::M, T::K, T::Bk, ii, kk, bb, 16);
        if (loc.lane == src_lane && loc.reg == want.reg && loc.sub == want.sub)
          return A[(bb * T::M + ii) * T::K + kk];
      }
  return 0.0f;
}

template <class T> float host_legf16_b_value(const std::vector<float> &B, int b, int k, int j) {
  auto want = fpsan::detail::input_loc(T::N, T::K, T::Bk, j, k, b, 16);
  const int src_lane = host_mfma_b_src_lane(want.lane, MfmaTraitBLGP<T>::value);
  for (int bb = 0; bb < T::Bk; ++bb)
    for (int jj = 0; jj < T::N; ++jj)
      for (int kk = 0; kk < T::K; ++kk) {
        auto loc = fpsan::detail::input_loc(T::N, T::K, T::Bk, jj, kk, bb, 16);
        if (loc.lane == src_lane && loc.reg == want.reg && loc.sub == want.sub)
          return B[(bb * T::K + kk) * T::N + jj];
      }
  return 0.0f;
}

template <class T> void run_legf16_layout() {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  LegF32Data<T> m = make_legf32<T>();
  std::vector<float> ref(T::Bk * T::M * T::N);
  for (int b = 0; b < T::Bk; ++b)
    for (int i = 0; i < T::M; ++i)
      for (int j = 0; j < T::N; ++j) {
        double acc = m.C[(b * T::M + i) * T::N + j];
        for (int k = 0; k < T::K; ++k)
          acc += static_cast<double>(host_legf16_a_value<T>(m.A, b, i, k)) *
                 static_cast<double>(host_legf16_b_value<T>(m.B, b, k, j));
        ref[(b * T::M + i) * T::N + j] = static_cast<float>(acc);
      }
  float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C), *dD;
  const int total = T::Bk * T::M * T::N;
  HIP_CHECK(hipMalloc(&dD, total * sizeof(float)));
  k_legf16<T><<<1, WAVE>>>(dA, dB, dC, dD);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> got(total);
  HIP_CHECK(hipMemcpy(got.data(), dD, total * sizeof(float), hipMemcpyDeviceToHost));
  for (int t = 0; t < total; ++t)
    EXPECT_EQ(bits_of(got[t]), bits_of(ref[t])) << "elem " << t;
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dD);
}

template <class T> void run_legf16_fpsan() {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  LegF32Data<T> m = make_legf32<T>();
  using AE = typename T::AElem;
  float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C);
  const int total = T::Bk * T::M * T::N;
  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VF = Value<float, S, kCC>;
    std::vector<std::uint32_t> ref(total);
    for (int b = 0; b < T::Bk; ++b)
      for (int i = 0; i < T::M; ++i)
        for (int j = 0; j < T::N; ++j) {
          VF acc(m.C[(b * T::M + i) * T::N + j]);
          for (int k = 0; k < T::K; ++k)
            acc =
                acc +
                fpsan::cast<float>(Value<AE, S, kCC>(AE(host_legf16_a_value<T>(m.A, b, i, k)))) *
                    fpsan::cast<float>(Value<AE, S, kCC>(AE(host_legf16_b_value<T>(m.B, b, k, j))));
          ref[(b * T::M + i) * T::N + j] = acc.fpsan_payload();
        }
    std::uint32_t *dD;
    HIP_CHECK(hipMalloc(&dD, total * sizeof(std::uint32_t)));
    k_legf16_p<T, S><<<1, WAVE>>>(dA, dB, dC, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(total);
    HIP_CHECK(hipMemcpy(got.data(), dD, total * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for (int t = 0; t < total; ++t)
      EXPECT_EQ(got[t], ref[t]) << "elem " << t;
    (void)hipFree(dD);
  });
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
}

#define LEGF16_TRAITS(Name, M_, N_, K_, B_, AV, AE, CV, WRAP)                                      \
  struct Name {                                                                                    \
    using AVec = AV;                                                                               \
    using AElem = AE;                                                                              \
    using CVec = CV;                                                                               \
    static constexpr int M = M_, N = N_, K = K_, Bk = B_;                                          \
    static constexpr int CBSZ = 0, ABID = 0, BLGP = 0;                                             \
    template <Semantics S, Conversions C>                                                          \
    __device__ static Value<CVec, S, C> call(Value<AVec, S, C> a, Value<AVec, S, C> b,             \
                                             Value<CVec, S, C> c) {                                \
      return fpsan::WRAP<CBSZ, ABID, BLGP, S, C>(a, b, c);                                         \
    }                                                                                              \
  };                                                                                               \
  TEST(Name, LayoutMatchesHardware) { run_legf16_layout<Name>(); }                                 \
  TEST(Name, FpsanMatchesScalarReference) { run_legf16_fpsan<Name>(); }

LEGF16_TRAITS(LegacyMfmaF16_16x16x16, 16, 16, 16, 1, fpsan::v4h_native, _Float16, v4f_native,
              amdgcn_mfma_f32_16x16x16f16)
LEGF16_TRAITS(LegacyMfmaF16_16x16x4, 16, 16, 4, 4, fpsan::v4h_native, _Float16, v16f_native,
              amdgcn_mfma_f32_16x16x4f16)
LEGF16_TRAITS(LegacyMfmaF16_32x32x8, 32, 32, 8, 1, fpsan::v4h_native, _Float16, v16f_native,
              amdgcn_mfma_f32_32x32x8f16)
LEGF16_TRAITS(LegacyMfmaF16_32x32x4, 32, 32, 4, 2, fpsan::v4h_native, _Float16, fpsan::v32f_native,
              amdgcn_mfma_f32_32x32x4f16)
LEGF16_TRAITS(LegacyMfmaF16_4x4x4, 4, 4, 4, 16, fpsan::v4h_native, _Float16, v4f_native,
              amdgcn_mfma_f32_4x4x4f16)
LEGF16_TRAITS(LegacyMfmaBF16_16x16x16_1k, 16, 16, 16, 1, fpsan::v4bf_native, __bf16, v4f_native,
              amdgcn_mfma_f32_16x16x16bf16_1k)
LEGF16_TRAITS(LegacyMfmaBF16_16x16x4_1k, 16, 16, 4, 4, fpsan::v4bf_native, __bf16, v16f_native,
              amdgcn_mfma_f32_16x16x4bf16_1k)
LEGF16_TRAITS(LegacyMfmaBF16_32x32x8_1k, 32, 32, 8, 1, fpsan::v4bf_native, __bf16, v16f_native,
              amdgcn_mfma_f32_32x32x8bf16_1k)
LEGF16_TRAITS(LegacyMfmaBF16_32x32x4_1k, 32, 32, 4, 2, fpsan::v4bf_native, __bf16,
              fpsan::v32f_native, amdgcn_mfma_f32_32x32x4bf16_1k)
LEGF16_TRAITS(LegacyMfmaBF16_4x4x4_1k, 4, 4, 4, 16, fpsan::v4bf_native, __bf16, v4f_native,
              amdgcn_mfma_f32_4x4x4bf16_1k)

struct LegacyMfmaF16_16x16x4_CBSZ2_ABID3_BLGP5 {
  using AVec = fpsan::v4h_native;
  using AElem = _Float16;
  using CVec = v16f_native;
  static constexpr int M = 16, N = 16, K = 4, Bk = 4;
  static constexpr int CBSZ = 2, ABID = 3, BLGP = 5;
  template <Semantics S, Conversions C>
  __device__ static Value<CVec, S, C> call(Value<AVec, S, C> a, Value<AVec, S, C> b,
                                           Value<CVec, S, C> c) {
    return fpsan::amdgcn_mfma_f32_16x16x4f16<CBSZ, ABID, BLGP, S, C>(a, b, c);
  }
};
TEST(LegacyMfmaF16_16x16x4_CBSZ2_ABID3_BLGP5, LayoutMatchesHardware) {
  run_legf16_layout<LegacyMfmaF16_16x16x4_CBSZ2_ABID3_BLGP5>();
}
TEST(LegacyMfmaF16_16x16x4_CBSZ2_ABID3_BLGP5, FpsanMatchesScalarReference) {
  run_legf16_fpsan<LegacyMfmaF16_16x16x4_CBSZ2_ABID3_BLGP5>();
}

struct LegacyMfmaBF16_32x32x4_1k_CBSZ1_ABID1_BLGP1 {
  using AVec = fpsan::v4bf_native;
  using AElem = __bf16;
  using CVec = fpsan::v32f_native;
  static constexpr int M = 32, N = 32, K = 4, Bk = 2;
  static constexpr int CBSZ = 1, ABID = 1, BLGP = 1;
  template <Semantics S, Conversions C>
  __device__ static Value<CVec, S, C> call(Value<AVec, S, C> a, Value<AVec, S, C> b,
                                           Value<CVec, S, C> c) {
    return fpsan::amdgcn_mfma_f32_32x32x4bf16_1k<CBSZ, ABID, BLGP, S, C>(a, b, c);
  }
};
TEST(LegacyMfmaBF16_32x32x4_1k_CBSZ1_ABID1_BLGP1, LayoutMatchesHardware) {
  run_legf16_layout<LegacyMfmaBF16_32x32x4_1k_CBSZ1_ABID1_BLGP1>();
}
TEST(LegacyMfmaBF16_32x32x4_1k_CBSZ1_ABID1_BLGP1, FpsanMatchesScalarReference) {
  run_legf16_fpsan<LegacyMfmaBF16_32x32x4_1k_CBSZ1_ABID1_BLGP1>();
}
