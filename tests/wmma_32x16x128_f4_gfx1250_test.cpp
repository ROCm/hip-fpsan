// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/wmma_32x16x128_f4_gfx1250_test.cpp
//
// gfx1250 wmma_f32_32x16x128_f4: a 32x16x128 WMMA with fp4 (E2M1) A/B operands
// and an f32 accumulator. A is 32x128 fp4 (512 bits/lane, v16i32), B is 128x16
// fp4 (256 bits/lane, v8i32), C/D is 32x16 f32 (16 regs/lane, v16f32).
//
// GROUNDING test: pins the (still unverified) 32x16 fragment layouts by staging
// A/B/C under a hypothesis and asserting bit-exact equality vs an independent
// host matmul on exact fp4-grid inputs, run through the real builtin (Float
// oracle). The hypothesis continues the validated 16x16x128 family to M=32:
//   D[m][n]: lane = n + 16*(m>>4),  reg = m & 15           (16 regs/lane)
//   A[m][k]: lane = m,  slot = k                            (all 128 k in-lane)
//   B[k][n]: lane = n + 16*((k>>2)&1),  slot = ab_index(k)  (== 16x16x128 B)
// fp4 nibble for slot s sits at bit 4*s of the per-lane register. Equality
// across distinct random inputs can only hold if these layouts equal the
// hardware fragment ABI.
//
// Requires gfx1250 hardware; built only under the gfx1250 test tier.
#include "fpsan/amdgcn_matrix.hpp"
#include "fpsan/detail/fp8.hpp"
#include "fpsan/fpsan.hpp"

#include "fpsan_semantics.hpp"
#include "hip_test_utils.hpp"
#include "subbyte_oracle.hpp"
#include "test_random.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;
using fpsan::detail::f32_to_narrow;
using fpsan::detail::kFp4E2M1;

static constexpr Conversions kCC = Conversions::Explicit;

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_wmma_f32_32x16x128_f4)

static constexpr int M = 32, N = 16, K = 128;

using v16i32_native = int __attribute__((ext_vector_type(16)));
using v8i32_native = int __attribute__((ext_vector_type(8)));
using v16f_native = float __attribute__((ext_vector_type(16)));

// B operand reuses the validated 16x16x128 ab_index slot ordering.
static int b_slot(int k) {
  const int reg = ((k >> 1) & 1) + 2 * ((k >> 3) & 1) + 4 * ((k >> 4) & 1) + 8 * ((k >> 5) & 1) +
                  16 * ((k >> 6) & 1);
  return 2 * reg + (k & 1);
}
static int b_lane(int n, int k) { return n + 16 * ((k >> 2) & 1); }

static void pack_nibble(std::uint32_t *reg, int slot, std::uint32_t code) {
  const int p = 4 * slot;
  reg[p >> 5] |= (code & 0xFu) << (p & 31);
}

// Silicon-grounded 32x16x128_f4 A fragment layout (probed via one-hot + binary-k
// sweeps). For logical A[m][k]:
//   g = m>>3; hslot = g&1; L = m - 8*((g+1)>>1)   (L in 0..15)
//   half = (k>>2)&1;  lane = L + 16*half
//   slot = 64*hslot + (k&3) + 4*(k>>3)
static int a_lane(int m, int k) {
  const int g = m >> 3;
  const int L = m - 8 * ((g + 1) >> 1);
  const int half = (k >> 2) & 1;
  return L + 16 * half;
}
static int a_slot(int m, int k) {
  const int hslot = (m >> 3) & 1;
  return 64 * hslot + (k & 3) + 4 * (k >> 3);
}

template <int A0, int B0>
__global__ void k_f4(const std::uint32_t *Apack, const std::uint32_t *Bpack, const float *C,
                     float *D) {
  const int lane = threadIdx.x;
  v16i32_native a;
  v8i32_native b;
  for (int i = 0; i < 16; ++i)
    a[i] = static_cast<int>(Apack[lane * 16 + i]);
  for (int i = 0; i < 8; ++i)
    b[i] = static_cast<int>(Bpack[lane * 8 + i]);
  v16f_native c;
  for (int r = 0; r < 16; ++r)
    c[r] = C[(r + 16 * (lane >> 4)) * N + (lane & 15)];
  v16f_native d = __builtin_amdgcn_wmma_f32_32x16x128_f4(a, b, static_cast<short>(0), c);
  for (int r = 0; r < 16; ++r)
    D[(r + 16 * (lane >> 4)) * N + (lane & 15)] = d[r];
  (void)A0;
  (void)B0;
}

static const float kGrid[] = {0.f,   0.5f, 1.f,   1.5f, 2.f,  3.f,  4.f, 6.f,
                              -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

// Silicon-grounded layout (reverse-engineered via the one-hot + binary-k probes
// below): A uses a_lane/a_slot, B reuses the 16x16x128 b_lane/b_slot, and the
// D/C accumulator is m = reg + 16*(lane>>4), n = lane&15. Asserts bit-exact vs an
// independent host matmul on the exact fp4 grid.
TEST(WmmaF4_32x16x128, LayoutMatchesHardware) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  std::mt19937 rng = fpsan_test::make_rng();
  std::uniform_int_distribution<int> pick(0, static_cast<int>(sizeof(kGrid) / sizeof(float)) - 1);
  std::vector<float> A(M * K), B(K * N), C(M * N);
  for (auto &x : A)
    x = kGrid[pick(rng)];
  for (auto &x : B)
    x = kGrid[pick(rng)];
  for (auto &x : C)
    x = kGrid[pick(rng)];

  std::vector<std::uint32_t> Apack(32 * 16, 0u), Bpack(32 * 8, 0u);
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k)
      pack_nibble(&Apack[a_lane(m, k) * 16], a_slot(m, k), f32_to_narrow(A[m * K + k], kFp4E2M1));
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      pack_nibble(&Bpack[b_lane(n, k) * 8], b_slot(k), f32_to_narrow(B[k * N + n], kFp4E2M1));

  std::vector<float> ref(M * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = C[m * N + n];
      for (int k = 0; k < K; ++k)
        acc += A[m * K + k] * B[k * N + n];
      ref[m * N + n] = acc;
    }

  std::uint32_t *dA = to_dev(Apack);
  std::uint32_t *dB = to_dev(Bpack);
  float *dC = to_dev(C);
  float *dD = nullptr;
  HIP_CHECK(hipMalloc(&dD, M * N * sizeof(float)));
  k_f4<0, 0><<<1, 32>>>(dA, dB, dC, dD);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> got(M * N);
  HIP_CHECK(hipMemcpy(got.data(), dD, M * N * sizeof(float), hipMemcpyDeviceToHost));
  int mism = 0;
  for (int i = 0; i < M * N; ++i)
    if (bits_of(got[i]) != bits_of(ref[i])) {
      if (mism < 8)
        ADD_FAILURE() << "mismatch at m=" << (i / N) << " n=" << (i % N) << " got=" << got[i]
                      << " ref=" << ref[i];
      ++mism;
    }
  EXPECT_EQ(mism, 0) << mism << " / " << (M * N) << " elements mismatched";
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dD);
}

// ===================== wrapper: Float + FPSan =====================
// FP4 is a packed finite subbyte format, not a scalar Value<> element type. The
// FPSan checks here pin canonical finite widening through the standard cast
// policy.
template <Semantics S> using VF = Value<float, S, kCC>;

template <Semantics S> static VF<S> f4_canonical_value(std::uint32_t code) {
  return fpsan_test::canonical_subbyte_widen<4, S, kCC>(code);
}

__global__ void k_wrap_float(const std::uint32_t *Apack, const std::uint32_t *Bpack, const float *C,
                             float *D) {
  const int lane = threadIdx.x;
  v16i32_native a;
  v8i32_native b;
  for (int i = 0; i < 16; ++i)
    a[i] = static_cast<int>(Apack[lane * 16 + i]);
  for (int i = 0; i < 8; ++i)
    b[i] = static_cast<int>(Bpack[lane * 8 + i]);
  v16f_native cn;
  for (int r = 0; r < 16; ++r)
    cn[r] = C[(r + 16 * (lane >> 4)) * N + (lane & 15)];
  Value<v16f_native, Semantics::Native, kCC> c(cn);
  auto d = fpsan::amdgcn_wmma_f32_32x16x128_f4<Semantics::Native, kCC>(a, b, c);
  for (int r = 0; r < 16; ++r)
    D[(r + 16 * (lane >> 4)) * N + (lane & 15)] = d.get(r).to_float();
}

template <Semantics S>
__global__ void k_wrap_fpsan(const std::uint32_t *Apack, const std::uint32_t *Bpack, const float *C,
                             std::uint32_t *Dpay) {
  const int lane = threadIdx.x;
  v16i32_native a;
  v8i32_native b;
  for (int i = 0; i < 16; ++i)
    a[i] = static_cast<int>(Apack[lane * 16 + i]);
  for (int i = 0; i < 8; ++i)
    b[i] = static_cast<int>(Bpack[lane * 8 + i]);
  v16f_native cn;
  for (int r = 0; r < 16; ++r)
    cn[r] = C[(r + 16 * (lane >> 4)) * N + (lane & 15)];
  Value<v16f_native, S, kCC> c(cn);
  auto d = fpsan::amdgcn_wmma_f32_32x16x128_f4<S, kCC>(a, b, c);
  for (int r = 0; r < 16; ++r)
    Dpay[(r + 16 * (lane >> 4)) * N + (lane & 15)] = d.get(r).fpsan_payload();
}

TEST(WmmaF4_32x16x128, WrapperFloatAndFPSan) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  std::mt19937 rng = fpsan_test::make_rng();
  std::uniform_int_distribution<int> pick(0, static_cast<int>(sizeof(kGrid) / sizeof(float)) - 1);
  std::vector<float> A(M * K), B(K * N), C(M * N);
  for (auto &x : A)
    x = kGrid[pick(rng)];
  for (auto &x : B)
    x = kGrid[pick(rng)];
  for (auto &x : C)
    x = kGrid[pick(rng)];

  std::vector<std::uint32_t> Apack(32 * 16, 0u), Bpack(32 * 8, 0u);
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k)
      pack_nibble(&Apack[a_lane(m, k) * 16], a_slot(m, k), f32_to_narrow(A[m * K + k], kFp4E2M1));
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      pack_nibble(&Bpack[b_lane(n, k) * 8], b_slot(k), f32_to_narrow(B[k * N + n], kFp4E2M1));

  std::vector<float> fref(M * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = C[m * N + n];
      for (int k = 0; k < K; ++k)
        acc += A[m * K + k] * B[k * N + n];
      fref[m * N + n] = acc;
    }
  std::uint32_t *dA = to_dev(Apack);
  std::uint32_t *dB = to_dev(Bpack);
  float *dC = to_dev(C);
  float *dDf = nullptr;
  HIP_CHECK(hipMalloc(&dDf, M * N * sizeof(float)));
  k_wrap_float<<<1, 32>>>(dA, dB, dC, dDf);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> gotf(M * N);
  HIP_CHECK(hipMemcpy(gotf.data(), dDf, M * N * sizeof(float), hipMemcpyDeviceToHost));
  int nf = 0;
  for (int i = 0; i < M * N; ++i)
    if (bits_of(gotf[i]) != bits_of(fref[i]) && nf++ < 4)
      ADD_FAILURE() << "float at m=" << i / N << " n=" << i % N << " got=" << gotf[i]
                    << " ref=" << fref[i];
  EXPECT_EQ(nf, 0) << nf << " float mismatches";

  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    std::vector<std::uint32_t> pref(M * N);
    for (int m = 0; m < M; ++m)
      for (int n = 0; n < N; ++n) {
        VF<S> acc(C[m * N + n]);
        for (int k = 0; k < K; ++k) {
          const std::uint32_t ca = f32_to_narrow(A[m * K + k], kFp4E2M1) & 0xF;
          const std::uint32_t cb = f32_to_narrow(B[k * N + n], kFp4E2M1) & 0xF;
          acc = acc + f4_canonical_value<S>(ca) * f4_canonical_value<S>(cb);
        }
        pref[m * N + n] = acc.fpsan_payload();
      }

    std::uint32_t *dDp = nullptr;
    HIP_CHECK(hipMalloc(&dDp, M * N * sizeof(std::uint32_t)));
    k_wrap_fpsan<S><<<1, 32>>>(dA, dB, dC, dDp);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> gotp(M * N);
    HIP_CHECK(hipMemcpy(gotp.data(), dDp, M * N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    int np = 0;
    for (int i = 0; i < M * N; ++i)
      if (gotp[i] != pref[i] && np++ < 4)
        ADD_FAILURE() << "payload at " << i << " got=" << gotp[i] << " ref=" << pref[i];
    EXPECT_EQ(np, 0) << np << " payload mismatches";
    (void)hipFree(dDp);
  });
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dDf);
}

// ===================== C-accumulator modifier (gfx1250) =====================
// D = A*B + mod(C); low 2 bits bit0=neg bit1=abs (abs-then-neg). Validates the
// f4 base wrapper threads Cmod into both the real builtin (Float) and the FPSan
// payload path, each against a fully independent host oracle. kGrid C spans both
// signs so abs/neg are genuinely exercised.
static float host_cmod_f4(float c, int cmod) {
  if (cmod & 2)
    c = std::fabs(c);
  if (cmod & 1)
    c = -c;
  return c;
}

template <int Cmod>
__global__ void k_f4_mod_float(const std::uint32_t *Apack, const std::uint32_t *Bpack,
                               const float *C, float *D) {
  const int lane = threadIdx.x;
  v16i32_native a;
  v8i32_native b;
  for (int i = 0; i < 16; ++i)
    a[i] = static_cast<int>(Apack[lane * 16 + i]);
  for (int i = 0; i < 8; ++i)
    b[i] = static_cast<int>(Bpack[lane * 8 + i]);
  v16f_native cn;
  for (int r = 0; r < 16; ++r)
    cn[r] = C[(r + 16 * (lane >> 4)) * N + (lane & 15)];
  Value<v16f_native, Semantics::Native, kCC> c(cn);
  auto d = fpsan::amdgcn_wmma_f32_32x16x128_f4<Semantics::Native, kCC, Cmod>(a, b, c);
  for (int r = 0; r < 16; ++r)
    D[(r + 16 * (lane >> 4)) * N + (lane & 15)] = d.get(r).to_float();
}

template <int Cmod, Semantics S>
__global__ void k_f4_mod_fpsan(const std::uint32_t *Apack, const std::uint32_t *Bpack,
                               const float *C, std::uint32_t *Dpay) {
  const int lane = threadIdx.x;
  v16i32_native a;
  v8i32_native b;
  for (int i = 0; i < 16; ++i)
    a[i] = static_cast<int>(Apack[lane * 16 + i]);
  for (int i = 0; i < 8; ++i)
    b[i] = static_cast<int>(Bpack[lane * 8 + i]);
  v16f_native cn;
  for (int r = 0; r < 16; ++r)
    cn[r] = C[(r + 16 * (lane >> 4)) * N + (lane & 15)];
  Value<v16f_native, S, kCC> c(cn);
  auto d = fpsan::amdgcn_wmma_f32_32x16x128_f4<S, kCC, Cmod>(a, b, c);
  for (int r = 0; r < 16; ++r)
    Dpay[(r + 16 * (lane >> 4)) * N + (lane & 15)] = d.get(r).fpsan_payload();
}

template <int Cmod> static void run_modifier_f4() {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  std::mt19937 rng = fpsan_test::make_rng();
  std::uniform_int_distribution<int> pick(0, static_cast<int>(sizeof(kGrid) / sizeof(float)) - 1);
  std::vector<float> A(M * K), B(K * N), C(M * N);
  for (auto &x : A)
    x = kGrid[pick(rng)];
  for (auto &x : B)
    x = kGrid[pick(rng)];
  for (auto &x : C)
    x = kGrid[pick(rng)];

  std::vector<std::uint32_t> Apack(32 * 16, 0u), Bpack(32 * 8, 0u);
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k)
      pack_nibble(&Apack[a_lane(m, k) * 16], a_slot(m, k), f32_to_narrow(A[m * K + k], kFp4E2M1));
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      pack_nibble(&Bpack[b_lane(n, k) * 8], b_slot(k), f32_to_narrow(B[k * N + n], kFp4E2M1));

  std::vector<float> fref(M * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = host_cmod_f4(C[m * N + n], Cmod);
      for (int k = 0; k < K; ++k)
        acc += A[m * K + k] * B[k * N + n];
      fref[m * N + n] = acc;
    }
  std::uint32_t *dA = to_dev(Apack);
  std::uint32_t *dB = to_dev(Bpack);
  float *dC = to_dev(C);
  float *dDf = nullptr;
  HIP_CHECK(hipMalloc(&dDf, M * N * sizeof(float)));
  k_f4_mod_float<Cmod><<<1, 32>>>(dA, dB, dC, dDf);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> gotf(M * N);
  HIP_CHECK(hipMemcpy(gotf.data(), dDf, M * N * sizeof(float), hipMemcpyDeviceToHost));
  int nf = 0;
  for (int i = 0; i < M * N; ++i)
    if (bits_of(gotf[i]) != bits_of(fref[i]) && nf++ < 4)
      ADD_FAILURE() << "Cmod=" << Cmod << " float at m=" << i / N << " n=" << i % N;
  EXPECT_EQ(nf, 0) << nf << " float mismatches";

  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    std::vector<std::uint32_t> pref(M * N);
    for (int m = 0; m < M; ++m)
      for (int n = 0; n < N; ++n) {
        VF<S> cm(C[m * N + n]);
        if (Cmod & 2)
          cm = fpsan::max(cm, -cm);
        if (Cmod & 1)
          cm = -cm;
        VF<S> acc = cm;
        for (int k = 0; k < K; ++k) {
          const std::uint32_t ca = f32_to_narrow(A[m * K + k], kFp4E2M1) & 0xF;
          const std::uint32_t cb = f32_to_narrow(B[k * N + n], kFp4E2M1) & 0xF;
          acc = acc + f4_canonical_value<S>(ca) * f4_canonical_value<S>(cb);
        }
        pref[m * N + n] = acc.fpsan_payload();
      }

    std::uint32_t *dDp = nullptr;
    HIP_CHECK(hipMalloc(&dDp, M * N * sizeof(std::uint32_t)));
    k_f4_mod_fpsan<Cmod, S><<<1, 32>>>(dA, dB, dC, dDp);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> gotp(M * N);
    HIP_CHECK(hipMemcpy(gotp.data(), dDp, M * N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    int np = 0;
    for (int i = 0; i < M * N; ++i)
      if (gotp[i] != pref[i] && np++ < 4)
        ADD_FAILURE() << "Cmod=" << Cmod << " payload at " << i;
    EXPECT_EQ(np, 0) << np << " payload mismatches";
    (void)hipFree(dDp);
  });
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dDf);
}

TEST(WmmaF4_32x16x128, ModifierNegC) { run_modifier_f4<1>(); }
TEST(WmmaF4_32x16x128, ModifierAbsC) { run_modifier_f4<2>(); }
TEST(WmmaF4_32x16x128, ModifierNegAbsC) { run_modifier_f4<3>(); }

// ===================== scale (i32) / scale16 (i64) wrappers =====================
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_wmma_scale_f32_32x16x128_f4)

static int scale_lane_a(int m) {
  const int g = m >> 3;
  return (g == 1) ? (m + 8) : (g == 2) ? (m - 8) : m;
}
static float e8m0f(int e) { return std::ldexp(1.f, e - 127); }

template <int SFMT>
__global__ void k_scale_float(const std::uint32_t *Apack, const std::uint32_t *Bpack,
                              const float *C, const unsigned *SA, const unsigned *SB, float *D) {
  const int lane = threadIdx.x;
  v16i32_native a;
  v8i32_native b;
  for (int i = 0; i < 16; ++i)
    a[i] = static_cast<int>(Apack[lane * 16 + i]);
  for (int i = 0; i < 8; ++i)
    b[i] = static_cast<int>(Bpack[lane * 8 + i]);
  v16f_native cn;
  for (int r = 0; r < 16; ++r)
    cn[r] = C[(r + 16 * (lane >> 4)) * N + (lane & 15)];
  Value<v16f_native, Semantics::Native, kCC> c(cn);
  auto d = fpsan::amdgcn_wmma_scale_f32_32x16x128_f4<Semantics::Native, kCC, 0, SFMT, SFMT>(
      a, b, c, static_cast<int>(SA[lane]), static_cast<int>(SB[lane]));
  for (int r = 0; r < 16; ++r)
    D[(r + 16 * (lane >> 4)) * N + (lane & 15)] = d.get(r).to_float();
}

template <int SFMT, Semantics S>
__global__ void k_scale_fpsan(const std::uint32_t *Apack, const std::uint32_t *Bpack,
                              const float *C, const unsigned *SA, const unsigned *SB,
                              std::uint32_t *Dpay) {
  const int lane = threadIdx.x;
  v16i32_native a;
  v8i32_native b;
  for (int i = 0; i < 16; ++i)
    a[i] = static_cast<int>(Apack[lane * 16 + i]);
  for (int i = 0; i < 8; ++i)
    b[i] = static_cast<int>(Bpack[lane * 8 + i]);
  v16f_native cn;
  for (int r = 0; r < 16; ++r)
    cn[r] = C[(r + 16 * (lane >> 4)) * N + (lane & 15)];
  Value<v16f_native, S, kCC> c(cn);
  auto d = fpsan::amdgcn_wmma_scale_f32_32x16x128_f4<S, kCC, 0, SFMT, SFMT>(
      a, b, c, static_cast<int>(SA[lane]), static_cast<int>(SB[lane]));
  for (int r = 0; r < 16; ++r)
    Dpay[(r + 16 * (lane >> 4)) * N + (lane & 15)] = d.get(r).fpsan_payload();
}

template <int SFMT>
__global__ void k_scale16_float(const std::uint32_t *Apack, const std::uint32_t *Bpack,
                                const float *C, const unsigned long long *SA,
                                const unsigned long long *SB, float *D) {
  const int lane = threadIdx.x;
  v16i32_native a;
  v8i32_native b;
  for (int i = 0; i < 16; ++i)
    a[i] = static_cast<int>(Apack[lane * 16 + i]);
  for (int i = 0; i < 8; ++i)
    b[i] = static_cast<int>(Bpack[lane * 8 + i]);
  v16f_native cn;
  for (int r = 0; r < 16; ++r)
    cn[r] = C[(r + 16 * (lane >> 4)) * N + (lane & 15)];
  Value<v16f_native, Semantics::Native, kCC> c(cn);
  auto d = fpsan::amdgcn_wmma_scale16_f32_32x16x128_f4<Semantics::Native, kCC, 0, SFMT, SFMT>(
      a, b, c, static_cast<long>(SA[lane]), static_cast<long>(SB[lane]));
  for (int r = 0; r < 16; ++r)
    D[(r + 16 * (lane >> 4)) * N + (lane & 15)] = d.get(r).to_float();
}

template <int SFMT, Semantics S>
__global__ void k_scale16_fpsan(const std::uint32_t *Apack, const std::uint32_t *Bpack,
                                const float *C, const unsigned long long *SA,
                                const unsigned long long *SB, std::uint32_t *Dpay) {
  const int lane = threadIdx.x;
  v16i32_native a;
  v8i32_native b;
  for (int i = 0; i < 16; ++i)
    a[i] = static_cast<int>(Apack[lane * 16 + i]);
  for (int i = 0; i < 8; ++i)
    b[i] = static_cast<int>(Bpack[lane * 8 + i]);
  v16f_native cn;
  for (int r = 0; r < 16; ++r)
    cn[r] = C[(r + 16 * (lane >> 4)) * N + (lane & 15)];
  Value<v16f_native, S, kCC> c(cn);
  auto d = fpsan::amdgcn_wmma_scale16_f32_32x16x128_f4<S, kCC, 0, SFMT, SFMT>(
      a, b, c, static_cast<long>(SA[lane]), static_cast<long>(SB[lane]));
  for (int r = 0; r < 16; ++r)
    Dpay[(r + 16 * (lane >> 4)) * N + (lane & 15)] = d.get(r).fpsan_payload();
}

// Decode a scale byte per the WMMA scale-format immediate (mirrors the wrapper):
// SFMT=0 -> E8M0FNU (2^(b-127)); SFMT=2 -> E4M3FN (full fp8 micro-scale).
static float scale_dec(unsigned byte, int sfmt) {
  return sfmt == 2 ? fpsan::detail::narrow_to_f32(byte & 0xFF, fpsan::detail::kFp8E4M3)
                   : e8m0f(static_cast<int>(byte & 0xFF));
}

// SCALE16=false: i32 (4 bytes, byte=2*(k>>6)+((k>>2)&1));
// SCALE16=true:  i64 (8 bytes, byte=4*(k>>6)+2*((k>>2)&1)+((k>>5)&1)).
// SFMT selects the per-block scale FORMAT: 0=E8M0FNU, 2=E4M3FN (both are valid
// f4xf4 scale formats). The scale bytes are
// generated and decoded in that format; the byte->k map is format-independent.
template <bool SCALE16, int SFMT> static void run_scale(const char *tag) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  const int NBY = SCALE16 ? 8 : 4;
  std::mt19937 rng = fpsan_test::make_rng();
  std::uniform_int_distribution<int> pick(0, static_cast<int>(sizeof(kGrid) / sizeof(float)) - 1);
  std::vector<float> A(M * K), B(K * N), C(M * N);
  for (auto &x : A)
    x = kGrid[pick(rng)];
  for (auto &x : B)
    x = kGrid[pick(rng)];
  for (auto &x : C)
    x = kGrid[pick(rng)];
  // Per-(row/col, block) scale bytes, generated in the active format.
  std::vector<int> eA(M * NBY), eB(N * NBY);
  if (SFMT == 2) {
    // E4M3FN-encoded exact micro-scales (powers of two + halves stay exact).
    const float sv[] = {0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f};
    std::uniform_int_distribution<int> ps(0, 5);
    for (auto &e : eA)
      e = static_cast<int>(f32_to_narrow(sv[ps(rng)], fpsan::detail::kFp8E4M3) & 0xFF);
    for (auto &e : eB)
      e = static_cast<int>(f32_to_narrow(sv[ps(rng)], fpsan::detail::kFp8E4M3) & 0xFF);
  } else {
    std::uniform_int_distribution<int> pexp(126, 129); // E8M0 exponent bytes
    for (auto &e : eA)
      e = pexp(rng);
    for (auto &e : eB)
      e = pexp(rng);
  }

  std::vector<std::uint32_t> Apack(32 * 16, 0u), Bpack(32 * 8, 0u);
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k)
      pack_nibble(&Apack[a_lane(m, k) * 16], a_slot(m, k), f32_to_narrow(A[m * K + k], kFp4E2M1));
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      pack_nibble(&Bpack[b_lane(n, k) * 8], b_slot(k), f32_to_narrow(B[k * N + n], kFp4E2M1));

  auto sbyte = [&](int k) {
    return SCALE16 ? (4 * (k >> 6) + 2 * ((k >> 2) & 1) + ((k >> 5) & 1))
                   : (2 * (k >> 6) + ((k >> 2) & 1));
  };

  std::vector<float> fref(M * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = C[m * N + n];
      for (int k = 0; k < K; ++k) {
        const int by = sbyte(k);
        acc += A[m * K + k] * B[k * N + n] * scale_dec(eA[m * NBY + by], SFMT) *
               scale_dec(eB[n * NBY + by], SFMT);
      }
      fref[m * N + n] = acc;
    }
  auto make_pref = [&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    std::vector<std::uint32_t> pref(M * N);
    for (int m = 0; m < M; ++m)
      for (int n = 0; n < N; ++n) {
        VF<S> acc(C[m * N + n]);
        for (int k = 0; k < K; ++k) {
          const int by = sbyte(k);
          const std::uint32_t ca = f32_to_narrow(A[m * K + k], kFp4E2M1) & 0xF;
          const std::uint32_t cb = f32_to_narrow(B[k * N + n], kFp4E2M1) & 0xF;
          acc = acc + f4_canonical_value<S>(ca) * f4_canonical_value<S>(cb) *
                          VF<S>(scale_dec(eA[m * NBY + by], SFMT)) *
                          VF<S>(scale_dec(eB[n * NBY + by], SFMT));
        }
        pref[m * N + n] = acc.fpsan_payload();
      }
    return pref;
  };

  // Stage per-lane scale operands: A row m -> lane scale_lane_a(m); B col n -> lane n.
  std::uint32_t *dA = to_dev(Apack);
  std::uint32_t *dB = to_dev(Bpack);
  float *dC = to_dev(C);
  float *dDf = nullptr;
  HIP_CHECK(hipMalloc(&dDf, M * N * sizeof(float)));
  int nf = 0;
  if constexpr (!SCALE16) {
    std::vector<unsigned> SA(32, 0u), SB(32, 0u);
    for (int m = 0; m < M; ++m) {
      unsigned w = 0;
      for (int by = 0; by < 4; ++by)
        w |= (unsigned(eA[m * 4 + by]) & 0xFF) << (8 * by);
      SA[scale_lane_a(m)] = w;
    }
    for (int n = 0; n < N; ++n) {
      unsigned w = 0;
      for (int by = 0; by < 4; ++by)
        w |= (unsigned(eB[n * 4 + by]) & 0xFF) << (8 * by);
      SB[n] = w;
    }
    unsigned *dSA = to_dev(SA);
    unsigned *dSB = to_dev(SB);
    k_scale_float<SFMT><<<1, 32>>>(dA, dB, dC, dSA, dSB, dDf);
    HIP_CHECK(hipDeviceSynchronize());
    fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
      constexpr Semantics S = decltype(sem)::value;
      std::vector<std::uint32_t> pref = make_pref(sem);
      std::uint32_t *dDp = nullptr;
      HIP_CHECK(hipMalloc(&dDp, M * N * sizeof(std::uint32_t)));
      k_scale_fpsan<SFMT, S><<<1, 32>>>(dA, dB, dC, dSA, dSB, dDp);
      HIP_CHECK(hipDeviceSynchronize());
      std::vector<std::uint32_t> gotp(M * N);
      HIP_CHECK(hipMemcpy(gotp.data(), dDp, M * N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
      int np = 0;
      for (int i = 0; i < M * N; ++i)
        if (gotp[i] != pref[i] && np++ < 4)
          ADD_FAILURE() << tag << " payload at " << i << " got=" << gotp[i] << " ref=" << pref[i];
      EXPECT_EQ(np, 0) << np << " payload mismatches";
      (void)hipFree(dDp);
    });
    (void)hipFree(dSA);
    (void)hipFree(dSB);
  } else {
    std::vector<unsigned long long> SA(32, 0ull), SB(32, 0ull);
    for (int m = 0; m < M; ++m) {
      unsigned long long w = 0;
      for (int by = 0; by < 8; ++by)
        w |= static_cast<unsigned long long>(eA[m * 8 + by] & 0xFF) << (8 * by);
      SA[scale_lane_a(m)] = w;
    }
    for (int n = 0; n < N; ++n) {
      unsigned long long w = 0;
      for (int by = 0; by < 8; ++by)
        w |= static_cast<unsigned long long>(eB[n * 8 + by] & 0xFF) << (8 * by);
      SB[n] = w;
    }
    unsigned long long *dSA = to_dev(SA);
    unsigned long long *dSB = to_dev(SB);
    k_scale16_float<SFMT><<<1, 32>>>(dA, dB, dC, dSA, dSB, dDf);
    HIP_CHECK(hipDeviceSynchronize());
    fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
      constexpr Semantics S = decltype(sem)::value;
      std::vector<std::uint32_t> pref = make_pref(sem);
      std::uint32_t *dDp = nullptr;
      HIP_CHECK(hipMalloc(&dDp, M * N * sizeof(std::uint32_t)));
      k_scale16_fpsan<SFMT, S><<<1, 32>>>(dA, dB, dC, dSA, dSB, dDp);
      HIP_CHECK(hipDeviceSynchronize());
      std::vector<std::uint32_t> gotp(M * N);
      HIP_CHECK(hipMemcpy(gotp.data(), dDp, M * N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
      int np = 0;
      for (int i = 0; i < M * N; ++i)
        if (gotp[i] != pref[i] && np++ < 4)
          ADD_FAILURE() << tag << " payload at " << i << " got=" << gotp[i] << " ref=" << pref[i];
      EXPECT_EQ(np, 0) << np << " payload mismatches";
      (void)hipFree(dDp);
    });
    (void)hipFree(dSA);
    (void)hipFree(dSB);
  }
  std::vector<float> gotf(M * N);
  HIP_CHECK(hipMemcpy(gotf.data(), dDf, M * N * sizeof(float), hipMemcpyDeviceToHost));
  for (int i = 0; i < M * N; ++i)
    if (bits_of(gotf[i]) != bits_of(fref[i]) && nf++ < 4)
      ADD_FAILURE() << tag << " float at m=" << i / N << " n=" << i % N << " got=" << gotf[i]
                    << " ref=" << fref[i];
  EXPECT_EQ(nf, 0) << nf << " float mismatches";
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dDf);
}

TEST(WmmaF4_32x16x128, ScaleFloatAndFPSan) { run_scale<false, 0>("scale"); }
TEST(WmmaF4_32x16x128, Scale16FloatAndFPSan) { run_scale<true, 0>("scale16"); }
// E4M3FN-format block scales (scale fmt immediate = 2). f4xf4 with both scales
// E4M3 is an accepted gfx1250 scaled-WMMA config; the E4M3 decode is silicon-
// probed. Checked against host oracles that decode the scale bytes as
// E4M3 in both Float and FPSan modes.
TEST(WmmaF4_32x16x128, ScaleE4M3FloatAndFPSan) { run_scale<false, 2>("scale_e4m3"); }
TEST(WmmaF4_32x16x128, Scale16E4M3FloatAndFPSan) { run_scale<true, 2>("scale16_e4m3"); }

#endif // __has_builtin(__builtin_amdgcn_wmma_scale_f32_32x16x128_f4)

// ===========================================================================
// Layout RE probes (raw (lane,reg) readout; value-encoded, layout-invariant
// all-ones operands). Run with --gtest_also_run_disabled_tests.
// ===========================================================================
__global__ void k_raw(const std::uint32_t *Apack, const std::uint32_t *Bpack, float *Draw) {
  const int lane = threadIdx.x;
  v16i32_native a;
  v8i32_native b;
  for (int i = 0; i < 16; ++i)
    a[i] = static_cast<int>(Apack[lane * 16 + i]);
  for (int i = 0; i < 8; ++i)
    b[i] = static_cast<int>(Bpack[lane * 8 + i]);
  v16f_native c{};
  v16f_native d = __builtin_amdgcn_wmma_f32_32x16x128_f4(a, b, static_cast<short>(0), c);
  for (int r = 0; r < 16; ++r)
    Draw[lane * 16 + r] = d[r];
}

// Probe 1: pin A (lane,slot) and the D-row map. Stage A so row m (lane m) holds
// exactly m fp4 ones (rest 0); B = all ones (layout-invariant). Then for the row
// landing at output (lane,reg): value == m. Clean integers 0..31 (each x16)
// confirm A lane==m and reveal (lane,reg)->m.
TEST(WmmaF4_32x16x128, DISABLED_ProbeArowDmap) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  const std::uint32_t one = f32_to_narrow(1.f, kFp4E2M1) & 0xF;
  std::vector<std::uint32_t> Apack(32 * 16, 0u), Bpack(32 * 8, 0u);
  for (int m = 0; m < M; ++m)
    for (int s = 0; s < m; ++s) // m ones in lane m
      pack_nibble(&Apack[m * 16], s, one);
  for (int lane = 0; lane < 32; ++lane)
    for (int s = 0; s < 64; ++s)
      pack_nibble(&Bpack[lane * 8], s, one);

  std::uint32_t *dA = to_dev(Apack);
  std::uint32_t *dB = to_dev(Bpack);
  float *dD = nullptr;
  HIP_CHECK(hipMalloc(&dD, 512 * sizeof(float)));
  k_raw<<<1, 32>>>(dA, dB, dD);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> draw(512);
  HIP_CHECK(hipMemcpy(draw.data(), dD, 512 * sizeof(float), hipMemcpyDeviceToHost));
  printf("Probe1 (lane,reg)->m:\n");
  for (int lane = 0; lane < 32; ++lane) {
    printf("lane%2d:", lane);
    for (int r = 0; r < 16; ++r)
      printf(" %3d", static_cast<int>(draw[lane * 16 + r]));
    printf("\n");
  }
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dD);
}

// Probe 2: pin B (lane,slot) and the D-col map. Stage B so column n holds (n+1)
// ones via the hypothesized B layout; A = all ones. Output value == n+1 reveals
// (lane,reg)->n and confirms the B layout if clean integers 1..16 (each x32).
TEST(WmmaF4_32x16x128, DISABLED_ProbeBcolDmap) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  const std::uint32_t one = f32_to_narrow(1.f, kFp4E2M1) & 0xF;
  std::vector<std::uint32_t> Apack(32 * 16, 0u), Bpack(32 * 8, 0u);
  for (int lane = 0; lane < 32; ++lane)
    for (int s = 0; s < 128; ++s)
      pack_nibble(&Apack[lane * 16], s, one);
  for (int n = 0; n < N; ++n)
    for (int k = 0; k <= n; ++k) // n+1 ones in column n
      pack_nibble(&Bpack[b_lane(n, k) * 8], b_slot(k), one);

  std::uint32_t *dA = to_dev(Apack);
  std::uint32_t *dB = to_dev(Bpack);
  float *dD = nullptr;
  HIP_CHECK(hipMalloc(&dD, 512 * sizeof(float)));
  k_raw<<<1, 32>>>(dA, dB, dD);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> draw(512);
  HIP_CHECK(hipMemcpy(draw.data(), dD, 512 * sizeof(float), hipMemcpyDeviceToHost));
  printf("Probe2 (lane,reg)->n+1:\n");
  for (int lane = 0; lane < 32; ++lane) {
    printf("lane%2d:", lane);
    for (int r = 0; r < 16; ++r)
      printf(" %3d", static_cast<int>(draw[lane * 16 + r]));
    printf("\n");
  }
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dD);
}

// Probe 3: true single-nibble one-hot A sweep over a physical lane. For each
// (lane,slot) with that one A nibble = 1 (B all ones, C=0), the lit outputs are
// exactly row m*'s 16 (lane,reg) slots. Groups slots by the lit-output set
// (= a D row-class) and prints, per class, its output positions and the source
// A slots -> reveals D (lane,reg)->(m,n) structure and A (lane,slot)->(m,k).
static void onehot_sweep(int srcLane) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  const std::uint32_t one = f32_to_narrow(1.f, kFp4E2M1) & 0xF;
  std::vector<std::uint32_t> Bpack(32 * 8, 0u);
  for (int lane = 0; lane < 32; ++lane)
    for (int s = 0; s < 64; ++s)
      pack_nibble(&Bpack[lane * 8], s, one);
  std::uint32_t *dB = to_dev(Bpack);
  float *dD = nullptr;
  HIP_CHECK(hipMalloc(&dD, 512 * sizeof(float)));

  std::vector<std::vector<int>> classes;    // each: sorted lit output indices
  std::vector<std::vector<int>> classSlots; // src slots feeding each class
  for (int s = 0; s < 128; ++s) {
    std::vector<std::uint32_t> Apack(32 * 16, 0u);
    pack_nibble(&Apack[srcLane * 16], s, one);
    std::uint32_t *dA = to_dev(Apack);
    k_raw<<<1, 32>>>(dA, dB, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> draw(512);
    HIP_CHECK(hipMemcpy(draw.data(), dD, 512 * sizeof(float), hipMemcpyDeviceToHost));
    std::vector<int> lit;
    for (int i = 0; i < 512; ++i)
      if (draw[i] != 0.f)
        lit.push_back(i);
    int ci = -1;
    for (int c = 0; c < static_cast<int>(classes.size()); ++c)
      if (classes[c] == lit) {
        ci = c;
        break;
      }
    if (ci < 0) {
      ci = classes.size();
      classes.push_back(lit);
      classSlots.emplace_back();
    }
    classSlots[ci].push_back(s);
    (void)hipFree(dA);
  }
  printf("one-hot A lane=%d: %d row-classes\n", srcLane, static_cast<int>(classes.size()));
  for (int c = 0; c < static_cast<int>(classes.size()); ++c) {
    printf("  class%2d outs[", c);
    for (int i : classes[c])
      printf("%d(%d.%d) ", i, i / 16, i % 16);
    printf("] <- slots{");
    for (int s : classSlots[c])
      printf("%d ", s);
    printf("}\n");
  }
  (void)hipFree(dB);
  (void)hipFree(dD);
}

TEST(WmmaF4_32x16x128, DISABLED_ProbeOneHotLane0) { onehot_sweep(0); }
TEST(WmmaF4_32x16x128, DISABLED_ProbeOneHotLane1) { onehot_sweep(1); }
TEST(WmmaF4_32x16x128, DISABLED_ProbeOneHotLane16) { onehot_sweep(16); }
// Probe 4: A-slot -> k for row 0. A one-hot at (srcLane, sa) (srcLane in {0,16},
// both feed row 0); B encodes bit b of k in column 0 via the assumed B layout
// (b_lane/b_slot). D[0][0] (raw index 0) == bit b of k(sa). 7 launches/slot
// recover k. Reveals A's k ordering relative to B's, so the two stage
// consistently.
static void kprobe_row0(int srcLane) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  const std::uint32_t one = f32_to_narrow(1.f, kFp4E2M1) & 0xF;
  float *dD = nullptr;
  HIP_CHECK(hipMalloc(&dD, 512 * sizeof(float)));
  // Pre-build 7 B buffers (bit b of k in column 0).
  std::vector<std::uint32_t *> dBbit(7, nullptr);
  for (int b = 0; b < 7; ++b) {
    std::vector<std::uint32_t> Bpack(32 * 8, 0u);
    for (int k = 0; k < K; ++k)
      if ((k >> b) & 1)
        pack_nibble(&Bpack[b_lane(0, k) * 8], b_slot(k), one);
    dBbit[b] = to_dev(Bpack);
  }
  int kof[64];
  for (int sa = 0; sa < 64; ++sa) {
    std::vector<std::uint32_t> Apack(32 * 16, 0u);
    pack_nibble(&Apack[srcLane * 16], sa, one);
    std::uint32_t *dA = to_dev(Apack);
    int kk = 0;
    for (int b = 0; b < 7; ++b) {
      k_raw<<<1, 32>>>(dA, dBbit[b], dD);
      HIP_CHECK(hipDeviceSynchronize());
      float v = 0;
      HIP_CHECK(hipMemcpy(&v, dD, sizeof(float), hipMemcpyDeviceToHost));
      if (v != 0.f)
        kk |= 1 << b;
    }
    kof[sa] = kk;
    (void)hipFree(dA);
  }
  printf("A-slot -> k (row0, srcLane=%d):\n", srcLane);
  for (int sa = 0; sa < 64; ++sa)
    printf("s=%2d k=%3d%s", sa, kof[sa], (sa % 8 == 7) ? "\n" : "  ");
  for (int b = 0; b < 7; ++b)
    (void)hipFree(dBbit[b]);
  (void)hipFree(dD);
}

TEST(WmmaF4_32x16x128, DISABLED_ProbeKrow0Lane0) { kprobe_row0(0); }
TEST(WmmaF4_32x16x128, DISABLED_ProbeKrow0Lane16) { kprobe_row0(16); }

TEST(WmmaF4_32x16x128, DISABLED_ProbeOneHotLane8) { onehot_sweep(8); }
TEST(WmmaF4_32x16x128, DISABLED_ProbeOneHotLane15) { onehot_sweep(15); }

// ===========================================================================
// Scale (i32) / scale16 (i64) layout RE for the 32x16x128_f4 shape.
// ===========================================================================
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_wmma_scale_f32_32x16x128_f4)

__global__ void k_raw_scale(const std::uint32_t *Apack, const std::uint32_t *Bpack,
                            const unsigned *SA, const unsigned *SB, float *Draw) {
  const int lane = threadIdx.x;
  v16i32_native a;
  v8i32_native b;
  for (int i = 0; i < 16; ++i)
    a[i] = static_cast<int>(Apack[lane * 16 + i]);
  for (int i = 0; i < 8; ++i)
    b[i] = static_cast<int>(Bpack[lane * 8 + i]);
  v16f_native c{};
  const int sa = static_cast<int>(SA[lane]);
  const int sb = static_cast<int>(SB[lane]);
  v16f_native d = __builtin_amdgcn_wmma_scale_f32_32x16x128_f4(a, b, static_cast<short>(0), c, 0, 0,
                                                               sa, 0, 0, sb, false, false);
  for (int r = 0; r < 16; ++r)
    Draw[lane * 16 + r] = d[r];
}

// Pass 1: scale-lane -> rows. All-ones A/B (decoded 1.0), scaleB=1. Set scale-
// operand lane L word to all factor-2 (0x80808080); any row m fed by lane L
// reads D[m][n]=256 (=2*128). Reveals scale-lane -> rows.
static void scale_probe() {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  const std::uint32_t one = f32_to_narrow(1.f, kFp4E2M1) & 0xF;
  std::vector<std::uint32_t> Apack(32 * 16, 0u), Bpack(32 * 8, 0u);
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k)
      pack_nibble(&Apack[a_lane(m, k) * 16], a_slot(m, k), one);
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      pack_nibble(&Bpack[b_lane(n, k) * 8], b_slot(k), one);
  std::uint32_t *dA = to_dev(Apack);
  std::uint32_t *dB = to_dev(Bpack);
  std::vector<unsigned> SB(32, 0x7f7f7f7fu);
  unsigned *dSB = to_dev(SB);
  float *dD = nullptr;
  HIP_CHECK(hipMalloc(&dD, 512 * sizeof(float)));
  printf("scale-lane -> rows (D doubled to 256):\n");
  for (int L = 0; L < 32; ++L) {
    std::vector<unsigned> SA(32, 0x7f7f7f7fu);
    SA[L] = 0x80808080u;
    unsigned *dSA = to_dev(SA);
    k_raw_scale<<<1, 32>>>(dA, dB, dSA, dSB, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> draw(512);
    HIP_CHECK(hipMemcpy(draw.data(), dD, 512 * sizeof(float), hipMemcpyDeviceToHost));
    printf("L=%2d rows{", L);
    for (int i = 0; i < 512; ++i)
      if (draw[i] == 256.f)
        printf("%d ", (i % 16) + 16 * ((i / 16) >> 4)); // m
    printf("}\n");
    (void)hipFree(dSA);
  }
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dD);
  (void)hipFree(dSB);
}

// byte->k probe for the A-scale of row 0. A one-hot decoded at (0,k0); B all
// ones; C=0; scaleB=1; scaleA lane = scaleLane (row 0's, from pass 1) with byte
// b doubled. raw[0]=(lane0,reg0)=(m0,n0); ==2 iff k0 is in byte b -> byte(k0).
static void scale_byte_probe(int scaleLane) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  const std::uint32_t one = f32_to_narrow(1.f, kFp4E2M1) & 0xF;
  std::vector<std::uint32_t> Bpack(32 * 8, 0u);
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      pack_nibble(&Bpack[b_lane(n, k) * 8], b_slot(k), one);
  std::uint32_t *dB = to_dev(Bpack);
  std::vector<unsigned> SB(32, 0x7f7f7f7fu);
  unsigned *dSB = to_dev(SB);
  float *dD = nullptr;
  HIP_CHECK(hipMalloc(&dD, 512 * sizeof(float)));
  int kbyte[128];
  for (int k0 = 0; k0 < K; ++k0) {
    std::vector<std::uint32_t> Apack(32 * 16, 0u);
    pack_nibble(&Apack[a_lane(0, k0) * 16], a_slot(0, k0), one);
    std::uint32_t *dA = to_dev(Apack);
    int byte = -1;
    for (int b = 0; b < 4; ++b) {
      std::vector<unsigned> SA(32, 0x7f7f7f7fu);
      SA[scaleLane] = (0x7f7f7f7fu & ~(0xFFu << (8 * b))) | (0x80u << (8 * b));
      unsigned *dSA = to_dev(SA);
      k_raw_scale<<<1, 32>>>(dA, dB, dSA, dSB, dD);
      HIP_CHECK(hipDeviceSynchronize());
      float v = 0;
      HIP_CHECK(hipMemcpy(&v, dD, sizeof(float), hipMemcpyDeviceToHost));
      if (v == 2.f)
        byte = b;
      (void)hipFree(dSA);
    }
    kbyte[k0] = byte;
    (void)hipFree(dA);
  }
  printf("k -> A-scale byte (row0, scaleLane=%d):\n", scaleLane);
  for (int k0 = 0; k0 < K; ++k0)
    printf("k=%3d byte=%d%s", k0, kbyte[k0], (k0 % 8 == 7) ? "\n" : "  ");
  (void)hipFree(dB);
  (void)hipFree(dD);
  (void)hipFree(dSB);
}

// Pass 1b: scaleB-lane -> col. B nonzero only in column n0 (decoded 1.0); A all
// ones; scaleA=1. For each lane L set scaleB[L]=all-2; whichever L doubles
// D[*][n0] is column n0's scale lane. Also reports the byte->#k for that lane.
static void scaleB_probe(int n0) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  const std::uint32_t one = f32_to_narrow(1.f, kFp4E2M1) & 0xF;
  std::vector<std::uint32_t> Apack(32 * 16, 0u), Bpack(32 * 8, 0u);
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k)
      pack_nibble(&Apack[a_lane(m, k) * 16], a_slot(m, k), one);
  for (int k = 0; k < K; ++k)
    pack_nibble(&Bpack[b_lane(n0, k) * 8], b_slot(k), one); // only column n0
  std::uint32_t *dA = to_dev(Apack);
  std::uint32_t *dB = to_dev(Bpack);
  std::vector<unsigned> SA(32, 0x7f7f7f7fu);
  unsigned *dSA = to_dev(SA);
  float *dD = nullptr;
  HIP_CHECK(hipMalloc(&dD, 512 * sizeof(float)));
  // raw index of D[0][n0] = lane n0 (m=0 -> lane=n0, reg0)
  const int idx = n0 * 16 + 0;
  printf("scaleB col n0=%d:\n", n0);
  for (int L = 0; L < 32; ++L) {
    std::vector<unsigned> SB(32, 0x7f7f7f7fu);
    SB[L] = 0x80808080u;
    unsigned *dSB = to_dev(SB);
    k_raw_scale<<<1, 32>>>(dA, dB, dSA, dSB, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> draw(512);
    HIP_CHECK(hipMemcpy(draw.data(), dD, 512 * sizeof(float), hipMemcpyDeviceToHost));
    if (draw[idx] != 128.f)
      printf("  lane L=%2d -> D[0][%d]=%g\n", L, n0, draw[idx]);
    (void)hipFree(dSB);
  }
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dD);
  (void)hipFree(dSA);
}

TEST(WmmaF4_32x16x128, DISABLED_ProbeScaleLane) { scale_probe(); }
TEST(WmmaF4_32x16x128, DISABLED_ProbeScaleBLane) {
  scaleB_probe(0);
  scaleB_probe(1);
  scaleB_probe(7);
}
TEST(WmmaF4_32x16x128, DISABLED_ProbeScaleByteRow0) { scale_byte_probe(0); }

#endif // __has_builtin(__builtin_amdgcn_wmma_scale_f32_32x16x128_f4)

#endif // __has_builtin(__builtin_amdgcn_wmma_f32_32x16x128_f4)
