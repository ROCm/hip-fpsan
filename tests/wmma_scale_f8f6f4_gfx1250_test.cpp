// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/wmma_scale_f8f6f4_gfx1250_test.cpp
//
// gfx1250 wmma_scale_f32_16x16x128_f8f6f4: the per-operand block-scaled variant
// of the f8f6f4 WMMA. K=128 is split into K-blocks; each (row, block) of A and
// (col, block) of B carries an E8M0 scale exponent, and the dot product of each
// block is multiplied by 2^(eA-127) * 2^(eB-127) before accumulation.
//
// GROUNDING test: pins the still-unverified scale-operand layout. The A/B
// fragment layout is the validated 8-bit Wmma16x16x128Layout (fp8/fp8, fmt 0/0);
// the only unknown is how the per-lane i32 scale operand encodes (row/col,
// block). Hypothesis (wave32-natural, distinct from the wave64 MFMA model): four
// K-blocks of 32, and the scale for row m / col n, block kb is byte kb of the
// scale operand supplied at lane m / lane n. The test stages distinct per-(row,
// block) exponents, runs the real builtin (Float oracle), and asserts bit-exact
// equality vs an independent host block-scaled matmul. A wrong scale layout
// fails on the distinct exponents.
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

#include <cmath>
#include <cstdint>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;
using fpsan::detail::e8m0_to_float;
using fpsan::detail::f32_to_narrow;
using fpsan::detail::FpFormat;
using fpsan::detail::kBf6E3M2;
using fpsan::detail::kFp4E2M1;
using fpsan::detail::kFp6E2M3;
using fpsan::detail::kFp8E4M3;
using fpsan::detail::kFp8E5M2;

#if !defined(__HIP_DEVICE_COMPILE__) ||                                                            \
    __has_builtin(__builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4)

static constexpr int M = 16, N = 16, K = 128, NB = 4, BK = 32; // 4 blocks of 32

using v16i32_native = int __attribute__((ext_vector_type(16)));
using v8f_native = float __attribute__((ext_vector_type(8)));

// Validated 8-bit K=128 A/B fragment layout (byte slot at bit 8*slot).
static int ab_lane(int rc, int k) { return rc + 16 * ((k >> 2) & 1); }
static int ab_slot(int k) {
  const int reg = ((k >> 1) & 1) + 2 * ((k >> 3) & 1) + 4 * ((k >> 4) & 1) + 8 * ((k >> 5) & 1) +
                  16 * ((k >> 6) & 1);
  return 2 * reg + (k & 1);
}
static void pack_byte(std::uint32_t *reg, int slot, std::uint32_t code) {
  reg[slot >> 2] |= (code & 0xFFu) << (8 * (slot & 3));
}

__global__ void k_scale(const std::uint32_t *Apack, const std::uint32_t *Bpack, const float *C,
                        const unsigned *SA, const unsigned *SB, float *D) {
  const int lane = threadIdx.x;
  v16i32_native a, b;
  for (int i = 0; i < 16; ++i) {
    a[i] = static_cast<int>(Apack[lane * 16 + i]);
    b[i] = static_cast<int>(Bpack[lane * 16 + i]);
  }
  v8f_native c;
  for (int e = 0; e < 8; ++e)
    c[e] = C[(e + 8 * (lane >> 4)) * N + (lane & 15)];
  const int sa = static_cast<int>(SA[lane]);
  const int sb = static_cast<int>(SB[lane]);
  v8f_native d = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
      0, a, 0, b, static_cast<short>(0), c, 0, 0, sa, 0, 0, sb, false, false);
  for (int e = 0; e < 8; ++e)
    D[(e + 8 * (lane >> 4)) * N + (lane & 15)] = d[e];
}

static const float kGrid[] = {0.f,   0.5f, 1.f,   1.5f, 2.f,  3.f,  4.f, 6.f,
                              -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

TEST(WmmaScaleF8f6f4_128, LayoutMatchesHardware) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  std::mt19937 rng = fpsan_test::make_rng();
  std::uniform_int_distribution<int> pick(0, static_cast<int>(sizeof(kGrid) / sizeof(float)) - 1);
  std::uniform_int_distribution<int> pexp(126, 129); // E8M0 exps: 2^-1..2^2
  std::vector<float> A(M * K), B(K * N), C(M * N);
  for (auto &x : A)
    x = kGrid[pick(rng)];
  for (auto &x : B)
    x = kGrid[pick(rng)];
  for (auto &x : C)
    x = kGrid[pick(rng)];

  // Per-(row, block) E8M0 exponents for A and per-(col, block) for B.
  std::vector<int> eA(M * NB), eB(N * NB);
  for (auto &e : eA)
    e = pexp(rng);
  for (auto &e : eB)
    e = pexp(rng);

  // Pack A/B fragments (8-bit, validated layout).
  std::vector<std::uint32_t> Apack(32 * 16, 0u), Bpack(32 * 16, 0u);
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k)
      pack_byte(&Apack[ab_lane(m, k) * 16], ab_slot(k), f32_to_narrow(A[m * K + k], kFp8E4M3));
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      pack_byte(&Bpack[ab_lane(n, k) * 16], ab_slot(k), f32_to_narrow(B[k * N + n], kFp8E4M3));

  // Hypothesis: scale for row m / col n, block kb is byte kb of lane m / lane n
  // (replicate to lanes 16..31).
  std::vector<unsigned> SA(32, 0u), SB(32, 0u);
  for (int r = 0; r < 16; ++r) {
    unsigned wa = 0, wb = 0;
    for (int kb = 0; kb < NB; ++kb) {
      wa |= (unsigned(eA[r * NB + kb]) & 0xFFu) << (8 * kb);
      wb |= (unsigned(eB[r * NB + kb]) & 0xFFu) << (8 * kb);
    }
    SA[r] = SA[r + 16] = wa;
    SB[r] = SB[r + 16] = wb;
  }

  std::vector<float> ref(M * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = C[m * N + n];
      for (int kb = 0; kb < NB; ++kb) {
        float blk = 0.f;
        for (int k = kb * BK; k < kb * BK + BK; ++k)
          blk += A[m * K + k] * B[k * N + n];
        const float fa = std::ldexp(1.f, eA[m * NB + kb] - 127);
        const float fb = std::ldexp(1.f, eB[n * NB + kb] - 127);
        acc += blk * fa * fb;
      }
      ref[m * N + n] = acc;
    }

  std::uint32_t *dA = to_dev(Apack);
  std::uint32_t *dB = to_dev(Bpack);
  float *dC = to_dev(C);
  unsigned *dSA = to_dev(SA);
  unsigned *dSB = to_dev(SB);
  float *dD = nullptr;
  HIP_CHECK(hipMalloc(&dD, M * N * sizeof(float)));
  k_scale<<<1, 32>>>(dA, dB, dC, dSA, dSB, dD);
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
  EXPECT_EQ(mism, 0) << mism << " / " << (M * N) << " mismatched";
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dSA);
  (void)hipFree(dSB);
  (void)hipFree(dD);
}

// ===================== Sub-byte block-scaled wrapper =====================
// Validate the shipped amdgcn_wmma_scale_f32_16x16x128_f8f6f4_sub wrapper for
// sub-byte operands: Native mode == host block-scaled matmul (decoded values),
// FPSan-family semantics == payload-ring block-scaled reference in the same
// semantics. FP4/FP6/BF6 are finite packed subbyte formats and canonicalize
// through the standard FPSan cast policy. FP8/BF8 mixed operands are public
// scalar Value<> types and widen through fpsan::cast.
static constexpr Conversions kCC = Conversions::Explicit;

static const FpFormat &fmt_of(int code) {
  switch (code) {
  case 0:
    return kFp8E4M3;
  case 1:
    return kFp8E5M2;
  case 2:
    return kFp6E2M3;
  case 3:
    return kBf6E3M2;
  default:
    return kFp4E2M1;
  }
}
static constexpr int width_of(int code) { return code <= 1 ? 8 : code <= 3 ? 6 : 4; }
static void pack_field(std::uint32_t *reg, int slot, std::uint32_t code, int width) {
  const int p = width * slot;
  for (int b = 0; b < width; ++b)
    if ((code >> b) & 1u)
      reg[(p + b) >> 5] |= 1u << ((p + b) & 31);
}
template <int FMT, Semantics S> static Value<float, S, kCC> widen_ref(std::uint32_t code) {
  if constexpr (FMT == 0)
    return fpsan::cast<float>(
        Value<fpsan::fp8_e4m3, S, kCC>::from_fpsan_payload(static_cast<std::uint8_t>(code)));
  else if constexpr (FMT == 1)
    return fpsan::cast<float>(
        Value<fpsan::fp8_e5m2, S, kCC>::from_fpsan_payload(static_cast<std::uint8_t>(code)));
  else
    return fpsan_test::canonical_subbyte_widen<width_of(FMT), S, kCC>(code);
}

// Decode a scale byte per the WMMA scale-format immediate (mirrors the wrapper):
// sfmt=0 -> E8M0FNU; sfmt=2 -> E4M3FN.
static float scale_dec(unsigned byte, int sfmt) {
  return sfmt == 2 ? fpsan::detail::narrow_to_f32(byte & 0xFF, kFp8E4M3)
                   : e8m0_to_float(byte & 0xFF);
}
// Generate one scale byte in the requested format.
static int gen_scale_byte(std::mt19937 &rng, int sfmt) {
  if (sfmt == 2) {
    const float sv[] = {0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f};
    std::uniform_int_distribution<int> ps(0, 5);
    return static_cast<int>(f32_to_narrow(sv[ps(rng)], kFp8E4M3) & 0xFF);
  }
  std::uniform_int_distribution<int> pexp(126, 129);
  return pexp(rng);
}

// Build the per-(row/col,block) scale operands (byte kb of lane r), each operand
// in its own scale FORMAT (asfmt/bsfmt: 0=E8M0FNU, 2=E4M3FN).
static void make_scales(std::mt19937 &rng, std::vector<int> &eA, std::vector<int> &eB,
                        std::vector<unsigned> &SA, std::vector<unsigned> &SB, int asfmt = 0,
                        int bsfmt = 0) {
  eA.assign(M * NB, 0);
  eB.assign(N * NB, 0);
  for (auto &e : eA)
    e = gen_scale_byte(rng, asfmt);
  for (auto &e : eB)
    e = gen_scale_byte(rng, bsfmt);
  SA.assign(32, 0u);
  SB.assign(32, 0u);
  for (int r = 0; r < 16; ++r) {
    unsigned wa = 0, wb = 0;
    for (int kb = 0; kb < NB; ++kb) {
      wa |= (unsigned(eA[r * NB + kb]) & 0xFFu) << (8 * kb);
      wb |= (unsigned(eB[r * NB + kb]) & 0xFFu) << (8 * kb);
    }
    SA[r] = SA[r + 16] = wa;
    SB[r] = SB[r + 16] = wb;
  }
}

template <int AFMT, int BFMT, int ASFMT, int BSFMT>
__global__ void k_sub_scale_float(const std::uint32_t *Apack, const std::uint32_t *Bpack,
                                  const float *C, const unsigned *SA, const unsigned *SB,
                                  float *D) {
  const int lane = threadIdx.x;
  v16i32_native a, b;
  for (int i = 0; i < 16; ++i) {
    a[i] = static_cast<int>(Apack[lane * 16 + i]);
    b[i] = static_cast<int>(Bpack[lane * 16 + i]);
  }
  v8f_native cn;
  for (int e = 0; e < 8; ++e)
    cn[e] = C[(e + 8 * (lane >> 4)) * N + (lane & 15)];
  Value<v8f_native, Semantics::Native, kCC> c(cn);
  auto d = fpsan::amdgcn_wmma_scale_f32_16x16x128_f8f6f4_sub<AFMT, BFMT, Semantics::Native, kCC, 0,
                                                             ASFMT, BSFMT>(
      a, b, c, static_cast<int>(SA[lane]), static_cast<int>(SB[lane]));
  for (int e = 0; e < 8; ++e)
    D[(e + 8 * (lane >> 4)) * N + (lane & 15)] = d.get(e).to_float();
}

template <Semantics S, int AFMT, int BFMT, int ASFMT, int BSFMT>
__global__ void k_sub_scale_fpsan(const std::uint32_t *Apack, const std::uint32_t *Bpack,
                                  const float *C, const unsigned *SA, const unsigned *SB,
                                  std::uint32_t *Dpay) {
  const int lane = threadIdx.x;
  v16i32_native a, b;
  for (int i = 0; i < 16; ++i) {
    a[i] = static_cast<int>(Apack[lane * 16 + i]);
    b[i] = static_cast<int>(Bpack[lane * 16 + i]);
  }
  v8f_native cn;
  for (int e = 0; e < 8; ++e)
    cn[e] = C[(e + 8 * (lane >> 4)) * N + (lane & 15)];
  Value<v8f_native, S, kCC> c(cn);
  auto d = fpsan::amdgcn_wmma_scale_f32_16x16x128_f8f6f4_sub<AFMT, BFMT, S, kCC, 0, ASFMT, BSFMT>(
      a, b, c, static_cast<int>(SA[lane]), static_cast<int>(SB[lane]));
  for (int e = 0; e < 8; ++e)
    Dpay[(e + 8 * (lane >> 4)) * N + (lane & 15)] = d.get(e).fpsan_payload();
}

template <Semantics S, int AFMT, int BFMT, int ASFMT = 0, int BSFMT = 0>
static void run_sub_scale(const char *tag) {
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
  std::vector<int> eA, eB;
  std::vector<unsigned> SA, SB;
  make_scales(rng, eA, eB, SA, SB, ASFMT, BSFMT);

  const int aw = width_of(AFMT), bw = width_of(BFMT);
  std::vector<std::uint32_t> Apack(32 * 16, 0u), Bpack(32 * 16, 0u);
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k)
      pack_field(&Apack[ab_lane(m, k) * 16], ab_slot(k), f32_to_narrow(A[m * K + k], fmt_of(AFMT)),
                 aw);
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      pack_field(&Bpack[ab_lane(n, k) * 16], ab_slot(k), f32_to_narrow(B[k * N + n], fmt_of(BFMT)),
                 bw);

  // Grounded sub-byte scale-byte index (matches detail::wmma_sub_scale_byte).
  auto sbyte = [](int k) { return 2 * (k >> 6) + ((k >> 2) & 1); };

  // Float reference: decoded per-element block-scaled matmul (exact grid -> any
  // accumulation order is bit-exact vs hardware).
  std::vector<float> fref(M * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = C[m * N + n];
      for (int k = 0; k < K; ++k) {
        const int byte = sbyte(k);
        acc += A[m * K + k] * B[k * N + n] * scale_dec(eA[m * NB + byte], ASFMT) *
               scale_dec(eB[n * NB + byte], BSFMT);
      }
      fref[m * N + n] = acc;
    }

  // FPSan reference: payload-ring per-element scaled, canonical subbyte widening.
  const std::uint32_t amask = (1u << aw) - 1u, bmask = (1u << bw) - 1u;
  std::vector<std::uint32_t> pref(M * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      using VF = Value<float, S, kCC>;
      VF acc(C[m * N + n]);
      for (int k = 0; k < K; ++k) {
        const int byte = sbyte(k);
        const std::uint32_t ca = f32_to_narrow(A[m * K + k], fmt_of(AFMT)) & amask;
        const std::uint32_t cb = f32_to_narrow(B[k * N + n], fmt_of(BFMT)) & bmask;
        const VF fa(scale_dec(unsigned(eA[m * NB + byte]) & 0xFFu, ASFMT));
        const VF fb(scale_dec(unsigned(eB[n * NB + byte]) & 0xFFu, BSFMT));
        acc = acc + widen_ref<AFMT, S>(ca) * widen_ref<BFMT, S>(cb) * fa * fb;
      }
      pref[m * N + n] = acc.fpsan_payload();
    }

  std::uint32_t *dA = to_dev(Apack);
  std::uint32_t *dB = to_dev(Bpack);
  float *dC = to_dev(C);
  unsigned *dSA = to_dev(SA);
  unsigned *dSB = to_dev(SB);
  float *dDf = nullptr;
  std::uint32_t *dDp = nullptr;
  HIP_CHECK(hipMalloc(&dDf, M * N * sizeof(float)));
  HIP_CHECK(hipMalloc(&dDp, M * N * sizeof(std::uint32_t)));
  k_sub_scale_float<AFMT, BFMT, ASFMT, BSFMT><<<1, 32>>>(dA, dB, dC, dSA, dSB, dDf);
  k_sub_scale_fpsan<S, AFMT, BFMT, ASFMT, BSFMT><<<1, 32>>>(dA, dB, dC, dSA, dSB, dDp);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> gotf(M * N);
  std::vector<std::uint32_t> gotp(M * N);
  HIP_CHECK(hipMemcpy(gotf.data(), dDf, M * N * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(gotp.data(), dDp, M * N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
  int nf = 0, np = 0;
  for (int i = 0; i < M * N; ++i) {
    if (bits_of(gotf[i]) != bits_of(fref[i]) && nf++ < 4)
      ADD_FAILURE() << tag << " float at m=" << i / N << " n=" << i % N << " got=" << gotf[i]
                    << " ref=" << fref[i];
    if (gotp[i] != pref[i] && np++ < 4)
      ADD_FAILURE() << tag << " payload at " << i << " got=" << gotp[i] << " ref=" << pref[i];
  }
  EXPECT_EQ(nf, 0) << nf << " float mismatches";
  EXPECT_EQ(np, 0) << np << " payload mismatches";
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dSA);
  (void)hipFree(dSB);
  (void)hipFree(dDf);
  (void)hipFree(dDp);
}

template <int AFMT, int BFMT, int ASFMT = 0, int BSFMT = 0>
static void run_sub_scale_all(const char *tag) {
  fpsan_test::for_matrix_fpsan_semantics(
      [&](auto sem) { run_sub_scale<decltype(sem)::value, AFMT, BFMT, ASFMT, BSFMT>(tag); });
}

// Probe: map (scale-operand lane La, byte Ba) -> (output row m, K-block kb) for
// fp4. A[m][k] = (k/32)+1 so block kb has the distinguishable value kb+1; B=1,
// C=0; all scales 1.0 except lane La byte Ba doubled. Baseline D = 32*(1+2+3+4)
// = 320; a doubled (m*,kb*) adds 32*(kb*+1) to row m*. Prints the mapping so the
// sub-byte scale layout can be pinned (fp8 used La==m, Ba==kb; refuted for fp4).
__global__ void k_probe(const std::uint32_t *Apack, const std::uint32_t *Bpack, const unsigned *SA,
                        float *D) {
  const int lane = threadIdx.x;
  v16i32_native a, b;
  for (int i = 0; i < 16; ++i) {
    a[i] = static_cast<int>(Apack[lane * 16 + i]);
    b[i] = static_cast<int>(Bpack[lane * 16 + i]);
  }
  v8f_native c{};
  const int sa = static_cast<int>(SA[lane]);
  const int sb = static_cast<int>(0x7f7f7f7fu);
  v8f_native d = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
      4, a, 4, b, static_cast<short>(0), c, 0, 0, sa, 0, 0, sb, false, false);
  for (int e = 0; e < 8; ++e)
    D[(e + 8 * (lane >> 4)) * N + (lane & 15)] = d[e];
}

template <int AFMT>
__global__ void k_probe2(const std::uint32_t *Apack, const std::uint32_t *Bpack, const unsigned *SA,
                         const unsigned *SB, float *D) {
  const int lane = threadIdx.x;
  v16i32_native a, b;
  for (int i = 0; i < 16; ++i) {
    a[i] = static_cast<int>(Apack[lane * 16 + i]);
    b[i] = static_cast<int>(Bpack[lane * 16 + i]);
  }
  v8f_native c{};
  const int sa = static_cast<int>(SA[lane]);
  const int sb = static_cast<int>(SB[lane]);
  v8f_native d = __builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4(
      AFMT, a, AFMT, b, static_cast<short>(0), c, 0, 0, sa, 0, 0, sb, false, false);
  for (int e = 0; e < 8; ++e)
    D[(e + 8 * (lane >> 4)) * N + (lane & 15)] = d[e];
}

template <int AFMT> static void run_probe(const char *tag) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  const int w = width_of(AFMT);
  std::vector<std::uint32_t> Apack(32 * 16, 0u);
  auto put = [&](std::uint32_t *reg, int slot, std::uint32_t code) {
    const int p = w * slot;
    for (int b = 0; b < w; ++b)
      if ((code >> b) & 1u)
        reg[(p + b) >> 5] |= 1u << ((p + b) & 31);
  };
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k)
      put(&Apack[ab_lane(m, k) * 16], ab_slot(k), f32_to_narrow(1.f, fmt_of(AFMT)));

  std::vector<unsigned> SA(32, 0u), SB(32, 0x7f7f7f7fu);
  for (int L = 0; L < 32; ++L) {
    unsigned word = 0;
    for (int b = 0; b < 4; ++b)
      word |= unsigned(128 + b) << (8 * b); // bytes -> factors 2,4,8,16
    SA[L] = word;
  }
  unsigned *dSA = to_dev(SA);
  unsigned *dSB = to_dev(SB);
  std::uint32_t *dA = to_dev(Apack);
  float *dD = nullptr;
  HIP_CHECK(hipMalloc(&dD, M * N * sizeof(float)));
  int kbyte[128];
  for (int k0 = 0; k0 < K; ++k0) {
    std::vector<std::uint32_t> B1(32 * 16, 0u);
    put(&B1[ab_lane(0, k0) * 16], ab_slot(k0), f32_to_narrow(1.f, fmt_of(AFMT)));
    std::uint32_t *dB1 = to_dev(B1);
    k_probe2<AFMT><<<1, 32>>>(dA, dB1, dSA, dSB, dD);
    HIP_CHECK(hipDeviceSynchronize());
    float d00 = 0;
    HIP_CHECK(hipMemcpy(&d00, dD, sizeof(float), hipMemcpyDeviceToHost));
    int byte = -1;
    for (int b = 0; b < 4; ++b)
      if (d00 == std::ldexp(1.f, b + 1))
        byte = b;
    kbyte[k0] = byte;
    (void)hipFree(dB1);
  }
  printf("logical k -> A-scale byte (%s):\n", tag);
  for (int k0 = 0; k0 < K; ++k0)
    printf("k=%3d byte=%d%s", k0, kbyte[k0], (k0 % 8 == 7) ? "\n" : "  ");
  (void)hipFree(dA);
  (void)hipFree(dD);
  (void)hipFree(dSA);
  (void)hipFree(dSB);
}

TEST(WmmaScaleF8f6f4_128, DISABLED_ProbeScaleLayoutFp4) { run_probe<4>("fp4"); }
TEST(WmmaScaleF8f6f4_128, DISABLED_ProbeScaleLayoutFp6) { run_probe<2>("fp6"); }
TEST(WmmaScaleF8f6f4_128, DISABLED_ProbeScaleLayoutFp8) { run_probe<0>("fp8"); }

// ---- scale16 (i64) probe: 8 E8M0 bytes, finer-granularity per-operand scale ----
template <int AFMT>
__global__ void k_probe16(const std::uint32_t *Apack, const std::uint32_t *Bpack,
                          const unsigned long long *SA, const unsigned long long *SB, float *D) {
  const int lane = threadIdx.x;
  v16i32_native a, b;
  for (int i = 0; i < 16; ++i) {
    a[i] = static_cast<int>(Apack[lane * 16 + i]);
    b[i] = static_cast<int>(Bpack[lane * 16 + i]);
  }
  v8f_native c{};
  const long sa = static_cast<long>(SA[lane]);
  const long sb = static_cast<long>(SB[lane]);
  v8f_native d = __builtin_amdgcn_wmma_scale16_f32_16x16x128_f8f6f4(
      AFMT, a, AFMT, b, static_cast<short>(0), c, 0, 0, sa, 0, 0, sb, false, false);
  for (int e = 0; e < 8; ++e)
    D[(e + 8 * (lane >> 4)) * N + (lane & 15)] = d[e];
}

template <int AFMT> static void run_probe16(const char *tag) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  const int w = width_of(AFMT);
  std::vector<std::uint32_t> Apack(32 * 16, 0u);
  auto put = [&](std::uint32_t *reg, int slot, std::uint32_t code) {
    const int p = w * slot;
    for (int b = 0; b < w; ++b)
      if ((code >> b) & 1u)
        reg[(p + b) >> 5] |= 1u << ((p + b) & 31);
  };
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k)
      put(&Apack[ab_lane(m, k) * 16], ab_slot(k), f32_to_narrow(1.f, fmt_of(AFMT)));

  // Distinct per-byte A factors 2,4,...,256 (exp 128..135) across all 8 bytes; B
  // scale = 1.0 (E8M0 0x7f in every byte).
  std::vector<unsigned long long> SA(32, 0ull), SB(32, 0x7f7f7f7f7f7f7f7full);
  for (int L = 0; L < 32; ++L) {
    unsigned long long word = 0;
    for (int b = 0; b < 8; ++b)
      word |= static_cast<unsigned long long>(128 + b) << (8 * b);
    SA[L] = word;
  }
  unsigned long long *dSA = to_dev(SA);
  unsigned long long *dSB = to_dev(SB);
  std::uint32_t *dA = to_dev(Apack);
  float *dD = nullptr;
  HIP_CHECK(hipMalloc(&dD, M * N * sizeof(float)));
  int kbyte[128];
  for (int k0 = 0; k0 < K; ++k0) {
    std::vector<std::uint32_t> B1(32 * 16, 0u);
    put(&B1[ab_lane(0, k0) * 16], ab_slot(k0), f32_to_narrow(1.f, fmt_of(AFMT)));
    std::uint32_t *dB1 = to_dev(B1);
    k_probe16<AFMT><<<1, 32>>>(dA, dB1, dSA, dSB, dD);
    HIP_CHECK(hipDeviceSynchronize());
    float d00 = 0;
    HIP_CHECK(hipMemcpy(&d00, dD, sizeof(float), hipMemcpyDeviceToHost));
    int byte = -1;
    for (int b = 0; b < 8; ++b)
      if (d00 == std::ldexp(1.f, b + 1))
        byte = b;
    kbyte[k0] = byte;
    (void)hipFree(dB1);
  }
  printf("logical k -> A-scale16 byte (%s):\n", tag);
  for (int k0 = 0; k0 < K; ++k0)
    printf("k=%3d byte=%d%s", k0, kbyte[k0], (k0 % 8 == 7) ? "\n" : "  ");
  (void)hipFree(dA);
  (void)hipFree(dD);
  (void)hipFree(dSA);
  (void)hipFree(dSB);
}

TEST(WmmaScaleF8f6f4_128, DISABLED_ProbeScale16LayoutFp4) { run_probe16<4>("fp4"); }
TEST(WmmaScaleF8f6f4_128, DISABLED_ProbeScale16LayoutFp6) { run_probe16<2>("fp6"); }
TEST(WmmaScaleF8f6f4_128, DISABLED_ProbeScale16LayoutFp8) { run_probe16<0>("fp8"); }

TEST(WmmaScaleF8f6f4_128, SubFp6Fp6) { run_sub_scale_all<2, 2>("fp6_fp6"); }
TEST(WmmaScaleF8f6f4_128, SubFp4Fp4) { run_sub_scale_all<4, 4>("fp4_fp4"); }
TEST(WmmaScaleF8f6f4_128, SubFp6Fp4) { run_sub_scale_all<2, 4>("fp6_fp4"); }
// E4M3FN-format scales (scale fmt immediate = 2). On gfx1250, E4M3 scales are
// only legal in matched configs: f4xf4 with both scales E4M3,
// and the f4 operand of an f{8,6} x f4 mix carrying an E4M3 scale (the f{8,6}
// side stays E8M0). Each covers a distinct decode path; checked against host
// oracles that decode the scale bytes in the matching format.
TEST(WmmaScaleF8f6f4_128, SubFp4Fp4ScaleE4M3) { run_sub_scale_all<4, 4, 2, 2>("fp4_fp4_e4m3"); }
TEST(WmmaScaleF8f6f4_128, SubFp6Fp4ScaleBE4M3) {
  run_sub_scale_all<2, 4, 0, 2>("fp6_fp4_sB_e4m3"); // A fp6/E8M0, B fp4/E4M3
}
TEST(WmmaScaleF8f6f4_128, SubFp4Fp6ScaleAE4M3) {
  run_sub_scale_all<4, 2, 2, 0>("fp4_fp6_sA_e4m3"); // A fp4/E4M3, B fp6/E8M0
}

// ===================== scale16 (i64) sub-byte wrapper =====================
// Validate amdgcn_wmma_scale16_f32_16x16x128_f8f6f4_sub: Float == host scaled
// matmul, FPSan == payload-ring reference, using the 8-byte/finer scale map
// (detail::wmma_sub_scale16_byte).
static constexpr int NB16 = 8;

template <int AFMT, int BFMT>
__global__ void k_sub_scale16_float(const std::uint32_t *Apack, const std::uint32_t *Bpack,
                                    const float *C, const unsigned long long *SA,
                                    const unsigned long long *SB, float *D) {
  const int lane = threadIdx.x;
  v16i32_native a, b;
  for (int i = 0; i < 16; ++i) {
    a[i] = static_cast<int>(Apack[lane * 16 + i]);
    b[i] = static_cast<int>(Bpack[lane * 16 + i]);
  }
  v8f_native cn;
  for (int e = 0; e < 8; ++e)
    cn[e] = C[(e + 8 * (lane >> 4)) * N + (lane & 15)];
  Value<v8f_native, Semantics::Native, kCC> c(cn);
  auto d = fpsan::amdgcn_wmma_scale16_f32_16x16x128_f8f6f4_sub<AFMT, BFMT, Semantics::Native, kCC>(
      a, b, c, static_cast<long>(SA[lane]), static_cast<long>(SB[lane]));
  for (int e = 0; e < 8; ++e)
    D[(e + 8 * (lane >> 4)) * N + (lane & 15)] = d.get(e).to_float();
}

template <Semantics S, int AFMT, int BFMT>
__global__ void k_sub_scale16_fpsan(const std::uint32_t *Apack, const std::uint32_t *Bpack,
                                    const float *C, const unsigned long long *SA,
                                    const unsigned long long *SB, std::uint32_t *Dpay) {
  const int lane = threadIdx.x;
  v16i32_native a, b;
  for (int i = 0; i < 16; ++i) {
    a[i] = static_cast<int>(Apack[lane * 16 + i]);
    b[i] = static_cast<int>(Bpack[lane * 16 + i]);
  }
  v8f_native cn;
  for (int e = 0; e < 8; ++e)
    cn[e] = C[(e + 8 * (lane >> 4)) * N + (lane & 15)];
  Value<v8f_native, S, kCC> c(cn);
  auto d = fpsan::amdgcn_wmma_scale16_f32_16x16x128_f8f6f4_sub<AFMT, BFMT, S, kCC>(
      a, b, c, static_cast<long>(SA[lane]), static_cast<long>(SB[lane]));
  for (int e = 0; e < 8; ++e)
    Dpay[(e + 8 * (lane >> 4)) * N + (lane & 15)] = d.get(e).fpsan_payload();
}

template <Semantics S, int AFMT, int BFMT> static void run_sub_scale16(const char *tag) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  std::mt19937 rng = fpsan_test::make_rng();
  std::uniform_int_distribution<int> pick(0, static_cast<int>(sizeof(kGrid) / sizeof(float)) - 1);
  std::uniform_int_distribution<int> pexp(126, 129);
  std::vector<float> A(M * K), B(K * N), C(M * N);
  for (auto &x : A)
    x = kGrid[pick(rng)];
  for (auto &x : B)
    x = kGrid[pick(rng)];
  for (auto &x : C)
    x = kGrid[pick(rng)];

  // 8 E8M0 exponents per row/col; pack into the per-lane i64 scale operand.
  std::vector<int> eA(M * NB16), eB(N * NB16);
  for (auto &e : eA)
    e = pexp(rng);
  for (auto &e : eB)
    e = pexp(rng);
  std::vector<unsigned long long> SA(32, 0ull), SB(32, 0ull);
  for (int r = 0; r < 16; ++r) {
    unsigned long long wa = 0, wb = 0;
    for (int b = 0; b < NB16; ++b) {
      wa |= static_cast<unsigned long long>(eA[r * NB16 + b] & 0xFF) << (8 * b);
      wb |= static_cast<unsigned long long>(eB[r * NB16 + b] & 0xFF) << (8 * b);
    }
    SA[r] = SA[r + 16] = wa;
    SB[r] = SB[r + 16] = wb;
  }

  const int aw = width_of(AFMT), bw = width_of(BFMT);
  std::vector<std::uint32_t> Apack(32 * 16, 0u), Bpack(32 * 16, 0u);
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k)
      pack_field(&Apack[ab_lane(m, k) * 16], ab_slot(k), f32_to_narrow(A[m * K + k], fmt_of(AFMT)),
                 aw);
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      pack_field(&Bpack[ab_lane(n, k) * 16], ab_slot(k), f32_to_narrow(B[k * N + n], fmt_of(BFMT)),
                 bw);

  // Grounded scale16 sub-byte byte index (matches detail::wmma_sub_scale16_byte).
  auto sbyte = [](int k) { return 4 * (k >> 6) + 2 * ((k >> 2) & 1) + ((k >> 5) & 1); };

  std::vector<float> fref(M * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = C[m * N + n];
      for (int k = 0; k < K; ++k) {
        const int byte = sbyte(k);
        acc += A[m * K + k] * B[k * N + n] * std::ldexp(1.f, eA[m * NB16 + byte] - 127) *
               std::ldexp(1.f, eB[n * NB16 + byte] - 127);
      }
      fref[m * N + n] = acc;
    }

  const std::uint32_t amask = (1u << aw) - 1u, bmask = (1u << bw) - 1u;
  std::vector<std::uint32_t> pref(M * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      using VF = Value<float, S, kCC>;
      VF acc(C[m * N + n]);
      for (int k = 0; k < K; ++k) {
        const int byte = sbyte(k);
        const std::uint32_t ca = f32_to_narrow(A[m * K + k], fmt_of(AFMT)) & amask;
        const std::uint32_t cb = f32_to_narrow(B[k * N + n], fmt_of(BFMT)) & bmask;
        const VF fa(e8m0_to_float(unsigned(eA[m * NB16 + byte]) & 0xFFu));
        const VF fb(e8m0_to_float(unsigned(eB[n * NB16 + byte]) & 0xFFu));
        acc = acc + widen_ref<AFMT, S>(ca) * widen_ref<BFMT, S>(cb) * fa * fb;
      }
      pref[m * N + n] = acc.fpsan_payload();
    }

  std::uint32_t *dA = to_dev(Apack);
  std::uint32_t *dB = to_dev(Bpack);
  float *dC = to_dev(C);
  unsigned long long *dSA = to_dev(SA);
  unsigned long long *dSB = to_dev(SB);
  float *dDf = nullptr;
  std::uint32_t *dDp = nullptr;
  HIP_CHECK(hipMalloc(&dDf, M * N * sizeof(float)));
  HIP_CHECK(hipMalloc(&dDp, M * N * sizeof(std::uint32_t)));
  k_sub_scale16_float<AFMT, BFMT><<<1, 32>>>(dA, dB, dC, dSA, dSB, dDf);
  k_sub_scale16_fpsan<S, AFMT, BFMT><<<1, 32>>>(dA, dB, dC, dSA, dSB, dDp);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> gotf(M * N);
  std::vector<std::uint32_t> gotp(M * N);
  HIP_CHECK(hipMemcpy(gotf.data(), dDf, M * N * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(gotp.data(), dDp, M * N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
  int nf = 0, np = 0;
  for (int i = 0; i < M * N; ++i) {
    if (bits_of(gotf[i]) != bits_of(fref[i]) && nf++ < 4)
      ADD_FAILURE() << tag << " float at m=" << i / N << " n=" << i % N << " got=" << gotf[i]
                    << " ref=" << fref[i];
    if (gotp[i] != pref[i] && np++ < 4)
      ADD_FAILURE() << tag << " payload at " << i << " got=" << gotp[i] << " ref=" << pref[i];
  }
  EXPECT_EQ(nf, 0) << nf << " float mismatches";
  EXPECT_EQ(np, 0) << np << " payload mismatches";
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dSA);
  (void)hipFree(dSB);
  (void)hipFree(dDf);
  (void)hipFree(dDp);
}

template <int AFMT, int BFMT> static void run_sub_scale16_all(const char *tag) {
  fpsan_test::for_matrix_fpsan_semantics(
      [&](auto sem) { run_sub_scale16<decltype(sem)::value, AFMT, BFMT>(tag); });
}

TEST(WmmaScaleF8f6f4_128, Scale16SubFp6Fp6) { run_sub_scale16_all<2, 2>("s16_fp6_fp6"); }
TEST(WmmaScaleF8f6f4_128, Scale16SubFp4Fp4) { run_sub_scale16_all<4, 4>("s16_fp4_fp4"); }
TEST(WmmaScaleF8f6f4_128, Scale16SubFp6Fp4) { run_sub_scale16_all<2, 4>("s16_fp6_fp4"); }

// ===================== MIXED 8-bit x sub-byte block-scaled =====================
// Validate amdgcn_wmma_scale{,16}_f32_16x16x128_f8f6f4_mixed: the 8-bit operand
// stages on the validated ab layout, the sub-byte operand on the mixed order
// (mix_lane/mix_slot). Both operands use the 8-bit scale-byte convention
// (detail::wmma_mix_scale_byte: i32 byte = k>>5; scale16 byte =
// 2*(k>>5)+((k>>2)&1)) -- grounded by probe_mixscale one-hot sweeps. Float ==
// host block-scaled matmul, FPSan == payload-ring block-scaled reference.
static int mix_lane(int rc, int k) { return rc + 16 * ((k >> 5) & 1); }
static int mix_slot(int k) {
  return 32 * ((k >> 6) & 1) + 16 * ((k >> 2) & 1) + 8 * ((k >> 4) & 1) + 4 * ((k >> 3) & 1) +
         2 * ((k >> 1) & 1) + (k & 1);
}
template <int AFMT, int BFMT>
static void stage_mixed(const std::vector<float> &A, const std::vector<float> &B,
                        std::vector<std::uint32_t> &Apack, std::vector<std::uint32_t> &Bpack) {
  Apack.assign(32 * 16, 0u);
  Bpack.assign(32 * 16, 0u);
  const bool aSub = AFMT >= 2, bSub = BFMT >= 2;
  const int aw = width_of(AFMT), bw = width_of(BFMT);
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k)
      pack_field(&Apack[(aSub ? mix_lane(m, k) : ab_lane(m, k)) * 16],
                 aSub ? mix_slot(k) : ab_slot(k), f32_to_narrow(A[m * K + k], fmt_of(AFMT)), aw);
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k)
      pack_field(&Bpack[(bSub ? mix_lane(n, k) : ab_lane(n, k)) * 16],
                 bSub ? mix_slot(k) : ab_slot(k), f32_to_narrow(B[k * N + n], fmt_of(BFMT)), bw);
}

template <int AFMT, int BFMT, int ASFMT, int BSFMT>
__global__ void k_mix_scale_float(const std::uint32_t *Apack, const std::uint32_t *Bpack,
                                  const float *C, const unsigned *SA, const unsigned *SB,
                                  float *D) {
  const int lane = threadIdx.x;
  v16i32_native a, b;
  for (int i = 0; i < 16; ++i) {
    a[i] = static_cast<int>(Apack[lane * 16 + i]);
    b[i] = static_cast<int>(Bpack[lane * 16 + i]);
  }
  v8f_native cn;
  for (int e = 0; e < 8; ++e)
    cn[e] = C[(e + 8 * (lane >> 4)) * N + (lane & 15)];
  Value<v8f_native, Semantics::Native, kCC> c(cn);
  auto d = fpsan::amdgcn_wmma_scale_f32_16x16x128_f8f6f4_mixed<AFMT, BFMT, Semantics::Native, kCC,
                                                               0, ASFMT, BSFMT>(
      a, b, c, static_cast<int>(SA[lane]), static_cast<int>(SB[lane]));
  for (int e = 0; e < 8; ++e)
    D[(e + 8 * (lane >> 4)) * N + (lane & 15)] = d.get(e).to_float();
}

template <Semantics S, int AFMT, int BFMT, int ASFMT, int BSFMT>
__global__ void k_mix_scale_fpsan(const std::uint32_t *Apack, const std::uint32_t *Bpack,
                                  const float *C, const unsigned *SA, const unsigned *SB,
                                  std::uint32_t *Dpay) {
  const int lane = threadIdx.x;
  v16i32_native a, b;
  for (int i = 0; i < 16; ++i) {
    a[i] = static_cast<int>(Apack[lane * 16 + i]);
    b[i] = static_cast<int>(Bpack[lane * 16 + i]);
  }
  v8f_native cn;
  for (int e = 0; e < 8; ++e)
    cn[e] = C[(e + 8 * (lane >> 4)) * N + (lane & 15)];
  Value<v8f_native, S, kCC> c(cn);
  auto d = fpsan::amdgcn_wmma_scale_f32_16x16x128_f8f6f4_mixed<AFMT, BFMT, S, kCC, 0, ASFMT, BSFMT>(
      a, b, c, static_cast<int>(SA[lane]), static_cast<int>(SB[lane]));
  for (int e = 0; e < 8; ++e)
    Dpay[(e + 8 * (lane >> 4)) * N + (lane & 15)] = d.get(e).fpsan_payload();
}

template <Semantics S, int AFMT, int BFMT, int ASFMT = 0, int BSFMT = 0>
static void run_mixed_scale(const char *tag) {
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
  std::vector<int> eA, eB;
  std::vector<unsigned> SA, SB;
  make_scales(rng, eA, eB, SA, SB, ASFMT, BSFMT);

  std::vector<std::uint32_t> Apack, Bpack;
  stage_mixed<AFMT, BFMT>(A, B, Apack, Bpack);

  // Mixed scale-byte map (both operands): byte = k>>5 (matches
  // detail::wmma_mix_scale_byte).
  auto sbyte = [](int k) { return k >> 5; };

  std::vector<float> fref(M * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = C[m * N + n];
      for (int k = 0; k < K; ++k) {
        const int byte = sbyte(k);
        acc += A[m * K + k] * B[k * N + n] * scale_dec(eA[m * NB + byte], ASFMT) *
               scale_dec(eB[n * NB + byte], BSFMT);
      }
      fref[m * N + n] = acc;
    }

  const int aw = width_of(AFMT), bw = width_of(BFMT);
  const std::uint32_t amask = (1u << aw) - 1u, bmask = (1u << bw) - 1u;
  std::vector<std::uint32_t> pref(M * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      using VF = Value<float, S, kCC>;
      VF acc(C[m * N + n]);
      for (int k = 0; k < K; ++k) {
        const int byte = sbyte(k);
        const std::uint32_t ca = f32_to_narrow(A[m * K + k], fmt_of(AFMT)) & amask;
        const std::uint32_t cb = f32_to_narrow(B[k * N + n], fmt_of(BFMT)) & bmask;
        const VF fa(scale_dec(unsigned(eA[m * NB + byte]) & 0xFFu, ASFMT));
        const VF fb(scale_dec(unsigned(eB[n * NB + byte]) & 0xFFu, BSFMT));
        acc = acc + widen_ref<AFMT, S>(ca) * widen_ref<BFMT, S>(cb) * fa * fb;
      }
      pref[m * N + n] = acc.fpsan_payload();
    }

  std::uint32_t *dA = to_dev(Apack);
  std::uint32_t *dB = to_dev(Bpack);
  float *dC = to_dev(C);
  unsigned *dSA = to_dev(SA);
  unsigned *dSB = to_dev(SB);
  float *dDf = nullptr;
  std::uint32_t *dDp = nullptr;
  HIP_CHECK(hipMalloc(&dDf, M * N * sizeof(float)));
  HIP_CHECK(hipMalloc(&dDp, M * N * sizeof(std::uint32_t)));
  k_mix_scale_float<AFMT, BFMT, ASFMT, BSFMT><<<1, 32>>>(dA, dB, dC, dSA, dSB, dDf);
  k_mix_scale_fpsan<S, AFMT, BFMT, ASFMT, BSFMT><<<1, 32>>>(dA, dB, dC, dSA, dSB, dDp);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> gotf(M * N);
  std::vector<std::uint32_t> gotp(M * N);
  HIP_CHECK(hipMemcpy(gotf.data(), dDf, M * N * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(gotp.data(), dDp, M * N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
  int nf = 0, np = 0;
  for (int i = 0; i < M * N; ++i) {
    if (bits_of(gotf[i]) != bits_of(fref[i]) && nf++ < 4)
      ADD_FAILURE() << tag << " float at m=" << i / N << " n=" << i % N << " got=" << gotf[i]
                    << " ref=" << fref[i];
    if (gotp[i] != pref[i] && np++ < 4)
      ADD_FAILURE() << tag << " payload at " << i << " got=" << gotp[i] << " ref=" << pref[i];
  }
  EXPECT_EQ(nf, 0) << nf << " float mismatches";
  EXPECT_EQ(np, 0) << np << " payload mismatches";
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dSA);
  (void)hipFree(dSB);
  (void)hipFree(dDf);
  (void)hipFree(dDp);
}

template <int AFMT, int BFMT, int ASFMT = 0, int BSFMT = 0>
static void run_mixed_scale_all(const char *tag) {
  fpsan_test::for_matrix_fpsan_semantics(
      [&](auto sem) { run_mixed_scale<decltype(sem)::value, AFMT, BFMT, ASFMT, BSFMT>(tag); });
}

TEST(WmmaScaleF8f6f4_128, MixFp8Fp4) { run_mixed_scale_all<0, 4>("mix_fp8_fp4"); }
TEST(WmmaScaleF8f6f4_128, MixFp8Fp6) { run_mixed_scale_all<0, 2>("mix_fp8_fp6"); }
TEST(WmmaScaleF8f6f4_128, MixBf8Fp4) { run_mixed_scale_all<1, 4>("mix_bf8_fp4"); }
TEST(WmmaScaleF8f6f4_128, MixFp4Fp8) { run_mixed_scale_all<4, 0>("mix_fp4_fp8"); }
// E4M3-format scale on the f4 operand of an 8-bit x f4 mix (the 8-bit side stays
// E8M0), the legal mixed E4M3 config.
TEST(WmmaScaleF8f6f4_128, MixFp8Fp4ScaleBE4M3) {
  run_mixed_scale_all<0, 4, 0, 2>("mix_fp8_fp4_sB_e4m3");
}
TEST(WmmaScaleF8f6f4_128, MixFp4Fp8ScaleAE4M3) {
  run_mixed_scale_all<4, 0, 2, 0>("mix_fp4_fp8_sA_e4m3");
}

// ---- scale16 (i64) MIXED 8-bit x sub-byte ----
template <int AFMT, int BFMT>
__global__ void k_mix_scale16_float(const std::uint32_t *Apack, const std::uint32_t *Bpack,
                                    const float *C, const unsigned long long *SA,
                                    const unsigned long long *SB, float *D) {
  const int lane = threadIdx.x;
  v16i32_native a, b;
  for (int i = 0; i < 16; ++i) {
    a[i] = static_cast<int>(Apack[lane * 16 + i]);
    b[i] = static_cast<int>(Bpack[lane * 16 + i]);
  }
  v8f_native cn;
  for (int e = 0; e < 8; ++e)
    cn[e] = C[(e + 8 * (lane >> 4)) * N + (lane & 15)];
  Value<v8f_native, Semantics::Native, kCC> c(cn);
  auto d =
      fpsan::amdgcn_wmma_scale16_f32_16x16x128_f8f6f4_mixed<AFMT, BFMT, Semantics::Native, kCC>(
          a, b, c, static_cast<long>(SA[lane]), static_cast<long>(SB[lane]));
  for (int e = 0; e < 8; ++e)
    D[(e + 8 * (lane >> 4)) * N + (lane & 15)] = d.get(e).to_float();
}

template <Semantics S, int AFMT, int BFMT>
__global__ void k_mix_scale16_fpsan(const std::uint32_t *Apack, const std::uint32_t *Bpack,
                                    const float *C, const unsigned long long *SA,
                                    const unsigned long long *SB, std::uint32_t *Dpay) {
  const int lane = threadIdx.x;
  v16i32_native a, b;
  for (int i = 0; i < 16; ++i) {
    a[i] = static_cast<int>(Apack[lane * 16 + i]);
    b[i] = static_cast<int>(Bpack[lane * 16 + i]);
  }
  v8f_native cn;
  for (int e = 0; e < 8; ++e)
    cn[e] = C[(e + 8 * (lane >> 4)) * N + (lane & 15)];
  Value<v8f_native, S, kCC> c(cn);
  auto d = fpsan::amdgcn_wmma_scale16_f32_16x16x128_f8f6f4_mixed<AFMT, BFMT, S, kCC>(
      a, b, c, static_cast<long>(SA[lane]), static_cast<long>(SB[lane]));
  for (int e = 0; e < 8; ++e)
    Dpay[(e + 8 * (lane >> 4)) * N + (lane & 15)] = d.get(e).fpsan_payload();
}

template <Semantics S, int AFMT, int BFMT> static void run_mixed_scale16(const char *tag) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  std::mt19937 rng = fpsan_test::make_rng();
  std::uniform_int_distribution<int> pick(0, static_cast<int>(sizeof(kGrid) / sizeof(float)) - 1);
  std::uniform_int_distribution<int> pexp(126, 129);
  std::vector<float> A(M * K), B(K * N), C(M * N);
  for (auto &x : A)
    x = kGrid[pick(rng)];
  for (auto &x : B)
    x = kGrid[pick(rng)];
  for (auto &x : C)
    x = kGrid[pick(rng)];

  std::vector<int> eA(M * NB16), eB(N * NB16);
  for (auto &e : eA)
    e = pexp(rng);
  for (auto &e : eB)
    e = pexp(rng);
  std::vector<unsigned long long> SA(32, 0ull), SB(32, 0ull);
  for (int r = 0; r < 16; ++r) {
    unsigned long long wa = 0, wb = 0;
    for (int b = 0; b < NB16; ++b) {
      wa |= static_cast<unsigned long long>(eA[r * NB16 + b] & 0xFF) << (8 * b);
      wb |= static_cast<unsigned long long>(eB[r * NB16 + b] & 0xFF) << (8 * b);
    }
    SA[r] = SA[r + 16] = wa;
    SB[r] = SB[r + 16] = wb;
  }

  std::vector<std::uint32_t> Apack, Bpack;
  stage_mixed<AFMT, BFMT>(A, B, Apack, Bpack);

  // Mixed scale16 byte map (both operands): byte = 2*(k>>5) + ((k>>2)&1)
  // (matches detail::wmma_mix_scale16_byte).
  auto sbyte = [](int k) { return 2 * (k >> 5) + ((k >> 2) & 1); };

  std::vector<float> fref(M * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = C[m * N + n];
      for (int k = 0; k < K; ++k) {
        const int byte = sbyte(k);
        acc += A[m * K + k] * B[k * N + n] * std::ldexp(1.f, eA[m * NB16 + byte] - 127) *
               std::ldexp(1.f, eB[n * NB16 + byte] - 127);
      }
      fref[m * N + n] = acc;
    }

  const int aw = width_of(AFMT), bw = width_of(BFMT);
  const std::uint32_t amask = (1u << aw) - 1u, bmask = (1u << bw) - 1u;
  std::vector<std::uint32_t> pref(M * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      using VF = Value<float, S, kCC>;
      VF acc(C[m * N + n]);
      for (int k = 0; k < K; ++k) {
        const int byte = sbyte(k);
        const std::uint32_t ca = f32_to_narrow(A[m * K + k], fmt_of(AFMT)) & amask;
        const std::uint32_t cb = f32_to_narrow(B[k * N + n], fmt_of(BFMT)) & bmask;
        const VF fa(e8m0_to_float(unsigned(eA[m * NB16 + byte]) & 0xFFu));
        const VF fb(e8m0_to_float(unsigned(eB[n * NB16 + byte]) & 0xFFu));
        acc = acc + widen_ref<AFMT, S>(ca) * widen_ref<BFMT, S>(cb) * fa * fb;
      }
      pref[m * N + n] = acc.fpsan_payload();
    }

  std::uint32_t *dA = to_dev(Apack);
  std::uint32_t *dB = to_dev(Bpack);
  float *dC = to_dev(C);
  unsigned long long *dSA = to_dev(SA);
  unsigned long long *dSB = to_dev(SB);
  float *dDf = nullptr;
  std::uint32_t *dDp = nullptr;
  HIP_CHECK(hipMalloc(&dDf, M * N * sizeof(float)));
  HIP_CHECK(hipMalloc(&dDp, M * N * sizeof(std::uint32_t)));
  k_mix_scale16_float<AFMT, BFMT><<<1, 32>>>(dA, dB, dC, dSA, dSB, dDf);
  k_mix_scale16_fpsan<S, AFMT, BFMT><<<1, 32>>>(dA, dB, dC, dSA, dSB, dDp);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> gotf(M * N);
  std::vector<std::uint32_t> gotp(M * N);
  HIP_CHECK(hipMemcpy(gotf.data(), dDf, M * N * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(gotp.data(), dDp, M * N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
  int nf = 0, np = 0;
  for (int i = 0; i < M * N; ++i) {
    if (bits_of(gotf[i]) != bits_of(fref[i]) && nf++ < 4)
      ADD_FAILURE() << tag << " float at m=" << i / N << " n=" << i % N << " got=" << gotf[i]
                    << " ref=" << fref[i];
    if (gotp[i] != pref[i] && np++ < 4)
      ADD_FAILURE() << tag << " payload at " << i << " got=" << gotp[i] << " ref=" << pref[i];
  }
  EXPECT_EQ(nf, 0) << nf << " float mismatches";
  EXPECT_EQ(np, 0) << np << " payload mismatches";
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dSA);
  (void)hipFree(dSB);
  (void)hipFree(dDf);
  (void)hipFree(dDp);
}

template <int AFMT, int BFMT> static void run_mixed_scale16_all(const char *tag) {
  fpsan_test::for_matrix_fpsan_semantics(
      [&](auto sem) { run_mixed_scale16<decltype(sem)::value, AFMT, BFMT>(tag); });
}

TEST(WmmaScaleF8f6f4_128, MixS16Fp8Fp4) { run_mixed_scale16_all<0, 4>("s16_mix_fp8_fp4"); }
TEST(WmmaScaleF8f6f4_128, MixS16Fp8Fp6) { run_mixed_scale16_all<0, 2>("s16_mix_fp8_fp6"); }
TEST(WmmaScaleF8f6f4_128, MixS16Fp4Fp8) { run_mixed_scale16_all<4, 0>("s16_mix_fp4_fp8"); }

#endif // __has_builtin(__builtin_amdgcn_wmma_scale_f32_16x16x128_f8f6f4)
