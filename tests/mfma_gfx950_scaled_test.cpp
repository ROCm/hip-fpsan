// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mfma_gfx950_parts.hpp"
#include "subbyte_oracle.hpp"

// ---------------------------------------------------------------------------
// Scaled f8f6f4 MFMA (16x16x128 / 32x32x64). The Float path forwards to the
// real builtin with the corrected 9-arg signature (A/B as v8i32, no ABID); the
// FPSan path runs the OCP-MX scaled dataflow (see amdgcn_mfma.hpp). This is a
// minimal Float smoke launch that asserts the builtin executes cleanly on
// MI350; the exhaustive Float-vs-hardware and FPSan-vs-scalar-reference checks
// live in the ScaledMfma16x16x128_* / ScaledMfma32x32x64_* suites further down.
// ---------------------------------------------------------------------------
using fpsan::v32e4m3_native;

template <Semantics S>
__global__ void k_scale_16x16x128(const fpsan::v32e4m3_native *A, const fpsan::v32e4m3_native *B,
                                  const v4f_native *C, v4f_native *D, int sa, int sb) {
  Value<fpsan::v32e4m3_native, S, kCC> a{A[0]}, b{B[0]};
  Value<v4f_native, S, kCC> c{C[0]};
  auto d = fpsan::amdgcn_mfma_scale_f32_16x16x128_f8f6f4<0, 0, 0, 0, 0, S, kCC>(a, b, c, sa, sb);
  if constexpr (S == Semantics::Native)
    D[0] = static_cast<v4f_native>(d);
}

template <Semantics S>
__global__ void k_scale_32x32x64(const fpsan::v32e4m3_native *A, const fpsan::v32e4m3_native *B,
                                 const v16f_native *C, v16f_native *D, int sa, int sb) {
  Value<fpsan::v32e4m3_native, S, kCC> a{A[0]}, b{B[0]};
  Value<v16f_native, S, kCC> c{C[0]};
  auto d = fpsan::amdgcn_mfma_scale_f32_32x32x64_f8f6f4<0, 0, 0, 0, 0, S, kCC>(a, b, c, sa, sb);
  if constexpr (S == Semantics::Native)
    D[0] = static_cast<v16f_native>(d);
}

TEST(ScaledMfma, F8F6F4_16x16x128_FloatLaunches) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  fpsan::v32e4m3_native *dA, *dB;
  v4f_native *dC, *dD;
  HIP_CHECK(hipMalloc(&dA, sizeof(*dA)));
  HIP_CHECK(hipMalloc(&dB, sizeof(*dB)));
  HIP_CHECK(hipMalloc(&dC, sizeof(*dC)));
  HIP_CHECK(hipMalloc(&dD, sizeof(*dD)));
  HIP_CHECK(hipMemset(dA, 0, sizeof(*dA)));
  HIP_CHECK(hipMemset(dB, 0, sizeof(*dB)));
  HIP_CHECK(hipMemset(dC, 0, sizeof(*dC)));
  k_scale_16x16x128<Semantics::Native><<<1, WAVE>>>(dA, dB, dC, dD, 0x7F, 0x7F);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipGetLastError());
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dD);
}

TEST(ScaledMfma, F8F6F4_32x32x64_FloatLaunches) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  fpsan::v32e4m3_native *dA, *dB;
  v16f_native *dC, *dD;
  HIP_CHECK(hipMalloc(&dA, sizeof(*dA)));
  HIP_CHECK(hipMalloc(&dB, sizeof(*dB)));
  HIP_CHECK(hipMalloc(&dC, sizeof(*dC)));
  HIP_CHECK(hipMalloc(&dD, sizeof(*dD)));
  HIP_CHECK(hipMemset(dA, 0, sizeof(*dA)));
  HIP_CHECK(hipMemset(dB, 0, sizeof(*dB)));
  HIP_CHECK(hipMemset(dC, 0, sizeof(*dC)));
  k_scale_32x32x64<Semantics::Native><<<1, WAVE>>>(dA, dB, dC, dD, 0x7F, 0x7F);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipGetLastError());
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dD);
}

// ---------------------------------------------------------------------------
// Scaled f8f6f4 MFMA 16x16x128 (E4M3), FPSan dataflow. Layout + scale are the
// silicon-verified model in amdgcn_mfma.hpp:
//   A[i][k] @ lane 16*((k%64)/16)+i, byte (k/64)*16+(k%16); B symmetric in j.
//   D[i][j] @ output_loc_32: reg=i%4, lane=(i/4)*16+j.
//   scale = E8M0 2^(byte-127), applied per row/col after the dot product.
// Both scale operands are exercised with non-trivial E8M0 exponents; the FPSan
// dataflow and the real builtin must agree bit-for-bit.
// ---------------------------------------------------------------------------
static constexpr int SK = 128, SM = 16, SN = 16;

// Element-generic scaled-MFMA kernel. AElem/BElem select the 8-bit operand
// format (fpsan::fp8_e4m3 or fpsan::fp8_e5m2); CBSZ/BLGP are the matching
// format immediates (E4M3=0, E5M2=1).
template <Semantics S, class Out, class AElem, class BElem, int CBSZ, int BLGP>
__global__ void k_scale16(const float *A, const float *B, const float *C, Out *D, int sa, int sb) {
  using AFrag = fpsan::detail::v32_fragment<AElem>;
  using BFrag = fpsan::detail::v32_fragment<BElem>;
  int lane = threadIdx.x;
  AFrag an{};
  BFrag bn{};
  for (int half = 0; half < 2; ++half)
    for (int kk = 0; kk < 16; ++kk) {
      int g = lane / 16, idx = lane % 16;
      int k = half * 64 + g * 16 + kk, byte = half * 16 + kk;
      an[byte] = AElem(A[idx * SK + k]);
      bn[byte] = BElem(B[k * SN + idx]);
    }
  fpsan::v4f_native cn{};
  for (int reg = 0; reg < 4; ++reg) {
    int i = 4 * (lane / 16) + reg, j = lane % 16;
    cn[reg] = C[i * SN + j];
  }
  Value<AFrag, S, kCC> a{an};
  Value<BFrag, S, kCC> b{bn};
  Value<fpsan::v4f_native, S, kCC> c{cn};
  auto d =
      fpsan::amdgcn_mfma_scale_f32_16x16x128_f8f6f4<CBSZ, 0, BLGP, 0, 0, S, kCC>(a, b, c, sa, sb);
  for (int reg = 0; reg < 4; ++reg) {
    int i = 4 * (lane / 16) + reg, j = lane % 16;
    if constexpr (S == Semantics::Native)
      D[i * SN + j] = d.get(reg).to_float();
    else
      D[i * SN + j] = d.get(reg).fpsan_payload();
  }
}

namespace {
struct ScaleMats {
  std::vector<float> A, B, C;
};
ScaleMats make_scale_mats() {
  ScaleMats m;
  m.A.resize(SM * SK);
  m.B.resize(SK * SN);
  m.C.resize(SM * SN);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : m.A)
    x = fpsan_test::pick_int_valued<float>(rng, -3, 3);
  for (auto &x : m.B)
    x = fpsan_test::pick_int_valued<float>(rng, -2, 2);
  for (auto &x : m.C)
    x = fpsan_test::pick_int_valued<float>(rng, -4, 4);
  return m;
}
} // namespace

// Float dataflow (== builtin) with both scales nonzero. Small-int inputs are
// exact in both e4m3 and e5m2, so the reference matmul is format-independent.
template <class AElem, class BElem, int CBSZ, int BLGP> void run_scale16_layout() {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  ScaleMats m = make_scale_mats();
  const int sa = 129, sb = 126; // 2^2 and 2^-1
  const float fa = fpsan::detail::e8m0_to_float(129), fb = fpsan::detail::e8m0_to_float(126);
  std::vector<float> ref(SM * SN);
  for (int i = 0; i < SM; ++i)
    for (int j = 0; j < SN; ++j) {
      double dot = 0;
      for (int k = 0; k < SK; ++k)
        dot += static_cast<double>(m.A[i * SK + k]) * static_cast<double>(m.B[k * SN + j]);
      ref[i * SN + j] = static_cast<float>(static_cast<double>(m.C[i * SN + j]) +
                                           dot * static_cast<double>(fa) * static_cast<double>(fb));
    }
  float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C), *dD;
  HIP_CHECK(hipMalloc(&dD, SM * SN * sizeof(float)));
  k_scale16<Semantics::Native, float, AElem, BElem, CBSZ, BLGP>
      <<<1, WAVE>>>(dA, dB, dC, dD, sa, sb);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> got(SM * SN);
  HIP_CHECK(hipMemcpy(got.data(), dD, SM * SN * sizeof(float), hipMemcpyDeviceToHost));
  for (int i = 0; i < SM * SN; ++i)
    EXPECT_EQ(bits_of(got[i]), bits_of(ref[i])) << "at " << i;
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dD);
}

template <class AElem, class BElem, int CBSZ, int BLGP> void run_scale16_fpsan() {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  ScaleMats m = make_scale_mats();
  const int sa = 128, sb = 130; // both scales nonzero
  float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C);
  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VF = Value<float, S, kCC>;
    using VA = Value<AElem, S, kCC>;
    using VB = Value<BElem, S, kCC>;
    const VF vsa(fpsan::detail::e8m0_to_float(128)), vsb(fpsan::detail::e8m0_to_float(130));
    std::vector<std::uint32_t> ref(SM * SN);
    for (int i = 0; i < SM; ++i)
      for (int j = 0; j < SN; ++j) {
        VF dot(0.0f);
        // Mirror the dataflow: embed as fp8, then cast to float in the ring.
        for (int k = 0; k < SK; ++k)
          dot = dot + fpsan::cast<float>(VA(AElem(m.A[i * SK + k]))) *
                          fpsan::cast<float>(VB(BElem(m.B[k * SN + j])));
        ref[i * SN + j] = (VF(m.C[i * SN + j]) + dot * vsa * vsb).fpsan_payload();
      }
    std::uint32_t *dD;
    HIP_CHECK(hipMalloc(&dD, SM * SN * sizeof(std::uint32_t)));
    k_scale16<S, std::uint32_t, AElem, BElem, CBSZ, BLGP><<<1, WAVE>>>(dA, dB, dC, dD, sa, sb);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(SM * SN);
    HIP_CHECK(hipMemcpy(got.data(), dD, SM * SN * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for (int i = 0; i < SM * SN; ++i)
      EXPECT_EQ(got[i], ref[i]) << "at " << i;
    (void)hipFree(dD);
  });
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
}

using fpsan::fp8_e4m3;
using fpsan::fp8_e5m2;

TEST(ScaledMfma16x16x128_E4M3, LayoutMatchesHardware) {
  run_scale16_layout<fp8_e4m3, fp8_e4m3, 0, 0>();
}
TEST(ScaledMfma16x16x128_E4M3, FpsanMatchesScalarReference) {
  run_scale16_fpsan<fp8_e4m3, fp8_e4m3, 0, 0>();
}
TEST(ScaledMfma16x16x128_E5M2, LayoutMatchesHardware) {
  run_scale16_layout<fp8_e5m2, fp8_e5m2, 1, 1>();
}
TEST(ScaledMfma16x16x128_E5M2, FpsanMatchesScalarReference) {
  run_scale16_fpsan<fp8_e5m2, fp8_e5m2, 1, 1>();
}
TEST(ScaledMfma16x16x128_Mixed, LayoutMatchesHardware) {
  run_scale16_layout<fp8_e4m3, fp8_e5m2, 0, 1>(); // A=E4M3, B=E5M2
}
TEST(ScaledMfma16x16x128_Mixed, FpsanMatchesScalarReference) {
  run_scale16_fpsan<fp8_e4m3, fp8_e5m2, 0, 1>();
}

// ---------------------------------------------------------------------------
// Scaled f8f6f4 MFMA 32x32x64. Layout (ISA 7.1.5):
//   A[m][k] @ lane 16*(2*((k%32)/16) + m/16) + (m%16), byte (k/32)*16+(k%16);
//   B symmetric in n. Output via output_loc_32(32,32,...), 16 regs/lane.
// ---------------------------------------------------------------------------
static constexpr int S2M = 32, S2N = 32, S2K = 64;

template <Semantics S, class Out, class AElem, class BElem, int CBSZ, int BLGP>
__global__ void k_scale32(const float *A, const float *B, const float *C, Out *D, int sa, int sb) {
  using AFrag = fpsan::detail::v32_fragment<AElem>;
  using BFrag = fpsan::detail::v32_fragment<BElem>;
  int lane = threadIdx.x;
  int mn = (lane % 16) + 16 * ((lane / 16) & 1);
  AFrag an{};
  BFrag bn{};
  for (int b = 0; b < 32; ++b) {
    int k = (b / 16) * 32 + ((lane / 16) / 2) * 16 + (b % 16);
    an[b] = AElem(A[mn * S2K + k]);
    bn[b] = BElem(B[k * S2N + mn]);
  }
  fpsan::v16f_native cn{};
  for (int i = 0; i < S2M; ++i)
    for (int j = 0; j < S2N; ++j) {
      auto loc = fpsan::detail::output_loc_32(S2M, S2N, i, j, 0);
      if (loc.lane == lane)
        cn[loc.reg] = C[i * S2N + j];
    }
  Value<AFrag, S, kCC> a{an};
  Value<BFrag, S, kCC> b{bn};
  Value<fpsan::v16f_native, S, kCC> c{cn};
  auto d =
      fpsan::amdgcn_mfma_scale_f32_32x32x64_f8f6f4<CBSZ, 0, BLGP, 0, 0, S, kCC>(a, b, c, sa, sb);
  for (int i = 0; i < S2M; ++i)
    for (int j = 0; j < S2N; ++j) {
      auto loc = fpsan::detail::output_loc_32(S2M, S2N, i, j, 0);
      if (loc.lane == lane) {
        if constexpr (S == Semantics::Native)
          D[i * S2N + j] = d.get(loc.reg).to_float();
        else
          D[i * S2N + j] = d.get(loc.reg).fpsan_payload();
      }
    }
}

namespace {
ScaleMats make_scale_mats32() {
  ScaleMats m;
  m.A.resize(S2M * S2K);
  m.B.resize(S2K * S2N);
  m.C.resize(S2M * S2N);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : m.A)
    x = fpsan_test::pick_int_valued<float>(rng, -3, 3);
  for (auto &x : m.B)
    x = fpsan_test::pick_int_valued<float>(rng, -2, 2);
  for (auto &x : m.C)
    x = fpsan_test::pick_int_valued<float>(rng, -4, 4);
  return m;
}
} // namespace

template <class AElem, class BElem, int CBSZ, int BLGP> void run_scale32_layout() {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  ScaleMats m = make_scale_mats32();
  const int sa = 129, sb = 126;
  const float fa = fpsan::detail::e8m0_to_float(129), fb = fpsan::detail::e8m0_to_float(126);
  std::vector<float> ref(S2M * S2N);
  for (int i = 0; i < S2M; ++i)
    for (int j = 0; j < S2N; ++j) {
      double dot = 0;
      for (int k = 0; k < S2K; ++k)
        dot += static_cast<double>(m.A[i * S2K + k]) * static_cast<double>(m.B[k * S2N + j]);
      ref[i * S2N + j] =
          static_cast<float>(static_cast<double>(m.C[i * S2N + j]) +
                             dot * static_cast<double>(fa) * static_cast<double>(fb));
    }
  float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C), *dD;
  HIP_CHECK(hipMalloc(&dD, S2M * S2N * sizeof(float)));
  k_scale32<Semantics::Native, float, AElem, BElem, CBSZ, BLGP>
      <<<1, WAVE>>>(dA, dB, dC, dD, sa, sb);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> got(S2M * S2N);
  HIP_CHECK(hipMemcpy(got.data(), dD, S2M * S2N * sizeof(float), hipMemcpyDeviceToHost));
  for (int i = 0; i < S2M * S2N; ++i)
    EXPECT_EQ(bits_of(got[i]), bits_of(ref[i])) << "at " << i;
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dD);
}

template <class AElem, class BElem, int CBSZ, int BLGP> void run_scale32_fpsan() {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  ScaleMats m = make_scale_mats32();
  const int sa = 128, sb = 130;
  float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C);
  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VF = Value<float, S, kCC>;
    using VA = Value<AElem, S, kCC>;
    using VB = Value<BElem, S, kCC>;
    const VF vsa(fpsan::detail::e8m0_to_float(128)), vsb(fpsan::detail::e8m0_to_float(130));
    std::vector<std::uint32_t> ref(S2M * S2N);
    for (int i = 0; i < S2M; ++i)
      for (int j = 0; j < S2N; ++j) {
        VF dot(0.0f);
        for (int k = 0; k < S2K; ++k)
          dot = dot + fpsan::cast<float>(VA(AElem(m.A[i * S2K + k]))) *
                          fpsan::cast<float>(VB(BElem(m.B[k * S2N + j])));
        ref[i * S2N + j] = (VF(m.C[i * S2N + j]) + dot * vsa * vsb).fpsan_payload();
      }
    std::uint32_t *dD;
    HIP_CHECK(hipMalloc(&dD, S2M * S2N * sizeof(std::uint32_t)));
    k_scale32<S, std::uint32_t, AElem, BElem, CBSZ, BLGP><<<1, WAVE>>>(dA, dB, dC, dD, sa, sb);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(S2M * S2N);
    HIP_CHECK(hipMemcpy(got.data(), dD, S2M * S2N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for (int i = 0; i < S2M * S2N; ++i)
      EXPECT_EQ(got[i], ref[i]) << "at " << i;
    (void)hipFree(dD);
  });
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
}

TEST(ScaledMfma32x32x64_E4M3, LayoutMatchesHardware) {
  run_scale32_layout<fp8_e4m3, fp8_e4m3, 0, 0>();
}
TEST(ScaledMfma32x32x64_E4M3, FpsanMatchesScalarReference) {
  run_scale32_fpsan<fp8_e4m3, fp8_e4m3, 0, 0>();
}
TEST(ScaledMfma32x32x64_E5M2, LayoutMatchesHardware) {
  run_scale32_layout<fp8_e5m2, fp8_e5m2, 1, 1>();
}
TEST(ScaledMfma32x32x64_E5M2, FpsanMatchesScalarReference) {
  run_scale32_fpsan<fp8_e5m2, fp8_e5m2, 1, 1>();
}
TEST(ScaledMfma32x32x64_Mixed, LayoutMatchesHardware) {
  run_scale32_layout<fp8_e5m2, fp8_e4m3, 1, 0>(); // A=E5M2, B=E4M3
}
TEST(ScaledMfma32x32x64_Mixed, FpsanMatchesScalarReference) {
  run_scale32_fpsan<fp8_e5m2, fp8_e4m3, 1, 0>();
}

// ---------------------------------------------------------------------------
// Sub-byte (fp6 / bf6 / fp4) scaled f8f6f4 MFMA. CBSZ/BLGP select the format:
// 2=E2M3 (fp6), 3=E3M2 (bf6), 4=E2M1 (fp4). Per-lane operands are the raw
// packed v8i32 (Width-bit field s at bits Width*s..). A and B must share a bit
// width. Layout reverse-engineered + silicon-verified on MI350 and pinned by
// amdgcn_mfma.hpp.
//
//   * LayoutMatchesHardware (Float): pack the exact fp6/fp4 hardware codes,
//     run the builtin via the wrapper, compare to a numeric reference. Inputs
//     are small exact integers, so the accumulation is exact.
//   * FpsanMatchesScalarReference (FPSan): pack arbitrary Width-bit payloads,
//     run the FPSan dataflow, compare to a host reference that applies canonical
//     finite subbyte widening and accumulates in the payload ring --
//     isolating the gather + extraction layout (which the Float test pins).
// ---------------------------------------------------------------------------
namespace {
// fp6 (E2M3, code 2), bf6 (E3M2, code 3), fp4 (E2M1, code 4) decode.
constexpr int sub_width(int code) { return code <= 3 ? 6 : 4; }
template <int Format, Semantics S> Value<float, S, kCC> sub_canonical_value(unsigned field) {
  return fpsan_test::canonical_subbyte_widen<sub_width(Format), S, kCC>(field);
}
float sub_decode(int code, unsigned c) {
  if (code == 2) { // E2M3
    int s = (c >> 5) & 1, e = (c >> 3) & 3, m = c & 7;
    float v = e == 0 ? m / 8.0f : (1 + m / 8.0f) * std::ldexp(1.0f, e - 1);
    return s ? -v : v;
  }
  if (code == 3) { // E3M2
    int s = (c >> 5) & 1, e = (c >> 2) & 7, m = c & 3;
    float v =
        e == 0 ? (m / 4.0f) * std::ldexp(1.0f, 1 - 3) : (1 + m / 4.0f) * std::ldexp(1.0f, e - 3);
    return s ? -v : v;
  }
  int s = (c >> 3) & 1, e = (c >> 1) & 3, m = c & 1; // E2M1
  float v = e == 0 ? m * 0.5f : (1 + 0.5f * m) * std::ldexp(1.0f, e - 1);
  return s ? -v : v;
}
unsigned sub_encode(int code, int val) {
  int n = 1 << sub_width(code);
  for (int c = 0; c < n; ++c)
    if (sub_decode(code, static_cast<unsigned>(c)) == static_cast<float>(val))
      return static_cast<unsigned>(c);
  return 0u;
}
void set_field(fpsan::v8i32_native &reg, int bitoff, unsigned val, int nb) {
  for (int b = 0; b < nb; ++b) {
    int bo = bitoff + b, w = bo / 32, sh = bo % 32;
    unsigned u = static_cast<unsigned>(reg[w]);
    u = (u & ~(1u << sh)) | (((val >> b) & 1u) << sh);
    reg[w] = static_cast<int>(u);
  }
}
} // namespace

template <Semantics S, int CBSZ, int BLGP, class Out>
__global__ void k_scale_sub16(const fpsan::v8i32_native *A, const fpsan::v8i32_native *B,
                              const float *C, Out *D, int sa, int sb) {
  int lane = threadIdx.x;
  fpsan::v8i32_native an = A[lane], bn = B[lane];
  fpsan::v4f_native cn{};
  for (int reg = 0; reg < 4; ++reg) {
    int i = 4 * (lane / 16) + reg, j = lane % 16;
    cn[reg] = C[i * SN + j];
  }
  Value<fpsan::v4f_native, S, kCC> c{cn};
  auto d = fpsan::amdgcn_mfma_scale_f32_16x16x128_f8f6f4_sub<CBSZ, BLGP, 0, 0, S, kCC>(an, bn, c,
                                                                                       sa, sb);
  for (int reg = 0; reg < 4; ++reg) {
    int i = 4 * (lane / 16) + reg, j = lane % 16;
    if constexpr (S == Semantics::Native)
      D[i * SN + j] = d.get(reg).to_float();
    else
      D[i * SN + j] = d.get(reg).fpsan_payload();
  }
}

template <int CBSZ, int BLGP> void run_scale_sub16_layout() {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  const int wa = sub_width(CBSZ), wb = sub_width(BLGP);
  std::mt19937 rng = fpsan_test::make_rng();
  const int sa = 129, sb = 126;
  const float fa = fpsan::detail::e8m0_to_float(129), fb = fpsan::detail::e8m0_to_float(126);
  std::vector<int> Acode(SM * SK), Bcode(SK * SN);
  for (auto &x : Acode)
    x = static_cast<int>(sub_encode(CBSZ, fpsan_test::pick_int(rng, -2, 3)));
  for (auto &x : Bcode)
    x = static_cast<int>(sub_encode(BLGP, fpsan_test::pick_int(rng, -2, 3)));
  std::vector<float> Cm(SM * SN);
  for (auto &x : Cm)
    x = fpsan_test::pick_int_valued<float>(rng, -8, 8);
  std::vector<fpsan::v8i32_native> A(WAVE, fpsan::v8i32_native{}), B(WAVE, fpsan::v8i32_native{});
  for (int i = 0; i < SM; ++i)
    for (int k = 0; k < SK; ++k) {
      int lane = 16 * ((k % 64) / 16) + i, s = (k / 64) * 16 + (k % 16);
      set_field(A[lane], wa * s, static_cast<unsigned>(Acode[i * SK + k]), wa);
    }
  for (int k = 0; k < SK; ++k)
    for (int j = 0; j < SN; ++j) {
      int lane = 16 * ((k % 64) / 16) + j, s = (k / 64) * 16 + (k % 16);
      set_field(B[lane], wb * s, static_cast<unsigned>(Bcode[k * SN + j]), wb);
    }
  std::vector<float> ref(SM * SN);
  for (int i = 0; i < SM; ++i)
    for (int j = 0; j < SN; ++j) {
      double dot = 0;
      for (int k = 0; k < SK; ++k)
        dot += static_cast<double>(sub_decode(CBSZ, static_cast<unsigned>(Acode[i * SK + k]))) *
               static_cast<double>(sub_decode(BLGP, static_cast<unsigned>(Bcode[k * SN + j])));
      ref[i * SN + j] = static_cast<float>(static_cast<double>(Cm[i * SN + j]) +
                                           dot * static_cast<double>(fa) * static_cast<double>(fb));
    }
  fpsan::v8i32_native *dA, *dB;
  float *dC, *dD;
  HIP_CHECK(hipMalloc(&dA, WAVE * sizeof(fpsan::v8i32_native)));
  HIP_CHECK(hipMalloc(&dB, WAVE * sizeof(fpsan::v8i32_native)));
  HIP_CHECK(hipMalloc(&dC, SM * SN * sizeof(float)));
  HIP_CHECK(hipMalloc(&dD, SM * SN * sizeof(float)));
  HIP_CHECK(hipMemcpy(dA, A.data(), WAVE * sizeof(fpsan::v8i32_native), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, B.data(), WAVE * sizeof(fpsan::v8i32_native), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dC, Cm.data(), SM * SN * sizeof(float), hipMemcpyHostToDevice));
  k_scale_sub16<Semantics::Native, CBSZ, BLGP, float><<<1, WAVE>>>(dA, dB, dC, dD, sa, sb);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> got(SM * SN);
  HIP_CHECK(hipMemcpy(got.data(), dD, SM * SN * sizeof(float), hipMemcpyDeviceToHost));
  for (int i = 0; i < SM * SN; ++i)
    EXPECT_EQ(bits_of(got[i]), bits_of(ref[i])) << "at " << i;
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dD);
}

template <int CBSZ, int BLGP> void run_scale_sub16_fpsan() {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  const int wa = sub_width(CBSZ), wb = sub_width(BLGP);
  std::mt19937 rng = fpsan_test::make_rng();
  const int sa = 128, sb = 130;
  // Arbitrary Width-bit payloads (the realistic FPSan content of sub-byte data;
  // not necessarily valid format codes).
  std::vector<int> Ap(SM * SK), Bp(SK * SN);
  for (auto &x : Ap)
    x = static_cast<int>(rng() & ((1u << wa) - 1u));
  for (auto &x : Bp)
    x = static_cast<int>(rng() & ((1u << wb) - 1u));
  std::vector<float> Cm(SM * SN);
  for (auto &x : Cm)
    x = fpsan_test::pick_int_valued<float>(rng, -8, 8);
  std::vector<fpsan::v8i32_native> A(WAVE, fpsan::v8i32_native{}), B(WAVE, fpsan::v8i32_native{});
  for (int i = 0; i < SM; ++i)
    for (int k = 0; k < SK; ++k) {
      int lane = 16 * ((k % 64) / 16) + i, s = (k / 64) * 16 + (k % 16);
      set_field(A[lane], wa * s, static_cast<unsigned>(Ap[i * SK + k]), wa);
    }
  for (int k = 0; k < SK; ++k)
    for (int j = 0; j < SN; ++j) {
      int lane = 16 * ((k % 64) / 16) + j, s = (k / 64) * 16 + (k % 16);
      set_field(B[lane], wb * s, static_cast<unsigned>(Bp[k * SN + j]), wb);
    }
  fpsan::v8i32_native *dA, *dB;
  float *dC;
  HIP_CHECK(hipMalloc(&dA, WAVE * sizeof(fpsan::v8i32_native)));
  HIP_CHECK(hipMalloc(&dB, WAVE * sizeof(fpsan::v8i32_native)));
  HIP_CHECK(hipMalloc(&dC, SM * SN * sizeof(float)));
  HIP_CHECK(hipMemcpy(dA, A.data(), WAVE * sizeof(fpsan::v8i32_native), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, B.data(), WAVE * sizeof(fpsan::v8i32_native), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dC, Cm.data(), SM * SN * sizeof(float), hipMemcpyHostToDevice));
  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VF = Value<float, S, kCC>;
    const VF vsa(fpsan::detail::e8m0_to_float(128)), vsb(fpsan::detail::e8m0_to_float(130));
    std::vector<std::uint32_t> ref(SM * SN);
    for (int i = 0; i < SM; ++i)
      for (int j = 0; j < SN; ++j) {
        VF dot(0.0f);
        for (int k = 0; k < SK; ++k)
          dot = dot + sub_canonical_value<CBSZ, S>(static_cast<unsigned>(Ap[i * SK + k])) *
                          sub_canonical_value<BLGP, S>(static_cast<unsigned>(Bp[k * SN + j]));
        ref[i * SN + j] = (VF(Cm[i * SN + j]) + dot * vsa * vsb).fpsan_payload();
      }
    std::uint32_t *dD;
    HIP_CHECK(hipMalloc(&dD, SM * SN * sizeof(std::uint32_t)));
    k_scale_sub16<S, CBSZ, BLGP, std::uint32_t><<<1, WAVE>>>(dA, dB, dC, dD, sa, sb);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(SM * SN);
    HIP_CHECK(hipMemcpy(got.data(), dD, SM * SN * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for (int i = 0; i < SM * SN; ++i)
      EXPECT_EQ(got[i], ref[i]) << "at " << i;
    (void)hipFree(dD);
  });
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
}

TEST(ScaledMfma16x16x128_FP6, LayoutMatchesHardware) { run_scale_sub16_layout<2, 2>(); }
TEST(ScaledMfma16x16x128_FP6, FpsanMatchesScalarReference) { run_scale_sub16_fpsan<2, 2>(); }
TEST(ScaledMfma16x16x128_BF6, LayoutMatchesHardware) { run_scale_sub16_layout<3, 3>(); }
TEST(ScaledMfma16x16x128_BF6, FpsanMatchesScalarReference) { run_scale_sub16_fpsan<3, 3>(); }
TEST(ScaledMfma16x16x128_FP6BF6, LayoutMatchesHardware) { run_scale_sub16_layout<2, 3>(); }
TEST(ScaledMfma16x16x128_FP6BF6, FpsanMatchesScalarReference) { run_scale_sub16_fpsan<2, 3>(); }
TEST(ScaledMfma16x16x128_FP4, LayoutMatchesHardware) { run_scale_sub16_layout<4, 4>(); }
TEST(ScaledMfma16x16x128_FP4, FpsanMatchesScalarReference) { run_scale_sub16_fpsan<4, 4>(); }

template <Semantics S, int CBSZ, int BLGP, class Out>
__global__ void k_scale_sub32(const fpsan::v8i32_native *A, const fpsan::v8i32_native *B,
                              const float *C, Out *D, int sa, int sb) {
  int lane = threadIdx.x;
  fpsan::v8i32_native an = A[lane], bn = B[lane];
  fpsan::v16f_native cn{};
  for (int i = 0; i < S2M; ++i)
    for (int j = 0; j < S2N; ++j) {
      auto loc = fpsan::detail::output_loc_32(S2M, S2N, i, j, 0);
      if (loc.lane == lane)
        cn[loc.reg] = C[i * S2N + j];
    }
  Value<fpsan::v16f_native, S, kCC> c{cn};
  auto d =
      fpsan::amdgcn_mfma_scale_f32_32x32x64_f8f6f4_sub<CBSZ, BLGP, 0, 0, S, kCC>(an, bn, c, sa, sb);
  for (int i = 0; i < S2M; ++i)
    for (int j = 0; j < S2N; ++j) {
      auto loc = fpsan::detail::output_loc_32(S2M, S2N, i, j, 0);
      if (loc.lane == lane) {
        if constexpr (S == Semantics::Native)
          D[i * S2N + j] = d.get(loc.reg).to_float();
        else
          D[i * S2N + j] = d.get(loc.reg).fpsan_payload();
      }
    }
}

template <int CBSZ, int BLGP>
void pack_sub32(std::vector<fpsan::v8i32_native> &A, std::vector<fpsan::v8i32_native> &B,
                const std::vector<int> &Af, const std::vector<int> &Bf) {
  const int wa = sub_width(CBSZ), wb = sub_width(BLGP);
  for (int m = 0; m < S2M; ++m)
    for (int k = 0; k < S2K; ++k) {
      int lane = 16 * (2 * ((k % 32) / 16) + m / 16) + (m % 16), s = (k / 32) * 16 + (k % 16);
      set_field(A[lane], wa * s, static_cast<unsigned>(Af[m * S2K + k]), wa);
    }
  for (int k = 0; k < S2K; ++k)
    for (int n = 0; n < S2N; ++n) {
      int lane = 16 * (2 * ((k % 32) / 16) + n / 16) + (n % 16), s = (k / 32) * 16 + (k % 16);
      set_field(B[lane], wb * s, static_cast<unsigned>(Bf[k * S2N + n]), wb);
    }
}

template <int CBSZ, int BLGP> void run_scale_sub32_layout() {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  std::mt19937 rng = fpsan_test::make_rng();
  const int sa = 129, sb = 126;
  const float fa = fpsan::detail::e8m0_to_float(129), fb = fpsan::detail::e8m0_to_float(126);
  std::vector<int> Acode(S2M * S2K), Bcode(S2K * S2N);
  for (auto &x : Acode)
    x = static_cast<int>(sub_encode(CBSZ, fpsan_test::pick_int(rng, -2, 3)));
  for (auto &x : Bcode)
    x = static_cast<int>(sub_encode(BLGP, fpsan_test::pick_int(rng, -2, 3)));
  std::vector<float> Cm(S2M * S2N);
  for (auto &x : Cm)
    x = fpsan_test::pick_int_valued<float>(rng, -8, 8);
  std::vector<fpsan::v8i32_native> A(WAVE, fpsan::v8i32_native{}), B(WAVE, fpsan::v8i32_native{});
  pack_sub32<CBSZ, BLGP>(A, B, Acode, Bcode);
  std::vector<float> ref(S2M * S2N);
  for (int i = 0; i < S2M; ++i)
    for (int j = 0; j < S2N; ++j) {
      double dot = 0;
      for (int k = 0; k < S2K; ++k)
        dot += static_cast<double>(sub_decode(CBSZ, static_cast<unsigned>(Acode[i * S2K + k]))) *
               static_cast<double>(sub_decode(BLGP, static_cast<unsigned>(Bcode[k * S2N + j])));
      ref[i * S2N + j] =
          static_cast<float>(static_cast<double>(Cm[i * S2N + j]) +
                             dot * static_cast<double>(fa) * static_cast<double>(fb));
    }
  fpsan::v8i32_native *dA, *dB;
  float *dC, *dD;
  HIP_CHECK(hipMalloc(&dA, WAVE * sizeof(fpsan::v8i32_native)));
  HIP_CHECK(hipMalloc(&dB, WAVE * sizeof(fpsan::v8i32_native)));
  HIP_CHECK(hipMalloc(&dC, S2M * S2N * sizeof(float)));
  HIP_CHECK(hipMalloc(&dD, S2M * S2N * sizeof(float)));
  HIP_CHECK(hipMemcpy(dA, A.data(), WAVE * sizeof(fpsan::v8i32_native), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, B.data(), WAVE * sizeof(fpsan::v8i32_native), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dC, Cm.data(), S2M * S2N * sizeof(float), hipMemcpyHostToDevice));
  k_scale_sub32<Semantics::Native, CBSZ, BLGP, float><<<1, WAVE>>>(dA, dB, dC, dD, sa, sb);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> got(S2M * S2N);
  HIP_CHECK(hipMemcpy(got.data(), dD, S2M * S2N * sizeof(float), hipMemcpyDeviceToHost));
  for (int i = 0; i < S2M * S2N; ++i)
    EXPECT_EQ(bits_of(got[i]), bits_of(ref[i])) << "at " << i;
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dD);
}

template <int CBSZ, int BLGP> void run_scale_sub32_fpsan() {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  const int wa = sub_width(CBSZ), wb = sub_width(BLGP);
  std::mt19937 rng = fpsan_test::make_rng();
  const int sa = 128, sb = 130;
  std::vector<int> Ap(S2M * S2K), Bp(S2K * S2N);
  for (auto &x : Ap)
    x = static_cast<int>(rng() & ((1u << wa) - 1u));
  for (auto &x : Bp)
    x = static_cast<int>(rng() & ((1u << wb) - 1u));
  std::vector<float> Cm(S2M * S2N);
  for (auto &x : Cm)
    x = fpsan_test::pick_int_valued<float>(rng, -8, 8);
  std::vector<fpsan::v8i32_native> A(WAVE, fpsan::v8i32_native{}), B(WAVE, fpsan::v8i32_native{});
  pack_sub32<CBSZ, BLGP>(A, B, Ap, Bp);
  fpsan::v8i32_native *dA, *dB;
  float *dC;
  HIP_CHECK(hipMalloc(&dA, WAVE * sizeof(fpsan::v8i32_native)));
  HIP_CHECK(hipMalloc(&dB, WAVE * sizeof(fpsan::v8i32_native)));
  HIP_CHECK(hipMalloc(&dC, S2M * S2N * sizeof(float)));
  HIP_CHECK(hipMemcpy(dA, A.data(), WAVE * sizeof(fpsan::v8i32_native), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, B.data(), WAVE * sizeof(fpsan::v8i32_native), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dC, Cm.data(), S2M * S2N * sizeof(float), hipMemcpyHostToDevice));
  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VF = Value<float, S, kCC>;
    const VF vsa(fpsan::detail::e8m0_to_float(128)), vsb(fpsan::detail::e8m0_to_float(130));
    std::vector<std::uint32_t> ref(S2M * S2N);
    for (int i = 0; i < S2M; ++i)
      for (int j = 0; j < S2N; ++j) {
        VF dot(0.0f);
        for (int k = 0; k < S2K; ++k)
          dot = dot + sub_canonical_value<CBSZ, S>(static_cast<unsigned>(Ap[i * S2K + k])) *
                          sub_canonical_value<BLGP, S>(static_cast<unsigned>(Bp[k * S2N + j]));
        ref[i * S2N + j] = (VF(Cm[i * S2N + j]) + dot * vsa * vsb).fpsan_payload();
      }
    std::uint32_t *dD;
    HIP_CHECK(hipMalloc(&dD, S2M * S2N * sizeof(std::uint32_t)));
    k_scale_sub32<S, CBSZ, BLGP, std::uint32_t><<<1, WAVE>>>(dA, dB, dC, dD, sa, sb);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(S2M * S2N);
    HIP_CHECK(hipMemcpy(got.data(), dD, S2M * S2N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for (int i = 0; i < S2M * S2N; ++i)
      EXPECT_EQ(got[i], ref[i]) << "at " << i;
    (void)hipFree(dD);
  });
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
}

TEST(ScaledMfma32x32x64_FP6, LayoutMatchesHardware) { run_scale_sub32_layout<2, 2>(); }
TEST(ScaledMfma32x32x64_FP6, FpsanMatchesScalarReference) { run_scale_sub32_fpsan<2, 2>(); }
TEST(ScaledMfma32x32x64_BF6, LayoutMatchesHardware) { run_scale_sub32_layout<3, 3>(); }
TEST(ScaledMfma32x32x64_BF6, FpsanMatchesScalarReference) { run_scale_sub32_fpsan<3, 3>(); }
TEST(ScaledMfma32x32x64_FP4, LayoutMatchesHardware) { run_scale_sub32_layout<4, 4>(); }
TEST(ScaledMfma32x32x64_FP4, FpsanMatchesScalarReference) { run_scale_sub32_fpsan<4, 4>(); }

// ---------------------------------------------------------------------------
// sub x sub DIFFERENT-width (fp6 x fp4) -- now supported (all sub formats share
// one physical k order, so equal field indices pair the same k). Reuses the
// sub harnesses, which already track wa/wb separately. Silicon-verified for
// mixed-width sub x sub cases at both 16x16x128 and 32x32x64 shapes.
// ---------------------------------------------------------------------------
TEST(ScaledMfma16x16x128_FP6FP4, LayoutMatchesHardware) { run_scale_sub16_layout<2, 4>(); }
TEST(ScaledMfma16x16x128_FP6FP4, FpsanMatchesScalarReference) { run_scale_sub16_fpsan<2, 4>(); }
TEST(ScaledMfma16x16x128_FP4FP6, LayoutMatchesHardware) { run_scale_sub16_layout<4, 2>(); }
TEST(ScaledMfma16x16x128_FP4FP6, FpsanMatchesScalarReference) { run_scale_sub16_fpsan<4, 2>(); }
TEST(ScaledMfma16x16x128_BF6FP4, LayoutMatchesHardware) { run_scale_sub16_layout<3, 4>(); }
TEST(ScaledMfma16x16x128_BF6FP4, FpsanMatchesScalarReference) { run_scale_sub16_fpsan<3, 4>(); }
TEST(ScaledMfma32x32x64_FP6FP4, LayoutMatchesHardware) { run_scale_sub32_layout<2, 4>(); }
TEST(ScaledMfma32x32x64_FP6FP4, FpsanMatchesScalarReference) { run_scale_sub32_fpsan<2, 4>(); }
TEST(ScaledMfma32x32x64_FP4FP6, LayoutMatchesHardware) { run_scale_sub32_layout<4, 2>(); }
TEST(ScaledMfma32x32x64_FP4FP6, FpsanMatchesScalarReference) { run_scale_sub32_fpsan<4, 2>(); }

// ===========================================================================
// Per-K-block (NON-UNIFORM) scale. Each 32-element K-block carries its own
// E8M0 scale, read from lane (stride*kb + row) of the per-lane scale operand
// (stride = 16 at 16x16x128, 32 at 32x32x64). Silicon-RE'd + verified with
// full-block-random coverage.
// These tests give DIFFERENT scales to each K-block (the uniform tests above
// are the special case where every lane carries the same scale operand).
// ===========================================================================
namespace {
// Random E8M0 exponent bytes near 127 (so 2^(b-127) stays exact-ish, keeping
// the small-int matmul exact).
std::vector<int> rand_exps(std::mt19937 &rng, int n) {
  std::vector<int> e(n);
  for (auto &x : e)
    x = 124 + static_cast<int>(rng() % 7); // 2^-3 .. 2^+3
  return e;
}
} // namespace

// ---- fp8 16x16x128 per-block scale ----------------------------------------
template <Semantics S, class Out, class AElem, class BElem, int CBSZ, int BLGP>
__global__ void k_scale16_pb(const float *A, const float *B, const float *C, Out *D, const int *SA,
                             const int *SB) {
  using AFrag = fpsan::detail::v32_fragment<AElem>;
  using BFrag = fpsan::detail::v32_fragment<BElem>;
  int lane = threadIdx.x;
  AFrag an{};
  BFrag bn{};
  for (int half = 0; half < 2; ++half)
    for (int kk = 0; kk < 16; ++kk) {
      int g = lane / 16, idx = lane % 16;
      int k = half * 64 + g * 16 + kk, byte = half * 16 + kk;
      an[byte] = AElem(A[idx * SK + k]);
      bn[byte] = BElem(B[k * SN + idx]);
    }
  fpsan::v4f_native cn{};
  for (int reg = 0; reg < 4; ++reg) {
    int i = 4 * (lane / 16) + reg, j = lane % 16;
    cn[reg] = C[i * SN + j];
  }
  Value<AFrag, S, kCC> a{an};
  Value<BFrag, S, kCC> b{bn};
  Value<fpsan::v4f_native, S, kCC> c{cn};
  auto d = fpsan::amdgcn_mfma_scale_f32_16x16x128_f8f6f4<CBSZ, 0, BLGP, 0, 0, S, kCC>(
      a, b, c, SA[lane], SB[lane]);
  for (int reg = 0; reg < 4; ++reg) {
    int i = 4 * (lane / 16) + reg, j = lane % 16;
    if constexpr (S == Semantics::Native)
      D[i * SN + j] = d.get(reg).to_float();
    else
      D[i * SN + j] = d.get(reg).fpsan_payload();
  }
}

template <class AElem, class BElem, int CBSZ, int BLGP> void run_scale16_perblock() {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  const int NB = SK / 32; // 4 K-blocks
  ScaleMats m = make_scale_mats();
  std::mt19937 rng = fpsan_test::make_rng();
  std::vector<int> eA = rand_exps(rng, SM * NB), eB = rand_exps(rng, SN * NB);
  // Scale operands: byte0 at lane 16*kb+row carries the (row,block) exponent.
  std::vector<int> SA(WAVE, 0), SB(WAVE, 0);
  for (int kb = 0; kb < NB; ++kb)
    for (int i = 0; i < SM; ++i)
      SA[16 * kb + i] = eA[i * NB + kb];
  for (int kb = 0; kb < NB; ++kb)
    for (int j = 0; j < SN; ++j)
      SB[16 * kb + j] = eB[j * NB + kb];
  auto e8 = [](int b) { return fpsan::detail::e8m0_to_float(static_cast<unsigned>(b)); };
  float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C);
  int *dSA = to_dev(SA), *dSB = to_dev(SB);
  // ---- Float vs host reference (pins the hardware per-block scale model) ----
  {
    std::vector<float> ref(SM * SN);
    for (int i = 0; i < SM; ++i)
      for (int j = 0; j < SN; ++j) {
        double acc = m.C[i * SN + j];
        for (int kb = 0; kb < NB; ++kb) {
          double blk = 0;
          for (int k = 32 * kb; k < 32 * kb + 32; ++k)
            blk += static_cast<double>(m.A[i * SK + k]) * static_cast<double>(m.B[k * SN + j]);
          acc += blk * static_cast<double>(e8(eA[i * NB + kb])) *
                 static_cast<double>(e8(eB[j * NB + kb]));
        }
        ref[i * SN + j] = static_cast<float>(acc);
      }
    float *dD;
    HIP_CHECK(hipMalloc(&dD, SM * SN * sizeof(float)));
    k_scale16_pb<Semantics::Native, float, AElem, BElem, CBSZ, BLGP>
        <<<1, WAVE>>>(dA, dB, dC, dD, dSA, dSB);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> got(SM * SN);
    HIP_CHECK(hipMemcpy(got.data(), dD, SM * SN * sizeof(float), hipMemcpyDeviceToHost));
    for (int i = 0; i < SM * SN; ++i)
      EXPECT_EQ(bits_of(got[i]), bits_of(ref[i])) << "Float at " << i;
    (void)hipFree(dD);
  }
  // ---- FPSan-family semantics vs payload-ring reference ---------------------
  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VF = Value<float, S, kCC>;
    using VA = Value<AElem, S, kCC>;
    using VB = Value<BElem, S, kCC>;
    std::vector<std::uint32_t> ref(SM * SN);
    for (int i = 0; i < SM; ++i)
      for (int j = 0; j < SN; ++j) {
        VF acc(m.C[i * SN + j]);
        for (int kb = 0; kb < NB; ++kb) {
          VF blk(0.0f);
          for (int k = 32 * kb; k < 32 * kb + 32; ++k)
            blk = blk + fpsan::cast<float>(VA(AElem(m.A[i * SK + k]))) *
                            fpsan::cast<float>(VB(BElem(m.B[k * SN + j])));
          acc = acc + blk * VF(e8(eA[i * NB + kb])) * VF(e8(eB[j * NB + kb]));
        }
        ref[i * SN + j] = acc.fpsan_payload();
      }
    std::uint32_t *dD;
    HIP_CHECK(hipMalloc(&dD, SM * SN * sizeof(std::uint32_t)));
    k_scale16_pb<S, std::uint32_t, AElem, BElem, CBSZ, BLGP><<<1, WAVE>>>(dA, dB, dC, dD, dSA, dSB);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(SM * SN);
    HIP_CHECK(hipMemcpy(got.data(), dD, SM * SN * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for (int i = 0; i < SM * SN; ++i)
      EXPECT_EQ(got[i], ref[i]) << "FPSan at " << i;
    (void)hipFree(dD);
  });
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dSA);
  (void)hipFree(dSB);
}

TEST(ScaledMfma16x16x128_PerBlockScale, E4M3) { run_scale16_perblock<fp8_e4m3, fp8_e4m3, 0, 0>(); }
TEST(ScaledMfma16x16x128_PerBlockScale, E5M2) { run_scale16_perblock<fp8_e5m2, fp8_e5m2, 1, 1>(); }

// ---- fp8 32x32x64 per-block scale -----------------------------------------
template <Semantics S, class Out, class AElem, class BElem, int CBSZ, int BLGP>
__global__ void k_scale32_pb(const float *A, const float *B, const float *C, Out *D, const int *SA,
                             const int *SB) {
  using AFrag = fpsan::detail::v32_fragment<AElem>;
  using BFrag = fpsan::detail::v32_fragment<BElem>;
  int lane = threadIdx.x;
  int mn = (lane % 16) + 16 * ((lane / 16) & 1);
  AFrag an{};
  BFrag bn{};
  for (int b = 0; b < 32; ++b) {
    int k = (b / 16) * 32 + ((lane / 16) / 2) * 16 + (b % 16);
    an[b] = AElem(A[mn * S2K + k]);
    bn[b] = BElem(B[k * S2N + mn]);
  }
  fpsan::v16f_native cn{};
  for (int i = 0; i < S2M; ++i)
    for (int j = 0; j < S2N; ++j) {
      auto loc = fpsan::detail::output_loc_32(S2M, S2N, i, j, 0);
      if (loc.lane == lane)
        cn[loc.reg] = C[i * S2N + j];
    }
  Value<AFrag, S, kCC> a{an};
  Value<BFrag, S, kCC> b{bn};
  Value<fpsan::v16f_native, S, kCC> c{cn};
  auto d = fpsan::amdgcn_mfma_scale_f32_32x32x64_f8f6f4<CBSZ, 0, BLGP, 0, 0, S, kCC>(
      a, b, c, SA[lane], SB[lane]);
  for (int i = 0; i < S2M; ++i)
    for (int j = 0; j < S2N; ++j) {
      auto loc = fpsan::detail::output_loc_32(S2M, S2N, i, j, 0);
      if (loc.lane == lane) {
        if constexpr (S == Semantics::Native)
          D[i * S2N + j] = d.get(loc.reg).to_float();
        else
          D[i * S2N + j] = d.get(loc.reg).fpsan_payload();
      }
    }
}

template <class AElem, class BElem, int CBSZ, int BLGP> void run_scale32_perblock() {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  const int NB = S2K / 32; // 2 K-blocks
  ScaleMats m = make_scale_mats32();
  std::mt19937 rng = fpsan_test::make_rng();
  std::vector<int> eA = rand_exps(rng, S2M * NB), eB = rand_exps(rng, S2N * NB);
  std::vector<int> SA(WAVE, 0), SB(WAVE, 0);
  for (int kb = 0; kb < NB; ++kb)
    for (int mrow = 0; mrow < S2M; ++mrow)
      SA[32 * kb + mrow] = eA[mrow * NB + kb];
  for (int kb = 0; kb < NB; ++kb)
    for (int n = 0; n < S2N; ++n)
      SB[32 * kb + n] = eB[n * NB + kb];
  auto e8 = [](int b) { return fpsan::detail::e8m0_to_float(static_cast<unsigned>(b)); };
  float *dA = to_dev(m.A), *dB = to_dev(m.B), *dC = to_dev(m.C);
  int *dSA = to_dev(SA), *dSB = to_dev(SB);
  {
    std::vector<float> ref(S2M * S2N);
    for (int i = 0; i < S2M; ++i)
      for (int j = 0; j < S2N; ++j) {
        double acc = m.C[i * S2N + j];
        for (int kb = 0; kb < NB; ++kb) {
          double blk = 0;
          for (int k = 32 * kb; k < 32 * kb + 32; ++k)
            blk += static_cast<double>(m.A[i * S2K + k]) * static_cast<double>(m.B[k * S2N + j]);
          acc += blk * static_cast<double>(e8(eA[i * NB + kb])) *
                 static_cast<double>(e8(eB[j * NB + kb]));
        }
        ref[i * S2N + j] = static_cast<float>(acc);
      }
    float *dD;
    HIP_CHECK(hipMalloc(&dD, S2M * S2N * sizeof(float)));
    k_scale32_pb<Semantics::Native, float, AElem, BElem, CBSZ, BLGP>
        <<<1, WAVE>>>(dA, dB, dC, dD, dSA, dSB);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> got(S2M * S2N);
    HIP_CHECK(hipMemcpy(got.data(), dD, S2M * S2N * sizeof(float), hipMemcpyDeviceToHost));
    for (int i = 0; i < S2M * S2N; ++i)
      EXPECT_EQ(bits_of(got[i]), bits_of(ref[i])) << "Float at " << i;
    (void)hipFree(dD);
  }
  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VF = Value<float, S, kCC>;
    using VA = Value<AElem, S, kCC>;
    using VB = Value<BElem, S, kCC>;
    std::vector<std::uint32_t> ref(S2M * S2N);
    for (int i = 0; i < S2M; ++i)
      for (int j = 0; j < S2N; ++j) {
        VF acc(m.C[i * S2N + j]);
        for (int kb = 0; kb < NB; ++kb) {
          VF blk(0.0f);
          for (int k = 32 * kb; k < 32 * kb + 32; ++k)
            blk = blk + fpsan::cast<float>(VA(AElem(m.A[i * S2K + k]))) *
                            fpsan::cast<float>(VB(BElem(m.B[k * S2N + j])));
          acc = acc + blk * VF(e8(eA[i * NB + kb])) * VF(e8(eB[j * NB + kb]));
        }
        ref[i * S2N + j] = acc.fpsan_payload();
      }
    std::uint32_t *dD;
    HIP_CHECK(hipMalloc(&dD, S2M * S2N * sizeof(std::uint32_t)));
    k_scale32_pb<S, std::uint32_t, AElem, BElem, CBSZ, BLGP><<<1, WAVE>>>(dA, dB, dC, dD, dSA, dSB);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(S2M * S2N);
    HIP_CHECK(hipMemcpy(got.data(), dD, S2M * S2N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for (int i = 0; i < S2M * S2N; ++i)
      EXPECT_EQ(got[i], ref[i]) << "FPSan at " << i;
    (void)hipFree(dD);
  });
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dSA);
  (void)hipFree(dSB);
}

TEST(ScaledMfma32x32x64_PerBlockScale, E4M3) { run_scale32_perblock<fp8_e4m3, fp8_e4m3, 0, 0>(); }

// ---- sub fp4 16x16x128 per-block scale (confirms the sub path reads the same
// per-block scale lanes) ----------------------------------------------------
template <Semantics S, int CBSZ, int BLGP, class Out>
__global__ void k_scale_sub16_pb(const fpsan::v8i32_native *A, const fpsan::v8i32_native *B,
                                 const float *C, Out *D, const int *SA, const int *SB) {
  int lane = threadIdx.x;
  fpsan::v8i32_native an = A[lane], bn = B[lane];
  fpsan::v4f_native cn{};
  for (int reg = 0; reg < 4; ++reg) {
    int i = 4 * (lane / 16) + reg, j = lane % 16;
    cn[reg] = C[i * SN + j];
  }
  Value<fpsan::v4f_native, S, kCC> c{cn};
  auto d = fpsan::amdgcn_mfma_scale_f32_16x16x128_f8f6f4_sub<CBSZ, BLGP, 0, 0, S, kCC>(
      an, bn, c, SA[lane], SB[lane]);
  for (int reg = 0; reg < 4; ++reg) {
    int i = 4 * (lane / 16) + reg, j = lane % 16;
    if constexpr (S == Semantics::Native)
      D[i * SN + j] = d.get(reg).to_float();
    else
      D[i * SN + j] = d.get(reg).fpsan_payload();
  }
}

template <int CBSZ, int BLGP> void run_scale_sub16_perblock_fpsan() {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  const int wa = sub_width(CBSZ), wb = sub_width(BLGP), NB = SK / 32;
  std::mt19937 rng = fpsan_test::make_rng();
  std::vector<int> Ap(SM * SK), Bp(SK * SN);
  for (auto &x : Ap)
    x = static_cast<int>(rng() & ((1u << wa) - 1u));
  for (auto &x : Bp)
    x = static_cast<int>(rng() & ((1u << wb) - 1u));
  std::vector<float> Cm(SM * SN);
  for (auto &x : Cm)
    x = fpsan_test::pick_int_valued<float>(rng, -8, 8);
  std::vector<int> eA = rand_exps(rng, SM * NB), eB = rand_exps(rng, SN * NB);
  std::vector<int> SA(WAVE, 0), SB(WAVE, 0);
  for (int kb = 0; kb < NB; ++kb)
    for (int i = 0; i < SM; ++i)
      SA[16 * kb + i] = eA[i * NB + kb];
  for (int kb = 0; kb < NB; ++kb)
    for (int j = 0; j < SN; ++j)
      SB[16 * kb + j] = eB[j * NB + kb];
  std::vector<fpsan::v8i32_native> A(WAVE, fpsan::v8i32_native{}), B(WAVE, fpsan::v8i32_native{});
  for (int i = 0; i < SM; ++i)
    for (int k = 0; k < SK; ++k) {
      int lane = 16 * ((k % 64) / 16) + i, s = (k / 64) * 16 + (k % 16);
      set_field(A[lane], wa * s, static_cast<unsigned>(Ap[i * SK + k]), wa);
    }
  for (int k = 0; k < SK; ++k)
    for (int j = 0; j < SN; ++j) {
      int lane = 16 * ((k % 64) / 16) + j, s = (k / 64) * 16 + (k % 16);
      set_field(B[lane], wb * s, static_cast<unsigned>(Bp[k * SN + j]), wb);
    }
  auto e8 = [](int b) { return fpsan::detail::e8m0_to_float(static_cast<unsigned>(b)); };
  fpsan::v8i32_native *dA, *dB;
  float *dC;
  HIP_CHECK(hipMalloc(&dA, WAVE * sizeof(fpsan::v8i32_native)));
  HIP_CHECK(hipMalloc(&dB, WAVE * sizeof(fpsan::v8i32_native)));
  HIP_CHECK(hipMalloc(&dC, SM * SN * sizeof(float)));
  HIP_CHECK(hipMemcpy(dA, A.data(), WAVE * sizeof(fpsan::v8i32_native), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, B.data(), WAVE * sizeof(fpsan::v8i32_native), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dC, Cm.data(), SM * SN * sizeof(float), hipMemcpyHostToDevice));
  int *dSA = to_dev(SA), *dSB = to_dev(SB);
  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VF = Value<float, S, kCC>;
    std::vector<std::uint32_t> ref(SM * SN);
    for (int i = 0; i < SM; ++i)
      for (int j = 0; j < SN; ++j) {
        VF acc(Cm[i * SN + j]);
        for (int kb = 0; kb < NB; ++kb) {
          VF blk(0.0f);
          for (int k = 32 * kb; k < 32 * kb + 32; ++k)
            blk = blk + sub_canonical_value<CBSZ, S>(static_cast<unsigned>(Ap[i * SK + k])) *
                            sub_canonical_value<BLGP, S>(static_cast<unsigned>(Bp[k * SN + j]));
          acc = acc + blk * VF(e8(eA[i * NB + kb])) * VF(e8(eB[j * NB + kb]));
        }
        ref[i * SN + j] = acc.fpsan_payload();
      }
    std::uint32_t *dD;
    HIP_CHECK(hipMalloc(&dD, SM * SN * sizeof(std::uint32_t)));
    k_scale_sub16_pb<S, CBSZ, BLGP, std::uint32_t><<<1, WAVE>>>(dA, dB, dC, dD, dSA, dSB);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(SM * SN);
    HIP_CHECK(hipMemcpy(got.data(), dD, SM * SN * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for (int i = 0; i < SM * SN; ++i)
      EXPECT_EQ(got[i], ref[i]) << "at " << i;
    (void)hipFree(dD);
  });
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dSA);
  (void)hipFree(dSB);
}

TEST(ScaledMfma16x16x128_PerBlockScale, FP4Sub) { run_scale_sub16_perblock_fpsan<4, 4>(); }

// ===========================================================================
// Mixed-WIDTH 8-bit x sub-byte (fp8 x fp4 / fp6, and the mirror). The sub
// operand follows the silicon-RE'd "mix model" that pairs it with fp8
// (different physical k order than fp8). Uniform scale here isolates the mixed
// LAYOUT (per-block scale is covered above). Proven with full-block-random
// coverage.
// ===========================================================================
namespace {
// 16x16x128 sub mix-model physical slot for logical k and output index.
void sub_mix_loc16(int k, int idx, int &lane, int &slot) {
  int q0 = 2 * ((k >> 6) & 1) + ((k >> 5) & 1);
  lane = 16 * q0 + idx;
  slot = 16 * ((k >> 4) & 1) + (k & 15);
}
// 32x32x64 sub mix-model physical slot.
void sub_mix_loc32(int k, int idx, int &lane, int &slot) {
  lane = 16 * (2 * (k / 32) + idx / 16) + (idx % 16);
  slot = 16 * ((k % 32) / 16) + (k % 16);
}
} // namespace

// A = fp8 (built from float matrix), B = sub-byte (raw v8i32, mix-packed).
template <Semantics S, class Out, class AElem, int CBSZ, int BLGP>
__global__ void k_scale16_mix_a8(const float *A, const fpsan::v8i32_native *B, const float *C,
                                 Out *D, int sa, int sb) {
  using AFrag = fpsan::detail::v32_fragment<AElem>;
  int lane = threadIdx.x;
  AFrag an{};
  for (int half = 0; half < 2; ++half)
    for (int kk = 0; kk < 16; ++kk) {
      int g = lane / 16, idx = lane % 16;
      int k = half * 64 + g * 16 + kk, byte = half * 16 + kk;
      an[byte] = AElem(A[idx * SK + k]);
    }
  fpsan::v8i32_native bn = B[lane];
  fpsan::v4f_native cn{};
  for (int reg = 0; reg < 4; ++reg) {
    int i = 4 * (lane / 16) + reg, j = lane % 16;
    cn[reg] = C[i * SN + j];
  }
  Value<AFrag, S, kCC> a{an};
  Value<fpsan::v4f_native, S, kCC> c{cn};
  auto d = fpsan::amdgcn_mfma_scale_f32_16x16x128_f8f6f4_mixed_a8<CBSZ, BLGP, 0, 0, S, kCC, AFrag>(
      a, bn, c, sa, sb);
  for (int reg = 0; reg < 4; ++reg) {
    int i = 4 * (lane / 16) + reg, j = lane % 16;
    if constexpr (S == Semantics::Native)
      D[i * SN + j] = d.get(reg).to_float();
    else
      D[i * SN + j] = d.get(reg).fpsan_payload();
  }
}

// A = sub-byte (raw v8i32, mix-packed), B = fp8.
template <Semantics S, class Out, class BElem, int CBSZ, int BLGP>
__global__ void k_scale16_mix_b8(const fpsan::v8i32_native *A, const float *B, const float *C,
                                 Out *D, int sa, int sb) {
  using BFrag = fpsan::detail::v32_fragment<BElem>;
  int lane = threadIdx.x;
  fpsan::v8i32_native an = A[lane];
  BFrag bn{};
  for (int half = 0; half < 2; ++half)
    for (int kk = 0; kk < 16; ++kk) {
      int g = lane / 16, idx = lane % 16;
      int k = half * 64 + g * 16 + kk, byte = half * 16 + kk;
      bn[byte] = BElem(B[k * SN + idx]);
    }
  fpsan::v4f_native cn{};
  for (int reg = 0; reg < 4; ++reg) {
    int i = 4 * (lane / 16) + reg, j = lane % 16;
    cn[reg] = C[i * SN + j];
  }
  Value<BFrag, S, kCC> b{bn};
  Value<fpsan::v4f_native, S, kCC> c{cn};
  auto d = fpsan::amdgcn_mfma_scale_f32_16x16x128_f8f6f4_mixed_b8<CBSZ, BLGP, 0, 0, S, kCC, BFrag>(
      an, b, c, sa, sb);
  for (int reg = 0; reg < 4; ++reg) {
    int i = 4 * (lane / 16) + reg, j = lane % 16;
    if constexpr (S == Semantics::Native)
      D[i * SN + j] = d.get(reg).to_float();
    else
      D[i * SN + j] = d.get(reg).fpsan_payload();
  }
}

// AIsSub=false: A fp8, B sub. AIsSub=true: A sub, B fp8.
template <bool AIsSub, class Fp8Elem, int CBSZ, int BLGP> void run_scale16_mixed() {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  const int subfmt = AIsSub ? CBSZ : BLGP, w = sub_width(subfmt);
  std::mt19937 rng = fpsan_test::make_rng();
  const int sa = 129, sb = 126;
  const float fa = fpsan::detail::e8m0_to_float(129), fb = fpsan::detail::e8m0_to_float(126);
  // fp8 matrix (small ints) + sub code matrix (small ints, exact in the format).
  std::vector<float> Afp8, Bfp8, Cm(SM * SN);
  std::vector<int> subcode(SM * SK); // [i][k] if A sub, else [k][j] reused
  for (auto &x : Cm)
    x = fpsan_test::pick_int_valued<float>(rng, -6, 6);
  // We always lay out A as [i][k] and B as [k][j] logically.
  std::vector<float> fp8mat;
  std::vector<int> submat;
  fp8mat.resize(AIsSub ? SK * SN : SM * SK);
  submat.resize(AIsSub ? SM * SK : SK * SN);
  for (auto &x : fp8mat)
    x = fpsan_test::pick_int_valued<float>(rng, -3, 3);
  for (auto &x : submat)
    x = static_cast<int>(sub_encode(subfmt, fpsan_test::pick_int(rng, -2, 3)));
  // Pack the sub operand in mix-model.
  std::vector<fpsan::v8i32_native> sub(WAVE, fpsan::v8i32_native{});
  if (AIsSub) {
    for (int i = 0; i < SM; ++i)
      for (int k = 0; k < SK; ++k) {
        int lane, slot;
        sub_mix_loc16(k, i, lane, slot);
        set_field(sub[lane], w * slot, static_cast<unsigned>(submat[i * SK + k]), w);
      }
  } else {
    for (int k = 0; k < SK; ++k)
      for (int j = 0; j < SN; ++j) {
        int lane, slot;
        sub_mix_loc16(k, j, lane, slot);
        set_field(sub[lane], w * slot, static_cast<unsigned>(submat[k * SN + j]), w);
      }
  }
  // Host reference (small ints exact): D = C + (sum_k A*B) * fa * fb.
  auto Aval = [&](int i, int k) {
    return AIsSub
               ? static_cast<double>(sub_decode(subfmt, static_cast<unsigned>(submat[i * SK + k])))
               : static_cast<double>(fp8mat[i * SK + k]);
  };
  auto Bval = [&](int k, int j) {
    return AIsSub
               ? static_cast<double>(fp8mat[k * SN + j])
               : static_cast<double>(sub_decode(subfmt, static_cast<unsigned>(submat[k * SN + j])));
  };
  std::vector<float> ref(SM * SN);
  for (int i = 0; i < SM; ++i)
    for (int j = 0; j < SN; ++j) {
      double dot = 0;
      for (int k = 0; k < SK; ++k)
        dot += Aval(i, k) * Bval(k, j);
      ref[i * SN + j] = static_cast<float>(static_cast<double>(Cm[i * SN + j]) + dot * fa * fb);
    }
  float *dF = to_dev(fp8mat), *dC = to_dev(Cm), *dD;
  fpsan::v8i32_native *dSub;
  HIP_CHECK(hipMalloc(&dSub, WAVE * sizeof(fpsan::v8i32_native)));
  HIP_CHECK(hipMemcpy(dSub, sub.data(), WAVE * sizeof(fpsan::v8i32_native), hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&dD, SM * SN * sizeof(float)));
  if constexpr (AIsSub)
    k_scale16_mix_b8<Semantics::Native, float, Fp8Elem, CBSZ, BLGP>
        <<<1, WAVE>>>(dSub, dF, dC, dD, sa, sb);
  else
    k_scale16_mix_a8<Semantics::Native, float, Fp8Elem, CBSZ, BLGP>
        <<<1, WAVE>>>(dF, dSub, dC, dD, sa, sb);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> got(SM * SN);
  HIP_CHECK(hipMemcpy(got.data(), dD, SM * SN * sizeof(float), hipMemcpyDeviceToHost));
  for (int i = 0; i < SM * SN; ++i)
    EXPECT_EQ(bits_of(got[i]), bits_of(ref[i])) << "at " << i;
  (void)hipFree(dF);
  (void)hipFree(dC);
  (void)hipFree(dD);
  (void)hipFree(dSub);
}

TEST(ScaledMfma16x16x128_Mixed8xSub, FP8xFP4_Layout) { run_scale16_mixed<false, fp8_e4m3, 0, 4>(); }
TEST(ScaledMfma16x16x128_Mixed8xSub, FP8xFP6_Layout) { run_scale16_mixed<false, fp8_e4m3, 0, 2>(); }
TEST(ScaledMfma16x16x128_Mixed8xSub, BF8xFP4_Layout) { run_scale16_mixed<false, fp8_e5m2, 1, 4>(); }
TEST(ScaledMfma16x16x128_Mixed8xSub, FP4xFP8_Layout) { run_scale16_mixed<true, fp8_e4m3, 4, 0>(); }
TEST(ScaledMfma16x16x128_Mixed8xSub, FP6xFP8_Layout) { run_scale16_mixed<true, fp8_e4m3, 2, 0>(); }

// FPSan-mode mixed 8 x sub: fp8 operand carries real fp8 values (cast widening);
// sub operand carries finite Width-bit subbyte codes (canonical widening).
template <bool AIsSub, class Fp8Elem, int CBSZ, int BLGP> void run_scale16_mixed_fpsan() {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  constexpr int SubFmt = AIsSub ? CBSZ : BLGP;
  const int w = sub_width(SubFmt);
  std::mt19937 rng = fpsan_test::make_rng();
  const int sa = 128, sb = 130;
  std::vector<float> fp8mat(AIsSub ? SK * SN : SM * SK), Cm(SM * SN);
  std::vector<int> subpay(AIsSub ? SM * SK : SK * SN);
  for (auto &x : fp8mat)
    x = fpsan_test::pick_int_valued<float>(rng, -3, 3);
  for (auto &x : subpay)
    x = static_cast<int>(rng() & ((1u << w) - 1u));
  for (auto &x : Cm)
    x = fpsan_test::pick_int_valued<float>(rng, -6, 6);
  std::vector<fpsan::v8i32_native> sub(WAVE, fpsan::v8i32_native{});
  if (AIsSub)
    for (int i = 0; i < SM; ++i)
      for (int k = 0; k < SK; ++k) {
        int lane, slot;
        sub_mix_loc16(k, i, lane, slot);
        set_field(sub[lane], w * slot, static_cast<unsigned>(subpay[i * SK + k]), w);
      }
  else
    for (int k = 0; k < SK; ++k)
      for (int j = 0; j < SN; ++j) {
        int lane, slot;
        sub_mix_loc16(k, j, lane, slot);
        set_field(sub[lane], w * slot, static_cast<unsigned>(subpay[k * SN + j]), w);
      }
  float *dF = to_dev(fp8mat), *dC = to_dev(Cm);
  fpsan::v8i32_native *dSub;
  HIP_CHECK(hipMalloc(&dSub, WAVE * sizeof(fpsan::v8i32_native)));
  HIP_CHECK(hipMemcpy(dSub, sub.data(), WAVE * sizeof(fpsan::v8i32_native), hipMemcpyHostToDevice));
  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VF = Value<float, S, kCC>;
    using VFp8 = Value<Fp8Elem, S, kCC>;
    const VF vsa(fpsan::detail::e8m0_to_float(128)), vsb(fpsan::detail::e8m0_to_float(130));
    auto Av = [&](int i, int k) {
      return AIsSub ? sub_canonical_value<SubFmt, S>(static_cast<unsigned>(subpay[i * SK + k]))
                    : fpsan::cast<float>(VFp8(Fp8Elem(fp8mat[i * SK + k])));
    };
    auto Bv = [&](int k, int j) {
      return AIsSub ? fpsan::cast<float>(VFp8(Fp8Elem(fp8mat[k * SN + j])))
                    : sub_canonical_value<SubFmt, S>(static_cast<unsigned>(subpay[k * SN + j]));
    };
    std::vector<std::uint32_t> ref(SM * SN);
    for (int i = 0; i < SM; ++i)
      for (int j = 0; j < SN; ++j) {
        VF dot(0.0f);
        for (int k = 0; k < SK; ++k)
          dot = dot + Av(i, k) * Bv(k, j);
        ref[i * SN + j] = (VF(Cm[i * SN + j]) + dot * vsa * vsb).fpsan_payload();
      }
    std::uint32_t *dD;
    HIP_CHECK(hipMalloc(&dD, SM * SN * sizeof(std::uint32_t)));
    if constexpr (AIsSub)
      k_scale16_mix_b8<S, std::uint32_t, Fp8Elem, CBSZ, BLGP>
          <<<1, WAVE>>>(dSub, dF, dC, dD, sa, sb);
    else
      k_scale16_mix_a8<S, std::uint32_t, Fp8Elem, CBSZ, BLGP>
          <<<1, WAVE>>>(dF, dSub, dC, dD, sa, sb);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(SM * SN);
    HIP_CHECK(hipMemcpy(got.data(), dD, SM * SN * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for (int i = 0; i < SM * SN; ++i)
      EXPECT_EQ(got[i], ref[i]) << "at " << i;
    (void)hipFree(dD);
  });
  (void)hipFree(dF);
  (void)hipFree(dC);
  (void)hipFree(dSub);
}

TEST(ScaledMfma16x16x128_Mixed8xSub, FP8xFP4_Fpsan) {
  run_scale16_mixed_fpsan<false, fp8_e4m3, 0, 4>();
}
TEST(ScaledMfma16x16x128_Mixed8xSub, FP4xFP8_Fpsan) {
  run_scale16_mixed_fpsan<true, fp8_e4m3, 4, 0>();
}
TEST(ScaledMfma16x16x128_Mixed8xSub, FP8xFP6_Fpsan) {
  run_scale16_mixed_fpsan<false, fp8_e4m3, 0, 2>();
}
TEST(ScaledMfma16x16x128_Mixed8xSub, BF8xFP4_Fpsan) {
  run_scale16_mixed_fpsan<false, fp8_e5m2, 1, 4>();
}

// ---- 32x32x64 mixed 8 x sub (Float layout) --------------------------------
template <Semantics S, class Out, class AElem, int CBSZ, int BLGP>
__global__ void k_scale32_mix_a8(const float *A, const fpsan::v8i32_native *B, const float *C,
                                 Out *D, int sa, int sb) {
  using AFrag = fpsan::detail::v32_fragment<AElem>;
  int lane = threadIdx.x;
  int mn = (lane % 16) + 16 * ((lane / 16) & 1);
  AFrag an{};
  for (int b = 0; b < 32; ++b) {
    int k = (b / 16) * 32 + ((lane / 16) / 2) * 16 + (b % 16);
    an[b] = AElem(A[mn * S2K + k]);
  }
  fpsan::v8i32_native bn = B[lane];
  fpsan::v16f_native cn{};
  for (int i = 0; i < S2M; ++i)
    for (int j = 0; j < S2N; ++j) {
      auto loc = fpsan::detail::output_loc_32(S2M, S2N, i, j, 0);
      if (loc.lane == lane)
        cn[loc.reg] = C[i * S2N + j];
    }
  Value<AFrag, S, kCC> a{an};
  Value<fpsan::v16f_native, S, kCC> c{cn};
  auto d = fpsan::amdgcn_mfma_scale_f32_32x32x64_f8f6f4_mixed_a8<CBSZ, BLGP, 0, 0, S, kCC, AFrag>(
      a, bn, c, sa, sb);
  for (int i = 0; i < S2M; ++i)
    for (int j = 0; j < S2N; ++j) {
      auto loc = fpsan::detail::output_loc_32(S2M, S2N, i, j, 0);
      if (loc.lane == lane) {
        if constexpr (S == Semantics::Native)
          D[i * S2N + j] = d.get(loc.reg).to_float();
        else
          D[i * S2N + j] = d.get(loc.reg).fpsan_payload();
      }
    }
}

template <Semantics S, class Out, class BElem, int CBSZ, int BLGP>
__global__ void k_scale32_mix_b8(const fpsan::v8i32_native *A, const float *B, const float *C,
                                 Out *D, int sa, int sb) {
  using BFrag = fpsan::detail::v32_fragment<BElem>;
  int lane = threadIdx.x;
  int mn = (lane % 16) + 16 * ((lane / 16) & 1);
  BFrag bn{};
  for (int b = 0; b < 32; ++b) {
    int k = (b / 16) * 32 + ((lane / 16) / 2) * 16 + (b % 16);
    bn[b] = BElem(B[k * S2N + mn]);
  }
  fpsan::v8i32_native an = A[lane];
  fpsan::v16f_native cn{};
  for (int i = 0; i < S2M; ++i)
    for (int j = 0; j < S2N; ++j) {
      auto loc = fpsan::detail::output_loc_32(S2M, S2N, i, j, 0);
      if (loc.lane == lane)
        cn[loc.reg] = C[i * S2N + j];
    }
  Value<BFrag, S, kCC> b{bn};
  Value<fpsan::v16f_native, S, kCC> c{cn};
  auto d = fpsan::amdgcn_mfma_scale_f32_32x32x64_f8f6f4_mixed_b8<CBSZ, BLGP, 0, 0, S, kCC, BFrag>(
      an, b, c, sa, sb);
  for (int i = 0; i < S2M; ++i)
    for (int j = 0; j < S2N; ++j) {
      auto loc = fpsan::detail::output_loc_32(S2M, S2N, i, j, 0);
      if (loc.lane == lane) {
        if constexpr (S == Semantics::Native)
          D[i * S2N + j] = d.get(loc.reg).to_float();
        else
          D[i * S2N + j] = d.get(loc.reg).fpsan_payload();
      }
    }
}

template <bool AIsSub, class Fp8Elem, int CBSZ, int BLGP> void run_scale32_mixed() {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  const int subfmt = AIsSub ? CBSZ : BLGP, w = sub_width(subfmt);
  std::mt19937 rng = fpsan_test::make_rng();
  const int sa = 129, sb = 126;
  const float fa = fpsan::detail::e8m0_to_float(129), fb = fpsan::detail::e8m0_to_float(126);
  std::vector<float> fp8mat(AIsSub ? S2K * S2N : S2M * S2K), Cm(S2M * S2N);
  std::vector<int> submat(AIsSub ? S2M * S2K : S2K * S2N);
  for (auto &x : fp8mat)
    x = fpsan_test::pick_int_valued<float>(rng, -3, 3);
  for (auto &x : submat)
    x = static_cast<int>(sub_encode(subfmt, fpsan_test::pick_int(rng, -2, 3)));
  for (auto &x : Cm)
    x = fpsan_test::pick_int_valued<float>(rng, -6, 6);
  std::vector<fpsan::v8i32_native> sub(WAVE, fpsan::v8i32_native{});
  if (AIsSub)
    for (int i = 0; i < S2M; ++i)
      for (int k = 0; k < S2K; ++k) {
        int lane, slot;
        sub_mix_loc32(k, i, lane, slot);
        set_field(sub[lane], w * slot, static_cast<unsigned>(submat[i * S2K + k]), w);
      }
  else
    for (int k = 0; k < S2K; ++k)
      for (int j = 0; j < S2N; ++j) {
        int lane, slot;
        sub_mix_loc32(k, j, lane, slot);
        set_field(sub[lane], w * slot, static_cast<unsigned>(submat[k * S2N + j]), w);
      }
  auto Aval = [&](int i, int k) {
    return AIsSub
               ? static_cast<double>(sub_decode(subfmt, static_cast<unsigned>(submat[i * S2K + k])))
               : static_cast<double>(fp8mat[i * S2K + k]);
  };
  auto Bval = [&](int k, int j) {
    return AIsSub ? static_cast<double>(fp8mat[k * S2N + j])
                  : static_cast<double>(
                        sub_decode(subfmt, static_cast<unsigned>(submat[k * S2N + j])));
  };
  std::vector<float> ref(S2M * S2N);
  for (int i = 0; i < S2M; ++i)
    for (int j = 0; j < S2N; ++j) {
      double dot = 0;
      for (int k = 0; k < S2K; ++k)
        dot += Aval(i, k) * Bval(k, j);
      ref[i * S2N + j] = static_cast<float>(static_cast<double>(Cm[i * S2N + j]) + dot * fa * fb);
    }
  float *dF = to_dev(fp8mat), *dC = to_dev(Cm), *dD;
  fpsan::v8i32_native *dSub;
  HIP_CHECK(hipMalloc(&dSub, WAVE * sizeof(fpsan::v8i32_native)));
  HIP_CHECK(hipMemcpy(dSub, sub.data(), WAVE * sizeof(fpsan::v8i32_native), hipMemcpyHostToDevice));
  HIP_CHECK(hipMalloc(&dD, S2M * S2N * sizeof(float)));
  if constexpr (AIsSub)
    k_scale32_mix_b8<Semantics::Native, float, Fp8Elem, CBSZ, BLGP>
        <<<1, WAVE>>>(dSub, dF, dC, dD, sa, sb);
  else
    k_scale32_mix_a8<Semantics::Native, float, Fp8Elem, CBSZ, BLGP>
        <<<1, WAVE>>>(dF, dSub, dC, dD, sa, sb);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> got(S2M * S2N);
  HIP_CHECK(hipMemcpy(got.data(), dD, S2M * S2N * sizeof(float), hipMemcpyDeviceToHost));
  for (int i = 0; i < S2M * S2N; ++i)
    EXPECT_EQ(bits_of(got[i]), bits_of(ref[i])) << "at " << i;
  (void)hipFree(dF);
  (void)hipFree(dC);
  (void)hipFree(dD);
  (void)hipFree(dSub);
}

template <bool AIsSub, class Fp8Elem, int CBSZ, int BLGP> void run_scale32_mixed_fpsan() {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  constexpr int SubFmt = AIsSub ? CBSZ : BLGP;
  const int w = sub_width(SubFmt);
  std::mt19937 rng = fpsan_test::make_rng();
  const int sa = 128, sb = 130;
  std::vector<float> fp8mat(AIsSub ? S2K * S2N : S2M * S2K), Cm(S2M * S2N);
  std::vector<int> subpay(AIsSub ? S2M * S2K : S2K * S2N);
  for (auto &x : fp8mat)
    x = fpsan_test::pick_int_valued<float>(rng, -3, 3);
  for (auto &x : subpay)
    x = static_cast<int>(rng() & ((1u << w) - 1u));
  for (auto &x : Cm)
    x = fpsan_test::pick_int_valued<float>(rng, -6, 6);
  std::vector<fpsan::v8i32_native> sub(WAVE, fpsan::v8i32_native{});
  if (AIsSub)
    for (int i = 0; i < S2M; ++i)
      for (int k = 0; k < S2K; ++k) {
        int lane, slot;
        sub_mix_loc32(k, i, lane, slot);
        set_field(sub[lane], w * slot, static_cast<unsigned>(subpay[i * S2K + k]), w);
      }
  else
    for (int k = 0; k < S2K; ++k)
      for (int j = 0; j < S2N; ++j) {
        int lane, slot;
        sub_mix_loc32(k, j, lane, slot);
        set_field(sub[lane], w * slot, static_cast<unsigned>(subpay[k * S2N + j]), w);
      }
  float *dF = to_dev(fp8mat), *dC = to_dev(Cm);
  fpsan::v8i32_native *dSub;
  HIP_CHECK(hipMalloc(&dSub, WAVE * sizeof(fpsan::v8i32_native)));
  HIP_CHECK(hipMemcpy(dSub, sub.data(), WAVE * sizeof(fpsan::v8i32_native), hipMemcpyHostToDevice));
  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VF = Value<float, S, kCC>;
    using VFp8 = Value<Fp8Elem, S, kCC>;
    const VF vsa(fpsan::detail::e8m0_to_float(128)), vsb(fpsan::detail::e8m0_to_float(130));
    auto Av = [&](int i, int k) {
      return AIsSub ? sub_canonical_value<SubFmt, S>(static_cast<unsigned>(subpay[i * S2K + k]))
                    : fpsan::cast<float>(VFp8(Fp8Elem(fp8mat[i * S2K + k])));
    };
    auto Bv = [&](int k, int j) {
      return AIsSub ? fpsan::cast<float>(VFp8(Fp8Elem(fp8mat[k * S2N + j])))
                    : sub_canonical_value<SubFmt, S>(static_cast<unsigned>(subpay[k * S2N + j]));
    };
    std::vector<std::uint32_t> ref(S2M * S2N);
    for (int i = 0; i < S2M; ++i)
      for (int j = 0; j < S2N; ++j) {
        VF dot(0.0f);
        for (int k = 0; k < S2K; ++k)
          dot = dot + Av(i, k) * Bv(k, j);
        ref[i * S2N + j] = (VF(Cm[i * S2N + j]) + dot * vsa * vsb).fpsan_payload();
      }
    std::uint32_t *dD;
    HIP_CHECK(hipMalloc(&dD, S2M * S2N * sizeof(std::uint32_t)));
    if constexpr (AIsSub)
      k_scale32_mix_b8<S, std::uint32_t, Fp8Elem, CBSZ, BLGP>
          <<<1, WAVE>>>(dSub, dF, dC, dD, sa, sb);
    else
      k_scale32_mix_a8<S, std::uint32_t, Fp8Elem, CBSZ, BLGP>
          <<<1, WAVE>>>(dF, dSub, dC, dD, sa, sb);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(S2M * S2N);
    HIP_CHECK(hipMemcpy(got.data(), dD, S2M * S2N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for (int i = 0; i < S2M * S2N; ++i)
      EXPECT_EQ(got[i], ref[i]) << "at " << i;
    (void)hipFree(dD);
  });
  (void)hipFree(dF);
  (void)hipFree(dC);
  (void)hipFree(dSub);
}

TEST(ScaledMfma32x32x64_Mixed8xSub, FP8xFP4_Layout) { run_scale32_mixed<false, fp8_e4m3, 0, 4>(); }
TEST(ScaledMfma32x32x64_Mixed8xSub, FP8xFP6_Layout) { run_scale32_mixed<false, fp8_e4m3, 0, 2>(); }
TEST(ScaledMfma32x32x64_Mixed8xSub, FP4xFP8_Layout) { run_scale32_mixed<true, fp8_e4m3, 4, 0>(); }
TEST(ScaledMfma32x32x64_Mixed8xSub, FP8xFP4_Fpsan) {
  run_scale32_mixed_fpsan<false, fp8_e4m3, 0, 4>();
}
TEST(ScaledMfma32x32x64_Mixed8xSub, FP4xFP8_Fpsan) {
  run_scale32_mixed_fpsan<true, fp8_e4m3, 4, 0>();
}
TEST(ScaledMfma32x32x64_Mixed8xSub, FP8xFP6_Fpsan) {
  run_scale32_mixed_fpsan<false, fp8_e4m3, 0, 2>();
}
