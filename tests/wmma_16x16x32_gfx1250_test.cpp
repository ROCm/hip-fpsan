// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/wmma_16x16x32_gfx1250_test.cpp
//
// Deep tests for the gfx1250 16x16x32 WMMA FPSan wrappers (K=32, 16-bit
// operands f16/bf16). Same two-property structure as wmma_test.cpp:
//
//  (1) Layout + dataflow vs real hardware. The software MMA run with real-float
//      arithmetic (Semantics::Native dataflow) must match the real builtin
//      bit-for-bit on EXACT-integer inputs. This validates the K=32 fragment
//      layout (Wmma16x16x32Layout) + cross-lane gather against the hardware ABI
//      -- the builtin path passes only if the test staging matches hardware, and
//      the dataflow path passes only if the wrapper gather matches it.
//
//  (2) Payload algebra. The shipped FPSan path must match an independent scalar
//      FPSan reference matmul computed on the host, payload for payload.
//
// Requires gfx1250 hardware; built only under the gfx1250 test tier.
#include "fpsan/amdgcn_matrix.hpp"
#include "fpsan/fpsan.hpp"

#include "fpsan_semantics.hpp"
#include "hip_test_utils.hpp"
#include "test_random.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::v16bf_native;
using fpsan::v16h_native;
using fpsan::v8bf_native;
using fpsan::v8f_native;
using fpsan::v8h_native;
using fpsan::Value;

static constexpr int M = 16, N = 16, K = 32;
static constexpr Conversions kCC = Conversions::Explicit;

// Inverse of Wmma16x16x32Layout: given lane and v16 fragment slot idx, the
// logical K-index this slot holds. Mirrors the forward map:
//   half=k&1; reg=((k>>1)&1)+2*((k>>3)&1)+4*((k>>4)&1); idx=2*reg+half;
//   lane bit = (k>>2)&1.
__device__ inline int frag_k_inv32(int lane, int idx) {
  int half = idx & 1;
  int reg = idx >> 1; // 0..7
  int b0 = half;
  int b1 = reg & 1;
  int b2 = (lane >> 4) & 1;
  int b3 = (reg >> 1) & 1;
  int b4 = (reg >> 2) & 1;
  return b0 | (b1 << 1) | (b2 << 2) | (b3 << 3) | (b4 << 4);
}

template <class Traits> struct Harness {
  using AVec = typename Traits::AVec;
  using BVec = typename Traits::BVec;
  using CVec = typename Traits::CVec;
  using AElem = fpsan::detail::vector_element_t<AVec>;
  using BElem = fpsan::detail::vector_element_t<BVec>;
  using CElem = fpsan::detail::vector_element_t<CVec>;
  using CBits = typename fpsan::detail::fp_traits<CElem>::bits_type;
};

template <class Traits, Semantics S>
__device__ void load_frags(const typename Harness<Traits>::AElem *A,
                           const typename Harness<Traits>::BElem *B,
                           const typename Harness<Traits>::CElem *C, int lane,
                           Value<typename Harness<Traits>::AVec, S, kCC> &a,
                           Value<typename Harness<Traits>::BVec, S, kCC> &b,
                           Value<typename Harness<Traits>::CVec, S, kCC> &c) {
  typename Harness<Traits>::AVec an;
  typename Harness<Traits>::BVec bn;
  typename Harness<Traits>::CVec cn;
  for (int idx = 0; idx < 16; ++idx) {
    int k = frag_k_inv32(lane, idx);
    an[idx] = A[(lane & 15) * K + k]; // A[m=lane&15][k]
    bn[idx] = B[k * N + (lane & 15)]; // B[k][n=lane&15]
  }
  for (int e = 0; e < 8; ++e) {
    int m = e + 8 * (lane >> 4), n = lane & 15;
    cn[e] = C[m * N + n];
  }
  a = Value<typename Harness<Traits>::AVec, S, kCC>(an);
  b = Value<typename Harness<Traits>::BVec, S, kCC>(bn);
  c = Value<typename Harness<Traits>::CVec, S, kCC>(cn);
}

template <class Traits>
__global__ void
k_builtin(const typename Harness<Traits>::AElem *A, const typename Harness<Traits>::BElem *B,
          const typename Harness<Traits>::CElem *C, typename Harness<Traits>::CElem *D) {
  int lane = threadIdx.x;
  Value<typename Harness<Traits>::AVec, Semantics::Native, kCC> a;
  Value<typename Harness<Traits>::BVec, Semantics::Native, kCC> b;
  Value<typename Harness<Traits>::CVec, Semantics::Native, kCC> c;
  load_frags<Traits, Semantics::Native>(A, B, C, lane, a, b, c);
  auto d = Traits::template call<Semantics::Native, kCC>(a, b, c);
  for (int e = 0; e < 8; ++e)
    D[(e + 8 * (lane >> 4)) * N + (lane & 15)] = d.get(e).to_float();
}

template <class Traits>
__global__ void
k_float_dataflow(const typename Harness<Traits>::AElem *A, const typename Harness<Traits>::BElem *B,
                 const typename Harness<Traits>::CElem *C, typename Harness<Traits>::CElem *D) {
  int lane = threadIdx.x;
  Value<typename Harness<Traits>::AVec, Semantics::Native, kCC> a;
  Value<typename Harness<Traits>::BVec, Semantics::Native, kCC> b;
  Value<typename Harness<Traits>::CVec, Semantics::Native, kCC> c;
  load_frags<Traits, Semantics::Native>(A, B, C, lane, a, b, c);
  auto d = fpsan::detail::wmma_16x16x32_dataflow(a, b, c);
  for (int e = 0; e < 8; ++e)
    D[(e + 8 * (lane >> 4)) * N + (lane & 15)] = d.get(e).to_float();
}

template <class Traits, Semantics S>
__global__ void
k_fpsan(const typename Harness<Traits>::AElem *A, const typename Harness<Traits>::BElem *B,
        const typename Harness<Traits>::CElem *C, typename Harness<Traits>::CBits *Dpay) {
  int lane = threadIdx.x;
  Value<typename Harness<Traits>::AVec, S, kCC> a;
  Value<typename Harness<Traits>::BVec, S, kCC> b;
  Value<typename Harness<Traits>::CVec, S, kCC> c;
  load_frags<Traits, S>(A, B, C, lane, a, b, c);
  auto d = Traits::template call<S, kCC>(a, b, c);
  for (int e = 0; e < 8; ++e)
    Dpay[(e + 8 * (lane >> 4)) * N + (lane & 15)] = d.get(e).fpsan_payload();
}

namespace {
template <class Traits> struct Mats {
  std::vector<typename Harness<Traits>::AElem> A;
  std::vector<typename Harness<Traits>::BElem> B;
  std::vector<typename Harness<Traits>::CElem> C;
};
template <class Traits> Mats<Traits> make_inputs() {
  using AE = typename Harness<Traits>::AElem;
  using BE = typename Harness<Traits>::BElem;
  using CE = typename Harness<Traits>::CElem;
  Mats<Traits> m;
  m.A.resize(M * K);
  m.B.resize(K * N);
  m.C.resize(M * N);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : m.A)
    x = fpsan_test::pick_int_valued<AE>(rng, Traits::a_lo, Traits::a_hi);
  for (auto &x : m.B)
    x = fpsan_test::pick_int_valued<BE>(rng, Traits::b_lo, Traits::b_hi);
  for (auto &x : m.C)
    x = fpsan_test::pick_int_valued<CE>(rng, Traits::c_lo, Traits::c_hi);
  return m;
}
} // namespace

template <class Traits> void run_layout_matches_hardware() {
  using CE = typename Harness<Traits>::CElem;
  using AE = typename Harness<Traits>::AElem;
  using BE = typename Harness<Traits>::BElem;
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  Mats<Traits> m = make_inputs<Traits>();
  AE *dA = to_dev(m.A);
  BE *dB = to_dev(m.B);
  CE *dC = to_dev(m.C);
  CE *dHw, *dOurs;
  HIP_CHECK(hipMalloc(&dHw, M * N * sizeof(CE)));
  HIP_CHECK(hipMalloc(&dOurs, M * N * sizeof(CE)));

  k_builtin<Traits><<<1, 32>>>(dA, dB, dC, dHw);
  k_float_dataflow<Traits><<<1, 32>>>(dA, dB, dC, dOurs);
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<CE> hw(M * N), ours(M * N);
  HIP_CHECK(hipMemcpy(hw.data(), dHw, M * N * sizeof(CE), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(ours.data(), dOurs, M * N * sizeof(CE), hipMemcpyDeviceToHost));
  for (int i = 0; i < M * N; ++i)
    EXPECT_EQ(bits_of(hw[i]), bits_of(ours[i])) << "mismatch at " << (i / N) << "," << (i % N);
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dHw);
  (void)hipFree(dOurs);
}

template <class Traits, Semantics S> void run_fpsan_matches_scalar_reference() {
  using AE = typename Harness<Traits>::AElem;
  using BE = typename Harness<Traits>::BElem;
  using CE = typename Harness<Traits>::CElem;
  using CBits = typename Harness<Traits>::CBits;
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  Mats<Traits> m = make_inputs<Traits>();

  using VA = Value<AE, S, kCC>;
  using VB = Value<BE, S, kCC>;
  using VC = Value<CE, S, kCC>;
  std::vector<CBits> ref(M * N);
  for (int mm = 0; mm < M; ++mm)
    for (int nn = 0; nn < N; ++nn) {
      VC acc(m.C[mm * N + nn]);
      for (int k = 0; k < K; ++k)
        acc = acc + fpsan::cast<CE>(VA(m.A[mm * K + k])) * fpsan::cast<CE>(VB(m.B[k * N + nn]));
      ref[mm * N + nn] = acc.fpsan_payload();
    }

  AE *dA = to_dev(m.A);
  BE *dB = to_dev(m.B);
  CE *dC = to_dev(m.C);
  CBits *dD;
  HIP_CHECK(hipMalloc(&dD, M * N * sizeof(CBits)));
  k_fpsan<Traits, S><<<1, 32>>>(dA, dB, dC, dD);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<CBits> got(M * N);
  HIP_CHECK(hipMemcpy(got.data(), dD, M * N * sizeof(CBits), hipMemcpyDeviceToHost));
  for (int i = 0; i < M * N; ++i)
    EXPECT_EQ(got[i], ref[i]) << "payload mismatch at " << (i / N) << "," << (i % N);
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dD);
}

// K=32 doubles the contraction length, so keep operand magnitudes small enough
// that 32-term integer sums stay exact in every output type (bf16 is exact for
// |x| <= 256: 32 * (2*2) + 4 = 132 < 256).
struct WmmaF32F16_32 {
  using AVec = v16h_native;
  using BVec = v16h_native;
  using CVec = v8f_native;
  static constexpr int a_lo = -2, a_hi = 2;
  static constexpr int b_lo = -2, b_hi = 2;
  static constexpr int c_lo = -4, c_hi = 4;
  template <Semantics S, Conversions C>
  __device__ static Value<CVec, S, C> call(Value<AVec, S, C> a, Value<BVec, S, C> b,
                                           Value<CVec, S, C> c) {
    return fpsan::amdgcn_wmma_f32_16x16x32_f16(a, b, c);
  }
};
TEST(WmmaF32F16_32, LayoutMatchesHardware) { run_layout_matches_hardware<WmmaF32F16_32>(); }
TEST(WmmaF32F16_32, FpsanMatchesScalarReference) {
  fpsan_test::for_matrix_fpsan_semantics(
      [](auto sem) { run_fpsan_matches_scalar_reference<WmmaF32F16_32, decltype(sem)::value>(); });
}

struct WmmaF16F16_32 {
  using AVec = v16h_native;
  using BVec = v16h_native;
  using CVec = v8h_native;
  static constexpr int a_lo = -2, a_hi = 2;
  static constexpr int b_lo = -2, b_hi = 2;
  static constexpr int c_lo = -4, c_hi = 4;
  template <Semantics S, Conversions C>
  __device__ static Value<CVec, S, C> call(Value<AVec, S, C> a, Value<BVec, S, C> b,
                                           Value<CVec, S, C> c) {
    return fpsan::amdgcn_wmma_f16_16x16x32_f16(a, b, c);
  }
};
TEST(WmmaF16F16_32, LayoutMatchesHardware) { run_layout_matches_hardware<WmmaF16F16_32>(); }
TEST(WmmaF16F16_32, FpsanMatchesScalarReference) {
  fpsan_test::for_matrix_fpsan_semantics(
      [](auto sem) { run_fpsan_matches_scalar_reference<WmmaF16F16_32, decltype(sem)::value>(); });
}

struct WmmaF32BF16_32 {
  using AVec = v16bf_native;
  using BVec = v16bf_native;
  using CVec = v8f_native;
  static constexpr int a_lo = -2, a_hi = 2;
  static constexpr int b_lo = -2, b_hi = 2;
  static constexpr int c_lo = -4, c_hi = 4;
  template <Semantics S, Conversions C>
  __device__ static Value<CVec, S, C> call(Value<AVec, S, C> a, Value<BVec, S, C> b,
                                           Value<CVec, S, C> c) {
    return fpsan::amdgcn_wmma_f32_16x16x32_bf16(a, b, c);
  }
};
TEST(WmmaF32BF16_32, LayoutMatchesHardware) { run_layout_matches_hardware<WmmaF32BF16_32>(); }
TEST(WmmaF32BF16_32, FpsanMatchesScalarReference) {
  fpsan_test::for_matrix_fpsan_semantics(
      [](auto sem) { run_fpsan_matches_scalar_reference<WmmaF32BF16_32, decltype(sem)::value>(); });
}

struct WmmaBF16BF16_32 {
  using AVec = v16bf_native;
  using BVec = v16bf_native;
  using CVec = v8bf_native;
  static constexpr int a_lo = -2, a_hi = 2;
  static constexpr int b_lo = -2, b_hi = 2;
  static constexpr int c_lo = -4, c_hi = 4;
  template <Semantics S, Conversions C>
  __device__ static Value<CVec, S, C> call(Value<AVec, S, C> a, Value<BVec, S, C> b,
                                           Value<CVec, S, C> c) {
    return fpsan::amdgcn_wmma_bf16_16x16x32_bf16(a, b, c);
  }
};
TEST(WmmaBF16BF16_32, LayoutMatchesHardware) { run_layout_matches_hardware<WmmaBF16BF16_32>(); }
TEST(WmmaBF16BF16_32, FpsanMatchesScalarReference) {
  fpsan_test::for_matrix_fpsan_semantics([](auto sem) {
    run_fpsan_matches_scalar_reference<WmmaBF16BF16_32, decltype(sem)::value>();
  });
}

// ---- WMMA C-accumulator modifier coverage (gfx1250) --------------------------
// Every gfx1250 FP WMMA carries an i16 C-modifier: D = A*B + mod(C), where the
// low 2 bits are bit0=neg, bit1=abs (applied abs-then-neg): 0:C 1:-C 2:|C| 3:-|C|.
// (Operand neg(A)/neg(B) is reserved-zero on all FP shapes -- the compiler
// rejects nonzero -- so the C-modifier is the only live numeric WMMA modifier.)
// These tests exercise non-default modifier values against fully independent
// host oracles, so they assert real intrinsic correctness on the device:
//  - Native mode: wrapper (real builtin, modifier applied in silicon) vs a host
//    float matmul that applies mod(C) with std::fabs/negate -- bit-for-bit.
//  - FPSan mode: shipped payload path vs a host FPSan matmul that pre-applies
//    mod(C) in the payload ring.
// f32<-f16 is the representative shape (f32 accumulate); the modifier is shape-
// independent (it only transforms C), so one shape pins the wrapper plumbing.
static float host_cmod(float c, int cmod) {
  if (cmod & 2)
    c = std::fabs(c);
  if (cmod & 1)
    c = -c;
  return c;
}

template <int Cmod>
__global__ void k_f16_mod_builtin(const _Float16 *A, const _Float16 *B, const float *C, float *D) {
  int lane = threadIdx.x;
  Value<v16h_native, Semantics::Native, kCC> a;
  Value<v16h_native, Semantics::Native, kCC> b;
  Value<v8f_native, Semantics::Native, kCC> c;
  load_frags<WmmaF32F16_32, Semantics::Native>(A, B, C, lane, a, b, c);
  auto d = fpsan::amdgcn_wmma_f32_16x16x32_f16<Semantics::Native, kCC, Cmod>(a, b, c);
  for (int e = 0; e < 8; ++e)
    D[(e + 8 * (lane >> 4)) * N + (lane & 15)] = d.get(e).to_float();
}

template <int Cmod, Semantics S>
__global__ void k_f16_mod_fpsan(const _Float16 *A, const _Float16 *B, const float *C,
                                std::uint32_t *D) {
  int lane = threadIdx.x;
  Value<v16h_native, S, kCC> a;
  Value<v16h_native, S, kCC> b;
  Value<v8f_native, S, kCC> c;
  load_frags<WmmaF32F16_32, S>(A, B, C, lane, a, b, c);
  auto d = fpsan::amdgcn_wmma_f32_16x16x32_f16<S, kCC, Cmod>(a, b, c);
  for (int e = 0; e < 8; ++e)
    D[(e + 8 * (lane >> 4)) * N + (lane & 15)] = d.get(e).fpsan_payload();
}

template <int Cmod, Semantics S> void run_modifier_f16() {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  Mats<WmmaF32F16_32> m = make_inputs<WmmaF32F16_32>();

  // Float oracle: host matmul with mod(C).
  std::vector<float> fref(M * N);
  for (int mm = 0; mm < M; ++mm)
    for (int nn = 0; nn < N; ++nn) {
      float acc = host_cmod(m.C[mm * N + nn], Cmod);
      for (int k = 0; k < K; ++k)
        acc += float(m.A[mm * K + k]) * float(m.B[k * N + nn]);
      fref[mm * N + nn] = acc;
    }
  // FPSan oracle: payload matmul with mod(C) pre-applied in the ring.
  using VA = Value<_Float16, S, kCC>;
  using VF = Value<float, S, kCC>;
  std::vector<std::uint32_t> pref(M * N);
  for (int mm = 0; mm < M; ++mm)
    for (int nn = 0; nn < N; ++nn) {
      VF cm(m.C[mm * N + nn]);
      if (Cmod & 2)
        cm = fpsan::max(cm, -cm);
      if (Cmod & 1)
        cm = -cm;
      VF acc = cm;
      for (int k = 0; k < K; ++k)
        acc =
            acc + fpsan::cast<float>(VA(m.A[mm * K + k])) * fpsan::cast<float>(VA(m.B[k * N + nn]));
      pref[mm * N + nn] = acc.fpsan_payload();
    }

  _Float16 *dA = to_dev(m.A);
  _Float16 *dB = to_dev(m.B);
  float *dC = to_dev(m.C);
  float *dF;
  std::uint32_t *dP;
  HIP_CHECK(hipMalloc(&dF, M * N * sizeof(float)));
  HIP_CHECK(hipMalloc(&dP, M * N * sizeof(std::uint32_t)));
  k_f16_mod_builtin<Cmod><<<1, 32>>>(dA, dB, dC, dF);
  k_f16_mod_fpsan<Cmod, S><<<1, 32>>>(dA, dB, dC, dP);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> gotF(M * N);
  std::vector<std::uint32_t> gotP(M * N);
  HIP_CHECK(hipMemcpy(gotF.data(), dF, M * N * sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(gotP.data(), dP, M * N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
  for (int i = 0; i < M * N; ++i) {
    EXPECT_EQ(bits_of(gotF[i]), bits_of(fref[i]))
        << "Float mod mismatch Cmod=" << Cmod << " at " << (i / N) << "," << (i % N);
    EXPECT_EQ(gotP[i], pref[i]) << "FPSan mod mismatch Cmod=" << Cmod << " at " << (i / N) << ","
                                << (i % N);
  }
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dF);
  (void)hipFree(dP);
}

TEST(WmmaF32F16_32, ModifierNegC) {
  fpsan_test::for_matrix_fpsan_semantics(
      [](auto sem) { run_modifier_f16<1, decltype(sem)::value>(); });
}
TEST(WmmaF32F16_32, ModifierAbsC) {
  fpsan_test::for_matrix_fpsan_semantics(
      [](auto sem) { run_modifier_f16<2, decltype(sem)::value>(); });
}
TEST(WmmaF32F16_32, ModifierNegAbsC) {
  fpsan_test::for_matrix_fpsan_semantics(
      [](auto sem) { run_modifier_f16<3, decltype(sem)::value>(); });
}

// ---- bf16f32 mixed variant: A,B bf16; C f32 accumulator; D bf16 output -------
// f32 accumulate over K=32, narrow to bf16 once at the end. Needs dedicated
// kernels because C (v8f) and D (v8bf) differ in type.
template <Semantics S>
__device__ void load_frags_bf16f32(const __bf16 *A, const __bf16 *B, const float *C, int lane,
                                   Value<v16bf_native, S, kCC> &a, Value<v16bf_native, S, kCC> &b,
                                   Value<v8f_native, S, kCC> &c) {
  v16bf_native an;
  v16bf_native bn;
  v8f_native cn;
  for (int idx = 0; idx < 16; ++idx) {
    int k = frag_k_inv32(lane, idx);
    an[idx] = A[(lane & 15) * K + k];
    bn[idx] = B[k * N + (lane & 15)];
  }
  for (int e = 0; e < 8; ++e)
    cn[e] = C[(e + 8 * (lane >> 4)) * N + (lane & 15)];
  a = Value<v16bf_native, S, kCC>(an);
  b = Value<v16bf_native, S, kCC>(bn);
  c = Value<v8f_native, S, kCC>(cn);
}

__global__ void k_bf16f32_builtin(const __bf16 *A, const __bf16 *B, const float *C, __bf16 *D) {
  int lane = threadIdx.x;
  Value<v16bf_native, Semantics::Native, kCC> a;
  Value<v16bf_native, Semantics::Native, kCC> b;
  Value<v8f_native, Semantics::Native, kCC> c;
  load_frags_bf16f32<Semantics::Native>(A, B, C, lane, a, b, c);
  auto d = fpsan::amdgcn_wmma_bf16f32_16x16x32_bf16<Semantics::Native, kCC>(a, b, c);
  for (int e = 0; e < 8; ++e)
    D[(e + 8 * (lane >> 4)) * N + (lane & 15)] = static_cast<__bf16>(d.get(e));
}

__global__ void k_bf16f32_dataflow(const __bf16 *A, const __bf16 *B, const float *C, __bf16 *D) {
  int lane = threadIdx.x;
  Value<v16bf_native, Semantics::Native, kCC> a;
  Value<v16bf_native, Semantics::Native, kCC> b;
  Value<v8f_native, Semantics::Native, kCC> c;
  load_frags_bf16f32<Semantics::Native>(A, B, C, lane, a, b, c);
  auto d = fpsan::detail::wmma_16x16x32_dataflow_cdiff<v8bf_native, v16bf_native, v16bf_native,
                                                       v8f_native>(a, b, c);
  for (int e = 0; e < 8; ++e)
    D[(e + 8 * (lane >> 4)) * N + (lane & 15)] = static_cast<__bf16>(d.get(e));
}

template <Semantics S>
__global__ void k_bf16f32_fpsan(const __bf16 *A, const __bf16 *B, const float *C,
                                std::uint16_t *D) {
  int lane = threadIdx.x;
  Value<v16bf_native, S, kCC> a;
  Value<v16bf_native, S, kCC> b;
  Value<v8f_native, S, kCC> c;
  load_frags_bf16f32<S>(A, B, C, lane, a, b, c);
  auto d = fpsan::amdgcn_wmma_bf16f32_16x16x32_bf16<S, kCC>(a, b, c);
  for (int e = 0; e < 8; ++e)
    D[(e + 8 * (lane >> 4)) * N + (lane & 15)] = d.get(e).fpsan_payload();
}

TEST(WmmaBF16F32_32, LayoutMatchesHardware) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  std::mt19937 rng = fpsan_test::make_rng();
  std::vector<__bf16> A(M * K), B(K * N);
  std::vector<float> C(M * N);
  for (auto &x : A)
    x = fpsan_test::pick_int_valued<__bf16>(rng, -2, 2);
  for (auto &x : B)
    x = fpsan_test::pick_int_valued<__bf16>(rng, -2, 2);
  for (auto &x : C)
    x = fpsan_test::pick_int_valued<float>(rng, -4, 4);
  auto *dA = to_dev(A);
  auto *dB = to_dev(B);
  float *dC = to_dev(C);
  __bf16 *dHw, *dOurs;
  HIP_CHECK(hipMalloc(&dHw, M * N * sizeof(__bf16)));
  HIP_CHECK(hipMalloc(&dOurs, M * N * sizeof(__bf16)));
  k_bf16f32_builtin<<<1, 32>>>(dA, dB, dC, dHw);
  k_bf16f32_dataflow<<<1, 32>>>(dA, dB, dC, dOurs);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<__bf16> hw(M * N), ours(M * N);
  HIP_CHECK(hipMemcpy(hw.data(), dHw, M * N * sizeof(__bf16), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(ours.data(), dOurs, M * N * sizeof(__bf16), hipMemcpyDeviceToHost));
  for (int i = 0; i < M * N; ++i)
    EXPECT_EQ(bits_of(hw[i]), bits_of(ours[i])) << "mismatch at " << (i / N) << "," << (i % N);
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dHw);
  (void)hipFree(dOurs);
}

template <Semantics S> void run_bf16f32_fpsan_matches_scalar_reference() {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  std::mt19937 rng = fpsan_test::make_rng();
  std::vector<__bf16> A(M * K), B(K * N);
  std::vector<float> C(M * N);
  for (auto &x : A)
    x = fpsan_test::pick_int_valued<__bf16>(rng, -2, 2);
  for (auto &x : B)
    x = fpsan_test::pick_int_valued<__bf16>(rng, -2, 2);
  for (auto &x : C)
    x = fpsan_test::pick_int_valued<float>(rng, -4, 4);

  using VBF = Value<__bf16, S, kCC>;
  using VF = Value<float, S, kCC>;
  std::vector<std::uint16_t> ref(M * N);
  for (int mm = 0; mm < M; ++mm)
    for (int nn = 0; nn < N; ++nn) {
      VF acc(C[mm * N + nn]);
      for (int k = 0; k < K; ++k)
        acc = acc + fpsan::cast<float>(VBF(A[mm * K + k])) * fpsan::cast<float>(VBF(B[k * N + nn]));
      ref[mm * N + nn] = fpsan::cast<__bf16>(acc).fpsan_payload();
    }
  auto *dA = to_dev(A);
  auto *dB = to_dev(B);
  float *dC = to_dev(C);
  std::uint16_t *dD;
  HIP_CHECK(hipMalloc(&dD, M * N * sizeof(std::uint16_t)));
  k_bf16f32_fpsan<S><<<1, 32>>>(dA, dB, dC, dD);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<std::uint16_t> got(M * N);
  HIP_CHECK(hipMemcpy(got.data(), dD, M * N * sizeof(std::uint16_t), hipMemcpyDeviceToHost));
  for (int i = 0; i < M * N; ++i)
    EXPECT_EQ(got[i], ref[i]) << "payload mismatch at " << (i / N) << "," << (i % N);
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dD);
}

TEST(WmmaBF16F32_32, FpsanMatchesScalarReference) {
  fpsan_test::for_matrix_fpsan_semantics(
      [](auto sem) { run_bf16f32_fpsan_matches_scalar_reference<decltype(sem)::value>(); });
}
