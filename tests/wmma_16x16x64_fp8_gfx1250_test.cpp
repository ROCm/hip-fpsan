// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/wmma_16x16x64_fp8_gfx1250_test.cpp
//
// Deep tests for the gfx1250 16x16x64 fp8/bf8 WMMA FPSan wrappers (K=64, 8-bit
// operands; f32 and f16 accumulators). Same two-property structure as
// wmma_test.cpp / wmma_16x16x32_gfx1250_test.cpp:
//
//  (1) LayoutMatchesHardware: software MMA (Float dataflow) vs the real builtin,
//      bit-exact on exact-integer inputs -- validates Wmma16x16x64Layout + gather.
//  (2) FpsanMatchesScalarReference: shipped FPSan path vs an independent host
//      scalar FPSan matmul, payload for payload.
//
// AMD naming: 'fp8' = OCP E4M3FN, 'bf8' = OCP E5M2. Requires gfx1250
// hardware; built only under the gfx1250 test tier.
#include "fpsan/amdgcn_matrix.hpp"
#include "fpsan/fpsan.hpp"

#include "fpsan_semantics.hpp"
#include "hip_test_utils.hpp"
#include "test_random.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::v32e4m3_native;
using fpsan::v32e5m2_native;
using fpsan::v8f_native;
using fpsan::v8h_native;
using fpsan::Value;

static constexpr int M = 16, N = 16, K = 64;
static constexpr Conversions kCC = Conversions::Explicit;

// Inverse of Wmma16x16x64Layout: given lane and v32 fragment slot idx, the
// logical K-index this slot holds.
//   half=k&1; reg=((k>>1)&1)+2*((k>>3)&1)+4*((k>>4)&1)+8*((k>>5)&1); idx=2*reg+half;
//   lane bit = (k>>2)&1.
__device__ inline int frag_k_inv64(int lane, int idx) {
  int half = idx & 1;
  int reg = idx >> 1; // 0..15
  return half | ((reg & 1) << 1) | (((lane >> 4) & 1) << 2) | (((reg >> 1) & 1) << 3) |
         (((reg >> 2) & 1) << 4) | (((reg >> 3) & 1) << 5);
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
  for (int idx = 0; idx < 32; ++idx) {
    int k = frag_k_inv64(lane, idx);
    an[idx] = A[(lane & 15) * K + k];
    bn[idx] = B[k * N + (lane & 15)];
  }
  for (int e = 0; e < 8; ++e)
    cn[e] = C[(e + 8 * (lane >> 4)) * N + (lane & 15)];
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
  auto d = fpsan::detail::wmma_16x16x64_dataflow(a, b, c);
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

// K=64 contraction: keep operand magnitudes small so 64-term integer sums stay
// exact in f32 and f16 (|sum| <= 64*4 + 4 = 260 < 2048).
#define FPSAN_WMMA64_TRAITS(NAME, AVEC, BVEC, CVEC, CALL)                                          \
  struct NAME {                                                                                    \
    using AVec = AVEC;                                                                             \
    using BVec = BVEC;                                                                             \
    using CVec = CVEC;                                                                             \
    static constexpr int a_lo = -2, a_hi = 2;                                                      \
    static constexpr int b_lo = -2, b_hi = 2;                                                      \
    static constexpr int c_lo = -4, c_hi = 4;                                                      \
    template <Semantics S, Conversions C>                                                          \
    __device__ static Value<CVec, S, C> call(Value<AVec, S, C> a, Value<BVec, S, C> b,             \
                                             Value<CVec, S, C> c) {                                \
      return fpsan::CALL(a, b, c);                                                                 \
    }                                                                                              \
  };                                                                                               \
  TEST(NAME, LayoutMatchesHardware) { run_layout_matches_hardware<NAME>(); }                       \
  TEST(NAME, FpsanMatchesScalarReference) {                                                        \
    fpsan_test::for_matrix_fpsan_semantics(                                                        \
        [](auto sem) { run_fpsan_matches_scalar_reference<NAME, decltype(sem)::value>(); });       \
  }

FPSAN_WMMA64_TRAITS(WmmaF32Fp8Fp8_64, v32e4m3_native, v32e4m3_native, v8f_native,
                    amdgcn_wmma_f32_16x16x64_fp8_fp8)
FPSAN_WMMA64_TRAITS(WmmaF32Fp8Bf8_64, v32e4m3_native, v32e5m2_native, v8f_native,
                    amdgcn_wmma_f32_16x16x64_fp8_bf8)
FPSAN_WMMA64_TRAITS(WmmaF32Bf8Fp8_64, v32e5m2_native, v32e4m3_native, v8f_native,
                    amdgcn_wmma_f32_16x16x64_bf8_fp8)
FPSAN_WMMA64_TRAITS(WmmaF32Bf8Bf8_64, v32e5m2_native, v32e5m2_native, v8f_native,
                    amdgcn_wmma_f32_16x16x64_bf8_bf8)
FPSAN_WMMA64_TRAITS(WmmaF16Fp8Fp8_64, v32e4m3_native, v32e4m3_native, v8h_native,
                    amdgcn_wmma_f16_16x16x64_fp8_fp8)
FPSAN_WMMA64_TRAITS(WmmaF16Fp8Bf8_64, v32e4m3_native, v32e5m2_native, v8h_native,
                    amdgcn_wmma_f16_16x16x64_fp8_bf8)
FPSAN_WMMA64_TRAITS(WmmaF16Bf8Fp8_64, v32e5m2_native, v32e4m3_native, v8h_native,
                    amdgcn_wmma_f16_16x16x64_bf8_fp8)
FPSAN_WMMA64_TRAITS(WmmaF16Bf8Bf8_64, v32e5m2_native, v32e5m2_native, v8h_native,
                    amdgcn_wmma_f16_16x16x64_bf8_bf8)
