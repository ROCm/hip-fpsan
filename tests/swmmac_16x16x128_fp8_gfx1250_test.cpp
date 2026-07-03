// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/swmmac_16x16x128_fp8_gfx1250_test.cpp
//
// Deep tests for the gfx1250 16x16x128 fp8/bf8 SWMMAC FPSan wrappers (2:4
// structured sparsity; K=128 dense / K=64 compressed; f32 and f16 accumulators).
// Same two-property structure as the dense WMMA suites:
//
//  (1) LayoutMatchesHardware: the software sparse dataflow (Semantics::Native)
//      must match the real builtin bit-for-bit on exact-integer inputs. This
//      validates the compressed-A layout (Wmma16x16x64Layout), the dense-B
//      layout (Wmma16x16x128Layout) and -- the new piece -- the sparse INDEX
//      layout (lane = i+16*(c/32), dword = (c/16)&1, bit = (c%16)*2, live
//      dense K = 4*(c/2)+selector), all against the hardware ABI.
//
//  (2) FpsanMatchesScalarReference: the shipped FPSan path equals an independent
//      host scalar FPSan sparse matmul, payload for payload.
//
// AMD naming: 'fp8' = OCP E4M3FN, 'bf8' = OCP E5M2. Requires a gfx1250
// hardware; built only under the gfx1250 test tier.
#include "fpsan/amdgcn_swmmac_gfx1250.hpp"
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
using fpsan::v2i32_native;
using fpsan::v32e4m3_native;
using fpsan::v32e5m2_native;
using fpsan::v64e4m3_native;
using fpsan::v64e5m2_native;
using fpsan::v8f_native;
using fpsan::v8h_native;
using fpsan::Value;

static constexpr int M = 16, N = 16, KD = 128, KC = 64;
static constexpr Conversions kCC = Conversions::Explicit;

// Inverse compressed-A layout (Wmma16x16x64Layout): lane + v32 slot -> compressed K.
// Inverse of Swmmac16x16x128CompressedALayout (empirically pinned): given lane
// and fragment slot idx (0..31), recover the compressed slot cc (0..63).
//   cc bit0 = idx&1 ; cc bit1 = lane half ; cc bits2.. = idx>>1.
__device__ inline int frag_cc_inv64(int lane, int idx) {
  return (idx & 1) | (((lane >> 4) & 1) << 1) | ((idx >> 1) << 2);
}
// Inverse dense-B layout (Wmma16x16x128Layout): lane + v64 slot -> dense K.
__device__ inline int frag_k_inv128(int lane, int idx) {
  int half = idx & 1;
  int reg = idx >> 1; // 0..31
  return half | ((reg & 1) << 1) | (((lane >> 4) & 1) << 2) | (((reg >> 1) & 1) << 3) |
         (((reg >> 2) & 1) << 4) | (((reg >> 3) & 1) << 5) | (((reg >> 4) & 1) << 6);
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
__device__ void load_frags(const typename Harness<Traits>::AElem *Acomp, // [M*KC]
                           const typename Harness<Traits>::BElem *B,     // [KD*N]
                           const typename Harness<Traits>::CElem *C,     // [M*N]
                           int lane, Value<typename Harness<Traits>::AVec, S, kCC> &a,
                           Value<typename Harness<Traits>::BVec, S, kCC> &b,
                           Value<typename Harness<Traits>::CVec, S, kCC> &c) {
  typename Harness<Traits>::AVec an;
  typename Harness<Traits>::BVec bn;
  typename Harness<Traits>::CVec cn;
  for (int idx = 0; idx < 32; ++idx) {
    int cc = frag_cc_inv64(lane, idx);
    an[idx] = Acomp[(lane & 15) * KC + cc];
  }
  for (int idx = 0; idx < 64; ++idx) {
    int k = frag_k_inv128(lane, idx);
    bn[idx] = B[k * N + (lane & 15)];
  }
  for (int e = 0; e < 8; ++e)
    cn[e] = C[(e + 8 * (lane >> 4)) * N + (lane & 15)];
  a = Value<typename Harness<Traits>::AVec, S, kCC>(an);
  b = Value<typename Harness<Traits>::BVec, S, kCC>(bn);
  c = Value<typename Harness<Traits>::CVec, S, kCC>(cn);
}

template <class Traits>
__global__ void k_builtin(const typename Harness<Traits>::AElem *A,
                          const typename Harness<Traits>::BElem *B,
                          const typename Harness<Traits>::CElem *C,
                          const int *IDX, // [32*2]
                          typename Harness<Traits>::CElem *D) {
  int lane = threadIdx.x;
  Value<typename Harness<Traits>::AVec, Semantics::Native, kCC> a;
  Value<typename Harness<Traits>::BVec, Semantics::Native, kCC> b;
  Value<typename Harness<Traits>::CVec, Semantics::Native, kCC> c;
  load_frags<Traits, Semantics::Native>(A, B, C, lane, a, b, c);
  v2i32_native idx;
  idx[0] = IDX[lane * 2 + 0];
  idx[1] = IDX[lane * 2 + 1];
  auto d = Traits::template call<Semantics::Native, kCC>(a, b, c, idx);
  for (int e = 0; e < 8; ++e)
    D[(e + 8 * (lane >> 4)) * N + (lane & 15)] = d.get(e).to_float();
}

template <class Traits>
__global__ void k_float_dataflow(const typename Harness<Traits>::AElem *A,
                                 const typename Harness<Traits>::BElem *B,
                                 const typename Harness<Traits>::CElem *C, const int *IDX,
                                 typename Harness<Traits>::CElem *D) {
  int lane = threadIdx.x;
  Value<typename Harness<Traits>::AVec, Semantics::Native, kCC> a;
  Value<typename Harness<Traits>::BVec, Semantics::Native, kCC> b;
  Value<typename Harness<Traits>::CVec, Semantics::Native, kCC> c;
  load_frags<Traits, Semantics::Native>(A, B, C, lane, a, b, c);
  v2i32_native idx;
  idx[0] = IDX[lane * 2 + 0];
  idx[1] = IDX[lane * 2 + 1];
  auto d = fpsan::detail::swmmac_software_16x16x128_fp8<
      typename Harness<Traits>::AVec, typename Harness<Traits>::BVec,
      typename Harness<Traits>::CVec, Semantics::Native, kCC>(a, b, c, idx);
  for (int e = 0; e < 8; ++e)
    D[(e + 8 * (lane >> 4)) * N + (lane & 15)] = d.get(e).to_float();
}

template <class Traits, Semantics S>
__global__ void k_fpsan(const typename Harness<Traits>::AElem *A,
                        const typename Harness<Traits>::BElem *B,
                        const typename Harness<Traits>::CElem *C, const int *IDX,
                        typename Harness<Traits>::CBits *Dpay) {
  int lane = threadIdx.x;
  Value<typename Harness<Traits>::AVec, S, kCC> a;
  Value<typename Harness<Traits>::BVec, S, kCC> b;
  Value<typename Harness<Traits>::CVec, S, kCC> c;
  load_frags<Traits, S>(A, B, C, lane, a, b, c);
  v2i32_native idx;
  idx[0] = IDX[lane * 2 + 0];
  idx[1] = IDX[lane * 2 + 1];
  auto d = Traits::template call<S, kCC>(a, b, c, idx);
  for (int e = 0; e < 8; ++e)
    Dpay[(e + 8 * (lane >> 4)) * N + (lane & 15)] = d.get(e).fpsan_payload();
}

namespace {
template <class Traits> struct Mats {
  std::vector<typename Harness<Traits>::AElem> A; // [M*KC]
  std::vector<typename Harness<Traits>::BElem> B; // [KD*N]
  std::vector<typename Harness<Traits>::CElem> C; // [M*N]
  std::vector<int> IDX;                           // [32*2]
  std::vector<std::uint8_t> sel;                  // [M*KC] selectors 0..3
};

template <class Traits> Mats<Traits> make_inputs() {
  using AE = typename Harness<Traits>::AElem;
  using BE = typename Harness<Traits>::BElem;
  using CE = typename Harness<Traits>::CElem;
  Mats<Traits> m;
  m.A.resize(M * KC);
  m.B.resize(KD * N);
  m.C.resize(M * N);
  m.IDX.assign(32 * 2, 0);
  m.sel.resize(M * KC);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : m.A)
    x = fpsan_test::pick_int_valued<AE>(rng, -2, 2);
  for (auto &x : m.B)
    x = fpsan_test::pick_int_valued<BE>(rng, -2, 2);
  for (auto &x : m.C)
    x = fpsan_test::pick_int_valued<CE>(rng, -4, 4);

  // For each (row, group g) pick two distinct dense positions p0<p1 in 0..3,
  // assign compressed slot 2g -> p0, 2g+1 -> p1, and stage the 2-bit
  // selectors into the per-lane v2i32 index per the layout.
  std::uniform_int_distribution<int> d6(0, 5);
  const int pairs[6][2] = {{0, 1}, {0, 2}, {0, 3}, {1, 2}, {1, 3}, {2, 3}};
  for (int i = 0; i < M; ++i)
    for (int g = 0; g < KC / 2; ++g) {
      const int *pr = pairs[d6(rng)];
      for (int s = 0; s < 2; ++s) {
        const int cc = 2 * g + s;
        const int selv = pr[s];
        m.sel[i * KC + cc] = static_cast<std::uint8_t>(selv);
        // selector location pinned empirically vs hardware; see
        // amdgcn_swmmac_gfx1250.hpp.
        const int L = i + 16 * (cc / 32);
        const int dw = (cc / 2) & 1;
        const int bit = 4 * ((cc / 4) % 8) + 2 * (cc & 1);
        m.IDX[L * 2 + dw] |= selv << bit;
      }
    }
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
  int *dI = to_dev(m.IDX);
  CE *dHw, *dOurs;
  HIP_CHECK(hipMalloc(&dHw, M * N * sizeof(CE)));
  HIP_CHECK(hipMalloc(&dOurs, M * N * sizeof(CE)));
  k_builtin<Traits><<<1, 32>>>(dA, dB, dC, dI, dHw);
  k_float_dataflow<Traits><<<1, 32>>>(dA, dB, dC, dI, dOurs);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<CE> hw(M * N), ours(M * N);
  HIP_CHECK(hipMemcpy(hw.data(), dHw, M * N * sizeof(CE), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(ours.data(), dOurs, M * N * sizeof(CE), hipMemcpyDeviceToHost));
  for (int i = 0; i < M * N; ++i)
    EXPECT_EQ(bits_of(hw[i]), bits_of(ours[i])) << "mismatch at " << (i / N) << "," << (i % N);
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dI);
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
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      VC acc(m.C[i * N + j]);
      for (int cc = 0; cc < KC; ++cc) {
        const int k = 4 * (cc / 2) + m.sel[i * KC + cc];
        acc = acc + fpsan::cast<CE>(VA(m.A[i * KC + cc])) * fpsan::cast<CE>(VB(m.B[k * N + j]));
      }
      ref[i * N + j] = acc.fpsan_payload();
    }

  AE *dA = to_dev(m.A);
  BE *dB = to_dev(m.B);
  CE *dC = to_dev(m.C);
  int *dI = to_dev(m.IDX);
  CBits *dD;
  HIP_CHECK(hipMalloc(&dD, M * N * sizeof(CBits)));
  k_fpsan<Traits, S><<<1, 32>>>(dA, dB, dC, dI, dD);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<CBits> got(M * N);
  HIP_CHECK(hipMemcpy(got.data(), dD, M * N * sizeof(CBits), hipMemcpyDeviceToHost));
  for (int i = 0; i < M * N; ++i)
    EXPECT_EQ(got[i], ref[i]) << "payload mismatch at " << (i / N) << "," << (i % N);
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dI);
  (void)hipFree(dD);
}

#define SWMMAC_TRAITS(NAME, AVEC, BVEC, CVEC, WRAP)                                                \
  struct NAME {                                                                                    \
    using AVec = AVEC;                                                                             \
    using BVec = BVEC;                                                                             \
    using CVec = CVEC;                                                                             \
    template <Semantics S, Conversions C>                                                          \
    __device__ static Value<CVec, S, C> call(Value<AVec, S, C> a, Value<BVec, S, C> b,             \
                                             Value<CVec, S, C> c, fpsan::v2i32_native idx) {       \
      return fpsan::WRAP(a, b, c, idx);                                                            \
    }                                                                                              \
  };                                                                                               \
  TEST(NAME, LayoutMatchesHardware) { run_layout_matches_hardware<NAME>(); }                       \
  TEST(NAME, FpsanMatchesScalarReference) {                                                        \
    fpsan_test::for_matrix_fpsan_semantics(                                                        \
        [](auto sem) { run_fpsan_matches_scalar_reference<NAME, decltype(sem)::value>(); });       \
  }

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_swmmac_f32_16x16x128_fp8_fp8)
SWMMAC_TRAITS(SwmmacF32Fp8Fp8, v32e4m3_native, v64e4m3_native, v8f_native,
              amdgcn_swmmac_f32_16x16x128_fp8_fp8)
SWMMAC_TRAITS(SwmmacF32Fp8Bf8, v32e4m3_native, v64e5m2_native, v8f_native,
              amdgcn_swmmac_f32_16x16x128_fp8_bf8)
SWMMAC_TRAITS(SwmmacF32Bf8Fp8, v32e5m2_native, v64e4m3_native, v8f_native,
              amdgcn_swmmac_f32_16x16x128_bf8_fp8)
SWMMAC_TRAITS(SwmmacF32Bf8Bf8, v32e5m2_native, v64e5m2_native, v8f_native,
              amdgcn_swmmac_f32_16x16x128_bf8_bf8)
SWMMAC_TRAITS(SwmmacF16Fp8Fp8, v32e4m3_native, v64e4m3_native, v8h_native,
              amdgcn_swmmac_f16_16x16x128_fp8_fp8)
SWMMAC_TRAITS(SwmmacF16Bf8Bf8, v32e5m2_native, v64e5m2_native, v8h_native,
              amdgcn_swmmac_f16_16x16x128_bf8_bf8)
#endif
