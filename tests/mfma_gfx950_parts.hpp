// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/mfma_gfx950_parts.hpp
//
// Shared utilities for the split gfx950 (CDNA4 / MI350) MFMA wrapper tests in
// fpsan/amdgcn_mfma.hpp. Same two-property structure as wmma_gfx12_test.cpp,
// but for the wave64 CDNA MFMA fragment layout (input_loc / output_loc,
// validated against real silicon -- these tests are the gfx950 source of truth,
// not any simulator):
//
//  (1) LayoutMatchesHardware (Native mode): we load the A/B/C fragments into the
//      per-lane registers using the ported input_loc / output_loc maps, call
//      the real __builtin_amdgcn_mfma_*, read D back through output_loc, and
//      compare against a host integer reference D = C + A*B.  Inputs are exact
//      small integers, so the accumulation is exact and the comparison is
//      bit-for-bit.  Because the builtin embodies the *true* hardware layout,
//      this passes only if both input_loc and output_loc match silicon.
//
//  (2) FpsanMatchesScalarReference (FPSan mode): the shipped FPSan software
//      dataflow (detail::mfma_software) must match an independent host scalar
//      FPSan reference matmul computed in the payload ring, payload for
//      payload. The maps cancel (used on both ends), so this isolates the
//      gather + payload arithmetic; the layout itself is pinned by (1).
//
// A templated Harness<Traits> drives both; each shape is a ~12-line Traits.
//
// Requires real MI350 (gfx950) hardware; built only under FPSAN_ENABLE_HIP with
// gfx950 in CMAKE_HIP_ARCHITECTURES.
#include "fpsan/amdgcn_mfma.hpp"
#include "fpsan/amdgcn_smfmac.hpp"
#include "fpsan/fpsan.hpp"

#include "fpsan_semantics.hpp"
#include "hip_test_utils.hpp"
#include "test_random.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

static constexpr Conversions kCC = Conversions::Explicit;
static constexpr int WAVE = 64;

template <class T, class = void> struct MfmaTraitCBSZ : std::integral_constant<int, 0> {};
template <class T>
struct MfmaTraitCBSZ<T, std::void_t<decltype(T::CBSZ)>> : std::integral_constant<int, T::CBSZ> {};
template <class T, class = void> struct MfmaTraitABID : std::integral_constant<int, 0> {};
template <class T>
struct MfmaTraitABID<T, std::void_t<decltype(T::ABID)>> : std::integral_constant<int, T::ABID> {};
template <class T, class = void> struct MfmaTraitBLGP : std::integral_constant<int, 0> {};
template <class T>
struct MfmaTraitBLGP<T, std::void_t<decltype(T::BLGP)>> : std::integral_constant<int, T::BLGP> {};

[[maybe_unused]] static int host_mfma_a_src_lane(int lane, int cbsz, int abid) {
  if (cbsz == 0)
    return lane;
  const int block = WAVE / (1 << cbsz);
  return (lane % block) + block * abid;
}

[[maybe_unused]] static int host_mfma_b_src_lane(int lane, int blgp) {
  switch (blgp) {
  case 0:
    return lane;
  case 1:
    return lane % 32;
  case 2:
    return (lane % 32) + 32;
  case 3:
    return (lane + 16) % WAVE;
  case 4:
    return lane % 16;
  case 5:
    return (lane % 16) + 16;
  case 6:
    return (lane % 16) + 32;
  case 7:
    return (lane % 16) + 48;
  default:
    return lane;
  }
}

// ---------------------------------------------------------------------------
// Harness shared by every dense MFMA shape whose A/B operands are packed
// per-lane vector fragments (v8 of f16/bf16/fp8).  A Traits declares the
// native fragment types, the shape (M,N,K) and operand width InBits, value
// ranges, and a semantics-generic call() dispatching to the fpsan:: wrapper.
// ---------------------------------------------------------------------------
template <class Traits> struct Harness {
  using AVec = typename Traits::AVec;
  using BVec = typename Traits::BVec;
  using CVec = typename Traits::CVec;
  using AElem = fpsan::detail::vector_element_t<AVec>;
  using BElem = fpsan::detail::vector_element_t<BVec>;
  using CElem = fpsan::detail::vector_element_t<CVec>;
  using CBits = typename fpsan::detail::fp_traits<CElem>::bits_type;
  static constexpr int M = Traits::M, N = Traits::N, K = Traits::K;
  static constexpr int InBits = Traits::InBits;
  static constexpr int per_dword = 32 / InBits;
  static constexpr int CBSZ = MfmaTraitCBSZ<Traits>::value;
  static constexpr int ABID = MfmaTraitABID<Traits>::value;
  static constexpr int BLGP = MfmaTraitBLGP<Traits>::value;
};

// Pack this lane's A/B/C fragments from row-major logical matrices, using the
// hardware fragment layout (input_loc for A/B, output_loc_32 for C). Shared by
// the Float and FPSan kernels; the Value ctor embeds for FPSan automatically.
template <class Traits, Semantics S>
__device__ void load_frags(const typename Harness<Traits>::AElem *A,
                           const typename Harness<Traits>::BElem *B,
                           const typename Harness<Traits>::CElem *C, int lane,
                           Value<typename Harness<Traits>::AVec, S, kCC> &a,
                           Value<typename Harness<Traits>::BVec, S, kCC> &b,
                           Value<typename Harness<Traits>::CVec, S, kCC> &c) {
  using H = Harness<Traits>;
  typename H::AVec an{};
  typename H::BVec bn{};
  typename H::CVec cn{};
  // A[i][k]: dim = M, outer index = i (row).
  for (int i = 0; i < H::M; ++i)
    for (int k = 0; k < H::K; ++k) {
      auto loc = fpsan::detail::input_loc(H::M, H::K, 1, i, k, 0, H::InBits);
      if (loc.lane == lane)
        an[H::per_dword * loc.reg + loc.sub] = A[i * H::K + k];
    }
  // B[k][j]: dim = N, outer index = j (column).
  for (int j = 0; j < H::N; ++j)
    for (int k = 0; k < H::K; ++k) {
      auto loc = fpsan::detail::input_loc(H::N, H::K, 1, j, k, 0, H::InBits);
      if (loc.lane == lane)
        bn[H::per_dword * loc.reg + loc.sub] = B[k * H::N + j];
    }
  // C[i][j]: output layout.
  for (int i = 0; i < H::M; ++i)
    for (int j = 0; j < H::N; ++j) {
      auto loc = fpsan::detail::output_loc_32(H::M, H::N, i, j, 0);
      if (loc.lane == lane)
        cn[loc.reg] = C[i * H::N + j];
    }
  a = Value<typename H::AVec, S, kCC>(an);
  b = Value<typename H::BVec, S, kCC>(bn);
  c = Value<typename H::CVec, S, kCC>(cn);
}

template <class Traits>
typename Harness<Traits>::AElem
host_mfma_a_value(const std::vector<typename Harness<Traits>::AElem> &A, int i, int k) {
  using H = Harness<Traits>;
  auto want = fpsan::detail::input_loc(H::M, H::K, 1, i, k, 0, H::InBits);
  const int src_lane = host_mfma_a_src_lane(want.lane, H::CBSZ, H::ABID);
  for (int ii = 0; ii < H::M; ++ii)
    for (int kk = 0; kk < H::K; ++kk) {
      auto loc = fpsan::detail::input_loc(H::M, H::K, 1, ii, kk, 0, H::InBits);
      if (loc.lane == src_lane && loc.reg == want.reg && loc.sub == want.sub)
        return A[ii * H::K + kk];
    }
  return typename H::AElem{};
}

template <class Traits>
typename Harness<Traits>::BElem
host_mfma_b_value(const std::vector<typename Harness<Traits>::BElem> &B, int k, int j) {
  using H = Harness<Traits>;
  auto want = fpsan::detail::input_loc(H::N, H::K, 1, j, k, 0, H::InBits);
  const int src_lane = host_mfma_b_src_lane(want.lane, H::BLGP);
  for (int jj = 0; jj < H::N; ++jj)
    for (int kk = 0; kk < H::K; ++kk) {
      auto loc = fpsan::detail::input_loc(H::N, H::K, 1, jj, kk, 0, H::InBits);
      if (loc.lane == src_lane && loc.reg == want.reg && loc.sub == want.sub)
        return B[kk * H::N + jj];
    }
  return typename H::BElem{};
}

// Native-mode kernel: calls the real builtin (Native mode) and writes D as
// element-type values to a row-major host buffer through output_loc_32.
template <class Traits>
__global__ void
k_builtin(const typename Harness<Traits>::AElem *A, const typename Harness<Traits>::BElem *B,
          const typename Harness<Traits>::CElem *C, typename Harness<Traits>::CElem *D) {
  using H = Harness<Traits>;
  int lane = threadIdx.x;
  Value<typename H::AVec, Semantics::Native, kCC> a;
  Value<typename H::BVec, Semantics::Native, kCC> b;
  Value<typename H::CVec, Semantics::Native, kCC> c;
  load_frags<Traits, Semantics::Native>(A, B, C, lane, a, b, c);
  auto d = Traits::template call<Semantics::Native, kCC>(a, b, c);
  for (int i = 0; i < H::M; ++i)
    for (int j = 0; j < H::N; ++j) {
      auto loc = fpsan::detail::output_loc_32(H::M, H::N, i, j, 0);
      if (loc.lane == lane)
        D[i * H::N + j] = d.get(loc.reg).to_float();
    }
}

// FPSan-mode kernel: writes per-element FPSan payloads through output_loc_32.
template <class Traits, Semantics S>
__global__ void
k_fpsan(const typename Harness<Traits>::AElem *A, const typename Harness<Traits>::BElem *B,
        const typename Harness<Traits>::CElem *C, typename Harness<Traits>::CBits *Dpay) {
  using H = Harness<Traits>;
  int lane = threadIdx.x;
  Value<typename H::AVec, S, kCC> a;
  Value<typename H::BVec, S, kCC> b;
  Value<typename H::CVec, S, kCC> c;
  load_frags<Traits, S>(A, B, C, lane, a, b, c);
  auto d = Traits::template call<S, kCC>(a, b, c);
  for (int i = 0; i < H::M; ++i)
    for (int j = 0; j < H::N; ++j) {
      auto loc = fpsan::detail::output_loc_32(H::M, H::N, i, j, 0);
      if (loc.lane == lane)
        Dpay[i * H::N + j] = d.get(loc.reg).fpsan_payload();
    }
}

namespace {

template <class Traits> struct Mats {
  std::vector<typename Harness<Traits>::AElem> A;
  std::vector<typename Harness<Traits>::BElem> B;
  std::vector<typename Harness<Traits>::CElem> C;
};
template <class Traits> Mats<Traits> make_inputs() {
  using H = Harness<Traits>;
  Mats<Traits> m;
  m.A.resize(H::M * H::K);
  m.B.resize(H::K * H::N);
  m.C.resize(H::M * H::N);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : m.A)
    x = fpsan_test::pick_int_valued<typename H::AElem>(rng, Traits::a_lo, Traits::a_hi);
  for (auto &x : m.B)
    x = fpsan_test::pick_int_valued<typename H::BElem>(rng, Traits::b_lo, Traits::b_hi);
  for (auto &x : m.C)
    x = fpsan_test::pick_int_valued<typename H::CElem>(rng, Traits::c_lo, Traits::c_hi);
  return m;
}
} // namespace

// (1) Layout test: real builtin (Float) vs host integer reference.
template <class Traits> void run_layout_matches_hardware() {
  using H = Harness<Traits>;
  using AE = typename H::AElem;
  using BE = typename H::BElem;
  using CE = typename H::CElem;
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  Mats<Traits> m = make_inputs<Traits>();

  // Host integer reference: exact because inputs are small integers.
  std::vector<CE> ref(H::M * H::N);
  for (int i = 0; i < H::M; ++i)
    for (int j = 0; j < H::N; ++j) {
      double acc = static_cast<double>(static_cast<float>(m.C[i * H::N + j]));
      for (int k = 0; k < H::K; ++k)
        acc += static_cast<double>(static_cast<float>(host_mfma_a_value<Traits>(m.A, i, k))) *
               static_cast<double>(static_cast<float>(host_mfma_b_value<Traits>(m.B, k, j)));
      ref[i * H::N + j] = static_cast<CE>(static_cast<float>(acc));
    }

  AE *dA = to_dev(m.A);
  BE *dB = to_dev(m.B);
  CE *dC = to_dev(m.C);
  CE *dD;
  HIP_CHECK(hipMalloc(&dD, H::M * H::N * sizeof(CE)));
  k_builtin<Traits><<<1, WAVE>>>(dA, dB, dC, dD);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<CE> got(H::M * H::N);
  HIP_CHECK(hipMemcpy(got.data(), dD, H::M * H::N * sizeof(CE), hipMemcpyDeviceToHost));
  for (int i = 0; i < H::M * H::N; ++i)
    EXPECT_EQ(bits_of(got[i]), bits_of(ref[i]))
        << "layout mismatch at " << (i / H::N) << "," << (i % H::N);
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dD);
}

// (2) Payload test: shipped FPSan dataflow vs host scalar FPSan reference, in one
// semantics S (self-consistency); looped over all FPSan-family semantics below.
template <class Traits, Semantics S> void run_fpsan_matches_scalar_reference() {
  using H = Harness<Traits>;
  using AE = typename H::AElem;
  using BE = typename H::BElem;
  using CE = typename H::CElem;
  using CBits = typename H::CBits;
  Mats<Traits> m = make_inputs<Traits>();

  using VA = Value<AE, S, kCC>;
  using VB = Value<BE, S, kCC>;
  using VC = Value<CE, S, kCC>;
  std::vector<CBits> ref(H::M * H::N);
  for (int i = 0; i < H::M; ++i)
    for (int j = 0; j < H::N; ++j) {
      VC acc(m.C[i * H::N + j]);
      for (int k = 0; k < H::K; ++k)
        acc = acc + fpsan::cast<CE>(VA(host_mfma_a_value<Traits>(m.A, i, k))) *
                        fpsan::cast<CE>(VB(host_mfma_b_value<Traits>(m.B, k, j)));
      ref[i * H::N + j] = acc.fpsan_payload();
    }

  AE *dA = to_dev(m.A);
  BE *dB = to_dev(m.B);
  CE *dC = to_dev(m.C);
  CBits *dD;
  HIP_CHECK(hipMalloc(&dD, H::M * H::N * sizeof(CBits)));
  k_fpsan<Traits, S><<<1, WAVE>>>(dA, dB, dC, dD);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<CBits> got(H::M * H::N);
  HIP_CHECK(hipMemcpy(got.data(), dD, H::M * H::N * sizeof(CBits), hipMemcpyDeviceToHost));
  for (int i = 0; i < H::M * H::N; ++i)
    EXPECT_EQ(got[i], ref[i]) << "payload mismatch at " << (i / H::N) << "," << (i % H::N);
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dD);
}

// Run the payload test for representative FPSan-family semantics (Triton + one
// algebraic field).
template <class Traits> void run_fpsan_matches_scalar_reference_all() {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  fpsan_test::for_matrix_fpsan_semantics(
      [](auto sem) { run_fpsan_matches_scalar_reference<Traits, decltype(sem)::value>(); });
}

// ---------------------------------------------------------------------------
// Per-shape Traits + TESTs.  Each gets both a LayoutMatchesHardware and a
// FpsanMatchesScalarReference test.
// ---------------------------------------------------------------------------
using fpsan::v16f_native;
using fpsan::v4f_native;
using fpsan::v8bf_native;
using fpsan::v8e4m3_native;
using fpsan::v8e5m2_native;
using fpsan::v8h_native;
