// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/mfma_cdna3_parts.hpp
//
// Shared utilities for the split CDNA3 / gfx94x MFMA wrapper tests. The concrete
// TEST instantiations live in the mfma_cdna3_*_test.cpp files.
#include "fpsan/amdgcn_mfma.hpp"
#include "fpsan/fpsan.hpp"

#include "fpsan_semantics.hpp"

#include "hip_test_utils.hpp"
#include "test_random.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <random>
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
// Dense vector-input MFMA shapes: f16, bf16, fp8/bf8. A/B are per-lane vector
// fragments, C/D are f32 vector accumulators. Bk is the independent block count
// used by the legacy small-K shapes.
// ---------------------------------------------------------------------------
template <class Traits> struct DenseVecHarness {
  using AVec = typename Traits::AVec;
  using BVec = typename Traits::BVec;
  using CVec = typename Traits::CVec;
  using AElem = fpsan::detail::vector_element_t<AVec>;
  using BElem = fpsan::detail::vector_element_t<BVec>;
  using CElem = fpsan::detail::vector_element_t<CVec>;
  using CBits = typename fpsan::detail::fp_traits<CElem>::bits_type;
  static constexpr int M = Traits::M, N = Traits::N, K = Traits::K, Bk = Traits::Bk;
  static constexpr int InBits = Traits::InBits;
  static constexpr int per_dword = 32 / InBits;
  static constexpr int CBSZ = MfmaTraitCBSZ<Traits>::value;
  static constexpr int ABID = MfmaTraitABID<Traits>::value;
  static constexpr int BLGP = MfmaTraitBLGP<Traits>::value;
};

template <class Traits> struct DenseVecData {
  using H = DenseVecHarness<Traits>;
  std::vector<typename H::AElem> A;
  std::vector<typename H::BElem> B;
  std::vector<typename H::CElem> C;
};

template <class Traits> DenseVecData<Traits> make_dense_vec_inputs() {
  using H = DenseVecHarness<Traits>;
  DenseVecData<Traits> d;
  d.A.resize(H::Bk * H::M * H::K);
  d.B.resize(H::Bk * H::K * H::N);
  d.C.resize(H::Bk * H::M * H::N);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : d.A)
    x = fpsan_test::pick_int_valued<typename H::AElem>(rng, Traits::a_lo, Traits::a_hi);
  for (auto &x : d.B)
    x = fpsan_test::pick_int_valued<typename H::BElem>(rng, Traits::b_lo, Traits::b_hi);
  for (auto &x : d.C)
    x = fpsan_test::pick_int_valued<typename H::CElem>(rng, Traits::c_lo, Traits::c_hi);
  return d;
}

template <class Traits, Semantics S>
__device__ void load_dense_vec(const typename DenseVecHarness<Traits>::AElem *A,
                               const typename DenseVecHarness<Traits>::BElem *B,
                               const typename DenseVecHarness<Traits>::CElem *C, int lane,
                               Value<typename DenseVecHarness<Traits>::AVec, S, kCC> &a,
                               Value<typename DenseVecHarness<Traits>::BVec, S, kCC> &b,
                               Value<typename DenseVecHarness<Traits>::CVec, S, kCC> &c) {
  using H = DenseVecHarness<Traits>;
  typename H::AVec an{};
  typename H::BVec bn{};
  typename H::CVec cn{};
  for (int blk = 0; blk < H::Bk; ++blk)
    for (int i = 0; i < H::M; ++i)
      for (int k = 0; k < H::K; ++k) {
        auto loc = fpsan::detail::input_loc(H::M, H::K, H::Bk, i, k, blk, H::InBits);
        if (loc.lane == lane)
          an[H::per_dword * loc.reg + loc.sub] = A[(blk * H::M + i) * H::K + k];
      }
  for (int blk = 0; blk < H::Bk; ++blk)
    for (int j = 0; j < H::N; ++j)
      for (int k = 0; k < H::K; ++k) {
        auto loc = fpsan::detail::input_loc(H::N, H::K, H::Bk, j, k, blk, H::InBits);
        if (loc.lane == lane)
          bn[H::per_dword * loc.reg + loc.sub] = B[(blk * H::K + k) * H::N + j];
      }
  for (int blk = 0; blk < H::Bk; ++blk)
    for (int i = 0; i < H::M; ++i)
      for (int j = 0; j < H::N; ++j) {
        auto loc = fpsan::detail::output_loc_32(H::M, H::N, i, j, blk);
        if (loc.lane == lane)
          cn[loc.reg] = C[(blk * H::M + i) * H::N + j];
      }
  a = Value<typename H::AVec, S, kCC>(an);
  b = Value<typename H::BVec, S, kCC>(bn);
  c = Value<typename H::CVec, S, kCC>(cn);
}

template <class Traits>
typename DenseVecHarness<Traits>::AElem
host_dense_vec_a_value(const std::vector<typename DenseVecHarness<Traits>::AElem> &A, int blk,
                       int i, int k) {
  using H = DenseVecHarness<Traits>;
  auto want = fpsan::detail::input_loc(H::M, H::K, H::Bk, i, k, blk, H::InBits);
  const int src_lane = host_mfma_a_src_lane(want.lane, H::CBSZ, H::ABID);
  for (int bb = 0; bb < H::Bk; ++bb)
    for (int ii = 0; ii < H::M; ++ii)
      for (int kk = 0; kk < H::K; ++kk) {
        auto loc = fpsan::detail::input_loc(H::M, H::K, H::Bk, ii, kk, bb, H::InBits);
        if (loc.lane == src_lane && loc.reg == want.reg && loc.sub == want.sub)
          return A[(bb * H::M + ii) * H::K + kk];
      }
  return typename H::AElem{};
}

template <class Traits>
typename DenseVecHarness<Traits>::BElem
host_dense_vec_b_value(const std::vector<typename DenseVecHarness<Traits>::BElem> &B, int blk,
                       int k, int j) {
  using H = DenseVecHarness<Traits>;
  auto want = fpsan::detail::input_loc(H::N, H::K, H::Bk, j, k, blk, H::InBits);
  const int src_lane = host_mfma_b_src_lane(want.lane, H::BLGP);
  for (int bb = 0; bb < H::Bk; ++bb)
    for (int jj = 0; jj < H::N; ++jj)
      for (int kk = 0; kk < H::K; ++kk) {
        auto loc = fpsan::detail::input_loc(H::N, H::K, H::Bk, jj, kk, bb, H::InBits);
        if (loc.lane == src_lane && loc.reg == want.reg && loc.sub == want.sub)
          return B[(bb * H::K + kk) * H::N + jj];
      }
  return typename H::BElem{};
}

template <class Traits>
__global__ void k_dense_vec_float(const typename DenseVecHarness<Traits>::AElem *A,
                                  const typename DenseVecHarness<Traits>::BElem *B,
                                  const typename DenseVecHarness<Traits>::CElem *C,
                                  typename DenseVecHarness<Traits>::CElem *D) {
  using H = DenseVecHarness<Traits>;
  Value<typename H::AVec, Semantics::Native, kCC> a;
  Value<typename H::BVec, Semantics::Native, kCC> b;
  Value<typename H::CVec, Semantics::Native, kCC> c;
  const int lane = threadIdx.x;
  load_dense_vec<Traits, Semantics::Native>(A, B, C, lane, a, b, c);
  auto d = Traits::template call<Semantics::Native, kCC>(a, b, c);
  for (int blk = 0; blk < H::Bk; ++blk)
    for (int i = 0; i < H::M; ++i)
      for (int j = 0; j < H::N; ++j) {
        auto loc = fpsan::detail::output_loc_32(H::M, H::N, i, j, blk);
        if (loc.lane == lane)
          D[(blk * H::M + i) * H::N + j] = d.get(loc.reg).to_float();
      }
}

template <class Traits, Semantics S>
__global__ void k_dense_vec_fpsan(const typename DenseVecHarness<Traits>::AElem *A,
                                  const typename DenseVecHarness<Traits>::BElem *B,
                                  const typename DenseVecHarness<Traits>::CElem *C,
                                  typename DenseVecHarness<Traits>::CBits *D) {
  using H = DenseVecHarness<Traits>;
  Value<typename H::AVec, S, kCC> a;
  Value<typename H::BVec, S, kCC> b;
  Value<typename H::CVec, S, kCC> c;
  const int lane = threadIdx.x;
  load_dense_vec<Traits, S>(A, B, C, lane, a, b, c);
  auto d = Traits::template call<S, kCC>(a, b, c);
  for (int blk = 0; blk < H::Bk; ++blk)
    for (int i = 0; i < H::M; ++i)
      for (int j = 0; j < H::N; ++j) {
        auto loc = fpsan::detail::output_loc_32(H::M, H::N, i, j, blk);
        if (loc.lane == lane)
          D[(blk * H::M + i) * H::N + j] = d.get(loc.reg).fpsan_payload();
      }
}

template <class Traits> void run_dense_vec_layout() {
  using H = DenseVecHarness<Traits>;
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  auto d = make_dense_vec_inputs<Traits>();
  std::vector<typename H::CElem> ref(H::Bk * H::M * H::N);
  for (int blk = 0; blk < H::Bk; ++blk)
    for (int i = 0; i < H::M; ++i)
      for (int j = 0; j < H::N; ++j) {
        double acc = static_cast<double>(static_cast<float>(d.C[(blk * H::M + i) * H::N + j]));
        for (int k = 0; k < H::K; ++k)
          acc += static_cast<double>(
                     static_cast<float>(host_dense_vec_a_value<Traits>(d.A, blk, i, k))) *
                 static_cast<double>(
                     static_cast<float>(host_dense_vec_b_value<Traits>(d.B, blk, k, j)));
        ref[(blk * H::M + i) * H::N + j] = static_cast<typename H::CElem>(static_cast<float>(acc));
      }
  auto *dA = to_dev(d.A);
  auto *dB = to_dev(d.B);
  auto *dC = to_dev(d.C);
  typename H::CElem *dD;
  HIP_CHECK(hipMalloc(&dD, ref.size() * sizeof(typename H::CElem)));
  k_dense_vec_float<Traits><<<1, WAVE>>>(dA, dB, dC, dD);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<typename H::CElem> got(ref.size());
  HIP_CHECK(
      hipMemcpy(got.data(), dD, got.size() * sizeof(typename H::CElem), hipMemcpyDeviceToHost));
  for (std::size_t i = 0; i < got.size(); ++i)
    EXPECT_EQ(bits_of(got[i]), bits_of(ref[i])) << "elem " << i;
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dD);
}

template <class Traits> void run_dense_vec_fpsan() {
  using H = DenseVecHarness<Traits>;
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  auto d = make_dense_vec_inputs<Traits>();
  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VA = Value<typename H::AElem, S, kCC>;
    using VB = Value<typename H::BElem, S, kCC>;
    using VC = Value<typename H::CElem, S, kCC>;
    std::vector<typename H::CBits> ref(H::Bk * H::M * H::N);
    for (int blk = 0; blk < H::Bk; ++blk)
      for (int i = 0; i < H::M; ++i)
        for (int j = 0; j < H::N; ++j) {
          VC acc(d.C[(blk * H::M + i) * H::N + j]);
          for (int k = 0; k < H::K; ++k)
            acc = acc + fpsan::cast<typename H::CElem>(
                            VA(host_dense_vec_a_value<Traits>(d.A, blk, i, k))) *
                            fpsan::cast<typename H::CElem>(
                                VB(host_dense_vec_b_value<Traits>(d.B, blk, k, j)));
          ref[(blk * H::M + i) * H::N + j] = acc.fpsan_payload();
        }
    auto *dA = to_dev(d.A);
    auto *dB = to_dev(d.B);
    auto *dC = to_dev(d.C);
    typename H::CBits *dD;
    HIP_CHECK(hipMalloc(&dD, ref.size() * sizeof(typename H::CBits)));
    k_dense_vec_fpsan<Traits, S><<<1, WAVE>>>(dA, dB, dC, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<typename H::CBits> got(ref.size());
    HIP_CHECK(
        hipMemcpy(got.data(), dD, got.size() * sizeof(typename H::CBits), hipMemcpyDeviceToHost));
    for (std::size_t i = 0; i < got.size(); ++i)
      EXPECT_EQ(got[i], ref[i]) << "elem " << i;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
  });
}

#define DENSE_VEC_TESTS(Name)                                                                      \
  TEST(Name, LayoutMatchesHardware) { run_dense_vec_layout<Name>(); }                              \
  TEST(Name, FpsanMatchesScalarReference) { run_dense_vec_fpsan<Name>(); }

#define DENSE_VEC_TRAITS(Name, M_, N_, K_, Bk_, InBits_, AV_, BV_, CV_, WRAP_)                     \
  struct Name {                                                                                    \
    using AVec = AV_;                                                                              \
    using BVec = BV_;                                                                              \
    using CVec = CV_;                                                                              \
    static constexpr int M = M_, N = N_, K = K_, Bk = Bk_, InBits = InBits_;                       \
    static constexpr int a_lo = -3, a_hi = 3, b_lo = -2, b_hi = 2, c_lo = -4, c_hi = 4;            \
    template <Semantics S, Conversions C>                                                          \
    __device__ static Value<CVec, S, C> call(Value<AVec, S, C> a, Value<BVec, S, C> b,             \
                                             Value<CVec, S, C> c) {                                \
      return fpsan::WRAP_<0, 0, 0, S, C>(a, b, c);                                                 \
    }                                                                                              \
  };                                                                                               \
  DENSE_VEC_TESTS(Name)

using fpsan::v16f_native;
using fpsan::v2f_native;
using fpsan::v32f_native;
using fpsan::v4bf_native;
using fpsan::v4f_native;
using fpsan::v4h_native;
using fpsan::v8amd_e4m3_native;
using fpsan::v8amd_e5m2_native;

// ---------------------------------------------------------------------------
// Dense scalar f32-input MFMA shapes.
// ---------------------------------------------------------------------------
template <class Traits> struct DenseF32Data {
  std::vector<float> A, B, C;
};

template <class Traits> DenseF32Data<Traits> make_dense_f32_inputs() {
  DenseF32Data<Traits> d;
  d.A.resize(Traits::Bk * Traits::M * Traits::K);
  d.B.resize(Traits::Bk * Traits::K * Traits::N);
  d.C.resize(Traits::Bk * Traits::M * Traits::N);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : d.A)
    x = fpsan_test::pick_int_valued<float>(rng, -3, 3);
  for (auto &x : d.B)
    x = fpsan_test::pick_int_valued<float>(rng, -2, 2);
  for (auto &x : d.C)
    x = fpsan_test::pick_int_valued<float>(rng, -4, 4);
  return d;
}

template <class Traits, Semantics S, class Out>
__global__ void k_dense_f32(const float *A, const float *B, const float *C, Out *D) {
  using T = Traits;
  int lane = threadIdx.x;
  float an = 0.0f, bn = 0.0f;
  for (int blk = 0; blk < T::Bk; ++blk)
    for (int i = 0; i < T::M; ++i)
      for (int k = 0; k < T::K; ++k) {
        auto loc = fpsan::detail::input_loc(T::M, T::K, T::Bk, i, k, blk, 32);
        if (loc.lane == lane)
          an = A[(blk * T::M + i) * T::K + k];
      }
  for (int blk = 0; blk < T::Bk; ++blk)
    for (int j = 0; j < T::N; ++j)
      for (int k = 0; k < T::K; ++k) {
        auto loc = fpsan::detail::input_loc(T::N, T::K, T::Bk, j, k, blk, 32);
        if (loc.lane == lane)
          bn = B[(blk * T::K + k) * T::N + j];
      }
  typename T::CVec cn{};
  for (int blk = 0; blk < T::Bk; ++blk)
    for (int i = 0; i < T::M; ++i)
      for (int j = 0; j < T::N; ++j) {
        auto loc = fpsan::detail::output_loc_32(T::M, T::N, i, j, blk);
        if (loc.lane == lane)
          cn[loc.reg] = C[(blk * T::M + i) * T::N + j];
      }
  Value<float, S, kCC> a{an}, b{bn};
  Value<typename T::CVec, S, kCC> c{cn};
  auto d = T::template call<S, kCC>(a, b, c);
  for (int blk = 0; blk < T::Bk; ++blk)
    for (int i = 0; i < T::M; ++i)
      for (int j = 0; j < T::N; ++j) {
        auto loc = fpsan::detail::output_loc_32(T::M, T::N, i, j, blk);
        if (loc.lane == lane) {
          if constexpr (S == Semantics::Native)
            D[(blk * T::M + i) * T::N + j] = d.get(loc.reg).to_float();
          else
            D[(blk * T::M + i) * T::N + j] = d.get(loc.reg).fpsan_payload();
        }
      }
}

template <class Traits> void run_dense_f32_layout() {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  auto d = make_dense_f32_inputs<Traits>();
  std::vector<float> ref(Traits::Bk * Traits::M * Traits::N);
  for (int blk = 0; blk < Traits::Bk; ++blk)
    for (int i = 0; i < Traits::M; ++i)
      for (int j = 0; j < Traits::N; ++j) {
        double acc = d.C[(blk * Traits::M + i) * Traits::N + j];
        for (int k = 0; k < Traits::K; ++k)
          acc += static_cast<double>(d.A[(blk * Traits::M + i) * Traits::K + k]) *
                 static_cast<double>(d.B[(blk * Traits::K + k) * Traits::N + j]);
        ref[(blk * Traits::M + i) * Traits::N + j] = static_cast<float>(acc);
      }
  float *dA = to_dev(d.A), *dB = to_dev(d.B), *dC = to_dev(d.C), *dD;
  HIP_CHECK(hipMalloc(&dD, ref.size() * sizeof(float)));
  k_dense_f32<Traits, Semantics::Native, float><<<1, WAVE>>>(dA, dB, dC, dD);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> got(ref.size());
  HIP_CHECK(hipMemcpy(got.data(), dD, got.size() * sizeof(float), hipMemcpyDeviceToHost));
  for (std::size_t i = 0; i < got.size(); ++i)
    EXPECT_EQ(bits_of(got[i]), bits_of(ref[i])) << "elem " << i;
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dD);
}

template <class Traits> void run_dense_f32_fpsan() {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  auto d = make_dense_f32_inputs<Traits>();
  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VF = Value<float, S, kCC>;
    std::vector<std::uint32_t> ref(Traits::Bk * Traits::M * Traits::N);
    for (int blk = 0; blk < Traits::Bk; ++blk)
      for (int i = 0; i < Traits::M; ++i)
        for (int j = 0; j < Traits::N; ++j) {
          VF acc(d.C[(blk * Traits::M + i) * Traits::N + j]);
          for (int k = 0; k < Traits::K; ++k)
            acc = acc + VF(d.A[(blk * Traits::M + i) * Traits::K + k]) *
                            VF(d.B[(blk * Traits::K + k) * Traits::N + j]);
          ref[(blk * Traits::M + i) * Traits::N + j] = acc.fpsan_payload();
        }
    float *dA = to_dev(d.A), *dB = to_dev(d.B), *dC = to_dev(d.C);
    std::uint32_t *dD;
    HIP_CHECK(hipMalloc(&dD, ref.size() * sizeof(std::uint32_t)));
    k_dense_f32<Traits, S, std::uint32_t><<<1, WAVE>>>(dA, dB, dC, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(ref.size());
    HIP_CHECK(hipMemcpy(got.data(), dD, got.size() * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for (std::size_t i = 0; i < got.size(); ++i)
      EXPECT_EQ(got[i], ref[i]) << "elem " << i;
    (void)hipFree(dA);
    (void)hipFree(dB);
    (void)hipFree(dC);
    (void)hipFree(dD);
  });
}

#define DENSE_F32_TRAITS(Name, M_, N_, K_, Bk_, CVec_, WRAP_)                                      \
  struct Name {                                                                                    \
    using CVec = CVec_;                                                                            \
    static constexpr int M = M_, N = N_, K = K_, Bk = Bk_;                                         \
    template <Semantics S, Conversions C>                                                          \
    __device__ static Value<CVec, S, C> call(Value<float, S, C> a, Value<float, S, C> b,           \
                                             Value<CVec, S, C> c) {                                \
      return fpsan::WRAP_<0, 0, 0, S, C>(a, b, c);                                                 \
    }                                                                                              \
  };                                                                                               \
  TEST(Name, LayoutMatchesHardware) { run_dense_f32_layout<Name>(); }                              \
  TEST(Name, FpsanMatchesScalarReference) { run_dense_f32_fpsan<Name>(); }
