// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/atomic_test.cpp
//
// GPU tests for the FP atomics in fpsan/amdgcn_atomic.hpp. Each test launches
// many lanes that hammer a single shared location with an atomic; we check
// both the final value and the bit-for-bit equivalence of the FPSan path with
// a host scalar reference applied in some canonical order. The point of FPSan
// for atomics: the FPSan answer doesn't depend on the contention-determined
// order, because the wrapper applies the selected Value<> operation atomically.
#include "fpsan/amdgcn_atomic.hpp"
#include "fpsan/fpsan.hpp"

#include "fpsan_semantics.hpp"
#include "hip_test_utils.hpp"
#include "test_random.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

static constexpr Conversions kCC = Conversions::Explicit;
static constexpr int LANES = 256;

// Each lane atomic-adds its f32 input into a single shared slot.
template <Semantics S> __global__ void k_atomic_fadd(Value<float, S, kCC> *slot, const float *in) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  (void)fpsan::amdgcn_atomic_fadd_f32(slot, Value<float, S, kCC>{in[i]});
}

template <Semantics S> __global__ void k_atomic_fmin(Value<float, S, kCC> *slot, const float *in) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  (void)fpsan::amdgcn_atomic_fmin_f32(slot, Value<float, S, kCC>{in[i]});
}

template <Semantics S> __global__ void k_atomic_fmax(Value<float, S, kCC> *slot, const float *in) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  (void)fpsan::amdgcn_atomic_fmax_f32(slot, Value<float, S, kCC>{in[i]});
}

// f64 variants (same atomics, 64-bit payload).
template <Semantics S>
__global__ void k_atomic_fadd64(Value<double, S, kCC> *slot, const double *in) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  (void)fpsan::amdgcn_atomic_fadd_f64(slot, Value<double, S, kCC>{in[i]});
}
template <Semantics S>
__global__ void k_atomic_fmin64(Value<double, S, kCC> *slot, const double *in) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  (void)fpsan::amdgcn_atomic_fmin_f64(slot, Value<double, S, kCC>{in[i]});
}
template <Semantics S>
__global__ void k_atomic_fmax64(Value<double, S, kCC> *slot, const double *in) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  (void)fpsan::amdgcn_atomic_fmax_f64(slot, Value<double, S, kCC>{in[i]});
}

// Packed v2f16 / v2bf16 atomic-add: each lane adds {in[2i], in[2i+1]} into a
// single packed slot. Element-wise accumulation, no cross-element carry.
using v2h_t = _Float16 __attribute__((ext_vector_type(2)));
using v2bf_t = __bf16 __attribute__((ext_vector_type(2)));

template <Semantics S>
__global__ void k_atomic_pk_f16(Value<v2h_t, S, kCC> *slot, const float *in) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  Value<v2h_t, S, kCC> v{};
  v.set(0, Value<_Float16, S, kCC>{static_cast<_Float16>(in[2 * i])});
  v.set(1, Value<_Float16, S, kCC>{static_cast<_Float16>(in[2 * i + 1])});
  (void)fpsan::amdgcn_atomic_pk_add_f16(slot, v);
}
template <Semantics S>
__global__ void k_atomic_pk_bf16(Value<v2bf_t, S, kCC> *slot, const float *in) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  Value<v2bf_t, S, kCC> v{};
  v.set(0, Value<__bf16, S, kCC>{static_cast<__bf16>(in[2 * i])});
  v.set(1, Value<__bf16, S, kCC>{static_cast<__bf16>(in[2 * i + 1])});
  (void)fpsan::amdgcn_atomic_pk_add_bf16(slot, v);
}

// ---- helpers ---------------------------------------------------------------
namespace {

std::vector<float> make_inputs() {
  std::vector<float> v(LANES);
  std::mt19937 rng = fpsan_test::make_rng();
  // Small exact integers; sums up to ~LANES*3 = 768 stay exact in f32.
  for (auto &x : v)
    x = fpsan_test::pick_int_valued<float>(rng, -3, 3);
  return v;
}

std::vector<double> make_inputs64() {
  std::vector<double> v(LANES);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : v)
    x = fpsan_test::pick_int_valued<double>(rng, -3, 3);
  return v;
}

} // namespace

// ---- atomic_fadd: exact integer reduction matches sequential sum -----------
TEST(Atomic, FaddFloatExact) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  auto in = make_inputs();
  float *dIn = to_dev(in);
  using V = Value<float, Semantics::Native, kCC>;
  V *dSlot;
  HIP_CHECK(hipMalloc(&dSlot, sizeof(V)));
  V init{0.f};
  HIP_CHECK(hipMemcpy(dSlot, &init, sizeof(V), hipMemcpyHostToDevice));
  k_atomic_fadd<Semantics::Native><<<1, LANES>>>(dSlot, dIn);
  HIP_CHECK(hipDeviceSynchronize());
  V got{0.f};
  HIP_CHECK(hipMemcpy(&got, dSlot, sizeof(V), hipMemcpyDeviceToHost));
  float ref = 0;
  for (float x : in)
    ref += x;
  EXPECT_EQ(static_cast<float>(got), ref);
  (void)hipFree(dIn);
  (void)hipFree(dSlot);
}

TEST(Atomic, FaddFpsanMatchesScalarRingSum) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  // The atomic_fadd wrapper uses a CAS loop with the per-semantics add, so it is
  // correct for every algebraic model (mod n), not just the Triton's mod 2^w.
  auto in = make_inputs();
  float *dIn = to_dev(in);
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    using V = Value<float, decltype(sem)::value, kCC>;
    V *dSlot;
    HIP_CHECK(hipMalloc(&dSlot, sizeof(V)));
    V init{0.f};
    HIP_CHECK(hipMemcpy(dSlot, &init, sizeof(V), hipMemcpyHostToDevice));
    k_atomic_fadd<decltype(sem)::value><<<1, LANES>>>(dSlot, dIn);
    HIP_CHECK(hipDeviceSynchronize());
    V got{0.f};
    HIP_CHECK(hipMemcpy(&got, dSlot, sizeof(V), hipMemcpyDeviceToHost));
    V acc{0.f};
    for (float x : in)
      acc = acc + V{x};
    EXPECT_EQ(got.fpsan_payload(), acc.fpsan_payload());
    (void)hipFree(dSlot);
  });
  (void)hipFree(dIn);
}

// ---- atomic_fmin / atomic_fmax: final value = min / max of all lanes -------
TEST(Atomic, FminFloatMatchesScalarMin) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  auto in = make_inputs();
  float *dIn = to_dev(in);
  using V = Value<float, Semantics::Native, kCC>;
  V *dSlot;
  HIP_CHECK(hipMalloc(&dSlot, sizeof(V)));
  // Seed with +INFINITY so the first observed value sets the min.
  V init{1e30f};
  HIP_CHECK(hipMemcpy(dSlot, &init, sizeof(V), hipMemcpyHostToDevice));
  k_atomic_fmin<Semantics::Native><<<1, LANES>>>(dSlot, dIn);
  HIP_CHECK(hipDeviceSynchronize());
  V got{0.f};
  HIP_CHECK(hipMemcpy(&got, dSlot, sizeof(V), hipMemcpyDeviceToHost));
  float ref = *std::min_element(in.begin(), in.end());
  EXPECT_EQ(static_cast<float>(got), ref);
  (void)hipFree(dIn);
  (void)hipFree(dSlot);
}

TEST(Atomic, FmaxFloatMatchesScalarMax) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  auto in = make_inputs();
  float *dIn = to_dev(in);
  using V = Value<float, Semantics::Native, kCC>;
  V *dSlot;
  HIP_CHECK(hipMalloc(&dSlot, sizeof(V)));
  V init{-1e30f};
  HIP_CHECK(hipMemcpy(dSlot, &init, sizeof(V), hipMemcpyHostToDevice));
  k_atomic_fmax<Semantics::Native><<<1, LANES>>>(dSlot, dIn);
  HIP_CHECK(hipDeviceSynchronize());
  V got{0.f};
  HIP_CHECK(hipMemcpy(&got, dSlot, sizeof(V), hipMemcpyDeviceToHost));
  float ref = *std::max_element(in.begin(), in.end());
  EXPECT_EQ(static_cast<float>(got), ref);
  (void)hipFree(dIn);
  (void)hipFree(dSlot);
}

TEST(Atomic, FminFpsanMatchesScalarMin) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  auto in = make_inputs();
  float *dIn = to_dev(in);
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    using V = Value<float, decltype(sem)::value, kCC>;
    V *dSlot;
    HIP_CHECK(hipMalloc(&dSlot, sizeof(V)));
    V init{1e30f};
    HIP_CHECK(hipMemcpy(dSlot, &init, sizeof(V), hipMemcpyHostToDevice));
    k_atomic_fmin<decltype(sem)::value><<<1, LANES>>>(dSlot, dIn);
    HIP_CHECK(hipDeviceSynchronize());
    V got{0.f};
    HIP_CHECK(hipMemcpy(&got, dSlot, sizeof(V), hipMemcpyDeviceToHost));
    V acc = init;
    for (float x : in)
      acc = fpsan::min(acc, V{x});
    EXPECT_EQ(got.fpsan_payload(), acc.fpsan_payload());
    (void)hipFree(dSlot);
  });
  (void)hipFree(dIn);
}

// ---- atomic_fadd returned-old correctness (single thread, no contention) ----
// The wrapper must return the OLD value at *addr (matching the AMD builtin
// convention). With a single thread, we can pin down the expected old exactly.
__global__ void k_atomic_fadd_returns_old_float(float *slot, float *old1, float *old2) {
  using V = Value<float, Semantics::Native, kCC>;
  auto *v_slot = reinterpret_cast<V *>(slot);
  // First call returns 0 (initial value), second returns 5 (after adding 5).
  *old1 = static_cast<float>(fpsan::amdgcn_atomic_fadd_f32(v_slot, V{5.f}));
  *old2 = static_cast<float>(fpsan::amdgcn_atomic_fadd_f32(v_slot, V{7.f}));
}

TEST(Atomic, FaddReturnsOldFloat) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  float *dSlot, *dOld1, *dOld2;
  HIP_CHECK(hipMalloc(&dSlot, sizeof(float)));
  HIP_CHECK(hipMalloc(&dOld1, sizeof(float)));
  HIP_CHECK(hipMalloc(&dOld2, sizeof(float)));
  float zero = 0.f;
  HIP_CHECK(hipMemcpy(dSlot, &zero, sizeof(float), hipMemcpyHostToDevice));
  k_atomic_fadd_returns_old_float<<<1, 1>>>(dSlot, dOld1, dOld2);
  HIP_CHECK(hipDeviceSynchronize());
  float old1, old2, final;
  HIP_CHECK(hipMemcpy(&old1, dOld1, sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&old2, dOld2, sizeof(float), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&final, dSlot, sizeof(float), hipMemcpyDeviceToHost));
  EXPECT_EQ(old1, 0.f);
  EXPECT_EQ(old2, 5.f);
  EXPECT_EQ(final, 12.f);
  (void)hipFree(dSlot);
  (void)hipFree(dOld1);
  (void)hipFree(dOld2);
}

// FPSan fmin/fmax returned-old correctness. These use explicit payloads so the
// reference is the selected Value<> order, not IEEE numeric order.
template <bool IsMin, Semantics S>
__global__ void k_atomic_fmm_returns_old32(Value<float, S, kCC> *slot, std::uint32_t *old1,
                                           std::uint32_t *old2, std::uint32_t *final,
                                           std::uint32_t rhs1, std::uint32_t rhs2) {
  using V = Value<float, S, kCC>;
  const V v1 = V::from_fpsan_payload(rhs1);
  const V v2 = V::from_fpsan_payload(rhs2);
  const V r1 =
      IsMin ? fpsan::amdgcn_atomic_fmin_f32(slot, v1) : fpsan::amdgcn_atomic_fmax_f32(slot, v1);
  const V r2 =
      IsMin ? fpsan::amdgcn_atomic_fmin_f32(slot, v2) : fpsan::amdgcn_atomic_fmax_f32(slot, v2);
  *old1 = r1.fpsan_payload();
  *old2 = r2.fpsan_payload();
  *final = slot->fpsan_payload();
}

template <bool IsMin, Semantics S> static void run_fpsan_fmm_returns_old32(const char *label) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";

  using V = Value<float, S, kCC>;
  using Bits = typename V::bits_type;
  const Bits initp = IsMin ? Bits{1} : Bits{0};
  const Bits rhs1p = IsMin ? Bits{0} : Bits{1};
  const Bits rhs2p = Bits{2};
  const V init = V::from_fpsan_payload(initp);
  const V after1 = IsMin ? fpsan::min(init, V::from_fpsan_payload(rhs1p))
                         : fpsan::max(init, V::from_fpsan_payload(rhs1p));
  const V after2 = IsMin ? fpsan::min(after1, V::from_fpsan_payload(rhs2p))
                         : fpsan::max(after1, V::from_fpsan_payload(rhs2p));

  V *dSlot;
  Bits *dOld1, *dOld2, *dFinal;
  HIP_CHECK(hipMalloc(&dSlot, sizeof(V)));
  HIP_CHECK(hipMalloc(&dOld1, sizeof(Bits)));
  HIP_CHECK(hipMalloc(&dOld2, sizeof(Bits)));
  HIP_CHECK(hipMalloc(&dFinal, sizeof(Bits)));
  HIP_CHECK(hipMemcpy(dSlot, &init, sizeof(V), hipMemcpyHostToDevice));
  k_atomic_fmm_returns_old32<IsMin, S><<<1, 1>>>(dSlot, dOld1, dOld2, dFinal, rhs1p, rhs2p);
  HIP_CHECK(hipDeviceSynchronize());

  Bits old1 = 0, old2 = 0, final = 0;
  HIP_CHECK(hipMemcpy(&old1, dOld1, sizeof(Bits), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&old2, dOld2, sizeof(Bits), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&final, dFinal, sizeof(Bits), hipMemcpyDeviceToHost));
  EXPECT_EQ(old1, init.fpsan_payload()) << label << " S=" << int(S);
  EXPECT_EQ(old2, after1.fpsan_payload()) << label << " S=" << int(S);
  EXPECT_EQ(final, after2.fpsan_payload()) << label << " S=" << int(S);

  (void)hipFree(dSlot);
  (void)hipFree(dOld1);
  (void)hipFree(dOld2);
  (void)hipFree(dFinal);
}

TEST(Atomic, FminFpsanReturnsOld) {
  fpsan_test::for_each_fpsan_semantics(
      [](auto sem) { run_fpsan_fmm_returns_old32<true, decltype(sem)::value>("fmin32"); });
}

TEST(Atomic, FmaxFpsanReturnsOld) {
  fpsan_test::for_each_fpsan_semantics(
      [](auto sem) { run_fpsan_fmm_returns_old32<false, decltype(sem)::value>("fmax32"); });
}

TEST(Atomic, FmaxFpsanMatchesScalarMax) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  auto in = make_inputs();
  float *dIn = to_dev(in);
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    using V = Value<float, decltype(sem)::value, kCC>;
    V *dSlot;
    HIP_CHECK(hipMalloc(&dSlot, sizeof(V)));
    V init{-1e30f};
    HIP_CHECK(hipMemcpy(dSlot, &init, sizeof(V), hipMemcpyHostToDevice));
    k_atomic_fmax<decltype(sem)::value><<<1, LANES>>>(dSlot, dIn);
    HIP_CHECK(hipDeviceSynchronize());
    V got{0.f};
    HIP_CHECK(hipMemcpy(&got, dSlot, sizeof(V), hipMemcpyDeviceToHost));
    V acc = init;
    for (float x : in)
      acc = fpsan::max(acc, V{x});
    EXPECT_EQ(got.fpsan_payload(), acc.fpsan_payload());
    (void)hipFree(dSlot);
  });
  (void)hipFree(dIn);
}

// ===========================================================================
// f64 atomics (atomic_fadd/fmin/fmax_f64).
// ===========================================================================
TEST(Atomic, Fadd64FloatExact) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  auto in = make_inputs64();
  double *dIn = to_dev(in);
  using V = Value<double, Semantics::Native, kCC>;
  V *dSlot;
  HIP_CHECK(hipMalloc(&dSlot, sizeof(V)));
  V init{0.0};
  HIP_CHECK(hipMemcpy(dSlot, &init, sizeof(V), hipMemcpyHostToDevice));
  k_atomic_fadd64<Semantics::Native><<<1, LANES>>>(dSlot, dIn);
  HIP_CHECK(hipDeviceSynchronize());
  V got{0.0};
  HIP_CHECK(hipMemcpy(&got, dSlot, sizeof(V), hipMemcpyDeviceToHost));
  double ref = 0;
  for (double x : in)
    ref += x;
  EXPECT_EQ(static_cast<double>(got), ref);
  (void)hipFree(dIn);
  (void)hipFree(dSlot);
}

TEST(Atomic, Fadd64FpsanMatchesScalarRingSum) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  // CAS-loop wrapper makes atomic_fadd64 correct for every algebraic model.
  auto in = make_inputs64();
  double *dIn = to_dev(in);
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    using V = Value<double, decltype(sem)::value, kCC>;
    V *dSlot;
    HIP_CHECK(hipMalloc(&dSlot, sizeof(V)));
    V init{0.0};
    HIP_CHECK(hipMemcpy(dSlot, &init, sizeof(V), hipMemcpyHostToDevice));
    k_atomic_fadd64<decltype(sem)::value><<<1, LANES>>>(dSlot, dIn);
    HIP_CHECK(hipDeviceSynchronize());
    V got{0.0};
    HIP_CHECK(hipMemcpy(&got, dSlot, sizeof(V), hipMemcpyDeviceToHost));
    V acc{0.0};
    for (double x : in)
      acc = acc + V{x};
    EXPECT_EQ(got.fpsan_payload(), acc.fpsan_payload());
    (void)hipFree(dSlot);
  });
  (void)hipFree(dIn);
}

TEST(Atomic, Fmin64FloatMatchesScalarMin) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  auto in = make_inputs64();
  double *dIn = to_dev(in);
  using V = Value<double, Semantics::Native, kCC>;
  V *dSlot;
  HIP_CHECK(hipMalloc(&dSlot, sizeof(V)));
  V init{1e300};
  HIP_CHECK(hipMemcpy(dSlot, &init, sizeof(V), hipMemcpyHostToDevice));
  k_atomic_fmin64<Semantics::Native><<<1, LANES>>>(dSlot, dIn);
  HIP_CHECK(hipDeviceSynchronize());
  V got{0.0};
  HIP_CHECK(hipMemcpy(&got, dSlot, sizeof(V), hipMemcpyDeviceToHost));
  double ref = *std::min_element(in.begin(), in.end());
  EXPECT_EQ(static_cast<double>(got), ref);
  (void)hipFree(dIn);
  (void)hipFree(dSlot);
}

TEST(Atomic, Fmax64FloatMatchesScalarMax) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  auto in = make_inputs64();
  double *dIn = to_dev(in);
  using V = Value<double, Semantics::Native, kCC>;
  V *dSlot;
  HIP_CHECK(hipMalloc(&dSlot, sizeof(V)));
  V init{-1e300};
  HIP_CHECK(hipMemcpy(dSlot, &init, sizeof(V), hipMemcpyHostToDevice));
  k_atomic_fmax64<Semantics::Native><<<1, LANES>>>(dSlot, dIn);
  HIP_CHECK(hipDeviceSynchronize());
  V got{0.0};
  HIP_CHECK(hipMemcpy(&got, dSlot, sizeof(V), hipMemcpyDeviceToHost));
  double ref = *std::max_element(in.begin(), in.end());
  EXPECT_EQ(static_cast<double>(got), ref);
  (void)hipFree(dIn);
  (void)hipFree(dSlot);
}

template <bool IsMin, Semantics S>
__global__ void k_atomic_fmm_returns_old64(Value<double, S, kCC> *slot, std::uint64_t *old1,
                                           std::uint64_t *old2, std::uint64_t *final,
                                           std::uint64_t rhs1, std::uint64_t rhs2) {
  using V = Value<double, S, kCC>;
  const V v1 = V::from_fpsan_payload(rhs1);
  const V v2 = V::from_fpsan_payload(rhs2);
  const V r1 =
      IsMin ? fpsan::amdgcn_atomic_fmin_f64(slot, v1) : fpsan::amdgcn_atomic_fmax_f64(slot, v1);
  const V r2 =
      IsMin ? fpsan::amdgcn_atomic_fmin_f64(slot, v2) : fpsan::amdgcn_atomic_fmax_f64(slot, v2);
  *old1 = r1.fpsan_payload();
  *old2 = r2.fpsan_payload();
  *final = slot->fpsan_payload();
}

template <bool IsMin, Semantics S> static void run_fpsan_fmm_returns_old64(const char *label) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";

  using V = Value<double, S, kCC>;
  using Bits = typename V::bits_type;
  const Bits initp = IsMin ? Bits{1} : Bits{0};
  const Bits rhs1p = IsMin ? Bits{0} : Bits{1};
  const Bits rhs2p = Bits{2};
  const V init = V::from_fpsan_payload(initp);
  const V after1 = IsMin ? fpsan::min(init, V::from_fpsan_payload(rhs1p))
                         : fpsan::max(init, V::from_fpsan_payload(rhs1p));
  const V after2 = IsMin ? fpsan::min(after1, V::from_fpsan_payload(rhs2p))
                         : fpsan::max(after1, V::from_fpsan_payload(rhs2p));

  V *dSlot;
  Bits *dOld1, *dOld2, *dFinal;
  HIP_CHECK(hipMalloc(&dSlot, sizeof(V)));
  HIP_CHECK(hipMalloc(&dOld1, sizeof(Bits)));
  HIP_CHECK(hipMalloc(&dOld2, sizeof(Bits)));
  HIP_CHECK(hipMalloc(&dFinal, sizeof(Bits)));
  HIP_CHECK(hipMemcpy(dSlot, &init, sizeof(V), hipMemcpyHostToDevice));
  k_atomic_fmm_returns_old64<IsMin, S><<<1, 1>>>(dSlot, dOld1, dOld2, dFinal, rhs1p, rhs2p);
  HIP_CHECK(hipDeviceSynchronize());

  Bits old1 = 0, old2 = 0, final = 0;
  HIP_CHECK(hipMemcpy(&old1, dOld1, sizeof(Bits), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&old2, dOld2, sizeof(Bits), hipMemcpyDeviceToHost));
  HIP_CHECK(hipMemcpy(&final, dFinal, sizeof(Bits), hipMemcpyDeviceToHost));
  EXPECT_EQ(old1, init.fpsan_payload()) << label << " S=" << int(S);
  EXPECT_EQ(old2, after1.fpsan_payload()) << label << " S=" << int(S);
  EXPECT_EQ(final, after2.fpsan_payload()) << label << " S=" << int(S);

  (void)hipFree(dSlot);
  (void)hipFree(dOld1);
  (void)hipFree(dOld2);
  (void)hipFree(dFinal);
}

TEST(Atomic, Fmin64FpsanReturnsOld) {
  fpsan_test::for_each_fpsan_semantics(
      [](auto sem) { run_fpsan_fmm_returns_old64<true, decltype(sem)::value>("fmin64"); });
}

TEST(Atomic, Fmax64FpsanReturnsOld) {
  fpsan_test::for_each_fpsan_semantics(
      [](auto sem) { run_fpsan_fmm_returns_old64<false, decltype(sem)::value>("fmax64"); });
}

TEST(Atomic, Fmin64FpsanMatchesScalarMin) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  auto in = make_inputs64();
  double *dIn = to_dev(in);
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    using V = Value<double, decltype(sem)::value, kCC>;
    V *dSlot;
    HIP_CHECK(hipMalloc(&dSlot, sizeof(V)));
    V init{1e300};
    HIP_CHECK(hipMemcpy(dSlot, &init, sizeof(V), hipMemcpyHostToDevice));
    k_atomic_fmin64<decltype(sem)::value><<<1, LANES>>>(dSlot, dIn);
    HIP_CHECK(hipDeviceSynchronize());
    V got{0.0};
    HIP_CHECK(hipMemcpy(&got, dSlot, sizeof(V), hipMemcpyDeviceToHost));
    V acc = init;
    for (double x : in)
      acc = fpsan::min(acc, V{x});
    EXPECT_EQ(got.fpsan_payload(), acc.fpsan_payload());
    (void)hipFree(dSlot);
  });
  (void)hipFree(dIn);
}

TEST(Atomic, Fmax64FpsanMatchesScalarMax) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  auto in = make_inputs64();
  double *dIn = to_dev(in);
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    using V = Value<double, decltype(sem)::value, kCC>;
    V *dSlot;
    HIP_CHECK(hipMalloc(&dSlot, sizeof(V)));
    V init{-1e300};
    HIP_CHECK(hipMemcpy(dSlot, &init, sizeof(V), hipMemcpyHostToDevice));
    k_atomic_fmax64<decltype(sem)::value><<<1, LANES>>>(dSlot, dIn);
    HIP_CHECK(hipDeviceSynchronize());
    V got{0.0};
    HIP_CHECK(hipMemcpy(&got, dSlot, sizeof(V), hipMemcpyDeviceToHost));
    V acc = init;
    for (double x : in)
      acc = fpsan::max(acc, V{x});
    EXPECT_EQ(got.fpsan_payload(), acc.fpsan_payload());
    (void)hipFree(dSlot);
  });
  (void)hipFree(dIn);
}

// ===========================================================================
// Packed v2f16 / v2bf16 atomic_pk_add: the two 16-bit elements accumulate
// INDEPENDENTLY (no cross-element carry). Exact-integer inputs keep both the
// device result and the host reference order-independent.
// ===========================================================================
template <class VEC, class Elem, Semantics S>
void run_pk_add(void (*k)(Value<VEC, S, kCC> *, const float *)) {
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev == 0)
    GTEST_SKIP() << "no HIP device";
  std::vector<float> in(2 * LANES);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : in)
    x = fpsan_test::pick_int_valued<float>(rng, -3, 3); // exact in f16/bf16
  float *dIn = to_dev(in);
  using V = Value<VEC, S, kCC>;
  V *dSlot;
  HIP_CHECK(hipMalloc(&dSlot, sizeof(V)));
  V init{};
  init.set(0, Value<Elem, S, kCC>{static_cast<Elem>(0)});
  init.set(1, Value<Elem, S, kCC>{static_cast<Elem>(0)});
  HIP_CHECK(hipMemcpy(dSlot, &init, sizeof(V), hipMemcpyHostToDevice));
  k<<<1, LANES>>>(dSlot, dIn);
  HIP_CHECK(hipDeviceSynchronize());
  V got{};
  HIP_CHECK(hipMemcpy(&got, dSlot, sizeof(V), hipMemcpyDeviceToHost));
  // Host reference: element-wise accumulation in the same (Value) semantics.
  V acc = init;
  for (int i = 0; i < LANES; ++i) {
    V v{};
    v.set(0, Value<Elem, S, kCC>{static_cast<Elem>(in[2 * i])});
    v.set(1, Value<Elem, S, kCC>{static_cast<Elem>(in[2 * i + 1])});
    acc = acc + v;
  }
  if constexpr (S == Semantics::Native) {
    EXPECT_EQ(static_cast<float>(got.get(0).to_float()), static_cast<float>(acc.get(0).to_float()));
    EXPECT_EQ(static_cast<float>(got.get(1).to_float()), static_cast<float>(acc.get(1).to_float()));
  } else {
    EXPECT_EQ(got.get(0).fpsan_payload(), acc.get(0).fpsan_payload());
    EXPECT_EQ(got.get(1).fpsan_payload(), acc.get(1).fpsan_payload());
  }
  (void)hipFree(dIn);
  (void)hipFree(dSlot);
}

TEST(Atomic, PkAddF16Float) {
  run_pk_add<v2h_t, _Float16, Semantics::Native>(k_atomic_pk_f16<Semantics::Native>);
}
TEST(Atomic, PkAddF16Fpsan) {
  fpsan_test::for_each_fpsan_semantics([](auto sem) {
    run_pk_add<v2h_t, _Float16, decltype(sem)::value>(k_atomic_pk_f16<decltype(sem)::value>);
  });
}
TEST(Atomic, PkAddBf16Float) {
  run_pk_add<v2bf_t, __bf16, Semantics::Native>(k_atomic_pk_bf16<Semantics::Native>);
}
TEST(Atomic, PkAddBf16Fpsan) {
  fpsan_test::for_each_fpsan_semantics([](auto sem) {
    run_pk_add<v2bf_t, __bf16, decltype(sem)::value>(k_atomic_pk_bf16<decltype(sem)::value>);
  });
}
