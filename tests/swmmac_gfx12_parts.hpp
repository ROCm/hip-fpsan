// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/swmmac_gfx12_parts.hpp
//
// Silicon tests for the gfx12 (RDNA4) sparse WMMA wrappers in
// fpsan/amdgcn_swmmac_gfx12.hpp. Three layers:
//
//   * *FloatMatchesBuiltin: the wrapper in Native mode is a bit-exact
//     pass-through to the underlying __builtin_amdgcn_swmmac_*.
//   * *LayoutMatchesHardware: the software dataflow in Native mode produces
//     the same D fragment as the hardware builtin, byte-for-byte. This is
//     the load-bearing layout proof -- if A/B/D/idx-lane mapping or the
//     sparse-K selection were off by one, this fails.
//   * *FpsanMatchesScalarReference: the dataflow in FPSan mode (payload ring)
//     matches a host-side scalar reference doing the equivalent sparse MAC
//     with the same FPSan scalar Value type. Verifies that the per-lane
//     payload algebra is plumbed correctly through the wave-cooperative
//     dataflow.
//
// Requires real gfx1201 hardware; built only under FPSAN_ENABLE_HIP with a
// gfx12 architecture.
#include "fpsan/amdgcn_swmmac_gfx12.hpp"
#include "fpsan/fpsan.hpp"
#include "fpsan_semantics.hpp"
#include "test_random.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

#define HIP_CHECK(e)                                                                               \
  do {                                                                                             \
    hipError_t e_ = (e);                                                                           \
    ASSERT_EQ(e_, hipSuccess) << hipGetErrorString(e_);                                            \
  } while (0)

static constexpr Conversions kCC = Conversions::Explicit;

namespace {
constexpr int WAVE = 32;
bool have_device() {
  int n = 0;
  return hipGetDeviceCount(&n) == hipSuccess && n > 0;
}
} // namespace
namespace {
constexpr int kSwM = 16, kSwN = 16, kSwK = 32;

struct SwData {
  std::vector<float> A;           // M x K dense, zero outside live positions
  std::vector<float> B;           // K x N
  std::vector<float> C;           // M x N
  std::vector<std::uint16_t> idx; // per-lane sparse index (WAVE)
  std::vector<int> p0;            // M*8: live K offset 0 per (row, group)
  std::vector<int> p1;            // M*8: live K offset 1 per (row, group)
};

// Build deterministic 2:4 sparse inputs. The encoding of per-lane idx
// depends on which lane holds the compression nibble for (row i, group g)
// -- different between f16 and fp8. encode_idx is provided by the caller.
template <class EncodeIdxFn> SwData make_sw_data(std::uint32_t seed, EncodeIdxFn encode_idx) {
  SwData d;
  d.A.assign(kSwM * kSwK, 0.f);
  d.B.resize(kSwK * kSwN);
  d.C.resize(kSwM * kSwN);
  d.idx.assign(WAVE, 0);
  d.p0.resize(kSwM * 8);
  d.p1.resize(kSwM * 8);
  std::mt19937 rng = fpsan_test::make_rng();
  rng.discard(seed); // cheap per-test diversity, still deterministic
  for (auto &x : d.B)
    x = fpsan_test::pick_int_valued<float>(rng, -2, 2);
  for (auto &x : d.C)
    x = fpsan_test::pick_int_valued<float>(rng, -4, 4);
  for (int i = 0; i < kSwM; ++i)
    for (int g = 0; g < 8; ++g) {
      int a0 = fpsan_test::pick_int(rng, 0, 3);
      int a1 = fpsan_test::pick_int(rng, 0, 3);
      while (a1 == a0)
        a1 = fpsan_test::pick_int(rng, 0, 3);
      if (a0 > a1) {
        int t = a0;
        a0 = a1;
        a1 = t;
      }
      d.p0[i * 8 + g] = a0;
      d.p1[i * 8 + g] = a1;
      d.A[i * kSwK + 4 * g + a0] = fpsan_test::pick_int_valued<float>(rng, -3, 3);
      d.A[i * kSwK + 4 * g + a1] = fpsan_test::pick_int_valued<float>(rng, -3, 3);
    }
  for (int L = 0; L < WAVE; ++L)
    d.idx[L] = encode_idx(L, d.p0, d.p1);
  return d;
}

// Encode per-lane idx for the f16/bf16/f16-out shared layout.
// Compression for (row i, group g) lives at lane i + 16*((g/2)%2), nibble
// (g%2) within byte (g/4) -> bit_off = 4*(g%2) + 8*(g/4).
//   * Lane L < 16 holds (i=L, g in {0,1,4,5}).
//   * Lane L >= 16 holds (i=L-16, g in {2,3,6,7}).
[[maybe_unused]] inline std::uint16_t encode_idx_h(int L, const std::vector<int> &p0,
                                                   const std::vector<int> &p1) {
  const int i = L & 15;
  const int high = L >> 4;
  std::uint16_t v = 0;
  for (int g = 0; g < 8; ++g) {
    const int side = (g >> 1) & 1;
    if (side != high)
      continue;
    const int bit_off = 4 * (g & 1) + 8 * (g >> 2);
    const int field = (p0[i * 8 + g] & 3) | ((p1[i * 8 + g] & 3) << 2);
    v |= static_cast<std::uint16_t>(field << bit_off);
  }
  return v;
}

// Encode per-lane idx for the fp8/bf8 layout.
// Compression for (row i, group g) lives at lane i + 16*(g/4), nibble
// (g%4) -> bit_off = 4*(g%4).
[[maybe_unused]] inline std::uint16_t encode_idx_fp8(int L, const std::vector<int> &p0,
                                                     const std::vector<int> &p1) {
  const int i = L & 15;
  const int high = L >> 4;
  std::uint16_t v = 0;
  for (int g = 0; g < 8; ++g) {
    const int side = (g >> 2) & 1;
    if (side != high)
      continue;
    const int bit_off = 4 * (g & 3);
    const int field = (p0[i * 8 + g] & 3) | ((p1[i * 8 + g] & 3) << 2);
    v |= static_cast<std::uint16_t>(field << bit_off);
  }
  return v;
}

template <class T> T *to_dev(const std::vector<T> &h) {
  T *d = nullptr;
  (void)hipMalloc(&d, h.size() * sizeof(T));
  (void)hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice);
  return d;
}

template <class T> std::uint64_t bits_of(T v) {
  std::uint64_t u = 0;
  std::memcpy(&u, &v, sizeof v);
  return u;
}
} // namespace

// ---- shared f16 / bf16 / f16-out kernels ----------------------------------
// Stage per-lane fragments from dense A. The (lane, slot) -> (i, k) mapping is
// the layout we are verifying; building the fragment from A_dense means a
// layout bug shows up as a hardware vs dataflow mismatch.
//
// Lane L holds row i = L%16. The 8 v8 slots map to dense K via:
//   side = L/16  (0 or 1)
//   For each g in 0..7 with ((g/2)%2) == side:
//     a_gpr  = 2*(g/4) + (g%2)
//     slot 0 = 2*a_gpr     -> A_dense[i][4g + p0(i,g)]
//     slot 1 = 2*a_gpr + 1 -> A_dense[i][4g + p1(i,g)]
template <Semantics S, class AScalar, class BScalar, class CScalar, class Out, class WrapFn>
__device__ inline void swmmac_h_dataflow(const float *Adense, const float *B, const float *C,
                                         const int *p0, const int *p1, const std::uint16_t *idx,
                                         Out *D, WrapFn fn) {
  using AV = AScalar __attribute__((ext_vector_type(8)));
  using BV = BScalar __attribute__((ext_vector_type(16)));
  using CV = CScalar __attribute__((ext_vector_type(8)));
  const int lane = threadIdx.x;
  const int i = lane & 15;
  const int side = lane >> 4;

  AV an{};
  for (int g = 0; g < 8; ++g) {
    if (((g >> 1) & 1) != side)
      continue;
    const int a_gpr = 2 * (g >> 2) + (g & 1);
    const int slot0 = 2 * a_gpr;
    an[slot0] = static_cast<AScalar>(Adense[i * kSwK + 4 * g + p0[i * 8 + g]]);
    an[slot0 + 1] = static_cast<AScalar>(Adense[i * kSwK + 4 * g + p1[i * 8 + g]]);
  }
  BV bn{};
  // Lane L holds (j = L%16, side = L/16). For each k with ((k/8)%2) == side:
  //   slot = 8*(k/16) + 2*((k/2)%4) + (k%2)
  const int j = lane & 15;
  for (int k = 0; k < kSwK; ++k) {
    if (((k >> 3) & 1) != side)
      continue;
    const int slot = 8 * (k / 16) + 2 * ((k / 2) % 4) + (k % 2);
    bn[slot] = static_cast<BScalar>(B[k * kSwN + j]);
  }
  CV cn{};
  for (int reg = 0; reg < 8; ++reg) {
    const int m = reg + 8 * side;
    cn[reg] = static_cast<CScalar>(C[m * kSwN + j]);
  }
  Value<AV, S, kCC> av(an);
  Value<BV, S, kCC> bv(bn);
  Value<CV, S, kCC> cv(cn);
  auto dv = fn(av, bv, cv, idx[lane]);
  for (int reg = 0; reg < 8; ++reg) {
    const int m = reg + 8 * side;
    if constexpr (S == Semantics::Native)
      D[m * kSwN + j] = static_cast<Out>(dv.get(reg).to_float());
    else
      D[m * kSwN + j] = static_cast<Out>(dv.get(reg).fpsan_payload());
  }
}

// Compute the sparse host-side reference D[m][n] = C + A_dense * B (the live
// values are already the only nonzeros in A_dense).
template <class CScalar> std::vector<CScalar> sw_reference_h(const SwData &d) {
  std::vector<CScalar> ref(kSwM * kSwN);
  for (int m = 0; m < kSwM; ++m)
    for (int n = 0; n < kSwN; ++n) {
      float acc = d.C[m * kSwN + n];
      for (int g = 0; g < 8; ++g) {
        acc += d.A[m * kSwK + 4 * g + d.p0[m * 8 + g]] * d.B[(4 * g + d.p0[m * 8 + g]) * kSwN + n];
        acc += d.A[m * kSwK + 4 * g + d.p1[m * 8 + g]] * d.B[(4 * g + d.p1[m * 8 + g]) * kSwN + n];
      }
      ref[m * kSwN + n] = static_cast<CScalar>(acc);
    }
  return ref;
}

// FPSan-domain reference using scalar Value arithmetic. Uses the same scalar
// types the kernel uses (AScalar/BScalar/CScalar = element types).
template <Semantics S, class AScalar, class BScalar, class CScalar>
std::vector<std::uint64_t> sw_reference_fpsan_h(const SwData &d) {
  using VA = Value<AScalar, S, kCC>;
  using VB = Value<BScalar, S, kCC>;
  using VC = Value<CScalar, S, kCC>;
  std::vector<std::uint64_t> ref(kSwM * kSwN);
  for (int m = 0; m < kSwM; ++m)
    for (int n = 0; n < kSwN; ++n) {
      VC acc(static_cast<CScalar>(d.C[m * kSwN + n]));
      for (int g = 0; g < 8; ++g) {
        const int k0 = 4 * g + d.p0[m * 8 + g];
        const int k1 = 4 * g + d.p1[m * 8 + g];
        acc = acc + fpsan::cast<CScalar>(VA(static_cast<AScalar>(d.A[m * kSwK + k0]))) *
                        fpsan::cast<CScalar>(VB(static_cast<BScalar>(d.B[k0 * kSwN + n])));
        acc = acc + fpsan::cast<CScalar>(VA(static_cast<AScalar>(d.A[m * kSwK + k1]))) *
                        fpsan::cast<CScalar>(VB(static_cast<BScalar>(d.B[k1 * kSwN + n])));
      }
      ref[m * kSwN + n] = static_cast<std::uint64_t>(acc.fpsan_payload());
    }
  return ref;
}

// ---- f16 layout + payload ---------------------------------------------------
__global__ void k_swmmac_f16_float(const float *A, const float *B, const float *C, const int *p0,
                                   const int *p1, const std::uint16_t *idx, float *D) {
  auto fn = [](auto av, auto bv, auto cv, std::uint16_t i) {
    return fpsan::amdgcn_swmmac_f32_16x16x32_f16_w32<Semantics::Native, kCC>(av, bv, cv, i);
  };
  swmmac_h_dataflow<Semantics::Native, _Float16, _Float16, float, float>(A, B, C, p0, p1, idx, D,
                                                                         fn);
}
template <Semantics S>
__global__ void k_swmmac_f16_fpsan(const float *A, const float *B, const float *C, const int *p0,
                                   const int *p1, const std::uint16_t *idx, std::uint32_t *D) {
  auto fn = [](auto av, auto bv, auto cv, std::uint16_t i) {
    return fpsan::amdgcn_swmmac_f32_16x16x32_f16_w32<S, kCC>(av, bv, cv, i);
  };
  swmmac_h_dataflow<S, _Float16, _Float16, float, std::uint32_t>(A, B, C, p0, p1, idx, D, fn);
}

// Layout / FPSan dataflow tests for fp8/bf8 (different per-lane layout).
// ============================================================================
//
// Lane L holds (i = L%16, side = L/16). A v8 packs 8 groups linearly,
// 2 bytes per group. For each g in 0..7 with (g/4) == side:
//   byte_off = 2*(g%4)
//   byte_off + 0 -> A_dense[i][4g + p0]
//   byte_off + 1 -> A_dense[i][4g + p1]
// B v16 holds k = 16*side + (byte_off) linearly: byte = k%16, lane = j + 16*(k/16).
//
// fp8/bf8 path: stage bytes directly, build per-lane fragments using
// AScalar/BScalar element types (fpsan::fp8_e4m3 / fp8_e5m2).
template <Semantics S, class AScalar, class BScalar, class Out, class WrapFn>
__global__ void k_swmmac_fp8_kernel(const float *Adense, const float *B, const float *C,
                                    const int *p0, const int *p1, const std::uint16_t *idx, Out *D,
                                    WrapFn fn) {
  using AFrag = fpsan::detail::v8_fragment<AScalar>;
  using BFrag = fpsan::detail::v16_byte_fragment<BScalar>;
  using v8f = fpsan::v8f_native;
  const int lane = threadIdx.x;
  const int i = lane & 15;
  const int j = lane & 15;
  const int side = lane >> 4;

  AFrag an{};
  for (int g = 0; g < 8; ++g) {
    if (((g >> 2) & 1) != side)
      continue;
    const int b0 = 2 * (g & 3);
    an[b0] = static_cast<AScalar>(Adense[i * kSwK + 4 * g + p0[i * 8 + g]]);
    an[b0 + 1] = static_cast<AScalar>(Adense[i * kSwK + 4 * g + p1[i * 8 + g]]);
  }
  BFrag bn{};
  for (int k = 0; k < kSwK; ++k) {
    if ((k >> 4) != side)
      continue;
    const int byte = k & 15;
    bn[byte] = static_cast<BScalar>(B[k * kSwN + j]);
  }
  v8f cn{};
  for (int reg = 0; reg < 8; ++reg) {
    const int m = reg + 8 * side;
    cn[reg] = C[m * kSwN + j];
  }
  Value<AFrag, S, kCC> av(an);
  Value<BFrag, S, kCC> bv(bn);
  Value<v8f, S, kCC> cv(cn);
  auto dv = fn(av, bv, cv, idx[lane]);
  for (int reg = 0; reg < 8; ++reg) {
    const int m = reg + 8 * side;
    if constexpr (S == Semantics::Native)
      D[m * kSwN + j] = static_cast<Out>(dv.get(reg).to_float());
    else
      D[m * kSwN + j] = static_cast<Out>(dv.get(reg).fpsan_payload());
  }
}

template <class AScalar, class BScalar> std::vector<float> sw_reference_fp8(const SwData &d) {
  std::vector<float> ref(kSwM * kSwN);
  for (int m = 0; m < kSwM; ++m)
    for (int n = 0; n < kSwN; ++n) {
      float acc = d.C[m * kSwN + n];
      for (int g = 0; g < 8; ++g) {
        const int k0 = 4 * g + d.p0[m * 8 + g];
        const int k1 = 4 * g + d.p1[m * 8 + g];
        const float av0 = static_cast<float>(static_cast<AScalar>(d.A[m * kSwK + k0]));
        const float av1 = static_cast<float>(static_cast<AScalar>(d.A[m * kSwK + k1]));
        const float bv0 = static_cast<float>(static_cast<BScalar>(d.B[k0 * kSwN + n]));
        const float bv1 = static_cast<float>(static_cast<BScalar>(d.B[k1 * kSwN + n]));
        acc += av0 * bv0;
        acc += av1 * bv1;
      }
      ref[m * kSwN + n] = acc;
    }
  return ref;
}

template <Semantics S, class AScalar, class BScalar>
std::vector<std::uint32_t> sw_reference_fp8_fpsan(const SwData &d) {
  using VA = Value<AScalar, S, kCC>;
  using VB = Value<BScalar, S, kCC>;
  using VF = Value<float, S, kCC>;
  std::vector<std::uint32_t> ref(kSwM * kSwN);
  for (int m = 0; m < kSwM; ++m)
    for (int n = 0; n < kSwN; ++n) {
      VF acc(d.C[m * kSwN + n]);
      for (int g = 0; g < 8; ++g) {
        const int k0 = 4 * g + d.p0[m * 8 + g];
        const int k1 = 4 * g + d.p1[m * 8 + g];
        acc = acc + fpsan::cast<float>(VA(static_cast<AScalar>(d.A[m * kSwK + k0]))) *
                        fpsan::cast<float>(VB(static_cast<BScalar>(d.B[k0 * kSwN + n])));
        acc = acc + fpsan::cast<float>(VA(static_cast<AScalar>(d.A[m * kSwK + k1]))) *
                        fpsan::cast<float>(VB(static_cast<BScalar>(d.B[k1 * kSwN + n])));
      }
      ref[m * kSwN + n] = acc.fpsan_payload();
    }
  return ref;
}

// Generic fp8/bf8 layout + fpsan test runner.
//
// `wf` is the Native-semantics wrapper lambda (the hardware oracle, run once).
// `make_wp` is a generic functor that, given a std::integral_constant<Semantics,
// S>, returns the FPSan wrapper lambda specialized to that semantics; it is
// invoked once per FPSan-family variant.
template <class AScalar, class BScalar, class WrapFloatFn, class MakeFpsanFn>
void run_fp8_layout_and_fpsan(std::uint32_t seed, WrapFloatFn wf, MakeFpsanFn make_wp) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  SwData d = make_sw_data(seed, encode_idx_fp8);
  std::vector<float> ref = sw_reference_fp8<AScalar, BScalar>(d);
  float *dA = to_dev(d.A), *dB = to_dev(d.B), *dC = to_dev(d.C);
  int *dp0 = to_dev(d.p0), *dp1 = to_dev(d.p1);
  std::uint16_t *dI = to_dev(d.idx);
  float *dDf = nullptr;
  HIP_CHECK(hipMalloc(&dDf, kSwM * kSwN * sizeof(float)));
  k_swmmac_fp8_kernel<Semantics::Native, AScalar, BScalar, float>
      <<<1, WAVE>>>(dA, dB, dC, dp0, dp1, dI, dDf, wf);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> got(kSwM * kSwN);
  HIP_CHECK(hipMemcpy(got.data(), dDf, kSwM * kSwN * sizeof(float), hipMemcpyDeviceToHost));
  for (int t = 0; t < kSwM * kSwN; ++t)
    EXPECT_EQ(bits_of(got[t]), bits_of(ref[t])) << "Layout at " << t;
  (void)hipFree(dDf);

  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    auto wp = make_wp(sem);
    std::vector<std::uint32_t> ref_p = sw_reference_fp8_fpsan<S, AScalar, BScalar>(d);
    std::uint32_t *dDp = nullptr;
    HIP_CHECK(hipMalloc(&dDp, kSwM * kSwN * sizeof(std::uint32_t)));
    k_swmmac_fp8_kernel<S, AScalar, BScalar, std::uint32_t>
        <<<1, WAVE>>>(dA, dB, dC, dp0, dp1, dI, dDp, wp);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got_p(kSwM * kSwN);
    HIP_CHECK(
        hipMemcpy(got_p.data(), dDp, kSwM * kSwN * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for (int t = 0; t < kSwM * kSwN; ++t)
      EXPECT_EQ(got_p[t], ref_p[t]) << "FPSan at " << t;
    (void)hipFree(dDp);
  });

  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dp0);
  (void)hipFree(dp1);
  (void)hipFree(dI);
}
