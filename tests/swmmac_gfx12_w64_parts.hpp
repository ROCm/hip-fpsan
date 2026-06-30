// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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
constexpr int WAVE = 64;
constexpr int M = 16, N = 16, K = 32, G = 8;

bool have_device() {
  int n = 0;
  return hipGetDeviceCount(&n) == hipSuccess && n > 0;
}

struct SwData {
  std::vector<float> A;
  std::vector<float> B;
  std::vector<float> C;
  std::vector<int> idx;
  std::vector<int> p0;
  std::vector<int> p1;
};

template <class EncodeIdxFn> SwData make_sw_data(std::uint32_t seed, EncodeIdxFn encode_idx) {
  SwData d;
  d.A.assign(M * K, 0.f);
  d.B.resize(K * N);
  d.C.resize(M * N);
  d.idx.assign(WAVE, 0);
  d.p0.resize(M * G);
  d.p1.resize(M * G);

  std::mt19937 rng = fpsan_test::make_rng();
  rng.discard(seed);
  for (auto &x : d.B)
    x = fpsan_test::pick_int_valued<float>(rng, -2, 2);
  for (auto &x : d.C)
    x = fpsan_test::pick_int_valued<float>(rng, -4, 4);
  for (int i = 0; i < M; ++i)
    for (int g = 0; g < G; ++g) {
      int a0 = fpsan_test::pick_int(rng, 0, 3);
      int a1 = fpsan_test::pick_int(rng, 0, 3);
      while (a1 == a0)
        a1 = fpsan_test::pick_int(rng, 0, 3);
      if (a0 > a1) {
        int t = a0;
        a0 = a1;
        a1 = t;
      }
      d.p0[i * G + g] = a0;
      d.p1[i * G + g] = a1;
      d.A[i * K + 4 * g + a0] = fpsan_test::pick_int_valued<float>(rng, -3, 3);
      d.A[i * K + 4 * g + a1] = fpsan_test::pick_int_valued<float>(rng, -3, 3);
    }

  for (int lane = 0; lane < WAVE; ++lane)
    d.idx[lane] = encode_idx(lane, d.p0, d.p1);
  return d;
}

[[maybe_unused]] int encode_idx_h_w64(int lane, const std::vector<int> &p0,
                                      const std::vector<int> &p1) {
  const int i = lane & 15;
  const int block = lane >> 4;
  int v = 0;
  for (int g = 0; g < G; ++g) {
    if ((g >> 1) != block)
      continue;
    const int bit_off = 4 * (g & 1);
    const int field = (p0[i * G + g] & 3) | ((p1[i * G + g] & 3) << 2);
    v |= field << bit_off;
  }
  return v;
}

[[maybe_unused]] int encode_idx_fp8_w64(int lane, const std::vector<int> &p0,
                                        const std::vector<int> &p1) {
  const int i = lane & 15;
  const int block = lane >> 4;
  int v = 0;
  for (int g = 0; g < G; ++g) {
    const int gblock = 2 * ((g >> 1) & 1) + (g >> 2);
    if (gblock != block)
      continue;
    const int bit_off = 4 * (g & 1);
    const int field = (p0[i * G + g] & 3) | ((p1[i * G + g] & 3) << 2);
    v |= field << bit_off;
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

template <class CScalar> std::vector<CScalar> sw_reference_h(const SwData &d) {
  std::vector<CScalar> ref(M * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = d.C[m * N + n];
      for (int g = 0; g < G; ++g) {
        const int k0 = 4 * g + d.p0[m * G + g];
        const int k1 = 4 * g + d.p1[m * G + g];
        acc += d.A[m * K + k0] * d.B[k0 * N + n];
        acc += d.A[m * K + k1] * d.B[k1 * N + n];
      }
      ref[m * N + n] = static_cast<CScalar>(acc);
    }
  return ref;
}

template <class AScalar, class BScalar> std::vector<float> sw_reference_fp8(const SwData &d) {
  std::vector<float> ref(M * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float acc = d.C[m * N + n];
      for (int g = 0; g < G; ++g) {
        const int k0 = 4 * g + d.p0[m * G + g];
        const int k1 = 4 * g + d.p1[m * G + g];
        const float av0 = static_cast<float>(static_cast<AScalar>(d.A[m * K + k0]));
        const float av1 = static_cast<float>(static_cast<AScalar>(d.A[m * K + k1]));
        const float bv0 = static_cast<float>(static_cast<BScalar>(d.B[k0 * N + n]));
        const float bv1 = static_cast<float>(static_cast<BScalar>(d.B[k1 * N + n]));
        acc += av0 * bv0;
        acc += av1 * bv1;
      }
      ref[m * N + n] = acc;
    }
  return ref;
}

template <class AScalar, class BScalar, class CScalar, Semantics S>
std::vector<std::uint64_t> sw_reference_fpsan_h(const SwData &d) {
  using VA = Value<AScalar, S, kCC>;
  using VB = Value<BScalar, S, kCC>;
  using VC = Value<CScalar, S, kCC>;
  std::vector<std::uint64_t> ref(M * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      VC acc(static_cast<CScalar>(d.C[m * N + n]));
      for (int g = 0; g < G; ++g) {
        const int k0 = 4 * g + d.p0[m * G + g];
        const int k1 = 4 * g + d.p1[m * G + g];
        acc = acc + fpsan::cast<CScalar>(VA(static_cast<AScalar>(d.A[m * K + k0]))) *
                        fpsan::cast<CScalar>(VB(static_cast<BScalar>(d.B[k0 * N + n])));
        acc = acc + fpsan::cast<CScalar>(VA(static_cast<AScalar>(d.A[m * K + k1]))) *
                        fpsan::cast<CScalar>(VB(static_cast<BScalar>(d.B[k1 * N + n])));
      }
      ref[m * N + n] = static_cast<std::uint64_t>(acc.fpsan_payload());
    }
  return ref;
}

template <class AScalar, class BScalar, Semantics S>
std::vector<std::uint32_t> sw_reference_fp8_fpsan(const SwData &d) {
  using VA = Value<AScalar, S, kCC>;
  using VB = Value<BScalar, S, kCC>;
  using VF = Value<float, S, kCC>;
  std::vector<std::uint32_t> ref(M * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      VF acc(d.C[m * N + n]);
      for (int g = 0; g < G; ++g) {
        const int k0 = 4 * g + d.p0[m * G + g];
        const int k1 = 4 * g + d.p1[m * G + g];
        acc = acc + fpsan::cast<float>(VA(static_cast<AScalar>(d.A[m * K + k0]))) *
                        fpsan::cast<float>(VB(static_cast<BScalar>(d.B[k0 * N + n])));
        acc = acc + fpsan::cast<float>(VA(static_cast<AScalar>(d.A[m * K + k1]))) *
                        fpsan::cast<float>(VB(static_cast<BScalar>(d.B[k1 * N + n])));
      }
      ref[m * N + n] = acc.fpsan_payload();
    }
  return ref;
}

struct SwF32F16 {
  using AScalar = _Float16;
  using BScalar = _Float16;
  using CScalar = float;
  using AVec = fpsan::v4h_native;
  using BVec = fpsan::v8h_native;
  using CVec = fpsan::v4f_native;
  using Out = float;
  using Payload = std::uint32_t;
  template <Semantics S>
  __device__ static Value<CVec, S, kCC> call(Value<AVec, S, kCC> a, Value<BVec, S, kCC> b,
                                             Value<CVec, S, kCC> c, int idx) {
    return fpsan::amdgcn_swmmac_f32_16x16x32_f16_w64<S, kCC>(a, b, c, idx);
  }
};

struct SwF32BF16 {
  using AScalar = __bf16;
  using BScalar = __bf16;
  using CScalar = float;
  using AVec = fpsan::v4bf_native;
  using BVec = fpsan::v8bf_native;
  using CVec = fpsan::v4f_native;
  using Out = float;
  using Payload = std::uint32_t;
  template <Semantics S>
  __device__ static Value<CVec, S, kCC> call(Value<AVec, S, kCC> a, Value<BVec, S, kCC> b,
                                             Value<CVec, S, kCC> c, int idx) {
    return fpsan::amdgcn_swmmac_f32_16x16x32_bf16_w64<S, kCC>(a, b, c, idx);
  }
};

struct SwF16F16 {
  using AScalar = _Float16;
  using BScalar = _Float16;
  using CScalar = _Float16;
  using AVec = fpsan::v4h_native;
  using BVec = fpsan::v8h_native;
  using CVec = fpsan::v4h_native;
  using Out = _Float16;
  using Payload = std::uint16_t;
  template <Semantics S>
  __device__ static Value<CVec, S, kCC> call(Value<AVec, S, kCC> a, Value<BVec, S, kCC> b,
                                             Value<CVec, S, kCC> c, int idx) {
    return fpsan::amdgcn_swmmac_f16_16x16x32_f16_w64<S, kCC>(a, b, c, idx);
  }
};

struct SwBF16BF16 {
  using AScalar = __bf16;
  using BScalar = __bf16;
  using CScalar = __bf16;
  using AVec = fpsan::v4bf_native;
  using BVec = fpsan::v8bf_native;
  using CVec = fpsan::v4bf_native;
  using Out = __bf16;
  using Payload = std::uint16_t;
  template <Semantics S>
  __device__ static Value<CVec, S, kCC> call(Value<AVec, S, kCC> a, Value<BVec, S, kCC> b,
                                             Value<CVec, S, kCC> c, int idx) {
    return fpsan::amdgcn_swmmac_bf16_16x16x32_bf16_w64<S, kCC>(a, b, c, idx);
  }
};

template <class Traits, Semantics S, class Out>
__global__ void k_swmmac_h(const float *A, const float *B, const float *C, const int *p0,
                           const int *p1, const int *idx, Out *D) {
  using AScalar = typename Traits::AScalar;
  using BScalar = typename Traits::BScalar;
  using CScalar = typename Traits::CScalar;
  using AVec = typename Traits::AVec;
  using BVec = typename Traits::BVec;
  using CVec = typename Traits::CVec;
  const int lane = threadIdx.x;
  const int row = lane & 15;
  const int col = lane & 15;
  const int block = lane >> 4;

  AVec av_raw{};
  for (int g = 0; g < G; ++g) {
    if ((g >> 1) != block)
      continue;
    const int slot = 2 * (g & 1);
    av_raw[slot] = static_cast<AScalar>(A[row * K + 4 * g + p0[row * G + g]]);
    av_raw[slot + 1] = static_cast<AScalar>(A[row * K + 4 * g + p1[row * G + g]]);
  }

  BVec bv_raw{};
  for (int k = 0; k < K; ++k) {
    if ((k >> 3) != block)
      continue;
    bv_raw[k & 7] = static_cast<BScalar>(B[k * N + col]);
  }

  CVec cv_raw{};
  for (int e = 0; e < 4; ++e) {
    const int m = e + 8 * (block & 1) + 4 * (block >> 1);
    cv_raw[e] = static_cast<CScalar>(C[m * N + col]);
  }

  Value<AVec, S, kCC> av(av_raw);
  Value<BVec, S, kCC> bv(bv_raw);
  Value<CVec, S, kCC> cv(cv_raw);
  auto dv = Traits::template call<S>(av, bv, cv, idx[lane]);
  for (int e = 0; e < 4; ++e) {
    const int m = e + 8 * (block & 1) + 4 * (block >> 1);
    if constexpr (S == Semantics::Native)
      D[m * N + col] = static_cast<Out>(dv.get(e).to_float());
    else
      D[m * N + col] = static_cast<Out>(dv.get(e).fpsan_payload());
  }
}

template <class AScalar_, class BScalar_, class AVec_, class BVec_, class Wrap> struct SwFp8 {
  using AScalar = AScalar_;
  using BScalar = BScalar_;
  using AVec = AVec_;
  using BVec = BVec_;
  using CVec = fpsan::v4f_native;
  using Out = float;
  using Payload = std::uint32_t;
  template <Semantics S>
  __device__ static Value<CVec, S, kCC> call(Value<AVec, S, kCC> a, Value<BVec, S, kCC> b,
                                             Value<CVec, S, kCC> c, int idx) {
    return Wrap{}.template operator()<S>(a, b, c, idx);
  }
};

struct Fp8Fp8Wrap {
  template <Semantics S>
  __device__ Value<fpsan::v4f_native, S, kCC>
  operator()(Value<fpsan::v4e4m3_native, S, kCC> a, Value<fpsan::v8e4m3_native, S, kCC> b,
             Value<fpsan::v4f_native, S, kCC> c, int idx) const {
    return fpsan::amdgcn_swmmac_f32_16x16x32_fp8_fp8_w64<S, kCC>(a, b, c, idx);
  }
};
struct Fp8Bf8Wrap {
  template <Semantics S>
  __device__ Value<fpsan::v4f_native, S, kCC>
  operator()(Value<fpsan::v4e4m3_native, S, kCC> a, Value<fpsan::v8e5m2_native, S, kCC> b,
             Value<fpsan::v4f_native, S, kCC> c, int idx) const {
    return fpsan::amdgcn_swmmac_f32_16x16x32_fp8_bf8_w64<S, kCC>(a, b, c, idx);
  }
};
struct Bf8Fp8Wrap {
  template <Semantics S>
  __device__ Value<fpsan::v4f_native, S, kCC>
  operator()(Value<fpsan::v4e5m2_native, S, kCC> a, Value<fpsan::v8e4m3_native, S, kCC> b,
             Value<fpsan::v4f_native, S, kCC> c, int idx) const {
    return fpsan::amdgcn_swmmac_f32_16x16x32_bf8_fp8_w64<S, kCC>(a, b, c, idx);
  }
};
struct Bf8Bf8Wrap {
  template <Semantics S>
  __device__ Value<fpsan::v4f_native, S, kCC>
  operator()(Value<fpsan::v4e5m2_native, S, kCC> a, Value<fpsan::v8e5m2_native, S, kCC> b,
             Value<fpsan::v4f_native, S, kCC> c, int idx) const {
    return fpsan::amdgcn_swmmac_f32_16x16x32_bf8_bf8_w64<S, kCC>(a, b, c, idx);
  }
};

using SwFp8Fp8 =
    SwFp8<fpsan::fp8_e4m3, fpsan::fp8_e4m3, fpsan::v4e4m3_native, fpsan::v8e4m3_native, Fp8Fp8Wrap>;
using SwFp8Bf8 =
    SwFp8<fpsan::fp8_e4m3, fpsan::fp8_e5m2, fpsan::v4e4m3_native, fpsan::v8e5m2_native, Fp8Bf8Wrap>;
using SwBf8Fp8 =
    SwFp8<fpsan::fp8_e5m2, fpsan::fp8_e4m3, fpsan::v4e5m2_native, fpsan::v8e4m3_native, Bf8Fp8Wrap>;
using SwBf8Bf8 =
    SwFp8<fpsan::fp8_e5m2, fpsan::fp8_e5m2, fpsan::v4e5m2_native, fpsan::v8e5m2_native, Bf8Bf8Wrap>;

template <class Traits, Semantics S, class Out>
__global__ void k_swmmac_fp8(const float *A, const float *B, const float *C, const int *p0,
                             const int *p1, const int *idx, Out *D) {
  using AScalar = typename Traits::AScalar;
  using BScalar = typename Traits::BScalar;
  using AVec = typename Traits::AVec;
  using BVec = typename Traits::BVec;
  using CVec = typename Traits::CVec;
  const int lane = threadIdx.x;
  const int row = lane & 15;
  const int col = lane & 15;
  const int block = lane >> 4;

  AVec av_raw{};
  for (int g = 0; g < G; ++g) {
    const int gblock = 2 * ((g >> 1) & 1) + (g >> 2);
    if (gblock != block)
      continue;
    const int byte = 2 * (g & 1);
    av_raw[byte] = static_cast<AScalar>(A[row * K + 4 * g + p0[row * G + g]]);
    av_raw[byte + 1] = static_cast<AScalar>(A[row * K + 4 * g + p1[row * G + g]]);
  }

  BVec bv_raw{};
  for (int k = 0; k < K; ++k) {
    const int kblock = 2 * ((k >> 3) & 1) + (k >> 4);
    if (kblock != block)
      continue;
    const int byte = 4 * ((k >> 2) & 1) + (k & 3);
    bv_raw[byte] = static_cast<BScalar>(B[k * N + col]);
  }

  CVec cv_raw{};
  for (int e = 0; e < 4; ++e) {
    const int m = e + 8 * (block & 1) + 4 * (block >> 1);
    cv_raw[e] = C[m * N + col];
  }

  Value<AVec, S, kCC> av(av_raw);
  Value<BVec, S, kCC> bv(bv_raw);
  Value<CVec, S, kCC> cv(cv_raw);
  auto dv = Traits::template call<S>(av, bv, cv, idx[lane]);
  for (int e = 0; e < 4; ++e) {
    const int m = e + 8 * (block & 1) + 4 * (block >> 1);
    if constexpr (S == Semantics::Native)
      D[m * N + col] = static_cast<Out>(dv.get(e).to_float());
    else
      D[m * N + col] = static_cast<Out>(dv.get(e).fpsan_payload());
  }
}

template <class Traits> void run_h_case(std::uint32_t seed) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  using Out = typename Traits::Out;
  using Payload = typename Traits::Payload;
  SwData d = make_sw_data(seed, encode_idx_h_w64);
  std::vector<Out> ref = sw_reference_h<Out>(d);

  float *dA = to_dev(d.A), *dB = to_dev(d.B), *dC = to_dev(d.C);
  int *dp0 = to_dev(d.p0), *dp1 = to_dev(d.p1), *dI = to_dev(d.idx);

  Out *dD = nullptr;
  HIP_CHECK(hipMalloc(&dD, M * N * sizeof(Out)));
  k_swmmac_h<Traits, Semantics::Native><<<1, WAVE>>>(dA, dB, dC, dp0, dp1, dI, dD);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<Out> got(M * N);
  HIP_CHECK(hipMemcpy(got.data(), dD, M * N * sizeof(Out), hipMemcpyDeviceToHost));
  for (int t = 0; t < M * N; ++t)
    EXPECT_EQ(bits_of(got[t]), bits_of(ref[t])) << "Layout at " << t;
  (void)hipFree(dD);

  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    std::vector<std::uint64_t> ref_p =
        sw_reference_fpsan_h<typename Traits::AScalar, typename Traits::BScalar, Out, S>(d);
    Payload *dP = nullptr;
    HIP_CHECK(hipMalloc(&dP, M * N * sizeof(Payload)));
    k_swmmac_h<Traits, S><<<1, WAVE>>>(dA, dB, dC, dp0, dp1, dI, dP);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<Payload> got_p(M * N);
    HIP_CHECK(hipMemcpy(got_p.data(), dP, M * N * sizeof(Payload), hipMemcpyDeviceToHost));
    for (int t = 0; t < M * N; ++t)
      EXPECT_EQ(static_cast<std::uint64_t>(got_p[t]), ref_p[t]) << "FPSan at " << t;
    (void)hipFree(dP);
  });

  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dp0);
  (void)hipFree(dp1);
  (void)hipFree(dI);
}

template <class Traits> void run_fp8_case(std::uint32_t seed) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  SwData d = make_sw_data(seed, encode_idx_fp8_w64);
  std::vector<float> ref = sw_reference_fp8<typename Traits::AScalar, typename Traits::BScalar>(d);

  float *dA = to_dev(d.A), *dB = to_dev(d.B), *dC = to_dev(d.C);
  int *dp0 = to_dev(d.p0), *dp1 = to_dev(d.p1), *dI = to_dev(d.idx);

  float *dD = nullptr;
  HIP_CHECK(hipMalloc(&dD, M * N * sizeof(float)));
  k_swmmac_fp8<Traits, Semantics::Native><<<1, WAVE>>>(dA, dB, dC, dp0, dp1, dI, dD);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> got(M * N);
  HIP_CHECK(hipMemcpy(got.data(), dD, M * N * sizeof(float), hipMemcpyDeviceToHost));
  for (int t = 0; t < M * N; ++t)
    EXPECT_EQ(bits_of(got[t]), bits_of(ref[t])) << "Layout at " << t;
  (void)hipFree(dD);

  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    std::vector<std::uint32_t> ref_p =
        sw_reference_fp8_fpsan<typename Traits::AScalar, typename Traits::BScalar, S>(d);
    std::uint32_t *dP = nullptr;
    HIP_CHECK(hipMalloc(&dP, M * N * sizeof(std::uint32_t)));
    k_swmmac_fp8<Traits, S><<<1, WAVE>>>(dA, dB, dC, dp0, dp1, dI, dP);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got_p(M * N);
    HIP_CHECK(hipMemcpy(got_p.data(), dP, M * N * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for (int t = 0; t < M * N; ++t)
      EXPECT_EQ(got_p[t], ref_p[t]) << "FPSan at " << t;
    (void)hipFree(dP);
  });

  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dp0);
  (void)hipFree(dp1);
  (void)hipFree(dI);
}
} // namespace
