// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/cvt_gfx950_test.cpp
//
// GPU tests for the gfx950 fp8 conversion wrappers in fpsan/amdgcn_cvt.hpp.
// (cvt_test.cpp is gfx12-only; these exercise the wrappers on CDNA4 silicon.)
//
// For the cvt_pk_f32_{fp8,bf8} unpack ops:
//   - Float: pack two exact-in-fp8 values into the selected 16-bit word,
//   unpack,
//     and require the originals back (round-trip is lossless for exact values).
//   - FPSan: the unpack of a register packed in FPSan mode must equal the
//   direct
//     payload-ring composition cast<float>(cast<fp8>(x)) -- i.e. the byte
//     selection + plumbing is correct.
#include "fpsan/amdgcn_cvt.hpp"
#include "fpsan/fpsan.hpp"

#include "fpsan_semantics.hpp"

#include "hip_test_utils.hpp"
#include "test_random.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

static constexpr Conversions kCC = Conversions::Explicit;
static constexpr int LANES = 64;

// Functors so a single kernel template covers fp8 and bf8 (builtins can't be
// passed as template args cleanly across the host/device boundary).
template <class FP8, bool WordSel> struct CvtPk;
template <bool WordSel> struct CvtPk<fpsan::fp8_e4m3, WordSel> {
  template <Semantics S>
  __device__ static int pack(Value<float, S, kCC> a, Value<float, S, kCC> b, int old) {
    return fpsan::amdgcn_cvt_pk_fp8_f32<WordSel == 0, S, kCC>(a, b, old);
  }
  template <Semantics S> __device__ static Value<fpsan::v2f_native, S, kCC> unpack(int p) {
    return fpsan::amdgcn_cvt_pk_f32_fp8<WordSel, S, kCC>(p);
  }
};
template <bool WordSel> struct CvtPk<fpsan::fp8_e5m2, WordSel> {
  template <Semantics S>
  __device__ static int pack(Value<float, S, kCC> a, Value<float, S, kCC> b, int old) {
    return fpsan::amdgcn_cvt_pk_bf8_f32<WordSel == 0, S, kCC>(a, b, old);
  }
  template <Semantics S> __device__ static Value<fpsan::v2f_native, S, kCC> unpack(int p) {
    return fpsan::amdgcn_cvt_pk_f32_bf8<WordSel, S, kCC>(p);
  }
};

template <Semantics S, class FP8, bool WordSel, class Out>
__global__ void k_run(const float *in, Out *out) {
  int l = threadIdx.x;
  Value<float, S, kCC> a{in[2 * l]}, b{in[2 * l + 1]};
  int packed = CvtPk<FP8, WordSel>::template pack<S>(a, b, 0);
  auto r = CvtPk<FP8, WordSel>::template unpack<S>(packed);
  if constexpr (S == Semantics::Native) {
    out[2 * l] = r.get(0).to_float();
    out[2 * l + 1] = r.get(1).to_float();
  } else {
    out[2 * l] = r.get(0).fpsan_payload();
    out[2 * l + 1] = r.get(1).fpsan_payload();
  }
}

template <class FP8, bool WordSel> void run_cvt_pk() {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  std::vector<float> in(2 * LANES);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : in)
    x = fpsan_test::pick_int_valued<float>(rng, -4, 4); // exact in fp8 + bf8
  float *dIn = to_dev(in);

  // Float: round-trip is lossless for exact-in-fp8 values.
  {
    float *dO;
    HIP_CHECK(hipMalloc(&dO, 2 * LANES * sizeof(float)));
    k_run<Semantics::Native, FP8, WordSel, float><<<1, LANES>>>(dIn, dO);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> got(2 * LANES);
    HIP_CHECK(hipMemcpy(got.data(), dO, 2 * LANES * sizeof(float), hipMemcpyDeviceToHost));
    for (int i = 0; i < 2 * LANES; ++i)
      EXPECT_EQ(got[i], in[i]) << "Float round-trip at " << i;
    (void)hipFree(dO);
  }
  // FPSan: must equal the direct payload-ring composition
  // cast<float>(cast<fp8>), for every FPSan-family semantics.
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VF = Value<float, S, kCC>;
    std::vector<std::uint32_t> ref(2 * LANES);
    for (int i = 0; i < 2 * LANES; ++i)
      ref[i] = fpsan::cast<float>(fpsan::cast<FP8>(VF(in[i]))).fpsan_payload();
    std::uint32_t *dO;
    HIP_CHECK(hipMalloc(&dO, 2 * LANES * sizeof(std::uint32_t)));
    k_run<S, FP8, WordSel, std::uint32_t><<<1, LANES>>>(dIn, dO);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(2 * LANES);
    HIP_CHECK(hipMemcpy(got.data(), dO, 2 * LANES * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for (int i = 0; i < 2 * LANES; ++i)
      EXPECT_EQ(got[i], ref[i]) << "FPSan payload at " << i;
    (void)hipFree(dO);
  });
  (void)hipFree(dIn);
}

TEST(CvtPkF32, Fp8_Word0) { run_cvt_pk<fpsan::fp8_e4m3, 0>(); }
TEST(CvtPkF32, Fp8_Word1) { run_cvt_pk<fpsan::fp8_e4m3, 1>(); }
TEST(CvtPkF32, Bf8_Word0) { run_cvt_pk<fpsan::fp8_e5m2, 0>(); }
TEST(CvtPkF32, Bf8_Word1) { run_cvt_pk<fpsan::fp8_e5m2, 1>(); }

// Cross-mode invariant that the gfx12 cvt_test missed: cvt_pk_fp8_f32<DstLo>
// must write the fp8 byte to the SAME word in Float and FPSan mode. DstLo=true
// -> low word (bytes 0,1), DstLo=false -> high word (bytes 2,3).
template <Semantics S> __global__ void k_pack_words(const float *in, int *lo, int *hi) {
  int l = threadIdx.x;
  Value<float, S, kCC> a{in[2 * l]}, b{in[2 * l + 1]};
  lo[l] = fpsan::amdgcn_cvt_pk_fp8_f32<true, S, kCC>(a, b, 0);  // -> low word
  hi[l] = fpsan::amdgcn_cvt_pk_fp8_f32<false, S, kCC>(a, b, 0); // -> high word
}

TEST(CvtPkF32, PackWritesSameWordInBothModes) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  std::vector<float> in(2 * LANES);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : in)
    x = fpsan_test::pick_int_valued<float>(rng, 1, 4); // nonzero fp8 bytes
  float *dIn = to_dev(in);

  // Float oracle: DstLo touches only the low word, DstHi only the high word.
  {
    int *dLoF, *dHiF;
    HIP_CHECK(hipMalloc(&dLoF, LANES * 4));
    HIP_CHECK(hipMalloc(&dHiF, LANES * 4));
    k_pack_words<Semantics::Native><<<1, LANES>>>(dIn, dLoF, dHiF);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<int> loF(LANES), hiF(LANES);
    HIP_CHECK(hipMemcpy(loF.data(), dLoF, LANES * 4, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hiF.data(), dHiF, LANES * 4, hipMemcpyDeviceToHost));
    for (int l = 0; l < LANES; ++l) {
      EXPECT_EQ(static_cast<unsigned>(loF[l]) & 0xFFFF0000u, 0u) << "Float lo " << l;
      EXPECT_EQ(static_cast<unsigned>(hiF[l]) & 0x0000FFFFu, 0u) << "Float hi " << l;
      EXPECT_NE(static_cast<unsigned>(loF[l]) & 0x0000FFFFu, 0u);
    }
    (void)hipFree(dLoF);
    (void)hipFree(dHiF);
  }
  // The same word-placement invariant must hold in every FPSan-family semantics.
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    int *dLoP, *dHiP;
    HIP_CHECK(hipMalloc(&dLoP, LANES * 4));
    HIP_CHECK(hipMalloc(&dHiP, LANES * 4));
    k_pack_words<S><<<1, LANES>>>(dIn, dLoP, dHiP);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<int> loP(LANES), hiP(LANES);
    HIP_CHECK(hipMemcpy(loP.data(), dLoP, LANES * 4, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hiP.data(), dHiP, LANES * 4, hipMemcpyDeviceToHost));
    for (int l = 0; l < LANES; ++l) {
      EXPECT_EQ(static_cast<unsigned>(loP[l]) & 0xFFFF0000u, 0u) << "FPSan lo " << l;
      EXPECT_EQ(static_cast<unsigned>(hiP[l]) & 0x0000FFFFu, 0u) << "FPSan hi " << l;
      EXPECT_NE(static_cast<unsigned>(loP[l]) & 0x0000FFFFu, 0u);
    }
    (void)hipFree(dLoP);
    (void)hipFree(dHiP);
  });
  (void)hipFree(dIn);
}

// Exercise the device cvt_pk_f32_fp8 / cvt_pk_f32_bf8 unpack over *every* fp8
// bit pattern, against the OCP-correct host decode (fp8_test.cpp locks down
// that host decode). The other tests here use small integer inputs that never
// reach the E4M3 high-exponent bytes 0x79-0x7F -- exactly where the
// NaN-detection rule matters (OCP E4M3FN: 0x79-0x7E are finite normals
// 288..448, only 0x7F is NaN). It catches any implementation whose decode gets
// the NaN boundary (or any byte) wrong.
template <class FP8> __global__ void k_unpack_all_bytes(const int *packed, float *out) {
  int l = threadIdx.x; // lane l carries fp8 byte l in word0/byte0.
  auto r = CvtPk<FP8, 0>::template unpack<Semantics::Native>(packed[l]);
  out[l] = r.get(0).to_float();
}

template <class FP8> void run_unpack_all_bytes() {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  std::vector<int> packed(256);
  for (int b = 0; b < 256; ++b)
    packed[b] = b; // fp8 byte b in byte0; bytes 1-3 zero, word0 element0 = fp8(b)
  int *dIn = to_dev(packed);
  float *dO;
  HIP_CHECK(hipMalloc(&dO, 256 * sizeof(float)));
  k_unpack_all_bytes<FP8><<<1, 256>>>(dIn, dO);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> got(256);
  HIP_CHECK(hipMemcpy(got.data(), dO, 256 * sizeof(float), hipMemcpyDeviceToHost));
  for (int b = 0; b < 256; ++b) {
    float ref = static_cast<float>(FP8(static_cast<std::uint8_t>(b)));
    if (std::isnan(ref))
      EXPECT_TRUE(std::isnan(got[b])) << "byte 0x" << std::hex << b;
    else
      EXPECT_EQ(got[b], ref) << "byte 0x" << std::hex << b;
  }
  (void)hipFree(dIn);
  (void)hipFree(dO);
}

TEST(CvtPkF32, Fp8UnpackAllBytes) { run_unpack_all_bytes<fpsan::fp8_e4m3>(); }
TEST(CvtPkF32, Bf8UnpackAllBytes) { run_unpack_all_bytes<fpsan::fp8_e5m2>(); }

// ---------------------------------------------------------------------------
// Coverage for the rest of the PR's non-scaled cvt (cvt_sr) and the fp8
// cvt_scalef32 ops, via round-trips through the already-validated unpack ops.
// All references are host-computed (independent of the device builtin), so any
// implementation that diverges from them shows up as a failure.
// ---------------------------------------------------------------------------
template <class FP8> struct CvtOps;
template <> struct CvtOps<fpsan::fp8_e4m3> {
  template <int B, Semantics S>
  __device__ static int sr(Value<float, S, kCC> v, int old, unsigned seed) {
    return fpsan::amdgcn_cvt_sr_fp8_f32<B, S, kCC>(v, old, seed);
  }
  template <int B, Semantics S> __device__ static Value<float, S, kCC> f32(int p) {
    return fpsan::amdgcn_cvt_f32_fp8<B, S, kCC>(p);
  }
  template <int B, Semantics S>
  __device__ static Value<float, S, kCC> sf32(int p, Value<float, S, kCC> sc) {
    return fpsan::amdgcn_cvt_scalef32_f32_fp8<B, S, kCC>(p, sc);
  }
};
template <> struct CvtOps<fpsan::fp8_e5m2> {
  template <int B, Semantics S>
  __device__ static int sr(Value<float, S, kCC> v, int old, unsigned seed) {
    return fpsan::amdgcn_cvt_sr_bf8_f32<B, S, kCC>(v, old, seed);
  }
  template <int B, Semantics S> __device__ static Value<float, S, kCC> f32(int p) {
    return fpsan::amdgcn_cvt_f32_bf8<B, S, kCC>(p);
  }
  template <int B, Semantics S>
  __device__ static Value<float, S, kCC> sf32(int p, Value<float, S, kCC> sc) {
    return fpsan::amdgcn_cvt_scalef32_f32_bf8<B, S, kCC>(p, sc);
  }
};

// cvt_sr<byte0>(v) then cvt_f32<byte0> -> v (exact in fp8). FPSan: the cast
// round-trip cast<float>(cast<fp8>(v)).
template <Semantics S, class FP8, class Out>
__global__ void k_sr_roundtrip(const float *in, Out *out) {
  int l = threadIdx.x;
  Value<float, S, kCC> v{in[l]};
  int packed = CvtOps<FP8>::template sr<0, S>(v, 0, 0x1234u + l);
  auto f = CvtOps<FP8>::template f32<0, S>(packed);
  if constexpr (S == Semantics::Native)
    out[l] = f.to_float();
  else
    out[l] = f.fpsan_payload();
}

template <class FP8> void run_cvt_sr() {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  std::vector<float> in(LANES);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : in)
    x = fpsan_test::pick_int_valued<float>(rng, -4, 4);
  float *dIn = to_dev(in);
  {
    float *dO;
    HIP_CHECK(hipMalloc(&dO, LANES * sizeof(float)));
    k_sr_roundtrip<Semantics::Native, FP8, float><<<1, LANES>>>(dIn, dO);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> got(LANES);
    HIP_CHECK(hipMemcpy(got.data(), dO, LANES * sizeof(float), hipMemcpyDeviceToHost));
    for (int i = 0; i < LANES; ++i)
      EXPECT_EQ(got[i], in[i]) << "sr round-trip " << i;
    (void)hipFree(dO);
  }
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VF = Value<float, S, kCC>;
    std::vector<std::uint32_t> ref(LANES);
    for (int i = 0; i < LANES; ++i)
      ref[i] = fpsan::cast<float>(fpsan::cast<FP8>(VF(in[i]))).fpsan_payload();
    std::uint32_t *dO;
    HIP_CHECK(hipMalloc(&dO, LANES * sizeof(std::uint32_t)));
    k_sr_roundtrip<S, FP8, std::uint32_t><<<1, LANES>>>(dIn, dO);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(LANES);
    HIP_CHECK(hipMemcpy(got.data(), dO, LANES * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for (int i = 0; i < LANES; ++i)
      EXPECT_EQ(got[i], ref[i]) << "sr FPSan " << i;
    (void)hipFree(dO);
  });
  (void)hipFree(dIn);
}

TEST(CvtSr, Fp8) { run_cvt_sr<fpsan::fp8_e4m3>(); }
TEST(CvtSr, Bf8) { run_cvt_sr<fpsan::fp8_e5m2>(); }

// cvt_scalef32_f32_{fp8,bf8}: build a register with fp8(v) in byte0 (via the
// SR pack), then unpack-with-scale -> v*scale. scale is a power of 2 so the
// product stays exact for our small-int v.
template <Semantics S, class FP8, class Out>
__global__ void k_scalef32_unpack(const float *in, Out *out, float scale) {
  int l = threadIdx.x;
  Value<float, S, kCC> v{in[l]};
  int packed = CvtOps<FP8>::template sr<0, S>(v, 0, 0u);
  auto f = CvtOps<FP8>::template sf32<0, S>(packed, Value<float, S, kCC>{scale});
  if constexpr (S == Semantics::Native)
    out[l] = f.to_float();
  else
    out[l] = f.fpsan_payload();
}

template <class FP8> void run_cvt_scalef32_unpack() {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  const float scale = 4.0f;
  std::vector<float> in(LANES);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : in)
    x = fpsan_test::pick_int_valued<float>(rng, -4, 4);
  float *dIn = to_dev(in);
  {
    float *dO;
    HIP_CHECK(hipMalloc(&dO, LANES * sizeof(float)));
    k_scalef32_unpack<Semantics::Native, FP8, float><<<1, LANES>>>(dIn, dO, scale);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> got(LANES);
    HIP_CHECK(hipMemcpy(got.data(), dO, LANES * sizeof(float), hipMemcpyDeviceToHost));
    for (int i = 0; i < LANES; ++i)
      EXPECT_EQ(got[i], in[i] * scale) << "scalef32 unpack " << i;
    (void)hipFree(dO);
  }
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VF = Value<float, S, kCC>;
    std::vector<std::uint32_t> ref(LANES);
    for (int i = 0; i < LANES; ++i)
      ref[i] = (fpsan::cast<float>(fpsan::cast<FP8>(VF(in[i]))) * VF(scale)).fpsan_payload();
    std::uint32_t *dO;
    HIP_CHECK(hipMalloc(&dO, LANES * sizeof(std::uint32_t)));
    k_scalef32_unpack<S, FP8, std::uint32_t><<<1, LANES>>>(dIn, dO, scale);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(LANES);
    HIP_CHECK(hipMemcpy(got.data(), dO, LANES * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for (int i = 0; i < LANES; ++i)
      EXPECT_EQ(got[i], ref[i]) << "scalef32 FPSan " << i;
    (void)hipFree(dO);
  });
  (void)hipFree(dIn);
}

TEST(CvtScalef32F32, Fp8) { run_cvt_scalef32_unpack<fpsan::fp8_e4m3>(); }
TEST(CvtScalef32F32, Bf8) { run_cvt_scalef32_unpack<fpsan::fp8_e5m2>(); }

// cvt_scalef32_pk_{fp8,bf8}_f32 (the scaled PACK). DIRECTION: pack stores
// value/scale (silicon-verified), so pack(x,s) then unpack(s) round-trips.
// Native mode forwards to the builtin (always correct); the FPSan branch must
// model the SAME divide-by-scale, else a kernel run under FPSan computes a
// different algorithm than on hardware. The FPSan reference below divides;
// this test fails if the wrapper multiplies (the prior bug).
template <class FP8> struct ScaledPackOps;
template <> struct ScaledPackOps<fpsan::fp8_e4m3> {
  template <Semantics S>
  __device__ static int pack(Value<float, S, kCC> a, Value<float, S, kCC> b,
                             Value<float, S, kCC> sc, int old) {
    return fpsan::amdgcn_cvt_scalef32_pk_fp8_f32<true, S, kCC>(a, b, sc, old);
  }
  template <Semantics S>
  __device__ static Value<float, S, kCC> unpack0(int p, Value<float, S, kCC> sc) {
    return fpsan::amdgcn_cvt_scalef32_f32_fp8<0, S, kCC>(p, sc);
  }
};
template <> struct ScaledPackOps<fpsan::fp8_e5m2> {
  template <Semantics S>
  __device__ static int pack(Value<float, S, kCC> a, Value<float, S, kCC> b,
                             Value<float, S, kCC> sc, int old) {
    return fpsan::amdgcn_cvt_scalef32_pk_bf8_f32<true, S, kCC>(a, b, sc, old);
  }
  template <Semantics S>
  __device__ static Value<float, S, kCC> unpack0(int p, Value<float, S, kCC> sc) {
    return fpsan::amdgcn_cvt_scalef32_f32_bf8<0, S, kCC>(p, sc);
  }
};

template <Semantics S, class FP8, class Out>
__global__ void k_scaled_pack(const float *in, Out *out, float scale) {
  int l = threadIdx.x;
  Value<float, S, kCC> v{in[l]};
  Value<float, S, kCC> sc{scale};
  int packed = ScaledPackOps<FP8>::pack(v, Value<float, S, kCC>{0.0f}, sc, 0);
  if constexpr (S == Semantics::Native) {
    auto f = ScaledPackOps<FP8>::unpack0(packed, sc);
    out[l] = f.to_float();
  } else {
    out[l] = static_cast<std::uint32_t>(packed) & 0xFFu; // byte0 payload
  }
}

template <class FP8> void run_cvt_scalef32_pack() {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  const float scale = 2.0f;
  std::vector<float> in(LANES);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : in)
    x = fpsan_test::pick_int_valued<float>(rng, -4, 4); // v/2 exact in fp8
  float *dIn = to_dev(in);
  {
    float *dO;
    HIP_CHECK(hipMalloc(&dO, LANES * sizeof(float)));
    k_scaled_pack<Semantics::Native, FP8, float><<<1, LANES>>>(dIn, dO, scale);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<float> got(LANES);
    HIP_CHECK(hipMemcpy(got.data(), dO, LANES * sizeof(float), hipMemcpyDeviceToHost));
    for (int i = 0; i < LANES; ++i)
      EXPECT_EQ(got[i], in[i]) << "scaled pack/unpack round-trip " << i;
    (void)hipFree(dO);
  }
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VF = Value<float, S, kCC>;
    std::vector<std::uint32_t> ref(LANES);
    for (int i = 0; i < LANES; ++i)
      ref[i] = static_cast<std::uint32_t>(fpsan::cast<FP8>(VF(in[i]) / VF(scale)).fpsan_payload()) &
               0xFFu;
    std::uint32_t *dO;
    HIP_CHECK(hipMalloc(&dO, LANES * sizeof(std::uint32_t)));
    k_scaled_pack<S, FP8, std::uint32_t><<<1, LANES>>>(dIn, dO, scale);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(LANES);
    HIP_CHECK(hipMemcpy(got.data(), dO, LANES * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for (int i = 0; i < LANES; ++i)
      EXPECT_EQ(got[i], ref[i]) << "scaled pack FPSan (divide-by-scale) " << i;
    (void)hipFree(dO);
  });
  (void)hipFree(dIn);
}

TEST(CvtScalef32Pack, Fp8) { run_cvt_scalef32_pack<fpsan::fp8_e4m3>(); }
TEST(CvtScalef32Pack, Bf8) { run_cvt_scalef32_pack<fpsan::fp8_e5m2>(); }
