// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/cvt_scalef32_sr_gfx950_test.cpp
//
// gfx950 tests for the final 14 cvt_scalef32 WRAPPERS in fpsan/amdgcn_cvt.hpp
// that close the suite to 100% FPSan coverage of the FP-relevant gfx950
// intrinsics:
//   (a) cvt_scalef32_f16_{fp8,bf8}  -- tied byte-select unpack to one f16 half
//   (b) cvt_scalef32_sr_{fp8,bf8}_{f32,f16,bf16} -- SR pack to an fp8/bf8 byte
//   (c) cvt_scalef32_sr_pk32_{fp6,bf6}_{f16,bf16} -- SR contiguous pack to
//   v6u32 (d) cvt_scalef32_sr_pk_fp4_{f16,bf16} -- SR pack to an fp4 nibble
//   pair
//
// Two independent oracles per op (so any implementation that diverges from the
// host reference fails):
//   Float : wrapper == hardware, checked against the INDEPENDENT host OCP
//           reference (detail::narrow_to_f32 / f32_to_narrow), NOT the builtin.
//   FPSan : wrapper == an independent payload-domain reference built from the
//           public Value arithmetic and canonical subbyte casts (pack DIVIDES
//           by scale, unpack MULTIPLIES -- the silicon-verified MX direction). The scale
//           is deliberately != 1 so a flipped mul/div direction is caught (a
//           pack->unpack round-trip would cancel it; these do not round-trip).
//           For the SR ops the seed is varied to confirm it is opaque to the
//           deterministic answer for exactly-representable inputs.
#include "fpsan/amdgcn_cvt.hpp"
#include "fpsan/fpsan.hpp"

#include "fpsan_semantics.hpp"
#include "subbyte_oracle.hpp"

#include "hip_test_utils.hpp"
#include "test_random.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

using fpsan::Conversions;
using fpsan::fp8_e4m3;
using fpsan::fp8_e5m2;
using fpsan::Semantics;
using fpsan::Value;
using fpsan::detail::f32_to_narrow;
using fpsan::detail::kBf6E3M2;
using fpsan::detail::kFp4E2M1;
using fpsan::detail::kFp6E2M3;
using fpsan::detail::kFp8E4M3;
using fpsan::detail::kFp8E5M2;
using fpsan::detail::narrow_to_f32;

static constexpr Conversions kCC = Conversions::Explicit;
using v6u = unsigned __attribute__((ext_vector_type(6)));

namespace {
// Independent host 6-bit bitstream pack (separate impl from the device side).
void host_pack6(const std::uint8_t *codes, int n, unsigned *out) {
  for (int i = 0; i < 6; ++i)
    out[i] = 0;
  for (int i = 0; i < n; ++i)
    for (int b = 0; b < 6; ++b)
      if ((codes[i] >> b) & 1) {
        int p = i * 6 + b;
        out[p / 32] |= 1u << (p % 32);
      }
}
} // namespace

// ===========================================================================
// (a) cvt_scalef32_f16_{fp8,bf8}: tied byte-select unpack -> one f16 half.
// ===========================================================================
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_scalef32_f16_fp8)

template <class FP8> struct F16Fp8;
template <> struct F16Fp8<fp8_e4m3> {
  template <int Byte, bool Hi, Semantics S>
  __device__ static Value<fpsan::v2h_native, S, kCC> call(Value<fpsan::v2h_native, S, kCC> old,
                                                          int src, Value<float, S, kCC> s) {
    return fpsan::amdgcn_cvt_scalef32_f16_fp8<Byte, Hi, S, kCC>(old, src, s);
  }
};
template <> struct F16Fp8<fp8_e5m2> {
  template <int Byte, bool Hi, Semantics S>
  __device__ static Value<fpsan::v2h_native, S, kCC> call(Value<fpsan::v2h_native, S, kCC> old,
                                                          int src, Value<float, S, kCC> s) {
    return fpsan::amdgcn_cvt_scalef32_f16_bf8<Byte, Hi, S, kCC>(old, src, s);
  }
};

// Float: byte 0 of src over all 256 values, Hi=false, scale=1; lane0 == host
// OCP decode widened to f16, lane1 preserved from old (=222).
template <class FP8> __global__ void k_a_float_all(const int *src, float *lane0, float *lane1) {
  int l = threadIdx.x;
  fpsan::v2h_native ov = {static_cast<_Float16>(111.0f), static_cast<_Float16>(222.0f)};
  auto r = F16Fp8<FP8>::template call<0, false, Semantics::Native>(
      Value<fpsan::v2h_native, Semantics::Native, kCC>(ov), src[l],
      Value<float, Semantics::Native, kCC>{1.0f});
  lane0[l] = static_cast<float>(static_cast<_Float16>(r.get(0)));
  lane1[l] = static_cast<float>(static_cast<_Float16>(r.get(1)));
}

template <class FP8> void run_a_float_all(const fpsan::detail::FpFormat &fmt, const char *name) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  std::vector<int> src(256);
  for (int b = 0; b < 256; ++b)
    src[b] = b; // fp8 byte b in byte0
  int *dS = to_dev(src);
  float *d0, *d1;
  HIP_CHECK(hipMalloc(&d0, 256 * sizeof(float)));
  HIP_CHECK(hipMalloc(&d1, 256 * sizeof(float)));
  k_a_float_all<FP8><<<1, 256>>>(dS, d0, d1);
  HIP_CHECK(hipDeviceSynchronize());
  auto g0 = from_dev(d0, 256), g1 = from_dev(d1, 256);
  for (int b = 0; b < 256; ++b) {
    float dec = narrow_to_f32(static_cast<unsigned>(b), fmt);
    float ref = static_cast<float>(static_cast<_Float16>(dec));
    if (std::isnan(ref))
      EXPECT_TRUE(std::isnan(g0[b])) << name << " byte 0x" << std::hex << b;
    else
      EXPECT_EQ(g0[b], ref) << name << " byte 0x" << std::hex << b;
    EXPECT_EQ(g1[b], 222.0f) << name << " preserve byte 0x" << std::hex << b;
  }
  (void)hipFree(dS);
  (void)hipFree(d0);
  (void)hipFree(d1);
}

TEST(CvtScalef32Sr, A_FloatUnpackAll_Fp8) { run_a_float_all<fp8_e4m3>(kFp8E4M3, "fp8"); }
TEST(CvtScalef32Sr, A_FloatUnpackAll_Bf8) { run_a_float_all<fp8_e5m2>(kFp8E5M2, "bf8"); }

// Float: byte-select + lo/hi placement + scale-multiply direction. src carries
// 4 distinct bytes; for each (Byte, Hi) the chosen half = decode(byte)*scale,
// the other half is preserved.
template <class FP8> __global__ void k_a_float_sel(int src, float scale, float *out) {
  fpsan::v2h_native ov = {static_cast<_Float16>(111.0f), static_cast<_Float16>(222.0f)};
  using VV = Value<fpsan::v2h_native, Semantics::Native, kCC>;
  using VS = Value<float, Semantics::Native, kCC>;
  auto w = [&](int idx, auto r) {
    out[2 * idx] = static_cast<float>(static_cast<_Float16>(r.get(0)));
    out[2 * idx + 1] = static_cast<float>(static_cast<_Float16>(r.get(1)));
  };
  w(0, (F16Fp8<FP8>::template call<0, false, Semantics::Native>(VV(ov), src, VS{scale})));
  w(1, (F16Fp8<FP8>::template call<0, true, Semantics::Native>(VV(ov), src, VS{scale})));
  w(2, (F16Fp8<FP8>::template call<1, false, Semantics::Native>(VV(ov), src, VS{scale})));
  w(3, (F16Fp8<FP8>::template call<2, true, Semantics::Native>(VV(ov), src, VS{scale})));
  w(4, (F16Fp8<FP8>::template call<3, false, Semantics::Native>(VV(ov), src, VS{scale})));
}

template <class FP8> void run_a_float_sel(const fpsan::detail::FpFormat &fmt, const char *name) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  // Four bytes that are exact, small, nonzero (so *scale is also exact in f16).
  std::uint8_t bytes[4];
  bytes[0] = static_cast<std::uint8_t>(f32_to_narrow(1.0f, fmt));
  bytes[1] = static_cast<std::uint8_t>(f32_to_narrow(-2.0f, fmt));
  bytes[2] = static_cast<std::uint8_t>(f32_to_narrow(0.5f, fmt));
  bytes[3] = static_cast<std::uint8_t>(f32_to_narrow(3.0f, fmt));
  int src = bytes[0] | (bytes[1] << 8) | (bytes[2] << 16) | (bytes[3] << 24);
  const float scale = 2.0f;
  float *dO;
  HIP_CHECK(hipMalloc(&dO, 10 * sizeof(float)));
  k_a_float_sel<FP8><<<1, 1>>>(src, scale, dO);
  HIP_CHECK(hipDeviceSynchronize());
  auto g = from_dev(dO, 10);
  auto dec = [&](int byteIdx) {
    return static_cast<float>(static_cast<_Float16>(narrow_to_f32(bytes[byteIdx], fmt) * scale));
  };
  // (Byte=0,Hi=false): lane0=dec(0), lane1=222
  EXPECT_EQ(g[0], dec(0)) << name;
  EXPECT_EQ(g[1], 222.0f) << name;
  // (Byte=0,Hi=true): lane0=111, lane1=dec(0)
  EXPECT_EQ(g[2], 111.0f) << name;
  EXPECT_EQ(g[3], dec(0)) << name;
  // (Byte=1,Hi=false)
  EXPECT_EQ(g[4], dec(1)) << name;
  EXPECT_EQ(g[5], 222.0f) << name;
  // (Byte=2,Hi=true)
  EXPECT_EQ(g[6], 111.0f) << name;
  EXPECT_EQ(g[7], dec(2)) << name;
  // (Byte=3,Hi=false)
  EXPECT_EQ(g[8], dec(3)) << name;
  EXPECT_EQ(g[9], 222.0f) << name;
  (void)hipFree(dO);
}

TEST(CvtScalef32Sr, A_FloatSelLoHiScale_Fp8) { run_a_float_sel<fp8_e4m3>(kFp8E4M3, "fp8"); }
TEST(CvtScalef32Sr, A_FloatSelLoHiScale_Bf8) { run_a_float_sel<fp8_e5m2>(kFp8E5M2, "bf8"); }

// FPSan: src holds fp8 PAYLOAD bytes; the chosen half = cast<_Float16>(
// cast<float>(payload) * scale) and the other half is preserved from old.
template <Semantics S, class FP8> __global__ void k_a_fpsan(int src, float scale, unsigned *out) {
  using VF = Value<float, S, kCC>;
  using VV = Value<fpsan::v2h_native, S, kCC>;
  VV old{};
  old.set(0, Value<_Float16, S, kCC>::from_fpsan_payload(0x1234));
  old.set(1, Value<_Float16, S, kCC>::from_fpsan_payload(0x5678));
  auto lo = F16Fp8<FP8>::template call<1, false, S>(old, src, VF{scale});
  auto hi = F16Fp8<FP8>::template call<2, true, S>(old, src, VF{scale});
  out[0] = lo.get(0).fpsan_payload();
  out[1] = lo.get(1).fpsan_payload();
  out[2] = hi.get(0).fpsan_payload();
  out[3] = hi.get(1).fpsan_payload();
}

template <class FP8> void run_a_fpsan() {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  // payload bytes 0xA3 (byte1) and 0x5C (byte2).
  unsigned byte1 = 0xA3, byte2 = 0x5C;
  int src = static_cast<int>((byte1 << 8) | (byte2 << 16));
  const float scale = 2.0f;
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VF = Value<float, S, kCC>;
    using V8 = Value<FP8, S, kCC>;
    unsigned *dO;
    HIP_CHECK(hipMalloc(&dO, 4 * sizeof(unsigned)));
    k_a_fpsan<S, FP8><<<1, 1>>>(src, scale, dO);
    HIP_CHECK(hipDeviceSynchronize());
    auto g = from_dev(dO, 4);
    unsigned ref_lo =
        fpsan::cast<_Float16>(fpsan::cast<float>(V8::from_fpsan_payload(byte1)) * VF{scale})
            .fpsan_payload();
    unsigned ref_hi =
        fpsan::cast<_Float16>(fpsan::cast<float>(V8::from_fpsan_payload(byte2)) * VF{scale})
            .fpsan_payload();
    EXPECT_EQ(g[0], ref_lo);  // lo: lane0 written
    EXPECT_EQ(g[1], 0x5678u); // lo: lane1 preserved
    EXPECT_EQ(g[2], 0x1234u); // hi: lane0 preserved
    EXPECT_EQ(g[3], ref_hi);  // hi: lane1 written
    (void)hipFree(dO);
  });
}

TEST(CvtScalef32Sr, A_FpsanUnpackMulAndLane_Fp8) { run_a_fpsan<fp8_e4m3>(); }
TEST(CvtScalef32Sr, A_FpsanUnpackMulAndLane_Bf8) { run_a_fpsan<fp8_e5m2>(); }

#endif // (a)

// ===========================================================================
// (b) cvt_scalef32_sr_{fp8,bf8}_{f32,f16,bf16}: SR pack to one fp8/bf8 byte.
// ===========================================================================
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_scalef32_sr_fp8_f32)

// Functor over (FP8, SrcElem) selecting the right wrapper.
template <class FP8, class SrcElem> struct SrFp8;
#define SRFP8_SPEC(FP8T, SRC, FN)                                                                  \
  template <> struct SrFp8<FP8T, SRC> {                                                            \
    template <int Byte, Semantics S>                                                               \
    __device__ static int call(int old, Value<SRC, S, kCC> v, unsigned seed,                       \
                               Value<float, S, kCC> sc) {                                          \
      return fpsan::FN<Byte, S, kCC>(old, v, seed, sc);                                            \
    }                                                                                              \
  }
SRFP8_SPEC(fp8_e4m3, float, amdgcn_cvt_scalef32_sr_fp8_f32);
SRFP8_SPEC(fp8_e4m3, _Float16, amdgcn_cvt_scalef32_sr_fp8_f16);
SRFP8_SPEC(fp8_e4m3, __bf16, amdgcn_cvt_scalef32_sr_fp8_bf16);
SRFP8_SPEC(fp8_e5m2, float, amdgcn_cvt_scalef32_sr_bf8_f32);
SRFP8_SPEC(fp8_e5m2, _Float16, amdgcn_cvt_scalef32_sr_bf8_f16);
SRFP8_SPEC(fp8_e5m2, __bf16, amdgcn_cvt_scalef32_sr_bf8_bf16);
#undef SRFP8_SPEC

template <Semantics S, class FP8, class SrcElem, class Out>
__global__ void k_b(int old, const float *vin, unsigned seed, float scale, Out *out) {
  int l = threadIdx.x;
  Value<SrcElem, S, kCC> v{static_cast<SrcElem>(vin[l])};
  // Byte index cycles 0..3 over lanes to exercise all slots.
  int r;
  switch (l & 3) {
  case 0:
    r = SrFp8<FP8, SrcElem>::template call<0, S>(old, v, seed, Value<float, S, kCC>{scale});
    break;
  case 1:
    r = SrFp8<FP8, SrcElem>::template call<1, S>(old, v, seed, Value<float, S, kCC>{scale});
    break;
  case 2:
    r = SrFp8<FP8, SrcElem>::template call<2, S>(old, v, seed, Value<float, S, kCC>{scale});
    break;
  default:
    r = SrFp8<FP8, SrcElem>::template call<3, S>(old, v, seed, Value<float, S, kCC>{scale});
    break;
  }
  out[l] = static_cast<Out>(r);
}

template <class FP8, class SrcElem> void run_b(const fpsan::detail::FpFormat &fmt) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  const int N = 64;
  const int old = static_cast<int>(0xDEADBEEFu);
  const float scale = 2.0f;
  std::mt19937 rng = fpsan_test::make_rng();
  std::vector<float> vin(N);
  for (auto &x : vin)
    x = fpsan_test::pick_int_valued<float>(rng, -4, 4); // x/2 exact in fp8
  float *dV = to_dev(vin);

  // ---- Float: byte `l&3` == host encode(SrcElem(x)/scale), others preserved.
  {
    int *dO;
    HIP_CHECK(hipMalloc(&dO, N * sizeof(int)));
    k_b<Semantics::Native, FP8, SrcElem, int><<<1, N>>>(old, dV, 0xABCDu, scale, dO);
    HIP_CHECK(hipDeviceSynchronize());
    auto g = from_dev(dO, N);
    for (int l = 0; l < N; ++l) {
      // Source is rounded through SrcElem first (matches the hardware src reg).
      float sv = static_cast<float>(static_cast<SrcElem>(vin[l]));
      std::uint8_t byte = static_cast<std::uint8_t>(f32_to_narrow(sv / scale, fmt));
      int sel = l & 3;
      unsigned mask = 0xFFu << (8 * sel);
      unsigned expect =
          (static_cast<unsigned>(old) & ~mask) | (static_cast<unsigned>(byte) << (8 * sel));
      EXPECT_EQ(static_cast<unsigned>(g[l]), expect) << "Float lane " << l;
    }
    (void)hipFree(dO);
  }
  // ---- FPSan: byte == cast<FP8>(cast<float>(v)/scale) payload; seed opaque.
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VF = Value<float, S, kCC>;
    using VS = Value<SrcElem, S, kCC>;
    unsigned *dO;
    HIP_CHECK(hipMalloc(&dO, N * sizeof(unsigned)));
    // A different seed must give the same answer (opaque).
    k_b<S, FP8, SrcElem, unsigned><<<1, N>>>(old, dV, 0x12345u, scale, dO);
    HIP_CHECK(hipDeviceSynchronize());
    auto g = from_dev(dO, N);
    for (int l = 0; l < N; ++l) {
      VS v{static_cast<SrcElem>(vin[l])};
      auto f8 = fpsan::cast<FP8>(fpsan::cast<float>(v) / VF{scale});
      std::uint8_t byte = static_cast<std::uint8_t>(f8.fpsan_payload());
      int sel = l & 3;
      unsigned mask = 0xFFu << (8 * sel);
      unsigned expect =
          (static_cast<unsigned>(old) & ~mask) | (static_cast<unsigned>(byte) << (8 * sel));
      EXPECT_EQ(g[l], expect) << "FPSan lane " << l;
    }
    (void)hipFree(dO);
  });
  (void)hipFree(dV);
}

TEST(CvtScalef32Sr, B_Fp8_F32) { run_b<fp8_e4m3, float>(kFp8E4M3); }
TEST(CvtScalef32Sr, B_Fp8_F16) { run_b<fp8_e4m3, _Float16>(kFp8E4M3); }
TEST(CvtScalef32Sr, B_Fp8_BF16) { run_b<fp8_e4m3, __bf16>(kFp8E4M3); }
TEST(CvtScalef32Sr, B_Bf8_F32) { run_b<fp8_e5m2, float>(kFp8E5M2); }
TEST(CvtScalef32Sr, B_Bf8_F16) { run_b<fp8_e5m2, _Float16>(kFp8E5M2); }
TEST(CvtScalef32Sr, B_Bf8_BF16) { run_b<fp8_e5m2, __bf16>(kFp8E5M2); }

#endif // (b)

// ===========================================================================
// (c) cvt_scalef32_sr_pk32_{fp6,bf6}_{f16,bf16}: SR contiguous pack -> v6u32.
// ===========================================================================
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_scalef32_sr_pk32_fp6_f16)

template <class SrcElem, bool IsFp6> struct SrPk32;
template <> struct SrPk32<_Float16, true> {
  template <Semantics S>
  __device__ static v6u call(Value<fpsan::v32h_native, S, kCC> v, unsigned seed,
                             Value<float, S, kCC> sc) {
    return fpsan::amdgcn_cvt_scalef32_sr_pk32_fp6_f16<S, kCC>(v, seed, sc);
  }
};
template <> struct SrPk32<__bf16, true> {
  template <Semantics S>
  __device__ static v6u call(Value<fpsan::v32bf_native, S, kCC> v, unsigned seed,
                             Value<float, S, kCC> sc) {
    return fpsan::amdgcn_cvt_scalef32_sr_pk32_fp6_bf16<S, kCC>(v, seed, sc);
  }
};
template <> struct SrPk32<_Float16, false> {
  template <Semantics S>
  __device__ static v6u call(Value<fpsan::v32h_native, S, kCC> v, unsigned seed,
                             Value<float, S, kCC> sc) {
    return fpsan::amdgcn_cvt_scalef32_sr_pk32_bf6_f16<S, kCC>(v, seed, sc);
  }
};
template <> struct SrPk32<__bf16, false> {
  template <Semantics S>
  __device__ static v6u call(Value<fpsan::v32bf_native, S, kCC> v, unsigned seed,
                             Value<float, S, kCC> sc) {
    return fpsan::amdgcn_cvt_scalef32_sr_pk32_bf6_bf16<S, kCC>(v, seed, sc);
  }
};

template <Semantics S, class SrcElem, class VEC, bool IsFp6>
__global__ void k_c(const float *in, unsigned seed, float scale, unsigned *out) {
  VEC v;
  for (int i = 0; i < 32; ++i)
    v[i] = static_cast<SrcElem>(in[i]);
  v6u r = SrPk32<SrcElem, IsFp6>::template call<S>(Value<VEC, S, kCC>(v), seed,
                                                   Value<float, S, kCC>{scale});
  for (int i = 0; i < 6; ++i)
    out[i] = r[i];
}

template <class SrcElem, class VEC, bool IsFp6> void run_c(const fpsan::detail::FpFormat &fmt) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  const float scale = 2.0f;
  std::mt19937 rng = fpsan_test::make_rng();
  std::vector<float> in(32);
  for (auto &x : in)
    x = fpsan_test::pick_int_valued<float>(rng, -6, 6); // x/2 representable
  float *dIn = to_dev(in);
  unsigned *dO;
  HIP_CHECK(hipMalloc(&dO, 6 * sizeof(unsigned)));

  // ---- Float: contiguous stream of host codes
  // f32_to_narrow(SrcElem(x)/scale).
  k_c<Semantics::Native, SrcElem, VEC, IsFp6><<<1, 1>>>(dIn, 0xABCDu, scale, dO);
  HIP_CHECK(hipDeviceSynchronize());
  {
    auto g = from_dev(dO, 6);
    std::uint8_t codes[32];
    for (int i = 0; i < 32; ++i) {
      float sv = static_cast<float>(static_cast<SrcElem>(in[i]));
      codes[i] = static_cast<std::uint8_t>(f32_to_narrow(sv / scale, fmt)) & 0x3F;
    }
    unsigned expect[6];
    host_pack6(codes, 32, expect);
    for (int i = 0; i < 6; ++i)
      EXPECT_EQ(g[i], expect[i]) << "Float word " << i;
  }
  in[5] = std::numeric_limits<float>::quiet_NaN();
  in[9] = std::numeric_limits<float>::infinity();
  in[23] = std::numeric_limits<float>::quiet_NaN();
  in[27] = -std::numeric_limits<float>::infinity();
  (void)hipFree(dIn);
  dIn = to_dev(in);
  // ---- FPSan: contiguous stream of canonical subbyte narrow codes; seed opaque.
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VF = Value<float, S, kCC>;
    k_c<S, SrcElem, VEC, IsFp6><<<1, 1>>>(dIn, 0x999u, scale, dO);
    HIP_CHECK(hipDeviceSynchronize());
    using VS = Value<SrcElem, S, kCC>;
    auto g = from_dev(dO, 6);
    std::uint8_t codes[32];
    for (int i = 0; i < 32; ++i) {
      VS v{static_cast<SrcElem>(in[i])};
      codes[i] = static_cast<std::uint8_t>(
          fpsan_test::canonical_subbyte_narrow_code<6>(fpsan::cast<float>(v) / VF{scale}));
    }
    unsigned expect[6];
    host_pack6(codes, 32, expect);
    for (int i = 0; i < 6; ++i)
      EXPECT_EQ(g[i], expect[i]) << "FPSan word " << i;
  });
  (void)hipFree(dIn);
  (void)hipFree(dO);
}

TEST(CvtScalef32Sr, C_Fp6_F16) { run_c<_Float16, fpsan::v32h_native, true>(kFp6E2M3); }
TEST(CvtScalef32Sr, C_Fp6_BF16) { run_c<__bf16, fpsan::v32bf_native, true>(kFp6E2M3); }
TEST(CvtScalef32Sr, C_Bf6_F16) { run_c<_Float16, fpsan::v32h_native, false>(kBf6E3M2); }
TEST(CvtScalef32Sr, C_Bf6_BF16) { run_c<__bf16, fpsan::v32bf_native, false>(kBf6E3M2); }

#endif // (c)

// ===========================================================================
// (d) cvt_scalef32_sr_pk_fp4_{f16,bf16}: SR pack to one fp4 nibble pair.
// ===========================================================================
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_scalef32_sr_pk_fp4_f16)

template <class SrcElem, class VEC> struct SrPkFp4;
template <> struct SrPkFp4<_Float16, fpsan::v2h_native> {
  template <int Sel, Semantics S>
  __device__ static unsigned call(unsigned old, Value<fpsan::v2h_native, S, kCC> v, unsigned seed,
                                  Value<float, S, kCC> sc) {
    return fpsan::amdgcn_cvt_scalef32_sr_pk_fp4_f16<Sel, S, kCC>(old, v, seed, sc);
  }
};
template <> struct SrPkFp4<__bf16, fpsan::v2bf_native> {
  template <int Sel, Semantics S>
  __device__ static unsigned call(unsigned old, Value<fpsan::v2bf_native, S, kCC> v, unsigned seed,
                                  Value<float, S, kCC> sc) {
    return fpsan::amdgcn_cvt_scalef32_sr_pk_fp4_bf16<Sel, S, kCC>(old, v, seed, sc);
  }
};

template <Semantics S, class SrcElem, class VEC>
__global__ void k_d(unsigned old, float a, float b, unsigned seed, float scale, unsigned *out) {
  VEC v;
  v[0] = static_cast<SrcElem>(a);
  v[1] = static_cast<SrcElem>(b);
  *out = SrPkFp4<SrcElem, VEC>::template call<0, S>(old, Value<VEC, S, kCC>(v), seed,
                                                    Value<float, S, kCC>{scale});
}

template <class SrcElem, class VEC> void run_d() {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  const float scale = 2.0f;
  const unsigned old = 0u;
  std::mt19937 rng = fpsan_test::make_rng();
  unsigned *dO;
  HIP_CHECK(hipMalloc(&dO, sizeof(unsigned)));
  // fp4 e2m1 representable magnitudes; a/scale must be EXACT or hardware SR
  // (stochastic) diverges from the RNE host reference.
  const float kFp4[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  auto pick_fp4 = [&]() {
    float m = kFp4[rng() % 8];
    return (rng() & 1) ? -m : m;
  };
  for (int t = 0; t < 32; ++t) {
    float a = pick_fp4() * scale;
    float b = pick_fp4() * scale;
    // ---- Float: byte0 = lo nibble enc(a/scale) | hi nibble enc(b/scale).
    k_d<Semantics::Native, SrcElem, VEC><<<1, 1>>>(old, a, b, 0xABCDu, scale, dO);
    HIP_CHECK(hipDeviceSynchronize());
    {
      unsigned g = from_dev(dO, 1)[0] & 0xFF;
      float sa = static_cast<float>(static_cast<SrcElem>(a));
      float sb = static_cast<float>(static_cast<SrcElem>(b));
      unsigned na = f32_to_narrow(sa / scale, kFp4E2M1) & 0xF;
      unsigned nb = f32_to_narrow(sb / scale, kFp4E2M1) & 0xF;
      EXPECT_EQ(g, na | (nb << 4)) << "Float a=" << a << " b=" << b;
    }
    // ---- FPSan: byte0 = canonical fp4 codes for a/scale and b/scale; seed opaque.
    fpsan_test::for_each_fpsan_semantics([&](auto sem) {
      constexpr Semantics S = decltype(sem)::value;
      using VF = Value<float, S, kCC>;
      k_d<S, SrcElem, VEC><<<1, 1>>>(old, a, b, 0x777u + t, scale, dO);
      HIP_CHECK(hipDeviceSynchronize());
      using VS = Value<SrcElem, S, kCC>;
      unsigned g = from_dev(dO, 1)[0] & 0xFF;
      unsigned na = fpsan_test::canonical_subbyte_narrow_code<4>(
          fpsan::cast<float>(VS{static_cast<SrcElem>(a)}) / VF{scale});
      unsigned nb = fpsan_test::canonical_subbyte_narrow_code<4>(
          fpsan::cast<float>(VS{static_cast<SrcElem>(b)}) / VF{scale});
      EXPECT_EQ(g, na | (nb << 4)) << "FPSan a=" << a << " b=" << b;
    });
  }
  const float special[] = {std::numeric_limits<float>::quiet_NaN(),
                           std::numeric_limits<float>::infinity(),
                           -std::numeric_limits<float>::infinity()};
  for (float a : special) {
    const float b = 2.0f;
    fpsan_test::for_each_fpsan_semantics([&](auto sem) {
      constexpr Semantics S = decltype(sem)::value;
      using VF = Value<float, S, kCC>;
      using VS = Value<SrcElem, S, kCC>;
      k_d<S, SrcElem, VEC><<<1, 1>>>(old, a, b, 0x777u, scale, dO);
      HIP_CHECK(hipDeviceSynchronize());
      unsigned g = from_dev(dO, 1)[0] & 0xFF;
      unsigned na = fpsan_test::canonical_subbyte_narrow_code<4>(
          fpsan::cast<float>(VS{static_cast<SrcElem>(a)}) / VF{scale});
      unsigned nb = fpsan_test::canonical_subbyte_narrow_code<4>(
          fpsan::cast<float>(VS{static_cast<SrcElem>(b)}) / VF{scale});
      EXPECT_EQ(g, na | (nb << 4)) << "FPSan non-finite source";
    });
  }
  (void)hipFree(dO);
}

TEST(CvtScalef32Sr, D_Fp4_F16) { run_d<_Float16, fpsan::v2h_native>(); }
TEST(CvtScalef32Sr, D_Fp4_BF16) { run_d<__bf16, fpsan::v2bf_native>(); }

#endif // (d)
