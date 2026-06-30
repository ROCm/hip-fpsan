// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/cvt_f16_fp8_gfx1250_test.cpp
//
// GPU tests for the gfx1250 f16<->fp8/bf8 conversion wrappers in
// fpsan/amdgcn_cvt.hpp (cvt_f16_fp8/bf8, cvt_pk_f16_fp8/bf8, cvt_pk_fp8/bf8_f16,
// cvt_sr_fp8/bf8_f16). Registered by CMake only for the gfx1250 tier.
//
// Authoritative-correctness policy: Native-mode references are computed HOST-SIDE
// from the OCP fp8 decode/encode (fpsan::fp8_e4m3 / fp8_e5m2, the same generic
// narrow<->f32 routines that fp8_test.cpp locks down), independent of the device
// builtin. fp8 (e4m3) and bf8 (e5m2) are exactly representable in f16, so the
// decode direction is lossless and the expected f16 equals the fp8 value. Any
// implementation that diverges from the host reference therefore shows up as a failure.
// FPSan-mode references use the deterministic payload-ring cast (computed
// independently of the device builtin).
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

namespace {
std::uint16_t f16_bits(_Float16 v) {
  std::uint16_t u = 0;
  std::memcpy(&u, &v, sizeof u);
  return u;
}
} // namespace

// ===========================================================================
// Decode: cvt_f16_fp8 / cvt_f16_bf8 over ALL 256 byte patterns, byte0.
// Float: lane b decodes fp8 byte b -> f16; compare to host (_Float16)fp8(b).
// ===========================================================================
template <class FP8> struct DecodeOps;
template <> struct DecodeOps<fpsan::fp8_e4m3> {
  template <int B, Semantics S> __device__ static Value<_Float16, S, kCC> f16(int p) {
    return fpsan::amdgcn_cvt_f16_fp8<B, S, kCC>(p);
  }
};
template <> struct DecodeOps<fpsan::fp8_e5m2> {
  template <int B, Semantics S> __device__ static Value<_Float16, S, kCC> f16(int p) {
    return fpsan::amdgcn_cvt_f16_bf8<B, S, kCC>(p);
  }
};

template <class FP8> __global__ void k_decode_all_bytes(const int *packed, _Float16 *out) {
  int l = threadIdx.x; // lane l carries fp8 byte l in byte0.
  out[l] = DecodeOps<FP8>::template f16<0, Semantics::Native>(packed[l]).to_float();
}

template <class FP8> void run_decode_all_bytes() {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  std::vector<int> packed(256);
  for (int b = 0; b < 256; ++b)
    packed[b] = b;
  int *dIn = to_dev(packed);
  _Float16 *dO;
  HIP_CHECK(hipMalloc(&dO, 256 * sizeof(_Float16)));
  k_decode_all_bytes<FP8><<<1, 256>>>(dIn, dO);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<_Float16> got(256);
  HIP_CHECK(hipMemcpy(got.data(), dO, 256 * sizeof(_Float16), hipMemcpyDeviceToHost));
  for (int b = 0; b < 256; ++b) {
    float fref = static_cast<float>(FP8(static_cast<std::uint8_t>(b)));
    _Float16 ref = static_cast<_Float16>(fref);
    if (std::isnan(fref))
      EXPECT_TRUE(std::isnan(static_cast<float>(got[b]))) << "byte 0x" << std::hex << b;
    else
      EXPECT_EQ(f16_bits(got[b]), f16_bits(ref)) << "byte 0x" << std::hex << b;
  }
  (void)hipFree(dIn);
  (void)hipFree(dO);
}

TEST(CvtF16Fp8, Fp8DecodeAllBytes) { run_decode_all_bytes<fpsan::fp8_e4m3>(); }
TEST(CvtF16Fp8, Bf8DecodeAllBytes) { run_decode_all_bytes<fpsan::fp8_e5m2>(); }

// Byte-select: place a known finite fp8 byte at byte ByteIdx, decode<ByteIdx>.
template <class FP8, int ByteIdx>
__global__ void k_decode_bytesel(const int *packed, _Float16 *out) {
  int l = threadIdx.x;
  out[l] = DecodeOps<FP8>::template f16<ByteIdx, Semantics::Native>(packed[l]).to_float();
}

template <class FP8, int ByteIdx> void run_decode_bytesel() {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  // Finite e4m3/e5m2 bytes only (0x00..0x7B span small finite values); place at
  // byte ByteIdx of each packed int.
  std::vector<int> packed(LANES);
  std::mt19937 rng = fpsan_test::make_rng();
  std::uniform_int_distribution<int> bdist(0, 0x70);
  std::vector<std::uint8_t> bytes(LANES);
  for (int i = 0; i < LANES; ++i) {
    bytes[i] = static_cast<std::uint8_t>(bdist(rng));
    packed[i] = static_cast<int>(static_cast<std::uint32_t>(bytes[i]) << (ByteIdx * 8));
  }
  int *dIn = to_dev(packed);
  _Float16 *dO;
  HIP_CHECK(hipMalloc(&dO, LANES * sizeof(_Float16)));
  k_decode_bytesel<FP8, ByteIdx><<<1, LANES>>>(dIn, dO);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<_Float16> got(LANES);
  HIP_CHECK(hipMemcpy(got.data(), dO, LANES * sizeof(_Float16), hipMemcpyDeviceToHost));
  for (int i = 0; i < LANES; ++i) {
    _Float16 ref = static_cast<_Float16>(static_cast<float>(FP8(bytes[i])));
    EXPECT_EQ(f16_bits(got[i]), f16_bits(ref)) << "lane " << i << " byteIdx " << ByteIdx;
  }
  (void)hipFree(dIn);
  (void)hipFree(dO);
}

TEST(CvtF16Fp8, Fp8DecodeByteSel) {
  run_decode_bytesel<fpsan::fp8_e4m3, 1>();
  run_decode_bytesel<fpsan::fp8_e4m3, 2>();
  run_decode_bytesel<fpsan::fp8_e4m3, 3>();
}
TEST(CvtF16Fp8, Bf8DecodeByteSel) {
  run_decode_bytesel<fpsan::fp8_e5m2, 1>();
  run_decode_bytesel<fpsan::fp8_e5m2, 3>();
}

// ===========================================================================
// Pack/unpack round trip: cvt_pk_fp8_f16 (encode) then cvt_pk_f16_fp8 (decode).
// For exact-in-fp8 inputs the round trip is the identity. Order-agnostic, so it
// authoritatively verifies pack/unpack are mutually consistent + lossless.
// FPSan mode: payload equals cast<f16>(cast<fp8>(x)) per element.
// ===========================================================================
template <class FP8> struct PkOps;
template <> struct PkOps<fpsan::fp8_e4m3> {
  template <Semantics S> __device__ static short pack(Value<fpsan::v2h_native, S, kCC> ab) {
    return fpsan::amdgcn_cvt_pk_fp8_f16<S, kCC>(ab);
  }
  template <Semantics S> __device__ static Value<fpsan::v2h_native, S, kCC> unpack(short p) {
    return fpsan::amdgcn_cvt_pk_f16_fp8<S, kCC>(p);
  }
};
template <> struct PkOps<fpsan::fp8_e5m2> {
  template <Semantics S> __device__ static short pack(Value<fpsan::v2h_native, S, kCC> ab) {
    return fpsan::amdgcn_cvt_pk_bf8_f16<S, kCC>(ab);
  }
  template <Semantics S> __device__ static Value<fpsan::v2h_native, S, kCC> unpack(short p) {
    return fpsan::amdgcn_cvt_pk_f16_bf8<S, kCC>(p);
  }
};

template <Semantics S, class FP8, class Out>
__global__ void k_pk_roundtrip(const _Float16 *in, Out *out) {
  using v2h = fpsan::v2h_native;
  int l = threadIdx.x;
  Value<v2h, S, kCC> ab{v2h{in[2 * l], in[2 * l + 1]}};
  short p = PkOps<FP8>::template pack<S>(ab);
  auto r = PkOps<FP8>::template unpack<S>(p);
  if constexpr (S == Semantics::Native) {
    out[2 * l] = r.get(0).to_float();
    out[2 * l + 1] = r.get(1).to_float();
  } else {
    out[2 * l] = r.get(0).fpsan_payload();
    out[2 * l + 1] = r.get(1).fpsan_payload();
  }
}

template <class FP8> void run_pk_roundtrip() {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  std::vector<_Float16> in(2 * LANES);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : in)
    x = static_cast<_Float16>(fpsan_test::pick_int_valued<float>(rng, -4, 4)); // exact in fp8
  _Float16 *dIn = to_dev(in);
  {
    _Float16 *dO;
    HIP_CHECK(hipMalloc(&dO, 2 * LANES * sizeof(_Float16)));
    k_pk_roundtrip<Semantics::Native, FP8, _Float16><<<1, LANES>>>(dIn, dO);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<_Float16> got(2 * LANES);
    HIP_CHECK(hipMemcpy(got.data(), dO, 2 * LANES * sizeof(_Float16), hipMemcpyDeviceToHost));
    for (int i = 0; i < 2 * LANES; ++i)
      EXPECT_EQ(f16_bits(got[i]), f16_bits(in[i])) << "Float round-trip at " << i;
    (void)hipFree(dO);
  }
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VH = Value<_Float16, S, kCC>;
    std::vector<std::uint16_t> ref(2 * LANES);
    for (int i = 0; i < 2 * LANES; ++i)
      ref[i] = static_cast<std::uint16_t>(
          fpsan::cast<_Float16>(fpsan::cast<FP8>(VH(in[i]))).fpsan_payload());
    std::uint16_t *dO;
    HIP_CHECK(hipMalloc(&dO, 2 * LANES * sizeof(std::uint16_t)));
    k_pk_roundtrip<S, FP8, std::uint16_t><<<1, LANES>>>(dIn, dO);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint16_t> got(2 * LANES);
    HIP_CHECK(hipMemcpy(got.data(), dO, 2 * LANES * sizeof(std::uint16_t), hipMemcpyDeviceToHost));
    for (int i = 0; i < 2 * LANES; ++i)
      EXPECT_EQ(got[i], ref[i]) << "FPSan payload at " << i;
    (void)hipFree(dO);
  });
  (void)hipFree(dIn);
}

TEST(CvtF16Fp8, Fp8PkRoundTrip) { run_pk_roundtrip<fpsan::fp8_e4m3>(); }
TEST(CvtF16Fp8, Bf8PkRoundTrip) { run_pk_roundtrip<fpsan::fp8_e5m2>(); }

// Pack absolute byte placement: element 0 -> low byte, element 1 -> high byte,
// against the host OCP encode. Confirms byte order, not just round-trip.
template <class FP8> __global__ void k_pack_bytes(const _Float16 *in, std::uint16_t *out) {
  using v2h = fpsan::v2h_native;
  int l = threadIdx.x;
  Value<v2h, Semantics::Native, kCC> ab{v2h{in[2 * l], in[2 * l + 1]}};
  out[l] = static_cast<std::uint16_t>(PkOps<FP8>::template pack<Semantics::Native>(ab));
}

template <class FP8> void run_pack_bytes() {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  std::vector<_Float16> in(2 * LANES);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : in)
    x = static_cast<_Float16>(fpsan_test::pick_int_valued<float>(rng, -4, 4));
  _Float16 *dIn = to_dev(in);
  std::uint16_t *dO;
  HIP_CHECK(hipMalloc(&dO, LANES * sizeof(std::uint16_t)));
  k_pack_bytes<FP8><<<1, LANES>>>(dIn, dO);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<std::uint16_t> got(LANES);
  HIP_CHECK(hipMemcpy(got.data(), dO, LANES * sizeof(std::uint16_t), hipMemcpyDeviceToHost));
  for (int l = 0; l < LANES; ++l) {
    std::uint8_t lo = FP8(static_cast<float>(in[2 * l])).bits;
    std::uint8_t hi = FP8(static_cast<float>(in[2 * l + 1])).bits;
    std::uint16_t expected = static_cast<std::uint16_t>(lo | (static_cast<std::uint16_t>(hi) << 8));
    EXPECT_EQ(got[l], expected) << "lane " << l;
  }
  (void)hipFree(dIn);
  (void)hipFree(dO);
}

TEST(CvtF16Fp8, Fp8PackByteOrder) { run_pack_bytes<fpsan::fp8_e4m3>(); }
TEST(CvtF16Fp8, Bf8PackByteOrder) { run_pack_bytes<fpsan::fp8_e5m2>(); }

// ===========================================================================
// SR encode: cvt_sr_fp8_f16 / cvt_sr_bf8_f16 into byte0, then decode back.
// For exact-in-fp8 inputs stochastic rounding is exact, so the round trip is
// the identity regardless of seed. FPSan: payload equals cast<f16>(cast<fp8>).
// ===========================================================================
template <class FP8> struct SrOps;
template <> struct SrOps<fpsan::fp8_e4m3> {
  template <Semantics S>
  __device__ static int sr(Value<_Float16, S, kCC> v, int old, unsigned seed) {
    return fpsan::amdgcn_cvt_sr_fp8_f16<0, S, kCC>(v, old, seed);
  }
  template <Semantics S> __device__ static Value<_Float16, S, kCC> dec(int p) {
    return fpsan::amdgcn_cvt_f16_fp8<0, S, kCC>(p);
  }
};
template <> struct SrOps<fpsan::fp8_e5m2> {
  template <Semantics S>
  __device__ static int sr(Value<_Float16, S, kCC> v, int old, unsigned seed) {
    return fpsan::amdgcn_cvt_sr_bf8_f16<0, S, kCC>(v, old, seed);
  }
  template <Semantics S> __device__ static Value<_Float16, S, kCC> dec(int p) {
    return fpsan::amdgcn_cvt_f16_bf8<0, S, kCC>(p);
  }
};

template <Semantics S, class FP8, class Out>
__global__ void k_sr_roundtrip(const _Float16 *in, Out *out) {
  int l = threadIdx.x;
  Value<_Float16, S, kCC> v{in[l]};
  int packed = SrOps<FP8>::template sr<S>(v, 0, 0x1234u + l);
  auto f = SrOps<FP8>::template dec<S>(packed);
  if constexpr (S == Semantics::Native)
    out[l] = f.to_float();
  else
    out[l] = f.fpsan_payload();
}

template <class FP8> void run_sr_roundtrip() {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  std::vector<_Float16> in(LANES);
  std::mt19937 rng = fpsan_test::make_rng();
  for (auto &x : in)
    x = static_cast<_Float16>(fpsan_test::pick_int_valued<float>(rng, -4, 4));
  _Float16 *dIn = to_dev(in);
  {
    _Float16 *dO;
    HIP_CHECK(hipMalloc(&dO, LANES * sizeof(_Float16)));
    k_sr_roundtrip<Semantics::Native, FP8, _Float16><<<1, LANES>>>(dIn, dO);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<_Float16> got(LANES);
    HIP_CHECK(hipMemcpy(got.data(), dO, LANES * sizeof(_Float16), hipMemcpyDeviceToHost));
    for (int i = 0; i < LANES; ++i)
      EXPECT_EQ(f16_bits(got[i]), f16_bits(in[i])) << "sr round-trip " << i;
    (void)hipFree(dO);
  }
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VH = Value<_Float16, S, kCC>;
    std::vector<std::uint16_t> ref(LANES);
    for (int i = 0; i < LANES; ++i)
      ref[i] = static_cast<std::uint16_t>(
          fpsan::cast<_Float16>(fpsan::cast<FP8>(VH(in[i]))).fpsan_payload());
    std::uint16_t *dO;
    HIP_CHECK(hipMalloc(&dO, LANES * sizeof(std::uint16_t)));
    k_sr_roundtrip<S, FP8, std::uint16_t><<<1, LANES>>>(dIn, dO);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint16_t> got(LANES);
    HIP_CHECK(hipMemcpy(got.data(), dO, LANES * sizeof(std::uint16_t), hipMemcpyDeviceToHost));
    for (int i = 0; i < LANES; ++i)
      EXPECT_EQ(got[i], ref[i]) << "sr FPSan " << i;
    (void)hipFree(dO);
  });
  (void)hipFree(dIn);
}

TEST(CvtF16Fp8, Fp8SrRoundTrip) { run_sr_roundtrip<fpsan::fp8_e4m3>(); }
TEST(CvtF16Fp8, Bf8SrRoundTrip) { run_sr_roundtrip<fpsan::fp8_e5m2>(); }
