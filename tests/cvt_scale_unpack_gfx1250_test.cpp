// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/cvt_scale_unpack_gfx1250_test.cpp
//
// GPU tests for the gfx1250 WIDE block-scale UNPACK wrappers in
// fpsan/amdgcn_cvt.hpp:
//   cvt_scale_pk8_{f32,f16,bf16}_{fp8,bf8}   (v2u32 8 bytes  -> v8)
//   cvt_scale_pk8_{f32,f16,bf16}_fp4         (u32   8 nibbles-> v8)
//   cvt_scale_pk16_{f32,f16,bf16}_{fp6,bf6}  (v3u32 16 codes -> v16)
//
// These are the decode siblings of the cvt_scalef32_pk8/pk16 PACK family. The
// block scale here is an E8M0 byte selected from a packed u32 `scale` operand
// by the _Constant ScaleSel (opsel); the byte is decoded as 2^(code-127).
//
// Authoritative-correctness policy (matches the gfx950/gfx1250 cvt suites):
//   Float : the wrapper forwards to the hardware builtin; expected element
//           values are computed HOST-SIDE from the OCP decoder
//           (detail::narrow_to_f32) times the host-decoded E8M0 scale -- NOT
//           from the builtin. Any implementation that diverges from the host
//           reference fails here.
//   FPSan : the wrapper equals an independent payload-domain reference
//           (FP8/BF8 payload cast, canonical subbyte FP4/FP6/BF6 widening,
//           NO scale -- block scale is Float-only).
// Each kernel runs a SINGLE lane (<<<1,1>>>), so ScaleSel=0 selects byte 0 of
// the scale operand for that lane, and every output element i is
// decode(code i) * scale, self-contained per lane.
//
// Inputs are chosen exactly representable in every narrow format AND in
// f16/bf16 (magnitudes on the fp4 grid {0,.5,1,1.5,2,3,4,6}), so the decode is
// exact and golden equality is well-defined.
#include "fpsan/amdgcn_cvt.hpp"
#include "fpsan/fpsan.hpp"

#include "fpsan_semantics.hpp"
#include "hip_test_utils.hpp"
#include "subbyte_oracle.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;
using fpsan::detail::f32_to_narrow;
using fpsan::detail::FpFormat;
using fpsan::detail::kBf6E3M2;
using fpsan::detail::kFp4E2M1;
using fpsan::detail::kFp6E2M3;
using fpsan::detail::kFp8E4M3;
using fpsan::detail::kFp8E5M2;
using fpsan::detail::narrow_to_f32;

static constexpr Conversions kCC = Conversions::Explicit;

namespace {
// E8M0 scale operand bytes: 0x7f == 2^0 == 1.0, 0x80 == 2^1 == 2.0.
constexpr std::uint32_t kE8M0_x1 = 0x0000007fu;
constexpr std::uint32_t kE8M0_x2 = 0x00000080u;

float e8m0_decode(std::uint32_t byte) {
  return std::ldexp(1.0f, static_cast<int>(byte & 0xFFu) - 127);
}

std::vector<float> exact8() { return {0.5f, -1.f, 1.5f, -2.f, 3.f, -4.f, 6.f, -0.5f}; }
std::vector<float> exact16() {
  return {0.5f, -1.f, 1.5f, -2.f,  3.f,  -4.f, 6.f, -0.5f,
          1.f,  2.f,  4.f,  -1.5f, -3.f, -6.f, 0.f, -2.f};
}

// Host pack of 8 narrow bytes -> v2u32 (byte i = code i, little-endian).
std::array<unsigned, 2> host_pack_bytes8(const std::vector<float> &in, const FpFormat &fmt) {
  std::array<unsigned, 2> w{0u, 0u};
  for (int i = 0; i < 8; ++i) {
    std::uint32_t code = f32_to_narrow(in[i], fmt) & 0xFFu;
    w[i / 4] |= code << (8 * (i % 4));
  }
  return w;
}
template <Semantics S, class FP8>
std::array<unsigned, 2> host_pack_payload_bytes8(const std::vector<float> &in,
                                                 const FpFormat &fmt) {
  using V = Value<FP8, S, kCC>;
  std::array<unsigned, 2> w{0u, 0u};
  for (int i = 0; i < 8; ++i) {
    const auto raw = static_cast<std::uint8_t>(f32_to_narrow(in[i], fmt) & 0xFFu);
    const auto p = static_cast<std::uint8_t>(V{FP8{raw}}.fpsan_payload());
    w[i / 4] |= static_cast<unsigned>(p) << (8 * (i % 4));
  }
  return w;
}
// Host pack of 8 fp4 nibbles -> u32 (nibble i = code i).
unsigned host_pack_nibbles8(const std::vector<float> &in) {
  unsigned out = 0;
  for (int i = 0; i < 8; ++i)
    out |= (f32_to_narrow(in[i], kFp4E2M1) & 0xFu) << (4 * i);
  return out;
}
unsigned host_pack_raw_nibbles8(const std::array<unsigned, 8> &codes) {
  unsigned out = 0;
  for (int i = 0; i < 8; ++i)
    out |= (codes[i] & 0xFu) << (4 * i);
  return out;
}
// Host pack of 16 6-bit codes -> v3u32 (contiguous little-endian).
std::array<unsigned, 3> host_pack_codes16(const std::vector<float> &in, const FpFormat &fmt) {
  std::array<unsigned, 3> w{0u, 0u, 0u};
  for (int i = 0; i < 16; ++i) {
    std::uint32_t code = f32_to_narrow(in[i], fmt) & 0x3Fu;
    for (int b = 0; b < 6; ++b)
      if ((code >> b) & 1u) {
        int p = i * 6 + b;
        w[p / 32] |= 1u << (p % 32);
      }
  }
  return w;
}
std::array<unsigned, 3> host_pack_raw_codes16(const std::array<unsigned, 16> &codes) {
  std::array<unsigned, 3> w{0u, 0u, 0u};
  for (int i = 0; i < 16; ++i) {
    std::uint32_t code = codes[i] & 0x3Fu;
    for (int b = 0; b < 6; ++b)
      if ((code >> b) & 1u) {
        int p = i * 6 + b;
        w[p / 32] |= 1u << (p % 32);
      }
  }
  return w;
}

template <class DstFT, int Width, Semantics S, std::size_t N>
void expect_canonical_device_payloads(const std::array<unsigned, N> &codes,
                                      const std::vector<unsigned> &got,
                                      const std::vector<unsigned> &got_sum, const char *label) {
  using Out = Value<DstFT, S, kCC>;
  for (std::size_t i = 0; i < N; ++i) {
    const auto want_payload =
        fpsan_test::canonical_subbyte_widen_payload<DstFT, Width, S, kCC>(codes[i]);
    const auto want = Out::from_fpsan_payload(want_payload);
    EXPECT_EQ(got[i], static_cast<unsigned>(want_payload))
        << label << " elem " << i << " code=0x" << std::hex << codes[i] << std::dec
        << " S=" << int(S);
    EXPECT_EQ(got_sum[i], static_cast<unsigned>((want + want).fpsan_payload()))
        << label << " elem " << i << " code=0x" << std::hex << codes[i] << std::dec
        << " follow-on add S=" << int(S);
  }
}
} // namespace

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_scale_pk8_f32_fp8)

// ============================ Native-mode golden =============================
// One real kernel + TEST per wrapper, scale operand pinned to E8M0 x1, decoding
// the packed stream and comparing each element to the host OCP decode.

#define PK8_BYTES_GOLDEN(CASE, WRAP, FMT, DST)                                                     \
  __global__ void CASE##_k(const unsigned *in, unsigned scale, float *out) {                       \
    fpsan::v2u32_native p;                                                                         \
    p[0] = in[0];                                                                                  \
    p[1] = in[1];                                                                                  \
    auto r = fpsan::WRAP<0, Semantics::Native, kCC>(p, scale);                                     \
    for (int i = 0; i < 8; ++i)                                                                    \
      out[i] = r.get(i).to_float();                                                                \
  }                                                                                                \
  TEST(CvtScalePk8Unpack, CASE) {                                                                  \
    if (!have_device())                                                                            \
      GTEST_SKIP() << "no HIP device";                                                             \
    auto in = exact8();                                                                            \
    auto w = host_pack_bytes8(in, FMT);                                                            \
    unsigned *dIn = to_dev(std::vector<unsigned>(w.begin(), w.end()));                             \
    float *dO;                                                                                     \
    HIP_CHECK(hipMalloc(&dO, 8 * sizeof(float)));                                                  \
    CASE##_k<<<1, 1>>>(dIn, kE8M0_x1, dO);                                                         \
    HIP_CHECK(hipDeviceSynchronize());                                                             \
    auto got = from_dev(dO, 8);                                                                    \
    for (int i = 0; i < 8; ++i)                                                                    \
      EXPECT_EQ(got[i], narrow_to_f32(f32_to_narrow(in[i], FMT) & 0xFFu, FMT)) << "elem " << i;    \
    (void)hipFree(dIn);                                                                            \
    (void)hipFree(dO);                                                                             \
  }

PK8_BYTES_GOLDEN(Fp8ToF32, amdgcn_cvt_scale_pk8_f32_fp8, kFp8E4M3, float)
PK8_BYTES_GOLDEN(Bf8ToF32, amdgcn_cvt_scale_pk8_f32_bf8, kFp8E5M2, float)
PK8_BYTES_GOLDEN(Fp8ToF16, amdgcn_cvt_scale_pk8_f16_fp8, kFp8E4M3, _Float16)
PK8_BYTES_GOLDEN(Bf8ToF16, amdgcn_cvt_scale_pk8_f16_bf8, kFp8E5M2, _Float16)
PK8_BYTES_GOLDEN(Fp8ToBf16, amdgcn_cvt_scale_pk8_bf16_fp8, kFp8E4M3, __bf16)
PK8_BYTES_GOLDEN(Bf8ToBf16, amdgcn_cvt_scale_pk8_bf16_bf8, kFp8E5M2, __bf16)
#undef PK8_BYTES_GOLDEN

#define PK8_NIB_GOLDEN(CASE, WRAP, DST)                                                            \
  __global__ void CASE##_k(unsigned packed, unsigned scale, float *out) {                          \
    auto r = fpsan::WRAP<0, Semantics::Native, kCC>(packed, scale);                                \
    for (int i = 0; i < 8; ++i)                                                                    \
      out[i] = r.get(i).to_float();                                                                \
  }                                                                                                \
  TEST(CvtScalePk8Unpack, CASE) {                                                                  \
    if (!have_device())                                                                            \
      GTEST_SKIP() << "no HIP device";                                                             \
    auto in = exact8();                                                                            \
    unsigned packed = host_pack_nibbles8(in);                                                      \
    float *dO;                                                                                     \
    HIP_CHECK(hipMalloc(&dO, 8 * sizeof(float)));                                                  \
    CASE##_k<<<1, 1>>>(packed, kE8M0_x1, dO);                                                      \
    HIP_CHECK(hipDeviceSynchronize());                                                             \
    auto got = from_dev(dO, 8);                                                                    \
    for (int i = 0; i < 8; ++i)                                                                    \
      EXPECT_EQ(got[i], narrow_to_f32(f32_to_narrow(in[i], kFp4E2M1) & 0xFu, kFp4E2M1))            \
          << "elem " << i;                                                                         \
    (void)hipFree(dO);                                                                             \
  }

PK8_NIB_GOLDEN(Fp4ToF32, amdgcn_cvt_scale_pk8_f32_fp4, float)
PK8_NIB_GOLDEN(Fp4ToF16, amdgcn_cvt_scale_pk8_f16_fp4, _Float16)
PK8_NIB_GOLDEN(Fp4ToBf16, amdgcn_cvt_scale_pk8_bf16_fp4, __bf16)
#undef PK8_NIB_GOLDEN

#define PK16_CODES_GOLDEN(CASE, WRAP, FMT, DST)                                                    \
  __global__ void CASE##_k(const unsigned *in, unsigned scale, float *out) {                       \
    fpsan::v3u32_native p;                                                                         \
    p[0] = in[0];                                                                                  \
    p[1] = in[1];                                                                                  \
    p[2] = in[2];                                                                                  \
    auto r = fpsan::WRAP<0, Semantics::Native, kCC>(p, scale);                                     \
    for (int i = 0; i < 16; ++i)                                                                   \
      out[i] = r.get(i).to_float();                                                                \
  }                                                                                                \
  TEST(CvtScalePk16Unpack, CASE) {                                                                 \
    if (!have_device())                                                                            \
      GTEST_SKIP() << "no HIP device";                                                             \
    auto in = exact16();                                                                           \
    auto w = host_pack_codes16(in, FMT);                                                           \
    unsigned *dIn = to_dev(std::vector<unsigned>(w.begin(), w.end()));                             \
    float *dO;                                                                                     \
    HIP_CHECK(hipMalloc(&dO, 16 * sizeof(float)));                                                 \
    CASE##_k<<<1, 1>>>(dIn, kE8M0_x1, dO);                                                         \
    HIP_CHECK(hipDeviceSynchronize());                                                             \
    auto got = from_dev(dO, 16);                                                                   \
    for (int i = 0; i < 16; ++i)                                                                   \
      EXPECT_EQ(got[i], narrow_to_f32(f32_to_narrow(in[i], FMT) & 0x3Fu, FMT)) << "elem " << i;    \
    (void)hipFree(dIn);                                                                            \
    (void)hipFree(dO);                                                                             \
  }

PK16_CODES_GOLDEN(Fp6ToF32, amdgcn_cvt_scale_pk16_f32_fp6, kFp6E2M3, float)
PK16_CODES_GOLDEN(Bf6ToF32, amdgcn_cvt_scale_pk16_f32_bf6, kBf6E3M2, float)
PK16_CODES_GOLDEN(Fp6ToF16, amdgcn_cvt_scale_pk16_f16_fp6, kFp6E2M3, _Float16)
PK16_CODES_GOLDEN(Bf6ToF16, amdgcn_cvt_scale_pk16_f16_bf6, kBf6E3M2, _Float16)
PK16_CODES_GOLDEN(Fp6ToBf16, amdgcn_cvt_scale_pk16_bf16_fp6, kFp6E2M3, __bf16)
PK16_CODES_GOLDEN(Bf6ToBf16, amdgcn_cvt_scale_pk16_bf16_bf6, kBf6E3M2, __bf16)
#undef PK16_CODES_GOLDEN

// ============================ Scale direction ===============================
// UNPACK multiplies by the (host-decoded) block scale: E8M0 0x80 == 2.0, so
// each f32 element is decode(code) * 2. The doubled magnitudes stay finite and
// exactly representable, so golden equality holds.
__global__ void k_pk8_scale_x2(const unsigned *in, unsigned scale, float *out) {
  fpsan::v2u32_native p;
  p[0] = in[0];
  p[1] = in[1];
  auto r = fpsan::amdgcn_cvt_scale_pk8_f32_fp8<0, Semantics::Native, kCC>(p, scale);
  for (int i = 0; i < 8; ++i)
    out[i] = r.get(i).to_float();
}

TEST(CvtScalePk8Unpack, Fp8ScaleMultiplies) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  auto in = exact8();
  auto w = host_pack_bytes8(in, kFp8E4M3);
  unsigned *dIn = to_dev(std::vector<unsigned>(w.begin(), w.end()));
  float *dO;
  HIP_CHECK(hipMalloc(&dO, 8 * sizeof(float)));
  k_pk8_scale_x2<<<1, 1>>>(dIn, kE8M0_x2, dO);
  HIP_CHECK(hipDeviceSynchronize());
  auto got = from_dev(dO, 8);
  for (int i = 0; i < 8; ++i)
    EXPECT_EQ(got[i], narrow_to_f32(f32_to_narrow(in[i], kFp8E4M3) & 0xFFu, kFp8E4M3) *
                          e8m0_decode(kE8M0_x2))
        << "elem " << i;
  (void)hipFree(dIn);
  (void)hipFree(dO);
}

// ============================ FPSan mode ====================================
// Payload-domain casts, NO scale (block scale is Float-only). The wrapper equals
// an independent payload reference built from the public Value algebra. FPSan
// tests pass E8M0 x2 to prove that the block scale is ignored here.

template <Semantics S>
__global__ void k_fpsan_pk8_fp8(const unsigned *in, unsigned scale, unsigned *out) {
  fpsan::v2u32_native p;
  p[0] = in[0];
  p[1] = in[1];
  auto r = fpsan::amdgcn_cvt_scale_pk8_f32_fp8<0, S, kCC>(p, scale);
  for (int i = 0; i < 8; ++i)
    out[i] = r.get(i).fpsan_payload();
}

template <Semantics S> void run_fpsan_pk8_fp8_payload() {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  auto in = exact8();
  auto w = host_pack_payload_bytes8<S, fpsan::fp8_e4m3>(in, kFp8E4M3);
  unsigned *dIn = to_dev(std::vector<unsigned>(w.begin(), w.end()));
  unsigned *dO;
  HIP_CHECK(hipMalloc(&dO, 8 * sizeof(unsigned)));
  k_fpsan_pk8_fp8<S><<<1, 1>>>(dIn, kE8M0_x2, dO);
  HIP_CHECK(hipDeviceSynchronize());
  auto got = from_dev(dO, 8);
  for (int i = 0; i < 8; ++i) {
    using Src = Value<fpsan::fp8_e4m3, S, kCC>;
    const auto raw = static_cast<std::uint8_t>(f32_to_narrow(in[i], kFp8E4M3) & 0xFFu);
    const auto src = Src{fpsan::fp8_e4m3{raw}};
    const auto want = static_cast<unsigned>(fpsan::cast<float>(src).fpsan_payload());
    EXPECT_EQ(got[i], want) << "elem " << i;
  }
  (void)hipFree(dIn);
  (void)hipFree(dO);
}

TEST(CvtScalePk8Unpack, FpsanFp8Payload) { FPSAN_RUN_FPSAN_SEMANTICS(run_fpsan_pk8_fp8_payload); }

template <Semantics S>
__global__ void k_fpsan_pk8_fp4_f32_canonical(unsigned packed, unsigned scale, unsigned *out) {
  auto r = fpsan::amdgcn_cvt_scale_pk8_f32_fp4<0, S, kCC>(packed, scale);
  for (int i = 0; i < 8; ++i) {
    const auto lane = r.get(i);
    out[i] = lane.fpsan_payload();
    out[8 + i] = (lane + lane).fpsan_payload();
  }
}

template <Semantics S>
__global__ void k_fpsan_pk8_fp4_f16_canonical(unsigned packed, unsigned scale, unsigned *out) {
  auto r = fpsan::amdgcn_cvt_scale_pk8_f16_fp4<0, S, kCC>(packed, scale);
  for (int i = 0; i < 8; ++i) {
    const auto lane = r.get(i);
    out[i] = lane.fpsan_payload();
    out[8 + i] = (lane + lane).fpsan_payload();
  }
}

template <Semantics S>
__global__ void k_fpsan_pk8_fp4_bf16_canonical(unsigned packed, unsigned scale, unsigned *out) {
  auto r = fpsan::amdgcn_cvt_scale_pk8_bf16_fp4<0, S, kCC>(packed, scale);
  for (int i = 0; i < 8; ++i) {
    const auto lane = r.get(i);
    out[i] = lane.fpsan_payload();
    out[8 + i] = (lane + lane).fpsan_payload();
  }
}

template <Semantics S> void run_fpsan_pk8_fp4_f32_canonical() {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  constexpr std::array<unsigned, 8> codes{0x0u, 0x1u, 0x7u, 0x8u, 0x9u, 0xAu, 0xEu, 0xFu};
  unsigned *dO;
  HIP_CHECK(hipMalloc(&dO, 16 * sizeof(unsigned)));
  k_fpsan_pk8_fp4_f32_canonical<S><<<1, 1>>>(host_pack_raw_nibbles8(codes), kE8M0_x2, dO);
  HIP_CHECK(hipDeviceSynchronize());
  auto got_all = from_dev(dO, 16);
  std::vector<unsigned> got(got_all.begin(), got_all.begin() + 8);
  std::vector<unsigned> got_sum(got_all.begin() + 8, got_all.end());
  expect_canonical_device_payloads<float, 4, S>(codes, got, got_sum, "gfx1250 pk8 fp4->f32");
  (void)hipFree(dO);
}

template <Semantics S> void run_fpsan_pk8_fp4_f16_canonical() {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  constexpr std::array<unsigned, 8> codes{0x0u, 0x1u, 0x7u, 0x8u, 0x9u, 0xAu, 0xEu, 0xFu};
  unsigned *dO;
  HIP_CHECK(hipMalloc(&dO, 16 * sizeof(unsigned)));
  k_fpsan_pk8_fp4_f16_canonical<S><<<1, 1>>>(host_pack_raw_nibbles8(codes), kE8M0_x2, dO);
  HIP_CHECK(hipDeviceSynchronize());
  auto got_all = from_dev(dO, 16);
  std::vector<unsigned> got(got_all.begin(), got_all.begin() + 8);
  std::vector<unsigned> got_sum(got_all.begin() + 8, got_all.end());
  expect_canonical_device_payloads<_Float16, 4, S>(codes, got, got_sum, "gfx1250 pk8 fp4->f16");
  (void)hipFree(dO);
}

template <Semantics S> void run_fpsan_pk8_fp4_bf16_canonical() {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  constexpr std::array<unsigned, 8> codes{0x0u, 0x1u, 0x7u, 0x8u, 0x9u, 0xAu, 0xEu, 0xFu};
  unsigned *dO;
  HIP_CHECK(hipMalloc(&dO, 16 * sizeof(unsigned)));
  k_fpsan_pk8_fp4_bf16_canonical<S><<<1, 1>>>(host_pack_raw_nibbles8(codes), kE8M0_x2, dO);
  HIP_CHECK(hipDeviceSynchronize());
  auto got_all = from_dev(dO, 16);
  std::vector<unsigned> got(got_all.begin(), got_all.begin() + 8);
  std::vector<unsigned> got_sum(got_all.begin() + 8, got_all.end());
  expect_canonical_device_payloads<__bf16, 4, S>(codes, got, got_sum, "gfx1250 pk8 fp4->bf16");
  (void)hipFree(dO);
}

TEST(CvtScalePk8Unpack, FpsanFp4ToF16CanonicalPayload) {
  FPSAN_RUN_FPSAN_SEMANTICS(run_fpsan_pk8_fp4_f16_canonical);
}

TEST(CvtScalePk8Unpack, FpsanFp4ToF32CanonicalPayload) {
  FPSAN_RUN_FPSAN_SEMANTICS(run_fpsan_pk8_fp4_f32_canonical);
}

TEST(CvtScalePk8Unpack, FpsanFp4ToBf16CanonicalPayload) {
  FPSAN_RUN_FPSAN_SEMANTICS(run_fpsan_pk8_fp4_bf16_canonical);
}

#define PK16_SUB6_CANONICAL_TEST(CASE, WRAP, DST, LABEL)                                           \
  template <Semantics S>                                                                           \
  __global__ void CASE##_k(const unsigned *in, unsigned scale, unsigned *out) {                    \
    fpsan::v3u32_native p;                                                                         \
    p[0] = in[0];                                                                                  \
    p[1] = in[1];                                                                                  \
    p[2] = in[2];                                                                                  \
    auto r = fpsan::WRAP<0, S, kCC>(p, scale);                                                     \
    for (int i = 0; i < 16; ++i) {                                                                 \
      const auto lane = r.get(i);                                                                  \
      out[i] = lane.fpsan_payload();                                                               \
      out[16 + i] = (lane + lane).fpsan_payload();                                                 \
    }                                                                                              \
  }                                                                                                \
  template <Semantics S> void CASE##_run() {                                                       \
    if (!have_device())                                                                            \
      GTEST_SKIP() << "no HIP device";                                                             \
    constexpr std::array<unsigned, 16> codes{0x00u, 0x01u, 0x1Fu, 0x20u, 0x21u, 0x2Au,             \
                                             0x30u, 0x3Fu, 0x02u, 0x07u, 0x10u, 0x18u,             \
                                             0x22u, 0x2Fu, 0x3Eu, 0x15u};                          \
    auto w = host_pack_raw_codes16(codes);                                                         \
    unsigned *dIn = to_dev(std::vector<unsigned>(w.begin(), w.end()));                             \
    unsigned *dO;                                                                                  \
    HIP_CHECK(hipMalloc(&dO, 32 * sizeof(unsigned)));                                              \
    CASE##_k<S><<<1, 1>>>(dIn, kE8M0_x2, dO);                                                      \
    HIP_CHECK(hipDeviceSynchronize());                                                             \
    auto got_all = from_dev(dO, 32);                                                               \
    std::vector<unsigned> got(got_all.begin(), got_all.begin() + 16);                              \
    std::vector<unsigned> got_sum(got_all.begin() + 16, got_all.end());                            \
    expect_canonical_device_payloads<DST, 6, S>(codes, got, got_sum, LABEL);                       \
    (void)hipFree(dIn);                                                                            \
    (void)hipFree(dO);                                                                             \
  }                                                                                                \
  TEST(CvtScalePk16Unpack, CASE) { FPSAN_RUN_FPSAN_SEMANTICS(CASE##_run); }

PK16_SUB6_CANONICAL_TEST(FpsanFp6ToF32CanonicalPayload, amdgcn_cvt_scale_pk16_f32_fp6, float,
                         "gfx1250 pk16 fp6->f32")
PK16_SUB6_CANONICAL_TEST(FpsanFp6ToF16CanonicalPayload, amdgcn_cvt_scale_pk16_f16_fp6, _Float16,
                         "gfx1250 pk16 fp6->f16")
PK16_SUB6_CANONICAL_TEST(FpsanFp6ToBf16CanonicalPayload, amdgcn_cvt_scale_pk16_bf16_fp6, __bf16,
                         "gfx1250 pk16 fp6->bf16")
PK16_SUB6_CANONICAL_TEST(FpsanBf6ToF32CanonicalPayload, amdgcn_cvt_scale_pk16_f32_bf6, float,
                         "gfx1250 pk16 bf6->f32")
PK16_SUB6_CANONICAL_TEST(FpsanBf6ToF16CanonicalPayload, amdgcn_cvt_scale_pk16_f16_bf6, _Float16,
                         "gfx1250 pk16 bf6->f16")
PK16_SUB6_CANONICAL_TEST(FpsanBf6ToBf16CanonicalPayload, amdgcn_cvt_scale_pk16_bf16_bf6, __bf16,
                         "gfx1250 pk16 bf6->bf16")
#undef PK16_SUB6_CANONICAL_TEST

#endif // has builtin
