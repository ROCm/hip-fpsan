// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/cvt_scalef32_pk_gfx1250_test.cpp
//
// GPU tests for the gfx1250 WIDE block-scale PACK wrappers in
// fpsan/amdgcn_cvt.hpp:
//   cvt_scalef32_pk8_{fp8,bf8}_{f32,f16,bf16}   (8 lanes -> v2u32, 8 bytes)
//   cvt_scalef32_pk8_fp4_{f32,f16,bf16}         (8 lanes -> u32, 8 nibbles)
//   cvt_scalef32_pk16_{fp6,bf6}_{f32,f16,bf16}  (16 lanes -> v3u32, 16 codes)
//   ... and the cvt_scalef32_sr_pk8/pk16 stochastic-rounding siblings.
//
// Authoritative-correctness policy (matches the gfx950 scaled-cvt suite):
//   Float : the wrapper forwards to the hardware builtin; the expected packed
//           bits are computed HOST-SIDE from the OCP encoders
//           (detail::f32_to_narrow for fp8/bf8/fp4/fp6/bf6) and the contiguous
//           little-endian packed-stream layout -- NOT from the
//           builtin. Any implementation that diverges from them fails here.
//   FPSan : the wrapper equals an independent payload-domain reference built
//           from the public Value algebra and canonical subbyte narrowing
//           (cast + divide-by-scale).
// Native inputs are chosen exactly representable in every narrow format AND in
// f16/bf16 (magnitudes in the fp4 grid {0,.5,1,1.5,2,3,4,6}), so at scale==1
// the pack is an exact, round-trip-stable encode. FPSan payload tests also cover
// source non-finites, including the algebraic finite-only narrow mappings.
#include "fpsan/amdgcn_cvt.hpp"
#include "fpsan/fpsan.hpp"

#include "fpsan_semantics.hpp"
#include "hip_test_utils.hpp"
#include "subbyte_oracle.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <limits>
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

static constexpr Conversions kCC = Conversions::Explicit;

namespace {
// Eight magnitudes on the fp4 grid (exact in fp4/fp6/fp8/f16/bf16).
std::vector<float> exact8() { return {0.5f, -1.f, 1.5f, -2.f, 3.f, -4.f, 6.f, -0.5f}; }
std::vector<float> exact16() {
  return {0.5f, -1.f, 1.5f, -2.f,  3.f,  -4.f, 6.f, -0.5f,
          1.f,  2.f,  4.f,  -1.5f, -3.f, -6.f, 0.f, -2.f};
}
std::vector<float> fpsan_nonfinite8() {
  auto in = exact8();
  in[1] = std::numeric_limits<float>::quiet_NaN();
  in[3] = std::numeric_limits<float>::infinity();
  in[5] = -std::numeric_limits<float>::infinity();
  in[6] = std::numeric_limits<float>::quiet_NaN();
  return in;
}
std::vector<float> fpsan_nonfinite16() {
  auto in = exact16();
  in[3] = std::numeric_limits<float>::quiet_NaN();
  in[5] = std::numeric_limits<float>::infinity();
  in[11] = std::numeric_limits<float>::quiet_NaN();
  in[13] = -std::numeric_limits<float>::infinity();
  return in;
}

// Host extract of contiguous little-endian 6-bit field i from a 96-bit stream.
std::uint32_t host_extract6(const std::vector<unsigned> &w, int i) {
  const int p = i * 6, wi = p >> 5, off = p & 31;
  std::uint32_t v = w[wi] >> off;
  if (off > 26)
    v |= w[static_cast<std::size_t>(wi) + 1] << (32 - off);
  return v & 0x3Fu;
}

void check_bytes8(const std::vector<unsigned> &got, const std::vector<float> &in,
                  const FpFormat &fmt, float scale = 1.0f) {
  ASSERT_EQ(got.size(), 2u);
  for (int i = 0; i < 8; ++i) {
    std::uint32_t want = f32_to_narrow(in[i] / scale, fmt) & 0xFFu;
    std::uint32_t have = (got[i / 4] >> (8 * (i % 4))) & 0xFFu;
    EXPECT_EQ(have, want) << "byte " << i;
  }
}
void check_nibbles8(const std::vector<unsigned> &got, const std::vector<float> &in,
                    const FpFormat &fmt, float scale = 1.0f) {
  ASSERT_EQ(got.size(), 1u);
  for (int i = 0; i < 8; ++i) {
    std::uint32_t want = f32_to_narrow(in[i] / scale, fmt) & 0xFu;
    std::uint32_t have = (got[0] >> (4 * i)) & 0xFu;
    EXPECT_EQ(have, want) << "nibble " << i;
  }
}
void check_codes16(const std::vector<unsigned> &got, const std::vector<float> &in,
                   const FpFormat &fmt, float scale = 1.0f) {
  ASSERT_EQ(got.size(), 3u);
  for (int i = 0; i < 16; ++i) {
    std::uint32_t want = f32_to_narrow(in[i] / scale, fmt) & 0x3Fu;
    std::uint32_t have = host_extract6(got, i);
    EXPECT_EQ(have, want) << "code " << i;
  }
}

template <int Width, class Lane, Semantics S>
std::uint32_t canonical_narrow_lane(float x, float scale) {
  using VF = Value<float, S, kCC>;
  using VS = Value<Lane, S, kCC>;
  return fpsan_test::canonical_subbyte_narrow_code<Width>(
      fpsan::cast<float>(VS{static_cast<Lane>(x)}) / VF{scale});
}
} // namespace

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_scalef32_pk8_fp8_f32)

// ============================ Native-mode golden =============================
// One real kernel + TEST per wrapper (scale == 1), checking the packed bits
// against the host OCP encode and the contiguous layout.

#define PK8_BYTES_GOLDEN(CASE, WRAP, FMT, LANE)                                                    \
  __global__ void CASE##_k(const float *in, unsigned *out) {                                       \
    using V = LANE __attribute__((ext_vector_type(8)));                                            \
    V v;                                                                                           \
    for (int i = 0; i < 8; ++i)                                                                    \
      v[i] = static_cast<LANE>(in[i]);                                                             \
    Value<V, Semantics::Native, kCC> vv{v};                                                        \
    fpsan::v2u32_native r =                                                                        \
        fpsan::WRAP<Semantics::Native, kCC>(vv, Value<float, Semantics::Native, kCC>{1.0f});       \
    out[0] = r[0];                                                                                 \
    out[1] = r[1];                                                                                 \
  }                                                                                                \
  TEST(CvtScalef32Pk8, CASE) {                                                                     \
    if (!have_device())                                                                            \
      GTEST_SKIP() << "no HIP device";                                                             \
    auto in = exact8();                                                                            \
    float *dIn = to_dev(in);                                                                       \
    unsigned *dO;                                                                                  \
    HIP_CHECK(hipMalloc(&dO, 2 * sizeof(unsigned)));                                               \
    CASE##_k<<<1, 1>>>(dIn, dO);                                                                   \
    HIP_CHECK(hipDeviceSynchronize());                                                             \
    check_bytes8(from_dev(dO, 2), in, FMT);                                                        \
    (void)hipFree(dIn);                                                                            \
    (void)hipFree(dO);                                                                             \
  }

PK8_BYTES_GOLDEN(Fp8FromF32, amdgcn_cvt_scalef32_pk8_fp8_f32, kFp8E4M3, float)
PK8_BYTES_GOLDEN(Bf8FromF32, amdgcn_cvt_scalef32_pk8_bf8_f32, kFp8E5M2, float)
PK8_BYTES_GOLDEN(Fp8FromF16, amdgcn_cvt_scalef32_pk8_fp8_f16, kFp8E4M3, _Float16)
PK8_BYTES_GOLDEN(Bf8FromF16, amdgcn_cvt_scalef32_pk8_bf8_f16, kFp8E5M2, _Float16)
PK8_BYTES_GOLDEN(Fp8FromBf16, amdgcn_cvt_scalef32_pk8_fp8_bf16, kFp8E4M3, __bf16)
PK8_BYTES_GOLDEN(Bf8FromBf16, amdgcn_cvt_scalef32_pk8_bf8_bf16, kFp8E5M2, __bf16)
#undef PK8_BYTES_GOLDEN

#define PK8_NIB_GOLDEN(CASE, WRAP, LANE)                                                           \
  __global__ void CASE##_k(const float *in, unsigned *out) {                                       \
    using V = LANE __attribute__((ext_vector_type(8)));                                            \
    V v;                                                                                           \
    for (int i = 0; i < 8; ++i)                                                                    \
      v[i] = static_cast<LANE>(in[i]);                                                             \
    Value<V, Semantics::Native, kCC> vv{v};                                                        \
    out[0] = fpsan::WRAP<Semantics::Native, kCC>(vv, Value<float, Semantics::Native, kCC>{1.0f});  \
  }                                                                                                \
  TEST(CvtScalef32Pk8, CASE) {                                                                     \
    if (!have_device())                                                                            \
      GTEST_SKIP() << "no HIP device";                                                             \
    auto in = exact8();                                                                            \
    float *dIn = to_dev(in);                                                                       \
    unsigned *dO;                                                                                  \
    HIP_CHECK(hipMalloc(&dO, sizeof(unsigned)));                                                   \
    CASE##_k<<<1, 1>>>(dIn, dO);                                                                   \
    HIP_CHECK(hipDeviceSynchronize());                                                             \
    check_nibbles8(from_dev(dO, 1), in, kFp4E2M1);                                                 \
    (void)hipFree(dIn);                                                                            \
    (void)hipFree(dO);                                                                             \
  }

PK8_NIB_GOLDEN(Fp4FromF32, amdgcn_cvt_scalef32_pk8_fp4_f32, float)
PK8_NIB_GOLDEN(Fp4FromF16, amdgcn_cvt_scalef32_pk8_fp4_f16, _Float16)
PK8_NIB_GOLDEN(Fp4FromBf16, amdgcn_cvt_scalef32_pk8_fp4_bf16, __bf16)
#undef PK8_NIB_GOLDEN

#define PK16_CODES_GOLDEN(CASE, WRAP, FMT, LANE)                                                   \
  __global__ void CASE##_k(const float *in, unsigned *out) {                                       \
    using V = LANE __attribute__((ext_vector_type(16)));                                           \
    V v;                                                                                           \
    for (int i = 0; i < 16; ++i)                                                                   \
      v[i] = static_cast<LANE>(in[i]);                                                             \
    Value<V, Semantics::Native, kCC> vv{v};                                                        \
    fpsan::v3u32_native r =                                                                        \
        fpsan::WRAP<Semantics::Native, kCC>(vv, Value<float, Semantics::Native, kCC>{1.0f});       \
    out[0] = r[0];                                                                                 \
    out[1] = r[1];                                                                                 \
    out[2] = r[2];                                                                                 \
  }                                                                                                \
  TEST(CvtScalef32Pk16, CASE) {                                                                    \
    if (!have_device())                                                                            \
      GTEST_SKIP() << "no HIP device";                                                             \
    auto in = exact16();                                                                           \
    float *dIn = to_dev(in);                                                                       \
    unsigned *dO;                                                                                  \
    HIP_CHECK(hipMalloc(&dO, 3 * sizeof(unsigned)));                                               \
    CASE##_k<<<1, 1>>>(dIn, dO);                                                                   \
    HIP_CHECK(hipDeviceSynchronize());                                                             \
    check_codes16(from_dev(dO, 3), in, FMT);                                                       \
    (void)hipFree(dIn);                                                                            \
    (void)hipFree(dO);                                                                             \
  }

PK16_CODES_GOLDEN(Fp6FromF32, amdgcn_cvt_scalef32_pk16_fp6_f32, kFp6E2M3, float)
PK16_CODES_GOLDEN(Bf6FromF32, amdgcn_cvt_scalef32_pk16_bf6_f32, kBf6E3M2, float)
PK16_CODES_GOLDEN(Fp6FromF16, amdgcn_cvt_scalef32_pk16_fp6_f16, kFp6E2M3, _Float16)
PK16_CODES_GOLDEN(Bf6FromF16, amdgcn_cvt_scalef32_pk16_bf6_f16, kBf6E3M2, _Float16)
PK16_CODES_GOLDEN(Fp6FromBf16, amdgcn_cvt_scalef32_pk16_fp6_bf16, kFp6E2M3, __bf16)
PK16_CODES_GOLDEN(Bf6FromBf16, amdgcn_cvt_scalef32_pk16_bf6_bf16, kBf6E3M2, __bf16)
#undef PK16_CODES_GOLDEN

// ---- SR variants: at exactly-representable inputs stochastic rounding is exact,
// so the packed bits equal the same host OCP encode regardless of seed.
#define SR_PK8_BYTES_GOLDEN(CASE, WRAP, FMT, LANE)                                                 \
  __global__ void CASE##_srb_k(const float *in, unsigned *out) {                                   \
    using V = LANE __attribute__((ext_vector_type(8)));                                            \
    V v;                                                                                           \
    for (int i = 0; i < 8; ++i)                                                                    \
      v[i] = static_cast<LANE>(in[i]);                                                             \
    Value<V, Semantics::Native, kCC> vv{v};                                                        \
    fpsan::v2u32_native r = fpsan::WRAP<Semantics::Native, kCC>(                                   \
        vv, 0x1234abcdu, Value<float, Semantics::Native, kCC>{1.0f});                              \
    out[0] = r[0];                                                                                 \
    out[1] = r[1];                                                                                 \
  }                                                                                                \
  TEST(CvtScalef32SrPk8, CASE) {                                                                   \
    if (!have_device())                                                                            \
      GTEST_SKIP() << "no HIP device";                                                             \
    auto in = exact8();                                                                            \
    float *dIn = to_dev(in);                                                                       \
    unsigned *dO;                                                                                  \
    HIP_CHECK(hipMalloc(&dO, 2 * sizeof(unsigned)));                                               \
    CASE##_srb_k<<<1, 1>>>(dIn, dO);                                                               \
    HIP_CHECK(hipDeviceSynchronize());                                                             \
    check_bytes8(from_dev(dO, 2), in, FMT);                                                        \
    (void)hipFree(dIn);                                                                            \
    (void)hipFree(dO);                                                                             \
  }

SR_PK8_BYTES_GOLDEN(Fp8FromF32, amdgcn_cvt_scalef32_sr_pk8_fp8_f32, kFp8E4M3, float)
SR_PK8_BYTES_GOLDEN(Bf8FromF32, amdgcn_cvt_scalef32_sr_pk8_bf8_f32, kFp8E5M2, float)
SR_PK8_BYTES_GOLDEN(Fp8FromF16, amdgcn_cvt_scalef32_sr_pk8_fp8_f16, kFp8E4M3, _Float16)
SR_PK8_BYTES_GOLDEN(Bf8FromF16, amdgcn_cvt_scalef32_sr_pk8_bf8_f16, kFp8E5M2, _Float16)
SR_PK8_BYTES_GOLDEN(Fp8FromBf16, amdgcn_cvt_scalef32_sr_pk8_fp8_bf16, kFp8E4M3, __bf16)
SR_PK8_BYTES_GOLDEN(Bf8FromBf16, amdgcn_cvt_scalef32_sr_pk8_bf8_bf16, kFp8E5M2, __bf16)
#undef SR_PK8_BYTES_GOLDEN

#define SR_PK8_NIB_GOLDEN(CASE, WRAP, LANE)                                                        \
  __global__ void CASE##_srn_k(const float *in, unsigned *out) {                                   \
    using V = LANE __attribute__((ext_vector_type(8)));                                            \
    V v;                                                                                           \
    for (int i = 0; i < 8; ++i)                                                                    \
      v[i] = static_cast<LANE>(in[i]);                                                             \
    Value<V, Semantics::Native, kCC> vv{v};                                                        \
    out[0] = fpsan::WRAP<Semantics::Native, kCC>(vv, 0x55u,                                        \
                                                 Value<float, Semantics::Native, kCC>{1.0f});      \
  }                                                                                                \
  TEST(CvtScalef32SrPk8, CASE) {                                                                   \
    if (!have_device())                                                                            \
      GTEST_SKIP() << "no HIP device";                                                             \
    auto in = exact8();                                                                            \
    float *dIn = to_dev(in);                                                                       \
    unsigned *dO;                                                                                  \
    HIP_CHECK(hipMalloc(&dO, sizeof(unsigned)));                                                   \
    CASE##_srn_k<<<1, 1>>>(dIn, dO);                                                               \
    HIP_CHECK(hipDeviceSynchronize());                                                             \
    check_nibbles8(from_dev(dO, 1), in, kFp4E2M1);                                                 \
    (void)hipFree(dIn);                                                                            \
    (void)hipFree(dO);                                                                             \
  }

SR_PK8_NIB_GOLDEN(Fp4FromF32, amdgcn_cvt_scalef32_sr_pk8_fp4_f32, float)
SR_PK8_NIB_GOLDEN(Fp4FromF16, amdgcn_cvt_scalef32_sr_pk8_fp4_f16, _Float16)
SR_PK8_NIB_GOLDEN(Fp4FromBf16, amdgcn_cvt_scalef32_sr_pk8_fp4_bf16, __bf16)
#undef SR_PK8_NIB_GOLDEN

#define SR_PK16_CODES_GOLDEN(CASE, WRAP, FMT, LANE)                                                \
  __global__ void CASE##_sr16_k(const float *in, unsigned *out) {                                  \
    using V = LANE __attribute__((ext_vector_type(16)));                                           \
    V v;                                                                                           \
    for (int i = 0; i < 16; ++i)                                                                   \
      v[i] = static_cast<LANE>(in[i]);                                                             \
    Value<V, Semantics::Native, kCC> vv{v};                                                        \
    fpsan::v3u32_native r = fpsan::WRAP<Semantics::Native, kCC>(                                   \
        vv, 0x99u, Value<float, Semantics::Native, kCC>{1.0f});                                    \
    out[0] = r[0];                                                                                 \
    out[1] = r[1];                                                                                 \
    out[2] = r[2];                                                                                 \
  }                                                                                                \
  TEST(CvtScalef32SrPk16, CASE) {                                                                  \
    if (!have_device())                                                                            \
      GTEST_SKIP() << "no HIP device";                                                             \
    auto in = exact16();                                                                           \
    float *dIn = to_dev(in);                                                                       \
    unsigned *dO;                                                                                  \
    HIP_CHECK(hipMalloc(&dO, 3 * sizeof(unsigned)));                                               \
    CASE##_sr16_k<<<1, 1>>>(dIn, dO);                                                              \
    HIP_CHECK(hipDeviceSynchronize());                                                             \
    check_codes16(from_dev(dO, 3), in, FMT);                                                       \
    (void)hipFree(dIn);                                                                            \
    (void)hipFree(dO);                                                                             \
  }

SR_PK16_CODES_GOLDEN(Fp6FromF32, amdgcn_cvt_scalef32_sr_pk16_fp6_f32, kFp6E2M3, float)
SR_PK16_CODES_GOLDEN(Bf6FromF32, amdgcn_cvt_scalef32_sr_pk16_bf6_f32, kBf6E3M2, float)
SR_PK16_CODES_GOLDEN(Fp6FromF16, amdgcn_cvt_scalef32_sr_pk16_fp6_f16, kFp6E2M3, _Float16)
SR_PK16_CODES_GOLDEN(Bf6FromF16, amdgcn_cvt_scalef32_sr_pk16_bf6_f16, kBf6E3M2, _Float16)
SR_PK16_CODES_GOLDEN(Fp6FromBf16, amdgcn_cvt_scalef32_sr_pk16_fp6_bf16, kFp6E2M3, __bf16)
SR_PK16_CODES_GOLDEN(Bf6FromBf16, amdgcn_cvt_scalef32_sr_pk16_bf6_bf16, kBf6E3M2, __bf16)
#undef SR_PK16_CODES_GOLDEN

// ============================ Scale direction ===============================
// PACK divides by scale (gfx950-verified MX convention): scale=2 stores
// encode(val/2). Inputs/2 stay on the exact fp8 grid, so golden equality holds.
__global__ void k_pk8_scale(const float *in, float scale, unsigned *out) {
  using V = float __attribute__((ext_vector_type(8)));
  V v;
  for (int i = 0; i < 8; ++i)
    v[i] = in[i];
  Value<V, Semantics::Native, kCC> vv{v};
  fpsan::v2u32_native r = fpsan::amdgcn_cvt_scalef32_pk8_fp8_f32<Semantics::Native, kCC>(
      vv, Value<float, Semantics::Native, kCC>{scale});
  out[0] = r[0];
  out[1] = r[1];
}

TEST(CvtScalef32Pk8, Fp8ScaleDivides) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  auto in = exact8();
  float *dIn = to_dev(in);
  unsigned *dO;
  HIP_CHECK(hipMalloc(&dO, 2 * sizeof(unsigned)));
  k_pk8_scale<<<1, 1>>>(dIn, 2.0f, dO);
  HIP_CHECK(hipDeviceSynchronize());
  check_bytes8(from_dev(dO, 2), in, kFp8E4M3, /*scale=*/2.0f);
  (void)hipFree(dIn);
  (void)hipFree(dO);
}

// ============================ FPSan mode ====================================
// Payload references built from the public Value algebra and canonical subbyte
// casts/narrows (independent of the wrapper's internal helpers). Finite cases
// use standard cast canonicalization; algebraic source non-finites use the
// native finite-only subbyte conventions.

template <Semantics S> __global__ void k_fpsan_pk8_fp8(const float *in, unsigned *out) {
  using V = float __attribute__((ext_vector_type(8)));
  V v;
  for (int i = 0; i < 8; ++i)
    v[i] = in[i];
  Value<V, S, kCC> vv{v};
  fpsan::v2u32_native r =
      fpsan::amdgcn_cvt_scalef32_pk8_fp8_f32<S, kCC>(vv, Value<float, S, kCC>{2.0f});
  out[0] = r[0];
  out[1] = r[1];
}

template <Semantics S> void run_fpsan_pk8_fp8_payload() {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  auto in = exact8();
  float *dIn = to_dev(in);
  unsigned *dO;
  HIP_CHECK(hipMalloc(&dO, 2 * sizeof(unsigned)));
  k_fpsan_pk8_fp8<S><<<1, 1>>>(dIn, dO);
  HIP_CHECK(hipDeviceSynchronize());
  auto got = from_dev(dO, 2);
  for (int i = 0; i < 8; ++i) {
    using VF = Value<float, S, kCC>;
    std::uint32_t want = static_cast<std::uint8_t>(
        fpsan::cast<fpsan::fp8_e4m3>(VF{in[i]} / VF{2.0f}).fpsan_payload());
    std::uint32_t have = (got[i / 4] >> (8 * (i % 4))) & 0xFFu;
    EXPECT_EQ(have, want) << "byte " << i;
  }
  (void)hipFree(dIn);
  (void)hipFree(dO);
}

TEST(CvtScalef32Pk8, FpsanFp8Payload) { FPSAN_RUN_FPSAN_SEMANTICS(run_fpsan_pk8_fp8_payload); }

#define FPSAN_DEFINE_PK8_FP4_CANONICAL_TEST(SUITE, CASE, WRAP, LANE)                               \
  template <Semantics S> __global__ void CASE##_k(const float *in, unsigned *out) {                \
    constexpr float scale = 2.0f;                                                                  \
    using V = LANE __attribute__((ext_vector_type(8)));                                            \
    V v;                                                                                           \
    for (int i = 0; i < 8; ++i)                                                                    \
      v[i] = static_cast<LANE>(in[i]);                                                             \
    Value<V, S, kCC> vv{v};                                                                        \
    out[0] = fpsan::WRAP<S, kCC>(vv, Value<float, S, kCC>{scale});                                 \
  }                                                                                                \
  template <Semantics S> void CASE##_run() {                                                       \
    if (!have_device())                                                                            \
      GTEST_SKIP() << "no HIP device";                                                             \
    constexpr float scale = 2.0f;                                                                  \
    auto in = fpsan_nonfinite8();                                                                  \
    float *dIn = to_dev(in);                                                                       \
    unsigned *dO;                                                                                  \
    HIP_CHECK(hipMalloc(&dO, sizeof(unsigned)));                                                   \
    CASE##_k<S><<<1, 1>>>(dIn, dO);                                                                \
    HIP_CHECK(hipDeviceSynchronize());                                                             \
    unsigned got = from_dev(dO, 1)[0];                                                             \
    for (int i = 0; i < 8; ++i) {                                                                  \
      std::uint32_t want = canonical_narrow_lane<4, LANE, S>(in[i], scale);                        \
      std::uint32_t have = (got >> (4 * i)) & 0xFu;                                                \
      EXPECT_EQ(have, want) << "nibble " << i << " S=" << int(S);                                  \
    }                                                                                              \
    (void)hipFree(dIn);                                                                            \
    (void)hipFree(dO);                                                                             \
  }                                                                                                \
  TEST(SUITE, CASE) { FPSAN_RUN_FPSAN_SEMANTICS(CASE##_run); }

#define FPSAN_DEFINE_SR_PK8_FP4_CANONICAL_TEST(SUITE, CASE, WRAP, LANE)                            \
  template <Semantics S> __global__ void CASE##_k(const float *in, unsigned *out) {                \
    constexpr float scale = 2.0f;                                                                  \
    using V = LANE __attribute__((ext_vector_type(8)));                                            \
    V v;                                                                                           \
    for (int i = 0; i < 8; ++i)                                                                    \
      v[i] = static_cast<LANE>(in[i]);                                                             \
    Value<V, S, kCC> vv{v};                                                                        \
    out[0] = fpsan::WRAP<S, kCC>(vv, 0xdeadbeefu, Value<float, S, kCC>{scale});                    \
  }                                                                                                \
  template <Semantics S> void CASE##_run() {                                                       \
    if (!have_device())                                                                            \
      GTEST_SKIP() << "no HIP device";                                                             \
    constexpr float scale = 2.0f;                                                                  \
    auto in = fpsan_nonfinite8();                                                                  \
    float *dIn = to_dev(in);                                                                       \
    unsigned *dO;                                                                                  \
    HIP_CHECK(hipMalloc(&dO, sizeof(unsigned)));                                                   \
    CASE##_k<S><<<1, 1>>>(dIn, dO);                                                                \
    HIP_CHECK(hipDeviceSynchronize());                                                             \
    unsigned got = from_dev(dO, 1)[0];                                                             \
    for (int i = 0; i < 8; ++i) {                                                                  \
      std::uint32_t want = canonical_narrow_lane<4, LANE, S>(in[i], scale);                        \
      std::uint32_t have = (got >> (4 * i)) & 0xFu;                                                \
      EXPECT_EQ(have, want) << "nibble " << i << " S=" << int(S);                                  \
    }                                                                                              \
    (void)hipFree(dIn);                                                                            \
    (void)hipFree(dO);                                                                             \
  }                                                                                                \
  TEST(SUITE, CASE) { FPSAN_RUN_FPSAN_SEMANTICS(CASE##_run); }

FPSAN_DEFINE_PK8_FP4_CANONICAL_TEST(CvtScalef32Pk8, FpsanFp4FromF32CanonicalPayload,
                                    amdgcn_cvt_scalef32_pk8_fp4_f32, float)
FPSAN_DEFINE_PK8_FP4_CANONICAL_TEST(CvtScalef32Pk8, FpsanFp4FromF16CanonicalPayload,
                                    amdgcn_cvt_scalef32_pk8_fp4_f16, _Float16)
FPSAN_DEFINE_PK8_FP4_CANONICAL_TEST(CvtScalef32Pk8, FpsanFp4FromBf16CanonicalPayload,
                                    amdgcn_cvt_scalef32_pk8_fp4_bf16, __bf16)
FPSAN_DEFINE_SR_PK8_FP4_CANONICAL_TEST(CvtScalef32SrPk8, FpsanSrFp4FromF32CanonicalPayload,
                                       amdgcn_cvt_scalef32_sr_pk8_fp4_f32, float)
FPSAN_DEFINE_SR_PK8_FP4_CANONICAL_TEST(CvtScalef32SrPk8, FpsanSrFp4FromF16CanonicalPayload,
                                       amdgcn_cvt_scalef32_sr_pk8_fp4_f16, _Float16)
FPSAN_DEFINE_SR_PK8_FP4_CANONICAL_TEST(CvtScalef32SrPk8, FpsanSrFp4FromBf16CanonicalPayload,
                                       amdgcn_cvt_scalef32_sr_pk8_fp4_bf16, __bf16)
#undef FPSAN_DEFINE_PK8_FP4_CANONICAL_TEST
#undef FPSAN_DEFINE_SR_PK8_FP4_CANONICAL_TEST

#define FPSAN_DEFINE_PK16_SUB6_CANONICAL_TEST(SUITE, CASE, WRAP, LANE)                             \
  template <Semantics S> __global__ void CASE##_k(const float *in, unsigned *out) {                \
    constexpr float scale = 2.0f;                                                                  \
    using V = LANE __attribute__((ext_vector_type(16)));                                           \
    V v;                                                                                           \
    for (int i = 0; i < 16; ++i)                                                                   \
      v[i] = static_cast<LANE>(in[i]);                                                             \
    Value<V, S, kCC> vv{v};                                                                        \
    fpsan::v3u32_native r = fpsan::WRAP<S, kCC>(vv, Value<float, S, kCC>{scale});                  \
    out[0] = r[0];                                                                                 \
    out[1] = r[1];                                                                                 \
    out[2] = r[2];                                                                                 \
  }                                                                                                \
  template <Semantics S> void CASE##_run() {                                                       \
    if (!have_device())                                                                            \
      GTEST_SKIP() << "no HIP device";                                                             \
    constexpr float scale = 2.0f;                                                                  \
    auto in = fpsan_nonfinite16();                                                                 \
    float *dIn = to_dev(in);                                                                       \
    unsigned *dO;                                                                                  \
    HIP_CHECK(hipMalloc(&dO, 3 * sizeof(unsigned)));                                               \
    CASE##_k<S><<<1, 1>>>(dIn, dO);                                                                \
    HIP_CHECK(hipDeviceSynchronize());                                                             \
    auto got = from_dev(dO, 3);                                                                    \
    for (int i = 0; i < 16; ++i) {                                                                 \
      std::uint32_t want = canonical_narrow_lane<6, LANE, S>(in[i], scale);                        \
      std::uint32_t have = host_extract6(got, i);                                                  \
      EXPECT_EQ(have, want) << "code " << i << " S=" << int(S);                                    \
    }                                                                                              \
    (void)hipFree(dIn);                                                                            \
    (void)hipFree(dO);                                                                             \
  }                                                                                                \
  TEST(SUITE, CASE) { FPSAN_RUN_FPSAN_SEMANTICS(CASE##_run); }

#define FPSAN_DEFINE_SR_PK16_SUB6_CANONICAL_TEST(SUITE, CASE, WRAP, LANE)                          \
  template <Semantics S> __global__ void CASE##_k(const float *in, unsigned *out) {                \
    constexpr float scale = 2.0f;                                                                  \
    using V = LANE __attribute__((ext_vector_type(16)));                                           \
    V v;                                                                                           \
    for (int i = 0; i < 16; ++i)                                                                   \
      v[i] = static_cast<LANE>(in[i]);                                                             \
    Value<V, S, kCC> vv{v};                                                                        \
    fpsan::v3u32_native r = fpsan::WRAP<S, kCC>(vv, 0x999u, Value<float, S, kCC>{scale});          \
    out[0] = r[0];                                                                                 \
    out[1] = r[1];                                                                                 \
    out[2] = r[2];                                                                                 \
  }                                                                                                \
  template <Semantics S> void CASE##_run() {                                                       \
    if (!have_device())                                                                            \
      GTEST_SKIP() << "no HIP device";                                                             \
    constexpr float scale = 2.0f;                                                                  \
    auto in = fpsan_nonfinite16();                                                                 \
    float *dIn = to_dev(in);                                                                       \
    unsigned *dO;                                                                                  \
    HIP_CHECK(hipMalloc(&dO, 3 * sizeof(unsigned)));                                               \
    CASE##_k<S><<<1, 1>>>(dIn, dO);                                                                \
    HIP_CHECK(hipDeviceSynchronize());                                                             \
    auto got = from_dev(dO, 3);                                                                    \
    for (int i = 0; i < 16; ++i) {                                                                 \
      std::uint32_t want = canonical_narrow_lane<6, LANE, S>(in[i], scale);                        \
      std::uint32_t have = host_extract6(got, i);                                                  \
      EXPECT_EQ(have, want) << "code " << i << " S=" << int(S);                                    \
    }                                                                                              \
    (void)hipFree(dIn);                                                                            \
    (void)hipFree(dO);                                                                             \
  }                                                                                                \
  TEST(SUITE, CASE) { FPSAN_RUN_FPSAN_SEMANTICS(CASE##_run); }

FPSAN_DEFINE_PK16_SUB6_CANONICAL_TEST(CvtScalef32Pk16, FpsanFp6FromF32CanonicalPayload,
                                      amdgcn_cvt_scalef32_pk16_fp6_f32, float)
FPSAN_DEFINE_PK16_SUB6_CANONICAL_TEST(CvtScalef32Pk16, FpsanBf6FromF32CanonicalPayload,
                                      amdgcn_cvt_scalef32_pk16_bf6_f32, float)
FPSAN_DEFINE_PK16_SUB6_CANONICAL_TEST(CvtScalef32Pk16, FpsanFp6FromF16CanonicalPayload,
                                      amdgcn_cvt_scalef32_pk16_fp6_f16, _Float16)
FPSAN_DEFINE_PK16_SUB6_CANONICAL_TEST(CvtScalef32Pk16, FpsanBf6FromF16CanonicalPayload,
                                      amdgcn_cvt_scalef32_pk16_bf6_f16, _Float16)
FPSAN_DEFINE_PK16_SUB6_CANONICAL_TEST(CvtScalef32Pk16, FpsanFp6FromBf16CanonicalPayload,
                                      amdgcn_cvt_scalef32_pk16_fp6_bf16, __bf16)
FPSAN_DEFINE_PK16_SUB6_CANONICAL_TEST(CvtScalef32Pk16, FpsanBf6FromBf16CanonicalPayload,
                                      amdgcn_cvt_scalef32_pk16_bf6_bf16, __bf16)
FPSAN_DEFINE_SR_PK16_SUB6_CANONICAL_TEST(CvtScalef32SrPk16, FpsanSrFp6FromF32CanonicalPayload,
                                         amdgcn_cvt_scalef32_sr_pk16_fp6_f32, float)
FPSAN_DEFINE_SR_PK16_SUB6_CANONICAL_TEST(CvtScalef32SrPk16, FpsanSrBf6FromF32CanonicalPayload,
                                         amdgcn_cvt_scalef32_sr_pk16_bf6_f32, float)
FPSAN_DEFINE_SR_PK16_SUB6_CANONICAL_TEST(CvtScalef32SrPk16, FpsanSrFp6FromF16CanonicalPayload,
                                         amdgcn_cvt_scalef32_sr_pk16_fp6_f16, _Float16)
FPSAN_DEFINE_SR_PK16_SUB6_CANONICAL_TEST(CvtScalef32SrPk16, FpsanSrBf6FromF16CanonicalPayload,
                                         amdgcn_cvt_scalef32_sr_pk16_bf6_f16, _Float16)
FPSAN_DEFINE_SR_PK16_SUB6_CANONICAL_TEST(CvtScalef32SrPk16, FpsanSrFp6FromBf16CanonicalPayload,
                                         amdgcn_cvt_scalef32_sr_pk16_fp6_bf16, __bf16)
FPSAN_DEFINE_SR_PK16_SUB6_CANONICAL_TEST(CvtScalef32SrPk16, FpsanSrBf6FromBf16CanonicalPayload,
                                         amdgcn_cvt_scalef32_sr_pk16_bf6_bf16, __bf16)
#undef FPSAN_DEFINE_PK16_SUB6_CANONICAL_TEST
#undef FPSAN_DEFINE_SR_PK16_SUB6_CANONICAL_TEST

#endif // has builtin
