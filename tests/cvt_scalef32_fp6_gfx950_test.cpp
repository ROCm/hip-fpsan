// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/cvt_scalef32_fp6_gfx950_test.cpp
//
// gfx950 tests for the fpsan:: fp6 (e2m3) / bf6 (e3m2) scaled-conversion
// WRAPPERS in fpsan/amdgcn_cvt.hpp (pk32_* unpack; 2xpk16_*_f32 + pk32_*_f16/
// bf16 + sr_pk32_*_f32 pack). Float vs the INDEPENDENT host OCP reference;
// FPSan canonicalizes finite fp6/bf6 codes through the standard cast policy;
// algebraic source non-finites narrow to fixed finite-only codes.
#include "fpsan/amdgcn_cvt.hpp"
#include "fpsan/fpsan.hpp"

#include "fpsan_semantics.hpp"
#include "subbyte_oracle.hpp"

#include "hip_test_utils.hpp"
#include "test_random.hpp"

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <vector>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;
using fpsan::detail::f32_to_narrow;
using fpsan::detail::kBf6E3M2;
using fpsan::detail::kFp6E2M3;
using fpsan::detail::narrow_to_f32;

static constexpr Conversions kCC = Conversions::Explicit;
using v6u = unsigned __attribute__((ext_vector_type(6)));

namespace {
// Independent host bitstream helpers (separate impl from the device side).
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
std::uint8_t host_extract6(const unsigned *w, int i) {
  std::uint8_t v = 0;
  for (int b = 0; b < 6; ++b) {
    int p = i * 6 + b;
    if ((w[p / 32] >> (p % 32)) & 1)
      v |= 1u << b;
  }
  return v;
}
} // namespace

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_scalef32_pk32_f32_fp6)

// ============================ Native mode ====================================

__global__ void k_f32_unpack_fp6(const unsigned *in6, float *out, float scale) {
  v6u s;
  for (int i = 0; i < 6; ++i)
    s[i] = in6[i];
  auto r = fpsan::amdgcn_cvt_scalef32_pk32_f32_fp6<Semantics::Native, kCC>(
      s, Value<float, Semantics::Native, kCC>{scale});
  for (int i = 0; i < 32; ++i)
    out[i] = r.get(i).to_float();
}
__global__ void k_f32_unpack_bf6(const unsigned *in6, float *out, float scale) {
  v6u s;
  for (int i = 0; i < 6; ++i)
    s[i] = in6[i];
  auto r = fpsan::amdgcn_cvt_scalef32_pk32_f32_bf6<Semantics::Native, kCC>(
      s, Value<float, Semantics::Native, kCC>{scale});
  for (int i = 0; i < 32; ++i)
    out[i] = r.get(i).to_float();
}

template <class Kern>
void float_unpack_all(Kern kern, const fpsan::detail::FpFormat &fmt, const char *name) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  float *dO;
  HIP_CHECK(hipMalloc(&dO, 32 * sizeof(float)));
  for (int base = 0; base < 64; base += 32) {
    std::uint8_t codes[32];
    for (int i = 0; i < 32; ++i)
      codes[i] = std::uint8_t(base + i);
    unsigned packed[6];
    host_pack6(codes, 32, packed);
    std::vector<unsigned> hp(packed, packed + 6);
    unsigned *dIn = to_dev(hp);
    kern<<<1, 1>>>(dIn, dO, 1.0f);
    HIP_CHECK(hipDeviceSynchronize());
    auto got = from_dev(dO, 32);
    for (int i = 0; i < 32; ++i)
      EXPECT_EQ(got[i], narrow_to_f32(unsigned(codes[i]), fmt))
          << name << " code " << int(codes[i]);
    (void)hipFree(dIn);
  }
  (void)hipFree(dO);
}

TEST(CvtScalef32Fp6Wrap, FloatUnpackFp6AllCodes) {
  float_unpack_all(k_f32_unpack_fp6, kFp6E2M3, "fp6");
}
TEST(CvtScalef32Fp6Wrap, FloatUnpackBf6AllCodes) {
  float_unpack_all(k_f32_unpack_bf6, kBf6E3M2, "bf6");
}

__global__ void k_f32_pack_fp6(const float *in, unsigned *out, float scale) {
  fpsan::v16f_native_cvt lo, hi;
  for (int i = 0; i < 16; ++i) {
    lo[i] = in[i];
    hi[i] = in[16 + i];
  }
  v6u r = fpsan::amdgcn_cvt_scalef32_2xpk16_fp6_f32<Semantics::Native, kCC>(
      Value<fpsan::v16f_native_cvt, Semantics::Native, kCC>{lo},
      Value<fpsan::v16f_native_cvt, Semantics::Native, kCC>{hi},
      Value<float, Semantics::Native, kCC>{scale});
  for (int i = 0; i < 6; ++i)
    out[i] = r[i];
}

TEST(CvtScalef32Fp6Wrap, FloatPack2xpk16ExactBits) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  std::vector<float> in(32);
  for (int c = 0; c < 32; ++c)
    in[c] = narrow_to_f32(unsigned(c), kFp6E2M3);
  float *dIn = to_dev(in);
  unsigned *dO;
  HIP_CHECK(hipMalloc(&dO, 6 * sizeof(unsigned)));
  k_f32_pack_fp6<<<1, 1>>>(dIn, dO, 1.0f);
  HIP_CHECK(hipDeviceSynchronize());
  auto got = from_dev(dO, 6);
  // Expected: interleaved codes, field 2k=lo[k], 2k+1=hi[k].
  std::uint8_t inter[32];
  for (int k = 0; k < 16; ++k) {
    inter[2 * k] = f32_to_narrow(in[k], kFp6E2M3) & 0x3F;
    inter[2 * k + 1] = f32_to_narrow(in[16 + k], kFp6E2M3) & 0x3F;
  }
  unsigned expect[6];
  host_pack6(inter, 32, expect);
  for (int i = 0; i < 6; ++i)
    EXPECT_EQ(got[i], expect[i]) << "word " << i;
  (void)hipFree(dIn);
  (void)hipFree(dO);
}

// ============================ FPSan mode ====================================

// Unpack: packed v6u32 holds finite fp6/bf6 codes. The wrapper widens, applies
// the f32 scale, and then casts to the destination type.
#define FPSAN_DEFINE_SUB6_UNPACK_CANONICAL_TEST(CASE, WRAP, DST, LABEL)                            \
  template <Semantics S>                                                                           \
  __global__ void CASE##_kernel(const unsigned *in6, float scale, unsigned *out) {                 \
    using VF = Value<float, S, kCC>;                                                               \
    v6u s;                                                                                         \
    for (int i = 0; i < 6; ++i)                                                                    \
      s[i] = in6[i];                                                                               \
    auto r = fpsan::WRAP<S, kCC>(s, VF{scale});                                                    \
    for (int i = 0; i < 32; ++i) {                                                                 \
      const auto lane = r.get(i);                                                                  \
      out[i] = lane.fpsan_payload();                                                               \
      out[32 + i] = (lane + lane).fpsan_payload();                                                 \
    }                                                                                              \
  }                                                                                                \
  template <Semantics S> void CASE##_run() {                                                       \
    if (!have_device())                                                                            \
      GTEST_SKIP() << "no HIP device";                                                             \
    const std::uint8_t codes[32] = {0x00u, 0x01u, 0x1Fu, 0x20u, 0x21u, 0x2Au, 0x30u, 0x3Fu,        \
                                    0x02u, 0x07u, 0x10u, 0x18u, 0x22u, 0x2Fu, 0x3Eu, 0x15u,        \
                                    0x04u, 0x08u, 0x0Fu, 0x11u, 0x1Eu, 0x23u, 0x31u, 0x3Du,        \
                                    0x05u, 0x09u, 0x12u, 0x19u, 0x24u, 0x2Cu, 0x38u, 0x3Cu};       \
    constexpr float scale = 2.0f;                                                                  \
    unsigned packed[6];                                                                            \
    host_pack6(codes, 32, packed);                                                                 \
    unsigned *dIn = to_dev(std::vector<unsigned>(packed, packed + 6));                             \
    unsigned *dO;                                                                                  \
    HIP_CHECK(hipMalloc(&dO, 64 * sizeof(unsigned)));                                              \
    CASE##_kernel<S><<<1, 1>>>(dIn, scale, dO);                                                    \
    HIP_CHECK(hipDeviceSynchronize());                                                             \
    auto got = from_dev(dO, 64);                                                                   \
    using VF = Value<float, S, kCC>;                                                               \
    using Out = Value<DST, S, kCC>;                                                                \
    for (int i = 0; i < 32; ++i) {                                                                 \
      const auto want =                                                                            \
          fpsan::cast<DST>(fpsan_test::canonical_subbyte_widen<6, S, kCC>(codes[i]) * VF{scale});  \
      EXPECT_EQ(got[i], static_cast<unsigned>(want.fpsan_payload()))                               \
          << LABEL << " elem " << i << " code=0x" << std::hex << unsigned(codes[i]);               \
      EXPECT_EQ(got[32 + i], static_cast<unsigned>((want + want).fpsan_payload()))                 \
          << LABEL << " follow-on add elem " << i << " code=0x" << std::hex << unsigned(codes[i]); \
      (void)sizeof(Out);                                                                           \
    }                                                                                              \
    (void)hipFree(dIn);                                                                            \
    (void)hipFree(dO);                                                                             \
  }                                                                                                \
  TEST(CvtScalef32Fp6Wrap, CASE) { FPSAN_RUN_FPSAN_SEMANTICS(CASE##_run); }

FPSAN_DEFINE_SUB6_UNPACK_CANONICAL_TEST(FpsanFp6ToF32CanonicalPayload,
                                        amdgcn_cvt_scalef32_pk32_f32_fp6, float, "fp6->f32")
FPSAN_DEFINE_SUB6_UNPACK_CANONICAL_TEST(FpsanBf6ToF32CanonicalPayload,
                                        amdgcn_cvt_scalef32_pk32_f32_bf6, float, "bf6->f32")
FPSAN_DEFINE_SUB6_UNPACK_CANONICAL_TEST(FpsanFp6ToF16CanonicalPayload,
                                        amdgcn_cvt_scalef32_pk32_f16_fp6, _Float16, "fp6->f16")
FPSAN_DEFINE_SUB6_UNPACK_CANONICAL_TEST(FpsanBf6ToF16CanonicalPayload,
                                        amdgcn_cvt_scalef32_pk32_f16_bf6, _Float16, "bf6->f16")
FPSAN_DEFINE_SUB6_UNPACK_CANONICAL_TEST(FpsanFp6ToBf16CanonicalPayload,
                                        amdgcn_cvt_scalef32_pk32_bf16_fp6, __bf16, "fp6->bf16")
FPSAN_DEFINE_SUB6_UNPACK_CANONICAL_TEST(FpsanBf6ToBf16CanonicalPayload,
                                        amdgcn_cvt_scalef32_pk32_bf16_bf6, __bf16, "bf6->bf16")
#undef FPSAN_DEFINE_SUB6_UNPACK_CANONICAL_TEST

// Pack (2xpk16): wrapper divides by scale, canonically narrows to 6-bit
// subbyte codes, then interleaves. Decode the result with the independent
// host_extract6 and compare to the canonical oracle.
template <Semantics S> __global__ void k_fpsan_pack(const float *in, unsigned *out, float scale) {
  using VF = Value<float, S, kCC>;
  fpsan::v16f_native_cvt lo, hi;
  for (int i = 0; i < 16; ++i) {
    lo[i] = in[i];
    hi[i] = in[16 + i];
  }
  v6u r = fpsan::amdgcn_cvt_scalef32_2xpk16_fp6_f32<S, kCC>(
      Value<fpsan::v16f_native_cvt, S, kCC>{lo}, Value<fpsan::v16f_native_cvt, S, kCC>{hi},
      VF{scale});
  for (int i = 0; i < 6; ++i)
    out[i] = r[i];
}

TEST(CvtScalef32Fp6Wrap, FpsanPackDivideInterleave) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VF = Value<float, S, kCC>;
    std::vector<float> in(32);
    std::mt19937 rng = fpsan_test::make_rng();
    for (auto &x : in)
      x = fpsan_test::pick_int_valued<float>(rng, -7, 7);
    in[3] = std::numeric_limits<float>::quiet_NaN();
    in[7] = std::numeric_limits<float>::infinity();
    in[19] = std::numeric_limits<float>::quiet_NaN();
    in[23] = -std::numeric_limits<float>::infinity();
    const float scale = 2.0f;
    float *dIn = to_dev(in);
    unsigned *dO;
    HIP_CHECK(hipMalloc(&dO, 6 * sizeof(unsigned)));
    k_fpsan_pack<S><<<1, 1>>>(dIn, dO, scale);
    HIP_CHECK(hipDeviceSynchronize());
    auto got = from_dev(dO, 6);
    for (int k = 0; k < 16; ++k) {
      unsigned refLo = fpsan_test::canonical_subbyte_narrow_code<6>(VF{in[k]} / VF{scale});
      unsigned refHi = fpsan_test::canonical_subbyte_narrow_code<6>(VF{in[16 + k]} / VF{scale});
      EXPECT_EQ(host_extract6(got.data(), 2 * k), refLo) << "lo " << k;
      EXPECT_EQ(host_extract6(got.data(), 2 * k + 1), refHi) << "hi " << k;
    }
    (void)hipFree(dIn);
    (void)hipFree(dO);
  });
}

#endif // has builtin
