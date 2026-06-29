// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/cvt_scalef32_fp4_gfx950_test.cpp
//
// gfx950 tests for the fpsan:: fp4 (e2m1) scaled-conversion WRAPPERS in
// fpsan/amdgcn_cvt.hpp (cvt_scalef32_pk_{f32,f16,bf16}_fp4 unpack;
// cvt_scalef32_pk_fp4_{f32,f16,bf16} + sr_pk_fp4_f32 pack). The sibling file
// cvt_scalef32_mx_gfx950_test.cpp exercises the raw builtins; this one drives
// the Value-typed wrappers in BOTH modes:
//
//   Float : wrapper == hardware, checked against the INDEPENDENT host OCP
//           reference (detail::narrow_to_f32 / f32_to_narrow) -- not the
//           builtin -- so any implementation that diverges from the host reference fails here.
//   FPSan : wrappers canonicalize finite fp4 codes through the standard cast
//           policy; algebraic source non-finites narrow to fixed finite-only
//           codes.
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
using fpsan::detail::kFp4E2M1;
using fpsan::detail::narrow_to_f32;

static constexpr Conversions kCC = Conversions::Explicit;

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_cvt_scalef32_pk_f32_fp4)

// ============================ Native mode ====================================

// Unpack every fp4 code through the WRAPPER (Sel=0), scale=1, vs OCP host ref.
__global__ void k_f32_unpack(const unsigned *packed, float *out) {
  int l = threadIdx.x;
  auto r = fpsan::amdgcn_cvt_scalef32_pk_f32_fp4<0, Semantics::Native, kCC>(
      packed[l], Value<float, Semantics::Native, kCC>{1.0f});
  out[2 * l] = r.get(0).to_float();
  out[2 * l + 1] = r.get(1).to_float();
}

TEST(CvtScalef32Fp4Wrap, FloatUnpackAllCodes) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  std::vector<unsigned> in(16);
  for (int c = 0; c < 16; ++c)
    in[c] = unsigned(c) | (unsigned(c) << 4);
  unsigned *dIn = to_dev(in);
  float *dO;
  HIP_CHECK(hipMalloc(&dO, 32 * sizeof(float)));
  k_f32_unpack<<<1, 16>>>(dIn, dO);
  HIP_CHECK(hipDeviceSynchronize());
  auto got = from_dev(dO, 32);
  for (int c = 0; c < 16; ++c) {
    float ref = narrow_to_f32(unsigned(c), kFp4E2M1);
    EXPECT_EQ(got[2 * c], ref) << "code " << c << " elem0";
    EXPECT_EQ(got[2 * c + 1], ref) << "code " << c << " elem1";
  }
  (void)hipFree(dIn);
  (void)hipFree(dO);
}

// Pack pairs of representable fp4 values through the wrapper (Sel=0, scale=1)
// vs OCP host encode; also confirm Sel placement + `old` preservation.
template <int Sel> __global__ void k_f32_pack(unsigned old, float a, float b, unsigned *out) {
  *out = fpsan::amdgcn_cvt_scalef32_pk_fp4_f32<Sel, Semantics::Native, kCC>(
      old, Value<float, Semantics::Native, kCC>{a}, Value<float, Semantics::Native, kCC>{b},
      Value<float, Semantics::Native, kCC>{1.0f});
}

TEST(CvtScalef32Fp4Wrap, FloatPackExactBitsAndSel) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  unsigned *dO;
  HIP_CHECK(hipMalloc(&dO, sizeof(unsigned)));
  const unsigned old = 0xA5A5A5A5u;
  for (int ca = 0; ca < 8; ++ca)
    for (int cb = 0; cb < 8; ++cb) {
      float a = narrow_to_f32(unsigned(ca), kFp4E2M1);
      float b = narrow_to_f32(unsigned(cb), kFp4E2M1);
      unsigned pair =
          (f32_to_narrow(a, kFp4E2M1) & 0xF) | ((f32_to_narrow(b, kFp4E2M1) & 0xF) << 4);
      k_f32_pack<0><<<1, 1>>>(old, a, b, dO);
      HIP_CHECK(hipDeviceSynchronize());
      unsigned got = from_dev(dO, 1)[0];
      unsigned expect = (old & 0xFFFFFF00u) | pair;
      EXPECT_EQ(got, expect) << "a=" << ca << " b=" << cb;
    }
  // Sel places the pair in byte Sel, preserving the rest.
  float a = narrow_to_f32(0x6u, kFp4E2M1), b = narrow_to_f32(0x7u, kFp4E2M1);
  unsigned pair = 0x6u | (0x7u << 4);
  auto run = [&](auto sel_tag, int sel) {
    k_f32_pack<decltype(sel_tag)::value><<<1, 1>>>(old, a, b, dO);
    HIP_CHECK(hipDeviceSynchronize());
    unsigned got = from_dev(dO, 1)[0];
    unsigned mask = 0xFFu << (8 * sel);
    EXPECT_EQ(got, (old & ~mask) | (pair << (8 * sel))) << "sel " << sel;
  };
  run(std::integral_constant<int, 0>{}, 0);
  run(std::integral_constant<int, 1>{}, 1);
  run(std::integral_constant<int, 2>{}, 2);
  run(std::integral_constant<int, 3>{}, 3);
  (void)hipFree(dO);
}

// Scale direction through the wrapper: unpack multiplies, pack divides.
__global__ void k_f32_scale(unsigned packed, float scale, float *outUnpack, float a, float pscale,
                            unsigned *outPack) {
  auto r = fpsan::amdgcn_cvt_scalef32_pk_f32_fp4<0, Semantics::Native, kCC>(
      packed, Value<float, Semantics::Native, kCC>{scale});
  outUnpack[0] = r.get(0).to_float();
  *outPack = fpsan::amdgcn_cvt_scalef32_pk_fp4_f32<0, Semantics::Native, kCC>(
      0u, Value<float, Semantics::Native, kCC>{a}, Value<float, Semantics::Native, kCC>{0.0f},
      Value<float, Semantics::Native, kCC>{pscale});
}

TEST(CvtScalef32Fp4Wrap, FloatScaleDirection) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  float *dU;
  unsigned *dP;
  HIP_CHECK(hipMalloc(&dU, sizeof(float)));
  HIP_CHECK(hipMalloc(&dP, sizeof(unsigned)));
  // packed nibble0 = code 2 (=1.0); unpack scale=4 -> 4.0.
  k_f32_scale<<<1, 1>>>(0x2u, 4.0f, dU, /*a=*/8.0f, /*pscale=*/2.0f, dP);
  HIP_CHECK(hipDeviceSynchronize());
  EXPECT_EQ(from_dev(dU, 1)[0], 4.0f); // 1.0 * 4
  // pack 8.0 / 2 = 4.0 -> code 6.
  EXPECT_EQ(from_dev(dP, 1)[0] & 0xF, unsigned(0x6));
  (void)hipFree(dU);
  (void)hipFree(dP);
}

// ============================ FPSan mode ====================================

// Unpack plumbing: packed holds finite fp4 codes; widening follows the standard
// cast policy for the active FPSan semantics.
template <Semantics S>
__global__ void k_fpsan_unpack_f32(unsigned packed, float scale, unsigned *out) {
  using VF = Value<float, S, kCC>;
  auto r = fpsan::amdgcn_cvt_scalef32_pk_f32_fp4<1, S, kCC>(packed, VF{scale});
  out[0] = r.get(0).fpsan_payload();
  out[1] = r.get(1).fpsan_payload();
  out[2] = (r.get(0) + r.get(0)).fpsan_payload();
  out[3] = (r.get(1) + r.get(1)).fpsan_payload();
}

template <Semantics S>
__global__ void k_fpsan_unpack_f16(unsigned packed, float scale, unsigned *out) {
  using VF = Value<float, S, kCC>;
  auto r = fpsan::amdgcn_cvt_scalef32_pk_f16_fp4<1, S, kCC>(packed, VF{scale});
  out[0] = r.get(0).fpsan_payload();
  out[1] = r.get(1).fpsan_payload();
  out[2] = (r.get(0) + r.get(0)).fpsan_payload();
  out[3] = (r.get(1) + r.get(1)).fpsan_payload();
}

template <Semantics S>
__global__ void k_fpsan_unpack_bf16(unsigned packed, float scale, unsigned *out) {
  using VF = Value<float, S, kCC>;
  auto r = fpsan::amdgcn_cvt_scalef32_pk_bf16_fp4<1, S, kCC>(packed, VF{scale});
  out[0] = r.get(0).fpsan_payload();
  out[1] = r.get(1).fpsan_payload();
  out[2] = (r.get(0) + r.get(0)).fpsan_payload();
  out[3] = (r.get(1) + r.get(1)).fpsan_payload();
}

template <class DstFT, Semantics S>
void expect_fp4_pair(const std::vector<unsigned> &got, std::uint32_t c0, std::uint32_t c1,
                     float scale) {
  using VF = Value<float, S, kCC>;
  using Out = Value<DstFT, S, kCC>;
  const auto src0 = fpsan_test::canonical_subbyte_widen<4, S, kCC>(c0) * VF{scale};
  const auto src1 = fpsan_test::canonical_subbyte_widen<4, S, kCC>(c1) * VF{scale};
  const auto want0 = fpsan::cast<DstFT>(src0);
  const auto want1 = fpsan::cast<DstFT>(src1);
  const auto got0 = Out::from_fpsan_payload(static_cast<typename Out::bits_type>(got[0]));
  const auto got1 = Out::from_fpsan_payload(static_cast<typename Out::bits_type>(got[1]));
  EXPECT_EQ(got[0], static_cast<unsigned>(want0.fpsan_payload())) << "code 0x" << std::hex << c0;
  EXPECT_EQ(got[1], static_cast<unsigned>(want1.fpsan_payload())) << "code 0x" << std::hex << c1;
  EXPECT_EQ(got[2], static_cast<unsigned>((got0 + got0).fpsan_payload()))
      << "follow-on add code 0x" << std::hex << c0;
  EXPECT_EQ(got[3], static_cast<unsigned>((got1 + got1).fpsan_payload()))
      << "follow-on add code 0x" << std::hex << c1;
  EXPECT_EQ(got[2], static_cast<unsigned>((want0 + want0).fpsan_payload()))
      << "canonical follow-on add code 0x" << std::hex << c0;
  EXPECT_EQ(got[3], static_cast<unsigned>((want1 + want1).fpsan_payload()))
      << "canonical follow-on add code 0x" << std::hex << c1;
}

template <class DstFT, Semantics S, class Launch> void run_fpsan_unpack_canonical(Launch launch) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  constexpr std::uint32_t cases[][2] = {{0x7u, 0xFu}, {0x8u, 0xBu}, {0xEu, 0x1u}};
  constexpr float scale = 2.0f;
  unsigned *dO;
  HIP_CHECK(hipMalloc(&dO, 4 * sizeof(unsigned)));
  for (const auto &pair : cases) {
    const unsigned packed = (pair[0] << 8) | (pair[1] << 12);
    launch(packed, scale, dO);
    HIP_CHECK(hipDeviceSynchronize());
    expect_fp4_pair<DstFT, S>(from_dev(dO, 4), pair[0], pair[1], scale);
  }
  (void)hipFree(dO);
}

TEST(CvtScalef32Fp4Wrap, FpsanUnpackF32CanonicalPayload) {
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    run_fpsan_unpack_canonical<float, S>([](unsigned packed, float scale, unsigned *dO) {
      k_fpsan_unpack_f32<S><<<1, 1>>>(packed, scale, dO);
    });
  });
}

TEST(CvtScalef32Fp4Wrap, FpsanUnpackF16CanonicalPayload) {
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    run_fpsan_unpack_canonical<_Float16, S>([](unsigned packed, float scale, unsigned *dO) {
      k_fpsan_unpack_f16<S><<<1, 1>>>(packed, scale, dO);
    });
  });
}

TEST(CvtScalef32Fp4Wrap, FpsanUnpackBf16CanonicalPayload) {
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    run_fpsan_unpack_canonical<__bf16, S>([](unsigned packed, float scale, unsigned *dO) {
      k_fpsan_unpack_bf16<S><<<1, 1>>>(packed, scale, dO);
    });
  });
}

template <Semantics S> __global__ void k_fpsan_fp4_roundtrip(unsigned packed, unsigned *out) {
  using VF = Value<float, S, kCC>;
  auto r = fpsan::amdgcn_cvt_scalef32_pk_f32_fp4<0, S, kCC>(packed, VF{1.0f});
  *out = fpsan::amdgcn_cvt_scalef32_pk_fp4_f32<0, S, kCC>(0u, r.get(0), r.get(1), VF{1.0f});
}

template <Semantics S> void run_fpsan_fp4_roundtrip() {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  unsigned *dO;
  HIP_CHECK(hipMalloc(&dO, sizeof(unsigned)));
  for (unsigned lo = 0; lo < 16; ++lo)
    for (unsigned hi = 0; hi < 16; ++hi) {
      const unsigned packed = lo | (hi << 4);
      k_fpsan_fp4_roundtrip<S><<<1, 1>>>(packed, dO);
      HIP_CHECK(hipDeviceSynchronize());
      const unsigned want_lo = fpsan_test::canonical_subbyte_code_from_residue<4, S>(
          fpsan_test::finite_subbyte_source_payload<4, S>(lo));
      const unsigned want_hi = fpsan_test::canonical_subbyte_code_from_residue<4, S>(
          fpsan_test::finite_subbyte_source_payload<4, S>(hi));
      EXPECT_EQ(from_dev(dO, 1)[0], want_lo | (want_hi << 4))
          << "lo=0x" << std::hex << lo << " hi=0x" << hi << std::dec;
    }
  (void)hipFree(dO);
}

TEST(CvtScalef32Fp4Wrap, FpsanFieldWithMulCastsRoundTripAllCodes) {
  run_fpsan_fp4_roundtrip<Semantics::FieldWithMulCasts>();
  run_fpsan_fp4_roundtrip<Semantics::FieldWithMulCasts2>();
}

// Pack plumbing + divide-by-scale: canonically narrow a/scale and b/scale into
// fp4 codes, place the selected pair, and preserve `old`.
template <Semantics S>
__global__ void k_fpsan_pack(unsigned old, float a, float b, float scale, unsigned *out) {
  using VF = Value<float, S, kCC>;
  *out = fpsan::amdgcn_cvt_scalef32_pk_fp4_f32<2, S, kCC>(old, VF{a}, VF{b}, VF{scale});
}

TEST(CvtScalef32Fp4Wrap, FpsanPackSelDivideAndPreserve) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VF = Value<float, S, kCC>;
    const unsigned old = 0xDEADBEEFu;
    std::mt19937 rng = fpsan_test::make_rng();
    unsigned *dO;
    HIP_CHECK(hipMalloc(&dO, sizeof(unsigned)));
    for (int t = 0; t < 32; ++t) {
      float a = fpsan_test::pick_int_valued<float>(rng, -6, 6);
      float b = fpsan_test::pick_int_valued<float>(rng, -6, 6);
      const float scale = 2.0f;
      k_fpsan_pack<S><<<1, 1>>>(old, a, b, scale, dO);
      HIP_CHECK(hipDeviceSynchronize());
      unsigned got = from_dev(dO, 1)[0];
      unsigned na = fpsan_test::canonical_subbyte_narrow_code<4>(VF{a} / VF{scale});
      unsigned nb = fpsan_test::canonical_subbyte_narrow_code<4>(VF{b} / VF{scale});
      unsigned pair = na | (nb << 4);
      unsigned mask = 0xFFu << (8 * 2);
      EXPECT_EQ(got, (old & ~mask) | (pair << (8 * 2))) << "a=" << a << " b=" << b;
    }
    const float special[] = {std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity()};
    for (float a : special) {
      const float b = 2.0f;
      const float scale = 2.0f;
      k_fpsan_pack<S><<<1, 1>>>(old, a, b, scale, dO);
      HIP_CHECK(hipDeviceSynchronize());
      unsigned got = from_dev(dO, 1)[0];
      unsigned na = fpsan_test::canonical_subbyte_narrow_code<4>(VF{a} / VF{scale});
      unsigned nb = fpsan_test::canonical_subbyte_narrow_code<4>(VF{b} / VF{scale});
      unsigned pair = na | (nb << 4);
      unsigned mask = 0xFFu << (8 * 2);
      EXPECT_EQ(got, (old & ~mask) | (pair << (8 * 2))) << "non-finite source pack";
    }
    (void)hipFree(dO);
  });
}

// sr_pk_fp4_f32: exact (seed-invariant) packing matches the deterministic pack.
template <Semantics S>
__global__ void k_fpsan_srpack(float a, float b, unsigned seed, unsigned *out) {
  using VF = Value<float, S, kCC>;
  *out = fpsan::amdgcn_cvt_scalef32_sr_pk_fp4_f32<0, S, kCC>(0u, VF{a}, VF{b}, seed, VF{1.0f});
}

TEST(CvtScalef32Fp4Wrap, FpsanSrPackMatchesDeterministic) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using VF = Value<float, S, kCC>;
    unsigned *dO;
    HIP_CHECK(hipMalloc(&dO, sizeof(unsigned)));
    for (int ca = 0; ca < 8; ++ca)
      for (int cb = 0; cb < 8; ++cb) {
        float a = narrow_to_f32(unsigned(ca), kFp4E2M1);
        float b = narrow_to_f32(unsigned(cb), kFp4E2M1);
        k_fpsan_srpack<S><<<1, 1>>>(a, b, 0xdeadbeefu, dO);
        HIP_CHECK(hipDeviceSynchronize());
        unsigned got = from_dev(dO, 1)[0] & 0xFF;
        unsigned na = fpsan_test::canonical_subbyte_narrow_code<4>(VF{a});
        unsigned nb = fpsan_test::canonical_subbyte_narrow_code<4>(VF{b});
        EXPECT_EQ(got, na | (nb << 4)) << "a=" << ca << " b=" << cb;
      }
    const float special[] = {std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity()};
    for (float a : special) {
      const float b = 1.0f;
      k_fpsan_srpack<S><<<1, 1>>>(a, b, 0xdeadbeefu, dO);
      HIP_CHECK(hipDeviceSynchronize());
      unsigned got = from_dev(dO, 1)[0] & 0xFF;
      unsigned na = fpsan_test::canonical_subbyte_narrow_code<4>(VF{a});
      unsigned nb = fpsan_test::canonical_subbyte_narrow_code<4>(VF{b});
      EXPECT_EQ(got, na | (nb << 4)) << "non-finite source sr pack";
    }
    (void)hipFree(dO);
  });
}

// f16/bf16 unpack/pack: Native mode vs OCP host ref (round-trip through the f32
// reference), confirming the wrappers compile and are hardware-correct.
__global__ void k_f16_unpack(unsigned packed, float *out) {
  auto r = fpsan::amdgcn_cvt_scalef32_pk_f16_fp4<0, Semantics::Native, kCC>(
      packed, Value<float, Semantics::Native, kCC>{1.0f});
  out[0] = static_cast<float>(static_cast<_Float16>(r.get(0)));
  out[1] = static_cast<float>(static_cast<_Float16>(r.get(1)));
}

TEST(CvtScalef32Fp4Wrap, FloatF16UnpackAllCodes) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  float *dO;
  HIP_CHECK(hipMalloc(&dO, 2 * sizeof(float)));
  for (int c = 0; c < 16; ++c) {
    unsigned packed = unsigned(c) | (unsigned(c) << 4);
    k_f16_unpack<<<1, 1>>>(packed, dO);
    HIP_CHECK(hipDeviceSynchronize());
    auto got = from_dev(dO, 2);
    float ref = narrow_to_f32(unsigned(c), kFp4E2M1); // exact in f16
    EXPECT_EQ(got[0], ref) << "f16 code " << c;
    EXPECT_EQ(got[1], ref) << "f16 code " << c;
  }
  (void)hipFree(dO);
}

#endif // has builtin
