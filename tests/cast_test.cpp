// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/cast_test.cpp
//
// fpsan::cast<ToFT> between scalar Values. Pure C++/Clang.
#include "fpsan/fpsan.hpp"

#include "fpsan_semantics.hpp"
#include "test_utils.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

using fpsan::cast;
using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

#if FPSAN_HAS_FLOAT16
// FPSan f16<->f32 cast matches Triton: embed -> signed-resize payload ->
// unembed.
TEST(Cast, FpsanF16ToF32FixedPoints) {
  fpsan_test::for_each_fpsan_semantics_all_variants([](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using H = Value<_Float16, S, Conversions::Explicit>;
    using F32 = Value<float, S, Conversions::Explicit>;
    using B = typename F32::bits_type;
    // 0 and 1 cross widths unchanged under any homomorphic (or Triton) cast.
    EXPECT_EQ(cast<float>(H(_Float16(0))).fpsan_payload(), B(0)) << "S=" << int(S);
    EXPECT_EQ(cast<float>(H(_Float16(1))).fpsan_payload(), B(1)) << "S=" << int(S);
    // -1 maps to the f32-width residue of -1 where the cast is value-faithful
    // (Triton's all-ones resize; an algebraic field's n-1). The composite
    // two-moduli rings carry no cast homomorphism (scorecard), so -1 is not
    // asserted there.
    if constexpr (fpsan_test::flavor_has_cast_homomorphism<float, S>()) {
      if constexpr (S == Semantics::Triton)
        EXPECT_EQ(cast<float>(H(_Float16(-1))).fpsan_payload(), B(0xFFFFFFFFu));
      else
        EXPECT_EQ(cast<float>(H(_Float16(-1))).fpsan_payload(),
                  static_cast<B>(F32::alg_cfg().n - 1));
    }
  });
}

TEST(Cast, FpsanF32ToF16ToF32RoundTrips) {
  // f16 -> f32 -> f16 recovers the payload wherever the cast is a homomorphism:
  // Triton's signed resize satisfies truncate(sign_extend(p)) == p, and a
  // single-prime field's cast tower is invertible. The composite two-moduli
  // rings carry no cast homomorphism, so the round trip is not asserted there.
  fpsan_test::for_each_fpsan_semantics_all_variants([](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    if constexpr (fpsan_test::flavor_has_cast_homomorphism<_Float16, S>()) {
      using H = Value<_Float16, S, Conversions::Explicit>;
      for (int i = 0; i < (1 << 16); ++i) {
        _Float16 v;
        uint16_t b = static_cast<uint16_t>(i);
        std::memcpy(&v, &b, sizeof v);
        H h(v);
        auto back = cast<_Float16>(cast<float>(h));
        ASSERT_EQ(back.fpsan_payload(), h.fpsan_payload())
            << "S=" << int(S) << " bits 0x" << std::hex << i;
      }
    }
  });
}

TEST(Cast, FloatModeIsNativeConversion) {
  using H = Value<_Float16, Semantics::Native, Conversions::Explicit>;
  using F = Value<float, Semantics::Native, Conversions::Explicit>;
  for (float x : {0.0f, 1.5f, -2.25f, 100.0f, 0.1f}) {
    F f(x);
    EXPECT_EQ(bits_of(static_cast<_Float16>(cast<_Float16>(f))), bits_of(static_cast<_Float16>(x)));
    H h(static_cast<_Float16>(x));
    EXPECT_EQ(bits_of(static_cast<float>(cast<float>(h))),
              bits_of(static_cast<float>(static_cast<_Float16>(x))));
  }
}
#endif

#if FPSAN_HAS_FLOAT16 && FPSAN_HAS_BF16
TEST(Cast, FieldWithMulCastsDistinguishesF16AndBf16) {
  using H = Value<_Float16, Semantics::FieldWithMulCasts, Conversions::Explicit>;
  using B = Value<__bf16, Semantics::FieldWithMulCasts, Conversions::Explicit>;
  using F32 = Value<float, Semantics::FieldWithMulCasts, Conversions::Explicit>;

  H h{_Float16(1.1f)};

  const B direct = cast<__bf16>(h);
  const B via32 = cast<__bf16>(cast<float>(h));
  EXPECT_EQ(via32.fpsan_payload(), direct.fpsan_payload());
  EXPECT_NE(direct.fpsan_payload(), static_cast<typename B::bits_type>(h.fpsan_payload()));
  EXPECT_EQ(cast<_Float16>(direct).fpsan_payload(), h.fpsan_payload());

  // This is the precision-loss case that same-width identity would miss:
  // the FieldWithMulCasts chain is observable, while a real rounded bf16 value
  // embeds as a different residue.
  const B rounded{static_cast<__bf16>(static_cast<float>(_Float16(1.1f)))};
  EXPECT_NE(via32.fpsan_payload(), rounded.fpsan_payload());

  // Same-format same-width casts remain the cheap identity.
  EXPECT_EQ(cast<float>(F32{1.25f}).fpsan_payload(), F32{1.25f}.fpsan_payload());
}
#endif

TEST(Cast, FieldWithMulCastsDistinguishesFp8Formats) {
  using E4 = Value<fpsan::fp8_e4m3, Semantics::FieldWithMulCasts, Conversions::Explicit>;
  using E5 = Value<fpsan::fp8_e5m2, Semantics::FieldWithMulCasts, Conversions::Explicit>;

  E4 a{fpsan::fp8_e4m3(1.5f)};
  E4 b{fpsan::fp8_e4m3(2.0f)};

  const E5 direct = cast<fpsan::fp8_e5m2>(a);
  const E5 via32 = cast<fpsan::fp8_e5m2>(cast<float>(a));
  EXPECT_EQ(via32.fpsan_payload(), direct.fpsan_payload());
  EXPECT_NE(direct.fpsan_payload(), static_cast<typename E5::bits_type>(a.fpsan_payload()));
  EXPECT_EQ(cast<fpsan::fp8_e4m3>(direct).fpsan_payload(), a.fpsan_payload());

  EXPECT_EQ(cast<fpsan::fp8_e5m2>(a * b).fpsan_payload(),
            (cast<fpsan::fp8_e5m2>(a) * cast<fpsan::fp8_e5m2>(b)).fpsan_payload());
  EXPECT_EQ(cast<fpsan::fp8_e4m3>(a).fpsan_payload(), a.fpsan_payload());
}
