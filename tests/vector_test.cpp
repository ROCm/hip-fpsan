// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/vector_test.cpp
//
// Value over Clang vector element types. The key checks tie the vector path to
// the already-ground-truthed scalar path: a vector Value must embed and compute
// lane-for-lane identically to the corresponding scalar Values. Pure C++/Clang
// (no GPU).
#include "fpsan/fpsan.hpp"

#include "fpsan_semantics.hpp"
#include "test_utils.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>

using fpsan::Conversions;
using fpsan::Semantics;
using fpsan::Value;

template <class T, int N> using vec = T __attribute__((ext_vector_type(N)));
using f4 = vec<float, 4>;
using d2 = vec<double, 2>;

// ---- a vector Value embeds lane-for-lane like scalar Values ----------------
TEST(Vector, EmbedMatchesScalarPerLane) {
  fpsan_test::for_each_fpsan_semantics([](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using Vf = Value<f4, S, Conversions::Explicit>;
    using Sf = Value<float, S, Conversions::Explicit>;
    f4 v = {1.5f, -2.0f, 0.0f, 3.14159f};
    Vf a(v);
    auto pay = a.fpsan_payload(); // uint32 x4
    for (int i = 0; i < 4; ++i)
      EXPECT_EQ(pay[i], Sf(v[i]).fpsan_payload()) << "lane " << i;
    // Unembedding the vector recovers, lane-for-lane, what unembedding the
    // scalar Value does. Only the bit-faithful flavors define to_float()
    // (Native/Triton); algebraic has no value-unembed, so its lane-vs-scalar
    // equivalence is the payload check above.
    if constexpr (!Vf::is_algebraic) {
      f4 back = static_cast<f4>(a);
      for (int i = 0; i < 4; ++i)
        EXPECT_EQ(bits_of(back[i]), bits_of(static_cast<float>(Sf(v[i])))) << "lane " << i;
    }
  });
}

// ---- ring arithmetic is lane-wise identical to scalar Values ---------------
TEST(Vector, FpsanArithmeticMatchesScalarPerLane) {
  fpsan_test::for_each_fpsan_semantics([](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using Vf = Value<f4, S, Conversions::Explicit>;
    using Sf = Value<float, S, Conversions::Explicit>;
    f4 x = {1.f, 2.f, -3.f, 0.5f}, y = {4.f, -5.f, 6.f, 7.f};
    Vf a(x), b(y);
    auto add = (a + b).fpsan_payload();
    auto sub = (a - b).fpsan_payload();
    auto mul = (a * b).fpsan_payload();
    auto neg = (-a).fpsan_payload();
    for (int i = 0; i < 4; ++i) {
      EXPECT_EQ(add[i], (Sf(x[i]) + Sf(y[i])).fpsan_payload());
      EXPECT_EQ(sub[i], (Sf(x[i]) - Sf(y[i])).fpsan_payload());
      EXPECT_EQ(mul[i], (Sf(x[i]) * Sf(y[i])).fpsan_payload());
      EXPECT_EQ(neg[i], (-Sf(x[i])).fpsan_payload());
    }
  });
}

// ---- FPSan ring laws hold lane-wise (exact) --------------------------------
TEST(Vector, FpsanRingLawsExact) {
  fpsan_test::for_each_fpsan_semantics([](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using Vf = Value<f4, S, Conversions::Explicit>;
    f4 xa = {1e8f, 1.f, -2.f, 3.f}, xb = {-1e8f, 2.f, 4.f, -5.f}, xc = {1.f, 3.f, -6.f, 7.f};
    Vf a(xa), b(xb), c(xc);
    auto all_true = [](auto mask) {
      for (int i = 0; i < 4; ++i)
        if (!mask[i])
          return false;
      return true;
    };
    EXPECT_TRUE(all_true((a + b) + c == a + (b + c))); // associativity, exact
    EXPECT_TRUE(all_true((a * b) * c == a * (b * c)));
    EXPECT_TRUE(all_true(a * (b + c) == a * b + a * c)); // distributivity
  });
}

// ---- mode=false vector Value is bit-exact native vector arithmetic ---------
TEST(Vector, ModeFalseMatchesNative) {
  using Vn = Value<f4, Semantics::Native, Conversions::Explicit>;
  f4 x = {1.1f, 2.2f, -3.3f, 4.4f}, y = {0.5f, -0.25f, 8.f, 0.1f};
  f4 add = static_cast<f4>(Vn(x) + Vn(y));
  f4 mul = static_cast<f4>(Vn(x) * Vn(y));
  f4 nx = x + y, ny = x * y;
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(bits_of(add[i]), bits_of(nx[i]));
    EXPECT_EQ(bits_of(mul[i]), bits_of(ny[i]));
  }
}

// ---- comparisons yield a per-lane mask (both modes) ------------------------
TEST(Vector, ComparisonMasks) {
  using Vn = Value<f4, Semantics::Native, Conversions::Explicit>;
  f4 x = {1.f, 5.f, 3.f, 4.f}, y = {2.f, 2.f, 3.f, 1.f};
  auto lt = (Vn(x) < Vn(y)); // native float order
  auto eq = (Vn(x) == Vn(y));
  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(static_cast<bool>(lt[i]), x[i] < y[i]);
    EXPECT_EQ(static_cast<bool>(eq[i]), x[i] == y[i]);
  }
  // FPSan mode: == is exact payload equality, lane-wise.
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using Vf = Value<f4, S, Conversions::Explicit>;
    auto feq = (Vf(x) == Vf(x));
    for (int i = 0; i < 4; ++i)
      EXPECT_TRUE(static_cast<bool>(feq[i]));
  });
}

template <Semantics S> void expect_math_helpers_match_scalar_per_lane() {
  using Vf = Value<f4, S, Conversions::Explicit>;
  using Sf = Value<float, S, Conversions::Explicit>;
  f4 x = {-4.f, -0.5f, 2.f, 3.25f};
  f4 y = {1.f, -7.f, 0.5f, -1.25f};
  Vf a(x), b(y);
  const Vf abs_a = fpsan::abs(a);
  const Vf fabs_a = fpsan::fabs(a);
  const Vf min_ab = fpsan::min(a, b);
  const Vf max_ab = fpsan::max(a, b);
  const Vf fmin_ab = fpsan::fmin(a, b);
  const Vf fmax_ab = fpsan::fmax(a, b);
  for (int i = 0; i < 4; ++i) {
    const Sf sx(x[i]), sy(y[i]);
    EXPECT_TRUE(abs_a.get(i) == fpsan::abs(sx)) << "lane " << i << " S=" << int(S);
    EXPECT_TRUE(fabs_a.get(i) == fpsan::fabs(sx)) << "lane " << i << " S=" << int(S);
    EXPECT_TRUE(min_ab.get(i) == fpsan::min(sx, sy)) << "lane " << i << " S=" << int(S);
    EXPECT_TRUE(max_ab.get(i) == fpsan::max(sx, sy)) << "lane " << i << " S=" << int(S);
    EXPECT_TRUE(fmin_ab.get(i) == fpsan::fmin(sx, sy)) << "lane " << i << " S=" << int(S);
    EXPECT_TRUE(fmax_ab.get(i) == fpsan::fmax(sx, sy)) << "lane " << i << " S=" << int(S);
  }
}

TEST(Vector, MathHelpersMatchScalarPerLane) {
  expect_math_helpers_match_scalar_per_lane<Semantics::Native>();
  fpsan_test::for_each_fpsan_semantics_all_variants(
      [](auto sem) { expect_math_helpers_match_scalar_per_lane<decltype(sem)::value>(); });
}

// ---- lane accessors get(i)/set(i) ------------------------------------------
TEST(Vector, GetSetLanes) {
  fpsan_test::for_each_fpsan_semantics([](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using Vf = Value<f4, S, Conversions::Explicit>;
    using Sf = Value<float, S, Conversions::Explicit>;
    Vf a{};
    for (int i = 0; i < 4; ++i)
      a.set(i, Sf(float(i) + 0.5f));
    for (int i = 0; i < 4; ++i)
      EXPECT_TRUE(a.get(i) == Sf(float(i) + 0.5f));
  });
}

// ---- double2 (64-bit lanes) sanity -----------------------------------------
TEST(Vector, Double2EmbedMatchesScalar) {
  fpsan_test::for_each_fpsan_semantics([](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using Vd = Value<d2, S, Conversions::Explicit>;
    using Sd = Value<double, S, Conversions::Explicit>;
    d2 v = {3.25, -7.5};
    auto pay = Vd(v).fpsan_payload(); // uint64 x2
    for (int i = 0; i < 2; ++i)
      EXPECT_EQ(pay[i], Sd(v[i]).fpsan_payload());
  });
}

#if FPSAN_HAS_FLOAT16
using h8 = vec<_Float16, 8>;
TEST(Vector, Half8EmbedMatchesScalar) {
  fpsan_test::for_each_fpsan_semantics([](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    using Vh = Value<h8, S, Conversions::Explicit>;
    using Sh = Value<_Float16, S, Conversions::Explicit>;
    h8 v;
    for (int i = 0; i < 8; ++i)
      v[i] = static_cast<_Float16>(i - 3) * static_cast<_Float16>(0.5f);
    auto pay = Vh(v).fpsan_payload(); // uint16 x8
    for (int i = 0; i < 8; ++i)
      EXPECT_EQ(pay[i], Sh(v[i]).fpsan_payload()) << "lane " << i;
  });
}
#endif
