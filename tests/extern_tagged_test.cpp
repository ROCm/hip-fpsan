// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/extern_tagged_test.cpp
//
// The generic extern/libdevice fallback fpsan::extern_tagged: the properties an
// opaque (unmodeled) call can honor -- determinism, symbol-distinctness,
// input-distinctness, argument-order sensitivity, NaN propagation (algebraic),
// and lane-wise consistency. Plus Triton parity: the symbol hash matches FNV-1a
// (Triton's stableStringHash) on standard vectors, and the operand-mixing
// formula matches Triton's fpsanVariadicExternTagged.
#include "fpsan/fpsan.hpp"

#include "fpsan_semantics.hpp"

#include <gtest/gtest.h>

#include <cstdint>

using fpsan::extern_tagged;
using fpsan::Value;
using fpsan::detail::stable_string_hash;
using Conv = fpsan::Conversions;

template <fpsan::Semantics S> using F = Value<float, S, Conv::Explicit>;
using Scr = F<fpsan::Semantics::Triton>;
using Alg = F<fpsan::Semantics::Field>;

template <class T, int N> using vec = T __attribute__((ext_vector_type(N)));
using f4 = vec<float, 4>;

// ---- Triton parity: the symbol hash IS FNV-1a-64 (== stableStringHash) ------
TEST(ExternTagged, FnvParityStandardVectors) {
  EXPECT_EQ(stable_string_hash(""), 14695981039346656037ull); // offset basis
  EXPECT_EQ(stable_string_hash("a"), 0xaf63dc4c8601ec8cull);
  EXPECT_EQ(stable_string_hash("foobar"), 0x85944171f73967e8ull);
  EXPECT_NE(stable_string_hash("__ocml_j0_f32"), stable_string_hash("__ocml_j1_f32"));
}

// Determinism, symbol-/input-distinctness, and argument-order sensitivity are
// properties an opaque tagged op honors in EVERY flavor, so loop them all.
TEST(ExternTagged, Deterministic) {
  const std::uint64_t H = stable_string_hash("__ocml_mystery_f32");
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    using FS = F<decltype(sem)::value>;
    EXPECT_TRUE(extern_tagged(H, FS{3.5f}) == extern_tagged(H, FS{3.5f}));
  });
}

TEST(ExternTagged, SymbolDistinct) {
  const std::uint64_t H1 = stable_string_hash("__ocml_mystery_f32");
  const std::uint64_t H2 = stable_string_hash("__ocml_other_f32");
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    using FS = F<decltype(sem)::value>;
    EXPECT_TRUE(extern_tagged(H1, FS{3.5f}) != extern_tagged(H2, FS{3.5f}));
  });
}

TEST(ExternTagged, InputDistinct) {
  const std::uint64_t H = stable_string_hash("__ocml_mystery_f32");
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    using FS = F<decltype(sem)::value>;
    EXPECT_TRUE(extern_tagged(H, FS{3.5f}) != extern_tagged(H, FS{3.6f}));
  });
}

// f(a,b) != f(b,a): the rotate-by-arg-index mixing makes operand order matter.
TEST(ExternTagged, ArgumentOrderSensitive) {
  const std::uint64_t H = stable_string_hash("__ocml_atan2_f32");
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    using FS = F<decltype(sem)::value>;
    EXPECT_TRUE(extern_tagged(H, FS{1.0f}, FS{2.0f}) != extern_tagged(H, FS{2.0f}, FS{1.0f}));
  });
}

// ---- Triton parity: the single-operand tag is exactly payload XOR nameHash --
TEST(ExternTagged, TritonSingleArgFormula) {
  const std::uint64_t H = stable_string_hash("__ocml_mystery_f32");
  Scr x{3.5f};
  const std::uint32_t expect = x.fpsan_payload() ^ static_cast<std::uint32_t>(H);
  EXPECT_EQ(extern_tagged(H, x).fpsan_payload(), expect);
}

// ---- Triton parity: two operands mix as (p0 + rotl(p1,1)) XOR nameHash ------
TEST(ExternTagged, TritonTwoArgFormula) {
  const std::uint64_t H = stable_string_hash("__ocml_atan2_f32");
  Scr a{1.25f}, b{6.0f};
  const std::uint32_t p0 = a.fpsan_payload();
  const std::uint32_t p1 = b.fpsan_payload();
  const std::uint32_t rot = (p1 << 1) | (p1 >> 31); // rotl by arg index 1, w=32
  const std::uint32_t expect = (p0 + rot) ^ static_cast<std::uint32_t>(H);
  EXPECT_EQ(extern_tagged(H, a, b).fpsan_payload(), expect);
}

// ---- algebraic: a non-finite operand poisons the tagged result to NaN -----
// Every algebraic flavor carries a projective pole and a nan_code; Triton has no
// NaN poisoning (a hard limitation), so this is asserted on the algebraic flavors.
TEST(ExternTagged, AlgebraicNaNPropagates) {
  const std::uint64_t H = stable_string_hash("__ocml_mystery_f32");
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    constexpr fpsan::Semantics S = decltype(sem)::value;
    if constexpr (fpsan::detail::is_algebraic_semantics(S)) {
      using FS = F<S>;
      FS inf = FS{1.0f} / FS{0.0f}; // the projective pole
      EXPECT_EQ(extern_tagged(H, inf).fpsan_payload(), FS::alg_cfg().nan_code);
      EXPECT_EQ(extern_tagged(H, FS{2.0f}, inf).fpsan_payload(), FS::alg_cfg().nan_code);
    }
  });
}

// ---- vector path is lane-wise identical to the scalar tag -------------------
TEST(ExternTagged, VectorLanewise) {
  const std::uint64_t H = stable_string_hash("__ocml_mystery_f32");
  f4 v = {1.5f, -2.0f, 0.25f, 3.14159f};
  fpsan_test::for_each_fpsan_semantics([&](auto sem) {
    constexpr fpsan::Semantics S = decltype(sem)::value;
    using Vf = Value<f4, S, Conv::Explicit>;
    using Sc = F<S>;
    Vf a(v);
    auto tag = extern_tagged(H, a).fpsan_payload();
    for (int i = 0; i < 4; ++i)
      EXPECT_EQ(tag[i], extern_tagged(H, Sc{v[i]}).fpsan_payload())
          << "S=" << int(S) << " lane " << i;
  });
}
