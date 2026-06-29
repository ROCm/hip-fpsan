// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "wmma_gfx11_parts.hpp"

template <bool Opsel, bool Tied = false> struct WmmaGfx11F16F16 {
  using AVec = v16h_wmma_native;
  using BVec = v16h_wmma_native;
  using CVec = std::conditional_t<kWaveSize == 64, v8h_native, v16h_wmma_native>;
  static constexpr bool has_opsel = true;
  static constexpr bool opsel = Opsel;
  static constexpr int wave_size = kWaveSize;
  static constexpr int a_lo = -3, a_hi = 3;
  static constexpr int b_lo = -2, b_hi = 2;
  static constexpr int c_lo = -4, c_hi = 4;
  template <Semantics S, Conversions Conv>
  __device__ static Value<CVec, S, Conv> call(Value<AVec, S, Conv> a, Value<BVec, S, Conv> b,
                                              Value<CVec, S, Conv> c) {
#if defined(FPSAN_TEST_GFX11_WAVE64)
    if constexpr (Tied)
      return fpsan::amdgcn_wmma_f16_16x16x16_f16_tied_w64<Opsel, S, Conv>(a, b, c);
    else
      return fpsan::amdgcn_wmma_f16_16x16x16_f16_w64<Opsel, S, Conv>(a, b, c);
#else
    if constexpr (Tied)
      return fpsan::amdgcn_wmma_f16_16x16x16_f16_tied_w32<Opsel, S, Conv>(a, b, c);
    else
      return fpsan::amdgcn_wmma_f16_16x16x16_f16_w32<Opsel, S, Conv>(a, b, c);
#endif
  }
};

using WmmaGfx11F16F16Tied0 = WmmaGfx11F16F16<false, true>;
using WmmaGfx11F16F16Tied1 = WmmaGfx11F16F16<true, true>;

#define FPSAN_WMMA_GFX11_CONCAT2(A, B) A##B
#define FPSAN_WMMA_GFX11_CONCAT(A, B) FPSAN_WMMA_GFX11_CONCAT2(A, B)
#if defined(FPSAN_TEST_GFX11_WAVE64)
#define FPSAN_WMMA_GFX11_SUITE(NAME) FPSAN_WMMA_GFX11_CONCAT(NAME, W64)
#else
#define FPSAN_WMMA_GFX11_SUITE(NAME) FPSAN_WMMA_GFX11_CONCAT(NAME, W32)
#endif

#define FPSAN_WMMA_GFX11_TESTS(NAME, TRAITS)                                                       \
  TEST(FPSAN_WMMA_GFX11_SUITE(NAME), LayoutMatchesHardware) {                                      \
    run_layout_matches_hardware<TRAITS>();                                                         \
  }                                                                                                \
  TEST(FPSAN_WMMA_GFX11_SUITE(NAME), FpsanMatchesScalarReference) {                                \
    run_fpsan_matches_scalar_reference<TRAITS>();                                                  \
  }

#define FPSAN_WMMA_GFX11_TIED_PRESERVE_TEST(NAME, TRAITS)                                          \
  TEST(FPSAN_WMMA_GFX11_SUITE(NAME), PreservesUnselectedHalf) {                                    \
    run_tied_preserves_unselected<TRAITS>();                                                       \
  }

FPSAN_WMMA_GFX11_TESTS(WmmaGfx11F16F16TiedOpsel0, WmmaGfx11F16F16Tied0)
FPSAN_WMMA_GFX11_TESTS(WmmaGfx11F16F16TiedOpsel1, WmmaGfx11F16F16Tied1)
FPSAN_WMMA_GFX11_TIED_PRESERVE_TEST(WmmaGfx11F16F16TiedOpsel0, WmmaGfx11F16F16Tied0)
FPSAN_WMMA_GFX11_TIED_PRESERVE_TEST(WmmaGfx11F16F16TiedOpsel1, WmmaGfx11F16F16Tied1)

#undef FPSAN_WMMA_GFX11_TIED_PRESERVE_TEST
#undef FPSAN_WMMA_GFX11_TESTS
#undef FPSAN_WMMA_GFX11_SUITE
#undef FPSAN_WMMA_GFX11_CONCAT
#undef FPSAN_WMMA_GFX11_CONCAT2
