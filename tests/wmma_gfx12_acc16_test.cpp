// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "wmma_gfx12_parts.hpp"

struct WmmaF32F16 {
  using AVec = std::conditional_t<kWaveSize == 64, v4h_native, v8h_native>;
  using BVec = std::conditional_t<kWaveSize == 64, v4h_native, v8h_native>;
  using CVec = std::conditional_t<kWaveSize == 64, v4f_native, v8f_native>;
  static constexpr int a_lo = -3, a_hi = 3;
  static constexpr int b_lo = -2, b_hi = 2;
  static constexpr int c_lo = -4, c_hi = 4;
  template <Semantics S, Conversions C>
  __device__ static Value<CVec, S, C> call(Value<AVec, S, C> a, Value<BVec, S, C> b,
                                           Value<CVec, S, C> c) {
#if defined(FPSAN_TEST_GFX12_WAVE64)
    return fpsan::amdgcn_wmma_f32_16x16x16_f16_w64(a, b, c);
#else
    return fpsan::amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#endif
  }
};

struct WmmaF16F16 {
  using AVec = std::conditional_t<kWaveSize == 64, v4h_native, v8h_native>;
  using BVec = std::conditional_t<kWaveSize == 64, v4h_native, v8h_native>;
  using CVec = std::conditional_t<kWaveSize == 64, v4h_native, v8h_native>;
  // f16 stays exact for sums up to a few thousand; keep ranges modest.
  static constexpr int a_lo = -3, a_hi = 3;
  static constexpr int b_lo = -2, b_hi = 2;
  static constexpr int c_lo = -4, c_hi = 4;
  template <Semantics S, Conversions C>
  __device__ static Value<CVec, S, C> call(Value<AVec, S, C> a, Value<BVec, S, C> b,
                                           Value<CVec, S, C> c) {
#if defined(FPSAN_TEST_GFX12_WAVE64)
    return fpsan::amdgcn_wmma_f16_16x16x16_f16_w64(a, b, c);
#else
    return fpsan::amdgcn_wmma_f16_16x16x16_f16_w32(a, b, c);
#endif
  }
};

struct WmmaF32BF16 {
  using AVec = std::conditional_t<kWaveSize == 64, v4bf_native, v8bf_native>;
  using BVec = std::conditional_t<kWaveSize == 64, v4bf_native, v8bf_native>;
  using CVec = std::conditional_t<kWaveSize == 64, v4f_native, v8f_native>;
  // bf16 has only 7 mantissa bits; small ints fit exactly.
  static constexpr int a_lo = -3, a_hi = 3;
  static constexpr int b_lo = -2, b_hi = 2;
  static constexpr int c_lo = -4, c_hi = 4;
  template <Semantics S, Conversions C>
  __device__ static Value<CVec, S, C> call(Value<AVec, S, C> a, Value<BVec, S, C> b,
                                           Value<CVec, S, C> c) {
#if defined(FPSAN_TEST_GFX12_WAVE64)
    return fpsan::amdgcn_wmma_f32_16x16x16_bf16_w64(a, b, c);
#else
    return fpsan::amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
#endif
  }
};

struct WmmaBF16BF16 {
  using AVec = std::conditional_t<kWaveSize == 64, v4bf_native, v8bf_native>;
  using BVec = std::conditional_t<kWaveSize == 64, v4bf_native, v8bf_native>;
  using CVec = std::conditional_t<kWaveSize == 64, v4bf_native, v8bf_native>;
  // bf16 accumulation: keep magnitudes small so length-16 sums stay exact.
  static constexpr int a_lo = -3, a_hi = 3;
  static constexpr int b_lo = -2, b_hi = 2;
  static constexpr int c_lo = -4, c_hi = 4;
  template <Semantics S, Conversions C>
  __device__ static Value<CVec, S, C> call(Value<AVec, S, C> a, Value<BVec, S, C> b,
                                           Value<CVec, S, C> c) {
#if defined(FPSAN_TEST_GFX12_WAVE64)
    return fpsan::amdgcn_wmma_bf16_16x16x16_bf16_w64(a, b, c);
#else
    return fpsan::amdgcn_wmma_bf16_16x16x16_bf16_w32(a, b, c);
#endif
  }
};

// ---- fp8 variants (AMD's naming: 'fp8' = OCP E4M3FN, 'bf8' = OCP E5M2) ------
// Per-lane fragment is 8 fp8s = 8 bytes = the v2i32 the builtin actually wants.

using fpsan::v8e4m3_native;
using fpsan::v8e5m2_native;

#define FPSAN_WMMA_GFX12_CONCAT2(A, B) A##B
#define FPSAN_WMMA_GFX12_CONCAT(A, B) FPSAN_WMMA_GFX12_CONCAT2(A, B)
#if defined(FPSAN_TEST_GFX12_WAVE64)
#define FPSAN_WMMA_GFX12_SUITE(NAME) FPSAN_WMMA_GFX12_CONCAT(NAME, W64)
#else
#define FPSAN_WMMA_GFX12_SUITE(NAME) FPSAN_WMMA_GFX12_CONCAT(NAME, W32)
#endif

#define FPSAN_WMMA_GFX12_TESTS(NAME)                                                               \
  TEST(FPSAN_WMMA_GFX12_SUITE(NAME), LayoutMatchesHardware) {                                      \
    run_layout_matches_hardware<NAME>();                                                           \
  }                                                                                                \
  TEST(FPSAN_WMMA_GFX12_SUITE(NAME), FpsanMatchesScalarReference) {                                \
    run_fpsan_matches_scalar_reference_all<NAME>();                                                \
  }

FPSAN_WMMA_GFX12_TESTS(WmmaF16F16)
FPSAN_WMMA_GFX12_TESTS(WmmaBF16BF16)
#undef FPSAN_WMMA_GFX12_TESTS
#undef FPSAN_WMMA_GFX12_SUITE
#undef FPSAN_WMMA_GFX12_CONCAT
#undef FPSAN_WMMA_GFX12_CONCAT2
