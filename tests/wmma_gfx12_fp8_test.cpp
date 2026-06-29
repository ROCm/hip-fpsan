// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "wmma_gfx12_parts.hpp"

using fpsan::v8e4m3_native;
using fpsan::v8e5m2_native;

struct WmmaF32Fp8Fp8 {
  using AVec = std::conditional_t<kWaveSize == 64, v4e4m3_native, v8e4m3_native>;
  using BVec = std::conditional_t<kWaveSize == 64, v4e4m3_native, v8e4m3_native>;
  using CVec = std::conditional_t<kWaveSize == 64, v4f_native, v8f_native>;
  // E4M3FN represents -3..3 exactly; sums of 16 products of |v|<=3*2 = 6 fit
  // exactly in f32, so the hardware-vs-dataflow comparison stays bit-exact.
  static constexpr int a_lo = -3, a_hi = 3;
  static constexpr int b_lo = -2, b_hi = 2;
  static constexpr int c_lo = -4, c_hi = 4;
  template <Semantics S, Conversions C>
  __device__ static Value<CVec, S, C> call(Value<AVec, S, C> a, Value<BVec, S, C> b,
                                           Value<CVec, S, C> c) {
#if defined(FPSAN_TEST_GFX12_WAVE64)
    return fpsan::amdgcn_wmma_f32_16x16x16_fp8_fp8_w64(a, b, c);
#else
    return fpsan::amdgcn_wmma_f32_16x16x16_fp8_fp8_w32(a, b, c);
#endif
  }
};

struct WmmaF32Fp8Bf8 {
  using AVec = std::conditional_t<kWaveSize == 64, v4e4m3_native, v8e4m3_native>;
  using BVec = std::conditional_t<kWaveSize == 64, v4e5m2_native, v8e5m2_native>;
  using CVec = std::conditional_t<kWaveSize == 64, v4f_native, v8f_native>;
  static constexpr int a_lo = -3, a_hi = 3;
  static constexpr int b_lo = -2, b_hi = 2;
  static constexpr int c_lo = -4, c_hi = 4;
  template <Semantics S, Conversions C>
  __device__ static Value<CVec, S, C> call(Value<AVec, S, C> a, Value<BVec, S, C> b,
                                           Value<CVec, S, C> c) {
#if defined(FPSAN_TEST_GFX12_WAVE64)
    return fpsan::amdgcn_wmma_f32_16x16x16_fp8_bf8_w64(a, b, c);
#else
    return fpsan::amdgcn_wmma_f32_16x16x16_fp8_bf8_w32(a, b, c);
#endif
  }
};

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

FPSAN_WMMA_GFX12_TESTS(WmmaF32Fp8Fp8)

#undef FPSAN_WMMA_GFX12_TESTS
#undef FPSAN_WMMA_GFX12_SUITE
#undef FPSAN_WMMA_GFX12_CONCAT
#undef FPSAN_WMMA_GFX12_CONCAT2
