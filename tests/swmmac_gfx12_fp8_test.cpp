// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "swmmac_gfx12_parts.hpp"

TEST(SwmmacGfx12, F32_FP8_FP8_LayoutAndFpsan) {
  auto wf = [](auto av, auto bv, auto cv, std::uint16_t i) {
    return fpsan::amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32<Semantics::Native, kCC>(av, bv, cv, i);
  };
  auto make_wp = [](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    return [](auto av, auto bv, auto cv, std::uint16_t i) {
      return fpsan::amdgcn_swmmac_f32_16x16x32_fp8_fp8_w32<S, kCC>(av, bv, cv, i);
    };
  };
  run_fp8_layout_and_fpsan<fpsan::fp8_e4m3, fpsan::fp8_e4m3>(0x400, wf, make_wp);
}
TEST(SwmmacGfx12, F32_FP8_BF8_LayoutAndFpsan) {
  auto wf = [](auto av, auto bv, auto cv, std::uint16_t i) {
    return fpsan::amdgcn_swmmac_f32_16x16x32_fp8_bf8_w32<Semantics::Native, kCC>(av, bv, cv, i);
  };
  auto make_wp = [](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    return [](auto av, auto bv, auto cv, std::uint16_t i) {
      return fpsan::amdgcn_swmmac_f32_16x16x32_fp8_bf8_w32<S, kCC>(av, bv, cv, i);
    };
  };
  run_fp8_layout_and_fpsan<fpsan::fp8_e4m3, fpsan::fp8_e5m2>(0x500, wf, make_wp);
}
