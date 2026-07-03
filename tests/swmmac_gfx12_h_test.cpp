// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "swmmac_gfx12_parts.hpp"

TEST(SwmmacGfx12, F32_F16_LayoutMatchesHardware) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  SwData d = make_sw_data(0x100, encode_idx_h);
  std::vector<float> ref = sw_reference_h<float>(d);
  float *dA = to_dev(d.A), *dB = to_dev(d.B), *dC = to_dev(d.C);
  int *dp0 = to_dev(d.p0), *dp1 = to_dev(d.p1);
  std::uint16_t *dI = to_dev(d.idx);
  float *dD = nullptr;
  HIP_CHECK(hipMalloc(&dD, kSwM * kSwN * sizeof(float)));
  k_swmmac_f16_float<<<1, WAVE>>>(dA, dB, dC, dp0, dp1, dI, dD);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> got(kSwM * kSwN);
  HIP_CHECK(hipMemcpy(got.data(), dD, kSwM * kSwN * sizeof(float), hipMemcpyDeviceToHost));
  for (int t = 0; t < kSwM * kSwN; ++t)
    EXPECT_EQ(bits_of(got[t]), bits_of(ref[t])) << "at " << t;
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dp0);
  (void)hipFree(dp1);
  (void)hipFree(dI);
  (void)hipFree(dD);
}

TEST(SwmmacGfx12, F32_F16_FpsanMatchesScalarReference) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  SwData d = make_sw_data(0x100, encode_idx_h);
  float *dA = to_dev(d.A), *dB = to_dev(d.B), *dC = to_dev(d.C);
  int *dp0 = to_dev(d.p0), *dp1 = to_dev(d.p1);
  std::uint16_t *dI = to_dev(d.idx);
  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    std::vector<std::uint64_t> ref = sw_reference_fpsan_h<S, _Float16, _Float16, float>(d);
    std::uint32_t *dD = nullptr;
    HIP_CHECK(hipMalloc(&dD, kSwM * kSwN * sizeof(std::uint32_t)));
    k_swmmac_f16_fpsan<S><<<1, WAVE>>>(dA, dB, dC, dp0, dp1, dI, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint32_t> got(kSwM * kSwN);
    HIP_CHECK(
        hipMemcpy(got.data(), dD, kSwM * kSwN * sizeof(std::uint32_t), hipMemcpyDeviceToHost));
    for (int t = 0; t < kSwM * kSwN; ++t)
      EXPECT_EQ(static_cast<std::uint64_t>(got[t]), ref[t]) << "at " << t;
    (void)hipFree(dD);
  });
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dC);
  (void)hipFree(dp0);
  (void)hipFree(dp1);
  (void)hipFree(dI);
}
