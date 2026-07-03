// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "swmmac_gfx12_parts.hpp"

__global__ void k_swmmac_f16h_float(const float *A, const float *B, const float *C, const int *p0,
                                    const int *p1, const std::uint16_t *idx, _Float16 *D) {
  auto fn = [](auto av, auto bv, auto cv, std::uint16_t i) {
    return fpsan::amdgcn_swmmac_f16_16x16x32_f16_w32<Semantics::Native, kCC>(av, bv, cv, i);
  };
  swmmac_h_dataflow<Semantics::Native, _Float16, _Float16, _Float16, _Float16>(A, B, C, p0, p1, idx,
                                                                               D, fn);
}

template <Semantics S>
__global__ void k_swmmac_f16h_fpsan(const float *A, const float *B, const float *C, const int *p0,
                                    const int *p1, const std::uint16_t *idx, std::uint16_t *D) {
  auto fn = [](auto av, auto bv, auto cv, std::uint16_t i) {
    return fpsan::amdgcn_swmmac_f16_16x16x32_f16_w32<S, kCC>(av, bv, cv, i);
  };
  swmmac_h_dataflow<S, _Float16, _Float16, _Float16, std::uint16_t>(A, B, C, p0, p1, idx, D, fn);
}

TEST(SwmmacGfx12, F16_F16_LayoutMatchesHardware) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  SwData d = make_sw_data(0x300, encode_idx_h);
  std::vector<_Float16> ref = sw_reference_h<_Float16>(d);
  float *dA = to_dev(d.A), *dB = to_dev(d.B), *dC = to_dev(d.C);
  int *dp0 = to_dev(d.p0), *dp1 = to_dev(d.p1);
  std::uint16_t *dI = to_dev(d.idx);
  _Float16 *dD = nullptr;
  HIP_CHECK(hipMalloc(&dD, kSwM * kSwN * sizeof(_Float16)));
  k_swmmac_f16h_float<<<1, WAVE>>>(dA, dB, dC, dp0, dp1, dI, dD);
  HIP_CHECK(hipDeviceSynchronize());
  std::vector<_Float16> got(kSwM * kSwN);
  HIP_CHECK(hipMemcpy(got.data(), dD, kSwM * kSwN * sizeof(_Float16), hipMemcpyDeviceToHost));
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

TEST(SwmmacGfx12, F16_F16_FpsanMatchesScalarReference) {
  if (!have_device())
    GTEST_SKIP() << "no HIP device";
  SwData d = make_sw_data(0x300, encode_idx_h);
  float *dA = to_dev(d.A), *dB = to_dev(d.B), *dC = to_dev(d.C);
  int *dp0 = to_dev(d.p0), *dp1 = to_dev(d.p1);
  std::uint16_t *dI = to_dev(d.idx);
  fpsan_test::for_matrix_fpsan_semantics([&](auto sem) {
    constexpr Semantics S = decltype(sem)::value;
    std::vector<std::uint64_t> ref = sw_reference_fpsan_h<S, _Float16, _Float16, _Float16>(d);
    std::uint16_t *dD = nullptr;
    HIP_CHECK(hipMalloc(&dD, kSwM * kSwN * sizeof(std::uint16_t)));
    k_swmmac_f16h_fpsan<S><<<1, WAVE>>>(dA, dB, dC, dp0, dp1, dI, dD);
    HIP_CHECK(hipDeviceSynchronize());
    std::vector<std::uint16_t> got(kSwM * kSwN);
    HIP_CHECK(
        hipMemcpy(got.data(), dD, kSwM * kSwN * sizeof(std::uint16_t), hipMemcpyDeviceToHost));
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
