// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/hip_test_utils.hpp
// ----------------------------------------------------------------------------
// HIP-side test helpers shared by the GPU test files: the HIP_CHECK assertion
// macro, a device-availability probe, and host<->device vector copies. Kept in
// the global namespace so the existing unqualified call sites need no change.
// Include only from tests compiled as HIP.
// ----------------------------------------------------------------------------
#ifndef FPSAN_TESTS_HIP_TEST_UTILS_HPP
#define FPSAN_TESTS_HIP_TEST_UTILS_HPP

#include "test_utils.hpp" // bits_of

#include <hip/hip_runtime.h>

#include <gtest/gtest.h>

#include <cstddef>
#include <string>
#include <vector>

// Assert a HIP call succeeded (prints the HIP error string on failure).
#define HIP_CHECK(e)                                                                               \
  do {                                                                                             \
    hipError_t e_ = (e);                                                                           \
    ASSERT_EQ(e_, hipSuccess) << hipGetErrorString(e_);                                            \
  } while (0)

// True when at least one HIP device is present (tests GTEST_SKIP otherwise).
inline bool have_device() {
  int n = 0;
  return hipGetDeviceCount(&n) == hipSuccess && n > 0;
}

// True when the current HIP device is gfx1250. gfx1250 is a DISTINCT
// architecture from the gfx12 / RDNA4 family (gfx1200/1201/1202) despite the
// shared "gfx12" prefix: it adds the permlane_{bcast,down,up,xor},
// permlane16_swap and permlane_idx_gen cross-lane ops that RDNA4 lacks. Those
// builtins are device-gated by __has_builtin in the wrappers, so their kernels
// are not emitted on RDNA4 -- a host that launches them there hits "cannot find
// symbol" at runtime. Tests of those ops must GTEST_SKIP unless this returns
// true. Match the full "gfx1250" token (not a "gfx12" prefix, which would
// wrongly accept RDNA4) and ignore any trailing :feature flags in gcnArchName.
inline bool device_is_gfx1250() {
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess)
    return false;
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, dev) != hipSuccess)
    return false;
  return std::string(prop.gcnArchName).compare(0, 7, "gfx1250") == 0;
}

inline bool device_is_gfx12() {
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess)
    return false;
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, dev) != hipSuccess)
    return false;
  return std::string(prop.gcnArchName).compare(0, 5, "gfx12") == 0;
}

inline bool device_is_gfx950() {
  int dev = 0;
  if (hipGetDevice(&dev) != hipSuccess)
    return false;
  hipDeviceProp_t prop{};
  if (hipGetDeviceProperties(&prop, dev) != hipSuccess)
    return false;
  return std::string(prop.gcnArchName).compare(0, 6, "gfx950") == 0;
}

// Allocate device memory and copy a host vector into it. Caller hipFree()s.
template <class T> T *to_dev(const std::vector<T> &h) {
  T *d = nullptr;
  (void)hipMalloc(&d, h.size() * sizeof(T));
  (void)hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice);
  return d;
}

// Copy n elements back from device memory into a host vector.
template <class T> std::vector<T> from_dev(const T *d, std::size_t n) {
  std::vector<T> h(n);
  (void)hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost);
  return h;
}

#endif // FPSAN_TESTS_HIP_TEST_UTILS_HPP
