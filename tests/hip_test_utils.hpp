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
#include <vector>

// Assert a HIP call succeeded (prints the HIP error string on failure).
#define HIP_CHECK(e)                                        \
    do                                                      \
    {                                                       \
        hipError_t e_ = (e);                                \
        ASSERT_EQ(e_, hipSuccess) << hipGetErrorString(e_); \
    } while(0)

// True when at least one HIP device is present (tests GTEST_SKIP otherwise).
inline bool have_device()
{
    int n = 0;
    return hipGetDeviceCount(&n) == hipSuccess && n > 0;
}

// Allocate device memory and copy a host vector into it. Caller hipFree()s.
template <class T>
T* to_dev(const std::vector<T>& h)
{
    T* d = nullptr;
    (void)hipMalloc(&d, h.size() * sizeof(T));
    (void)hipMemcpy(d, h.data(), h.size() * sizeof(T), hipMemcpyHostToDevice);
    return d;
}

// Copy n elements back from device memory into a host vector.
template <class T>
std::vector<T> from_dev(const T* d, std::size_t n)
{
    std::vector<T> h(n);
    (void)hipMemcpy(h.data(), d, n * sizeof(T), hipMemcpyDeviceToHost);
    return h;
}

#endif // FPSAN_TESTS_HIP_TEST_UTILS_HPP
