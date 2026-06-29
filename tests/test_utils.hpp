// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/test_utils.hpp
// ----------------------------------------------------------------------------
// Tiny host-side helpers shared by the test suite (pure C++, no HIP). Kept in
// the global namespace so the existing unqualified call sites need no change.
// ----------------------------------------------------------------------------
#ifndef FPSAN_TESTS_TEST_UTILS_HPP
#define FPSAN_TESTS_TEST_UTILS_HPP

#include <cstdint>
#include <cstring>

// Raw bit pattern of a scalar as a u64, for exact (==) comparison of float
// bits regardless of element type (f32/f64/f16/bf16/fp8).
template <class T> std::uint64_t bits_of(T v) {
  std::uint64_t u = 0;
  std::memcpy(&u, &v, sizeof v);
  return u;
}

#if defined(__clang__) && defined(__is_identifier)
#if !__is_identifier(__bf16)
inline std::uint64_t bits_of(__bf16 v) { return __builtin_bit_cast(std::uint16_t, v); }
#endif
#endif

#endif // FPSAN_TESTS_TEST_UTILS_HPP
