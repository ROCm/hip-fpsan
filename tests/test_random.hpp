// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/test_random.hpp
//
// Deterministic pseudo-random test inputs, shared by the tests that need whole
// arrays of numbers (as opposed to a few curated scalars).
//
// Generating array inputs by hand is surprisingly error-prone: an index formula
// like (i*7)%7 looks varied but is constant, which once made an entire WMMA
// operand identically 0 and silently masked a layout bug. A PRNG gives general,
// non-degenerate data -- but only if it is also fully specified and
// reproducible so a failure is the same on every machine and every run.
//
// Ground rules (all deliberate):
//   * std::mt19937 with its standard default seed (5489). The engine and its
//     output sequence are fully specified by the C++ standard, so they are
//     identical on every conforming implementation.
//   * Range mapping is done by hand (lo + draw % span). We do NOT use
//     std::*_distribution -- those are implementation-defined and differ
//     between standard libraries for the same engine/seed.
//   * No std::default_random_engine (implementation-defined), no LCG/xorshift
//     engines (too prone to special-case low bits), no nondeterministic seeds.
// The tiny modulo bias is irrelevant here: we want spread and generality, not
// statistical uniformity.
#ifndef FPSAN_TESTS_TEST_RANDOM_HPP
#define FPSAN_TESTS_TEST_RANDOM_HPP

#include <cstdint>
#include <random>

namespace fpsan_test {

// A fresh engine seeded with the standard default (5489). Each test owns its
// own engine, so tests are order-independent and individually reproducible.
inline std::mt19937 make_rng() { return std::mt19937{}; }

// One draw mapped to the integer range [lo, hi], fully specified. mt19937
// yields values in [0, 2^32), so they fit a uint32_t exactly and this modulo is
// portable.
inline int pick_int(std::mt19937 &rng, int lo, int hi) {
  const std::uint32_t span = static_cast<std::uint32_t>(hi - lo + 1);
  return lo + static_cast<int>(static_cast<std::uint32_t>(rng()) % span);
}

// A small exact-integer value in [lo, hi] as type T. Integers are representable
// exactly in every float type we test, so products and sums of them stay exact
// (needed where a test compares bit-for-bit against hardware or native ops).
// Routed via float so it works for types whose only converting constructor is
// from float (e.g. fpsan::fp8_e4m3, fpsan::fp8_e5m2).
template <class T> T pick_int_valued(std::mt19937 &rng, int lo, int hi) {
  return static_cast<T>(static_cast<float>(pick_int(rng, lo, hi)));
}

// A value of the form k/4 for integer k in [lo4, hi4], as type T. Quarters are
// exact in binary floating point, so this adds non-integer mantissa coverage
// while keeping arithmetic exact.
template <class T> T pick_quarter(std::mt19937 &rng, int lo4, int hi4) {
  return static_cast<T>(pick_int(rng, lo4, hi4)) / static_cast<T>(4);
}

} // namespace fpsan_test

#endif // FPSAN_TESTS_TEST_RANDOM_HPP
