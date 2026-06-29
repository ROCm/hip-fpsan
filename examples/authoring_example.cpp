// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// examples/authoring_example.cpp
//
// Tutorial example: authoring new code directly against Value.
//
// We write a small numeric routine once, generic over the scalar type, then
// instantiate it two ways:
//   * Scalar = float                          -> ordinary IEEE arithmetic
//   * Scalar = Value<float, fpsan::Semantics::Triton,
//   fpsan::Conversions::Implicit> -> FPSan integer-payload algebra
//
// The routine computes the same sum two ways that are equal in exact real
// arithmetic but differ under IEEE rounding. Under FPSan the two orderings
// produce the *identical* payload, which is the property FPSan is built to
// check. See docs/tutorial-authoring.md.
//
// Build (from this directory):
//   c++ -std=c++17 -I../include authoring_example.cpp -o authoring_example
#include <fpsan/fpsan.hpp>
#include <fpsan/io.hpp>

#include <cstdio>
#include <vector>

template <class Scalar> Scalar sum_forward(const std::vector<float> &v) {
  Scalar acc(0);
  for (float x : v)
    acc = acc + Scalar(x);
  return acc;
}
template <class Scalar> Scalar sum_reverse(const std::vector<float> &v) {
  Scalar acc(0);
  for (auto it = v.rbegin(); it != v.rend(); ++it)
    acc = acc + Scalar(*it);
  return acc;
}

int main() {
  // A sum where order matters in floating point: a huge value, its negation,
  // and many small values.
  std::vector<float> v = {1e8f, 1.0f, 2.0f, 3.0f, -1e8f, 0.5f, 0.25f};

  float f_fwd = sum_forward<float>(v);
  float f_rev = sum_reverse<float>(v);
  std::printf("plain float : forward=%.6f reverse=%.6f  %s\n", f_fwd, f_rev,
              f_fwd == f_rev ? "equal" : "DIFFER (rounding)");

  using FpsanF = fpsan::Value<float, fpsan::Semantics::Triton, fpsan::Conversions::Implicit>;
  FpsanF s_fwd = sum_forward<FpsanF>(v);
  FpsanF s_rev = sum_reverse<FpsanF>(v);
  std::printf("fpsan       : forward payload=%u reverse payload=%u  %s\n", s_fwd.fpsan_payload(),
              s_rev.fpsan_payload(), s_fwd == s_rev ? "EQUAL" : "differ");

  // The whole point: FPSan makes the reassociation exact.
  if (!(s_fwd == s_rev)) {
    std::printf("ERROR: expected FPSan sums to be equal\n");
    return 1;
  }
  return 0;
}
