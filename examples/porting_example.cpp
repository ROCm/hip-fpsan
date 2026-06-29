// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// examples/porting_example.cpp
//
// Tutorial example: porting an existing codebase incrementally. We keep one
// kernel and move a single `Scalar` typedef through the four stages, showing
// that each stage still compiles and runs. See
// docs/tutorial-porting.md.
//
//   Stage 0: Scalar = float                            (original code)
//   Stage 1: Scalar = Value<float, fpsan::Semantics::Native,
//   fpsan::Conversions::Implicit>  (drop-in, identical math) Stage 2: Scalar =
//   Value<float, fpsan::Semantics::Native, fpsan::Conversions::Explicit>
//   (explicit conversions:
//            find accidental float<->wrapper mixing at compile time)
//   Stage 3: Scalar = Value<float, fpsan::Semantics::Triton,
//   fpsan::Conversions::Explicit>   (FPSan semantics on)
//
// (Semantics::Triton with Conversions::Implicit is also legal; turning on
// explicit conversions first is just a recommended safety step.)
//
// Build (from this directory):
//   c++ -std=c++17 -I../include porting_example.cpp -o porting_example
#include <fpsan/fpsan.hpp>

#include <cstdio>
#include <vector>

// The kernel under porting. Note: written to be conversion-clean, so it also
// compiles at Stage 2/3 where implicit float<->Scalar conversions are off.
// (Scalar literals are constructed explicitly; reductions stay in Scalar.)
template <class Scalar>
Scalar weighted_norm(const std::vector<float> &xs, const std::vector<float> &ws) {
  Scalar acc(0);
  for (std::size_t i = 0; i < xs.size(); ++i) {
    Scalar x(xs[i]);
    Scalar w(ws[i]);
    acc = acc + w * (x * x);
  }
  return acc;
}

int main() {
  std::vector<float> xs = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> ws = {0.5f, 0.25f, 0.125f, 0.0625f};

  // Stage 0 + 1: identical results (drop-in).
  float s0 = weighted_norm<float>(xs, ws);
  using S1 = fpsan::Value<float, fpsan::Semantics::Native, fpsan::Conversions::Implicit>;
  float s1 = static_cast<float>(weighted_norm<S1>(xs, ws));
  std::printf("stage0 float            = %.6f\n", s0);
  std::printf("stage1 <float,0,0>      = %.6f  (%s)\n", s1, s0 == s1 ? "identical" : "DIFFERS");

  // Stage 2: explicit conversions on; same numbers, stricter type checking.
  using S2 = fpsan::Value<float, fpsan::Semantics::Native, fpsan::Conversions::Explicit>;
  float s2 = static_cast<float>(weighted_norm<S2>(xs, ws));
  std::printf("stage2 <float,0,1>      = %.6f  (%s)\n", s2, s0 == s2 ? "identical" : "DIFFERS");

  // Stage 3: FPSan semantics. Numbers are now scrambled payloads; what matters
  // is that algebraically-equivalent variants agree. We just show it runs.
  using S3 = fpsan::Value<float, fpsan::Semantics::Triton, fpsan::Conversions::Explicit>;
  S3 s3 = weighted_norm<S3>(xs, ws);
  std::printf("stage3 <float,1,1>      = payload %u (numerically meaningless)\n",
              s3.fpsan_payload());

  return (s0 == s1 && s0 == s2) ? 0 : 1;
}
