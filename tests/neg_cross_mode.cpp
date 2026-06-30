// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Expected to FAIL compilation: mixing semantics in one binary op.
#include "fpsan/fpsan.hpp"
int main() {
  fpsan::Value<float, fpsan::Semantics::Triton, fpsan::Conversions::Implicit> a(1.0f);
  fpsan::Value<float, fpsan::Semantics::Native, fpsan::Conversions::Implicit> b(1.0f);
  auto c = a + b; // static_assert: operands must be the same type
  (void)c;
}
