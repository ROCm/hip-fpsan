// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Expected to FAIL: implicit construction when conversions == Conversions::Explicit.
#include "fpsan/fpsan.hpp"
int main() {
  fpsan::Value<float, fpsan::Semantics::Native, fpsan::Conversions::Explicit> x =
      1.0f; // copy-init needs implicit ctor
  (void)x;
}
