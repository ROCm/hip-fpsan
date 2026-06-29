// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Expected to FAIL compilation: Value's element type must be a supported
// floating-point type. Integer matrix intrinsics (e.g. WMMA_I32_*_I8) don't
// fit the Value<float-type> framework -- the static_assert documents that
// explicitly and the build refuses to instantiate the template.
#include "fpsan/fpsan.hpp"
int main() {
  fpsan::Value<int, fpsan::Semantics::Native, fpsan::Conversions::Explicit> a(0);
  (void)a;
}
