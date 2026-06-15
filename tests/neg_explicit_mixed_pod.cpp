// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Expected to FAIL: mixing with a POD scalar when conversions == Conversions::Explicit.
#include "fpsan/fpsan.hpp"
int main()
{
    fpsan::Value<float, fpsan::Semantics::Native, fpsan::Conversions::Explicit> x(1.0f);
    auto y = x + 2.0f; // no implicit conversion, no mixed-POD operator
    (void)y;
}
