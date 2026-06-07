// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan/fpsan.hpp
// ----------------------------------------------------------------------------
// Umbrella header for the FPSan C++ library.  Include this to get everything.
//
//   #include <fpsan/fpsan.hpp>
//   using fpsan::Value;
//
// The library turns floating-point code into a form that can switch between
// ordinary IEEE arithmetic and Triton-style "FPSan" integer-payload arithmetic
// by flipping a template parameter.  See README.md and the tutorials in
// docs/.
// ----------------------------------------------------------------------------
#ifndef FPSAN_FPSAN_HPP
#define FPSAN_FPSAN_HPP

#include "fpsan/cast.hpp"
#include "fpsan/detail/config.hpp"
#include "fpsan/detail/fp8.hpp"
#include "fpsan/math.hpp"
#include "fpsan/numeric_limits.hpp"
#include "fpsan/value.hpp"
// This umbrella pulls the full scalar/vector surface (the Value type, math,
// casts, numeric_limits, fp8). Two things are intentionally left opt-in:
//   * fpsan/io.hpp (host-only iostream operator<<) -- include it explicitly.
//   * fpsan/amdgcn_*.hpp (HIP device intrinsic wrappers) -- include the ones
//     your kernel needs.

#endif // FPSAN_FPSAN_HPP
