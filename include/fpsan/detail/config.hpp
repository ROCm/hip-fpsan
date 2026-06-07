// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan/detail/config.hpp
// ----------------------------------------------------------------------------
// Portability shims shared by the whole library.
//
// The library is header-only and intended to compile both as ordinary C++ and
// as HIP C++ (host and device).  This header centralizes the few constructs
// that differ between those worlds so the rest of the code stays clean.
// ----------------------------------------------------------------------------
#ifndef FPSAN_DETAIL_CONFIG_HPP
#define FPSAN_DETAIL_CONFIG_HPP

// C++17 is required. C++17 is the default standard of the HIP/ROCm clang
// (amdclang++) for both host and device, so requiring it keeps the library
// usable from a default HIP compile. Guard the requirement once, here, in the
// foundational header every other header pulls in. Use _MSVC_LANG when present
// (MSVC reports a stale value in __cplusplus unless /Zc:__cplusplus is set).
#if defined(_MSVC_LANG)
#define FPSAN_CPLUSPLUS _MSVC_LANG
#else
#define FPSAN_CPLUSPLUS __cplusplus
#endif
#if FPSAN_CPLUSPLUS < 201703L
#error "fpsan requires C++17 or later (e.g. -std=c++17)."
#endif

// FPSAN_HOST_DEVICE: mark a function callable from both host and device when
// compiling for a GPU target, and a no-op otherwise.  __HIP__ is defined by
// hipcc / `clang -x hip`; __CUDACC__ is defined by nvcc / clang CUDA.
#if defined(__HIP__) || defined(__CUDACC__)
#define FPSAN_HOST_DEVICE __host__ __device__
#define FPSAN_DEVICE __device__
#else
#define FPSAN_HOST_DEVICE
#define FPSAN_DEVICE
#endif

#define FPSAN_VERSION_MAJOR 0
#define FPSAN_VERSION_MINOR 1
#define FPSAN_VERSION_PATCH 0

#endif // FPSAN_DETAIL_CONFIG_HPP
