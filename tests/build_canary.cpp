// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// tests/build_canary.cpp
//
// A dependency-free "does the build system work at all" canary: it includes the
// umbrella header, checks the version macros, and returns success.  It builds
// (and ctest-runs) even before GoogleTest is available, and under both the CXX
// and HIP test languages.  The real feature tests use GoogleTest.
#include "fpsan/fpsan.hpp"

#include <cstdio>

int main()
{
    static_assert(FPSAN_VERSION_MAJOR == 0, "unexpected version");
    std::printf("fpsan build canary OK (v%d.%d.%d)\n",
                FPSAN_VERSION_MAJOR,
                FPSAN_VERSION_MINOR,
                FPSAN_VERSION_PATCH);
    return 0;
}
