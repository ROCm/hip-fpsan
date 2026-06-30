# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# CMakeLists.txt
#
# Build system for the header-only FPSan C++ library.
#
# The library itself needs no building; this exists to (a) expose an INTERFACE
# target other projects can link, and (b) drive the test suite. Configure into a
# build directory (a nested ./build is fine), e.g.
#
#   cmake -G Ninja -S . -B build
#   cmake --build build
#   ctest --test-dir build
#
# Options:
#   -DFPSAN_BUILD_TESTS=ON|OFF      (default ON)  build and register the tests
#   -DFPSAN_BUILD_EXAMPLES=ON|OFF   (default ON)  build the tutorial examples
#   -DFPSAN_BUILD_BENCHMARKS=ON|OFF (default OFF) build benchmark binaries
#   -DFPSAN_BUILD_PYTHON=ON|OFF     (default OFF) build Python bindings
#   -DFPSAN_TEST_MATRIX_SEMANTIC_PROFILE=full|representative|triton
#                                           (default representative)
#
# Tests and examples compile as HIP C++ whenever a HIP compiler is available --
# there is no separate enable toggle to drift out of sync. CMAKE_HIP_COMPILER is
# the single source of truth: it is derived from ROCM_PATH (resolved as
# -DROCM_PATH= > $ENV{ROCM_PATH} > `rocm-sdk path --root` > /opt/rocm), set
# explicitly with -DCMAKE_HIP_COMPILER=..., or found on PATH by
# check_language(HIP). For a pure-C++ build on a machine that has a HIP
# toolchain, force it empty:  -DCMAKE_HIP_COMPILER=
#
cmake_minimum_required(VERSION 3.21)

project(fpsan LANGUAGES CXX)

# Generate compile_commands.json, for tools like clangd.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---- reject a literal in-source build --------------------------------------
# A nested build directory (e.g. ./build) is fine; only configuring directly
# into the source root is rejected, since that scatters CMake's generated files
# over the tree.
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
    message(
        FATAL_ERROR
        "fpsan: in-source builds are not allowed. Configure into a separate build "
        "directory, e.g.  cmake -G Ninja -S . -B build"
    )
endif()

option(FPSAN_BUILD_TESTS "Build the FPSan test suite" ON)
option(FPSAN_BUILD_EXAMPLES "Build the tutorial examples" ON)
option(FPSAN_BUILD_BENCHMARKS "Build FPSan benchmark binaries" OFF)
option(FPSAN_BUILD_PYTHON "Build FPSan Python bindings" OFF)

# ---- HIP toolchain detection -----------------------------------------------
# Follow the ROCm-library convention (rocPRIM, rocThrust, hipBLASLt): discover
# the toolchain under a ROCM_PATH root resolved with the same priority those
# projects use -- an explicit -DROCM_PATH=..., then $ENV{ROCM_PATH}, then the
# ROCm Python SDK (`rocm-sdk path --root`), then the canonical install location
# (/opt/rocm on Linux). From that root we set the standard CMAKE_HIP_COMPILER
# (ROCm's amdclang++ at lib/llvm/bin, matching rocm-libraries'
# default_amdclang.cmake) and add the root to CMAKE_PREFIX_PATH, then
# check_language(HIP) confirms it (falling back to a PATH search). Whether
# tests/examples compile as HIP then flows from CMAKE_HIP_COMPILER rather than
# from a bespoke always-present toggle.
if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH
            "$ENV{ROCM_PATH}"
            CACHE PATH
            "Path to the ROCm installation"
        )
    else()
        # Ask the ROCm Python SDK before falling back to the canonical location, the
        # same priority the ROCm libraries use (rocm-libraries stinkytofu/hipblaslt):
        #   ROCM_PATH env  >  `rocm-sdk path --root`  >  /opt/rocm
        set(_fpsan_rocm_root "")
        find_program(FPSAN_ROCM_SDK rocm-sdk)
        if(FPSAN_ROCM_SDK)
            execute_process(
                COMMAND "${FPSAN_ROCM_SDK}" path --root
                OUTPUT_VARIABLE _fpsan_rocm_root
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
            )
        endif()
        if(_fpsan_rocm_root AND EXISTS "${_fpsan_rocm_root}")
            set(ROCM_PATH
                "${_fpsan_rocm_root}"
                CACHE PATH
                "Path to the ROCm installation"
            )
        else()
            set(ROCM_PATH
                "/opt/rocm"
                CACHE PATH
                "Path to the ROCm installation"
            )
        endif()
        unset(_fpsan_rocm_root)
    endif()
endif()
if(NOT "${ROCM_PATH}" IN_LIST CMAKE_PREFIX_PATH)
    list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
endif()
# Derive the HIP compiler (ROCm's amdclang++) from ROCM_PATH only when the user
# has said nothing about it. An explicit -DCMAKE_HIP_COMPILER=... is respected --
# including an explicit empty value, which is how you force a pure-C++ build on a
# box that does have a HIP toolchain (it suppresses both this derivation and the
# check_language probe below, so nothing resurrects it).
if(
    NOT DEFINED CMAKE_HIP_COMPILER
    AND EXISTS "${ROCM_PATH}/lib/llvm/bin/amdclang++"
)
    set(CMAKE_HIP_COMPILER
        "${ROCM_PATH}/lib/llvm/bin/amdclang++"
        CACHE FILEPATH
        "HIP compiler"
    )
    if(NOT CMAKE_HIP_COMPILER_ROCM_ROOT)
        set(CMAKE_HIP_COMPILER_ROCM_ROOT
            "${ROCM_PATH}"
            CACHE PATH
            "ROCm root containing lib/cmake/hip-lang"
        )
    endif()
endif()
# Probe for a HIP compiler unless one was explicitly disabled (defined-but-empty).
if(NOT DEFINED CMAKE_HIP_COMPILER OR CMAKE_HIP_COMPILER)
    include(CheckLanguage)
    check_language(HIP) # confirms CMAKE_HIP_COMPILER, or finds it on PATH if unset
endif()

# CMAKE_HIP_COMPILER is the single predicate: tests and examples build as HIP
# whenever a HIP compiler is available, and as pure C++ when it is not. No
# separate cached enable toggle -- that only adds stale build-dir state.
if(CMAKE_HIP_COMPILER)
    # Bake the ROCm runtime lib dirs into test/example RPATHs so the binaries find
    # libamdhip64 etc. without LD_LIBRARY_PATH (the dist is not installed
    # system-wide).
    if(CMAKE_HIP_COMPILER_ROCM_ROOT)
        list(
            APPEND CMAKE_BUILD_RPATH
            "${CMAKE_HIP_COMPILER_ROCM_ROOT}/lib"
            "${CMAKE_HIP_COMPILER_ROCM_ROOT}/lib/llvm/lib"
        )
    endif()
    # Enable HIP at the top level so every subdirectory (tests, examples) can tag
    # sources with the HIP language.
    enable_language(HIP)
endif()

# ---- the header-only library ------------------------------------------------
add_library(fpsan INTERFACE)
add_library(fpsan::fpsan ALIAS fpsan)
target_include_directories(
    fpsan
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_compile_features(fpsan INTERFACE cxx_std_17)

# ---- warnings ---------------------------------------------------------------
# Keep strict warnings local to this repository's own tests/examples/tools.
# Do not attach these to fpsan itself: fpsan is the public dependency target,
# and consumers must not inherit warning policy or warnings-as-errors.
add_library(fpsan_project_warnings INTERFACE)
target_compile_options(
    fpsan_project_warnings
    INTERFACE
        $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall
        -Wextra
        -Werror>
        $<$<COMPILE_LANG_AND_ID:HIP,Clang>:-Wall
        -Wextra
        -Werror>
)

# ---- tests ------------------------------------------------------------------
if(FPSAN_BUILD_TESTS OR FPSAN_BUILD_PYTHON)
    enable_testing()
endif()

if(FPSAN_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(FPSAN_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

if(FPSAN_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

if(FPSAN_BUILD_PYTHON)
    add_subdirectory(python)
endif()
