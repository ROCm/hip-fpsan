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
#   -DFPSAN_ENABLE_HIP=ON|OFF       (default: ON when a HIP toolchain is found,
#                                    else OFF)    compile tests/examples as HIP.
#       The default flows from CMAKE_HIP_COMPILER, derived from ROCM_PATH
#       (resolved as -DROCM_PATH= > $ENV{ROCM_PATH} > `rocm-sdk path --root` >
#       /opt/rocm) or set explicitly; set -DFPSAN_ENABLE_HIP=OFF to force a
#       pure-C++ build.
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
  message(FATAL_ERROR
    "fpsan: in-source builds are not allowed. Configure into a separate build "
    "directory, e.g.  cmake -G Ninja -S . -B build")
endif()

option(FPSAN_BUILD_TESTS "Build the FPSan test suite" ON)
option(FPSAN_BUILD_EXAMPLES "Build the tutorial examples" ON)

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
    set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "Path to the ROCm installation")
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
        ERROR_QUIET)
    endif()
    if(_fpsan_rocm_root AND EXISTS "${_fpsan_rocm_root}")
      set(ROCM_PATH "${_fpsan_rocm_root}" CACHE PATH "Path to the ROCm installation")
    else()
      set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to the ROCm installation")
    endif()
    unset(_fpsan_rocm_root)
  endif()
endif()
if(NOT "${ROCM_PATH}" IN_LIST CMAKE_PREFIX_PATH)
  list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
endif()
# Derive the HIP compiler (ROCm's amdclang++) from ROCM_PATH when the user has
# not set one explicitly; leave an explicit -DCMAKE_HIP_COMPILER untouched.
if(NOT CMAKE_HIP_COMPILER AND EXISTS "${ROCM_PATH}/lib/llvm/bin/amdclang++")
  set(CMAKE_HIP_COMPILER "${ROCM_PATH}/lib/llvm/bin/amdclang++" CACHE FILEPATH "HIP compiler")
  if(NOT CMAKE_HIP_COMPILER_ROCM_ROOT)
    set(CMAKE_HIP_COMPILER_ROCM_ROOT "${ROCM_PATH}" CACHE PATH
        "ROCm root containing lib/cmake/hip-lang")
  endif()
endif()
include(CheckLanguage)
check_language(HIP) # confirms CMAKE_HIP_COMPILER, or finds it on PATH if unset

# The HIP path is on by default when a HIP compiler is available and forced off
# otherwise; pass -DFPSAN_ENABLE_HIP=OFF to force a pure-C++ build even where a
# HIP toolchain exists.
include(CMakeDependentOption)
cmake_dependent_option(FPSAN_ENABLE_HIP
  "Compile the tests and examples as HIP C++ (auto-on when a HIP toolchain is available)"
  ON "CMAKE_HIP_COMPILER" OFF)

if(FPSAN_ENABLE_HIP)
  # Bake the ROCm runtime lib dirs into test/example RPATHs so the binaries find
  # libamdhip64 etc. without LD_LIBRARY_PATH (the dist is not installed
  # system-wide).
  if(CMAKE_HIP_COMPILER_ROCM_ROOT)
    list(APPEND CMAKE_BUILD_RPATH "${CMAKE_HIP_COMPILER_ROCM_ROOT}/lib"
         "${CMAKE_HIP_COMPILER_ROCM_ROOT}/lib/llvm/lib")
  endif()
  # Enable HIP at the top level so every subdirectory (tests, examples) can tag
  # sources with the HIP language.
  enable_language(HIP)
endif()

# ---- the header-only library ------------------------------------------------
add_library(fpsan INTERFACE)
add_library(fpsan::fpsan ALIAS fpsan)
target_include_directories(fpsan INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_compile_features(fpsan INTERFACE cxx_std_17)

# ---- warnings ---------------------------------------------------------------
# -Wall -Wextra -Werror on GCC/Clang-compatible compilers, applied to anything
# that links the library (tests, examples).  HIP source files use the same
# Clang frontend, so the same flags apply to device code too.  MSVC is not in
# scope for the test/example build.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(fpsan INTERFACE -Wall -Wextra -Werror)
endif()

# ---- tests ------------------------------------------------------------------
if(FPSAN_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

if(FPSAN_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()
