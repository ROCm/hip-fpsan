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
#       The default flows from CMAKE_HIP_COMPILER (set explicitly, auto-detected
#       from a TheRock/therock-venv install, or found on PATH by
#       check_language(HIP)); set -DFPSAN_ENABLE_HIP=OFF to force pure C++.
#
cmake_minimum_required(VERSION 3.21)

project(fpsan LANGUAGES CXX)

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
# Whether the tests/examples compile as HIP C++ flows from the standard
# CMAKE_HIP_COMPILER variable, mirroring recent ROCm libraries (rocPRIM,
# rocThrust: check_language(HIP) plus a CMAKE_HIP_COMPILER-gated option) rather
# than a bespoke always-present toggle. If the user has not pointed
# CMAKE_HIP_COMPILER at a toolchain, try a from-source TheRock build then the
# therock venv SDK, and finally let check_language(HIP) search PATH. Override any
# of this with -DCMAKE_HIP_COMPILER=... / -DCMAKE_HIP_COMPILER_ROCM_ROOT=...
if(NOT CMAKE_HIP_COMPILER)
  # Each candidate: "<rocm-root>|<clang++ relative path>".
  set(_fpsan_rocm_candidates
      "$ENV{HOME}/therock-7.13|lib/llvm/bin/clang++"
      "$ENV{HOME}/workspace/TheRock-build/dist/rocm|lib/llvm/bin/clang++"
      "$ENV{HOME}/therock_venv/lib/python3.13/site-packages/_rocm_sdk_devel|llvm/bin/clang++")
  foreach(_cand ${_fpsan_rocm_candidates})
    string(REPLACE "|" ";" _cand "${_cand}")
    list(GET _cand 0 _rocm_root)
    list(GET _cand 1 _rocm_clang_rel)
    if(EXISTS "${_rocm_root}/${_rocm_clang_rel}")
      message(STATUS "fpsan: found ROCm/HIP toolchain at ${_rocm_root}")
      set(CMAKE_HIP_COMPILER "${_rocm_root}/${_rocm_clang_rel}" CACHE FILEPATH "HIP compiler")
      set(CMAKE_HIP_COMPILER_ROCM_ROOT "${_rocm_root}" CACHE PATH
          "ROCm root containing lib/cmake/hip-lang")
      list(APPEND CMAKE_PREFIX_PATH "${_rocm_root}")
      break()
    endif()
  endforeach()
endif()

include(CheckLanguage)
check_language(HIP) # fills CMAKE_HIP_COMPILER from PATH if still unset

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
