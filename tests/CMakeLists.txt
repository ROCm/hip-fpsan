# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# tests/CMakeLists.txt

# When FPSAN_ENABLE_HIP is ON, compile every test source as HIP so the same
# tests exercise the device path.  We tag sources with the HIP language rather
# than swapping the compiler, which is the established CMake practice.
# (enable_language(HIP) happens at the top level so examples/ sees it too.)
if(FPSAN_ENABLE_HIP)
  set(FPSAN_TEST_LANG HIP)
else()
  set(FPSAN_TEST_LANG CXX)
endif()

# Per-case (ON) registers every TEST() as its own ctest entry via
# gtest_discover_tests: fine-grained ctest -j parallelism and isolated failures
# (a crash fails only that case), at the cost of one process launch -- and full
# HIP runtime init -- per case. Per-binary (OFF) registers each executable as a
# single ctest entry: one process runs all its TEST()s, far lower launch/init
# latency, but a crash/failure takes down the whole binary's entry and -j is only
# across binaries. Default ON to keep granular CI reporting; set OFF for fast
# local runs where everything is expected to pass.
option(FPSAN_CTEST_PER_CASE
       "Register each gtest TEST() as its own ctest entry (vs one entry per binary)"
       ON)

# Helper: define a test executable from one source file, link the library, set
# its language, and register it with ctest.
function(fpsan_add_test name src)
  cmake_parse_arguments(FPSAN_TEST "" "TEST_PREFIX" "" ${ARGN})
  add_executable(${name} ${src})
  set_source_files_properties(${src} PROPERTIES LANGUAGE ${FPSAN_TEST_LANG})
  target_link_libraries(${name} PRIVATE fpsan::fpsan)
  if(TARGET GTest::gtest_main AND FPSAN_CTEST_PER_CASE)
    target_link_libraries(${name} PRIVATE GTest::gtest_main)
    include(GoogleTest)
    # Each TEST() becomes its own ctest entry, so ctest -j parallelizes across
    # them and a crash fails only that case.
    set(_fpsan_gtest_args)
    if(FPSAN_TEST_TEST_PREFIX)
      list(APPEND _fpsan_gtest_args TEST_PREFIX "${FPSAN_TEST_TEST_PREFIX}")
    endif()
    gtest_discover_tests(${name} ${_fpsan_gtest_args})
  elseif(TARGET GTest::gtest_main)
    target_link_libraries(${name} PRIVATE GTest::gtest_main)
    # One ctest entry per binary: all of its TEST()s run in a single process.
    add_test(NAME ${name} COMMAND ${name})
  else()
    # Canary / no-gtest fallback: register the executable as a single test.
    add_test(NAME ${name} COMMAND ${name})
  endif()
endfunction()

# ---- GoogleTest -------------------------------------------------------------
# Prefer a GoogleTest already available to CMake (system package, vcpkg/Conan,
# or a -DCMAKE_PREFIX_PATH install). If none is found, fetch a pinned release at
# configure time with FetchContent. For an offline build against a local
# checkout, point CMake at it with -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=<path>.
find_package(GTest QUIET)
if(NOT TARGET GTest::gtest_main)
  include(FetchContent)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.2
    GIT_SHALLOW TRUE)
  FetchContent_MakeAvailable(googletest)
endif()

# Negative compile test: the source MUST fail to compile. We build an
# EXCLUDE_FROM_ALL target on demand via ctest and mark the test WILL_FAIL, so a
# successful compile is reported as a test failure. This is how we lock in the
# library's compile-time guarantees (a key requirement for AI-agent porting).
function(fpsan_add_compile_fail_test name src)
  add_executable(${name} EXCLUDE_FROM_ALL ${src})
  set_source_files_properties(${src} PROPERTIES LANGUAGE ${FPSAN_TEST_LANG})
  target_link_libraries(${name} PRIVATE fpsan::fpsan)
  set_target_properties(${name} PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD TRUE)
  add_test(NAME ${name}
           COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                   --target ${name} --config $<CONFIG>)
  set_tests_properties(${name} PROPERTIES WILL_FAIL TRUE)
endfunction()

function(fpsan_enable_wave64 name)
  target_compile_definitions(${name} PRIVATE FPSAN_TEST_FORCE_WAVE_SIZE=64)
  target_compile_options(${name}
                         PRIVATE $<$<COMPILE_LANGUAGE:HIP>:-mwavefrontsize64>)
endfunction()

function(fpsan_enable_gfx11_wave64 name)
  fpsan_enable_wave64(${name})
endfunction()

# Helper: a standalone host test that owns its main() and reports failure via a
# nonzero exit code (no gtest). Compiled as plain C++ -- these are host-only and
# need no GPU -- and registered to run directly.
function(fpsan_add_plain_test name src)
  add_executable(${name} ${src})
  target_link_libraries(${name} PRIVATE fpsan::fpsan)
  target_compile_features(${name} PRIVATE cxx_std_17)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

# Helper: a compile-only codegen test -- the source MUST compile (here, forcing
# device codegen of an intrinsic through every Semantics). Built on demand via
# ctest as an OBJECT (the source has no main); a compile failure fails the test.
function(fpsan_add_codegen_test name src)
  add_library(${name} OBJECT EXCLUDE_FROM_ALL ${src})
  set_source_files_properties(${src} PROPERTIES LANGUAGE ${FPSAN_TEST_LANG})
  target_link_libraries(${name} PRIVATE fpsan::fpsan)
  set_target_properties(${name} PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD TRUE)
  add_test(NAME ${name}
           COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR}
                   --target ${name} --config $<CONFIG>)
endfunction()

# ---- build canary: proves the toolchain + INTERFACE target + ctest work even
# before gtest is present.  Always built. -----------------------------------
fpsan_add_test(fpsan_build_canary build_canary.cpp)

# ---- compile-time guarantee tests (do not require gtest) -------------------
fpsan_add_compile_fail_test(fpsan_neg_cross_mode neg_cross_mode.cpp)
fpsan_add_compile_fail_test(fpsan_neg_explicit_ctor neg_explicit_ctor.cpp)
fpsan_add_compile_fail_test(fpsan_neg_explicit_mixed_pod neg_explicit_mixed_pod.cpp)
fpsan_add_compile_fail_test(fpsan_neg_non_float_type neg_non_float_type.cpp)

# ---- algebraic tests (host-only, own main(), no gtest) ---------
# Exhaustive checks of the algebraic Semantics across ALL variants including the
# independent-prime "2" twins (the dedicated home for full-variant coverage).
fpsan_add_plain_test(fpsan_algebraic_proto_test algebraic_proto_test.cpp)
fpsan_add_plain_test(fpsan_algebraic_value_test algebraic_value_test.cpp)

# ---- real test suite (requires gtest) --------------------------------------
if(TARGET GTest::gtest_main)
  fpsan_add_test(fpsan_core_test core_test.cpp)
  fpsan_add_test(fpsan_drop_in_test drop_in_test.cpp)
  fpsan_add_test(fpsan_math_test math_test.cpp)
  fpsan_add_test(fpsan_extern_tagged_test extern_tagged_test.cpp)
  fpsan_add_test(fpsan_vector_test vector_test.cpp)
  fpsan_add_test(fpsan_cast_test cast_test.cpp)
  fpsan_add_test(fpsan_fp8_test fp8_test.cpp)
  # Host<->device parity of the core algebra from a single source body. Built in
  # both configs: pure C++ checks the host invariants; HIP additionally runs the
  # same body in a kernel and asserts device == host bit-for-bit.
  fpsan_add_test(fpsan_parity_test parity_test.cpp)
  # GPU execution tests: only meaningful when compiling as HIP.
  if(FPSAN_ENABLE_HIP)
    # Tests whose kernels touch only gfx9-era / portable builtins (shfl,
    # mbcnt, basic ds_*, classify, atomic_fadd...) build on every supported
    # arch.
    fpsan_add_test(fpsan_hip_device_test hip_device_test.cpp)
    fpsan_add_test(fpsan_atomic_test atomic_test.cpp)
    fpsan_add_test(fpsan_classify_test classify_test.cpp)
    # Architecture-family detection. gfx1250 is in the gfx12 family for shared
    # builtins, but is not RDNA4: it has a distinct WMMA/SWMMAC family and no
    # dot9-12 fdot2/dot4 instructions.
    set(_fpsan_has_gfx12 FALSE)
    set(_fpsan_gfx12_arches)
    set(_fpsan_has_gfx11 FALSE)
    set(_fpsan_gfx11_arches)
    set(_fpsan_has_non_gfx11 FALSE)
    set(_fpsan_has_cdna3 FALSE)
    set(_fpsan_has_rdna4 FALSE)
    set(_fpsan_has_gfx950 FALSE)
    set(_fpsan_rdna4_arches)
    set(_fpsan_fp8_cvt_arches)
    set(_fpsan_xlane_arches)
    set(_fpsan_gfx950_arches)
    foreach(_arch IN LISTS CMAKE_HIP_ARCHITECTURES)
      if(_arch MATCHES "^gfx12")
        set(_fpsan_has_gfx12 TRUE)
        list(APPEND _fpsan_gfx12_arches ${_arch})
        list(APPEND _fpsan_fp8_cvt_arches ${_arch})
        list(APPEND _fpsan_xlane_arches ${_arch})
      endif()
      if(_arch MATCHES "^gfx11")
        set(_fpsan_has_gfx11 TRUE)
        list(APPEND _fpsan_gfx11_arches ${_arch})
        list(APPEND _fpsan_xlane_arches ${_arch})
      else()
        set(_fpsan_has_non_gfx11 TRUE)
      endif()
      if(_arch MATCHES "^gfx94[0-2]$")
        set(_fpsan_has_cdna3 TRUE)
        list(APPEND _fpsan_fp8_cvt_arches ${_arch})
      endif()
      if(_arch MATCHES "^gfx120")
        set(_fpsan_has_rdna4 TRUE)
        list(APPEND _fpsan_rdna4_arches ${_arch})
      endif()
      if(_arch STREQUAL "gfx950")
        set(_fpsan_has_gfx950 TRUE)
        list(APPEND _fpsan_gfx950_arches ${_arch})
      endif()
    endforeach()
    if(_fpsan_has_gfx11 OR _fpsan_has_gfx12 OR _fpsan_has_cdna3)
      # cvt_pkrtz is a portable AMDGPU FP conversion and should run on gfx11 as
      # well. FP8/BF8 conversion cases are a separate target restricted to the
      # architecture subset where those builtins lower.
      fpsan_add_test(fpsan_cvt_test cvt_test.cpp)
      if(_fpsan_has_gfx12 OR _fpsan_has_cdna3)
        fpsan_add_test(fpsan_cvt_fp8_test cvt_test.cpp)
        target_compile_definitions(fpsan_cvt_fp8_test
                                   PRIVATE FPSAN_TEST_ENABLE_PKRTZ_CVT=0
                                           FPSAN_TEST_ENABLE_FP8_CVT=1)
        set_target_properties(fpsan_cvt_fp8_test
                              PROPERTIES HIP_ARCHITECTURES "${_fpsan_fp8_cvt_arches}")
      endif()
    endif()
    if(_fpsan_has_gfx11 OR _fpsan_has_gfx12)
      fpsan_add_test(fpsan_xlane_test xlane_test.cpp)
      set_target_properties(fpsan_xlane_test
                            PROPERTIES HIP_ARCHITECTURES "${_fpsan_xlane_arches}")
    endif()
    if(_fpsan_has_gfx950)
      fpsan_add_test(fpsan_xlane_gfx950_test xlane_test.cpp TEST_PREFIX GFX950.)
      set_target_properties(fpsan_xlane_gfx950_test
                            PROPERTIES HIP_ARCHITECTURES "${_fpsan_gfx950_arches}")
      target_compile_definitions(fpsan_xlane_gfx950_test
                                 PRIVATE FPSAN_TEST_FORCE_WAVE_SIZE=64)
    endif()
    # Shared across the gfx12 family (build on RDNA4 and gfx1250).
    if(_fpsan_has_gfx12)
      fpsan_add_test(fpsan_global_load_tr_test global_load_tr_test.cpp)
      set_target_properties(fpsan_global_load_tr_test
                            PROPERTIES HIP_ARCHITECTURES "${_fpsan_gfx12_arches}")
    endif()
    # RDNA4-only: K=16 WMMA, the fdot2/dot4 (dot9-12) families, and the K=32
    # SWMMAC shapes (swmmac_*_16x16x32_*_w32). gfx1250 has a different WMMA family
    # (added below) and no dot9-12, and its SWMMAC shapes are K=64 (f16/bf16) /
    # K=128 (fp8) -- the K=32 shape is absent from the gfx1250 ISA, so the gfx12
    # K=32 builtin emits an instruction gfx1250 cannot decode, so it is excluded
    # on gfx1250.
    if(_fpsan_has_rdna4)
      fpsan_add_test(fpsan_global_load_tr_gfx12_w64_test
                     global_load_tr_test.cpp TEST_PREFIX GFX12W64.)
      fpsan_add_test(fpsan_wmma_gfx12_test wmma_gfx12_test.cpp)
      fpsan_add_test(fpsan_wmma_gfx12_w64_test wmma_gfx12_test.cpp)
      fpsan_add_test(fpsan_swmmac_gfx12_test swmmac_gfx12_test.cpp)
      fpsan_add_test(fpsan_swmmac_gfx12_w64_test swmmac_gfx12_w64_test.cpp)
      fpsan_add_test(fpsan_xlane_gfx12_w64_test xlane_test.cpp TEST_PREFIX GFX12W64.)
      fpsan_add_test(fpsan_classify_gfx12_w64_test classify_test.cpp TEST_PREFIX GFX12W64.)
      fpsan_add_test(fpsan_wave_gfx12_w64_test wave_test.cpp TEST_PREFIX GFX12W64.)
      # Compile-only: force the RDNA4 K=16 WMMA codegen through every Semantics
      # (incl. the algebraic variants and their "2" twins) -- orthogonality at
      # the GPU level. The kernel uses amdgcn_wmma_f32_16x16x16_f16_w32, the
      # RDNA4-only K=16 builtin, so it is gated with the other RDNA4 WMMA tests.
      fpsan_add_codegen_test(fpsan_algebraic_device_test algebraic_device_test.cpp)
      set_target_properties(fpsan_wmma_gfx12_test
                            fpsan_global_load_tr_gfx12_w64_test
                            fpsan_wmma_gfx12_w64_test
                            fpsan_swmmac_gfx12_test
                            fpsan_swmmac_gfx12_w64_test
                            fpsan_xlane_gfx12_w64_test
                            fpsan_classify_gfx12_w64_test
                            fpsan_wave_gfx12_w64_test
                            fpsan_algebraic_device_test
                            PROPERTIES HIP_ARCHITECTURES "${_fpsan_rdna4_arches}")
      target_compile_definitions(fpsan_wmma_gfx12_w64_test
                                 PRIVATE FPSAN_TEST_GFX12_WAVE64=1)
      target_compile_definitions(fpsan_global_load_tr_gfx12_w64_test
                                 PRIVATE FPSAN_TEST_GLOBAL_LOAD_TR_W64=1)
      target_compile_definitions(fpsan_wave_gfx12_w64_test
                                 PRIVATE FPSAN_TEST_ENABLE_WAVE_F64=1
                                         FPSAN_TEST_SKIP_FLOAT_WAVE_REDUCE=1)
      fpsan_enable_wave64(fpsan_global_load_tr_gfx12_w64_test)
      fpsan_enable_wave64(fpsan_wmma_gfx12_w64_test)
      fpsan_enable_wave64(fpsan_swmmac_gfx12_w64_test)
      fpsan_enable_wave64(fpsan_xlane_gfx12_w64_test)
      fpsan_enable_wave64(fpsan_classify_gfx12_w64_test)
      fpsan_enable_wave64(fpsan_wave_gfx12_w64_test)
    endif()
    if(_fpsan_has_gfx11)
      fpsan_add_test(fpsan_wmma_gfx11_test wmma_gfx11_test.cpp)
      fpsan_add_test(fpsan_wmma_gfx11_w64_test wmma_gfx11_test.cpp)
      set_target_properties(fpsan_wmma_gfx11_test fpsan_wmma_gfx11_w64_test
                            PROPERTIES HIP_ARCHITECTURES "${_fpsan_gfx11_arches}")
      target_compile_definitions(fpsan_wmma_gfx11_w64_test
                                 PRIVATE FPSAN_TEST_GFX11_WAVE64=1)
      fpsan_enable_gfx11_wave64(fpsan_wmma_gfx11_w64_test)
    endif()
    if(_fpsan_has_gfx11)
      fpsan_add_test(fpsan_xlane_w64_test xlane_test.cpp TEST_PREFIX W64.)
      set_target_properties(fpsan_xlane_w64_test
                            PROPERTIES HIP_ARCHITECTURES "${_fpsan_gfx11_arches}")
      fpsan_enable_gfx11_wave64(fpsan_xlane_w64_test)

      fpsan_add_test(fpsan_classify_w64_test classify_test.cpp TEST_PREFIX W64.)
      set_target_properties(fpsan_classify_w64_test
                            PROPERTIES HIP_ARCHITECTURES "${_fpsan_gfx11_arches}")
      fpsan_enable_gfx11_wave64(fpsan_classify_w64_test)

      fpsan_add_test(fpsan_wave_gfx11_w64_test wave_test.cpp TEST_PREFIX W64.)
      set_target_properties(fpsan_wave_gfx11_w64_test
                            PROPERTIES HIP_ARCHITECTURES "${_fpsan_gfx11_arches}")
      target_compile_definitions(fpsan_wave_gfx11_w64_test
                                 PRIVATE FPSAN_TEST_ENABLE_WAVE_F64=1
                                         FPSAN_TEST_SKIP_FLOAT_WAVE_REDUCE=1)
      fpsan_enable_gfx11_wave64(fpsan_wave_gfx11_w64_test)
    endif()
    if(_fpsan_has_cdna3)
      fpsan_add_test(fpsan_mfma_cdna3_test mfma_cdna3_test.cpp)
      fpsan_add_test(fpsan_mfma_cdna3_f16_16x16x4_test
                     mfma_cdna3_f16_16x16x4_test.cpp)
      fpsan_add_test(fpsan_mfma_cdna3_f16_wide_test
                     mfma_cdna3_f16_wide_test.cpp)
      fpsan_add_test(fpsan_mfma_cdna3_f16_32x32x4_test
                     mfma_cdna3_f16_32x32x4_test.cpp)
      fpsan_add_test(fpsan_mfma_cdna3_f16_4x4x4_test
                     mfma_cdna3_f16_4x4x4_test.cpp)
      fpsan_add_test(fpsan_mfma_cdna3_bf16_test mfma_cdna3_bf16_test.cpp)
      fpsan_add_test(fpsan_mfma_cdna3_bf16_16x16x4_test
                     mfma_cdna3_bf16_16x16x4_test.cpp)
      fpsan_add_test(fpsan_mfma_cdna3_bf16_wide_test
                     mfma_cdna3_bf16_wide_test.cpp)
      fpsan_add_test(fpsan_mfma_cdna3_bf16_32x32x4_test
                     mfma_cdna3_bf16_32x32x4_test.cpp)
      fpsan_add_test(fpsan_mfma_cdna3_bf16_4x4x4_test
                     mfma_cdna3_bf16_4x4x4_test.cpp)
      fpsan_add_test(fpsan_mfma_cdna3_fp8_test mfma_cdna3_fp8_test.cpp)
      fpsan_add_test(fpsan_mfma_cdna3_fp8_16x16x32_bf8_test
                     mfma_cdna3_fp8_16x16x32_bf8_test.cpp)
      fpsan_add_test(fpsan_mfma_cdna3_fp8_32x32x16_fp8_test
                     mfma_cdna3_fp8_32x32x16_fp8_test.cpp)
      fpsan_add_test(fpsan_mfma_cdna3_fp8_32x32x16_bf8_test
                     mfma_cdna3_fp8_32x32x16_bf8_test.cpp)
      fpsan_add_test(fpsan_mfma_cdna3_xf32_test mfma_cdna3_xf32_test.cpp)
      fpsan_add_test(fpsan_mfma_cdna3_f32_test mfma_cdna3_f32_test.cpp)
      fpsan_add_test(fpsan_mfma_cdna3_f32_32x32_test
                     mfma_cdna3_f32_32x32_test.cpp)
      fpsan_add_test(fpsan_mfma_cdna3_f32_4x4_test
                     mfma_cdna3_f32_4x4_test.cpp)
      fpsan_add_test(fpsan_mfma_cdna3_f64_test mfma_cdna3_f64_test.cpp)
      fpsan_add_test(fpsan_mfma_cdna3_f64_4x4_test
                     mfma_cdna3_f64_4x4_test.cpp)
      fpsan_add_test(fpsan_smfmac_cdna3_test smfmac_cdna3_test.cpp)
    endif()
    # gfx1250-only tests (builtins exclusive to gfx1250: bf16 transcendentals,
    # bf16/fp8<->f16 cvt, block-scale cvt, gfx1250 WMMA/SWMMAC, ...).
    set(_fpsan_has_gfx1250 FALSE)
    foreach(_arch IN LISTS CMAKE_HIP_ARCHITECTURES)
      if(_arch MATCHES "^gfx125")
        set(_fpsan_has_gfx1250 TRUE)
      endif()
    endforeach()
    if(_fpsan_has_gfx1250)
      fpsan_add_test(fpsan_amdgcn_bf16_trans_test amdgcn_bf16_trans_test.cpp)
      fpsan_add_test(fpsan_cvt_f16_fp8_gfx1250_test cvt_f16_fp8_gfx1250_test.cpp)
      fpsan_add_test(fpsan_cvt_scalef32_pk_gfx1250_test
                     cvt_scalef32_pk_gfx1250_test.cpp)
      fpsan_add_test(fpsan_cvt_scale_unpack_gfx1250_test
                     cvt_scale_unpack_gfx1250_test.cpp)
      fpsan_add_test(fpsan_cvt_fp8_e5m3_gfx1250_test
                     cvt_fp8_e5m3_gfx1250_test.cpp)
      fpsan_add_test(fpsan_cvt_sr_pack_gfx1250_test
                     cvt_sr_pack_gfx1250_test.cpp)
      fpsan_add_test(fpsan_cvt_sr_pk_f16_gfx1250_test
                     cvt_sr_pk_f16_gfx1250_test.cpp)
      fpsan_add_test(fpsan_load_tr16_gfx1250_test
                     load_tr16_gfx1250_test.cpp)
      fpsan_add_test(fpsan_wmma_16x16x4_gfx1250_test
                     wmma_16x16x4_gfx1250_test.cpp)
      fpsan_add_test(fpsan_wmma_16x16x32_gfx1250_test
                     wmma_16x16x32_gfx1250_test.cpp)
      fpsan_add_test(fpsan_wmma_16x16x64_fp8_gfx1250_test
                     wmma_16x16x64_fp8_gfx1250_test.cpp)
      fpsan_add_test(fpsan_wmma_16x16x128_fp8_gfx1250_test
                     wmma_16x16x128_fp8_gfx1250_test.cpp)
      fpsan_add_test(fpsan_wmma_16x16x128_f8f6f4_gfx1250_test
                     wmma_16x16x128_f8f6f4_gfx1250_test.cpp)
      fpsan_add_test(fpsan_wmma_32x16x128_f4_gfx1250_test
                     wmma_32x16x128_f4_gfx1250_test.cpp)
      fpsan_add_test(fpsan_wmma_scale_f8f6f4_gfx1250_test
                     wmma_scale_f8f6f4_gfx1250_test.cpp)
      fpsan_add_test(fpsan_swmmac_16x16x128_fp8_gfx1250_test
                     swmmac_16x16x128_fp8_gfx1250_test.cpp)
      fpsan_add_test(fpsan_swmmac_16x16x64_16_gfx1250_test
                     swmmac_16x16x64_16_gfx1250_test.cpp)
    endif()
    # gfx950 (CDNA4) MFMA / smfmac / scaled-cvt tests.
    if(_fpsan_has_gfx950)
      fpsan_add_test(fpsan_mfma_gfx950_test mfma_gfx950_test.cpp)
      fpsan_add_test(fpsan_mfma_gfx950_scaled_test
                     mfma_gfx950_scaled_test.cpp)
      fpsan_add_test(fpsan_mfma_gfx950_smfmac_test
                     mfma_gfx950_smfmac_test.cpp)
      fpsan_add_test(fpsan_ds_read_tr_test ds_read_tr_test.cpp)
      fpsan_add_test(fpsan_cvt_gfx950_test cvt_gfx950_test.cpp)
      fpsan_add_test(fpsan_cvt_scalef32_mx_gfx950_test
                     cvt_scalef32_mx_gfx950_test.cpp)
      fpsan_add_test(fpsan_cvt_scalef32_fp4_gfx950_test
                     cvt_scalef32_fp4_gfx950_test.cpp)
      fpsan_add_test(fpsan_cvt_scalef32_fp6_gfx950_test
                     cvt_scalef32_fp6_gfx950_test.cpp)
      fpsan_add_test(fpsan_cvt_scalef32_fp8vec_gfx950_test
                     cvt_scalef32_fp8vec_gfx950_test.cpp)
      fpsan_add_test(fpsan_cvt_scalef32_sr_gfx950_test
                     cvt_scalef32_sr_gfx950_test.cpp)
    endif()
    # wave_test exercises the f32 wave-reduce wrappers on RDNA/CDNA and, for
    # gfx11-only builds, the f64 wave-reduce wrappers that lower on gfx1100. The
    # test is wave-size agnostic (one full wave, references reduced over warpSize
    # lanes), so register it whenever the arch list contains one of those families.
    # Registered once to avoid duplicate targets in mixed-arch builds.
    if(_fpsan_has_gfx11 OR _fpsan_has_gfx12 OR _fpsan_has_cdna3 OR _fpsan_has_gfx950)
      fpsan_add_test(fpsan_wave_test wave_test.cpp)
      if(_fpsan_has_gfx11 AND NOT _fpsan_has_non_gfx11)
        target_compile_definitions(fpsan_wave_test PRIVATE FPSAN_TEST_ENABLE_WAVE_F64=1)
      endif()
      # ldexp{f,,h} builtins exist + lower on RDNA, CDNA3, and CDNA4.
      fpsan_add_test(fpsan_amdgcn_ldexp_test amdgcn_ldexp_test.cpp)
      # Common scalar math builtins lower on RDNA3, RDNA4, CDNA3, and CDNA4. The same
      # source also contains dot-product tests gated by compile definitions so
      # the gfx12-only cases do not leak into CDNA/gfx1250 builds.
      fpsan_add_test(fpsan_amdgcn_math_test amdgcn_math_test.cpp)
      if(_fpsan_has_gfx11 OR _fpsan_has_rdna4 OR _fpsan_has_cdna3)
        target_compile_definitions(fpsan_amdgcn_math_test
                                   PRIVATE FPSAN_TEST_ENABLE_FDOT2=1)
      endif()
      if(_fpsan_has_gfx11 OR _fpsan_has_rdna4)
        target_compile_definitions(fpsan_amdgcn_math_test
                                   PRIVATE FPSAN_TEST_ENABLE_FDOT2_EXTENDED=1)
      endif()
      if(_fpsan_has_rdna4)
        target_compile_definitions(fpsan_amdgcn_math_test
                                   PRIVATE FPSAN_TEST_ENABLE_GFX12_DOT_MATH=1)
      endif()
    endif()
  endif()
endif()
