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
  add_executable(${name} ${src})
  set_source_files_properties(${src} PROPERTIES LANGUAGE ${FPSAN_TEST_LANG})
  target_link_libraries(${name} PRIVATE fpsan::fpsan)
  if(TARGET GTest::gtest_main AND FPSAN_CTEST_PER_CASE)
    target_link_libraries(${name} PRIVATE GTest::gtest_main)
    include(GoogleTest)
    # Each TEST() becomes its own ctest entry, so ctest -j parallelizes across
    # them and a crash fails only that case.
    gtest_discover_tests(${name})
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

# ---- build canary: proves the toolchain + INTERFACE target + ctest work even
# before gtest is present.  Always built. -----------------------------------
fpsan_add_test(fpsan_build_canary build_canary.cpp)

# ---- compile-time guarantee tests (do not require gtest) -------------------
fpsan_add_compile_fail_test(fpsan_neg_cross_mode neg_cross_mode.cpp)
fpsan_add_compile_fail_test(fpsan_neg_explicit_ctor neg_explicit_ctor.cpp)
fpsan_add_compile_fail_test(fpsan_neg_explicit_mixed_pod neg_explicit_mixed_pod.cpp)
fpsan_add_compile_fail_test(fpsan_neg_non_float_type neg_non_float_type.cpp)

# ---- real test suite (requires gtest) --------------------------------------
if(TARGET GTest::gtest_main)
  fpsan_add_test(fpsan_core_test core_test.cpp)
  fpsan_add_test(fpsan_drop_in_test drop_in_test.cpp)
  fpsan_add_test(fpsan_math_test math_test.cpp)
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
    # gfx12 family detection.  Two tiers behave differently and must not be
    # conflated:
    #   _fpsan_has_gfx12  -- any gfx12* (RDNA4 gfx1200/1201 OR gfx1250). Used for
    #                        the builtins shared across the whole family
    #                        (permlane64, mov_dpp8, ballot_w32, scalar math,
    #                        gfx12 cvt/swmmac/global_load_tr).
    #   _fpsan_has_rdna4  -- only gfx1200/gfx1201. These have the K=16 WMMA
    #                        (wmma-128b-insts) and the dot9-12 fdot2/dot4 families
    #                        that gfx1250 dropped, so the tests that exercise
    #                        those builtins are RDNA4-only.
    set(_fpsan_has_gfx12 FALSE)
    set(_fpsan_has_cdna3 FALSE)
    set(_fpsan_has_rdna4 FALSE)
    foreach(_arch IN LISTS CMAKE_HIP_ARCHITECTURES)
      if(_arch MATCHES "^gfx12")
        set(_fpsan_has_gfx12 TRUE)
      endif()
      if(_arch MATCHES "^gfx94[0-2]$")
        set(_fpsan_has_cdna3 TRUE)
      endif()
      if(_arch MATCHES "^gfx120")
        set(_fpsan_has_rdna4 TRUE)
      endif()
    endforeach()
    if(_fpsan_has_gfx12 OR _fpsan_has_cdna3)
      # Baseline fp8/bf8 conversion builtins are shared by gfx12 and CDNA3.
      # Register once so mixed-architecture HIP builds do not create duplicate
      # CMake targets.
      fpsan_add_test(fpsan_cvt_test cvt_test.cpp)
    endif()
    # Shared across the gfx12 family (build on RDNA4 and gfx1250).
    if(_fpsan_has_gfx12)
      fpsan_add_test(fpsan_xlane_test xlane_test.cpp)
      fpsan_add_test(fpsan_global_load_tr_test global_load_tr_test.cpp)
    endif()
    # RDNA4-only: K=16 WMMA, the fdot2/dot4 (dot9-12) families, and the K=32
    # SWMMAC shapes (swmmac_*_16x16x32_*_w32). gfx1250 has a different WMMA family
    # (added below) and no dot9-12, and its SWMMAC shapes are K=64 (f16/bf16) /
    # K=128 (fp8) -- the K=32 shape is absent from the gfx1250 ISA, so the gfx12
    # K=32 builtin emits an instruction gfx1250 cannot decode, so it is excluded
    # on gfx1250.
    if(_fpsan_has_rdna4)
      fpsan_add_test(fpsan_wmma_test wmma_test.cpp)
      fpsan_add_test(fpsan_swmmac_gfx12_test swmmac_gfx12_test.cpp)
    endif()
    if(_fpsan_has_cdna3)
      fpsan_add_test(fpsan_mfma_cdna3_test mfma_cdna3_test.cpp)
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
    set(_fpsan_has_gfx950 FALSE)
    foreach(_arch IN LISTS CMAKE_HIP_ARCHITECTURES)
      if(_arch STREQUAL "gfx950")
        set(_fpsan_has_gfx950 TRUE)
      endif()
    endforeach()
    if(_fpsan_has_gfx950)
      fpsan_add_test(fpsan_mfma_test mfma_test.cpp)
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
    # wave_test exercises the f32 wave-reduce wrappers
    # (amdgcn_wave_reduce_f{add,sub,min,max}_f32). Contrary to an earlier
    # assumption, these builtins exist AND lower on BOTH RDNA (gfx10+, wave32)
    # and CDNA4/gfx950 (wave64) -- verified on silicon. The test is wave-size
    # agnostic (one full wave, references reduced over warpSize lanes), so we
    # register it whenever the arch list has a gfx12, CDNA3, or a gfx950. Registered
    # once to avoid a duplicate target when both are present.
    if(_fpsan_has_gfx12 OR _fpsan_has_cdna3 OR _fpsan_has_gfx950)
      fpsan_add_test(fpsan_wave_test wave_test.cpp)
      # ldexp{f,,h} builtins exist + lower on RDNA, CDNA3, and CDNA4.
      fpsan_add_test(fpsan_amdgcn_ldexp_test amdgcn_ldexp_test.cpp)
      # Common scalar math builtins lower on RDNA4, CDNA3, and CDNA4. The same
      # source also contains dot-product tests gated by compile definitions so
      # the gfx12-only cases do not leak into CDNA builds.
      fpsan_add_test(fpsan_amdgcn_math_test amdgcn_math_test.cpp)
      # fdot2 / dot4 (dot9-12) exist on RDNA4 and CDNA3 but NOT gfx1250, which
      # dropped the dot families. Gate on rdna4/cdna3 (not the whole gfx12
      # family) so the gfx1250 build of this shared file stays scalar-only.
      if(_fpsan_has_rdna4 OR _fpsan_has_cdna3)
        target_compile_definitions(fpsan_amdgcn_math_test
                                   PRIVATE FPSAN_TEST_ENABLE_FDOT2=1)
      endif()
      if(_fpsan_has_rdna4)
        target_compile_definitions(fpsan_amdgcn_math_test
                                   PRIVATE FPSAN_TEST_ENABLE_GFX12_DOT_MATH=1)
      endif()
    endif()
  endif()
endif()
