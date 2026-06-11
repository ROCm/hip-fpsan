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

# Helper: define a test executable from one source file, link the library, set
# its language, and register it with ctest.
function(fpsan_add_test name src)
  add_executable(${name} ${src})
  set_source_files_properties(${src} PROPERTIES LANGUAGE ${FPSAN_TEST_LANG})
  target_link_libraries(${name} PRIVATE fpsan::fpsan)
  if(TARGET GTest::gtest_main)
    target_link_libraries(${name} PRIVATE GTest::gtest_main)
    include(GoogleTest)
    # gtest_discover_tests registers each TEST() individually, so ctest -j can
    # parallelize across them.
    gtest_discover_tests(${name})
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

# Tool target for refreshing checked-in silicon golden fixtures. Excluded from
# normal builds and ctest; invoke explicitly on the target architecture.
function(fpsan_add_hip_tool name src)
  if(FPSAN_ENABLE_HIP)
    add_executable(${name} EXCLUDE_FROM_ALL ${src})
    set_source_files_properties(${src} PROPERTIES LANGUAGE HIP)
    target_link_libraries(${name} PRIVATE fpsan::fpsan)
    set_target_properties(${name} PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD TRUE)
  endif()
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
    # Tests whose kernels touch gfx12-specific builtins (WMMA, permlane64,
    # mov_dpp8, ballot_w32, fdot2_f16_f16). Register them only when the
    # arch list contains a gfx12.
    set(_fpsan_has_gfx12 FALSE)
    set(_fpsan_has_cdna3 FALSE)
    foreach(_arch IN LISTS CMAKE_HIP_ARCHITECTURES)
      if(_arch MATCHES "^gfx12")
        set(_fpsan_has_gfx12 TRUE)
      endif()
      if(_arch MATCHES "^gfx94[0-2]$")
        set(_fpsan_has_cdna3 TRUE)
      endif()
    endforeach()
    if(_fpsan_has_gfx12 OR _fpsan_has_cdna3)
      # Baseline fp8/bf8 conversion builtins are shared by gfx12 and CDNA3.
      # Register once so mixed-architecture HIP builds do not create duplicate
      # CMake targets.
      fpsan_add_test(fpsan_cvt_test cvt_test.cpp)
    endif()
    if(_fpsan_has_gfx12)
      fpsan_add_test(fpsan_wmma_test wmma_test.cpp)
      fpsan_add_test(fpsan_xlane_test xlane_test.cpp)
      fpsan_add_test(fpsan_global_load_tr_test global_load_tr_test.cpp)
      fpsan_add_test(fpsan_swmmac_gfx12_test swmmac_gfx12_test.cpp)
    endif()
    if(_fpsan_has_cdna3)
      fpsan_add_test(fpsan_mfma_cdna3_test mfma_cdna3_test.cpp)
      fpsan_add_test(fpsan_smfmac_cdna3_test smfmac_cdna3_test.cpp)
      fpsan_add_hip_tool(fpsan_generate_goldens generate_goldens.cpp)
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
      if(_fpsan_has_gfx12 OR _fpsan_has_cdna3)
        target_compile_definitions(fpsan_amdgcn_math_test
                                   PRIVATE FPSAN_TEST_ENABLE_FDOT2=1)
      endif()
      if(_fpsan_has_gfx12)
        target_compile_definitions(fpsan_amdgcn_math_test
                                   PRIVATE FPSAN_TEST_ENABLE_GFX12_DOT_MATH=1)
      endif()
    endif()
  endif()
endif()
