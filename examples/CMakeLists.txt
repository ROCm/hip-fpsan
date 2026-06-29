# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# examples/CMakeLists.txt
#
# Tutorial examples. They are built and (when tests are enabled) registered with
# ctest so the tutorial code can never silently rot. They run on the host even
# under the HIP configuration.

function(fpsan_add_example name src)
    add_executable(${name} ${src})
    if(CMAKE_HIP_COMPILER)
        set_source_files_properties(${src} PROPERTIES LANGUAGE HIP)
    endif()
    target_link_libraries(${name} PRIVATE fpsan::fpsan fpsan_project_warnings)
    if(FPSAN_BUILD_TESTS)
        add_test(NAME ${name} COMMAND ${name})
    endif()
endfunction()

fpsan_add_example(fpsan_example_authoring authoring_example.cpp)
fpsan_add_example(fpsan_example_porting porting_example.cpp)

# GPU demo: reduction three ways via CPU naive / wave reduce / matrix bf16.
# The matrix path is single-source with a device-arch `#if` -- K=16 WMMA on
# gfx12 (RDNA4), K=32 WMMA on gfx1250, MFMA on gfx950 -- and is a hard compile
# error on architectures without a matrix code path. No arch is selected here;
# that choice lives in the source and follows CMAKE_HIP_ARCHITECTURES.
if(CMAKE_HIP_COMPILER)
    fpsan_add_example(fpsan_example_reduction_demo reduction_demo.cpp)
endif()
