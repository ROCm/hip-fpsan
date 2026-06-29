# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# python/CMakeLists.txt

find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

# Prefer a nanobind package already visible to CMake. If none is found, fetch a
# pinned release checkout at configure time. For offline builds against a local
# checkout, pass -DFETCHCONTENT_SOURCE_DIR_NANOBIND=<path>.
find_package(nanobind CONFIG QUIET)
if(NOT nanobind_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        nanobind
        GIT_REPOSITORY https://github.com/wjakob/nanobind.git
        GIT_TAG v2.6.1
        GIT_SHALLOW TRUE
        GIT_SUBMODULES_RECURSE TRUE
    )
    FetchContent_MakeAvailable(nanobind)
endif()

nanobind_add_module(_fpsan STABLE_ABI LTO fpsan_python.cpp)
target_link_libraries(_fpsan PRIVATE fpsan::fpsan fpsan_project_warnings)
target_compile_features(_fpsan PRIVATE cxx_std_17)
set_target_properties(
    _fpsan
    PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fpsan"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/fpsan"
)

configure_file(
    fpsan/__init__.py
    "${CMAKE_CURRENT_BINARY_DIR}/fpsan/__init__.py"
    COPYONLY
)

add_test(
    NAME fpsan_python_test
    COMMAND
        "${Python_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_fpsan_python.py"
)
set_tests_properties(
    fpsan_python_test
    PROPERTIES ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}"
)
