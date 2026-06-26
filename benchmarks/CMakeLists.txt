# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# benchmarks/CMakeLists.txt

add_executable(fpsan_cpu_bench cpu_bench.cpp)
target_link_libraries(fpsan_cpu_bench PRIVATE fpsan::fpsan fpsan_project_warnings)
target_compile_features(fpsan_cpu_bench PRIVATE cxx_std_17)

if(CMAKE_HIP_COMPILER)
  add_executable(fpsan_hip_bench hip_bench.cpp)
  set_source_files_properties(hip_bench.cpp PROPERTIES LANGUAGE HIP)
  target_link_libraries(fpsan_hip_bench PRIVATE fpsan::fpsan fpsan_project_warnings)
  target_compile_features(fpsan_hip_bench PRIVATE cxx_std_17)
endif()
