// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mfma_cdna3_parts.hpp"

DENSE_VEC_TRAITS(MfmaFP8_32x32x16_fp8_fp8, 32, 32, 16, 1, 8, v8amd_e4m3_native, v8amd_e4m3_native,
                 v16f_native, amdgcn_mfma_f32_32x32x16_fp8_fp8)

DENSE_VEC_TRAITS(MfmaFP8_32x32x16_fp8_bf8, 32, 32, 16, 1, 8, v8amd_e4m3_native, v8amd_e5m2_native,
                 v16f_native, amdgcn_mfma_f32_32x32x16_fp8_bf8)
