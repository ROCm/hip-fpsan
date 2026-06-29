// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mfma_cdna3_parts.hpp"

DENSE_VEC_TRAITS(MfmaFP8_16x16x32_bf8_fp8, 16, 16, 32, 1, 8, v8amd_e5m2_native, v8amd_e4m3_native,
                 v4f_native, amdgcn_mfma_f32_16x16x32_bf8_fp8)

DENSE_VEC_TRAITS(MfmaFP8_16x16x32_bf8_bf8, 16, 16, 32, 1, 8, v8amd_e5m2_native, v8amd_e5m2_native,
                 v4f_native, amdgcn_mfma_f32_16x16x32_bf8_bf8)
