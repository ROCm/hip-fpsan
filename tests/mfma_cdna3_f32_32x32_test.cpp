// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mfma_cdna3_parts.hpp"

DENSE_F32_TRAITS(MfmaF32_32x32x2, 32, 32, 2, 1, v16f_native, amdgcn_mfma_f32_32x32x2f32)

DENSE_F32_TRAITS(MfmaF32_32x32x1, 32, 32, 1, 2, v32f_native, amdgcn_mfma_f32_32x32x1f32)
