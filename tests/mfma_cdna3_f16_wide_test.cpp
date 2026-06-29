// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mfma_cdna3_parts.hpp"

DENSE_VEC_TRAITS(MfmaF16_32x32x8, 32, 32, 8, 1, 16, v4h_native, v4h_native, v16f_native,
                 amdgcn_mfma_f32_32x32x8f16)
