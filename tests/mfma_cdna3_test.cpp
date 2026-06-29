// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mfma_cdna3_parts.hpp"

DENSE_VEC_TRAITS(MfmaF16_16x16x16, 16, 16, 16, 1, 16, v4h_native, v4h_native, v4f_native,
                 amdgcn_mfma_f32_16x16x16f16)
