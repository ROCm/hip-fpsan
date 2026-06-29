// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mfma_cdna3_parts.hpp"

DENSE_VEC_TRAITS(MfmaBF16_1k_16x16x4, 16, 16, 4, 4, 16, v4bf_native, v4bf_native, v16f_native,
                 amdgcn_mfma_f32_16x16x4bf16_1k)
