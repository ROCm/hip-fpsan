// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mfma_cdna3_parts.hpp"

DENSE_VEC_TRAITS(MfmaBF16_1k_16x16x16, 16, 16, 16, 1, 16, v4bf_native, v4bf_native, v4f_native,
                 amdgcn_mfma_f32_16x16x16bf16_1k)
