// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "mfma_cdna3_parts.hpp"

DENSE_VEC_TRAITS(MfmaXF32_16x16x8, 16, 16, 8, 1, 32, v2f_native, v2f_native, v4f_native,
                 amdgcn_mfma_f32_16x16x8_xf32)
DENSE_VEC_TRAITS(MfmaXF32_32x32x4, 32, 32, 4, 1, 32, v2f_native, v2f_native, v16f_native,
                 amdgcn_mfma_f32_32x32x4_xf32)

// xF32 shapes have one MFMA block here, so non-zero CBSZ would exceed the
// ISA's log2(blocks) limit. Keep coverage to the legal default modifiers.
