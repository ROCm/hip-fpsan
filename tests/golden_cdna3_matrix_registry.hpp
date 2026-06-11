// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Shared CDNA3 matrix intrinsic registry used by the LayoutMatchesHardware
// tests and by the silicon golden generator. Each entry is deliberately
// declarative: the including file supplies the macro that expands it into a
// traits struct, a test, or a generator registration.
#ifndef FPSAN_TESTS_GOLDEN_CDNA3_MATRIX_REGISTRY_HPP
#define FPSAN_TESTS_GOLDEN_CDNA3_MATRIX_REGISTRY_HPP

#define FPSAN_CDNA3_DENSE_VEC_MATRIX_INTRINSICS(X) \
    X(MfmaF16_16x16x16,                            \
      16,                                          \
      16,                                          \
      16,                                          \
      1,                                           \
      16,                                          \
      v4h_native,                                  \
      v4h_native,                                  \
      v4f_native,                                  \
      amdgcn_mfma_f32_16x16x16f16)                 \
    X(MfmaF16_16x16x4,                             \
      16,                                          \
      16,                                          \
      4,                                           \
      4,                                           \
      16,                                          \
      v4h_native,                                  \
      v4h_native,                                  \
      v16f_native,                                 \
      amdgcn_mfma_f32_16x16x4f16)                  \
    X(MfmaF16_32x32x8,                             \
      32,                                          \
      32,                                          \
      8,                                           \
      1,                                           \
      16,                                          \
      v4h_native,                                  \
      v4h_native,                                  \
      v16f_native,                                 \
      amdgcn_mfma_f32_32x32x8f16)                  \
    X(MfmaF16_32x32x4,                             \
      32,                                          \
      32,                                          \
      4,                                           \
      2,                                           \
      16,                                          \
      v4h_native,                                  \
      v4h_native,                                  \
      v32f_native,                                 \
      amdgcn_mfma_f32_32x32x4f16)                  \
    X(MfmaF16_4x4x4,                               \
      4,                                           \
      4,                                           \
      4,                                           \
      16,                                          \
      16,                                          \
      v4h_native,                                  \
      v4h_native,                                  \
      v4f_native,                                  \
      amdgcn_mfma_f32_4x4x4f16)                    \
    X(MfmaBF16_1k_16x16x16,                        \
      16,                                          \
      16,                                          \
      16,                                          \
      1,                                           \
      16,                                          \
      v4bf_native,                                 \
      v4bf_native,                                 \
      v4f_native,                                  \
      amdgcn_mfma_f32_16x16x16bf16_1k)             \
    X(MfmaBF16_1k_16x16x4,                         \
      16,                                          \
      16,                                          \
      4,                                           \
      4,                                           \
      16,                                          \
      v4bf_native,                                 \
      v4bf_native,                                 \
      v16f_native,                                 \
      amdgcn_mfma_f32_16x16x4bf16_1k)              \
    X(MfmaBF16_1k_32x32x8,                         \
      32,                                          \
      32,                                          \
      8,                                           \
      1,                                           \
      16,                                          \
      v4bf_native,                                 \
      v4bf_native,                                 \
      v16f_native,                                 \
      amdgcn_mfma_f32_32x32x8bf16_1k)              \
    X(MfmaBF16_1k_32x32x4,                         \
      32,                                          \
      32,                                          \
      4,                                           \
      2,                                           \
      16,                                          \
      v4bf_native,                                 \
      v4bf_native,                                 \
      v32f_native,                                 \
      amdgcn_mfma_f32_32x32x4bf16_1k)              \
    X(MfmaBF16_1k_4x4x4,                           \
      4,                                           \
      4,                                           \
      4,                                           \
      16,                                          \
      16,                                          \
      v4bf_native,                                 \
      v4bf_native,                                 \
      v4f_native,                                  \
      amdgcn_mfma_f32_4x4x4bf16_1k)                \
    X(MfmaFP8_16x16x32_fp8_fp8,                    \
      16,                                          \
      16,                                          \
      32,                                          \
      1,                                           \
      8,                                           \
      v8amd_e4m3_native,                           \
      v8amd_e4m3_native,                           \
      v4f_native,                                  \
      amdgcn_mfma_f32_16x16x32_fp8_fp8)            \
    X(MfmaFP8_16x16x32_fp8_bf8,                    \
      16,                                          \
      16,                                          \
      32,                                          \
      1,                                           \
      8,                                           \
      v8amd_e4m3_native,                           \
      v8amd_e5m2_native,                           \
      v4f_native,                                  \
      amdgcn_mfma_f32_16x16x32_fp8_bf8)            \
    X(MfmaFP8_16x16x32_bf8_fp8,                    \
      16,                                          \
      16,                                          \
      32,                                          \
      1,                                           \
      8,                                           \
      v8amd_e5m2_native,                           \
      v8amd_e4m3_native,                           \
      v4f_native,                                  \
      amdgcn_mfma_f32_16x16x32_bf8_fp8)            \
    X(MfmaFP8_16x16x32_bf8_bf8,                    \
      16,                                          \
      16,                                          \
      32,                                          \
      1,                                           \
      8,                                           \
      v8amd_e5m2_native,                           \
      v8amd_e5m2_native,                           \
      v4f_native,                                  \
      amdgcn_mfma_f32_16x16x32_bf8_bf8)            \
    X(MfmaFP8_32x32x16_fp8_fp8,                    \
      32,                                          \
      32,                                          \
      16,                                          \
      1,                                           \
      8,                                           \
      v8amd_e4m3_native,                           \
      v8amd_e4m3_native,                           \
      v16f_native,                                 \
      amdgcn_mfma_f32_32x32x16_fp8_fp8)            \
    X(MfmaFP8_32x32x16_fp8_bf8,                    \
      32,                                          \
      32,                                          \
      16,                                          \
      1,                                           \
      8,                                           \
      v8amd_e4m3_native,                           \
      v8amd_e5m2_native,                           \
      v16f_native,                                 \
      amdgcn_mfma_f32_32x32x16_fp8_bf8)            \
    X(MfmaFP8_32x32x16_bf8_fp8,                    \
      32,                                          \
      32,                                          \
      16,                                          \
      1,                                           \
      8,                                           \
      v8amd_e5m2_native,                           \
      v8amd_e4m3_native,                           \
      v16f_native,                                 \
      amdgcn_mfma_f32_32x32x16_bf8_fp8)            \
    X(MfmaFP8_32x32x16_bf8_bf8,                    \
      32,                                          \
      32,                                          \
      16,                                          \
      1,                                           \
      8,                                           \
      v8amd_e5m2_native,                           \
      v8amd_e5m2_native,                           \
      v16f_native,                                 \
      amdgcn_mfma_f32_32x32x16_bf8_bf8)            \
    X(MfmaXF32_16x16x8,                            \
      16,                                          \
      16,                                          \
      8,                                           \
      1,                                           \
      32,                                          \
      v2f_native,                                  \
      v2f_native,                                  \
      v4f_native,                                  \
      amdgcn_mfma_f32_16x16x8_xf32)                \
    X(MfmaXF32_32x32x4,                            \
      32,                                          \
      32,                                          \
      4,                                           \
      1,                                           \
      32,                                          \
      v2f_native,                                  \
      v2f_native,                                  \
      v16f_native,                                 \
      amdgcn_mfma_f32_32x32x4_xf32)

#define FPSAN_CDNA3_DENSE_F32_MATRIX_INTRINSICS(X)                            \
    X(MfmaF32_16x16x4, 16, 16, 4, 1, v4f_native, amdgcn_mfma_f32_16x16x4f32)  \
    X(MfmaF32_16x16x1, 16, 16, 1, 4, v16f_native, amdgcn_mfma_f32_16x16x1f32) \
    X(MfmaF32_32x32x2, 32, 32, 2, 1, v16f_native, amdgcn_mfma_f32_32x32x2f32) \
    X(MfmaF32_32x32x1, 32, 32, 1, 2, v32f_native, amdgcn_mfma_f32_32x32x1f32) \
    X(MfmaF32_4x4x1, 4, 4, 1, 16, v4f_native, amdgcn_mfma_f32_4x4x1f32)

#define FPSAN_CDNA3_SMF_H_MATRIX_INTRINSICS(X)  \
    X(SmfmacF16_16x16x32, _Float16, 16, 16, 32) \
    X(SmfmacBf16_16x16x32, __bf16, 16, 16, 32)  \
    X(SmfmacF16_32x32x16, _Float16, 32, 32, 16) \
    X(SmfmacBf16_32x32x16, __bf16, 32, 32, 16)

#define FPSAN_CDNA3_SMF_FP8_MATRIX_INTRINSICS(X)                                    \
    X(SmfmacFp8_16x16x64_FP8_FP8, fpsan::amd_fp8_e4m3, fpsan::amd_fp8_e4m3, 16, 64) \
    X(SmfmacFp8_16x16x64_FP8_BF8, fpsan::amd_fp8_e4m3, fpsan::amd_fp8_e5m2, 16, 64) \
    X(SmfmacFp8_16x16x64_BF8_FP8, fpsan::amd_fp8_e5m2, fpsan::amd_fp8_e4m3, 16, 64) \
    X(SmfmacFp8_16x16x64_BF8_BF8, fpsan::amd_fp8_e5m2, fpsan::amd_fp8_e5m2, 16, 64) \
    X(SmfmacFp8_32x32x32_FP8_FP8, fpsan::amd_fp8_e4m3, fpsan::amd_fp8_e4m3, 32, 32) \
    X(SmfmacFp8_32x32x32_FP8_BF8, fpsan::amd_fp8_e4m3, fpsan::amd_fp8_e5m2, 32, 32) \
    X(SmfmacFp8_32x32x32_BF8_FP8, fpsan::amd_fp8_e5m2, fpsan::amd_fp8_e4m3, 32, 32) \
    X(SmfmacFp8_32x32x32_BF8_BF8, fpsan::amd_fp8_e5m2, fpsan::amd_fp8_e5m2, 32, 32)

#endif // FPSAN_TESTS_GOLDEN_CDNA3_MATRIX_REGISTRY_HPP
