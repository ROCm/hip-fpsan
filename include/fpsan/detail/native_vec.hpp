// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan/detail/native_vec.hpp
// ----------------------------------------------------------------------------
// Packed 2-element native vector aliases shared by the AMDGPU wrappers that
// take/return v2 packed operands: the packed-conversion family (amdgcn_cvt),
// fdot2 (amdgcn_math), and the packed FP atomics (amdgcn_atomic).
//
// This header is only ever pulled in transitively by those GPU-only wrappers
// (each guarded with its own `#error` for non-HIP/CUDA builds), so _Float16 and
// __bf16 are guaranteed to be available by the HIP/CUDA frontend.
// ----------------------------------------------------------------------------
#ifndef FPSAN_DETAIL_NATIVE_VEC_HPP
#define FPSAN_DETAIL_NATIVE_VEC_HPP

namespace fpsan {
using v2h_native = _Float16 __attribute__((ext_vector_type(2)));
using v2bf_native = __bf16 __attribute__((ext_vector_type(2)));
using v2i16_native = short __attribute__((ext_vector_type(2)));
} // namespace fpsan

#endif // FPSAN_DETAIL_NATIVE_VEC_HPP
