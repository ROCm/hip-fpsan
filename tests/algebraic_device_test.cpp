// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Device-codegen orthogonality proof: the *real* gfx12 WMMA intrinsic wrapper,
// instantiated for the algebraic Semantics, with ZERO changes to the intrinsic.
// Compile-only check (codegen, not execution):
//   amdclang++ -x hip --offload-arch=gfx1201 -c -I include \
//       tests/algebraic_device_test.cpp -o /tmp/algd.o
// ----------------------------------------------------------------------------
#include "fpsan/amdgcn_matrix.hpp"

#include <hip/hip_runtime.h>

using namespace fpsan;

// One generic kernel over Semantics -- exactly what the library promises: the
// intrinsic is semantics-agnostic, so the algebraic variants flow through it.
template <Semantics S> __global__ void wmma_kernel(const _Float16 *A, const _Float16 *B, float *C) {
  using AV = Value<v8h_native, S, Conversions::Explicit>;
  using CV = Value<v8f_native, S, Conversions::Explicit>;
  v8h_native a, b;
  v8f_native c;
  for (int i = 0; i < 8; ++i) {
    a[i] = A[i];
    b[i] = B[i];
    c[i] = C[i];
  }
  CV d = amdgcn_wmma_f32_16x16x16_f16_w32(AV(a), AV(b), CV(c));
  // store the storage bits (payload in payload modes, float bits in Native
  // mode) so the WMMA call is not dead-code-eliminated.
  auto bits = d.to_storage_bits();
  for (int i = 0; i < 8; ++i)
    C[i] = __builtin_bit_cast(float, static_cast<unsigned>(bits[i]));
}

// Force device codegen for Native, Triton, and the algebraic variants through the
// same intrinsic. If the algebraic instantiations compile, orthogonality holds
// at the actual GPU-codegen level.
template __global__ void wmma_kernel<Semantics::Native>(const _Float16 *, const _Float16 *,
                                                        float *);
template __global__ void wmma_kernel<Semantics::Triton>(const _Float16 *, const _Float16 *,
                                                        float *);
template __global__ void wmma_kernel<Semantics::Field>(const _Float16 *, const _Float16 *, float *);
template __global__ void wmma_kernel<Semantics::Field2>(const _Float16 *, const _Float16 *,
                                                        float *);
template __global__ void wmma_kernel<Semantics::FieldFast>(const _Float16 *, const _Float16 *,
                                                           float *);
template __global__ void wmma_kernel<Semantics::FieldFast2>(const _Float16 *, const _Float16 *,
                                                            float *);
template __global__ void wmma_kernel<Semantics::FieldWithMulCasts>(const _Float16 *,
                                                                   const _Float16 *, float *);
template __global__ void wmma_kernel<Semantics::FieldWithMulCasts2>(const _Float16 *,
                                                                    const _Float16 *, float *);
template __global__ void wmma_kernel<Semantics::SophieGermainRing>(const _Float16 *,
                                                                   const _Float16 *, float *);
template __global__ void wmma_kernel<Semantics::SophieGermainRing2>(const _Float16 *,
                                                                    const _Float16 *, float *);
template __global__ void wmma_kernel<Semantics::PythagoreanRing>(const _Float16 *, const _Float16 *,
                                                                 float *);
template __global__ void wmma_kernel<Semantics::PythagoreanRing2>(const _Float16 *,
                                                                  const _Float16 *, float *);
