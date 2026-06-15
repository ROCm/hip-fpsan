// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan/amdgcn_classify.hpp
// ----------------------------------------------------------------------------
// FPSan wrappers for AMDGPU classification / float-compare intrinsics. Opt-in
// (not pulled by <fpsan/fpsan.hpp>).
//
// These are inherently "leak" ops -- they inspect the IEEE bit structure of
// the value, which is exactly what the FPSan abstraction is hiding. So the
// FPSan-mode implementation has to unembed the payload back to the represented
// float, then apply the same builtin. The classification result itself
// (bool / wavefront mask) is the same kind of object in both modes; only the
// path differs.
// ----------------------------------------------------------------------------
#ifndef FPSAN_AMDGCN_CLASSIFY_HPP
#define FPSAN_AMDGCN_CLASSIFY_HPP

#include "fpsan/value.hpp"

#if !defined(__HIP__) && !defined(__CUDACC__)
#error "fpsan/amdgcn_classify.hpp is GPU-only; compile as HIP (or CUDA)."
#endif

namespace fpsan
{

// ---- class / classf / classh (per-lane bool, classifies the value's IEEE
// category against the mask) -------------------------------------------------
#define FPSAN_DEFINE_AMDGCN_CLASS(name, type, BUILTIN)                                \
    template <Semantics S = Semantics::Native, Conversions C = Conversions::Explicit> \
    FPSAN_DEVICE bool name(Value<type, S, C> v, int mask)                             \
    {                                                                                 \
        return BUILTIN(v.to_float(), mask);                                           \
    }

    FPSAN_DEFINE_AMDGCN_CLASS(amdgcn_classf, float, __builtin_amdgcn_classf)
    FPSAN_DEFINE_AMDGCN_CLASS(amdgcn_class, double, __builtin_amdgcn_class)
    FPSAN_DEFINE_AMDGCN_CLASS(amdgcn_classh, _Float16, __builtin_amdgcn_classh)

#undef FPSAN_DEFINE_AMDGCN_CLASS

// ---- fcmp / fcmpf (wavefront ballot of float-compare predicate) -------------
// Returns the same-shape ballot mask the AMD builtin returns; both modes call
// the same builtin on the represented float (Value::to_float() handles the
// FPSan-mode unembed transparently).
#define FPSAN_DEFINE_AMDGCN_FCMP(name, type, BUILTIN)                                           \
    template <int Pred, Semantics S = Semantics::Native, Conversions C = Conversions::Explicit> \
    FPSAN_DEVICE std::uint64_t name(Value<type, S, C> a, Value<type, S, C> b)                   \
    {                                                                                           \
        return BUILTIN(a.to_float(), b.to_float(), Pred);                                       \
    }

    FPSAN_DEFINE_AMDGCN_FCMP(amdgcn_fcmpf, float, __builtin_amdgcn_fcmpf)
    FPSAN_DEFINE_AMDGCN_FCMP(amdgcn_fcmp, double, __builtin_amdgcn_fcmp)

#undef FPSAN_DEFINE_AMDGCN_FCMP

} // namespace fpsan

#endif // FPSAN_AMDGCN_CLASSIFY_HPP
