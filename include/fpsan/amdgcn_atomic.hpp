// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan/amdgcn_atomic.hpp
// ----------------------------------------------------------------------------
// FPSan wrappers for AMDGPU FP atomics. GPU-only, opt-in (not pulled by
// <fpsan/fpsan.hpp>).
//
// The two identities behind every atomic in this header:
//
//   - FPSan addition on f32/f64 is integer add mod 2^w on the payload (see
//     [[mix.hpp]]). The corresponding FPSan atomic_fadd is therefore a
//     hardware INTEGER atomicAdd on the payload word -- lock-free, exact, and
//     identical to the FPSan + operation done sequentially.
//
//   - FPSan order is signed-int order on the payload ([[value.hpp]]:less()).
//     Hardware signed-int atomicMin/atomicMax on the payload IS the FPSan
//     atomic_fmin/fmax -- again lock-free and exact.
//
// We use HIP's atomicAdd/atomicMin/atomicMax overloads as the underlying
// primitive: HIP picks the right hardware instruction (global/flat/ds) based
// on the address space the pointer actually references, so a single wrapper
// covers all three pointer kinds without duplicated code or address-space
// gymnastics.
//
// Float-mode fmin/fmax need a CAS loop because there is no hardware
// integer-typed atomic on a float (and the dedicated atomic_fmin/fmax_f32
// builtins are buffer-resource forms that need explicit buffer descriptors).
// The CAS loop is straightforward: read old, compute fmin/fmax in float, CAS
// until success.
// ----------------------------------------------------------------------------
#ifndef FPSAN_AMDGCN_ATOMIC_HPP
#define FPSAN_AMDGCN_ATOMIC_HPP

#include "fpsan/detail/native_vec.hpp"
#include "fpsan/value.hpp"

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

#if !defined(__HIP__) && !defined(__CUDACC__)
#error "fpsan/amdgcn_atomic.hpp is GPU-only; compile as HIP (or CUDA)."
#endif

namespace fpsan
{

    // ---- atomic_fadd_f32 --------------------------------------------------------
    // Returns the OLD value at *addr (matching the AMD builtin convention and
    // HIP atomicAdd). Address space is whatever the pointer references; HIP picks
    // the appropriate hardware instruction.
    template <Semantics S, Conversions C>
    FPSAN_DEVICE Value<float, S, C> amdgcn_atomic_fadd_f32(Value<float, S, C>* addr,
                                                           Value<float, S, C>  v)
    {
        if constexpr(S == Semantics::Native)
        {
            float old = atomicAdd(reinterpret_cast<float*>(addr), v.to_float());
            return Value<float, S, C>(old);
        }
        else
        {
            unsigned old = atomicAdd(reinterpret_cast<unsigned*>(addr),
                                     static_cast<unsigned>(v.fpsan_payload()));
            return Value<float, S, C>::from_fpsan_payload(static_cast<std::uint32_t>(old));
        }
    }

    // ---- atomic_fmin_f32 / atomic_fmax_f32 --------------------------------------
    // FPSan mode reduces to hardware SIGNED-INT atomicMin/atomicMax on the
    // payload. Float mode runs a CAS loop on the float bits.
    namespace detail
    {

        // Type-correct libdevice min/max: fminf/fmaxf for float, fmin/fmax for
        // double (so the float path never silently promotes to the double builtin).
        FPSAN_DEVICE inline float fmm_min(float a, float b)
        {
            return fminf(a, b);
        }
        FPSAN_DEVICE inline float fmm_max(float a, float b)
        {
            return fmaxf(a, b);
        }
        FPSAN_DEVICE inline double fmm_min(double a, double b)
        {
            return fmin(a, b);
        }
        FPSAN_DEVICE inline double fmm_max(double a, double b)
        {
            return fmax(a, b);
        }

        // Float-mode fmin/fmax CAS loop on the raw bits, shared by f32 and f64:
        // read old, compute the FP min/max, CAS until success. UInt is the
        // same-width unsigned integer the hardware atomicCAS operates on (there is
        // no hardware integer-typed atomic on a float/double directly).
        template <bool IsMin, class Float, class UInt>
        FPSAN_DEVICE Float atomic_fmm(Float* p, Float v)
        {
            auto* ip       = reinterpret_cast<UInt*>(p);
            UInt  old_bits = *ip;
            for(;;)
            {
                const Float old_val  = __builtin_bit_cast(Float, old_bits);
                const Float new_val  = IsMin ? fmm_min(old_val, v) : fmm_max(old_val, v);
                const UInt  new_bits = __builtin_bit_cast(UInt, new_val);
                const UInt  witness  = atomicCAS(ip, old_bits, new_bits);
                if(witness == old_bits)
                    return old_val;
                old_bits = witness;
            }
        }

    } // namespace detail

    template <Semantics S, Conversions C>
    FPSAN_DEVICE Value<float, S, C> amdgcn_atomic_fmin_f32(Value<float, S, C>* addr,
                                                           Value<float, S, C>  v)
    {
        if constexpr(S == Semantics::Native)
        {
            float old = detail::atomic_fmm<true, float, unsigned>(reinterpret_cast<float*>(addr),
                                                                  v.to_float());
            return Value<float, S, C>(old);
        }
        else
        {
            int old = atomicMin(reinterpret_cast<int*>(addr), static_cast<int>(v.fpsan_payload()));
            return Value<float, S, C>::from_fpsan_payload(static_cast<std::uint32_t>(old));
        }
    }

    template <Semantics S, Conversions C>
    FPSAN_DEVICE Value<float, S, C> amdgcn_atomic_fmax_f32(Value<float, S, C>* addr,
                                                           Value<float, S, C>  v)
    {
        if constexpr(S == Semantics::Native)
        {
            float old = detail::atomic_fmm<false, float, unsigned>(reinterpret_cast<float*>(addr),
                                                                   v.to_float());
            return Value<float, S, C>(old);
        }
        else
        {
            int old = atomicMax(reinterpret_cast<int*>(addr), static_cast<int>(v.fpsan_payload()));
            return Value<float, S, C>::from_fpsan_payload(static_cast<std::uint32_t>(old));
        }
    }

    // ---- atomic_fadd_f64 --------------------------------------------------------
    // Same identity as f32, widened to 64 bits: FPSan add on f64 is integer add
    // mod 2^64 on the payload, so FPSan atomic_fadd is a hardware 64-bit integer
    // atomicAdd on the payload word.
    template <Semantics S, Conversions C>
    FPSAN_DEVICE Value<double, S, C> amdgcn_atomic_fadd_f64(Value<double, S, C>* addr,
                                                            Value<double, S, C>  v)
    {
        if constexpr(S == Semantics::Native)
        {
            double old = atomicAdd(reinterpret_cast<double*>(addr), v.to_float());
            return Value<double, S, C>(old);
        }
        else
        {
            unsigned long long old = atomicAdd(reinterpret_cast<unsigned long long*>(addr),
                                               static_cast<unsigned long long>(v.fpsan_payload()));
            return Value<double, S, C>::from_fpsan_payload(static_cast<std::uint64_t>(old));
        }
    }

    // ---- atomic_fmin_f64 / atomic_fmax_f64 -------------------------------------
    // FPSan order is signed-int order on the (64-bit) payload, so FPSan reduces to
    // a hardware signed 64-bit atomicMin/atomicMax. Float mode runs a CAS loop on
    // the 64-bit float bits (no hardware integer-typed atomic on a double).
    template <Semantics S, Conversions C>
    FPSAN_DEVICE Value<double, S, C> amdgcn_atomic_fmin_f64(Value<double, S, C>* addr,
                                                            Value<double, S, C>  v)
    {
        if constexpr(S == Semantics::Native)
        {
            double old = detail::atomic_fmm<true, double, unsigned long long>(
                reinterpret_cast<double*>(addr), v.to_float());
            return Value<double, S, C>(old);
        }
        else
        {
            long long old = atomicMin(reinterpret_cast<long long*>(addr),
                                      static_cast<long long>(v.fpsan_payload()));
            return Value<double, S, C>::from_fpsan_payload(static_cast<std::uint64_t>(old));
        }
    }

    template <Semantics S, Conversions C>
    FPSAN_DEVICE Value<double, S, C> amdgcn_atomic_fmax_f64(Value<double, S, C>* addr,
                                                            Value<double, S, C>  v)
    {
        if constexpr(S == Semantics::Native)
        {
            double old = detail::atomic_fmm<false, double, unsigned long long>(
                reinterpret_cast<double*>(addr), v.to_float());
            return Value<double, S, C>(old);
        }
        else
        {
            long long old = atomicMax(reinterpret_cast<long long*>(addr),
                                      static_cast<long long>(v.fpsan_payload()));
            return Value<double, S, C>::from_fpsan_payload(static_cast<std::uint64_t>(old));
        }
    }

    // ---- packed atomic_pk_add (v2f16 / v2bf16) ---------------------------------
    // gfx950 has packed FP-add atomics (global/flat/ds _atomic_pk_add_{f16,bf16}).
    // Unlike f32/f64, we CANNOT use a hardware integer atomicAdd on the packed
    // 32-bit word: that would carry across the 16-bit element boundary, whereas the
    // packed add must add the two 16-bit elements INDEPENDENTLY. So both modes use
    // a 32-bit CAS loop and let Value's element-wise `+` do the work:
    //   * Float: `oldV + v` is a native 2-lane f16/bf16 vector add (two RNE adds),
    //     matching the hardware packed-add element semantics.
    //   * FPSan: `oldV + v` is the per-element ring add (mod 2^16 per lane) -- the
    //     same value the sequential FPSan `+` would produce, order-independent.
    // The CAS loop is lock-free and, because atomicCAS picks the instruction from
    // the pointer's address space, one wrapper covers ds / flat / global.
    // v2h_native / v2bf_native come from fpsan/detail/native_vec.hpp.

    namespace detail
    {
        template <class VEC, Semantics S, Conversions C>
        FPSAN_DEVICE Value<VEC, S, C> atomic_pk_add(Value<VEC, S, C>* addr, Value<VEC, S, C> v)
        {
            using V = Value<VEC, S, C>;
            static_assert(sizeof(V) == sizeof(unsigned),
                          "packed atomic expects a 32-bit (2x16-bit) Value");
            auto*    p   = reinterpret_cast<unsigned*>(addr);
            unsigned old = *p;
            for(;;)
            {
                const V oldV = V::from_storage_bits(__builtin_bit_cast(typename V::bits_type, old));
                const V newV = oldV + v;
                const unsigned newbits = __builtin_bit_cast(unsigned, newV.to_storage_bits());
                const unsigned witness = atomicCAS(p, old, newbits);
                if(witness == old)
                    return oldV;
                old = witness;
            }
        }
    } // namespace detail

    template <Semantics S, Conversions C>
    FPSAN_DEVICE Value<v2h_native, S, C> amdgcn_atomic_pk_add_f16(Value<v2h_native, S, C>* addr,
                                                                  Value<v2h_native, S, C>  v)
    {
        return detail::atomic_pk_add<v2h_native, S, C>(addr, v);
    }

    template <Semantics S, Conversions C>
    FPSAN_DEVICE Value<v2bf_native, S, C> amdgcn_atomic_pk_add_bf16(Value<v2bf_native, S, C>* addr,
                                                                    Value<v2bf_native, S, C>  v)
    {
        return detail::atomic_pk_add<v2bf_native, S, C>(addr, v);
    }

} // namespace fpsan

#endif // FPSAN_AMDGCN_ATOMIC_HPP
