// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan/amdgcn_global_load.hpp
// ----------------------------------------------------------------------------
// FPSan wrappers for the gfx12 (RDNA4) matrix-transposed global loads:
//
//   global_load_tr_b128 (wave32, v8 of f16 / bf16 per lane;
//                        wave64, v4 of f16 / bf16 per lane)
//   global_load_tr_b64  (wave64, v4 of fp8 / bf8 per lane)
//
// These instructions read 128 bits per lane from global memory with the
// cooperative-transpose lane mapping the WMMA operand layout expects, so a
// matrix tile staged in global memory can be fed straight into the WMMA
// wrappers in fpsan/amdgcn_matrix.hpp without a manual transpose. They are
// the gfx12 analogue of gfx950's ds_read_tr* family, just with the LDS
// pointer replaced by a global one.
//
// Like ds_read_tr*, this is *pure data movement*: no arithmetic touches the
// loaded values, the hardware just relocates whole 16-bit slots across lanes
// in a defined permutation. So Float mode and FPSan mode share one
// implementation -- the lane's stored bits (IEEE float bits in Float mode,
// the FPSan payload in FPSan mode) are moved verbatim and rebuilt with
// Value::from_storage_bits. Whatever was staged in global memory (a Value's
// storage) is exactly what comes back, transposed.
//
// ABI note. The builtins take a pointer to a vector *in global memory*
// (address space 1) of the result element type. HIP hands out generic
// pointers for global data; the wrapper takes a generic Value<scalar>*
// pointer and does the addrspace + vector-type cast itself in the body.
// That cast is only valid when the pointer actually points into global
// memory -- the instruction's contract. The builtin call is gated to the
// device pass; in the HIP host pass the wrapper is a visible stub.
//
// Wavefront size. RDNA4 exposes the b128 form as v8 f16/bf16 on wave32 and
// v4 f16/bf16 on wave64. RDNA4 also exposes a wave64 b64 form as one i32 per
// lane, which is the 4-byte fp8/bf8 fragment shape. Each wrapper is gated on
// the matching builtin so wave32-only and wave64-only compiler surfaces stay
// separate.
//
// HIP/device-only. Opt-in (not pulled by <fpsan/fpsan.hpp>).
// ----------------------------------------------------------------------------
#ifndef FPSAN_AMDGCN_GLOBAL_LOAD_HPP
#define FPSAN_AMDGCN_GLOBAL_LOAD_HPP

#include "fpsan/amdgcn_matrix.hpp" // fragment aliases used by matrix loads
#include "fpsan/detail/traits.hpp" // detail::vector_element_t
#include "fpsan/value.hpp"

#include <cstdint>

#if !defined(__HIP__) && !defined(__CUDACC__)
#error "fpsan/amdgcn_global_load.hpp is GPU-only; compile as HIP (or CUDA)."
#endif

namespace fpsan
{

#define FPSAN_GL_GLOBAL __attribute__((address_space(1)))

// Run `block` only in the device pass (the host pass keeps a visible stub so
// the wrappers parse, but never type-checks the global-addrspace builtin call).
#ifdef __HIP_DEVICE_COMPILE__
#define FPSAN_GL_DEVICE_ONLY(block) block
#else
#define FPSAN_GL_DEVICE_ONLY(block)
#endif

    // ---- global_load_tr_b128 : 16-bit elements (f16 / bf16), wave32 ------------
    // Reads 8 consecutive 16-bit values per lane from global memory, transposed.
    // `gptr` points at this lane's slot in a Value<FT> array (length >= 8) in
    // global memory. Returns the transposed v8 fragment as a Value<vec>.
    //
    // The half builtin is typed on Clang's `__fp16` (vendor half-precision
    // ABI type), which is layout-compatible with but spelled differently from
    // our `_Float16` -- so the pointer cast goes through `__fp16 ext_vector`
    // before reaching the builtin.
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_global_load_tr_b128_v8f16)
    template <Semantics S, Conversions C>
    FPSAN_DEVICE Value<v8h_native, S, C>
                 amdgcn_global_load_tr_b128_f16(const Value<_Float16, S, C>* gptr)
    {
        using Bits = typename Value<v8h_native, S, C>::bits_type;
        Bits bits{};
        (void)gptr;
        FPSAN_GL_DEVICE_ONLY({
            using v8fp16 = __fp16 __attribute__((ext_vector_type(8)));
            auto raw = __builtin_amdgcn_global_load_tr_b128_v8f16((v8fp16 FPSAN_GL_GLOBAL*)(gptr));
            bits     = __builtin_bit_cast(Bits, raw);
        })
        return Value<v8h_native, S, C>::from_storage_bits(bits);
    }

    template <Semantics S, Conversions C>
    FPSAN_DEVICE Value<v8bf_native, S, C>
                 amdgcn_global_load_tr_b128_bf16(const Value<__bf16, S, C>* gptr)
    {
        using Bits = typename Value<v8bf_native, S, C>::bits_type;
        Bits bits{};
        (void)gptr;
        FPSAN_GL_DEVICE_ONLY({
            auto raw
                = __builtin_amdgcn_global_load_tr_b128_v8bf16((v8bf_native FPSAN_GL_GLOBAL*)(gptr));
            bits = __builtin_bit_cast(Bits, raw);
        })
        return Value<v8bf_native, S, C>::from_storage_bits(bits);
    }
#endif

    // ---- global_load_tr_b128 : 16-bit elements (f16 / bf16), wave64 -----------
    // Reads 4 consecutive 16-bit values per lane from global memory, transposed.
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_global_load_tr_b128_v4f16)
    template <Semantics S, Conversions C>
    FPSAN_DEVICE Value<v4h_native, S, C>
                 amdgcn_global_load_tr_b128_f16_w64(const Value<_Float16, S, C>* gptr)
    {
        using Bits = typename Value<v4h_native, S, C>::bits_type;
        Bits bits{};
        (void)gptr;
        FPSAN_GL_DEVICE_ONLY({
            using v4fp16 = __fp16 __attribute__((ext_vector_type(4)));
            auto raw = __builtin_amdgcn_global_load_tr_b128_v4f16((v4fp16 FPSAN_GL_GLOBAL*)(gptr));
            bits     = __builtin_bit_cast(Bits, raw);
        })
        return Value<v4h_native, S, C>::from_storage_bits(bits);
    }
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_global_load_tr_b128_v4bf16)
    template <Semantics S, Conversions C>
    FPSAN_DEVICE Value<v4bf_native, S, C>
                 amdgcn_global_load_tr_b128_bf16_w64(const Value<__bf16, S, C>* gptr)
    {
        using Bits = typename Value<v4bf_native, S, C>::bits_type;
        Bits bits{};
        (void)gptr;
        FPSAN_GL_DEVICE_ONLY({
            auto raw
                = __builtin_amdgcn_global_load_tr_b128_v4bf16((v4bf_native FPSAN_GL_GLOBAL*)(gptr));
            bits = __builtin_bit_cast(Bits, raw);
        })
        return Value<v4bf_native, S, C>::from_storage_bits(bits);
    }
#endif

    // ---- global_load_tr_b64 : 8-bit elements (fp8 / bf8), wave64 -------------
    // Reads 4 consecutive bytes per lane transposed into one v4 fp8/bf8 fragment.
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_global_load_tr_b64_i32)
#define FPSAN_DEFINE_GLOBAL_LOAD_TR_B64_W64(NAME, FRAG)                                          \
    template <Semantics S, Conversions C>                                                        \
    FPSAN_DEVICE Value<FRAG, S, C> NAME(const Value<detail::vector_element_t<FRAG>, S, C>* gptr) \
    {                                                                                            \
        using Bits = typename Value<FRAG, S, C>::bits_type;                                      \
        Bits bits{};                                                                             \
        (void)gptr;                                                                              \
        FPSAN_GL_DEVICE_ONLY({                                                                   \
            auto raw = __builtin_amdgcn_global_load_tr_b64_i32((int FPSAN_GL_GLOBAL*)(gptr));    \
            bits     = __builtin_bit_cast(Bits, raw);                                            \
        })                                                                                       \
        return Value<FRAG, S, C>::from_storage_bits(bits);                                       \
    }
    FPSAN_DEFINE_GLOBAL_LOAD_TR_B64_W64(amdgcn_global_load_tr_b64_fp8_w64, v4e4m3_native)
    FPSAN_DEFINE_GLOBAL_LOAD_TR_B64_W64(amdgcn_global_load_tr_b64_bf8_w64, v4e5m2_native)
#undef FPSAN_DEFINE_GLOBAL_LOAD_TR_B64_W64
#endif

    // ---- global_load_tr16_b128 : 16-bit elements (f16 / bf16), wave32 ----------
    // gfx1250's renamed/re-encoded successor to gfx12's global_load_tr_b128. Same
    // contract: 128 bits (8 x 16-bit) per lane from global memory with the WMMA
    // cooperative-transpose lane mapping, pure data movement (Float and FPSan move
    // bits identically). gfx1250 spells it `global_load_tr16_b128_v8{f16,bf16}`
    // (the "16" = 16-bit element transpose granule); kept as a distinct gfx1250
    // wrapper because the gfx12 builtin is not available under --offload-arch=gfx1250.
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_global_load_tr16_b128_v8f16)
    template <Semantics S, Conversions C>
    FPSAN_DEVICE Value<v8h_native, S, C>
                 amdgcn_global_load_tr16_b128_f16(const Value<_Float16, S, C>* gptr)
    {
        using Bits = typename Value<v8h_native, S, C>::bits_type;
        Bits bits{};
        (void)gptr;
        FPSAN_GL_DEVICE_ONLY({
            using v8fp16 = __fp16 __attribute__((ext_vector_type(8)));
            auto raw
                = __builtin_amdgcn_global_load_tr16_b128_v8f16((v8fp16 FPSAN_GL_GLOBAL*)(gptr));
            bits = __builtin_bit_cast(Bits, raw);
        })
        return Value<v8h_native, S, C>::from_storage_bits(bits);
    }

    template <Semantics S, Conversions C>
    FPSAN_DEVICE Value<v8bf_native, S, C>
                 amdgcn_global_load_tr16_b128_bf16(const Value<__bf16, S, C>* gptr)
    {
        using Bits = typename Value<v8bf_native, S, C>::bits_type;
        Bits bits{};
        (void)gptr;
        FPSAN_GL_DEVICE_ONLY({
            auto raw = __builtin_amdgcn_global_load_tr16_b128_v8bf16(
                (v8bf_native FPSAN_GL_GLOBAL*)(gptr));
            bits = __builtin_bit_cast(Bits, raw);
        })
        return Value<v8bf_native, S, C>::from_storage_bits(bits);
    }
#endif

    // ---- global_load_tr8_b64 : 8-bit elements (fp8 / bf8), wave32 -------------
    // gfx1250's 64-bit (8-byte) transposed global load, the global-memory analogue
    // of ds_load_tr8_b64 and the fp8/bf8 counterpart to global_load_tr16_b128.
    // Reads 8 consecutive bytes per lane transposed into one v8 fp8/bf8 fragment
    // (the shape the fp8 WMMA wrappers consume). Pure data movement: Float and
    // FPSan move bits identically. `gptr` points at this lane's slot in a
    // Value<elem> array (length >= 8) in global memory.
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_global_load_tr8_b64_v2i32)
#define FPSAN_DEFINE_GLOBAL_LOAD_TR8(NAME, FRAG)                                                   \
    template <Semantics S, Conversions C>                                                          \
    FPSAN_DEVICE Value<FRAG, S, C> NAME(const Value<detail::vector_element_t<FRAG>, S, C>* gptr)   \
    {                                                                                              \
        using Bits = typename Value<FRAG, S, C>::bits_type;                                        \
        Bits bits{};                                                                               \
        (void)gptr;                                                                                \
        FPSAN_GL_DEVICE_ONLY({                                                                     \
            using v2i32 = int __attribute__((ext_vector_type(2)));                                 \
            auto raw = __builtin_amdgcn_global_load_tr8_b64_v2i32((v2i32 FPSAN_GL_GLOBAL*)(gptr)); \
            bits     = __builtin_bit_cast(Bits, raw);                                              \
        })                                                                                         \
        return Value<FRAG, S, C>::from_storage_bits(bits);                                         \
    }
    FPSAN_DEFINE_GLOBAL_LOAD_TR8(amdgcn_global_load_tr8_b64_fp8, v8e4m3_native)
    FPSAN_DEFINE_GLOBAL_LOAD_TR8(amdgcn_global_load_tr8_b64_bf8, v8e5m2_native)
#undef FPSAN_DEFINE_GLOBAL_LOAD_TR8
#endif

#undef FPSAN_GL_DEVICE_ONLY
#undef FPSAN_GL_GLOBAL

} // namespace fpsan

#endif // FPSAN_AMDGCN_GLOBAL_LOAD_HPP
