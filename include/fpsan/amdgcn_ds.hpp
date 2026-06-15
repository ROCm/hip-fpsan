// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan/amdgcn_ds.hpp
// ----------------------------------------------------------------------------
// FPSan wrappers for the gfx950 (CDNA4) matrix-transposed LDS reads:
//
//   ds_read_tr16_b64  (16-bit elements: f16 / bf16)
//   ds_read_tr8_b64   (8-bit elements: fp8 / bf8)
//   ds_read_tr4_b64   (4-bit elements: fp4 e2m1; 16 codes / 64 bits per lane)
//   ds_read_tr6_b96   (6-bit elements: fp6 e2m3 / bf6 e3m2; 16 codes / 96 bits)
//
// These instructions read a per-lane fragment from LDS (shared memory) with the
// cooperative-transpose lane mapping the MFMA operand layout expects, so a
// matrix tile staged in LDS can be fed straight into the MFMA wrappers in
// fpsan/amdgcn_mfma.hpp without a manual transpose.
//
// They are *pure data movement*: no arithmetic touches the loaded values, the
// hardware just relocates bits across lanes. So Float mode and FPSan mode share
// one implementation -- the lane's stored bits (IEEE float bits in Float mode,
// the FPSan payload in FPSan mode) are moved verbatim and rebuilt with
// Value::from_storage_bits. Whatever was staged in LDS (a Value's storage) is
// exactly what comes back, transposed.
//
// ABI note. The builtins take a pointer to the result vector *in LDS*
// (address space 3). HIP hands out generic pointers for __shared__ data, and a
// generic pointer does not implicitly convert to addrspace(3) across a function
// boundary, so each wrapper takes a generic Value pointer and does the
// addrspace conversion itself with a C-style cast inside the body. That cast is
// only valid when `lds` actually points into __shared__ memory -- the
// instruction's contract. The builtin call is gated to the device pass; in the
// HIP host pass the wrapper is a visible stub (host code never executes it but
// must parse it), matching the convention used across the amdgcn_* headers.
//
// Wave64.  HIP/device-only.  Opt-in (not pulled by <fpsan/fpsan.hpp>).
// ----------------------------------------------------------------------------
#ifndef FPSAN_AMDGCN_DS_HPP
#define FPSAN_AMDGCN_DS_HPP

#include "fpsan/amdgcn_matrix.hpp" // v8e4m3_native / v8e5m2_native + fp8 types
#include "fpsan/detail/traits.hpp" // detail::vector_element_t
#include "fpsan/value.hpp"

#include <cstdint>

#if !defined(__HIP__) && !defined(__CUDACC__)
#error "fpsan/amdgcn_ds.hpp is GPU-only; compile as HIP (or CUDA)."
#endif

namespace fpsan
{

    // 4-element native vectors for the 16-bit transposed reads.
    using v4h_native  = _Float16 __attribute__((ext_vector_type(4)));
    using v4bf_native = __bf16 __attribute__((ext_vector_type(4)));

#define FPSAN_DS_LDS __attribute__((address_space(3)))

// Run `block` only in the device pass (the host pass keeps a visible stub so
// the wrappers parse, but never type-checks the LDS builtin call).
#ifdef __HIP_DEVICE_COMPILE__
#define FPSAN_DS_DEVICE_ONLY(block) block
#else
#define FPSAN_DS_DEVICE_ONLY(block)
#endif

// ---- ds_read_tr16_b64 : 16-bit elements (f16 / bf16) -----------------------
// Reads 4 consecutive 16-bit values per lane from LDS, transposed. `lds` points
// at this lane's slot in a Value<FT> array (length >= 4) in __shared__ memory.
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_ds_read_tr16_b64_v4i16)
#define FPSAN_DEFINE_DS_READ_TR16(NAME, FT, VEC)                                               \
    template <Semantics S, Conversions C>                                                      \
    FPSAN_DEVICE Value<VEC, S, C> NAME(const Value<FT, S, C>* lds)                             \
    {                                                                                          \
        using Bits = typename Value<VEC, S, C>::bits_type;                                     \
        Bits bits{};                                                                           \
        (void)lds;                                                                             \
        FPSAN_DS_DEVICE_ONLY({                                                                 \
            using v4i16 = short __attribute__((ext_vector_type(4)));                           \
            auto raw    = __builtin_amdgcn_ds_read_tr16_b64_v4i16((v4i16 FPSAN_DS_LDS*)(lds)); \
            bits        = __builtin_bit_cast(Bits, raw);                                       \
        })                                                                                     \
        return Value<VEC, S, C>::from_storage_bits(bits);                                      \
    }
    FPSAN_DEFINE_DS_READ_TR16(amdgcn_ds_read_tr16_b64_f16, _Float16, v4h_native)
    FPSAN_DEFINE_DS_READ_TR16(amdgcn_ds_read_tr16_b64_bf16, __bf16, v4bf_native)
#undef FPSAN_DEFINE_DS_READ_TR16
#endif

// ---- ds_load_tr16_b128 : 16-bit elements (f16 / bf16), wave32 --------------
// gfx1250's LDS transpose load: 128 bits (8 x 16-bit) per lane, the LDS analogue
// of gfx1250's global_load_tr16_b128 and the wave32/b128 successor to gfx950's
// b64 ds_read_tr16 (4 elements). Still PURE bit movement, so Float and FPSan
// share one body. `lds` points at this lane's slot in a Value<FT> array
// (length >= 8) in __shared__ memory; returns the transposed v8 fragment.
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_ds_load_tr16_b128_v8f16)
#define FPSAN_DEFINE_DS_LOAD_TR16_B128(NAME, FT, VEC, BUILTIN, RAWVEC) \
    template <Semantics S, Conversions C>                              \
    FPSAN_DEVICE Value<VEC, S, C> NAME(const Value<FT, S, C>* lds)     \
    {                                                                  \
        using Bits = typename Value<VEC, S, C>::bits_type;             \
        Bits bits{};                                                   \
        (void)lds;                                                     \
        FPSAN_DS_DEVICE_ONLY({                                         \
            auto raw = BUILTIN((RAWVEC FPSAN_DS_LDS*)(lds));           \
            bits     = __builtin_bit_cast(Bits, raw);                  \
        })                                                             \
        return Value<VEC, S, C>::from_storage_bits(bits);              \
    }
    // The half builtin is typed on Clang's vendor `__fp16`, layout-compatible
    // with our `_Float16`; route the pointer cast through `__fp16` ext_vector.
    using v8fp16_ds = __fp16 __attribute__((ext_vector_type(8)));
    FPSAN_DEFINE_DS_LOAD_TR16_B128(amdgcn_ds_load_tr16_b128_f16,
                                   _Float16,
                                   v8h_native,
                                   __builtin_amdgcn_ds_load_tr16_b128_v8f16,
                                   v8fp16_ds)
    FPSAN_DEFINE_DS_LOAD_TR16_B128(amdgcn_ds_load_tr16_b128_bf16,
                                   __bf16,
                                   v8bf_native,
                                   __builtin_amdgcn_ds_load_tr16_b128_v8bf16,
                                   v8bf_native)
#undef FPSAN_DEFINE_DS_LOAD_TR16_B128
#endif

// ---- ds_read_tr8_b64 : 8-bit elements (fp8 / bf8) --------------------------
// Reads 8 consecutive bytes per lane from LDS, transposed -- one v8 fp8/bf8
// operand fragment, the shape the fp8 MFMA wrappers consume.
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_ds_read_tr8_b64_v2i32)
#define FPSAN_DEFINE_DS_READ_TR8(NAME, FRAG)                                                    \
    template <Semantics S, Conversions C>                                                       \
    FPSAN_DEVICE Value<FRAG, S, C> NAME(const Value<detail::vector_element_t<FRAG>, S, C>* lds) \
    {                                                                                           \
        using Bits = typename Value<FRAG, S, C>::bits_type;                                     \
        Bits bits{};                                                                            \
        (void)lds;                                                                              \
        FPSAN_DS_DEVICE_ONLY({                                                                  \
            using v2i32 = int __attribute__((ext_vector_type(2)));                              \
            auto raw    = __builtin_amdgcn_ds_read_tr8_b64_v2i32((v2i32 FPSAN_DS_LDS*)(lds));   \
            bits        = __builtin_bit_cast(Bits, raw);                                        \
        })                                                                                      \
        return Value<FRAG, S, C>::from_storage_bits(bits);                                      \
    }
    FPSAN_DEFINE_DS_READ_TR8(amdgcn_ds_read_tr8_b64_fp8, v8e4m3_native)
    FPSAN_DEFINE_DS_READ_TR8(amdgcn_ds_read_tr8_b64_bf8, v8e5m2_native)
#undef FPSAN_DEFINE_DS_READ_TR8
#endif

// ---- ds_load_tr8_b64 : 8-bit elements (fp8 / bf8), gfx1250 -----------------
// gfx1250's re-encoded successor to gfx950's ds_read_tr8_b64: still 64 bits
// (8 bytes) per lane transposed into one v8 fp8/bf8 fragment, the shape the
// fp8 WMMA wrappers consume. Pure bit movement (Float and FPSan share a body);
// kept distinct because the gfx950 ds_read_tr8 builtin is not available under
// --offload-arch=gfx1250.
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_ds_load_tr8_b64_v2i32)
#define FPSAN_DEFINE_DS_LOAD_TR8(NAME, FRAG)                                                    \
    template <Semantics S, Conversions C>                                                       \
    FPSAN_DEVICE Value<FRAG, S, C> NAME(const Value<detail::vector_element_t<FRAG>, S, C>* lds) \
    {                                                                                           \
        using Bits = typename Value<FRAG, S, C>::bits_type;                                     \
        Bits bits{};                                                                            \
        (void)lds;                                                                              \
        FPSAN_DS_DEVICE_ONLY({                                                                  \
            using v2i32 = int __attribute__((ext_vector_type(2)));                              \
            auto raw    = __builtin_amdgcn_ds_load_tr8_b64_v2i32((v2i32 FPSAN_DS_LDS*)(lds));   \
            bits        = __builtin_bit_cast(Bits, raw);                                        \
        })                                                                                      \
        return Value<FRAG, S, C>::from_storage_bits(bits);                                      \
    }
    FPSAN_DEFINE_DS_LOAD_TR8(amdgcn_ds_load_tr8_b64_fp8, v8e4m3_native)
    FPSAN_DEFINE_DS_LOAD_TR8(amdgcn_ds_load_tr8_b64_bf8, v8e5m2_native)
#undef FPSAN_DEFINE_DS_LOAD_TR8
#endif

    // ---- ds_read_tr4_b64 / ds_read_tr6_b96 : sub-byte (fp4 / fp6, bf6) ---------
    // These transpose 4-bit (tr4) and 6-bit (tr6) matrix elements across lanes,
    // exactly as tr8/tr16 do for 8-/16-bit elements -- still PURE bit movement, the
    // hardware just relocates whole sub-byte slots. fp4/fp6 are not byte multiples,
    // so the library has no `Value<fp4>`/`Value<fp6>` element type; sub-byte data
    // lives PACKED in dword containers (the same representation the fp4/fp6
    // cvt_scalef32 wrappers produce/consume): tr4 = 16 fp4 codes in 64 bits
    // (v2u32), tr6 = 16 fp6 codes in 96 bits (v3u32). In FPSan mode those slots
    // hold 4-/6-bit FPSan payloads occupying the identical bit positions an fp4/fp6
    // IEEE code would, so the same element-granular transpose relocates payloads
    // and codes identically -- hence one body for both semantics. The wrappers
    // therefore operate on the raw packed dwords (no Value element); the <S, C>
    // params are kept only for call-site uniformity with the rest of the library.
    using v2u32_native = unsigned __attribute__((ext_vector_type(2)));
    using v3u32_native = unsigned __attribute__((ext_vector_type(3)));

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_ds_read_tr4_b64_v2i32)
    // `lds` points at this lane's 64-bit (2-dword) packed-fp4 slot in __shared__.
    template <Semantics S, Conversions C>
    FPSAN_DEVICE v2u32_native amdgcn_ds_read_tr4_b64(const std::uint32_t* lds)
    {
        v2u32_native bits{};
        (void)lds;
        FPSAN_DS_DEVICE_ONLY({
            using v2i32 = int __attribute__((ext_vector_type(2)));
            auto raw    = __builtin_amdgcn_ds_read_tr4_b64_v2i32((v2i32 FPSAN_DS_LDS*)(lds));
            bits        = __builtin_bit_cast(v2u32_native, raw);
        })
        return bits;
    }
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_ds_read_tr6_b96_v3i32)
    // `lds` points at this lane's 96-bit (3-dword) packed-fp6/bf6 slot in
    // __shared__.
    template <Semantics S, Conversions C>
    FPSAN_DEVICE v3u32_native amdgcn_ds_read_tr6_b96(const std::uint32_t* lds)
    {
        v3u32_native bits{};
        (void)lds;
        FPSAN_DS_DEVICE_ONLY({
            using v3i32 = int __attribute__((ext_vector_type(3)));
            auto raw    = __builtin_amdgcn_ds_read_tr6_b96_v3i32((v3i32 FPSAN_DS_LDS*)(lds));
            bits        = __builtin_bit_cast(v3u32_native, raw);
        })
        return bits;
    }
#endif

    // ---- ds_load_tr4_b64 / ds_load_tr6_b96 : sub-byte (fp4 / fp6, bf6), gfx1250
    // gfx1250's re-encoded successors to gfx950's ds_read_tr4/tr6. Same packed
    // sub-byte transpose contract (tr4 = 16 fp4 codes in 64 bits / 2 dwords,
    // tr6 = 16 fp6 codes in 96 bits / 3 dwords), pure bit movement on raw
    // packed dwords; kept distinct because the gfx950 ds_read_tr4/tr6 builtins
    // are not available under --offload-arch=gfx1250.
#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_ds_load_tr4_b64_v2i32)
    // `lds` points at this lane's 64-bit (2-dword) packed-fp4 slot in __shared__.
    template <Semantics S, Conversions C>
    FPSAN_DEVICE v2u32_native amdgcn_ds_load_tr4_b64(const std::uint32_t* lds)
    {
        v2u32_native bits{};
        (void)lds;
        FPSAN_DS_DEVICE_ONLY({
            using v2i32 = int __attribute__((ext_vector_type(2)));
            auto raw    = __builtin_amdgcn_ds_load_tr4_b64_v2i32((v2i32 FPSAN_DS_LDS*)(lds));
            bits        = __builtin_bit_cast(v2u32_native, raw);
        })
        return bits;
    }
#endif

#if !defined(__HIP_DEVICE_COMPILE__) || __has_builtin(__builtin_amdgcn_ds_load_tr6_b96_v3i32)
    // `lds` points at this lane's 96-bit (3-dword) packed-fp6/bf6 slot in
    // __shared__.
    template <Semantics S, Conversions C>
    FPSAN_DEVICE v3u32_native amdgcn_ds_load_tr6_b96(const std::uint32_t* lds)
    {
        v3u32_native bits{};
        (void)lds;
        FPSAN_DS_DEVICE_ONLY({
            using v3i32 = int __attribute__((ext_vector_type(3)));
            auto raw    = __builtin_amdgcn_ds_load_tr6_b96_v3i32((v3i32 FPSAN_DS_LDS*)(lds));
            bits        = __builtin_bit_cast(v3u32_native, raw);
        })
        return bits;
    }
#endif

#undef FPSAN_DS_DEVICE_ONLY
#undef FPSAN_DS_LDS

} // namespace fpsan

#endif // FPSAN_AMDGCN_DS_HPP
