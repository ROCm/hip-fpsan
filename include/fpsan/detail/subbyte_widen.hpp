// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan/detail/subbyte_widen.hpp
// ----------------------------------------------------------------------------
// Deterministic widen of a sub-byte storage payload field to a 32-bit f32
// payload.
//
// FP4/FP6 are not scalar Value element types, so their packed gfx intrinsics use
// a storage-level convention rather than a value-faithful algebraic cast: treat
// the low Width payload bits as a signed storage integer, then encode that
// integer in the destination payload semantics. This helper implements that
// convention for the scaled-conversion wrappers (amdgcn_cvt.hpp: fp4/fp6 unpack)
// and the scaled/sub-byte matrix dataflows, which both decode packed fp4/fp6
// codes. Do not use it for FP8/BF8 Value types; those go through fpsan::cast.
// ----------------------------------------------------------------------------
#ifndef FPSAN_DETAIL_SUBBYTE_WIDEN_HPP
#define FPSAN_DETAIL_SUBBYTE_WIDEN_HPP

#include "fpsan/value.hpp"

#include <cstdint>

namespace fpsan
{
    namespace detail
    {
        // Sign-extend a storage payload field into the destination payload bits.
        // This is a storage-level resize, so even algebraic semantics preserve the
        // raw sign-extended bits instead of reducing them modulo the algebraic ring.
        // Width is the source storage field's bit count; the
        // high bits of `field` are ignored. Width==8 is reserved for storage-only
        // formats such as gfx1250 E5M3. Public FP8/BF8 Value formats must use
        // fpsan::cast so algebraic same-width format identity is preserved.
        template <int Width, Semantics S, Conversions C>
        FPSAN_HOST_DEVICE Value<float, S, C> storage_payload_widen(std::uint32_t field)
        {
            const std::int32_t e = static_cast<std::int32_t>(field << (32 - Width)) >> (32 - Width);
            return Value<float, S, C>::from_fpsan_payload(static_cast<std::uint32_t>(e));
        }

        // Sub-byte formats are storage-only, not scalar Value element types.
        template <int Width, Semantics S, Conversions C>
        FPSAN_HOST_DEVICE Value<float, S, C> subbyte_widen(std::uint32_t field)
        {
            static_assert(Width < 8, "public FP8/BF8 payloads must be widened through fpsan::cast");
            return storage_payload_widen<Width, S, C>(field);
        }

        template <class DstFT, int Width, Semantics S, Conversions C>
        FPSAN_HOST_DEVICE Value<DstFT, S, C> subbyte_widen_to(std::uint32_t field)
        {
            using Out = Value<DstFT, S, C>;
            return Out::from_fpsan_payload(static_cast<typename Out::bits_type>(
                subbyte_widen<Width, S, C>(field).fpsan_payload()));
        }

        // Element bit width of an f8f6f4 format immediate: fp8/bf8 (0,1) -> 8,
        // fp6/bf6 (2,3) -> 6, fp4 (4) -> 4. Shared by the gfx950 MFMA and gfx1250
        // WMMA f8f6f4 paths (both decode the same packed sub-byte fragments).
        FPSAN_HOST_DEVICE constexpr int f8f6f4_width(int code)
        {
            return (code <= 1) ? 8 : (code <= 3) ? 6 : 4;
        }

        // OCP-MX E8M0 block-scale -> float. E8M0 stores only an exponent (bias
        // 127): byte b -> 2^(b-127); 0xFF is NaN. Built by bit construction so it
        // is exact (no libm) and host/device safe. Shared by the gfx950 MFMA and
        // gfx1250 WMMA block-scaled paths.
        FPSAN_HOST_DEVICE inline float e8m0_to_float(unsigned byte)
        {
            if(byte == 0xFFu)
                return __builtin_nanf("");
            if(byte == 0u)
                return __builtin_bit_cast(float, std::uint32_t(0x00400000u)); // 2^-127
            return __builtin_bit_cast(float, static_cast<std::uint32_t>(byte) << 23);
        }
    } // namespace detail
} // namespace fpsan

#endif // FPSAN_DETAIL_SUBBYTE_WIDEN_HPP
