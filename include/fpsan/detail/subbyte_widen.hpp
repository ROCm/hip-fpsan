// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan/detail/subbyte_widen.hpp
// ----------------------------------------------------------------------------
// FPSan widen of a sub-byte payload field to a 32-bit f32 payload.
//
// In FPSan mode a narrow->wide float conversion is a signed resize of the
// payload (the Triton ExtSI model), purely integer-level. This single helper
// implements that for any field width and is shared by the scaled-conversion
// wrappers (amdgcn_cvt.hpp: fp4/fp6 unpack) and the scaled/sub-byte MFMA
// dataflow (amdgcn_mfma.hpp), which both decode packed fp4/fp6 codes.
// ----------------------------------------------------------------------------
#ifndef FPSAN_DETAIL_SUBBYTE_WIDEN_HPP
#define FPSAN_DETAIL_SUBBYTE_WIDEN_HPP

#include "fpsan/value.hpp"

#include <cstdint>

namespace fpsan
{
    namespace detail
    {
        // Sign-resize a Width-bit payload (low Width bits of `field`) to a 32-bit
        // f32 payload. Width is the source format's bit count (4 for fp4, 6 for
        // fp6); the high bits of `field` are ignored.
        template <int Width, Semantics S, Conversions C>
        FPSAN_DEVICE Value<float, S, C> subbyte_widen(std::uint32_t field)
        {
            const std::int32_t e = static_cast<std::int32_t>(field << (32 - Width)) >> (32 - Width);
            return Value<float, S, C>::from_fpsan_payload(static_cast<std::uint32_t>(e));
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
