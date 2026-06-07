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
    } // namespace detail
} // namespace fpsan

#endif // FPSAN_DETAIL_SUBBYTE_WIDEN_HPP
