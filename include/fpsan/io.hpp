// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan/io.hpp
// ----------------------------------------------------------------------------
// Optional host-only iostream support. Kept out of the umbrella header so the
// core stays free of <ostream> for device builds; include explicitly:
//   #include <fpsan/io.hpp>
// Prints the represented float; in FPSan mode also the integer payload, since
// the float value alone is scrambled and not informative.
// ----------------------------------------------------------------------------
#ifndef FPSAN_IO_HPP
#define FPSAN_IO_HPP

#include "fpsan/value.hpp"

#include <ostream>

namespace fpsan
{

    template <class FT, Semantics S, Conversions C>
    std::ostream& operator<<(std::ostream& os, const Value<FT, S, C>& v)
    {
        if constexpr(S == Semantics::Triton)
            os << "fpsan(payload=" << +v.fpsan_payload()
               << ", unembed=" << static_cast<double>(v.to_float()) << ")";
        else
            os << static_cast<double>(v.to_float());
        return os;
    }

} // namespace fpsan

#endif // FPSAN_IO_HPP
