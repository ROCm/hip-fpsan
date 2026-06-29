// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan/io.hpp
// ----------------------------------------------------------------------------
// Optional host-only iostream support. Kept out of the umbrella header so the
// core stays free of <ostream> for device builds; include explicitly:
//   #include <fpsan/io.hpp>
// What gets printed is the payload in the terms that are meaningful for the
// flavor -- never a reconstructed "float", which the sanitizing payloads do not
// generally carry:
//   * Native    : the underlying float value (it IS a real float).
//   * Triton    : the payload as raw hex bits (a scramble in the free ring
//                 Z/2^w; there is no value to recover).
//   * algebraic : the payload as "x mod n" -- the residue x in its finite ring
//                 Z/nZ together with the modulus n.
// ----------------------------------------------------------------------------
#ifndef FPSAN_IO_HPP
#define FPSAN_IO_HPP

#include "fpsan/value.hpp"

#include <ios>
#include <ostream>

namespace fpsan {

template <class FT, Semantics S, Conversions C>
std::ostream &operator<<(std::ostream &os, const Value<FT, S, C> &v) {
  // Lead with the exact Semantics enum-constant name, then the payload in the
  // terms meaningful for that flavor.
  os << semantics_name(S) << "(";
  if constexpr (S == Semantics::Native) {
    os << static_cast<double>(v.to_float());
  } else if constexpr (S == Semantics::Triton) {
    const std::ios_base::fmtflags save = os.flags();
    os << "payload=0x" << std::hex << +v.fpsan_payload();
    os.flags(save);
  } else // algebraic: the payload is a residue in Z/nZ
  {
    os << "payload=" << +v.fpsan_payload() << " mod " << Value<FT, S, C>::alg_cfg().n;
  }
  os << ")";
  return os;
}

} // namespace fpsan

#endif // FPSAN_IO_HPP
