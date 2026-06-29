// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan/cast.hpp
// ----------------------------------------------------------------------------
// Precision casts between scalar Values, fpsan::cast<ToFT>(Value<FromFT,...>).
//
//   Semantics::Native : the native conversion (static_cast<ToFT>).
//   Semantics::Triton : Triton's fp-cast semantics for arith.extf / arith.truncf
//                      / tt.fp_to_fp, namely
//                        embed(src)  ->  signed-resize the payload to the
//                        destination width  ->  unembed(dst)
//                      i.e. ExtSI when widening, truncate when narrowing.
//
// This is what makes mixed-precision FPSan code (e.g. an f16-input,
// f32-accumulate matmul) agree with a Triton/scalar reference.
// ----------------------------------------------------------------------------
#ifndef FPSAN_CAST_HPP
#define FPSAN_CAST_HPP

#include "fpsan/detail/config.hpp"
#include "fpsan/value.hpp"

#include <type_traits>

namespace fpsan {

template <class ToFT, class FromFT, Semantics S, Conversions C>
FPSAN_HOST_DEVICE Value<ToFT, S, C> cast(Value<FromFT, S, C> v) {
  using To = Value<ToFT, S, C>;
  static_assert(!Value<FromFT, S, C>::is_vector && !To::is_vector,
                "fpsan::cast currently supports scalar Values only");
  if constexpr (S == Semantics::Native) {
    return To(static_cast<ToFT>(v.to_float()));
  } else if constexpr (detail::is_algebraic_semantics(S)) {
    // Algebraic variants: a per-width modulus makes casts un-faithful by
    // construction (widening is information-theoretically uncomputable from
    // the narrow residue; see algebraic-fpsan.md). The
    // FieldWithMulCasts variants opt into the expensive
    // multiplicative cast tower, including format-distinguishing maps for
    // same-width families such as f16/bf16 and e4m3/e5m2. All other
    // algebraic variants use a deterministic, in-range convention -- Inf/NaN
    // map across, and a finite residue reduces mod the destination modulus.
    using ToBits = typename To::bits_type;
    const auto from = Value<FromFT, S, C>::alg_cfg();
    const auto to = To::alg_cfg();
    const auto p = v.fpsan_payload();
    if constexpr (detail::has_multiplicative_field_casts(S)) {
      return To::from_fpsan_payload(static_cast<ToBits>(detail::alg_cast1(from, to, p)));
    } else {
      if (from.has_inf_nan && p == from.inf_code)
        return To::from_fpsan_payload(static_cast<ToBits>(to.inf_code));
      if (from.has_inf_nan && p == from.nan_code)
        return To::from_fpsan_payload(static_cast<ToBits>(to.nan_code));
      return To::from_fpsan_payload(static_cast<ToBits>(p % to.n));
    }
  } else {
    using FromBits = typename Value<FromFT, S, C>::bits_type;
    using ToBits = typename To::bits_type;
    // Reinterpret the source payload as signed, then convert to the destination
    // signed width (sign-extends on widen, truncates on narrow), then back to
    // unsigned -- matching Triton's castSignedIntValueToType on the payload.
    const auto signed_src = static_cast<std::make_signed_t<FromBits>>(v.fpsan_payload());
    const auto resized = static_cast<std::make_signed_t<ToBits>>(signed_src);
    return To::from_fpsan_payload(static_cast<ToBits>(resized));
  }
}

} // namespace fpsan

#endif // FPSAN_CAST_HPP
