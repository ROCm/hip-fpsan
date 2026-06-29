// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan/numeric_limits.hpp
// ----------------------------------------------------------------------------
// std::numeric_limits<Value<...>> so that ported code querying limits (max(),
// epsilon(), infinity(), ...) keeps working. Values forward to the underlying
// float type and are returned wrapped (embedded in FPSan mode). The
// boundary/special values are the underlying float's; what changes under FPSan
// is the *arithmetic*, not which floats exist.
// ----------------------------------------------------------------------------
#ifndef FPSAN_NUMERIC_LIMITS_HPP
#define FPSAN_NUMERIC_LIMITS_HPP

#include "fpsan/value.hpp"

#include <limits>

namespace std {

template <class FT, fpsan::Semantics S, fpsan::Conversions C>
class numeric_limits<fpsan::Value<FT, S, C>> {
  using W = fpsan::Value<FT, S, C>;
  using base = std::numeric_limits<FT>;

public:
  static constexpr bool is_specialized = base::is_specialized;
  static constexpr bool is_signed = base::is_signed;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = base::has_infinity;
  static constexpr bool has_quiet_NaN = base::has_quiet_NaN;
  static constexpr bool has_signaling_NaN = base::has_signaling_NaN;
  static constexpr std::float_denorm_style has_denorm = base::has_denorm;
  static constexpr bool is_bounded = base::is_bounded;
  // A wrapper is not literally the IEC559 type, and FPSan mode is certainly
  // not.
  static constexpr bool is_iec559 = base::is_iec559 && (S == fpsan::Semantics::Native);
  static constexpr int digits = base::digits;
  static constexpr int digits10 = base::digits10;
  static constexpr int max_digits10 = base::max_digits10;
  static constexpr int radix = base::radix;
  static constexpr int min_exponent = base::min_exponent;
  static constexpr int max_exponent = base::max_exponent;
  static constexpr int min_exponent10 = base::min_exponent10;
  static constexpr int max_exponent10 = base::max_exponent10;
  static constexpr bool traps = base::traps;

  FPSAN_HOST_DEVICE static constexpr W min() noexcept { return W(base::min()); }
  FPSAN_HOST_DEVICE static constexpr W max() noexcept { return W(base::max()); }
  FPSAN_HOST_DEVICE static constexpr W lowest() noexcept { return W(base::lowest()); }
  FPSAN_HOST_DEVICE static constexpr W epsilon() noexcept { return W(base::epsilon()); }
  FPSAN_HOST_DEVICE static constexpr W round_error() noexcept { return W(base::round_error()); }
  FPSAN_HOST_DEVICE static constexpr W infinity() noexcept { return W(base::infinity()); }
  FPSAN_HOST_DEVICE static constexpr W quiet_NaN() noexcept { return W(base::quiet_NaN()); }
  FPSAN_HOST_DEVICE static constexpr W signaling_NaN() noexcept { return W(base::signaling_NaN()); }
  FPSAN_HOST_DEVICE static constexpr W denorm_min() noexcept { return W(base::denorm_min()); }
};

// cv-qualified specializations forward to the unqualified one.
template <class FT, fpsan::Semantics S, fpsan::Conversions C>
class numeric_limits<const fpsan::Value<FT, S, C>> : public numeric_limits<fpsan::Value<FT, S, C>> {
};
template <class FT, fpsan::Semantics S, fpsan::Conversions C>
class numeric_limits<volatile fpsan::Value<FT, S, C>>
    : public numeric_limits<fpsan::Value<FT, S, C>> {};
template <class FT, fpsan::Semantics S, fpsan::Conversions C>
class numeric_limits<const volatile fpsan::Value<FT, S, C>>
    : public numeric_limits<fpsan::Value<FT, S, C>> {};

} // namespace std

#endif // FPSAN_NUMERIC_LIMITS_HPP
