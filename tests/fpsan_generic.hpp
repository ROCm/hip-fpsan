// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan_generic.hpp
// ----------------------------------------------------------------------------
// Independent ground-truth reference for the FPSan payload algebra, vendored
// into the test tree so that the tests depend on no code outside the repo.
//
// This is a plain run-time-parameterized reimplementation of the SAME FPSan
// recipe the library implements, ported verbatim from the Triton sources:
//   lib/Conversion/TritonInstrumentToLLVM/FpSanToLLVM.cpp  (phi / phi^{-1})
//   lib/Dialect/TritonInstrument/Transforms/FpSanitizer.cpp (div / exp / exp2)
// A `FPFormat` describes the bit layout and a `FPSanFloat` carries a payload
// tagged with its format. The library's tests cross-check their payloads
// against this second, independent implementation for every enumerable bit
// pattern, so the library is validated against Triton's algorithm rather than
// only against itself.
//
// Self-contained: only the standard library, C++17. Namespace fpsan_generic is
// kept for the historical name used throughout the tests.
// ----------------------------------------------------------------------------
#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace fpsan_generic {

// ---- modular inverse of an odd 64-bit number (same as the float32 reference) -------
inline uint64_t inv_odd_u64(uint64_t a) {
  uint64_t x = 2 - a;
  for (unsigned correctBits = 2; correctBits < 64; correctBits *= 2)
    x *= 2 - a * x;
  return x;
}

// ---------------------------------------------------------------------------
//  FPFormat: the bit layout of an IEEE-754-style binary float.
//
//  A code word is  [sign : 1][exponent : expBits][mantissa : mantBits].
//  Value of a normal (exp != 0, and exp != all-ones for inf/nan types):
//      (-1)^sign * 2^(exp - bias) * (1 + mantissa / 2^mantBits)
//  Subnormal (exp == 0):
//      (-1)^sign * 2^(1 - bias) * (mantissa / 2^mantBits)
//  If hasInfNan, the all-ones exponent encodes +/-Inf (mantissa 0) or NaN.
// ---------------------------------------------------------------------------
struct FPFormat {
  const char *name;
  unsigned bitWidth;
  unsigned expBits;
  unsigned mantBits;
  int bias;
  bool hasInfNan;

  // The IEEE bit pattern of +1.0 is  bias << mantBits  (exp = bias, mantissa
  // 0). Its number of trailing zero bits is mantBits, the FPSan xorshift
  // amount.
  uint32_t oneBits() const { return static_cast<uint32_t>(bias) << mantBits; }
};

namespace formats {
// FP4, OCP "MXFP4" element type: 1 sign, 2 exponent, 1 mantissa, bias 1, no
// inf/nan.  16 values: {0, .5, 1, 1.5, 2, 3, 4, 6} and their negatives.
inline constexpr FPFormat E2M1{"E2M1 (FP4)", 4, 2, 1, 1, false};
// FP8 E5M2: 1 sign, 5 exponent, 2 mantissa, bias 15, with inf/nan.  This is the
// "true IEEE-754-style" FP8 (same exponent/mantissa split philosophy as
// binary16/binary32, just narrower).
inline constexpr FPFormat E5M2{"E5M2 (FP8)", 8, 5, 2, 15, true};
// binary32, for cross-checking against the float32 reference.
inline constexpr FPFormat F32{"binary32", 32, 8, 23, 127, true};
} // namespace formats

// ---------------------------------------------------------------------------
//  Mixing configuration -- the width-generic version of MixConfig in
//  the float32 reference.  Magnitude arithmetic is modulo 2^(bitWidth-1); the sign bit
//  is carried separately.
// ---------------------------------------------------------------------------
struct MixConfig {
  unsigned bitWidth;
  uint64_t signMask; // top bit
  uint64_t magMask;  // low (bitWidth-1) bits = 2^(bitWidth-1) - 1
  uint64_t fullMask; // all bitWidth bits = 2^bitWidth - 1 (the ring modulus - 1)
  unsigned shift;    // xorshift amount = mantissa width
  uint64_t mulA, mulAInv, mulBPos, mulBNeg, mulBPosInv, mulBNegInv;
};

inline MixConfig make_mix_config(const FPFormat &fmt) {
  MixConfig c;
  c.bitWidth = fmt.bitWidth;
  c.signMask = uint64_t{1} << (fmt.bitWidth - 1);
  c.magMask = c.signMask - 1;
  c.fullMask = (uint64_t{1} << fmt.bitWidth) - 1;

  uint64_t oneBits = fmt.oneBits();
  // C++17: std::countr_zero is C++20; __builtin_ctzll matches it for nonzero
  // input (oneBits is the bit pattern of 1.0, always nonzero here).
  c.shift = oneBits ? __builtin_ctzll(oneBits) : 64;

  c.mulA = 922291u & c.magMask;
  uint64_t oneMixed = (oneBits * c.mulA) & c.magMask;
  oneMixed ^= oneMixed >> c.shift;
  assert((oneMixed & 1) == 1 && "expected odd mixed 1.0");

  c.mulBPos = inv_odd_u64(oneMixed) & c.magMask;
  // magMask == -1 mod 2^(bitWidth-1), so mulBNeg == -mulBPos: this is what
  // makes phi(-x) = -phi(x).
  c.mulBNeg = (c.mulBPos * c.magMask) & c.magMask;

  c.mulAInv = inv_odd_u64(c.mulA) & c.magMask;
  c.mulBPosInv = inv_odd_u64(c.mulBPos) & c.magMask;
  c.mulBNegInv = inv_odd_u64(c.mulBNeg) & c.magMask;
  return c;
}

// Two formats yield the same MixConfig iff their layout fields agree (the
// `name` is descriptive only and make_mix_config ignores it).
inline bool same_layout(const FPFormat &a, const FPFormat &b) {
  return a.bitWidth == b.bitWidth && a.expBits == b.expBits && a.mantBits == b.mantBits &&
         a.bias == b.bias && a.hasInfNan == b.hasInfNan;
}

// Memoized lookup so repeated ops on a format don't rebuild the config.  Keyed
// by layout *value*, not by address: callers routinely pass short-lived
// FPFormat temporaries, and distinct temporaries can reuse the same stack
// address -- an address key then returns a stale, wrong-format config (and the
// dangling pointer is only compared, never dereferenced, so ASan stays silent).
inline const MixConfig &mix_config_for(const FPFormat &fmt) {
  static std::vector<std::pair<FPFormat, MixConfig>> cache;
  for (auto &e : cache)
    if (same_layout(e.first, fmt))
      return e.second;
  cache.emplace_back(fmt, make_mix_config(fmt));
  return cache.back().second;
}

// ---- phi / phi^{-1}, width-generic (cf. mixFloatToInt / unmixIntToFloat) ---
inline uint32_t mix_float_bits_to_int(const MixConfig &c, uint32_t u) {
  uint64_t signFlip = (u & c.signMask) ? c.signMask : 0u;
  uint64_t x = u ^ signFlip;
  uint64_t y = (x * c.mulA) & c.magMask;
  uint64_t z = y ^ (y >> c.shift);
  uint64_t mulB = (u & c.signMask) ? c.mulBNeg : c.mulBPos;
  uint64_t w = (z * mulB) & c.magMask;
  return static_cast<uint32_t>(w ^ signFlip);
}

inline uint32_t unmix_int_to_float_bits(const MixConfig &c, uint32_t v) {
  uint64_t signFlip = (v & c.signMask) ? c.signMask : 0u;
  uint64_t w = v ^ signFlip;
  uint64_t mulBInv = (v & c.signMask) ? c.mulBNegInv : c.mulBPosInv;
  uint64_t z = (w * mulBInv) & c.magMask;
  uint64_t y = z;
  for (unsigned shift = c.shift; shift < c.bitWidth; shift *= 2)
    y = y ^ (y >> shift);
  uint64_t x = (y * c.mulAInv) & c.magMask;
  return static_cast<uint32_t>(x ^ signFlip);
}

// ---- division helper inv() (cf. fpsanIntInv), width-generic ----------------
inline uint32_t fpsan_int_inv(const MixConfig &c, uint32_t u) {
  uint64_t a = u | 1u;
  uint64_t x = 2u - a;
  for (unsigned correctBits = 2; correctBits < c.bitWidth; correctBits *= 2)
    x = (x * (2u - a * x)) & c.fullMask;
  uint64_t evenPart = x & (c.fullMask ^ 1u); // drop low bit
  uint64_t originalParity = u & 1u;
  return static_cast<uint32_t>(evenPart | originalParity);
}

// ---- modular exponentiation (cf. fpsanExp2FromInt / fpsanExp) --------------
// The 32-bit constants are reused, truncated to the format width.  Note
// 0xA343836D == 5 (mod 8), and truncation preserves the low bits, so the
// generator stays == 5 (mod 8) for every width.
inline uint32_t fpsan_exp2_payload(const MixConfig &c, uint32_t xI) {
  uint64_t C = 0xA343836Du & c.fullMask;
  uint64_t y = 1u;
  for (int i = c.bitWidth - 1; i >= 0; --i) {
    y = (y * y) & c.fullMask;
    uint64_t bit = uint64_t{1} << i;
    uint64_t factor = (xI & bit) ? C : 1u;
    y = (y * factor) & c.fullMask;
  }
  return static_cast<uint32_t>(y);
}
inline uint32_t fpsan_exp_payload(const MixConfig &c, uint32_t xI) {
  uint64_t kRcpLog2 = 0x236EE9BFu & c.fullMask;
  uint32_t scaled = static_cast<uint32_t>((uint64_t{xI} * kRcpLog2) & c.fullMask);
  return fpsan_exp2_payload(c, scaled);
}

// ===========================================================================
//  FPSanFloat: a value in the FPSan payload algebra of a chosen format.
//  Holds the format plus the integer payload; arithmetic is modular integer
//  arithmetic mod 2^bitWidth, exactly as in the float32 reference but width-generic.
// ===========================================================================
class FPSanFloat {
public:
  FPSanFloat(const FPFormat &fmt, uint32_t payload)
      : fmt_(&fmt), payload_(payload & mix_config_for(fmt).fullMask) {}

  // phi: embed a raw bit pattern (the IEEE code word) of `fmt`.
  static FPSanFloat embed(const FPFormat &fmt, uint32_t rawBits) {
    const MixConfig &c = mix_config_for(fmt);
    return FPSanFloat(fmt, mix_float_bits_to_int(c, rawBits & c.fullMask));
  }

  const FPFormat &format() const { return *fmt_; }
  uint32_t payload() const { return payload_; }
  // phi^{-1}: the raw bit pattern this payload unembeds to.
  uint32_t raw_bits() const { return unmix_int_to_float_bits(cfg(), payload_); }

  FPSanFloat operator+(const FPSanFloat &o) const { return wrap(payload_ + o.payload_); }
  FPSanFloat operator-(const FPSanFloat &o) const { return wrap(payload_ - o.payload_); }
  FPSanFloat operator*(const FPSanFloat &o) const { return wrap(payload_ * o.payload_); }
  FPSanFloat operator/(const FPSanFloat &o) const {
    return wrap(payload_ * fpsan_int_inv(cfg(), o.payload_));
  }
  FPSanFloat operator-() const { return wrap(0u - payload_); }

  FPSanFloat exp() const { return wrap(fpsan_exp_payload(cfg(), payload_)); }
  FPSanFloat exp2() const { return wrap(fpsan_exp2_payload(cfg(), payload_)); }

  bool operator==(const FPSanFloat &o) const { return fmt_ == o.fmt_ && payload_ == o.payload_; }
  bool operator!=(const FPSanFloat &o) const { return !(*this == o); }

private:
  const MixConfig &cfg() const { return mix_config_for(*fmt_); }
  FPSanFloat wrap(uint64_t p) const {
    return FPSanFloat(*fmt_, static_cast<uint32_t>(p & cfg().fullMask));
  }
  const FPFormat *fmt_;
  uint32_t payload_;
};

// ===========================================================================
//  Decoding helpers: turn a raw code word into the real number it denotes (and
//  a printable name).  Used by the demo / table generator.
// ===========================================================================
struct Decoded {
  double value = 0.0; // meaningful only when finite
  bool sign = false;
  unsigned expField = 0;
  uint32_t mantField = 0;
  bool isZero = false;
  bool isSubnormal = false;
  bool isInf = false;
  bool isNaN = false;
};

inline Decoded decode(const FPFormat &fmt, uint32_t bits) {
  Decoded d;
  unsigned bw = fmt.bitWidth;
  uint32_t full = static_cast<uint32_t>((uint64_t{1} << bw) - 1);
  bits &= full;
  d.sign = (bits >> (bw - 1)) & 1u;
  d.expField = (bits >> fmt.mantBits) & ((1u << fmt.expBits) - 1u);
  d.mantField = bits & ((1u << fmt.mantBits) - 1u);
  unsigned expMax = (1u << fmt.expBits) - 1u;
  double mantFrac = static_cast<double>(d.mantField) / (1u << fmt.mantBits);

  if (d.expField == expMax && fmt.hasInfNan) {
    if (d.mantField == 0)
      d.isInf = true;
    else
      d.isNaN = true;
    return d;
  }
  double mag;
  if (d.expField == 0) {
    d.isSubnormal = (d.mantField != 0);
    d.isZero = (d.mantField == 0);
    // 2^(1 - bias) * mantFrac
    mag = mantFrac;
    int e = 1 - fmt.bias;
    mag = mag * (e >= 0 ? double(uint64_t{1} << e) : 1.0 / double(uint64_t{1} << -e));
  } else {
    int e = int(d.expField) - fmt.bias;
    mag = (1.0 + mantFrac);
    mag = mag * (e >= 0 ? double(uint64_t{1} << e) : 1.0 / double(uint64_t{1} << -e));
  }
  d.value = d.sign ? -mag : mag;
  return d;
}

inline std::string value_name(const FPFormat &fmt, uint32_t bits) {
  Decoded d = decode(fmt, bits);
  if (d.isInf)
    return d.sign ? "-Inf" : "+Inf";
  if (d.isNaN)
    return "NaN";
  if (d.isZero)
    return d.sign ? "-0" : "0";
  char buf[64];
  std::snprintf(buf, sizeof buf, "%g", d.value);
  return buf;
}

// Brute-force "encode the representable value closest to v" by scanning all
// code words.  Only sensible for narrow formats (bitWidth small); used to set
// up worked examples.  Ties and special values are skipped.
inline uint32_t encode_nearest(const FPFormat &fmt, double v) {
  uint32_t full = static_cast<uint32_t>((uint64_t{1} << fmt.bitWidth) - 1);
  uint32_t best = 0;
  double bestErr = 1e300;
  for (uint32_t bits = 0; bits <= full; ++bits) {
    Decoded d = decode(fmt, bits);
    if (d.isInf || d.isNaN)
      continue;
    double err = d.value - v;
    if (err < 0)
      err = -err;
    if (err < bestErr) {
      bestErr = err;
      best = bits;
    }
  }
  return best;
}

} // namespace fpsan_generic
