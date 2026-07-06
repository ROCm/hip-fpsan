// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// fpsan/detail/algebraic.hpp
// ----------------------------------------------------------------------------
// Finite-ring implementation for the algebraic FPSan semantics. See
// docs/algebraic-fpsan.md for the user-facing guide and
// docs/reducing-floats-mod-p.{tex,pdf} for the math write-up. Where
// Semantics::Triton scrambles the float bits into Z/2^w (mix.hpp's ring_*), the
// algebraic semantics carry the genuine residue
//
//     phi_n(m * 2^e) = m * 2^e   in  Z/nZ,
//
// interpreting negative e using 2^{-1} mod n, of the value's exact dyadic
// rational, for a per-element-width odd modulus n.
//
// This header is the ONE place the finite-ring constants and operations live.
// Public Semantics can share one of these algebra-layer variants while choosing
// different operation policies above it; e.g. Field, FieldFast and
// FieldWithMulCasts all reuse Field1 primes and embeddings.
//   * Field-family semantics use n = p, a prime just below 2^w.
//   * SophieGermainRing-family semantics use n = p*d with p = 2d+1, carrying
//     exp(v) = g^(v mod d) so exp(a+b)=exp(a)exp(b), at the cost of rare
//     zero-divisors.
//   * PythagoreanRing-family semantics use n = p*d with p = 4d+1, keeping the
//     exp/log channel and adding an order-d rotor for sin/cos angle-addition.
//
// Per the design: everything keys on the scalar element width. Composite
// exp/log/trig channels only make sense >= 8 bits; below that this layer falls
// back to the matching Field prime (two_moduli = false).
// Vector Values apply all of this LANE-WISE; vector width != element width.
//
// Cross-width casts are policy, not part of the leaf embedding. A residue modulo
// one prime does not determine a residue modulo another. cast.hpp therefore
// chooses either a cheap deterministic convention or, for FieldWithMulCasts, the
// multiplicative tower implemented by alg_cast1 below.
// ----------------------------------------------------------------------------
#ifndef FPSAN_DETAIL_ALGEBRAIC_HPP
#define FPSAN_DETAIL_ALGEBRAIC_HPP

#include "fpsan/detail/config.hpp"
#include "fpsan/detail/traits.hpp"

#include <cstdint>
#include <limits>

namespace fpsan {
namespace detail {
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using u128 = unsigned __int128;

// ---- per-width integer traits for the modular layer ----------------------
// The modular arithmetic only ever needs a "2x-widened" type to hold a
// transient product a*b (< n^2 < 2^(2w)); everything else stays in the
// natural payload type U (the bits element type, e.g. uint16_t for
// _Float16). Keeping U narrow -- instead of widening every lane to u64 with a
// u128 intermediate -- keeps register pressure proportional to the float
// width, which matters on 32-bit-register GPUs. The residues and the two
// sentinels (n = Inf, n+1 = NaN) all fit in U by construction: the Field
// primes are chosen just below 2^w leaving >= 2 free codes.
//
//   wider_t<U>:        uint8->uint16, uint16->uint32, uint32->uint64,
//                      uint64->unsigned __int128  (for a*b products)
//   signed_wide_t<U>:  int16 / int32 / int64 / __int128
//                      (for the Bezout coefficients in extended Euclid;
//                       bounded by n in magnitude, so signed-2x is exact)
template <class U> struct alg_wider;
template <> struct alg_wider<std::uint8_t> {
  using type = std::uint16_t;
  using signed_type = std::int16_t;
};
template <> struct alg_wider<std::uint16_t> {
  using type = std::uint32_t;
  using signed_type = std::int32_t;
};
template <> struct alg_wider<std::uint32_t> {
  using type = std::uint64_t;
  using signed_type = std::int64_t;
};
template <> struct alg_wider<std::uint64_t> {
  using type = unsigned __int128;
  using signed_type = __int128;
};
template <class U> using wider_t = typename alg_wider<U>::type;
template <class U> using signed_wide_t = typename alg_wider<U>::signed_type;

// The divergence point inside this implementation layer. The public
// Semantics names are mapped onto these internal variants in value.hpp;
// keeping the enum here avoids depending on value.hpp from this detail
// header.
enum class AlgVariant {
  Field1,
  Field2,
  SophieGermain1,
  SophieGermain2,
  Pythagorean1,
  Pythagorean2
};

struct AlgModulus {
  u64 n = 0; // modulus; residues in [0, n)
  u64 g = 0; // exp/log generator (order d); unused if !two_moduli
  u64 d = 0; // exp/log/trig exponent modulus; unused if !two_moduli
  bool two_moduli = false;
  // order-d rotation element of (Z/n)[i] (i^2=-1) for sin/cos: a genuine
  // rotation in the F_p factor, identity in the F_d factor. Used only by
  // the PythagoreanRing-family semantics.
  u64 omega_re = 0;
  u64 omega_im = 0;
  bool has_sin_cos = false;
};

// ---- the constants table (the only per-(variant x width) data) ----------
// Field-family primes leave >= 2 codes free (Inf, NaN sentinels). All are
// p == 11 (mod 12), which gives two algebraic structures:
//   * sqrt as a multiplicative map with 1/2 round-trip coverage (p==3 mod 4)
//   * cbrt as a PERFECT multiplicative cube root (3 coprime to p-1, p==2 mod 3)
//
// The same primes are also chosen so p_w - 1 forms a coprime tower:
//   (p_4-1) | (p_8-1) | (p_16-1) | (p_32-1) | (p_64-1),
// with each step cofactor coprime to the smaller order; fp6 is standalone.
// FieldWithMulCasts uses this in alg_cast1 to make widening and narrowing
// multiplicative homomorphisms that form commutative diagrams in log
// coordinates. Field and FieldFast share the exact same primes for input
// compatibility, but their casts do not pay for those logarithms.
//
// Variant 1 and 2 are two independent towers sharing only fp4 = 11 (the
// only 11-mod-12 prime that fits 4 bits). Sophie Germain pairs use p = 2d+1
// with g of order d in (Z/n)^*. 64-bit arithmetic uses the same tower
// pattern, and alg_mulmod uses a 128-bit intermediate to carry n near 2^64.
FPSAN_HOST_DEVICE constexpr u64 alg_field_prime(AlgVariant v, unsigned w) {
  const bool a =
      (v == AlgVariant::Field1 || v == AlgVariant::SophieGermain1 || v == AlgVariant::Pythagorean1);
  switch (w) {
  case 4:
    return 11u; // shared: only 11-mod-12 prime that fits 4 bits
  case 6:
    return a ? 59u : 47u; // standalone (not in the cast chain)
  case 8:
    return a ? 191u : 131u;
  case 16:
    return a ? 65171u : 64871u;
  case 32:
    return a ? 4284862331u : 4291215371u;
  // 64-bit: 11-mod-12 primes in the cast tower (p_32-1 | p_64-1, cofactor
  // coprime to p_32-1), just below 2^64.
  case 64:
    return a ? 18446743887391934171ull : 18446743217995397111ull;
  default:
    return 0;
  }
}
// A primitive root (generator of F_p^*) for each Field-family prime. The
// FieldWithMulCasts cast tower uses it in alg_cast1: the cast sends
// g_narrow to a generator of the order-(p_narrow-1) subgroup of F_p_wide^*.
FPSAN_HOST_DEVICE constexpr u64 alg_field_root(AlgVariant v, unsigned w) {
  const bool a =
      (v == AlgVariant::Field1 || v == AlgVariant::SophieGermain1 || v == AlgVariant::Pythagorean1);
  switch (w) {
  case 4:
    return 2u; // p=11
  case 6:
    return a ? 2u : 5u; // 59 -> 2, 47 -> 5
  case 8:
    return a ? 19u : 2u; // 191 -> 19, 131 -> 2
  case 16:
    return a ? 2u : 7u; // 65171 -> 2, 64871 -> 7
  case 32:
    return 2u; // 4284862331 -> 2, 4291215371 -> 2
  case 64:
    return a ? 2u : 7u; // primitive roots of the two 64-bit primes
  default:
    return 0;
  }
}
// PythagoreanRing-family moduli: p = 4d+1 (so p == 1 mod 4 -> the circle
// group has order p-1, divisible by d, so a genuine order-d rotation exists).
// g is the order-d exp/log generator (d | p-1), omega the order-d rotation
// element of (Z/n)[i] -- a rotation in the F_p factor, identity in the F_d
// factor. The checked-in d factors are 7 mod 24, which keeps the selected
// qr-order factor 2-positive. d is ~sqrt(2) smaller than in the
// SophieGermainRing family at the same width, which is the collision cost paid
// for sin/cos laws. The number-theoretic constraints are discussed in
// docs/reducing-floats-mod-p.tex.
FPSAN_HOST_DEVICE constexpr AlgModulus alg_pythagorean_pair(AlgVariant v, unsigned w) {
  const bool p1 = (v == AlgVariant::Pythagorean1);
  switch (w) {
  case 8:
    return p1 ? AlgModulus{203u, 190u, 7u, true, 134u, 140u, true}
              : AlgModulus{203u, 190u, 7u, true, 134u, 140u, true};
  case 16:
    return p1 ? AlgModulus{64643u, 57024u, 127u, true, 36831u, 62992u, true}
              : AlgModulus{25043u, 20304u, 79u, true, 13431u, 17854u, true};
  case 32:
    return p1 ? AlgModulus{4263339083u, 4261380264u, 32647u, true, 2663668731u, 1327688196u, true}
              : AlgModulus{4219574243u, 4217625504u, 32479u, true, 2636320431u, 181102904u, true};
  case 64:
    // p = 4d+1, d == 7 (mod 24) so the r^(d+1) log projection lands in <g>;
    // omega is the order-d rotor in (Z/n)[i] (identity in the F_d factor).
    return p1 ? AlgModulus{18446679324986898443ull, 18446679196138105704ull, 2147479879ull, true,
                           11529174517718939931ull, 8531841841793833212ull,  true}
              : AlgModulus{18446671903297182683ull, 18446671774448415864ull, 2147479447ull, true,
                           11529169879162879731ull, 2934161963579244570ull,  true};
  default:
    return {};
  }
}
FPSAN_HOST_DEVICE constexpr AlgModulus alg_sophie_germain_pair(AlgVariant v, unsigned w) {
  // Sophie Germain pairs (p = 2d+1) with d == 11 mod 12, so the safe-prime
  // qr-order factor makes both 2 and 3 positive. The 8-bit second variant
  // reuses the only such pair that fits; wider widths use distinct pairs.
  // g has order d in (Z/n)^*.
  const bool sg1 = (v == AlgVariant::SophieGermain1);
  switch (w) {
  case 8:
    return sg1 ? AlgModulus{253u, 188u, 11u, true} : AlgModulus{253u, 188u, 11u, true};
  case 16:
    return sg1 ? AlgModulus{64261u, 63188u, 179u, true} : AlgModulus{34453u, 33668u, 131u, true};
  case 32:
    return sg1 ? AlgModulus{4268741401u, 4268464208u, 46199u, true}
               : AlgModulus{4226711653u, 4226435828u, 45971u, true};
  case 64:
    // p = 2d+1 (Sophie Germain), d == 11 (mod 12) so 3 stays coprime to the
    // group exponent, cbrt remains a perfect power map, and 2/3 are positive.
    return sg1 ? AlgModulus{18446723947803676141ull, 18446723929581683084ull, 3036998843ull, true}
               : AlgModulus{18446715930127601161ull, 18446715911905612064ull, 3036998183ull, true};
  default:
    return {};
  }
}

FPSAN_HOST_DEVICE constexpr AlgModulus alg_modulus(AlgVariant v, unsigned w) {
  const bool is_sophie_germain =
      (v == AlgVariant::SophieGermain1 || v == AlgVariant::SophieGermain2);
  const bool is_pythagorean = (v == AlgVariant::Pythagorean1 || v == AlgVariant::Pythagorean2);
  if (is_sophie_germain && w >= 8)
    return alg_sophie_germain_pair(v, w);
  if (is_pythagorean && w >= 8)
    return alg_pythagorean_pair(v, w);
  // Field-family semantics, or composite-channel requests below 8 bits:
  // use a field prime with no exp/log/trig channel. g carries a primitive
  // root of the prime for the FieldWithMulCasts tower.
  return {alg_field_prime(v, w), alg_field_root(v, w), 0u, false};
}

FPSAN_HOST_DEVICE constexpr u64 alg_gcd(u64 a, u64 b) {
  while (b) {
    u64 t = a % b;
    a = b;
    b = t;
  }
  return a;
}

// ---- the per-Value configuration (analogous to MixConfig) ----------------
struct AlgConfig {
  u64 n = 0;
  u64 inv2 = 0; // 2^{-1} mod n
  u64 g = 0;
  u64 cast_g = 0; // format-specific primitive root for Field casts
  u64 d = 0;
  u64 inf_code = 0; // = n
  u64 nan_code = 0; // = n + 1
  bool two_moduli = false;
  u64 omega_re = 0; // order-d rotation element of (Z/n)[i], for sin/cos
  u64 omega_im = 0;
  bool has_sin_cos = false;
  // multiplicative root exponents (power maps x^e on units): sqrt and its
  // reciprocal rsqrt always available; cbrt only where 3 is coprime to the
  // group exponent. PythagoreanRing-family moduli deliberately do not have
  // this property, so cbrt falls back to a tag there.
  u64 sqrt_exp = 0;
  u64 rsqrt_exp = 0;
  u64 cbrt_exp = 0;
  bool has_cbrt = false;
  // decoded float format of the element type:
  unsigned bit_width = 0;
  unsigned mant_bits = 0;
  u64 exp_max = 0; // all-ones exponent field
  u64 mant_mask = 0;
  int bias = 0;
  bool has_inf_nan = false; // the algebraic carrier reserves Inf/NaN sentinels
  bool format_has_infinity = false;
  bool format_has_nan = false;
  bool format_nan_as_neg_zero = false;
};

// Modular multiply through a 2x-widened intermediate: the only product in
// the modular layer that can exceed the payload type. For U = u64 this is
// the old u128 path (n up to ~2^64); for narrower U it stays in the
// matching narrow widened type (uint16/uint32/uint64), so f8/f16/f32 never
// touch 64/128-bit arithmetic here. Inputs are residues in [0, n).
template <class U> FPSAN_HOST_DEVICE constexpr U alg_mulmod(U a, U b, U n) {
  // numeric_limits is fine here for U itself, since U is just (uint8/16/32/64)
  constexpr U lim = std::numeric_limits<U>::max() >> (sizeof(U) * 4);
  // Some callers intentionally use a wider-than necessary U-type (eg alg_cast1)
  if (n <= lim) {
    a %= n;
    b %= n;
    return (a * b) % n;
  }
  return static_cast<U>((static_cast<wider_t<U>>(a) * static_cast<wider_t<U>>(b)) % n);
}

template <class ElementType> FPSAN_HOST_DEVICE constexpr AlgConfig make_alg_config(AlgVariant v) {
  using T = fp_traits<ElementType>;
  static_assert(T::bit_width <= 64, "fpsan algebraic: element types wider than 64 bits are "
                                    "not supported.");
  static_assert(T::cast_tag <= 1, "fpsan algebraic: multiplicative casts currently distinguish at "
                                  "most two same-width format classes");
  AlgConfig c;
  const AlgModulus m = alg_modulus(v, T::bit_width);
  c.n = m.n;
  c.g = m.g;
  c.cast_g = (!m.two_moduli && m.g != 0 && T::cast_tag == 1)
                 ? alg_mulmod(alg_mulmod(m.g, m.g, m.n), m.g, m.n)
                 : m.g;
  c.d = m.d;
  c.two_moduli = m.two_moduli;
  c.omega_re = m.omega_re;
  c.omega_im = m.omega_im;
  c.has_sin_cos = m.has_sin_cos;
  c.inv2 = (m.n + 1) / 2; // inverse of 2 mod odd n
  c.inf_code = m.n;
  c.nan_code = m.n + 1;
  c.bit_width = T::bit_width;
  c.mant_bits = T::mantissa_bits;
  c.bias = T::bias;
  c.mant_mask = (u64{1} << T::mantissa_bits) - 1;
  c.exp_max = (u64{1} << T::exponent_bits) - 1;
  c.has_inf_nan = true;
  c.format_has_infinity = T::has_infinity;
  c.format_has_nan = T::has_nan;
  c.format_nan_as_neg_zero = T::nan_as_neg_zero;
  // Root power maps. lam = exponent of the unit group (Carmichael):
  // n-1 for a prime field, lcm(p-1, d-1) for the composite CRT rings.
  const u64 pf = c.two_moduli ? (c.n / c.d) : c.n; // F_p factor
  const u64 lam = c.two_moduli ? (pf - 1) / alg_gcd(pf - 1, c.d - 1) * (c.d - 1) : (c.n - 1);
  u64 lam_odd = lam;
  while (lam_odd % 2 == 0)
    lam_odd /= 2;
  c.sqrt_exp = (lam_odd + 1) / 2; // 2^{-1} mod (odd part): best sqrt coverage
  c.rsqrt_exp = lam - c.sqrt_exp; // x^rsqrt_exp == sqrt(x)^{-1} on units
  c.has_cbrt = (lam % 3 != 0);    // 3 invertible mod lam -> perfect cbrt
  // 3^{-1} mod lam, written to avoid overflow at lam ~ 2^64: for lam == 1
  // (mod 3) the textbook (2*lam+1)/3 would overflow, so use the equal
  // value 2*((lam-1)/3) + 1; for lam == 2 (mod 3), (lam+1)/3 fits (lam+1 = n).
  c.cbrt_exp = !c.has_cbrt ? 0 : (lam % 3 == 1 ? 2 * ((lam - 1) / 3) + 1 : (lam + 1) / 3);
  return c;
}

// ---- scalar modular arithmetic -------------------------------------------
// Overflow-safe modular add/sub for moduli up to ~2^64. At 64 bits a + b can
// exceed 2^64, so a u64 wrap (s < a) must be handled like the s >= n case --
// s - n then yields the correct residue modulo 2^64. (For n < 2^32, as in the
// narrower widths, the wrap never happens and this matches the old s>=n form.)
// addmod stays entirely in the narrow type U: a + b can be up to 2n-2,
// which overflows U when n is near 2^w, but the wrap check (s < a) detects
// exactly that overflow -- the same trick the u64 code used for n near 2^64
// -- so s - n recovers the right residue modulo 2^w. No widening needed.
template <class U> FPSAN_HOST_DEVICE constexpr U alg_addmod(U a, U b, U n) {
  U s = static_cast<U>((a) + b);
  return (s < a || s >= n) ? static_cast<U>((s - n)) : s;
}
template <class U> FPSAN_HOST_DEVICE constexpr U alg_submod(U a, U b, U n) {
  return (a >= b) ? static_cast<U>((a - b))
                  : static_cast<U>((n - (b - a))); // a, b in [0, n): no overflow
}
template <class U> FPSAN_HOST_DEVICE constexpr U alg_powmod(U b, U e, U n) {
  U r = static_cast<U>((1) % n);
  b = static_cast<U>((b) % n);
  while (e) {
    if (e & 1)
      r = alg_mulmod<U>(r, b, n);
    b = alg_mulmod<U>(b, b, n);
    e >>= 1;
  }
  return r;
}
// Returns the inverse, or n (an out-of-range sentinel) if a is a non-unit.
// Remainders stay in u64 (< n <= ~2^64); the Bezout coefficients are kept in
// a signed 128-bit type so this is exact for moduli up to ~2^64 -- the old
// int64 coefficients overflowed once n exceeded 2^63.
// Remainders stay in U (< n <= ~2^w); the Bezout coefficients are kept in
// the SIGNED 2x-widened type so this is exact for moduli up to ~2^w -- a
// signed type the same width as U could overflow once n exceeds 2^(w-1).
template <class U> FPSAN_HOST_DEVICE constexpr U alg_inv(U a, U n) {
  using S = signed_wide_t<U>;
  S t = 0, newt = 1;
  U r = n, newr = static_cast<U>((a) % n);
  while (newr != 0) {
    U q = static_cast<U>((r) / newr);
    S tmp = static_cast<S>(t - static_cast<S>(q) * newt);
    t = newt;
    newt = tmp;
    U rr = static_cast<U>(r - static_cast<U>(q * newr));
    r = newr;
    newr = rr;
  }
  if (r != 1)
    return n; // not invertible (zero-divisor)
  if (t < 0)
    t += static_cast<S>(n);
  return static_cast<U>(t);
}

// A cheap operation-tagged scramble: the "free generator" for operations
// with no honored identity in the selected algebraic semantics.
// For <=32-bit element widths the 32-bit murmur3 finalizer is a bijection
// over the payload domain, so its residue mod n is as uniform -- the
// collision rate is set by n, not the mixer width -- as the 64-bit
// splitmix64 used at 64 bits, at ~1/3 the GPU registers (op_tag is folded
// to 32 bits). The 64-bit path is unchanged, so f64 tags are bit-identical;
// <=32-bit tag fingerprints are rerolled (still deterministic, op-distinct,
// same collision rate -- tags carry no algebraic identity, so the exact
// values are arbitrary).
template <class U> FPSAN_HOST_DEVICE constexpr U alg_tag(u64 op_tag, U payload, U n) {
  if constexpr (sizeof(U) <= 4) {
    const u32 o = static_cast<u32>(op_tag) ^ static_cast<u32>(op_tag >> 32);
    u32 z = static_cast<u32>(payload) + 1u;
    z = z * 0x9E3779B9u + o;
    z ^= z >> 16;
    z *= 0x85EBCA6Bu;
    z ^= z >> 13;
    z *= 0xC2B2AE35u;
    z ^= z >> 16;
    z ^= o * 0x85EBCA6Bu;
    z ^= z >> 16;
    z *= 0x85EBCA6Bu;
    z ^= z >> 13;
    z *= 0xC2B2AE35u;
    z ^= z >> 16;
    return static_cast<U>((z) % static_cast<u32>(n));
  }
  u64 z = (static_cast<u64>(payload) + 1) * 0x9E3779B97F4A7C15ull + op_tag;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  z ^= z >> 31;
  return static_cast<U>((z) % n);
}

// ---- scalar payload ops on residues (incl. the Inf/NaN projective rules) -
// Codes: [0,n) finite; n = Inf; n+1 = NaN.
//
// These leaf ops are templated on the payload type U (the bits element type,
// e.g. uint16_t for _Float16). The AlgConfig stores its fields as u64; each
// op casts the fields it needs (n, inf_code = n, nan_code = n+1, g, d, inv2,
// ...) down to U. This is safe because every residue and sentinel is < 2^w
// by construction (the prime n sits just below 2^w with >= 2 free codes, so
// n and n+1 still fit in U). Default U = u64 keeps the external u64 callers
// (proto test, cast tower) working unchanged.
template <class U = u64> FPSAN_HOST_DEVICE constexpr bool alg_is_inf(const AlgConfig &c, U p) {
  return p == static_cast<U>(c.inf_code);
}
template <class U = u64> FPSAN_HOST_DEVICE constexpr bool alg_is_nan(const AlgConfig &c, U p) {
  return p == static_cast<U>(c.nan_code);
}
template <class U = u64> FPSAN_HOST_DEVICE constexpr bool alg_is_fin(const AlgConfig &c, U p) {
  return p < static_cast<U>(c.n);
}
FPSAN_HOST_DEVICE constexpr u64 alg_qr_order_modulus(const AlgConfig &c) {
  return c.two_moduli ? (c.has_sin_cos ? c.d : c.n / c.d) : c.n;
}
FPSAN_HOST_DEVICE constexpr bool alg_has_qr_order(const AlgConfig &c) {
  return alg_qr_order_modulus(c) % 4 == 3;
}
template <class U = u64>
FPSAN_HOST_DEVICE constexpr bool alg_is_nonzero_qr(const AlgConfig &c, U p) {
  // The qr-order only needs a sign-canonical positive class, not full
  // quadratic residuosity in every CRT component. Field semantics use their
  // prime field. SophieGermain rings use the safe-prime factor p=n/d where
  // -1 is a non-square. Pythagorean rings use the d factor when it has that
  // property. Thus for ordinary units exactly one of x and -x is qr-positive,
  // matching the field behavior whenever alg_has_qr_order(c) holds.
  const U q = static_cast<U>(alg_qr_order_modulus(c));
  const U r = static_cast<U>(p % q);
  if (!alg_is_fin<U>(c, p) || r == 0)
    return false;
  return alg_powmod<U>(r, static_cast<U>((q - 1) / 2), q) == 1;
}
template <class U = u64>
FPSAN_HOST_DEVICE constexpr bool alg_qr_less(const AlgConfig &c, U a, U b) {
  return !alg_is_nonzero_qr<U>(c, a) && alg_is_nonzero_qr<U>(c, b);
}

template <class U = u64> FPSAN_HOST_DEVICE constexpr U alg_neg1(const AlgConfig &c, U a) {
  if (!alg_is_fin<U>(c, a))
    return a; // -Inf == Inf (unsigned), -NaN == NaN
  return a == 0 ? U{0} : static_cast<U>(static_cast<U>(c.n) - a);
}
template <class U = u64> FPSAN_HOST_DEVICE constexpr U alg_add1(const AlgConfig &c, U a, U b) {
  if (alg_is_nan<U>(c, a) || alg_is_nan<U>(c, b))
    return static_cast<U>(c.nan_code);
  if (alg_is_inf<U>(c, a) || alg_is_inf<U>(c, b))
    return (alg_is_inf<U>(c, a) && alg_is_inf<U>(c, b)) ? static_cast<U>(c.nan_code)
                                                        : static_cast<U>(c.inf_code);
  return alg_addmod<U>(a, b, static_cast<U>(c.n));
}
template <class U = u64> FPSAN_HOST_DEVICE constexpr U alg_sub1(const AlgConfig &c, U a, U b) {
  return alg_add1<U>(c, a, alg_neg1<U>(c, b));
}
template <class U = u64> FPSAN_HOST_DEVICE constexpr U alg_mul1(const AlgConfig &c, U a, U b) {
  if (alg_is_nan<U>(c, a) || alg_is_nan<U>(c, b))
    return static_cast<U>(c.nan_code);
  const bool ai = alg_is_inf<U>(c, a), bi = alg_is_inf<U>(c, b);
  const bool az = (alg_is_fin<U>(c, a) && a == 0), bz = (alg_is_fin<U>(c, b) && b == 0);
  if (ai || bi)
    return (az || bz) ? static_cast<U>(c.nan_code) : static_cast<U>(c.inf_code); // 0*Inf -> NaN
  return alg_mulmod<U>(a, b, static_cast<U>(c.n));
}
template <class U = u64> FPSAN_HOST_DEVICE constexpr U alg_div1(const AlgConfig &c, U a, U b) {
  if (alg_is_nan<U>(c, a) || alg_is_nan<U>(c, b))
    return static_cast<U>(c.nan_code);
  const bool ai = alg_is_inf<U>(c, a), bi = alg_is_inf<U>(c, b);
  if (ai && bi)
    return static_cast<U>(c.nan_code); // Inf/Inf
  if (ai)
    return static_cast<U>(c.inf_code); // Inf/finite
  if (bi)
    return static_cast<U>(0); // finite/Inf -> 0
  if (b == 0)
    return a == 0 ? static_cast<U>(c.nan_code)
                  : static_cast<U>(c.inf_code); // 0/0 -> NaN, x/0 -> Inf
  const U inv = alg_inv<U>(b, static_cast<U>(c.n));
  if (inv == static_cast<U>(c.n))
    return static_cast<U>(c.nan_code); // zero-divisor (CRT variant) -> poison
  return alg_mulmod<U>(a, inv, static_cast<U>(c.n));
}
template <class U = u64> FPSAN_HOST_DEVICE constexpr U alg_fast_div1(const AlgConfig &c, U a, U b) {
  if (alg_is_nan<U>(c, a) || alg_is_nan<U>(c, b))
    return static_cast<U>(c.nan_code);
  const bool ai = alg_is_inf<U>(c, a), bi = alg_is_inf<U>(c, b);
  if (ai && bi)
    return static_cast<U>(c.nan_code); // Inf/Inf
  if (ai)
    return static_cast<U>(c.inf_code); // Inf/finite
  if (bi)
    return static_cast<U>(0); // finite/Inf -> 0
  if (b == 0)
    return a == 0 ? static_cast<U>(c.nan_code)
                  : static_cast<U>(c.inf_code); // 0/0 -> NaN, x/0 -> Inf
  if (a == 0)
    return static_cast<U>(0); // zero divided by a finite nonzero value stays zero
  if (b == 1)
    return a; // x/1 stays x without paying for an inverse
  return alg_tag<U>(
      0x646976ull /*"div"*/,
      alg_addmod<U>(a, alg_tag<U>(0x6279ull /*"by"*/, b, static_cast<U>(c.n)), static_cast<U>(c.n)),
      static_cast<U>(c.n));
}
template <class U = u64> FPSAN_HOST_DEVICE constexpr U alg_exp1(const AlgConfig &c, U a) {
  if (!alg_is_fin<U>(c, a))
    return static_cast<U>(c.nan_code); // exp(Inf) ambiguous (unsigned), exp(NaN)=NaN
  if (c.two_moduli)
    return alg_powmod<U>(static_cast<U>(c.g), static_cast<U>((a) % static_cast<U>(c.d)),
                         static_cast<U>(c.n)); // g^(v mod d)
  return alg_tag<U>(/*tag "exp"*/ 0x657870ull, a, static_cast<U>(c.n));
}

// phi_n of a raw float bit-pattern of the element type (one lane).
template <class U = u64> FPSAN_HOST_DEVICE constexpr U alg_embed1(const AlgConfig &c, U raw) {
  const U sign = static_cast<U>(((raw) >> (c.bit_width - 1)) & 1);
  const U expf = static_cast<U>(((raw) >> c.mant_bits) & static_cast<U>(c.exp_max));
  const U mantf = static_cast<U>((raw) & static_cast<U>(c.mant_mask));
  const U sign_mask = static_cast<U>((U{1}) << (c.bit_width - 1));
  if (c.format_has_nan && c.format_nan_as_neg_zero && raw == sign_mask)
    return static_cast<U>(c.nan_code);
  if (expf == static_cast<U>(c.exp_max)) {
    if (c.format_has_infinity)
      return mantf == 0 ? static_cast<U>(c.inf_code) : static_cast<U>(c.nan_code);
    if (c.format_has_nan && !c.format_nan_as_neg_zero && mantf == static_cast<U>(c.mant_mask))
      return static_cast<U>(c.nan_code);
  }
  U mag = 0;
  int e = 0;
  if (expf == 0) {
    if (mantf == 0)
      return 0; // +/-0 -> residue 0
    mag = mantf;
    e = 1 - c.bias - static_cast<int>(c.mant_bits);
  } else {
    mag = static_cast<U>(((U{1}) << c.mant_bits) | mantf);
    e = static_cast<int>(expf) - c.bias - static_cast<int>(c.mant_bits);
  }
  U r = static_cast<U>((mag) % static_cast<U>(c.n));
  U pw = e >= 0 ? alg_powmod<U>(static_cast<U>(2), static_cast<U>(e), static_cast<U>(c.n))
                : alg_powmod<U>(static_cast<U>(c.inv2), static_cast<U>((-e)), static_cast<U>(c.n));
  r = alg_mulmod<U>(r, pw, static_cast<U>(c.n));
  return sign ? static_cast<U>((r == 0) ? 0 : static_cast<U>(c.n - r)) : r;
}

// phi_n^{-1} is NOT well-defined (the residue does not determine the value),
// so this is a best-effort, non-faithful decode used only by to_float() for
// display: sentinels map to native Inf/NaN bit patterns when the element
// format has them, or to a native NaN where a format has NaN but no Inf.
// A finite residue is returned as-is (a deterministic but meaningless bit
// pattern).
// Algebraic Values are meant to be compared by payload, not unembedded.
template <class U = u64> FPSAN_HOST_DEVICE constexpr U alg_unembed1(const AlgConfig &c, U p) {
  const U sign_mask = static_cast<U>((U{1}) << (c.bit_width - 1));
  const U nan_bits = c.format_nan_as_neg_zero
                         ? sign_mask
                         : static_cast<U>((static_cast<U>(c.exp_max) << c.mant_bits) |
                                          static_cast<U>(c.mant_mask));
  if (c.has_inf_nan && p == static_cast<U>(c.inf_code)) {
    if (c.format_has_infinity)
      return static_cast<U>(static_cast<U>(c.exp_max) << c.mant_bits); // +Inf
    if (c.format_has_nan)
      return nan_bits;
  }
  if (c.has_inf_nan && p == static_cast<U>(c.nan_code))
    return c.format_has_nan ? nan_bits : static_cast<U>(0);
  const U width_mask = (c.bit_width >= (sizeof(U) * 8)) ? static_cast<U>(~U{0})
                                                        : static_cast<U>((U{1} << c.bit_width) - 1);
  return static_cast<U>((p)&width_mask);
}

// ---- vector wrappers: apply the scalar core lane-wise (cf. ring_div) -----
// The lambdas below pass the NATURAL lane element type L (the bits element
// type, e.g. uint16_t for _Float16) to op, not u64 -- so the templated leaf
// ops instantiate at the narrow width and never widen a lane to u64.
template <class Bits, class Op> FPSAN_HOST_DEVICE constexpr Bits alg_lanewise1(Bits a, Op op) {
  if constexpr (!is_clang_vector_v<Bits>)
    return static_cast<Bits>(op(a));
  else {
    using L = bits_lane_t<Bits>;
    Bits out{};
    for (unsigned i = 0; i < sizeof(Bits) / sizeof(L); ++i)
      out[i] = static_cast<L>(op(static_cast<L>(a[i])));
    return out;
  }
}
template <class Bits, class Op>
FPSAN_HOST_DEVICE constexpr Bits alg_lanewise2(Bits a, Bits b, Op op) {
  if constexpr (!is_clang_vector_v<Bits>)
    return static_cast<Bits>(op(a, b));
  else {
    using L = bits_lane_t<Bits>;
    Bits out{};
    for (unsigned i = 0; i < sizeof(Bits) / sizeof(L); ++i)
      out[i] = static_cast<L>(op(static_cast<L>(a[i]), static_cast<L>(b[i])));
    return out;
  }
}

template <class Bits> FPSAN_HOST_DEVICE constexpr Bits alg_embed(const AlgConfig &c, Bits raw) {
  return alg_lanewise1(raw, [&](auto x) { return alg_embed1(c, x); });
}
template <class Bits> FPSAN_HOST_DEVICE constexpr Bits alg_unembed(const AlgConfig &c, Bits p) {
  return alg_lanewise1(p, [&](auto x) { return alg_unembed1(c, x); });
}
template <class Bits> FPSAN_HOST_DEVICE constexpr Bits alg_add(const AlgConfig &c, Bits a, Bits b) {
  return alg_lanewise2(a, b, [&](auto x, auto y) { return alg_add1(c, x, y); });
}
template <class Bits> FPSAN_HOST_DEVICE constexpr Bits alg_sub(const AlgConfig &c, Bits a, Bits b) {
  return alg_lanewise2(a, b, [&](auto x, auto y) { return alg_sub1(c, x, y); });
}
template <class Bits> FPSAN_HOST_DEVICE constexpr Bits alg_mul(const AlgConfig &c, Bits a, Bits b) {
  return alg_lanewise2(a, b, [&](auto x, auto y) { return alg_mul1(c, x, y); });
}
template <class Bits> FPSAN_HOST_DEVICE constexpr Bits alg_div(const AlgConfig &c, Bits a, Bits b) {
  return alg_lanewise2(a, b, [&](auto x, auto y) { return alg_div1(c, x, y); });
}
template <class Bits>
FPSAN_HOST_DEVICE constexpr Bits alg_fast_div(const AlgConfig &c, Bits a, Bits b) {
  return alg_lanewise2(a, b, [&](auto x, auto y) { return alg_fast_div1(c, x, y); });
}
template <class Bits> FPSAN_HOST_DEVICE constexpr Bits alg_neg(const AlgConfig &c, Bits a) {
  return alg_lanewise1(a, [&](auto x) { return alg_neg1(c, x); });
}
template <class Bits> FPSAN_HOST_DEVICE constexpr Bits alg_qr_abs(const AlgConfig &c, Bits a) {
  return alg_lanewise1(a, [&](auto x) { return alg_is_nonzero_qr(c, x) ? x : alg_neg1(c, x); });
}
template <class Bits>
FPSAN_HOST_DEVICE constexpr Bits alg_qr_min(const AlgConfig &c, Bits a, Bits b) {
  return alg_lanewise2(a, b, [&](auto x, auto y) {
    const bool x_pos = alg_is_nonzero_qr(c, x);
    const bool y_pos = alg_is_nonzero_qr(c, y);
    if (x_pos != y_pos)
      return x_pos ? y : x;
    return x < y ? x : y;
  });
}
template <class Bits>
FPSAN_HOST_DEVICE constexpr Bits alg_qr_max(const AlgConfig &c, Bits a, Bits b) {
  return alg_lanewise2(a, b, [&](auto x, auto y) {
    const bool x_pos = alg_is_nonzero_qr(c, x);
    const bool y_pos = alg_is_nonzero_qr(c, y);
    if (x_pos != y_pos)
      return x_pos ? x : y;
    return x < y ? y : x;
  });
}
template <class Bits> FPSAN_HOST_DEVICE constexpr Bits alg_exp(const AlgConfig &c, Bits a) {
  return alg_lanewise1(a, [&](auto x) { return alg_exp1(c, x); });
}

// Op-tagged free generator for operations with no honored identity in the
// selected algebraic semantics: deterministic and op-distinct, but not an
// algebraic law.
template <class U> FPSAN_HOST_DEVICE constexpr U alg_tagged1(const AlgConfig &c, U a, u64 tag) {
  return alg_is_fin<U>(c, a) ? alg_tag<U>(tag, a, static_cast<U>(c.n)) : static_cast<U>(c.nan_code);
}
template <class Bits>
FPSAN_HOST_DEVICE constexpr Bits alg_tagged(const AlgConfig &c, Bits a, u64 tag) {
  return alg_lanewise1(a, [&](auto x) { return alg_tagged1(c, x, tag); });
}

// ---- multiplicative roots: sqrt, rsqrt, cbrt (power maps x^e on units) ----
// sqrt/cbrt are ALGEBRAIC (not transcendental): a fixed-exponent power map,
// so sqrt(x*y)==sqrt(x)*sqrt(y) and cbrt(x*y)==cbrt(x)*cbrt(y) hold exactly
// for every modulus, and rsqrt==1/sqrt is consistent. The round-trip
// sqrt(x)^2==x holds on the square residues (~1/2 of a prime field), and
// cbrt(x)^3==x holds for ALL x where has_cbrt (3 coprime to the group
// exponent). Where 3 divides it, cbrt falls back to a tag.
template <class U> FPSAN_HOST_DEVICE constexpr U alg_sqrt1(const AlgConfig &c, U x) {
  if (alg_is_nan<U>(c, x))
    return static_cast<U>(c.nan_code);
  if (alg_is_inf<U>(c, x))
    return static_cast<U>(c.inf_code);                                      // sqrt(Inf) = Inf
  return alg_powmod<U>(x, static_cast<U>(c.sqrt_exp), static_cast<U>(c.n)); // sqrt(0) = 0
}
template <class U> FPSAN_HOST_DEVICE constexpr U alg_rsqrt1(const AlgConfig &c, U x) {
  if (alg_is_nan<U>(c, x))
    return static_cast<U>(c.nan_code);
  if (alg_is_inf<U>(c, x))
    return 0; // 1/sqrt(Inf) = 0
  if (x == 0)
    return static_cast<U>(c.inf_code); // 1/sqrt(0) = Inf
  return alg_powmod<U>(x, static_cast<U>(c.rsqrt_exp), static_cast<U>(c.n));
}
template <class U> FPSAN_HOST_DEVICE constexpr U alg_cbrt1(const AlgConfig &c, U x) {
  if (!c.has_cbrt)
    return alg_tagged1<U>(c, x, 0x63627274ull /*"cbrt"*/);
  if (alg_is_nan<U>(c, x))
    return static_cast<U>(c.nan_code);
  if (alg_is_inf<U>(c, x))
    return static_cast<U>(c.inf_code);                                      // cbrt(Inf) = Inf
  return alg_powmod<U>(x, static_cast<U>(c.cbrt_exp), static_cast<U>(c.n)); // cbrt(0) = 0
}
template <class Bits> FPSAN_HOST_DEVICE constexpr Bits alg_sqrt(const AlgConfig &c, Bits x) {
  return alg_lanewise1(x, [&](auto v) { return alg_sqrt1(c, v); });
}
template <class Bits> FPSAN_HOST_DEVICE constexpr Bits alg_rsqrt(const AlgConfig &c, Bits x) {
  return alg_lanewise1(x, [&](auto v) { return alg_rsqrt1(c, v); });
}
template <class Bits> FPSAN_HOST_DEVICE constexpr Bits alg_cbrt(const AlgConfig &c, Bits x) {
  return alg_lanewise1(x, [&](auto v) { return alg_cbrt1(c, v); });
}

// Binary tag (e.g. fmod): deterministic in both operands.
template <class U>
FPSAN_HOST_DEVICE constexpr U alg_tagged2_1(const AlgConfig &c, U a, U b, u64 tag) {
  if (!alg_is_fin<U>(c, a) || !alg_is_fin<U>(c, b))
    return static_cast<U>(c.nan_code);
  return alg_tag<U>(tag, static_cast<U>(alg_tag<U>(tag, a, static_cast<U>(c.n)) + b),
                    static_cast<U>(c.n));
}
template <class Bits>
FPSAN_HOST_DEVICE constexpr Bits alg_tagged2(const AlgConfig &c, Bits a, Bits b, u64 tag) {
  return alg_lanewise2(a, b, [&](auto x, auto y) { return alg_tagged2_1(c, x, y, tag); });
}

// Generic extern/libdevice fallback in the residue ring: a deterministic,
// symbol-distinct, argument-order-sensitive tag for any unmodeled call,
// keyed by a symbol-name hash. The algebraic analogue of
// payload_extern_tagged (which is itself a transcription of Triton's extern
// tagging -- see fpsan/detail/math.hpp). NaN-propagating. The arg index is
// folded into each operation tag, so f(a,b) != f(b,a).
template <class... P>
FPSAN_HOST_DEVICE constexpr u64 alg_extern_tagged1(const AlgConfig &c, u64 name_hash,
                                                   P... operands) {
  bool fin = true;
  ((fin = fin && alg_is_fin(c, static_cast<u64>(operands))), ...);
  if (!fin)
    return c.nan_code;
  u64 acc = name_hash % c.n;
  unsigned i = 0;
  ((acc = alg_tag(acc + i++, static_cast<u64>(operands), c.n)), ...);
  return acc;
}

// Pohlig-Hellman discrete log for a SMOOTH group order m: find k in [0,m)
// with b^k == x in F_q, where b has order m. m is factored by trial division
// at runtime -- cheap because the Field primes are chosen so p-1 is smooth
// (largest prime factor a few thousand), so the loop terminates there. Used
// for the fp32<->fp64 cast, where m = p_32-1 ~ 2^32 makes the O(m) scan
// hopeless; returns the same unique k the scan would. The per-prime-power
// work is digit-by-digit (order-p subgroup dlogs by brute scan, p small)
// then CRT-combined; O(sum of prime factors), O(1) memory, device-friendly.
FPSAN_HOST_DEVICE constexpr u64 alg_dlog_ph(u64 x, u64 b, u64 m, u64 q) {
  u64 k = 0, M = 1, rem = m;
  for (u64 d = 2; rem > 1; ++d) {
    if (d * d > rem)
      d = rem; // the remaining cofactor is prime
    if (rem % d != 0)
      continue;
    u64 e = 0, pe = 1;
    while (rem % d == 0) {
      rem /= d;
      pe *= d;
      ++e;
    }
    const u64 cof = m / pe;
    const u64 gi = alg_powmod(b, cof, q); // order pe
    const u64 hi = alg_powmod(x, cof, q); // in <gi>
    const u64 giinv = alg_inv(gi, q);
    const u64 gamma = alg_powmod(gi, pe / d, q); // order d (prime)
    u64 ki = 0, pj = 1;
    for (u64 j = 0; j < e; ++j) {
      const u64 t = alg_mulmod(alg_powmod(giinv, ki, q), hi, q);
      const u64 hj = alg_powmod(t, pe / (pj * d), q); // order | d
      u64 dj = 0, cur = 1 % q;
      for (u64 s = 0; s < d; ++s) {
        if (cur == hj) {
          dj = s;
          break;
        }
        cur = alg_mulmod(cur, gamma, q);
      }
      ki += dj * pj;
      pj *= d;
    }
    // CRT-combine (k mod M) with (ki mod pe); M, pe coprime.
    const u64 inv = alg_inv(M % pe, pe);
    const u64 diff = (ki + pe - k % pe) % pe;
    k += M * alg_mulmod(diff, inv, pe);
    M *= pe;
  }
  return k % m;
}

// Discrete log in F_q of x to base b, where b has order m (the answer is in
// [0, m)). Brute force O(m) wins for the fp4 order 10. Pohlig-Hellman wins
// overwhelmingly for fp16/fp32 orders, since every Field p-1 is smooth by
// construction. The fp8 orders are the crossover point, so the dispatch below
// follows measurements on CPU and gfx1201.
FPSAN_HOST_DEVICE constexpr u64 alg_dlog_base(u64 x, u64 b, u64 m, u64 q) {
#if defined(__HIP_DEVICE_COMPILE__)
  // On gfx1201, PH also wins for order-190 Field fp8 logs, and for the
  // order-130 Field2 fp8 logs once they live as subgroups in a wider field.
  if (m > (u64{1} << 10) || m == 190 || (m == 130 && q > 256))
    return alg_dlog_ph(x, b, m, q);
#else
  // On CPU, PH wins for order 190 except in the 64-bit modulus case.
  // Order 130 is still faster by brute force in every host cast context.
  if (m > (u64{1} << 10) || (m == 190 && q <= 0xffffffffull))
    return alg_dlog_ph(x, b, m, q);
#endif
  u64 cur = 1 % q;
  for (u64 k = 0; k < m; ++k) {
    if (cur == x)
      return k;
    cur = alg_mulmod(cur, b, q);
  }
  return 0; // unreachable when x is in <b>
}

// Cast between widths. A value-FAITHFUL cast is impossible across coprime
// per-width primes (the narrow residue can't determine the wide one; see
// algebraic-fpsan.md). But a *multiplicative* cast is, and because the Field
// primes form a COPRIME TOWER (p_narrow-1 | p_wide-1 with coprime cofactor),
// widening and narrowing form a commutative diagram in log coordinates
// L_p(x) = dlog_{g_p}(x):
//   * widen  (narrow N -> wide W):  L_W = CRT-section of L_N -- the lift that
//     is L_N mod (p_N-1) and 0 mod the cofactor. So cast(x) = h^{L_N(x)} with
//     h = g_W^{(s*s^{-1} mod (p_N-1)) mod (p_W-1)}, s = (p_W-1)/(p_N-1).
//   * narrow (wide W -> narrow N):  L_N = L_W mod (p_N-1) -- the quotient.
//     Computed cheaply as a dlog over the order-(p_N-1) subgroup.
// Then widening composes, narrowing composes, and narrow(widen(x)) == x. The
// same log-coordinate rule also distinguishes same-width formats: f16 and
// bf16, or e4m3 and e5m2, use different primitive roots in the same F_p.
// Both directions satisfy cast(x*y)==cast(x)*cast(y) and cast(0)==0. Off the
// chain (composite CRT rings, fp6, or non-chain pairs) it is the plain
// reduce-mod convention. Inf/NaN map across.
FPSAN_HOST_DEVICE constexpr u64 alg_cast1(const AlgConfig &from, const AlgConfig &to, u64 p) {
  if (from.has_inf_nan && p == from.inf_code)
    return to.inf_code;
  if (from.has_inf_nan && p == from.nan_code)
    return to.nan_code;
  if (p == 0)
    return 0;
  const bool field = !from.two_moduli && !to.two_moduli && from.cast_g != 0 && to.cast_g != 0;
  if (field && to.n == from.n) {
    if (to.cast_g == from.cast_g)
      return p % to.n;
    const u64 k = alg_dlog_base(p, from.cast_g, from.n - 1, from.n);
    return alg_powmod(to.cast_g, k, to.n);
  }
  if (field && to.n > from.n && (to.n - 1) % (from.n - 1) == 0) {
    // widen N=from -> W=to
    const u64 s = (to.n - 1) / (from.n - 1);
    const u64 sinv = alg_inv(s % (from.n - 1), from.n - 1); // coprime tower => exists
    const u64 h = alg_powmod(to.cast_g, (s * sinv) % (to.n - 1), to.n);
    const u64 k = alg_dlog_base(p, from.cast_g, from.n - 1, from.n);
    return alg_powmod(h, k, to.n);
  }
  if (field && to.n < from.n && (from.n - 1) % (to.n - 1) == 0) {
    // narrow W=from -> N=to
    const u64 s = (from.n - 1) / (to.n - 1);
    const u64 H = alg_powmod(from.cast_g, s, from.n); // order p_N-1 in F_W
    const u64 proj = alg_powmod(p, s, from.n);        // project onto that subgroup
    const u64 k = alg_dlog_base(proj, H, to.n - 1, from.n);
    return alg_powmod(to.cast_g, k, to.n);
  }
  return p % to.n;
}

// log: the inverse of exp on the order-d channel, the dual of g^(v mod d).
// exp embeds Z/d into the MULTIPLICATIVE order-d subgroup <g>; log embeds
// it into the ADDITIVE order-d subgroup {0, n/d, 2n/d, ...} (= multiples
// of p, since n=p*d), which is closed under mod-n addition and isomorphic
// to Z/d -- so log(x*y) = log(x)+log(y) holds EXACTLY in Z/n. Concretely
//   log(r) = (n/d) * dlog_g( (r mod p)^(d+1) )     in [0, n)
// where (r mod p)^(d+1) is r's order-d component. This works for both
// composite families because the chosen d makes d+1 kill the complementary
// cofactor and act as the identity on the order-d subgroup. Only the
// composite CRT families have the d-channel; Field-family semantics fall
// back to a tag. Undefined at a true zero (-> Inf pole) and where the value
// vanishes in the F_p factor (-> NaN). The brute-force dlog is O(d);
// a production device path would precompute a d-entry table.
// Raw discrete log on the order-d channel: the unique k in [0, d) with
// g^k == (r mod p)^(d+1) (r's order-d component) in F_p, or c.d as an
// out-of-range sentinel when r vanishes in the F_p factor. Shared by log
// and log2; the brute-force scan is O(d) (a device path would table it).
// Pollard's rho discrete log in a cyclic group <gp> of PRIME order d in F_p:
// find k in [0,d) with gp^k == target. O(sqrt d) time and O(1) memory -- the
// device-friendly alternative to the O(d) scan, needed once d ~ 2^31 (the
// 64-bit Sophie Germain / Pythagorean channel; the scan would be ~3e9 steps).
// The walk stays inside <gp> (target is in it), tracking x = gp^a * target^b;
// a Floyd collision yields k = (a'-a)/(b-b') mod d (d prime => exact inverse).
// Deterministic restarts cover the rare degenerate (b == b') collision.
FPSAN_HOST_DEVICE constexpr void alg_rho_step(u64 &x, u64 &a, u64 &b, u64 gp, u64 target, u64 d,
                                              u64 p) {
  switch (x % 3) {
  case 0:
    x = alg_mulmod(x, target, p);
    b = (b + 1) % d;
    break;
  case 1:
    x = alg_mulmod(x, x, p);
    a = (2 * a) % d;
    b = (2 * b) % d;
    break;
  default:
    x = alg_mulmod(x, gp, p);
    a = (a + 1) % d;
    break;
  }
}
FPSAN_HOST_DEVICE constexpr u64 alg_dlog_rho(u64 target, u64 gp, u64 d, u64 p) {
  if (target == 1 % p)
    return 0;
  const u64 cap = (u64{1} << 21); // >> sqrt(d) for any 64-bit d; safety bound
  for (u64 attempt = 1; attempt <= 16; ++attempt) {
    u64 a = attempt % d, b = 0;
    u64 x = alg_powmod(gp, a, p);
    u64 A = a, B = b, X = x;
    for (u64 i = 0; i < cap; ++i) {
      alg_rho_step(x, a, b, gp, target, d, p);
      alg_rho_step(X, A, B, gp, target, d, p);
      alg_rho_step(X, A, B, gp, target, d, p);
      if (x == X) {
        const u64 bb = (b + d - B) % d;
        if (bb == 0)
          break; // degenerate collision -- restart with a new offset
        const u64 aa = (A + d - a) % d;
        const u64 binv = alg_powmod(bb, d - 2, d); // d prime: b^(d-2)=b^-1
        return alg_mulmod(aa, binv, d);
      }
    }
  }
  return 0; // astronomically unlikely across 16 walks; payload then poisoned
}
// The original linear log scan: Find k such that gp ^ k mod p = target
FPSAN_HOST_DEVICE constexpr u64 alg_dlog_scan(u64 target, u64 gp, u64 d, u64 p) {
  u64 cur = 1 % p;
  for (u64 k = 0; k < d; ++k) {
    if (cur == target)
      return k;
    cur = alg_mulmod(cur, gp, p);
  }
  return d; // unreachable when target lies in <g>
}
FPSAN_HOST_DEVICE constexpr u64 alg_dlog1(const AlgConfig &c, u64 r) {
  const u64 p = c.n / c.d; // prime field factor (n = p*d)
  const u64 rp = r % p;
  if (rp == 0)
    return c.d; // sentinel: value vanishes in the F_p factor
  if (rp == 1)
    return 0;
  const u64 gp = c.g % p;                        // order-d generator in F_p^*
  const u64 target = alg_powmod(rp, c.d + 1, p); // r's order-d component
  if (target == 1 % p)
    return 0;
  // Small order: the exact O(d) scan (proven, used by fp8/16/32). Large
  // order (64-bit, d ~ 2^31): Pollard's rho, O(sqrt d). Both return the
  // same unique k in [0, d).
  if (c.d > (u64{1} << 20)) {
    const u64 k = alg_dlog_rho(target, gp, c.d, p);
    return alg_powmod(gp, k, p) == target ? k : c.d;
  }
  // Try pollard's rho here for medium size orders as well
  if (c.d > (u64{1} << 10)) {
    const u64 k = alg_dlog_rho(target, gp, c.d, p);
    if (k < c.d && alg_powmod(gp, k, p) == target)
      return k;
  }
  return alg_dlog_scan(target, gp, c.d, p);
}
FPSAN_HOST_DEVICE constexpr u64 alg_log1(const AlgConfig &c, u64 r) {
  if (!c.two_moduli)
    return alg_tagged1(c, r, 0x6C6F67ull /*"log"*/);
  if (!alg_is_fin(c, r))
    return c.nan_code; // log(Inf/NaN)
  if (r == 0)
    return c.inf_code; // log(0) = -inf (unsigned pole)
  if (r == 1)
    return 0;
  const u64 k = alg_dlog1(c, r);
  if (k >= c.d)
    return c.nan_code; // vanishes in the F_p factor
  // (n/d)*k, in the ADDITIVE order-d subgroup {0, n/d, ...} ~ Z/d.
  return ((c.n / c.d) * k) % c.n;
}
template <class Bits> FPSAN_HOST_DEVICE constexpr Bits alg_log(const AlgConfig &c, Bits r) {
  return alg_lanewise1(r, [&](u64 x) { return alg_log1(c, x); });
}

// exp2 / log2: a second exp/log homomorphism pair on the SAME order-d
// channel, related to exp/log by a fixed base change exp2(v) = exp(K*v),
// i.e. exp2(x) == exp(x)^K. The true inter-base constant log2(e) is
// irrational, hence unrepresentable, so K is a fixed pseudo-random unit
// mod d (exactly the role Triton's rcpLog2 magic constant plays): exp2
// honors its OWN homomorphism exp2(a+b)==exp2(a)*exp2(b) and log2 is its
// exact inverse log2(x*y)==log2(x)+log2(y), but NO numeric relation to
// exp/log is claimed. Field variants (no d-channel) fall back to tags.
// exp_b / log_b family: a base-b exponential on the same order-d channel,
// exp_b(v) = g^(K_b * v mod d) for a fixed per-base unit K_b, with log_b its
// exact inverse. The true inter-base constant (log_b e) is irrational, so K_b
// is a magic number (exactly the role Triton's rcpLog2 plays): each base keeps
// its own homomorphism and inverse, and NO numeric relation between bases is
// claimed. Base e is K=1 (alg_exp1/alg_log1); base 2 and base 10 use the
// distinct salted constants below. Semantics with no d-channel fall back to
// tags.
// A pseudo-random unit in [2, d-1] (so != 0 and != 1, the base-e multiplier).
// The order-d subgroup is cyclic of PRIME order, so all d-1 non-identity
// elements are generators -- one per base. d < 3 (only PythagoreanRing2 at
// fp8, d=3) has a single non-trivial unit, so its bases coincide; for every other modulus the
// bases are kept distinct. If d==0, callers return a tag.
FPSAN_HOST_DEVICE constexpr u64 alg_base_unit(u64 d, u64 magic) {
  return (d < 3) ? 0 : 2 + magic % (d - 2);
}
FPSAN_HOST_DEVICE constexpr u64 alg_exp2_base(u64 d) { return alg_base_unit(d, 2654435761ull); }
FPSAN_HOST_DEVICE constexpr u64 alg_exp10_base(u64 d) {
  if (d < 3)
    return 0;
  const u64 k = alg_base_unit(d, 3266489917ull);
  // keep base 10 distinct from base 2 (and from base e = 1; k >= 2 already)
  return (k == alg_exp2_base(d)) ? (2 + (k - 1) % (d - 2)) : k;
}

template <class U> FPSAN_HOST_DEVICE constexpr U alg_expb_1(const AlgConfig &c, U a, U K, u64 tag) {
  if (!c.two_moduli)
    return alg_tagged1<U>(c, a, tag);
  if (!alg_is_fin<U>(c, a))
    return static_cast<U>(c.nan_code);
  // exponent (K * (a mod d)) mod d: the product needs the widened type.
  return alg_powmod<U>(
      static_cast<U>(c.g),
      alg_mulmod<U>(K, static_cast<U>((a) % static_cast<U>(c.d)), static_cast<U>(c.d)),
      static_cast<U>(c.n)); // g^(K*v mod d)
}
template <class U> FPSAN_HOST_DEVICE constexpr U alg_logb_1(const AlgConfig &c, U r, U K, u64 tag) {
  if (!c.two_moduli)
    return alg_tagged1<U>(c, r, tag);
  if (!alg_is_fin<U>(c, r))
    return static_cast<U>(c.nan_code);
  if (r == 0)
    return static_cast<U>(c.inf_code);
  if (r == static_cast<U>(1))
    return static_cast<U>(0);
  const U k = static_cast<U>(alg_dlog1(c, static_cast<u64>(r)));
  if (k >= static_cast<U>(c.d))
    return static_cast<U>(c.nan_code);
  const U Kinv = alg_powmod<U>(K, static_cast<U>((c.d - 2)),
                               static_cast<U>(c.d)); // K^(d-2) = K^-1 mod prime d
  // (n/d) * (Kinv*k mod d), reduced mod n: widen the outer product.
  return static_cast<U>((static_cast<wider_t<U>>(static_cast<U>(c.n / c.d)) *
                         alg_mulmod<U>(Kinv, k, static_cast<U>(c.d))) %
                        static_cast<U>(c.n));
}
template <class U> FPSAN_HOST_DEVICE constexpr U alg_exp2_1(const AlgConfig &c, U a) {
  return alg_expb_1<U>(c, a, static_cast<U>(alg_exp2_base(c.d)), 0x65787032ull /*"exp2"*/);
}
template <class U> FPSAN_HOST_DEVICE constexpr U alg_log2_1(const AlgConfig &c, U r) {
  return alg_logb_1<U>(c, r, static_cast<U>(alg_exp2_base(c.d)), 0x6C6F6732ull /*"log2"*/);
}
template <class U> FPSAN_HOST_DEVICE constexpr U alg_exp10_1(const AlgConfig &c, U a) {
  return alg_expb_1<U>(c, a, static_cast<U>(alg_exp10_base(c.d)), 0x6578703130ull /*"exp10"*/);
}
template <class U> FPSAN_HOST_DEVICE constexpr U alg_log10_1(const AlgConfig &c, U r) {
  return alg_logb_1<U>(c, r, static_cast<U>(alg_exp10_base(c.d)), 0x6C6F673130ull /*"log10"*/);
}

template <class Bits> FPSAN_HOST_DEVICE constexpr Bits alg_exp2(const AlgConfig &c, Bits a) {
  return alg_lanewise1(a, [&](auto x) { return alg_exp2_1(c, x); });
}
template <class Bits> FPSAN_HOST_DEVICE constexpr Bits alg_log2(const AlgConfig &c, Bits r) {
  return alg_lanewise1(r, [&](auto x) { return alg_log2_1(c, x); });
}
template <class Bits> FPSAN_HOST_DEVICE constexpr Bits alg_exp10(const AlgConfig &c, Bits a) {
  return alg_lanewise1(a, [&](auto x) { return alg_exp10_1(c, x); });
}
template <class Bits> FPSAN_HOST_DEVICE constexpr Bits alg_log10(const AlgConfig &c, Bits r) {
  return alg_lanewise1(r, [&](auto x) { return alg_log10_1(c, x); });
}

// ---- sin / cos via an order-d rotation in (Z/n)[i], i^2 = -1 -----------
// cos(x)=Re(omega^(x mod d)), sin(x)=Im(omega^(x mod d)). Since omega has
// order d and the complex multiplication realizes the rotation, the
// angle-addition formulas hold exactly in Z/n. Semantics without this
// rotor keep sin/cos as tags.
template <class U> struct AlgC {
  U re = 0, im = 0;
};
template <class U> FPSAN_HOST_DEVICE constexpr AlgC<U> alg_cmul(AlgC<U> a, AlgC<U> b, U n) {
  // (ar+ai i)(br+bi i) = (ar br - ai bi) + (ar bi + ai br) i  mod n
  const U re = alg_submod<U>(alg_mulmod<U>(a.re, b.re, n), alg_mulmod<U>(a.im, b.im, n), n);
  const U im = alg_addmod<U>(alg_mulmod<U>(a.re, b.im, n), alg_mulmod<U>(a.im, b.re, n), n);
  return {re, im};
}
template <class U> FPSAN_HOST_DEVICE constexpr AlgC<U> alg_cpow(AlgC<U> base, U e, U n) {
  AlgC<U> r{static_cast<U>((1) % n), 0};
  while (e) {
    if (e & 1)
      r = alg_cmul<U>(r, base, n);
    base = alg_cmul<U>(base, base, n);
    e >>= 1;
  }
  return r;
}
template <class U> FPSAN_HOST_DEVICE constexpr AlgC<U> alg_rotor(const AlgConfig &c, U r) {
  return alg_cpow<U>(AlgC<U>{static_cast<U>(c.omega_re), static_cast<U>(c.omega_im)},
                     static_cast<U>(r % static_cast<U>(c.d)), static_cast<U>(c.n));
}
template <class U> FPSAN_HOST_DEVICE constexpr U alg_cos1(const AlgConfig &c, U r) {
  if (!c.has_sin_cos)
    return alg_tagged1<U>(c, r, 0x636F73ull /*"cos"*/);
  if (!alg_is_fin<U>(c, r))
    return static_cast<U>(c.nan_code);
  return alg_rotor<U>(c, r).re;
}
template <class U> FPSAN_HOST_DEVICE constexpr U alg_sin1(const AlgConfig &c, U r) {
  if (!c.has_sin_cos)
    return alg_tagged1<U>(c, r, 0x73696Eull /*"sin"*/);
  if (!alg_is_fin<U>(c, r))
    return static_cast<U>(c.nan_code);
  return alg_rotor<U>(c, r).im;
}
template <class Bits> FPSAN_HOST_DEVICE constexpr Bits alg_cos(const AlgConfig &c, Bits r) {
  return alg_lanewise1(r, [&](auto x) { return alg_cos1(c, x); });
}
template <class Bits> FPSAN_HOST_DEVICE constexpr Bits alg_sin(const AlgConfig &c, Bits r) {
  return alg_lanewise1(r, [&](auto x) { return alg_sin1(c, x); });
}

} // namespace detail
} // namespace fpsan

#endif // FPSAN_DETAIL_ALGEBRAIC_HPP
