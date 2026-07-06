// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Standalone verification of the algebraic-fpsan payload algebra
// (fpsan/detail/algebraic.hpp), prior to wiring it into Value.
//
//   c++ -std=c++17 -I include tests/algebraic_proto_test.cpp -o /tmp/algt && /tmp/algt
// ----------------------------------------------------------------------------
#include "fpsan/detail/algebraic.hpp"

#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace fpsan::detail;

static long pass = 0, fail = 0;
static void check(bool ok, const char *msg) {
  if (ok)
    ++pass;
  else {
    ++fail;
    std::printf("  FAIL: %s\n", msg);
  }
}

template <class FT> static u64 raw_bits(FT v) {
  using B = typename fp_traits<FT>::bits_type;
  B b{};
  std::memcpy(&b, &v, sizeof b);
  return static_cast<u64>(b);
}
template <class FT> static u64 emb(const AlgConfig &c, FT v) { return alg_embed1(c, raw_bits(v)); }

template <class FT> static void test_width(const char *name, AlgVariant field, AlgVariant exp) {
  std::printf("== %s ==\n", name);
  AlgConfig cf = make_alg_config<FT>(field);
  AlgConfig ce = make_alg_config<FT>(exp);

  // encoding basics
  check(emb<FT>(cf, static_cast<FT>(0)) == 0, "phi(0)=0");
  check(emb<FT>(cf, static_cast<FT>(1)) == 1, "phi(1)=1");
  check(emb<FT>(cf, static_cast<FT>(2)) == 2 % cf.n, "phi(2)=2");
  check(emb<FT>(cf, static_cast<FT>(-1)) == cf.n - 1, "phi(-1)=-1");
  check(emb<FT>(cf, static_cast<FT>(0.5)) == cf.inv2, "phi(1/2)=inv2");

  // qr-order primitives: nonzero residues that are squares in the selected
  // prime factor are the "positive" half. The selected factor has -1 as a
  // non-square, so exactly one of x and -x is qr for ordinary nonzero x.
  u64 non_qr = 0;
  for (u64 r = 2; r < cf.n && non_qr == 0; ++r)
    if (!alg_is_nonzero_qr(cf, r))
      non_qr = r;
  check(alg_is_nonzero_qr(cf, u64{1}), "1 is qr-positive");
  check(!alg_is_nonzero_qr(cf, u64{0}), "0 is not qr-positive");
  check(non_qr != 0, "found a finite non-qr residue");
  check(alg_is_nonzero_qr(cf, alg_neg1(cf, non_qr)), "negative of non-qr is qr-positive");
  check(alg_qr_abs(cf, u64{1}) == 1, "qr_abs keeps qr-positive residues");
  check(alg_qr_abs(cf, non_qr) == alg_neg1(cf, non_qr), "qr_abs negates non-qr residues");
  check(alg_qr_less(cf, u64{0}, u64{1}), "qr order: zero < qr");
  check(!alg_qr_less(cf, u64{0}, non_qr), "qr order: zero is not < non-qr");
  check(!alg_qr_less(cf, non_qr, u64{0}), "qr order: non-qr is not < zero");
  check(alg_qr_less(cf, non_qr, u64{1}), "qr order: non-qr < qr");
  check(!alg_qr_less(cf, u64{1}, non_qr), "qr order: qr is not < non-qr");
  check(alg_qr_min(cf, u64{1}, non_qr) == non_qr, "qr min selects non-qr");
  check(alg_qr_max(cf, u64{1}, non_qr) == 1, "qr max selects qr");

  // ring homomorphism + value-faithfulness on exactly-representable integers
  long hadd = 0, hmul = 0, n = 0;
  for (int i = -8; i <= 8; ++i)
    for (int j = -8; j <= 8; ++j, ++n) {
      hadd += alg_add1(cf, emb<FT>(cf, static_cast<FT>(i)), emb<FT>(cf, static_cast<FT>(j))) ==
              emb<FT>(cf, static_cast<FT>(i + j));
      hmul += alg_mul1(cf, emb<FT>(cf, static_cast<FT>(i)), emb<FT>(cf, static_cast<FT>(j))) ==
              emb<FT>(cf, static_cast<FT>(i * j));
    }
  check(hadd == n, "phi(i+j)=phi(i)+phi(j)");
  check(hmul == n, "phi(i*j)=phi(i)*phi(j)");
  check(emb<FT>(cf, static_cast<FT>(2)) != 0 &&
            alg_add1(cf, emb<FT>(cf, static_cast<FT>(2)), emb<FT>(cf, static_cast<FT>(2))) ==
                emb<FT>(cf, static_cast<FT>(4)),
        "2+2==4 (value-faithful)");

  // field: x/x == 1 for every nonzero value (Field variant has no zero-divisors)
  long xx = 0, cnt = 0;
  for (int i = 1; i < 200; ++i, ++cnt) {
    u64 r = emb<FT>(cf, static_cast<FT>(i));
    xx += alg_div1(cf, r, r) == 1;
  }
  check(xx == cnt, "x/x == 1 (field)");

  // Inf / NaN table. The leaf ops are templated on the payload type U; pass
  // u64-typed arguments so U deduces to u64 here (raw int literals would deduce
  // a signed U with no wider_t mapping).
  check(alg_div1(cf, u64{1}, u64{0}) == cf.inf_code, "1/0 = Inf");
  check(alg_div1(cf, u64{1}, cf.inf_code) == 0, "1/Inf = 0");
  check(alg_add1(cf, cf.inf_code, cf.inf_code) == cf.nan_code, "Inf+Inf = NaN");
  check(alg_mul1(cf, u64{0}, cf.inf_code) == cf.nan_code, "0*Inf = NaN");
  check(alg_add1(cf, cf.nan_code, u64{5}) == cf.nan_code, "NaN absorbs");

  // exp homomorphism (SophieGermain variant, >= 8 bits): exp(a+b) = exp(a)*exp(b)
  check(ce.two_moduli, "SophieGermain variant has exp at this width");
  long eh = 0, en = 0;
  u64 seed = 12345;
  for (int k = 0; k < 5000; ++k, ++en) {
    seed = seed * 6364136223846793005ull + 1;
    u64 a = (seed >> 11) % ce.n;
    seed = seed * 6364136223846793005ull + 1;
    u64 b = (seed >> 11) % ce.n;
    eh += alg_exp1(ce, alg_add1(ce, a, b)) == alg_mul1(ce, alg_exp1(ce, a), alg_exp1(ce, b));
  }
  check(eh == en, "exp(a+b)=exp(a)*exp(b) (CRT)");
  check(alg_exp1(ce, u64{0}) == 1, "exp(0)=1");

  // exp2 is a second homomorphism on the same channel; log2 inverts it
  long e2 = 0, l2 = 0;
  seed = 999;
  for (int k = 0; k < 5000; ++k) {
    seed = seed * 6364136223846793005ull + 1;
    u64 a = (seed >> 11) % ce.n;
    seed = seed * 6364136223846793005ull + 1;
    u64 b = (seed >> 11) % ce.n;
    e2 += alg_exp2_1(ce, alg_add1(ce, a, b)) == alg_mul1(ce, alg_exp2_1(ce, a), alg_exp2_1(ce, b));
    l2 += alg_exp2_1(ce, alg_log2_1(ce, alg_exp2_1(ce, a))) == alg_exp2_1(ce, a);
  }
  check(e2 == en, "exp2(a+b)=exp2(a)*exp2(b) (CRT)");
  check(l2 == en, "exp2(log2(exp2 v))=exp2 v (log2 inverts exp2)");
  check(alg_exp2_1(ce, u64{0}) == 1, "exp2(0)=1");
  check(alg_exp2_1(ce, u64{1}) != alg_exp1(ce, u64{1}), "exp2 != exp (distinct base change)");

  // Field variant does NOT get the homomorphism (hash-based tag)
  check(!cf.two_moduli, "Field variant has no exp");
}

int main() {
  // sub-byte fallback: SophieGermain variants behave like Field below 8 bits
  check(!alg_modulus(AlgVariant::SophieGermain1, 4).two_moduli,
        "SophieGermain1 @4-bit: no exp (fallback)");
  check(!alg_modulus(AlgVariant::SophieGermain1, 6).two_moduli,
        "SophieGermain1 @6-bit: no exp (fallback)");
  check(alg_modulus(AlgVariant::SophieGermain1, 8).two_moduli, "SophieGermain1 @8-bit: has exp");
  check(alg_modulus(AlgVariant::SophieGermain1, 4).n == alg_modulus(AlgVariant::Field1, 4).n,
        "SophieGermain1 @4-bit reuses Field1's prime");

  test_width<float>("float (32-bit)", AlgVariant::Field1, AlgVariant::SophieGermain1);
#if FPSAN_HAS_FLOAT16
  test_width<_Float16>("_Float16 (16-bit)", AlgVariant::Field1, AlgVariant::SophieGermain1);
#endif

  std::printf("\npassed %ld, failed %ld\n", pass, fail);
  return fail == 0 ? 0 : 1;
}
