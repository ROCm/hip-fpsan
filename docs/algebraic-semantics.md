# Algebraic semantics in detail

This document gives the precise working model behind
`Semantics::Field`, `Semantics::FieldFast`,
`Semantics::FieldWithMulCasts`, `Semantics::SophieGermainRing`, and
`Semantics::PythagoreanRing`. It is meant for engineers who are comfortable with
finite rings and want to understand what the implementation promises, without
turning the user guide into a math paper.

For the short decision guide and scorecard, see
[algebraic-fpsan.md](algebraic-fpsan.md). For a more formal mathematical
development, see [reducing-floats-mod-p.pdf](reducing-floats-mod-p.pdf). For
the Triton construction that algebraic FPSan contrasts with, see
[triton-fpsan.md](triton-fpsan.md).

## Orientation

Triton-style FPSan and algebraic FPSan both replace floating-point arithmetic by
a deterministic fingerprint semantics. They differ at the AST leaves: the
places where native floating-point inputs and constants are first converted into
fingerprints.

Triton uses a tuned bijection from IEEE-754 bit patterns to `w`-bit fingerprints and
then evaluates arithmetic in `Z/2^w`. That bijection is designed to make a few
special values behave well, notably `0`, `1`, and negation, but it is not a
homomorphism from the original exact values. For example, the fingerprint of
`2.0f` is not the integer `2`, so `x + x` and `2*x` are not the same fingerprint
in general.

Algebraic FPSan changes that leaf map: a finite binary float is treated as its
exact dyadic value and reduced into a finite ring. The operations then run in
that ring. In the faithful algebraic variants, this makes every
rational-function identity true of the exact values also true of the
fingerprints, within the limits of the chosen finite modulus. `FieldFast` keeps
the same leaf map but deliberately tags division and roots.

Neither model attempts to preserve IEEE-754 rounding. Both are checking exact
algebraic equivalence, not bitwise floating-point equivalence.

## Public variants

| Public semantics | Internal variant | Modulus | Main structure |
|---|---|---|---|
| `Field` | `Field1` | prime `p` | finite field, cheap casts |
| `Field2` | `Field2` | different prime `p` | second collision set for `Field` |
| `FieldFast` | `Field1` | same prime `p` as `Field` | finite field encoding, fast tagged division/roots |
| `FieldFast2` | `Field2` | same prime `p` as `Field2` | second collision set for `FieldFast` |
| `FieldWithMulCasts` | `Field1` | same prime `p` as `Field` | finite field, multiplicative casts |
| `FieldWithMulCasts2` | `Field2` | same prime `p` as `Field2` | second collision set with multiplicative casts |
| `SophieGermainRing` | `Exp1` | `n = p*d`, `p = 2*d + 1` | exp/log channel |
| `SophieGermainRing2` | `Exp2` | different `p*d` | second collision set for Sophie Germain |
| `PythagoreanRing` | `Trig1` | `n = p*d`, `p = 4*d + 1` | exp/log plus sin/cos channel |
| `PythagoreanRing2` | `Trig2` | different `p*d` | second collision set for Pythagorean |

The Field-family variants deliberately share prime tables: `Field`, `FieldFast`,
and `FieldWithMulCasts` have the same finite fingerprints for input
values, and likewise for their `2` twins. That makes it possible to switch only
the semantics of expensive operations or casts while keeping the collision set
fixed.

The `2` variants are not semantically different families. They use independent
moduli satisfying the same design constraints, so a user can rerun a check with
a different collision set.

## Finite values

Every finite binary floating-point value has the form:

```text
x =  m * 2^e
```

where `m` is an integer significand and `e` may be negative. Since every
algebraic modulus `n` is odd, `gcd(2, n) = 1`, so `2` is a unit in `Z/nZ`. In
other words, there is a residue `2^-1 mod n` such that:

```text
2 * 2^-1 == 1 mod n
```

That is the key point that makes negative exponents meaningful in a finite
residue ring. The finite fingerprint is:

```text
phi_n(m * 2^e) = m * 2^e mod n
```

For `e >= 0`, the factor `2^e` is the usual power of `2`. For `e < 0`, it means
`(2^-1)^(-e)` modulo `n`. Thus subnormal values and small normal values with
negative binary exponents are handled by the same formula as integers and large
normal values.

That one formula is why the algebraic semantics preserve exact additive and
multiplicative value identities:

```text
phi_n(x + y) = phi_n(x) + phi_n(y)
phi_n(x * y) = phi_n(x) * phi_n(y)
```

The implementation lives in
[`include/fpsan/detail/algebraic.hpp`](../include/fpsan/detail/algebraic.hpp).
`Value` dispatches to it when `detail::is_algebraic_semantics(S)` is true.

## Non-finite extension

Finite residues use the range `[0, n)`. Two extra codes are reserved:

```text
n     infinity
n + 1 NaN
```

That is why the chosen modulus must leave at least two unused codes in the
underlying fingerprint width.

The infinity is unsigned. Both `+inf` and `-inf` map to the same fingerprint. That
means the model intentionally emits `NaN` for `inf + inf` as well as `inf - inf`,
because it does not carry the sign needed to distinguish the IEEE-754 cases.

The important safety property is that a nonzero finite float never embeds as the
zero residue. Since:

```text
phi_n(m * 2^e) == 0  iff  n divides m
```

and the moduli are much larger than the significand range for the corresponding
format, only `m == 0` can hit zero. Division by zero therefore means division by
a real zero, not an accidental fingerprint collision.

## Field semantics

`Semantics::Field` uses a prime modulus near the top of the available fingerprint
range. The result is a finite field `F_p`, so every nonzero finite fingerprint
has an inverse. `Semantics::FieldFast` and
`Semantics::FieldWithMulCasts` use the same primes and the same
finite-value fingerprints; they differ only in which expensive operations opt
into the full algebraic structure.

The primes are not arbitrary "largest primes below `2^w`". They satisfy several
constraints at once:

- `p` is odd, so dyadic values reduce modulo `p`;
- `p <= 2^w - 2`, so `p` and `p+1` can be the infinity and NaN codes;
- `p` is large enough that nonzero finite significands cannot reduce to zero;
- `p == 3 mod 4`, so `sqrt` can be implemented as a multiplicative power map in
  the faithful Field variants;
- `p == 2 mod 3`, so `cbrt` can be a perfect multiplicative cube root in the
  faithful Field variants;
- across the `fp4 -> fp8 -> fp16 -> fp32 -> fp64` chain, the values `p_w - 1`
  form a divisibility tower with coprime cofactors, enabling the optional
  multiplicative casts.

The first two root constraints combine to `p == 11 mod 12`.

### Roots

In a prime field with `p == 3 mod 4`, the map:

```text
sqrt(x) = x^((p+1)/4)
```

is multiplicative. It satisfies:

```text
sqrt(x*y) == sqrt(x) * sqrt(y)
```

for every residue. The stronger identity `sqrt(x)^2 == x` holds on the square
residues and gives `-x` on the nonsquares. That is not a bug; a homomorphic
square-root choice cannot be a two-sided inverse to squaring on all of a group
whose order is even.

For `cbrt`, the exponent `3` must be invertible modulo the group order. The
`p == 2 mod 3` part of `p == 11 mod 12` gives that, so `cbrt(x)^3 == x` holds for
all residues in `Field`.

`FieldFast` intentionally does not use these root power maps. It shares the
prime choices so its finite fingerprints match `Field`, but `sqrt`, `rsqrt`, and
`cbrt` are deterministic tags there.

### FieldFast semantics

`Semantics::FieldFast` keeps the same leaf map and the same `+`, `-`, and `*`
operations as `Field`. It gives up the expensive field inverse used by division
and the expensive root power maps. For finite nonzero operands, division returns
a deterministic op-tagged fingerprint instead of `a * inverse(b)`, except for
the cheap special cases `0/x == 0` and `x/1 == x`.

The practical consequence is that polynomial and linear-algebraic identities
still work:

```text
2 + 2 == 4
x + x == 2*x
(a+b)^2 == a^2 + 2ab + b^2
```

but field-inverse and root identities do not:

```text
x/x == 1
(a/b)*b == a
sqrt(x*y) == sqrt(x)*sqrt(y)
```

Use it when benchmarks show that division or roots dominate and those identities
are not part of the check you care about.

### Casts

A cast between different floating widths cannot be a ring homomorphism in
Field-family semantics: if the source and destination primes differ, every additive-group
homomorphism between the two finite fields is trivial. That already rules out
any nonzero ring homomorphism.

The default `Field` and `FieldFast` variants therefore use cheap deterministic
casts: finite residues reduce modulo the destination prime, while zero and
non-finite sentinels map to their corresponding destination codes.

`FieldWithMulCasts` instead provides multiplicative casts on nonzero
finite fingerprints. The multiplicative groups `F_p^*` are cyclic, and casts are
defined in discrete-log coordinates. The prime choices make the group orders
line up:

```text
p_fp4 - 1  divides  p_fp8 - 1  divides  p_fp16 - 1
          divides  p_fp32 - 1 divides  p_fp64 - 1
```

with coprime cofactors. That lets the implementation define widening and
narrowing maps that:

```text
cast(x*y) == cast(x) * cast(y)
cast<T>(cast<U>(x)) == cast<T>(x)
narrow(widen(x)) == x
```

inside the supported chain. `fp6` is standalone and not part of this cast tower.
Same-width format changes in the tower use the same idea with different
primitive roots for the two format classes at that width. Thus `f16` and `bf16`
are not silently coalesced, and neither are the `e4m3` and `e5m2` fp8 families:
the cast is a multiplicative automorphism of the same finite field, not the
identity. Same-format same-width casts remain identity.
The `fp32`/`fp64` edge uses Pohlig-Hellman instead of a linear discrete-log scan,
because `p_fp32 - 1` is roughly `2^32`; the group order is deliberately smooth
enough for that. The plain `Field` and `FieldFast` variants keep the same primes
so users can switch to or from `FieldWithMulCasts` without changing
any finite input fingerprints.

## Why composite rings exist

In a prime field alone, there is no useful exponential homomorphism from
addition to multiplication:

```text
(F_p, +) -> (F_p^*, *)
```

The additive group has order `p`; the multiplicative group has order `p-1`.
Those orders are coprime, so every group homomorphism is trivial.

To model `exp(a+b) == exp(a)*exp(b)`, the fingerprint must encode a second residue
modulo some `d` for which the multiplicative side has an order-`d` subgroup.
The compact way to carry both residues is the Chinese Remainder Theorem:

```text
Z/(p*d)  ~=  F_p x F_d
```

The cost is that `Z/(p*d)` is a ring with zero-divisors, not a field. Division by
a zero-divisor returns the algebraic NaN fingerprint.

## Sophie Germain ring

`Semantics::SophieGermainRing` uses:

```text
n = p*d,   p = 2*d + 1,   p and d prime.
```

Strictly, `d` is the Sophie Germain prime and `p` is the corresponding safe
prime; the public name refers to the pair.

The implementation chooses an element `g` that has order `d` in the `F_p`
factor and is `1` in the `F_d` factor. Then:

```text
exp(x) = g^(x mod d)
```

Because the exponent only depends on `x mod d`, this gives:

```text
exp(a+b) == exp(a) * exp(b)
```

`exp2` and `exp10` use the same channel with fixed nonzero scale factors:

```text
exp_b(x) = g^(K_b * x mod d)
```

The constants `K_b` are fingerprints for the bases, not numerical logarithms.
For example, `exp2(1)` is not the residue of the integer `2`. The goal is to
preserve the base's own law, not to model the real value of the base.

`log`, `log2`, and `log10` are the discrete-log duals of these exponentials. They
map the order-`d` multiplicative subgroup back into the additive order-`d`
subgroup of `Z/(p*d)`, so:

```text
log(x*y) == log(x) + log(y)
exp(log(exp(x))) == exp(x)
```

The discrete log is a small scan for the 8-, 16-, and 32-bit cases. For the
64-bit cases, where `d` is about `2^31`, the implementation uses Pollard's rho
for the order-`d` channel.

## Pythagorean ring

`Semantics::PythagoreanRing` uses:

```text
n = p*d,   p = 4*d + 1,   p and d prime.
```

This keeps an order-`d` exp/log channel and additionally makes `-1` a square in
the `F_p` factor. That gives a finite-ring analogue of complex rotations.

The implementation chooses an order-`d` rotor:

```text
omega = omega_re + i*omega_im,  i^2 = -1
```

in `(Z/nZ)[i]`, with norm `1`. It defines:

```text
cos(x) = Re(omega^(x mod d))
sin(x) = Im(omega^(x mod d))
```

Complex multiplication then gives the angle-addition laws exactly:

```text
cos(a+b) == cos(a)*cos(b) - sin(a)*sin(b)
sin(a+b) == sin(a)*cos(b) + cos(a)*sin(b)
cos(x)^2 + sin(x)^2 == 1
```

The Pythagorean choices give up perfect `cbrt`. For prime `d > 3`, the condition
`p = 4*d + 1` forces `d == 1 mod 3`, so `d-1` is divisible by `3`; for the small
`d = 3` case, `p-1` is divisible by `3`. Either way the unit-group exponent has a
factor of `3`, so no global cube-root power map exists. `cbrt` is therefore a
tag in this variant.

The Pythagorean `d` is also smaller than the Sophie Germain `d` at the same
fingerprint width, so exp/trig image collisions are somewhat more likely.

## Tags

Some functions have no algebraic structure this finite model can use, or are
intentionally made opaque in a faster semantics such as `FieldFast`. For those, the
implementation creates a deterministic tag:

```text
same op + same operands     -> same tag
different op or operands    -> usually different tag
```

Tags are not a weak implementation fallback. For arbitrary extern/libdevice
calls and genuinely independent transcendental values, an opaque fresh generator
is the right sanitizer model.

`fpsan::extern_tagged` exposes this path for unmodeled calls. In
`Semantics::Triton` it matches Triton's extern-tagging formula; in algebraic
semantics it uses the same
symbol-distinct idea inside the algebraic residue ring.

## Collisions and the `2` variants

The algebraic leaf map is not injective. It cannot be: a `w`-bit fingerprint cannot
uniquely encode all finite values plus non-finite sentinels while also being a
homomorphic residue model.

For the usual comparison pattern, this is often acceptable. If two expressions
use the same AST input leaves, meaning the same native floating-point inputs or
constants first converted into fingerprints, leaf collisions are shared by both
sides. A false match between genuinely different expressions happens when the
chosen modulus divides the exact difference between the two modeled expressions.

The `2` variants give a fixed second modulus under the same constraints. If a
test unexpectedly passes or fails in one algebraic variant, rerun the matching
`2` variant to check whether the result is stable under a different modulus.

## What is intentionally not modeled

- IEEE-754 rounding and exception flags.
- Overflow from a large finite expression to infinity.
- Signed zero and signed infinity.
- IEEE-754 numeric ordering for comparisons, `min`, or `max`.
- Recovering a useful float from an algebraic fingerprint.

These are deliberate boundaries of the model. Algebraic FPSan answers "did these
two computations have the same exact algebraic fingerprint under this semantics?"
It does not answer "would IEEE-754 arithmetic produce the same final bits?"
