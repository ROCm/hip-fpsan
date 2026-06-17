# Triton FPSan from first principles

A pedagogical reconstruction of Triton's floating-point sanitizer **FPSan**,
built up one ingredient at a time. Each ingredient is motivated by a concrete
property the previous version doesn't have.

FPSan was invented by the [Triton compiler
project](https://github.com/triton-lang/triton); we are not the original
authors. There is also an [excellent blog
post](https://cp4space.hatsya.com/2026/05/03/schanuels-conjecture-and-the-semantics-of-fpsan/)
by one of its authors covering the math. The present document is our own
attempt to understand the design by deriving it from scratch, motivated
entirely by concrete numerical-engineering needs and front-loading as little
mathematical machinery as possible. A real, complete implementation of these
ideas is this library — see the [README](../README.md) and the headers under
[`include/fpsan/`](../include/fpsan).

For the value-residue alternatives to Triton FPSan, see
[Algebraic FPSan](algebraic-fpsan.md). For the operational details of
those semantics, see [Algebraic FPSan semantics](algebraic-semantics.md).

## The problem FPSan solves

You're maintaining a numerical kernel and you refactor it for performance.
Maybe you replaced

```
((a + b) + c) + d
```

with

```
(a + b) + (c + d)
```

to enable two-way SIMD, or you traded a `mul` for an `fma`, or you reordered
a softmax reduction to improve register pressure. You diff the output bits
against the old kernel and they differ.

Now what? You're stuck with a question that, in pure IEEE-754 float, has no easy
answer: **did the refactor change the meaning of the calculation, or did it
just rearrange operations in a way that gives a different rounding error?**

You could write a tolerance-based test, but tolerances drift and accumulate;
you could derive an algebraic proof of equivalence by hand, but that doesn't
scale. What you really want is a parallel arithmetic — the same operations,
the same data flow — where **algebraically equivalent expressions produce
bit-identical results**, while genuinely different expressions almost never
collide. Then your test reduces to a single `memcmp` of the kernel's outputs.

That parallel arithmetic is FPSan. The rest of this document is one way to
discover how to build it.

We'll iteratively construct an arithmetic for a `Value` type that stands in
for `float`, the usual single-precision 32-bit `float` type; everything
generalizes to other widths.

## Iteration 1: bit_cast and integer arithmetic

The simplest non-trivial design. A `Value` holds a `uint32_t`. Construction
from a `float` is `std::bit_cast`. The four basic operators are *integer*
operations on the bits:

```cpp
class Value {
    uint32_t bits_;
public:
    explicit Value(float x) : bits_(std::bit_cast<uint32_t>(x)) {}

    friend Value operator+(Value a, Value b) { return from_bits(a.bits_ + b.bits_); }
    friend Value operator*(Value a, Value b) { return from_bits(a.bits_ * b.bits_); }
    friend bool  operator==(Value a, Value b) { return a.bits_ == b.bits_; }
    // ...
};
```

Equality is bit-equality, as it must be.

### What works

The headline property is already here. Take any three values:

```cpp
Value a{1.7f}, b{2.3f}, c{4.9f};
assert((a + b) + c == a + (b + c));   // associativity
assert(a + b == b + a);                // commutativity
assert(a * (b + c) == a*b + a*c);      // distributivity
```

All three hold, because of the familiar rules of integer arithmetic mod
$2^{32}$: integer addition and multiplication are associative and
commutative, and multiplication distributes over addition. Any two
expressions that reduce to the same sum-of-products via these rules will
produce identical 32-bit payloads. That already covers an enormous fraction
of the refactors that motivate FPSan in the first place: reassociation,
factoring, vectorization-driven reordering, loop unrolling — anything
purely structural.

> [!NOTE]
> **Why `uint32_t` and not `int32_t`?** C and C++ make signed integer
> overflow *undefined behaviour*.
> Unsigned integer arithmetic, by contrast, is defined to wrap mod $2^{32}$
> exactly. We need that wraparound: it's what makes our + and × behave like
> well-defined operations on arbitrary bit patterns.

### What doesn't work

Two important kinds of identities fail.

**Multiplicative identity.** The bit pattern of `1.0f` is `0x3F800000`, not
`1`. So:

```cpp
Value x{7.0f};
Value one{1.0f};
assert(one * x == x);   // FAILS
```

Any optimization that simplifies `1 * x → x` (and there are many: unrolling a
power loop, eliminating a multiply by a precomputed constant that happened to
be 1, etc.) will be flagged as a meaning change. False positive.

**Additive inverse.** Floating-point negation is just a sign-bit flip:
`bit_cast(-x) = bit_cast(x) XOR 0x80000000`. Integer negation works nothing
like that. So when you compute `a + (-a)` under iteration 1, you're
integer-adding two bit patterns that differ only in the top bit — a result
that lands far from zero:

```cpp
Value a{3.14f};        // bits = 0x4048F5C3
Value neg_a{-3.14f};   // bits = 0xC048F5C3  (just the sign bit flipped)
assert(a + neg_a == Value{0.0f});   // FAILS:
                                    //   payload = 0x4048F5C3 + 0xC048F5C3
                                    //           = 0x0091EB86 mod 2^32
                                    //   not 0.
```

So `a + (-a)` does not collapse to zero. Any compiler that spots `x - x` or
`x + (-x)` and simplifies it to `0` would be flagged. More false
positives.

Aside: the additive identity works for `+0.0f` in iteration 1, by
accident. `+0.0f` has bit pattern `0x00000000`, which *is* the integer `0`,
so `Value{+0.0f} + x == x` holds. But `-0.0f` has bit pattern `0x80000000`
— a perfectly good additive identity in IEEE-754 float, where `-0.0f + x` also
equals `x` — and that bit pattern is nothing close to zero as an integer,
so `Value{-0.0f} + x == x` fails. IEEE-754 has two zeros and iteration 1 only
handles one of them. Iteration 2 will handle both, as a special case of a
more general construction.

## Iteration 2: A scrambling encoding that handles zero, one, and negation

We want a representation in which `0.0`, `1.0`, and `-x` behave the way the
zero, one, and additive-inverse of integer arithmetic do:

- $\varphi(0.0) = 0$
- $\varphi(1.0) = 1$
- $\varphi(-x) = -\varphi(x)$

The function $\varphi : \mathrm{bits}(\mathtt{float}) \to \mathtt{uint32\_t}$
is what we'll call the **encoding**. A `Value` will now hold
$\varphi(\mathrm{bits}(x))$ instead of $\mathrm{bits}(x)$ directly. We keep
the arithmetic *between* payloads exactly as it was in iteration 1 — plain
integer + and × mod $2^{32}$ — so all of iteration 1's good properties
(associativity, commutativity, distributivity) come along for free; we
never touch the operations themselves, only the input/output encoding.

The interesting question is *how to construct $\varphi$* so the three
constraints above hold.

> [!NOTE]
> **A bit of math vocabulary, used sparingly.** We need $\varphi$ to be
> *injective*: no two distinct float bit patterns may produce the same
> payload, otherwise unrelated inputs would collide at the encoding stage
> already. Since $\varphi$ maps a $2^{32}$-element set to itself, an
> injective $\varphi$ is automatically a **bijection** — invertibility
> falls out for free, even though no FPSan arithmetic actually invokes the
> inverse. The only place the inverse shows up in a real FPSan
> implementation is debug printing of a `Value`, where the recovered
> `float` is only meaningful when the value happens to be a direct
> embedding of a known input; for anything produced by FPSan arithmetic,
> the recovered `float` is a meaningless scramble.

### Why not just leave the bits alone?

The obvious question before designing $\varphi$: why bother? Why not take
$\varphi$ to be the identity, fixing up only the three constraints above
locally and leaving the rest of the bits alone?

The sanitizer's whole job is to make distinct expressions produce distinct
payloads. With $\varphi$ near the identity, the payload of `a + b` is
roughly `bits(a) + bits(b)`: close to either input, predictable from its
neighbors, and *very* prone to colliding with other expressions whose bit
patterns happen to land nearby. We'd be back to false positives, just of a
subtler kind than iteration 1's.

What we want from $\varphi$ is exactly what we want from a **hash
function**: spread inputs across the output range as uniformly as we can
manage, so that collisions between unrelated inputs are vanishingly rare.
The standard recipe for building such a hash from $w$-bit words to $w$-bit
words is to compose a handful of cheap bit-mixing steps such as multiplies by odd constants and xor-shifts.

### Constructing $\varphi$

Here is one concrete construction (essentially Triton's). It works on the
magnitude and sign separately to satisfy the negation constraint cleanly.

```
phi(u):                                       # u is the 32-bit float bit pattern
    s     = top bit of u                      # the IEEE-754 sign bit
    m     = u with the sign bit cleared       # the remaining 31 bits
    y     = (m * A) mod 2^31                  # multiplicative scramble
    z     = y XOR (y >> 23)                   # xor-shift mix (23 = mantissa width)
    w     = (z * B_s) mod 2^31                # second multiplicative scramble
    return w with the top bit set to s
```

Three free parameters: a constant $A$, and two constants $B_+, B_-$ chosen
for the positive and negative branches.

Two of those parameters are pinned by our constraints:

- **$\varphi(1.0) = 1$.** The bit pattern of $1.0$ is $\mathrm{bias} \cdot
  2^{23}$ in single precision (mantissa zero, exponent equal to the bias
  127). Walk through the recipe with $s = 0$ and $m = 127 \cdot 2^{23}$;
  the output is $(m A \mathbin{\oplus} (mA \gg 23)) \cdot B_+$. Pick any
  odd $A$, compute the intermediate, then choose $B_+$ to be its modular
  inverse. Now $\varphi(1.0) = 1$ by construction.

- **$\varphi(-x) = -\varphi(x)$.** Equivalent to $B_- = -B_+$ modulo $2^{31}$.
  This makes the second multiplicative scramble flip sign when the sign bit
  is set, so the whole encoding becomes negation-equivariant.

The third "constraint", $\varphi(0.0) = 0$, comes for free: $m = 0$ in,
multiplications and xor-shifts of zero are zero, $w = 0$ out.

That's the entire encoding. Each step is individually reversible — a
multiplication by $A$ is inverted by multiplying by $A^{-1}$, and an
xor-shift is inverted by composing it with itself a few times — so the
encoding as a whole is invertible. Evaluation is identical on host and on
GPU and uses nothing beyond a handful of integer multiplies and one xor-shift;
no runtime state.

### What this buys

The three motivating identities all hold:

```cpp
Value x{7.0f};
assert(Value{1.0f}   * x == x);                  // multiplicative identity
assert(Value{0.0f}   + x == x);                  // additive identity
assert(Value{3.14f}  + Value{-3.14f} == Value{0.0f});   // additive inverse
```

And iteration 1's structural properties (associativity, commutativity,
distributivity) continue to hold automatically, because the arithmetic
*between* payloads is still plain integer + and ×.

### What still doesn't fold

Iteration 2 is not magic. Constants do not "compute":

```cpp
assert(Value{1.0f} + Value{2.0f} == Value{3.0f});   // FAILS
assert(Value{2.0f} * Value{3.0f} == Value{6.0f});   // FAILS
```

The payload of `1.0f + 2.0f` is $1 + \varphi(\mathrm{bits}(2.0))$, which is
some scrambled-looking integer. The payload of `3.0f` is
$\varphi(\mathrm{bits}(3.0))$, a different scrambled-looking integer. There
is no reason for them to be equal.

This is a deliberate non-feature. FPSan is not an evaluator: it never claims
that `1 + 2` should equal `3` as payloads. It tests *algebraic* equivalence,
not *arithmetic* evaluation. If your refactor changes `1 + 2` into `3` by
constant-folding, FPSan will flag it — and that is correct, because at the
level of abstract syntax those two expressions are different. Optimizers
that wish to constant-fold must do so *before* the FPSan-instrumented run,
or the two runs being compared.

### What about division and subtraction?

Subtraction is trivial: `a - b = a + (-b)`, and we have negation working.

Division is the awkward one. We want `(a / b) * b == a`. With integer
multiplication mod $2^{32}$, that means division-by-`b` has to be
multiplication by some integer `b⁻¹` satisfying `b * b⁻¹ == 1` mod $2^{32}$.
Such a `b⁻¹` exists **only when `b` is odd** — even payloads have no
multiplicative inverse mod a power of two.

FPSan handles this by defining a "reciprocal" function that:

- Returns the true modular inverse when the payload is odd, so that
  `(a / b) * b == a` whenever the inverse exists.
- For even payloads, returns a fixed function $\iota$ that satisfies
  $\iota(\iota(x)) = x$ — so that `1 / (1 / x) == x` always holds, even
  though `(x / x) == 1` may not hold for even $x$.

This is the one place where FPSan's algebraic guarantees become conditional.
The condition (odd payload) is satisfied for "most" floats — odd payloads
are roughly half the representation space, scattered by the scrambling — so
in practice division works.

## Iteration 3: Mixed precision and conversions

Casts between floating-point formats are common, so FPSan supports them too.
Widening a value and then narrowing it back should recover the original:

```cpp
Value<__bf16> h{1.5f};
Value<float>  f    = cast<float>(h);   // widen
Value<__bf16> back = cast<__bf16>(f);  // narrow back
assert(back == h);
```

Iteration 2 had only one encoding $\varphi_{32}$ for 32-bit floats. We need
an encoding $\varphi_w$ for each format width $w$, and a rule for converting
between them.

The fix is uniform: parameterize the encoding by the format. The recipe
from iteration 2 already had two width-dependent parameters — the magnitude
mask (low $w-1$ bits) and the xor-shift amount (the mantissa width). Set
$A$ as a fixed constant, derive $B_+$ from the
$\varphi_w(1.0) = 1$ constraint per width, and we get one $\varphi_w$ per
format.

Casts between formats operate directly on the payload, never reconstructing
a float. Read the source payload as a signed integer and resize it to the
destination width — sign-extend to widen, truncate to narrow:

```
cast<U>(Value<T> v):
    p = asSigned(payload_T(v))      // source payload, read as a signed integer
    p = signedResize(p, width(U))   // sign-extend to widen, truncate to narrow
    return payload_U(p)             // store p as the destination payload
```

This is the rule Triton's sanitizer uses for `arith.extf` / `arith.truncf`.
Note what it does *not* do: it never inverts $\varphi$ to recover a float,
never performs an IEEE-754 conversion, and never rounds. The cast stays entirely
in payload space — which is the point, since reconstructing the float
mid-kernel is exactly what FPSan exists to avoid.

The widen-then-narrow round-trip above is bit-exact for a purely integer
reason: $\mathrm{truncate}(\mathrm{signExtend}(p)) = p$ for every payload $p$,
at every width — no appeal to IEEE-754 rounding. (The other direction,
narrow-then-widen, loses the high payload bits and does *not* round-trip,
exactly as a lossy numeric cast wouldn't.) The fixed points carry across
widths — $0 \mapsto 0$, $1 \mapsto 1$, $-1 \mapsto$ all-ones — because signed
resize preserves them.

This is a mechanical extension of iteration 2: one $\varphi_w$ per width and an
integer resize between them, no new algebra.

## Iteration 4: Transcendentals

So far we have $+, -, \times, \div$ and casts. There are also `exp`, `log`,
`sin`, `cos`, `sqrt`, `tanh`, and the rest. Unlike the four basic operations,
these have no natural counterpart in integer arithmetic — we have to *define*
how FPSan implements each one.

The defining property we want, for each transcendental, is **whatever
algebraic identities the real function obeys, the FPSan version should
obey them too**.

For some transcendentals this is rich:

- **`exp(x + y) == exp(x) * exp(y)`** and **`exp(0) == 1`** — the
  exponential turns sums into products.
- Likewise `exp2`.
- **`sin(x + y) == sin(x)*cos(y) + cos(x)*sin(y)`** etc. — the
  angle-addition formulas.

There's a simple construction. Pick any odd 64-bit integer $g$, and define

$$\mathrm{exp}(p) \;=\; g^p \bmod 2^{64}.$$

Then by the rules of exponents,
$\mathrm{exp}(p+q) = g^{p+q} = g^p \cdot g^q = \mathrm{exp}(p) \cdot
\mathrm{exp}(q)$ — the identity we wanted — and
$\mathrm{exp}(0) = g^0 = 1$.

We need $g$ to be odd so that $\mathrm{exp}(p)$ is always odd — only odd
integers mod $2^{64}$ have a multiplicative inverse, and we need that
inverse for the identity $\mathrm{exp}(-p) = 1 / \mathrm{exp}(p)$ to hold.
We also need $g$ to avoid obvious degenerate choices like $g = 1$, which
would make `exp` collapse everything to $1$. Pick a large odd integer with
no special structure and the resulting function looks "random" on arguments
other than $0$.

So `exp` is no mystery: it's `g^p` for a fixed `g`. `sin` and `cos` admit a
similar construction that satisfies the angle-addition formulas, by
piggy-backing on `exp` via Euler's identity $e^{ix} = \cos x + i \sin x$
and treating payloads as complex numbers built from pairs of integers.

For others — `log`, `sqrt`, `erf`, `floor`, `ceil`, `tanh`, etc. — the
identities we'd *want* are out of reach, and it's worth being honest that this
is a forced limitation rather than something FPSan has no use for. It would be
genuinely useful to honor `log(x*y) = log x + log y`, or
`sqrt(x*y) = sqrt x · sqrt y` — a refactor that applied either would then
survive the rewrite. But neither is achievable in $\mathbb{Z}/2^w$:

- A `log` obeying its law would be a homomorphism from *multiplication* to
  *addition*, and the zero element forbids any nontrivial one:
  `log(0·x) = log 0 = log 0 + log x` forces `log x = 0` for all `x`.
- A `sqrt` that is both multiplicative *and* an actual square root
  (`sqrt(x)² = x`) can't exist either: squaring mod $2^w$ is neither injective
  nor onto, so most payloads have no square root at all.

The asymmetry with `exp` is the point: *addition* has no such obstruction, so
`g^p` works, but the reverse direction (product → sum) is blocked by zero and by
multiplication not being invertible. So for these FPSan doesn't *choose* to skip
the identities — it *can't* honor them, and settles for the weakest thing still
worth having: a **tagged scramble**. Each such op gets a hash-like permutation
of the payload, tagged by the operation's identity, so that:

- `log(x) == log(x)` always — same input gives same output.
- `log(x) != sqrt(x)` for any $x$ (with overwhelming probability) — different
  operations don't collide.

That is all the structure these operations carry under FPSan. They don't
compute anything you can use numerically, but they do let you detect "did
this transcendental get called or not, and was it called on the same
argument both times" — which is exactly the kind of question a refactor-vs-
original comparison needs to answer.

Next: [authoring code with `Value`](tutorial-authoring.md), [porting an
existing codebase](tutorial-porting.md), or [choosing an algebraic
semantics](algebraic-fpsan.md).
