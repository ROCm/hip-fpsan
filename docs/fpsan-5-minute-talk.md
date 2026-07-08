# FPSan in 5 minutes

## The problem that FPSan addresses

FPSan is an idea from the Triton compiler project. The implementation in this
repository includes both Triton-style FPSan and algebraic variants that build on
that idea.

How do we verify that an optimization is correct when it changes the rounding
errors in a floating-point workload?

Put differently: how do we show that even though the rounding errors have
changed, the underlying math is still the same?

Because of rounding, floating-point arithmetic does not *exactly* satisfy
certain algebraic properties, such as associativity of `+` and `*`, and
distributivity of `*` over `+`:

* `(a + b) + c != a + (b + c)`
* `(a * b) * c != a * (b * c)`
* `(a + b) * c != (a * c) + (b * c)`

The idea of FPSan is to replace floating-point arithmetic by arithmetic on
*fingerprints* that:

* is NOT going to be of any use to compute the actual results;
* but IS going to *exactly* satisfy these algebraic properties.

If the optimization preserved the underlying math, then the fingerprints should
agree exactly.

Thus tests can perform *exact* comparisons of fingerprints instead of *fuzzy*
comparisons of floats.

## Step 1: the tiny version of the idea

The simplest realization of this idea is to just bit-cast floats into unsigned
integers of the same bit-width.

The unsigned integers are then the fingerprints. The arithmetic of fingerprints
is just the usual wrapping arithmetic of unsigned integers.

```cpp
class Value {
    uint32_t payload;

public:
    explicit Value(float x) : payload(std::bit_cast<uint32_t>(x)) {}

    friend Value operator+(Value a, Value b) {
        return from_payload(a.payload + b.payload); // wraps mod 2^32
    }

    friend Value operator*(Value a, Value b) {
        return from_payload(a.payload * b.payload); // wraps mod 2^32
    }
};
```

This already satisfies the basic requirement of exact associativity and
distributivity because that's what wrapping unsigned integer arithmetic does.

## What Triton FPSan does

The actual Triton FPSan design refines the tiny version above in multiple ways.

The first refinement is recovering *some* value-level algebraic facts. The above
bit-cast does not preserve even basic facts about constants and negation, which
leads to surprising results.

For example, the bit pattern of `1.0f` is not integer `1`, so:

```cpp
Value{1.0f} * x == x   // false in the toy version
```

Also, float negation is a sign-bit flip, not integer negation, so:

```cpp
x + (-x) == 0          // false in the toy version
```

So the first thing that Triton FPSan does is to replace the naive bit-cast with a
carefully tuned scrambling function that recovers some basic algebraic features:

```text
0.0  -> 0
1.0  -> 1
-x   -> -fingerprint(x)
```

Some other things that Triton FPSan does:

1. Further tune the fingerprinting to give it more hash-function-like properties (reduce collisions).
2. Implement `exp`, `cos`, and `sin` in the fingerprint arithmetic in a way that
   respects some basic properties, such as `exp(x + y) == exp(x) * exp(y)`.

## One hip-fpsan usage example

The `hip-fpsan` API idea is just: run the reference and optimized code with
`fpsan::Value` instead of `float`, then compare exactly.

```cpp
#include <fpsan/fpsan.hpp>

template <class T>
T reference(const float* x, int n);

template <class T>
T optimized(const float* x, int n);

using F = fpsan::Value<float,
                       fpsan::Semantics::Triton,
                       fpsan::Conversions::Explicit>;

F a = reference<F>(ptr, n);
F b = optimized<F>(ptr, n);

assert(a == b);
```

## Algebraic FPSan: the variants we add

Triton FPSan scrambles float bit patterns. That is great for collision behavior,
but it does not know ordinary value facts:

```cpp
using T = fpsan::Value<float, fpsan::Semantics::Triton,
                       fpsan::Conversions::Explicit>;

T{2.0f} + T{2.0f} == T{4.0f}; // false
```

Algebraic FPSan is the idea that it is actually possible to design fingerprinting
functions that transport more of the arithmetic, so that the above "ordinary
value facts" like `2 + 2 == 4` are now preserved.

Instead of treating a finite float mostly as a bit pattern, it treats it as its
exact value. Every finite binary float is:

```text
integer * power_of_two
```

So algebraic FPSan reduces that exact value modulo a carefully chosen finite
modulus:

```text
2^e * m  ->  2^e * m mod n
```

When the exponent `e` is negative, this relies on `2` being invertible modulo `n`.
This requires `n` to be odd.

When `n` is prime, not only `2` but all integers that are not 0 modulo `n`
become invertible modulo `n`. This allows Field-style algebraic FPSan to model
division.

The practical effect:

```cpp
using A = fpsan::Value<float, fpsan::Semantics::Field,
                       fpsan::Conversions::Explicit>;

A{2.0f} + A{2.0f} == A{4.0f}; // true

A x{3.0f};
x + x == A{2.0f} * x; // true, structurally
```

So algebraic FPSan preserves many exact rational identities that Triton FPSan
intentionally does not preserve.

## Example: catching a changed reduction

Suppose an optimized reduction accidentally drops one term on a boundary tile.

Floating-point output might still be close enough to pass a loose tolerance,
especially in a noisy ML test.

FPSan output should usually change, because the expression fingerprint changed:

```cpp
// reference:
sum = (((x0 + x1) + x2) + x3);

// buggy optimized:
sum = ((x0 + x1) + x3); // x2 vanished
```

The point is not the exact numeric error. The point is that the dataflow changed.

## Example: accepting a harmless reassociation

Now suppose the optimized version only changes the tree shape:

```cpp
// reference:
sum = (((x0 + x1) + x2) + x3);

// optimized:
sum = ((x0 + x1) + (x2 + x3));
```

IEEE-754 bits may differ because rounding points moved.

FPSan fingerprints can match because the finite-ring arithmetic is associative.

That is exactly the kind of distinction we want when debugging compiler or
kernel transformations.

## One sentence to remember

FPSan runs your numerical code in a same-width fingerprint arithmetic where many
algebraic rewrites become exact equalities, so exact fingerprint mismatches are a
strong signal that the optimized program changed the computation’s structure.

## Pointers for after the 5 minutes

- [Triton FPSan from first principles](triton-fpsan.md)
- [Algebraic FPSan](algebraic-fpsan.md)
- [Algebraic FPSan semantics](algebraic-semantics.md)
- [Authoring code with `Value`](tutorial-authoring.md)
- [Porting an existing codebase](tutorial-porting.md)
