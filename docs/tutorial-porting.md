# Tutorial: porting an existing codebase incrementally

The goal: take code written in terms of `float` (or `double`) and bring it under
FPSan checking **gradually**, without a big-bang rewrite and without ever having
both semantics mixed in one arithmetic expression. The companion program is
[`examples/porting_example.cpp`](../examples/porting_example.cpp).

The enabling idea is that `Value` is a *type*, and `semantics` / `conversions`
are *template parameters* — so different variants are distinct types that can
coexist in one program (no ODR hazard), and you flip them on a chosen subset of
the code at a time. This is why these are template parameters rather than a
global `#ifdef`.

## The recommended four stages

Introduce a single alias and move it through these stages (with
`using namespace fpsan;` for brevity):

```cpp
// Step 0 (original):
using Scalar = float;

// Step 1 — drop-in. Identical arithmetic; just proves the wrapper compiles
// and behaves exactly like float everywhere it is used.
using Scalar = Value<float, Semantics::Float, Conversions::Implicit>;

// Step 2 — turn ON explicit conversions. No numeric change, but now every
// accidental implicit float<->Scalar conversion becomes a COMPILE ERROR, so you
// can find and make deliberate every place values cross the boundary.
using Scalar = Value<float, Semantics::Float, Conversions::Explicit>;

// Step 3 — turn ON FPSan semantics.
using Scalar = Value<float, Semantics::FPSan, Conversions::Explicit>;
```

(`Semantics::FPSan` with `Conversions::Implicit` is also valid; turning explicit
conversions on first is just a safety step so that, once FPSan math is active,
no value silently slips in or out through an implicit cast — which would quietly
escape the checking.)

## Stage 1 — drop-in

Replace the typedef and rebuild. `Value<float, fpsan::Semantics::Float, fpsan::Conversions::Implicit>` is a bit-exact
stand-in for `float`: it has the same operators, implicit conversions, and
`std::numeric_limits`, and a generic kernel produces **identical bits**. If
something fails to compile here, it usually means the code relied on `float`
being a builtin in a way the wrapper does not yet cover — fix locally and
continue. The example asserts `stage1 == stage0`.

## Stage 2 — make conversions explicit

Set `conversions` to `Conversions::Explicit`. Now:

- `Scalar x = 1.0f;` (copy-init) fails — write `Scalar x(1.0f);`.
- `float y = x;` fails — write `float y = static_cast<float>(x);`.
- `x + 2.0f` fails — the scalar must be wrapped, or use `x + Scalar(2.0f)`.

Each error marks a real boundary between "plain float" and "tracked" values.
Make each one explicit. Numerically nothing changes (still `Semantics::Float`),
so you can do this file-by-file and keep a working build. The example kernel is
written conversion-clean, so it already compiles at this stage.

## Stage 3 — turn on FPSan

Set `semantics` to `Semantics::FPSan`. Arithmetic is now the integer payload algebra.
Numbers become meaningless in isolation, so your *tests* change shape: instead
of comparing against expected numeric values, compare two
supposed-to-be-equivalent computations (e.g. an optimized kernel vs a reference,
or two operation orderings) and assert their **payloads are equal**. Compare
FPSan results only against other FPSan results.

## Mixing granularity and safety rails

- You can port one module to `Scalar = Value<...>` while the rest stays
  `float`, as long as a single arithmetic expression never mixes the two. At the
  boundary, convert explicitly.
- Mixing two different `Value` instantiations in one binary op (different
  `semantics`, `float_type`, or `conversions`) is a **compile-time
  error** with a clear message — so an incremental port cannot accidentally
  combine the wrong semantics.

## HIP / GPU

The same code compiles as HIP C++ for device kernels (everything is
`__host__ __device__` and constexpr). See the top-level README for the `hip`
build preset; `tests/hip_device_test.cpp` shows Value running in a kernel.
