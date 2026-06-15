# Tutorial: authoring code with `Value`

This walks through writing new numeric code directly against the library. The
companion program is [`examples/authoring_example.cpp`](../examples/authoring_example.cpp).

> Reminder: FPSan is a *checking* tool, not a faster float. In FPSan-family modes the
> numeric values are replaced by fingerprints on purpose; what is preserved is
> the algebra promised by the selected `Semantics`.

## 1. Pick a scalar type

```cpp
#include <fpsan/fpsan.hpp>
using namespace fpsan;
using Scalar = Value<float, Semantics::Triton, Conversions::Implicit>;
```

The three template parameters are:

| parameter | meaning |
|-----------|---------|
| `float_type` | the underlying real type: `float`, `double`, `_Float16`, `__bf16`, `fpsan::fp8_e4m3`/`fp8_e5m2`, or a Clang/GCC ext-vector of any of these |
| `semantics` | `Semantics::Native` = ordinary IEEE-754 arithmetic; [`Semantics::Triton`](triton-fpsan.md) = Triton FPSan fingerprint arithmetic; [algebraic variants](algebraic-fpsan.md) such as `Semantics::Field` = value-residue fingerprint arithmetic |
| `conversions` | `Conversions::Implicit` = implicit casts like a POD; `Conversions::Explicit` = every conversion/1-arg ctor is `explicit` |

## 2. Write ordinary-looking code

`Value` supports the usual operators (`+ - * /`, unary `-`, compound
assignment, comparisons) and the standard math functions in `<fpsan/math.hpp>`
(`exp`, `exp2`, `log`, `sin`, `cos`, `sqrt`, `fma`, `min`, `max`, ...), found by
ADL:

```cpp
Scalar f(Scalar x) {
  using fpsan::exp;            // so unqualified exp(x) finds the FPSan overload
  return exp(x * x) / (x + Scalar(1));
}
```

Construct from numbers, convert back with a cast (or `.to_float()`):

```cpp
Scalar a = 3.0f;              // implicit when conversions == Conversions::Implicit
float  y = static_cast<float>(f(a));
```

## 3. Check an equivalence

The idiom is to compute the same quantity two ways and compare fingerprints. In the
example we sum a vector forward and backward:

```cpp
template <class S> S sum_forward(const std::vector<float>& v) { /* ... */ }
template <class S> S sum_reverse(const std::vector<float>& v) { /* ... */ }
```

With `float`, `sum_forward != sum_reverse` for a catastrophic input
(`{1e8, 1, 2, 3, -1e8, ...}`). With `Value<float, fpsan::Semantics::Triton, fpsan::Conversions::Implicit>`, the two
fingerprints are **equal** — addition is exactly associative in the fingerprint ring.

## 4. Inspecting values

Include `<fpsan/io.hpp>` for `operator<<` (host only). In FPSan-family modes it
prints the stored fingerprint in the terms meaningful for that semantics:

```cpp
#include <fpsan/io.hpp>
std::cout << f(a) << "\n";   // Triton(payload=0x...) or Field(payload=... mod ...)
```

`x.fpsan_payload()` returns the raw stored integer (only defined in FPSan-family modes);
`x.to_float()` returns the represented float in `Native` mode and in
[`Triton`](triton-fpsan.md) mode. [Algebraic fingerprints](algebraic-fpsan.md)
are residues, not recoverable floats, so `to_float()` is intentionally
unavailable for algebraic semantics.

## 5. Build

The library is header-only; just add `include` to your include path, or
link the CMake target:

```cmake
target_link_libraries(your_target PRIVATE fpsan::fpsan)
```

Next: [porting an existing codebase](tutorial-porting.md). For semantics
selection, see [Algebraic FPSan semantics](algebraic-fpsan.md) and
[Triton FPSan from first principles](triton-fpsan.md).
