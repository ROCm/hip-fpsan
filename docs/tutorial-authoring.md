# Tutorial: authoring code with `Value`

This walks through writing new numeric code directly against the library. The
companion program is [`examples/authoring_example.cpp`](../examples/authoring_example.cpp).

> Reminder: FPSan is a *checking* tool, not a faster float. In FPSan mode the
> numeric values are scrambled on purpose; what is preserved is the **algebra**
> (associativity, distributivity, the exp/trig identities). Two computations
> that are equal as real-number expressions produce the **same payload**, even
> when IEEE rounding would make them differ.

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
| `semantics` | `Semantics::Native` = ordinary IEEE arithmetic; `Semantics::Triton` = FPSan payload algebra |
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

The idiom is to compute the same quantity two ways and compare payloads. In the
example we sum a vector forward and backward:

```cpp
template <class S> S sum_forward(const std::vector<float>& v) { /* ... */ }
template <class S> S sum_reverse(const std::vector<float>& v) { /* ... */ }
```

With `float`, `sum_forward != sum_reverse` for a catastrophic input
(`{1e8, 1, 2, 3, -1e8, ...}`). With `Value<float, fpsan::Semantics::Triton, fpsan::Conversions::Implicit>`, the two
payloads are **equal** — addition is exactly associative in the payload ring.

## 4. Inspecting values

Include `<fpsan/io.hpp>` for `operator<<` (host only). In FPSan mode it prints
both the integer payload and the (scrambled) unembedded float:

```cpp
#include <fpsan/io.hpp>
std::cout << f(a) << "\n";   // fpsan(payload=..., unembed=...)
```

`x.fpsan_payload()` returns the raw payload (only defined in FPSan mode);
`x.to_float()` returns the represented float. A round trip with no arithmetic —
`static_cast<float>(Scalar(x))` — recovers `x` exactly, because the embedding is
a bijection.

## 5. Build

The library is header-only; just add `include` to your include path, or
link the CMake target:

```cmake
target_link_libraries(your_target PRIVATE fpsan::fpsan)
```

Next: [porting an existing codebase](tutorial-porting.md).
