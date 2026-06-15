# hip-fpsan

A header-only C++17 library providing **`Value<float_type, semantics,
conversions>`** — a floating-point scalar type that can switch, by a template
parameter, between ordinary IEEE-754 arithmetic and Triton-style **FPSan**
integer-fingerprint arithmetic. It is designed for porting existing numerical code
(including HIP GPU kernels) into a form where you can check algebraic
equivalence of computations. "FPSan" here is the floating-point sanitizer from
the [Triton compiler](https://github.com/triton-lang/triton) — the ideas explained in
[this blog post](https://cp4space.hatsya.com/2026/05/03/schanuels-conjecture-and-the-semantics-of-fpsan/);
this library is just a C++/HIP library offering a `Value` type implementing these
semantics, and overloads of intrinsics for various AMD GPUs allowing to port
entire programs, including compute kernels using such intrinsics.

## API

```cpp
#include <fpsan/fpsan.hpp>

enum class fpsan::Semantics {
  Native, Triton,
  Field, // ...more algebraic variants; see below.
};
enum class fpsan::Conversions { Implicit, Explicit };

template <class float_type, fpsan::Semantics semantics,
          fpsan::Conversions conversions>
class fpsan::Value;
```

| parameter | values |
|-----------|--------|
| `semantics` | `Semantics::Native` = native arithmetic; `Semantics::Triton` = Triton FPSan integer fingerprints; `Semantics::Field` and friends = algebraic fingerprints |
| `conversions` | `Conversions::Implicit` = implicit casts to/from numbers (like a POD); `Conversions::Explicit` = every conversion / 1-arg ctor is `explicit` |

`float_type` may be `float`, `double`, `_Float16`, `__bf16`,
`fpsan::fp8_e4m3`, `fpsan::fp8_e5m2`, or a Clang/GCC ext-vector of those
scalars. Sub-byte MX formats are handled by the packed AMDGPU intrinsic
wrappers rather than by scalar `Value<fp6>` / `Value<fp4>` types.

The `semantics` and `conversions` parameters enable an **incremental conversion
path**: start from plain `float` and migrate in small, compiler-checked steps —

```
float
  → Value<float, Semantics::Native, Conversions::Implicit>   // bit-exact drop-in for float
  → Value<float, Semantics::Native, Conversions::Explicit>   // same results; no silent casts
  → Value<float, Semantics::Triton, Conversions::Explicit>   // algebraic-equivalence checking
```

`Semantics::Native` keeps native IEEE-754 results; `Conversions::Explicit`
flushes out implicit conversions; `Semantics::Triton` and the algebraic
semantics replace arithmetic with comparable fingerprints. For algebraic
semantics beyond Triton, start with
[`docs/algebraic-fpsan.md`](docs/algebraic-fpsan.md), then see
[`docs/algebraic-semantics.md`](docs/algebraic-semantics.md) and
[`docs/reducing-floats-mod-p.pdf`](docs/reducing-floats-mod-p.pdf).

## Quick start

```cpp
#include <fpsan/fpsan.hpp>
using S = fpsan::Value<float, fpsan::Semantics::Triton, fpsan::Conversions::Implicit>;
S a = 1e8f, b = -1e8f, c = 1.0f;
bool exact = ((a + b) + c == a + (b + c)); // true under FPSan (false for float)
```

Header-only: add `include` to your include path, or use the CMake target
`fpsan::fpsan`.

- Operators: `+ - * /`, unary `+`/`-`, compound assignment, all comparisons.
- `<fpsan/math.hpp>`: standard-library-style math on `Value` (ADL free
  functions) — see [Math functions](#math-functions) for semantics notes.
- `<fpsan/numeric_limits.hpp>` (auto-included): `std::numeric_limits` support.
- `<fpsan/io.hpp>` (opt-in, host only): `operator<<`.

See the tutorials: [authoring](docs/tutorial-authoring.md) ·
[incremental porting](docs/tutorial-porting.md).

## Math functions

`<fpsan/math.hpp>` provides standard-library-style math as ADL free functions on
`Value` (write `using std::exp; exp(x);` in generic code, or `fpsan::exp(x)`).
`Semantics::Native` forwards to `std::` where applicable. FPSan-family
semantics implement deterministic fingerprint operations: some preserve useful
identities, while the rest produce operation-distinguishing tags. Triton
behavior is introduced in [`docs/triton-fpsan.md`](docs/triton-fpsan.md);
algebraic behavior is summarized in
[`docs/algebraic-fpsan.md`](docs/algebraic-fpsan.md).

## AMD GPU intrinsic support

Beyond the scalar/vector `Value` type, the library ships FPSan-aware overloads of
AMD GPU device intrinsics (opt-in headers, not pulled by `<fpsan/fpsan.hpp>`) so a
whole compute kernel can be ported. Each wrapper has a `Semantics::Native` path
that forwards to the real builtin and FPSan-family paths that reproduce the
intrinsic dataflow on fingerprints.

**Supported architectures:**

- RDNA3: gfx1100
- RDNA4: gfx1200 / gfx1201
- CDNA3: gfx942
- CDNA4: gfx950
- gfx1250

## Building the tests and examples

CMake presets are provided:

```bash
# pure C++ (system clang++)
cmake --preset cxx && cmake --build --preset cxx && ctest --preset cxx

# HIP C++ (ROCm clang; compiles the same tests as device code, gfx1201 / RDNA4)
cmake --preset hip && cmake --build --preset hip && ctest --preset hip

# HIP C++ for CDNA3/CDNA4/gfx1250; running tests needs matching hardware.
cmake --preset hip-gfx942   && cmake --build --preset hip-gfx942
cmake --preset hip-gfx950   && cmake --build --preset hip-gfx950
cmake --preset hip-gfx1250  && cmake --build --preset hip-gfx1250

# optional microbenchmarks
cmake --preset cxx -DFPSAN_BUILD_BENCHMARKS=ON
cmake --build --preset cxx --target fpsan_cpu_bench
```

The HIP presets find the ROCm toolchain under `ROCM_PATH`, resolved as
`-DROCM_PATH=...` > `$ENV{ROCM_PATH}` > `rocm-sdk path --root` > `/opt/rocm`
(the same priority the ROCm libraries use). Point them at a different install
with `-DROCM_PATH=...` (or `$ENV{ROCM_PATH}`),
or override the pieces directly with `-DCMAKE_HIP_COMPILER=...`,
`-DCMAKE_HIP_COMPILER_ROCM_ROOT=...`, `-DCMAKE_HIP_ARCHITECTURES=...`.

CMake options: `FPSAN_BUILD_TESTS` (ON), `FPSAN_BUILD_EXAMPLES` (ON),
`FPSAN_BUILD_BENCHMARKS` (OFF), `FPSAN_ENABLE_HIP`. The latter defaults to ON
when a HIP toolchain is available and OFF otherwise: it follows the standard
`CMAKE_HIP_COMPILER` variable, which is derived from `ROCM_PATH` (resolved as
`-DROCM_PATH=...` > `$ENV{ROCM_PATH}` > `rocm-sdk path --root` > `/opt/rocm`),
set explicitly, or found on `PATH` by `check_language(HIP)`. Set
`-DFPSAN_ENABLE_HIP=OFF` to force a pure-C++ build even where a HIP toolchain
exists.
