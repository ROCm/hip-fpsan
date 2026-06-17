# hip-fpsan

`hip-fpsan` is a header-only C++17/HIP library for running numerical code under
FPSan-family semantics.

FPSan was introduced by the [Triton compiler](https://github.com/triton-lang/triton)
as a way to test floating-point rewrites without requiring bit-identical rounded
results. An FPSan run replaces each floating-point value by a same-width
fingerprint, evaluates the program in deterministic finite arithmetic, and
compares the resulting fingerprints. This is useful when a compiler, library, or
GPU intrinsic changes expression shape through reassociation, factoring, FMA
formation, or a different accumulation order.

This library implements both the original Triton-style FPSan model and algebraic
FPSan variants:

- `Semantics::Native`: ordinary native arithmetic, useful for incremental
  porting and for flushing out implicit conversions.
- `Semantics::Triton`: Triton FPSan integer-fingerprint arithmetic.
- `Semantics::Field` and the other algebraic semantics: value-residue
  fingerprints that preserve more exact finite-value identities in their
  supported algebraic fragments.

The main API is a scalar/vector `Value<float_type, semantics, conversions>` type,
plus FPSan-aware overloads for many AMD GPU device intrinsics so whole HIP
kernels can be ported.

Good starting points:

- Hands-on guides:
  - [Authoring code with `Value`](docs/tutorial-authoring.md)
  - [Incremental porting](docs/tutorial-porting.md)
- Understanding FPSan:
  - [Triton FPSan from first principles](docs/triton-fpsan.md)
  - Algebraic FPSan:
    - [Algebraic FPSan](docs/algebraic-fpsan.md): general introduction and
      comparison of FPSan variants.
    - [Algebraic FPSan semantics](docs/algebraic-semantics.md): operational
      details for math-comfortable engineers.
    - [Reducing floats mod p](docs/reducing-floats-mod-p.pdf): mathematical
      write-up of algebraic FPSan.

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

## API at a glance

```cpp
#include <fpsan/fpsan.hpp>

using S = fpsan::Value<float, fpsan::Semantics::Triton,
                       fpsan::Conversions::Explicit>;
```

`float_type` may be `float`, `double`, `_Float16`, `__bf16`,
`fpsan::fp8_e4m3`, `fpsan::fp8_e5m2`, or a Clang/GCC ext-vector of those
scalars. Sub-byte MX formats are handled by packed AMDGPU intrinsic wrappers.

`Conversions::Implicit` makes `Value` behave more like a drop-in scalar.
`Conversions::Explicit` makes conversions opt-in, which is usually better once
FPSan checking is enabled.

For an incremental port, move in small steps:

```text
float
  -> Value<float, Semantics::Native, Conversions::Implicit>
  -> Value<float, Semantics::Native, Conversions::Explicit>
  -> Value<float, Semantics::Triton, Conversions::Explicit>
```

The final step can use `Semantics::Triton` or one of the
[algebraic semantics](docs/algebraic-fpsan.md), depending on what identities you
want the fingerprint arithmetic to preserve.

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
