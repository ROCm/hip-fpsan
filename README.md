# hip-fpsan

`hip-fpsan` supports C++17, HIP, and Python for running numerical code under
FPSan-family semantics. The core C++17/HIP library is header-only; the Python
bindings are optional and target interactive experiments.

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
  - [Python bindings](docs/python.md): interactive shell and notebook use.
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
`Semantics::Native` forwards to `std::` where applicable, including `abs`/`fabs`.
FPSan-family semantics implement deterministic fingerprint operations: some
preserve useful identities, while the rest produce operation-distinguishing tags.
Triton behavior is introduced in [`docs/triton-fpsan.md`](docs/triton-fpsan.md);
algebraic behavior, including Field-family quadratic-residue-positive
`abs`/selection, is summarized in
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

Configure out of tree with ordinary CMake command lines:

```bash
# Pure C++. -DCMAKE_HIP_COMPILER= forces the host path even if this box has a
# HIP toolchain; on a non-ROCm machine it is unnecessary (no compiler is found).
cmake -S . -B build/cxx -G Ninja -DCMAKE_HIP_COMPILER=
cmake --build build/cxx
ctest --test-dir build/cxx --output-on-failure

# HIP C++ for RDNA4 / gfx1201. HIP turns on automatically once the ROCm
# toolchain is found (see ROCM_PATH below); there is no separate enable flag.
cmake -S . -B build/hip-gfx1201 -G Ninja \
  -DCMAKE_HIP_ARCHITECTURES=gfx1201
cmake --build build/hip-gfx1201
ctest --test-dir build/hip-gfx1201 --output-on-failure

# HIP C++ for CDNA4 / gfx950. Running tests needs matching hardware.
cmake -S . -B build/hip-gfx950 -G Ninja \
  -DCMAKE_HIP_ARCHITECTURES=gfx950
cmake --build build/hip-gfx950

# Optional microbenchmarks.
cmake -S . -B build/cxx-bench -G Ninja \
  -DCMAKE_HIP_COMPILER= \
  -DFPSAN_BUILD_BENCHMARKS=ON
cmake --build build/cxx-bench --target fpsan_cpu_bench

# Optional Python bindings.
cmake -S . -B build/python -G Ninja \
  -DCMAKE_HIP_COMPILER= \
  -DFPSAN_BUILD_PYTHON=ON
cmake --build build/python
ctest --test-dir build/python -R fpsan_python_test --output-on-failure
```

The HIP configure finds the ROCm toolchain under `ROCM_PATH`, resolved as
`-DROCM_PATH=...` > `$ENV{ROCM_PATH}` > `rocm-sdk path --root` > `/opt/rocm`
(the same priority the ROCm libraries use). Point it at a different install
with `-DROCM_PATH=...` (or `$ENV{ROCM_PATH}`),
or override the pieces directly with `-DCMAKE_HIP_COMPILER=...`,
`-DCMAKE_HIP_COMPILER_ROCM_ROOT=...`, `-DCMAKE_HIP_ARCHITECTURES=...`.

CMake options: `FPSAN_BUILD_TESTS` (ON), `FPSAN_BUILD_EXAMPLES` (ON),
`FPSAN_BUILD_BENCHMARKS` (OFF), `FPSAN_BUILD_PYTHON` (OFF).

There is no separate HIP enable toggle. Tests and examples build as HIP C++
whenever a HIP compiler is available, and as pure C++ otherwise -- the single
source of truth is the standard `CMAKE_HIP_COMPILER`, derived from `ROCM_PATH`
(resolved as `-DROCM_PATH=...` > `$ENV{ROCM_PATH}` > `rocm-sdk path --root` >
`/opt/rocm`), set explicitly with `-DCMAKE_HIP_COMPILER=...`, or found on `PATH`
by `check_language(HIP)`. To force a pure-C++ build on a machine that *has* a
HIP toolchain, set it empty: `-DCMAKE_HIP_COMPILER=`.
