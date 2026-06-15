# hip-fpsan

A header-only C++17 library providing **`Value<float_type, semantics,
conversions>`** — a floating-point scalar type that can switch, by a template
parameter, between ordinary IEEE arithmetic and Triton-style **FPSan**
integer-payload arithmetic. It is designed for porting existing numerical code
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

enum class fpsan::Semantics { Float, FPSan };
enum class fpsan::Conversions { Implicit, Explicit };

template <class float_type, fpsan::Semantics semantics,
          fpsan::Conversions conversions>
class fpsan::Value;
```

| parameter | values |
|-----------|--------|
| `semantics` | `Semantics::Native` = native arithmetic of `float_type` (drop-in); `Semantics::Triton` = FPSan integer-payload arithmetic |
| `conversions` | `Conversions::Implicit` = implicit casts to/from numbers (like a POD); `Conversions::Explicit` = every conversion / 1-arg ctor is `explicit` |

As a scalar, `float_type` is any IEEE-style binary float the library has a layout
for: `float`, `double`, `_Float16`, `__bf16` (the last two where the toolchain
supplies them), and the OCP FP8 types `fpsan::fp8_e4m3` / `fpsan::fp8_e5m2` —
i.e. the 8/16/32/64-bit formats (`sizeof(float_type)` ∈ {1, 2, 4, 8}). It may
also be a Clang/GCC ext-vector of any of those scalars, e.g.
`Value<float __attribute__((ext_vector_type(8))), …>`, a SIMD bundle whose
payload algebra runs lane-wise. The sub-8-bit MX formats (fp6 e2m3, bf6 e3m2,
fp4 e2m1) are supported too, carried packed several-to-a-dword in vector
containers.

What the sub-byte formats lack is a *scalar* type: being non-byte-multiples they
have no `Value<fp6>` / `Value<fp4>`, so instead of plain arithmetic they are
produced, consumed and FPSan-checked in that packed form by the gfx950 device
intrinsic wrappers (`amdgcn_cvt.hpp`, `amdgcn_mfma.hpp`, `amdgcn_ds.hpp`). Each
`Value` combination is a distinct *type*, so variants coexist with no ODR hazard,
and mixing different instantiations in one operation is a clean compile-time
error.

The two parameters exist to enable an **incremental conversion path**: start from
plain `float` and migrate in small, compiler-checked steps —

```
float
  → Value<float, Semantics::Native, Conversions::Implicit>   // bit-exact drop-in for float
  → Value<float, Semantics::Native, Conversions::Explicit>   // same results; no silent casts
  → Value<float, Semantics::Triton, Conversions::Explicit>   // algebraic-equivalence checking
```

`Semantics::Native` keeps native IEEE results; `Conversions::Explicit` flushes out
implicit conversions; `Semantics::Triton` then replaces arithmetic with
integer-payload algebra that obeys *exact* laws (associativity, distributivity,
`exp(x+y) = exp(x)·exp(y)`), so algebraically-equal expressions produce the
**same payload** — e.g. `(a + b) + c == a + (b + c)` holds on the nose. The
payloads themselves are scrambled and only meaningful when compared against other
FPSan payloads; see the blog post linked above for the why.

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
  functions) — see [Math functions](#math-functions) for the full list and
  status.
- `<fpsan/numeric_limits.hpp>` (auto-included): `std::numeric_limits` support.
- `<fpsan/io.hpp>` (opt-in, host only): `operator<<`.

See the tutorials: [authoring](docs/tutorial-authoring.md) ·
[incremental porting](docs/tutorial-porting.md).

## Math functions

`<fpsan/math.hpp>` provides standard-library-style math as ADL free functions on
`Value` (write `using std::exp; exp(x);` in generic code, or `fpsan::exp(x)`). In
**Float** mode each calls the matching `std::` function (a drop-in); in **FPSan**
mode each maps to Triton's payload handler. These handlers are **pure
integer-payload arithmetic** — no AMD intrinsics, identical on host and on every
gfx target — and fall into three classes:

- **Algebraic** — real identities hold in the payload ring, e.g.
  `exp2(x+y) == exp2(x)·exp2(y)` and the angle-addition formulas.
- **Tagged scramble** — *not* real math: a deterministic, op-distinguishing
  permutation of the payload. The only guarantees are that equal inputs give
  equal outputs and different ops give different outputs (so you can detect
  "a transcendental was called here" and check dataflow equivalence).
- **Modular** — plain integer ring ops on the payload.

The FPSan handlers are transcribed from Triton's `FpSanitizer.cpp` and verified
**bit-for-bit** against Triton's own reference (every ✅ below was checked across
100k+ cases for `float`; the same constants are reused, truncated to width, for
the other formats).

| Function | Class | FPSan ↔ Triton |
|---|---|:---:|
| `exp`, `exp2` | algebraic | ✅ identical |
| `sin`, `cos` | algebraic | ✅ identical |
| `log`, `log2` | tagged scramble | ✅ identical |
| `sqrt`, `rsqrt` | tagged scramble | ✅ identical |
| `precise_sqrt` | tagged scramble | ✅ identical |
| `erf` | tagged scramble | ✅ identical |
| `floor`, `ceil` | tagged scramble | ✅ identical |
| `fma` | modular (`a·b + c`) | ✅ identical |
| `fmod` | modular (signed `srem`) | ✅ identical |
| `fmin`, `fmax`, `min`, `max` | modular (signed order) | ✅ identical |
| `fmed3` (median-of-3) | modular (composed min/max) | ➕ extension *(no Triton op)* |
| `rcp`, `fract` | tagged scramble | ➕ extension *(no Triton op)* |
| `tanh` | tagged scramble | ⚠️ diverges *(Triton lowers `tanh` as an extern libdevice/ocml call with a different symbol-hash tag)* |

✅ = bit-identical to Triton's FPSan handler · ➕ = library extension Triton has no
equivalent for · ⚠️ = both implement it but by different schemes, so payloads
differ.

Notes on parity with Triton's surface:

- The algebraic and tagged-scramble semantics, the op tag (`murmur64` of an op
  id), and the division/remainder handling all match Triton exactly.
- `rcp`, `fract`, and `fmed3` are conveniences this library adds; Triton has no
  first-class equivalent.
- `tanh` is the one intentional divergence: Triton treats `tanh` (along with
  `tan`, `log1p`, `cbrt`, `round`, …) as an *extern* elementwise op tagged by a
  hash of the libdevice/ocml symbol name, whereas here it is a first-class
  tagged-scramble unary. Both are valid FPSan tags, but they are not the same
  bits.

## Caveats (matching FPSan's own docs)

- This is **not** an IEEE-754 reference implementation: no real ordering,
  rounding, NaN/Inf, subnormals, or exceptions.
- `x / x == 1` holds only for **odd** payloads; for even payloads `inv` is a
  parity-preserving involution, not a true inverse.
- FPSan results are only meaningful when compared against *other* FPSan results.

## AMD GPU intrinsic support

Beyond the scalar/vector `Value` type, the library ships FPSan-aware overloads of
AMD GPU device intrinsics (opt-in headers, not pulled by `<fpsan/fpsan.hpp>`) so a
whole compute kernel — not just its scalar math — can be ported. Each wrapper has
a **Float** path (forwards to the real `__builtin_amdgcn_*`) and an **FPSan** path
(a payload-ring software dataflow using the hardware fragment layout).

**Supported architectures:**

- **RDNA3** — gfx1100 (WMMA matrix path, wave32 and wave64)
- **RDNA4** — gfx1200 / gfx1201 (WMMA matrix path, K=16, wave32)
- **CDNA3** — gfx942 (gfx9-family MFMA path, wave64)
- **CDNA4** — gfx950 (MFMA + scaled / sub-byte matrix path, wave64)
- **gfx1250** — distinct WMMA family, K=32/64/128 shapes (wave32)

The CDNA / gfx9-family targets carry the MFMA/SMFMAC matrix instructions; the
RDNA targets carry WMMA/SWMMAC where native to the architecture — each family
implements the matrix ops native to it. Every in-scope FP-relevant intrinsic the
architecture exposes is wrapped in both Float and FPSan modes (or deferred with
an explicit rationale).

Integer MFMA/SMFMAC, graphics ops (cubemap, interp, image_bvh, fp→int format
packs), and the IEEE bit-twiddling micro-ops (`frexp`/`div_scale`/`trig_preop`/…)
are intentionally out of scope: a symbolic FPSan payload has no faithful image
for them.

## Building the tests and examples

CMake presets are provided:

```bash
# pure C++ (system clang++)
cmake --preset cxx && cmake --build --preset cxx && ctest --preset cxx

# HIP C++ (ROCm clang; compiles the same tests as device code, gfx1201 / RDNA4)
cmake --preset hip && cmake --build --preset hip && ctest --preset hip

# HIP C++ for CDNA4 (gfx950) or gfx1250 -- compile-checks the arch wrappers;
# running the tests needs the matching hardware.
cmake --preset hip-gfx950   && cmake --build --preset hip-gfx950
cmake --preset hip-gfx1250  && cmake --build --preset hip-gfx1250
```

The HIP presets find the ROCm toolchain under `ROCM_PATH`, resolved as
`-DROCM_PATH=...` > `$ENV{ROCM_PATH}` > `rocm-sdk path --root` > `/opt/rocm`
(the same priority the ROCm libraries use). Point them at a different install
with `-DROCM_PATH=...` (or `$ENV{ROCM_PATH}`),
or override the pieces directly with `-DCMAKE_HIP_COMPILER=...`,
`-DCMAKE_HIP_COMPILER_ROCM_ROOT=...`, `-DCMAKE_HIP_ARCHITECTURES=...`.

### GoogleTest

The unit tests use GoogleTest. CMake first tries `find_package(GTest)`; if no
installed GoogleTest is found it fetches a pinned release (v1.15.2) at configure
time via `FetchContent` (needs network access). For an offline build, point CMake
at a local checkout with
`-DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/path/to/googletest`.

CMake options: `FPSAN_BUILD_TESTS` (ON), `FPSAN_BUILD_EXAMPLES` (ON),
`FPSAN_ENABLE_HIP`. The latter defaults to ON when a HIP toolchain is available
and OFF otherwise: it follows the standard `CMAKE_HIP_COMPILER` variable, which
is derived from `ROCM_PATH` (resolved as `-DROCM_PATH=...` > `$ENV{ROCM_PATH}` >
`rocm-sdk path --root` > `/opt/rocm`), set explicitly, or found on `PATH` by
`check_language(HIP)`. Set `-DFPSAN_ENABLE_HIP=OFF` to force a pure-C++ build
even where a HIP toolchain exists.

## Layout

```
include/fpsan/        fpsan.hpp (umbrella), value.hpp, math.hpp, cast.hpp,
                      numeric_limits.hpp, io.hpp,
                      amdgcn_*.hpp (HIP device intrinsic wrappers),
                      detail/{config,traits,mix,math,fp8}.hpp
tests/                GoogleTest suites + compile-fail tests + host/device
                      parity & HIP device tests
                      + fpsan_generic.hpp (independent ground-truth reference)
examples/             tutorial programs (also run under ctest)
docs/                 tutorials
```
