# Python bindings

`hip-fpsan` has optional Python bindings for interactive experiments with the
C++ scalar `fpsan::Value` type. They are meant for a Python shell or notebook:
try small expressions, inspect fingerprints, and compare Triton FPSan with the
algebraic variants.

The bindings are deliberately narrower than the C++ library. They cover scalar
`Value` types, arithmetic operators, comparisons, and the math functions from
`<fpsan/math.hpp>`. They do not bind AMD GPU intrinsics, HIP kernels, vector
`Value` types, packed MX formats such as fp6/fp4, or NumPy array operations.
Native FP8 classes are available for construction and inspection, but arithmetic
on FP8 is useful through FPSan-family semantics rather than `Semantics::Native`,
matching the C++ surface.

## Building

The bindings are off by default:

```bash
cmake -S . -B build/python -G Ninja \
  -DFPSAN_ENABLE_HIP=OFF \
  -DFPSAN_BUILD_PYTHON=ON
cmake --build build/python
ctest --test-dir build/python -R fpsan_python_test --output-on-failure
```

For a Python-only build that skips the C++ test suite, add
`-DFPSAN_BUILD_TESTS=OFF`.

CMake first tries `find_package(nanobind CONFIG)`. If that does not find a
system, vcpkg, or local nanobind package, it falls back to `FetchContent` and a
pinned nanobind release checkout. For offline builds, point CMake at a local
nanobind checkout:

```bash
cmake -S . -B build/python -G Ninja \
  -DFPSAN_ENABLE_HIP=OFF \
  -DFPSAN_BUILD_PYTHON=ON \
  -DFPSAN_BUILD_TESTS=OFF \
  -DFETCHCONTENT_SOURCE_DIR_NANOBIND=/path/to/nanobind
```

From the build tree, put the generated package on `PYTHONPATH`:

```bash
export PYTHONPATH=$PWD/build/python/python:$PYTHONPATH
python
```

Adjust the build directory to match the `-B` directory you used.

## First examples

```python
>>> import fpsan
>>> T = fpsan.Float32Triton
>>> A = fpsan.Float32Field
>>> T(2.0) + T(2.0) == T(4.0)
False
>>> A(2.0) + A(2.0) == A(4.0)
True
```

Algebraic values do not convert back to floats, because their payload is a
residue in a finite ring rather than an encoding of a recoverable floating-point
value:

```python
>>> x = A(3.0) * A(7.0)
>>> x.payload
21
>>> A.from_payload(x.payload) == x
True
```

Triton and native values can be converted back:

```python
>>> float(fpsan.Float32Native(1.25) + fpsan.Float32Native(2.5))
3.75
>>> float(fpsan.Float32Triton(1.25))
1.25
```

## Choosing a `Value` type

Direct class names use the pattern `<dtype><Semantics>`, with
`Conversions::Explicit` as the Python default:

```python
fpsan.Float32Triton
fpsan.Float32Field
fpsan.Float64FieldFast
fpsan.FP8E5M2PythagoreanRing
```

The C++ `Conversions` template parameter is still represented:

```python
fpsan.Float32Triton          # Explicit, the Python default
fpsan.Float32TritonExplicit  # alias for the same class
fpsan.Float32TritonImplicit
```

In Python, there are no C++ implicit conversions during operator overload
resolution, so `Conversions::Explicit` is the natural default. The implicit
classes are provided mostly to mirror the C++ template surface.

You can also use the factory:

```python
>>> F = fpsan.value_type("float32", fpsan.Semantics.Field)
>>> F is fpsan.Float32Field
True
```

If NumPy is installed, NumPy scalar types and dtype objects work for the common
IEEE formats:

```python
>>> import numpy as np
>>> fpsan.value_type(np.float32, fpsan.Semantics.Triton)
<class 'fpsan.Float32Triton'>
>>> fpsan.value_type(np.dtype("float64"), fpsan.Semantics.Field)
<class 'fpsan.Float64Field'>
```

The dtype strings accepted by the factory include `"float16"`, `"float32"`,
`"float64"`, `"bfloat16"` when the C++ compiler supports `__bf16`,
`"fp8_e4m3"`, and `"fp8_e5m2"`.

## Math functions

The module exposes overloads for the same scalar math names as C++:

```python
>>> F = fpsan.Float32Field
>>> fpsan.sqrt(F(2.0) * F(8.0)) == fpsan.sqrt(F(2.0)) * fpsan.sqrt(F(8.0))
True
>>> fpsan.fma(F(2.0), F(8.0), F(1.0)) == F(2.0) * F(8.0) + F(1.0)
True
```

Available unary functions include `exp`, `exp2`, `exp10`, `log`, `log2`,
`log10`, `sin`, `cos`, `sqrt`, `precise_sqrt`, `rsqrt`, `cbrt`, `erf`,
`floor`, `ceil`, `rcp`, `fract`, and `tanh`.

Available multi-argument functions include `fma`, `fmod`, `fmin`, `fmax`,
`min`, `max`, and `fmed3`.
