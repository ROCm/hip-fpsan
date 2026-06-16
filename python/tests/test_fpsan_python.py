# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import math

import fpsan


def test_factory_accepts_numpy_dtype_objects_if_numpy_is_present():
    try:
        import numpy as np
    except Exception:
        np = None

    assert fpsan.value_type("float32", fpsan.Semantics.Triton) is fpsan.Float32Triton
    if np is not None:
        assert fpsan.value_type(np.float32, fpsan.Semantics.Triton) is fpsan.Float32Triton
        assert fpsan.value_type(np.dtype("float32"), fpsan.Semantics.Field) is fpsan.Float32Field


def test_triton_and_field_differ_on_value_facts():
    triton = fpsan.Float32Triton
    field = fpsan.Float32Field

    assert triton(2.0) + triton(2.0) != triton(4.0)
    assert field(2.0) + field(2.0) == field(4.0)
    assert field(3.0) * field(3.0) == field(9.0)


def test_payload_construction_and_math():
    field = fpsan.value_type("float32", fpsan.Semantics.Field)
    x = field(2.0)
    y = field(8.0)

    assert field.from_payload(x.payload) == x
    assert fpsan.sqrt(x * y) == fpsan.sqrt(x) * fpsan.sqrt(y)
    assert fpsan.fma(x, y, field(1.0)) == x * y + field(1.0)


def test_native_and_triton_are_convertible_back_to_python_float():
    native = fpsan.Float32Native
    triton = fpsan.Float32Triton

    assert math.isclose(float(native(1.25) + native(2.5)), 3.75)
    assert math.isclose(float(triton(1.25)), 1.25)


def test_conversions_parameter_is_reflected():
    assert (
        fpsan.value_type("float32", fpsan.Semantics.Triton, fpsan.Conversions.Explicit)
        is fpsan.Float32Triton
    )
    assert (
        fpsan.value_type("float32", fpsan.Semantics.Triton, fpsan.Conversions.Implicit)
        is fpsan.Float32TritonImplicit
    )
    assert fpsan.Float32Triton(1.0).conversions is fpsan.Conversions.Explicit
    assert fpsan.Float32TritonImplicit(1.0).conversions is fpsan.Conversions.Implicit


def test_fp8_scalar_types_are_available():
    fp8 = fpsan.value_type("fp8_e5m2", fpsan.Semantics.Field)
    assert fp8(2.0) + fp8(2.0) == fp8(4.0)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
