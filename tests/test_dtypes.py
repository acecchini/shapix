"""Tests for _dtypes.py — dtype spec matching and extraction."""

from __future__ import annotations

import numpy as np
import pytest

from shapix._dtypes import (
    BOOL,
    COMPLEX128,
    FLOAT,
    FLOAT32,
    FLOAT64,
    INT,
    INT32,
    INT64,
    INTEGER,
    NUM,
    SHAPED,
    UINT8,
    DtypeSpec,
    extract_dtype_str,
)


class TestExtractDtypeStr:
    def test_numpy_float32(self) -> None:
        arr = np.zeros((2,), dtype=np.float32)
        assert extract_dtype_str(arr) == "float32"

    def test_numpy_int64(self) -> None:
        arr = np.zeros((2,), dtype=np.int64)
        assert extract_dtype_str(arr) == "int64"

    def test_numpy_bool(self) -> None:
        arr = np.zeros((2,), dtype=np.bool_)
        assert extract_dtype_str(arr) in ("bool", "bool_")

    def test_numpy_complex128(self) -> None:
        arr = np.zeros((2,), dtype=np.complex128)
        assert extract_dtype_str(arr) == "complex128"

    def test_numpy_uint8(self) -> None:
        arr = np.zeros((2,), dtype=np.uint8)
        assert extract_dtype_str(arr) == "uint8"

    def test_no_dtype(self) -> None:
        assert extract_dtype_str(42) == ""

    def test_numpy_structured(self) -> None:
        dt = np.dtype([("x", np.float32), ("y", np.int32)])
        arr = np.zeros((2,), dtype=dt)
        # Structured arrays have dtype.type.__name__ == "void"
        # Falls back to str(dtype)
        result = extract_dtype_str(arr)
        assert "x" in result or "void" in result


class TestDtypeSpecMatches:
    def test_float32_matches(self) -> None:
        arr = np.zeros((2,), dtype=np.float32)
        assert FLOAT32.matches(arr)

    def test_float32_rejects_int(self) -> None:
        arr = np.zeros((2,), dtype=np.int32)
        assert not FLOAT32.matches(arr)

    def test_float_group(self) -> None:
        assert FLOAT.matches(np.zeros((2,), dtype=np.float32))
        assert FLOAT.matches(np.zeros((2,), dtype=np.float64))
        assert FLOAT.matches(np.zeros((2,), dtype=np.float16))
        assert not FLOAT.matches(np.zeros((2,), dtype=np.int32))

    def test_int_group(self) -> None:
        assert INT.matches(np.zeros((2,), dtype=np.int32))
        assert INT.matches(np.zeros((2,), dtype=np.int64))
        assert not INT.matches(np.zeros((2,), dtype=np.uint8))

    def test_integer_group(self) -> None:
        assert INTEGER.matches(np.zeros((2,), dtype=np.int32))
        assert INTEGER.matches(np.zeros((2,), dtype=np.uint8))
        assert not INTEGER.matches(np.zeros((2,), dtype=np.float32))

    def test_num_group(self) -> None:
        assert NUM.matches(np.zeros((2,), dtype=np.float32))
        assert NUM.matches(np.zeros((2,), dtype=np.int64))
        assert NUM.matches(np.zeros((2,), dtype=np.complex128))
        assert not NUM.matches(np.zeros((2,), dtype=np.bool_))

    def test_shaped_group(self) -> None:
        assert SHAPED.matches(np.zeros((2,), dtype=np.float32))
        assert SHAPED.matches(np.zeros((2,), dtype=np.bool_))

    def test_bool(self) -> None:
        assert BOOL.matches(np.zeros((2,), dtype=np.bool_))
        assert not BOOL.matches(np.zeros((2,), dtype=np.int32))


class TestCustomDtypeSpec:
    def test_custom_spec(self) -> None:
        spec = DtypeSpec("MyType", frozenset({"float32", "float64"}))
        assert spec.matches(np.zeros((2,), dtype=np.float32))
        assert spec.matches(np.zeros((2,), dtype=np.float64))
        assert not spec.matches(np.zeros((2,), dtype=np.int32))
