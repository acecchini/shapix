"""Tests for _array_types.py — array type factory and ShapeChecker."""

from __future__ import annotations

import numpy as np
import pytest
from beartype import beartype

from shapix import N, C, H, W, __, _N
from shapix._array_types import _ArrayFactory, make_array_type
from shapix._dtypes import FLOAT32, INT64, SHAPED
from shapix.numpy import Float32Array, Int64Array, ShapedArray


class TestMakeArrayType:
    def test_returns_factory(self) -> None:
        factory = make_array_type(np.ndarray, FLOAT32)
        assert isinstance(factory, _ArrayFactory)

    def test_factory_name(self) -> None:
        factory = make_array_type(np.ndarray, FLOAT32)
        assert factory.__name__ == "Float32Array"

    def test_subscript_returns_annotated(self) -> None:
        factory = make_array_type(np.ndarray, FLOAT32)
        hint = factory[N, C]
        # Should be an Annotated type
        assert hasattr(hint, "__metadata__")


class TestDtypeChecking:
    def test_correct_dtype(self) -> None:
        @beartype
        def f(x: Float32Array[N]) -> Float32Array[N]:
            return x

        f(np.ones((5,), dtype=np.float32))

    def test_wrong_dtype(self) -> None:
        @beartype
        def f(x: Float32Array[N]) -> Float32Array[N]:
            return x

        with pytest.raises(Exception):
            f(np.ones((5,), dtype=np.int32))

    def test_category_dtype(self) -> None:
        @beartype
        def f(x: ShapedArray[N]) -> ShapedArray[N]:
            return x

        # ShapedArray accepts any dtype
        f(np.ones((5,), dtype=np.float32))
        f(np.ones((5,), dtype=np.int64))
        f(np.ones((5,), dtype=np.bool_))


class TestShapeChecking:
    def test_correct_shape(self) -> None:
        @beartype
        def f(x: Float32Array[N, C]) -> Float32Array[N, C]:
            return x

        result = f(np.ones((4, 3), dtype=np.float32))
        assert result.shape == (4, 3)

    def test_wrong_rank(self) -> None:
        @beartype
        def f(x: Float32Array[N, C]) -> Float32Array[N, C]:
            return x

        with pytest.raises(Exception):
            f(np.ones((4,), dtype=np.float32))

    def test_fixed_dim(self) -> None:
        @beartype
        def f(x: Float32Array[3, N]) -> Float32Array[3, N]:
            return x

        f(np.ones((3, 5), dtype=np.float32))

        with pytest.raises(Exception):
            f(np.ones((4, 5), dtype=np.float32))

    def test_anonymous_dim(self) -> None:
        @beartype
        def f(x: Float32Array[_N, C]) -> Float32Array[_N, C]:
            return x

        # _N is anonymous — matches anything, first arg can have different first dim
        f(np.ones((3, 5), dtype=np.float32))
        f(np.ones((99, 5), dtype=np.float32))

    def test_return_value_checked(self) -> None:
        @beartype
        def f(x: Float32Array[N, C]) -> Float32Array[N, C]:
            return np.zeros((1, 1), dtype=np.float32)

        with pytest.raises(Exception):
            f(np.ones((4, 3), dtype=np.float32))


class TestSymbolicDims:
    def test_addition(self) -> None:
        @beartype
        def f(x: Float32Array[N], y: Float32Array[N + 1]) -> Float32Array[N]:
            return x

        f(np.ones((5,), dtype=np.float32), np.ones((6,), dtype=np.float32))

    def test_addition_mismatch(self) -> None:
        @beartype
        def f(x: Float32Array[N], y: Float32Array[N + 1]) -> Float32Array[N]:
            return x

        with pytest.raises(Exception):
            f(np.ones((5,), dtype=np.float32), np.ones((5,), dtype=np.float32))

    def test_multiplication(self) -> None:
        @beartype
        def f(x: Float32Array[N], y: Float32Array[2 * N]) -> Float32Array[N]:
            return x

        f(np.ones((5,), dtype=np.float32), np.ones((10,), dtype=np.float32))
