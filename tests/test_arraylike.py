"""Tests for recursive ArrayLike types — scalar, array, or nested sequences."""

from __future__ import annotations

import numpy as np
from beartype import beartype

import pytest
from beartype.roar import BeartypeCallHintParamViolation

from shapix.numpy import ArrayLike, F32Like, I64Like, Float32Like


class TestRecursiveArrayLike:
    def test_scalar(self) -> None:
        @beartype
        def f(x: F32Like) -> float:
            return float(np.asarray(x).sum())

        assert f(3.14) == 3.14

    def test_1d_list(self) -> None:
        @beartype
        def f(x: F32Like) -> float:
            return float(np.asarray(x).sum())

        assert f([1.0, 2.0, 3.0]) == 6.0

    def test_2d_nested_list(self) -> None:
        @beartype
        def f(x: F32Like) -> float:
            return float(np.asarray(x).sum())

        assert f([[1.0, 2.0], [3.0, 4.0]]) == 10.0

    def test_3d_nested_list(self) -> None:
        @beartype
        def f(x: F32Like) -> float:
            return float(np.asarray(x).sum())

        f([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])

    def test_ndarray(self) -> None:
        @beartype
        def f(x: F32Like) -> float:
            return float(np.asarray(x).sum())

        assert f(np.ones((3, 4), dtype=np.float32)) == 12.0

    def test_deep_nesting(self) -> None:
        """6D+ nesting should work with recursive type (no 5D limit)."""
        @beartype
        def f(x: F32Like) -> float:
            return float(np.asarray(x).sum())

        # 6D nesting: would fail with old _ArrayLike5DLess
        data = [[[[[[1.0]]]]]]
        f(data)

    def test_int_arraylike(self) -> None:
        @beartype
        def f(x: I64Like) -> int:
            return int(np.asarray(x).sum())

        assert f(42) == 42
        assert f([1, 2, 3]) == 6
        assert f([[1, 2], [3, 4]]) == 10

    def test_custom_arraylike(self) -> None:
        """Users can construct custom ArrayLike types."""
        type MyLike = ArrayLike[float, np.ndarray]

        @beartype
        def f(x: MyLike) -> float:
            return float(np.asarray(x).sum())

        assert f(3.14) == 3.14
        assert f([1.0, 2.0]) == 3.0
        assert f(np.array([1.0, 2.0])) == 3.0

    def test_numpy_scalar(self) -> None:
        """np.float32 scalars are accepted by F32Like."""
        @beartype
        def f(x: F32Like) -> float:
            return float(np.asarray(x).sum())

        assert f(np.float32(2.5)) == 2.5

    def test_empty_list(self) -> None:
        """Empty list is a valid Sequence."""
        @beartype
        def f(x: F32Like) -> float:
            return float(np.asarray(x).sum())

        assert f([]) == 0.0

    def test_tuple_input(self) -> None:
        """Tuples are Sequences, should be accepted."""
        @beartype
        def f(x: F32Like) -> float:
            return float(np.asarray(x).sum())

        assert f((1.0, 2.0, 3.0)) == 6.0

    def test_nested_tuple(self) -> None:
        """Nested tuples should work like nested lists."""
        @beartype
        def f(x: F32Like) -> float:
            return float(np.asarray(x).sum())

        assert f(((1.0, 2.0), (3.0, 4.0))) == 10.0

    def test_wrong_type_rejected(self) -> None:
        """Non-numeric types should be rejected."""
        @beartype
        def f(x: F32Like) -> float:
            return float(np.asarray(x).sum())

        with pytest.raises(BeartypeCallHintParamViolation):
            f(object())  # type: ignore[arg-type]

    def test_0d_ndarray(self) -> None:
        """0D arrays (scalars wrapped in ndarray) should be accepted."""
        @beartype
        def f(x: F32Like) -> float:
            return float(np.asarray(x).sum())

        assert f(np.array(5.0, dtype=np.float32)) == 5.0
