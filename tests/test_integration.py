"""End-to-end integration tests for shapix with @beartype."""

from __future__ import annotations

import numpy as np
import pytest
from beartype import beartype

from shapix import N, C, H, W, Any, __, sB, hN
from shapix.numpy import Float32Array, Float64Array, IntArray, Int32Array


class TestBasicBeartype:
    """Tests that shapix works correctly with standard @beartype."""

    def test_simple_function(self) -> None:
        @beartype
        def normalize(x: Float32Array[N, C]) -> Float32Array[N, C]:
            return x / x.sum()

        result = normalize(np.ones((4, 3), dtype=np.float32))
        assert result.shape == (4, 3)

    def test_cross_arg_same_dims(self) -> None:
        @beartype
        def add(
            x: Float32Array[N, C, H, W],
            y: Float32Array[N, C, H, W],
        ) -> Float32Array[N, C, H, W]:
            return x + y

        a = np.ones((2, 3, 28, 28), dtype=np.float32)
        b = np.ones((2, 3, 28, 28), dtype=np.float32)
        result = add(a, b)
        assert result.shape == (2, 3, 28, 28)

    def test_cross_arg_shared_dim(self) -> None:
        @beartype
        def matmul(
            a: Float32Array[N, C], b: Float32Array[C, H]
        ) -> Float32Array[N, H]:
            return a @ b

        result = matmul(
            np.ones((4, 3), dtype=np.float32),
            np.ones((3, 5), dtype=np.float32),
        )
        assert result.shape == (4, 5)


class TestErrorDetection:
    def test_shape_mismatch(self) -> None:
        @beartype
        def f(x: Float32Array[N, C], y: Float32Array[N, C]) -> Float32Array[N, C]:
            return x + y

        with pytest.raises(Exception):
            f(
                np.ones((4, 3), dtype=np.float32),
                np.ones((5, 3), dtype=np.float32),  # N mismatch
            )

    def test_dtype_mismatch(self) -> None:
        @beartype
        def f(x: Float32Array[N]) -> Float32Array[N]:
            return x

        with pytest.raises(Exception):
            f(np.ones((5,), dtype=np.float64))

    def test_rank_mismatch(self) -> None:
        @beartype
        def f(x: Float32Array[N, C]) -> Float32Array[N, C]:
            return x

        with pytest.raises(Exception):
            f(np.ones((5,), dtype=np.float32))

    def test_return_type_wrong_shape(self) -> None:
        @beartype
        def f(x: Float32Array[N, C]) -> Float32Array[N, C]:
            return np.zeros((1, 1), dtype=np.float32)

        with pytest.raises(Exception):
            f(np.ones((4, 3), dtype=np.float32))

    def test_return_type_wrong_dtype(self) -> None:
        @beartype
        def f(x: Float32Array[N]) -> Float32Array[N]:
            return np.ones((5,), dtype=np.int32)

        with pytest.raises(Exception):
            f(np.ones((5,), dtype=np.float32))


class TestSequentialCalls:
    def test_different_shapes(self) -> None:
        @beartype
        def f(x: Float32Array[N]) -> Float32Array[N]:
            return x

        for size in [3, 7, 1, 100, 42]:
            result = f(np.ones((size,), dtype=np.float32))
            assert result.shape == (size,)

    def test_different_shapes_multi_dim(self) -> None:
        @beartype
        def f(
            x: Float32Array[N, C], y: Float32Array[N, C]
        ) -> Float32Array[N, C]:
            return x + y

        for n, c in [(2, 3), (10, 20), (1, 1), (50, 7)]:
            result = f(
                np.ones((n, c), dtype=np.float32),
                np.ones((n, c), dtype=np.float32),
            )
            assert result.shape == (n, c)


class TestNestedCalls:
    def test_inner_outer_independent(self) -> None:
        @beartype
        def inner(x: Float32Array[N]) -> Float32Array[N]:
            return x * 2

        @beartype
        def outer(x: Float32Array[N, C]) -> Float32Array[N]:
            row = x[:, 0]
            return inner(row)

        result = outer(np.ones((4, 3), dtype=np.float32))
        assert result.shape == (4,)

    def test_deep_nesting(self) -> None:
        @beartype
        def a(x: Float32Array[N]) -> Float32Array[N]:
            return x

        @beartype
        def b(x: Float32Array[N, C]) -> Float32Array[N]:
            return a(x[:, 0])

        @beartype
        def c(x: Float32Array[N, C, H]) -> Float32Array[N]:
            return b(x[:, :, 0])

        result = c(np.ones((4, 3, 5), dtype=np.float32))
        assert result.shape == (4,)


class TestVariadicDims:
    def test_any_prefix(self) -> None:
        @beartype
        def f(x: Float32Array[Any, C]) -> Float32Array[Any, C]:
            return x

        f(np.ones((3,), dtype=np.float32))
        f(np.ones((2, 3), dtype=np.float32))
        f(np.ones((1, 2, 3), dtype=np.float32))

    def test_variadic_named(self) -> None:
        @beartype
        def f(
            x: Float32Array[sB, C], y: Float32Array[sB, C]
        ) -> Float32Array[sB, C]:
            return x + y

        result = f(
            np.ones((2, 3, 4), dtype=np.float32),
            np.ones((2, 3, 4), dtype=np.float32),
        )
        assert result.shape == (2, 3, 4)

    def test_variadic_mismatch(self) -> None:
        @beartype
        def f(
            x: Float32Array[sB, C], y: Float32Array[sB, C]
        ) -> Float32Array[sB, C]:
            return x + y

        with pytest.raises(Exception):
            f(
                np.ones((2, 3, 4), dtype=np.float32),
                np.ones((5, 3, 4), dtype=np.float32),
            )


class TestBroadcastableDims:
    def test_size_1_matches(self) -> None:
        @beartype
        def f(x: Float32Array[hN, C]) -> Float32Array[hN, C]:
            return x

        # hN allows size 1 always
        f(np.ones((1, 5), dtype=np.float32))

    def test_normal_size(self) -> None:
        @beartype
        def f(x: Float32Array[hN, C]) -> Float32Array[hN, C]:
            return x

        f(np.ones((10, 5), dtype=np.float32))


class TestFixedDims:
    def test_literal_int(self) -> None:
        @beartype
        def f(x: Float32Array[3, N]) -> Float32Array[3, N]:
            return x

        f(np.ones((3, 10), dtype=np.float32))

        with pytest.raises(Exception):
            f(np.ones((4, 10), dtype=np.float32))


class TestAnonymousDims:
    def test_underscore(self) -> None:
        @beartype
        def f(x: Float32Array[__, C]) -> Float32Array[__, C]:
            return x

        # __ matches any value, no consistency check
        f(np.ones((3, 5), dtype=np.float32))
        f(np.ones((99, 5), dtype=np.float32))


class TestMultipleDtypes:
    def test_float64(self) -> None:
        @beartype
        def f(x: Float64Array[N]) -> Float64Array[N]:
            return x

        f(np.ones((5,), dtype=np.float64))

    def test_int32(self) -> None:
        @beartype
        def f(x: Int32Array[N]) -> Int32Array[N]:
            return x

        f(np.ones((5,), dtype=np.int32))

    def test_int_category(self) -> None:
        @beartype
        def f(x: IntArray[N]) -> IntArray[N]:
            return x

        f(np.ones((5,), dtype=np.int32))
        f(np.ones((5,), dtype=np.int64))

    def test_mixed_dtypes_in_signature(self) -> None:
        @beartype
        def f(x: Float32Array[N, C], y: Int32Array[N]) -> Float32Array[N, C]:
            return x

        f(
            np.ones((4, 3), dtype=np.float32),
            np.ones((4,), dtype=np.int32),
        )
