"""End-to-end integration tests for shapix with @beartype."""

from __future__ import annotations

import numpy as np
import pytest
from beartype import beartype

from shapix import N, C, H, W, B, Any, _
from shapix.numpy import F32, F64, Int, I32


class TestBasicBeartype:
    """Tests that shapix works correctly with standard @beartype."""

    def test_simple_function(self) -> None:
        @beartype
        def normalize(x: F32[N, C]) -> F32[N, C]:
            return x / x.sum()

        result = normalize(np.ones((4, 3), dtype=np.float32))
        assert result.shape == (4, 3)

    def test_cross_arg_same_dims(self) -> None:
        @beartype
        def add(
            x: F32[N, C, H, W],
            y: F32[N, C, H, W],
        ) -> F32[N, C, H, W]:
            return x + y

        a = np.ones((2, 3, 28, 28), dtype=np.float32)
        b = np.ones((2, 3, 28, 28), dtype=np.float32)
        result = add(a, b)
        assert result.shape == (2, 3, 28, 28)

    def test_cross_arg_shared_dim(self) -> None:
        @beartype
        def matmul(
            a: F32[N, C], b: F32[C, H]
        ) -> F32[N, H]:
            return a @ b

        result = matmul(
            np.ones((4, 3), dtype=np.float32),
            np.ones((3, 5), dtype=np.float32),
        )
        assert result.shape == (4, 5)


class TestErrorDetection:
    def test_shape_mismatch(self) -> None:
        @beartype
        def f(x: F32[N, C], y: F32[N, C]) -> F32[N, C]:
            return x + y

        with pytest.raises(Exception):
            f(
                np.ones((4, 3), dtype=np.float32),
                np.ones((5, 3), dtype=np.float32),  # N mismatch
            )

    def test_dtype_mismatch(self) -> None:
        @beartype
        def f(x: F32[N]) -> F32[N]:
            return x

        with pytest.raises(Exception):
            f(np.ones((5,), dtype=np.float64))

    def test_rank_mismatch(self) -> None:
        @beartype
        def f(x: F32[N, C]) -> F32[N, C]:
            return x

        with pytest.raises(Exception):
            f(np.ones((5,), dtype=np.float32))

    def test_return_type_wrong_shape(self) -> None:
        @beartype
        def f(x: F32[N, C]) -> F32[N, C]:
            return np.zeros((1, 1), dtype=np.float32)

        with pytest.raises(Exception):
            f(np.ones((4, 3), dtype=np.float32))

    def test_return_type_wrong_dtype(self) -> None:
        @beartype
        def f(x: F32[N]) -> F32[N]:
            return np.ones((5,), dtype=np.int32)

        with pytest.raises(Exception):
            f(np.ones((5,), dtype=np.float32))


class TestSequentialCalls:
    def test_different_shapes(self) -> None:
        @beartype
        def f(x: F32[N]) -> F32[N]:
            return x

        for size in [3, 7, 1, 100, 42]:
            result = f(np.ones((size,), dtype=np.float32))
            assert result.shape == (size,)

    def test_different_shapes_multi_dim(self) -> None:
        @beartype
        def f(
            x: F32[N, C], y: F32[N, C]
        ) -> F32[N, C]:
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
        def inner(x: F32[N]) -> F32[N]:
            return x * 2

        @beartype
        def outer(x: F32[N, C]) -> F32[N]:
            row = x[:, 0]
            return inner(row)

        result = outer(np.ones((4, 3), dtype=np.float32))
        assert result.shape == (4,)

    def test_deep_nesting(self) -> None:
        @beartype
        def a(x: F32[N]) -> F32[N]:
            return x

        @beartype
        def b(x: F32[N, C]) -> F32[N]:
            return a(x[:, 0])

        @beartype
        def c(x: F32[N, C, H]) -> F32[N]:
            return b(x[:, :, 0])

        result = c(np.ones((4, 3, 5), dtype=np.float32))
        assert result.shape == (4,)


class TestVariadicDims:
    def test_anonymous_variadic(self) -> None:
        @beartype
        def f(x: F32[~_, C]) -> F32[~_, C]:
            return x

        f(np.ones((3,), dtype=np.float32))
        f(np.ones((2, 3), dtype=np.float32))
        f(np.ones((1, 2, 3), dtype=np.float32))

    def test_any_alias(self) -> None:
        """Any is a backward-compat alias for ~_."""
        @beartype
        def f(x: F32[Any, C]) -> F32[Any, C]:
            return x

        f(np.ones((3,), dtype=np.float32))
        f(np.ones((2, 3), dtype=np.float32))

    def test_variadic_named(self) -> None:
        @beartype
        def f(
            x: F32[~B, C], y: F32[~B, C]
        ) -> F32[~B, C]:
            return x + y

        result = f(
            np.ones((2, 3, 4), dtype=np.float32),
            np.ones((2, 3, 4), dtype=np.float32),
        )
        assert result.shape == (2, 3, 4)

    def test_variadic_mismatch(self) -> None:
        @beartype
        def f(
            x: F32[~B, C], y: F32[~B, C]
        ) -> F32[~B, C]:
            return x + y

        with pytest.raises(Exception):
            f(
                np.ones((2, 3, 4), dtype=np.float32),
                np.ones((5, 3, 4), dtype=np.float32),
            )


class TestBroadcastableDims:
    def test_size_1_matches(self) -> None:
        @beartype
        def f(x: F32[+N, C]) -> F32[+N, C]:
            return x

        # +N allows size 1 always
        f(np.ones((1, 5), dtype=np.float32))

    def test_normal_size(self) -> None:
        @beartype
        def f(x: F32[+N, C]) -> F32[+N, C]:
            return x

        f(np.ones((10, 5), dtype=np.float32))


class TestFixedDims:
    def test_literal_int(self) -> None:
        @beartype
        def f(x: F32[3, N]) -> F32[3, N]:
            return x

        f(np.ones((3, 10), dtype=np.float32))

        with pytest.raises(Exception):
            f(np.ones((4, 10), dtype=np.float32))


class TestAnonymousDims:
    def test_underscore(self) -> None:
        @beartype
        def f(x: F32[_, C]) -> F32[_, C]:
            return x

        # _ matches any value, no consistency check
        f(np.ones((3, 5), dtype=np.float32))
        f(np.ones((99, 5), dtype=np.float32))


class TestMultipleDtypes:
    def test_float64(self) -> None:
        @beartype
        def f(x: F64[N]) -> F64[N]:
            return x

        f(np.ones((5,), dtype=np.float64))

    def test_int32(self) -> None:
        @beartype
        def f(x: I32[N]) -> I32[N]:
            return x

        f(np.ones((5,), dtype=np.int32))

    def test_int_category(self) -> None:
        @beartype
        def f(x: Int[N]) -> Int[N]:
            return x

        f(np.ones((5,), dtype=np.int32))
        f(np.ones((5,), dtype=np.int64))

    def test_mixed_dtypes_in_signature(self) -> None:
        @beartype
        def f(x: F32[N, C], y: I32[N]) -> F32[N, C]:
            return x

        f(
            np.ones((4, 3), dtype=np.float32),
            np.ones((4,), dtype=np.int32),
        )
