"""Comprehensive runtime checking tests for NumPy arrays with beartype.

Covers dtype acceptance, shape matching, cross-argument consistency, symbolic
dimensions, variadic/broadcastable/anonymous dims, return checking, violation
detection, ArrayLike types, and edge cases.
"""

from __future__ import annotations

import numpy as np
import pytest
from beartype import beartype
from beartype.door import is_bearable
from beartype.roar import (
  BeartypeCallHintParamViolation,
  BeartypeCallHintReturnViolation,
)

import shapix
from shapix import B, C, H, N, Dimension, __
from shapix._array_types import _ArrayFactory, make_array_type
from shapix._dtypes import FLOAT32
from shapix.numpy import (
  ArrayLike,
  Bool,
  BoolLk,
  F16,
  F32,
  F32Like,
  F64,
  Float,
  I32,
  I64,
  I64Like,
  Int,
  Num,
  Shaped,
  U8,
)


# =====================================================================
# Factory internals
# =====================================================================


class TestFactory:
  def test_returns_factory(self) -> None:
    factory = make_array_type(np.ndarray, FLOAT32)
    assert isinstance(factory, _ArrayFactory)

  def test_factory_name(self) -> None:
    factory = make_array_type(np.ndarray, FLOAT32)
    assert factory.__name__ == "Float32Array"

  def test_subscript_returns_annotated(self) -> None:
    hint = F32[N, C]
    assert hasattr(hint, "__metadata__")


# =====================================================================
# Dtype acceptance / rejection
# =====================================================================


class TestDtypeAcceptance:
  @pytest.mark.parametrize(
    "ArrayType, good_dtype, bad_dtypes",
    [
      (F32, np.float32, [np.float64, np.float16, np.int32, np.bool_]),
      (F64, np.float64, [np.float32]),
      (F16, np.float16, [np.float32]),
      (I32, np.int32, [np.float32, np.int64]),
      (I64, np.int64, [np.float32, np.uint8]),
      (U8, np.uint8, [np.int8, np.int32]),
      (Bool, np.bool_, [np.int32, np.float32]),
    ],
  )
  def test_accepts_correct_rejects_wrong(
    self, ArrayType: _ArrayFactory, good_dtype: type, bad_dtypes: list[type]
  ) -> None:
    @beartype
    def f(x: ArrayType[N]) -> ArrayType[N]:  # type: ignore[valid-type]
      return x

    f(np.ones(3, dtype=good_dtype))
    for bad in bad_dtypes:
      with pytest.raises(BeartypeCallHintParamViolation):
        f(np.ones(3, dtype=bad))


class TestDtypeCategoryTypes:
  @pytest.mark.parametrize(
    "CategoryType, good_dtypes, bad_dtypes",
    [
      (Float, [np.float32, np.float64, np.float16], [np.int64, np.bool_]),
      (Int, [np.int32, np.int64], [np.float32, np.uint32]),
      (Num, [np.float32, np.int64, np.complex128], [np.bool_]),
    ],
  )
  def test_category_accept_reject(
    self, CategoryType: _ArrayFactory, good_dtypes: list[type], bad_dtypes: list[type]
  ) -> None:
    @beartype
    def f(x: CategoryType[N]) -> CategoryType[N]:  # type: ignore[valid-type]
      return x

    for g in good_dtypes:
      f(np.ones(3, dtype=g))
    for b in bad_dtypes:
      with pytest.raises(BeartypeCallHintParamViolation):
        f(np.ones(3, dtype=b))

  def test_shaped_accepts_all(self) -> None:
    @beartype
    def f(x: Shaped[N]) -> Shaped[N]:
      return x

    for dtype in [
      np.float32,
      np.float64,
      np.int32,
      np.int64,
      np.uint8,
      np.bool_,
      np.complex64,
    ]:
      f(np.ones(3, dtype=dtype))


class TestDtypeReturnViolations:
  def test_return_wrong_concrete_dtype(self) -> None:
    @beartype
    def f(x: F32[N]) -> F32[N]:
      return x.astype(np.float64)  # type: ignore

    with pytest.raises(BeartypeCallHintReturnViolation):
      f(np.ones(3, dtype=np.float32))

  def test_return_wrong_category_dtype(self) -> None:
    @beartype
    def f(x: F32[N]) -> Float[N]:
      return x.astype(np.int32)  # type: ignore

    with pytest.raises(BeartypeCallHintReturnViolation):
      f(np.ones(3, dtype=np.float32))


# =====================================================================
# Shape matching
# =====================================================================


class TestShapeMatching:
  def test_correct_2d(self) -> None:
    @beartype
    def f(x: F32[N, C]) -> F32[N, C]:
      return x

    assert f(np.ones((4, 3), dtype=np.float32)).shape == (4, 3)

  @pytest.mark.parametrize(
    "shape, ndim_expected",
    [
      ((5,), 2),  # 1D for 2D spec
      ((2, 3, 4), 2),  # 3D for 2D spec
    ],
  )
  def test_wrong_rank(self, shape: tuple[int, ...], ndim_expected: int) -> None:
    @beartype
    def f(x: F32[N, C]) -> F32[N, C]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones(shape, dtype=np.float32))

  def test_0d_scalar_rejected(self) -> None:
    @beartype
    def f(x: F32[N]) -> F32[N]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.float32(1.0))  # type: ignore[arg-type]

  def test_non_array_rejected(self) -> None:
    @beartype
    def f(x: F32[N]) -> F32[N]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f([1.0, 2.0, 3.0])  # type: ignore[arg-type]

  @pytest.mark.parametrize(
    "fixed, actual, should_pass",
    [(3, 3, True), (3, 4, False), (3, 2, False), (0, 0, True), (0, 1, False)],
  )
  def test_fixed_dim(self, fixed: int, actual: int, should_pass: bool) -> None:
    @beartype
    def f(x: F32[fixed, N]) -> F32[fixed, N]:  # type: ignore[valid-type]
      return x

    if should_pass:
      f(np.ones((actual, 5), dtype=np.float32))
    else:
      with pytest.raises(BeartypeCallHintParamViolation):
        f(np.ones((actual, 5), dtype=np.float32))

  def test_multiple_fixed_dims(self) -> None:
    @beartype
    def f(x: F32[3, 4, 5]) -> F32[3, 4, 5]:
      return x

    f(np.ones((3, 4, 5), dtype=np.float32))
    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones((3, 4, 6), dtype=np.float32))

  def test_return_wrong_rank(self) -> None:
    @beartype
    def f(x: F32[N, C]) -> F32[N, C]:
      return x.ravel()

    with pytest.raises(BeartypeCallHintReturnViolation):
      f(np.ones((3, 4), dtype=np.float32))


# =====================================================================
# Cross-argument consistency
# =====================================================================


class TestCrossArgConsistency:
  def test_same_dims_pass(self) -> None:
    @beartype
    def f(x: F32[N, C], y: F32[N, C]) -> F32[N, C]:
      return x + y

    assert f(
      np.ones((4, 3), dtype=np.float32), np.ones((4, 3), dtype=np.float32)
    ).shape == (4, 3)

  @pytest.mark.parametrize(
    "shape_x, shape_y",
    [
      ((4, 3), (5, 3)),  # N mismatch
      ((4, 3), (4, 7)),  # C mismatch
    ],
  )
  def test_dim_mismatch(
    self, shape_x: tuple[int, ...], shape_y: tuple[int, ...]
  ) -> None:
    @beartype
    def f(x: F32[N, C], y: F32[N, C]) -> F32[N, C]:
      return x + y

    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones(shape_x, dtype=np.float32), np.ones(shape_y, dtype=np.float32))

  def test_shared_dim_matmul(self) -> None:
    @beartype
    def f(x: F32[N, C], y: F32[C, H]) -> F32[N, H]:
      return x @ y

    assert f(
      np.ones((4, 3), dtype=np.float32), np.ones((3, 5), dtype=np.float32)
    ).shape == (4, 5)

  def test_shared_dim_matmul_mismatch(self) -> None:
    @beartype
    def f(x: F32[N, C], y: F32[C, H]) -> F32[N, H]:
      return x @ y

    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones((4, 3), dtype=np.float32), np.ones((7, 5), dtype=np.float32))

  def test_three_args(self) -> None:
    @beartype
    def f(x: F32[N, C], y: F32[C, H], z: F32[N]) -> F32[N, H]:
      return x @ y

    with pytest.raises(BeartypeCallHintParamViolation):
      f(
        np.ones((4, 3), dtype=np.float32),
        np.ones((3, 5), dtype=np.float32),
        np.ones((99,), dtype=np.float32),
      )

  def test_return_violates_bound_dim(self) -> None:
    @beartype
    def f(x: F32[N, C]) -> F32[N, C]:
      return np.ones((99, 99), dtype=np.float32)

    with pytest.raises(BeartypeCallHintReturnViolation):
      f(np.ones((4, 3), dtype=np.float32))

  def test_mixed_dtypes_shared_dim(self) -> None:
    @beartype
    def f(x: F32[N, C], idx: I32[N]) -> F32[N, C]:
      return x

    f(np.ones((4, 3), dtype=np.float32), np.ones(4, dtype=np.int32))

  def test_mixed_dtypes_mismatch(self) -> None:
    @beartype
    def f(x: F32[N, C], idx: I32[N]) -> F32[N, C]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones((4, 3), dtype=np.float32), np.ones(7, dtype=np.int32))


# =====================================================================
# Symbolic dimensions
# =====================================================================


class TestSymbolicDims:
  @pytest.mark.parametrize(
    "expr_dim, n_size, y_size, should_pass",
    [
      (N + 1, 5, 6, True),
      (N + 1, 5, 5, False),
      (N - 1, 10, 9, True),
      (N - 1, 10, 10, False),
      (2 * N, 5, 10, True),
      (2 * N, 5, 9, False),
      (N // 2, 10, 5, True),
      (N // 2, 10, 4, False),
    ],
  )
  def test_input_expressions(
    self, expr_dim: Dimension, n_size: int, y_size: int, should_pass: bool
  ) -> None:
    @beartype
    def f(x: F32[N], y: F32[expr_dim]) -> F32[N]:  # type: ignore[valid-type]
      return x

    if should_pass:
      f(np.ones(n_size, dtype=np.float32), np.ones(y_size, dtype=np.float32))
    else:
      with pytest.raises(BeartypeCallHintParamViolation):
        f(np.ones(n_size, dtype=np.float32), np.ones(y_size, dtype=np.float32))

  def test_multi_variable_expression(self) -> None:
    @beartype
    def f(x: F32[N, C], y: F32[N * C]) -> F32[N, C]:
      return x

    f(np.ones((3, 4), dtype=np.float32), np.ones(12, dtype=np.float32))
    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones((3, 4), dtype=np.float32), np.ones(10, dtype=np.float32))

  def test_return_expression(self) -> None:
    @beartype
    def f(x: F32[N]) -> F32[N + 2]:
      return np.pad(x, 1)

    assert f(np.ones(5, dtype=np.float32)).shape == (7,)

  def test_return_expression_mismatch(self) -> None:
    @beartype
    def f(x: F32[N]) -> F32[N + 1]:
      return x

    with pytest.raises(BeartypeCallHintReturnViolation):
      f(np.ones(5, dtype=np.float32))

  def test_mul_return(self) -> None:
    @beartype
    def f(x: F32[N, C]) -> F32[N * C]:
      return x.reshape(-1)

    assert f(np.ones((3, 4), dtype=np.float32)).shape == (12,)

  def test_mul_return_mismatch(self) -> None:
    @beartype
    def f(x: F32[N, C]) -> F32[N * C]:
      return np.ones(99, dtype=np.float32)

    with pytest.raises(BeartypeCallHintReturnViolation):
      f(np.ones((3, 4), dtype=np.float32))


# =====================================================================
# Variadic dimensions
# =====================================================================


class TestVariadicDims:
  def test_zero_batch_dims(self) -> None:
    @beartype
    def f(x: F32[~B, C]) -> F32[~B, C]:
      return x

    f(np.ones(3, dtype=np.float32))

  def test_many_batch_dims(self) -> None:
    @beartype
    def f(x: F32[~B, C]) -> F32[~B, C]:
      return x

    f(np.ones((2, 3, 4, 5), dtype=np.float32))

  def test_cross_arg_match(self) -> None:
    @beartype
    def f(x: F32[~B, C], y: F32[~B, C]) -> F32[~B, C]:
      return x + y

    assert f(
      np.ones((2, 3, 4), dtype=np.float32), np.ones((2, 3, 4), dtype=np.float32)
    ).shape == (2, 3, 4)

  def test_cross_arg_shape_mismatch(self) -> None:
    @beartype
    def f(x: F32[~B, C], y: F32[~B, C]) -> F32[~B, C]:
      return x + y

    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones((2, 3, 4), dtype=np.float32), np.ones((5, 3, 4), dtype=np.float32))

  def test_cross_arg_rank_mismatch(self) -> None:
    @beartype
    def f(x: F32[~B, C], y: F32[~B, C]) -> F32[~B, C]:
      return x + y

    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones((2, 3, 4), dtype=np.float32), np.ones(4, dtype=np.float32))

  def test_prefix_and_suffix(self) -> None:
    @beartype
    def f(x: F32[N, ~B, C]) -> F32[N, ~B, C]:
      return x

    f(np.ones((4, 28, 28, 3), dtype=np.float32))
    f(np.ones((4, 3), dtype=np.float32))

  def test_insufficient_rank(self) -> None:
    @beartype
    def f(x: F32[N, ~B, C]) -> F32[N, ~B, C]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones(4, dtype=np.float32))

  def test_anonymous_variadic_no_binding(self) -> None:
    @beartype
    def f(x: F32[~__, C], y: F32[~__, C]) -> F32[~__, C]:
      return x

    f(np.ones((2, 3, 4), dtype=np.float32), np.ones(4, dtype=np.float32))

  def test_anonymous_variadic_return(self) -> None:
    @beartype
    def f(x: F32[~__, C]) -> F32[~__, C]:
      return x

    assert f(np.ones((2, 3, 5), dtype=np.float32)).shape == (2, 3, 5)


# =====================================================================
# Broadcastable dimensions
# =====================================================================


class TestBroadcastableDims:
  def test_size_1_passes(self) -> None:
    @beartype
    def f(x: F32[N, C], y: F32[+N, C]) -> F32[N, C]:
      return x + y

    f(np.ones((4, 3), dtype=np.float32), np.ones((1, 3), dtype=np.float32))

  def test_matching_size(self) -> None:
    @beartype
    def f(x: F32[N, C], y: F32[+N, C]) -> F32[N, C]:
      return x + y

    f(np.ones((4, 3), dtype=np.float32), np.ones((4, 3), dtype=np.float32))

  def test_wrong_size(self) -> None:
    @beartype
    def f(x: F32[N, C], y: F32[+N, C]) -> F32[N, C]:
      return x + y

    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones((4, 3), dtype=np.float32), np.ones((2, 3), dtype=np.float32))

  def test_binds_when_not_1(self) -> None:
    @beartype
    def f(x: F32[+N], y: F32[+N]) -> F32[+N]:
      return x

    f(np.ones(5, dtype=np.float32), np.ones(5, dtype=np.float32))
    f(np.ones(5, dtype=np.float32), np.ones(1, dtype=np.float32))

  def test_both_size_1(self) -> None:
    @beartype
    def f(x: F32[+N], y: F32[+N]) -> F32[+N]:
      return x

    f(np.ones(1, dtype=np.float32), np.ones(1, dtype=np.float32))

  def test_broadcastable_then_strict_fails(self) -> None:
    @beartype
    def f(x: F32[+N, C]) -> F32[N, C]:
      return np.ones((1, 5), dtype=np.float32)

    with pytest.raises(BeartypeCallHintReturnViolation):
      f(np.ones((4, 5), dtype=np.float32))


# =====================================================================
# Anonymous dimensions
# =====================================================================


class TestAnonymousDims:
  def test_no_cross_arg_binding(self) -> None:
    @beartype
    def f(x: F32[__, C], y: F32[__, C]) -> F32[__, C]:
      return x

    f(np.ones((4, 3), dtype=np.float32), np.ones((99, 3), dtype=np.float32))

  def test_still_checks_rank(self) -> None:
    @beartype
    def f(x: F32[__, __]) -> F32[__, __]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones(5, dtype=np.float32))

  def test_mixed_with_named(self) -> None:
    @beartype
    def f(x: F32[__, N, __], y: F32[N]) -> F32[__, N, __]:
      return x

    f(np.ones((2, 4, 5), dtype=np.float32), np.ones(4, dtype=np.float32))
    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones((2, 4, 5), dtype=np.float32), np.ones(7, dtype=np.float32))


# =====================================================================
# Return violations
# =====================================================================


class TestReturnViolations:
  def test_wrong_shape(self) -> None:
    @beartype
    def f(x: F32[N, C]) -> F32[N, C]:
      return np.ones((1, 1), dtype=np.float32)

    with pytest.raises(BeartypeCallHintReturnViolation):
      f(np.ones((4, 3), dtype=np.float32))

  def test_wrong_dtype(self) -> None:
    @beartype
    def f(x: F32[N]) -> F32[N]:
      return x.astype(np.int32)

    with pytest.raises(BeartypeCallHintReturnViolation):
      f(np.ones(5, dtype=np.float32))

  def test_wrong_rank(self) -> None:
    @beartype
    def f(x: F32[N, C]) -> F32[N]:
      return x

    with pytest.raises(BeartypeCallHintReturnViolation):
      f(np.ones((3, 4), dtype=np.float32))

  def test_none_rejected(self) -> None:
    @beartype
    def f(x: F32[N]) -> F32[N]:
      return None  # type: ignore[return-value]

    with pytest.raises(BeartypeCallHintReturnViolation):
      f(np.ones(5, dtype=np.float32))

  def test_fixed_dim_mismatch(self) -> None:
    @beartype
    def f(x: F32[N]) -> F32[3]:
      return x

    with pytest.raises(BeartypeCallHintReturnViolation):
      f(np.ones(5, dtype=np.float32))


# =====================================================================
# Sequential and nested calls
# =====================================================================


class TestSequentialCalls:
  def test_different_shapes(self) -> None:
    @beartype
    def f(x: F32[N]) -> F32[N]:
      return x

    for size in [3, 7, 1, 100, 42]:
      assert f(np.ones(size, dtype=np.float32)).shape == (size,)

  def test_multi_dim_sequential(self) -> None:
    @beartype
    def f(x: F32[N, C], y: F32[N, C]) -> F32[N, C]:
      return x + y

    for n, c in [(2, 3), (10, 20), (1, 1), (50, 7)]:
      assert f(
        np.ones((n, c), dtype=np.float32), np.ones((n, c), dtype=np.float32)
      ).shape == (n, c)

  def test_sequential_independence(self) -> None:
    @beartype
    def f(x: F32[N, C], y: F32[N, C]) -> F32[N, C]:
      return x + y

    f(np.ones((4, 3), dtype=np.float32), np.ones((4, 3), dtype=np.float32))
    f(np.ones((99, 7), dtype=np.float32), np.ones((99, 7), dtype=np.float32))


class TestNestedCalls:
  def test_inner_outer(self) -> None:
    @shapix.check
    @beartype
    def inner(x: F32[N]) -> F32[N]:
      return x * 2

    @shapix.check
    @beartype
    def outer(x: F32[N, C]) -> F32[N]:
      return inner(x[:, 0])

    assert outer(np.ones((4, 3), dtype=np.float32)).shape == (4,)

  def test_deep_nesting(self) -> None:
    @shapix.check
    @beartype
    def a(x: F32[N]) -> F32[N]:
      return x

    @shapix.check
    @beartype
    def b(x: F32[N, C]) -> F32[N]:
      return a(x[:, 0])

    @shapix.check
    @beartype
    def c(x: F32[N, C, H]) -> F32[N]:
      return b(x[:, :, 0])

    assert c(np.ones((4, 3, 5), dtype=np.float32)).shape == (4,)


# =====================================================================
# Custom dimensions
# =====================================================================


class TestCustomDimensions:
  def test_binds_and_enforces(self) -> None:
    Vocab = Dimension("Vocab")
    Embed = Dimension("Embed")

    @beartype
    def f(x: F32[Vocab, Embed], y: F32[Vocab]) -> F32[Vocab, Embed]:
      return x

    f(np.ones((100, 64), dtype=np.float32), np.ones(100, dtype=np.float32))

  def test_mismatch(self) -> None:
    Vocab = Dimension("Vocab")
    Embed = Dimension("Embed")

    @beartype
    def f(x: F32[Vocab, Embed], y: F32[Vocab]) -> F32[Vocab, Embed]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones((100, 64), dtype=np.float32), np.ones(50, dtype=np.float32))

  def test_variadic(self) -> None:
    Batch = Dimension("Batch")

    @beartype
    def f(x: F32[~Batch, N]) -> F32[~Batch, N]:
      return x

    f(np.ones(5, dtype=np.float32))
    f(np.ones((2, 3, 5), dtype=np.float32))

  def test_broadcastable(self) -> None:
    Seq = Dimension("Seq")

    @beartype
    def f(x: F32[N, Seq], y: F32[N, +Seq]) -> F32[N, Seq]:
      return x + y

    f(np.ones((4, 10), dtype=np.float32), np.ones((4, 1), dtype=np.float32))
    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones((4, 10), dtype=np.float32), np.ones((4, 5), dtype=np.float32))

  def test_dimension_wrapped_int_pass(self) -> None:
    THREE = Dimension("3")

    @beartype
    def f(x: F32[N, THREE]) -> F32[N, THREE]:
      return x

    f(np.ones((4, 3), dtype=np.float32))

  def test_dimension_wrapped_int_reject(self) -> None:
    THREE = Dimension("3")

    @beartype
    def f(x: F32[N, THREE]) -> F32[N, THREE]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones((4, 5), dtype=np.float32))


# =====================================================================
# Zero-sized and large arrays
# =====================================================================


class TestZeroSizedArrays:
  def test_zero_first_dim(self) -> None:
    @beartype
    def f(x: F32[N, C]) -> F32[N, C]:
      return x

    assert f(np.ones((0, 3), dtype=np.float32)).shape == (0, 3)

  def test_zero_dim_cross_arg(self) -> None:
    @beartype
    def f(x: F32[N, C], y: F32[N]) -> F32[N, C]:
      return x

    f(np.ones((0, 3), dtype=np.float32), np.ones(0, dtype=np.float32))
    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones((0, 3), dtype=np.float32), np.ones(1, dtype=np.float32))


class TestLargeDimensions:
  def test_large_dim_passes(self) -> None:
    @beartype
    def f(x: F32[N], y: F32[N]) -> F32[N]:
      return x

    big = np.ones(1_000_000, dtype=np.float32)
    f(big, big)

  def test_large_dim_mismatch(self) -> None:
    @beartype
    def f(x: F32[N], y: F32[N]) -> F32[N]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones(1_000_000, dtype=np.float32), np.ones(1_000_001, dtype=np.float32))


# =====================================================================
# Multiple violations
# =====================================================================


class TestMultipleViolations:
  def test_wrong_dtype_and_shape(self) -> None:
    @beartype
    def f(x: F32[3, N]) -> F32[3, N]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones((4, 5), dtype=np.int32))

  def test_both_wrong(self) -> None:
    @beartype
    def f(x: F32[N]) -> F32[N]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones(3, dtype=np.int64))


# =====================================================================
# ArrayLike acceptance
# =====================================================================


class TestArrayLikeAcceptance:
  @pytest.mark.parametrize(
    "value",
    [
      3.14,
      np.float32(2.5),
      [1.0, 2.0, 3.0],
      [[1.0, 2.0], [3.0, 4.0]],
      [[[1.0]]],
      [[[[[[1.0]]]]]],
      (1.0, 2.0, 3.0),
      ((1.0, 2.0), (3.0, 4.0)),
      [],
      np.ones((3, 4), dtype=np.float32),
      np.array(5.0, dtype=np.float32),
    ],
    ids=[
      "scalar",
      "np_scalar",
      "1d_list",
      "2d_list",
      "3d_list",
      "6d_deep",
      "tuple",
      "nested_tuple",
      "empty_list",
      "ndarray",
      "0d_ndarray",
    ],
  )
  def test_f32like_accepted(self, value: object) -> None:
    @beartype
    def f(x: F32Like) -> float:
      return float(np.asarray(x).sum())

    f(value)


class TestArrayLikeRejection:
  @pytest.mark.parametrize("value", [object(), {"a": 1.0}], ids=["object", "dict"])
  def test_f32like_rejected(self, value: object) -> None:
    @beartype
    def f(x: F32Like) -> float:
      return float(np.asarray(x).sum())

    with pytest.raises(BeartypeCallHintParamViolation):
      f(value)  # type: ignore[arg-type]


class TestArrayLikeVariousTypes:
  def test_i64like(self) -> None:
    @beartype
    def f(x: I64Like) -> int:
      return int(np.asarray(x).sum())

    assert f(42) == 42
    assert f([1, 2, 3]) == 6
    assert f([[1, 2], [3, 4]]) == 10

  def test_boollk(self) -> None:
    assert is_bearable(True, BoolLk)
    assert is_bearable([True, False], BoolLk)

  def test_custom_arraylike(self) -> None:
    type MyLike = ArrayLike[float, np.ndarray]

    @beartype
    def f(x: MyLike) -> float:
      return float(np.asarray(x).sum())

    assert f(3.14) == 3.14
    assert f([1.0, 2.0]) == 3.0
    assert f(np.array([1.0, 2.0])) == 3.0
