# pyright: reportArgumentType=false, reportGeneralTypeIssues=false, reportReturnType=false
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
from shapix import B, C, H, N, Dimension, Value, __
from shapix._array_types import _ArrayFactory, make_array_like_type, make_array_type
from shapix._dtypes import COMPLEX256, FLOAT32, FLOAT128, DtypeSpec
from shapix.numpy import (
  Bool,
  BoolLike,
  Bytes,
  DT64,
  F16,
  F32,
  F32Like,
  F32ScalarLike,
  F64,
  F128,
  F128Like,
  Float,
  I8ScalarLike,
  I16ScalarLike,
  I32,
  I64,
  I64ScalarLike,
  I64Like,
  InexactScalarLike,
  Int,
  Num,
  Obj,
  Shaped,
  Str,
  StringLike,
  TD64,
  U8,
  U8ScalarLike,
  U16ScalarLike,
  U64ScalarLike,
  V,
  C256,
  C256Like,
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


class TestExtendedPrecisionDtypes:
  @pytest.mark.skipif(
    np.dtype(np.longdouble).name not in FLOAT128.allowed,
    reason="platform does not expose float128/longdouble distinctly",
  )
  def test_f128(self) -> None:
    @beartype
    def f(x: F128[N]) -> F128[N]:
      return x

    arr = np.ones(3, dtype=np.longdouble)
    assert f(arr).dtype == arr.dtype
    assert is_bearable(arr, F128Like[...])

  @pytest.mark.skipif(
    np.dtype(np.clongdouble).name not in COMPLEX256.allowed,
    reason="platform does not expose complex256/clongdouble distinctly",
  )
  def test_c256(self) -> None:
    @beartype
    def f(x: C256[N]) -> C256[N]:
      return x

    arr = np.ones(3, dtype=np.clongdouble)
    assert f(arr).dtype == arr.dtype
    assert is_bearable(arr, C256Like[...])

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


class TestValueExpressions:
  # ----- Assertions (correct usage) -----

  def test_value_expr_from_argument(self) -> None:
    Size = Value("size")

    @beartype
    def f(size: int, x: F32[Size]) -> F32[Size]:
      return x

    arr = np.ones(4, dtype=np.float32)
    assert f(4, arr).shape == (4,)

  def test_value_expr_from_self_attribute(self) -> None:
    class SomeClass:
      size = 5
      FullSize = Value("self.size + 3")

      @beartype
      def full(self) -> F32[FullSize]:
        return np.ones(self.size + 3, dtype=np.float32)

    assert SomeClass().full().shape == (8,)

  def test_value_expr_can_mix_scope_and_bound_dims(self) -> None:
    Padded = Value("N + pad")

    @beartype
    def f(x: F32[N], pad: int) -> F32[Padded]:
      return np.ones(x.shape[0] + pad, dtype=np.float32)

    assert f(np.ones(4, dtype=np.float32), 2).shape == (6,)

  def test_value_add_dimension_in_annotation(self) -> None:
    """Value + Dimension arithmetic works in type annotations."""

    @beartype
    def f(x: F32[N], pad: int) -> F32[N + Value("pad")]:  # type: ignore[valid-type]
      return np.ones(x.shape[0] + pad, dtype=np.float32)

    assert f(np.ones(4, dtype=np.float32), 2).shape == (6,)

  def test_dimension_add_value_in_annotation(self) -> None:
    """Dimension + Value arithmetic works in type annotations."""

    @beartype
    def f(x: F32[N], pad: int) -> F32[Value("pad") + N]:  # type: ignore[valid-type]
      return np.ones(pad + x.shape[0], dtype=np.float32)

    assert f(np.ones(3, dtype=np.float32), 5).shape == (8,)

  def test_value_mul_dimension_in_annotation(self) -> None:
    """Value * Dimension arithmetic works in type annotations."""

    @beartype
    def f(x: F32[N], factor: int) -> F32[Value("factor") * N]:  # type: ignore[valid-type]
      return np.ones(factor * x.shape[0], dtype=np.float32)

    assert f(np.ones(3, dtype=np.float32), 4).shape == (12,)

  def test_value_add_value_in_annotation(self) -> None:
    """Value + Value arithmetic works in type annotations."""

    @beartype
    def f(a: int, b: int) -> F32[Value("a") + Value("b")]:  # type: ignore[valid-type]
      return np.ones(a + b, dtype=np.float32)

    assert f(3, 5).shape == (8,)

  def test_int_add_value_in_annotation(self) -> None:
    """int + Value arithmetic works in type annotations."""

    @beartype
    def f(x: F32[N]) -> F32[3 + Value("N")]:  # type: ignore[valid-type]
      return np.ones(x.shape[0] + 3, dtype=np.float32)

    assert f(np.ones(4, dtype=np.float32)).shape == (7,)

  def test_chained_value_arithmetic(self) -> None:
    """Chained (N + Value) * int works end-to-end."""

    @beartype
    def f(x: F32[N], pad: int) -> F32[(N + Value("pad")) * 2]:  # type: ignore[valid-type]
      return np.ones((x.shape[0] + pad) * 2, dtype=np.float32)

    assert f(np.ones(4, dtype=np.float32), 1).shape == (10,)

  def test_broadcastable_value(self) -> None:
    """Broadcastable Value(+) accepts size 1."""

    @beartype
    def f(size: int, x: F32[+Value("size")]) -> None:  # type: ignore[valid-type]
      pass

    f(10, np.ones(10, dtype=np.float32))
    f(10, np.ones(1, dtype=np.float32))

  # ----- Violations (incorrect usage → beartype raises) -----

  def test_value_wrong_param_shape_rejected(self) -> None:
    Size = Value("size")

    @beartype
    def f(size: int, x: F32[Size]) -> F32[Size]:
      return x

    arr = np.ones(4, dtype=np.float32)
    with pytest.raises(BeartypeCallHintParamViolation):
      f(5, arr)

  def test_value_wrong_return_shape_rejected(self) -> None:
    @beartype
    def f(size: int) -> F32[Value("size")]:  # type: ignore[valid-type]
      return np.ones(999, dtype=np.float32)

    with pytest.raises(BeartypeCallHintReturnViolation):
      f(4)

  def test_value_add_dim_wrong_return_rejected(self) -> None:
    @beartype
    def f(x: F32[N], pad: int) -> F32[N + Value("pad")]:  # type: ignore[valid-type]
      return np.ones(999, dtype=np.float32)

    with pytest.raises(BeartypeCallHintReturnViolation):
      f(np.ones(4, dtype=np.float32), 2)

  def test_value_mul_dim_wrong_return_rejected(self) -> None:
    @beartype
    def f(x: F32[N], factor: int) -> F32[Value("factor") * N]:  # type: ignore[valid-type]
      return np.ones(999, dtype=np.float32)

    with pytest.raises(BeartypeCallHintReturnViolation):
      f(np.ones(3, dtype=np.float32), 4)

  def test_value_self_attr_wrong_return_rejected(self) -> None:
    class Broken:
      size = 5
      FullSize = Value("self.size + 3")

      @beartype
      def full(self) -> F32[FullSize]:
        return np.ones(999, dtype=np.float32)

    with pytest.raises(BeartypeCallHintReturnViolation):
      Broken().full()

  def test_broadcastable_value_non_1_wrong_size_rejected(self) -> None:
    """Broadcastable Value(+) with non-1 wrong size is rejected."""

    @beartype
    def f(size: int, x: F32[+Value("size")]) -> None:  # type: ignore[valid-type]
      pass

    with pytest.raises(BeartypeCallHintParamViolation):
      f(10, np.ones(5, dtype=np.float32))


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
# Ellipsis as anonymous variadic alias
# =====================================================================


class TestEllipsisDims:
  def test_ellipsis_variadic_middle(self) -> None:
    """F32[N, ..., C] should work like F32[N, ~__, C]."""

    @beartype
    def f(x: F32[N, ..., C]) -> F32[N, ..., C]:
      return x

    f(np.ones((4, 3), dtype=np.float32))  # no middle dims
    f(np.ones((4, 5, 3), dtype=np.float32))  # 1 middle dim
    f(np.ones((4, 5, 6, 3), dtype=np.float32))  # 2 middle dims

  def test_ellipsis_at_start(self) -> None:
    """F32[..., C] — any number of leading dims."""

    @beartype
    def f(x: F32[..., C]) -> F32[..., C]:
      return x

    f(np.ones(3, dtype=np.float32))  # just C
    f(np.ones((4, 3), dtype=np.float32))  # one leading + C
    f(np.ones((2, 4, 3), dtype=np.float32))  # two leading + C

  def test_ellipsis_at_end(self) -> None:
    """F32[N, ...] — any number of trailing dims."""

    @beartype
    def f(x: F32[N, ...]) -> F32[N, ...]:
      return x

    f(np.ones(3, dtype=np.float32))  # just N
    f(np.ones((4, 3), dtype=np.float32))  # N + one trailing
    f(np.ones((4, 3, 5), dtype=np.float32))  # N + two trailing

  def test_ellipsis_matches_zero_dims(self) -> None:
    """F32[N, ..., C] with shape (3, 4) — N=3, C=4, no middle."""

    @beartype
    def f(x: F32[N, ..., C]) -> F32[N, ..., C]:
      return x

    assert f(np.ones((3, 4), dtype=np.float32)).shape == (3, 4)

  def test_ellipsis_no_cross_arg_binding(self) -> None:
    """... (like ~__) doesn't bind, so no cross-arg consistency."""

    @beartype
    def f(x: F32[..., C], y: F32[..., C]) -> F32[..., C]:
      return x

    f(np.ones((2, 3, 4), dtype=np.float32), np.ones(4, dtype=np.float32))

  def test_bare_ellipsis_any_shape(self) -> None:
    """F32[...] matches any shape."""

    @beartype
    def f(x: F32[...]) -> F32[...]:
      return x

    f(np.ones(3, dtype=np.float32))
    f(np.ones((3, 4), dtype=np.float32))
    f(np.ones((2, 3, 4, 5), dtype=np.float32))


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

  def test_broadcastable_fixed_dim_accepts_1(self) -> None:
    D3 = Dimension(3)
    hint = F32[+D3]
    assert is_bearable(np.ones(1, dtype=np.float32), hint)

  def test_broadcastable_fixed_dim_accepts_exact(self) -> None:
    D3 = Dimension(3)
    hint = F32[+D3]
    assert is_bearable(np.ones(3, dtype=np.float32), hint)

  def test_broadcastable_fixed_dim_rejects_other(self) -> None:
    D3 = Dimension(3)
    hint = F32[+D3]
    assert not is_bearable(np.ones(5, dtype=np.float32), hint)

  def test_broadcastable_symbolic_accepts_1(self) -> None:
    hint = F32[N, +(N + 1)]
    arr = np.ones((5, 1), dtype=np.float32)
    assert is_bearable(arr, hint)

  def test_broadcastable_symbolic_accepts_matching(self) -> None:
    hint = F32[N, +(N + 1)]
    arr = np.ones((5, 6), dtype=np.float32)
    assert is_bearable(arr, hint)

  def test_broadcastable_symbolic_rejects_mismatch(self) -> None:
    hint = F32[N, +(N + 1)]
    arr = np.ones((5, 7), dtype=np.float32)
    assert not is_bearable(arr, hint)

  def test_variadic_on_fixed_raises(self) -> None:
    D3 = Dimension(3)
    with pytest.raises(TypeError, match="Cannot apply ~ .* fixed numeric"):
      F32[~D3]  # type: ignore[operator]


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


class TestCustomStructuredDtypes:
  def test_same_structured_dtype_passes(self) -> None:
    dt = np.dtype([("x", np.float32), ("y", np.int32)])
    StructXY = make_array_type(np.ndarray, DtypeSpec.structured(dt))

    @beartype
    def f(x: StructXY[N], y: StructXY[N]) -> StructXY[N]:  # type: ignore[valid-type]
      return x

    f(np.zeros(3, dtype=dt), np.zeros(3, dtype=dt))

  def test_different_structured_dtype_rejected(self) -> None:
    dt_xy = np.dtype([("x", np.float32), ("y", np.int32)])
    dt_xz = np.dtype([("x", np.float32), ("z", np.int32)])
    StructXY = make_array_type(np.ndarray, DtypeSpec.structured(dt_xy))

    @beartype
    def f(x: StructXY[N], y: StructXY[N]) -> StructXY[N]:  # type: ignore[valid-type]
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.zeros(3, dtype=dt_xy), np.zeros(3, dtype=dt_xz))


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
      f(np.ones((4, 5), dtype=np.int32))  # type: ignore

  def test_both_wrong(self) -> None:
    @beartype
    def f(x: F32[N]) -> F32[N]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones(3, dtype=np.int64))  # type: ignore


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
    def f(x: F32Like[...]) -> float:
      return float(np.asarray(x).sum())

    f(value)  # type: ignore


class TestArrayLikeRejection:
  @pytest.mark.parametrize("value", [object(), {"a": 1.0}], ids=["object", "dict"])
  def test_f32like_rejected(self, value: object) -> None:
    @beartype
    def f(x: F32Like[...]) -> float:
      return float(np.asarray(x).sum())

    with pytest.raises(BeartypeCallHintParamViolation):
      f(value)  # type: ignore[arg-type]


class TestArrayLikeVariousTypes:
  def test_i64like(self) -> None:
    @beartype
    def f(x: I64Like[...]) -> int:
      return int(np.asarray(x).sum())

    assert f(42) == 42
    assert f([1, 2, 3]) == 6
    assert f([[1, 2], [3, 4]]) == 10

  def test_boollike(self) -> None:
    assert is_bearable(True, BoolLike[...])
    assert is_bearable([True, False], BoolLike[...])

  def test_custom_arraylike(self) -> None:
    from shapix._dtypes import FLOAT64

    MyLike = make_array_like_type(FLOAT64, name="MyLike")

    @beartype
    def f(x: MyLike[...]) -> float:
      return float(np.asarray(x).sum())

    assert f(3.14) == 3.14
    assert f([1.0, 2.0]) == 3.0
    assert f(np.array([1.0, 2.0])) == 3.0


# =====================================================================
# Multiple variadic rejection
# =====================================================================


class TestMultipleVariadicRejected:
  def test_two_named_variadics(self) -> None:
    with pytest.raises(TypeError, match="At most one variadic"):
      F32[~B, ~N]

  def test_double_ellipsis(self) -> None:
    with pytest.raises(TypeError, match="At most one variadic"):
      F32[..., ...]

  def test_named_variadic_plus_ellipsis(self) -> None:
    with pytest.raises(TypeError, match="At most one variadic"):
      F32[~B, ..., C]

  def test_numpy_rejects_torch(self) -> None:
    torch = __import__("pytest").importorskip("torch")

    @beartype
    def f(x: F32[N]) -> F32[N]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(torch.ones(3, dtype=torch.float32))

  def test_numpy_rejects_jax(self) -> None:
    jax = __import__("pytest").importorskip("jax")
    jnp = jax.numpy

    @beartype
    def f(x: F32[N]) -> F32[N]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(jnp.ones(3, dtype=jnp.float32))

  def test_single_variadic_still_works(self) -> None:
    """Verify single variadics are not rejected."""

    @beartype
    def f(x: F32[~B, C]) -> F32[~B, C]:
      return x

    f(np.ones((2, 3, 4), dtype=np.float32))


# =====================================================================
# ScalarLike boundary tests
# =====================================================================


class TestScalarLikeBoundaries:
  def test_i8_scalar_like_boundaries(self) -> None:
    assert is_bearable(-128, I8ScalarLike)
    assert is_bearable(127, I8ScalarLike)
    assert not is_bearable(-129, I8ScalarLike)
    assert not is_bearable(128, I8ScalarLike)

  def test_u8_scalar_like_boundaries(self) -> None:
    assert is_bearable(0, U8ScalarLike)
    assert is_bearable(255, U8ScalarLike)
    assert not is_bearable(-1, U8ScalarLike)
    assert not is_bearable(256, U8ScalarLike)

  def test_scalar_like_rejects_wrong_type(self) -> None:
    assert not is_bearable("hello", I8ScalarLike)
    assert not is_bearable(3.14, U8ScalarLike)

  def test_scalar_like_numpy_integers(self) -> None:
    assert is_bearable(np.int8(42), I8ScalarLike)
    assert is_bearable(np.uint8(200), U8ScalarLike)

  def test_i64_scalar_like_rejects_out_of_range_numpy_integer(self) -> None:
    assert not is_bearable(np.uint64(2**63), I64ScalarLike)

  def test_inexact_scalar_like_rejects_int(self) -> None:
    assert not is_bearable(1, InexactScalarLike)


# =====================================================================
# ArrayLike shape constraint violations
# =====================================================================


class TestArrayLikeShapeViolations:
  def test_f32like_with_shape_constraint_passes(self) -> None:
    @beartype
    def f(x: F32Like[N, C]) -> float:
      return float(np.asarray(x).sum())

    assert f([[1.0, 2.0], [3.0, 4.0]]) == 10.0

  def test_f32like_wrong_rank_rejected(self) -> None:
    @beartype
    def f(x: F32Like[N, C]) -> float:
      return float(np.asarray(x).sum())

    with pytest.raises(BeartypeCallHintParamViolation):
      f([1.0, 2.0, 3.0])

  def test_f32like_cross_arg_consistency(self) -> None:
    @shapix.check
    @beartype
    def f(x: F32Like[N, C], y: F32Like[N, C]) -> float:
      return 0.0

    f([[1.0, 2.0]], [[3.0, 4.0]])  # N=1, C=2 in both
    with pytest.raises(BeartypeCallHintParamViolation):
      f([[1.0, 2.0]], [[3.0, 4.0, 5.0]])  # C=2 vs C=3

  def test_f32like_fixed_dim_rejected(self) -> None:
    @beartype
    def f(x: F32Like[3]) -> float:
      return float(np.asarray(x).sum())

    f([1.0, 2.0, 3.0])
    with pytest.raises(BeartypeCallHintParamViolation):
      f([1.0, 2.0])


class TestArrayLikeDtypeViolations:
  def test_f32like_rejects_complex_array(self) -> None:
    assert not is_bearable(np.ones(3, dtype=np.complex128), F32Like[...])

  def test_f32like_rejects_string(self) -> None:
    assert not is_bearable("hello", F32Like[...])

  def test_i64like_rejects_complex(self) -> None:
    assert not is_bearable(np.ones(3, dtype=np.complex128), I64Like[...])

  def test_boollike_rejects_float(self) -> None:
    assert not is_bearable(3.14, BoolLike[...])


class TestArrayLikeMemoRestore:
  def test_failed_like_check_does_not_pollute_memo(self) -> None:
    """ArrayLike shape failure should restore memo state."""
    with shapix.check_context():
      # First check binds N=3
      assert is_bearable(np.ones(3, dtype=np.float32), F32Like[N])
      # Second check should fail (wrong dtype), but N=3 should remain
      assert not is_bearable(np.ones(5, dtype=np.complex128), F32Like[N])
      # Third check should still see N=3
      assert is_bearable(np.ones(3, dtype=np.float32), F32Like[N])
      assert not is_bearable(np.ones(4, dtype=np.float32), F32Like[N])


# =====================================================================
# Endianness integration
# =====================================================================


class TestEndiannessIntegration:
  def test_f32le_accepts_le(self) -> None:
    from shapix._dtypes import FLOAT32_LE

    F32LE = make_array_type(np.ndarray, FLOAT32_LE)

    @beartype
    def f(x: F32LE[N]) -> F32LE[N]:  # type: ignore[valid-type]
      return x

    f(np.ones(3, dtype="<f4"))

  def test_f32le_rejects_be(self) -> None:
    from shapix._dtypes import FLOAT32_LE

    F32LE = make_array_type(np.ndarray, FLOAT32_LE)

    @beartype
    def f(x: F32LE[N]) -> F32LE[N]:  # type: ignore[valid-type]
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones(3, dtype=">f4"))

  def test_i64be_accepts_be(self) -> None:
    from shapix._dtypes import INT64_BE

    I64BE = make_array_type(np.ndarray, INT64_BE)

    @beartype
    def f(x: I64BE[N]) -> I64BE[N]:  # type: ignore[valid-type]
      return x

    f(np.ones(3, dtype=">i8"))

  def test_i64be_rejects_le(self) -> None:
    from shapix._dtypes import INT64_BE

    I64BE = make_array_type(np.ndarray, INT64_BE)

    @beartype
    def f(x: I64BE[N]) -> I64BE[N]:  # type: ignore[valid-type]
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones(3, dtype="<i8"))

  def test_native_endianness(self) -> None:
    from shapix._dtypes import FLOAT32_N

    F32N = make_array_type(np.ndarray, FLOAT32_N)

    @beartype
    def f(x: F32N[N]) -> F32N[N]:  # type: ignore[valid-type]
      return x

    f(np.ones(3, dtype=np.float32))

  def test_endianness_cross_arg_consistency(self) -> None:
    from shapix._dtypes import FLOAT32_LE

    F32LE = make_array_type(np.ndarray, FLOAT32_LE)

    @beartype
    def f(x: F32LE[N], y: F32LE[N]) -> F32LE[N]:  # type: ignore[valid-type]
      return x + y

    f(np.ones(3, dtype="<f4"), np.ones(3, dtype="<f4"))
    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones(3, dtype="<f4"), np.ones(5, dtype="<f4"))


# =====================================================================
# Structured dtype integration
# =====================================================================


class TestStructuredDtypeIntegration:
  def test_structured_with_shape_checking(self) -> None:
    from shapix.numpy import Structured

    Point = Structured([("x", np.float32), ("y", np.float32)])

    @beartype
    def f(points: Point[N]) -> Point[N]:  # type: ignore[valid-type]
      return points

    dt = np.dtype([("x", np.float32), ("y", np.float32)])
    f(np.zeros(5, dtype=dt))

  def test_structured_rejects_wrong_fields(self) -> None:
    from shapix.numpy import Structured

    Point = Structured([("x", np.float32), ("y", np.float32)])
    wrong_dt = np.dtype([("a", np.float32), ("b", np.float32)])

    @beartype
    def f(points: Point[N]) -> Point[N]:  # type: ignore[valid-type]
      return points

    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.zeros(5, dtype=wrong_dt))

  def test_structured_cross_arg(self) -> None:
    from shapix.numpy import Structured

    Point = Structured([("x", np.float32), ("y", np.float32)])
    dt = np.dtype([("x", np.float32), ("y", np.float32)])

    @beartype
    def f(a: Point[N], b: Point[N]) -> Point[N]:  # type: ignore[valid-type]
      return a

    f(np.zeros(3, dtype=dt), np.zeros(3, dtype=dt))
    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.zeros(3, dtype=dt), np.zeros(5, dtype=dt))


# =====================================================================
# Like casting variants (programmatic via make_array_like_type)
# =====================================================================


class TestLikeCastingVariants:
  """Test casting × dtype combinations for Like types."""

  @pytest.mark.parametrize(
    "casting,accept_extra,reject_extra",
    [
      ("no", np.float32, np.float64),
      ("equiv", np.float32, np.int32),
      ("safe", np.float32, np.complex128),
      ("same_kind", np.float32, np.complex128),
      ("unsafe", np.float32, None),
    ],
  )
  def test_f32like_casting_levels(
    self,
    casting: str,
    accept_extra: np.dtype[np.generic],
    reject_extra: np.dtype[np.generic] | None,
  ) -> None:
    T = make_array_like_type(FLOAT32, casting=casting, name=f"F32Like_{casting}")
    arr = np.ones(3, dtype=accept_extra)
    assert is_bearable(arr, T[...])
    if reject_extra is not None:
      bad = np.ones(3, dtype=reject_extra)
      assert not is_bearable(bad, T[...])

  @pytest.mark.parametrize(
    "dtype_spec,accept_dtype,reject_dtype",
    [
      (FLOAT32, np.float32, np.complex128),
      (DtypeSpec("Int8", frozenset({"int8"})), np.int8, np.float32),
    ],
  )
  def test_like_casting_no_exact_match(
    self,
    dtype_spec: DtypeSpec,
    accept_dtype: np.dtype[np.generic],
    reject_dtype: np.dtype[np.generic],
  ) -> None:
    """casting='no' only accepts exact dtype."""
    T = make_array_like_type(dtype_spec, casting="no")
    assert is_bearable(np.ones(3, dtype=accept_dtype), T[...])
    assert not is_bearable(np.ones(3, dtype=reject_dtype), T[...])

  def test_like_safe_allows_int_to_float(self) -> None:
    """int32 → float64 is safe."""
    from shapix._dtypes import FLOAT64

    T = make_array_like_type(FLOAT64, casting="safe")
    assert is_bearable(np.ones(3, dtype=np.int32), T[...])

  def test_like_safe_rejects_float_to_int(self) -> None:
    """float32 → int32 is NOT safe."""
    from shapix._dtypes import INT32

    T = make_array_like_type(INT32, casting="safe")
    assert not is_bearable(np.ones(3, dtype=np.float32), T[...])

  def test_like_same_kind_int_to_float(self) -> None:
    """int32 → float32 is same_kind (both numeric, int→float OK)."""
    T = make_array_like_type(FLOAT32, casting="same_kind")
    assert is_bearable(np.ones(3, dtype=np.int32), T[...])

  def test_like_same_kind_complex_rejected(self) -> None:
    """complex → float is NOT same_kind."""
    T = make_array_like_type(FLOAT32, casting="same_kind")
    assert not is_bearable(np.ones(3, dtype=np.complex128), T[...])

  def test_like_unsafe_accepts_complex_to_float(self) -> None:
    """complex → float IS unsafe-castable."""
    T = make_array_like_type(FLOAT32, casting="unsafe")
    assert is_bearable(np.ones(3, dtype=np.complex128), T[...])

  def test_like_empty_array(self) -> None:
    T = make_array_like_type(FLOAT32, casting="same_kind")
    assert is_bearable(np.ones(0, dtype=np.float32), T[...])

  def test_like_0d_array(self) -> None:
    T = make_array_like_type(FLOAT32, casting="same_kind")
    assert is_bearable(np.array(1.0, dtype=np.float32), T[...])

  def test_like_scalar(self) -> None:
    T = make_array_like_type(FLOAT32, casting="same_kind")
    assert is_bearable(1.0, T[...])

  def test_like_nested_list(self) -> None:
    T = make_array_like_type(FLOAT32, casting="same_kind")
    assert is_bearable([[1.0, 2.0], [3.0, 4.0]], T[...])


# =====================================================================
# Endianness array types (programmatic via make_array_type)
# =====================================================================


class TestEndiannessArrayTypes:
  """Test endianness × dtype combinations via make_array_type."""

  @pytest.mark.parametrize(
    "endian_spec_name,byte_char,format_str",
    [
      ("FLOAT32_LE", "<", "f4"),
      ("FLOAT32_BE", ">", "f4"),
      ("FLOAT32_N", "=", "f4"),
      ("INT16_LE", "<", "i2"),
      ("INT16_BE", ">", "i2"),
      ("INT32_LE", "<", "i4"),
      ("INT32_BE", ">", "i4"),
      ("INT64_LE", "<", "i8"),
      ("INT64_BE", ">", "i8"),
      ("UINT16_LE", "<", "u2"),
      ("UINT32_LE", "<", "u4"),
      ("UINT64_LE", "<", "u8"),
      ("FLOAT16_LE", "<", "f2"),
      ("FLOAT64_LE", "<", "f8"),
      ("COMPLEX64_LE", "<", "c8"),
      ("COMPLEX128_LE", "<", "c16"),
    ],
  )
  def test_endianness_array_accept(
    self, endian_spec_name: str, byte_char: str, format_str: str
  ) -> None:
    import shapix._dtypes as dtypes_mod

    spec = getattr(dtypes_mod, endian_spec_name)
    T = make_array_type(np.ndarray, spec)
    arr = np.ones(3, dtype=np.dtype(f"{byte_char}{format_str}"))
    assert is_bearable(arr, T[...])

  @pytest.mark.parametrize(
    "endian_spec_name,wrong_char,format_str",
    [
      ("FLOAT32_LE", ">", "f4"),
      ("FLOAT32_BE", "<", "f4"),
      ("INT32_LE", ">", "i4"),
      ("INT64_BE", "<", "i8"),
    ],
  )
  def test_endianness_array_reject(
    self, endian_spec_name: str, wrong_char: str, format_str: str
  ) -> None:
    import shapix._dtypes as dtypes_mod

    spec = getattr(dtypes_mod, endian_spec_name)
    T = make_array_type(np.ndarray, spec)
    arr = np.ones(3, dtype=np.dtype(f"{wrong_char}{format_str}"))
    assert not is_bearable(arr, T[...])

  def test_category_endianness(self) -> None:
    """Category specs with endianness accept matching types."""
    from shapix._dtypes import FLOAT_LE

    T = make_array_type(np.ndarray, FLOAT_LE)
    assert is_bearable(np.ones(3, dtype="<f4"), T[...])
    assert is_bearable(np.ones(3, dtype="<f8"), T[...])
    assert not is_bearable(np.ones(3, dtype=">f4"), T[...])

  def test_shaped_endianness(self) -> None:
    """SHAPED_LE accepts any dtype but only little-endian."""
    from shapix._dtypes import SHAPED_LE

    T = make_array_type(np.ndarray, SHAPED_LE)
    assert is_bearable(np.ones(3, dtype="<f4"), T[...])
    assert is_bearable(np.ones(3, dtype="<i4"), T[...])
    assert not is_bearable(np.ones(3, dtype=">f4"), T[...])


# =====================================================================
# Endianness Like types (programmatic)
# =====================================================================


class TestEndiannessLikeTypes:
  """Test endianness Like types via make_array_like_type."""

  @pytest.mark.parametrize("casting", ["no", "equiv", "safe", "same_kind", "unsafe"])
  def test_f32le_like_accept_reject(self, casting: str) -> None:
    from shapix._dtypes import FLOAT32_LE

    T = make_array_like_type(FLOAT32_LE, casting=casting)
    # Correct endianness accepted
    assert is_bearable(np.ones(3, dtype="<f4"), T[...])
    # Wrong endianness rejected
    assert not is_bearable(np.ones(3, dtype=">f4"), T[...])

  def test_endianness_like_same_kind_int_to_float_le(self) -> None:
    """int32 LE → float32 LE should be accepted under same_kind."""
    from shapix._dtypes import FLOAT32_LE

    T = make_array_like_type(FLOAT32_LE, casting="same_kind")
    # int32 LE array should cast to float32 by same_kind, but endianness check
    # is on the source array, not the target — source is LE int32
    arr = np.ones(3, dtype="<i4")
    assert is_bearable(arr, T[...])

  def test_endianness_like_rejects_wrong_endian_scalar(self) -> None:
    """Scalar (no endianness) should pass byteorder check."""
    from shapix._dtypes import FLOAT32_LE

    T = make_array_like_type(FLOAT32_LE, casting="same_kind")
    # Scalars have no byte order, so _check_byteorder returns True
    assert is_bearable(1.0, T[...])


# =====================================================================
# ScalarLike casting variants (programmatic)
# =====================================================================


class TestScalarLikeCastingVariants:
  """Test make_scalar_like_type with various casting rules."""

  def test_no_casting_strict(self) -> None:
    from shapix.numpy import make_scalar_like_type

    T = make_scalar_like_type(np.float32, casting="no")
    assert is_bearable(np.float32(1.0), T)
    assert not is_bearable(1.0, T)  # Python float != np.float32

  def test_safe_casting(self) -> None:
    from shapix.numpy import make_scalar_like_type

    T = make_scalar_like_type(np.float32, casting="safe")
    assert is_bearable(np.float16(1.0), T)  # float16 → float32 is safe
    assert not is_bearable(np.complex64(1.0), T)  # complex → float not safe

  def test_same_kind_casting(self) -> None:
    from shapix.numpy import make_scalar_like_type

    T = make_scalar_like_type(np.float32, casting="same_kind")
    assert is_bearable(1.0, T)  # Python float → float32 is same_kind
    assert not is_bearable(1 + 0j, T)  # complex → float not same_kind

  def test_unsafe_casting(self) -> None:
    from shapix.numpy import make_scalar_like_type

    T = make_scalar_like_type(np.float32, casting="unsafe")
    assert is_bearable(1 + 0j, T)  # complex → float is unsafe-castable
    assert not is_bearable("hello", T)  # strings not castable

  def test_int8_no_casting(self) -> None:
    from shapix.numpy import make_scalar_like_type

    T = make_scalar_like_type(np.int8, casting="no")
    assert is_bearable(np.int8(1), T)
    assert not is_bearable(1, T)  # Python int != np.int8

  def test_int8_same_kind_casting(self) -> None:
    from shapix.numpy import make_scalar_like_type

    T = make_scalar_like_type(np.int8, casting="same_kind")
    assert is_bearable(1, T)  # Python int → int8 same_kind
    assert not is_bearable(1.0, T)  # float → int not same_kind

  def test_int8_unsafe_casting(self) -> None:
    from shapix.numpy import make_scalar_like_type

    T = make_scalar_like_type(np.int8, casting="unsafe")
    assert is_bearable(1.0, T)  # float → int8 is unsafe-castable
    assert not is_bearable("hello", T)

  @pytest.mark.parametrize(
    "target,accept,reject",
    [
      (np.float64, np.float32(1.0), np.complex128(1.0)),
      (np.int32, np.int16(1), np.float32(1.0)),
      (np.uint8, np.uint8(1), np.int8(-1)),
    ],
  )
  def test_safe_casting_matrix(
    self, target: type[np.generic], accept: object, reject: object
  ) -> None:
    from shapix.numpy import make_scalar_like_type

    T = make_scalar_like_type(target, casting="safe")
    assert is_bearable(accept, T)
    assert not is_bearable(reject, T)

  def test_string_target(self) -> None:
    """Target can be a string like 'float32'."""
    from shapix.numpy import make_scalar_like_type

    T = make_scalar_like_type("float32", casting="same_kind")
    assert is_bearable(1.0, T)

  def test_dtype_target(self) -> None:
    """Target can be a np.dtype object."""
    from shapix.numpy import make_scalar_like_type

    T = make_scalar_like_type(np.dtype(np.float32), casting="same_kind")
    assert is_bearable(1.0, T)

  def test_rejects_non_scalar_base(self) -> None:
    """Non-scalar types like list should be rejected."""
    from shapix.numpy import make_scalar_like_type

    T = make_scalar_like_type(np.float32, casting="unsafe")
    assert not is_bearable([1.0, 2.0], T)
    assert not is_bearable(np.ones(3), T)

  def test_make_scalar_like_type_not_reexported_from_init(self) -> None:
    assert not hasattr(shapix, "make_scalar_like_type")


class TestNumericScalarBooleanRejection:
  """Numeric scalar aliases must reject booleans."""

  @pytest.mark.parametrize(
    "alias_name",
    [
      "I8ScalarLike",
      "I16ScalarLike",
      "I32ScalarLike",
      "I64ScalarLike",
      "U8ScalarLike",
      "U16ScalarLike",
      "U32ScalarLike",
      "U64ScalarLike",
      "F16ScalarLike",
      "F32ScalarLike",
      "F64ScalarLike",
      "F128ScalarLike",
      "C64ScalarLike",
      "C128ScalarLike",
      "C256ScalarLike",
      "IntScalarLike",
      "UIntScalarLike",
      "IntegerScalarLike",
      "FloatScalarLike",
      "RealScalarLike",
      "ComplexScalarLike",
      "InexactScalarLike",
      "NumScalarLike",
    ],
  )
  def test_numeric_alias_rejects_bool(self, alias_name: str) -> None:
    import shapix.numpy as snp

    alias = getattr(snp, alias_name)
    assert not is_bearable(True, alias), f"{alias_name} should reject True"
    assert not is_bearable(False, alias), f"{alias_name} should reject False"
    assert not is_bearable(np.bool_(True), alias), (
      f"{alias_name} should reject np.bool_(True)"
    )

  def test_bool_scalar_like_accepts_booleans(self) -> None:
    from shapix.numpy import BoolScalarLike

    assert is_bearable(True, BoolScalarLike)
    assert is_bearable(False, BoolScalarLike)

  def test_shaped_scalar_like_accepts_booleans(self) -> None:
    from shapix.numpy import ShapedScalarLike

    assert is_bearable(True, ShapedScalarLike)
    assert is_bearable(False, ShapedScalarLike)

  def test_make_scalar_like_type_rejects_bool_for_numeric(self) -> None:
    from shapix.numpy import make_scalar_like_type

    T = make_scalar_like_type(np.uint8)
    assert not is_bearable(True, T)

    T2 = make_scalar_like_type(np.float32)
    assert not is_bearable(True, T2)

  def test_make_scalar_like_type_accepts_bool_for_bool_target(self) -> None:
    from shapix.numpy import make_scalar_like_type

    T = make_scalar_like_type(np.bool_)
    assert is_bearable(True, T)


# =====================================================================
# DT64 / TD64 runtime integration
# =====================================================================


class TestDatetimeTimedelta:
  """Full @beartype pipeline for DT64 and TD64 array types."""

  def test_dt64_accepts_datetime64_ns(self) -> None:
    arr = np.array(["2021-01-01", "2021-06-15"], dtype="datetime64[ns]")
    assert is_bearable(arr, DT64[N])

  def test_dt64_accepts_datetime64_D(self) -> None:
    arr = np.array(["2021-01-01", "2021-06-15"], dtype="datetime64[D]")
    assert is_bearable(arr, DT64[N])

  def test_dt64_rejects_float32(self) -> None:
    arr = np.ones(3, dtype=np.float32)
    assert not is_bearable(arr, DT64[N])

  def test_dt64_rejects_int64(self) -> None:
    arr = np.ones(3, dtype=np.int64)
    assert not is_bearable(arr, DT64[N])

  def test_td64_accepts_timedelta64_ms(self) -> None:
    arr = np.array([1, 2, 3], dtype="timedelta64[ms]")
    assert is_bearable(arr, TD64[N])

  def test_td64_accepts_timedelta64_s(self) -> None:
    arr = np.array([10, 20], dtype="timedelta64[s]")
    assert is_bearable(arr, TD64[N])

  def test_td64_rejects_float32(self) -> None:
    arr = np.ones(3, dtype=np.float32)
    assert not is_bearable(arr, TD64[N])

  def test_td64_rejects_int64(self) -> None:
    arr = np.ones(3, dtype=np.int64)
    assert not is_bearable(arr, TD64[N])

  def test_dt64_cross_arg_consistency(self) -> None:
    """N binds correctly across two DT64 args."""

    @beartype
    def f(x: DT64[N], y: DT64[N]) -> None:
      pass

    a = np.array(["2021-01-01", "2021-06-15"], dtype="datetime64[ns]")
    b = np.array(["2022-03-01", "2022-12-31"], dtype="datetime64[D]")
    f(a, b)  # same N=2

  def test_dt64_cross_arg_mismatch_raises(self) -> None:
    @beartype
    def f(x: DT64[N], y: DT64[N]) -> None:
      pass

    a = np.array(["2021-01-01", "2021-06-15"], dtype="datetime64[ns]")
    b = np.array(["2022-03-01"], dtype="datetime64[D]")
    with pytest.raises(BeartypeCallHintParamViolation):
      f(a, b)  # N=2 vs N=1

  def test_td64_shape_checking(self) -> None:
    @beartype
    def f(x: TD64[N, C]) -> None:
      pass

    f(np.ones((3, 4), dtype="timedelta64[ms]"))
    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones(3, dtype="timedelta64[ms]"))


# =====================================================================
# V / Str / Bytes / Obj runtime integration
# =====================================================================


class TestSpecialDtypes:
  """Full @beartype pipeline for V, Str, Bytes, and Obj array types."""

  def test_void_accepts_void_array(self) -> None:
    arr = np.zeros(3, dtype="V8")
    assert is_bearable(arr, V[N])

  def test_void_rejects_float(self) -> None:
    assert not is_bearable(np.ones(3, dtype=np.float32), V[N])

  def test_str_accepts_str_array(self) -> None:
    arr = np.array(["a", "b", "c"], dtype=np.str_)
    assert is_bearable(arr, Str[N])

  def test_str_rejects_float(self) -> None:
    assert not is_bearable(np.ones(3, dtype=np.float32), Str[N])

  def test_bytes_accepts_bytes_array(self) -> None:
    arr = np.array([b"a", b"b"], dtype=np.bytes_)
    assert is_bearable(arr, Bytes[N])

  def test_bytes_rejects_float(self) -> None:
    assert not is_bearable(np.ones(3, dtype=np.float32), Bytes[N])

  def test_obj_accepts_object_array(self) -> None:
    arr = np.array([object(), object()], dtype=object)
    assert is_bearable(arr, Obj[N])

  def test_obj_rejects_float(self) -> None:
    assert not is_bearable(np.ones(3, dtype=np.float32), Obj[N])

  def test_str_shape_checking(self) -> None:
    @beartype
    def f(x: Str[N, C]) -> None:
      pass

    f(np.array([["a", "b"], ["c", "d"]], dtype=np.str_))
    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.array(["a", "b"], dtype=np.str_))


# =====================================================================
# Endianness @beartype integration
# =====================================================================


class TestEndiannessBeartype:
  """Endianness variants through the full @beartype decorator pipeline."""

  def test_f32le_accepts_little_endian(self) -> None:
    from shapix._dtypes import FLOAT32_LE

    T = make_array_type(np.ndarray, FLOAT32_LE)

    @beartype
    def f(x: T[N]) -> None:  # type: ignore[valid-type]
      pass

    f(np.ones(3, dtype="<f4"))

  def test_f32le_rejects_big_endian(self) -> None:
    from shapix._dtypes import FLOAT32_LE

    T = make_array_type(np.ndarray, FLOAT32_LE)

    @beartype
    def f(x: T[N]) -> None:  # type: ignore[valid-type]
      pass

    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones(3, dtype=">f4"))

  def test_f32be_accepts_big_endian(self) -> None:
    from shapix._dtypes import FLOAT32_BE

    T = make_array_type(np.ndarray, FLOAT32_BE)

    @beartype
    def f(x: T[N]) -> None:  # type: ignore[valid-type]
      pass

    f(np.ones(3, dtype=">f4"))

  def test_f32be_rejects_little_endian(self) -> None:
    from shapix._dtypes import FLOAT32_BE

    T = make_array_type(np.ndarray, FLOAT32_BE)

    @beartype
    def f(x: T[N]) -> None:  # type: ignore[valid-type]
      pass

    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones(3, dtype="<f4"))

  def test_f32n_accepts_native_endian(self) -> None:
    from shapix._dtypes import FLOAT32_N

    T = make_array_type(np.ndarray, FLOAT32_N)

    @beartype
    def f(x: T[N]) -> None:  # type: ignore[valid-type]
      pass

    f(np.ones(3, dtype=np.float32))

  def test_endianness_cross_arg(self) -> None:
    """N binds correctly across endian-constrained args."""
    from shapix._dtypes import FLOAT32_LE

    T = make_array_type(np.ndarray, FLOAT32_LE)

    @beartype
    def f(x: T[N], y: T[N]) -> None:  # type: ignore[valid-type]
      pass

    f(np.ones(5, dtype="<f4"), np.ones(5, dtype="<f4"))
    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones(5, dtype="<f4"), np.ones(3, dtype="<f4"))


# =====================================================================
# ScalarLike extended boundary tests
# =====================================================================


class TestScalarLikeBoundariesExtended:
  """Boundary-value tests for ScalarLike types beyond I8/U8."""

  def test_i16_scalar_like_boundaries(self) -> None:
    assert is_bearable(-32768, I16ScalarLike)
    assert is_bearable(32767, I16ScalarLike)
    assert not is_bearable(-32769, I16ScalarLike)
    assert not is_bearable(32768, I16ScalarLike)

  def test_i64_scalar_like_boundaries(self) -> None:
    min_val = np.iinfo(np.int64).min
    max_val = np.iinfo(np.int64).max
    assert is_bearable(min_val, I64ScalarLike)
    assert is_bearable(max_val, I64ScalarLike)
    assert not is_bearable(min_val - 1, I64ScalarLike)
    assert not is_bearable(max_val + 1, I64ScalarLike)

  def test_u16_scalar_like_boundaries(self) -> None:
    assert is_bearable(0, U16ScalarLike)
    assert is_bearable(65535, U16ScalarLike)
    assert not is_bearable(-1, U16ScalarLike)
    assert not is_bearable(65536, U16ScalarLike)

  def test_u64_scalar_like_boundaries(self) -> None:
    max_val = np.iinfo(np.uint64).max
    assert is_bearable(0, U64ScalarLike)
    assert is_bearable(max_val, U64ScalarLike)
    assert not is_bearable(-1, U64ScalarLike)
    assert not is_bearable(max_val + 1, U64ScalarLike)

  def test_f32_scalar_like_boundaries(self) -> None:
    min_val = float(np.finfo(np.float32).min)
    max_val = float(np.finfo(np.float32).max)
    assert is_bearable(min_val, F32ScalarLike)
    assert is_bearable(max_val, F32ScalarLike)
    assert is_bearable(0.0, F32ScalarLike)

  def test_i16_numpy_scalar(self) -> None:
    assert is_bearable(np.int16(100), I16ScalarLike)
    assert not is_bearable(np.float32(1.0), I16ScalarLike)

  def test_u64_numpy_scalar(self) -> None:
    assert is_bearable(np.uint64(42), U64ScalarLike)
    assert not is_bearable(np.float64(1.0), U64ScalarLike)


# =====================================================================
# StringLike smoke tests
# =====================================================================


class TestStringLike:
  def test_accepts_python_str(self) -> None:
    assert is_bearable("hello", StringLike)

  def test_accepts_numpy_str(self) -> None:
    assert is_bearable(np.str_("hi"), StringLike)

  def test_rejects_int(self) -> None:
    assert not is_bearable(42, StringLike)

  def test_rejects_bytes(self) -> None:
    assert not is_bearable(b"hello", StringLike)
