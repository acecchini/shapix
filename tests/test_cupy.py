# pyright: reportArgumentType=false, reportGeneralTypeIssues=false, reportUnusedImport=false
"""Tests for shapix.cupy — CuPy ndarray types, Like types, and dtype handling."""

from __future__ import annotations

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

from beartype import beartype
from beartype.door import is_bearable
from beartype.roar import (
  BeartypeCallHintParamViolation,
  BeartypeCallHintReturnViolation,
)

import shapix
from shapix import B, C, Dimension, N, Value, __
from shapix.cupy import (
  F16,
  F32,
  F64,
  I32,
  I64,
  U8,
  Bool,
  BoolLike,
  F32Like,
  Float,
  I64Like,
  Int,
  Num,
  Shaped,
)

# =====================================================================
# Dtype acceptance / rejection
# =====================================================================


class TestCuPyDtypeAcceptance:
  @pytest.mark.parametrize(
    "ArrayType, good_dtype, bad_dtypes",
    [
      (F32, cp.float32, [cp.float64, cp.float16, cp.int32, cp.bool_]),
      (F64, cp.float64, [cp.float32]),
      (F16, cp.float16, [cp.float32]),
      (I32, cp.int32, [cp.float32, cp.int64]),
      (I64, cp.int64, [cp.float32, cp.uint8]),
      (U8, cp.uint8, [cp.int8, cp.int32]),
      (Bool, cp.bool_, [cp.int32, cp.float32]),
    ],
  )
  def test_accepts_correct_rejects_wrong(
    self, ArrayType: object, good_dtype: object, bad_dtypes: list[object]
  ) -> None:
    @beartype
    def f(x: ArrayType[N]) -> ArrayType[N]:  # type: ignore[valid-type]
      return x

    f(cp.ones(3, dtype=good_dtype))
    for bad in bad_dtypes:
      with pytest.raises(BeartypeCallHintParamViolation):
        f(cp.ones(3, dtype=bad))


class TestCuPyCategoryTypes:
  def test_float_accepts_f32_f64_f16(self) -> None:
    @beartype
    def f(x: Float[N]) -> Float[N]:
      return x

    for dtype in [cp.float32, cp.float64, cp.float16]:
      f(cp.ones(3, dtype=dtype))

  def test_float_rejects_int(self) -> None:
    @beartype
    def f(x: Float[N]) -> Float[N]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(cp.ones(3, dtype=cp.int32))

  def test_int_rejects_unsigned(self) -> None:
    @beartype
    def f(x: Int[N]) -> Int[N]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(cp.ones(3, dtype=cp.uint8))

  def test_shaped_accepts_all(self) -> None:
    @beartype
    def f(x: Shaped[N]) -> Shaped[N]:
      return x

    for dtype in [cp.float32, cp.int32, cp.uint8, cp.bool_]:
      f(cp.ones(3, dtype=dtype))

  def test_num_rejects_bool(self) -> None:
    @beartype
    def f(x: Num[N]) -> Num[N]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(cp.ones(3, dtype=cp.bool_))


# =====================================================================
# Shape checking
# =====================================================================


class TestCuPyShapeChecking:
  def test_basic_2d(self) -> None:
    @beartype
    def f(x: F32[N, C]) -> F32[N, C]:
      return x

    assert f(cp.ones((4, 3), dtype=cp.float32)).shape == (4, 3)

  def test_wrong_rank(self) -> None:
    @beartype
    def f(x: F32[N, C]) -> F32[N, C]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(cp.ones(5, dtype=cp.float32))

  def test_cross_arg_consistency(self) -> None:
    @beartype
    def f(x: F32[N, C], y: F32[N, C]) -> F32[N, C]:
      return x + y

    assert f(
      cp.ones((4, 3), dtype=cp.float32), cp.ones((4, 3), dtype=cp.float32)
    ).shape == (4, 3)

  def test_cross_arg_mismatch(self) -> None:
    @beartype
    def f(x: F32[N, C], y: F32[N, C]) -> F32[N, C]:
      return x + y

    with pytest.raises(BeartypeCallHintParamViolation):
      f(cp.ones((4, 3), dtype=cp.float32), cp.ones((5, 3), dtype=cp.float32))

  def test_fixed_dim(self) -> None:
    @beartype
    def f(x: F32[3, N]) -> F32[3, N]:
      return x

    f(cp.ones((3, 5), dtype=cp.float32))
    with pytest.raises(BeartypeCallHintParamViolation):
      f(cp.ones((4, 5), dtype=cp.float32))

  def test_symbolic_dim(self) -> None:
    @beartype
    def f(x: F32[N], y: F32[N + 1]) -> F32[N]:
      return x

    f(cp.ones(5, dtype=cp.float32), cp.ones(6, dtype=cp.float32))
    with pytest.raises(BeartypeCallHintParamViolation):
      f(cp.ones(5, dtype=cp.float32), cp.ones(5, dtype=cp.float32))

  def test_variadic(self) -> None:
    @beartype
    def f(x: F32[~B, C]) -> F32[~B, C]:
      return x

    f(cp.ones(3, dtype=cp.float32))
    f(cp.ones((2, 3, 4), dtype=cp.float32))

  def test_anonymous_variadic(self) -> None:
    @beartype
    def f(x: F32[~__, C]) -> F32[~__, C]:
      return x

    f(cp.ones(3, dtype=cp.float32))
    f(cp.ones((2, 3, 4), dtype=cp.float32))

  def test_broadcastable(self) -> None:
    @beartype
    def f(x: F32[N, C], y: F32[+N, C]) -> F32[N, C]:
      return x + y

    f(cp.ones((4, 3), dtype=cp.float32), cp.ones((1, 3), dtype=cp.float32))
    f(cp.ones((4, 3), dtype=cp.float32), cp.ones((4, 3), dtype=cp.float32))
    with pytest.raises(BeartypeCallHintParamViolation):
      f(cp.ones((4, 3), dtype=cp.float32), cp.ones((2, 3), dtype=cp.float32))

  def test_return_shape_violation(self) -> None:
    @beartype
    def f(x: F32[N, C]) -> F32[N, C]:
      return cp.ones((1, 1), dtype=cp.float32)

    with pytest.raises(BeartypeCallHintReturnViolation):
      f(cp.ones((4, 3), dtype=cp.float32))

  def test_return_dtype_violation(self) -> None:
    @beartype
    def f(x: F32[N]) -> F32[N]:
      return x.astype(cp.float64)

    with pytest.raises(BeartypeCallHintReturnViolation):
      f(cp.ones(3, dtype=cp.float32))

  def test_sequential_calls_independent(self) -> None:
    @beartype
    def f(x: F32[N]) -> F32[N]:
      return x

    f(cp.ones(3, dtype=cp.float32))
    f(cp.ones(99, dtype=cp.float32))

  def test_anonymous_dim(self) -> None:
    @beartype
    def f(x: F32[__, C], y: F32[__, C]) -> F32[__, C]:
      return x

    f(cp.ones((4, 3), dtype=cp.float32), cp.ones((99, 3), dtype=cp.float32))


# =====================================================================
# Cross-backend rejection
# =====================================================================


class TestCuPyCrossBackendRejection:
  def test_cupy_type_rejects_numpy(self) -> None:
    @beartype
    def f(x: F32[N]) -> F32[N]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones(3, dtype=np.float32))

  def test_numpy_type_rejects_cupy(self) -> None:
    from shapix.numpy import F32 as NpF32

    @beartype
    def f(x: NpF32[N]) -> NpF32[N]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(cp.ones(3, dtype=cp.float32))


# =====================================================================
# Like types
# =====================================================================


class TestCuPyLikeTypes:
  @pytest.mark.parametrize(
    "value",
    [1.5, [1.0, 2.0], [[1.0, 2.0], [3.0, 4.0]], (1.0, 2.0)],
    ids=["scalar", "list", "nested_list", "tuple"],
  )
  def test_f32like_accepts_scalars_and_sequences(self, value: object) -> None:
    assert is_bearable(value, F32Like[...])

  def test_f32like_accepts_numpy(self) -> None:
    assert is_bearable(np.ones(3, dtype=np.float32), F32Like[...])

  def test_f32like_accepts_cupy(self) -> None:
    assert is_bearable(cp.ones(3, dtype=cp.float32), F32Like[...])

  def test_f32like_rejects_non_numeric(self) -> None:
    assert not is_bearable(object(), F32Like[...])

  def test_i64like(self) -> None:
    assert is_bearable(42, I64Like[...])
    assert is_bearable([1, 2, 3], I64Like[...])
    assert is_bearable(cp.ones(3, dtype=cp.int64), I64Like[...])

  def test_boollike(self) -> None:
    assert is_bearable(True, BoolLike[...])
    assert is_bearable([True, False], BoolLike[...])

  def test_spoofed_shape_dtype_rejected(self) -> None:
    """Object with .shape/.dtype but not a real array must be rejected."""

    class SpoofedArray:
      def __init__(self) -> None:
        self.shape = (3,)
        self.dtype = np.dtype(np.float32)

      def __array__(self, *_a: object, **_kw: object) -> None:  # noqa: PLW3201
        msg = "not convertible"
        raise TypeError(msg)

    assert not is_bearable(SpoofedArray(), F32Like[...])


class TestCuPyLikeDiagnostics:
  def test_f32like_runtime_hint_module_is_backend_correct(self) -> None:
    assert F32Like[N].__module__ == "shapix.cupy"

  def test_f32like_violation_uses_cupy_backend_label(self) -> None:
    @beartype
    def f(x: F32Like[N]) -> None:  # type: ignore[valid-type]
      pass

    with pytest.raises(BeartypeCallHintParamViolation) as exc_info:
      f(cp.ones((2, 2), dtype=cp.float32))

    text = str(exc_info.value)
    assert "<class 'shapix.cupy.F32Like[N]'>" in text
    assert "numpy.F32Like[N]" not in text


class TestCuPyLikeEdgeCases:
  def test_deep_nesting(self) -> None:
    assert is_bearable([[[[[[1.0]]]]]], F32Like[...])

  def test_empty_list(self) -> None:
    assert is_bearable([], F32Like[...])

  def test_0d_cupy_array(self) -> None:
    assert is_bearable(cp.array(1.5, dtype=cp.float32), F32Like[...])

  def test_0d_numpy_array(self) -> None:
    assert is_bearable(np.array(1.5, dtype=np.float32), F32Like[...])

  def test_dict_rejected(self) -> None:
    assert not is_bearable({"a": 1.0}, F32Like[...])


class TestCuPyLikeTrustScope:
  """CuPy Like fast path trusts only np.ndarray and cupy.ndarray."""

  def test_cupy_ndarray_is_fast_path_trusted(self) -> None:
    from shapix.cupy import _CUPY_TRUSTED

    assert cp.ndarray in _CUPY_TRUSTED
    assert np.ndarray in _CUPY_TRUSTED


# =====================================================================
# Decorator integration
# =====================================================================


class TestCuPyDecoratorIntegration:
  def test_shapix_check(self) -> None:
    @shapix.check
    @beartype
    def f(x: F32[N], y: F32[N]) -> F32[N]:
      return x + y

    f(cp.ones(3, dtype=cp.float32), cp.ones(3, dtype=cp.float32))
    with pytest.raises(BeartypeCallHintParamViolation):
      f(cp.ones(3, dtype=cp.float32), cp.ones(5, dtype=cp.float32))

  def test_check_context(self) -> None:
    with shapix.check_context():
      x = cp.ones((4, 3), dtype=cp.float32)
      y = cp.ones((5, 3), dtype=cp.float32)
      assert is_bearable(x, F32[N, C])
      assert not is_bearable(y, F32[N, C])


# =====================================================================
# Custom dimensions with CuPy
# =====================================================================


class TestCuPyCustomDimensions:
  def test_custom_dim(self) -> None:
    Seq = Dimension("Seq")

    @beartype
    def f(x: F32[N, Seq]) -> F32[N, Seq]:
      return x

    f(cp.ones((4, 10), dtype=cp.float32))

  def test_custom_variadic(self) -> None:
    Batch = Dimension("Batch")

    @beartype
    def f(x: F32[~Batch, C]) -> F32[~Batch, C]:
      return x

    f(cp.ones(3, dtype=cp.float32))
    f(cp.ones((2, 3, 4), dtype=cp.float32))


# =====================================================================
# Return type violations specific to CuPy
# =====================================================================


class TestCuPyReturnViolations:
  def test_return_wrong_dtype(self) -> None:
    @beartype
    def f(x: F32[N]) -> F32[N]:
      return x.astype(cp.int32)

    with pytest.raises(BeartypeCallHintReturnViolation):
      f(cp.ones(3, dtype=cp.float32))

  def test_return_none(self) -> None:
    @beartype
    def f(x: F32[N]) -> F32[N]:
      return None  # type: ignore[return-value]

    with pytest.raises(BeartypeCallHintReturnViolation):
      f(cp.ones(3, dtype=cp.float32))

  def test_return_wrong_rank(self) -> None:
    @beartype
    def f(x: F32[N, C]) -> F32[N]:
      return x

    with pytest.raises(BeartypeCallHintReturnViolation):
      f(cp.ones((3, 4), dtype=cp.float32))


# =====================================================================
# ScalarLike re-exports
# =====================================================================


class TestCuPyScalarLikeReexports:
  """ScalarLike types re-exported from numpy are identical objects."""

  @pytest.mark.parametrize(
    "name",
    [
      "StringLike",
      "BoolScalarLike",
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
      "ShapedScalarLike",
    ],
  )
  def test_identity(self, name: str) -> None:
    import shapix.cupy as cupy_mod
    import shapix.numpy as np_mod

    assert getattr(cupy_mod, name) is getattr(np_mod, name)

  def test_make_scalar_like_type_reexport(self) -> None:
    from shapix.cupy import make_scalar_like_type

    T = make_scalar_like_type(np.float32, casting="same_kind")
    assert is_bearable(1.0, T)

  def test_i8_scalar_like_from_cupy(self) -> None:
    from shapix.cupy import I8ScalarLike

    assert is_bearable(-128, I8ScalarLike)
    assert is_bearable(127, I8ScalarLike)
    assert not is_bearable(128, I8ScalarLike)


# =====================================================================
# Value resolution with CuPy
# =====================================================================


class TestCuPyValueResolution:
  def test_value_with_check(self) -> None:
    """Value("size") resolves under @check with CuPy arrays."""

    @shapix.check
    @beartype
    def f(size: int) -> F32[Value("size")]:  # type: ignore[valid-type]
      return cp.ones(size, dtype=cp.float32)

    assert f(4).shape == (4,)

  def test_value_cross_arg(self) -> None:
    """Value + dim under @check with CuPy arrays."""

    @shapix.check
    @beartype
    def f(x: F32[N], pad: int) -> F32[N + Value("pad")]:  # type: ignore[valid-type]
      return cp.ones(x.shape[0] + pad, dtype=cp.float32)

    result = f(cp.ones(4, dtype=cp.float32), 2)
    assert result.shape == (6,)


class TestCuPyNumericScalarBoolRejection:
  def test_i64_scalar_rejects_bool(self) -> None:
    from shapix.cupy import I64ScalarLike

    assert not is_bearable(True, I64ScalarLike)
