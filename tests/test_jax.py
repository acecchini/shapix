# pyright: reportArgumentType=false, reportGeneralTypeIssues=false, reportUnusedImport=false
"""Tests for shapix.jax — JAX array types, Like types, and dtype handling."""

from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from beartype import beartype
from beartype.door import is_bearable
from beartype.roar import (
  BeartypeCallHintParamViolation,
  BeartypeCallHintReturnViolation,
)

import shapix
from shapix import B, C, N, __, Dimension  # noqa: F811 — B used in ~B annotations
from shapix.jax import (
  BF16,
  BF16Like,
  Bool,
  F16,
  F32,
  F32Like,
  Float,
  I32,
  I64Like,
  Int,
  Num,
  Shaped,
  U8,
  BoolLike,
)


# =====================================================================
# Dtype acceptance / rejection
# =====================================================================


class TestJaxDtypeAcceptance:
  @pytest.mark.parametrize(
    "ArrayType, good_dtype, bad_dtypes",
    [
      (F32, jnp.float32, [jnp.float16, jnp.int32, jnp.bool_]),
      (F16, jnp.float16, [jnp.float32]),
      (BF16, jnp.bfloat16, [jnp.float32, jnp.float16]),
      (I32, jnp.int32, [jnp.float32, jnp.int16]),
      (U8, jnp.uint8, [jnp.int8, jnp.int32]),
      (Bool, jnp.bool_, [jnp.int32, jnp.float32]),
    ],
  )
  def test_accepts_correct_rejects_wrong(
    self, ArrayType: object, good_dtype: object, bad_dtypes: list[object]
  ) -> None:
    @beartype
    def f(x: ArrayType[N]) -> ArrayType[N]:  # type: ignore[valid-type]
      return x

    f(jnp.ones(3, dtype=good_dtype))
    for bad in bad_dtypes:
      with pytest.raises(BeartypeCallHintParamViolation):
        f(jnp.ones(3, dtype=bad))


class TestJaxCategoryTypes:
  def test_float_accepts_f32_f16_bf16(self) -> None:
    @beartype
    def f(x: Float[N]) -> Float[N]:
      return x

    f(jnp.ones(3, dtype=jnp.float32))
    f(jnp.ones(3, dtype=jnp.float16))
    f(jnp.ones(3, dtype=jnp.bfloat16))

  def test_float_rejects_int(self) -> None:
    @beartype
    def f(x: Float[N]) -> Float[N]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(jnp.ones(3, dtype=jnp.int32))

  def test_int_rejects_unsigned(self) -> None:
    @beartype
    def f(x: Int[N]) -> Int[N]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(jnp.ones(3, dtype=jnp.uint8))

  def test_shaped_accepts_all(self) -> None:
    @beartype
    def f(x: Shaped[N]) -> Shaped[N]:
      return x

    for dtype in [jnp.float32, jnp.int32, jnp.uint8, jnp.bool_, jnp.bfloat16]:
      f(jnp.ones(3, dtype=dtype))

  def test_num_rejects_bool(self) -> None:
    @beartype
    def f(x: Num[N]) -> Num[N]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(jnp.ones(3, dtype=jnp.bool_))


# =====================================================================
# Shape checking (smoke tests — shape logic tested exhaustively in test_numpy)
# =====================================================================


class TestJaxShapeChecking:
  def test_basic_2d(self) -> None:
    @beartype
    def f(x: F32[N, C]) -> F32[N, C]:
      return x

    assert f(jnp.ones((4, 3), dtype=jnp.float32)).shape == (4, 3)

  def test_wrong_rank(self) -> None:
    @beartype
    def f(x: F32[N, C]) -> F32[N, C]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(jnp.ones(5, dtype=jnp.float32))

  def test_cross_arg_consistency(self) -> None:
    @beartype
    def f(x: F32[N, C], y: F32[N, C]) -> F32[N, C]:
      return x + y

    assert f(
      jnp.ones((4, 3), dtype=jnp.float32), jnp.ones((4, 3), dtype=jnp.float32)
    ).shape == (4, 3)

  def test_cross_arg_mismatch(self) -> None:
    @beartype
    def f(x: F32[N, C], y: F32[N, C]) -> F32[N, C]:
      return x + y

    with pytest.raises(BeartypeCallHintParamViolation):
      f(jnp.ones((4, 3), dtype=jnp.float32), jnp.ones((5, 3), dtype=jnp.float32))

  def test_fixed_dim(self) -> None:
    @beartype
    def f(x: F32[3, N]) -> F32[3, N]:
      return x

    f(jnp.ones((3, 5), dtype=jnp.float32))
    with pytest.raises(BeartypeCallHintParamViolation):
      f(jnp.ones((4, 5), dtype=jnp.float32))

  def test_symbolic_dim(self) -> None:
    @beartype
    def f(x: F32[N], y: F32[N + 1]) -> F32[N]:
      return x

    f(jnp.ones(5, dtype=jnp.float32), jnp.ones(6, dtype=jnp.float32))
    with pytest.raises(BeartypeCallHintParamViolation):
      f(jnp.ones(5, dtype=jnp.float32), jnp.ones(5, dtype=jnp.float32))

  def test_variadic(self) -> None:
    @beartype
    def f(x: F32[~B, C]) -> F32[~B, C]:
      return x

    f(jnp.ones(3, dtype=jnp.float32))
    f(jnp.ones((2, 3, 4), dtype=jnp.float32))

  def test_anonymous_variadic(self) -> None:
    @beartype
    def f(x: F32[~__, C]) -> F32[~__, C]:
      return x

    f(jnp.ones(3, dtype=jnp.float32))
    f(jnp.ones((2, 3, 4), dtype=jnp.float32))

  def test_broadcastable(self) -> None:
    @beartype
    def f(x: F32[N, C], y: F32[+N, C]) -> F32[N, C]:
      return x + y

    f(jnp.ones((4, 3), dtype=jnp.float32), jnp.ones((1, 3), dtype=jnp.float32))
    f(jnp.ones((4, 3), dtype=jnp.float32), jnp.ones((4, 3), dtype=jnp.float32))
    with pytest.raises(BeartypeCallHintParamViolation):
      f(jnp.ones((4, 3), dtype=jnp.float32), jnp.ones((2, 3), dtype=jnp.float32))

  def test_return_shape_violation(self) -> None:
    @beartype
    def f(x: F32[N, C]) -> F32[N, C]:
      return jnp.ones((1, 1), dtype=jnp.float32)

    with pytest.raises(BeartypeCallHintReturnViolation):
      f(jnp.ones((4, 3), dtype=jnp.float32))

  def test_return_dtype_violation(self) -> None:
    @beartype
    def f(x: F32[N]) -> F32[N]:
      return x.astype(jnp.int32)

    with pytest.raises(BeartypeCallHintReturnViolation):
      f(jnp.ones(3, dtype=jnp.float32))

  def test_sequential_calls_independent(self) -> None:
    @beartype
    def f(x: F32[N]) -> F32[N]:
      return x

    f(jnp.ones(3, dtype=jnp.float32))
    f(jnp.ones(99, dtype=jnp.float32))

  def test_anonymous_dim(self) -> None:
    @beartype
    def f(x: F32[__, C], y: F32[__, C]) -> F32[__, C]:
      return x

    f(jnp.ones((4, 3), dtype=jnp.float32), jnp.ones((99, 3), dtype=jnp.float32))


# =====================================================================
# BFloat16 (JAX/PyTorch-specific)
# =====================================================================


class TestJaxBFloat16:
  def test_bf16_basic(self) -> None:
    @beartype
    def f(x: BF16[N, C]) -> BF16[N, C]:
      return x

    assert f(jnp.ones((4, 3), dtype=jnp.bfloat16)).shape == (4, 3)

  def test_bf16_cross_arg(self) -> None:
    @beartype
    def f(x: BF16[N], y: BF16[N]) -> BF16[N]:
      return x + y

    f(jnp.ones(5, dtype=jnp.bfloat16), jnp.ones(5, dtype=jnp.bfloat16))
    with pytest.raises(BeartypeCallHintParamViolation):
      f(jnp.ones(5, dtype=jnp.bfloat16), jnp.ones(7, dtype=jnp.bfloat16))

  def test_bf16_rejects_f32(self) -> None:
    @beartype
    def f(x: BF16[N]) -> BF16[N]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(jnp.ones(3, dtype=jnp.float32))


# =====================================================================
# 64-bit dtypes (require jax_enable_x64)
# =====================================================================


class TestJax64Bit:
  @pytest.fixture(autouse=True)
  def _enable_x64(self) -> None:
    jax.config.update("jax_enable_x64", True)
    yield  # type: ignore[func-returns-value]
    jax.config.update("jax_enable_x64", False)

  def test_f64(self) -> None:
    from shapix.jax import F64

    @beartype
    def f(x: F64[N]) -> F64[N]:
      return x

    f(jnp.ones(5, dtype=jnp.float64))

  def test_i64(self) -> None:
    from shapix.jax import I64

    @beartype
    def f(x: I64[N]) -> I64[N]:
      return x

    f(jnp.ones(5, dtype=jnp.int64))


# =====================================================================
# Cross-backend rejection
# =====================================================================


class TestJaxCrossBackendRejection:
  def test_jax_type_rejects_numpy(self) -> None:
    @beartype
    def f(x: F32[N]) -> F32[N]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones(3, dtype=np.float32))

  def test_numpy_type_rejects_jax(self) -> None:
    from shapix.numpy import F32 as NpF32

    @beartype
    def f(x: NpF32[N]) -> NpF32[N]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(jnp.ones(3, dtype=jnp.float32))


# =====================================================================
# Like types
# =====================================================================


class TestJaxLikeTypes:
  @pytest.mark.parametrize(
    "value",
    [1.5, [1.0, 2.0], [[1.0, 2.0], [3.0, 4.0]], (1.0, 2.0)],
    ids=["scalar", "list", "nested_list", "tuple"],
  )
  def test_f32like_accepts_scalars_and_sequences(self, value: object) -> None:
    assert is_bearable(value, F32Like[...])

  def test_f32like_accepts_numpy(self) -> None:
    assert is_bearable(np.ones(3, dtype=np.float32), F32Like[...])

  def test_f32like_accepts_jax(self) -> None:
    assert is_bearable(jnp.ones(3, dtype=jnp.float32), F32Like[...])

  def test_f32like_rejects_non_numeric(self) -> None:
    assert not is_bearable(object(), F32Like[...])

  def test_i64like(self) -> None:
    assert is_bearable(42, I64Like[...])
    assert is_bearable([1, 2, 3], I64Like[...])

  def test_boollike(self) -> None:
    assert is_bearable(True, BoolLike[...])
    assert is_bearable([True, False], BoolLike[...])

  def test_bf16like_accepts_jax_bfloat16(self) -> None:
    arr = jnp.ones(3, dtype=jnp.bfloat16)
    assert is_bearable(arr, BF16Like[...])

  def test_bf16like_rejects_non_numeric_dtype(self) -> None:
    # Under same_kind casting (default), all numeric types can cast to bfloat16
    # when JAX is loaded, so we use a string array for rejection
    arr = np.array(["a", "b"], dtype=np.str_)
    assert not is_bearable(arr, BF16Like[...])

  def test_bf16like_shape_checking(self) -> None:
    @beartype
    def f(x: BF16Like[N]) -> None:  # type: ignore[valid-type]
      pass

    f(jnp.ones(5, dtype=jnp.bfloat16))
    with pytest.raises(Exception):
      f(jnp.ones((3, 4), dtype=jnp.bfloat16))


class TestJaxArrayProtocol:
  def test_jax_array_protocol_accepted(self) -> None:
    """Objects implementing __jax_array__ should be accepted by JAX Like types."""

    class MyArray:
      """Custom type that JAX can convert via __jax_array__."""

      def __init__(self, data: object) -> None:
        self._data = data

      def __jax_array__(self) -> jax.Array:  # noqa: PLW3201
        return jnp.asarray(self._data, dtype=jnp.float32)

    wrapper = MyArray([1.0, 2.0, 3.0])
    # Should NOT have .shape/.dtype, forcing the slow path
    assert not hasattr(wrapper, "shape")
    assert is_bearable(wrapper, F32Like[...])

  def test_jax_array_protocol_wrong_dtype_rejected(self) -> None:
    """__jax_array__ object with wrong dtype should be rejected."""
    from shapix.jax import I64Like

    class MyArray:
      def __init__(self, data: object) -> None:
        self._data = data

      def __jax_array__(self) -> jax.Array:  # noqa: PLW3201
        return jnp.asarray(self._data, dtype=jnp.float32)

    wrapper = MyArray([1.0, 2.0])
    # I64Like expects integer, but __jax_array__ returns float32
    assert not is_bearable(wrapper, I64Like[...])


class TestJaxLikeEdgeCases:
  def test_deep_nesting(self) -> None:
    assert is_bearable([[[[[[1.0]]]]]], F32Like[...])

  def test_empty_list(self) -> None:
    assert is_bearable([], F32Like[...])

  def test_0d_jax_array(self) -> None:
    assert is_bearable(jnp.array(1.5, dtype=jnp.float32), F32Like[...])

  def test_0d_numpy_array(self) -> None:
    assert is_bearable(np.array(1.5, dtype=np.float32), F32Like[...])

  def test_dict_rejected(self) -> None:
    assert not is_bearable({"a": 1.0}, F32Like[...])

  def test_f32like_accepts_torch_tensor_via_numpy_arraylike(self) -> None:
    torch = pytest.importorskip("torch")
    assert is_bearable(torch.ones(3, dtype=torch.float32), F32Like[...])


# =====================================================================
# Decorator integration
# =====================================================================


class TestJaxDecoratorIntegration:
  def test_shapix_check(self) -> None:
    @shapix.check
    @beartype
    def f(x: F32[N], y: F32[N]) -> F32[N]:
      return x + y

    f(jnp.ones(3, dtype=jnp.float32), jnp.ones(3, dtype=jnp.float32))
    with pytest.raises(BeartypeCallHintParamViolation):
      f(jnp.ones(3, dtype=jnp.float32), jnp.ones(5, dtype=jnp.float32))

  def test_check_context(self) -> None:
    with shapix.check_context():
      x = jnp.ones((4, 3), dtype=jnp.float32)
      y = jnp.ones((5, 3), dtype=jnp.float32)
      assert is_bearable(x, F32[N, C])
      assert not is_bearable(y, F32[N, C])


# =====================================================================
# Custom dimensions with JAX
# =====================================================================


class TestJaxCustomDimensions:
  def test_custom_dim(self) -> None:
    Seq = Dimension("Seq")

    @beartype
    def f(x: F32[N, Seq]) -> F32[N, Seq]:
      return x

    f(jnp.ones((4, 10), dtype=jnp.float32))

  def test_custom_variadic(self) -> None:
    Batch = Dimension("Batch")

    @beartype
    def f(x: F32[~Batch, C]) -> F32[~Batch, C]:
      return x

    f(jnp.ones(3, dtype=jnp.float32))
    f(jnp.ones((2, 3, 4), dtype=jnp.float32))


# =====================================================================
# Mixed JAX + NumPy in same function
# =====================================================================


class TestJaxMixedAnnotations:
  def test_jax_input_numpy_return(self) -> None:
    """Function takes JAX input, returns numpy — return type must match."""
    from shapix.numpy import F32 as NpF32

    @beartype
    def f(x: F32[N]) -> NpF32[N]:
      return np.asarray(x)

    result = f(jnp.ones(5, dtype=jnp.float32))
    assert isinstance(result, np.ndarray)
    assert result.shape == (5,)


# =====================================================================
# ScalarLike re-exports
# =====================================================================


class TestJaxScalarLikeReexports:
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
      "C64ScalarLike",
      "C128ScalarLike",
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
    import shapix.jax as jax_mod
    import shapix.numpy as np_mod

    assert getattr(jax_mod, name) is getattr(np_mod, name)

  def test_make_scalar_like_type_reexport(self) -> None:
    from shapix.jax import make_scalar_like_type

    T = make_scalar_like_type(np.float32, casting="same_kind")
    assert is_bearable(1.0, T)

  def test_u8_scalar_like_from_jax(self) -> None:
    from shapix.jax import U8ScalarLike

    assert is_bearable(0, U8ScalarLike)
    assert is_bearable(255, U8ScalarLike)
    assert not is_bearable(256, U8ScalarLike)


# =====================================================================
# Like casting spot checks with JAX arrays
# =====================================================================


class TestJaxLikeCastingVariants:
  def test_like_same_kind_jax_array(self) -> None:
    from shapix._array_types import make_array_like_type
    from shapix._dtypes import FLOAT32

    T = make_array_like_type(FLOAT32, casting="same_kind")
    assert is_bearable(jnp.ones(3, dtype=jnp.float32), T[...])
    assert not is_bearable(jnp.ones(3, dtype=jnp.complex64), T[...])

  def test_like_no_casting_jax_array(self) -> None:
    from shapix._array_types import make_array_like_type
    from shapix._dtypes import FLOAT32

    T = make_array_like_type(FLOAT32, casting="no")
    assert is_bearable(jnp.ones(3, dtype=jnp.float32), T[...])
    assert not is_bearable(jnp.ones(3, dtype=jnp.int32), T[...])
