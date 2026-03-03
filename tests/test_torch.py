"""Tests for shapix.torch — PyTorch tensor types, Like types, and dtype handling."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from beartype import beartype
from beartype.door import is_bearable
from beartype.roar import (
  BeartypeCallHintParamViolation,
  BeartypeCallHintReturnViolation,
)

import shapix
from shapix import B, C, N, __, Dimension
from shapix.torch import (
  BF16,
  Bool,
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
  BoolLk,
)


# =====================================================================
# Dtype acceptance / rejection
# =====================================================================


class TestTorchDtypeAcceptance:
  @pytest.mark.parametrize(
    "ArrayType, good_dtype, bad_dtypes",
    [
      (F32, torch.float32, [torch.float64, torch.float16, torch.int32, torch.bool]),
      (F64, torch.float64, [torch.float32]),
      (F16, torch.float16, [torch.float32]),
      (BF16, torch.bfloat16, [torch.float32, torch.float16]),
      (I32, torch.int32, [torch.float32, torch.int64]),
      (I64, torch.int64, [torch.float32, torch.uint8]),
      (U8, torch.uint8, [torch.int8, torch.int32]),
      (Bool, torch.bool, [torch.int32, torch.float32]),
    ],
  )
  def test_accepts_correct_rejects_wrong(
    self, ArrayType: object, good_dtype: object, bad_dtypes: list[object]
  ) -> None:
    @beartype
    def f(x: ArrayType[N]) -> ArrayType[N]:  # type: ignore[valid-type]
      return x

    f(torch.ones(3, dtype=good_dtype))
    for bad in bad_dtypes:
      with pytest.raises(BeartypeCallHintParamViolation):
        f(torch.ones(3, dtype=bad))


class TestTorchCategoryTypes:
  def test_float_accepts_f32_f64_f16_bf16(self) -> None:
    @beartype
    def f(x: Float[N]) -> Float[N]:
      return x

    for dtype in [torch.float32, torch.float64, torch.float16, torch.bfloat16]:
      f(torch.ones(3, dtype=dtype))

  def test_float_rejects_int(self) -> None:
    @beartype
    def f(x: Float[N]) -> Float[N]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(torch.ones(3, dtype=torch.int32))

  def test_int_rejects_unsigned(self) -> None:
    @beartype
    def f(x: Int[N]) -> Int[N]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(torch.ones(3, dtype=torch.uint8))

  def test_shaped_accepts_all(self) -> None:
    @beartype
    def f(x: Shaped[N]) -> Shaped[N]:
      return x

    for dtype in [torch.float32, torch.int32, torch.uint8, torch.bool, torch.bfloat16]:
      f(torch.ones(3, dtype=dtype))

  def test_num_rejects_bool(self) -> None:
    @beartype
    def f(x: Num[N]) -> Num[N]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(torch.ones(3, dtype=torch.bool))


# =====================================================================
# Shape checking (smoke tests — shape logic tested exhaustively in test_numpy)
# =====================================================================


class TestTorchShapeChecking:
  def test_basic_2d(self) -> None:
    @beartype
    def f(x: F32[N, C]) -> F32[N, C]:
      return x

    assert f(torch.ones(4, 3, dtype=torch.float32)).shape == (4, 3)

  def test_wrong_rank(self) -> None:
    @beartype
    def f(x: F32[N, C]) -> F32[N, C]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(torch.ones(5, dtype=torch.float32))

  def test_cross_arg_consistency(self) -> None:
    @beartype
    def f(x: F32[N, C], y: F32[N, C]) -> F32[N, C]:
      return x + y

    assert f(
      torch.ones(4, 3, dtype=torch.float32), torch.ones(4, 3, dtype=torch.float32)
    ).shape == (4, 3)

  def test_cross_arg_mismatch(self) -> None:
    @beartype
    def f(x: F32[N, C], y: F32[N, C]) -> F32[N, C]:
      return x + y

    with pytest.raises(BeartypeCallHintParamViolation):
      f(torch.ones(4, 3, dtype=torch.float32), torch.ones(5, 3, dtype=torch.float32))

  def test_fixed_dim(self) -> None:
    @beartype
    def f(x: F32[3, N]) -> F32[3, N]:
      return x

    f(torch.ones(3, 5, dtype=torch.float32))
    with pytest.raises(BeartypeCallHintParamViolation):
      f(torch.ones(4, 5, dtype=torch.float32))

  def test_symbolic_dim(self) -> None:
    @beartype
    def f(x: F32[N], y: F32[N + 1]) -> F32[N]:
      return x

    f(torch.ones(5, dtype=torch.float32), torch.ones(6, dtype=torch.float32))
    with pytest.raises(BeartypeCallHintParamViolation):
      f(torch.ones(5, dtype=torch.float32), torch.ones(5, dtype=torch.float32))

  def test_variadic(self) -> None:
    @beartype
    def f(x: F32[~B, C]) -> F32[~B, C]:
      return x

    f(torch.ones(3, dtype=torch.float32))
    f(torch.ones(2, 3, 4, dtype=torch.float32))

  def test_anonymous_variadic(self) -> None:
    @beartype
    def f(x: F32[~__, C]) -> F32[~__, C]:
      return x

    f(torch.ones(3, dtype=torch.float32))
    f(torch.ones(2, 3, 4, dtype=torch.float32))

  def test_broadcastable(self) -> None:
    @beartype
    def f(x: F32[N, C], y: F32[+N, C]) -> F32[N, C]:
      return x + y

    f(torch.ones(4, 3, dtype=torch.float32), torch.ones(1, 3, dtype=torch.float32))
    f(torch.ones(4, 3, dtype=torch.float32), torch.ones(4, 3, dtype=torch.float32))
    with pytest.raises(BeartypeCallHintParamViolation):
      f(torch.ones(4, 3, dtype=torch.float32), torch.ones(2, 3, dtype=torch.float32))

  def test_return_shape_violation(self) -> None:
    @beartype
    def f(x: F32[N, C]) -> F32[N, C]:
      return torch.ones(1, 1, dtype=torch.float32)

    with pytest.raises(BeartypeCallHintReturnViolation):
      f(torch.ones(4, 3, dtype=torch.float32))

  def test_return_dtype_violation(self) -> None:
    @beartype
    def f(x: F32[N]) -> F32[N]:
      return x.to(torch.float64)

    with pytest.raises(BeartypeCallHintReturnViolation):
      f(torch.ones(3, dtype=torch.float32))

  def test_sequential_calls_independent(self) -> None:
    @beartype
    def f(x: F32[N]) -> F32[N]:
      return x

    f(torch.ones(3, dtype=torch.float32))
    f(torch.ones(99, dtype=torch.float32))

  def test_anonymous_dim(self) -> None:
    @beartype
    def f(x: F32[__, C], y: F32[__, C]) -> F32[__, C]:
      return x

    f(torch.ones(4, 3, dtype=torch.float32), torch.ones(99, 3, dtype=torch.float32))


# =====================================================================
# BFloat16 (PyTorch/JAX-specific)
# =====================================================================


class TestTorchBFloat16:
  def test_bf16_basic(self) -> None:
    @beartype
    def f(x: BF16[N, C]) -> BF16[N, C]:
      return x

    assert f(torch.ones(4, 3, dtype=torch.bfloat16)).shape == (4, 3)

  def test_bf16_cross_arg(self) -> None:
    @beartype
    def f(x: BF16[N], y: BF16[N]) -> BF16[N]:
      return x + y

    f(torch.ones(5, dtype=torch.bfloat16), torch.ones(5, dtype=torch.bfloat16))
    with pytest.raises(BeartypeCallHintParamViolation):
      f(torch.ones(5, dtype=torch.bfloat16), torch.ones(7, dtype=torch.bfloat16))

  def test_bf16_rejects_f32(self) -> None:
    @beartype
    def f(x: BF16[N]) -> BF16[N]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(torch.ones(3, dtype=torch.float32))


# =====================================================================
# Cross-backend rejection
# =====================================================================


class TestTorchCrossBackendRejection:
  def test_torch_type_rejects_numpy(self) -> None:
    @beartype
    def f(x: F32[N]) -> F32[N]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones(3, dtype=np.float32))

  def test_numpy_type_rejects_torch(self) -> None:
    from shapix.numpy import F32 as NpF32

    @beartype
    def f(x: NpF32[N]) -> NpF32[N]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(torch.ones(3, dtype=torch.float32))

  def test_torch_type_rejects_jax(self) -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    @beartype
    def f(x: F32[N]) -> F32[N]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(jnp.ones(3, dtype=jnp.float32))


# =====================================================================
# Like types
# =====================================================================


class TestTorchLikeTypes:
  @pytest.mark.parametrize(
    "value",
    [1.5, [1.0, 2.0], [[1.0, 2.0], [3.0, 4.0]], (1.0, 2.0)],
    ids=["scalar", "list", "nested_list", "tuple"],
  )
  def test_f32like_accepts_scalars_and_sequences(self, value: object) -> None:
    assert is_bearable(value, F32Like)

  def test_f32like_accepts_numpy(self) -> None:
    assert is_bearable(np.ones(3, dtype=np.float32), F32Like)

  def test_f32like_accepts_torch(self) -> None:
    assert is_bearable(torch.ones(3, dtype=torch.float32), F32Like)

  def test_f32like_rejects_non_numeric(self) -> None:
    assert not is_bearable(object(), F32Like)

  def test_i64like(self) -> None:
    assert is_bearable(42, I64Like)
    assert is_bearable([1, 2, 3], I64Like)
    assert is_bearable(torch.ones(3, dtype=torch.int64), I64Like)

  def test_boollk(self) -> None:
    assert is_bearable(True, BoolLk)
    assert is_bearable([True, False], BoolLk)


class TestTorchLikeEdgeCases:
  def test_deep_nesting(self) -> None:
    assert is_bearable([[[[[[1.0]]]]]], F32Like)

  def test_empty_list(self) -> None:
    assert is_bearable([], F32Like)

  def test_0d_torch_tensor(self) -> None:
    assert is_bearable(torch.tensor(1.5), F32Like)

  def test_0d_numpy_array(self) -> None:
    assert is_bearable(np.array(1.5, dtype=np.float32), F32Like)

  def test_dict_rejected(self) -> None:
    assert not is_bearable({"a": 1.0}, F32Like)

  def test_f32like_rejects_jax_array(self) -> None:
    pytest.importorskip("jax")
    import jax.numpy as jnp

    assert not is_bearable(jnp.ones(3, dtype=jnp.float32), F32Like)


# =====================================================================
# Decorator integration
# =====================================================================


class TestTorchDecoratorIntegration:
  def test_shapix_check(self) -> None:
    @shapix.check
    @beartype
    def f(x: F32[N], y: F32[N]) -> F32[N]:
      return x + y

    f(torch.ones(3, dtype=torch.float32), torch.ones(3, dtype=torch.float32))
    with pytest.raises(BeartypeCallHintParamViolation):
      f(torch.ones(3, dtype=torch.float32), torch.ones(5, dtype=torch.float32))

  def test_check_context(self) -> None:
    with shapix.check_context():
      x = torch.ones(4, 3, dtype=torch.float32)
      y = torch.ones(5, 3, dtype=torch.float32)
      assert is_bearable(x, F32[N, C])
      assert not is_bearable(y, F32[N, C])


# =====================================================================
# Custom dimensions with PyTorch
# =====================================================================


class TestTorchCustomDimensions:
  def test_custom_dim(self) -> None:
    Seq = Dimension("Seq")

    @beartype
    def f(x: F32[N, Seq]) -> F32[N, Seq]:
      return x

    f(torch.ones(4, 10, dtype=torch.float32))

  def test_custom_variadic(self) -> None:
    Batch = Dimension("Batch")

    @beartype
    def f(x: F32[~Batch, C]) -> F32[~Batch, C]:
      return x

    f(torch.ones(3, dtype=torch.float32))
    f(torch.ones(2, 3, 4, dtype=torch.float32))


# =====================================================================
# Mixed Torch + NumPy in same function
# =====================================================================


class TestTorchMixedAnnotations:
  def test_torch_input_numpy_return(self) -> None:
    from shapix.numpy import F32 as NpF32

    @beartype
    def f(x: F32[N]) -> NpF32[N]:
      return x.numpy()

    result = f(torch.ones(5, dtype=torch.float32))
    assert isinstance(result, np.ndarray)
    assert result.shape == (5,)


# =====================================================================
# Return type violations specific to PyTorch
# =====================================================================


class TestTorchReturnViolations:
  def test_return_wrong_dtype(self) -> None:
    @beartype
    def f(x: F32[N]) -> F32[N]:
      return x.to(torch.int32)

    with pytest.raises(BeartypeCallHintReturnViolation):
      f(torch.ones(3, dtype=torch.float32))

  def test_return_none(self) -> None:
    @beartype
    def f(x: F32[N]) -> F32[N]:
      return None  # type: ignore[return-value]

    with pytest.raises(BeartypeCallHintReturnViolation):
      f(torch.ones(3, dtype=torch.float32))

  def test_return_wrong_rank(self) -> None:
    @beartype
    def f(x: F32[N, C]) -> F32[N]:
      return x

    with pytest.raises(BeartypeCallHintReturnViolation):
      f(torch.ones(3, 4, dtype=torch.float32))
