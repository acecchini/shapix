"""Tests for _dimensions.py — dimension symbols and arithmetic."""

from __future__ import annotations

import pytest

from shapix._dimensions import Dimension, Value
from shapix._shape import (
  ANONYMOUS,
  ANONYMOUS_VARIADIC,
  FixedDim,
  NamedDim,
  SymbolicDim,
  ValueDim,
  VariadicDim,
)


class TestDimensionCreation:
  def test_simple_name(self) -> None:
    d = Dimension("N")
    assert str(d) == "N"

  def test_is_str(self) -> None:
    d = Dimension("N")
    assert isinstance(d, str)


class TestDimensionArithmetic:
  def test_add_int(self) -> None:
    d = Dimension("N") + 1
    assert str(d) == "(N+1)"
    assert isinstance(d, Dimension)

  def test_radd_int(self) -> None:
    d = 1 + Dimension("N")
    assert str(d) == "(1+N)"

  def test_sub_int(self) -> None:
    d = Dimension("N") - 1
    assert str(d) == "(N-1)"

  def test_rsub_int(self) -> None:
    d = 10 - Dimension("N")
    assert str(d) == "(10-N)"

  def test_mul_int(self) -> None:
    d = 2 * Dimension("N")
    assert str(d) == "(2*N)"

  def test_truediv(self) -> None:
    d = Dimension("N") / 2
    assert str(d) == "(N/2)"

  def test_rtruediv(self) -> None:
    d = 10 / Dimension("N")
    assert str(d) == "(10/N)"

  def test_floordiv(self) -> None:
    d = Dimension("N") // 2
    assert str(d) == "(N//2)"

  def test_rfloordiv(self) -> None:
    d = 10 // Dimension("N")
    assert str(d) == "(10//N)"

  def test_pow(self) -> None:
    d = Dimension("N") ** 2
    assert str(d) == "(N**2)"

  def test_rpow(self) -> None:
    d = 2 ** Dimension("N")
    assert str(d) == "(2**N)"

  def test_mod(self) -> None:
    d = Dimension("N") % 3
    assert str(d) == "(N%3)"

  def test_rmod(self) -> None:
    d = 11 % Dimension("N")
    assert str(d) == "(11%N)"

  def test_neg(self) -> None:
    d = -Dimension("N")
    assert str(d) == "-N"

  def test_chained(self) -> None:
    d = (Dimension("N") + 1) * 2
    assert str(d) == "((N+1)*2)"


class TestDimSpec:
  def test_named(self) -> None:
    d = Dimension("N")
    spec = d._dim_spec
    assert isinstance(spec, NamedDim)
    assert spec.name == "N"
    assert spec.broadcastable is False

  def test_broadcastable(self) -> None:
    d = Dimension("+N")
    spec = d._dim_spec
    assert isinstance(spec, NamedDim)
    assert spec.name == "N"
    assert spec.broadcastable is True

  def test_variadic(self) -> None:
    d = Dimension("~batch")
    spec = d._dim_spec
    assert isinstance(spec, VariadicDim)
    assert spec.name == "batch"
    assert spec.broadcastable is False

  def test_variadic_broadcastable(self) -> None:
    d = Dimension("~+batch")
    spec = d._dim_spec
    assert isinstance(spec, VariadicDim)
    assert spec.name == "batch"
    assert spec.broadcastable is True

  def test_fixed_int(self) -> None:
    d = Dimension("42")
    spec = d._dim_spec
    assert isinstance(spec, FixedDim)
    assert spec.size == 42

  def test_symbolic(self) -> None:
    d = Dimension("N") + 1
    spec = d._dim_spec
    assert isinstance(spec, SymbolicDim)
    assert spec.expr == "(N+1)"

  def test_anonymous_underscore(self) -> None:
    d = Dimension("__")
    spec = d._dim_spec
    assert spec is ANONYMOUS

  def test_anonymous_variadic(self) -> None:
    d = Dimension("~__")
    spec = d._dim_spec
    assert spec is ANONYMOUS_VARIADIC

  def test_scalar(self) -> None:
    d = Dimension("")
    assert d._dim_spec is None


class TestUnaryOperators:
  def test_invert_makes_variadic(self) -> None:
    d = ~Dimension("N")
    assert str(d) == "~N"
    spec = d._dim_spec
    assert isinstance(spec, VariadicDim)
    assert spec.name == "N"
    assert spec.broadcastable is False

  def test_invert_idempotent(self) -> None:
    d = ~Dimension("~N")
    assert str(d) == "~N"

  def test_pos_makes_broadcastable(self) -> None:
    d = +Dimension("N")
    assert str(d) == "+N"
    spec = d._dim_spec
    assert isinstance(spec, NamedDim)
    assert spec.name == "N"
    assert spec.broadcastable is True

  def test_pos_idempotent(self) -> None:
    d = +Dimension("+N")
    assert str(d) == "+N"

  def test_invert_anonymous(self) -> None:
    """~__ should produce anonymous variadic."""
    d = ~Dimension("__")
    assert str(d) == "~__"
    spec = d._dim_spec
    assert spec is ANONYMOUS_VARIADIC

  def test_invert_on_custom_dim(self) -> None:
    Batch = Dimension("Batch")
    d = ~Batch
    assert str(d) == "~Batch"
    spec = d._dim_spec
    assert isinstance(spec, VariadicDim)
    assert spec.name == "Batch"

  def test_pos_on_custom_dim(self) -> None:
    Batch = Dimension("Batch")
    d = +Batch
    assert str(d) == "+Batch"
    spec = d._dim_spec
    assert isinstance(spec, NamedDim)
    assert spec.name == "Batch"
    assert spec.broadcastable is True


class TestDimSpecEdgeCases:
  def test_zero_fixed_dim(self) -> None:
    d = Dimension("0")
    spec = d._dim_spec
    assert isinstance(spec, FixedDim)
    assert spec.size == 0

  def test_negative_fixed_dim(self) -> None:
    d = Dimension("-3")
    spec = d._dim_spec
    assert isinstance(spec, FixedDim)
    assert spec.size == -3

  def test_neg_operator_symbolic(self) -> None:
    d = -Dimension("N")
    assert str(d) == "-N"
    spec = d._dim_spec
    assert isinstance(spec, SymbolicDim)
    assert spec.expr == "-N"

  def test_dim_plus_dim(self) -> None:
    d = Dimension("N") + Dimension("C")
    assert str(d) == "(N+C)"
    spec = d._dim_spec
    assert isinstance(spec, SymbolicDim)
    assert spec.expr == "(N+C)"


class TestValueExpressions:
  def test_value_expr_requires_string(self) -> None:
    with pytest.raises(TypeError, match=r"Value\(\.\.\.\) expects a string expression"):
      Value(1)  # type: ignore[arg-type]

  def test_value_expr(self) -> None:
    spec = Value("size")._dim_spec  # noqa: SLF001
    assert isinstance(spec, ValueDim)
    assert spec.expr == "size"
    assert spec.broadcastable is False

  def test_broadcastable_value_expr(self) -> None:
    spec = (+Value("size"))._dim_spec  # noqa: SLF001
    assert isinstance(spec, ValueDim)
    assert spec.expr == "size"
    assert spec.broadcastable is True

  def test_broadcastable_idempotent(self) -> None:
    v = +Value("size")
    assert (+v) is v

  def test_value_is_str(self) -> None:
    v = Value("size")
    assert isinstance(v, str)
    assert str(v) == "size"

  def test_value_repr(self) -> None:
    assert repr(Value("size")) == 'Value("size")'
    assert repr(+Value("size")) == '+Value("size")'

  def test_value_not_dimension(self) -> None:
    v = Value("size")
    assert not isinstance(v, Dimension)

  def test_invert_raises(self) -> None:
    with pytest.raises(TypeError, match="variadic"):
      ~Value("size")  # type: ignore[misc]


class TestValueArithmetic:
  """Value + Dimension / int / Value arithmetic."""

  def test_value_add_int(self) -> None:
    r = Value("x") + 3
    assert str(r) == "(x+3)"
    assert isinstance(r._dim_spec, ValueDim)  # noqa: SLF001

  def test_int_add_value(self) -> None:
    r = 3 + Value("x")
    assert str(r) == "(3+x)"

  def test_value_sub_int(self) -> None:
    r = Value("x") - 1
    assert str(r) == "(x-1)"

  def test_int_sub_value(self) -> None:
    r = 10 - Value("x")
    assert str(r) == "(10-x)"

  def test_value_mul_int(self) -> None:
    r = Value("x") * 2
    assert str(r) == "(x*2)"

  def test_int_mul_value(self) -> None:
    r = 2 * Value("x")
    assert str(r) == "(2*x)"

  def test_value_truediv_int(self) -> None:
    r = Value("x") / 2
    assert str(r) == "(x/2)"

  def test_int_truediv_value(self) -> None:
    r = 10 / Value("x")
    assert str(r) == "(10/x)"

  def test_value_floordiv_int(self) -> None:
    r = Value("x") // 2
    assert str(r) == "(x//2)"

  def test_int_floordiv_value(self) -> None:
    r = 10 // Value("x")
    assert str(r) == "(10//x)"

  def test_value_pow_int(self) -> None:
    r = Value("x") ** 2
    assert str(r) == "(x**2)"

  def test_int_pow_value(self) -> None:
    r = 2 ** Value("x")
    assert str(r) == "(2**x)"

  def test_value_mod_int(self) -> None:
    r = Value("x") % 3
    assert str(r) == "(x%3)"

  def test_int_mod_value(self) -> None:
    r = 10 % Value("x")
    assert str(r) == "(10%x)"

  def test_value_neg(self) -> None:
    r = -Value("x")
    assert str(r) == "(-x)"

  def test_value_add_dimension(self) -> None:
    N = Dimension("N")
    r = Value("x") + N
    assert str(r) == "(x+N)"
    assert isinstance(r._dim_spec, ValueDim)  # noqa: SLF001

  def test_dimension_add_value(self) -> None:
    N = Dimension("N")
    r = N + Value("x")
    assert str(r) == "(N+x)"
    assert isinstance(r._dim_spec, ValueDim)  # noqa: SLF001

  def test_value_add_value(self) -> None:
    r = Value("a") + Value("b")
    assert str(r) == "(a+b)"
    assert isinstance(r._dim_spec, ValueDim)  # noqa: SLF001

  def test_dimension_mul_value(self) -> None:
    N = Dimension("N")
    r = N * Value("x")
    assert str(r) == "(N*x)"
    assert isinstance(r._dim_spec, ValueDim)  # noqa: SLF001

  def test_chained_dim_value_int(self) -> None:
    N = Dimension("N")
    r = (N + Value("pad")) * 2
    assert str(r) == "((N+pad)*2)"

  def test_dim_dim_still_dimension(self) -> None:
    """Arithmetic between Dimensions (no Value) should stay Dimension."""
    N = Dimension("N")
    C = Dimension("C")
    r = N + C
    assert isinstance(r, Dimension)
    from shapix._dimensions import _ValueExpr

    assert not isinstance(r, _ValueExpr)


class TestMixedScalarRejection:
  """Scalar must be the only shape token — mixed use must raise."""

  def test_n_scalar_raises(self) -> None:
    from shapix import N, Scalar
    from shapix.numpy import F32

    with pytest.raises(TypeError, match="Scalar must be the only shape token"):
      F32[N, Scalar]

  def test_scalar_n_raises(self) -> None:
    from shapix import N, Scalar
    from shapix.numpy import F32

    with pytest.raises(TypeError, match="Scalar must be the only shape token"):
      F32[Scalar, N]

  def test_ellipsis_scalar_raises(self) -> None:
    from shapix import Scalar
    from shapix.numpy import F32

    with pytest.raises(TypeError, match="Scalar must be the only shape token"):
      F32[..., Scalar]

  def test_scalar_ellipsis_raises(self) -> None:
    from shapix import Scalar
    from shapix.numpy import F32

    with pytest.raises(TypeError, match="Scalar must be the only shape token"):
      F32[Scalar, ...]

  def test_scalar_int_raises(self) -> None:
    from shapix import Scalar
    from shapix.numpy import F32

    with pytest.raises(TypeError, match="Scalar must be the only shape token"):
      F32[Scalar, 3]

  def test_scalar_alone_works(self) -> None:
    from shapix import Scalar
    from shapix.numpy import F32

    hint = F32[Scalar]
    assert hasattr(hint, "__metadata__")

  def test_dimension_empty_n_raises(self) -> None:
    """Dimension("") (equivalent to Scalar) mixed with N must raise."""
    from shapix import N
    from shapix.numpy import F32

    with pytest.raises(TypeError, match="Scalar must be the only shape token"):
      F32[Dimension(""), N]

  def test_n_dimension_empty_raises(self) -> None:
    from shapix import N
    from shapix.numpy import F32

    with pytest.raises(TypeError, match="Scalar must be the only shape token"):
      F32[N, Dimension("")]

  def test_ellipsis_dimension_empty_raises(self) -> None:
    from shapix.numpy import F32

    with pytest.raises(TypeError, match="Scalar must be the only shape token"):
      F32[..., Dimension("")]

  def test_dimension_empty_ellipsis_raises(self) -> None:
    from shapix.numpy import F32

    with pytest.raises(TypeError, match="Scalar must be the only shape token"):
      F32[Dimension(""), ...]

  def test_dimension_empty_alone_works(self) -> None:
    """Dimension("") by itself is valid (equivalent to Scalar)."""
    from shapix.numpy import F32

    hint = F32[Dimension("")]
    assert hasattr(hint, "__metadata__")

  def test_negative_fixed_dim_rejected_in_shape_spec(self) -> None:
    """Dimension('-3') must be rejected when used in array shape spec."""
    from shapix._array_types import _to_shape_spec

    with pytest.raises(TypeError, match="Negative dimension"):
      _to_shape_spec((Dimension("-3"),))

  def test_negative_dim_in_mixed_spec_rejected(self) -> None:
    """Dimension('-1') mixed with named dims is rejected."""
    from shapix._array_types import _to_shape_spec

    with pytest.raises(TypeError, match="Negative dimension"):
      _to_shape_spec((Dimension("-1"), Dimension("N")))

  def test_array_factory_rejects_negative_dimension(self) -> None:
    """F32[Dimension('-3')] must raise TypeError like F32[-3]."""
    from shapix.numpy import F32

    with pytest.raises(TypeError, match="Negative dimension"):
      F32[Dimension("-3")]
