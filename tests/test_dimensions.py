"""Tests for _dimensions.py — dimension symbols and arithmetic."""

from __future__ import annotations

from shapix._dimensions import Dimension
from shapix._shape import (
  ANONYMOUS,
  ANONYMOUS_VARIADIC,
  FixedDim,
  NamedDim,
  SymbolicDim,
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
