"""Tests for _shape.py — dimension spec types and matching logic."""

from __future__ import annotations

from shapix._memo import ShapeMemo
from shapix._shape import (
  ANONYMOUS,
  ANONYMOUS_VARIADIC,
  FixedDim,
  NamedDim,
  SymbolicDim,
  VariadicDim,
  check_shape,
)


class TestFixedDim:
  def test_match(self) -> None:
    memo = ShapeMemo()
    assert check_shape((3, 4), (FixedDim(3), FixedDim(4)), memo) == ""

  def test_mismatch(self) -> None:
    memo = ShapeMemo()
    err = check_shape((3, 5), (FixedDim(3), FixedDim(4)), memo)
    assert "expected 4 but got 5" in err

  def test_wrong_rank(self) -> None:
    memo = ShapeMemo()
    err = check_shape((3, 4, 5), (FixedDim(3), FixedDim(4)), memo)
    assert "expected 2 dimensions but got 3" in err


class TestNamedDim:
  def test_bind_and_match(self) -> None:
    memo = ShapeMemo()
    assert check_shape((8,), (NamedDim("N"),), memo) == ""
    assert memo.single["N"] == 8
    # Second check with same N must match
    assert check_shape((8,), (NamedDim("N"),), memo) == ""

  def test_bind_and_mismatch(self) -> None:
    memo = ShapeMemo()
    assert check_shape((8,), (NamedDim("N"),), memo) == ""
    err = check_shape((5,), (NamedDim("N"),), memo)
    assert "expected 8 but got 5" in err

  def test_multiple_names(self) -> None:
    memo = ShapeMemo()
    spec = (NamedDim("N"), NamedDim("C"), NamedDim("N"))
    assert check_shape((4, 3, 4), spec, memo) == ""
    assert memo.single == {"N": 4, "C": 3}

  def test_cross_arg_consistency(self) -> None:
    memo = ShapeMemo()
    spec = (NamedDim("N"), NamedDim("C"))
    assert check_shape((4, 3), spec, memo) == ""
    assert check_shape((4, 3), spec, memo) == ""
    err = check_shape((5, 3), spec, memo)
    assert "expected 4 but got 5" in err

  def test_broadcastable_size_1(self) -> None:
    memo = ShapeMemo()
    spec = (NamedDim("N", broadcastable=True),)
    assert check_shape((1,), spec, memo) == ""
    # Broadcastable with size 1 should not bind
    assert "N" not in memo.single

  def test_broadcastable_normal_size(self) -> None:
    memo = ShapeMemo()
    spec = (NamedDim("N", broadcastable=True),)
    assert check_shape((8,), spec, memo) == ""
    assert memo.single["N"] == 8


class TestSymbolicDim:
  def test_expression(self) -> None:
    memo = ShapeMemo()
    memo.single["N"] = 5
    assert check_shape((6,), (SymbolicDim("N+1"),), memo) == ""

  def test_expression_mismatch(self) -> None:
    memo = ShapeMemo()
    memo.single["N"] = 5
    err = check_shape((10,), (SymbolicDim("N+1"),), memo)
    assert "evaluated to 6 but got 10" in err

  def test_complex_expression(self) -> None:
    memo = ShapeMemo()
    memo.single["H"] = 4
    memo.single["W"] = 5
    assert check_shape((20,), (SymbolicDim("H*W"),), memo) == ""

  def test_unbound_name(self) -> None:
    memo = ShapeMemo()
    err = check_shape((5,), (SymbolicDim("X+1"),), memo)
    assert "cannot evaluate" in err

  def test_broadcastable_size_1(self) -> None:
    memo = ShapeMemo()
    memo.single["N"] = 10
    assert check_shape((1,), (SymbolicDim("N+1", broadcastable=True),), memo) == ""


class TestAnonymous:
  def test_matches_any_size(self) -> None:
    memo = ShapeMemo()
    spec = (ANONYMOUS, ANONYMOUS, ANONYMOUS)
    assert check_shape((1, 999, 42), spec, memo) == ""
    # Nothing bound
    assert memo.single == {}

  def test_wrong_rank(self) -> None:
    memo = ShapeMemo()
    err = check_shape((1, 2), (ANONYMOUS,), memo)
    assert "expected 1 dimensions but got 2" in err


class TestAnonymousVariadic:
  def test_zero_dims(self) -> None:
    memo = ShapeMemo()
    spec = (ANONYMOUS_VARIADIC, NamedDim("C"))
    assert check_shape((3,), spec, memo) == ""
    assert memo.single["C"] == 3

  def test_many_dims(self) -> None:
    memo = ShapeMemo()
    spec = (ANONYMOUS_VARIADIC, NamedDim("C"))
    assert check_shape((1, 2, 3, 4), spec, memo) == ""
    assert memo.single["C"] == 4


class TestVariadicDim:
  def test_bind(self) -> None:
    memo = ShapeMemo()
    spec = (VariadicDim("spatial"), NamedDim("C"))
    assert check_shape((28, 28, 3), spec, memo) == ""
    assert memo.variadic["spatial"] == (False, (28, 28))
    assert memo.single["C"] == 3

  def test_consistency(self) -> None:
    memo = ShapeMemo()
    spec = (VariadicDim("spatial"), NamedDim("C"))
    assert check_shape((28, 28, 3), spec, memo) == ""
    assert check_shape((28, 28, 3), spec, memo) == ""
    err = check_shape((32, 32, 3), spec, memo)
    assert "does not match" in err

  def test_zero_dims(self) -> None:
    memo = ShapeMemo()
    spec = (VariadicDim("batch"), NamedDim("N"))
    assert check_shape((5,), spec, memo) == ""
    assert memo.variadic["batch"] == (False, ())
    assert memo.single["N"] == 5

  def test_prefix_and_suffix(self) -> None:
    memo = ShapeMemo()
    spec = (NamedDim("N"), VariadicDim("spatial"), NamedDim("C"))
    assert check_shape((4, 28, 28, 3), spec, memo) == ""
    assert memo.single == {"N": 4, "C": 3}
    assert memo.variadic["spatial"] == (False, (28, 28))

  def test_broadcastable(self) -> None:
    memo = ShapeMemo()
    spec = (VariadicDim("s", broadcastable=True), NamedDim("C"))
    assert check_shape((28, 28, 3), spec, memo) == ""
    # Second shape broadcasts
    assert check_shape((1, 28, 3), spec, memo) == ""


class TestMixedSpecs:
  def test_fixed_and_named(self) -> None:
    memo = ShapeMemo()
    spec = (FixedDim(3), NamedDim("N"), NamedDim("N"))
    assert check_shape((3, 10, 10), spec, memo) == ""

  def test_variadic_with_fixed(self) -> None:
    memo = ShapeMemo()
    spec = (FixedDim(3), ANONYMOUS_VARIADIC, FixedDim(5))
    assert check_shape((3, 1, 2, 5), spec, memo) == ""

  def test_min_rank_check(self) -> None:
    memo = ShapeMemo()
    spec = (NamedDim("N"), ANONYMOUS_VARIADIC, NamedDim("C"))
    err = check_shape((5,), spec, memo)
    assert "expected at least 2 dimensions but got 1" in err


class TestSymbolicEdgeCases:
  def test_symbolic_division_by_zero(self) -> None:
    memo = ShapeMemo()
    memo.single["N"] = 10
    err = check_shape((5,), (SymbolicDim("N//0"),), memo)
    assert "cannot evaluate" in err

  def test_symbolic_float_result_rejects(self) -> None:
    memo = ShapeMemo()
    memo.single["N"] = 3
    err = check_shape((1,), (SymbolicDim("N/2"),), memo)
    assert err != ""

  def test_symbolic_float_result_passes(self) -> None:
    memo = ShapeMemo()
    memo.single["N"] = 4
    assert check_shape((2,), (SymbolicDim("N/2"),), memo) == ""

  def test_empty_shape_empty_spec(self) -> None:
    memo = ShapeMemo()
    assert check_shape((), (), memo) == ""

  def test_symbolic_broadcastable_non_1_requires_match(self) -> None:
    memo = ShapeMemo()
    memo.single["N"] = 5
    err = check_shape((3,), (SymbolicDim("N+1", broadcastable=True),), memo)
    assert "evaluated to 6 but got 3" in err


class TestVariadicEdgeCases:
  def test_variadic_empty_suffix(self) -> None:
    """Variadic at end with no suffix dims."""
    memo = ShapeMemo()
    spec = (NamedDim("N"), VariadicDim("B"))
    assert check_shape((3, 5, 7), spec, memo) == ""
    assert memo.single["N"] == 3
    assert memo.variadic["B"] == (False, (5, 7))

  def test_variadic_mismatch_non_broadcastable(self) -> None:
    """Named variadic that doesn't match and isn't broadcastable."""
    memo = ShapeMemo()
    spec = (VariadicDim("B"), NamedDim("C"))
    assert check_shape((2, 3, 4), spec, memo) == ""
    err = check_shape((5, 6, 4), spec, memo)
    assert "does not match" in err

  def test_variadic_broadcastable_both_ways(self) -> None:
    """Broadcastable variadic: both old and new are broadcastable."""
    memo = ShapeMemo()
    spec = (VariadicDim("B", broadcastable=True), NamedDim("C"))
    assert check_shape((1, 4), spec, memo) == ""
    assert check_shape((3, 4), spec, memo) == ""

  def test_variadic_shape_exact_match_reuses(self) -> None:
    """Same variadic shape on second check should pass."""
    memo = ShapeMemo()
    spec = (VariadicDim("B"), NamedDim("C"))
    assert check_shape((2, 3, 4), spec, memo) == ""
    assert check_shape((2, 3, 4), spec, memo) == ""

  def test_variadic_min_rank_with_prefix_and_suffix(self) -> None:
    """Shape too short for prefix + suffix around variadic."""
    memo = ShapeMemo()
    spec = (NamedDim("N"), VariadicDim("B"), NamedDim("C"), NamedDim("H"))
    err = check_shape((3, 4), spec, memo)
    assert "expected at least 3 dimensions but got 2" in err


class TestNamedDimEdgeCases:
  def test_broadcastable_binds_then_matches(self) -> None:
    """Broadcastable dim with non-1 size should bind, then match same size."""
    memo = ShapeMemo()
    spec = (NamedDim("N", broadcastable=True),)
    assert check_shape((5,), spec, memo) == ""
    assert memo.single["N"] == 5
    assert check_shape((5,), spec, memo) == ""

  def test_broadcastable_binds_then_size_1_passes(self) -> None:
    """Broadcastable dim bound to 5 should still accept size 1."""
    memo = ShapeMemo()
    spec = (NamedDim("N", broadcastable=True),)
    assert check_shape((5,), spec, memo) == ""
    assert check_shape((1,), spec, memo) == ""

  def test_broadcastable_binds_then_wrong_size_fails(self) -> None:
    """Broadcastable dim bound to 5 should reject non-1, non-5."""
    memo = ShapeMemo()
    spec = (NamedDim("N", broadcastable=True),)
    assert check_shape((5,), spec, memo) == ""
    err = check_shape((3,), spec, memo)
    assert "expected 5 but got 3" in err

  def test_multiple_anonymous_no_rank_check_conflict(self) -> None:
    """Multiple anonymous dims don't interfere with each other."""
    memo = ShapeMemo()
    spec = (ANONYMOUS, ANONYMOUS)
    assert check_shape((99, 1), spec, memo) == ""
    assert memo.single == {}


class TestReprOutput:
  def test_named_dim_repr(self) -> None:
    assert repr(NamedDim("N")) == "N"
    assert repr(NamedDim("N", broadcastable=True)) == "+N"

  def test_fixed_dim_repr(self) -> None:
    assert repr(FixedDim(3)) == "3"

  def test_symbolic_dim_repr(self) -> None:
    assert repr(SymbolicDim("N+1")) == "N+1"

  def test_variadic_dim_repr(self) -> None:
    assert repr(VariadicDim("B")) == "~B"
    assert repr(VariadicDim("B", broadcastable=True)) == "~+B"

  def test_anonymous_repr(self) -> None:
    assert repr(ANONYMOUS) == "__"
    assert repr(ANONYMOUS_VARIADIC) == "~__"
