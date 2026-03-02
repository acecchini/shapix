"""Tests for _dimensions.py — dimension symbols and arithmetic."""

from __future__ import annotations

from shapix._dimensions import Dimension
from shapix._shape import FixedDim, NamedDim, SymbolicDim, VariadicDim


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

    def test_mul_int(self) -> None:
        d = 2 * Dimension("N")
        assert str(d) == "(2*N)"

    def test_truediv(self) -> None:
        d = Dimension("N") / 2
        assert str(d) == "(N/2)"

    def test_floordiv(self) -> None:
        d = Dimension("N") // 2
        assert str(d) == "(N//2)"

    def test_pow(self) -> None:
        d = Dimension("N") ** 2
        assert str(d) == "(N**2)"

    def test_mod(self) -> None:
        d = Dimension("N") % 3
        assert str(d) == "(N%3)"

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
        d = Dimension("#N")
        spec = d._dim_spec
        assert isinstance(spec, NamedDim)
        assert spec.name == "N"
        assert spec.broadcastable is True

    def test_variadic(self) -> None:
        d = Dimension("*batch")
        spec = d._dim_spec
        assert isinstance(spec, VariadicDim)
        assert spec.name == "batch"
        assert spec.broadcastable is False

    def test_variadic_broadcastable(self) -> None:
        d = Dimension("*#batch")
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
        d = Dimension("_")
        spec = d._dim_spec
        assert isinstance(spec, NamedDim)
        assert spec.name == "_"

    def test_ellipsis(self) -> None:
        d = Dimension("...")
        spec = d._dim_spec
        assert isinstance(spec, VariadicDim)
        assert spec.name == "_"
