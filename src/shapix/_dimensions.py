"""Dimension symbols for readable array shape annotations.

Dimensions are the building blocks of shapix type annotations. Import
pre-defined symbols or create your own::

    from shapix import N, C, H, W, Dimension

    Vocab = Dimension("Vocab")
    Embed = Dimension("Embed")

Dimensions support full Python arithmetic to express derived shapes::

    from shapix import N
    from shapix.numpy import F32


    @beartype
    def pad(x: F32[N]) -> F32[N + 2]: ...
    def flatten(x: F32[N, C]) -> F32[N * C]: ...

Unary operators control matching behaviour:

- ``~N`` — Variadic: match zero or more contiguous dims.
- ``+N`` — Broadcastable: size 1 always matches.
- ``__`` — Anonymous: match any single dim, no binding.
- ``~__`` — Anonymous variadic: match any number of dims, no binding.

Plain ``int`` values (e.g. ``3``) are also accepted as fixed dimension sizes
when subscripting array types.

Use ``Value["expr"]`` for dimensions that depend on runtime parameters or
``self`` attributes rather than previously bound shape names.
"""

from __future__ import annotations

import typing as tp

from ._shape import (
  ANONYMOUS,
  ANONYMOUS_VARIADIC,
  DimSpec,
  FixedDim,
  NamedDim,
  SymbolicDim,
  ValueDim,
  VariadicDim,
)

__all__ = [
  "Dimension",
  "Value",
  # Common named dimensions
  "B",
  "N",
  "P",
  "L",
  "C",
  "D",
  "K",
  "H",
  "W",
  # Anonymous
  "__",
  # Special
  "Scalar",
]


class Dimension(str):
  """A named dimension symbol that doubles as a shape spec element.

  Behaves like a ``str`` for display but carries semantic meaning for
  runtime shape checking.  Arithmetic produces :class:`SymbolicDim`-backed
  expressions::

      >>> N = Dimension("N")
      >>> N + 1
      Dimension('(N+1)')
      >>> 2 * N
      Dimension('(2*N)')
  """

  # ------------------------------------------------------------------
  # Arithmetic → symbolic dimension expressions
  # ------------------------------------------------------------------

  def __add__(self, other: object) -> Dimension:
    return self.__class__(f"({self}+{other})")

  def __radd__(self, other: object) -> Dimension:
    return self.__class__(f"({other}+{self})")

  def __sub__(self, other: object) -> Dimension:
    return self.__class__(f"({self}-{other})")

  def __rsub__(self, other: object) -> Dimension:
    return self.__class__(f"({other}-{self})")

  def __mul__(self, other: object) -> Dimension:
    return self.__class__(f"({self}*{other})")

  def __rmul__(self, other: object) -> Dimension:
    return self.__class__(f"({other}*{self})")

  def __truediv__(self, other: object) -> Dimension:
    return self.__class__(f"({self}/{other})")

  def __rtruediv__(self, other: object) -> Dimension:
    return self.__class__(f"({other}/{self})")

  def __floordiv__(self, other: object) -> Dimension:
    return self.__class__(f"({self}//{other})")

  def __rfloordiv__(self, other: object) -> Dimension:
    return self.__class__(f"({other}//{self})")

  def __pow__(self, other: object) -> Dimension:
    return self.__class__(f"({self}**{other})")

  def __rpow__(self, other: object) -> Dimension:
    return self.__class__(f"({other}**{self})")

  def __mod__(self, other: object) -> Dimension:
    return self.__class__(f"({self}%{other})")

  def __rmod__(self, other: object) -> Dimension:
    return self.__class__(f"({other}%{self})")

  def __neg__(self) -> Dimension:
    return self.__class__(f"-{self}")

  def __invert__(self) -> Dimension:
    """``~N`` → variadic dimension (matches zero or more contiguous dims)."""
    raw = str(self)
    if raw.startswith("~"):
      return self
    return self.__class__(f"~{raw}")

  def __pos__(self) -> Dimension:
    """``+N`` → broadcastable dimension (size 1 always matches)."""
    raw = str(self)
    if raw.startswith("+"):
      return self
    return self.__class__(f"+{raw}")

  # ------------------------------------------------------------------
  # Conversion to internal dim spec
  # ------------------------------------------------------------------

  @property
  def _dim_spec(self) -> DimSpec | None:
    """Convert this symbol to an internal shape-spec object.

    Returns ``None`` for the scalar sentinel (empty string).
    """
    raw = str(self)

    # Empty → scalar (no dims)
    if raw == "":
      return None

    # Anonymous single dim
    if raw == "__":
      return ANONYMOUS

    # Variadic: ~name or ~+name
    if raw.startswith("~"):
      rest = raw[1:]
      if rest == "__":
        return ANONYMOUS_VARIADIC
      if rest.startswith("+"):
        return VariadicDim(rest[1:], broadcastable=True)
      return VariadicDim(rest, broadcastable=False)

    # Broadcastable: +name
    if raw.startswith("+"):
      return NamedDim(raw[1:], broadcastable=True)

    # Pure integer
    if raw.lstrip("-").isdigit():
      return FixedDim(int(raw))

    # Simple identifier → named dim
    if raw.isidentifier():
      return NamedDim(raw, broadcastable=False)

    # Everything else → symbolic expression (arithmetic result)
    return SymbolicDim(raw)


class _ValueExpr:
  """Runtime value expression used by ``Value[...]`` in shape subscripts."""

  __slots__ = ("expr", "broadcastable")

  def __init__(self, expr: str, *, broadcastable: bool = False) -> None:
    self.expr = expr
    self.broadcastable = broadcastable

  def __pos__(self) -> _ValueExpr:
    if self.broadcastable:
      return self
    return self.__class__(self.expr, broadcastable=True)

  @property
  def _dim_spec(self) -> ValueDim:
    return ValueDim(self.expr, broadcastable=self.broadcastable)

  def __repr__(self) -> str:
    prefix = "+" if self.broadcastable else ""
    return f'{prefix}Value["{self.expr}"]'


# ---------------------------------------------------------------------------
# Pre-defined dimension symbols
# ---------------------------------------------------------------------------

if tp.TYPE_CHECKING:
  # Declared as ``Dimension`` so type checkers see the full operator set
  # (``__add__``, ``__invert__``, ``__pos__``, …) and ``F32[N, C]`` subscripts.
  class Value:
    @classmethod
    def __class_getitem__(cls, expr: str) -> Dimension: ...

  Scalar: Dimension
  B: Dimension
  N: Dimension
  P: Dimension
  L: Dimension
  C: Dimension
  D: Dimension
  K: Dimension
  H: Dimension
  W: Dimension
  __: Dimension
else:

  class Value:
    """Explicit runtime value expression for shape annotations.

    Use ``Value["size"]`` or ``Value["self.some_value + 3"]`` when a shape
    depends on a runtime parameter rather than a previously bound dimension.
    """

    __slots__ = ()

    def __class_getitem__(cls, expr: object) -> _ValueExpr:  # type: ignore[misc]
      if not isinstance(expr, str):
        msg = "Value[...] expects a string expression"
        raise TypeError(msg)
      return _ValueExpr(expr)

  # Common named dimensions
  Scalar = Dimension("")
  B = Dimension("B")
  N = Dimension("N")
  P = Dimension("P")
  L = Dimension("L")
  C = Dimension("C")
  D = Dimension("D")
  K = Dimension("K")
  H = Dimension("H")
  W = Dimension("W")

  # Anonymous (match anything, no binding)
  __ = Dimension("__")
