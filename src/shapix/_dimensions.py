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

Use ``Value("expr")`` for dimensions that depend on runtime parameters or
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


# ---------------------------------------------------------------------------
# Arithmetic helper
# ---------------------------------------------------------------------------


def _binop(left: object, op: str, right: object) -> Dimension:
  """Build a binary expression.  Returns ``Value`` if either operand is one."""
  if isinstance(left, Dimension) and str(left) == "":
    msg = "Cannot use Scalar in arithmetic expressions"
    raise TypeError(msg)
  if isinstance(right, Dimension) and str(right) == "":
    msg = "Cannot use Scalar in arithmetic expressions"
    raise TypeError(msg)
  if isinstance(left, Value) or isinstance(right, Value):
    return Value(f"({left}{op}{right})")  # type: ignore[return-value]
  return Dimension(f"({left}{op}{right})")


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

  def __new__(cls, value: str | int, /) -> Dimension:
    if isinstance(value, bool):
      msg = f"Dimension does not accept bool ({value!r}); use an int or str"
      raise TypeError(msg)
    if not isinstance(value, (str, int)):  # pyright: ignore[reportUnnecessaryIsInstance]
      msg = f"Dimension requires str or int, got {type(value).__name__}"  # type: ignore[unreachable]
      raise TypeError(msg)
    return str.__new__(cls, str(value) if isinstance(value, int) else value)

  # ------------------------------------------------------------------
  # Arithmetic → symbolic dimension expressions
  # ------------------------------------------------------------------

  def __add__(self, other: object) -> Dimension:
    return _binop(self, "+", other)

  def __radd__(self, other: object) -> Dimension:
    return _binop(other, "+", self)

  def __sub__(self, other: object) -> Dimension:
    return _binop(self, "-", other)

  def __rsub__(self, other: object) -> Dimension:
    return _binop(other, "-", self)

  def __mul__(self, other: object) -> Dimension:
    return _binop(self, "*", other)

  def __rmul__(self, other: object) -> Dimension:
    return _binop(other, "*", self)

  def __truediv__(self, other: object) -> Dimension:
    return _binop(self, "/", other)

  def __rtruediv__(self, other: object) -> Dimension:
    return _binop(other, "/", self)

  def __floordiv__(self, other: object) -> Dimension:
    return _binop(self, "//", other)

  def __rfloordiv__(self, other: object) -> Dimension:
    return _binop(other, "//", self)

  def __pow__(self, other: object) -> Dimension:
    return _binop(self, "**", other)

  def __rpow__(self, other: object) -> Dimension:
    return _binop(other, "**", self)

  def __mod__(self, other: object) -> Dimension:
    return _binop(self, "%", other)

  def __rmod__(self, other: object) -> Dimension:
    return _binop(other, "%", self)

  def __neg__(self) -> Dimension:
    if str(self) == "":
      msg = "Cannot use Scalar in arithmetic expressions"
      raise TypeError(msg)
    return Dimension(f"-{self}")

  def __invert__(self) -> Dimension:
    """``~N`` → variadic dimension (matches zero or more contiguous dims)."""
    raw = str(self)
    if raw.startswith("~"):
      return self
    if raw.lstrip("-").isdigit():
      msg = f"Cannot apply ~ (variadic) to a fixed numeric dimension {raw!r}"
      raise TypeError(msg)
    if not raw.isidentifier() and not (raw.startswith("+") and raw[1:].isidentifier()):
      msg = f"Cannot apply ~ (variadic) to expression {raw!r}; variadic requires a named dimension"
      raise TypeError(msg)
    return Dimension(f"~{raw}")

  def __pos__(self) -> Dimension:
    """``+N`` → broadcastable dimension (size 1 always matches)."""
    raw = str(self)
    if raw.startswith("+"):
      return self
    if raw == "":
      msg = "Cannot apply + (broadcastable) to Scalar"
      raise TypeError(msg)
    if raw == "__":
      msg = "Cannot apply + (broadcastable) to anonymous dimension __"
      raise TypeError(msg)
    if raw.startswith("~"):
      msg = (
        f"Cannot apply + (broadcastable) to variadic dimension {raw!r}; "
        "use ~+name for broadcastable variadic"
      )
      raise TypeError(msg)
    return Dimension(f"+{raw}")

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
        name = rest[1:]
        if name.lstrip("-").isdigit():
          msg = f"Cannot use variadic (~) with fixed numeric dimension {name!r}"
          raise TypeError(msg)
        if not name.isidentifier():
          msg = f"Cannot use variadic (~) with expression {name!r}; variadic requires a named dimension"
          raise TypeError(msg)
        return VariadicDim(name, broadcastable=True)
      if rest.lstrip("-").isdigit():
        msg = f"Cannot use variadic (~) with fixed numeric dimension {rest!r}"
        raise TypeError(msg)
      if not rest.isidentifier():
        msg = f"Cannot use variadic (~) with expression {rest!r}; variadic requires a named dimension"
        raise TypeError(msg)
      return VariadicDim(rest, broadcastable=False)

    # Broadcastable: +name or +number
    if raw.startswith("+"):
      rest = raw[1:]
      if not rest:
        msg = "Invalid dimension '+'; broadcastable requires a base dimension"
        raise TypeError(msg)
      if rest == "__":
        msg = "Cannot use broadcastable (+) with anonymous dimension __"
        raise TypeError(msg)
      if rest.lstrip("-").isdigit():
        size = int(rest)
        if size < 0:
          msg = f"Negative dimension {size} is invalid; array shapes are non-negative"
          raise TypeError(msg)
        return FixedDim(size, broadcastable=True)
      if rest.isidentifier():
        return NamedDim(rest, broadcastable=True)
      return SymbolicDim(rest, broadcastable=True)

    # Pure integer
    if raw.lstrip("-").isdigit():
      return FixedDim(int(raw))

    # Simple identifier → named dim
    if raw.isidentifier():
      return NamedDim(raw, broadcastable=False)

    # Everything else → symbolic expression (arithmetic result)
    return SymbolicDim(raw)


class _ValueExpr(str):
  """Runtime value expression used by ``Value("...")`` in shape subscripts.

  Subclasses :class:`str` (like :class:`Dimension`) with the raw expression as
  its string value.  This keeps ``Dimension`` arithmetic return types compatible
  with the ``str`` base class and makes ``str(value_expr)`` produce the raw
  expression.

  Any arithmetic involving a ``_ValueExpr`` produces another ``_ValueExpr``,
  since the result still needs runtime scope access.
  """

  broadcastable: bool

  def __new__(cls, expr: object, *, broadcastable: bool = False) -> _ValueExpr:
    if not isinstance(expr, str):
      msg = "Value(...) expects a string expression"
      raise TypeError(msg)
    obj = str.__new__(cls, expr)
    obj.broadcastable = broadcastable
    return obj

  # ------------------------------------------------------------------
  # Unary
  # ------------------------------------------------------------------

  def __pos__(self) -> _ValueExpr:
    """``+Value("x")`` → broadcastable value dim (size 1 always matches)."""
    if self.broadcastable:
      return self
    return _ValueExpr(str(self), broadcastable=True)

  def __neg__(self) -> _ValueExpr:
    return _ValueExpr(f"(-{self})")

  def __invert__(self) -> tp.Never:
    """``~Value(...)`` is not supported — variadic requires a name."""
    msg = "Value expressions cannot be variadic (~); use a named Dimension instead"
    raise TypeError(msg)

  # ------------------------------------------------------------------
  # Arithmetic → _ValueExpr (always, since scope is needed)
  # ------------------------------------------------------------------

  def __add__(self, other: object) -> _ValueExpr:
    return _ValueExpr(f"({self}+{other})")

  def __radd__(self, other: object) -> _ValueExpr:
    return _ValueExpr(f"({other}+{self})")

  def __sub__(self, other: object) -> _ValueExpr:
    return _ValueExpr(f"({self}-{other})")

  def __rsub__(self, other: object) -> _ValueExpr:
    return _ValueExpr(f"({other}-{self})")

  def __mul__(self, other: object) -> _ValueExpr:
    return _ValueExpr(f"({self}*{other})")

  def __rmul__(self, other: object) -> _ValueExpr:
    return _ValueExpr(f"({other}*{self})")

  def __truediv__(self, other: object) -> _ValueExpr:
    return _ValueExpr(f"({self}/{other})")

  def __rtruediv__(self, other: object) -> _ValueExpr:
    return _ValueExpr(f"({other}/{self})")

  def __floordiv__(self, other: object) -> _ValueExpr:
    return _ValueExpr(f"({self}//{other})")

  def __rfloordiv__(self, other: object) -> _ValueExpr:
    return _ValueExpr(f"({other}//{self})")

  def __pow__(self, other: object) -> _ValueExpr:
    return _ValueExpr(f"({self}**{other})")

  def __rpow__(self, other: object) -> _ValueExpr:
    return _ValueExpr(f"({other}**{self})")

  def __mod__(self, other: object) -> _ValueExpr:
    return _ValueExpr(f"({self}%{other})")

  def __rmod__(self, other: object) -> _ValueExpr:
    return _ValueExpr(f"({other}%{self})")

  # ------------------------------------------------------------------
  # Dim spec + repr
  # ------------------------------------------------------------------

  @property
  def _dim_spec(self) -> ValueDim:
    return ValueDim(str(self), broadcastable=self.broadcastable)

  def __repr__(self) -> str:
    prefix = "+" if self.broadcastable else ""
    return f'{prefix}Value("{self}")'


# ---------------------------------------------------------------------------
# Pre-defined dimension symbols
# ---------------------------------------------------------------------------

if tp.TYPE_CHECKING:

  class Value:
    def __new__(cls, expr: str) -> Dimension: ...  # type: ignore[misc]

    def __pos__(self) -> Dimension: ...

  Scalar = tp.Literal["Scalar"]
  B = tp.Literal["B"]
  N = tp.Literal["N"]
  P = tp.Literal["P"]
  L = tp.Literal["L"]
  C = tp.Literal["C"]
  D = tp.Literal["D"]
  K = tp.Literal["K"]
  H = tp.Literal["H"]
  W = tp.Literal["W"]
  __ = tp.Literal["__"]

else:

  class Value(_ValueExpr):
    """Explicit runtime value expression for shape annotations.

    Use ``Value("size")`` or ``Value("self.some_value + 3")`` when a shape
    depends on a runtime parameter rather than a previously bound dimension.
    """

    __slots__ = ()

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
