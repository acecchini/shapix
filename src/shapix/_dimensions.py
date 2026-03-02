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
    VariadicDim,
)

__all__ = [
    "Dimension",
    # Common named dimensions
    "B",
    "N",
    "P",
    "L",
    "C",
    "H",
    "W",
    "T",
    # Anonymous
    "__",
    # Special
    "Scalar",
    "Any",
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
        if raw == "_":
            return ANONYMOUS

        # Variadic: ~name or ~+name
        if raw.startswith("~"):
            rest = raw[1:]
            if rest == "_" or rest.startswith("_"):
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


# ---------------------------------------------------------------------------
# Pre-defined dimension symbols
# ---------------------------------------------------------------------------

if tp.TYPE_CHECKING:
    # Type aliases so pyright accepts them in type expressions like
    # ``F32[N, C]``.  At type-checking time the dims resolve to ``int``
    # which is harmless — the *runtime* ``Dimension`` instances carry the
    # actual semantics.
    type Scalar = int
    type B = int
    type N = int
    type P = int
    type L = int
    type C = int
    type H = int
    type W = int
    type T = int
    type __ = int
    type Any = int
else:
    # Common named dimensions
    Scalar = Dimension("")
    B = Dimension("B")
    N = Dimension("N")
    P = Dimension("P")
    L = Dimension("L")
    C = Dimension("C")
    H = Dimension("H")
    W = Dimension("W")
    T = Dimension("T")

    # Anonymous (match anything, no binding)
    __ = Dimension("_")

    # Anonymous variadic (match any number of dims, no binding)
    Any = Dimension("~_")
