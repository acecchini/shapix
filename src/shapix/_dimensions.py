"""Dimension symbols for readable array shape annotations.

Dimensions are the building blocks of shapix type annotations. Import
pre-defined symbols or create your own::

    from shapix import N, C, H, W, Dimension

    Vocab = Dimension("Vocab")
    Embed = Dimension("Embed")

Dimensions support full Python arithmetic to express derived shapes::

    from shapix import N

    @beartype
    def pad(x: Float32Array[N]) -> Float32Array[N + 2]: ...
    def flatten(x: Float32Array[N, C]) -> Float32Array[N * C]: ...

Prefixes control matching behaviour:

- *(none)* — Named: bind on first use, enforce on subsequent (``N``, ``C``).
- ``*`` — Variadic: match zero or more contiguous dims (``sB`` = ``*B``).
- ``#`` — Broadcastable: size 1 always matches (``hN`` = ``#N``).
- ``_`` — Anonymous: match any size, no binding (``_N``, ``__``).
- ``...`` — Anonymous variadic: match any number of dims (``Any``).

Plain ``int`` values (e.g. ``3``) are also accepted as fixed dimension sizes
when subscripting array types.
"""

from __future__ import annotations

from ._shape import (
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
    # Variadic
    "sB",
    "sN",
    "sL",
    "sC",
    # Broadcastable
    "hN",
    "hL",
    "hC",
    # Anonymous
    "_B",
    "_N",
    "_L",
    "_C",
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
    # Construction helpers
    # ------------------------------------------------------------------

    def __class_getitem__(cls, item: str | tuple[str, ...]) -> Dimension:  # type: ignore[override]
        if isinstance(item, tuple):
            return cls(" ".join(cls(i) for i in item))
        return cls(item)

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

    def __abs__(self) -> Dimension:
        return self.__class__(f"abs({self})")

    # ------------------------------------------------------------------
    # Conversion to internal dim spec
    # ------------------------------------------------------------------

    @property
    def _dim_spec(self) -> NamedDim | FixedDim | SymbolicDim | VariadicDim:
        """Convert this symbol to an internal shape-spec object."""
        raw = str(self)

        # Empty → scalar (no dims)
        if raw == "":
            return FixedDim(-1)  # sentinel, handled by array factory

        # Ellipsis → anonymous variadic
        if raw == "...":
            return VariadicDim("_", broadcastable=False)

        # Variadic: *name or *#name
        if raw.startswith("*"):
            rest = raw[1:]
            if rest.startswith("#"):
                return VariadicDim(rest[1:], broadcastable=True)
            return VariadicDim(rest, broadcastable=False)

        # Broadcastable: #name
        if raw.startswith("#"):
            return NamedDim(raw[1:], broadcastable=True)

        # Anonymous: bare _ or _name
        if raw == "_":
            return NamedDim("_", broadcastable=False)  # treated as anonymous in check

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

# Common named dimensions
Scalar = Dimension("")
B = Dimension("B")
N = Dimension("N")
P = Dimension("P")
L = Dimension("L")
C = Dimension("C")
H = Dimension("H")
W = Dimension("W")

# Variadic (match zero or more dims)
sB = Dimension("*B")
sN = Dimension("*N")
sL = Dimension("*L")
sC = Dimension("*C")

# Broadcastable (size 1 always matches)
hN = Dimension("#N")
hL = Dimension("#L")
hC = Dimension("#C")

# Anonymous (match anything, no binding)
_B = Dimension("_B")
_N = Dimension("_N")
_L = Dimension("_L")
_C = Dimension("_C")
__ = Dimension("_")

# Special
Any = Dimension("...")

# Trees
T = Dimension("T")
