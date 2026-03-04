"""Shapix — elegant runtime shape and dtype checking for array annotations.

Works with standard ``@beartype`` decorators and ``beartype.claw`` import hooks.
No custom decorator required for basic usage.

Quick start::

    from beartype import beartype
    from shapix import N, C, H, W
    from shapix.numpy import F32


    @beartype
    def conv(x: F32[N, C, H, W]) -> F32[N, C, H, W]: ...

Dimension symbols (``N``, ``C``, ``H``, ``W``, …) are bound on first use
within a function call and enforced on subsequent occurrences. This gives
automatic cross-argument shape consistency with no extra boilerplate.

Exports
-------
Dimension symbols
    ``B``, ``N``, ``P``, ``L``, ``C``, ``H``, ``W`` — named dimensions.
    ``__`` — anonymous (match any single dim, no binding).
    ``Scalar`` — scalar (no dimensions).

Tree structure symbols
    ``T``, ``S`` — named tree structure symbols.
    :class:`Structure` — create custom structure symbols.

Unary operators (apply to any dimension)
    ``~N`` — variadic (match zero or more contiguous dims).
    ``+N`` — broadcastable (size 1 always matches).
    ``~__`` — anonymous variadic (match any number of dims, no binding).
    ``...`` — alias for ``~__`` (Ellipsis in subscripts).

Classes
    :class:`Dimension` — create custom dimension symbols with arithmetic support.
    :class:`DtypeSpec` — describe a set of allowed dtypes by string name.
    ``Tree`` — subscriptable tree annotation, import from ``shapix.optree``
    or ``shapix.jax``.

Functions
    :func:`make_array_type` — create subscriptable array type factories for
    custom array classes.
    :func:`check` — optional decorator for explicit memo management.

Context managers
    :class:`check_context` — shared dimension memo for manual
    ``is_bearable()`` checks.
"""

from ._array_types import make_array_type as make_array_type
from ._decorator import check as check
from ._decorator import check_context as check_context
from ._tree import S as S
from ._tree import Structure as Structure
from ._tree import T as T
from ._dimensions import __ as __
from ._dimensions import B as B
from ._dimensions import C as C
from ._dimensions import Dimension as Dimension
from ._dimensions import H as H
from ._dimensions import L as L
from ._dimensions import N as N
from ._dimensions import P as P
from ._dimensions import Scalar as Scalar
from ._dimensions import W as W
from ._dtypes import DtypeSpec as DtypeSpec
