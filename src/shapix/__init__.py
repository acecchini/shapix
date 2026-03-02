"""Shapix — elegant runtime shape and dtype checking for array annotations.

Works with standard ``@beartype`` decorators and ``beartype.claw`` import hooks.
No custom decorator required for basic usage.

Quick start::

    from beartype import beartype
    from shapix import N, C, H, W
    from shapix.numpy import F32

    @beartype
    def conv(x: F32[N, C, H, W]) -> F32[N, C, H, W]:
        ...

Dimension symbols (``N``, ``C``, ``H``, ``W``, …) are bound on first use
within a function call and enforced on subsequent occurrences. This gives
automatic cross-argument shape consistency with no extra boilerplate.

Exports
-------
Dimension symbols
    ``B``, ``N``, ``P``, ``L``, ``C``, ``H``, ``W``, ``T`` — named dimensions.
    ``vB``, ``vN``, ``vL``, ``vC`` — variadic (match zero or more dims).
    ``bN``, ``bL``, ``bC`` — broadcastable (size 1 always matches).
    ``_B``, ``_N``, ``_L``, ``_C``, ``__`` — anonymous (match any, no binding).
    ``Scalar`` — scalar (no dimensions), ``Any`` — anonymous variadic.

Classes
    :class:`Dimension` — create custom dimension symbols with arithmetic support.
    :class:`DtypeSpec` — describe a set of allowed dtypes by string name.

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
from ._dimensions import Any as Any
from ._dimensions import B as B
from ._dimensions import C as C
from ._dimensions import Dimension as Dimension
from ._dimensions import H as H
from ._dimensions import L as L
from ._dimensions import N as N
from ._dimensions import P as P
from ._dimensions import Scalar as Scalar
from ._dimensions import T as T
from ._dimensions import W as W
from ._dimensions import __ as __
from ._dimensions import _B as _B
from ._dimensions import _C as _C
from ._dimensions import _L as _L
from ._dimensions import _N as _N
from ._dimensions import bC as bC
from ._dimensions import bL as bL
from ._dimensions import bN as bN
from ._dimensions import vB as vB
from ._dimensions import vC as vC
from ._dimensions import vL as vL
from ._dimensions import vN as vN
from ._dtypes import DtypeSpec as DtypeSpec
