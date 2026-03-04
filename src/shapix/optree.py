# pyright: reportMissingImports=false
"""Tree annotations backed by the ``optree`` library.

Usage::

    from shapix import N, C, T, S, Structure
    from shapix.optree import Tree
    from shapix.numpy import F32


    @beartype
    def f(x: Tree[F32[N, C]], y: Tree[F32[N, C], T]): ...

Requires ``optree``. Install with ``pip install optree``.
"""

from __future__ import annotations

import typing as tp

from ._tree import Structure as Structure
from ._tree import _TreeFactory

__all__ = ["Tree", "Structure"]


def _get_optree() -> tp.Any:
  import optree

  return optree


if tp.TYPE_CHECKING:
  from ._tree import Tree as Tree
else:
  Tree = _TreeFactory(_get_optree, name="Tree")
