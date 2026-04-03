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

from ._imports import require_module
from ._tree import Structure as Structure
from ._tree import _TreeFactory

__all__ = ["Structure", "Tree"]


def _get_optree() -> tp.Any:
  return require_module(
    "optree",
    install_hint=(
      "shapix.optree requires 'optree' at runtime. "
      "Install it with `pip install optree`."
    ),
  )


if tp.TYPE_CHECKING:
  _T = tp.TypeVar("_T")

  class Tree(tp.Generic[_T]):
    """Static type stub — ``Tree[LeafType]`` for type checkers."""

    def __class_getitem__(cls, item: object) -> type: ...

else:
  Tree = _TreeFactory(_get_optree, name="Tree")
