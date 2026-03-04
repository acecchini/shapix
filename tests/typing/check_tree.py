"""Verify Tree type annotations work with type checkers.

Tested with: pyright, mypy, ty
"""

from __future__ import annotations

import typing as tp

from shapix import Dimension, Structure
from shapix.optree import Tree

# ---------------------------------------------------------------------------
# Import validation
# ---------------------------------------------------------------------------

_pt = Tree  # Tree should be importable
_st = Structure  # Structure should be importable

# ---------------------------------------------------------------------------
# Custom dimensions for type-level usage
# ---------------------------------------------------------------------------

if tp.TYPE_CHECKING:
  N: Dimension
  C: Dimension
else:
  N = Dimension("N")
  C = Dimension("C")

# ---------------------------------------------------------------------------
# Structure symbols
# ---------------------------------------------------------------------------

if tp.TYPE_CHECKING:
  T: Structure
  S: Structure
else:
  T = Structure("T")
  S = Structure("S")

# ---------------------------------------------------------------------------
# Function signatures with Tree
# ---------------------------------------------------------------------------


def process(data: Tree[object]) -> Tree[object]:
  return data


def transform(x: Tree[object], y: Tree[object]) -> Tree[object]:
  return x


def mixed(x: Tree[object], scale: float) -> Tree[object]:
  return x


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------


def identity(x: Tree[object]) -> Tree[object]:
  return x


def to_none(x: Tree[object]) -> None:
  pass
