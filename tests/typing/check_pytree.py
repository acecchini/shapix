"""Verify PyTree type annotations work with type checkers.

Tested with: pyright, mypy, ty
"""

from __future__ import annotations

import typing as tp

from shapix import Dimension, PyTree, Structure

# ---------------------------------------------------------------------------
# Import validation
# ---------------------------------------------------------------------------

_pt = PyTree  # PyTree should be importable
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
# Function signatures with PyTree
# ---------------------------------------------------------------------------


def process(data: PyTree[object]) -> PyTree[object]:
  return data


def transform(x: PyTree[object], y: PyTree[object]) -> PyTree[object]:
  return x


def mixed(x: PyTree[object], scale: float) -> PyTree[object]:
  return x


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------


def identity(x: PyTree[object]) -> PyTree[object]:
  return x


def to_none(x: PyTree[object]) -> None:
  pass
