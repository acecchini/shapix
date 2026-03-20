"""Shape specification types and matching logic.

This module defines the internal dimension spec types that the shape checker
operates on, and the :func:`check_shape` function that validates a concrete
shape tuple against a spec tuple while maintaining dimension bindings in a
:class:`~shapix._memo.ShapeMemo`.

Dimension spec types:

- :class:`NamedDim` — binds to a name on first occurrence, enforces match on
  subsequent ones (``N``, ``batch``). Optionally broadcastable (size 1 OK).
- :class:`FixedDim` — must match an exact integer (``3``, ``224``).
- :class:`SymbolicDim` — arithmetic expression evaluated against bound
  dimensions (``N+1``, ``H*W``). Optionally broadcastable.
- :class:`ValueDim` — arithmetic expression evaluated against runtime call
  scope (``Value("size")``, ``Value("self.width + 3")``). Optionally
  broadcastable.
- :class:`VariadicDim` — matches zero or more contiguous dims and binds the
  matched sub-shape (``~spatial``). Optionally broadcastable.
- :data:`ANONYMOUS` — matches any single dim without binding (``__``).
- :data:`ANONYMOUS_VARIADIC` — matches any number of dims without binding
  (``~__``).

The :func:`check_shape` function returns ``""`` on success or a human-readable
error message explaining the mismatch.
"""

from __future__ import annotations

import ast
import functools
import numbers
import operator
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ._memo import ShapeMemo

__all__ = [
  "NamedDim",
  "FixedDim",
  "SymbolicDim",
  "ValueDim",
  "VariadicDim",
  "ANONYMOUS",
  "ANONYMOUS_VARIADIC",
  "DimSpec",
  "check_shape",
]


# ---------------------------------------------------------------------------
# Dimension spec types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NamedDim:
  """A named dimension that binds to / validates against the memo."""

  name: str
  broadcastable: bool = False

  def __repr__(self) -> str:
    prefix = "+" if self.broadcastable else ""
    return f"{prefix}{self.name}"


@dataclass(frozen=True, slots=True)
class FixedDim:
  """A dimension that must match an exact size."""

  size: int
  broadcastable: bool = False

  def __repr__(self) -> str:
    prefix = "+" if self.broadcastable else ""
    return f"{prefix}{self.size}"


@dataclass(frozen=True, slots=True)
class SymbolicDim:
  """An arithmetic expression evaluated against bound dimensions."""

  expr: str
  broadcastable: bool = False

  def __repr__(self) -> str:
    return self.expr


@dataclass(frozen=True, slots=True)
class ValueDim:
  """A runtime value expression evaluated against the current call scope."""

  expr: str
  broadcastable: bool = False

  def __repr__(self) -> str:
    prefix = "+" if self.broadcastable else ""
    return f'{prefix}Value("{self.expr}")'


@dataclass(frozen=True, slots=True)
class VariadicDim:
  """Matches zero or more contiguous dimensions."""

  name: str
  broadcastable: bool = False

  def __repr__(self) -> str:
    prefix = "+" if self.broadcastable else ""
    return f"~{prefix}{self.name}"


class _Anonymous:
  """Matches any single dimension without binding."""

  __slots__ = ()

  def __repr__(self) -> str:
    return "__"


class _AnonymousVariadic:
  """Matches any number of dimensions without binding."""

  __slots__ = ()

  def __repr__(self) -> str:
    return "~__"


ANONYMOUS = _Anonymous()
ANONYMOUS_VARIADIC = _AnonymousVariadic()

type DimSpec = (
  NamedDim
  | FixedDim
  | SymbolicDim
  | ValueDim
  | VariadicDim
  | _Anonymous
  | _AnonymousVariadic
)


_BINOPS: dict[type, Callable[..., object]] = {
  ast.Add: operator.add,
  ast.Sub: operator.sub,
  ast.Mult: operator.mul,
  ast.Div: operator.truediv,
  ast.FloorDiv: operator.floordiv,
  ast.Pow: operator.pow,
  ast.Mod: operator.mod,
}
_UNARYOPS: dict[type, Callable[..., object]] = {
  ast.UAdd: operator.pos,
  ast.USub: operator.neg,
}


# ---------------------------------------------------------------------------
# Shape checking
# ---------------------------------------------------------------------------


def check_shape(
  shape: tuple[int, ...],
  spec: tuple[DimSpec, ...],
  memo: ShapeMemo,
  scope: Mapping[str, object] | None = None,
) -> str:
  """Validate *shape* against *spec*, updating *memo* with bindings.

  Parameters
  ----------
  shape:
      The concrete array shape, e.g. ``(4, 3, 28, 28)``.
  spec:
      A tuple of :data:`DimSpec` objects describing the expected shape.
  memo:
      A :class:`ShapeMemo` that accumulates dimension bindings. Named
      dimensions are bound on first occurrence and validated on subsequent
      ones, enabling cross-argument consistency.

  Returns
  -------
  str
      ``""`` on success. A human-readable error message on mismatch.
  """
  # Find the variadic dim (at most one allowed)
  variadic_idx: int | None = None
  for i, dim in enumerate(spec):
    if isinstance(dim, (VariadicDim, _AnonymousVariadic)):
      variadic_idx = i
      break

  if variadic_idx is None:
    # No variadic — exact rank match required
    if len(shape) != len(spec):
      return f"expected {len(spec)} dimensions but got {len(shape)} (shape={shape})"
    return _check_fixed_dims(spec, shape, memo, scope)

  # Split around the variadic dim
  n_prefix = variadic_idx
  n_suffix = len(spec) - variadic_idx - 1
  min_rank = n_prefix + n_suffix

  if len(shape) < min_rank:
    return (
      f"expected at least {min_rank} dimensions but got {len(shape)} (shape={shape})"
    )

  # Check prefix
  err = _check_fixed_dims(spec[:n_prefix], shape[:n_prefix], memo, scope)
  if err:
    return err

  # Check suffix
  if n_suffix > 0:
    err = _check_fixed_dims(spec[-n_suffix:], shape[-n_suffix:], memo, scope)
    if err:
      return err

  # Check variadic middle
  variadic_dim = spec[variadic_idx]
  suffix_start = len(shape) - n_suffix if n_suffix else len(shape)
  middle_shape = shape[n_prefix:suffix_start]

  if isinstance(variadic_dim, _AnonymousVariadic):
    return ""

  assert isinstance(variadic_dim, VariadicDim)
  return _check_variadic(variadic_dim, middle_shape, memo)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_fixed_dims(
  spec: tuple[DimSpec, ...],
  shape: tuple[int, ...],
  memo: ShapeMemo,
  scope: Mapping[str, object] | None,
) -> str:
  for dim, size in zip(spec, shape):
    err = _check_one(dim, size, memo, scope)
    if err:
      return err
  return ""


def _check_one(
  dim: DimSpec, size: int, memo: ShapeMemo, scope: Mapping[str, object] | None
) -> str:
  if isinstance(dim, _Anonymous):
    return ""

  if isinstance(dim, FixedDim):
    if dim.broadcastable and size == 1:
      return ""
    if dim.size != size:
      return f"dimension expected {dim.size} but got {size}"
    return ""

  if isinstance(dim, NamedDim):
    if dim.broadcastable and size == 1:
      return ""
    prev = memo.single.get(dim.name)
    if prev is None:
      memo.single[dim.name] = size
      return ""
    if prev != size:
      return f"dimension '{dim.name}' expected {prev} but got {size}"
    return ""

  if isinstance(dim, SymbolicDim):
    if dim.broadcastable and size == 1:
      return ""
    try:
      expected = _evaluate_dim_expr(dim.expr, memo.single, allow_attr=False)
    except Exception as e:
      return f"cannot evaluate '{dim.expr}': {e}"
    err = _check_expected_value(dim.expr, expected, size)
    if err:
      return err
    return ""

  if isinstance(dim, ValueDim):
    if dim.broadcastable and size == 1:
      return ""
    try:
      names: dict[str, object] = dict(memo.single)
      if scope is not None:
        names.update(scope)
      expected = _evaluate_dim_expr(dim.expr, names, allow_attr=True)
    except Exception as e:
      return f"cannot evaluate {dim!r}: {e}"
    err = _check_expected_value(repr(dim), expected, size)
    if err:
      return err
    return ""

  return f"internal error: unrecognized dim spec {dim!r}"


def _check_variadic(dim: VariadicDim, shape: tuple[int, ...], memo: ShapeMemo) -> str:
  prev = memo.variadic.get(dim.name)
  if prev is None:
    memo.variadic[dim.name] = (dim.broadcastable, shape)
    return ""

  prev_broadcastable, prev_shape = prev
  if prev_shape == shape:
    return ""

  # Broadcasting logic
  if prev_broadcastable or dim.broadcastable:
    try:
      import numpy as np

      broadcast = np.broadcast_shapes(shape, prev_shape)
    except ValueError:
      return (
        f"variadic '~{dim.name}' shape {shape} cannot broadcast "
        f"with existing {prev_shape}"
      )
    # Update to the broadcast result
    memo.variadic[dim.name] = (prev_broadcastable and dim.broadcastable, broadcast)
    return ""

  return f"variadic '~{dim.name}' shape {shape} does not match existing {prev_shape}"


def _check_expected_value(label: str, expected: object, size: int) -> str:
  if isinstance(expected, bool) or not isinstance(expected, numbers.Real):
    return f"dimension '{label}' resolved to non-real value {expected!r}"
  if expected != size:
    return f"dimension '{label}' evaluated to {expected} but got {size}"
  return ""


# 512 entries covers typical usage; in long-running processes with many unique
# expressions, eviction causes minor re-parse overhead but no correctness issue.
@functools.lru_cache(maxsize=512)
def _parse_expr(expr: str, *, allow_attr: bool) -> ast.Expression:
  try:
    tree = ast.parse(expr, mode="eval")
  except SyntaxError as e:  # pragma: no cover - exercised via public errors
    raise ValueError(f"invalid syntax: {e.msg}") from e
  _validate_expr(tree, allow_attr=allow_attr)
  return tree


def _validate_expr(node: ast.AST, *, allow_attr: bool) -> None:
  if isinstance(node, ast.Expression):
    _validate_expr(node.body, allow_attr=allow_attr)
    return

  if isinstance(node, ast.BinOp):
    if type(node.op) not in _BINOPS:
      raise ValueError(f"unsupported operator '{type(node.op).__name__}'")
    _validate_expr(node.left, allow_attr=allow_attr)
    _validate_expr(node.right, allow_attr=allow_attr)
    return

  if isinstance(node, ast.UnaryOp):
    if type(node.op) not in _UNARYOPS:
      raise ValueError(f"unsupported unary operator '{type(node.op).__name__}'")
    _validate_expr(node.operand, allow_attr=allow_attr)
    return

  if isinstance(node, ast.Name):
    return

  if isinstance(node, ast.Attribute):
    if not allow_attr:
      raise ValueError("attribute access is not allowed here")
    if node.attr.startswith("__"):
      raise ValueError("dunder attribute access is not allowed")
    _validate_expr(node.value, allow_attr=allow_attr)
    return

  if isinstance(node, ast.Constant):
    value = node.value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
      raise ValueError(f"unsupported literal {value!r}")
    return

  raise ValueError(
    f"unsupported expression form '{ast.dump(node, include_attributes=False)}'"
  )


def _evaluate_dim_expr(
  expr: str, names: Mapping[str, object], *, allow_attr: bool
) -> object:
  tree = _parse_expr(expr, allow_attr=allow_attr)
  return _evaluate_expr_node(tree.body, names)


def _evaluate_expr_node(node: ast.AST, names: Mapping[str, object]) -> object:
  if isinstance(node, ast.BinOp):
    left = _evaluate_expr_node(node.left, names)
    right = _evaluate_expr_node(node.right, names)
    return _BINOPS[type(node.op)](left, right)

  if isinstance(node, ast.UnaryOp):
    operand = _evaluate_expr_node(node.operand, names)
    return _UNARYOPS[type(node.op)](operand)

  if isinstance(node, ast.Name):
    try:
      return names[node.id]
    except KeyError as e:
      raise NameError(f"unknown name '{node.id}'") from e

  if isinstance(node, ast.Attribute):
    base = _evaluate_expr_node(node.value, names)
    try:
      return getattr(base, node.attr)
    except AttributeError as e:
      raise AttributeError(f"object {base!r} has no attribute '{node.attr}'") from e

  if isinstance(node, ast.Constant):
    return node.value

  raise ValueError(f"unsupported expression node '{type(node).__name__}'")
