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
- :class:`VariadicDim` — matches zero or more contiguous dims and binds the
  matched sub-shape (``~spatial``). Optionally broadcastable.
- :data:`ANONYMOUS` — matches any single dim without binding (``__``).
- :data:`ANONYMOUS_VARIADIC` — matches any number of dims without binding
  (``~__``).

The :func:`check_shape` function returns ``""`` on success or a human-readable
error message explaining the mismatch.
"""

from __future__ import annotations

from dataclasses import dataclass

from ._memo import ShapeMemo

__all__ = [
  "NamedDim",
  "FixedDim",
  "SymbolicDim",
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

  def __repr__(self) -> str:
    return str(self.size)


@dataclass(frozen=True, slots=True)
class SymbolicDim:
  """An arithmetic expression evaluated against bound dimensions."""

  expr: str
  broadcastable: bool = False

  def __repr__(self) -> str:
    return self.expr


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
  NamedDim | FixedDim | SymbolicDim | VariadicDim | _Anonymous | _AnonymousVariadic
)


# ---------------------------------------------------------------------------
# Shape checking
# ---------------------------------------------------------------------------


def check_shape(
  shape: tuple[int, ...], spec: tuple[DimSpec, ...], memo: ShapeMemo
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
    return _check_fixed_dims(spec, shape, memo)

  # Split around the variadic dim
  n_prefix = variadic_idx
  n_suffix = len(spec) - variadic_idx - 1
  min_rank = n_prefix + n_suffix

  if len(shape) < min_rank:
    return (
      f"expected at least {min_rank} dimensions but got {len(shape)} (shape={shape})"
    )

  # Check prefix
  err = _check_fixed_dims(spec[:n_prefix], shape[:n_prefix], memo)
  if err:
    return err

  # Check suffix
  if n_suffix > 0:
    err = _check_fixed_dims(spec[-n_suffix:], shape[-n_suffix:], memo)
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
  spec: tuple[DimSpec, ...] | list[DimSpec], shape: tuple[int, ...], memo: ShapeMemo
) -> str:
  for dim, size in zip(spec, shape):
    err = _check_one(dim, size, memo)
    if err:
      return err
  return ""


def _check_one(dim: DimSpec, size: int, memo: ShapeMemo) -> str:
  if isinstance(dim, _Anonymous):
    return ""

  if isinstance(dim, FixedDim):
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
      expected = eval(dim.expr, {"__builtins__": {}}, memo.single)  # noqa: S307
    except Exception as e:
      return f"cannot evaluate '{dim.expr}': {e}"
    if expected != size:
      return f"dimension '{dim.expr}' evaluated to {expected} but got {size}"
    return ""

  return ""


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
