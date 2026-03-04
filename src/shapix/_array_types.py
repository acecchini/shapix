"""Core array type factory — the heart of shapix.

Transforms ``Float32Array[N, C, H, W]`` into
``Annotated[np.ndarray, Is[_ShapeChecker(...)]]`` at runtime so that standard
``@beartype`` performs both dtype and shape validation automatically.

The two main components are:

- :class:`_ShapeChecker` — a callable validator invoked by beartype's ``Is[]``
  mechanism. It checks the dtype, then validates the shape against the spec
  while maintaining cross-argument dimension consistency via the memo system.

- :class:`_ArrayFactory` — a subscriptable object that converts
  ``Float32Array[N, C]`` into the appropriate ``Annotated`` type hint.

Use :func:`make_array_type` to create new array type factories for custom
array classes or dtype combinations.
"""

from __future__ import annotations

from typing import Annotated

from beartype.vale import Is

from ._dimensions import Dimension
from ._dtypes import DtypeSpec, extract_dtype_str
from ._memo import get_memo
from ._shape import (
  ANONYMOUS_VARIADIC,
  DimSpec,
  FixedDim,
  NamedDim,
  VariadicDim,
  _AnonymousVariadic,
  check_shape,
)

__all__ = ["make_array_type"]


# ---------------------------------------------------------------------------
# Validator callable (used inside beartype's Is[...])
# ---------------------------------------------------------------------------


class _ShapeChecker:
  """Callable invoked by beartype to validate dtype + shape at runtime.

  Instances are created once per unique annotation (e.g.
  ``Float32Array[N, C]``) and reused across all functions that share it.
  """

  __slots__ = ("_dtype_spec", "_shape_spec", "_repr", "_fail_obj")

  def __init__(self, dtype_spec: DtypeSpec, shape_spec: tuple[DimSpec, ...]) -> None:
    self._dtype_spec = dtype_spec
    self._shape_spec = shape_spec
    self._fail_obj: object = None

    # Pre-compute repr for beartype error messages
    dims = ", ".join(repr(d) for d in shape_spec)
    self._repr = f"{dtype_spec.name}[{dims}]"

  def __call__(self, obj: object) -> bool:
    # When beartype's error-generation code re-invokes us, it does so
    # from a different call-stack frame, which creates a fresh memo.
    # That memo lacks the bindings from prior params, so a previously
    # failing check would incorrectly pass.  To stay consistent, replay
    # the failure for one re-check, then clear it.  We use ``is`` identity
    # (not ``id()``) to avoid false matches from recycled addresses.
    if self._fail_obj is not None and self._fail_obj is obj:
      self._fail_obj = None
      return False

    # Dtype check
    if extract_dtype_str(obj) not in self._dtype_spec.allowed:
      return False

    # Shape check with memo (auto-detects call context)
    shape = getattr(obj, "shape", None)
    if shape is None:
      return False

    memo = get_memo(_depth=3)

    # Snapshot memo state so we can restore on failure (avoid polluting
    # the memo with partial bindings from a bad argument).
    single_snap = memo.single.copy()
    variadic_snap = memo.variadic.copy()
    structures_snap = memo.structures.copy()

    result = check_shape(tuple(shape), self._shape_spec, memo) == ""

    if not result:
      memo.single.clear()
      memo.single.update(single_snap)
      memo.variadic.clear()
      memo.variadic.update(variadic_snap)
      memo.structures.clear()
      memo.structures.update(structures_snap)
      self._fail_obj = obj

    return result

  def __repr__(self) -> str:
    return self._repr


# ---------------------------------------------------------------------------
# Array factory
# ---------------------------------------------------------------------------


class _ArrayFactory:
  """Subscriptable factory: ``Float32Array[N, C]`` → ``Annotated[ndarray, Is[...]]``.

  Created via :func:`make_array_type` and not instantiated directly.
  """

  __slots__ = ("_array_type", "_dtype_spec", "__name__")

  def __init__(self, array_type: type, dtype_spec: DtypeSpec) -> None:
    self._array_type = array_type
    self._dtype_spec = dtype_spec
    self.__name__ = f"{dtype_spec.name}Array"

  def __getitem__(self, dims: object) -> type:
    if not isinstance(dims, tuple):
      dims = (dims,)

    shape_spec = _to_shape_spec(dims)
    checker = _ShapeChecker(self._dtype_spec, shape_spec)
    return Annotated[self._array_type, Is[checker]]  # type: ignore[return-value]

  def __repr__(self) -> str:
    return self.__name__


def make_array_type(array_type: type, dtype_spec: DtypeSpec) -> _ArrayFactory:
  """Create a subscriptable array type factory for a given base type and dtype.

  Parameters
  ----------
  array_type:
      The base array class (e.g. ``np.ndarray``, ``jax.Array``,
      ``torch.Tensor``, or any class with ``.dtype`` and ``.shape``).
  dtype_spec:
      A :class:`~shapix._dtypes.DtypeSpec` defining the allowed dtypes.

  Returns
  -------
  _ArrayFactory
      A subscriptable factory. ``factory[N, C]`` produces
      ``Annotated[array_type, Is[checker]]``.

  Example::

      import numpy as np
      from shapix._dtypes import FLOAT32

      Float32Array = make_array_type(np.ndarray, FLOAT32)
      Float32Array[N, C, H, W]  # → Annotated[ndarray, Is[...]]
  """
  return _ArrayFactory(array_type, dtype_spec)


# ---------------------------------------------------------------------------
# Dimension conversion
# ---------------------------------------------------------------------------


def _to_shape_spec(dims: tuple[object, ...]) -> tuple[DimSpec, ...]:
  """Convert a tuple of user-facing dim objects to internal DimSpec."""
  specs: list[DimSpec] = []
  for d in dims:
    if d is Ellipsis:
      specs.append(ANONYMOUS_VARIADIC)
    elif isinstance(d, int):
      specs.append(FixedDim(d))
    elif isinstance(d, Dimension):
      spec = d._dim_spec  # noqa: SLF001
      if spec is not None:
        specs.append(spec)
    else:
      specs.append(NamedDim(str(d), broadcastable=False))

  variadic_count = sum(
    1 for s in specs if isinstance(s, (VariadicDim, _AnonymousVariadic))
  )
  if variadic_count > 1:
    msg = "At most one variadic dimension allowed per shape spec"
    raise TypeError(msg)

  return tuple(specs)
