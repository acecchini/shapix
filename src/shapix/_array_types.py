"""Core array type factories — the heart of shapix.

Transforms ``F32[N, C, H, W]`` into ``Annotated[np.ndarray, Is[checker]]``
at runtime so that standard ``@beartype`` performs both dtype and shape
validation automatically.

The main components are:

- :class:`_StructChecker` — a callable validator invoked by beartype's ``Is[]``
  mechanism for strict array types.  Checks dtype then validates shape against
  the spec while maintaining cross-argument dimension consistency via the memo.

- :class:`_ArrayLikeChecker` — similar validator for array-like inputs
  (scalars, sequences, arrays).  Converts inputs via ``np.asarray`` and checks
  dtype compatibility using ``np.can_cast`` with configurable casting level.

- :class:`_ArrayFactory` / :class:`_ArrayLikeFactory` — subscriptable objects
  that convert ``F32[N, C]`` or ``F32Like[N, C]`` into the appropriate
  ``Annotated`` type hint.

Use :func:`make_array_type` and :func:`make_array_like_type` to create new
factories for custom array classes or dtype combinations.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from beartype.vale import Is

from ._dimensions import Dimension, _ValueExpr
from ._dtypes import DtypeSpec
from ._dtypes import extract_dtype_str as extract_dtype_str
from ._memo import ShapeMemo as ShapeMemo
from ._memo import get_memo, get_scope
from ._shape import (
  ANONYMOUS,
  ANONYMOUS_VARIADIC,
  DimSpec,
  FixedDim,
  NamedDim,
  SymbolicDim,
  ValueDim,
  VariadicDim,
  _AnonymousVariadic,
  check_shape,
)

__all__ = ["make_array_type", "make_array_like_type"]

_VALID_CASTINGS = frozenset({"no", "equiv", "safe", "same_kind", "unsafe"})


# ---------------------------------------------------------------------------
# Validator callable (used inside beartype's Is[...])
# ---------------------------------------------------------------------------


class _StructChecker:
  """Callable invoked by beartype to validate dtype + shape at runtime.

  Instances are created once per unique annotation (e.g.
  ``Float32Array[N, C]``) and reused across all functions that share it.
  """

  __slots__ = ("_dtype_spec", "_shape_spec", "_repr", "_fail_obj")

  def __init__(self, dtype_spec: DtypeSpec, shape_spec: tuple[DimSpec, ...]) -> None:
    self._dtype_spec = dtype_spec
    self._shape_spec = shape_spec
    self._fail_obj: object | None = None

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
    if not self._dtype_spec.matches(obj):
      return False

    # Shape check with memo (auto-detects call context)
    shape = getattr(obj, "shape", None)
    if shape is None:
      return False

    memo = get_memo(_depth=3)
    scope = get_scope(_depth=3)
    snap = memo.snapshot()

    result = check_shape(tuple(shape), self._shape_spec, memo, scope) == ""

    if not result:
      memo.restore(snap)
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
    checker = _StructChecker(self._dtype_spec, shape_spec)
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
# ArrayLike checker
# ---------------------------------------------------------------------------


class _ArrayLikeChecker:
  """Callable invoked by beartype to validate array-like objects with dtype + shape.

  Handles arrays (objects with ``.shape`` + ``.dtype``), scalars, nested
  sequences, ``__array__`` protocol objects, and buffer protocol objects
  by converting to an array and checking dtype + shape.

  The *casting* parameter controls dtype strictness using numpy casting rules:
  ``no < equiv < safe < same_kind < unsafe``.

  The optional *asarray* callable allows backends to provide their own
  conversion function (e.g. ``jnp.asarray``, ``torch.as_tensor``) so that
  objects implementing backend-specific protocols (``__jax_array__``,
  ``__torch_function__``) are accepted.
  """

  __slots__ = (
    "_dtype_spec",
    "_shape_spec",
    "_casting",
    "_asarray",
    "_repr",
    "_fail_obj",
  )

  def __init__(
    self,
    dtype_spec: DtypeSpec,
    shape_spec: tuple[DimSpec, ...],
    *,
    casting: str,
    name: str,
    asarray: Callable[[object], object] | None = None,
  ) -> None:
    self._dtype_spec = dtype_spec
    self._shape_spec = shape_spec
    self._casting = casting
    self._asarray = asarray
    self._fail_obj: object | None = None

    dims = ", ".join(repr(d) for d in shape_spec)
    self._repr = f"{name}[{dims}]"

  def __call__(self, obj: object) -> bool:
    # Replay failure for beartype error-generation re-invocation
    # (same pattern as _StructChecker).
    if self._fail_obj is not None and self._fail_obj is obj:
      self._fail_obj = None
      return False

    memo = get_memo(_depth=3)
    scope = get_scope(_depth=3)

    # Fast path: obj already has shape + dtype (np.ndarray, Tensor, jax.Array)
    shape = getattr(obj, "shape", None)
    if shape is not None and getattr(obj, "dtype", None) is not None:
      result = self._check(obj, tuple(shape), memo, scope)
      if not result:
        self._fail_obj = obj
      return result

    # Slow path: convert scalar / sequence / protocol object to array.
    # Try the backend-specific converter first (jnp.asarray, torch.as_tensor),
    # then fall back to np.asarray for scalars and nested sequences.
    arr = self._convert(obj)
    if arr is None:
      self._fail_obj = obj
      return False

    result = self._check(arr, tuple(arr.shape), memo, scope)  # type: ignore[union-attr]
    if not result:
      self._fail_obj = obj
    return result

  def _convert(self, obj: object) -> object | None:
    """Convert *obj* to an array with ``.shape`` and ``.dtype``, or None."""
    if self._asarray is not None:
      try:
        return self._asarray(obj)  # type: ignore[operator]
      except Exception:  # noqa: BLE001
        pass  # fall through to numpy

    try:
      import numpy as np

      return np.asarray(obj)
    except (TypeError, ValueError):
      return None

  def _check(
    self, obj: object, shape: tuple[int, ...], memo: ShapeMemo, scope: dict[str, object]
  ) -> bool:
    """Validate dtype (with casting rules) then shape (with memo)."""
    if not self._check_dtype(obj):
      return False

    snap = memo.snapshot()
    result = check_shape(shape, self._shape_spec, memo, scope) == ""
    if not result:
      memo.restore(snap)
    return result

  def _check_dtype(self, obj: object) -> bool:
    """Check dtype using numpy casting rules."""
    # Strictest level: exact dtype match only
    if self._casting == "no":
      return self._dtype_spec.matches(obj)

    source = extract_dtype_str(obj)
    if not source:
      return False

    # Wildcard ("*") accepts any dtype (used by SHAPED)
    if "*" in self._dtype_spec.allowed:
      return self._dtype_spec._check_byteorder(obj)

    # Exact string match always passes (handles non-numpy dtypes like bfloat16
    # where np.can_cast would raise TypeError for an unknown dtype string).
    if source in self._dtype_spec.allowed:
      return self._dtype_spec._check_byteorder(obj)

    import numpy as np

    for target in self._dtype_spec.allowed:
      try:
        if np.can_cast(source, target, casting=self._casting):  # pyright: ignore[reportArgumentType]
          return self._dtype_spec._check_byteorder(obj)
      except TypeError:
        continue
    return False

  def __repr__(self) -> str:
    return self._repr


# ---------------------------------------------------------------------------
# ArrayLike factory
# ---------------------------------------------------------------------------


class _ArrayLikeFactory:
  """Subscriptable factory: ``F32Like[N, C]`` → ``Annotated[object, Is[...]]``.

  Created via :func:`make_array_like_type` and not instantiated directly.
  Mirrors :class:`_ArrayFactory` but validates array-like inputs (scalars,
  sequences, arrays) with dtype casting awareness.
  """

  __slots__ = ("_dtype_spec", "_casting", "_asarray", "__name__")

  def __init__(
    self,
    dtype_spec: DtypeSpec,
    *,
    casting: str,
    name: str,
    asarray: Callable[[object], object] | None = None,
  ) -> None:
    self._dtype_spec = dtype_spec
    self._casting = casting
    self._asarray = asarray
    self.__name__ = name

  def __getitem__(self, dims: object) -> type:
    if not isinstance(dims, tuple):
      dims = (dims,)

    shape_spec = _to_shape_spec(dims)
    checker = _ArrayLikeChecker(
      self._dtype_spec,
      shape_spec,
      casting=self._casting,
      name=self.__name__,
      asarray=self._asarray,
    )
    return Annotated[object, Is[checker]]  # type: ignore[return-value]

  def __repr__(self) -> str:
    return self.__name__


def make_array_like_type(
  dtype_spec: DtypeSpec,
  *,
  casting: str = "same_kind",
  name: str = "ArrayLike",
  asarray: Callable[[object], object] | None = None,
) -> _ArrayLikeFactory:
  """Create a subscriptable array-like type factory.

  Parameters
  ----------
  dtype_spec:
      A :class:`~shapix._dtypes.DtypeSpec` defining the allowed dtypes.
  casting:
      NumPy casting rule: ``"no"`` | ``"equiv"`` | ``"safe"``
      | ``"same_kind"`` | ``"unsafe"``.  Controls how strictly dtype
      compatibility is checked.  Default ``"same_kind"``.
  name:
      Human-readable name for error messages.
  asarray:
      Optional callable ``(obj) -> array`` for backend-specific conversion.
      When provided, it is tried before ``np.asarray`` on the slow path
      (objects without ``.shape`` / ``.dtype``).  Use this to support
      protocols like ``__jax_array__`` or ``__torch_function__``.

  Returns
  -------
  _ArrayLikeFactory
      A subscriptable factory.  ``factory[N, C]`` produces
      ``Annotated[object, Is[checker]]``.

  Example::

      from shapix._dtypes import FLOAT32

      F32Like = make_array_like_type(FLOAT32, name="F32Like")
      F32Like[N, C, H, W]  # → Annotated[object, Is[...]]
  """
  if casting not in _VALID_CASTINGS:
    msg = f"Invalid casting {casting!r}, must be one of {sorted(_VALID_CASTINGS)}"
    raise ValueError(msg)
  return _ArrayLikeFactory(dtype_spec, casting=casting, name=name, asarray=asarray)


# ---------------------------------------------------------------------------
# Dimension conversion
# ---------------------------------------------------------------------------


def _to_shape_spec(dims: tuple[object, ...]) -> tuple[DimSpec, ...]:
  """Convert a tuple of user-facing dim objects to internal DimSpec."""
  specs: list[DimSpec] = []
  for d in dims:
    if d is Ellipsis:
      specs.append(ANONYMOUS_VARIADIC)
    elif d is ANONYMOUS:
      specs.append(ANONYMOUS)
    elif d is ANONYMOUS_VARIADIC:
      specs.append(ANONYMOUS_VARIADIC)
    elif isinstance(d, int):
      specs.append(FixedDim(d))
    elif isinstance(d, (FixedDim, NamedDim, SymbolicDim, ValueDim, VariadicDim)):
      specs.append(d)
    elif isinstance(d, Dimension):
      spec = d._dim_spec  # noqa: SLF001
      if spec is not None:
        specs.append(spec)
    elif isinstance(d, _ValueExpr):
      specs.append(d._dim_spec)  # noqa: SLF001
    else:
      msg = (
        "Invalid shape token "
        f"{d!r}; expected int, Ellipsis, Dimension, Value(...), or a DimSpec"
      )
      raise TypeError(msg)

  variadic_count = sum(
    1 for s in specs if isinstance(s, (VariadicDim, _AnonymousVariadic))
  )
  if variadic_count > 1:
    msg = "At most one variadic dimension allowed per shape spec"
    raise TypeError(msg)

  return tuple(specs)
