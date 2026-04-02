"""Core array type factories — the heart of shapix.

Transforms ``F32[N, C, H, W]`` into custom runtime hint classes so that
standard ``@beartype`` performs both dtype and shape validation automatically
and can surface readable diagnostics on failure.

The main components are:

- :class:`_ArrayChecker` — validator attached to a shapix runtime hint for
  strict array types. Checks dtype, then validates shape against the spec while
  maintaining cross-argument dimension consistency via the memo.

- :class:`_ArrayLikeChecker` — similar validator for array-like inputs
  (scalars, sequences, arrays).  Converts inputs via ``np.asarray`` and checks
  dtype compatibility using ``np.can_cast`` with configurable casting level.

- :class:`_ArrayFactory` / :class:`_ArrayLikeFactory` — subscriptable objects
  that convert ``F32[N, C]`` or ``F32Like[N, C]`` into the appropriate
  runtime hint class.

Use :func:`make_array_type` and :func:`make_array_like_type` to create new
factories for custom array classes or dtype combinations.
"""

from __future__ import annotations

import typing as tp
from collections.abc import Callable

from ._dimensions import Dimension, _ValueExpr
from ._dtypes import DtypeSpec
from ._dtypes import extract_dtype_str as extract_dtype_str
from ._memo import ShapeMemo as ShapeMemo
from ._memo import get_memo, get_scope, has_untagged_memo
from ._runtime_hints import ReplayFailureState, ValidationFailure, make_runtime_hint
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
# Trusted array type cache (for ArrayLike fast-path gating)
# ---------------------------------------------------------------------------


def _is_trusted_array_type(cls: type[object]) -> bool:
  """True if *cls* is a known array class (or subclass of one)."""
  cached = _TRUSTED_ARRAY_TYPE_CACHE.get(cls)
  if cached is not None:
    return cached

  import sys

  import numpy as np

  trusted: tuple[type, ...] = (np.ndarray,)

  jax_mod = sys.modules.get("jax")
  if jax_mod is not None:
    jax_array = getattr(jax_mod, "Array", None)
    if jax_array is not None:
      trusted = (*trusted, jax_array)

  torch_mod = sys.modules.get("torch")
  if torch_mod is not None:
    tensor = getattr(torch_mod, "Tensor", None)
    if tensor is not None:
      trusted = (*trusted, tensor)

  cupy_mod = sys.modules.get("cupy")
  if cupy_mod is not None:
    cupy_array = getattr(cupy_mod, "ndarray", None)
    if cupy_array is not None:
      trusted = (*trusted, cupy_array)

  result = issubclass(cls, trusted)
  _TRUSTED_ARRAY_TYPE_CACHE[cls] = result
  return result


_TRUSTED_ARRAY_TYPE_CACHE: dict[type[object], bool] = {}


def _is_trusted_array(obj: object) -> bool:
  """True if *obj* is an instance of a known array class."""
  return _is_trusted_array_type(type(obj))


def _infer_array_hint_module(array_type: type[object]) -> str:
  return getattr(array_type, "__module__", __name__)


def _infer_arraylike_hint_module(
  asarray: Callable[[object], object] | None, trusted_types: tuple[type, ...] | None
) -> str:
  # Backend Like factories pass a backend-scoped converter wrapper (for example
  # ``shapix.jax._jax_asarray``) but often keep ``np.ndarray`` first in
  # ``trusted_types``. Prefer the explicit converter module so diagnostics
  # identify the owning shapix backend rather than falling back to NumPy.
  if asarray is not None:
    return getattr(asarray, "__module__", __name__)
  if trusted_types:
    return getattr(trusted_types[0], "__module__", __name__)
  return __name__


def _describe_actual_dtype(obj: object) -> str:
  source = extract_dtype_str(obj)
  dtype = getattr(obj, "dtype", None)
  if dtype is None:
    return f"{type(obj).__name__} has no dtype"
  if source:
    dt_str = getattr(dtype, "str", "")
    if isinstance(dt_str, str) and dt_str:
      return f"{source} ({dt_str})"
    return source
  return repr(dtype)


def _byteorder_detail(dtype_spec: DtypeSpec) -> str:
  byteorder = dtype_spec.byteorder
  if byteorder == "native":
    return "native byte order"
  return f"{byteorder}-endian byte order"


def _dtype_mismatch(
  dtype_spec: DtypeSpec, obj: object, *, casting: str | None = None
) -> ValidationFailure:
  actual = _describe_actual_dtype(obj)
  if dtype_spec.byteorder != "any" and dtype_spec._check_byteorder(obj) is False:
    return ValidationFailure(
      f"expected dtype {dtype_spec.name} with {_byteorder_detail(dtype_spec)} but got {actual}"
    )
  if casting is not None and casting != "no":
    return ValidationFailure(
      f"expected dtype compatible with {dtype_spec.name} under casting='{casting}' but got {actual}"
    )
  return ValidationFailure(f"expected dtype {dtype_spec.name} but got {actual}")


# ---------------------------------------------------------------------------
# Runtime validators used by shapix's beartype hint classes
# ---------------------------------------------------------------------------


class _ArrayChecker:
  """Callable invoked by beartype to validate dtype + shape at runtime.

  Instances are created once per unique annotation (e.g.
  ``Float32Array[N, C]``) and reused across all functions that share it.
  """

  __slots__ = ("_array_type", "_dtype_spec", "_shape_spec", "_repr", "_fail_state")

  def __init__(
    self,
    array_type: type[object] | DtypeSpec,
    dtype_spec: DtypeSpec | tuple[DimSpec, ...],
    shape_spec: tuple[DimSpec, ...] | None = None,
  ) -> None:
    if shape_spec is None:
      self._array_type = None
      self._dtype_spec = tp.cast(DtypeSpec, array_type)
      self._shape_spec = tp.cast(tuple[DimSpec, ...], dtype_spec)
    else:
      self._array_type = tp.cast(type[object], array_type)
      self._dtype_spec = tp.cast(DtypeSpec, dtype_spec)
      self._shape_spec = shape_spec
    self._fail_state = ReplayFailureState()

    # Pre-compute repr for beartype error messages
    dims = ", ".join(repr(d) for d in self._shape_spec)
    self._repr = f"{self._dtype_spec.name}[{dims}]"

  def __call__(self, obj: object) -> bool:
    return self.instancecheck(obj)

  def instancecheck(self, obj: object) -> bool:
    if self._fail_state.should_replay(obj):
      return False

    memo = get_memo()
    scope = get_scope()
    snap = memo.snapshot()
    has_prior = any(snap)
    failure = self._validate(obj, memo, scope)
    if failure is None:
      self._fail_state.clear()
      return True

    memo.restore(snap)
    if has_prior and not has_untagged_memo():
      self._fail_state.record(obj, memo, failure)
    else:
      self._fail_state.clear()
    return False

  def instancecheck_str(self, obj: object) -> str:
    detail = self._fail_state.detail_for(obj)
    if detail is None:
      memo = get_memo()
      scope = get_scope()
      snap = memo.snapshot()
      failure = self._validate(obj, memo, scope)
      memo.restore(snap)
      if failure is None:
        detail = ValidationFailure(f"unexpectedly accepted {obj!r} for {self!r}")
      else:
        detail = failure
    self._fail_state.clear()
    return detail.message

  def _validate(
    self, obj: object, memo: ShapeMemo, scope: dict[str, object]
  ) -> ValidationFailure | None:
    array_type = self._array_type
    if array_type is not None and not isinstance(obj, array_type):
      expected = f"{array_type.__module__}.{array_type.__name__}"
      actual = f"{type(obj).__module__}.{type(obj).__name__}"
      return ValidationFailure(f"expected {expected} but got {actual}")

    if not self._dtype_spec.matches(obj):
      return _dtype_mismatch(self._dtype_spec, obj)

    shape = getattr(obj, "shape", None)
    if shape is None:
      return ValidationFailure(
        f"expected object with shape for {self._repr} but got {type(obj).__name__}"
      )

    err = check_shape(tuple(shape), self._shape_spec, memo, scope)
    if err:
      return ValidationFailure(err)
    return None

  def __repr__(self) -> str:
    return self._repr


# ---------------------------------------------------------------------------
# Array factory
# ---------------------------------------------------------------------------


class _ArrayFactory:
  """Subscriptable factory: ``Float32Array[N, C]`` → runtime hint class.

  Created via :func:`make_array_type` and not instantiated directly.
  """

  __slots__ = ("_array_type", "_dtype_spec", "__name__", "_cache", "_module")

  def __init__(self, array_type: type, dtype_spec: DtypeSpec) -> None:
    self._array_type = array_type
    self._dtype_spec = dtype_spec
    self.__name__ = f"{dtype_spec.name}Array"
    self._cache: dict[tuple[DimSpec, ...], type] = {}
    self._module = _infer_array_hint_module(array_type)

  def __getitem__(self, dims: object) -> type:
    if not isinstance(dims, tuple):
      dims = (dims,)

    shape_spec = _to_shape_spec(dims)
    cached = self._cache.get(shape_spec)
    if cached is not None:
      return cached

    checker = _ArrayChecker(self._array_type, self._dtype_spec, shape_spec)
    hint = make_runtime_hint(
      repr(checker), checker, module=self._module, origin=self._array_type
    )
    self._cache[shape_spec] = hint
    return hint

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
      A subscriptable factory. ``factory[N, C]`` produces a runtime hint
      class validated by beartype.

  Example::

      import numpy as np
      from shapix._dtypes import FLOAT32

      Float32Array = make_array_type(np.ndarray, FLOAT32)
      Float32Array[N, C, H, W]  # → runtime hint class
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
  backend-specific inputs (e.g. ``__jax_array__`` protocol objects) are
  accepted.
  """

  __slots__ = (
    "_dtype_spec",
    "_shape_spec",
    "_casting",
    "_is_structured",
    "_asarray",
    "_trusted_types",
    "_repr",
    "_fail_state",
  )

  def __init__(
    self,
    dtype_spec: DtypeSpec,
    shape_spec: tuple[DimSpec, ...],
    *,
    casting: str,
    name: str,
    asarray: Callable[[object], object] | None = None,
    trusted_types: tuple[type, ...] | None = None,
  ) -> None:
    self._dtype_spec = dtype_spec
    self._shape_spec = shape_spec
    self._casting = casting
    self._is_structured: bool = dtype_spec._structured is not None  # noqa: SLF001
    self._asarray = asarray
    self._trusted_types = trusted_types
    self._fail_state = ReplayFailureState()

    dims = ", ".join(repr(d) for d in shape_spec)
    self._repr = f"{name}[{dims}]"

  def __call__(self, obj: object) -> bool:
    return self.instancecheck(obj)

  def instancecheck(self, obj: object) -> bool:
    if self._fail_state.should_replay(obj):
      return False

    memo = get_memo()
    scope = get_scope()
    snap = memo.snapshot()
    has_prior = any(snap)
    failure = self._validate(obj, memo, scope)
    if failure is None:
      self._fail_state.clear()
      return True

    memo.restore(snap)
    if has_prior and not has_untagged_memo():
      self._fail_state.record(obj, memo, failure)
    else:
      self._fail_state.clear()
    return False

  def instancecheck_str(self, obj: object) -> str:
    detail = self._fail_state.detail_for(obj)
    if detail is None:
      memo = get_memo()
      scope = get_scope()
      snap = memo.snapshot()
      failure = self._validate(obj, memo, scope)
      memo.restore(snap)
      if failure is None:
        detail = ValidationFailure(f"unexpectedly accepted {obj!r} for {self!r}")
      else:
        detail = failure
    self._fail_state.clear()
    return detail.message

  def _validate(
    self, obj: object, memo: ShapeMemo, scope: dict[str, object]
  ) -> ValidationFailure | None:
    # Fast path: known array types with shape + dtype skip conversion.
    # Spoofed objects with .shape/.dtype fall through to the slow path
    # where _convert() verifies actual convertibility.
    # When trusted_types is set, only those types get the fast path;
    # foreign-backend arrays fall through to the slow path where
    # _convert() verifies actual convertibility.
    shape = getattr(obj, "shape", None)
    if shape is not None and getattr(obj, "dtype", None) is not None:
      trusted = self._trusted_types
      if (trusted is not None and isinstance(obj, trusted)) or (
        trusted is None and _is_trusted_array(obj)
      ):
        return self._check(obj, tuple(shape), memo, scope)

    # Slow path: convert scalar / sequence / protocol object to array.
    # Try the backend-specific converter first (jnp.asarray, torch.as_tensor),
    # then fall back to np.asarray for scalars and nested sequences.
    arr, failure = self._convert(obj)
    if arr is None:
      return failure

    return self._check(arr, tuple(arr.shape), memo, scope)  # type: ignore[attr-defined]

  def _convert(self, obj: object) -> tuple[object | None, ValidationFailure | None]:
    """Convert *obj* to an array with ``.shape`` and ``.dtype``."""
    errors: list[str] = []

    if self._asarray is not None:
      try:
        return self._asarray(obj), None
      except Exception as e:  # noqa: BLE001
        if str(e):
          errors.append(str(e))

    try:
      import numpy as np

      return np.asarray(obj), None
    except Exception as e:  # noqa: BLE001
      if str(e):
        errors.append(str(e))

    detail = "; ".join(dict.fromkeys(errors))
    if detail:
      return None, ValidationFailure(f"could not convert value to array: {detail}")
    return None, ValidationFailure("could not convert value to array")

  def _check(
    self, obj: object, shape: tuple[int, ...], memo: ShapeMemo, scope: dict[str, object]
  ) -> ValidationFailure | None:
    """Validate dtype (with casting rules) then shape (with memo)."""
    failure = self._check_dtype(obj)
    if failure is not None:
      return failure

    err = check_shape(shape, self._shape_spec, memo, scope)
    if err:
      return ValidationFailure(err)
    return None

  def _check_dtype(self, obj: object) -> ValidationFailure | None:
    """Check dtype using numpy casting rules."""
    # Strictest level: exact dtype match only
    if self._casting == "no":
      if self._dtype_spec.matches(obj):
        return None
      return _dtype_mismatch(self._dtype_spec, obj)

    # Structured dtypes require exact field layout comparison regardless of
    # casting mode — np.can_cast("void", "void") would accept any void dtype.
    if self._is_structured:
      if self._dtype_spec.matches(obj):
        return None
      return _dtype_mismatch(self._dtype_spec, obj)

    source = extract_dtype_str(obj)
    if not source:
      return _dtype_mismatch(self._dtype_spec, obj, casting=self._casting)

    # Wildcard ("*") accepts any dtype (used by SHAPED)
    if "*" in self._dtype_spec.allowed:
      if self._dtype_spec._check_byteorder(obj):
        return None
      return _dtype_mismatch(self._dtype_spec, obj, casting=self._casting)

    # Exact string match always passes (handles non-numpy dtypes like bfloat16
    # where np.can_cast would raise TypeError for an unknown dtype string).
    if source in self._dtype_spec.allowed:
      if self._dtype_spec._check_byteorder(obj):
        return None
      return _dtype_mismatch(self._dtype_spec, obj, casting=self._casting)

    import numpy as np

    for target in self._dtype_spec.allowed:
      try:
        if np.can_cast(source, target, casting=self._casting):  # type: ignore[arg-type]  # pyright: ignore[reportArgumentType]
          if self._dtype_spec._check_byteorder(obj):
            return None
          return _dtype_mismatch(self._dtype_spec, obj, casting=self._casting)
      except TypeError:
        continue

    return _dtype_mismatch(self._dtype_spec, obj, casting=self._casting)

  def __repr__(self) -> str:
    return self._repr


# ---------------------------------------------------------------------------
# ArrayLike factory
# ---------------------------------------------------------------------------


class _ArrayLikeFactory:
  """Subscriptable factory: ``F32Like[N, C]`` → runtime hint class.

  Created via :func:`make_array_like_type` and not instantiated directly.
  Mirrors :class:`_ArrayFactory` but validates array-like inputs (scalars,
  sequences, arrays) with dtype casting awareness.
  """

  __slots__ = (
    "_dtype_spec",
    "_casting",
    "_asarray",
    "_trusted_types",
    "__name__",
    "_cache",
    "_module",
  )

  def __init__(
    self,
    dtype_spec: DtypeSpec,
    *,
    casting: str,
    name: str,
    asarray: Callable[[object], object] | None = None,
    trusted_types: tuple[type, ...] | None = None,
  ) -> None:
    self._dtype_spec = dtype_spec
    self._casting = casting
    self._asarray = asarray
    self._trusted_types = trusted_types
    self.__name__ = name
    self._cache: dict[tuple[DimSpec, ...], type] = {}
    self._module = _infer_arraylike_hint_module(asarray, trusted_types)

  def __getitem__(self, dims: object) -> type:
    if not isinstance(dims, tuple):
      dims = (dims,)

    shape_spec = _to_shape_spec(dims)
    cached = self._cache.get(shape_spec)
    if cached is not None:
      return cached

    checker = _ArrayLikeChecker(
      self._dtype_spec,
      shape_spec,
      casting=self._casting,
      name=self.__name__,
      asarray=self._asarray,
      trusted_types=self._trusted_types,
    )
    hint = make_runtime_hint(repr(checker), checker, module=self._module, origin=object)
    self._cache[shape_spec] = hint
    return hint

  def __repr__(self) -> str:
    return self.__name__


def make_array_like_type(
  dtype_spec: DtypeSpec,
  *,
  casting: str = "same_kind",
  name: str = "ArrayLike",
  asarray: Callable[[object], object] | None = None,
  trusted_types: tuple[type, ...] | None = None,
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
      backend-specific protocols like ``__jax_array__``.
  trusted_types:
      Optional tuple of types that get the fast path (skip conversion).
      When ``None`` (default), all known array backends are trusted.
      Set this to scope the fast path to a specific backend so that
      foreign-backend arrays fall through to the slow path where
      actual convertibility is verified.

  Returns
  -------
  _ArrayLikeFactory
      A subscriptable factory.  ``factory[N, C]`` produces a runtime hint
      class validated by beartype.

  Example::

      from shapix._dtypes import FLOAT32

      F32Like = make_array_like_type(FLOAT32, name="F32Like")
      F32Like[N, C, H, W]  # → runtime hint class
  """
  if casting not in _VALID_CASTINGS:
    msg = f"Invalid casting {casting!r}, must be one of {sorted(_VALID_CASTINGS)}"
    raise ValueError(msg)
  return _ArrayLikeFactory(
    dtype_spec, casting=casting, name=name, asarray=asarray, trusted_types=trusted_types
  )


# ---------------------------------------------------------------------------
# Dimension conversion
# ---------------------------------------------------------------------------


def _is_scalar_token(d: object) -> bool:
  """True if *d* represents the scalar sentinel (zero-dim shape)."""
  return isinstance(d, Dimension) and d._dim_spec is None  # noqa: SLF001


def _to_shape_spec(dims: tuple[object, ...]) -> tuple[DimSpec, ...]:
  """Convert a tuple of user-facing dim objects to internal DimSpec."""
  if any(_is_scalar_token(d) for d in dims) and len(dims) > 1:
    msg = (
      "Scalar must be the only shape token; mixed use like F32[N, Scalar] is invalid"
    )
    raise TypeError(msg)
  if not dims:
    msg = "Empty shape spec is not allowed; use F32[...] for any-rank matching"
    raise TypeError(msg)
  specs: list[DimSpec] = []
  for d in dims:
    if d is Ellipsis:
      specs.append(ANONYMOUS_VARIADIC)
    elif d is ANONYMOUS:
      specs.append(ANONYMOUS)
    elif d is ANONYMOUS_VARIADIC:
      specs.append(ANONYMOUS_VARIADIC)
    elif isinstance(d, bool):
      msg = f"bool ({d!r}) is not a valid shape token; use an int or Dimension"
      raise TypeError(msg)
    elif isinstance(d, int):
      if d < 0:
        msg = f"Negative dimension {d} is invalid; array shapes are non-negative"
        raise TypeError(msg)
      specs.append(FixedDim(d))
    elif isinstance(d, (FixedDim, NamedDim, SymbolicDim, ValueDim, VariadicDim)):
      specs.append(d)
    elif isinstance(d, Dimension):
      spec = d._dim_spec  # noqa: SLF001
      if spec is not None:
        if isinstance(spec, FixedDim) and spec.size < 0:
          msg = (
            f"Negative dimension {spec.size} is invalid; array shapes are non-negative"
          )
          raise TypeError(msg)
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
