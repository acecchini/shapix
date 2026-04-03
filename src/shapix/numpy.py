"""NumPy array type annotations with runtime shape and dtype checking.

Array types (e.g. ``F32``, ``Int``) are subscriptable with dimension symbols::

    from shapix import N, C, H, W
    from shapix.numpy import F32


    @beartype
    def conv(x: F32[N, C, H, W]) -> F32[N, C, H, W]: ...

Endianness variants are available programmatically via
``make_array_type(np.ndarray, FLOAT32_LE)`` using DtypeSpec constants from
``shapix._dtypes``.

Additional dtypes: ``V`` (void), ``Str`` (string), ``Bytes``, ``Obj`` (object),
``DT64`` (datetime64), ``TD64`` (timedelta64).

``Like`` types (e.g. ``F32Like``) accept scalars, nested sequences, or arrays
— use them for function inputs that will be converted to arrays.
Like types **must be subscripted**: ``F32Like[N, C]`` or ``F32Like[...]``::

    from shapix.numpy import F32Like


    @beartype
    def f(x: F32Like[N, C]) -> F32[N, C]: ...

``ScalarLike`` types (e.g. ``I8ScalarLike``, ``U8ScalarLike``) validate individual
scalar values with range checking — no shape, just value::

    from shapix.numpy import U8ScalarLike


    @beartype
    def pixel(value: U8ScalarLike) -> int: ...  # [0, 255]

``make_scalar_like_type`` creates casting-aware scalar types programmatically::

    from shapix.numpy import make_scalar_like_type

    F32Scalar = make_scalar_like_type(np.float32, casting="safe")

``Structured()`` creates array types for NumPy structured (record) dtypes::

    from shapix.numpy import Structured

    Point = Structured([("x", np.float32), ("y", np.float32)])

``ArrayLike`` is a public recursive type template for static type checking.
"""

from __future__ import annotations

import typing as tp
from typing import Annotated

import numpy as np
from beartype.vale import Is
from numpy._typing import _NestedSequence, _SupportsArray

__all__ = [
  # Array types — base
  "Bool",
  "I8",
  "I16",
  "I32",
  "I64",
  "U8",
  "U16",
  "U32",
  "U64",
  "F16",
  "F32",
  "F64",
  "F128",
  "C64",
  "C128",
  "C256",
  # Array types — categories
  "Int",
  "UInt",
  "Integer",
  "Float",
  "Real",
  "Complex",
  "Inexact",
  "Num",
  "Shaped",
  # Array types — new dtypes
  "V",
  "Str",
  "Bytes",
  "Obj",
  "DT64",
  "TD64",
  # Structured dtype helper
  "Structured",
  # Scalar Like types
  "StringLike",
  "BoolScalarLike",
  "I8ScalarLike",
  "I16ScalarLike",
  "I32ScalarLike",
  "I64ScalarLike",
  "U8ScalarLike",
  "U16ScalarLike",
  "U32ScalarLike",
  "U64ScalarLike",
  "F16ScalarLike",
  "F32ScalarLike",
  "F64ScalarLike",
  "F128ScalarLike",
  "C64ScalarLike",
  "C128ScalarLike",
  "C256ScalarLike",
  "IntScalarLike",
  "UIntScalarLike",
  "IntegerScalarLike",
  "FloatScalarLike",
  "RealScalarLike",
  "ComplexScalarLike",
  "InexactScalarLike",
  "NumScalarLike",
  "ShapedScalarLike",
  # Scalar Like factory
  "make_scalar_like_type",
  # ArrayLike template
  "ArrayLike",
  # Array Like types
  "BoolLike",
  "I8Like",
  "I16Like",
  "I32Like",
  "I64Like",
  "U8Like",
  "U16Like",
  "U32Like",
  "U64Like",
  "F16Like",
  "F32Like",
  "F64Like",
  "F128Like",
  "C64Like",
  "C128Like",
  "C256Like",
  "IntLike",
  "UIntLike",
  "IntegerLike",
  "FloatLike",
  "RealLike",
  "ComplexLike",
  "InexactLike",
  "NumLike",
  "ShapedLike",
]

from ._array_types import _is_valid_casting, make_array_type
from ._array_types import make_array_like_type as _make_array_like_type
from ._dtypes import (
  BOOL,
  BYTES,
  COMPLEX,
  COMPLEX64,
  COMPLEX128,
  COMPLEX256,
  DATETIME64,
  FLOAT,
  FLOAT16,
  FLOAT32,
  FLOAT64,
  FLOAT128,
  INEXACT,
  INT,
  INT8,
  INT16,
  INT32,
  INT64,
  INTEGER,
  NUM,
  OBJECT,
  REAL,
  SHAPED,
  STRING,
  TIMEDELTA64,
  UINT,
  UINT8,
  UINT16,
  UINT32,
  UINT64,
  VOID,
  DtypeSpec,
)

# ---------------------------------------------------------------------------
# Backend-scoped fast-path trust: only np.ndarray skips conversion.
# Foreign-backend arrays (e.g. torch.Tensor, jax.Array) fall through to the
# slow path where np.asarray() verifies actual convertibility.
# ---------------------------------------------------------------------------

_NUMPY_TRUSTED: tuple[type, ...] = (np.ndarray,)


def make_array_like_type(
  dtype_spec: DtypeSpec,
  *,
  casting: str = "same_kind",
  name: str = "ArrayLike",
  asarray: tp.Callable[[object], object] | None = None,
  trusted_types: tuple[type[object], ...] | None = _NUMPY_TRUSTED,
) -> tp.Any:
  """NumPy-aware version of :func:`shapix.make_array_like_type`.

  The fast path only trusts ``np.ndarray``; other backend arrays
  (e.g. ``torch.Tensor``, ``jax.Array``) go through the slow path
  where ``np.asarray()`` verifies actual convertibility.
  """
  return _make_array_like_type(
    dtype_spec, casting=casting, name=name, asarray=asarray, trusted_types=trusted_types
  )


# ---------------------------------------------------------------------------
# Numeric bounds (used by ScalarLike range validation at runtime)
# ---------------------------------------------------------------------------

_INT8_MIN, _INT8_MAX = np.iinfo(np.int8).min, np.iinfo(np.int8).max
_INT16_MIN, _INT16_MAX = np.iinfo(np.int16).min, np.iinfo(np.int16).max
_INT32_MIN, _INT32_MAX = np.iinfo(np.int32).min, np.iinfo(np.int32).max
_INT64_MIN, _INT64_MAX = np.iinfo(np.int64).min, np.iinfo(np.int64).max
_UINT8_MAX = np.iinfo(np.uint8).max
_UINT16_MAX = np.iinfo(np.uint16).max
_UINT32_MAX = np.iinfo(np.uint32).max
_UINT64_MAX = np.iinfo(np.uint64).max
_FLOAT16_MIN, _FLOAT16_MAX = np.finfo(np.float16).min, np.finfo(np.float16).max
_FLOAT32_MIN, _FLOAT32_MAX = np.finfo(np.float32).min, np.finfo(np.float32).max
_FLOAT64_MIN, _FLOAT64_MAX = np.finfo(np.float64).min, np.finfo(np.float64).max
_FLOAT128_MIN, _FLOAT128_MAX = (
  np.finfo(np.longdouble).min,
  np.finfo(np.longdouble).max,
)


def _ge(v: object) -> tp.Any:
  return Is[lambda x: x >= v]


def _le(v: object) -> tp.Any:
  return Is[lambda x: x <= v]


def _not_bool(x: object) -> bool:
  return not isinstance(x, bool | np.bool_)


_nb = Is[_not_bool]


# ---------------------------------------------------------------------------
# Scalar "Like" types (range-validated via beartype)
# ---------------------------------------------------------------------------

StringLike: tp.TypeAlias = str | np.str_
BoolScalarLike: tp.TypeAlias = bool | np.bool_

I8ScalarLike: tp.TypeAlias = Annotated[
  int | np.integer[tp.Any], _ge(_INT8_MIN) & _le(_INT8_MAX) & _nb
]
I16ScalarLike: tp.TypeAlias = Annotated[
  int | np.integer[tp.Any], _ge(_INT16_MIN) & _le(_INT16_MAX) & _nb
]
I32ScalarLike: tp.TypeAlias = Annotated[
  int | np.integer[tp.Any], _ge(_INT32_MIN) & _le(_INT32_MAX) & _nb
]
I64ScalarLike: tp.TypeAlias = Annotated[
  int | np.integer[tp.Any], _ge(_INT64_MIN) & _le(_INT64_MAX) & _nb
]

U8ScalarLike: tp.TypeAlias = Annotated[
  int | np.integer[tp.Any], _ge(0) & _le(_UINT8_MAX) & _nb
]
U16ScalarLike: tp.TypeAlias = Annotated[
  int | np.integer[tp.Any], _ge(0) & _le(_UINT16_MAX) & _nb
]
U32ScalarLike: tp.TypeAlias = Annotated[
  int | np.integer[tp.Any], _ge(0) & _le(_UINT32_MAX) & _nb
]
U64ScalarLike: tp.TypeAlias = Annotated[
  int | np.integer[tp.Any], _ge(0) & _le(_UINT64_MAX) & _nb
]

F16ScalarLike: tp.TypeAlias = Annotated[
  float | np.floating[tp.Any], _ge(_FLOAT16_MIN) & _le(_FLOAT16_MAX) & _nb
]
F32ScalarLike: tp.TypeAlias = Annotated[
  float | np.floating[tp.Any], _ge(_FLOAT32_MIN) & _le(_FLOAT32_MAX) & _nb
]
F64ScalarLike: tp.TypeAlias = Annotated[
  float | np.floating[tp.Any], _ge(_FLOAT64_MIN) & _le(_FLOAT64_MAX) & _nb
]
F128ScalarLike: tp.TypeAlias = Annotated[
  float | np.floating[tp.Any], _ge(_FLOAT128_MIN) & _le(_FLOAT128_MAX) & _nb
]

C64ScalarLike: tp.TypeAlias = Annotated[
  complex | np.complexfloating[tp.Any, tp.Any], _nb
]
C128ScalarLike: tp.TypeAlias = Annotated[
  complex | np.complexfloating[tp.Any, tp.Any], _nb
]
C256ScalarLike: tp.TypeAlias = Annotated[
  complex | np.complexfloating[tp.Any, tp.Any], _nb
]

IntScalarLike: tp.TypeAlias = I64ScalarLike
UIntScalarLike: tp.TypeAlias = U64ScalarLike
IntegerScalarLike: tp.TypeAlias = Annotated[int | np.integer[tp.Any], _nb]
FloatScalarLike: tp.TypeAlias = F64ScalarLike
RealScalarLike: tp.TypeAlias = Annotated[
  int | float | np.integer[tp.Any] | np.floating[tp.Any], _nb
]
ComplexScalarLike: tp.TypeAlias = Annotated[
  complex | np.complexfloating[tp.Any, tp.Any], _nb
]
InexactScalarLike: tp.TypeAlias = Annotated[float | complex | np.inexact[tp.Any], _nb]
NumScalarLike: tp.TypeAlias = Annotated[int | float | complex | np.number[tp.Any], _nb]
ShapedScalarLike: tp.TypeAlias = bool | np.bool_ | NumScalarLike

# ---------------------------------------------------------------------------
# ArrayLike template (static type checking only)
# ---------------------------------------------------------------------------

_Scalar = tp.TypeVar("_Scalar")
_Dtype = tp.TypeVar("_Dtype", bound=np.generic)

ArrayLike: tp.TypeAlias = (
  _Scalar
  | _SupportsArray[np.dtype[_Dtype]]
  | _NestedSequence[_Scalar | _SupportsArray[np.dtype[_Dtype]]]
  | tuple[_Scalar | _SupportsArray[np.dtype[_Dtype]], ...]
)

# ---------------------------------------------------------------------------
# Structured dtype helper
# ---------------------------------------------------------------------------


def Structured(  # noqa: N802
  fields: object,
) -> object:
  """Create a subscriptable array type for a structured numpy dtype.

  Example::

      Point = Structured([("x", np.float32), ("y", np.float32)])


      @beartype
      def f(points: Point[N]) -> Point[N]: ...
  """
  return make_array_type(np.ndarray, DtypeSpec.structured(fields))


# ---------------------------------------------------------------------------
# Array types (shape-checked via shapix runtime hints)
# ---------------------------------------------------------------------------

if tp.TYPE_CHECKING:
  from numpy.typing import NDArray

  _Dims = tp.TypeVarTuple("_Dims")

  # --- Base types ---
  Bool = tp.TypeAliasType("Bool", NDArray[np.bool_], type_params=(_Dims,))

  I8 = tp.TypeAliasType("I8", NDArray[np.int8], type_params=(_Dims,))
  I16 = tp.TypeAliasType("I16", NDArray[np.int16], type_params=(_Dims,))
  I32 = tp.TypeAliasType("I32", NDArray[np.int32], type_params=(_Dims,))
  I64 = tp.TypeAliasType("I64", NDArray[np.int64], type_params=(_Dims,))

  U8 = tp.TypeAliasType("U8", NDArray[np.uint8], type_params=(_Dims,))
  U16 = tp.TypeAliasType("U16", NDArray[np.uint16], type_params=(_Dims,))
  U32 = tp.TypeAliasType("U32", NDArray[np.uint32], type_params=(_Dims,))
  U64 = tp.TypeAliasType("U64", NDArray[np.uint64], type_params=(_Dims,))

  F16 = tp.TypeAliasType("F16", NDArray[np.float16], type_params=(_Dims,))
  F32 = tp.TypeAliasType("F32", NDArray[np.float32], type_params=(_Dims,))
  F64 = tp.TypeAliasType("F64", NDArray[np.float64], type_params=(_Dims,))
  F128 = tp.TypeAliasType("F128", NDArray[np.longdouble], type_params=(_Dims,))

  C64 = tp.TypeAliasType("C64", NDArray[np.complex64], type_params=(_Dims,))
  C128 = tp.TypeAliasType("C128", NDArray[np.complex128], type_params=(_Dims,))
  C256 = tp.TypeAliasType("C256", NDArray[np.clongdouble], type_params=(_Dims,))

  Int = tp.TypeAliasType("Int", NDArray[np.signedinteger[tp.Any]], type_params=(_Dims,))
  UInt = tp.TypeAliasType(
    "UInt", NDArray[np.unsignedinteger[tp.Any]], type_params=(_Dims,)
  )
  Integer = tp.TypeAliasType(
    "Integer", NDArray[np.integer[tp.Any]], type_params=(_Dims,)
  )
  Float = tp.TypeAliasType("Float", NDArray[np.floating[tp.Any]], type_params=(_Dims,))
  Real = tp.TypeAliasType(
    "Real", NDArray[np.integer[tp.Any] | np.floating[tp.Any]], type_params=(_Dims,)
  )
  Complex = tp.TypeAliasType(
    "Complex", NDArray[np.complexfloating[tp.Any, tp.Any]], type_params=(_Dims,)
  )
  Inexact = tp.TypeAliasType(
    "Inexact", NDArray[np.inexact[tp.Any]], type_params=(_Dims,)
  )
  Num = tp.TypeAliasType("Num", NDArray[np.number[tp.Any]], type_params=(_Dims,))
  Shaped = tp.TypeAliasType(
    "Shaped", NDArray[np.bool_ | np.number[tp.Any]], type_params=(_Dims,)
  )

  # --- New dtypes ---
  V = tp.TypeAliasType("V", NDArray[np.void], type_params=(_Dims,))
  Str = tp.TypeAliasType("Str", NDArray[np.str_], type_params=(_Dims,))
  Bytes = tp.TypeAliasType("Bytes", NDArray[np.bytes_], type_params=(_Dims,))
  Obj = tp.TypeAliasType("Obj", NDArray[np.object_], type_params=(_Dims,))
  DT64 = tp.TypeAliasType("DT64", NDArray[np.datetime64], type_params=(_Dims,))
  TD64 = tp.TypeAliasType("TD64", NDArray[np.timedelta64], type_params=(_Dims,))

  # --- Like types (static: ArrayLike template with bare scalar types) ---
  BoolLike = tp.TypeAliasType(
    "BoolLike", ArrayLike[bool, np.bool_], type_params=(_Dims,)
  )

  I8Like = tp.TypeAliasType("I8Like", ArrayLike[int, np.int8], type_params=(_Dims,))
  I16Like = tp.TypeAliasType("I16Like", ArrayLike[int, np.int16], type_params=(_Dims,))
  I32Like = tp.TypeAliasType("I32Like", ArrayLike[int, np.int32], type_params=(_Dims,))
  I64Like = tp.TypeAliasType("I64Like", ArrayLike[int, np.int64], type_params=(_Dims,))

  U8Like = tp.TypeAliasType("U8Like", ArrayLike[int, np.uint8], type_params=(_Dims,))
  U16Like = tp.TypeAliasType("U16Like", ArrayLike[int, np.uint16], type_params=(_Dims,))
  U32Like = tp.TypeAliasType("U32Like", ArrayLike[int, np.uint32], type_params=(_Dims,))
  U64Like = tp.TypeAliasType("U64Like", ArrayLike[int, np.uint64], type_params=(_Dims,))

  F16Like = tp.TypeAliasType(
    "F16Like", ArrayLike[float, np.float16], type_params=(_Dims,)
  )
  F32Like = tp.TypeAliasType(
    "F32Like", ArrayLike[float, np.float32], type_params=(_Dims,)
  )
  F64Like = tp.TypeAliasType(
    "F64Like", ArrayLike[float, np.float64], type_params=(_Dims,)
  )
  F128Like = tp.TypeAliasType(
    "F128Like", ArrayLike[float, np.longdouble], type_params=(_Dims,)
  )

  C64Like = tp.TypeAliasType(
    "C64Like", ArrayLike[complex, np.complex64], type_params=(_Dims,)
  )
  C128Like = tp.TypeAliasType(
    "C128Like", ArrayLike[complex, np.complex128], type_params=(_Dims,)
  )
  C256Like = tp.TypeAliasType(
    "C256Like", ArrayLike[complex, np.clongdouble], type_params=(_Dims,)
  )

  IntLike = tp.TypeAliasType(
    "IntLike", ArrayLike[int, np.signedinteger[tp.Any]], type_params=(_Dims,)
  )
  UIntLike = tp.TypeAliasType(
    "UIntLike", ArrayLike[int, np.unsignedinteger[tp.Any]], type_params=(_Dims,)
  )
  IntegerLike = tp.TypeAliasType(
    "IntegerLike", ArrayLike[int, np.integer[tp.Any]], type_params=(_Dims,)
  )
  FloatLike = tp.TypeAliasType(
    "FloatLike", ArrayLike[float, np.floating[tp.Any]], type_params=(_Dims,)
  )
  RealLike = tp.TypeAliasType(
    "RealLike",
    ArrayLike[int | float, np.integer[tp.Any] | np.floating[tp.Any]],
    type_params=(_Dims,),
  )
  ComplexLike = tp.TypeAliasType(
    "ComplexLike",
    ArrayLike[complex, np.complexfloating[tp.Any, tp.Any]],
    type_params=(_Dims,),
  )
  InexactLike = tp.TypeAliasType(
    "InexactLike", ArrayLike[float | complex, np.inexact[tp.Any]], type_params=(_Dims,)
  )
  NumLike = tp.TypeAliasType(
    "NumLike", ArrayLike[int | float | complex, np.number[tp.Any]], type_params=(_Dims,)
  )
  ShapedLike = tp.TypeAliasType(
    "ShapedLike",
    ArrayLike[bool | int | float | complex, np.bool_ | np.number[tp.Any]],
    type_params=(_Dims,),
  )

else:
  # --- Base types ---
  Bool = make_array_type(np.ndarray, BOOL)

  I8 = make_array_type(np.ndarray, INT8)
  I16 = make_array_type(np.ndarray, INT16)
  I32 = make_array_type(np.ndarray, INT32)
  I64 = make_array_type(np.ndarray, INT64)

  U8 = make_array_type(np.ndarray, UINT8)
  U16 = make_array_type(np.ndarray, UINT16)
  U32 = make_array_type(np.ndarray, UINT32)
  U64 = make_array_type(np.ndarray, UINT64)

  F16 = make_array_type(np.ndarray, FLOAT16)
  F32 = make_array_type(np.ndarray, FLOAT32)
  F64 = make_array_type(np.ndarray, FLOAT64)
  F128 = make_array_type(np.ndarray, FLOAT128)

  C64 = make_array_type(np.ndarray, COMPLEX64)
  C128 = make_array_type(np.ndarray, COMPLEX128)
  C256 = make_array_type(np.ndarray, COMPLEX256)

  Int = make_array_type(np.ndarray, INT)
  UInt = make_array_type(np.ndarray, UINT)
  Integer = make_array_type(np.ndarray, INTEGER)
  Float = make_array_type(np.ndarray, FLOAT)
  Real = make_array_type(np.ndarray, REAL)
  Complex = make_array_type(np.ndarray, COMPLEX)
  Inexact = make_array_type(np.ndarray, INEXACT)
  Num = make_array_type(np.ndarray, NUM)
  Shaped = make_array_type(np.ndarray, SHAPED)

  # --- New dtypes ---
  V = make_array_type(np.ndarray, VOID)
  Str = make_array_type(np.ndarray, STRING)
  Bytes = make_array_type(np.ndarray, BYTES)
  Obj = make_array_type(np.ndarray, OBJECT)
  DT64 = make_array_type(np.ndarray, DATETIME64)
  TD64 = make_array_type(np.ndarray, TIMEDELTA64)

  # --- Like types (runtime: factory with dtype + casting) ---
  BoolLike = make_array_like_type(BOOL, name="BoolLike")

  I8Like = make_array_like_type(INT8, name="I8Like")
  I16Like = make_array_like_type(INT16, name="I16Like")
  I32Like = make_array_like_type(INT32, name="I32Like")
  I64Like = make_array_like_type(INT64, name="I64Like")

  U8Like = make_array_like_type(UINT8, name="U8Like")
  U16Like = make_array_like_type(UINT16, name="U16Like")
  U32Like = make_array_like_type(UINT32, name="U32Like")
  U64Like = make_array_like_type(UINT64, name="U64Like")

  F16Like = make_array_like_type(FLOAT16, name="F16Like")
  F32Like = make_array_like_type(FLOAT32, name="F32Like")
  F64Like = make_array_like_type(FLOAT64, name="F64Like")
  F128Like = make_array_like_type(FLOAT128, name="F128Like")

  C64Like = make_array_like_type(COMPLEX64, name="C64Like")
  C128Like = make_array_like_type(COMPLEX128, name="C128Like")
  C256Like = make_array_like_type(COMPLEX256, name="C256Like")

  IntLike = make_array_like_type(INT, name="IntLike")
  UIntLike = make_array_like_type(UINT, name="UIntLike")
  IntegerLike = make_array_like_type(INTEGER, name="IntegerLike")
  FloatLike = make_array_like_type(FLOAT, name="FloatLike")
  RealLike = make_array_like_type(REAL, name="RealLike")
  ComplexLike = make_array_like_type(COMPLEX, name="ComplexLike")
  InexactLike = make_array_like_type(INEXACT, name="InexactLike")
  NumLike = make_array_like_type(NUM, name="NumLike")
  ShapedLike = make_array_like_type(SHAPED, name="ShapedLike")


# ---------------------------------------------------------------------------
# ScalarLike factory (casting-aware scalar type creation)
# ---------------------------------------------------------------------------

_SCALAR_BASE = bool | int | float | complex | np.generic


def make_scalar_like_type(
  target_dtype: np.dtype[tp.Any] | type[np.generic] | str,
  *,
  casting: str = "same_kind",
  name: str = "ScalarLike",
) -> type:
  """Create a scalar-like type with casting validation.

  Uses ``np.can_cast(value, target_dtype, casting=casting)`` at runtime.

  Parameters
  ----------
  target_dtype : dtype-like
      Target NumPy dtype (e.g. ``np.float32``, ``"int8"``).
  casting : str
      NumPy casting rule: ``"no"`` | ``"equiv"`` | ``"safe"``
      | ``"same_kind"`` | ``"unsafe"``.
  name : str
      Human-readable name for error messages.

  Returns
  -------
  type
      An ``Annotated`` type that beartype validates at runtime.

  """
  from ._array_types import _VALID_CASTINGS

  if not _is_valid_casting(casting):
    msg = f"Invalid casting {casting!r}, must be one of {sorted(_VALID_CASTINGS)}"
    raise ValueError(msg)

  target = np.dtype(target_dtype)

  def _check(value: object) -> bool:
    if isinstance(value, bool | np.bool_) and target != np.dtype(np.bool_):
      return False
    try:
      arr = np.asarray(value)
      return np.can_cast(arr.dtype, target, casting=casting)
    except (TypeError, ValueError):
      return False

  _check.__name__ = name
  _check.__qualname__ = name
  return tp.cast("type", Annotated[_SCALAR_BASE, Is[_check]])
