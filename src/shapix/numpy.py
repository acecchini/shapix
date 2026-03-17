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
from typing import Annotated as A

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

from ._array_types import make_array_like_type, make_array_type
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


def _ge(v: object):  # noqa: ANN202
  return Is[lambda x: x >= v]


def _le(v: object):  # noqa: ANN202
  return Is[lambda x: x <= v]


# ---------------------------------------------------------------------------
# Scalar "Like" types (range-validated via beartype)
# ---------------------------------------------------------------------------

type StringLike = str | np.str_
type BoolScalarLike = bool | np.bool_

type I8ScalarLike = A[int | np.integer[tp.Any], _ge(_INT8_MIN) & _le(_INT8_MAX)]  # type: ignore[type-arg]
type I16ScalarLike = A[int | np.integer[tp.Any], _ge(_INT16_MIN) & _le(_INT16_MAX)]  # type: ignore[type-arg]
type I32ScalarLike = A[int | np.integer[tp.Any], _ge(_INT32_MIN) & _le(_INT32_MAX)]  # type: ignore[type-arg]
type I64ScalarLike = A[int | np.integer[tp.Any], _ge(_INT64_MIN) & _le(_INT64_MAX)]  # type: ignore[type-arg]

type U8ScalarLike = A[int | np.integer[tp.Any], _ge(0) & _le(_UINT8_MAX)]  # type: ignore[type-arg]
type U16ScalarLike = A[int | np.integer[tp.Any], _ge(0) & _le(_UINT16_MAX)]  # type: ignore[type-arg]
type U32ScalarLike = A[int | np.integer[tp.Any], _ge(0) & _le(_UINT32_MAX)]  # type: ignore[type-arg]
type U64ScalarLike = A[int | np.integer[tp.Any], _ge(0) & _le(_UINT64_MAX)]  # type: ignore[type-arg]

type F16ScalarLike = A[
  float | np.floating[tp.Any], _ge(_FLOAT16_MIN) & _le(_FLOAT16_MAX)
]  # type: ignore[type-arg]
type F32ScalarLike = A[
  float | np.floating[tp.Any], _ge(_FLOAT32_MIN) & _le(_FLOAT32_MAX)
]  # type: ignore[type-arg]
type F64ScalarLike = A[
  float | np.floating[tp.Any], _ge(_FLOAT64_MIN) & _le(_FLOAT64_MAX)
]  # type: ignore[type-arg]
type F128ScalarLike = A[
  float | np.floating[tp.Any], _ge(_FLOAT128_MIN) & _le(_FLOAT128_MAX)
]  # type: ignore[type-arg]

type C64ScalarLike = complex | np.complexfloating[tp.Any, tp.Any]
type C128ScalarLike = complex | np.complexfloating[tp.Any, tp.Any]
type C256ScalarLike = complex | np.complexfloating[tp.Any, tp.Any]

type IntScalarLike = I64ScalarLike
type UIntScalarLike = U64ScalarLike
type IntegerScalarLike = int | np.integer[tp.Any]
type FloatScalarLike = F64ScalarLike
type RealScalarLike = int | float | np.integer[tp.Any] | np.floating[tp.Any]
type ComplexScalarLike = complex | np.complexfloating[tp.Any, tp.Any]
type InexactScalarLike = float | complex | np.inexact[tp.Any]
type NumScalarLike = int | float | complex | np.number[tp.Any]
type ShapedScalarLike = bool | np.bool_ | NumScalarLike

# ---------------------------------------------------------------------------
# ArrayLike template (static type checking only)
# ---------------------------------------------------------------------------

type ArrayLike[_Scalar, _Dtype: np.generic] = (
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
# Array types (shape-checked via beartype Is[])
# ---------------------------------------------------------------------------

if tp.TYPE_CHECKING:
  from numpy.typing import NDArray

  # --- Base types ---
  type Bool[*Dims] = NDArray[np.bool_]

  type I8[*Dims] = NDArray[np.int8]
  type I16[*Dims] = NDArray[np.int16]
  type I32[*Dims] = NDArray[np.int32]
  type I64[*Dims] = NDArray[np.int64]

  type U8[*Dims] = NDArray[np.uint8]
  type U16[*Dims] = NDArray[np.uint16]
  type U32[*Dims] = NDArray[np.uint32]
  type U64[*Dims] = NDArray[np.uint64]

  type F16[*Dims] = NDArray[np.float16]
  type F32[*Dims] = NDArray[np.float32]
  type F64[*Dims] = NDArray[np.float64]
  type F128[*Dims] = NDArray[np.longdouble]

  type C64[*Dims] = NDArray[np.complex64]
  type C128[*Dims] = NDArray[np.complex128]
  type C256[*Dims] = NDArray[np.clongdouble]

  type Int[*Dims] = NDArray[np.signedinteger[tp.Any]]
  type UInt[*Dims] = NDArray[np.unsignedinteger[tp.Any]]
  type Integer[*Dims] = NDArray[np.integer[tp.Any]]
  type Float[*Dims] = NDArray[np.floating[tp.Any]]
  type Real[*Dims] = NDArray[np.integer[tp.Any] | np.floating[tp.Any]]
  type Complex[*Dims] = NDArray[np.complexfloating[tp.Any, tp.Any]]
  type Inexact[*Dims] = NDArray[np.inexact[tp.Any]]
  type Num[*Dims] = NDArray[np.number[tp.Any]]
  type Shaped[*Dims] = NDArray[np.bool_ | np.number[tp.Any]]

  # --- New dtypes ---
  type V[*Dims] = NDArray[np.void]
  type Str[*Dims] = NDArray[np.str_]
  type Bytes[*Dims] = NDArray[np.bytes_]
  type Obj[*Dims] = NDArray[np.object_]
  type DT64[*Dims] = NDArray[np.datetime64]
  type TD64[*Dims] = NDArray[np.timedelta64]

  # --- Like types (static: ArrayLike template with bare scalar types) ---
  type BoolLike[*Dims] = ArrayLike[bool, np.bool_]

  type I8Like[*Dims] = ArrayLike[int, np.int8]
  type I16Like[*Dims] = ArrayLike[int, np.int16]
  type I32Like[*Dims] = ArrayLike[int, np.int32]
  type I64Like[*Dims] = ArrayLike[int, np.int64]

  type U8Like[*Dims] = ArrayLike[int, np.uint8]
  type U16Like[*Dims] = ArrayLike[int, np.uint16]
  type U32Like[*Dims] = ArrayLike[int, np.uint32]
  type U64Like[*Dims] = ArrayLike[int, np.uint64]

  type F16Like[*Dims] = ArrayLike[float, np.float16]
  type F32Like[*Dims] = ArrayLike[float, np.float32]
  type F64Like[*Dims] = ArrayLike[float, np.float64]
  type F128Like[*Dims] = ArrayLike[float, np.longdouble]

  type C64Like[*Dims] = ArrayLike[complex, np.complex64]
  type C128Like[*Dims] = ArrayLike[complex, np.complex128]
  type C256Like[*Dims] = ArrayLike[complex, np.clongdouble]

  type IntLike[*Dims] = ArrayLike[int, np.signedinteger[tp.Any]]
  type UIntLike[*Dims] = ArrayLike[int, np.unsignedinteger[tp.Any]]
  type IntegerLike[*Dims] = ArrayLike[int, np.integer[tp.Any]]
  type FloatLike[*Dims] = ArrayLike[float, np.floating[tp.Any]]
  type RealLike[*Dims] = ArrayLike[
    int | float, np.integer[tp.Any] | np.floating[tp.Any]
  ]
  type ComplexLike[*Dims] = ArrayLike[complex, np.complexfloating[tp.Any, tp.Any]]
  type InexactLike[*Dims] = ArrayLike[float | complex, np.inexact[tp.Any]]
  type NumLike[*Dims] = ArrayLike[int | float | complex, np.number[tp.Any]]
  type ShapedLike[*Dims] = ArrayLike[
    bool | int | float | complex, np.bool_ | np.number[tp.Any]
  ]

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

  if casting not in _VALID_CASTINGS:
    msg = f"Invalid casting {casting!r}, must be one of {sorted(_VALID_CASTINGS)}"
    raise ValueError(msg)

  target = np.dtype(target_dtype)

  def _check(value: object) -> bool:
    try:
      arr = np.asarray(value)
      return np.can_cast(arr.dtype, target, casting=casting)  # pyright: ignore[reportArgumentType]
    except (TypeError, ValueError):
      return False

  _check.__name__ = name
  _check.__qualname__ = name
  return A[_SCALAR_BASE, Is[_check]]  # type: ignore[return-value]
