"""NumPy array type annotations with runtime shape and dtype checking.

Scalar types (e.g. ``Int8Like``, ``Float32Like``) use beartype's
``Annotated[T, Is[...]]`` to validate value ranges at runtime.

Array types (e.g. ``F32``, ``Int``) are subscriptable with dimension symbols::

    from shapix import N, C, H, W
    from shapix.numpy import F32


    @beartype
    def conv(x: F32[N, C, H, W]) -> F32[N, C, H, W]: ...

``Like`` types (e.g. ``F32Like``) accept scalars, nested sequences, or arrays
— use them for function inputs that will be converted to arrays.
"""

from __future__ import annotations

import typing as tp
from collections.abc import Sequence
from typing import Annotated as A

import numpy as np
from beartype.vale import Is

from ._array_types import make_array_type
from ._dtypes import (
  BOOL,
  COMPLEX,
  COMPLEX64,
  COMPLEX128,
  FLOAT,
  FLOAT16,
  FLOAT32,
  FLOAT64,
  INT,
  INT8,
  INT16,
  INT32,
  INT64,
  INTEGER,
  INEXACT,
  NUM,
  REAL,
  SHAPED,
  UINT,
  UINT8,
  UINT16,
  UINT32,
  UINT64,
)

# ---------------------------------------------------------------------------
# Numeric bounds
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


def _ge(v: int | float):  # noqa: ANN202
  return Is[lambda x: x >= v]


def _le(v: int | float):  # noqa: ANN202
  return Is[lambda x: x <= v]


# ---------------------------------------------------------------------------
# Scalar "Like" types (range-validated via beartype)
# ---------------------------------------------------------------------------

type StringLike = str | np.str_
type BoolLike = bool | np.bool_

type Int8Like = A[int | np.integer[tp.Any], _ge(_INT8_MIN) & _le(_INT8_MAX)]  # type: ignore[type-arg]
type Int16Like = A[int | np.integer[tp.Any], _ge(_INT16_MIN) & _le(_INT16_MAX)]  # type: ignore[type-arg]
type Int32Like = A[int | np.integer[tp.Any], _ge(_INT32_MIN) & _le(_INT32_MAX)]  # type: ignore[type-arg]
type Int64Like = (
  A[int, _ge(_INT64_MIN) & _le(_INT64_MAX)] | A[np.integer[tp.Any], _le(_INT64_MAX)]
)  # type: ignore[type-arg]

type UInt8Like = A[int | np.integer[tp.Any], _ge(0) & _le(_UINT8_MAX)]  # type: ignore[type-arg]
type UInt16Like = A[int | np.integer[tp.Any], _ge(0) & _le(_UINT16_MAX)]  # type: ignore[type-arg]
type UInt32Like = A[int | np.integer[tp.Any], _ge(0) & _le(_UINT32_MAX)]  # type: ignore[type-arg]
type UInt64Like = A[int, _ge(0) & _le(_UINT64_MAX)] | A[np.integer[tp.Any], _ge(0)]  # type: ignore[type-arg]

type Float16Like = A[float | np.floating[tp.Any], _ge(_FLOAT16_MIN) & _le(_FLOAT16_MAX)]  # type: ignore[type-arg]
type Float32Like = A[float | np.floating[tp.Any], _ge(_FLOAT32_MIN) & _le(_FLOAT32_MAX)]  # type: ignore[type-arg]
type Float64Like = A[float | np.floating[tp.Any], _ge(_FLOAT64_MIN) & _le(_FLOAT64_MAX)]  # type: ignore[type-arg]

type Complex64Like = complex | np.complexfloating[tp.Any, tp.Any]
type Complex128Like = complex | np.complexfloating[tp.Any, tp.Any]

type IntLike = Int64Like
type UIntLike = UInt64Like
type IntegerLike = int | np.integer[tp.Any]
type FloatLike = Float64Like
type RealLike = int | float | np.integer[tp.Any] | np.floating[tp.Any]
type ComplexLike = complex | np.complexfloating[tp.Any, tp.Any]
type InexactLike = int | float | complex | np.inexact[tp.Any]
type NumLike = int | float | complex | np.number[tp.Any]
type ShapedLike = bool | np.bool_ | NumLike

type Fraction = A[FloatLike, _ge(0.0) & _le(1.0)]  # type: ignore[type-arg]
type Seed = np.uint64
type SeedLike = UInt64Like

# ---------------------------------------------------------------------------
# Recursive ArrayLike template (scalar | array | nested sequences of any depth)
# ---------------------------------------------------------------------------

type ArrayLike[_Scalar, _Array] = (
  _Scalar | _Array | Sequence[ArrayLike[_Scalar, _Array]]
)

# ---------------------------------------------------------------------------
# Array types (shape-checked via beartype Is[])
# ---------------------------------------------------------------------------

if tp.TYPE_CHECKING:
  from numpy.typing import NDArray

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

  type C64[*Dims] = NDArray[np.complex64]
  type C128[*Dims] = NDArray[np.complex128]

  type Int[*Dims] = NDArray[np.signedinteger[tp.Any]]
  type UInt[*Dims] = NDArray[np.unsignedinteger[tp.Any]]
  type Integer[*Dims] = NDArray[np.integer[tp.Any]]
  type Float[*Dims] = NDArray[np.floating[tp.Any]]
  type Real[*Dims] = NDArray[np.integer[tp.Any] | np.floating[tp.Any]]
  type Complex[*Dims] = NDArray[np.complexfloating[tp.Any, tp.Any]]
  type Inexact[*Dims] = NDArray[np.inexact[tp.Any]]
  type Num[*Dims] = NDArray[np.number[tp.Any]]
  type Shaped[*Dims] = NDArray[np.bool_ | np.number[tp.Any]]

else:
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

  C64 = make_array_type(np.ndarray, COMPLEX64)
  C128 = make_array_type(np.ndarray, COMPLEX128)

  Int = make_array_type(np.ndarray, INT)
  UInt = make_array_type(np.ndarray, UINT)
  Integer = make_array_type(np.ndarray, INTEGER)
  Float = make_array_type(np.ndarray, FLOAT)
  Real = make_array_type(np.ndarray, REAL)
  Complex = make_array_type(np.ndarray, COMPLEX)
  Inexact = make_array_type(np.ndarray, INEXACT)
  Num = make_array_type(np.ndarray, NUM)
  Shaped = make_array_type(np.ndarray, SHAPED)

# ---------------------------------------------------------------------------
# Like types (scalar | array | nested sequences — for input validation)
# ---------------------------------------------------------------------------

type BoolLk = ArrayLike[BoolLike, np.ndarray]

type I8Like = ArrayLike[Int8Like, np.ndarray]
type I16Like = ArrayLike[Int16Like, np.ndarray]
type I32Like = ArrayLike[Int32Like, np.ndarray]
type I64Like = ArrayLike[Int64Like, np.ndarray]

type U8Like = ArrayLike[UInt8Like, np.ndarray]
type U16Like = ArrayLike[UInt16Like, np.ndarray]
type U32Like = ArrayLike[UInt32Like, np.ndarray]
type U64Like = ArrayLike[UInt64Like, np.ndarray]

type F16Like = ArrayLike[Float16Like, np.ndarray]
type F32Like = ArrayLike[Float32Like, np.ndarray]
type F64Like = ArrayLike[Float64Like, np.ndarray]

type C64Like = ArrayLike[Complex64Like, np.ndarray]
type C128Like = ArrayLike[Complex128Like, np.ndarray]

type IntLk = ArrayLike[IntLike, np.ndarray]
type UIntLk = ArrayLike[UIntLike, np.ndarray]
type IntegerLk = ArrayLike[IntegerLike, np.ndarray]
type FloatLk = ArrayLike[FloatLike, np.ndarray]
type RealLk = ArrayLike[RealLike, np.ndarray]
type ComplexLk = ArrayLike[ComplexLike, np.ndarray]
type InexactLk = ArrayLike[InexactLike, np.ndarray]
type NumLk = ArrayLike[NumLike, np.ndarray]
type ShapedLk = ArrayLike[ShapedLike, np.ndarray]
