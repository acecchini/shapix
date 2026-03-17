"""NumPy array type annotations with runtime shape and dtype checking.

Array types (e.g. ``F32``, ``Int``) are subscriptable with dimension symbols::

    from shapix import N, C, H, W
    from shapix.numpy import F32


    @beartype
    def conv(x: F32[N, C, H, W]) -> F32[N, C, H, W]: ...

Endianness variants use ``LE`` / ``BE`` / ``N`` suffixes::

    from shapix.numpy import F32LE, I64BE, I32N

``Like`` types (e.g. ``F32Like``) accept scalars, nested sequences, or arrays
— use them for function inputs that will be converted to arrays::

    from shapix.numpy import F32Like


    @beartype
    def f(x: F32Like[N, C]) -> F32[N, C]: ...
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
  "C64",
  "C128",
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
  # Array types — endianness (LE)
  "I16LE",
  "I32LE",
  "I64LE",
  "U16LE",
  "U32LE",
  "U64LE",
  "F16LE",
  "F32LE",
  "F64LE",
  "C64LE",
  "C128LE",
  "IntLE",
  "UIntLE",
  "IntegerLE",
  "FloatLE",
  "RealLE",
  "ComplexLE",
  "InexactLE",
  "NumLE",
  "ShapedLE",
  # Array types — endianness (BE)
  "I16BE",
  "I32BE",
  "I64BE",
  "U16BE",
  "U32BE",
  "U64BE",
  "F16BE",
  "F32BE",
  "F64BE",
  "C64BE",
  "C128BE",
  "IntBE",
  "UIntBE",
  "IntegerBE",
  "FloatBE",
  "RealBE",
  "ComplexBE",
  "InexactBE",
  "NumBE",
  "ShapedBE",
  # Array types — endianness (native)
  "I16N",
  "I32N",
  "I64N",
  "U16N",
  "U32N",
  "U64N",
  "F16N",
  "F32N",
  "F64N",
  "C64N",
  "C128N",
  "IntN",
  "UIntN",
  "IntegerN",
  "FloatN",
  "RealN",
  "ComplexN",
  "InexactN",
  "NumN",
  "ShapedN",
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
  "C64ScalarLike",
  "C128ScalarLike",
  "IntScalarLike",
  "UIntScalarLike",
  "IntegerScalarLike",
  "FloatScalarLike",
  "RealScalarLike",
  "ComplexScalarLike",
  "InexactScalarLike",
  "NumScalarLike",
  "ShapedScalarLike",
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
  "C64Like",
  "C128Like",
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
  COMPLEX64_BE,
  COMPLEX64_LE,
  COMPLEX64_N,
  COMPLEX128_BE,
  COMPLEX128_LE,
  COMPLEX128_N,
  COMPLEX_BE,
  COMPLEX_LE,
  COMPLEX_N,
  DATETIME64,
  FLOAT,
  FLOAT16,
  FLOAT32,
  FLOAT64,
  FLOAT16_BE,
  FLOAT16_LE,
  FLOAT16_N,
  FLOAT32_BE,
  FLOAT32_LE,
  FLOAT32_N,
  FLOAT64_BE,
  FLOAT64_LE,
  FLOAT64_N,
  FLOAT_BE,
  FLOAT_LE,
  FLOAT_N,
  INEXACT,
  INEXACT_BE,
  INEXACT_LE,
  INEXACT_N,
  INT,
  INT8,
  INT16,
  INT32,
  INT64,
  INT16_BE,
  INT16_LE,
  INT16_N,
  INT32_BE,
  INT32_LE,
  INT32_N,
  INT64_BE,
  INT64_LE,
  INT64_N,
  INT_BE,
  INT_LE,
  INT_N,
  INTEGER,
  INTEGER_BE,
  INTEGER_LE,
  INTEGER_N,
  NUM,
  NUM_BE,
  NUM_LE,
  NUM_N,
  OBJECT,
  REAL,
  REAL_BE,
  REAL_LE,
  REAL_N,
  SHAPED,
  SHAPED_BE,
  SHAPED_LE,
  SHAPED_N,
  STRING,
  TIMEDELTA64,
  UINT,
  UINT8,
  UINT16,
  UINT32,
  UINT64,
  UINT16_BE,
  UINT16_LE,
  UINT16_N,
  UINT32_BE,
  UINT32_LE,
  UINT32_N,
  UINT64_BE,
  UINT64_LE,
  UINT64_N,
  UINT_BE,
  UINT_LE,
  UINT_N,
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
type I64ScalarLike = (
  A[int, _ge(_INT64_MIN) & _le(_INT64_MAX)] | np.integer[tp.Any]
)  # Any numpy integer is in int64 range

type U8ScalarLike = A[int | np.integer[tp.Any], _ge(0) & _le(_UINT8_MAX)]  # type: ignore[type-arg]
type U16ScalarLike = A[int | np.integer[tp.Any], _ge(0) & _le(_UINT16_MAX)]  # type: ignore[type-arg]
type U32ScalarLike = A[int | np.integer[tp.Any], _ge(0) & _le(_UINT32_MAX)]  # type: ignore[type-arg]
type U64ScalarLike = A[int, _ge(0) & _le(_UINT64_MAX)] | A[np.integer[tp.Any], _ge(0)]  # type: ignore[type-arg]

type F16ScalarLike = A[
  float | np.floating[tp.Any], _ge(_FLOAT16_MIN) & _le(_FLOAT16_MAX)
]  # type: ignore[type-arg]
type F32ScalarLike = A[
  float | np.floating[tp.Any], _ge(_FLOAT32_MIN) & _le(_FLOAT32_MAX)
]  # type: ignore[type-arg]
type F64ScalarLike = A[
  float | np.floating[tp.Any], _ge(_FLOAT64_MIN) & _le(_FLOAT64_MAX)
]  # type: ignore[type-arg]

type C64ScalarLike = complex | np.complexfloating[tp.Any, tp.Any]
type C128ScalarLike = complex | np.complexfloating[tp.Any, tp.Any]

type IntScalarLike = I64ScalarLike
type UIntScalarLike = U64ScalarLike
type IntegerScalarLike = int | np.integer[tp.Any]
type FloatScalarLike = F64ScalarLike
type RealScalarLike = int | float | np.integer[tp.Any] | np.floating[tp.Any]
type ComplexScalarLike = complex | np.complexfloating[tp.Any, tp.Any]
type InexactScalarLike = int | float | complex | np.inexact[tp.Any]
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

  # --- New dtypes ---
  type V[*Dims] = NDArray[np.void]
  type Str[*Dims] = NDArray[np.str_]
  type Bytes[*Dims] = NDArray[np.bytes_]
  type Obj[*Dims] = NDArray[np.object_]
  type DT64[*Dims] = NDArray[np.datetime64]
  type TD64[*Dims] = NDArray[np.timedelta64]

  # --- LE variants (static type = same as base; runtime checks endianness) ---
  type I16LE[*Dims] = NDArray[np.int16]
  type I32LE[*Dims] = NDArray[np.int32]
  type I64LE[*Dims] = NDArray[np.int64]
  type U16LE[*Dims] = NDArray[np.uint16]
  type U32LE[*Dims] = NDArray[np.uint32]
  type U64LE[*Dims] = NDArray[np.uint64]
  type F16LE[*Dims] = NDArray[np.float16]
  type F32LE[*Dims] = NDArray[np.float32]
  type F64LE[*Dims] = NDArray[np.float64]
  type C64LE[*Dims] = NDArray[np.complex64]
  type C128LE[*Dims] = NDArray[np.complex128]
  type IntLE[*Dims] = NDArray[np.signedinteger[tp.Any]]
  type UIntLE[*Dims] = NDArray[np.unsignedinteger[tp.Any]]
  type IntegerLE[*Dims] = NDArray[np.integer[tp.Any]]
  type FloatLE[*Dims] = NDArray[np.floating[tp.Any]]
  type RealLE[*Dims] = NDArray[np.integer[tp.Any] | np.floating[tp.Any]]
  type ComplexLE[*Dims] = NDArray[np.complexfloating[tp.Any, tp.Any]]
  type InexactLE[*Dims] = NDArray[np.inexact[tp.Any]]
  type NumLE[*Dims] = NDArray[np.number[tp.Any]]
  type ShapedLE[*Dims] = NDArray[np.bool_ | np.number[tp.Any]]

  # --- BE variants ---
  type I16BE[*Dims] = NDArray[np.int16]
  type I32BE[*Dims] = NDArray[np.int32]
  type I64BE[*Dims] = NDArray[np.int64]
  type U16BE[*Dims] = NDArray[np.uint16]
  type U32BE[*Dims] = NDArray[np.uint32]
  type U64BE[*Dims] = NDArray[np.uint64]
  type F16BE[*Dims] = NDArray[np.float16]
  type F32BE[*Dims] = NDArray[np.float32]
  type F64BE[*Dims] = NDArray[np.float64]
  type C64BE[*Dims] = NDArray[np.complex64]
  type C128BE[*Dims] = NDArray[np.complex128]
  type IntBE[*Dims] = NDArray[np.signedinteger[tp.Any]]
  type UIntBE[*Dims] = NDArray[np.unsignedinteger[tp.Any]]
  type IntegerBE[*Dims] = NDArray[np.integer[tp.Any]]
  type FloatBE[*Dims] = NDArray[np.floating[tp.Any]]
  type RealBE[*Dims] = NDArray[np.integer[tp.Any] | np.floating[tp.Any]]
  type ComplexBE[*Dims] = NDArray[np.complexfloating[tp.Any, tp.Any]]
  type InexactBE[*Dims] = NDArray[np.inexact[tp.Any]]
  type NumBE[*Dims] = NDArray[np.number[tp.Any]]
  type ShapedBE[*Dims] = NDArray[np.bool_ | np.number[tp.Any]]

  # --- Native variants ---
  type I16N[*Dims] = NDArray[np.int16]
  type I32N[*Dims] = NDArray[np.int32]
  type I64N[*Dims] = NDArray[np.int64]
  type U16N[*Dims] = NDArray[np.uint16]
  type U32N[*Dims] = NDArray[np.uint32]
  type U64N[*Dims] = NDArray[np.uint64]
  type F16N[*Dims] = NDArray[np.float16]
  type F32N[*Dims] = NDArray[np.float32]
  type F64N[*Dims] = NDArray[np.float64]
  type C64N[*Dims] = NDArray[np.complex64]
  type C128N[*Dims] = NDArray[np.complex128]
  type IntN[*Dims] = NDArray[np.signedinteger[tp.Any]]
  type UIntN[*Dims] = NDArray[np.unsignedinteger[tp.Any]]
  type IntegerN[*Dims] = NDArray[np.integer[tp.Any]]
  type FloatN[*Dims] = NDArray[np.floating[tp.Any]]
  type RealN[*Dims] = NDArray[np.integer[tp.Any] | np.floating[tp.Any]]
  type ComplexN[*Dims] = NDArray[np.complexfloating[tp.Any, tp.Any]]
  type InexactN[*Dims] = NDArray[np.inexact[tp.Any]]
  type NumN[*Dims] = NDArray[np.number[tp.Any]]
  type ShapedN[*Dims] = NDArray[np.bool_ | np.number[tp.Any]]

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

  type C64Like[*Dims] = ArrayLike[complex, np.complex64]
  type C128Like[*Dims] = ArrayLike[complex, np.complex128]

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

  # --- New dtypes ---
  V = make_array_type(np.ndarray, VOID)
  Str = make_array_type(np.ndarray, STRING)
  Bytes = make_array_type(np.ndarray, BYTES)
  Obj = make_array_type(np.ndarray, OBJECT)
  DT64 = make_array_type(np.ndarray, DATETIME64)
  TD64 = make_array_type(np.ndarray, TIMEDELTA64)

  # --- LE variants ---
  I16LE = make_array_type(np.ndarray, INT16_LE)
  I32LE = make_array_type(np.ndarray, INT32_LE)
  I64LE = make_array_type(np.ndarray, INT64_LE)
  U16LE = make_array_type(np.ndarray, UINT16_LE)
  U32LE = make_array_type(np.ndarray, UINT32_LE)
  U64LE = make_array_type(np.ndarray, UINT64_LE)
  F16LE = make_array_type(np.ndarray, FLOAT16_LE)
  F32LE = make_array_type(np.ndarray, FLOAT32_LE)
  F64LE = make_array_type(np.ndarray, FLOAT64_LE)
  C64LE = make_array_type(np.ndarray, COMPLEX64_LE)
  C128LE = make_array_type(np.ndarray, COMPLEX128_LE)
  IntLE = make_array_type(np.ndarray, INT_LE)
  UIntLE = make_array_type(np.ndarray, UINT_LE)
  IntegerLE = make_array_type(np.ndarray, INTEGER_LE)
  FloatLE = make_array_type(np.ndarray, FLOAT_LE)
  RealLE = make_array_type(np.ndarray, REAL_LE)
  ComplexLE = make_array_type(np.ndarray, COMPLEX_LE)
  InexactLE = make_array_type(np.ndarray, INEXACT_LE)
  NumLE = make_array_type(np.ndarray, NUM_LE)
  ShapedLE = make_array_type(np.ndarray, SHAPED_LE)

  # --- BE variants ---
  I16BE = make_array_type(np.ndarray, INT16_BE)
  I32BE = make_array_type(np.ndarray, INT32_BE)
  I64BE = make_array_type(np.ndarray, INT64_BE)
  U16BE = make_array_type(np.ndarray, UINT16_BE)
  U32BE = make_array_type(np.ndarray, UINT32_BE)
  U64BE = make_array_type(np.ndarray, UINT64_BE)
  F16BE = make_array_type(np.ndarray, FLOAT16_BE)
  F32BE = make_array_type(np.ndarray, FLOAT32_BE)
  F64BE = make_array_type(np.ndarray, FLOAT64_BE)
  C64BE = make_array_type(np.ndarray, COMPLEX64_BE)
  C128BE = make_array_type(np.ndarray, COMPLEX128_BE)
  IntBE = make_array_type(np.ndarray, INT_BE)
  UIntBE = make_array_type(np.ndarray, UINT_BE)
  IntegerBE = make_array_type(np.ndarray, INTEGER_BE)
  FloatBE = make_array_type(np.ndarray, FLOAT_BE)
  RealBE = make_array_type(np.ndarray, REAL_BE)
  ComplexBE = make_array_type(np.ndarray, COMPLEX_BE)
  InexactBE = make_array_type(np.ndarray, INEXACT_BE)
  NumBE = make_array_type(np.ndarray, NUM_BE)
  ShapedBE = make_array_type(np.ndarray, SHAPED_BE)

  # --- Native variants ---
  I16N = make_array_type(np.ndarray, INT16_N)
  I32N = make_array_type(np.ndarray, INT32_N)
  I64N = make_array_type(np.ndarray, INT64_N)
  U16N = make_array_type(np.ndarray, UINT16_N)
  U32N = make_array_type(np.ndarray, UINT32_N)
  U64N = make_array_type(np.ndarray, UINT64_N)
  F16N = make_array_type(np.ndarray, FLOAT16_N)
  F32N = make_array_type(np.ndarray, FLOAT32_N)
  F64N = make_array_type(np.ndarray, FLOAT64_N)
  C64N = make_array_type(np.ndarray, COMPLEX64_N)
  C128N = make_array_type(np.ndarray, COMPLEX128_N)
  IntN = make_array_type(np.ndarray, INT_N)
  UIntN = make_array_type(np.ndarray, UINT_N)
  IntegerN = make_array_type(np.ndarray, INTEGER_N)
  FloatN = make_array_type(np.ndarray, FLOAT_N)
  RealN = make_array_type(np.ndarray, REAL_N)
  ComplexN = make_array_type(np.ndarray, COMPLEX_N)
  InexactN = make_array_type(np.ndarray, INEXACT_N)
  NumN = make_array_type(np.ndarray, NUM_N)
  ShapedN = make_array_type(np.ndarray, SHAPED_N)

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

  C64Like = make_array_like_type(COMPLEX64, name="C64Like")
  C128Like = make_array_like_type(COMPLEX128, name="C128Like")

  IntLike = make_array_like_type(INT, name="IntLike")
  UIntLike = make_array_like_type(UINT, name="UIntLike")
  IntegerLike = make_array_like_type(INTEGER, name="IntegerLike")
  FloatLike = make_array_like_type(FLOAT, name="FloatLike")
  RealLike = make_array_like_type(REAL, name="RealLike")
  ComplexLike = make_array_like_type(COMPLEX, name="ComplexLike")
  InexactLike = make_array_like_type(INEXACT, name="InexactLike")
  NumLike = make_array_like_type(NUM, name="NumLike")
  ShapedLike = make_array_like_type(SHAPED, name="ShapedLike")
