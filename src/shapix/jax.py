# pyright: reportMissingImports=false
"""JAX array type annotations with runtime shape and dtype checking.

Usage::

    from shapix import N, C, H, W
    from shapix.jax import F32, BF16


    @beartype
    def forward(x: F32[N, C, H, W]) -> BF16[N, C, H, W]: ...
"""

from __future__ import annotations

import typing as tp

import numpy as np
from jax import Array as JaxArray

from ._array_types import make_array_type
from ._dtypes import (
  BFLOAT16,
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
# Array types (shape-checked via beartype Is[])
# ---------------------------------------------------------------------------

if tp.TYPE_CHECKING:
  type Bool[*Dims] = JaxArray

  type I8[*Dims] = JaxArray
  type I16[*Dims] = JaxArray
  type I32[*Dims] = JaxArray
  type I64[*Dims] = JaxArray

  type U8[*Dims] = JaxArray
  type U16[*Dims] = JaxArray
  type U32[*Dims] = JaxArray
  type U64[*Dims] = JaxArray

  type F16[*Dims] = JaxArray
  type F32[*Dims] = JaxArray
  type F64[*Dims] = JaxArray
  type BF16[*Dims] = JaxArray

  type C64[*Dims] = JaxArray
  type C128[*Dims] = JaxArray

  type Int[*Dims] = JaxArray
  type UInt[*Dims] = JaxArray
  type Integer[*Dims] = JaxArray
  type Float[*Dims] = JaxArray
  type Real[*Dims] = JaxArray
  type Complex[*Dims] = JaxArray
  type Inexact[*Dims] = JaxArray
  type Num[*Dims] = JaxArray
  type Shaped[*Dims] = JaxArray

else:
  Bool = make_array_type(JaxArray, BOOL)

  I8 = make_array_type(JaxArray, INT8)
  I16 = make_array_type(JaxArray, INT16)
  I32 = make_array_type(JaxArray, INT32)
  I64 = make_array_type(JaxArray, INT64)

  U8 = make_array_type(JaxArray, UINT8)
  U16 = make_array_type(JaxArray, UINT16)
  U32 = make_array_type(JaxArray, UINT32)
  U64 = make_array_type(JaxArray, UINT64)

  F16 = make_array_type(JaxArray, FLOAT16)
  F32 = make_array_type(JaxArray, FLOAT32)
  F64 = make_array_type(JaxArray, FLOAT64)
  BF16 = make_array_type(JaxArray, BFLOAT16)

  C64 = make_array_type(JaxArray, COMPLEX64)
  C128 = make_array_type(JaxArray, COMPLEX128)

  Int = make_array_type(JaxArray, INT)
  UInt = make_array_type(JaxArray, UINT)
  Integer = make_array_type(JaxArray, INTEGER)
  Float = make_array_type(JaxArray, FLOAT)
  Real = make_array_type(JaxArray, REAL)
  Complex = make_array_type(JaxArray, COMPLEX)
  Inexact = make_array_type(JaxArray, INEXACT)
  Num = make_array_type(JaxArray, NUM)
  Shaped = make_array_type(JaxArray, SHAPED)

# ---------------------------------------------------------------------------
# Like types (scalar | array | nested sequences — for input validation)
# ---------------------------------------------------------------------------

from .numpy import (
  ArrayLike,
  BoolLike,
  Complex64Like,
  Complex128Like,
  ComplexLike,
  Float16Like,
  Float32Like,
  Float64Like,
  FloatLike,
  InexactLike,
  Int8Like,
  Int16Like,
  Int32Like,
  Int64Like,
  IntegerLike,
  IntLike,
  NumLike,
  RealLike,
  ShapedLike,
  UInt8Like,
  UInt16Like,
  UInt32Like,
  UInt64Like,
  UIntLike,
)

type BoolLk = ArrayLike[BoolLike, JaxArray | np.ndarray]

type I8Like = ArrayLike[Int8Like, JaxArray | np.ndarray]
type I16Like = ArrayLike[Int16Like, JaxArray | np.ndarray]
type I32Like = ArrayLike[Int32Like, JaxArray | np.ndarray]
type I64Like = ArrayLike[Int64Like, JaxArray | np.ndarray]

type U8Like = ArrayLike[UInt8Like, JaxArray | np.ndarray]
type U16Like = ArrayLike[UInt16Like, JaxArray | np.ndarray]
type U32Like = ArrayLike[UInt32Like, JaxArray | np.ndarray]
type U64Like = ArrayLike[UInt64Like, JaxArray | np.ndarray]

type F16Like = ArrayLike[Float16Like, JaxArray | np.ndarray]
type F32Like = ArrayLike[Float32Like, JaxArray | np.ndarray]
type F64Like = ArrayLike[Float64Like, JaxArray | np.ndarray]

type C64Like = ArrayLike[Complex64Like, JaxArray | np.ndarray]
type C128Like = ArrayLike[Complex128Like, JaxArray | np.ndarray]

type IntLk = ArrayLike[IntLike, JaxArray | np.ndarray]
type UIntLk = ArrayLike[UIntLike, JaxArray | np.ndarray]
type IntegerLk = ArrayLike[IntegerLike, JaxArray | np.ndarray]
type FloatLk = ArrayLike[FloatLike, JaxArray | np.ndarray]
type RealLk = ArrayLike[RealLike, JaxArray | np.ndarray]
type ComplexLk = ArrayLike[ComplexLike, JaxArray | np.ndarray]
type InexactLk = ArrayLike[InexactLike, JaxArray | np.ndarray]
type NumLk = ArrayLike[NumLike, JaxArray | np.ndarray]
type ShapedLk = ArrayLike[ShapedLike, JaxArray | np.ndarray]
