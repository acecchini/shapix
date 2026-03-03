# pyright: reportMissingImports=false
"""PyTorch tensor type annotations with runtime shape and dtype checking.

Usage::

    from shapix import N, C, H, W
    from shapix.torch import F32


    @beartype
    def forward(x: F32[N, C, H, W]) -> F32[N, C, H, W]: ...
"""

from __future__ import annotations

import typing as tp

import numpy as np
from torch import Tensor

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
# Tensor types (shape-checked via beartype Is[])
# ---------------------------------------------------------------------------

if tp.TYPE_CHECKING:
  type Bool[*Dims] = Tensor

  type I8[*Dims] = Tensor
  type I16[*Dims] = Tensor
  type I32[*Dims] = Tensor
  type I64[*Dims] = Tensor

  type U8[*Dims] = Tensor
  type U16[*Dims] = Tensor
  type U32[*Dims] = Tensor
  type U64[*Dims] = Tensor

  type F16[*Dims] = Tensor
  type F32[*Dims] = Tensor
  type F64[*Dims] = Tensor
  type BF16[*Dims] = Tensor

  type C64[*Dims] = Tensor
  type C128[*Dims] = Tensor

  type Int[*Dims] = Tensor
  type UInt[*Dims] = Tensor
  type Integer[*Dims] = Tensor
  type Float[*Dims] = Tensor
  type Real[*Dims] = Tensor
  type Complex[*Dims] = Tensor
  type Inexact[*Dims] = Tensor
  type Num[*Dims] = Tensor
  type Shaped[*Dims] = Tensor

else:
  Bool = make_array_type(Tensor, BOOL)

  I8 = make_array_type(Tensor, INT8)
  I16 = make_array_type(Tensor, INT16)
  I32 = make_array_type(Tensor, INT32)
  I64 = make_array_type(Tensor, INT64)

  U8 = make_array_type(Tensor, UINT8)
  U16 = make_array_type(Tensor, UINT16)
  U32 = make_array_type(Tensor, UINT32)
  U64 = make_array_type(Tensor, UINT64)

  F16 = make_array_type(Tensor, FLOAT16)
  F32 = make_array_type(Tensor, FLOAT32)
  F64 = make_array_type(Tensor, FLOAT64)
  BF16 = make_array_type(Tensor, BFLOAT16)

  C64 = make_array_type(Tensor, COMPLEX64)
  C128 = make_array_type(Tensor, COMPLEX128)

  Int = make_array_type(Tensor, INT)
  UInt = make_array_type(Tensor, UINT)
  Integer = make_array_type(Tensor, INTEGER)
  Float = make_array_type(Tensor, FLOAT)
  Real = make_array_type(Tensor, REAL)
  Complex = make_array_type(Tensor, COMPLEX)
  Inexact = make_array_type(Tensor, INEXACT)
  Num = make_array_type(Tensor, NUM)
  Shaped = make_array_type(Tensor, SHAPED)

# ---------------------------------------------------------------------------
# Like types (scalar | tensor | nested sequences — for input validation)
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

type BoolLk = ArrayLike[BoolLike, Tensor | np.ndarray]

type I8Like = ArrayLike[Int8Like, Tensor | np.ndarray]
type I16Like = ArrayLike[Int16Like, Tensor | np.ndarray]
type I32Like = ArrayLike[Int32Like, Tensor | np.ndarray]
type I64Like = ArrayLike[Int64Like, Tensor | np.ndarray]

type U8Like = ArrayLike[UInt8Like, Tensor | np.ndarray]
type U16Like = ArrayLike[UInt16Like, Tensor | np.ndarray]
type U32Like = ArrayLike[UInt32Like, Tensor | np.ndarray]
type U64Like = ArrayLike[UInt64Like, Tensor | np.ndarray]

type F16Like = ArrayLike[Float16Like, Tensor | np.ndarray]
type F32Like = ArrayLike[Float32Like, Tensor | np.ndarray]
type F64Like = ArrayLike[Float64Like, Tensor | np.ndarray]

type C64Like = ArrayLike[Complex64Like, Tensor | np.ndarray]
type C128Like = ArrayLike[Complex128Like, Tensor | np.ndarray]

type IntLk = ArrayLike[IntLike, Tensor | np.ndarray]
type UIntLk = ArrayLike[UIntLike, Tensor | np.ndarray]
type IntegerLk = ArrayLike[IntegerLike, Tensor | np.ndarray]
type FloatLk = ArrayLike[FloatLike, Tensor | np.ndarray]
type RealLk = ArrayLike[RealLike, Tensor | np.ndarray]
type ComplexLk = ArrayLike[ComplexLike, Tensor | np.ndarray]
type InexactLk = ArrayLike[InexactLike, Tensor | np.ndarray]
type NumLk = ArrayLike[NumLike, Tensor | np.ndarray]
type ShapedLk = ArrayLike[ShapedLike, Tensor | np.ndarray]
