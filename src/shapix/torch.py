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

from torch import Tensor

try:
  import numpy as _np  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ModuleNotFoundError as exc:
  msg = (
    "shapix.torch requires 'numpy' at runtime. "
    "Install it alongside shapix (e.g. `pip install shapix numpy torch`)."
  )
  raise ModuleNotFoundError(msg) from exc

__all__ = [
  # Array types
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
  "BF16",
  "C64",
  "C128",
  "Int",
  "UInt",
  "Integer",
  "Float",
  "Real",
  "Complex",
  "Inexact",
  "Num",
  "Shaped",
  # Like types
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
  BoolLike as BoolLikeNumpy,
  C64Like as C64LikeNumpy,
  C128Like as C128LikeNumpy,
  ComplexLike as ComplexLikeNumpy,
  F16Like as F16LikeNumpy,
  F32Like as F32LikeNumpy,
  F64Like as F64LikeNumpy,
  FloatLike as FloatLikeNumpy,
  InexactLike as InexactLikeNumpy,
  I8Like as I8LikeNumpy,
  I16Like as I16LikeNumpy,
  I32Like as I32LikeNumpy,
  I64Like as I64LikeNumpy,
  IntegerLike as IntegerLikeNumpy,
  IntLike as IntLikeNumpy,
  NumLike as NumLikeNumpy,
  RealLike as RealLikeNumpy,
  ShapedLike as ShapedLikeNumpy,
  U8Like as U8LikeNumpy,
  U16Like as U16LikeNumpy,
  U32Like as U32LikeNumpy,
  U64Like as U64LikeNumpy,
  UIntLike as UIntLikeNumpy,
)

if tp.TYPE_CHECKING:
  type BoolLike[*Dims] = Bool[*Dims] | BoolLikeNumpy[*Dims]

  type I8Like[*Dims] = I8[*Dims] | I8LikeNumpy[*Dims]
  type I16Like[*Dims] = I16[*Dims] | I16LikeNumpy[*Dims]
  type I32Like[*Dims] = I32[*Dims] | I32LikeNumpy[*Dims]
  type I64Like[*Dims] = I64[*Dims] | I64LikeNumpy[*Dims]

  type U8Like[*Dims] = U8[*Dims] | U8LikeNumpy[*Dims]
  type U16Like[*Dims] = U16[*Dims] | U16LikeNumpy[*Dims]
  type U32Like[*Dims] = U32[*Dims] | U32LikeNumpy[*Dims]
  type U64Like[*Dims] = U64[*Dims] | U64LikeNumpy[*Dims]

  type F16Like[*Dims] = F16[*Dims] | F16LikeNumpy[*Dims]
  type F32Like[*Dims] = F32[*Dims] | F32LikeNumpy[*Dims]
  type F64Like[*Dims] = F64[*Dims] | F64LikeNumpy[*Dims]

  type C64Like[*Dims] = C64[*Dims] | C64LikeNumpy[*Dims]
  type C128Like[*Dims] = C128[*Dims] | C128LikeNumpy[*Dims]

  type IntLike[*Dims] = Int[*Dims] | IntLikeNumpy[*Dims]
  type UIntLike[*Dims] = UInt[*Dims] | UIntLikeNumpy[*Dims]
  type IntegerLike[*Dims] = Integer[*Dims] | IntegerLikeNumpy[*Dims]
  type FloatLike[*Dims] = Float[*Dims] | FloatLikeNumpy[*Dims]
  type RealLike[*Dims] = Real[*Dims] | RealLikeNumpy[*Dims]
  type ComplexLike[*Dims] = Complex[*Dims] | ComplexLikeNumpy[*Dims]
  type InexactLike[*Dims] = Inexact[*Dims] | InexactLikeNumpy[*Dims]
  type NumLike[*Dims] = Num[*Dims] | NumLikeNumpy[*Dims]
  type ShapedLike[*Dims] = Shaped[*Dims] | ShapedLikeNumpy[*Dims]

else:
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
