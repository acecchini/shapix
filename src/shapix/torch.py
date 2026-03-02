"""PyTorch tensor type annotations with runtime shape and dtype checking.

Usage::

    from shapix import N, C, H, W
    from shapix.torch import Float32Tensor

    @beartype
    def forward(x: Float32Tensor[N, C, H, W]) -> Float32Tensor[N, C, H, W]:
        ...
"""

from __future__ import annotations

import typing as tp

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

if tp.TYPE_CHECKING:
    type BoolTensor[*Dims] = Tensor

    type Int8Tensor[*Dims] = Tensor
    type Int16Tensor[*Dims] = Tensor
    type Int32Tensor[*Dims] = Tensor
    type Int64Tensor[*Dims] = Tensor

    type UInt8Tensor[*Dims] = Tensor
    type UInt16Tensor[*Dims] = Tensor
    type UInt32Tensor[*Dims] = Tensor
    type UInt64Tensor[*Dims] = Tensor

    type Float16Tensor[*Dims] = Tensor
    type Float32Tensor[*Dims] = Tensor
    type Float64Tensor[*Dims] = Tensor
    type BFloat16Tensor[*Dims] = Tensor

    type Complex64Tensor[*Dims] = Tensor
    type Complex128Tensor[*Dims] = Tensor

    type IntTensor[*Dims] = Tensor
    type UIntTensor[*Dims] = Tensor
    type IntegerTensor[*Dims] = Tensor
    type FloatTensor[*Dims] = Tensor
    type RealTensor[*Dims] = Tensor
    type ComplexTensor[*Dims] = Tensor
    type InexactTensor[*Dims] = Tensor
    type NumTensor[*Dims] = Tensor
    type ShapedTensor[*Dims] = Tensor

else:
    BoolTensor = make_array_type(Tensor, BOOL)

    Int8Tensor = make_array_type(Tensor, INT8)
    Int16Tensor = make_array_type(Tensor, INT16)
    Int32Tensor = make_array_type(Tensor, INT32)
    Int64Tensor = make_array_type(Tensor, INT64)

    UInt8Tensor = make_array_type(Tensor, UINT8)
    UInt16Tensor = make_array_type(Tensor, UINT16)
    UInt32Tensor = make_array_type(Tensor, UINT32)
    UInt64Tensor = make_array_type(Tensor, UINT64)

    Float16Tensor = make_array_type(Tensor, FLOAT16)
    Float32Tensor = make_array_type(Tensor, FLOAT32)
    Float64Tensor = make_array_type(Tensor, FLOAT64)
    BFloat16Tensor = make_array_type(Tensor, BFLOAT16)

    Complex64Tensor = make_array_type(Tensor, COMPLEX64)
    Complex128Tensor = make_array_type(Tensor, COMPLEX128)

    IntTensor = make_array_type(Tensor, INT)
    UIntTensor = make_array_type(Tensor, UINT)
    IntegerTensor = make_array_type(Tensor, INTEGER)
    FloatTensor = make_array_type(Tensor, FLOAT)
    RealTensor = make_array_type(Tensor, REAL)
    ComplexTensor = make_array_type(Tensor, COMPLEX)
    InexactTensor = make_array_type(Tensor, INEXACT)
    NumTensor = make_array_type(Tensor, NUM)
    ShapedTensor = make_array_type(Tensor, SHAPED)
