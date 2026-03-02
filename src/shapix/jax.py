"""JAX array type annotations with runtime shape and dtype checking.

Usage::

    from shapix import N, C, H, W
    from shapix.jax import Float32Array, BFloat16Array

    @beartype
    def forward(x: Float32Array[N, C, H, W]) -> BFloat16Array[N, C, H, W]:
        ...
"""

from __future__ import annotations

import typing as tp

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

if tp.TYPE_CHECKING:
    type BoolArray[*Dims] = JaxArray

    type Int8Array[*Dims] = JaxArray
    type Int16Array[*Dims] = JaxArray
    type Int32Array[*Dims] = JaxArray
    type Int64Array[*Dims] = JaxArray

    type UInt8Array[*Dims] = JaxArray
    type UInt16Array[*Dims] = JaxArray
    type UInt32Array[*Dims] = JaxArray
    type UInt64Array[*Dims] = JaxArray

    type Float16Array[*Dims] = JaxArray
    type Float32Array[*Dims] = JaxArray
    type Float64Array[*Dims] = JaxArray
    type BFloat16Array[*Dims] = JaxArray

    type Complex64Array[*Dims] = JaxArray
    type Complex128Array[*Dims] = JaxArray

    type IntArray[*Dims] = JaxArray
    type UIntArray[*Dims] = JaxArray
    type IntegerArray[*Dims] = JaxArray
    type FloatArray[*Dims] = JaxArray
    type RealArray[*Dims] = JaxArray
    type ComplexArray[*Dims] = JaxArray
    type InexactArray[*Dims] = JaxArray
    type NumArray[*Dims] = JaxArray
    type ShapedArray[*Dims] = JaxArray

else:
    BoolArray = make_array_type(JaxArray, BOOL)

    Int8Array = make_array_type(JaxArray, INT8)
    Int16Array = make_array_type(JaxArray, INT16)
    Int32Array = make_array_type(JaxArray, INT32)
    Int64Array = make_array_type(JaxArray, INT64)

    UInt8Array = make_array_type(JaxArray, UINT8)
    UInt16Array = make_array_type(JaxArray, UINT16)
    UInt32Array = make_array_type(JaxArray, UINT32)
    UInt64Array = make_array_type(JaxArray, UINT64)

    Float16Array = make_array_type(JaxArray, FLOAT16)
    Float32Array = make_array_type(JaxArray, FLOAT32)
    Float64Array = make_array_type(JaxArray, FLOAT64)
    BFloat16Array = make_array_type(JaxArray, BFLOAT16)

    Complex64Array = make_array_type(JaxArray, COMPLEX64)
    Complex128Array = make_array_type(JaxArray, COMPLEX128)

    IntArray = make_array_type(JaxArray, INT)
    UIntArray = make_array_type(JaxArray, UINT)
    IntegerArray = make_array_type(JaxArray, INTEGER)
    FloatArray = make_array_type(JaxArray, FLOAT)
    RealArray = make_array_type(JaxArray, REAL)
    ComplexArray = make_array_type(JaxArray, COMPLEX)
    InexactArray = make_array_type(JaxArray, INEXACT)
    NumArray = make_array_type(JaxArray, NUM)
    ShapedArray = make_array_type(JaxArray, SHAPED)
