"""NumPy array type annotations with runtime shape and dtype checking.

Scalar "Like" types (e.g. ``Int8Like``, ``Float32Like``) use beartype's
``Annotated[T, Is[...]]`` to validate value ranges at runtime.

Array types (e.g. ``Float32Array``, ``IntArray``) are subscriptable with
dimension symbols::

    from shapix import N, C, H, W
    from shapix.numpy import Float32Array

    @beartype
    def conv(x: Float32Array[N, C, H, W]) -> Float32Array[N, C, H, W]:
        ...
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
type Int64Like = A[int, _ge(_INT64_MIN) & _le(_INT64_MAX)] | A[np.integer[tp.Any], _le(_INT64_MAX)]  # type: ignore[type-arg]

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
# ArrayLike recursive types (scalar-or-nested-sequence, for input validation)
# ---------------------------------------------------------------------------

type _ArrayLike1D[_Scalar, _Array] = Sequence[_Scalar] | _Array
type _ArrayLike2D[_Scalar, _Array] = Sequence[_ArrayLike1D[_Scalar, _Array]] | _Array
type _ArrayLike3D[_Scalar, _Array] = Sequence[_ArrayLike2D[_Scalar, _Array]] | _Array
type _ArrayLike4D[_Scalar, _Array] = Sequence[_ArrayLike3D[_Scalar, _Array]] | _Array
type _ArrayLike5D[_Scalar, _Array] = Sequence[_ArrayLike4D[_Scalar, _Array]] | _Array
type _ArrayLike1DLess[_Scalar, _Array] = _Scalar | _ArrayLike1D[_Scalar, _Array]
type _ArrayLike2DLess[_Scalar, _Array] = _ArrayLike1DLess[_Scalar, _Array] | _ArrayLike2D[_Scalar, _Array]
type _ArrayLike3DLess[_Scalar, _Array] = _ArrayLike2DLess[_Scalar, _Array] | _ArrayLike3D[_Scalar, _Array]
type _ArrayLike4DLess[_Scalar, _Array] = _ArrayLike3DLess[_Scalar, _Array] | _ArrayLike4D[_Scalar, _Array]
type _ArrayLike5DLess[_Scalar, _Array] = _ArrayLike4DLess[_Scalar, _Array] | _ArrayLike5D[_Scalar, _Array]

# ---------------------------------------------------------------------------
# Array types
# ---------------------------------------------------------------------------

if tp.TYPE_CHECKING:
    from numpy.typing import NDArray

    type BoolArray[*Dims] = NDArray[np.bool_]

    type Int8Array[*Dims] = NDArray[np.int8]
    type Int16Array[*Dims] = NDArray[np.int16]
    type Int32Array[*Dims] = NDArray[np.int32]
    type Int64Array[*Dims] = NDArray[np.int64]

    type UInt8Array[*Dims] = NDArray[np.uint8]
    type UInt16Array[*Dims] = NDArray[np.uint16]
    type UInt32Array[*Dims] = NDArray[np.uint32]
    type UInt64Array[*Dims] = NDArray[np.uint64]

    type Float16Array[*Dims] = NDArray[np.float16]
    type Float32Array[*Dims] = NDArray[np.float32]
    type Float64Array[*Dims] = NDArray[np.float64]

    type Complex64Array[*Dims] = NDArray[np.complex64]
    type Complex128Array[*Dims] = NDArray[np.complex128]

    type IntArray[*Dims] = NDArray[np.signedinteger[tp.Any]]
    type UIntArray[*Dims] = NDArray[np.unsignedinteger[tp.Any]]
    type IntegerArray[*Dims] = NDArray[np.integer[tp.Any]]
    type FloatArray[*Dims] = NDArray[np.floating[tp.Any]]
    type RealArray[*Dims] = NDArray[np.integer[tp.Any] | np.floating[tp.Any]]
    type ComplexArray[*Dims] = NDArray[np.complexfloating[tp.Any, tp.Any]]
    type InexactArray[*Dims] = NDArray[np.inexact[tp.Any]]
    type NumArray[*Dims] = NDArray[np.number[tp.Any]]
    type ShapedArray[*Dims] = NDArray[np.bool_ | np.number[tp.Any]]

else:
    BoolArray = make_array_type(np.ndarray, BOOL)

    Int8Array = make_array_type(np.ndarray, INT8)
    Int16Array = make_array_type(np.ndarray, INT16)
    Int32Array = make_array_type(np.ndarray, INT32)
    Int64Array = make_array_type(np.ndarray, INT64)

    UInt8Array = make_array_type(np.ndarray, UINT8)
    UInt16Array = make_array_type(np.ndarray, UINT16)
    UInt32Array = make_array_type(np.ndarray, UINT32)
    UInt64Array = make_array_type(np.ndarray, UINT64)

    Float16Array = make_array_type(np.ndarray, FLOAT16)
    Float32Array = make_array_type(np.ndarray, FLOAT32)
    Float64Array = make_array_type(np.ndarray, FLOAT64)

    Complex64Array = make_array_type(np.ndarray, COMPLEX64)
    Complex128Array = make_array_type(np.ndarray, COMPLEX128)

    IntArray = make_array_type(np.ndarray, INT)
    UIntArray = make_array_type(np.ndarray, UINT)
    IntegerArray = make_array_type(np.ndarray, INTEGER)
    FloatArray = make_array_type(np.ndarray, FLOAT)
    RealArray = make_array_type(np.ndarray, REAL)
    ComplexArray = make_array_type(np.ndarray, COMPLEX)
    InexactArray = make_array_type(np.ndarray, INEXACT)
    NumArray = make_array_type(np.ndarray, NUM)
    ShapedArray = make_array_type(np.ndarray, SHAPED)
