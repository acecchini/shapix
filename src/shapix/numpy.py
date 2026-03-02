import typing as tp
from collections.abc import Sequence
from typing import Annotated as A

import numpy as np
from beartype.vale import Is
from jaxtyping import (
  Bool,
  Complex,
  Complex64,
  Complex128,
  Float,
  Float16,
  Float32,
  Float64,
  Inexact,
  Int,
  Int8,
  Int16,
  Int32,
  Int64,
  Integer,
  Num,
  Real,
  Shaped,
  UInt,
  UInt8,
  UInt16,
  UInt32,
  UInt64,
)
from numpy import (
  bool_,
  complex64,
  complex128,
  complexfloating,
  dtype,
  float16,
  float32,
  float64,
  floating,
  inexact,
  int8,
  int16,
  int32,
  int64,
  integer,
  number,
  signedinteger,
  str_,
  uint8,
  uint16,
  uint32,
  uint64,
  unsignedinteger,
)

# from numpy._typing._array_like import _DualArrayLike
from numpy._typing._array_like import _SupportsArray
from numpy._typing._nested_sequence import _NestedSequence
from numpy.typing import NDArray

from ..jaxtyping import ArrayLikeType, ArrayType, String
from ..numbers import (
  FLOAT16_MAX,
  FLOAT16_MIN,
  FLOAT32_MAX,
  FLOAT32_MIN,
  FLOAT64_MAX,
  FLOAT64_MIN,
  INT8_MAX,
  INT8_MIN,
  INT16_MAX,
  INT16_MIN,
  INT32_MAX,
  INT32_MIN,
  INT64_MAX,
  INT64_MIN,
  UINT8_MAX,
  UINT16_MAX,
  UINT32_MAX,
  UINT64_MAX,
)
from ..operator import ge, le

type _DualArrayLike[DType: dtype[tp.Any], T] = (
  _SupportsArray[DType] | T | _NestedSequence[_SupportsArray[DType] | T]
)

type StringLike = str | np.str_
type BoolLike = bool | np.bool_

type Int8Like = A[int | np.integer[tp.Any], Is[ge(INT8_MIN)] & Is[le(INT8_MAX)]]
type Int16Like = A[int | np.integer[tp.Any], Is[ge(INT16_MIN)] & Is[le(INT16_MAX)]]
type Int32Like = A[int | np.integer[tp.Any], Is[ge(INT32_MIN)] & Is[le(INT32_MAX)]]
# any np.integer is >= INT64_MIN
type Int64Like = (
  A[int, Is[ge(INT64_MIN)] & Is[le(INT64_MAX)]]
  | A[np.integer[tp.Any], Is[le(INT64_MAX)]]
)

# any np.integer is <= UINT64_MAX
type UInt8Like = A[int | np.integer[tp.Any], Is[ge(0)] & Is[le(UINT8_MAX)]]
type UInt16Like = A[int | np.integer[tp.Any], Is[ge(0)] & Is[le(UINT16_MAX)]]
type UInt32Like = A[int | np.integer[tp.Any], Is[ge(0)] & Is[le(UINT32_MAX)]]
type UInt64Like = (
  A[int, Is[ge(0)] & Is[le(UINT64_MAX)]] | A[np.integer[tp.Any], Is[ge(0)]]
)

type Float16Like = A[
  float | np.floating[tp.Any], Is[ge(FLOAT16_MIN)] & Is[le(FLOAT16_MAX)]
]
type Float32Like = A[
  float | np.floating[tp.Any], Is[ge(FLOAT32_MIN)] & Is[le(FLOAT32_MAX)]
]
type Float64Like = A[
  float | np.floating[tp.Any], Is[ge(FLOAT64_MIN)] & Is[le(FLOAT64_MAX)]
]
# Float128 exists but we limit to 64 for now

# TODO: Add a max_radius as a vale constraint?
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

type Fraction = A[FloatLike, Is[ge(0.0)] & Is[le(1.0)]]
type Seed = np.uint64
type SeedLike = UInt64Like

type _ArrayLike1D[_Scalar, _Array] = Sequence[_Scalar] | _Array
type _ArrayLike2D[_Scalar, _Array] = Sequence[_ArrayLike1D[_Scalar, _Array]] | _Array
type _ArrayLike3D[_Scalar, _Array] = Sequence[_ArrayLike2D[_Scalar, _Array]] | _Array
type _ArrayLike4D[_Scalar, _Array] = Sequence[_ArrayLike3D[_Scalar, _Array]] | _Array
type _ArrayLike5D[_Scalar, _Array] = Sequence[_ArrayLike4D[_Scalar, _Array]] | _Array
type _ArrayLike1DLess[_Scalar, _Array] = _Scalar | _ArrayLike1D[_Scalar, _Array]
type _ArrayLike2DLess[_Scalar, _Array] = (
  _ArrayLike1DLess[_Scalar, _Array] | _ArrayLike2D[_Scalar, _Array]
)
type _ArrayLike3DLess[_Scalar, _Array] = (
  _ArrayLike2DLess[_Scalar, _Array] | _ArrayLike3D[_Scalar, _Array]
)
type _ArrayLike4DLess[_Scalar, _Array] = (
  _ArrayLike3DLess[_Scalar, _Array] | _ArrayLike4D[_Scalar, _Array]
)
type _ArrayLike5DLess[_Scalar, _Array] = (
  _ArrayLike4DLess[_Scalar, _Array] | _ArrayLike5D[_Scalar, _Array]
)

if tp.TYPE_CHECKING:
  type StringArray[*Dims] = NDArray[np.str_]
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
  type NumArray[*Dims] = NDArray[number[tp.Any]]
  type ShapedArray[*Dims] = NDArray[bool_ | number[tp.Any]]

  type StringArrayLike[*Dims] = _DualArrayLike[dtype[str_], str]
  type BoolArrayLike[*Dims] = _DualArrayLike[dtype[bool_], bool]

  type Int8ArrayLike[*Dims] = _DualArrayLike[dtype[int8], int]
  type Int16ArrayLike[*Dims] = _DualArrayLike[dtype[int16], int]
  type Int32ArrayLike[*Dims] = _DualArrayLike[dtype[int32], int]
  type Int64ArrayLike[*Dims] = _DualArrayLike[dtype[int64], int]

  type UInt8ArrayLike[*Dims] = _DualArrayLike[dtype[uint8], int]
  type UInt16ArrayLike[*Dims] = _DualArrayLike[dtype[uint16], int]
  type UInt32ArrayLike[*Dims] = _DualArrayLike[dtype[uint32], int]
  type UInt64ArrayLike[*Dims] = _DualArrayLike[dtype[uint64], int]

  type Float16ArrayLike[*Dims] = _DualArrayLike[dtype[float16], float]
  type Float32ArrayLike[*Dims] = _DualArrayLike[dtype[float32], float]
  type Float64ArrayLike[*Dims] = _DualArrayLike[dtype[float64], float]

  type Complex64ArrayLike[*Dims] = _DualArrayLike[dtype[complex64], complex]
  type Complex128ArrayLike[*Dims] = _DualArrayLike[dtype[complex128], complex]

  type IntArrayLike[*Dims] = _DualArrayLike[dtype[signedinteger[tp.Any]], int]
  type UIntArrayLike[*Dims] = _DualArrayLike[dtype[unsignedinteger[tp.Any]], int]
  type IntegerArrayLike[*Dims] = _DualArrayLike[dtype[integer[tp.Any]], int]
  type FloatArrayLike[*Dims] = _DualArrayLike[dtype[floating[tp.Any]], float]
  type RealArrayLike[*Dims] = _DualArrayLike[
    dtype[floating[tp.Any] | integer[tp.Any]], int | float
  ]
  type ComplexArrayLike[*Dims] = _DualArrayLike[
    dtype[complexfloating[tp.Any, tp.Any]], complex
  ]
  type InexactArrayLike[*Dims] = _DualArrayLike[dtype[inexact[tp.Any]], float | complex]
  type NumArrayLike[*Dims] = _DualArrayLike[
    dtype[number[tp.Any]], int | float | complex
  ]
  type ShapedArrayLike[*Dims] = _DualArrayLike[
    dtype[number[tp.Any] | bool_], bool | int | float | complex
  ]

else:
  Array = ArrayType[np.ndarray]
  ArrayLike = ArrayLikeType[np.ndarray]

  StringArray = Array[String]
  BoolArray = Array[Bool]

  Int8Array = Array[Int8]
  Int16Array = Array[Int16]
  Int32Array = Array[Int32]
  Int64Array = Array[Int64]

  UInt8Array = Array[UInt8]
  UInt16Array = Array[UInt16]
  UInt32Array = Array[UInt32]
  UInt64Array = Array[UInt64]

  Float16Array = Array[Float16]
  Float32Array = Array[Float32]
  Float64Array = Array[Float64]

  Complex64Array = Array[Complex64]
  Complex128Array = Array[Complex128]

  IntArray = Array[Int]
  UIntArray = Array[UInt]
  IntegerArray = Array[Integer]
  FloatArray = Array[Float]
  RealArray = Array[Real]
  ComplexArray = Array[Complex]
  InexactArray = Array[Inexact]
  NumArray = Array[Num]
  ShapedArray = Array[Shaped]

  BoolArrayLike = ArrayLike[Bool]
  StringArrayLike = ArrayLike[String]

  Int8ArrayLike = ArrayLike[Int8]
  Int16ArrayLike = ArrayLike[Int16]
  Int32ArrayLike = ArrayLike[Int32]
  Int64ArrayLike = ArrayLike[Int64]

  UInt8ArrayLike = ArrayLike[UInt8]
  UInt16ArrayLike = ArrayLike[UInt16]
  UInt32ArrayLike = ArrayLike[UInt32]
  UInt64ArrayLike = ArrayLike[UInt64]

  Float16ArrayLike = ArrayLike[Float16]
  Float32ArrayLike = ArrayLike[Float32]
  Float64ArrayLike = ArrayLike[Float64]

  Complex64ArrayLike = ArrayLike[Complex64]
  Complex128ArrayLike = ArrayLike[Complex128]

  IntArrayLike = ArrayLike[Int]
  UIntArrayLike = ArrayLike[UInt]
  IntegerArrayLike = ArrayLike[Integer]
  FloatArrayLike = ArrayLike[Float]
  RealArrayLike = ArrayLike[Real]
  ComplexArrayLike = ArrayLike[Complex]
  InexactArrayLike = ArrayLike[Inexact]
  NumArrayLike = ArrayLike[Num]
  ShapedArrayLike = ArrayLike[Shaped]
