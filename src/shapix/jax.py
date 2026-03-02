import typing as tp

from jax import Array as JaxArray
from jaxtyping import (
  BFloat16,
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

from ..jaxtyping import ArrayLikeType, ArrayType
from ..numpy import BoolArray as NumpyBoolArray
from ..numpy import (
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
from ..numpy import Complex64Array as NumpyComplex64Array
from ..numpy import Complex128Array as NumpyComplex128Array
from ..numpy import ComplexArrayLike as NumpyComplexArrayLike
from ..numpy import Float16Array as NumpyFloat16Array
from ..numpy import Float32Array as NumpyFloat32Array
from ..numpy import Float64Array as NumpyFloat64Array
from ..numpy import FloatArrayLike as NumpyFloatArrayLike
from ..numpy import InexactArrayLike as NumpyInexactArrayLike
from ..numpy import Int8Array as NumpyInt8Array
from ..numpy import Int16Array as NumpyInt16Array
from ..numpy import Int32Array as NumpyInt32Array
from ..numpy import Int64Array as NumpyInt64Array
from ..numpy import IntArrayLike as NumpyIntArrayLike
from ..numpy import IntegerArrayLike as NumpyIntegerArrayLike
from ..numpy import NumArrayLike as NumpyNumArrayLike
from ..numpy import RealArrayLike as NumpyRealArrayLike
from ..numpy import ShapedArrayLike as NumpyShapedArrayLike
from ..numpy import UInt8Array as NumpyUInt8Array
from ..numpy import UInt16Array as NumpyUInt16Array
from ..numpy import UInt32Array as NumpyUInt32Array
from ..numpy import UInt64Array as NumpyUInt64Array
from ..numpy import UIntArrayLike as NumpyUIntArrayLike
from ..numpy.typing import _ArrayLike5DLess

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

  type BoolArrayLike[*Dims] = _ArrayLike5DLess[
    BoolLike, NumpyBoolArray[*Dims] | BoolArray[*Dims]
  ]

  type Int8ArrayLike[*Dims] = _ArrayLike5DLess[
    Int8Like, NumpyInt8Array[*Dims] | Int8Array[*Dims]
  ]
  type Int16ArrayLike[*Dims] = _ArrayLike5DLess[
    Int16Like, NumpyInt16Array[*Dims] | Int16Array[*Dims]
  ]
  type Int32ArrayLike[*Dims] = _ArrayLike5DLess[
    Int32Like, NumpyInt32Array[*Dims] | Int32Array[*Dims]
  ]
  type Int64ArrayLike[*Dims] = _ArrayLike5DLess[
    Int64Like, NumpyInt64Array[*Dims] | Int64Array[*Dims]
  ]

  type UInt8ArrayLike[*Dims] = _ArrayLike5DLess[
    UInt8Like, NumpyUInt8Array[*Dims] | UInt8Array[*Dims]
  ]
  type UInt16ArrayLike[*Dims] = _ArrayLike5DLess[
    UInt16Like, NumpyUInt16Array[*Dims] | UInt16Array[*Dims]
  ]
  type UInt32ArrayLike[*Dims] = _ArrayLike5DLess[
    UInt32Like, NumpyUInt32Array[*Dims] | UInt32Array[*Dims]
  ]
  type UInt64ArrayLike[*Dims] = _ArrayLike5DLess[
    UInt64Like, NumpyUInt64Array[*Dims] | UInt64Array[*Dims]
  ]

  type Float16ArrayLike[*Dims] = _ArrayLike5DLess[
    Float16Like, NumpyFloat16Array[*Dims] | Float16Array[*Dims]
  ]
  type Float32ArrayLike[*Dims] = _ArrayLike5DLess[
    Float32Like, NumpyFloat32Array[*Dims] | Float32Array[*Dims]
  ]
  type Float64ArrayLike[*Dims] = _ArrayLike5DLess[
    Float64Like, NumpyFloat64Array[*Dims] | Float64Array[*Dims]
  ]
  type Complex64ArrayLike[*Dims] = _ArrayLike5DLess[
    Complex64Like, NumpyComplex64Array[*Dims] | Complex64Array[*Dims]
  ]
  type Complex128ArrayLike[*Dims] = _ArrayLike5DLess[
    Complex128Like, NumpyComplex128Array[*Dims] | Complex128Array[*Dims]
  ]

  type IntArrayLike[*Dims] = _ArrayLike5DLess[
    IntLike, NumpyIntArrayLike | IntArray[*Dims]
  ]
  type UIntArrayLike[*Dims] = _ArrayLike5DLess[
    UIntLike, NumpyUIntArrayLike | UIntArray[*Dims]
  ]
  type IntegerArrayLike[*Dims] = _ArrayLike5DLess[
    IntegerLike, NumpyIntegerArrayLike | IntegerArray[*Dims]
  ]
  type FloatArrayLike[*Dims] = _ArrayLike5DLess[
    FloatLike, NumpyFloatArrayLike | FloatArray[*Dims]
  ]
  type RealArrayLike[*Dims] = _ArrayLike5DLess[
    RealLike, NumpyRealArrayLike | RealArray[*Dims]
  ]
  type ComplexArrayLike[*Dims] = _ArrayLike5DLess[
    ComplexLike, NumpyComplexArrayLike | ComplexArray[*Dims]
  ]
  type InexactArrayLike[*Dims] = _ArrayLike5DLess[
    InexactLike, NumpyInexactArrayLike | InexactArray[*Dims]
  ]
  type NumArrayLike[*Dims] = _ArrayLike5DLess[
    NumLike, NumpyNumArrayLike | NumArray[*Dims]
  ]
  type ShapedArrayLike[*Dims] = _ArrayLike5DLess[
    ShapedLike, NumpyShapedArrayLike | ShapedArray[*Dims]
  ]

else:
  Array = ArrayType[JaxArray]
  ArrayLike = ArrayLikeType[JaxArray]

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

  BFloat16Array = Array[BFloat16]

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
