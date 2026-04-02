"""Verify scalar Like types and ArrayLike type aliases work with type checkers.

Tested with: pyright, mypy, ty
"""

from shapix.numpy import (
  ArrayLike,
  # Scalar Like types
  BoolScalarLike,
  C64ScalarLike,
  C128ScalarLike,
  C256ScalarLike,
  ComplexScalarLike,
  F16ScalarLike,
  F32ScalarLike,
  F64ScalarLike,
  F128ScalarLike,
  FloatScalarLike,
  I8ScalarLike,
  I16ScalarLike,
  I32ScalarLike,
  I64ScalarLike,
  InexactScalarLike,
  IntegerScalarLike,
  IntScalarLike,
  NumScalarLike,
  RealScalarLike,
  ShapedScalarLike,
  StringLike,
  U8ScalarLike,
  U16ScalarLike,
  U32ScalarLike,
  U64ScalarLike,
  UIntScalarLike,
  # Array Like types
  BoolLike,
  C64Like,
  C128Like,
  ComplexLike,
  F16Like,
  F32Like,
  F64Like,
  FloatLike,
  I8Like,
  I16Like,
  I32Like,
  I64Like,
  InexactLike,
  IntegerLike,
  IntLike,
  NumLike,
  RealLike,
  ShapedLike,
  U8Like,
  U16Like,
  U32Like,
  U64Like,
  UIntLike,
)

# ---------------------------------------------------------------------------
# Scalar Like types — just verify they're importable and usable as annotations
# ---------------------------------------------------------------------------

_bool: BoolScalarLike = True
_str: StringLike = "hello"

# ---------------------------------------------------------------------------
# Array Like types — nested sequences should be valid
# ---------------------------------------------------------------------------

_arr_like = ArrayLike  # type: ignore[type-arg]  # ArrayLike is a generic type alias

# Float types
_f16: F16ScalarLike = 1.0
_f32: F32ScalarLike = 1.0
_f64: F64ScalarLike = 1.0
_f128: F128ScalarLike = 1.0
_fl: FloatScalarLike = 1.0

# Int types
_i8: I8ScalarLike = 1
_i16: I16ScalarLike = 1
_i32: I32ScalarLike = 1
_i64: I64ScalarLike = 1
_il: IntScalarLike = 1

# Uint types
_u8: U8ScalarLike = 1
_u16: U16ScalarLike = 1
_u32: U32ScalarLike = 1
_u64: U64ScalarLike = 1
_ul: UIntScalarLike = 1

# Category types
_integer: IntegerScalarLike = 1
_real: RealScalarLike = 1.0
_complex: ComplexScalarLike = 1 + 0j
_inexact: InexactScalarLike = 1.0
_num: NumScalarLike = 1
_shaped: ShapedScalarLike = True

# Complex types
_c64: C64ScalarLike = 1 + 0j
_c128: C128ScalarLike = 1 + 0j
_c256: C256ScalarLike = 1 + 0j

# ---------------------------------------------------------------------------
# ArrayLike types — verify import
# ---------------------------------------------------------------------------

array_like_aliases = [
  BoolLike,
  IntLike,
  UIntLike,
  IntegerLike,
  FloatLike,
  RealLike,
  ComplexLike,
  InexactLike,
  NumLike,
  ShapedLike,
  I8Like,
  I16Like,
  I32Like,
  I64Like,
  U8Like,
  U16Like,
  U32Like,
  U64Like,
  F16Like,
  F32Like,
  F64Like,
  C64Like,
  C128Like,
]
