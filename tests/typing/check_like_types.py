"""Verify scalar Like types and ArrayLike type aliases work with type checkers.

Tested with: pyright, mypy, ty
"""

import numpy as np

from shapix.numpy import (
  ArrayLike,
  BoolLike,
  BoolLk,
  C128Like,
  C64Like,
  Complex64Like,
  Complex128Like,
  ComplexLike,
  ComplexLk,
  F16Like,
  F32Like,
  F64Like,
  Float16Like,
  Float32Like,
  Float64Like,
  FloatLike,
  FloatLk,
  Fraction,
  I16Like,
  I32Like,
  I64Like,
  I8Like,
  InexactLike,
  InexactLk,
  Int16Like,
  Int32Like,
  Int64Like,
  Int8Like,
  IntegerLike,
  IntegerLk,
  IntLike,
  IntLk,
  NumLike,
  NumLk,
  RealLike,
  RealLk,
  Seed,
  SeedLike,
  ShapedLike,
  ShapedLk,
  StringLike,
  U16Like,
  U32Like,
  U64Like,
  U8Like,
  UInt16Like,
  UInt32Like,
  UInt64Like,
  UInt8Like,
  UIntLike,
  UIntLk,
)

# ---------------------------------------------------------------------------
# Scalar Like types — just verify they're importable and usable as annotations
# ---------------------------------------------------------------------------

_bool: BoolLike = True
_str: StringLike = "hello"

# ---------------------------------------------------------------------------
# Array Like types — nested sequences should be valid
# ---------------------------------------------------------------------------

_arr_like = ArrayLike  # ArrayLike is a generic type alias

# Float types
_f16: Float16Like = 1.0
_f32: Float32Like = 1.0
_f64: Float64Like = 1.0
_fl: FloatLike = 1.0

# Int types
_i8: Int8Like = 1
_i16: Int16Like = 1
_i32: Int32Like = 1
_i64: Int64Like = 1
_il: IntLike = 1

# Uint types
_u8: UInt8Like = 1
_u16: UInt16Like = 1
_u32: UInt32Like = 1
_u64: UInt64Like = 1
_ul: UIntLike = 1

# Category types
_integer: IntegerLike = 1
_real: RealLike = 1.0
_complex: ComplexLike = 1 + 0j
_inexact: InexactLike = 1.0
_num: NumLike = 1
_shaped: ShapedLike = True

# Complex types
_c64: Complex64Like = 1 + 0j
_c128: Complex128Like = 1 + 0j

# Special types
_frac: Fraction = 0.5
_seed: Seed = np.uint64(42)

# ---------------------------------------------------------------------------
# ArrayLike Lk types — verify import
# ---------------------------------------------------------------------------

_blk: type = BoolLk  # type: ignore[assignment]
_ilk: type = IntLk  # type: ignore[assignment]
_ulk: type = UIntLk  # type: ignore[assignment]
_iglk: type = IntegerLk  # type: ignore[assignment]
_flk: type = FloatLk  # type: ignore[assignment]
_rlk: type = RealLk  # type: ignore[assignment]
_clk: type = ComplexLk  # type: ignore[assignment]
_inlk: type = InexactLk  # type: ignore[assignment]
_nlk: type = NumLk  # type: ignore[assignment]
_slk: type = ShapedLk  # type: ignore[assignment]

_i8lk: type = I8Like  # type: ignore[assignment]
_i16lk: type = I16Like  # type: ignore[assignment]
_i32lk: type = I32Like  # type: ignore[assignment]
_i64lk: type = I64Like  # type: ignore[assignment]
_u8lk: type = U8Like  # type: ignore[assignment]
_u16lk: type = U16Like  # type: ignore[assignment]
_u32lk: type = U32Like  # type: ignore[assignment]
_u64lk: type = U64Like  # type: ignore[assignment]
_f16lk: type = F16Like  # type: ignore[assignment]
_f32lk: type = F32Like  # type: ignore[assignment]
_f64lk: type = F64Like  # type: ignore[assignment]
_c64lk: type = C64Like  # type: ignore[assignment]
_c128lk: type = C128Like  # type: ignore[assignment]

_ = SeedLike
