# pyright: reportInvalidTypeForm=false
"""Verify F32[N, C] annotation pattern (pyright-specific).

This tests the full shapix annotation pattern including dimension-subscripted
array types. Only pyright/Pylance supports this through TYPE_CHECKING stubs.

Tested with: pyright only
"""

import numpy as np
from beartype import beartype

from shapix import B, C, Dimension, H, N, Scalar, Value, W, __, check, check_context
from shapix.numpy import (
  Bool,
  C64,
  C128,
  F16,
  F32,
  F64,
  I8,
  I16,
  I32,
  I64,
  Int,
  Integer,
  Num,
  Shaped,
  U8,
  U16,
  U32,
  U64,
)

# ---------------------------------------------------------------------------
# Basic typed functions — various dtypes
# ---------------------------------------------------------------------------


@check
@beartype
def add_f32(x: F32[N, C], y: F32[N, C]) -> F32[N, C]:
  return x + y


result = add_f32(np.ones((4, 3), dtype=np.float32), np.ones((4, 3), dtype=np.float32))
assert result.shape == (4, 3)


@beartype
def cast_i64_to_f32(x: I64[N]) -> F32[N]:
  return x.astype(np.float32)


# ---------------------------------------------------------------------------
# All concrete dtype variants in annotations
# ---------------------------------------------------------------------------


@beartype
def bool_fn(x: Bool[N]) -> Bool[N]:
  return x


@beartype
def i8_fn(x: I8[N]) -> I8[N]:
  return x


@beartype
def i16_fn(x: I16[N]) -> I16[N]:
  return x


@beartype
def i32_fn(x: I32[N]) -> I32[N]:
  return x


@beartype
def u8_fn(x: U8[N]) -> U8[N]:
  return x


@beartype
def u16_fn(x: U16[N]) -> U16[N]:
  return x


@beartype
def u32_fn(x: U32[N]) -> U32[N]:
  return x


@beartype
def u64_fn(x: U64[N]) -> U64[N]:
  return x


@beartype
def f16_fn(x: F16[N]) -> F16[N]:
  return x


@beartype
def f64_fn(x: F64[N]) -> F64[N]:
  return x


@beartype
def c64_fn(x: C64[N]) -> C64[N]:
  return x


@beartype
def c128_fn(x: C128[N]) -> C128[N]:
  return x


# ---------------------------------------------------------------------------
# Category dtype variants
# ---------------------------------------------------------------------------


@beartype
def int_fn(x: Int[N, C]) -> Int[N, C]:
  return x


@beartype
def integer_fn(x: Integer[N]) -> Integer[N]:
  return x


@beartype
def num_fn(x: Num[N]) -> Num[N]:
  return x


@beartype
def shaped_fn(x: Shaped[N, C]) -> Shaped[N, C]:
  return x


# ---------------------------------------------------------------------------
# Multiple dimensions: 1D, 2D, 3D, 4D
# ---------------------------------------------------------------------------


@beartype
def fn_1d(x: F32[N]) -> F32[N]:
  return x


@beartype
def fn_2d(x: F32[N, C]) -> F32[N, C]:
  return x


@beartype
def fn_3d(x: F32[N, H, W]) -> F32[N, H, W]:
  return x


@beartype
def fn_4d(x: F32[N, C, H, W]) -> F32[N, C, H, W]:
  return x


# ---------------------------------------------------------------------------
# Scalar annotation
# ---------------------------------------------------------------------------


@beartype
def scalar_fn(x: F32[Scalar]) -> F32[Scalar]:
  return x


# ---------------------------------------------------------------------------
# Dimension arithmetic in subscripts
# ---------------------------------------------------------------------------


@beartype
def pad(x: F32[N]) -> F32[N + 2]:
  return np.pad(x, 1).astype(np.float32)


@beartype
def double_channels(x: F32[N, C]) -> F32[N, C * 2]:
  return np.concatenate([x, x], axis=1).astype(np.float32)


FromSize = Value("size")
FromSelf = Value("self.offset + size")


@beartype
def from_value(size: int) -> F32[FromSize]:
  return np.ones(size, dtype=np.float32)


class SomeClass:
  offset = 3

  @beartype
  def from_self(self, size: int) -> F32[FromSelf]:
    return np.ones(self.offset + size, dtype=np.float32)


# ---------------------------------------------------------------------------
# Variadic and broadcast dims in subscripts
# ---------------------------------------------------------------------------


@beartype
def batched(x: F32[B, N, C]) -> F32[B, N, C]:
  return x


@beartype
def with_anon(x: F32[__, C]) -> F32[__, C]:
  return x


# ---------------------------------------------------------------------------
# Custom dimensions in annotations
# ---------------------------------------------------------------------------

Vocab = Dimension("Vocab")
Embed = Dimension("Embed")


@beartype
def embed_lookup(x: I64[N], table: F32[Vocab, Embed]) -> F32[N, Embed]:
  return table[x]


# ---------------------------------------------------------------------------
# check_context — verifies context manager typing
# ---------------------------------------------------------------------------


def test_check_context_typing() -> None:
  with check_context():
    x = np.ones((3, 4), dtype=np.float32)
    assert x.shape == (3, 4)


# ---------------------------------------------------------------------------
# Mixed @check + @beartype
# ---------------------------------------------------------------------------


@check
@beartype
def checked_fn(x: F32[N, C], y: F32[N, C]) -> F32[N, C]:
  return x + y
