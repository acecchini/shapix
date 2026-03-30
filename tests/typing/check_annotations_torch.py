# pyright: reportMissingImports=false, reportInvalidTypeForm=false
"""Verify PyTorch tensor type annotations type-check cleanly.

Tests the full shapix annotation pattern with PyTorch backend types.

Tested with: pyright, mypy, ty
"""

from typing import TYPE_CHECKING

from beartype import beartype

from shapix import B, C, Dimension, H, N, Scalar, W, __, check
from shapix.torch import (
  BF16,
  BF16Like,
  Bool,
  BoolLike,
  F16,
  F32,
  F32Like,
  F64,
  I32,
  I64,
  I64Like,
  Int,
  Num,
  Shaped,
  U8,
)

# ---------------------------------------------------------------------------
# Basic typed functions
# ---------------------------------------------------------------------------


@check
@beartype
def add_f32(x: F32[N, C], y: F32[N, C]) -> F32[N, C]:
  return x + y


@beartype
def bf16_fn(x: BF16[N, C]) -> BF16[N, C]:
  return x


# ---------------------------------------------------------------------------
# All concrete dtype variants
# ---------------------------------------------------------------------------


@beartype
def bool_fn(x: Bool[N]) -> Bool[N]:
  return x


@beartype
def i32_fn(x: I32[N]) -> I32[N]:
  return x


@beartype
def i64_fn(x: I64[N]) -> I64[N]:
  return x


@beartype
def u8_fn(x: U8[N]) -> U8[N]:
  return x


@beartype
def f16_fn(x: F16[N]) -> F16[N]:
  return x


@beartype
def f64_fn(x: F64[N]) -> F64[N]:
  return x


# ---------------------------------------------------------------------------
# Category types
# ---------------------------------------------------------------------------


@beartype
def int_fn(x: Int[N, C]) -> Int[N, C]:
  return x


@beartype
def num_fn(x: Num[N]) -> Num[N]:
  return x


@beartype
def shaped_fn(x: Shaped[N, C]) -> Shaped[N, C]:
  return x


# ---------------------------------------------------------------------------
# Dimensions: variadic, broadcast, anonymous, custom, arithmetic
# ---------------------------------------------------------------------------


@beartype
def batched(x: F32[B, N, C]) -> F32[B, N, C]:
  return x


@beartype
def with_anon(x: F32[__, C]) -> F32[__, C]:
  return x


@beartype
def fn_4d(x: F32[N, C, H, W]) -> F32[N, C, H, W]:
  return x


@beartype
def scalar_fn(x: F32[Scalar]) -> F32[Scalar]:
  return x


if TYPE_CHECKING:
  type Vocab = int
  type PaddedN = int
else:
  Vocab = Dimension("Vocab")
  PaddedN = N + 2


@beartype
def vocab_fn(x: I64[N, Vocab]) -> I64[N, Vocab]:
  return x


@beartype
def pad(x: F32[N]) -> F32[PaddedN]:
  return x


# ---------------------------------------------------------------------------
# Like types — verify import, subscript, and annotation usage
# ---------------------------------------------------------------------------


@beartype
def like_fn(x: F32Like[N, C]) -> F32Like[N, C]:
  return x


@beartype
def bf16_like_fn(x: BF16Like[N]) -> BF16Like[N]:
  return x


@beartype
def i64_like_fn(x: I64Like[N]) -> I64Like[N]:
  return x


@beartype
def bool_like_fn(x: BoolLike[N]) -> BoolLike[N]:
  return x
