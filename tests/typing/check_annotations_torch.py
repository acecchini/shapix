# pyright: reportMissingImports=false, reportInvalidTypeForm=false
"""Verify PyTorch tensor type annotations (pyright-specific).

Tests the full shapix annotation pattern with PyTorch backend types.

Tested with: pyright only
"""

from beartype import beartype

from shapix import B, C, Dimension, H, N, Scalar, W, __, check
from shapix.torch import BF16, Bool, F16, F32, F64, I32, I64, Int, Num, Shaped, U8

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


Vocab = Dimension("Vocab")


@beartype
def vocab_fn(x: I64[N, Vocab]) -> I64[N, Vocab]:
  return x


@beartype
def pad(x: F32[N]) -> F32[N + 2]:
  return x
