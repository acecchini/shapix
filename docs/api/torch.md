---
title: "shapix.torch"
description: PyTorch tensor aliases, Like aliases, and ScalarLike re-exports.
---

# `shapix.torch`

`shapix.torch` provides PyTorch-native array aliases based on `torch.Tensor`.

```python
from shapix.torch import (
  F32, BF16, Int, Shaped,
  F32Like, BF16Like,
  U8ScalarLike, make_scalar_like_type,
)
```

## What it exports

Strict array aliases:

- concrete families such as `Bool`, `I32`, `I64`, `F16`, `F32`, `F64`, `BF16`, `C64`, `C128`
- category families such as `Int`, `UInt`, `Integer`, `Float`, `Real`, `Complex`, `Inexact`, `Num`, `Shaped`

`Like` aliases:

- `BF16Like`
- `BoolLike`, `I8Like` through `I64Like`, `U8Like` through `U64Like`
- `F16Like`, `F32Like`, `F64Like`
- `C64Like`, `C128Like`
- category aliases such as `IntLike`, `FloatLike`, `NumLike`, `ShapedLike`

Other exports:

- NumPy-defined `ScalarLike` aliases re-exported for convenience
- `make_scalar_like_type`

## Backend limits

`shapix.torch` does **not** export NumPy-only aliases such as:

- `F128`
- `C256`
- `V`
- `Str`
- `Bytes`
- `Obj`
- `DT64`
- `TD64`

It also requires `numpy` alongside `torch` at runtime.

## `Like` behavior

Torch `Like` aliases use `torch.as_tensor` on the slow path, so they can accept:

- real tensors
- NumPy arrays
- Python scalars and nested sequences

Static type checkers still see the result as `torch.Tensor`.

## `ScalarLike` re-exports

`ScalarLike` aliases are re-exported from `shapix.numpy`. They validate Python and NumPy scalar values, not Torch 0-D tensors.

For Torch scalar tensors, prefer a `Like` alias with `Scalar`, for example `F32Like[Scalar]`.

## Trees

`shapix.torch` does not export `Tree`.

If you want tree annotations in a Torch project, import:

- `Tree` from `shapix.optree` for an explicit OpTree backend
- or `Tree` from `shapix.jax` if your project already depends on JAX's tree utilities
