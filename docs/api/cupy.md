---
title: "shapix.cupy"
description: CuPy array aliases, Like aliases, and ScalarLike re-exports.
---

# `shapix.cupy`

`shapix.cupy` provides CuPy-native array aliases based on `cupy.ndarray`.

```python
from shapix.cupy import (
  F32, I64, Int, Shaped,
  F32Like, I64Like,
  U8ScalarLike, make_scalar_like_type,
)
```

## What it exports

Strict array aliases:

- concrete families such as `Bool`, `I32`, `I64`, `F16`, `F32`, `F64`, `C64`, `C128`
- category families such as `Int`, `UInt`, `Integer`, `Float`, `Real`, `Complex`, `Inexact`, `Num`, `Shaped`

`Like` aliases:

- `BoolLike`, `I8Like` through `I64Like`, `U8Like` through `U64Like`
- `F16Like`, `F32Like`, `F64Like`
- `C64Like`, `C128Like`
- category aliases such as `IntLike`, `FloatLike`, `NumLike`, `ShapedLike`

Other exports:

- NumPy-defined `ScalarLike` aliases re-exported for convenience
- `make_scalar_like_type`

## Backend limits

`shapix.cupy` does **not** export:

- `BF16`
- `F128`
- `C256`
- `V`
- `Str`
- `Bytes`
- `Obj`
- `DT64`
- `TD64`

It also requires `numpy` alongside `cupy` at runtime.

## `Like` behavior

CuPy `Like` aliases use `cupy.asarray` on the slow path, so they can accept:

- real CuPy arrays
- NumPy arrays
- Python scalars and nested sequences

Static type checkers still see the result as `cupy.ndarray`.

## `ScalarLike` re-exports

`ScalarLike` aliases are re-exported from `shapix.numpy`. They validate Python and NumPy scalar values, not CuPy 0-D arrays.

For CuPy scalar arrays, prefer a `Like` alias with `Scalar`, for example `F32Like[Scalar]`.
