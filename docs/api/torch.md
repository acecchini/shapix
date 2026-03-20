---
title: "shapix.torch"
description: PyTorch tensor types and Like types.
---

# `shapix.torch`

PyTorch tensor types with dtype and shape checking. Base type: `torch.Tensor`.

```python
from shapix.torch import F32, BF16, Shaped
from shapix.torch import F32Like, F32Lk
```

---

## Array Types

Same type names as `shapix.numpy`, plus **`BF16`** (bfloat16).

### Concrete dtypes

| Type | Dtype |
|------|-------|
| `Bool` | `bool` |
| `I8`, `I16`, `I32`, `I64` | `int8` – `int64` |
| `U8`, `U16`, `U32`, `U64` | `uint8` – `uint64` |
| `BF16` | `bfloat16` |
| `F16`, `F32`, `F64` | `float16` – `float64` |
| `C64`, `C128` | `complex64`, `complex128` |

### Category dtypes

Same as NumPy: `Int`, `UInt`, `Integer`, `Float`, `Real`, `Complex`, `Inexact`, `Num`, `Shaped`.

---

## Like Types

All scalar `Like` types are re-exported from `shapix.numpy`.

### Array-Like (Lk) types

Lk types use `torch.Tensor | np.ndarray` as the array component:

| Type | Scalar | Array |
|------|--------|-------|
| `BoolLk` | `BoolLike` | `Tensor \| np.ndarray` |
| `I8Lk` – `I64Lk` | `IntNLike` | `Tensor \| np.ndarray` |
| `F16Lk` – `F64Lk` | `FloatNLike` | `Tensor \| np.ndarray` |
| `BF16Lk` | `FloatLike` | `Tensor \| np.ndarray` |
| `IntLk` – `ShapedLk` | Category scalar | `Tensor \| np.ndarray` |

---

## Usage

```python
import torch
from beartype import beartype
from shapix import N, C, H, W
from shapix.torch import F32, BF16

@beartype
def attention(q: F32[N, L, C], k: F32[N, L, C], v: F32[N, L, C]) -> F32[N, L, C]:
    scores = torch.matmul(q, k.transpose(-2, -1)) / (C ** 0.5)
    return torch.matmul(torch.softmax(scores, dim=-1), v)

@beartype
def mixed_precision(x: BF16[N, C]) -> BF16[N, C]:
    return x
```

!!! note "Tree support"
    PyTorch users can use tree annotations via `shapix.optree.Tree` or `shapix.Tree` (which auto-detects optree or jax).
