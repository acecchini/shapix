---
title: "shapix.jax"
description: JAX array types, Tree annotations, and Like types.
---

# `shapix.jax`

JAX array types with dtype and shape checking. Base type: `jax.Array`.

```python
from shapix.jax import F32, BF16, Shaped
from shapix.jax import Tree, Structure
from shapix.jax import F32Like, F32Lk
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

## Tree

```python
from shapix.jax import Tree
```

Tree annotations backed explicitly by `jax.tree_util`. See [Tree Annotations](../features/tree-annotations.md) for full usage.

```python
from beartype import beartype
from shapix.jax import Tree, F32
from shapix import N, T

@beartype
def process(params: Tree[F32[N], T], grads: Tree[F32[N], T]):
    ...
```

Also re-exports `Structure` for custom structure symbols.

---

## Like Types

All scalar `Like` types are re-exported from `shapix.numpy`.

### Array-Like (Lk) types

Lk types use `jax.Array | np.ndarray` as the array component:

| Type | Scalar | Array |
|------|--------|-------|
| `BoolLk` | `BoolLike` | `jax.Array \| np.ndarray` |
| `I8Lk` – `I64Lk` | `IntNLike` | `jax.Array \| np.ndarray` |
| `F16Lk` – `F64Lk` | `FloatNLike` | `jax.Array \| np.ndarray` |
| `BF16Lk` | `FloatLike` | `jax.Array \| np.ndarray` |
| `IntLk` – `ShapedLk` | Category scalar | `jax.Array \| np.ndarray` |
