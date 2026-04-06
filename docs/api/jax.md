---
description: JAX array aliases, Like aliases, ScalarLike re-exports, and JAX Tree support.
---

# `shapix.jax`

`shapix.jax` provides JAX-native array aliases based on `jax.Array`.

```python
from shapix.jax import (
  F32, BF16, Int, Shaped,
  F32Like, BF16Like,
  U8ScalarLike, make_scalar_like_type,
  Tree, Structure,
)
```

## What it exports

Strict array aliases:

- concrete families such as `Bool`, `I32`, `I64`, `F16`, `F32`, `F64`, `BF16`,
    `C64`, `C128`
- category families such as `Int`, `UInt`, `Integer`, `Float`, `Real`,
    `Complex`, `Inexact`, `Num`, `Shaped`

`Like` aliases:

- `BF16Like`
- `BoolLike`, `I8Like` through `I64Like`, `U8Like` through `U64Like`
- `F16Like`, `F32Like`, `F64Like`
- `C64Like`, `C128Like`
- category aliases such as `IntLike`, `FloatLike`, `NumLike`, `ShapedLike`

Other exports:

- NumPy-defined `ScalarLike` aliases re-exported for convenience
- `make_scalar_like_type`
- `Tree`
- `Structure`

## Backend limits

`shapix.jax` does **not** export NumPy-only aliases such as:

- `F128`
- `C256`
- `V`
- `Str`
- `Bytes`
- `Obj`
- `DT64`
- `TD64`

It also requires `numpy` alongside `jax` at runtime.

## `Like` behavior

JAX `Like` aliases use `jnp.asarray` on the slow path, so they can accept:

- real JAX arrays
- NumPy arrays
- Python scalars and nested sequences
- objects implementing `__jax_array__`

Static type checkers still see the result as `jax.Array`.

## `ScalarLike` re-exports

`ScalarLike` aliases are re-exported from `shapix.numpy`. They validate Python
and NumPy scalar values, not JAX 0-D arrays.

For JAX scalar arrays, prefer a `Like` alias with `Scalar`, for example
`F32Like[Scalar]`.

## `Tree`

`shapix.jax.Tree` is the JAX-backed pytree annotation.

```python
from beartype import beartype
from shapix import N, T
from shapix.jax import F32, Tree

@beartype
def process(params: Tree[F32[N], T],
            grads: Tree[F32[N], T]) -> Tree[F32[N]]:  # type: ignore[valid-type]
  ...
```

Leaf-only annotations such as `Tree[F32[N]]` are checker-friendly.
Structure-bearing forms such as `Tree[..., T]` are runtime-only.
