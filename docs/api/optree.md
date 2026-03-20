---
title: "shapix.optree"
description: Tree annotations backed by optree.
---

# `shapix.optree`

Tree annotations using [optree](https://github.com/metaopt/optree) as the backend for tree operations.

```python
from shapix.optree import Tree, Structure
```

---

## `Tree`

Subscriptable tree type factory backed explicitly by `optree`. Functionally identical to `shapix.Tree` when optree is the detected backend.

See [Tree Annotations](../features/tree-annotations.md) for full usage patterns.

```python
from beartype import beartype
from shapix.optree import Tree
from shapix import N, T, Structure
from shapix.numpy import F32

Params = Structure("Params")

@beartype
def update(params: Tree[F32[N], Params], grads: Tree[F32[N], Params]):
    ...
```

---

## `Structure`

Re-exported from `shapix._tree`. See [`shapix.Structure`](shapix.md#structure).

---

## When to use `shapix.optree`

| Import | Backend |
|--------|---------|
| `from shapix import Tree` | Auto-detect (optree > jax) |
| `from shapix.optree import Tree` | Always optree |
| `from shapix.jax import Tree` | Always jax.tree_util |

Use `shapix.optree` when you want to **explicitly** use optree regardless of whether JAX is installed. This is useful in pure-NumPy or PyTorch projects that don't depend on JAX.
