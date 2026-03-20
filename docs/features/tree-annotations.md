---
title: Tree Annotations
description: Validate leaves and enforce structure consistency in nested containers.
---

# Tree Annotations

Tree annotations validate all leaves in a nested structure (dicts, lists, tuples, namedtuples). Import `Tree` from an explicit backend module:

## Importing Tree

=== "optree"

    ```python
    from shapix.optree import Tree
    ```

=== "JAX"

    ```python
    from shapix.jax import Tree
    ```

## Basic leaf checking

`Tree[LeafType]` validates that every leaf in the structure matches the given type:

```python
import numpy as np
from beartype import beartype
from shapix import N, C
from shapix.optree import Tree  # or: from shapix.jax import Tree
from shapix.numpy import F32

@beartype
def process(data: Tree[F32[N, C]]) -> Tree[F32[N, C]]:
    ...

# All leaves must be F32 arrays with consistent N and C
process({"params": np.ones((3, 4), dtype=np.float32),
         "state": np.ones((3, 4), dtype=np.float32)})
```

Dimension bindings are shared across leaves — `N` and `C` must be consistent across the entire tree.

## Structure binding

Named structure symbols (`T`, `S`) enforce that multiple arguments share identical tree shapes:

```python
from shapix import T, N
from shapix.optree import Tree
from shapix.numpy import F32

@beartype
def add_trees(x: Tree[F32[N], T], y: Tree[F32[N], T]) -> Tree[F32[N]]:
    ...

x = {"a": np.ones((3,), dtype=np.float32),
     "b": np.ones((3,), dtype=np.float32)}

y = {"a": np.ones((3,), dtype=np.float32),
     "b": np.ones((3,), dtype=np.float32)}

add_trees(x, y)  # OK — same structure {"a": ..., "b": ...}

add_trees({"a": np.ones((3,), dtype=np.float32)},
          [np.ones((3,), dtype=np.float32)])  # Raises — different structure
```

## Multi-level structure matching

Structure names are listed left-to-right from outer to inner. The behavior depends on whether `...` is present:

### Full structure binding

```python
# T = full tree structure (all levels)
@beartype
def f(x: Tree[F32[N], T], y: Tree[F32[N], T]):
    ...
```

### Top-level only

Trailing `...` makes each name capture only one level, with inner levels unchecked:

```python
# T = top-level only, subtrees can differ
@beartype
def f(x: Tree[F32[N], T, ...], y: Tree[F32[N], T, ...]):
    ...
```

### Bottom-level only

Leading `...` matches names from the bottom up:

```python
# T = bottom-level only (leaf-adjacent container)
@beartype
def f(x: Tree[F32[N], ..., T], y: Tree[F32[N], ..., T]):
    ...
```

### Two-level matching

```python
# T = top level (one level), S = full remaining structure below
@beartype
def f(x: Tree[int, T, S], y: Tree[int, T, S]):
    ...

# T = top, S = next, inner levels unchecked
@beartype
def f(x: Tree[F32[N], T, S, ...]):
    ...

# S = bottom, T = second-from-bottom
@beartype
def f(x: Tree[F32[N], ..., T, S]):
    ...
```

## Custom structure symbols

Create your own with `Structure`:

```python
from shapix import Structure

Params = Structure("Params")
State = Structure("State")

@beartype
def train(params: Tree[F32[N], Params], state: Tree[I64[N], State]):
    ...
```

## Summary

| Pattern | Meaning |
|---------|---------|
| `Tree[LeafType]` | Leaf checking only |
| `Tree[LeafType, T]` | Full structure binding |
| `Tree[LeafType, T, ...]` | Top-level only |
| `Tree[LeafType, ..., T]` | Bottom-level only |
| `Tree[LeafType, T, S]` | T = top (one level), S = full remaining |
| `Tree[LeafType, T, S, ...]` | T = top, S = next, inner unchecked |
| `Tree[LeafType, ..., T, S]` | S = bottom, T = second-from-bottom |
