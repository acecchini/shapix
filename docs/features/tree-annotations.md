---
title: Tree Annotations
description: Validate pytree leaves and optionally bind tree structure across arguments.
---

# Tree Annotations

Tree annotations validate nested container structures such as dicts, lists, tuples, namedtuples, and other pytree-compatible objects.

Import `Tree` from an explicit backend module:

=== "optree"

    ```python
    from shapix.optree import Tree
    ```

=== "JAX"

    ```python
    from shapix.jax import Tree
    ```

The root `shapix` module exports `Structure`, `T`, and `S`, but not `Tree` itself.

## Basic leaf checking

`Tree[LeafType]` means "every leaf in the pytree must satisfy `LeafType`".

```python
import numpy as np
from beartype import beartype
from shapix import N, C
from shapix.numpy import F32
from shapix.optree import Tree

@beartype
def process(data: Tree[F32[N, C]]) -> Tree[F32[N, C]]:
  ...

process({
  "params": np.ones((3, 4), dtype=np.float32),
  "state": np.ones((3, 4), dtype=np.float32),
})
```

Dimension bindings are shared across the whole tree, so `N` and `C` must agree across all leaves.

## Structure binding

Named structure symbols (`T`, `S`) enforce that multiple arguments share identical tree shapes:

```python
import numpy as np
from beartype import beartype
from shapix import N, T
from shapix.numpy import F32
from shapix.optree import Tree

@beartype
def add_trees(x: Tree[F32[N], T], y: Tree[F32[N], T]) -> Tree[F32[N]]:  # type: ignore[valid-type]
  ...
```

Structure symbols are runtime-only. Static type checkers understand `Tree[F32[N]]`, but not the extra structure arguments, so those function signatures need a targeted `# type: ignore`.

## Multi-level structure matching

Structure names are interpreted from outer to inner unless `...` changes the direction or truncates the match.

### Full structure binding

```python
@beartype
def f(x: Tree[F32[N], T], y: Tree[F32[N], T]):  # type: ignore[valid-type]
  ...
```

### Top-level only

Trailing `...` makes each name capture only one level, with inner levels unchecked:

```python
@beartype
def f(x: Tree[F32[N], T, ...], y: Tree[F32[N], T, ...]):  # type: ignore[valid-type]
  ...
```

### Bottom-level only

Leading `...` matches names from the bottom up:

```python
@beartype
def f(x: Tree[F32[N], ..., T], y: Tree[F32[N], ..., T]):  # type: ignore[valid-type]
  ...
```

### Two-level matching

```python
@beartype
def f(x: Tree[int, T, S], y: Tree[int, T, S]):  # type: ignore[valid-type]
  ...

@beartype
def g(x: Tree[F32[N], T, S, ...]):  # type: ignore[valid-type]
  ...

@beartype
def h(x: Tree[F32[N], ..., T, S]):  # type: ignore[valid-type]
  ...
```

## Custom structure symbols

Create your own with `Structure`:

```python
from beartype import beartype
from shapix import N, Structure
from shapix.numpy import F32, I64
from shapix.optree import Tree

Params = Structure("Params")
State = Structure("State")

@beartype
def train(params: Tree[F32[N], Params],
          state: Tree[I64[N], State]):  # type: ignore[valid-type]
  ...
```

## Static typing split

Checker-friendly:

- `Tree[object]`
- `Tree[int]`
- `Tree[F32[N]]`
- `Tree[F32[N, C]]`

Runtime-only add-ons:

- `Tree[F32[N], T]`
- `Tree[F32[N], T, ...]`
- `Tree[F32[N], ..., T]`
- any custom `Structure` symbol inside the subscript

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
