---
title: Quick Start
description: Write your first shape-checked function with Shapix.
---

# Quick Start

## Basic shape and dtype checking

Import dimension symbols and array types, then annotate your functions:

```python
import numpy as np
from beartype import beartype
from shapix import N, C
from shapix.numpy import F32

@beartype
def normalize(x: F32[N, C]) -> F32[N, C]:
    return x / x.sum(axis=1, keepdims=True)

normalize(np.ones((4, 3), dtype=np.float32))   # OK
normalize(np.ones((4, 3), dtype=np.float64))   # Raises — wrong dtype
normalize(np.ones((4,), dtype=np.float32))     # Raises — wrong rank
```

`F32[N, C]` means: a `float32` array with exactly 2 dimensions. `N` and `C` are **named dimensions** — they bind to concrete sizes at runtime and enforce consistency.

## Cross-argument consistency

Named dimensions are tracked within each function call. If `N` is bound to 4 by the first argument, all subsequent arguments must agree:

```python
@beartype
def add(x: F32[N, C], y: F32[N, C]) -> F32[N, C]:
    return x + y

add(np.ones((4, 3), dtype=np.float32),
    np.ones((4, 3), dtype=np.float32))   # OK — N=4, C=3 in both

add(np.ones((4, 3), dtype=np.float32),
    np.ones((5, 3), dtype=np.float32))   # Raises — N=4 vs N=5
```

## Sequential calls are independent

Each function invocation gets a fresh set of dimension bindings:

```python
@beartype
def f(x: F32[N]) -> F32[N]:
    return x

f(np.ones((3,), dtype=np.float32))    # N=3
f(np.ones((100,), dtype=np.float32))  # N=100 — no conflict
```

## Return type checking

Return values are validated too:

```python
@beartype
def bad_reshape(x: F32[N, C]) -> F32[N, C]:
    return x.reshape(-1)  # Returns 1D — raises!
```

## What's next?

<div class="grid cards" markdown>

-   :material-ruler:{ .lg .middle } **Dimensions**

    ---

    Named, fixed, variadic, broadcastable, symbolic — the full dimension system.

    [:octicons-arrow-right-24: Dimensions](../features/dimensions.md)

-   :material-grid:{ .lg .middle } **Array Types**

    ---

    Concrete dtypes, categories, and custom types for any backend.

    [:octicons-arrow-right-24: Array types](../features/array-types.md)

-   :material-file-tree:{ .lg .middle } **Tree Annotations**

    ---

    Validate leaves and structure of nested dicts, lists, and tuples.

    [:octicons-arrow-right-24: Trees](../features/tree-annotations.md)

</div>
