---
title: Quick Start
description: Write your first shape-checked function and learn the core execution model.
---

# Quick Start

This page uses NumPy for concreteness, but the same shape language works across `shapix.jax`, `shapix.torch`, and `shapix.cupy`.

## 1. Start with standard `@beartype`

Import dimension symbols from the root module and array aliases from a backend module:

```python
import numpy as np
from beartype import beartype
from shapix import N, C
from shapix.numpy import F32

@beartype
def normalize(x: F32[N, C]) -> F32[N, C]:
  return x / x.sum(axis=1, keepdims=True)

normalize(np.ones((4, 3), dtype=np.float32))  # OK
normalize(np.ones((4, 3), dtype=np.float64))  # Raises: wrong dtype
normalize(np.ones((4,), dtype=np.float32))  # Raises: wrong rank
```

`F32[N, C]` means:

- the object must be a NumPy `float32` array
- it must have rank 2
- the first axis is named `N`
- the second axis is named `C`

`N` and `C` are runtime dimension symbols. They bind on first use inside one call and are checked everywhere else in that call.

## 2. Cross-argument consistency is automatic

You do not need a separate decorator to share dimension bindings between parameters:

```python
@beartype
def add(x: F32[N, C], y: F32[N, C]) -> F32[N, C]:
  return x + y

add(np.ones((4, 3), dtype=np.float32),
    np.ones((4, 3), dtype=np.float32))  # OK

add(np.ones((4, 3), dtype=np.float32),
    np.ones((5, 3), dtype=np.float32))  # Raises: N mismatch
```

## 3. Return values are checked too

Return annotations are part of the same contract:

```python
@beartype
def bad_flatten(x: F32[N, C]) -> F32[N, C]:
  return x.reshape(-1)
```

Calling `bad_flatten` raises because the returned array no longer has shape `(N, C)`.

## 4. Calls are independent

Each function invocation gets a fresh set of dimension bindings:

```python
@beartype
def f(x: F32[N]) -> F32[N]:
  return x

f(np.ones((3,), dtype=np.float32))  # N = 3
f(np.ones((100,), dtype=np.float32))  # N = 100, no conflict with the prior call
```

## 5. Add `@shapix.check` only when you need explicit memo scope

Most code should stop at `@beartype`. Add `@shapix.check` when you want explicit memo management:

```python
import shapix
from beartype import beartype
from shapix import Value
from shapix.numpy import F32

@shapix.check
@beartype
async def make_batch(size: int) -> F32[Value("size")]:  # type: ignore[valid-type]
  ...
```

That helper matters most when:

- extra decorators or framework wrappers make call-stack detection brittle
- you want `Value(...)` scope to be explicit across an `await`
- you want to pair memo management with `BeartypeConf`

## What's next

!!! tip "Next steps"

    - :material-ruler: **[Dimensions](../features/dimensions.md)** — Named, fixed, variadic, broadcastable, symbolic, `Scalar`, and `Value(...)`.
    - :material-check-decagram: **[Static Typing](../features/static-typing.md)** — What works directly on pyright, mypy, and ty, and where targeted ignores are still required.
    - :material-grid: **[Array Types](../features/array-types.md)** — Dtype families, structured dtypes, endianness, and backend differences.
    - :material-file-tree: **[Tree Annotations](../features/tree-annotations.md)** — Leaf checking and structure binding for pytrees.
