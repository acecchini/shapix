---
title: Examples
description: Copyable examples covering the main shapix runtime and typing patterns.
---

# Examples

## Shapix Tour Notebook

The tour notebook is still the broadest runnable walkthrough in the repository:

- basic `@beartype` usage
- named dimensions and cross-argument consistency
- return checking
- fixed, variadic, broadcastable, anonymous, and symbolic dimensions
- `Value(...)`
- custom array types
- explicit memo helpers
- tree annotations

[:material-notebook: View on GitHub](https://github.com/acecchini/shapix/blob/main/examples/shapix_tour.ipynb){ .md-button .md-button--primary }

## Example 1: Plain `@beartype`

```python
import numpy as np
from beartype import beartype
from shapix import C, N
from shapix.numpy import F32

@beartype
def normalize(x: F32[N, C]) -> F32[N, C]:
  return x / x.sum(axis=1, keepdims=True)

normalize(np.ones((4, 3), dtype=np.float32))  # OK
normalize(np.ones((4,), dtype=np.float32))  # Raises
```

## Example 2: `@shapix.check` for explicit runtime scope

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

## Example 3: Custom dimensions that stay checker-friendly

```python
import typing as tp
from beartype import beartype
from shapix import Dimension, N
from shapix.numpy import F32, I64

if tp.TYPE_CHECKING:
  type Vocab = int
  type Embed = int
else:
  Vocab = Dimension("Vocab")
  Embed = Dimension("Embed")

@beartype
def embed_lookup(tokens: I64[N], table: F32[Vocab, Embed]) -> F32[N, Embed]:
  return table[tokens]
```

## Example 4: Tree leaf and structure checking

```python
from beartype import beartype
from shapix import N, T
from shapix.numpy import F32
from shapix.optree import Tree

@beartype
def accumulate(params: Tree[F32[N], T],
               grads: Tree[F32[N], T]) -> Tree[F32[N]]:  # type: ignore[valid-type]
  ...
```

Use leaf-only `Tree[F32[N]]` when you want cleaner static typing. Add structure symbols like `T` when you want runtime structure equality too.

## Example 5: Like inputs and scalar ranges

```python
from beartype import beartype
from shapix import Scalar
from shapix.numpy import F32Like, U8ScalarLike

@beartype
def to_scalar_array(x: F32Like[Scalar]) -> float:
  ...

@beartype
def clamp_pixel(value: U8ScalarLike) -> int:
  return int(value)
```
