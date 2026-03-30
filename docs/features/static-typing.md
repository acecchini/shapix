---
title: Static Typing
description: How shapix annotations map onto pyright, mypy, and ty.
---

# Static Typing

Shapix supports **pyright**, **mypy**, and **ty**. The repository runs all three against the typing fixtures in `tests/typing/` via `tests/test_typecheck.py`.

At a high level:

- under `TYPE_CHECKING`, backend array aliases resolve to real static array types such as `numpy.typing.NDArray`, `jax.Array`, `torch.Tensor`, or `cupy.ndarray`
- pre-defined dimensions such as `N`, `C`, and `Scalar` are represented in a checker-friendly way
- some syntax is still inherently runtime-only and needs either targeted ignores or checker-only aliases

## Works directly

These patterns are part of the tested public typing surface:

```python
from beartype import beartype
from shapix import C, N, Scalar, __, check
from shapix.numpy import F32

@beartype
def f(x: F32[N, C]) -> F32[N, C]:
  return x

@beartype
def scalar(x: F32[Scalar]) -> F32[Scalar]:
  return x

@beartype
def keep_last(x: F32[__, C]) -> F32[__, C]:
  return x

@check
async def async_identity(x: F32[N]) -> F32[N]:
  return x
```

This also extends to:

- backend aliases from `shapix.jax`, `shapix.torch`, and `shapix.cupy`
- `Like` aliases such as `F32Like[N, C]`
- leaf-only tree annotations such as `Tree[F32[N, C]]`
- the public `ArrayLike` template and backend `ScalarLike` aliases

## What is runtime-only

The following syntax is valid at runtime but is still beyond what the static checkers model directly:

| Pattern | Example | Typical workaround |
|---------|---------|--------------------|
| Fixed integer literal dims | `F32[N, 3, H, W]` | targeted `# type: ignore` or checker-only alias |
| Arithmetic dims | `F32[N + 2]` | targeted `# type: ignore` or checker-only alias |
| `Value(...)` dims | `F32[Value("size")]` | targeted `# type: ignore` |
| Variadic dims | `F32[~B, C]` | targeted `# type: ignore` or checker-only alias |
| Broadcastable dims | `F32[+N, C]` | targeted `# type: ignore` or checker-only alias |
| Tree structure args | `Tree[F32[N], T]` | targeted `# type: ignore` |

Example:

```python
from beartype import beartype
from shapix import N, Value
from shapix.numpy import F32

@beartype
def pad(x: F32[N]) -> F32[N + 2]:  # type: ignore[valid-type]
  ...

@beartype
def sized(size: int) -> F32[Value("size")]:  # type: ignore[valid-type]
  ...
```

Prefer narrow, annotation-local ignores like these instead of weakening global checker strictness for an entire project.

That is also the main repo-tested baseline in `tests/typing/`.

## Checker-only aliases for runtime-only tokens

When you use runtime-only tokens frequently, you can keep signatures cleaner by defining a checker-only placeholder under `TYPE_CHECKING` and binding it to the real runtime token in the `else` branch.

### Fixed integer literals

```python
import typing as tp
from beartype import beartype
from shapix import Dimension, H, N, W
from shapix.numpy import F32

if tp.TYPE_CHECKING:
  Three = tp.Literal[3]
else:
  Three = Dimension(3)

@beartype
def process_rgb(x: F32[N, Three, H, W]) -> F32[N, Three, H, W]:
  return x
```

### Variadic, broadcastable, and symbolic aliases

```python
import typing as tp
from beartype import beartype
from shapix import B, C, N
from shapix.numpy import F32

if tp.TYPE_CHECKING:
  VariadicBatch = tp.Literal["VariadicBatch"]
  BroadcastN = tp.Literal["BroadcastN"]
  PaddedN = tp.Literal["PaddedN"]
else:
  VariadicBatch = ~B
  BroadcastN = +N
  PaddedN = N + 2

@beartype
def softmax(x: F32[VariadicBatch, C]) -> F32[VariadicBatch, C]:
  return x

@beartype
def broadcast_add(x: F32[N, C], y: F32[BroadcastN, C]) -> F32[N, C]:
  return x

@beartype
def pad(x: F32[N]) -> F32[PaddedN]:
  return x
```

Notes:

- the checker-side placeholder name is arbitrary; it just needs to be a stable alias object
- this is an advanced convenience pattern, not the only supported approach
- if you only need the syntax occasionally, a targeted `# type: ignore` is simpler and more explicit

## Custom dimensions

Custom dimensions are runtime objects. To make them usable in annotations across all three checkers, define a checker-only alias:

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
def embed(tokens: I64[N], table: F32[Vocab, Embed]) -> F32[N, Embed]:
  return table[tokens]
```

This is the simplest checker-only alias pattern and the one most users should keep in their toolbox first.

## Tree annotations

`Tree` has a split contract:

- `Tree[F32[N, C]]` is checker-friendly and tested
- `Tree[F32[N], T]`, `Tree[F32[N], T, ...]`, and similar structure-bearing forms are runtime-only and need a targeted ignore

```python
from shapix import N, T
from shapix.numpy import F32
from shapix.optree import Tree

def leaves_only(x: Tree[F32[N]]) -> Tree[F32[N]]:
  return x

def structure_checked(x: Tree[F32[N], T]) -> Tree[F32[N]]:  # type: ignore[valid-type]
  return x
```

## Backend notes

The typing model differs slightly from runtime behavior:

- `shapix.jax.F32Like[...]` and `shapix.torch.F32Like[...]` accept scalars and nested sequences at runtime, but static checkers see them as `jax.Array` and `torch.Tensor`
- `Shaped[...]` and `ShapedLike[...]` accept any dtype at runtime, while their static aliases are approximations
- backend `ScalarLike` aliases validate Python and NumPy scalar values, not backend-native 0-D arrays

For backend-native scalar arrays, annotate the input as a `Like` type with `Scalar`, for example `F32Like[Scalar]`.
