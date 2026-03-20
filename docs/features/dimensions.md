---
title: Dimensions
description: The complete dimension system — named, fixed, variadic, broadcastable, anonymous, and symbolic dimensions.
---

# Dimensions

Dimensions are the core building blocks of Shapix's shape specifications. They describe what each axis of an array should look like and how axes should relate across arguments.

## Named dimensions

Named dimensions **bind** to a size on first occurrence and **enforce** that size on every subsequent use within the same function call.

```python
from beartype import beartype
from shapix import N, C, H, W
from shapix.numpy import F32

@beartype
def forward(x: F32[N, C, H, W]) -> F32[N, C, H, W]:
    ...
```

**Pre-defined symbols:**

| Symbol | Typical use |
|--------|-------------|
| `N` | Batch size, count |
| `B` | Batch |
| `C` | Channels |
| `D` | Embedding dimension |
| `K` | Number of heads |
| `H` | Height |
| `W` | Width |
| `L` | Sequence length |
| `P` | Points / parameters |

## Fixed dimensions

Use plain integers for dimensions that must match an exact size:

```python
@beartype
def rgb_to_gray(x: F32[N, 3, H, W]) -> F32[N, 1, H, W]:
    ...
```

!!! tip "Static type checker note"
    If type checkers flag integer literals, wrap them in `Dimension()`: `Dimension("3")`. See [static type checkers](#static-type-checkers-pyright-mypy-ty) below.

## Symbolic dimensions

Dimensions support arithmetic. Expressions are evaluated against bound dimension values:

```python
from shapix import N, C

@beartype
def pad(x: F32[N]) -> F32[N + 2]:
    ...

@beartype
def flatten(x: F32[N, C]) -> F32[N * C]:
    return x.reshape(-1)
```

**Supported operators:** `+`, `-`, `*`, `/`, `//`, `**`, `%`

Symbolic dimensions are intentionally limited to arithmetic over dimension names and numeric literals. Attribute access and function calls are rejected.

## Runtime value dimensions

Use `Value("expr")` when a shape depends on a runtime parameter or `self` attribute rather than a previously bound dimension:

```python
from beartype import beartype
from shapix import Value
from shapix.numpy import F32
import numpy as np

Size = Value("size")
WidthPlus3 = Value("self.width + 3")


@beartype
def full(size: int) -> F32[Size]:
  return np.full((size,), 1.0, dtype=np.float32)


class SomeClass:
  width = 5

  @beartype
  def full(self) -> F32[WidthPlus3]:
    return np.full((self.width + 3,), 1.0, dtype=np.float32)
```

`Value(...)` uses a restricted arithmetic grammar. It allows names, attribute access, numeric literals, and arithmetic operators, but rejects calls, indexing, and other arbitrary Python expressions.

## Scalar dimension

`Scalar` represents a zero-dimensional array (a scalar array):

```python
from shapix import Scalar
from shapix.numpy import F32

@beartype
def dot(x: F32[N], y: F32[N]) -> F32[Scalar]:
  return np.dot(x, y)  # returns shape ()
```

!!! warning
    `Scalar` must be the only shape token. Mixed forms like `F32[N, Scalar]` or `F32[Scalar, ...]` raise `TypeError` at hint construction time.

## Variadic dimensions

Apply `~` (tilde) to make a dimension **variadic** — matching zero or more contiguous dimensions:

```python
from shapix import B, C
from shapix.numpy import F32
import numpy as np

@beartype
def normalize(x: F32[~B, C]) -> F32[~B, C]:
    return x / x.sum(axis=-1, keepdims=True)

normalize(np.ones((3,), dtype=np.float32))        # *B=(),     C=3
normalize(np.ones((4, 3), dtype=np.float32))       # *B=(4,),   C=3
normalize(np.ones((2, 4, 3), dtype=np.float32))    # *B=(2,4),  C=3
```

Named variadic dimensions enforce cross-argument consistency on the matched shape:

```python
@beartype
def add(x: F32[~B, C], y: F32[~B, C]) -> F32[~B, C]:
    return x + y
```

### Anonymous variadic

Use `~__` when you don't need consistency across arguments:

```python
from shapix import __, C

@beartype
def last_dim(x: F32[~__, C]) -> F32[~__, C]:
    return x
```

`...` (Ellipsis) is an alias for `~__`:

```python
@beartype
def last_dim(x: F32[..., C]) -> F32[..., C]:
    return x
```

!!! note "One variadic per spec"
    Only one variadic dimension is allowed per shape specification.

## Broadcastable dimensions

Apply `+` (unary plus) to make a dimension **broadcastable** — size 1 always matches, regardless of the bound value:

```python
from shapix import N, C

@beartype
def broadcast_add(x: F32[N, C], y: F32[+N, C]) -> F32[N, C]:
    return x + y

broadcast_add(np.ones((4, 3), dtype=np.float32),
              np.ones((1, 3), dtype=np.float32))   # OK — +N allows size 1
```

Broadcastable also works with variadic dimensions: `~+B` matches zero or more dims where each can be 1 or the bound value.

## Anonymous dimensions

`__` matches any single dimension without binding — no cross-argument consistency:

```python
from shapix import __, C

@beartype
def f(x: F32[__, C]) -> F32[__, C]:
    return x
# __ matches anything, only C is cross-checked
```

## Custom dimensions

Create your own with `Dimension`:

```python
from shapix import Dimension

Vocab = Dimension("Vocab")
Embed = Dimension("Embed")
Seq = Dimension("Seq")

@beartype
def embed(tokens: I64[N, Seq], table: F32[Vocab, Embed]) -> F32[N, Seq, Embed]:
    ...
```

Unary operators work on custom dimensions too: `~Vocab` (variadic), `+Vocab` (broadcastable).

### TYPE_CHECKING pattern

To silence pyright/Pylance on custom dimensions:

```python
import typing as tp
from shapix import Dimension

if tp.TYPE_CHECKING:
    type Vocab = int
    type Embed = int
else:
    Vocab = Dimension("Vocab")
    Embed = Dimension("Embed")
```

## Summary

| Syntax | Meaning | Example | Behavior |
|--------|---------|---------|----------|
| *(none)* | Named | `N` | Bind & enforce |
| `int` | Fixed | `3` | Exact match |
| `~` | Variadic | `~B` | Zero or more dims |
| `+` | Broadcastable | `+N` | Size 1 always OK |
| `~+` | Broadcastable variadic | `~+B` | Variadic + broadcast |
| `Scalar` | Scalar | `Scalar` | Zero dimensions |
| `__` | Anonymous | `__` | Match any, no binding |
| `~__` | Anonymous variadic | `~__` | Zero or more, no binding |
| `...` | Ellipsis (alias) | `...` | Same as `~__` |
| arithmetic | Symbolic | `N + 1` | Expression |

## Static type checkers (pyright, mypy, ty)

Shapix supports **pyright**, **mypy**, and **ty**. Pre-defined dimension symbols resolve to `TypeVar` under `TYPE_CHECKING`, so core annotations like `F32[N, C]` work across all three checkers.

Some patterns are fundamentally runtime-only:

| Pattern | Example | Workaround |
|---------|---------|------------|
| Integer literals | `F32[N, 3, H, W]` | Wrap in `Dimension("3")` |
| Unary operators | `F32[~B, C]`, `F32[+N, C]` | `# type: ignore` |
| Arithmetic | `F32[N + 2]` | `# type: ignore` |
| Custom dimensions | `F32[Vocab, Embed]` | `# type: ignore` or `TYPE_CHECKING` pattern |
| `Value(...)` | `F32[Value("size")]` | `# type: ignore` |
| Tree structure args | `Tree[F32[N], T]` | `# type: ignore` — leaf-only `Tree[F32[N, C]]` works |

### Recommended config

=== "pyright"

    ```toml
    [tool.pyright]
    reportInvalidTypeForm = false
    ```

=== "mypy"

    ```toml
    [tool.mypy]
    ignore_missing_imports = true
    ```

=== "ty"

    No special configuration needed.

For patterns all checkers reject, use blanket `# type: ignore`:

```python
@beartype
def pad(x: F32[N]) -> F32[N + 2]:  # type: ignore
    ...
```
