---
title: Dimensions
description: Named, fixed, variadic, broadcastable, anonymous, scalar, symbolic, and runtime value dimensions.
---

# Dimensions

Dimensions are the core shape language in shapix. They describe:

- how many axes an array should have
- which axes must agree across parameters
- where fixed sizes, symbolic expressions, or runtime values are expected

Every array alias such as `F32[...]`, `I64[...]`, or `BF16Like[...]` is subscripted with these tokens.

## Named dimensions

Named dimensions bind on first use and are enforced on every later use inside the same call.

```python
from beartype import beartype
from shapix import N, C, H, W
from shapix.numpy import F32

@beartype
def forward(x: F32[N, C, H, W]) -> F32[N, C, H, W]:
  ...
```

Pre-defined symbols:

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

Use plain integers when an axis must have an exact size:

```python
@beartype
def rgb_to_gray(x: F32[N, 3, H, W]) -> F32[N, 1, H, W]:
  ...
```

Static type checkers generally do **not** understand integer literals in these shape positions. Keep them for runtime correctness and either:

- add a targeted `# type: ignore`, or
- use the notebook-style checker-only alias pattern from [Static Typing](static-typing.md), for example `Three = tp.Literal[3]` under `TYPE_CHECKING` and `Three = Dimension(3)` at runtime

## Symbolic dimensions

Dimensions support arithmetic. Expressions are evaluated against already-bound shape names:

```python
from shapix import N, C

@beartype
def pad(x: F32[N]) -> F32[N + 2]:
  ...

@beartype
def flatten(x: F32[N, C]) -> F32[N * C]:
  return x.reshape(-1)
```

Supported operators: `+`, `-`, `*`, `/`, `//`, `**`, `%`

These expressions are intentionally narrow:

- allowed: named dimensions and numeric literals
- rejected: attribute access, indexing, function calls, arbitrary Python code

For static typing, arithmetic dimensions are runtime-only and typically need `# type: ignore`. If you use them heavily, you can also bind a checker-only alias such as `PaddedN` under `TYPE_CHECKING` and assign `PaddedN = N + 2` at runtime.

## Runtime value dimensions

Use `Value("expr")` when the shape depends on a runtime parameter or `self` attribute rather than a previously bound shape symbol.

```python
import numpy as np
from beartype import beartype
from shapix import Value
from shapix.numpy import F32

Size = Value("size")
WidthPlus3 = Value("self.width + 3")

@beartype
def full(size: int) -> F32[Size]:  # type: ignore[valid-type]
  return np.full((size,), 1.0, dtype=np.float32)

class SomeClass:
  width = 5

  @beartype
  def full(self) -> F32[WidthPlus3]:  # type: ignore[valid-type]
    return np.full((self.width + 3,), 1.0, dtype=np.float32)
```

`Value(...)` supports:

- names from the current call scope
- attribute access such as `self.width`
- numeric literals
- arithmetic operators

It rejects calls, indexing, comprehensions, and arbitrary evaluation.

When `Value(...)` appears in an async function, use `@shapix.check` if you want the scope to be explicitly preserved across the await.

## Scalar dimension

`Scalar` means "zero-dimensional array" or shape `()`.

```python
import numpy as np
from beartype import beartype
from shapix import N, Scalar
from shapix.numpy import F32

@beartype
def dot(x: F32[N], y: F32[N]) -> F32[Scalar]:
  return np.dot(x, y)
```

!!! warning
    `Scalar` must be the only shape token. Mixed forms like `F32[N, Scalar]` or `F32[Scalar, ...]` raise `TypeError` at hint construction time.

## Variadic dimensions

Apply `~` to a named dimension to match zero or more contiguous axes.

```python
import numpy as np
from beartype import beartype
from shapix import B, C
from shapix.numpy import F32

@beartype
def normalize(x: F32[~B, C]) -> F32[~B, C]:
  return x / x.sum(axis=-1, keepdims=True)

normalize(np.ones((3,), dtype=np.float32))  # B = (), C = 3
normalize(np.ones((4, 3), dtype=np.float32))  # B = (4,), C = 3
normalize(np.ones((2, 4, 3), dtype=np.float32))  # B = (2, 4), C = 3
```

Named variadic dimensions enforce cross-argument consistency on the matched sub-shape:

```python
@beartype
def add(x: F32[~B, C], y: F32[~B, C]) -> F32[~B, C]:
  return x + y
```

Static type checkers still treat `~B` as runtime-only syntax. The same notebook-style alias trick works here too: bind a placeholder such as `VariadicBatch` under `TYPE_CHECKING`, and assign `VariadicBatch = ~B` at runtime.

### Anonymous variadic

Use `~__` when you do not need cross-argument binding:

```python
from beartype import beartype
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

Apply unary `+` to make a dimension broadcastable. Size `1` always matches the bound value.

```python
import numpy as np
from beartype import beartype
from shapix import N, C
from shapix.numpy import F32

@beartype
def broadcast_add(x: F32[N, C], y: F32[+N, C]) -> F32[N, C]:
  return x + y

broadcast_add(np.ones((4, 3), dtype=np.float32),
              np.ones((1, 3), dtype=np.float32))  # OK
```

Broadcastable also works with variadic dimensions: `~+B` matches zero or more dims where each can be `1` or the bound value.

## Anonymous dimensions

`__` matches any single axis without binding it.

```python
from beartype import beartype
from shapix import __, C
from shapix.numpy import F32

@beartype
def f(x: F32[__, C]) -> F32[__, C]:
  return x
```

Unlike variadic and broadcastable syntax, `__` is part of the checker-tested static surface and can be used directly with pyright, mypy, and ty.

## Custom dimensions

Create your own with `Dimension`:

```python
from beartype import beartype
from shapix import Dimension, N
from shapix.numpy import F32, I64

Vocab = Dimension("Vocab")
Embed = Dimension("Embed")
Seq = Dimension("Seq")

@beartype
def embed(tokens: I64[N, Seq], table: F32[Vocab, Embed]) -> F32[N, Seq, Embed]:
  ...
```

Custom dimensions work at runtime immediately. For static typing, use the `TYPE_CHECKING` pattern:

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

Unary operators work on custom dimensions too: `~Vocab`, `+Vocab`, `~+Vocab`.

## Summary table

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

## Static typing notes

The checker-supported subset is documented in [Static Typing](static-typing.md). The short version:

- `F32[N, C]`, `F32[Scalar]`, and `F32[__, C]` are part of the tested cross-checker surface
- custom dimensions need the `TYPE_CHECKING` pattern above
- arithmetic, fixed integer literals, and unary operator forms such as `~B` and `+N` are still runtime-only syntax for static checkers
- for those runtime-only forms, either use targeted ignores or the notebook-style checker-only alias pattern
