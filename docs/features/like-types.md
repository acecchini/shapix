---
title: Like Types
description: Input validation types that accept scalars, arrays, or nested sequences.
---

# Like Types

`Like` types accept **scalars**, **arrays**, or **nested sequences of any depth** — use them for function inputs that will be converted to arrays.

## Basic usage

Like types **must be subscripted** — use `[...]` (Ellipsis) to accept any shape, or `[N, C]` to enforce specific dimensions:

```python
import numpy as np
from beartype import beartype
from shapix import N, C
from shapix.numpy import F32, F32Like

@beartype
def to_array(x: F32Like[...]) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)

to_array(3.14)                          # scalar
to_array([1.0, 2.0, 3.0])              # 1D list
to_array([[1.0, 2.0], [3.0, 4.0]])     # 2D nested list
to_array(np.ones((3, 4)))              # ndarray
to_array([[[[[[1.0]]]]]])              # 6D+ — no depth limit

@beartype
def process(x: F32Like[N, C]) -> F32[N, C]:
    return np.asarray(x, dtype=np.float32)
```

Dtype compatibility uses NumPy's `same_kind` casting rules by default: `int32` can be passed where `float32` is expected (safe upcast), but `complex128` cannot.

## Available Like types

**Concrete:** `BoolLike`, `I8Like`–`I64Like`, `U8Like`–`U64Like`, `F16Like`–`F128Like`, `C64Like`–`C256Like`

**Category:** `IntLike`, `FloatLike`, `NumLike`, `ShapedLike`, etc.

Like types are also available in JAX and PyTorch backends:

```python
from shapix.jax import F32Like    # accepts jax.Array, ndarray, scalars, sequences
from shapix.torch import F32Like  # accepts Tensor, ndarray, scalars, sequences
```

## ScalarLike types (range-validated scalars)

ScalarLike types validate individual scalar values with range checking — no shape, just value.

```python
from shapix.numpy import U8ScalarLike, F32ScalarLike

@beartype
def clamp_pixel(value: U8ScalarLike) -> int:
    """Accepts int in [0, 255] range."""
    return int(value)

clamp_pixel(128)   # OK
clamp_pixel(256)   # Raises — out of uint8 range
clamp_pixel(-1)    # Raises — negative not allowed for unsigned
```

!!! warning "Boolean exclusion"
    Numeric scalar aliases (`I8ScalarLike`, `F32ScalarLike`, `NumScalarLike`, etc.) reject `bool` and `np.bool_` values. Python `bool` is a subclass of `int`, but shapix treats booleans as non-numeric. Use `BoolScalarLike` for boolean scalars.

**Available:** `BoolScalarLike`, `I8ScalarLike`–`I64ScalarLike`, `U8ScalarLike`–`U64ScalarLike`, `F16ScalarLike`–`F128ScalarLike`, `C64ScalarLike`, `C128ScalarLike`, `C256ScalarLike`, plus category aliases `IntScalarLike`, `FloatScalarLike`, `NumScalarLike`, etc.

Also: `StringLike` (`str | np.str_`).

ScalarLike types are available from all backends:

```python
from shapix.numpy import U8ScalarLike   # defined here
from shapix.jax import U8ScalarLike     # re-exported
from shapix.torch import U8ScalarLike   # re-exported
```

### Custom ScalarLike types

Use `make_scalar_like_type` for custom casting rules:

```python
from shapix.numpy import make_scalar_like_type

F32ScalarStrict = make_scalar_like_type(np.float32, casting="no")     # exact np.float32 only
F32ScalarSafe = make_scalar_like_type(np.float32, casting="safe")     # float16 OK, complex rejected
```

## ArrayLike template

The `ArrayLike` recursive type alias is public for custom static type combinations:

```python
from shapix.numpy import ArrayLike

type MyInputType = ArrayLike[float, np.float32]
```

This uses PEP 695 recursive type aliases — no depth limit on nesting.
