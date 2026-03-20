---
title: Array Types
description: Dtype-checked array types for NumPy, JAX, PyTorch, and custom backends.
---

# Array Types

Shapix provides concise, readable array type annotations for every major backend. Each type enforces both **dtype** and **shape** at runtime.

## NumPy

```python
from shapix.numpy import F32, I64, Shaped  # and many more
```

### Concrete dtypes

| Type | Dtype |
|------|-------|
| `Bool` | `bool` |
| `I8`, `I16`, `I32`, `I64` | `int8` – `int64` |
| `U8`, `U16`, `U32`, `U64` | `uint8` – `uint64` |
| `F16`, `F32`, `F64`, `F128` | `float16` – `float128` / `longdouble` |
| `C64`, `C128`, `C256` | `complex64` – `complex256` / `clongdouble` |

### Additional dtypes

| Type | Dtype |
|------|-------|
| `V` | `void` |
| `Str` | `string` |
| `Bytes` | `bytes` |
| `Obj` | `object` |
| `DT64` | `datetime64` (accepts unit-qualified dtypes like `datetime64[ns]`) |
| `TD64` | `timedelta64` (accepts unit-qualified dtypes like `timedelta64[ms]`) |

### Category dtypes

Category types accept any dtype within the category:

| Type | Includes |
|------|----------|
| `Int` | All signed integers |
| `UInt` | All unsigned integers |
| `Integer` | All integers (signed + unsigned) |
| `Float` | All floats (including bfloat16) |
| `Real` | Integers + floats |
| `Complex` | `complex64`, `complex128` |
| `Inexact` | Floats + complex |
| `Num` | All numeric types |
| `Shaped` | Any dtype (shape-only checking) |

### Structured dtypes

Use `Structured()` for NumPy structured (record) dtypes:

```python
from shapix.numpy import Structured

Point = Structured([("x", np.float32), ("y", np.float32)])

@beartype
def process_points(pts: Point[N]) -> Point[N]:
    return pts

# Exact field match required
pts = np.zeros(10, dtype=[("x", np.float32), ("y", np.float32)])
process_points(pts)  # OK

wrong = np.zeros(10, dtype=[("a", np.float32), ("b", np.float32)])
process_points(wrong)  # Raises — field names don't match
```

### Endianness variants

For multi-byte dtypes, create endianness-constrained types programmatically using `make_array_type` with endianness `DtypeSpec` constants:

```python
from shapix import make_array_type, N, C
from shapix._dtypes import FLOAT32_LE, INT64_BE

F32LE = make_array_type(np.ndarray, FLOAT32_LE)

@beartype
def process_le(x: F32LE[N, C]) -> F32LE[N, C]:
    """Only accepts little-endian float32 arrays."""
    return x

process_le(np.ones((4, 3), dtype="<f4"))  # OK — little-endian
process_le(np.ones((4, 3), dtype=">f4"))  # Raises — big-endian
```

Available suffixes: `_LE` (little-endian), `_BE` (big-endian), `_N` (native).

### Usage

```python
import numpy as np
from beartype import beartype
from shapix import N, C
from shapix.numpy import F32, Integer, Shaped

@beartype
def matmul(x: F32[N, C], y: F32[C, N]) -> F32[N, N]:
    return x @ y

@beartype
def any_int(x: Integer[N]) -> Integer[N]:
    return x  # Accepts int8, int16, int32, int64, uint8, ...

@beartype
def any_shape(x: Shaped[N, C]) -> Shaped[N, C]:
    return x  # Any dtype, just check shape
```

## JAX

```python
from shapix.jax import F32, BF16
```

Most NumPy type names, plus **`BF16`** (bfloat16). Base type is `jax.Array`. Also exports `Like` types and `Tree`.

```python
import jax.numpy as jnp
from beartype import beartype
from shapix import N, C
from shapix.jax import BF16

@beartype
def forward(x: BF16[N, C]) -> BF16[N, C]:
    return jnp.relu(x)
```

## PyTorch

```python
from shapix.torch import F32, BF16
```

Most NumPy type names, plus **`BF16`** (bfloat16). Base type is `torch.Tensor`. Also exports `Like` types.

```python
import torch
from beartype import beartype
from shapix import N, C
from shapix.torch import F32

@beartype
def linear(x: F32[N, C], w: F32[C, C]) -> F32[N, C]:
    return x @ w.T
```

## Custom array types

Use `make_array_type` and `make_array_like_type` to create types for custom array classes or dtype combinations:

```python
from shapix import make_array_type, make_array_like_type, DtypeSpec

# Custom dtype: only float32 or float16 (e.g. for mixed-precision training)
MIXED = DtypeSpec("MixedPrecision", frozenset({"float16", "float32"}))

# Strict array type — only accepts np.ndarray with matching dtype
MixedArray = make_array_type(np.ndarray, MIXED)

# Like type — accepts scalars, sequences, arrays with dtype casting
MixedLike = make_array_like_type(MIXED, name="MixedLike")

# Like type with strict casting (exact match only, no upcasting)
MixedExact = make_array_like_type(MIXED, casting="no", name="MixedExact")
```

### Casting rules

The `casting` parameter on `make_array_like_type` controls dtype strictness using NumPy casting rules:

| Casting | Meaning | Example (target=float32) |
|---------|---------|--------------------------|
| `"no"` | Exact dtype only | Only `float32` accepted |
| `"equiv"` | Same kind, same size | `float32` only |
| `"safe"` | No data loss | `int16` OK, `float64` rejected |
| `"same_kind"` | Same kind allowed (default) | `int32` OK, `complex64` rejected |
| `"unsafe"` | Any cast | `complex128` OK, strings rejected |
