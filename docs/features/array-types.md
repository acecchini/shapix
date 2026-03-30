---
title: Array Types
description: Dtype-checked array aliases for NumPy, JAX, PyTorch, CuPy, and custom array classes.
---

# Array Types

Array aliases such as `F32[N, C]` or `DT64[...]` are the core "strict array" side of shapix. They enforce both:

- the concrete backend array class
- the allowed dtype family
- the shape specification in the subscript

At runtime these aliases become `Annotated[ArrayType, Is[validator]]`, so standard `@beartype` does the checking.

## Built-in backends

=== "NumPy"

    ```python
    from shapix.numpy import F32, I64, Shaped
    ```

    Concrete dtypes:

    | Type | Dtype |
    |------|-------|
    | `Bool` | `bool` |
    | `I8`, `I16`, `I32`, `I64` | `int8` – `int64` |
    | `U8`, `U16`, `U32`, `U64` | `uint8` – `uint64` |
    | `F16`, `F32`, `F64`, `F128` | `float16` – `float128` / `longdouble` |
    | `C64`, `C128`, `C256` | `complex64` – `complex256` / `clongdouble` |

    Additional dtypes:

    | Type | Dtype |
    |------|-------|
    | `V` | `void` |
    | `Str` | `string` |
    | `Bytes` | `bytes` |
    | `Obj` | `object` |
    | `DT64` | `datetime64` |
    | `TD64` | `timedelta64` |

    Category dtypes:

    | Type | Meaning |
    |------|---------|
    | `Int` | Any signed integer dtype |
    | `UInt` | Any unsigned integer dtype |
    | `Integer` | Any integer dtype |
    | `Float` | Any floating dtype |
    | `Real` | Any integer or floating dtype |
    | `Complex` | Any complex dtype |
    | `Inexact` | Any float or complex dtype |
    | `Num` | Any numeric dtype |
    | `Shaped` | Any dtype at runtime, shape-only checking |

    Notes:

    - `DT64` and `TD64` accept unit-qualified NumPy dtypes such as `datetime64[ns]`, `datetime64[D]`, `timedelta64[ms]`, and `timedelta64[s]`.
    - `Shaped` checks shape only at runtime. Its static alias is necessarily an approximation, not a true "any dtype ndarray" model.

=== "JAX"

    ```python
    from shapix.jax import F32, BF16
    ```

    `shapix.jax` uses `jax.Array` as the base array class.

    It exports:

    - most NumPy numeric aliases
    - `BF16` for JAX bfloat16 arrays
    - JAX `Like` aliases
    - `Tree` and `Structure`

    It does **not** expose NumPy-only aliases such as `F128`, `C256`, `V`, `Str`, `Bytes`, `Obj`, `DT64`, or `TD64`.

=== "PyTorch"

    ```python
    from shapix.torch import F32, BF16
    ```

    `shapix.torch` uses `torch.Tensor` as the base array class.

    It exports:

    - most NumPy numeric aliases
    - `BF16`
    - Torch `Like` aliases
    - NumPy-defined `ScalarLike` re-exports

    It does **not** expose NumPy-only aliases such as `F128`, `C256`, `V`, `Str`, `Bytes`, `Obj`, `DT64`, or `TD64`.

=== "CuPy"

    ```python
    from shapix.cupy import F32, I64
    ```

    `shapix.cupy` uses `cupy.ndarray` as the base array class.

    It exports:

    - most NumPy numeric aliases
    - CuPy `Like` aliases
    - NumPy-defined `ScalarLike` re-exports

    It does **not** expose `BF16`, `F128`, `C256`, or the NumPy-only non-numeric aliases `V`, `Str`, `Bytes`, `Obj`, `DT64`, and `TD64`.

### Structured dtypes

Use `Structured()` for NumPy structured (record) dtypes:

```python
import numpy as np
from beartype import beartype
from shapix import N
from shapix.numpy import Structured

Point = Structured([("x", np.float32), ("y", np.float32)])

@beartype
def process_points(pts: Point[N]) -> Point[N]:
  return pts

pts = np.zeros(10, dtype=[("x", np.float32), ("y", np.float32)])
process_points(pts)  # OK

wrong = np.zeros(10, dtype=[("a", np.float32), ("b", np.float32)])
process_points(wrong)  # Raises
```

### Endianness variants

For multi-byte dtypes, create endianness-constrained types programmatically using `make_array_type` with endianness `DtypeSpec` constants:

```python
import numpy as np
from beartype import beartype
from shapix import N, C, make_array_type
from shapix._dtypes import FLOAT32_LE

F32LE = make_array_type(np.ndarray, FLOAT32_LE)

@beartype
def process_le(x: F32LE[N, C]) -> F32LE[N, C]:
  return x

process_le(np.ones((4, 3), dtype="<f4"))  # OK
process_le(np.ones((4, 3), dtype=">f4"))  # Raises
```

Available suffixes: `_LE` (little-endian), `_BE` (big-endian), `_N` (native).

## Custom array types

Use `make_array_type` and `make_array_like_type` to create types for custom array classes or dtype combinations:

```python
import numpy as np
from shapix import DtypeSpec, make_array_like_type, make_array_type

MIXED = DtypeSpec("MixedPrecision", frozenset({"float16", "float32"}))

MixedArray = make_array_type(np.ndarray, MIXED)
MixedLike = make_array_like_type(MIXED, name="MixedLike")
MixedExact = make_array_like_type(MIXED, casting="no", name="MixedExact")
```

`make_array_type` is the strict array factory. `make_array_like_type` is the broader input-contract factory and is covered in more detail on [Like Types](like-types.md).

### `DtypeSpec`

`DtypeSpec` is the root abstraction behind these aliases:

- `allowed` is the set of canonical dtype names
- `byteorder` can restrict acceptance to `"little"`, `"big"`, or `"native"`
- `DtypeSpec.structured(...)` matches one exact NumPy structured dtype layout

That makes it the right tool when the built-in alias set is close but not quite what you need.

## When to use array types vs Like types

- Use array types such as `F32[N, C]` when the function truly requires a real backend array object.
- Use `Like` types such as `F32Like[N, C]` when callers may pass scalars, lists, tuples, or convertible foreign arrays that you immediately normalize yourself.
