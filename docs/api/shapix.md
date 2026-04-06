---
description: Lightweight root API for dimensions, dtype specs, memo helpers, and factories.
---

# `shapix`

The root module is intentionally small and optional-dependency-safe. It is the
place to import:

- dimension symbols such as `N`, `C`, `Scalar`, and `Value`
- tree structure symbols such as `T`, `S`, and `Structure`
- `DtypeSpec`
- `make_array_type` and `make_array_like_type`
- `check` and `check_context`
- `__version__`

It does **not** export `Tree` or `make_scalar_like_type`.

## Typical imports

```python
import shapix
from shapix import (
  B, C, D, H, K, L, N, P, S, T, W, __,
  Scalar, Dimension, Value, Structure,
  DtypeSpec, make_array_type, make_array_like_type,
  check, check_context,
)
```

## `__version__`

```python
import shapix

print(shapix.__version__)
```

In an installed package this is the package version. In a plain source checkout
without installed metadata it falls back to a non-empty string such as
`0+unknown`.

## Pre-defined dimensions

<!-- markdownlint-disable MD013 -->

| Name | Meaning | | ------------------------------------------- |
---------------------------------------- | | `N`, `B`, `C`, `D`, `K`, `H`, `W`,
`L`, `P` | Named dimensions | | `__` | Anonymous single dimension | | `Scalar` |
Zero-dimensional array shape | | `Value("expr")` | Runtime value-based dimension
expression |

<!-- markdownlint-enable MD013 -->

Use `Dimension("Name")` to create your own.

## Pre-defined structure symbols

<!-- markdownlint-disable MD013 -->

| Name | Meaning | | ------------------- | ---------------------------- | | `T`,
`S` | Named tree structure symbols | | `Structure("Name")` | Custom tree
structure symbol |

<!-- markdownlint-enable MD013 -->

Remember that `Tree` itself lives in `shapix.optree` or `shapix.jax`.

## `Dimension`

`Dimension` is a `str` subclass used to build the user-facing shape language.

```python
from shapix import Dimension

Vocab = Dimension("Vocab")
Embed = Dimension("Embed")
```

Supported operators:

- `~N` for variadic dimensions
- `+N` for broadcastable dimensions
- arithmetic such as `N + 1`, `H * W`, or `2 * C`

`Scalar` cannot participate in arithmetic.

## `Value`

`Value("expr")` is the runtime-value dimension helper.

```python
from shapix import Value

Batch = Value("batch")
WidthPlus3 = Value("self.width + 3")
```

It supports names, attribute access, numeric literals, and arithmetic. It
rejects calls, indexing, and arbitrary evaluation.

## `DtypeSpec`

`DtypeSpec` describes an allowed dtype set and optional byte-order constraint.

```python
from shapix import DtypeSpec

BF16_OR_F32 = DtypeSpec("BF16orF32", frozenset({"bfloat16", "float32"}))
```

Key features:

- `allowed`: canonical dtype names
- `byteorder`: `"any"`, `"little"`, `"big"`, or `"native"`
- `matches(obj)`: runtime predicate used by array aliases
- `DtypeSpec.structured(dtype)`: exact structured-dtype matcher

## `make_array_type`

`make_array_type(array_type, dtype_spec)` creates a strict array factory for
custom array classes.

```python
import numpy as np
from beartype import beartype
from shapix import N, C, make_array_type
from shapix._dtypes import FLOAT32

MyF32 = make_array_type(np.ndarray, FLOAT32)

@beartype
def f(x: MyF32[N, C]) -> MyF32[N, C]:
  return x
```

The `array_type` only needs `.shape` and `.dtype` at runtime.

## `make_array_like_type`

`make_array_like_type(dtype_spec, *, casting="same_kind", name="ArrayLike")`
creates a broader input-contract factory for values that will be converted
before use.

```python
from shapix import make_array_like_type
from shapix._dtypes import FLOAT32

F32Input = make_array_like_type(FLOAT32, name="F32Input")
F32StrictInput = make_array_like_type(FLOAT32, casting="no", name="F32StrictInput")
```

Use backend modules when you want the built-in NumPy/JAX/Torch/CuPy `Like`
aliases. Use the root factory when you want custom dtype combinations or custom
conversion hooks.

## `check`

`@shapix.check` is the explicit memo-management decorator.

```python
import shapix
from beartype import beartype

@shapix.check
@beartype
def f(x: F32[N], y: F32[N]) -> F32[N]:
  ...
```

Or combine memo management with `BeartypeConf`:

```python
from beartype import BeartypeConf

@shapix.check(conf=BeartypeConf())
def f(x: F32[N]) -> F32[N]:
  ...
```

`check` supports sync and async functions. Generator functions are rejected.

## `check_context`

`check_context()` shares one memo across manual `is_bearable()` calls.

```python
from beartype.door import is_bearable
import shapix

with shapix.check_context():
  assert is_bearable(x, F32[N, C])
  assert is_bearable(y, F32[N, C])
```

It also supports `async with`.
