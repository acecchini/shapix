---
title: Like Types
description: Input-validation types for array-like data and range-validated scalar values.
---

# Like Types

Shapix has two broad input-contract families:

- `Like` types such as `F32Like[N, C]`
- `ScalarLike` types such as `U8ScalarLike`

Use them when your API accepts values that will be converted, normalized, or range-checked before real work begins.

## Basic usage

`Like` types accept scalars, backend arrays, NumPy arrays, and nested sequences. They must always be subscripted.

```python
import numpy as np
from beartype import beartype
from shapix import N, C
from shapix.numpy import F32, F32Like

@beartype
def to_array(x: F32Like[...]) -> np.ndarray:
  return np.asarray(x, dtype=np.float32)

to_array(3.14)  # scalar
to_array([1.0, 2.0, 3.0])  # 1D list
to_array([[1.0, 2.0], [3.0, 4.0]])  # 2D nested list
to_array(np.ones((3, 4)))  # ndarray
to_array([[[[[[1.0]]]]]])  # arbitrarily deep nesting

@beartype
def process(x: F32Like[N, C]) -> F32[N, C]:
  return np.asarray(x, dtype=np.float32)
```

Use `[...]` to mean "any rank" or a full dimension spec when the input shape itself matters.

All built-in `Like` aliases are created with `make_array_like_type(..., casting="same_kind")`, so dtype compatibility follows NumPy's `"same_kind"` casting rules by default. For example, `int32` can flow into `F32Like[...]`, but `complex128` cannot.

## Built-in Like aliases

NumPy exports the broadest set:

- concrete aliases such as `BoolLike`, `I8Like`, `F32Like`, `C128Like`
- category aliases such as `IntLike`, `FloatLike`, `NumLike`, `ShapedLike`

JAX, PyTorch, and CuPy export parallel `Like` families with backend-specific conversion behavior.

That default matters in practice:

- `F32Like[...]` accepts integer inputs that can be cast to `float32` under `"same_kind"`
- `F32Like[...]` rejects complex inputs, because complex-to-float is not `"same_kind"`
- `IntLike[...]` accepts integer-like inputs but not floating-point values under the same rule

## ScalarLike types (range-validated scalars)

`ScalarLike` types validate individual scalar values. There is no shape component, only value- and dtype-family constraints.

```python
from beartype import beartype
from shapix.numpy import U8ScalarLike

@beartype
def clamp_pixel(value: U8ScalarLike) -> int:
  return int(value)

clamp_pixel(128)  # OK
clamp_pixel(256)  # Raises
clamp_pixel(-1)  # Raises
```

!!! warning "Boolean exclusion"
    Numeric scalar aliases (`I8ScalarLike`, `F32ScalarLike`, `NumScalarLike`, etc.) reject `bool` and `np.bool_` values. Python `bool` is a subclass of `int`, but shapix treats booleans as non-numeric. Use `BoolScalarLike` for boolean scalars.

Available families include:

- concrete aliases: `BoolScalarLike`, `I8ScalarLike` through `I64ScalarLike`, `U8ScalarLike` through `U64ScalarLike`, `F16ScalarLike` through `F128ScalarLike`, `C64ScalarLike`, `C128ScalarLike`, `C256ScalarLike`
- category aliases: `IntScalarLike`, `FloatScalarLike`, `RealScalarLike`, `NumScalarLike`, `ShapedScalarLike`, and others
- `StringLike` for `str | np.str_`

Backend modules re-export these NumPy-defined scalar types:

```python
from shapix.numpy import U8ScalarLike
from shapix.jax import U8ScalarLike
from shapix.torch import U8ScalarLike
from shapix.cupy import U8ScalarLike
```

!!! note
    Backend-native 0-D arrays such as `jnp.array(1.0)` or `torch.tensor(1.0)` are not `ScalarLike`. Use a `Like` alias with `Scalar`, for example `F32Like[Scalar]`.

## Backend-specific conversion behavior

The `Like` family is intentionally backend-aware:

- `shapix.numpy` slow-path conversion uses `np.asarray`
- `shapix.jax` slow-path conversion uses `jnp.asarray`, so objects implementing `__jax_array__` are accepted
- `shapix.torch` slow-path conversion uses `torch.as_tensor`
- `shapix.cupy` slow-path conversion uses `cupy.asarray`

Static type checkers only see the backend array type, not the broader runtime acceptance of scalars and nested sequences.

## Custom `ScalarLike` types

Use `make_scalar_like_type` when the built-in scalar families are close but not quite right for your API:

```python
import numpy as np
from shapix.numpy import make_scalar_like_type

F32ScalarDefault = make_scalar_like_type(np.float32)  # same_kind
F32ScalarStrict = make_scalar_like_type(np.float32, casting="no")
F32ScalarSafe = make_scalar_like_type(np.float32, casting="safe")
F32ScalarUnsafe = make_scalar_like_type(np.float32, casting="unsafe")
```

Useful interpretations:

- `casting="no"` means "exact target dtype only"
- `casting="safe"` means "no information loss"
- `casting="same_kind"` means "same numeric family" and is the default
- `casting="unsafe"` is the most permissive option and should be used deliberately

`target_dtype` may be passed as a NumPy scalar type, a dtype object, or a canonical dtype string such as `"float32"`.

Numeric `ScalarLike` factories still reject booleans by design:

```python
U8Scalar = make_scalar_like_type(np.uint8)
BoolScalar = make_scalar_like_type(np.bool_)

# U8Scalar rejects True / False
# BoolScalar accepts True / False
```

`make_scalar_like_type` is intentionally documented on backend modules such as `shapix.numpy`; it is not part of the lightweight root `shapix` import surface.

## Casting rules

Both `make_array_like_type` and `make_scalar_like_type` use NumPy casting semantics:

| Casting | Meaning | Example for target `float32` |
|---------|---------|------------------------------|
| `"no"` | Exact dtype only | only `float32` |
| `"equiv"` | Same kind and size | `float32` but not `float64` |
| `"safe"` | No information loss | `int16` yes, `float64` no |
| `"same_kind"` | Same-kind conversion | `int32` yes, `complex64` no |
| `"unsafe"` | Any cast NumPy allows | very permissive |

## Default used by built-in `Like` aliases

The built-in aliases such as `F32Like`, `I64Like`, `IntLike`, `FloatLike`, and `NumLike` all use:

```python
casting="same_kind"
```

That is the library default for `make_array_like_type(...)`, and it is the behavior you get unless you build a custom alias yourself.

If you need stricter or looser input acceptance, make a custom alias instead of relying on the built-ins:

```python
from shapix import make_array_like_type
from shapix._dtypes import FLOAT32

F32Exact = make_array_like_type(FLOAT32, casting="no", name="F32Exact")
F32Unsafe = make_array_like_type(FLOAT32, casting="unsafe", name="F32Unsafe")
```

## `ArrayLike` template

`shapix.numpy.ArrayLike` is a public recursive type alias for custom static typing combinations:

```python
import numpy as np
from shapix.numpy import ArrayLike

type MyInputType = ArrayLike[float, np.float32]
```

That template is most useful when you want your own checker-friendly alias but still follow shapix's "scalar or nested sequence or array" model.
