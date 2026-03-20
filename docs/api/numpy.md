---
title: "shapix.numpy"
description: NumPy array types, scalar Like types, and ArrayLike.
---

# `shapix.numpy`

NumPy array types with dtype and shape checking. Base type: `np.ndarray`.

```python
from shapix.numpy import F32, I64, Bool, Shaped, Float, DT64, Structured
from shapix.numpy import F32Like, IntLike, F32ScalarLike
from shapix.numpy import make_scalar_like_type, ArrayLike
```

---

## Array Types

All array types are subscriptable with [dimension symbols](../features/dimensions.md). At runtime they produce `Annotated[np.ndarray, Is[checker]]`.

### Concrete dtypes

| Type | Dtype | Example |
|------|-------|---------|
| `Bool` | `bool` | `Bool[N]` |
| `I8` | `int8` | `I8[N, C]` |
| `I16` | `int16` | `I16[N]` |
| `I32` | `int32` | `I32[N, C]` |
| `I64` | `int64` | `I64[N]` |
| `U8` | `uint8` | `U8[H, W]` |
| `U16` | `uint16` | `U16[N]` |
| `U32` | `uint32` | `U32[N]` |
| `U64` | `uint64` | `U64[N]` |
| `F16` | `float16` | `F16[N, C]` |
| `F32` | `float32` | `F32[N, C, H, W]` |
| `F64` | `float64` | `F64[N]` |
| `F128` | `float128` / `longdouble` | `F128[N]` |
| `C64` | `complex64` | `C64[N]` |
| `C128` | `complex128` | `C128[N]` |
| `C256` | `complex256` / `clongdouble` | `C256[N]` |
| `V` | `void` | `V[N]` |
| `Str` | `string` | `Str[N]` |
| `Bytes` | `bytes` | `Bytes[N]` |
| `Obj` | `object` | `Obj[N]` |
| `DT64` | `datetime64` | `DT64[N]` |
| `TD64` | `timedelta64` | `TD64[N]` |

### Category dtypes

| Type | Includes |
|------|----------|
| `Int` | `int8`, `int16`, `int32`, `int64` |
| `UInt` | `uint8`, `uint16`, `uint32`, `uint64` |
| `Integer` | `Int` + `UInt` |
| `Float` | `bfloat16`, `float16`, `float32`, `float64` |
| `Real` | `Integer` + `Float` |
| `Complex` | `complex64`, `complex128` |
| `Inexact` | `Float` + `Complex` |
| `Num` | `Integer` + `Float` + `Complex` |
| `Shaped` | Any dtype (shape-only checking) |

---

## Like Types (Array-Like)

Subscriptable types accepting scalars, arrays, or nested sequences. Must be subscripted with shape dims or `[...]`.

**Concrete:** `BoolLike`, `I8Like`–`I64Like`, `U8Like`–`U64Like`, `F16Like`–`F128Like`, `C64Like`–`C256Like`

**Category:** `IntLike`, `UIntLike`, `IntegerLike`, `FloatLike`, `RealLike`, `ComplexLike`, `InexactLike`, `NumLike`, `ShapedLike`

---

## ScalarLike Types

Range-validated scalar types. Numeric scalars reject `bool` values.

**Concrete:** `BoolScalarLike`, `I8ScalarLike`–`I64ScalarLike`, `U8ScalarLike`–`U64ScalarLike`, `F16ScalarLike`–`F128ScalarLike`, `C64ScalarLike`, `C128ScalarLike`, `C256ScalarLike`

**Category:** `IntScalarLike`, `UIntScalarLike`, `IntegerScalarLike`, `FloatScalarLike`, `RealScalarLike`, `ComplexScalarLike`, `InexactScalarLike`, `NumScalarLike`, `ShapedScalarLike`

**Other:** `StringLike` (`str | np.str_`)

---

## `make_scalar_like_type`

```python
def make_scalar_like_type(
    target_dtype: np.dtype,
    *,
    casting: str = "same_kind",
    name: str = "ScalarLike",
) -> type:
    """Create a casting-aware scalar type using np.can_cast at runtime."""
```

---

## `Structured`

```python
def Structured(fields) -> _ArrayFactory:
    """Create a subscriptable array type for NumPy structured/record dtypes.

    Parameters
    ----------
    fields
        NumPy dtype fields, e.g. [("x", np.float32), ("y", np.float32)].
    """
```

---

## `ArrayLike`

```python
type ArrayLike[_Scalar, _Array] = _Scalar | _Array | Sequence[ArrayLike[_Scalar, _Array]]
```

PEP 695 recursive type alias — accepts scalars, arrays, or nested sequences of any depth.

```python
from shapix.numpy import ArrayLike

type MyInput = ArrayLike[float, np.ndarray]
# float | np.ndarray | Sequence[float | np.ndarray | Sequence[...]]
```
