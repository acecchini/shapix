---
title: "shapix"
description: Core module API reference.
---

# `shapix`

The core module exports dimension symbols, the decorator system, tree annotations, and the array type factory.

```python
import shapix
from shapix import N, B, C, D, K, H, W, L, P, __, Scalar
from shapix import Dimension, Value, DtypeSpec
from shapix import check, check_context
from shapix import T, S, Structure
from shapix import make_array_type, make_array_like_type
```

---

## Dimension Symbols

### Pre-defined dimensions

| Symbol | Value | Typical use |
|--------|-------|-------------|
| `N` | `Dimension("N")` | Batch size, count |
| `B` | `Dimension("B")` | Batch |
| `C` | `Dimension("C")` | Channels |
| `D` | `Dimension("D")` | Embedding dimension |
| `K` | `Dimension("K")` | Number of heads |
| `H` | `Dimension("H")` | Height |
| `W` | `Dimension("W")` | Width |
| `L` | `Dimension("L")` | Sequence length |
| `P` | `Dimension("P")` | Points / parameters |
| `__` | `Dimension("__")` | Anonymous (match any single dim) |
| `Scalar` | `Dimension("")` | No dimensions (scalar array) |

---

## `Dimension`

::: shapix.Dimension

```python
class Dimension(str):
    """A named dimension symbol that doubles as a shape spec element.

    Behaves like a str for display but carries semantic meaning for
    runtime shape checking. Arithmetic produces SymbolicDim-backed
    expressions.
    """
```

### Creating dimensions

```python
from shapix import Dimension

Vocab = Dimension("Vocab")
Embed = Dimension("Embed")
```

### Operators

| Operator | Example | Result |
|----------|---------|--------|
| `~` (invert) | `~N` | Variadic: match zero or more dims |
| `+` (pos) | `+N` | Broadcastable: size 1 always matches |
| `+` (add) | `N + 1` | Symbolic expression |
| `-` (sub) | `N - 1` | Symbolic expression |
| `*` (mul) | `N * C` | Symbolic expression |
| `/` (div) | `N / 2` | Symbolic expression |
| `//` (floordiv) | `N // 2` | Symbolic expression |
| `**` (pow) | `N ** 2` | Symbolic expression |
| `%` (mod) | `N % 2` | Symbolic expression |
| `-` (neg) | `-N` | Symbolic expression |

---

## `Value`

```python
class Value:
    """Runtime value expression for shape dimensions.

    Use when a shape depends on a runtime parameter or self attribute
    rather than a previously bound dimension.
    """
```

### Example

```python
from shapix import Value

Size = Value("size")
WidthPlus3 = Value("self.width + 3")

@beartype
def full(size: int) -> F32[Size]:
    return np.full((size,), 1.0, dtype=np.float32)
```

Supports `+Value(...)` for broadcastable, but rejects `~Value(...)` (variadic).

---

## `DtypeSpec`

```python
@dataclass(frozen=True, slots=True)
class DtypeSpec:
    """Describes a set of allowed dtypes by their canonical string names."""

    name: str
    allowed: frozenset[str]

    def matches(self, obj: object) -> bool:
        """Return True if obj's dtype is in the allowed set."""
```

### Example

```python
from shapix import DtypeSpec

BF16_OR_F32 = DtypeSpec("BF16orF32", frozenset({"bfloat16", "float32"}))
```

---

## `make_array_type`

```python
def make_array_type(array_type: type, dtype_spec: DtypeSpec) -> _ArrayFactory:
    """Create a subscriptable array type factory for a given base type and dtype.

    Parameters
    ----------
    array_type
        The base array class (e.g. np.ndarray, jax.Array, torch.Tensor,
        or any class with .dtype and .shape attributes).
    dtype_spec
        A DtypeSpec defining the allowed dtypes.

    Returns
    -------
    _ArrayFactory
        Subscripting it with dimensions produces Annotated[array_type, Is[checker]].
    """
```

### Example

```python
from shapix import make_array_type
from shapix._dtypes import FLOAT32

MyF32 = make_array_type(MyArrayClass, FLOAT32)

@beartype
def f(x: MyF32[N, C]) -> MyF32[N, C]:
    ...
```

---

## `make_array_like_type`

```python
def make_array_like_type(
    dtype_spec: DtypeSpec,
    *,
    casting: str = "same_kind",
    name: str = "ArrayLike",
) -> _ArrayFactory:
    """Create a subscriptable array-like type factory with configurable dtype casting.

    Parameters
    ----------
    dtype_spec
        A DtypeSpec defining the target dtypes.
    casting
        NumPy casting rule: "no", "equiv", "safe", "same_kind", "unsafe".
    name
        Display name for error messages.
    """
```

---

## `check`

```python
@overload
def check(fn: Callable[P, R], /) -> Callable[P, R]: ...
@overload
def check(*, conf: object = ...) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

def check(...):
    """Decorator that manages the dimension memo around a function call.

    Two usage modes:
    1. Memo only (pair with @beartype):
       @shapix.check
       @beartype
       def f(x: F32[N, C]) -> F32[N, C]: ...

    2. Memo + beartype combined (pass conf=):
       @shapix.check(conf=BeartypeConf(...))
       def f(x: F32[N, C]) -> F32[N, C]: ...
    """
```

---

## `check_context`

```python
class check_context:
    """Context manager for manual isinstance checks with shared memo.

    Usage:
        with shapix.check_context():
            assert is_bearable(x, F32[N, C])  # Binds N=4
            assert is_bearable(y, F32[N, C])  # Checks N=4
    """

    def __enter__(self) -> check_context: ...
    def __exit__(self, *_: object) -> None: ...
```

---

## `Structure`

```python
class Structure(str):
    """Named tree structure symbol for Tree annotations."""
```

### Pre-defined structures

| Symbol | Value |
|--------|-------|
| `T` | `Structure("T")` |
| `S` | `Structure("S")` |

### Custom structures

```python
from shapix import Structure

Params = Structure("Params")
State = Structure("State")
```
