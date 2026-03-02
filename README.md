# Shapix

Elegant runtime shape and dtype checking for NumPy, JAX, and PyTorch arrays — powered by [beartype](https://github.com/beartype/beartype).

```python
from beartype import beartype
from shapix import N, C, H, W
from shapix.numpy import Float32Array

@beartype
def conv2d(x: Float32Array[N, C, H, W], weight: Float32Array[C, C, 3, 3]) -> Float32Array[N, C, H, W]:
    ...
```

Shapix turns array shape annotations into **Python objects** that beartype validates at runtime. Dimensions like `N` and `C` are checked for consistency across all parameters — if `x` has batch size 8, then `weight` must agree, automatically.

## Features

- **Zero boilerplate** — works with standard `@beartype` decorators and `beartype.claw` import hooks. No custom decorator required.
- **Cross-argument consistency** — named dimensions are enforced across all parameters and the return value within a single function call.
- **Static type checker friendly** — under `TYPE_CHECKING`, array types resolve to proper `NDArray` / `Array` / `Tensor` aliases. Pyright sees real types.
- **Readable annotations** — `Float32Array[N, C, H, W]` reads like documentation.
- **Full `BeartypeConf` support** — unlike jaxtyping, shapix doesn't replace your beartype configuration.
- **Thread-safe** — each thread gets independent dimension bindings.
- **Multi-backend** — NumPy, JAX, and PyTorch out of the box, plus a factory for custom array types.

## Installation

```bash
pip install shapix
```

Optional backends:

```bash
pip install shapix[jax]    # JAX support
pip install shapix[torch]  # PyTorch support
```

## Quick start

### Basic shape and dtype checking

```python
import numpy as np
from beartype import beartype
from shapix import N, C
from shapix.numpy import Float32Array

@beartype
def normalize(x: Float32Array[N, C]) -> Float32Array[N, C]:
    return x / x.sum(axis=1, keepdims=True)

normalize(np.ones((4, 3), dtype=np.float32))   # OK
normalize(np.ones((4, 3), dtype=np.float64))   # Raises — wrong dtype
normalize(np.ones((4,), dtype=np.float32))     # Raises — wrong rank
```

### Cross-argument consistency

Named dimensions are tracked within each function call. If `N` is bound to 4 by the first argument, all subsequent arguments must agree:

```python
@beartype
def add(x: Float32Array[N, C], y: Float32Array[N, C]) -> Float32Array[N, C]:
    return x + y

add(np.ones((4, 3), dtype=np.float32),
    np.ones((4, 3), dtype=np.float32))   # OK — N=4, C=3 in both

add(np.ones((4, 3), dtype=np.float32),
    np.ones((5, 3), dtype=np.float32))   # Raises — N=4 vs N=5
```

### Sequential calls are independent

Each function invocation gets a fresh set of dimension bindings:

```python
@beartype
def f(x: Float32Array[N]) -> Float32Array[N]:
    return x

f(np.ones((3,), dtype=np.float32))    # N=3
f(np.ones((100,), dtype=np.float32))  # N=100 — no conflict with previous call
```

## Dimension symbols

Shapix provides pre-defined dimension symbols for common use cases. You can also create your own.

### Named dimensions

Bind to a size on first occurrence and enforce consistency on subsequent ones.

| Symbol | Typical use |
|--------|-------------|
| `N` | Batch size, count |
| `B` | Batch |
| `C` | Channels |
| `H` | Height |
| `W` | Width |
| `L` | Sequence length |
| `T` | Time / tree |
| `P` | Points / parameters |

```python
from shapix import N, C, H, W

@beartype
def forward(x: Float32Array[N, C, H, W]) -> Float32Array[N, C, H, W]:
    ...
```

### Fixed dimensions

Use plain integers for dimensions that must match an exact size:

```python
@beartype
def rgb_to_gray(x: Float32Array[N, 3, H, W]) -> Float32Array[N, 1, H, W]:
    ...
```

### Symbolic dimensions

Dimensions support arithmetic. Expressions are evaluated against bound dimension values:

```python
from shapix import N, C

@beartype
def pad(x: Float32Array[N]) -> Float32Array[N + 2]:
    ...

@beartype
def flatten(x: Float32Array[N, C]) -> Float32Array[N * C]:
    return x.reshape(-1)
```

Supported operators: `+`, `-`, `*`, `/`, `//`, `**`, `%`.

### Variadic dimensions

Match zero or more contiguous dimensions. Prefixed with `s` (for "star"):

```python
from shapix import sB, C
from shapix.numpy import Float32Array

@beartype
def normalize(x: Float32Array[sB, C]) -> Float32Array[sB, C]:
    return x / x.sum(axis=-1, keepdims=True)

normalize(np.ones((3,), dtype=np.float32))        # *B=(),     C=3
normalize(np.ones((4, 3), dtype=np.float32))       # *B=(4,),   C=3
normalize(np.ones((2, 4, 3), dtype=np.float32))    # *B=(2,4),  C=3
```

Named variadic dimensions enforce cross-argument consistency on the matched shape:

```python
@beartype
def add(x: Float32Array[sB, C], y: Float32Array[sB, C]) -> Float32Array[sB, C]:
    return x + y
```

Use `Any` (an anonymous variadic) when you don't need consistency:

```python
from shapix import Any, C

@beartype
def last_dim(x: Float32Array[Any, C]) -> Float32Array[Any, C]:
    return x
```

### Broadcastable dimensions

Prefixed with `h` (for "hash"). Size 1 always matches, regardless of the bound value:

```python
from shapix import N, C, hN

@beartype
def broadcast_add(x: Float32Array[N, C], y: Float32Array[hN, C]) -> Float32Array[N, C]:
    return x + y

broadcast_add(np.ones((4, 3), dtype=np.float32),
              np.ones((1, 3), dtype=np.float32))   # OK — hN allows size 1
```

### Anonymous dimensions

Prefixed with `_`. Match any size without binding — no cross-argument consistency:

```python
from shapix import _N, C, __

@beartype
def f(x: Float32Array[_N, C]) -> Float32Array[_N, C]:
    return x
# _N matches anything, only C is cross-checked

@beartype
def g(x: Float32Array[__, __]) -> Float32Array[__, __]:
    return x
# Both dims unchecked
```

### Custom dimensions

Create your own with `Dimension`:

```python
from shapix import Dimension

Vocab = Dimension("Vocab")
Embed = Dimension("Embed")
Seq = Dimension("Seq")

@beartype
def embed(tokens: Int64Array[N, Seq], table: Float32Array[Vocab, Embed]) -> Float32Array[N, Seq, Embed]:
    ...
```

### Summary

| Prefix | Meaning | Example | Behavior |
|--------|---------|---------|----------|
| *(none)* | Named | `N` | Bind & enforce |
| `int` | Fixed | `3` | Exact match |
| `s` | Variadic | `sB` | Zero or more dims |
| `h` | Broadcastable | `hN` | Size 1 always OK |
| `_` | Anonymous | `_N`, `__` | Match any, no binding |
| `...` | Any variadic | `Any` | Zero or more, no binding |
| arithmetic | Symbolic | `N + 1` | Expression |

## Array types

### NumPy

```python
from shapix.numpy import Float32Array, Int64Array, ShapedArray  # and many more
```

**Concrete dtypes:** `BoolArray`, `Int8Array`, `Int16Array`, `Int32Array`, `Int64Array`, `UInt8Array`, `UInt16Array`, `UInt32Array`, `UInt64Array`, `Float16Array`, `Float32Array`, `Float64Array`, `Complex64Array`, `Complex128Array`

**Category dtypes:** `IntArray` (signed), `UIntArray` (unsigned), `IntegerArray` (all int), `FloatArray`, `RealArray` (int + float), `ComplexArray`, `InexactArray` (float + complex), `NumArray` (all numeric), `ShapedArray` (any dtype)

### JAX

```python
from shapix.jax import Float32Array, BFloat16Array
```

Same type names as NumPy, plus `BFloat16Array`. Base type is `jax.Array`.

### PyTorch

```python
from shapix.torch import Float32Tensor, BFloat16Tensor
```

Same dtype variants, but suffixed with `Tensor` instead of `Array`. Base type is `torch.Tensor`.

### Custom array types

Use `make_array_type` to create types for any array-like class:

```python
from shapix import make_array_type
from shapix._dtypes import FLOAT32, DtypeSpec

# For a custom array class with .dtype and .shape attributes:
MyFloat32Array = make_array_type(MyArrayClass, FLOAT32)

# Or define your own dtype spec:
BF16_OR_F32 = DtypeSpec("BF16orF32", frozenset({"bfloat16", "float32"}))
MixedArray = make_array_type(np.ndarray, BF16_OR_F32)
```

## Advanced usage

### Package-wide instrumentation with `beartype.claw`

Instead of decorating each function with `@beartype`, you can instrument an entire package:

```python
# In your_package/__init__.py
from beartype.claw import beartype_this_package
beartype_this_package()

# Or via shapix's wrapper:
from shapix.claw import shapix_this_package
shapix_this_package()
```

Every function in your package that uses shapix type annotations will be checked automatically.

### Explicit memo management with `@shapix.check`

The frame-based memo works automatically with `@beartype` in virtually all cases. For exotic call-stack scenarios, or to combine memo management with custom `BeartypeConf`, use the explicit decorator:

```python
import shapix
from beartype import beartype

# Option 1: Memo only — pair with @beartype
@shapix.check
@beartype
def f(x: Float32Array[N, C]) -> Float32Array[N, C]:
    ...

# Option 2: Memo + beartype combined with custom config
from beartype._conf.confmain import BeartypeConf

@shapix.check(conf=BeartypeConf())
def f(x: Float32Array[N, C]) -> Float32Array[N, C]:
    ...
```

### Manual checks with `check_context`

For `isinstance`-style checks outside of decorated functions, use `check_context` with beartype's `is_bearable`:

```python
from beartype.door import is_bearable
import shapix
from shapix import N, C
from shapix.numpy import Float32Array

with shapix.check_context():
    assert is_bearable(x, Float32Array[N, C])  # Binds N and C
    assert is_bearable(y, Float32Array[N, C])  # Must match same N, C
```

## How it works

Shapix uses three key mechanisms:

1. **`Annotated[T, Is[validator]]`** — Each array type annotation (e.g., `Float32Array[N, C]`) produces a `typing.Annotated` type with a beartype `Is[...]` validator. This lets beartype handle all the dispatch natively.

2. **Frame-based memo management** — beartype's `Is[validator]` call stack is deterministic: `validator → _is_valid_bool → beartype_wrapper`. All parameter checks for one function call share the same wrapper frame. Shapix identifies this frame via `sys._getframe()` and associates a dimension memo (name → size bindings) with it. This is how cross-argument consistency works with zero boilerplate.

3. **Thread-local storage** — Each thread gets its own memo stack via `threading.local()`, ensuring thread safety.

## Compared to jaxtyping

| | jaxtyping | shapix |
|---|---|---|
| Decorator | Custom `@jaxtyped` replaces `@beartype` | Standard `@beartype` |
| Shape syntax | String-based: `"batch channels"` | Python objects: `N, C` |
| BeartypeConf | Not supported (decorator conflict) | Fully supported |
| Type checker | Metaclass magic (confuses pyright) | `Annotated` aliases (clean) |
| Backends | NumPy, JAX | NumPy, JAX, PyTorch |
| Dependencies | jaxtyping + beartype | beartype only |
| Custom decorator | Required | Not required |

## License

MIT
