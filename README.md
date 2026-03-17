# Shapix

Elegant runtime shape and dtype checking for NumPy, JAX, and PyTorch arrays — powered by [beartype](https://github.com/beartype/beartype).

```python
from beartype import beartype
from shapix import N, C, H, W
from shapix.numpy import F32

@beartype
def conv2d(x: F32[N, C, H, W], weight: F32[C, C, 3, 3]) -> F32[N, C, H, W]:
    ...
```

Shapix turns array shape annotations into **Python objects** that beartype validates at runtime. Dimensions like `N` and `C` are checked for consistency across all parameters — if `x` has batch size 8, then `weight` must agree, automatically.

## Features

- **Zero boilerplate** — works with standard `@beartype` decorators and `beartype.claw` import hooks. No custom decorator required.
- **Cross-argument consistency** — named dimensions are enforced across all parameters and the return value within a single function call.
- **Static type checker friendly** — under `TYPE_CHECKING`, array types resolve to proper `NDArray` / `Array` / `Tensor` aliases. Pyright sees real types.
- **Readable annotations** — `F32[N, C, H, W]` reads like documentation.
- **Full `BeartypeConf` support** — unlike jaxtyping, shapix doesn't replace your beartype configuration.
- **Thread-safe** — each thread gets independent dimension bindings.
- **Multi-backend** — NumPy, JAX, and PyTorch out of the box, plus a factory for custom array types.

## Installation

```bash
pip install shapix
```

Shapix has one dependency: [beartype](https://github.com/beartype/beartype). Install your preferred array framework separately:

Install optional dependencies alongside `shapix` with plain package names.
Avoid extras-style installs such as `shapix[numpy]` or `shapix[torch]` (shapix intentionally does not provide extras).

```bash
pip install shapix numpy          # NumPy
pip install shapix torch          # PyTorch
pip install shapix jax            # JAX
pip install shapix numpy optree   # NumPy + tree support (optree or jax)
```

For `shapix.jax` and `shapix.torch`, install `numpy` alongside the backend:

```bash
pip install shapix numpy jax
pip install shapix numpy torch
```

## Quick start

### Basic shape and dtype checking

```python
import numpy as np
from beartype import beartype
from shapix import N, C
from shapix.numpy import F32

@beartype
def normalize(x: F32[N, C]) -> F32[N, C]:
    return x / x.sum(axis=1, keepdims=True)

normalize(np.ones((4, 3), dtype=np.float32))   # OK
normalize(np.ones((4, 3), dtype=np.float64))   # Raises — wrong dtype
normalize(np.ones((4,), dtype=np.float32))     # Raises — wrong rank
```

### Cross-argument consistency

Named dimensions are tracked within each function call. If `N` is bound to 4 by the first argument, all subsequent arguments must agree:

```python
@beartype
def add(x: F32[N, C], y: F32[N, C]) -> F32[N, C]:
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
def f(x: F32[N]) -> F32[N]:
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
| `D` | Embedding dimension |
| `K` | Number of heads |
| `H` | Height |
| `W` | Width |
| `L` | Sequence length |
| `P` | Points / parameters |

```python
from shapix import N, C, H, W
from shapix.numpy import F32

@beartype
def forward(x: F32[N, C, H, W]) -> F32[N, C, H, W]:
    ...
```

### Fixed dimensions

Use plain integers for dimensions that must match an exact size:

```python
@beartype
def rgb_to_gray(x: F32[N, 3, H, W]) -> F32[N, 1, H, W]:
    ...
```

### Symbolic dimensions

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

Supported operators: `+`, `-`, `*`, `/`, `//`, `**`, `%`.

Symbolic dimensions are intentionally limited to arithmetic over dimension names
and numeric literals. Attribute access and function calls are rejected.

### Runtime value dimensions

Use `Value["expr"]` when a shape depends on a runtime parameter or `self`
attribute rather than a previously bound dimension:

```python
from beartype import beartype
from shapix import Value
from shapix.numpy import F32
import numpy as np

@beartype
def full(size: int) -> F32[Value["size"]]:
    return np.full((size,), 1.0, dtype=np.float32)

class SomeClass:
    width = 5

    @beartype
    def full(self) -> F32[Value["self.width + 3"]]:
        return np.full((self.width + 3,), 1.0, dtype=np.float32)
```

`Value[...]` uses a restricted arithmetic grammar as well. It allows names,
attribute access, numeric literals, and arithmetic operators, but rejects calls,
indexing, and other arbitrary Python expressions.

### Variadic dimensions

Apply `~` (tilde) to any dimension to make it **variadic** — matching zero or more contiguous dimensions:

```python
from shapix import B, C
from shapix.numpy import F32

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

Use `~__` (anonymous variadic) when you don't need consistency:

```python
from shapix import __, C

@beartype
def last_dim(x: F32[~__, C]) -> F32[~__, C]:
    return x
```

### Broadcastable dimensions

Apply `+` (unary plus) to any dimension to make it **broadcastable** — size 1 always matches, regardless of the bound value:

```python
from shapix import N, C

@beartype
def broadcast_add(x: F32[N, C], y: F32[+N, C]) -> F32[N, C]:
    return x + y

broadcast_add(np.ones((4, 3), dtype=np.float32),
              np.ones((1, 3), dtype=np.float32))   # OK — +N allows size 1
```

Broadcastable also works with variadic dimensions: `~+B` matches zero or more dims where each can be 1 or the bound value.

### Anonymous dimensions

`__` matches any single dimension without binding — no cross-argument consistency:

```python
from shapix import __, C

@beartype
def f(x: F32[__, C]) -> F32[__, C]:
    return x
# __ matches anything, only C is cross-checked
```

### Scalar dimension

`Scalar` represents a zero-dimensional array (a scalar array). Use it when a function accepts or returns a scalar:

```python
from shapix import Scalar
from shapix.numpy import F32

@beartype
def dot(x: F32[N], y: F32[N]) -> F32[Scalar]:
    return np.dot(x, y)  # returns shape ()
```

### Custom dimensions

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

To silence pyright/Pylance on custom dimensions, use the `TYPE_CHECKING` pattern:

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

### Summary

| Syntax | Meaning | Example | Behavior |
|--------|---------|---------|----------|
| *(none)* | Named | `N` | Bind & enforce |
| `int` | Fixed | `3` | Exact match |
| `~` | Variadic | `~B` | Zero or more dims |
| `+` | Broadcastable | `+N` | Size 1 always OK |
| `~+` | Broadcastable variadic | `~+B` | Variadic + broadcast |
| `__` | Anonymous | `__` | Match any, no binding |
| `~__` | Anonymous variadic | `~__` | Zero or more, no binding |
| `...` | Ellipsis (alias) | `...` | Same as `~__` |
| `Scalar` | Scalar | `Scalar` | Zero dimensions |
| arithmetic | Symbolic | `N + 1` | Expression |

## Array types

Concise names for fast, readable annotations:

### NumPy

```python
from shapix.numpy import F32, I64, Shaped  # and many more
```

**Concrete dtypes:** `Bool`, `I8`, `I16`, `I32`, `I64`, `U8`, `U16`, `U32`, `U64`, `F16`, `F32`, `F64`, `F128`, `C64`, `C128`, `C256`

**Category dtypes:** `Int` (signed), `UInt` (unsigned), `Integer` (all int), `Float`, `Real` (int + float), `Complex`, `Inexact` (float + complex), `Num` (all numeric), `Shaped` (any dtype)

**Additional dtypes:** `V` (void), `Str` (string), `Bytes` (bytes), `Obj` (object), `DT64` (datetime64), `TD64` (timedelta64)

### JAX

```python
from shapix.jax import F32, BF16
```

Most NumPy type names, plus `BF16` and `BF16Like`. Base type is `jax.Array`. JAX does not expose NumPy-only extended-precision array aliases such as `F128` / `C256`. Also exports `Like` types and `Tree`.

### PyTorch

```python
from shapix.torch import F32, BF16
```

Most NumPy type names, plus `BF16` and `BF16Like`. Base type is `torch.Tensor`. PyTorch does not expose NumPy-only extended-precision array aliases such as `F128` / `C256`. Also exports `Like` types.

### Endianness variants

For multi-byte dtypes, create endianness-constrained types programmatically using `make_array_type` with endianness `DtypeSpec` constants from `shapix._dtypes`:

```python
from shapix import make_array_type, N, C
from shapix._dtypes import FLOAT32_LE, INT64_BE, INT32_N

F32LE = make_array_type(np.ndarray, FLOAT32_LE)
I64BE = make_array_type(np.ndarray, INT64_BE)

@beartype
def process_le(x: F32LE[N, C]) -> F32LE[N, C]:
    """Only accepts little-endian float32 arrays."""
    return x

process_le(np.ones((4, 3), dtype="<f4"))   # OK — little-endian
process_le(np.ones((4, 3), dtype=">f4"))   # Raises — big-endian
```

Available endianness specs for all multi-byte types: `INT16_LE`/`INT16_BE`/`INT16_N`, `INT32_LE`/`INT32_BE`/`INT32_N`, etc., plus category groups (`INT_LE`, `FLOAT_BE`, `REAL_N`, `SHAPED_LE`, etc.).

Single-byte types (`Bool`, `I8`, `U8`) have no endianness variants (byte order is irrelevant).

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
process_points(pts)   # OK

wrong = np.zeros(10, dtype=[("a", np.float32), ("b", np.float32)])
process_points(wrong)  # Raises — field names don't match
```

## Like types (input validation)

`Like` types accept scalars, nested sequences of any depth, or arrays — use them for function inputs that will be converted:

```python
from shapix.numpy import F32Like

@beartype
def to_array(x: F32Like[...]) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)

to_array(3.14)                          # scalar
to_array([1.0, 2.0, 3.0])              # 1D list
to_array([[1.0, 2.0], [3.0, 4.0]])     # 2D nested list
to_array(np.ones((3, 4)))              # ndarray
to_array([[[[[[1.0]]]]]])              # 6D+ — no depth limit
```

Like types **must be subscripted** — use `[...]` (Ellipsis) to accept any shape, or `[N, C]` to enforce specific dimensions:

```python
@beartype
def process(x: F32Like[N, C]) -> F32[N, C]:
    return np.asarray(x, dtype=np.float32)
```

Dtype compatibility uses NumPy's `same_kind` casting rules by default: `int32` can be passed where `float32` is expected (safe upcast), but `complex128` cannot.

**Available:** `BoolLike`, `I8Like`, `I16Like`, `I32Like`, `I64Like`, `U8Like`–`U64Like`, `F16Like`, `F32Like`, `F64Like`, `F128Like`, `C64Like`, `C128Like`, `C256Like`, plus category aliases `IntLike`, `FloatLike`, `NumLike`, `ShapedLike`, etc.

Like types are also available in JAX and PyTorch backends:

```python
from shapix.jax import F32Like     # accepts jax.Array, ndarray, scalars, sequences
from shapix.torch import F32Like   # accepts Tensor, ndarray, scalars, sequences
```

### ScalarLike types (range-validated scalars)

ScalarLike types validate individual scalar values with range checking — no shape, just value:

```python
from shapix.numpy import I8ScalarLike, F32ScalarLike, U8ScalarLike

@beartype
def clamp_pixel(value: U8ScalarLike) -> int:
    """Accepts int in [0, 255] range."""
    return int(value)

clamp_pixel(128)    # OK
clamp_pixel(256)    # Raises — out of uint8 range
clamp_pixel(-1)     # Raises — negative not allowed for unsigned
```

**Available:** `BoolScalarLike`, `I8ScalarLike`–`I64ScalarLike`, `U8ScalarLike`–`U64ScalarLike`, `F16ScalarLike`–`F128ScalarLike`, `C64ScalarLike`, `C128ScalarLike`, `C256ScalarLike`, plus category aliases `IntScalarLike`, `FloatScalarLike`, `NumScalarLike`, etc.

Also: `StringLike` (str | np.str_).

ScalarLike types are available from all backends:

```python
from shapix.numpy import U8ScalarLike   # defined here
from shapix.jax import U8ScalarLike     # re-exported
from shapix.torch import U8ScalarLike   # re-exported
```

For custom casting rules, use `make_scalar_like_type`:

```python
from shapix.numpy import make_scalar_like_type

F32ScalarStrict = make_scalar_like_type(np.float32, casting="no")   # exact np.float32 only
F32ScalarSafe = make_scalar_like_type(np.float32, casting="safe")   # float16 OK, complex rejected
```

The `ArrayLike` template is also public for custom static type combinations:

```python
from shapix.numpy import ArrayLike

type MyInputType = ArrayLike[float, np.float32]
```

### Casting rules

The `casting` parameter on `make_array_like_type` and `make_scalar_like_type` controls dtype strictness using NumPy casting rules:

| Casting | Meaning | Example (target=float32) |
|---------|---------|--------------------------|
| `"no"` | Exact dtype only | Only `float32` accepted |
| `"equiv"` | Same kind, same size | `float32` only (no size change) |
| `"safe"` | No data loss | `int16` OK, `float64` rejected |
| `"same_kind"` | Same kind allowed | `int32` OK, `complex64` rejected |
| `"unsafe"` | Any cast | `complex128` OK, strings rejected |

```python
from shapix import make_array_like_type
from shapix._dtypes import FLOAT32
from shapix.numpy import make_scalar_like_type

# Strict: only exact float32 arrays
F32Strict = make_array_like_type(FLOAT32, casting="no", name="F32Strict")

# Permissive: accept anything castable
F32Unsafe = make_array_like_type(FLOAT32, casting="unsafe", name="F32Unsafe")

# Scalar with safe casting
F32ScalarSafe = make_scalar_like_type(np.float32, casting="safe")
```

### Custom array types and Like types

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

The `casting` parameter controls dtype strictness using NumPy casting rules: `"no"` < `"equiv"` < `"safe"` < `"same_kind"` (default) < `"unsafe"`.

## Tree annotations

Tree annotations validate all leaves in a nested structure (dicts, lists, tuples, namedtuples). Import `Tree` from an explicit backend module:

```python
from shapix.optree import Tree   # backed by optree
from shapix.jax import Tree      # backed by jax.tree_util
```

### Basic leaf checking

```python
from shapix import T, S, N, C
from shapix.optree import Tree  # or: from shapix.jax import Tree
from shapix.numpy import F32

@beartype
def process(data: Tree[F32[N, C]]) -> Tree[F32[N, C]]:
    ...

# All leaves must be F32 arrays with consistent N and C
process({"params": np.ones((3, 4), dtype=np.float32),
         "state": np.ones((3, 4), dtype=np.float32)})
```

### Structure binding

Named structure symbols (`T`, `S`) enforce that multiple arguments share identical tree shapes:

```python
@beartype
def add_trees(x: Tree[F32[N], T], y: Tree[F32[N], T]) -> Tree[F32[N]]:
    ...

add_trees({"a": x1, "b": x2}, {"a": y1, "b": y2})  # OK — same structure
add_trees({"a": x1}, [y1, y2])                       # Raises — different structure
```

### Multi-level structure matching

Structure names are listed left-to-right from outer to inner. Without `...`, each name except the last captures ONE level; the last captures the full remaining structure. Trailing `...` makes all names one-level-only with inner levels unchecked. Leading `...` matches names from the bottom up.

```python
# T = full tree structure (all levels)
def f(x: Tree[F32[N], T], y: Tree[F32[N], T]): ...

# T = top-level only, subtrees are arbitrary
def f(x: Tree[F32[N], T, ...], y: Tree[F32[N], T, ...]): ...

# T = bottom-level only (leaf-adjacent container)
def f(x: Tree[F32[N], ..., T], y: Tree[F32[N], ..., T]): ...

# T = top level, S = full remaining structure below
def f(x: Tree[int, T], y: Tree[int, S], z: Tree[int, T, S]): ...

# T = top, S = next, inner levels unchecked
def f(x: Tree[F32[N], T, S, ...]): ...

# S = bottom, T = second-from-bottom
def f(x: Tree[F32[N], ..., T, S]): ...
```

### Custom structure symbols

Create your own with `Structure`:

```python
from shapix import Structure

Params = Structure("Params")
State = Structure("State")

@beartype
def train(params: Tree[F32[N], Params], state: Tree[I64[N], State]): ...
```

## Advanced usage

### `from __future__ import annotations` (PEP 563)

Shapix is fully compatible with `from __future__ import annotations`. The library itself uses it in every source file.

The one rule: **every dimension symbol used in an annotation must be imported in the module scope.** This is true with or without the future import — the difference is only the error you get if you forget:

```python
from __future__ import annotations
from shapix import C           # B is NOT imported
from shapix.numpy import F32

@beartype
def f(x: F32[~B, C]): ...     # BeartypeDecorHintForwardRefException — B is not in scope
```

Fix: import `B`:

```python
from shapix import B, C
```

This applies to all dimension symbols — named (`N`, `B`), custom (`Vocab = Dimension("Vocab")`), and any symbol used with operators (`~B`, `+N`). The operators (`~`, `+`) are evaluated on the imported object, so the base symbol must be available.

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

### How cross-argument checking works (the memo)

A **dimension memo** maps dimension names to sizes (e.g., `N->4`, `C->3`). Each function call gets a fresh memo. All parameter checks within that call share the same memo — that's how shapix knows `N=4` in `x` must match `N=4` in `y`.

This happens **automatically** with `@beartype`. Shapix detects the beartype wrapper frame via `sys._getframe()` and associates a memo with it. No extra decorator needed.

### `@shapix.check` — explicit memo management

To understand `@shapix.check`, you need to understand the problem it solves.

**The problem: sharing state across parameter checks.** When beartype validates `f(x, y)`, it checks `x` and `y` independently — it calls the `Is[...]` validator once per parameter. But shapix needs all those validators to share the same dimension memo, so that `N=4` bound by `x` is enforced on `y`. Something has to connect them.

**The automatic approach** (no extra decorator needed) is frame-based detection. Shapix walks up the call stack with `sys._getframe()` to find the beartype wrapper frame. Since all parameter checks within one `f(x, y)` call share the same wrapper frame, shapix can key a memo to it. This just works:

```python
@beartype  # Shapix auto-detects this frame — nothing else needed
def f(x: F32[N, C], y: F32[N, C]) -> F32[N, C]: ...
```

**`@shapix.check`** takes a different approach: instead of detecting the frame, it explicitly pushes a fresh memo onto a stack before the call and pops it after. All validators see this explicit memo first (it takes priority over frame detection):

```python
@shapix.check   # Pushes memo before call, pops after
@beartype        # Validates parameters using that memo
def f(x: F32[N, C], y: F32[N, C]) -> F32[N, C]: ...
```

**Combined mode** — `@shapix.check` can also apply `@beartype` for you with a custom `BeartypeConf`:

```python
from beartype import BeartypeConf

@shapix.check(conf=BeartypeConf())   # Applies @beartype + memo management
def f(x: F32[N, C], y: F32[N, C]) -> F32[N, C]: ...
```

Both approaches produce identical results in normal usage. You only need `@shapix.check` in specific situations:

#### 1. Extra decorators between beartype and the call site

Frame-based detection counts a fixed number of frames up from the validator. If something adds extra frames between beartype's wrapper and the actual function call, the detection can land on the wrong frame:

```python
# This works — beartype is the outermost wrapper, frame detection is stable
@beartype
def f(x: F32[N, C], y: F32[N, C]) -> F32[N, C]: ...

# This might not — a middleware decorator adds extra frames
@some_middleware   # Adds frames between beartype's wrapper and the caller
@beartype
def f(x: F32[N, C], y: F32[N, C]) -> F32[N, C]: ...

# Fix: @shapix.check bypasses frame detection entirely
@some_middleware
@shapix.check
@beartype
def f(x: F32[N, C], y: F32[N, C]) -> F32[N, C]: ...
```

#### 2. Defensive coding

If you want a guarantee that cross-argument checking works regardless of how your code is called (by test runners, async frameworks, deep middleware stacks), `@shapix.check` removes all dependence on call-stack structure.

**When you don't need it:** If you're using plain `@beartype` and your tests pass, the frame-based detection is working. Most applications never need `@shapix.check`.

### Manual checks with `check_context`

For `isinstance`-style checks outside of decorated functions, use `check_context` with beartype's `is_bearable`. Without it, each check gets an independent memo — dimensions aren't cross-checked:

```python
from beartype.door import is_bearable
import shapix
from shapix import N, C
from shapix.numpy import F32

# Without check_context — each check is independent, N is NOT cross-checked
is_bearable(x, F32[N, C])  # Binds N=4 in a temporary memo (discarded)
is_bearable(y, F32[N, C])  # Binds N=999 in a new memo — no error!

# With check_context — checks share a memo, N IS cross-checked
with shapix.check_context():
    assert is_bearable(x, F32[N, C])  # Binds N=4
    assert is_bearable(y, F32[N, C])  # Checks N=4 — raises if y has N=999
```

## How it works

Shapix uses three key mechanisms:

1. **`Annotated[T, Is[validator]]`** — Each array type annotation (e.g., `F32[N, C]`) produces a `typing.Annotated` type with a beartype `Is[...]` validator. This lets beartype handle all the dispatch natively.

2. **Frame-based memo** — beartype's `Is[validator]` call stack is deterministic: `validator -> _is_valid_bool -> beartype_wrapper`. All parameter checks for one function call share the same wrapper frame. Shapix identifies this frame via `sys._getframe()` and associates a dimension memo (name -> size bindings) with it. This is how cross-argument consistency works with zero boilerplate.

3. **Thread-local storage** — Each thread gets its own memo stack via `threading.local()`, ensuring thread safety.

## Static type checkers (pyright / Pylance)

Shapix's pre-defined dimension symbols (`N`, `C`, `H`, `W`, ...) work out of the box with pyright and Pylance — under `TYPE_CHECKING` they resolve to `int` type aliases, so annotations like `F32[N, C]` are fully valid type expressions.

However, some patterns produce type checker errors because they place **runtime values** where pyright expects **types**:

| Pattern | Example | Pyright rule |
|---------|---------|--------------|
| Integer literals | `F32[N, 3, H, W]` | `reportGeneralTypeIssues` |
| Unary operators | `F32[~B, C]`, `F32[+N, C]` | `reportInvalidTypeForm` |
| Arithmetic | `F32[N + 2]` | `reportInvalidTypeForm` |
| Custom dimensions | `F32[Vocab, Embed]` | `reportInvalidTypeForm` (or use `TYPE_CHECKING` pattern) |

### Option A — Suppress `reportInvalidTypeForm` and wrap integers (recommended)

Add one line to your pyright config to silence operators, custom dimensions, and arithmetic globally. Then wrap bare integer literals in `Dimension()` to shift them to the same suppressed rule:

```jsonc
// pyrightconfig.json (recommended — Pylance always reads this)
{ "reportInvalidTypeForm": false }
```

Or equivalently in `pyproject.toml`:

```toml
[tool.pyright]
reportInvalidTypeForm = false
```

```python
from shapix import N, H, W, Dimension

# Integer literals — wrap in Dimension() to avoid reportGeneralTypeIssues
@beartype
def rgb_to_gray(x: F32[N, Dimension("3"), H, W]) -> F32[N, Dimension("1"), H, W]:
    ...

# Operators, arithmetic, custom dimensions — all covered by the suppression
@beartype
def pad(x: F32[N]) -> F32[N + 2]:
    ...
```

### Option B — Inline `# type: ignore`

If you prefer not to change your pyright config, silence individual lines:

```python
@beartype
def rgb_to_gray(x: F32[N, 3, H, W]) -> F32[N, 1, H, W]:  # type: ignore[reportGeneralTypeIssues]
    ...

@beartype
def f(x: F32[~B, C]) -> F32[~B, C]:  # type: ignore[reportInvalidTypeForm]
    ...
```

## Compared to jaxtyping

| | jaxtyping | shapix |
|---|---|---|
| Decorator | Custom `@jaxtyped` replaces `@beartype` | Standard `@beartype` |
| Shape syntax | String-based: `"batch channels"` | Python objects: `N, C` |
| BeartypeConf | Not supported (decorator conflict) | Fully supported |
| Type checker | Metaclass magic (confuses pyright) | `Annotated` aliases (clean) |
| Backends | NumPy, JAX | NumPy, JAX, PyTorch |
| Tree | Built-in with structure binding | Built-in with structure binding (via optree) |
| Dependencies | jaxtyping + beartype | beartype only |
| Custom decorator | Required | Not required |
| Endianness | Not supported | Programmatic LE/BE/N variants |
| Structured dtypes | Not supported | `Structured()` helper |
| ArrayLike | Not supported | `F32Like`, `IntLike`, etc. |
| ScalarLike | Not supported | Range-validated scalar types |

## Complete API reference

### Dimension symbols (`shapix`)

`N`, `B`, `C`, `D`, `K`, `H`, `W`, `L`, `P` — named dimensions
`__` — anonymous dimension
`Scalar` — scalar (zero dimensions)
`T`, `S` — tree structure symbols
`Dimension("Name")` — custom dimension
`Structure("Name")` — custom structure symbol

### Array types (`shapix.numpy`)

**Concrete:** `Bool`, `I8`, `I16`, `I32`, `I64`, `U8`, `U16`, `U32`, `U64`, `F16`, `F32`, `F64`, `C64`, `C128`
**Categories:** `Int`, `UInt`, `Integer`, `Float`, `Real`, `Complex`, `Inexact`, `Num`, `Shaped`
**Other:** `V`, `Str`, `Bytes`, `Obj`, `DT64`, `TD64`
**Endianness:** Programmatic via `make_array_type(np.ndarray, FLOAT32_LE)` using `DtypeSpec` constants

### Like types (`shapix.numpy`)

**ArrayLike:** `BoolLike`, `I8Like`–`I64Like`, `U8Like`–`U64Like`, `F16Like`–`F128Like`, `C64Like`, `C128Like`, `C256Like`, `IntLike`, `FloatLike`, `NumLike`, `ShapedLike`, etc.
**ScalarLike:** `BoolScalarLike`, `I8ScalarLike`–`I64ScalarLike`, `U8ScalarLike`–`U64ScalarLike`, `F16ScalarLike`–`F128ScalarLike`, `C64ScalarLike`, `C128ScalarLike`, `C256ScalarLike`, `IntScalarLike`, `FloatScalarLike`, `NumScalarLike`, etc.
**Other:** `StringLike`, `ArrayLike[scalar, dtype]` (template)

### JAX/PyTorch (`shapix.jax`, `shapix.torch`)

Most NumPy array types, plus `BF16` and `BF16Like`. NumPy-only extended-precision array aliases such as `F128` / `C256` stay in `shapix.numpy`. Both export `Like` types, `ScalarLike` types (re-exported from numpy), and `make_scalar_like_type`. JAX also exports `Tree`.

### Factories (`shapix`)

`make_array_type(array_class, dtype_spec)` — custom array type
`make_array_like_type(dtype_spec, *, casting="same_kind", name="ArrayLike")` — custom Like type
`make_scalar_like_type(target_dtype, *, casting="same_kind", name="ScalarLike")` — custom ScalarLike type
`DtypeSpec(name, allowed)` — custom dtype specification
`DtypeSpec.structured(fields)` — structured dtype specification

### Decorators & context managers (`shapix`)

`@shapix.check` — explicit memo management
`@shapix.check(conf=BeartypeConf())` — combined memo + beartype
`shapix.check_context()` — context manager for manual checks

## License

MIT
