---
description: NumPy array aliases, Like aliases, ScalarLike aliases, and NumPy-specific helpers.
---

# `shapix.numpy`

`shapix.numpy` is the main NumPy-facing module. It exports:

- strict array aliases such as `F32[N, C]`
- `Like` aliases such as `F32Like[...]`
- `ScalarLike` aliases such as `U8ScalarLike`
- `Structured(...)`
- `ArrayLike`
- `make_scalar_like_type(...)`

```python
from shapix.numpy import (
  F32, I64, DT64, TD64, Shaped, Structured,
  F32Like, IntLike,
  U8ScalarLike, ArrayLike, make_scalar_like_type,
)
```

## Strict array aliases

Concrete families:

- `Bool`
- `I8`, `I16`, `I32`, `I64`
- `U8`, `U16`, `U32`, `U64`
- `F16`, `F32`, `F64`, `F128`
- `C64`, `C128`, `C256`

Category families:

- `Int`, `UInt`, `Integer`
- `Float`, `Real`, `Complex`, `Inexact`, `Num`
- `Shaped`

Additional NumPy-only aliases:

- `V`
- `Str`
- `Bytes`
- `Obj`
- `DT64`
- `TD64`

Notes:

- `DT64` and `TD64` accept unit-qualified NumPy dtypes such as `datetime64[ns]`
    and `timedelta64[ms]`.
- `Shaped` accepts any dtype at runtime and checks only shape.

## `Structured`

`Structured(fields)` creates a strict array alias for one exact NumPy structured
dtype.

```python
import numpy as np
from shapix import N
from shapix.numpy import Structured

Point = Structured([("x", np.float32), ("y", np.float32)])
```

This is a convenience wrapper over `DtypeSpec.structured(...)`.

## `Like` aliases

`Like` aliases are subscripted input contracts. Examples:

- `F32Like[...]`
- `I64Like[N]`
- `NumLike[N, C]`
- `ShapedLike[...]`

At runtime they accept scalars, nested sequences, arrays, and convertible
values. Static type checkers see a narrower model.

Built-in `Like` aliases use the default
`make_array_like_type(..., casting="same_kind")` behavior. That is why
`F32Like[...]` accepts integer inputs but rejects complex inputs by default.

## `ScalarLike` aliases

`ScalarLike` aliases validate individual scalar values:

- `BoolScalarLike`
- `I8ScalarLike` through `I64ScalarLike`
- `U8ScalarLike` through `U64ScalarLike`
- `F16ScalarLike` through `F128ScalarLike`
- `C64ScalarLike`, `C128ScalarLike`, `C256ScalarLike`
- category aliases such as `IntScalarLike`, `FloatScalarLike`, `NumScalarLike`
- `StringLike`

Numeric scalar aliases intentionally reject booleans.

## `make_scalar_like_type`

Use
`make_scalar_like_type(target_dtype, *, casting="same_kind", name="ScalarLike")`
when the built-in scalar aliases are close but not exact enough.

```python
import numpy as np
from shapix.numpy import make_scalar_like_type

F32ScalarStrict = make_scalar_like_type(np.float32, casting="no")
F32ScalarSafe = make_scalar_like_type(np.float32, casting="safe")
```

Common interpretations:

- `"no"`: exact target dtype only
- `"safe"`: no information loss
- `"same_kind"`: same numeric family, and the default
- `"unsafe"`: most permissive

Numeric factories still reject booleans. For example,
`make_scalar_like_type(np.float32)` rejects `True`, while
`make_scalar_like_type(np.bool_)` accepts it.

`make_scalar_like_type` is deliberately documented here rather than on the root
module, because the root import stays NumPy-optional.

## `ArrayLike`

`ArrayLike` is the public recursive typing template behind the static `Like`
surface:

```python
import numpy as np
from shapix.numpy import ArrayLike

type MyInput = ArrayLike[float, np.float32]
```

It is useful when you want a checker-friendly custom alias that still follows
the shapix "scalar or nested sequence or array" model.
