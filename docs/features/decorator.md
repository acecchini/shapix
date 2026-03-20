---
title: Decorator & Memo
description: Explicit memo management with @shapix.check and check_context.
---

# Decorator & Memo

## How cross-argument checking works

A **dimension memo** maps dimension names to sizes (e.g., `N→4`, `C→3`). Each function call gets a fresh memo. All parameter checks within that call share the same memo — that's how shapix knows `N=4` in `x` must match `N=4` in `y`.

This happens **automatically** with `@beartype`. Shapix detects the beartype wrapper frame via `sys._getframe()` and associates a memo with it. No extra decorator needed.

```python
@beartype  # Shapix auto-detects this frame — nothing else needed
def f(x: F32[N, C], y: F32[N, C]) -> F32[N, C]:
    ...
```

## `@shapix.check`

`@shapix.check` provides **explicit** memo management. Instead of detecting the frame, it pushes a fresh memo onto a stack before the call and pops it after.

### Usage mode 1: memo only

Pair with `@beartype`:

```python
import shapix
from beartype import beartype

@shapix.check
@beartype
def f(x: F32[N, C], y: F32[N, C]) -> F32[N, C]:
    ...
```

### Usage mode 2: memo + beartype combined

Pass a `BeartypeConf` to combine both:

```python
from beartype import BeartypeConf, BeartypeStrategy

@shapix.check(conf=BeartypeConf(strategy=BeartypeStrategy.On))
def f(x: F32[N, C], y: F32[N, C]) -> F32[N, C]:
    ...
```

### When do you need it?

Both approaches (auto-detection and explicit) produce identical results in normal usage. You only need `@shapix.check` in specific situations:

#### 1. Extra decorators between beartype and the call site

Frame-based detection counts a fixed number of frames up from the validator. If something adds extra frames, detection can land on the wrong frame:

```python
# This might break — middleware adds extra frames
@some_middleware
@beartype
def f(x: F32[N, C], y: F32[N, C]) -> F32[N, C]: ...

# Fix: @shapix.check bypasses frame detection entirely
@some_middleware
@shapix.check
@beartype
def f(x: F32[N, C], y: F32[N, C]) -> F32[N, C]: ...
```

#### 2. Defensive coding

If you want a guarantee that cross-argument checking works regardless of how your code is called (test runners, async frameworks, deep middleware stacks), `@shapix.check` removes all dependence on call-stack structure.

### Async support

`@shapix.check` supports both sync and async functions. For async functions, the memo scope covers the full awaited execution, and `inspect.iscoroutinefunction()` is preserved on the decorated function.

!!! tip "When you don't need it"
    If you're using plain `@beartype` and your tests pass, the frame-based detection is working. Most applications never need `@shapix.check`.

## `check_context`

For `isinstance`-style checks outside of decorated functions, use `check_context` with beartype's `is_bearable`:

```python
from beartype.door import is_bearable
import shapix
from shapix import N, C
from shapix.numpy import F32

# Without check_context — each check is independent
is_bearable(x, F32[N, C])  # Binds N=4 in a temporary memo (discarded)
is_bearable(y, F32[N, C])  # Binds N=999 in a new memo — no error!

# With check_context — checks share a memo
with shapix.check_context():
    assert is_bearable(x, F32[N, C])  # Binds N=4
    assert is_bearable(y, F32[N, C])  # Checks N=4 — raises if y has N=999
```

All `is_bearable` calls within the context manager share the same dimension memo, enabling cross-argument consistency for manual checks.

## Internals: the memo

The `ShapeMemo` tracks three kinds of bindings:

| Field | Type | Purpose |
|-------|------|---------|
| `single` | `dict[str, int]` | Named dimension bindings: `{"N": 4, "C": 3}` |
| `variadic` | `dict[str, tuple[bool, tuple[int, ...]]]` | Variadic bindings: `{"B": (False, (2, 4))}` |
| `structures` | `dict[str, object]` | Tree structure bindings |

The memo uses **snapshot-and-rollback**: before checking each argument, the current state is saved. On failure, the memo rolls back to prevent partial bindings from polluting subsequent checks.

## Thread and async safety

- **Frame-based auto-detection** uses `threading.local()` for thread isolation.
- **Explicit memo stack** (`@shapix.check`, `check_context()`) uses `contextvars.ContextVar`, making it both thread-safe and async-task-safe.

!!! note
    Child tasks inheriting an active parent context share the same live memo by reference. For full task isolation, each task should enter its own `check_context()`.

## `from __future__ import annotations` (PEP 563)

Shapix is fully compatible with `from __future__ import annotations`. The one rule: **every dimension symbol used in an annotation must be imported in the module scope.**

```python
from __future__ import annotations
from shapix import B, C  # Both B and C must be imported
from shapix.numpy import F32

@beartype
def f(x: F32[~B, C]): ...
```
