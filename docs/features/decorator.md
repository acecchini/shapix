---
title: Decorator & Memo
description: Explicit memo management with @shapix.check and check_context.
---

# Decorator & Memo

## How cross-argument checking works

A dimension memo maps names such as `N` and `C` to the concrete sizes seen during one call. That shared memo is why:

- `x: F32[N, C]` can bind `N = 4`
- `y: F32[N, C]` is then required to use the same `N = 4`
- the return value can be checked against the same bindings

```python
from beartype import beartype
from shapix import N, C
from shapix.numpy import F32

@beartype
def f(x: F32[N, C], y: F32[N, C]) -> F32[N, C]:
  ...
```

Plain `@beartype` is the normal entry point. Most users never need more than that.

## Which decorator should you choose?

Use plain `@beartype` by default.

Add `@shapix.check` only when you need explicit memo scope rather than frame discovery.

| Choice | What it does | When it is the right fit |
|--------|---------------|--------------------------|
| `@beartype` | lets shapix find the shared memo by walking the beartype call stack | normal application code with no unusual wrapper layers |
| `@shapix.check` + `@beartype` | pushes one memo explicitly before the call and pops it after | middleware-heavy stacks, wrapper decorators, async `Value(...)`, or defensive correctness |
| `@shapix.check(conf=...)` | same explicit memo handling, and applies beartype for you | when you also want to pass `BeartypeConf` without stacking both decorators manually |

What `@shapix.check` changes:

- how the shared memo is found
- how long that memo lives around the call

What it does **not** change:

- the array, dtype, or shape semantics themselves
- whether return values are checked
- whether plain `@beartype` remains the simplest choice for ordinary code

## `@shapix.check`

`@shapix.check` provides explicit memo management. Instead of discovering the correct beartype frame dynamically, it pushes a memo before the call and pops it afterwards.

### Usage mode 1: memo only

```python
import shapix
from beartype import beartype
from shapix import N, C
from shapix.numpy import F32

@shapix.check
@beartype
def f(x: F32[N, C], y: F32[N, C]) -> F32[N, C]:
  ...
```

### Usage mode 2: memo + beartype combined

```python
import shapix
from beartype import BeartypeConf, BeartypeStrategy
from shapix import N, C
from shapix.numpy import F32

@shapix.check(conf=BeartypeConf(strategy=BeartypeStrategy.On))
def f(x: F32[N, C], y: F32[N, C]) -> F32[N, C]:
  ...
```

### When do you need it?

Use it when you want memo scope to stop depending on call-stack shape.

Typical reasons:

- extra decorators or framework wrappers add their own call frames
- `Value(...)` needs an explicit scope across async execution
- you want `BeartypeConf` and memo handling from one decorator
- you want a defensive guarantee that exotic stack layouts will not matter

Concrete examples:

- web frameworks or middleware that wrap handlers before beartype sees them
- utility decorators around model code that add their own frames
- async code where a `Value("size")` expression should keep using the original bound scope until the coroutine finishes
- tests or runtime environments where you do not want frame-layout assumptions to be part of correctness

## Async support

`@shapix.check` supports both sync and async callables:

- `inspect.iscoroutinefunction()` remains `True`
- the memo lifetime covers the awaited execution, not just coroutine creation
- parameter mismatches and return mismatches are still raised normally
- `@shapix.check(conf=...)` works for async functions too

Generator functions are intentionally rejected:

- sync generators raise `TypeError`
- async generators raise `TypeError`

!!! tip "When you don't need it"
    If plain `@beartype` is already working in your codebase, keep it simple. `@shapix.check` is an explicit escape hatch, not the default style.

## `check_context`

For manual `is_bearable()` checks, use `check_context` so multiple validations share one memo.

```python
from beartype.door import is_bearable
import shapix
from shapix import N, C
from shapix.numpy import F32

is_bearable(x, F32[N, C])  # independent temporary memo
is_bearable(y, F32[N, C])  # independent temporary memo

with shapix.check_context():
  assert is_bearable(x, F32[N, C])
  assert is_bearable(y, F32[N, C])
```

`check_context` supports both:

- `with shapix.check_context():`
- `async with shapix.check_context():`

## Thread and async safety

- frame-based auto-detection uses `threading.local()` for thread isolation
- the explicit memo stack used by `@shapix.check` and `check_context()` uses `contextvars.ContextVar`

!!! note
    Child tasks inheriting an active parent context share the same live memo by reference. For full task isolation, each task should enter its own `check_context()`.

## `from __future__ import annotations`

Shapix works with `from __future__ import annotations`, but every symbol used inside the annotation must still be imported in module scope:

```python
from __future__ import annotations
from beartype import beartype
from shapix import B, C
from shapix.numpy import F32

@beartype
def f(x: F32[~B, C]):  # type: ignore[valid-type]
  ...
```

The same rule applies to custom dimensions and structure symbols: if the annotation refers to a runtime object, that symbol must exist in module scope when beartype resolves the annotation.
