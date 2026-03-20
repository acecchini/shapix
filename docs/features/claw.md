---
title: Import Hook
description: Package-wide instrumentation with beartype.claw — no per-function decorators.
---

# Import Hook (beartype.claw)

Instead of decorating each function with `@beartype`, you can instrument an entire package at once. Every function using shapix type annotations will be checked automatically.

## Usage

=== "Via shapix"

    ```python
    # In your_package/__init__.py
    from shapix.claw import shapix_this_package
    shapix_this_package()
    ```

=== "Via beartype directly"

    ```python
    # In your_package/__init__.py
    from beartype.claw import beartype_this_package
    beartype_this_package()
    ```

Both are equivalent. `shapix_this_package` is a thin wrapper around `beartype_this_package` that exists as a semantic entry point.

## How it works

Once activated in your package's `__init__.py`, **all subsequently imported submodules** get `@beartype` applied automatically:

```python
# your_package/__init__.py
from shapix.claw import shapix_this_package
shapix_this_package()

# your_package/model.py — automatically instrumented
from shapix import N, C
from shapix.numpy import F32

def forward(x: F32[N, C]) -> F32[N, C]:  # checked at runtime!
    ...
```

## Custom configuration

Pass a `BeartypeConf` to customize the checking behavior:

```python
from beartype import BeartypeConf
from shapix.claw import shapix_this_package

shapix_this_package(conf=BeartypeConf(
    is_color=False,  # disable colored error messages
))
```

## When to use

| Approach | Best for |
|----------|----------|
| `@beartype` per function | Fine-grained control, specific functions |
| `beartype.claw` / `shapix_this_package` | Entire packages, library-wide checking |

!!! tip
    The import hook approach is especially useful for large codebases where adding `@beartype` to every function would be tedious. It also catches functions that might otherwise be missed.
