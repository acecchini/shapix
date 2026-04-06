---
description: Install shapix, pick a backend, and learn the runtime and typing model.
---

# Getting Started

Shapix turns annotations like `F32[N, C]` into runtime-validated contracts by
generating small runtime hint classes for arrays and trees. Beartype validates
those hints through `__instancecheck__()`, and shapix provides readable failures
through `__instancecheck_str__()`.

The root `shapix` module is intentionally lightweight: it exports dimension
symbols, tree structure symbols, `DtypeSpec`, the custom array factories, and
the memo helpers. Backend modules such as `shapix.numpy`, `shapix.jax`,
`shapix.torch`, and `shapix.cupy` add concrete array aliases once you install
those dependencies.

## Start here

- :material-download: **[Installation](installation.md)** — Python requirements,
    backend-specific install commands, and optional dependency boundaries.
- :material-rocket-launch: **[Quick Start](quickstart.md)** — Your first
    `@beartype`-checked function, cross-argument consistency, and return
    validation.
- :material-check-decagram: **[Static Typing](../features/static-typing.md)** —
    What passes on pyright, mypy, and ty, and what still needs targeted
    runtime-only workarounds.

## What Shapix gives you

- Standard `@beartype` support. No custom decorator is required for normal shape
    checking.
- Cross-argument memo binding. If one parameter binds `N = 32`, later parameters
    and the return annotation must agree.
- Backend-specific array aliases for NumPy, JAX, PyTorch, and CuPy.
- `Like` and `ScalarLike` input contracts for data-conversion APIs.
- Explicit memo tools for tricky call stacks, async functions, and manual
    `is_bearable()` checks.

## Where to go next

- [Dimensions](../features/dimensions.md) explains named, fixed, variadic,
    broadcastable, anonymous, symbolic, and `Value(...)`-based shape tokens.
- [Array Types](../features/array-types.md) covers dtype families, structured
    dtypes, endianness, and backend differences.
- [Tree Annotations](../features/tree-annotations.md) covers leaf validation
    plus structure matching for pytrees.
