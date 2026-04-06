---
description: Explore Shapix's runtime contracts, typing model, and backend-specific tools.
---

# Features

Shapix layers several pieces together:

- a shape language built from Python objects such as `N`, `Scalar`, and
    `Value("size")`
- backend-specific array aliases such as `F32`, `BF16`, and `DT64`
- input-contract aliases such as `F32Like[...]` and `U8ScalarLike`
- memo helpers for explicit scope control when plain `@beartype` is not enough
- a static typing surface exercised by pyright, mypy, and ty

## Guides

- :material-ruler: **[Dimensions](dimensions.md)** — Named, fixed, variadic,
    broadcastable, anonymous, and symbolic dimensions.
- :material-check-decagram: **[Static Typing](static-typing.md)** — How the
    public annotation surface maps onto pyright, mypy, and ty, plus the
    runtime-only patterns that still need targeted workarounds.
- :material-grid: **[Array Types](array-types.md)** — Dtype-checked array
    aliases for NumPy, JAX, PyTorch, CuPy, and custom array classes.
- :material-keyboard: **[Like Types](like-types.md)** — Input validation types
    that accept scalars, arrays, or nested sequences, plus range-validated
    `ScalarLike` aliases.
- :material-file-tree: **[Tree Annotations](tree-annotations.md)** — Validate
    all leaves and enforce structure consistency in nested containers.
- :material-wrench: **[Decorator & Memo](decorator.md)** — `@shapix.check`,
    `check_context`, async behavior, and when explicit memo scope is worth it.
- :material-hook: **[Import Hook](claw.md)** — Package-wide instrumentation with
    `beartype.claw`.
