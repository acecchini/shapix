---
title: Features
description: Explore Shapix's core features for runtime array checking.
---

# Features

Shapix provides a comprehensive set of tools for runtime shape and dtype validation. Each feature integrates seamlessly with beartype's `Annotated[T, Is[...]]` mechanism.

---

- :material-ruler: **[Dimensions](dimensions.md)** — Named, fixed, variadic, broadcastable, anonymous, and symbolic dimensions — the complete dimension system.

- :material-grid: **[Array Types](array-types.md)** — Concise dtype-checked array types for NumPy, JAX, and PyTorch — plus a factory for custom backends.

- :material-keyboard: **[Like Types](like-types.md)** — Input validation types that accept scalars, arrays, or nested sequences of any depth.

- :material-file-tree: **[Tree Annotations](tree-annotations.md)** — Validate all leaves and enforce structure consistency in nested containers.

- :material-wrench: **[Decorator & Memo](decorator.md)** — `@shapix.check`, `check_context`, and how cross-argument consistency works under the hood.

- :material-hook: **[Import Hook](claw.md)** — Package-wide instrumentation with `beartype.claw` — no per-function decorators needed.
