---
title: Features
description: Explore Shapix's core features for runtime array checking.
---

# Features

Shapix provides a comprehensive set of tools for runtime shape and dtype validation. Each feature integrates seamlessly with beartype's `Annotated[T, Is[...]]` mechanism.

<div class="grid cards" markdown>

-   :material-ruler:{ .lg .middle } **Dimensions**

    ---

    Named, fixed, variadic, broadcastable, anonymous, and symbolic dimensions — the complete dimension system.

    [:octicons-arrow-right-24: Dimensions](dimensions.md)

-   :material-grid:{ .lg .middle } **Array Types**

    ---

    Concise dtype-checked array types for NumPy, JAX, and PyTorch — plus a factory for custom backends.

    [:octicons-arrow-right-24: Array types](array-types.md)

-   :material-keyboard:{ .lg .middle } **Like Types**

    ---

    Input validation types that accept scalars, arrays, or nested sequences of any depth.

    [:octicons-arrow-right-24: Like types](like-types.md)

-   :material-file-tree:{ .lg .middle } **Tree Annotations**

    ---

    Validate all leaves and enforce structure consistency in nested containers.

    [:octicons-arrow-right-24: Tree annotations](tree-annotations.md)

-   :material-wrench:{ .lg .middle } **Decorator & Memo**

    ---

    `@shapix.check`, `check_context`, and how cross-argument consistency works under the hood.

    [:octicons-arrow-right-24: Decorator & memo](decorator.md)

-   :material-hook:{ .lg .middle } **Import Hook**

    ---

    Package-wide instrumentation with `beartype.claw` — no per-function decorators needed.

    [:octicons-arrow-right-24: Import hook](claw.md)

</div>
