---
title: Examples
description: Hands-on examples and notebooks demonstrating Shapix.
---

# Examples

## Shapix Tour Notebook

The interactive tour notebook covers all major features with runnable examples:

1. **Basic usage** — `@beartype` + `F32[N, C]`, wrong dtype/rank caught
2. **Cross-argument consistency** — `N` and `C` enforced across `x` and `y`
3. **Sequential calls** — each invocation gets fresh bindings
4. **Return type checking** — return values validated against spec
5. **Fixed dimensions** — exact-size matching with integers
6. **Symbolic dimensions** — `N + 2`, `N * C` arithmetic
7. **Variadic dimensions** — `~B` for zero or more batch dims
8. **Broadcastable dimensions** — `+N` allows size 1
9. **Anonymous dimensions** — `__` matches anything
10. **Ellipsis** — `...` as alias for `~__`
11. **Custom dimensions** — `Dimension("Vocab")` with `TYPE_CHECKING` pattern
12. **Nested function calls** — independent memos
13. **Multiple dtype arrays** — `F32`, `I32`, `F64`, `Float` category
14. **Custom array types** — `make_array_type` + `DtypeSpec`
15. **Explicit memo** — `@shapix.check` + `check_context`
16. **Full example** — mini neural network with `linear`, `relu`, `classifier`
17. **Tree annotations** — `Tree[F32[N, C]]`, `Tree[F32[N], T]`, `Tree[F32[N], T, ...]`

[:material-notebook: View on GitHub](https://github.com/acecchini/shapix/blob/main/examples/shapix_tour.ipynb){ .md-button .md-button--primary }

---

## Mini Neural Network

A complete example combining multiple features:

```python
import numpy as np
from beartype import beartype
from shapix import N, C, H, W, Dimension
from shapix.numpy import F32

# Custom dimensions
Features = Dimension("Features")
Classes = Dimension("Classes")

@beartype
def linear(x: F32[N, C], w: F32[C, Features], b: F32[Features]) -> F32[N, Features]:
    return x @ w + b

@beartype
def relu(x: F32[N, Features]) -> F32[N, Features]:
    return np.maximum(x, 0)

@beartype
def classifier(x: F32[N, C]) -> F32[N, Classes]:
    # Two-layer network: C -> 128 -> 10
    w1 = np.random.randn(x.shape[1], 128).astype(np.float32)
    b1 = np.zeros(128, dtype=np.float32)
    w2 = np.random.randn(128, 10).astype(np.float32)
    b2 = np.zeros(10, dtype=np.float32)

    h = relu(linear(x, w1, b1))
    return linear(h, w2, b2)

# All shapes are checked at runtime!
batch = np.random.randn(32, 784).astype(np.float32)
logits = classifier(batch)  # F32[32, 10]
```

## Tree Operations

Working with nested structures:

```python
import numpy as np
from beartype import beartype
from shapix import T, N
from shapix.optree import Tree
from shapix.numpy import F32

@beartype
def tree_add(x: Tree[F32[N], T], y: Tree[F32[N], T]) -> Tree[F32[N]]:
    """Add two trees with matching structure and shapes."""
    import optree
    x_leaves, x_struct = optree.tree_flatten(x)
    y_leaves, _ = optree.tree_flatten(y)
    return optree.tree_unflatten(
        x_struct,
        [a + b for a, b in zip(x_leaves, y_leaves)]
    )

params = {
    "layer1": {"weight": np.ones((64,), dtype=np.float32),
               "bias": np.zeros((64,), dtype=np.float32)},
    "layer2": {"weight": np.ones((64,), dtype=np.float32),
               "bias": np.zeros((64,), dtype=np.float32)},
}

grads = {
    "layer1": {"weight": np.random.randn(64).astype(np.float32),
               "bias": np.random.randn(64).astype(np.float32)},
    "layer2": {"weight": np.random.randn(64).astype(np.float32),
               "bias": np.random.randn(64).astype(np.float32)},
}

updated = tree_add(params, grads)  # Structure T enforced to match!
```

## Comparison with jaxtyping

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
