---
title: Installation
description: Install shapix with your preferred array backend.
---

# Installation

## Requirements

- **Python** >= 3.10
- **beartype** >= 0.20, installed automatically with `shapix`

## Install with pip

```bash
pip install shapix
```

Shapix intentionally does **not** use extras such as `shapix[numpy]`. Install `shapix` and your backend packages explicitly.

=== "NumPy"

    ```bash
    pip install shapix numpy
    ```

=== "PyTorch"

    ```bash
    pip install shapix numpy torch
    ```

=== "JAX"

    ```bash
    pip install shapix numpy jax
    ```

=== "CuPy"

    ```bash
    pip install shapix numpy cupy
    ```

=== "NumPy + Trees"

    ```bash
    pip install shapix numpy optree  # or install jax and use shapix.jax.Tree
    ```

!!! note
    `shapix.jax`, `shapix.torch`, and `shapix.cupy` require `numpy` alongside the backend. The lightweight root import `import shapix` does not.

## Install with uv

```bash
uv add shapix
```

## Optional dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | NumPy array aliases, `ScalarLike`, and backend dtype helpers |
| `torch` | PyTorch tensor aliases and Torch `Like` types |
| `jax` | JAX array aliases, JAX `Like` types, and JAX `Tree` |
| `cupy` | CuPy array aliases and CuPy `Like` types |
| `optree` | Explicit pytree backend via `shapix.optree.Tree` |

## Import boundaries

The root package is designed to stay optional-dependency-safe:

```python
import shapix

print(shapix.__version__)
print(shapix.N, shapix.C)
```

That works even in a plain source checkout without installed package metadata. In that case `__version__` falls back to a non-empty string such as `0+unknown`.

Backend modules are stricter:

- `shapix.numpy` needs `numpy`
- `shapix.jax` needs `jax` and `numpy`
- `shapix.torch` needs `torch` and `numpy`
- `shapix.cupy` needs `cupy` and `numpy`
- `shapix.optree` needs `optree`

## Verify installation

```python
import shapix
print(shapix.__version__)
```

Then verify the backend you actually plan to use:

```python
from shapix import N, C
from shapix.numpy import F32
```
