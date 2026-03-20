---
title: Installation
description: Install Shapix with your preferred array backend.
---

# Installation

## Requirements

- **Python** >= 3.12
- **beartype** >= 0.20 (installed automatically)

## Install with pip

```bash
pip install shapix
```

Shapix has one dependency: [beartype](https://github.com/beartype/beartype). Install your preferred array framework separately — shapix does not provide extras-style installs:

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

=== "NumPy + Trees"

    ```bash
    pip install shapix numpy optree  # or jax for jax.tree_util
    ```

!!! note
    For `shapix.jax` and `shapix.torch`, install `numpy` alongside the backend.

## Install with uv

```bash
uv add shapix
```

## Optional dependencies

| Package | Purpose |
|---------|---------|
| `numpy` | NumPy array support, ScalarLike types |
| `torch` | PyTorch tensor support |
| `jax` | JAX array + tree support |
| `optree` | Tree annotations (alternative to JAX) |

## Verify installation

```python
import shapix
print(shapix.__version__)
```
