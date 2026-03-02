# Shapix

Runtime shape and dtype checking for NumPy, JAX, and PyTorch arrays, powered by beartype.

## Architecture

```
src/shapix/
├── __init__.py        # Public API: dimensions, check, make_array_type
├── _memo.py           # Frame-based memo management (thread-safe)
├── _shape.py          # Shape spec types and matching logic
├── _dtypes.py         # Dtype specifications and matching
├── _array_types.py    # Array type factory → Annotated[T, Is[checker]]
├── _dimensions.py     # Dimension symbols (N, C, vB, bN, etc.)
├── _decorator.py      # Optional @shapix.check + check_context
├── numpy.py           # NumPy: F32, I64, F32Like, ArrayLike, etc.
├── jax.py             # JAX: F32, BF16, etc.
├── torch.py           # PyTorch: F32, BF16, etc.
└── claw.py            # Import hook wrapping beartype.claw
```

## Key design decisions

- Frame-based memo via `sys._getframe()` + `f_lasti` for sequential call detection
- `Annotated[T, Is[validator]]` so beartype handles dispatch natively
- Under `TYPE_CHECKING`: array types are NDArray aliases, dimensions are `type X = int`
- Recursive `ArrayLike` type (PEP 695) — no depth limit
- Concise names: `F32`, `I64`, `vB`, `bN` instead of `Float32Array`, `sB`, `hN`

## Naming conventions

- Array types: `F32`, `I64`, `Bool`, `Shaped`, `Int`, `Float`, etc.
- Like types: `F32Like`, `I64Like` (scalar | array | nested sequence)
- Dimensions: `N`, `C`, `H`, `W` (named); `vB`, `vN` (variadic); `bN`, `bL` (broadcast); `_N`, `__` (anonymous)

## Running tests

```bash
uv run pytest tests/
```
