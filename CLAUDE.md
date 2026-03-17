# Shapix

Runtime shape and dtype checking for NumPy, JAX, and PyTorch arrays, powered by beartype.

## Architecture

```
src/shapix/
├── __init__.py        # Public API: dimensions, check, make_array_type (Tree in backend modules)
├── _memo.py           # Frame-based memo management (thread-safe)
├── _shape.py          # Shape spec types and matching logic
├── _dtypes.py         # Dtype specifications and matching
├── _array_types.py    # Array type factory → Annotated[T, Is[checker]]
├── _dimensions.py     # Dimension symbols (N, C, ~B, +N, __, etc.)
├── _decorator.py      # Optional @shapix.check + check_context
├── _tree.py           # Tree annotations with leaf-type and structure checking
├── numpy.py           # NumPy: F32, I64, F32Like, ArrayLike, etc.
├── jax.py             # JAX: F32, BF16, Tree (jax.tree_util), etc.
├── torch.py           # PyTorch: F32, BF16, etc.
├── optree.py          # Tree backed by optree
└── claw.py            # Import hook wrapping beartype.claw
```

## Key design decisions

- Frame-based memo via `sys._getframe()` + `f_lasti` for sequential call detection
- `Annotated[T, Is[validator]]` so beartype handles dispatch natively
- Under `TYPE_CHECKING`: array types are NDArray aliases, dimensions are `type X = int`
- Recursive `ArrayLike` type (PEP 695) — no depth limit
- Unary operators: `~N` (variadic), `+N` (broadcastable), `__` (anonymous), `~__` (anonymous variadic)
- Internal string representation matches user-facing syntax: `~N` not `*N`, `+N` not `#N`

## Naming conventions

- Array types: `F32`, `I64`, `Bool`, `Shaped`, `Int`, `Float`, etc.
- Like types: `F32Like`, `I64Like` (scalar | array | nested sequence) — must be subscripted: `F32Like[N, C]` or `F32Like[...]`
- ScalarLike types: `F32ScalarLike`, `I8ScalarLike` (range-validated scalars, no shape)
- Endianness variants: `F32LE`, `I64BE`, `I32N` (LE/BE/N suffixes)
- Dimensions: `N`, `C`, `H`, `W` (named); `~B` (variadic); `+N` (broadcast); `__` (anonymous); `~__` (anonymous variadic)

## Running tests

```bash
uv run pytest tests/              # all tests locally
uv run tox run -e dev             # locked env with all deps
uv run tox run -e py312-bt022-numpy24   # factor-based compat env
```

## Tox factor scheme

Runtime envs: `{python}-{beartype}-{backend}` (e.g. `py312-bt022-numpy24`)
Typecheck envs: `{python}-{beartype}-type-{checker}` (e.g. `py312-bt022-type-pyright1408`)

Factor deps are merged by tox. `tests/conftest.py` filters tests by backend via `TOX_ENV_NAME`.
`tools/validate_tox_env.py` validates env names in `commands_pre`.

## Type checking

- pyright: `strict` mode with targeted suppressions for beartype internals
- mypy: `strict = true` with `ignore_missing_imports = true`
- ty: defaults only (`ty.toml` sets python-version)
