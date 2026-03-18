# Shapix

Runtime shape and dtype checking for NumPy, JAX, and PyTorch arrays, powered by beartype.

> Current release posture: **hold release until the blocker, the high-severity runtime issues, and full static type-checker parity are fixed, covered by regression tests, and documented accurately.**

## Mission

Shapix turns array annotations such as `F32[N, C]` into runtime-validated contracts using `typing.Annotated` + beartype validators.

This repository is correctness-sensitive in two dimensions:

1. **Runtime behavior** must be correct.
2. **Static typing behavior** must be coherent across supported checkers.

When working in this repo, optimize for:

1. **Runtime correctness**
2. **Full static type-checker parity across pyright, mypy, and ty**
3. **Public API stability**
4. **Accurate docs that match the real supported surface**
5. **Regression-proof tests in the right backend envs**
6. **Small, explicit fixes over clever rewrites**

## Immediate release blockers and must-fix contracts

These are the current non-negotiable contracts for the codebase:

### 1) `@shapix.check` must be async-safe
- Decorating `async def` must preserve coroutine behavior.
- `inspect.iscoroutinefunction()` should still report `True`.
- Explicit memo lifetime must cover the awaited execution, not just coroutine creation.
- If async support ever becomes intentionally unsupported, reject it explicitly at decoration time. Do **not** silently wrap async callables incorrectly.

### 2) `Scalar` may not silently disappear inside mixed shape specs
- `Scalar` is valid only as the sole shape token.
- Mixed forms such as `F32[N, Scalar]`, `F32[Scalar, N]`, `F32[..., Scalar]`, or `F32[Scalar, ...]` must raise clearly at hint construction time.
- Silent reinterpretation is unacceptable.

### 3) `DT64` / `TD64` must accept real NumPy datetime/timedelta arrays
- Unit-qualified dtypes like `datetime64[ns]`, `datetime64[D]`, `timedelta64[ms]`, and `timedelta64[s]` are normal NumPy dtypes and must match the corresponding shapix aliases.

### 4) Numeric `ScalarLike` semantics must exclude booleans
- `BoolScalarLike` accepts booleans.
- Numeric scalar aliases and numeric scalar factory outputs must **not** accept `True` / `False` merely because Python `bool` subclasses `int`.
- `ShapedScalarLike` may still include booleans if that remains the intended umbrella contract.

### 5) Root import must remain lightweight and optional-dependency-safe
- `import shapix` must not require NumPy.
- Root `shapix` must **not** start exporting `make_scalar_like_type` if that would violate the import boundary.
- Prefer fixing docs to match the existing root API rather than widening the root API.

### 6) Source-tree imports must survive missing installed metadata
- `__version__` lookup must not make `import shapix` fail in a plain source checkout.
- Fallback behavior must still provide a non-empty version string.

### 7) Full static type-checker parity is a release requirement
- The flagship annotation surface must type-check on **pyright, mypy, and ty**.
- `F32[N, C]`, `F32[Scalar]`, `F32[N + 2]`, `F32[__, C]`, `I64[N, Vocab]`, backend variants, and `Tree[...]` must all be accepted by all three checkers.
- `@shapix.check` must preserve usable static signatures for all three checkers, including async functions.
- The typing test harness must not keep a pyright-only bucket for the flagship annotation files.
- Do **not** “solve” checker parity by deleting the strongest annotation examples, downgrading the public syntax, or moving failures behind checker-specific omissions.

## Architecture

```text
src/shapix/
├── __init__.py        # Root public API, version export, public docs surface
├── _memo.py           # Frame-based memo management + explicit memo stack
├── _shape.py          # Shape spec types, expression evaluation, rank/shape matching
├── _dtypes.py         # Dtype extraction + DtypeSpec matching
├── _array_types.py    # Array / ArrayLike factories and validator plumbing
├── _dimensions.py     # Dimension symbols, Scalar, Value(...), arithmetic
├── _decorator.py      # @shapix.check + check_context
├── _tree.py           # Tree validators and structure binding
├── numpy.py           # NumPy public surface + typing stubs
├── jax.py             # JAX public surface + Tree + typing stubs
├── torch.py           # PyTorch public surface + typing stubs
├── optree.py          # optree-backed Tree
└── claw.py            # beartype.claw convenience wrapper
```

## High-risk areas

### Memo system
`_memo.py` is clever and brittle by nature. It depends on frame inspection and beartype call-stack assumptions. Treat any change here as high-risk.

Rules:
- Preserve thread-local behavior.
- Preserve explicit push/pop symmetry on success and failure.
- Prefer adding regression tests before touching memo behavior.
- Do not broaden frame heuristics casually.

### Decorator behavior
`_decorator.py` is public API. It is not just convenience glue.

Rules:
- Preserve function metadata.
- Preserve signature ergonomics.
- Preserve explicit memo scope for `Value(...)`.
- Support both sync and async correctly.
- Preserve static typing fidelity for decorated sync and async functions.

### Shape token parsing
`_array_types._to_shape_spec()` is a contract boundary.

Rules:
- Reject invalid or ambiguous specs eagerly.
- Never silently drop or reinterpret user tokens except where explicitly documented.
- `Scalar` is the key trap here.

### Dtype normalization
`_dtypes.extract_dtype_str()` is the canonical dtype bridge across NumPy, JAX, and Torch.

Rules:
- Normalize platform/runtime differences carefully.
- Add tests for every new normalization rule.
- Do not regress structured dtype or endianness behavior while fixing datetime/timedelta.

### Static typing surface
The `TYPE_CHECKING` branches and typing fixtures are part of the public API.

Rules:
- Design typing stubs for **all three** supported checkers, not just pyright.
- Prefer checker-agnostic `TYPE_CHECKING` representations over checker-specific hacks.
- Preserve the public syntax users actually write: `F32[N, C]`, not a weaker fallback syntax.
- Keep runtime behavior and static-only scaffolding clearly separated.
- If predefined dimensions or factories need different static representations, update typing fixtures accordingly instead of preserving stale assumptions.

### Docs as contract
README, root module docs, backend docs, and examples are part of the public API.

Rules:
- If code/tests/docs disagree, fix the contract explicitly.
- Prefer narrowing docs over widening API when optional-dependency boundaries matter.
- Do not merge docs that overclaim checker support before the unified type-check suite passes.
- Do not leave pyright-specific wording behind once parity has been achieved.

## Intentional semantics to preserve

These behaviors are currently intentional or at least test-encoded. Do not “fix” them accidentally:

- Frame-based memo detection remains the default cross-argument mechanism.
- `check_context()` is for shared manual `is_bearable()` checks.
- Like-type endianness is checked on the **source object’s** byte order, not on a hypothetical cast target.
- `Value(...)` supports a constrained expression language; do not broaden it into arbitrary eval.
- Attribute access is allowed only in the `Value(...)` path and should stay constrained.
- Root `shapix` import should stay independent from NumPy at runtime.
- The public shape-annotation syntax itself should stay stable while typing parity is improved.

## Public API decisions

### Root module (`shapix`)
Root `shapix` is intentionally small:
- dimension symbols
- tree symbols / `Structure`
- `DtypeSpec`
- `make_array_type`
- `make_array_like_type`
- `check`
- `check_context`

Do **not** add `make_scalar_like_type` to root just to make docs “nicer.”
That would work against the tested no-NumPy root import boundary.

### Backend modules
- `shapix.numpy` owns `make_scalar_like_type`.
- `shapix.jax` and `shapix.torch` may re-export scalar-like types and the scalar factory.
- Backend-specific runtime behavior must be tested in backend-specific test files.
- Backend-specific static typing must also be validated in backend-specific typing fixtures.

### Static type-checker contract
The target public contract is simple:
- shapix supports **pyright, mypy, and ty**
- the full flagship annotation surface type-checks on all three
- examples and docs should use one public syntax, not checker-specific alternates

Do **not** preserve a pyright-only support story as a steady state.

## Test placement rules

This repo uses backend-based test filtering in `tests/conftest.py`. That means test location matters.

Put tests here:

- generic decorator/runtime tests: `tests/test_decorator.py`
- generic dtype extraction/matching tests: `tests/test_dtypes.py`
- generic shape-token parsing tests: `tests/test_dimensions.py` or `tests/test_numpy.py`
- JAX-specific runtime coverage: `tests/test_jax.py`
- Torch-specific runtime coverage: `tests/test_torch.py`
- optree-specific runtime coverage: `tests/test_tree.py`
- static typing contract: `tests/test_typecheck.py` and `tests/typing/*`

Typing-specific rule:
- the flagship annotation fixtures must be exercised by **all three** checkers
- do not keep a `PYRIGHT_ONLY_FILES` split once parity work is complete

## Required regression coverage

Every fix in a high-risk area needs a regression test.

Minimum expected additions for the current patch series:

### Runtime
- async `@shapix.check`
- async `@shapix.check(conf=...)`
- async return-shape violation under `@check`
- async `Value(...)` resolution under `@check`
- mixed `Scalar` misuse raises
- datetime64 extraction/matching
- timedelta64 extraction/matching
- boolean rejection in numeric scalar aliases
- boolean rejection in numeric `make_scalar_like_type(...)`
- root import fallback when package metadata is unavailable
- JAX `Value(...)` coverage
- Torch `Value(...)` coverage
- JAX `Tree` smoke coverage if `Tree` is exported there

### Static typing
- `tests/test_typecheck.py` runs the same flagship annotation files under pyright, mypy, and ty
- `tests/typing/check_annotations.py` passes on all three
- `tests/typing/check_annotations_jax.py` passes on all three
- `tests/typing/check_annotations_torch.py` passes on all three
- typing coverage includes predefined dimensions, `Scalar`, arithmetic dims, anonymous dims, `@check`, and backend imports
- add an async-`@check` typing fixture if needed to prove the decorated async signature is acceptable to all three
- remove “pyright-only” assumptions from typing fixture headers/comments

## Development workflow

### 1) Reproduce first
Before changing code, understand:
- current behavior,
- intended behavior,
- existing tests that encode related semantics,
- current checker failures on the full flagship typing suite.

### 2) Write or update a failing regression test
Especially for:
- decorator behavior,
- shape-token parsing,
- dtype normalization,
- docs/export drift,
- checker parity gaps.

### 3) Make the smallest fix that restores the contract
Avoid broad refactors unless the tests prove they are needed.

### 4) Update docs in the same change
If a public behavior changes or becomes clarified, docs must land with the code.
For typing parity work, update docs only after the unified checker suite passes.

### 5) Validate with targeted and full checks
Suggested commands:

```bash
uv sync

# targeted runtime checks
uv run pytest -n0 tests/test_decorator.py tests/test_dtypes.py tests/test_dimensions.py
uv run pytest -n0 tests/test_jax.py tests/test_torch.py tests/test_tree.py

# unified type-check contract
uv run pytest -n0 tests/test_typecheck.py
uv run pyright tests/typing
uv run mypy tests/typing
uv run ty check tests/typing

# full repo check
uv run tox run -e dev
```

Use `-n0` for targeted debugging because the default pytest config uses xdist.

## Coding standards

- Python 3.12+
- 2-space indentation
- keep public error messages clear and specific
- prefer explicit helpers over dense inline lambdas when semantics matter
- avoid new dependencies unless there is a strong reason
- keep optional backend imports optional
- do not use raw `eval`
- keep fixes local unless a shared helper clearly reduces duplication
- do not weaken checker strictness just to get green results

## Docs standards

When editing docs:

- remove or correct stale root-level factory claims
- document async support of `@shapix.check`
- clarify `Scalar` is only valid by itself
- clarify datetime/timedelta aliases accept unit-qualified NumPy dtypes
- clarify numeric scalar aliases exclude booleans
- describe shapix as supporting pyright, mypy, and ty once the unified suite proves it
- replace pyright-specific or pyright/Pylance-only wording with checker-agnostic language where parity has been achieved

Also check for duplicated text in:
- `README.md`
- root module docstring in `src/shapix/__init__.py`
- backend module docstrings
- examples / notebook snippets
- `tests/typing/*` file headers and comments

## What not to do

- do not widen the root API just to satisfy docs drift
- do not silently reinterpret invalid shape syntax
- do not “fix” endianness semantics unless a test proves the current contract is wrong
- do not broaden expression evaluation beyond the current constrained model
- do not preserve a pyright-only bucket for flagship annotations
- do not weaken strict checker settings or hide failures with broad ignores
- do not make root import depend on NumPy
- do not make speculative CI or packaging churn without a concrete incompatibility
- do not advertise full checker parity before the unified suite actually passes

## Done definition for this patch series

A patch set is done only when:

1. The async decorator blocker is fixed.
2. Mixed `Scalar` specs fail fast.
3. `DT64` / `TD64` accept real NumPy datetime/timedelta arrays.
4. Numeric scalar semantics reject booleans consistently.
5. Root import survives missing package metadata.
6. Docs match actual exports and runtime behavior.
7. The flagship typing suite passes on pyright, mypy, and ty.
8. `tests/test_typecheck.py` no longer treats the strongest annotation files as pyright-only.
9. Docs/examples no longer carry stale pyright-only support wording.
10. Targeted tests, unified type-check tests, and at least one full dev run pass.

If there is any remaining known limitation, document it explicitly in code comments and docs instead of leaving it implicit.
