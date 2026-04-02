# Shapix

Runtime shape and dtype checking for NumPy, JAX, PyTorch, CuPy, and tree-structured containers, powered by beartype.

> Engineering posture: treat Shapix as a correctness-sensitive library with a small public surface, strict optional-dependency boundaries, and docs/tests that define the product contract.

## Mission

Shapix turns annotations such as `F32[N, C]`, `F32Like[~B, C]`, `F32[Value("size")]`, and `Tree[F32[N], T]` into runtime-validated contracts while keeping a coherent static typing story across supported checkers.

The repository has two equal product surfaces:

1. **Runtime behavior**: validators, decorators, import boundaries, and error messages.
2. **Static typing behavior**: the annotation syntax that pyright, mypy, and ty are expected to accept.

When working in this repo, optimize for:

1. **Runtime correctness and readability of failures**
2. **Public API and annotation-syntax stability**
3. **Optional-dependency safety and lightweight imports**
4. **Static type-checker coherence across pyright, mypy, and ty**
5. **Docs, examples, and typing fixtures that match real behavior**
6. **Focused regression coverage for risky changes**
7. **Small, explicit fixes over clever rewrites**

## Repository map

```text
src/shapix/
├── __init__.py        # Root public API and root module docstring
├── _dimensions.py     # Dimension symbols, Scalar, Value(...), arithmetic, unary markers
├── _shape.py          # Shape-spec types, expression validation/evaluation, rank matching
├── _dtypes.py         # DtypeSpec, dtype extraction, byte-order and structured dtype logic
├── _array_types.py    # Strict array and ArrayLike factories plus validator plumbing
├── _memo.py           # Frame-based memo discovery + explicit memo stack
├── _decorator.py      # @shapix.check and check_context
├── _tree.py           # Tree validators and structure binding
├── numpy.py           # NumPy aliases, Like aliases, ScalarLike aliases, Structured
├── jax.py             # JAX aliases, JAX Like aliases, Tree, optional import boundary
├── torch.py           # Torch aliases, Torch Like aliases, optional import boundary
├── cupy.py            # CuPy aliases, CuPy Like aliases, optional import boundary
├── optree.py          # optree-backed Tree
└── claw.py            # beartype.claw convenience wrapper

tests/
├── test_decorator.py  # @check, check_context, async/sync decorator semantics
├── test_memo.py       # Frame-based and explicit memo behavior
├── test_dimensions.py # Dimension syntax and Value(...) behavior
├── test_shape.py      # Internal shape-matching behavior
├── test_dtypes.py     # Dtype extraction and DtypeSpec matching
├── test_numpy.py      # NumPy aliases, Like types, ScalarLike, structured dtypes
├── test_jax.py        # JAX-specific runtime behavior
├── test_torch.py      # Torch-specific runtime behavior
├── test_cupy.py       # CuPy-specific runtime behavior
├── test_tree.py       # Tree validation and structure matching
├── test_coverage_edges.py
├── test_typecheck.py  # pyright, mypy, ty integration
└── typing/            # Public typing fixtures, treated as contract

docs/                  # User-facing docs site
README.md              # High-level product contract
CONTRIBUTING.md        # Dev workflow and tox matrix notes
CHANGELOG.md           # User-visible change log
tox.toml               # Version/back-end/checker test matrix
pyproject.toml         # Packaging metadata and checker settings
```

## Enduring project rules

### 1) Announced behavior must be real behavior

- If README, docs, examples, docstrings, tests, or typing fixtures imply support for something, verify that the code actually provides it.
- If code, tests, and docs disagree, resolve the contract explicitly. Do not leave the mismatch implicit.
- Prefer narrowing docs over widening the API when the existing import boundary or semantics are intentional.

### 2) Public syntax must stay stable

- Preserve the flagship annotation surface users actually write: named dims, `Scalar`, anonymous dims, fixed integer dims, arithmetic dims, `Value(...)`, variadic dims, broadcastable dims, backend aliases, `Like[...]`, and `Tree[...]`.
- Invalid or ambiguous syntax must fail clearly at hint construction time, not be silently reinterpreted.
- Do not weaken public examples just to satisfy one checker unless the repo-wide typing contract has changed intentionally.

### 3) Optional dependency boundaries are part of the API

- `import shapix` must stay lightweight and must not require NumPy.
- Backend modules may require their own runtime dependencies and should fail with clear install hints.
- Do not make unrelated imports start depending on optional backends.
- Keep backend-specific conversion behavior backend-scoped rather than centralizing it into a generic path that widens imports.

### 4) Runtime semantics must be explicit and unsurprising

- Strict array types validate instance type, dtype, and shape.
- `Like` types validate convertibility using backend-aware conversion and casting rules.
- `ScalarLike` types validate scalar values and ranges; they are not just aliases for 0-D arrays.
- Tree annotations validate leaf types and optionally structure bindings.
- Error messages are part of usability. Preserve or improve specificity when changing failure paths.

### 5) Static typing is a maintained product surface

- The supported checker story is whatever `tests/test_typecheck.py` and `tests/typing/*` prove for pyright, mypy, and ty.
- Prefer checker-agnostic `TYPE_CHECKING` scaffolding over checker-specific hacks.
- Keep narrow, annotation-local ignores or checker-only aliases local to runtime-only syntax. Do not weaken global checker settings to make a change pass.

### 6) Docs, examples, and typing fixtures are contract artifacts

- Treat `README.md`, `docs/`, module docstrings, notebooks, and `tests/typing/*` as part of the public API surface.
- Keep import examples accurate with respect to root-vs-backend ownership.
- Replace temporary or release-specific wording with durable behavioral guidance when editing docs.

## Public API boundaries

### Root module (`shapix`)

The root module is intentionally small. It exports:

- dimension symbols and `Dimension`
- `Scalar` and `Value`
- structure symbols and `Structure`
- `DtypeSpec`
- `make_array_type`
- `make_array_like_type`
- `check`
- `check_context`
- `__version__`

Do **not** widen the root API just to make examples or docs shorter.

### Backend modules

- `shapix.numpy` owns the broadest runtime surface: strict arrays, Like types, ScalarLike types, `Structured`, `ArrayLike`, and `make_scalar_like_type`.
- `shapix.jax`, `shapix.torch`, and `shapix.cupy` provide backend-specific array and Like aliases, plus NumPy ScalarLike re-exports.
- `Tree` is imported from `shapix.optree` or `shapix.jax`, not from the root module.
- Backend-specific dtype availability and conversion behavior are intentional differences and should stay documented.

### Typing contract

- Leaf-only tree annotations are part of the checker-friendly surface.
- Some runtime syntax is intentionally runtime-only and relies on local ignores or checker-only aliases.
- If you change what is statically supported, update the fixtures and the user-facing typing docs together.

## Intentional semantics to preserve

These are design choices or test-encoded behaviors. Do not change them casually:

- Standard `@beartype` should work with shapix annotations without requiring a custom decorator for normal usage.
- Frame-based memo discovery remains the default cross-argument mechanism for plain `@beartype`.
- `@shapix.check` and `check_context()` exist for explicit memo management and manual `is_bearable()` workflows.
- Memo lifetime must be correct across argument validation, return validation, exceptions, and async paths.
- `Value(...)` uses a constrained expression language; do not broaden it into arbitrary Python evaluation.
- Attribute access is allowed only in the `Value(...)` expression path and should remain constrained.
- Like-type endianness checks are performed on the source object's actual byte order.
- Backend `Like` conversion behavior is backend-specific and tested that way.
- Tree structure arguments are runtime-visible but not fully modeled by static checkers.
- Root import remains NumPy-independent at runtime.

## High-risk areas

### Memo system

`src/shapix/_memo.py` is inherently brittle because it combines frame inspection, thread/task-local behavior, and replay handling for beartype diagnostics.

Rules:

- Preserve thread-safety and task-safety.
- Preserve push/pop symmetry on success and failure.
- Add focused regression tests before changing frame heuristics or memo visibility rules.
- Do not broaden wrapper-frame detection casually.

### Decorator behavior

`src/shapix/_decorator.py` is public API, not convenience glue.

Rules:

- Preserve function metadata and signatures.
- Preserve both decorator modes: `@check` and `@check(conf=...)`.
- Preserve sync and async behavior.
- Preserve explicit memo scope for `Value(...)`.
- Preserve typing fidelity for decorated functions.

### Shape-token parsing and evaluation

`src/shapix/_array_types.py` and `src/shapix/_shape.py` define the contract for how annotations are interpreted.

Rules:

- Reject invalid or ambiguous specs eagerly.
- Never silently drop user tokens.
- Keep symbolic, variadic, broadcastable, and `Value(...)` semantics explicit.
- Keep error messages clear enough that misuse is immediately obvious.

### Dtype normalization

`src/shapix/_dtypes.py` is the dtype bridge across backends.

Rules:

- Normalize platform and backend differences carefully.
- Add tests for any new normalization or byte-order rule.
- Do not regress structured dtype, endianness, or datetime/timedelta behavior while fixing something else.

### Tree validation

`src/shapix/_tree.py` bridges leaf validation, structure binding, and memo sharing.

Rules:

- Preserve memo reuse between outer array checks and tree leaf checks.
- Preserve structure-binding semantics, including wildcard positions.
- Keep the static/runtime split for tree structure arguments explicit in docs and fixtures.

### Static typing scaffolding

`TYPE_CHECKING` branches in backend modules and files under `tests/typing/` are part of the supported surface.

Rules:

- Design for pyright, mypy, and ty together.
- Prefer checker-neutral representations.
- Update type fixtures and source stubs together.
- Do not hide regressions with broad ignores.

## Task-specific expectations

### Runtime bug fixes

- Reproduce the current behavior first.
- Add the smallest failing regression test you can.
- Fix the bug locally without changing unrelated semantics.
- Verify diagnostics if the failure path is user-visible.

### New features or new public syntax

- Make sure the new surface fits the existing import boundaries and naming scheme.
- Add runtime tests in the correct backend or generic test file.
- Add typing coverage if the feature is meant to be checker-visible.
- Update README and docs in the same change if the surface is user-facing.

### Typing work

- Update both source-level `TYPE_CHECKING` scaffolding and `tests/typing/*` as needed.
- Run all three checkers, not just one.
- Keep runtime behavior and static-only aliases clearly separated.

### Docs or examples work

- Verify that examples import from the correct module.
- Keep root-vs-backend boundaries explicit.
- If you clarify a limitation, make sure the limitation is also reflected in tests or fixtures when appropriate.
- Avoid documenting support that the runtime or typing suites do not prove.

### Packaging, import-boundary, or dependency work

- Preserve root import lightness.
- Preserve optional backend imports and clear install hints.
- Check `pyproject.toml`, `tox.toml`, docs, and tests together when changing dependency ranges or supported versions.
- Prefer concrete compatibility fixes over speculative dependency churn.

### User-visible maintenance

- Update `CHANGELOG.md` for user-visible behavior changes when appropriate.
- Keep `CONTRIBUTING.md` and docs aligned with the actual tox matrix and local workflow.

## Test placement rules

This repo filters backend tests in `tests/conftest.py`, so test location matters.

Put tests here:

- generic decorator and memo behavior: `tests/test_decorator.py`, `tests/test_memo.py`
- generic dimensions and shape behavior: `tests/test_dimensions.py`, `tests/test_shape.py`
- generic dtype behavior: `tests/test_dtypes.py`
- NumPy arrays, Like types, ScalarLike, structured dtypes: `tests/test_numpy.py`
- JAX-specific runtime behavior: `tests/test_jax.py`
- Torch-specific runtime behavior: `tests/test_torch.py`
- CuPy-specific runtime behavior: `tests/test_cupy.py`
- tree behavior: `tests/test_tree.py`
- edge and regression coverage that spans internals: `tests/test_coverage_edges.py`
- type-checker integration: `tests/test_typecheck.py` and `tests/typing/*`

## Development workflow

### 1) Understand the current contract

Before changing code, inspect:

- current behavior
- intended behavior
- nearby tests and typing fixtures
- docs and examples that mention the touched surface
- import and dependency boundaries if backends are involved

### 2) Add or update the right regression coverage

Use the narrowest test that proves the behavior:

- runtime tests for runtime semantics
- typing fixtures for checker-visible syntax
- docs updates for user-visible contract changes

### 3) Make the smallest change that restores or extends the contract

Avoid broad refactors unless the tests show they are necessary.

### 4) Update contract artifacts in the same change

If the change is user-visible, review:

- `README.md`
- relevant docs pages in `docs/`
- root or backend module docstrings
- `tests/typing/*`
- `CHANGELOG.md` when appropriate

### 5) Validate at the right level

Suggested commands:

```bash
uv sync

# targeted runtime checks
uv run pytest -n0 tests/test_decorator.py tests/test_memo.py
uv run pytest -n0 tests/test_dimensions.py tests/test_shape.py tests/test_dtypes.py
uv run pytest -n0 tests/test_numpy.py tests/test_jax.py tests/test_torch.py tests/test_cupy.py tests/test_tree.py

# unified type-check contract
uv run pytest -n0 tests/test_typecheck.py
uv run pyright src tests/typing
uv run mypy src tests/typing
uv run ty check src tests/typing

# full locked dev environment
uv run tox run -e dev
```

Use `-n0` for targeted debugging because the default pytest setup uses xdist.

## Coding standards

- Support the Python versions declared in `pyproject.toml`; do not assume only the newest interpreter matters.
- Use 2-space indentation.
- Keep public error messages clear and specific.
- Prefer explicit helpers over dense inline lambdas when semantics matter.
- Avoid new dependencies unless there is a strong reason.
- Keep optional backend imports optional.
- Do not use raw `eval`.
- Keep fixes local unless a shared helper clearly reduces duplication without widening risk.
- Do not weaken checker strictness just to get green results.

## What not to do

- Do not widen the root API just to satisfy docs drift.
- Do not silently reinterpret invalid shape syntax.
- Do not broaden `Value(...)` evaluation beyond the current constrained model.
- Do not regress root import safety by making `import shapix` depend on NumPy or backend packages.
- Do not centralize backend-specific behavior in a way that erases intentional differences.
- Do not preserve stale docs, examples, or typing comments after the code changes.
- Do not advertise support that the runtime and typing suites do not prove.
- Do not paper over checker regressions with broad ignores or relaxed settings.

## Done definition

Work is done when the changed surface is explicit and verified:

1. Runtime behavior is correct and regression-covered where risk warrants it.
2. The public API and import boundary remain coherent.
3. Relevant docs, examples, and typing fixtures match the new reality.
4. Relevant runtime tests and checker runs pass for the touched area.
5. Any remaining limitation is documented explicitly instead of being left implicit.

If a change touches memo handling, decorator behavior, shape parsing, dtype normalization, tree validation, or static typing scaffolding, assume the burden of proof is higher and validate more than the minimum happy path.
