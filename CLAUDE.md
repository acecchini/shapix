# Shapix

Runtime shape and dtype checking for NumPy, JAX, PyTorch, and CuPy arrays, powered by beartype.

> Engineering posture: **production hardening**. Prefer closing gaps between documented behavior and verified behavior over widening surface area or adding clever abstractions.

## Mission

Shapix turns array annotations such as `F32[N, C]` into runtime-validated contracts using `typing.Annotated` plus beartype validators.

This repository is correctness-sensitive in two dimensions:

1. **Runtime behavior** must be correct and unsurprising.
2. **Static typing behavior** must be coherent across supported checkers.

When working in this repo, optimize for:

1. **Runtime correctness**
2. **Static type-checker parity across pyright, mypy, and ty**
3. **Public API stability**
4. **Accurate docs that match the real supported surface**
5. **Regression-proof tests in the right backend envs**
6. **Small, explicit fixes over clever rewrites**

## Production-hardening priorities

The next iterations should focus on making the library behave exactly as users would expect from the public docs and examples.

Treat the following as standing engineering priorities:

### 1) Announced behavior must be real behavior
- If README, module docstrings, examples, and tests imply a feature exists, verify that it actually works.
- If code, tests, and docs disagree, resolve the contract explicitly. Do not leave the mismatch implicit.
- Prefer narrowing docs over widening the API when optional-dependency boundaries or import costs are involved.

### 2) Public syntax must stay stable
- Preserve the flagship annotation syntax users actually write: `F32[N, C]`, `F32[Scalar]`, `F32[N + 2]`, `F32[__, C]`, backend variants, and `Tree[...]`.
- Do not “solve” compatibility issues by weakening the public syntax or removing strong examples.
- Invalid or ambiguous syntax must fail clearly at construction time, not be silently reinterpreted.

### 3) Runtime semantics must be explicit and unsurprising
- Decorator behavior, dtype matching, shape parsing, scalar-like behavior, and tree validation are contract boundaries.
- Optional backend behavior must remain optional at import time.
- If behavior is intentionally constrained, document the constraint instead of relying on users to infer it.

### 4) Static typing is a product requirement
- The flagship annotation surface should type-check on **pyright, mypy, and ty**.
- Keep checker-specific hacks to a minimum and prefer checker-agnostic `TYPE_CHECKING` scaffolding.
- Do not advertise checker support that the unified type-check suite does not prove.

### 5) Regressions are unacceptable in high-risk areas
- Any bug fix or feature completion in memo handling, decorator behavior, shape parsing, dtype normalization, or typing stubs needs focused regression coverage.
- Prefer adding the failing test first, then making the smallest change that restores the contract.

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
├── cupy.py            # CuPy public surface + typing stubs
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
- Preserve correct behavior for both sync and async callables.
- Preserve static typing fidelity for decorated sync and async functions.

### Shape token parsing
`_array_types._to_shape_spec()` is a contract boundary.

Rules:
- Reject invalid or ambiguous specs eagerly.
- Never silently drop or reinterpret user tokens except where explicitly documented.
- Keep error messages clear enough that misuse is immediately obvious.

### Dtype normalization
`_dtypes.extract_dtype_str()` is the canonical dtype bridge across NumPy, JAX, Torch, and CuPy.

Rules:
- Normalize platform/runtime differences carefully.
- Add tests for every new normalization rule.
- Do not regress structured dtype, endianness, or datetime/timedelta behavior while fixing other cases.

### Static typing surface
The `TYPE_CHECKING` branches and typing fixtures are part of the public API.

Rules:
- Design typing stubs for **all three** supported checkers, not just pyright.
- Prefer checker-agnostic representations over checker-specific omissions.
- Preserve the public syntax users actually write instead of introducing weaker static-only alternatives.
- Keep runtime behavior and static-only scaffolding clearly separated.

### Docs as contract
README, module docstrings, examples, and typing fixtures are part of the public API.

Rules:
- If code/tests/docs disagree, fix the contract explicitly.
- Prefer narrowing docs over widening API when optional-dependency boundaries matter.
- Do not merge docs that overclaim checker support or backend support before the relevant suite passes.
- Remove stale one-off release language once the underlying work is done.

## Intentional semantics to preserve

These behaviors are intentional or at least test-encoded. Do not “fix” them accidentally:

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
- tree symbols and `Structure`
- `DtypeSpec`
- `make_array_type`
- `make_array_like_type`
- `check`
- `check_context`

Do **not** widen the root API just to make docs or examples look nicer.
That works against the tested optional-dependency and lightweight-import boundary.

### Backend modules
- `shapix.numpy` owns `make_scalar_like_type`.
- `shapix.jax`, `shapix.torch`, and `shapix.cupy` may re-export scalar-like types and the scalar factory.
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
- CuPy-specific runtime coverage: `tests/test_cupy.py` if present, otherwise the closest backend-specific file
- optree-specific runtime coverage: `tests/test_tree.py`
- static typing contract: `tests/test_typecheck.py` and `tests/typing/*`

Typing-specific rule:
- the flagship annotation fixtures must be exercised by **all three** checkers
- do not keep a pyright-only bucket for the main supported annotation surface

## Regression expectations

Every fix in a high-risk area needs a regression test.

Expected coverage patterns:

### Decorator and memo changes
- cover sync and async callables when relevant
- cover memo lifetime across argument and return validation
- cover `@check`, `@check(conf=...)`, and signature preservation where relevant

### Shape parsing changes
- cover valid public syntax
- cover invalid or ambiguous forms failing at hint construction time
- verify that no token is silently dropped or reinterpreted

### Dtype and scalar-like changes
- test real backend values and arrays, not only mocked dtype strings
- verify structured dtype, endianness, and datetime/timedelta behavior remain correct
- make boolean acceptance or rejection explicit in tests for numeric and boolean scalar-like aliases

### Import-boundary and packaging changes
- verify `import shapix` still works from a source tree without installed metadata
- verify missing optional backends do not break unrelated imports
- keep root import lightweight and NumPy-independent

### Static typing changes
- run the same flagship annotation files under pyright, mypy, and ty
- include predefined dimensions, `Scalar`, arithmetic dims, anonymous dims, `@check`, backend imports, and `Tree[...]`
- update typing fixtures and comments together so the fixtures describe the current contract accurately

## Development workflow

### 1) Reproduce first
Before changing code, understand:
- current behavior
- intended behavior
- existing tests that encode related semantics
- current checker behavior on the affected typing fixtures

### 2) Add or update the failing regression test
Especially for:
- decorator behavior
- memo behavior
- shape-token parsing
- dtype normalization
- import-boundary drift
- docs/export drift
- checker parity gaps

### 3) Make the smallest fix that restores the contract
Avoid broad refactors unless the tests prove they are needed.

### 4) Update docs in the same change
If a public behavior changes or becomes clarified, docs must land with the code.
If checker support changes, update docs only after the unified checker suite passes for that surface.

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

- make claims that match the real exported surface
- keep root-vs-backend API boundaries explicit
- document important behavioral constraints rather than leaving them implicit
- describe checker support only to the extent the unified suite proves it
- replace stale, temporary release language with durable contract language

Also check for duplicated or stale text in:
- `README.md`
- root module docstring in `src/shapix/__init__.py`
- backend module docstrings
- examples and notebook snippets
- `tests/typing/*` file headers and comments

## What not to do

- do not widen the root API just to satisfy docs drift
- do not silently reinterpret invalid shape syntax
- do not “fix” endianness semantics unless a test proves the current contract is wrong
- do not broaden expression evaluation beyond the current constrained model
- do not preserve a checker-specific bucket for the main supported annotation surface
- do not weaken strict checker settings or hide failures with broad ignores
- do not make root import depend on NumPy
- do not make speculative CI or packaging churn without a concrete incompatibility
- do not advertise support that the actual runtime and type-check suites do not prove

## Done definition for an iteration

An iteration is done only when:

1. The intended runtime behavior is explicit in tests.
2. The affected public API or semantics are documented accurately.
3. Targeted runtime tests pass for the touched area.
4. Relevant typing fixtures pass on pyright, mypy, and ty.
5. Optional-dependency boundaries remain intact.
6. The change does not silently weaken the public syntax or supported surface.
7. Any remaining limitation is documented explicitly instead of left implicit.

If a change touches a high-risk area, assume the burden of proof is higher: add focused regression coverage and validate more than the minimum happy path.
