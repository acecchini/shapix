# Frame Resolution and Validator Diagnostics Plan

This document captures the next runtime-hardening work for shapix. It is
intentionally planning-only. This branch does not implement runtime changes
yet.

Related public reference:
- beartype discussion on `__instancecheck_str__()`: <https://github.com/beartype/beartype/discussions/317>

## Goals

- Preserve the public annotation surface:
  - `F32[N, C]`
  - `F32Like[...]`
  - `Tree[...]`
  - `@shapix.check`
  - `check_context()`
- Replace brittle internals before they fail under future beartype wrapper
  changes.
- Expose human-readable tensor and tree violation messages instead of the
  current boolean-only `False == beartype.vale.Is[...]` output.
- Keep pyright, mypy, and ty support intact.

## Findings From Current Code

### 1. Frame resolution is only partially hardened

`src/shapix/_memo.py` currently has two depth-sensitive entry points:

- `get_memo(_depth=...)`
- `get_scope(_depth=...)`

The two functions do not have the same failure profile:

- `get_memo()` uses the passed depth as a starting point and then manages a
  frame-keyed stack. This is hacky, but the implementation already tolerates
  some variation in the stack layout.
- `get_scope()` assumes that the frame at the exact passed depth is the beartype
  wrapper frame, then looks for `__beartype_func`, `args`, and `kwargs` in that
  frame. That is the brittle part that needs hardening.

Current call sites hardcode `_depth=3` from:

- `src/shapix/_array_types.py`
- `src/shapix/_tree.py`

That assumption is not a stable contract.

### 2. Shape diagnostics are generated and then discarded

`src/shapix/_shape.py` already produces readable failure strings through
`check_shape(...)`.

Those strings are then collapsed to booleans in runtime validators:

- `_ArrayChecker.__call__()`
- `_ArrayLikeChecker._check()`

The current pattern is effectively:

```python
result = check_shape(...) == ""
```

That means users see generic beartype validator failures instead of the useful
shape mismatch details already computed internally.

### 3. The current runtime hook is the wrong abstraction for rich messages

The array and tree factories currently return runtime hints based on:

- `typing.Annotated[...]`
- `beartype.vale.Is[...]`

That path works for boolean validation, but it is a poor fit for custom
violation text. Beartype's metaclass plugin API points to a different
extension point:

- `__instancecheck__()` on a metaclass for validation
- `__instancecheck_str__()` on that same metaclass for human-readable failures

This implies a runtime architecture change for the array and tree validators.

### 4. Replay-guard machinery is duplicated and probably transitional

All three runtime checker paths currently carry replay state:

- `_ArrayChecker`
- `_ArrayLikeChecker`
- `_TreeChecker`

They all maintain variants of:

- `_fail_obj`
- `_fail_memo`
- `_fail_replays`

This exists because the current `Is[...]` path forces shapix to survive
beartype's error-generation replays without losing cross-argument memo state.

Once shapix moves onto the metaclass plugin API, this machinery needs to be
re-evaluated rather than copied forward unchanged.

### 5. The change is runtime-internal, but typing compatibility is still a hard requirement

The backend modules currently separate runtime behavior from static behavior
with `TYPE_CHECKING` aliases. That means the refactor must preserve:

- the runtime surface users already write
- the checker-friendly aliases in `src/shapix/numpy.py`
- the analogous backend typing surfaces in `src/shapix/jax.py`,
  `src/shapix/torch.py`, and `src/shapix/cupy.py`
- the unified pyright, mypy, and ty test suite in `tests/test_typecheck.py`

This refactor should not weaken the public syntax just to simplify runtime
internals.

## Required Refactor Workstreams

### A. Replace exact-depth scope lookup with iterative frame search

Refactor `src/shapix/_memo.py` so that frame-sensitive lookup is based on
searching upward for the nearest valid beartype wrapper context rather than
blindly indexing into one assumed frame depth.

Expected changes:

- Add a shared helper that walks up frames looking for the wrapper locals shapix
  actually needs, such as `__beartype_func`, `args`, and `kwargs`.
- Reuse that helper from `get_scope()`.
- Audit whether `get_memo()` should also switch from a caller-provided exact
  depth to a common frame-resolution helper, even if its current behavior is
  less fragile.
- Remove or minimize `_depth=3` constants at the call sites in
  `src/shapix/_array_types.py` and `src/shapix/_tree.py`.

Expected deletions:

- Magic call-site stack-depth assumptions that are treated like a stable API.

### B. Introduce a first-class failure-detail pipeline

Refactor the validator flow so that dtype, shape, conversion, and tree-structure
failures produce structured diagnostics instead of only booleans.

Expected changes:

- Keep `check_shape()` as the canonical source of shape mismatch text, or lift
  it to a small failure object if a structured form is more useful.
- Add a consistent internal representation for validator failure details.
- Preserve memo rollback semantics when validation fails.
- Make sure `Value(...)` resolution failures, rank mismatches, symbolic
  evaluation failures, and structure mismatches all produce explicit messages.

Expected deletions:

- Boolean-only shape checking paths that throw away the message returned by
  `check_shape()`.

### C. Migrate array and tree runtime hints off `Is[...]` where message quality matters

Refactor the runtime factories to use custom classes or metaclasses that expose:

- `__instancecheck__()`
- `__instancecheck_str__()`

This work applies to:

- strict array annotations from `make_array_type()`
- array-like annotations from `make_array_like_type()`
- tree annotations from `src/shapix/_tree.py`

The public syntax should stay unchanged. The internal representation of the
runtime type hints is what changes.

Expected changes:

- Add internal runtime hint types dedicated to beartype integration.
- Preserve current `repr()` quality so error messages still name the same public
  hint surface.
- Keep backend imports optional and scoped the same way they are today.

Expected deletions:

- Runtime dependence on `beartype.vale.Is[...]` for the array and tree paths.

### D. Re-evaluate or remove replay guards after the metaclass migration

The `_fail_obj` / `_fail_memo` / `_fail_replays` pattern should not be treated
as permanent architecture.

Expected changes:

- Verify how beartype invokes `__instancecheck__()` and
  `__instancecheck_str__()` in failure scenarios.
- Decide whether the current replay guard becomes unnecessary, can be reduced to
  a shared helper, or needs to survive in a smaller form.
- Avoid carrying three copies of near-identical replay logic forward if the new
  plugin path makes that unnecessary.

Expected deletions:

- Duplicated replay-guard code in `_ArrayChecker`, `_ArrayLikeChecker`, and
  `_TreeChecker`, if the new API makes it redundant.

### E. Keep typing and public surface stable throughout

The implementation must preserve:

- the checker-only aliases in backend modules
- decorator typing behavior in `src/shapix/_decorator.py`
- root import boundaries
- backend optionality

This is not a license to simplify by narrowing supported syntax.

## Test Work Required

### Runtime regression coverage

Add or update focused tests for:

- frame lookup through plain `@beartype`
- frame lookup through `@shapix.check`
- frame lookup for return validation using `Value(...)`
- nested calls where outer and inner wrappers should not share the wrong scope
- sync and async decorated functions
- tree validation paths that share memo and scope with leaf checks

### Message-quality coverage

Add new assertions that the violation text contains the real failure reason for:

- named-dimension mismatch
- rank mismatch
- fixed-dimension mismatch
- `Value(...)` mismatch
- symbolic-expression evaluation failure
- array-like conversion failure
- tree-structure mismatch

These tests should replace the current effective contract of "the message is not
contradictory" with the stronger contract of "the message is actually useful."

### Typing coverage

Re-run and keep green:

- `tests/test_typecheck.py`
- pyright on `src` and `tests/typing`
- mypy on `src` and `tests/typing`
- ty on `src` and `tests/typing`

## Recommended Implementation Sequence

1. Harden frame and scope resolution first, with targeted memo and decorator
   regressions.
2. Introduce internal failure-detail plumbing without changing the public API.
3. Migrate strict array hints onto the metaclass plugin API and add readable
   message assertions.
4. Migrate array-like hints onto the same infrastructure.
5. Migrate tree hints and reconcile memo-sharing behavior with the new message
   path.
6. Remove obsolete replay code and stale comments once the new flow is proven.
7. Update README and module docstrings in the same implementation series so docs
   match the verified behavior.

## Non-Goals For The First Implementation PR

- Do not change user-facing annotation syntax.
- Do not widen the root `shapix` import surface.
- Do not broaden `Value(...)` expression semantics.
- Do not opportunistically refactor scalar-like aliases unless the metaclass
  migration clearly needs shared infrastructure from that area.

## Files Expected To Change In Later Implementation PRs

Primary runtime work:

- `src/shapix/_memo.py`
- `src/shapix/_array_types.py`
- `src/shapix/_tree.py`
- potentially a new internal helper module for beartype runtime hint classes

Test work:

- `tests/test_memo.py`
- `tests/test_decorator.py`
- `tests/test_numpy.py`
- `tests/test_jax.py`
- `tests/test_torch.py`
- `tests/test_cupy.py`
- `tests/test_tree.py`
- `tests/test_typecheck.py`

Docs to update when code lands:

- `README.md`
- `src/shapix/__init__.py`
- backend module docstrings where the runtime implementation details are now out
  of date
