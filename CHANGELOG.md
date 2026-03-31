# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.0.1] - 2026-03-31

### Added

- Runtime shape and dtype validation for NumPy, JAX, PyTorch, and CuPy arrays,
  powered by `typing.Annotated` and beartype validators.
- Symbolic dimension syntax including named dimensions, anonymous dimensions,
  `Scalar`, arithmetic dimension expressions, and constrained `Value(...)`
  checks.
- `Tree[...]` validation helpers, explicit `@shapix.check` support, and
  `check_context()` for shared manual bearability checks.
- Multi-checker typing coverage across pyright, mypy, and ty, with CI, tox, and
  pre-commit validation for runtime and typing behavior.
- A documentation website covering installation, API boundaries, supported
  backends, and practical usage patterns.

### Notes

- The PyPI distribution for this release is `shapix-rt`, where `rt` means
  `runtime`; the import path remains `shapix`.
- The root `shapix` module intentionally stays lightweight and
  optional-dependency-safe; backend-specific aliases and factories live in
  `shapix.numpy`, `shapix.jax`, `shapix.torch`, and `shapix.cupy`.
- CuPy support remains optional at install time and requires a compatible CuPy
  environment when used at runtime.
