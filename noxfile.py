"""Nox sessions for multi-version testing, linting, and type-checking."""

from __future__ import annotations

import nox

nox.options.default_venv_backend = "uv"
nox.options.reuse_existing_virtualenvs = True

PYTHONS = ["3.12", "3.13", "3.14"]
TORCH_CPU_INDEX = "https://download.pytorch.org/whl/cpu"


@nox.session(python=PYTHONS)
def tests(session: nox.Session) -> None:
  """Core deps only (numpy + beartype)."""
  session.install("-e", ".", "pytest")
  session.run("pytest", "tests/", *session.posargs)


@nox.session(python=PYTHONS)
def tests_torch(session: nox.Session) -> None:
  """With PyTorch."""
  session.install("-e", ".", "pytest")
  session.install(f"--index-url={TORCH_CPU_INDEX}", "torch>=2.6.0")
  session.run("pytest", "tests/", *session.posargs)


@nox.session(python=PYTHONS)
def tests_jax(session: nox.Session) -> None:
  """With JAX."""
  session.install("-e", ".", "pytest", "jax[cpu]>=0.5.0")
  session.run("pytest", "tests/", *session.posargs)


@nox.session(python=PYTHONS)
def tests_all(session: nox.Session) -> None:
  """With all optional deps."""
  session.install("-e", ".", "pytest", "jax[cpu]>=0.5.0")
  session.install(f"--index-url={TORCH_CPU_INDEX}", "torch>=2.6.0")
  session.run("pytest", "tests/", *session.posargs)


@nox.session
def lint(session: nox.Session) -> None:
  """Run ruff linter and formatter checks."""
  session.install("ruff>=0.14.0")
  session.run("ruff", "check", "src/", "tests/")
  session.run("ruff", "format", "--check", "src/", "tests/")


@nox.session
def typecheck(session: nox.Session) -> None:
  """Run pyright type checker."""
  session.install("-e", ".", "pyright>=1.1.408", "pytest")
  session.run("pyright", "src/")
