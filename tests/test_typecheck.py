"""Type-checker compatibility tests.

Runs pyright, mypy, and ty on sample files under ``tests/typing/`` to verify
that shapix's public API type-checks cleanly across multiple checkers.

These tests require ``pyright``, ``mypy``, and ``ty`` to be installed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

TYPING_DIR = Path(__file__).parent / "typing"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(
  cmd: list[str], *, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
  return subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=cwd)


def _skip_if_missing(tool: str) -> None:
  if shutil.which(tool) is None:
    pytest.skip(f"{tool} not installed")


# ---------------------------------------------------------------------------
# Tests — common (all checkers should pass)
# ---------------------------------------------------------------------------

COMMON_FILES = ["check_imports.py", "check_decorator.py"]


@pytest.mark.typecheck
@pytest.mark.parametrize("filename", COMMON_FILES)
class TestPyright:
  def test_pyright(self, filename: str) -> None:
    _skip_if_missing("pyright")
    result = _run(["pyright", str(TYPING_DIR / filename)])
    assert result.returncode == 0, f"pyright failed:\n{result.stdout}\n{result.stderr}"


@pytest.mark.typecheck
@pytest.mark.parametrize("filename", COMMON_FILES)
class TestMypy:
  def test_mypy(self, filename: str) -> None:
    _skip_if_missing("mypy")
    result = _run(["mypy", str(TYPING_DIR / filename), "--ignore-missing-imports"])
    assert result.returncode == 0, f"mypy failed:\n{result.stdout}\n{result.stderr}"


@pytest.mark.typecheck
@pytest.mark.parametrize("filename", COMMON_FILES)
class TestTy:
  def test_ty(self, filename: str) -> None:
    _skip_if_missing("ty")
    result = _run(["ty", "check", str(TYPING_DIR / filename)])
    assert result.returncode == 0, f"ty failed:\n{result.stdout}\n{result.stderr}"


# ---------------------------------------------------------------------------
# Tests — pyright-only (full F32[N, C] annotation pattern)
# ---------------------------------------------------------------------------

PYRIGHT_FILES = ["check_annotations.py"]


@pytest.mark.typecheck
@pytest.mark.parametrize("filename", PYRIGHT_FILES)
class TestPyrightAnnotations:
  def test_pyright_annotations(self, filename: str) -> None:
    _skip_if_missing("pyright")
    result = _run(["pyright", str(TYPING_DIR / filename)])
    assert result.returncode == 0, f"pyright failed:\n{result.stdout}\n{result.stderr}"
