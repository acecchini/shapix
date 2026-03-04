"""Type-checker compatibility tests.

Runs pyright, mypy, and ty on sample files under ``tests/typing/`` to verify
that shapix's public API type-checks cleanly across multiple checkers.

Test matrix:
- **Common files** (all checkers): imports, decorator, like types, make_array_type
- **Pyright-only files**: full F32[N, C] annotations (TYPE_CHECKING stubs)

These tests require ``pyright``, ``mypy``, and ``ty`` to be installed.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

TYPING_DIR = Path(__file__).parent / "typing"

# ---------------------------------------------------------------------------
# File categories
# ---------------------------------------------------------------------------

# Files that should pass cleanly on ALL type checkers
COMMON_FILES = [
  "check_imports.py",
  "check_decorator.py",
  "check_like_types.py",
  "check_make_array_type.py",
  "check_tree.py",
]

# Files that only pyright supports (F32[N, C] subscript annotations)
PYRIGHT_ONLY_FILES = [
  "check_annotations.py",
  "check_annotations_jax.py",
  "check_annotations_torch.py",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(
  cmd: list[str], *, cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
  return subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=cwd)


def _skip_if_missing(tool: str) -> None:
  if shutil.which(tool) is None:
    pytest.skip(f"{tool} not installed")


# ---------------------------------------------------------------------------
# Pyright — common + pyright-only files
# ---------------------------------------------------------------------------


@pytest.mark.typecheck
class TestPyright:
  @pytest.mark.parametrize("filename", COMMON_FILES + PYRIGHT_ONLY_FILES)
  def test_pyright(self, filename: str) -> None:
    _skip_if_missing("pyright")
    result = _run(["pyright", str(TYPING_DIR / filename)])
    assert result.returncode == 0, (
      f"pyright failed on {filename}:\n{result.stdout}\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Mypy — common files only
# ---------------------------------------------------------------------------


@pytest.mark.typecheck
class TestMypy:
  @pytest.mark.parametrize("filename", COMMON_FILES)
  def test_mypy(self, filename: str) -> None:
    _skip_if_missing("mypy")
    result = _run(["mypy", str(TYPING_DIR / filename)])
    assert result.returncode == 0, (
      f"mypy failed on {filename}:\n{result.stdout}\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# ty — common files only
# ---------------------------------------------------------------------------


@pytest.mark.typecheck
class TestTy:
  @pytest.mark.parametrize("filename", COMMON_FILES)
  def test_ty(self, filename: str) -> None:
    _skip_if_missing("ty")
    result = _run(["ty", "check", str(TYPING_DIR / filename)])
    assert result.returncode == 0, (
      f"ty failed on {filename}:\n{result.stdout}\n{result.stderr}"
    )
