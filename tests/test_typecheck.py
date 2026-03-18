"""Type-checker compatibility tests.

Runs pyright, mypy, and ty on sample files under ``tests/typing/`` to verify
that shapix's public API type-checks cleanly across all three checkers.

All flagship annotation files are tested by all checkers.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

TYPING_DIR = Path(__file__).parent / "typing"

# ---------------------------------------------------------------------------
# All files tested by all three checkers
# ---------------------------------------------------------------------------

ALL_FILES = [
  "check_imports.py",
  "check_decorator.py",
  "check_like_types.py",
  "check_make_array_type.py",
  "check_tree.py",
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
# Pyright
# ---------------------------------------------------------------------------


@pytest.mark.typecheck
class TestPyright:
  @pytest.mark.parametrize("filename", ALL_FILES)
  def test_pyright(self, filename: str) -> None:
    _skip_if_missing("pyright")
    result = _run(["pyright", str(TYPING_DIR / filename)])
    assert result.returncode == 0, (
      f"pyright failed on {filename}:\n{result.stdout}\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Mypy
# ---------------------------------------------------------------------------


@pytest.mark.typecheck
class TestMypy:
  @pytest.mark.parametrize("filename", ALL_FILES)
  def test_mypy(self, filename: str) -> None:
    _skip_if_missing("mypy")
    result = _run(["mypy", str(TYPING_DIR / filename)])
    assert result.returncode == 0, (
      f"mypy failed on {filename}:\n{result.stdout}\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# ty
# ---------------------------------------------------------------------------


@pytest.mark.typecheck
class TestTy:
  @pytest.mark.parametrize("filename", ALL_FILES)
  def test_ty(self, filename: str) -> None:
    _skip_if_missing("ty")
    result = _run(["ty", "check", str(TYPING_DIR / filename)])
    assert result.returncode == 0, (
      f"ty failed on {filename}:\n{result.stdout}\n{result.stderr}"
    )
