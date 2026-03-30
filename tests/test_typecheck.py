"""Type-checker compatibility tests.

Runs pyright, mypy, and ty against both:

- sample files under ``tests/typing/`` that exercise the public annotation
  surface and its documented ``TYPE_CHECKING`` workarounds
- the ``src/`` tree itself, so source-level checker regressions are caught too
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TYPING_DIR = ROOT / "tests" / "typing"
WHOLE_TREE_TARGETS = ["src", "tests/typing"]

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
  "check_annotations_cupy.py",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool_path(tool: str) -> str | None:
  env_tool = Path(sys.executable).resolve().parent / tool
  if env_tool.exists():
    return str(env_tool)
  return shutil.which(tool)


def _skip_if_missing(tool: str) -> None:
  if _tool_path(tool) is None:
    pytest.skip(f"{tool} not installed")


def _run(tool: str, *args: str, cwd: Path = ROOT) -> subprocess.CompletedProcess[str]:
  executable = _tool_path(tool)
  assert executable is not None, f"{tool} not installed"
  return subprocess.run(
    [executable, *args], capture_output=True, text=True, timeout=120, cwd=cwd
  )


# ---------------------------------------------------------------------------
# Pyright
# ---------------------------------------------------------------------------


@pytest.mark.typecheck
class TestPyright:
  @pytest.mark.parametrize("filename", ALL_FILES)
  def test_pyright(self, filename: str) -> None:
    _skip_if_missing("pyright")
    result = _run("pyright", str(TYPING_DIR / filename))
    assert result.returncode == 0, (
      f"pyright failed on {filename}:\n{result.stdout}\n{result.stderr}"
    )

  def test_pyright_source_tree(self) -> None:
    _skip_if_missing("pyright")
    result = _run("pyright", *WHOLE_TREE_TARGETS)
    assert result.returncode == 0, (
      f"pyright failed on source tree:\n{result.stdout}\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# Mypy
# ---------------------------------------------------------------------------


@pytest.mark.typecheck
class TestMypy:
  @pytest.mark.parametrize("filename", ALL_FILES)
  def test_mypy(self, filename: str) -> None:
    _skip_if_missing("mypy")
    result = _run("mypy", str(TYPING_DIR / filename))
    assert result.returncode == 0, (
      f"mypy failed on {filename}:\n{result.stdout}\n{result.stderr}"
    )

  def test_mypy_source_tree(self) -> None:
    _skip_if_missing("mypy")
    result = _run("mypy", *WHOLE_TREE_TARGETS)
    assert result.returncode == 0, (
      f"mypy failed on source tree:\n{result.stdout}\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# ty
# ---------------------------------------------------------------------------


@pytest.mark.typecheck
class TestTy:
  @pytest.mark.parametrize("filename", ALL_FILES)
  def test_ty(self, filename: str) -> None:
    _skip_if_missing("ty")
    result = _run("ty", "check", str(TYPING_DIR / filename))
    assert result.returncode == 0, (
      f"ty failed on {filename}:\n{result.stdout}\n{result.stderr}"
    )

  def test_ty_source_tree(self) -> None:
    _skip_if_missing("ty")
    result = _run("ty", "check", *WHOLE_TREE_TARGETS)
    assert result.returncode == 0, (
      f"ty failed on source tree:\n{result.stdout}\n{result.stderr}"
    )
