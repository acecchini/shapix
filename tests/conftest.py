"""Tox-aware test filtering.

When running under tox with a factor-based env name (e.g. ``py312-bt022-numpy24``),
only the tests relevant to the active backend are collected. When running
``uv run pytest`` directly (no ``TOX_ENV_NAME``), all tests run unfiltered.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  import pytest

_BACKEND_TESTS: dict[str, set[str]] = {
  "numpy": {
    "test_numpy.py",
    "test_dtypes.py",
    "test_decorator.py",
    "test_memo.py",
    "test_shape.py",
    "test_dimensions.py",
    "test_coverage_edges.py",
  },
  "jax": {"test_jax.py", "test_dtypes.py"},
  "torch": {"test_torch.py", "test_dtypes.py"},
  "cupy": {"test_cupy.py", "test_dtypes.py"},
  "optree": {"test_tree.py", "test_dtypes.py"},
}


def pytest_collection_modifyitems(
  config: pytest.Config, items: list[pytest.Item]
) -> None:
  env_name = os.environ.get("TOX_ENV_NAME", "")
  if not env_name or env_name == "dev":
    return

  factors = set(env_name.split("-"))

  if "type" in factors:
    allowed = {"test_typecheck.py"}
  else:
    backend = next(
      (b for b in _BACKEND_TESTS if any(f.startswith(b) for f in factors)), None
    )
    if not backend:
      return
    allowed = _BACKEND_TESTS[backend]

  selected = [i for i in items if i.path.name in allowed]
  deselected = [i for i in items if i.path.name not in allowed]
  items[:] = selected
  if deselected:
    config.hook.pytest_deselected(items=deselected)
