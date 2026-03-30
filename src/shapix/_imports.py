"""Helpers for optional runtime imports.

These helpers keep backend imports explicit at runtime while avoiding direct
import statements in modules that are intended to remain statically analyzable
in environments where some optional backends are not installed.
"""

from __future__ import annotations

import importlib
import types

__all__ = ["require_attr", "require_module"]


def require_module(module_name: str, *, install_hint: str) -> types.ModuleType:
  """Import an optional module or raise a clear installation error.

  Only wraps ``ModuleNotFoundError`` when the missing module is the requested
  optional dependency itself. Nested import failures from within that module
  are re-raised unchanged so real dependency issues are not masked.
  """
  try:
    return importlib.import_module(module_name)
  except ModuleNotFoundError as exc:
    missing = exc.name or ""
    if not missing and module_name in str(exc):
      raise ModuleNotFoundError(install_hint) from exc
    if missing == module_name or missing.startswith(f"{module_name}."):
      raise ModuleNotFoundError(install_hint) from exc
    raise


def require_attr(module_name: str, attr: str, *, install_hint: str) -> object:
  """Import *module_name* and return *attr* from it."""
  module = require_module(module_name, install_hint=install_hint)
  return getattr(module, attr)
