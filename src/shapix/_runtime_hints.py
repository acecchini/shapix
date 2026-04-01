"""Shared runtime hint helpers for beartype integration."""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

__all__ = [
  "ValidationFailure",
  "get_runtime_validator",
  "hint_label",
  "make_runtime_hint",
]


@dataclass(frozen=True, slots=True)
class ValidationFailure:
  """Readable validation failure detail for beartype-backed runtime hints."""

  message: str

  def __str__(self) -> str:
    return self.message


class _RuntimeValidator(tp.Protocol):
  def instancecheck(self, obj: object) -> bool: ...

  def instancecheck_str(self, obj: object) -> str: ...


def get_runtime_validator(hint: object) -> _RuntimeValidator | None:
  validator = getattr(hint, "__shapix_validator__", None)
  if validator is None:
    return None
  if not hasattr(validator, "instancecheck") or not hasattr(
    validator, "instancecheck_str"
  ):
    return None
  return tp.cast(_RuntimeValidator, validator)


def hint_label(hint: object) -> str:
  validator = get_runtime_validator(hint)
  if validator is not None:
    return repr(validator)
  return repr(hint)


class _ShapixRuntimeHintMeta(type):
  def __instancecheck__(cls, obj: object) -> bool:
    validator = get_runtime_validator(cls)
    assert validator is not None
    return validator.instancecheck(obj)

  def __instancecheck_str__(cls, obj: object) -> str:  # noqa: PLW3201
    validator = get_runtime_validator(cls)
    assert validator is not None
    return validator.instancecheck_str(obj)


def make_runtime_hint(
  name: str, validator: _RuntimeValidator, *, module: str, origin: object
) -> type:
  """Create a beartype-friendly runtime hint class with a custom validator."""
  namespace = {
    "__module__": module,
    "__qualname__": name,
    "__slots__": (),
    "__args__": (origin, validator),
    "__origin__": origin,
    "__metadata__": (validator,),
    "__shapix_validator__": validator,
  }
  return _ShapixRuntimeHintMeta(name, (), namespace)
