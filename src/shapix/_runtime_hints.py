"""Shared runtime hint helpers for beartype integration."""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass

__all__ = [
  "ReplayFailureState",
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


@dataclass(slots=True)
class ReplayFailureState:
  """One stored validator failure for beartype's error-message replay.

  Beartype may re-invoke ``__instancecheck__`` after an initial failure while it
  formats the final violation message. When the original failure happened under
  a tagged memo, replaying the boolean check would otherwise see an empty memo
  and incorrectly pass. This helper preserves the original failure detail for
  that narrow replay window and clears itself as soon as the context stops
  matching the original failing call.
  """

  obj: object | None = None
  memo: object | None = None
  detail: ValidationFailure | None = None

  def detail_for(self, obj: object) -> ValidationFailure | None:
    if self.obj is obj:
      return self.detail
    return None

  def record(self, obj: object, memo: object, failure: ValidationFailure) -> None:
    self.obj = obj
    self.memo = memo
    self.detail = failure

  def clear(self) -> None:
    self.obj = None
    self.memo = None
    self.detail = None

  def should_replay(self, obj: object) -> bool:
    if self.obj is None or self.obj is not obj:
      return False

    from ._memo import get_memo, has_untagged_memo

    if has_untagged_memo():
      self.clear()
      return False

    current_memo = get_memo()
    if current_memo is self.memo:
      self.clear()
      return False
    if current_memo.single or current_memo.variadic or current_memo.structures:
      self.clear()
      return False
    return True


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
  return tp.cast("_RuntimeValidator", validator)


def _require_runtime_validator(hint: object) -> _RuntimeValidator:
  validator = get_runtime_validator(hint)
  if validator is None:
    msg = f"{hint!r} is not a shapix runtime hint"
    raise TypeError(msg)
  return validator


def hint_label(hint: object) -> str:
  validator = get_runtime_validator(hint)
  if validator is not None:
    return repr(validator)
  return repr(hint)


class _ShapixRuntimeHintMeta(type):
  def __instancecheck__(cls, obj: object) -> bool:
    validator = _require_runtime_validator(cls)
    return validator.instancecheck(obj)

  def __instancecheck_str__(cls, obj: object) -> str:  # noqa: PLW3201
    validator = _require_runtime_validator(cls)
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
