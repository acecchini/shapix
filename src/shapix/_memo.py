"""Thread-safe memo and scope management for shape checking.

Two modes of operation:

1. **Automatic (frame-based)**: When used with standard ``@beartype``, the memo
   context is detected automatically using the beartype wrapper's frame identity.
   All parameter checks within one function call share the same memo.

2. **Explicit (push/pop)**: When used with ``@shapix.check``, the memo is managed
   via an explicit stack. This is guaranteed correct regardless of call-stack depth.

If no memo context exists (e.g. a bare ``isinstance`` check), a temporary memo is
created per check — dimensions are validated individually but not cross-checked.
"""

from __future__ import annotations

import sys
import threading
import inspect
from dataclasses import dataclass, field

__all__ = ["ShapeMemo", "get_memo", "get_scope", "push_memo", "pop_memo"]

_FRAME_STACK_GC_THRESHOLD = 8


@dataclass
class ShapeMemo:
  """Tracks bound dimension names within a single function-call scope."""

  single: dict[str, int] = field(default_factory=dict)
  """Named dimension bindings: ``{"batch": 32, "channels": 3}``."""

  variadic: dict[str, tuple[bool, tuple[int, ...]]] = field(default_factory=dict)
  """Variadic dimension bindings: ``{"spatial": (False, (28, 28))}``."""

  structures: dict[str, object] = field(default_factory=dict)
  """Tree structure bindings: ``{"T": <TreeSpec>}``."""

  def snapshot(
    self,
  ) -> tuple[
    dict[str, int], dict[str, tuple[bool, tuple[int, ...]]], dict[str, object]
  ]:
    """Capture a copy of all current bindings."""
    return self.single.copy(), self.variadic.copy(), self.structures.copy()

  def restore(
    self,
    snap: tuple[
      dict[str, int], dict[str, tuple[bool, tuple[int, ...]]], dict[str, object]
    ],
  ) -> None:
    """Roll back all bindings to a previous snapshot."""
    single, variadic, structures = snap
    self.single.clear()
    self.single.update(single)
    self.variadic.clear()
    self.variadic.update(variadic)
    self.structures.clear()
    self.structures.update(structures)


_local = threading.local()

# ---------------------------------------------------------------------------
# Explicit stack (used by @shapix.check / check_context)
# ---------------------------------------------------------------------------


def push_memo(
  memo: ShapeMemo | None = None, *, scope: dict[str, object] | None = None
) -> ShapeMemo:
  """Push a memo onto the explicit stack. Pair with :func:`pop_memo`."""
  stack = _get_explicit_stack()
  if memo is None:
    memo = ShapeMemo()
  stack.append(memo)
  _get_explicit_scope_stack().append(scope)
  return memo


def pop_memo() -> None:
  """Pop the most recent explicit memo."""
  _get_explicit_stack().pop()
  _get_explicit_scope_stack().pop()


def _get_explicit_stack() -> list[ShapeMemo]:
  try:
    return _local.explicit_stack  # type: ignore[has-type]
  except AttributeError:
    stack: list[ShapeMemo] = []
    _local.explicit_stack = stack
    return stack


def _get_explicit_scope_stack() -> list[dict[str, object] | None]:
  try:
    return _local.explicit_scope_stack  # type: ignore[has-type]
  except AttributeError:
    stack: list[dict[str, object] | None] = []
    _local.explicit_scope_stack = stack
    return stack


# ---------------------------------------------------------------------------
# Frame-based auto-detection (used by Is[_ShapeChecker] validators)
# ---------------------------------------------------------------------------


def get_memo(_depth: int = 2) -> ShapeMemo:
  """Return the memo for the current checking context.

  Resolution order:

  1. If an explicit memo was pushed (via ``@shapix.check``), use it.
  2. Otherwise, identify the call context from the beartype wrapper frame
     and reuse or create a memo keyed by that frame's identity.
  3. If all else fails, return a fresh temporary memo (no cross-arg checking).

  Parameters
  ----------
  _depth:
      Internal. Number of frames to skip when locating the beartype wrapper.
      Default ``2`` accounts for: our validator → beartype's ``_is_valid_bool``
      → beartype wrapper.
  """
  # 1. Explicit stack takes priority
  explicit = _get_explicit_stack()
  if explicit:
    return explicit[-1]

  # 2. Frame-based detection
  try:
    frame = sys._getframe(_depth)
  except ValueError:
    return ShapeMemo()

  frame_id = id(frame)
  code = frame.f_code

  if not hasattr(_local, "frame_stack"):
    _local.frame_stack = []  # list[tuple[int, object, int, ShapeMemo]]

  # Stack entries: (frame_id, code_obj, max_lasti, memo)
  # We store the code object alongside id(frame) to prevent false matches
  # when CPython recycles a frame object at the same address for a different
  # function (common in parametrised test loops and tight call sequences).
  stack: list[tuple[int, object, int, ShapeMemo]] = _local.frame_stack
  lasti: int = frame.f_lasti

  # Fast path: same frame as last check (next param in same call)
  if stack and stack[-1][0] == frame_id and stack[-1][1] is code:
    _, _, prev_lasti, prev_memo = stack[-1]
    if lasti >= prev_lasti:
      # Same call, advancing through params — update max_lasti
      stack[-1] = (frame_id, code, lasti, prev_memo)
      return prev_memo
    # f_lasti went backwards → frame-id was reused (new call to same fn).
    # Discard the stale entry and fall through to create a fresh memo.
    stack.pop()

  # Check deeper in stack (returning to outer call after inner completed)
  for i in range(len(stack) - 2, -1, -1):
    if stack[i][0] == frame_id and stack[i][1] is code:
      _, _, prev_lasti, prev_memo = stack[i]
      if lasti >= prev_lasti:
        del stack[i + 1 :]
        stack[i] = (frame_id, code, lasti, prev_memo)
        return prev_memo
      # Frame-id reuse at a deeper level — discard everything from i onward
      del stack[i:]
      break

  # New call context — clean stale entries when stack grows large
  if len(stack) > _FRAME_STACK_GC_THRESHOLD:
    active: set[int] = set()
    f = sys._getframe(0)
    while f is not None:
      active.add(id(f))
      f = f.f_back
    stack[:] = [(fid, c, li, m) for fid, c, li, m in stack if fid in active]

  memo = ShapeMemo()
  stack.append((frame_id, code, lasti, memo))
  return memo


def get_scope(_depth: int = 2) -> dict[str, object]:
  """Return the current runtime scope for ``Value(...)`` expressions."""
  explicit = _get_explicit_scope_stack()
  if explicit and explicit[-1] is not None:
    return explicit[-1]

  try:
    frame = sys._getframe(_depth)
  except ValueError:
    return {}

  locals_map = dict(frame.f_locals)

  # beartype wrappers expose runtime arguments as ``args`` / ``kwargs`` plus
  # the wrapped function object. Rebind those to parameter names so
  # ``Value("size")`` and ``Value("self.attr")`` work for both param and
  # return validation.
  fn = locals_map.get("__beartype_func")
  args = locals_map.get("args")
  kwargs = locals_map.get("kwargs")
  if callable(fn) and isinstance(args, tuple) and isinstance(kwargs, dict):
    try:
      bound = inspect.signature(fn).bind_partial(*args, **kwargs)
    except TypeError:
      pass
    else:
      bound.apply_defaults()
      return dict(bound.arguments)

  return locals_map


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------


def bindings_str(memo: ShapeMemo) -> str:
  """Human-readable string of current dimension bindings."""
  parts: list[str] = []
  for name, size in memo.single.items():
    parts.append(f"{name}={size}")
  for name, (_, shape) in memo.variadic.items():
    parts.append(f"~{name}={shape}")
  return ", ".join(parts)
