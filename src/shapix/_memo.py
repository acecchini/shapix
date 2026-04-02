"""Thread-safe and async-safe memo and scope management for shape checking.

Two modes of operation:

1. **Automatic (frame-based)**: When used with standard ``@beartype``, the memo
   context is detected automatically using the beartype wrapper's frame identity.
   All parameter checks within one function call share the same memo.

2. **Explicit (push/pop)**: When used with ``@shapix.check``, the memo is managed
   via an explicit stack stored in a :class:`contextvars.ContextVar`.  This is
   guaranteed correct regardless of call-stack depth.  The stack structure is
   async-task-safe (push/pop in one task cannot mutate another's stack), but
   ``ShapeMemo`` objects are shared by reference — child tasks that inherit an
   active parent context see the same live memo.  For full isolation, each task
   should create its own ``check_context``.

If no memo context exists (e.g. a bare ``isinstance`` check), a temporary memo is
created per check — dimensions are validated individually but not cross-checked.
"""

from __future__ import annotations

import contextvars
import inspect
import sys
import threading
import typing as tp
import types
from dataclasses import dataclass, field

__all__ = [
  "ShapeMemo",
  "get_memo",
  "get_scope",
  "push_memo",
  "pop_memo",
  "has_untagged_memo",
]

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


# ---------------------------------------------------------------------------
# Explicit stack (used by @shapix.check / check_context)
#
# Stored as immutable tuples in ContextVar so that asyncio task fork
# cannot modify the parent's stack (push/pop are isolated).  Note that
# ShapeMemo objects within the tuple are shared by reference; child tasks
# that inherit an active explicit stack see the same live memo.  For full
# task isolation, each task should create its own check_context.
# ---------------------------------------------------------------------------

_explicit_stack: contextvars.ContextVar[tuple[ShapeMemo, ...]] = contextvars.ContextVar(
  "_explicit_stack", default=()
)
_explicit_scope_stack: contextvars.ContextVar[tuple[dict[str, object] | None, ...]] = (
  contextvars.ContextVar("_explicit_scope_stack", default=())
)
_explicit_owner_stack: contextvars.ContextVar[tuple[types.CodeType | None, ...]] = (
  contextvars.ContextVar("_explicit_owner_stack", default=())
)


def push_memo(
  memo: ShapeMemo | None = None,
  *,
  scope: dict[str, object] | None = None,
  owner_code: types.CodeType | None = None,
) -> ShapeMemo:
  """Push a memo onto the explicit stack. Pair with :func:`pop_memo`.

  Parameters
  ----------
  owner_code:
      When set, this entry is only visible to frames whose ``f_code``
      matches.  Untagged entries (``None``) are unconditional and used
      by :class:`check_context` and :class:`_TreeChecker`.
  """
  if memo is None:
    memo = ShapeMemo()
  _explicit_stack.set(_explicit_stack.get() + (memo,))
  _explicit_scope_stack.set(_explicit_scope_stack.get() + (scope,))
  _explicit_owner_stack.set(_explicit_owner_stack.get() + (owner_code,))
  return memo


def pop_memo() -> None:
  """Pop the most recent explicit memo."""
  _explicit_stack.set(_explicit_stack.get()[:-1])
  _explicit_scope_stack.set(_explicit_scope_stack.get()[:-1])
  _explicit_owner_stack.set(_explicit_owner_stack.get()[:-1])


def has_untagged_memo() -> bool:
  """True when an untagged explicit memo is active (check_context scope).

  Untagged entries are pushed by :class:`check_context` (and :class:`_TreeChecker`)
  and are unconditionally visible to all validators.  When such a memo is active,
  beartype's error-generation re-invocation still sees the real memo with bindings,
  so the replay guard is unnecessary.
  """
  stack = _explicit_stack.get()
  if not stack:
    return False
  owners = _explicit_owner_stack.get()
  return owners[-1] is None


# ---------------------------------------------------------------------------
# Frame-based auto-detection (used by Is[_ShapeChecker] validators)
# ---------------------------------------------------------------------------

_local = threading.local()


def _frame_at_or_none(_depth: int) -> types.FrameType | None:
  try:
    return sys._getframe(_depth)
  except ValueError:
    return None


def _iter_frames(_depth: int) -> tp.Iterator[types.FrameType]:
  frame = _frame_at_or_none(_depth)
  while frame is not None:
    yield frame
    frame = frame.f_back


def _is_beartype_wrapper_frame(frame: types.FrameType) -> bool:
  locals_map = frame.f_locals
  fn = locals_map.get("__beartype_func")
  args = locals_map.get("args")
  kwargs = locals_map.get("kwargs")
  if callable(fn) and isinstance(args, tuple) and isinstance(kwargs, dict):
    return True

  # beartype.door.is_bearable() compiles per-call checker functions named like
  # ``__beartype_checker_0`` that expose the validated object as one or more
  # ``__beartype_pith_*`` locals. These frames are the right memo boundary for
  # standalone bearability checks; falling back past them to shared internals
  # can leak stale bindings across unrelated checks.
  return frame.f_code.co_name.startswith("__beartype_checker_") and any(
    name.startswith("__beartype_pith_") for name in locals_map
  )


def _find_beartype_wrapper_frame(*, _depth: int) -> types.FrameType | None:
  for frame in _iter_frames(_depth):
    if _is_beartype_wrapper_frame(frame):
      return frame
  return None


def _nearest_runtime_frame(_depth: int) -> types.FrameType | None:
  frame = _find_beartype_wrapper_frame(_depth=_depth)
  if frame is not None:
    return frame
  return _frame_at_or_none(_depth)


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
  # 1. Explicit stack takes priority (if owner matches or untagged)
  explicit = _explicit_stack.get()
  if explicit:
    owners = _explicit_owner_stack.get()
    owner_code = owners[-1] if owners else None
    if owner_code is None:
      # Untagged entry (check_context / TreeChecker) — always visible
      return explicit[-1]
    # Tagged entry — only visible if the caller's frame matches
    frame = _nearest_runtime_frame(_depth)
    if frame is not None and frame.f_code is owner_code:
      return explicit[-1]
    # Fall through to frame-based detection

  # 2. Frame-based detection
  frame = _find_beartype_wrapper_frame(_depth=_depth)
  if frame is None:
    frame = _frame_at_or_none(_depth)
  if frame is None:
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
    f: types.FrameType | None = sys._getframe(0)
    while f is not None:
      active.add(id(f))
      f = f.f_back
    stack[:] = [(fid, c, li, m) for fid, c, li, m in stack if fid in active]

  memo = ShapeMemo()
  stack.append((frame_id, code, lasti, memo))
  return memo


def get_scope(_depth: int = 2) -> dict[str, object]:
  """Return the current runtime scope for ``Value(...)`` expressions."""
  explicit = _explicit_scope_stack.get()
  if explicit and explicit[-1] is not None:
    owners = _explicit_owner_stack.get()
    owner_code = owners[-1] if owners else None
    if owner_code is None:
      # Untagged entry — always visible
      return explicit[-1]
    # Tagged entry — only visible if the caller's frame matches
    frame = _nearest_runtime_frame(_depth)
    if frame is not None and frame.f_code is owner_code:
      return explicit[-1]
    # Fall through to frame-based detection

  frame = _find_beartype_wrapper_frame(_depth=_depth)
  if frame is None:
    frame = _frame_at_or_none(_depth)
  if frame is None:
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
