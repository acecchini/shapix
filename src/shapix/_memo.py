"""Thread-safe memo management for cross-argument dimension consistency.

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
from dataclasses import dataclass, field

__all__ = [
    "ShapeMemo",
    "get_memo",
    "push_memo",
    "pop_memo",
]


@dataclass
class ShapeMemo:
    """Tracks bound dimension names within a single function-call scope."""

    single: dict[str, int] = field(default_factory=dict)
    """Named dimension bindings: ``{"batch": 32, "channels": 3}``."""

    variadic: dict[str, tuple[bool, tuple[int, ...]]] = field(default_factory=dict)
    """Variadic dimension bindings: ``{"spatial": (False, (28, 28))}``."""


_local = threading.local()

# ---------------------------------------------------------------------------
# Explicit stack (used by @shapix.check / check_context)
# ---------------------------------------------------------------------------

def push_memo() -> ShapeMemo:
    """Push a fresh memo onto the explicit stack. Pair with :func:`pop_memo`."""
    stack = _get_explicit_stack()
    memo = ShapeMemo()
    stack.append(memo)
    return memo


def pop_memo() -> None:
    """Pop the most recent explicit memo."""
    _get_explicit_stack().pop()


def _get_explicit_stack() -> list[ShapeMemo]:
    try:
        return _local.explicit_stack  # type: ignore[has-type]
    except AttributeError:
        stack: list[ShapeMemo] = []
        _local.explicit_stack = stack
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

    if not hasattr(_local, "frame_stack"):
        _local.frame_stack = []  # list[tuple[int, int, ShapeMemo]]

    # Stack entries: (frame_id, max_lasti, memo)
    stack: list[tuple[int, int, ShapeMemo]] = _local.frame_stack
    lasti: int = frame.f_lasti

    # Fast path: same frame as last check (next param in same call)
    if stack and stack[-1][0] == frame_id:
        _, prev_lasti, prev_memo = stack[-1]
        if lasti >= prev_lasti:
            # Same call, advancing through params — update max_lasti
            stack[-1] = (frame_id, lasti, prev_memo)
            return prev_memo
        # f_lasti went backwards → frame-id was reused (new call to same fn).
        # Discard the stale entry and fall through to create a fresh memo.
        stack.pop()

    # Check deeper in stack (returning to outer call after inner completed)
    for i in range(len(stack) - 2, -1, -1):
        if stack[i][0] == frame_id:
            _, prev_lasti, prev_memo = stack[i]
            if lasti >= prev_lasti:
                del stack[i + 1 :]
                stack[i] = (frame_id, lasti, prev_memo)
                return prev_memo
            # Frame-id reuse at a deeper level — discard everything from i onward
            del stack[i:]
            break

    # New call context — clean stale entries when stack grows large
    if len(stack) > 8:
        active: set[int] = set()
        f = sys._getframe(0)
        while f is not None:
            active.add(id(f))
            f = f.f_back
        stack[:] = [(fid, li, m) for fid, li, m in stack if fid in active]

    memo = ShapeMemo()
    stack.append((frame_id, lasti, memo))
    return memo


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
