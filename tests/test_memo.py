"""Tests for _memo.py — frame-based and explicit memo management."""

from __future__ import annotations

import threading

import numpy as np
import pytest
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation

from shapix import N, C
from shapix._memo import ShapeMemo, get_memo, pop_memo, push_memo
from shapix.numpy import F32


class TestExplicitMemo:
  def test_push_pop(self) -> None:
    memo = push_memo()
    assert isinstance(memo, ShapeMemo)
    assert memo.single == {}
    pop_memo()

  def test_nested_push_pop(self) -> None:
    outer = push_memo()
    outer.single["N"] = 10
    inner = push_memo()
    # Inner memo is independent
    assert inner.single == {}
    inner.single["N"] = 5
    pop_memo()
    # Outer is restored
    assert outer.single["N"] == 10
    pop_memo()

  def test_explicit_takes_priority(self) -> None:
    memo = push_memo()
    # get_memo should return the explicit one
    got = get_memo(_depth=0)
    assert got is memo
    pop_memo()


class TestFrameBasedMemo:
  def test_sequential_calls_get_fresh_memos(self) -> None:
    """Calling the same function twice should not reuse stale bindings."""

    @beartype
    def g(x: F32[N]) -> F32[N]:
      return x

    # First call: N=3
    g(np.ones((3,), dtype=np.float32))
    # Second call: N=7 — must NOT fail against N=3
    g(np.ones((7,), dtype=np.float32))
    # Third call: N=1
    g(np.ones((1,), dtype=np.float32))

  def test_sequential_cross_arg(self) -> None:
    """Sequential calls with multiple args each get independent memos."""

    @beartype
    def f(x: F32[N, C], y: F32[N, C]) -> F32[N, C]:
      return x + y

    f(np.ones((2, 5), dtype=np.float32), np.ones((2, 5), dtype=np.float32))
    f(np.ones((10, 20), dtype=np.float32), np.ones((10, 20), dtype=np.float32))

  def test_nested_calls_independent(self) -> None:
    """Nested function calls get their own memos."""

    @beartype
    def inner(x: F32[N]) -> F32[N]:
      return x * 2

    @beartype
    def outer(x: F32[N, C], y: F32[N]) -> F32[N]:
      return inner(y)

    result = outer(np.ones((4, 3), dtype=np.float32), np.ones((4,), dtype=np.float32))
    assert result.shape == (4,)

  def test_cross_arg_mismatch_detected(self) -> None:
    @beartype
    def f(x: F32[N], y: F32[N]) -> F32[N]:
      return x + y

    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones((3,), dtype=np.float32), np.ones((5,), dtype=np.float32))


class TestThreadSafety:
  def test_threads_have_independent_memos(self) -> None:
    """Each thread should have its own memo."""
    results: list[bool] = []
    errors: list[str] = []

    @beartype
    def f(x: F32[N]) -> F32[N]:
      return x

    def worker(size: int) -> None:
      try:
        f(np.ones((size,), dtype=np.float32))
        results.append(True)
      except Exception as e:
        errors.append(str(e))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(1, 6)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()

    assert len(errors) == 0, f"Thread errors: {errors}"
    assert len(results) == 5


class TestMemoEdgeCases:
  def test_bindings_str_empty(self) -> None:
    from shapix._memo import bindings_str

    memo = ShapeMemo()
    assert bindings_str(memo) == ""

  def test_bindings_str_with_structures(self) -> None:
    from shapix._memo import bindings_str

    memo = ShapeMemo(single={"N": 3}, structures={"T": "some_spec"})
    formatted = bindings_str(memo)
    assert "N=3" in formatted
    # structures are not included in bindings_str
    assert "T" not in formatted

  def test_pop_empty_raises(self) -> None:
    from shapix._memo import _get_explicit_stack, pop_memo

    # Ensure stack is empty
    stack = _get_explicit_stack()
    orig_len = len(stack)
    if orig_len == 0:
      with pytest.raises(IndexError):
        pop_memo()
