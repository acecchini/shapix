# pyright: reportArgumentType=false, reportGeneralTypeIssues=false
"""Tests for _decorator.py — @shapix.check and check_context."""

from __future__ import annotations

import numpy as np
import pytest
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation

import shapix
from shapix import N, C
from shapix.numpy import F32


class TestCheckDecorator:
  def test_basic_usage(self) -> None:
    @shapix.check
    @beartype
    def f(x: F32[N, C], y: F32[N, C]) -> F32[N, C]:
      return x + y

    result = f(np.ones((4, 3), dtype=np.float32), np.ones((4, 3), dtype=np.float32))
    assert result.shape == (4, 3)

  def test_cross_arg_mismatch(self) -> None:
    @shapix.check
    @beartype
    def f(x: F32[N], y: F32[N]) -> F32[N]:
      return x + y

    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones((3,), dtype=np.float32), np.ones((5,), dtype=np.float32))

  def test_with_conf(self) -> None:
    from beartype import BeartypeConf

    @shapix.check(conf=BeartypeConf())
    def f(x: F32[N]) -> F32[N]:
      return x

    result = f(np.ones((5,), dtype=np.float32))
    assert result.shape == (5,)

  def test_sequential_calls(self) -> None:
    @shapix.check
    @beartype
    def g(x: F32[N]) -> F32[N]:
      return x

    g(np.ones((3,), dtype=np.float32))
    g(np.ones((7,), dtype=np.float32))


class TestCheckContext:
  def test_isinstance_check(self) -> None:
    from beartype.door import is_bearable

    arr = np.ones((4, 3), dtype=np.float32)
    with shapix.check_context():
      assert is_bearable(arr, F32[N, C])

  def test_isinstance_cross_check(self) -> None:
    from beartype.door import is_bearable

    x = np.ones((4, 3), dtype=np.float32)
    y = np.ones((4, 5), dtype=np.float32)
    with shapix.check_context():
      assert is_bearable(x, F32[N, C])
      # y has N=4 (matches) but different C=5 (should fail)
      assert not is_bearable(y, F32[N, C])

  def test_context_cleanup(self) -> None:
    """After exiting the context, a new context gets a fresh memo."""
    from beartype.door import is_bearable

    with shapix.check_context():
      x = np.ones((4,), dtype=np.float32)
      assert is_bearable(x, F32[N])

    # New context — N should not be bound to 4
    with shapix.check_context():
      y = np.ones((10,), dtype=np.float32)
      assert is_bearable(y, F32[N])

  def test_check_context_returns_self(self) -> None:
    """check_context() works as a context manager and returns itself."""
    ctx = shapix.check_context()
    result = ctx.__enter__()
    assert result is ctx
    ctx.__exit__(None, None, None)


class TestDecoratorEdgeCases:
  def test_exception_cleanup(self) -> None:
    """@check cleans memo even when the decorated function raises."""

    @shapix.check
    @beartype
    def boom(x: F32[N]) -> F32[N]:
      raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
      boom(np.ones(5, dtype=np.float32))

    # Should not leak memo state — next call should work independently
    @shapix.check
    @beartype
    def ok(x: F32[N]) -> F32[N]:
      return x

    ok(np.ones(3, dtype=np.float32))

  def test_metadata_preserved(self) -> None:
    """@check preserves __name__ and __doc__."""

    @shapix.check
    @beartype
    def my_func(x: F32[N]) -> F32[N]:
      """My docstring."""
      return x

    assert my_func.__name__ == "my_func"
    assert my_func.__doc__ == "My docstring."

  def test_nested_check_context(self) -> None:
    """Two nested contexts have independent bindings."""
    from beartype.door import is_bearable

    with shapix.check_context():
      assert is_bearable(np.ones(4, dtype=np.float32), F32[N])
      with shapix.check_context():
        # Inner context: N should not be bound to 4
        assert is_bearable(np.ones(10, dtype=np.float32), F32[N])
      # Outer context: N should still be 4
      assert not is_bearable(np.ones(10, dtype=np.float32), F32[N])

  def test_empty_check_context(self) -> None:
    """Enter/exit with no checks should not raise."""
    with shapix.check_context():
      pass
