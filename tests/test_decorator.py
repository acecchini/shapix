# pyright: reportArgumentType=false, reportGeneralTypeIssues=false
"""Tests for _decorator.py — @shapix.check and check_context."""

from __future__ import annotations

import numpy as np
import pytest
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation

import shapix
from shapix import N, C, Value  # noqa: F401
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


class TestValueWithCheckDecorator:
  def test_value_with_shapix_check(self) -> None:
    """Value expressions work with @shapix.check (explicit scope)."""

    @shapix.check
    @beartype
    def f(size: int) -> F32[Value("size")]:  # type: ignore[valid-type]
      return np.ones(size, dtype=np.float32)

    assert f(4).shape == (4,)

  def test_value_with_shapix_check_violation(self) -> None:
    from beartype.roar import BeartypeCallHintReturnViolation

    @shapix.check
    @beartype
    def f(size: int) -> F32[Value("size")]:  # type: ignore[valid-type]
      return np.ones(999, dtype=np.float32)

    with pytest.raises(BeartypeCallHintReturnViolation):
      f(4)

  def test_value_add_dim_with_shapix_check(self) -> None:
    """Value + Dimension arithmetic works under @shapix.check."""

    @shapix.check
    @beartype
    def f(x: F32[N], pad: int) -> F32[N + Value("pad")]:  # type: ignore[valid-type]
      return np.ones(x.shape[0] + pad, dtype=np.float32)

    assert f(np.ones(4, dtype=np.float32), 2).shape == (6,)

  def test_value_add_dim_with_shapix_check_violation(self) -> None:
    from beartype.roar import BeartypeCallHintReturnViolation

    @shapix.check
    @beartype
    def f(x: F32[N], pad: int) -> F32[N + Value("pad")]:  # type: ignore[valid-type]
      return np.ones(999, dtype=np.float32)

    with pytest.raises(BeartypeCallHintReturnViolation):
      f(np.ones(4, dtype=np.float32), 2)

  def test_value_self_attr_with_shapix_check(self) -> None:
    """Value("self.x") works with @shapix.check."""

    class Obj:
      size = 5

      @shapix.check
      @beartype
      def f(self) -> F32[Value("self.size")]:  # type: ignore[valid-type]
        return np.ones(self.size, dtype=np.float32)

    assert Obj().f().shape == (5,)

  def test_value_with_conf(self) -> None:
    """Value works with @shapix.check(conf=...)."""
    from beartype import BeartypeConf

    @shapix.check(conf=BeartypeConf())
    def f(size: int) -> F32[Value("size")]:  # type: ignore[valid-type]
      return np.ones(size, dtype=np.float32)

    assert f(7).shape == (7,)


class TestMemoIsolation:
  """Explicit memo from @shapix.check must not leak to nested @beartype helpers."""

  def test_check_outer_plain_beartype_inner_independent_dims(self) -> None:
    """Outer @shapix.check binds N=4, inner plain @beartype binds N=7."""

    @beartype
    def inner(x: F32[N]) -> F32[N]:
      return x

    @shapix.check
    @beartype
    def outer(x: F32[N]) -> F32[N]:
      # Call inner with a different N — must succeed independently
      inner(np.ones(7, dtype=np.float32))
      return x

    outer(np.ones(4, dtype=np.float32))

  def test_check_outer_plain_beartype_inner_value(self) -> None:
    """Inner plain @beartype with Value resolves from its own frame."""

    @beartype
    def inner(size: int) -> F32[Value("size")]:  # type: ignore[valid-type]
      return np.ones(size, dtype=np.float32)

    @shapix.check
    @beartype
    def outer(dummy: int) -> F32[Value("dummy")]:  # type: ignore[valid-type]
      inner(7)
      return np.ones(dummy, dtype=np.float32)

    outer(4)

  def test_check_context_shares_memo_with_beartype_call(self) -> None:
    """Inside check_context(), @beartype calls share the same memo (by design)."""

    @beartype
    def helper(x: F32[N]) -> F32[N]:
      return x

    with shapix.check_context():
      from beartype.door import is_bearable

      assert is_bearable(np.ones(4, dtype=np.float32), F32[N])
      # check_context is untagged — helper shares the memo, so N=4 is bound
      helper(np.ones(4, dtype=np.float32))  # same N — OK
      with pytest.raises(BeartypeCallHintParamViolation):
        helper(np.ones(7, dtype=np.float32))  # different N — fails

  def test_async_check_outer_plain_beartype_inner(self) -> None:
    """Async variant: outer @shapix.check does not leak to inner @beartype."""
    import asyncio

    @beartype
    def inner(x: F32[N]) -> F32[N]:
      return x

    @shapix.check
    @beartype
    async def outer(x: F32[N]) -> F32[N]:
      inner(np.ones(7, dtype=np.float32))
      return x

    asyncio.run(outer(np.ones(4, dtype=np.float32)))

  def test_async_child_task_independent_memo(self) -> None:
    """Child task spawned inside @shapix.check gets isolated memo after parent returns."""
    import asyncio

    @shapix.check
    @beartype
    async def parent(x: F32[N]) -> F32[N]:
      return x

    async def run() -> None:
      await parent(np.ones(4, dtype=np.float32))
      # After parent returns, a new call with different N works
      await parent(np.ones(10, dtype=np.float32))

    asyncio.run(run())


class TestAsyncCheckContext:
  def test_async_check_context(self) -> None:
    """async with check_context() works correctly."""
    import asyncio

    from beartype.door import is_bearable

    async def run() -> None:
      async with shapix.check_context():
        assert is_bearable(np.ones((4, 3), dtype=np.float32), F32[N, C])
        assert not is_bearable(np.ones((4, 5), dtype=np.float32), F32[N, C])

    asyncio.run(run())

  def test_async_check_context_cleanup(self) -> None:
    """After exiting async context, a new context gets a fresh memo."""
    import asyncio

    from beartype.door import is_bearable

    async def run() -> None:
      async with shapix.check_context():
        assert is_bearable(np.ones((4,), dtype=np.float32), F32[N])

      async with shapix.check_context():
        assert is_bearable(np.ones((10,), dtype=np.float32), F32[N])

    asyncio.run(run())

  def test_async_concurrent_check_context(self) -> None:
    """Overlapping async with check_context() tasks are isolated."""
    import asyncio

    from beartype.door import is_bearable

    async def task_a() -> bool:
      async with shapix.check_context():
        await asyncio.sleep(0)
        return is_bearable(np.ones((4,), dtype=np.float32), F32[N])

    async def task_b() -> bool:
      async with shapix.check_context():
        await asyncio.sleep(0)
        return is_bearable(np.ones((10,), dtype=np.float32), F32[N])

    async def run() -> None:
      a, b = await asyncio.gather(task_a(), task_b())
      assert a
      assert b

    asyncio.run(run())


class TestAsyncCheckDecorator:
  def test_async_basic(self) -> None:
    """Decorated async fn still works, iscoroutinefunction is True."""
    import asyncio
    import inspect

    @shapix.check
    @beartype
    async def f(x: F32[N, C], y: F32[N, C]) -> F32[N, C]:
      return x + y

    assert inspect.iscoroutinefunction(f)
    result = asyncio.run(
      f(np.ones((4, 3), dtype=np.float32), np.ones((4, 3), dtype=np.float32))
    )
    assert result.shape == (4, 3)

  def test_async_cross_arg_mismatch(self) -> None:
    """Shape mismatch raises for async functions."""
    import asyncio

    @shapix.check
    @beartype
    async def f(x: F32[N], y: F32[N]) -> F32[N]:
      return x + y

    with pytest.raises(BeartypeCallHintParamViolation):
      asyncio.run(f(np.ones((3,), dtype=np.float32), np.ones((5,), dtype=np.float32)))

  def test_async_return_mismatch(self) -> None:
    """Return shape violation raises for async functions."""
    import asyncio

    from beartype.roar import BeartypeCallHintReturnViolation

    @shapix.check
    @beartype
    async def f(x: F32[N]) -> F32[N]:
      return np.ones(999, dtype=np.float32)

    with pytest.raises(BeartypeCallHintReturnViolation):
      asyncio.run(f(np.ones(3, dtype=np.float32)))

  def test_async_with_conf(self) -> None:
    """@check(conf=BeartypeConf()) on async function."""
    import asyncio
    import inspect

    from beartype import BeartypeConf

    @shapix.check(conf=BeartypeConf())
    async def f(x: F32[N]) -> F32[N]:
      return x

    assert inspect.iscoroutinefunction(f)
    result = asyncio.run(f(np.ones((5,), dtype=np.float32)))
    assert result.shape == (5,)

  def test_async_value_resolution(self) -> None:
    """Value("size") resolves during await."""
    import asyncio

    @shapix.check
    @beartype
    async def f(size: int) -> F32[Value("size")]:  # type: ignore[valid-type]
      return np.ones(size, dtype=np.float32)

    result = asyncio.run(f(4))
    assert result.shape == (4,)

  def test_async_concurrent_memo_isolation(self) -> None:
    """Concurrent @check tasks see independent memos."""
    import asyncio

    @shapix.check
    @beartype
    async def task_fn(x: F32[N]) -> F32[N]:
      await asyncio.sleep(0)
      return x

    async def run() -> None:
      a = np.ones((4,), dtype=np.float32)
      b = np.ones((10,), dtype=np.float32)
      ra, rb = await asyncio.gather(task_fn(a), task_fn(b))
      assert ra.shape == (4,)
      assert rb.shape == (10,)

    asyncio.run(run())

  def test_async_concurrent_value_resolution(self) -> None:
    """Concurrent tasks using Value("self.attr") on different instances."""
    import asyncio

    class Obj:
      def __init__(self, size: int) -> None:
        self.size = size

      @shapix.check
      @beartype
      async def f(self) -> F32[Value("self.size")]:  # type: ignore[valid-type]
        await asyncio.sleep(0)
        return np.ones(self.size, dtype=np.float32)

    async def run() -> None:
      a, b = await asyncio.gather(Obj(3).f(), Obj(7).f())
      assert a.shape == (3,)
      assert b.shape == (7,)

    asyncio.run(run())

  def test_async_exception_cleanup(self) -> None:
    """Memo is cleaned up when async function raises."""
    import asyncio

    @shapix.check
    @beartype
    async def boom(x: F32[N]) -> F32[N]:
      raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
      asyncio.run(boom(np.ones(5, dtype=np.float32)))

    # Should not leak — next call works independently
    @shapix.check
    @beartype
    async def ok(x: F32[N]) -> F32[N]:
      return x

    result = asyncio.run(ok(np.ones(3, dtype=np.float32)))
    assert result.shape == (3,)

  def test_async_cancellation(self) -> None:
    """Memo is cleaned up when task is cancelled."""
    import asyncio

    @shapix.check
    @beartype
    async def slow(x: F32[N]) -> F32[N]:
      await asyncio.sleep(100)
      return x

    async def run() -> None:
      task = asyncio.create_task(slow(np.ones(5, dtype=np.float32)))
      await asyncio.sleep(0)
      task.cancel()
      with pytest.raises(asyncio.CancelledError):
        await task

      # Memo should not be leaked — a new call works fine
      @shapix.check
      @beartype
      async def ok(x: F32[N]) -> F32[N]:
        return x

      result = await ok(np.ones(3, dtype=np.float32))
      assert result.shape == (3,)

    asyncio.run(run())
