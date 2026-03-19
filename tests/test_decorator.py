# pyright: reportArgumentType=false, reportGeneralTypeIssues=false
"""Tests for _decorator.py — @shapix.check and check_context."""

from __future__ import annotations

import numpy as np
import pytest
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation

import shapix
from shapix import N, C, T, Value  # noqa: F401
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

  def test_async_child_task_inside_check_context_shares_memo(self) -> None:
    """Child task inherits parent's memo by reference (documented behavior).

    A child task spawned inside an active ``check_context`` sees the same
    live ``ShapeMemo`` — this is intentional (shared by reference).  For
    full isolation, each task should enter its own ``check_context``.
    """
    import asyncio

    from shapix._memo import _explicit_stack

    async def run() -> None:
      async with shapix.check_context():
        from beartype.door import is_bearable

        # Bind N=4 in the parent context
        x = np.ones((4, 3), dtype=np.float32)
        assert is_bearable(x, F32[N, C])

        # Capture parent memo object identity
        parent_stack = _explicit_stack.get()
        assert len(parent_stack) > 0
        parent_memo = parent_stack[-1]

        child_saw: dict[str, object] = {}

        async def child() -> None:
          child_stack = _explicit_stack.get()
          if child_stack:
            child_saw["memo"] = child_stack[-1]
            child_saw["N"] = child_stack[-1].single.get("N")

        await asyncio.create_task(child())

        # Child inherits the same memo object (shared by reference)
        assert child_saw.get("memo") is parent_memo
        assert child_saw.get("N") == 4

    asyncio.run(run())


class TestCheckContextFreshRecheck:
  """Replay guards must not leak _fail_obj across fresh check_context() scopes.

  When an untagged explicit memo is active (check_context), beartype's
  error-generation re-invocation still sees the real memo, so the replay
  guard should be bypassed — allowing the same object to be re-validated
  in a fresh context where it may legitimately pass.
  """

  def test_struct_checker_same_object_fresh_context(self) -> None:
    """Object that fails in one check_context must pass in a fresh one."""
    from beartype.door import is_bearable

    hint = F32[N]

    arr = np.ones((4,), dtype=np.float32)

    # Context 1: bind N=3 first, then arr (shape 4) must fail
    with shapix.check_context():
      assert is_bearable(np.ones((3,), dtype=np.float32), hint)  # bind N=3
      assert not is_bearable(arr, hint)  # fails: N=3 but shape is 4

    # Context 2: fresh memo, N is unbound — arr (shape 4) should pass
    with shapix.check_context():
      assert is_bearable(arr, hint)  # must not be poisoned

  def test_arraylike_checker_same_object_fresh_context(self) -> None:
    """ArrayLike object that fails in one context must pass in a fresh one."""
    from beartype.door import is_bearable

    from shapix.numpy import F32Like

    hint = F32Like[N]

    arr = np.ones((4,), dtype=np.float32)

    # Context 1: bind N=3, then arr (shape 4) fails
    with shapix.check_context():
      assert is_bearable(np.ones((3,), dtype=np.float32), hint)  # bind N=3
      assert not is_bearable(arr, hint)  # fails: shape mismatch

    # Context 2: fresh memo — arr should pass
    with shapix.check_context():
      assert is_bearable(arr, hint)  # must not be poisoned

  def test_tree_checker_same_object_fresh_context(self) -> None:
    """Tree object that fails in one context must pass in a fresh one."""
    pytest.importorskip("optree")
    from beartype.door import is_bearable

    from shapix import T
    from shapix.optree import Tree

    hint = Tree[F32[N], T]  # type: ignore[type-arg]

    x_dict = {"a": np.ones(3, dtype=np.float32)}
    y_list = [np.ones(3, dtype=np.float32)]

    # Context 1: bind T to dict structure via x_dict, then y_list fails
    with shapix.check_context():
      assert is_bearable(x_dict, hint)  # bind T = dict structure
      assert not is_bearable(y_list, hint)  # fails: list != dict

    # Context 2: fresh memo — y_list should pass (T unbound)
    with shapix.check_context():
      assert is_bearable(y_list, hint)  # must not be poisoned


class TestReplayGuardRevalidation:
  """Replay guard must not prevent revalidation of mutated objects."""

  def test_struct_checker_revalidates_mutated_object(self) -> None:
    """Array that fails then is resized must pass on recheck."""
    from beartype.door import is_bearable

    hint = F32[3]
    arr = np.ones(4, dtype=np.float32)
    assert not is_bearable(arr, hint)
    arr.resize((3,), refcheck=False)
    assert is_bearable(arr, hint)

  def test_arraylike_checker_revalidates_mutated_object(self) -> None:
    """List that fails then is shortened must pass on recheck."""
    from beartype.door import is_bearable

    from shapix.numpy import F32Like

    hint = F32Like[3]
    lst = [1.0, 2.0, 3.0, 4.0]
    assert not is_bearable(lst, hint)
    lst.pop()
    assert is_bearable(lst, hint)

  def test_tree_checker_revalidates_mutated_object(self) -> None:
    """Tree with bad leaves that are replaced must pass on recheck."""
    pytest.importorskip("optree")
    from beartype.door import is_bearable

    from shapix.optree import Tree

    hint = Tree[F32[3]]  # type: ignore[type-arg]
    tree = {"a": np.ones(4, dtype=np.float32)}
    assert not is_bearable(tree, hint)
    tree["a"] = np.ones(3, dtype=np.float32)
    assert is_bearable(tree, hint)


class TestReplayGuardCrossArg:
  """Guard must not block revalidation in fresh @beartype calls after cross-arg failure."""

  def test_struct_checker_cross_arg_revalidation(self) -> None:
    """Object that fails cross-arg check passes in a fresh call with compatible binding."""
    b = np.ones(3, dtype=np.float32)  # shape (3,)

    @shapix.check
    @beartype
    def f(a: F32[N], b: F32[N]) -> None:
      pass

    # a binds N=4, b has shape (3,) → cross-arg failure, guard arms on b.
    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones(4, dtype=np.float32), b)

    # Fresh call: a binds N=3, b has shape (3,) → must pass.
    f(np.ones(3, dtype=np.float32), b)

  def test_arraylike_checker_cross_arg_revalidation(self) -> None:
    """ArrayLike object that fails cross-arg passes after mutation in a fresh call."""
    from shapix.numpy import F32Like

    lst = [1.0, 2.0, 3.0, 4.0]

    @shapix.check
    @beartype
    def f(a: F32Like[N], b: F32Like[N]) -> None:
      pass

    with pytest.raises(BeartypeCallHintParamViolation):
      f([1.0, 2.0, 3.0], lst)

    lst.pop()
    f([1.0, 2.0, 3.0], lst)  # must pass

  def test_tree_checker_cross_arg_revalidation(self) -> None:
    """Tree that fails cross-structure check passes in a fresh call."""
    pytest.importorskip("optree")
    from shapix.optree import Tree

    y = [np.ones(3, dtype=np.float32)]

    @shapix.check
    @beartype
    def f(x: Tree[F32[N], T], y: Tree[F32[N], T]) -> None:  # type: ignore[valid-type]
      pass

    with pytest.raises(BeartypeCallHintParamViolation):
      f({"a": np.ones(3, dtype=np.float32)}, y)  # y fails: list != dict

    # Same y object, but in a fresh call where T is unbound — should pass
    @shapix.check
    @beartype
    def g(x: Tree[F32[N], T]) -> None:  # type: ignore[valid-type]
      pass

    g(y)  # must pass


class TestReplayGuardReusedHint:
  """Prebuilt hint must not poison standalone validation after cross-arg failure."""

  def test_struct_reused_hint_standalone(self) -> None:
    """is_bearable passes for reused F32[N] hint after cross-arg failure."""
    from beartype.door import is_bearable

    Hint = F32[N]
    b = np.ones(3, dtype=np.float32)

    @shapix.check
    @beartype
    def pair(a: Hint, b: Hint) -> None:
      pass

    with pytest.raises(BeartypeCallHintParamViolation):
      pair(np.ones(4, dtype=np.float32), b)  # b fails: N=4 vs shape (3,)

    assert is_bearable(b, Hint)  # must pass: N unbound, shape (3,) binds N=3

  def test_arraylike_reused_hint_standalone(self) -> None:
    """is_bearable passes for reused F32Like[N] hint after cross-arg failure."""
    from beartype.door import is_bearable
    from shapix.numpy import F32Like

    Hint = F32Like[N]
    lst = [1.0, 2.0, 3.0]

    @shapix.check
    @beartype
    def pair(a: Hint, b: Hint) -> None:
      pass

    with pytest.raises(BeartypeCallHintParamViolation):
      pair([1.0, 2.0, 3.0, 4.0], lst)  # lst fails: N=4 vs len 3

    assert is_bearable(lst, Hint)  # must pass

  def test_tree_reused_hint_standalone(self) -> None:
    """is_bearable passes for reused Tree hint after cross-structure failure."""
    pytest.importorskip("optree")
    from beartype.door import is_bearable
    from shapix.optree import Tree

    Hint = Tree[F32[N], T]  # type: ignore[type-arg]
    y = [np.ones(3, dtype=np.float32)]

    @shapix.check
    @beartype
    def pair(x: Hint, y: Hint) -> None:  # type: ignore[valid-type]
      pass

    with pytest.raises(BeartypeCallHintParamViolation):
      pair({"a": np.ones(3, dtype=np.float32)}, y)

    assert is_bearable(y, Hint)  # must pass: T unbound, list structure OK


class TestReplayGuardCheckContext:
  """Prebuilt hint must not poison standalone validation after check_context failure."""

  def test_struct_check_context_standalone(self) -> None:
    """is_bearable passes standalone after check_context cross-dim failure."""
    from beartype.door import is_bearable

    Hint = F32[N]
    arr = np.ones(4, dtype=np.float32)

    with shapix.check_context():
      assert is_bearable(np.ones(3, dtype=np.float32), Hint)  # binds N=3
      assert not is_bearable(arr, Hint)  # fails: N=3 vs (4,)

    assert is_bearable(arr, Hint)  # must pass: N unbound, shape (4,) binds N=4

  def test_arraylike_check_context_standalone(self) -> None:
    """ArrayLike object passes standalone after check_context failure."""
    from beartype.door import is_bearable

    from shapix.numpy import F32Like

    Hint = F32Like[N]
    lst = [1.0, 2.0, 3.0, 4.0]

    with shapix.check_context():
      assert is_bearable([1.0, 2.0, 3.0], Hint)  # binds N=3
      assert not is_bearable(lst, Hint)  # fails: N=3 vs len 4

    assert is_bearable(lst, Hint)  # must pass

  def test_tree_check_context_standalone(self) -> None:
    """Tree object passes standalone after check_context structure failure."""
    pytest.importorskip("optree")
    from beartype.door import is_bearable

    from shapix.optree import Tree

    Hint = Tree[F32[N], T]  # type: ignore[type-arg]
    y = [np.ones(3, dtype=np.float32)]

    with shapix.check_context():
      x_dict = {"a": np.ones(3, dtype=np.float32)}
      assert is_bearable(x_dict, Hint)  # binds T=dict
      assert not is_bearable(y, Hint)  # fails: list != dict

    assert is_bearable(y, Hint)  # must pass: T unbound
