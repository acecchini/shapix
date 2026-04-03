"""Optional decorator and context manager for explicit memo management.

Most users will not need these — the frame-based memo in :mod:`._memo` works
automatically with ``@beartype``.  These are provided for:

- Guaranteed correctness in exotic call-stack scenarios
- Combining memo management with ``BeartypeConf`` in a single decorator
- Manual ``is_bearable()`` checks at module scope
"""

from __future__ import annotations

import functools
import inspect
import typing as tp
from collections.abc import Callable
from typing import overload

from ._memo import pop_memo, push_memo

if tp.TYPE_CHECKING:
  from beartype import BeartypeConf

P = tp.ParamSpec("P")
R = tp.TypeVar("R")

__all__ = ["check", "check_context"]


@overload
def check(fn: Callable[P, R], /) -> Callable[P, R]: ...
@overload
def check(
  *, conf: BeartypeConf = ...
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def check(
  fn: Callable[P, tp.Any] | None = None, /, *, conf: BeartypeConf | None = None
) -> tp.Any:
  """Decorator that manages the dimension memo around a function call.

  Usage::

      # Memo only — pair with @beartype
      @shapix.check
      @beartype
      def f(x: Float32Array[N, C]) -> Float32Array[N, C]: ...


      # Memo + beartype combined
      @shapix.check(conf=BeartypeConf(strategy=BeartypeStrategy.On))
      def f(x: Float32Array[N, C]) -> Float32Array[N, C]: ...

  .. note::
     Generator functions (sync and async) are not supported.
     Decorate regular functions or coroutine functions only.
  """

  def decorator(fn: Callable[P, tp.Any]) -> Callable[P, tp.Any]:
    if inspect.isasyncgenfunction(fn):
      msg = "@shapix.check does not support async generator functions"
      raise TypeError(msg)
    if inspect.isgeneratorfunction(fn):
      msg = "@shapix.check does not support generator functions"
      raise TypeError(msg)

    inner: Callable[P, tp.Any] = fn
    signature = inspect.signature(fn)
    if conf is not None:
      from beartype import beartype

      inner = beartype(conf=conf)(fn)

    if inspect.iscoroutinefunction(fn):
      inner_code = getattr(inner, "__code__", None)
      inner_async = tp.cast("Callable[P, tp.Awaitable[tp.Any]]", inner)

      @functools.wraps(fn)
      async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> tp.Any:
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        push_memo(scope=dict(bound.arguments), owner_code=inner_code)
        try:
          return await inner_async(*args, **kwargs)
        finally:
          pop_memo()

      return async_wrapper

    inner_code = getattr(inner, "__code__", None)

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> tp.Any:
      bound = signature.bind_partial(*args, **kwargs)
      bound.apply_defaults()
      push_memo(scope=dict(bound.arguments), owner_code=inner_code)
      try:
        return inner(*args, **kwargs)
      finally:
        pop_memo()

    return wrapper

  if fn is not None:
    return decorator(fn)
  return decorator


class check_context:
  """Context manager for manual ``is_bearable()`` checks with shared memo.

  Supports both sync and async contexts::

      from beartype.door import is_bearable

      with shapix.check_context():
        assert is_bearable(x, Float32Array[N, C])
        assert is_bearable(y, Float32Array[N])  # same N

      async with shapix.check_context():
        assert is_bearable(x, Float32Array[N, C])

  .. note::
     Child async tasks spawned inside an active context inherit the same
     mutable memo by reference.  For full task isolation, each task should
     enter its own ``check_context()``.
  """

  __slots__ = ()

  def __enter__(self) -> check_context:
    push_memo()
    return self

  def __exit__(self, *_: object) -> None:
    pop_memo()

  async def __aenter__(self) -> check_context:
    push_memo()
    return self

  async def __aexit__(self, *_: object) -> None:
    pop_memo()
