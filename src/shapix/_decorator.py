"""Optional decorator and context manager for explicit memo management.

Most users will not need these — the frame-based memo in :mod:`._memo` works
automatically with ``@beartype``.  These are provided for:

- Guaranteed correctness in exotic call-stack scenarios
- Combining memo management with ``BeartypeConf`` in a single decorator
- Manual ``isinstance`` checks at module scope
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from typing import overload

from ._memo import pop_memo, push_memo

__all__ = ["check", "check_context"]


@overload
def check[**P, R](fn: Callable[P, R], /) -> Callable[P, R]: ...
@overload
def check[**P, R](
  *, conf: object = ...
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def check[**P, R](
  fn: Callable[P, R] | None = None, /, *, conf: object | None = None
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
  """Decorator that manages the dimension memo around a function call.

  Usage::

      # Memo only — pair with @beartype
      @shapix.check
      @beartype
      def f(x: Float32Array[N, C]) -> Float32Array[N, C]: ...


      # Memo + beartype combined
      @shapix.check(conf=BeartypeConf(strategy=BeartypeStrategy.On))
      def f(x: Float32Array[N, C]) -> Float32Array[N, C]: ...
  """

  def decorator(fn: Callable[P, R]) -> Callable[P, R]:
    inner = fn
    signature = inspect.signature(fn)
    if conf is not None:
      from beartype import beartype

      inner = beartype(fn, conf=conf)  # type: ignore[arg-type]

    if inspect.iscoroutinefunction(fn):

      @functools.wraps(fn)
      async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        bound = signature.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        push_memo(scope=dict(bound.arguments))
        try:
          return await inner(*args, **kwargs)  # type: ignore[misc]
        finally:
          pop_memo()

      return async_wrapper  # type: ignore[return-value]

    @functools.wraps(fn)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
      bound = signature.bind_partial(*args, **kwargs)
      bound.apply_defaults()
      push_memo(scope=dict(bound.arguments))
      try:
        return inner(*args, **kwargs)  # type: ignore[return-value]
      finally:
        pop_memo()

    return wrapper  # type: ignore[return-value]

  if fn is not None:
    return decorator(fn)
  return decorator  # type: ignore[return-value]


class check_context:
  """Context manager for manual ``isinstance`` checks with shared memo.

  Usage::

      with shapix.check_context():
        assert isinstance(x, Float32Array[N, C])
        assert isinstance(y, Float32Array[N])  # same N
  """

  __slots__ = ()

  def __enter__(self) -> check_context:
    push_memo()
    return self

  def __exit__(self, *_: object) -> None:
    pop_memo()
