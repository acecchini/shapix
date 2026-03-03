"""Verify shapix.check preserves function signatures for type checkers.

Tested with: pyright, mypy, ty
"""

import numpy as np
from numpy.typing import NDArray

from shapix import check, check_context

# ---------------------------------------------------------------------------
# Basic @check — preserves NDArray signature
# ---------------------------------------------------------------------------


@check
def identity(x: NDArray[np.float32]) -> NDArray[np.float32]:
  return x


result: NDArray[np.float32] = identity(np.ones((4, 3), dtype=np.float32))
assert result.shape == (4, 3)


# ---------------------------------------------------------------------------
# @check — multi-parameter function
# ---------------------------------------------------------------------------


@check
def add(x: NDArray[np.float32], y: NDArray[np.float32]) -> NDArray[np.float32]:
  return x + y


result2: NDArray[np.float32] = add(
  np.ones((4,), dtype=np.float32), np.ones((4,), dtype=np.float32)
)


# ---------------------------------------------------------------------------
# @check — various return types
# ---------------------------------------------------------------------------


@check
def get_shape(x: NDArray[np.float32]) -> tuple[int, ...]:
  return x.shape


shape_result: tuple[int, ...] = get_shape(np.ones((4, 3), dtype=np.float32))


@check
def count_elements(x: NDArray[np.float32]) -> int:
  return int(x.size)


int_result: int = count_elements(np.ones((4,), dtype=np.float32))


@check
def describe(x: NDArray[np.float32]) -> str:
  return f"shape={x.shape}"


str_result: str = describe(np.ones((4,), dtype=np.float32))


@check
def do_nothing(x: NDArray[np.float32]) -> None:
  _ = x


none_result: None = do_nothing(np.ones((4,), dtype=np.float32))


# ---------------------------------------------------------------------------
# @check — no arguments / keyword arguments
# ---------------------------------------------------------------------------


@check
def make_array(size: int = 10) -> NDArray[np.float32]:
  return np.ones((size,), dtype=np.float32)


arr: NDArray[np.float32] = make_array()
arr2: NDArray[np.float32] = make_array(size=5)


# ---------------------------------------------------------------------------
# @check — *args and **kwargs
# ---------------------------------------------------------------------------


@check
def concat(*arrays: NDArray[np.float32]) -> NDArray[np.float32]:
  return np.concatenate(arrays)


concat_result: NDArray[np.float32] = concat(
  np.ones((2,), dtype=np.float32), np.ones((3,), dtype=np.float32)
)


# ---------------------------------------------------------------------------
# check_context — context manager type
# ---------------------------------------------------------------------------


def use_check_context() -> None:
  with check_context() as ctx:
    _ = ctx
    arr = np.ones((3, 4), dtype=np.float32)
    assert arr.shape == (3, 4)
