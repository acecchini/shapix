"""Verify shapix.check preserves function signatures for type checkers."""

import numpy as np
from numpy.typing import NDArray

from shapix import check


@check
def identity(x: NDArray[np.float32]) -> NDArray[np.float32]:
  return x


# Type checker should see the preserved signature
result: NDArray[np.float32] = identity(np.ones((4, 3), dtype=np.float32))
assert result.shape == (4, 3)


@check
def add(x: NDArray[np.float32], y: NDArray[np.float32]) -> NDArray[np.float32]:
  return x + y


result2: NDArray[np.float32] = add(
  np.ones((4,), dtype=np.float32), np.ones((4,), dtype=np.float32)
)
