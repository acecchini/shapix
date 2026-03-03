"""Verify F32[N, C] annotation pattern (pyright-specific).

This tests the full shapix annotation pattern including dimension-subscripted
array types. Only pyright/Pylance supports this through TYPE_CHECKING stubs.
"""

import numpy as np
from beartype import beartype

from shapix import N, C, check
from shapix.numpy import F32, I64


@check
@beartype
def add(x: F32[N, C], y: F32[N, C]) -> F32[N, C]:
  return x + y


result = add(np.ones((4, 3), dtype=np.float32), np.ones((4, 3), dtype=np.float32))
assert result.shape == (4, 3)


@beartype
def cast(x: I64[N]) -> F32[N]:
  return x.astype(np.float32)
