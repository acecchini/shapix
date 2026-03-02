import typing as tp
from typing import Annotated as A

import numpy as np
import numpy._typing as npt
from jaxtyping import AbstractDtype, make_numpy_struct_dtype

# jaxtyping AbstractDtype for numpy string arrays
if tp.TYPE_CHECKING:
  type String = A

else:

  class String(AbstractDtype):
    dtypes = ["str_", "unicode_"]


class StructDtype:
  """
  Creates a jaxtyping AbstractDtype for a structured NumPy array
  using the syntax: StructDtype['Name', [('field1', dtype1), ...]]
  """

  # The argument received is the tuple: ('Name', [('field1', dtype1), ...])
  @classmethod
  def __class_getitem__(
    cls, fields: tuple[str, npt._VoidDTypeLike]
  ) -> type[AbstractDtype]:
    # Unpack the received tuple
    name, dtype_like = fields

    # 1. First, construct the actual np.dtype object from the field definitions
    dtype = np.dtype(dtype_like)

    # 2. Call make_numpy_struct_dtype with the correct arguments:
    #    (numpy_struct_dtype, name)
    return make_numpy_struct_dtype(dtype, name)
