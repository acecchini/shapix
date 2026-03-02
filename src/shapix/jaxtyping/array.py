from collections.abc import Collection
from importlib import import_module

import jax
import numpy as np
from jaxtyping import AbstractDtype
from torch import Tensor

arr_type_to_lib_name = {np.ndarray: "numpy", jax.Array: "jax", Tensor: "torch"}


def array(
  arr_type: type[np.ndarray] | type[jax.Array] | type[Tensor],
  dtype_type: type[AbstractDtype],
  shape: tuple[str | int, ...],
):
  """Build a jaxtyping annotation for a concrete array type/dtype/shape trio."""
  shape_str = " ".join(str(x) for x in shape)
  return dtype_type[arr_type, shape_str]


# Create a default Array type hint with ellipsis in it


# TODO: Could accept | and & ?
class ArrayType:
  """Factory that mirrors ``jaxtyping.Array`` but with reusable dimension symbols."""

  @classmethod
  def __class_getitem__(
    cls, arr_type: type[np.ndarray] | type[jax.Array] | type[Tensor]
  ):
    class Array:
      @classmethod
      def __class_getitem__(cls, dtype_type: type[AbstractDtype]):
        class ArrayDtyped:
          @classmethod
          def __class_getitem__(cls, shape: str | int | tuple[str | int, ...]):
            shape_tuple = shape if isinstance(shape, tuple) else (shape,)
            return array(arr_type, dtype_type, shape_tuple)

        return ArrayDtyped

    return Array


# TODO: Add other type of array like (i.e. the ones that support __array__ method!!
# TODO: Is *N, ...., and #N handled?
# TODO: Could accept | and & ?
# TODO: For now only checks only type and partially shapes for arrays
class ArrayLikeType:
  """Array-like annotations including nested ``Sequence`` fallbacks."""

  @classmethod
  def __class_getitem__(
    cls, arr_type: type[np.ndarray] | type[jax.Array] | type[Tensor]
  ):
    lib_name = arr_type_to_lib_name[arr_type]
    rk_module = import_module("shapix.numpy.typing")

    class ArrayLike:
      @classmethod
      def __class_getitem__(cls, dtype_type):
        scalar_like = getattr(rk_module, f"{dtype_type.__name__}Like")

        class ArrayLikeDtyped:
          @classmethod
          def __class_getitem__(cls, shape: str | int | tuple[str | int, ...]):
            if not isinstance(shape, tuple):
              if shape == "":
                annot = scalar_like | array(arr_type, dtype_type, ())
                return (
                  annot
                  if lib_name == "numpy"
                  else annot | array(np.ndarray, dtype_type, ())
                )
              return cls[(shape,)]
            shape_list = []
            for dim in shape:
              dim_str = str(dim)
              shape_list += dim_str.split()
            shape_tuple = tuple(shape_list)
            del shape_list

            if len(shape_tuple) == 1:
              annot = Collection[scalar_like] | array(arr_type, dtype_type, shape_tuple)
              return (
                annot
                if lib_name == "numpy"
                else annot | array(np.ndarray, dtype_type, shape_tuple)
              )
            # There are at least 2 dimensions
            subarray_like = cls[shape_tuple[1:]]
            annot = Collection[subarray_like] | array(arr_type, dtype_type, shape_tuple)
            return (
              annot
              if lib_name == "numpy"
              else annot | array(np.ndarray, dtype_type, shape_tuple)
            )

        return ArrayLikeDtyped

    return ArrayLike
