# pyright: reportMissingImports=false
# ruff: noqa: E402
"""PyTorch tensor type annotations with runtime shape and dtype checking.

Usage::

    from shapix import N, C, H, W
    from shapix.torch import F32


    @beartype
    def forward(x: F32[N, C, H, W]) -> F32[N, C, H, W]: ...

ScalarLike types (range-validated scalars) and ``make_scalar_like_type``
are re-exported from ``shapix.numpy`` for convenience.
"""

from __future__ import annotations

import typing as tp

from ._imports import require_attr, require_module

_TORCH_INSTALL_HINT = (
  "shapix.torch requires 'torch' at runtime. "
  "Install it alongside shapix (e.g. `pip install shapix numpy torch`)."
)
_NUMPY_INSTALL_HINT = (
  "shapix.torch requires 'numpy' at runtime. "
  "Install it alongside shapix (e.g. `pip install shapix numpy torch`)."
)

if tp.TYPE_CHECKING:
  from torch import Tensor
else:
  Tensor = tp.cast(
    type[object], require_attr("torch", "Tensor", install_hint=_TORCH_INSTALL_HINT)
  )

try:
  import numpy as _np  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ModuleNotFoundError as exc:
  raise ModuleNotFoundError(_NUMPY_INSTALL_HINT) from exc

from ._array_types import make_array_like_type as _make_array_like_type
from ._array_types import make_array_type
from ._dtypes import (
  BFLOAT16,
  BOOL,
  COMPLEX,
  COMPLEX64,
  COMPLEX128,
  FLOAT,
  FLOAT16,
  FLOAT32,
  FLOAT64,
  INT,
  INT8,
  INT16,
  INT32,
  INT64,
  INTEGER,
  INEXACT,
  NUM,
  REAL,
  SHAPED,
  UINT,
  UINT8,
  UINT16,
  UINT32,
  UINT64,
)

# ScalarLike types + factory (re-exported from numpy — no shape, just value)
from .numpy import BoolScalarLike as BoolScalarLike
from .numpy import C64ScalarLike as C64ScalarLike
from .numpy import C128ScalarLike as C128ScalarLike
from .numpy import C256ScalarLike as C256ScalarLike
from .numpy import ComplexScalarLike as ComplexScalarLike
from .numpy import F16ScalarLike as F16ScalarLike
from .numpy import F32ScalarLike as F32ScalarLike
from .numpy import F64ScalarLike as F64ScalarLike
from .numpy import F128ScalarLike as F128ScalarLike
from .numpy import FloatScalarLike as FloatScalarLike
from .numpy import I8ScalarLike as I8ScalarLike
from .numpy import I16ScalarLike as I16ScalarLike
from .numpy import I32ScalarLike as I32ScalarLike
from .numpy import I64ScalarLike as I64ScalarLike
from .numpy import InexactScalarLike as InexactScalarLike
from .numpy import IntegerScalarLike as IntegerScalarLike
from .numpy import IntScalarLike as IntScalarLike
from .numpy import NumScalarLike as NumScalarLike
from .numpy import RealScalarLike as RealScalarLike
from .numpy import ShapedScalarLike as ShapedScalarLike
from .numpy import StringLike as StringLike
from .numpy import U8ScalarLike as U8ScalarLike
from .numpy import U16ScalarLike as U16ScalarLike
from .numpy import U32ScalarLike as U32ScalarLike
from .numpy import U64ScalarLike as U64ScalarLike
from .numpy import UIntScalarLike as UIntScalarLike
from .numpy import make_scalar_like_type as make_scalar_like_type

# PyTorch does not support float128, complex256, void, string, bytes, object,
# datetime64, or timedelta64 dtypes.  Those types are NumPy-only (see numpy.py).
# BF16 is available here but not in numpy.py (NumPy has no native bfloat16).
__all__ = [
  # Array types
  "Bool",
  "I8",
  "I16",
  "I32",
  "I64",
  "U8",
  "U16",
  "U32",
  "U64",
  "F16",
  "F32",
  "F64",
  "BF16",
  "C64",
  "C128",
  "Int",
  "UInt",
  "Integer",
  "Float",
  "Real",
  "Complex",
  "Inexact",
  "Num",
  "Shaped",
  # Like types
  "BF16Like",
  "BoolLike",
  "I8Like",
  "I16Like",
  "I32Like",
  "I64Like",
  "U8Like",
  "U16Like",
  "U32Like",
  "U64Like",
  "F16Like",
  "F32Like",
  "F64Like",
  "C64Like",
  "C128Like",
  "IntLike",
  "UIntLike",
  "IntegerLike",
  "FloatLike",
  "RealLike",
  "ComplexLike",
  "InexactLike",
  "NumLike",
  "ShapedLike",
  # ScalarLike types (re-exported from numpy)
  "StringLike",
  "BoolScalarLike",
  "I8ScalarLike",
  "I16ScalarLike",
  "I32ScalarLike",
  "I64ScalarLike",
  "U8ScalarLike",
  "U16ScalarLike",
  "U32ScalarLike",
  "U64ScalarLike",
  "F16ScalarLike",
  "F32ScalarLike",
  "F64ScalarLike",
  "F128ScalarLike",
  "C64ScalarLike",
  "C128ScalarLike",
  "C256ScalarLike",
  "IntScalarLike",
  "UIntScalarLike",
  "IntegerScalarLike",
  "FloatScalarLike",
  "RealScalarLike",
  "ComplexScalarLike",
  "InexactScalarLike",
  "NumScalarLike",
  "ShapedScalarLike",
  # ScalarLike factory
  "make_scalar_like_type",
]

# ---------------------------------------------------------------------------
# Backend-specific conversion (tensors, ndarray, scalars, sequences)
# ---------------------------------------------------------------------------


def _torch_asarray(obj: object) -> tp.Any:
  torch = require_module("torch", install_hint=_TORCH_INSTALL_HINT)
  return torch.as_tensor(obj)


# Backend-scoped fast-path trust: only np.ndarray and torch.Tensor skip
# conversion.  Foreign-backend arrays (e.g. jax.Array) fall through
# to the slow path where torch.as_tensor() verifies actual convertibility.
_TORCH_TRUSTED: tuple[type, ...] = (_np.ndarray, Tensor)


def make_array_like_type(
  dtype_spec: object,
  *,
  casting: str = "same_kind",
  name: str = "ArrayLike",
  asarray: object | None = _torch_asarray,
  trusted_types: object | None = _TORCH_TRUSTED,
) -> tp.Any:
  """Torch-aware version of :func:`shapix.make_array_like_type`.

  Defaults to ``torch.as_tensor`` for the slow path, so tensors, NumPy arrays,
  Python scalars, and nested sequences are accepted.
  The fast path only trusts ``np.ndarray`` and ``torch.Tensor``; other
  backend arrays (e.g. ``jax.Array``) go through the slow path.
  """
  return _make_array_like_type(
    dtype_spec,  # type: ignore[arg-type]
    casting=casting,
    name=name,
    asarray=asarray,  # type: ignore[arg-type]
    trusted_types=trusted_types,  # type: ignore[arg-type]
  )


# ---------------------------------------------------------------------------
# Tensor types (shape-checked via beartype Is[])
# ---------------------------------------------------------------------------

if tp.TYPE_CHECKING:
  _Dims = tp.TypeVarTuple("_Dims")

  Bool = tp.TypeAliasType("Bool", Tensor, type_params=(_Dims,))

  I8 = tp.TypeAliasType("I8", Tensor, type_params=(_Dims,))
  I16 = tp.TypeAliasType("I16", Tensor, type_params=(_Dims,))
  I32 = tp.TypeAliasType("I32", Tensor, type_params=(_Dims,))
  I64 = tp.TypeAliasType("I64", Tensor, type_params=(_Dims,))

  U8 = tp.TypeAliasType("U8", Tensor, type_params=(_Dims,))
  U16 = tp.TypeAliasType("U16", Tensor, type_params=(_Dims,))
  U32 = tp.TypeAliasType("U32", Tensor, type_params=(_Dims,))
  U64 = tp.TypeAliasType("U64", Tensor, type_params=(_Dims,))

  F16 = tp.TypeAliasType("F16", Tensor, type_params=(_Dims,))
  F32 = tp.TypeAliasType("F32", Tensor, type_params=(_Dims,))
  F64 = tp.TypeAliasType("F64", Tensor, type_params=(_Dims,))
  BF16 = tp.TypeAliasType("BF16", Tensor, type_params=(_Dims,))

  C64 = tp.TypeAliasType("C64", Tensor, type_params=(_Dims,))
  C128 = tp.TypeAliasType("C128", Tensor, type_params=(_Dims,))

  Int = tp.TypeAliasType("Int", Tensor, type_params=(_Dims,))
  UInt = tp.TypeAliasType("UInt", Tensor, type_params=(_Dims,))
  Integer = tp.TypeAliasType("Integer", Tensor, type_params=(_Dims,))
  Float = tp.TypeAliasType("Float", Tensor, type_params=(_Dims,))
  Real = tp.TypeAliasType("Real", Tensor, type_params=(_Dims,))
  Complex = tp.TypeAliasType("Complex", Tensor, type_params=(_Dims,))
  Inexact = tp.TypeAliasType("Inexact", Tensor, type_params=(_Dims,))
  Num = tp.TypeAliasType("Num", Tensor, type_params=(_Dims,))
  Shaped = tp.TypeAliasType("Shaped", Tensor, type_params=(_Dims,))

else:
  Bool = make_array_type(Tensor, BOOL)

  I8 = make_array_type(Tensor, INT8)
  I16 = make_array_type(Tensor, INT16)
  I32 = make_array_type(Tensor, INT32)
  I64 = make_array_type(Tensor, INT64)

  U8 = make_array_type(Tensor, UINT8)
  U16 = make_array_type(Tensor, UINT16)
  U32 = make_array_type(Tensor, UINT32)
  U64 = make_array_type(Tensor, UINT64)

  F16 = make_array_type(Tensor, FLOAT16)
  F32 = make_array_type(Tensor, FLOAT32)
  F64 = make_array_type(Tensor, FLOAT64)
  BF16 = make_array_type(Tensor, BFLOAT16)

  C64 = make_array_type(Tensor, COMPLEX64)
  C128 = make_array_type(Tensor, COMPLEX128)

  Int = make_array_type(Tensor, INT)
  UInt = make_array_type(Tensor, UINT)
  Integer = make_array_type(Tensor, INTEGER)
  Float = make_array_type(Tensor, FLOAT)
  Real = make_array_type(Tensor, REAL)
  Complex = make_array_type(Tensor, COMPLEX)
  Inexact = make_array_type(Tensor, INEXACT)
  Num = make_array_type(Tensor, NUM)
  Shaped = make_array_type(Tensor, SHAPED)

# ---------------------------------------------------------------------------
# Like types — runtime: scalar | tensor | nested sequences; static: Tensor
# ---------------------------------------------------------------------------

if tp.TYPE_CHECKING:
  BF16Like = tp.TypeAliasType("BF16Like", Tensor, type_params=(_Dims,))
  BoolLike = tp.TypeAliasType("BoolLike", Tensor, type_params=(_Dims,))

  I8Like = tp.TypeAliasType("I8Like", Tensor, type_params=(_Dims,))
  I16Like = tp.TypeAliasType("I16Like", Tensor, type_params=(_Dims,))
  I32Like = tp.TypeAliasType("I32Like", Tensor, type_params=(_Dims,))
  I64Like = tp.TypeAliasType("I64Like", Tensor, type_params=(_Dims,))

  U8Like = tp.TypeAliasType("U8Like", Tensor, type_params=(_Dims,))
  U16Like = tp.TypeAliasType("U16Like", Tensor, type_params=(_Dims,))
  U32Like = tp.TypeAliasType("U32Like", Tensor, type_params=(_Dims,))
  U64Like = tp.TypeAliasType("U64Like", Tensor, type_params=(_Dims,))

  F16Like = tp.TypeAliasType("F16Like", Tensor, type_params=(_Dims,))
  F32Like = tp.TypeAliasType("F32Like", Tensor, type_params=(_Dims,))
  F64Like = tp.TypeAliasType("F64Like", Tensor, type_params=(_Dims,))

  C64Like = tp.TypeAliasType("C64Like", Tensor, type_params=(_Dims,))
  C128Like = tp.TypeAliasType("C128Like", Tensor, type_params=(_Dims,))

  IntLike = tp.TypeAliasType("IntLike", Tensor, type_params=(_Dims,))
  UIntLike = tp.TypeAliasType("UIntLike", Tensor, type_params=(_Dims,))
  IntegerLike = tp.TypeAliasType("IntegerLike", Tensor, type_params=(_Dims,))
  FloatLike = tp.TypeAliasType("FloatLike", Tensor, type_params=(_Dims,))
  RealLike = tp.TypeAliasType("RealLike", Tensor, type_params=(_Dims,))
  ComplexLike = tp.TypeAliasType("ComplexLike", Tensor, type_params=(_Dims,))
  InexactLike = tp.TypeAliasType("InexactLike", Tensor, type_params=(_Dims,))
  NumLike = tp.TypeAliasType("NumLike", Tensor, type_params=(_Dims,))
  ShapedLike = tp.TypeAliasType("ShapedLike", Tensor, type_params=(_Dims,))

else:
  BF16Like = make_array_like_type(BFLOAT16, name="BF16Like")
  BoolLike = make_array_like_type(BOOL, name="BoolLike")

  I8Like = make_array_like_type(INT8, name="I8Like")
  I16Like = make_array_like_type(INT16, name="I16Like")
  I32Like = make_array_like_type(INT32, name="I32Like")
  I64Like = make_array_like_type(INT64, name="I64Like")

  U8Like = make_array_like_type(UINT8, name="U8Like")
  U16Like = make_array_like_type(UINT16, name="U16Like")
  U32Like = make_array_like_type(UINT32, name="U32Like")
  U64Like = make_array_like_type(UINT64, name="U64Like")

  F16Like = make_array_like_type(FLOAT16, name="F16Like")
  F32Like = make_array_like_type(FLOAT32, name="F32Like")
  F64Like = make_array_like_type(FLOAT64, name="F64Like")

  C64Like = make_array_like_type(COMPLEX64, name="C64Like")
  C128Like = make_array_like_type(COMPLEX128, name="C128Like")

  IntLike = make_array_like_type(INT, name="IntLike")
  UIntLike = make_array_like_type(UINT, name="UIntLike")
  IntegerLike = make_array_like_type(INTEGER, name="IntegerLike")
  FloatLike = make_array_like_type(FLOAT, name="FloatLike")
  RealLike = make_array_like_type(REAL, name="RealLike")
  ComplexLike = make_array_like_type(COMPLEX, name="ComplexLike")
  InexactLike = make_array_like_type(INEXACT, name="InexactLike")
  NumLike = make_array_like_type(NUM, name="NumLike")
  ShapedLike = make_array_like_type(SHAPED, name="ShapedLike")
