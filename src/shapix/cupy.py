# pyright: reportMissingImports=false
"""CuPy array type annotations with runtime shape and dtype checking.

Usage::

    from shapix import N, C, H, W
    from shapix.cupy import F32


    @beartype
    def forward(x: F32[N, C, H, W]) -> F32[N, C, H, W]: ...

ScalarLike types (range-validated scalars) and ``make_scalar_like_type``
are re-exported from ``shapix.numpy`` for convenience.
"""

from __future__ import annotations

import typing as tp

from cupy import ndarray as CuPyArray

try:
  import numpy as _np  # noqa: F401  # pyright: ignore[reportUnusedImport]
except ModuleNotFoundError as exc:
  msg = (
    "shapix.cupy requires 'numpy' at runtime. "
    "Install it alongside shapix (e.g. `pip install shapix numpy cupy`)."
  )
  raise ModuleNotFoundError(msg) from exc

# CuPy does not support float128, complex256, void, string, bytes, object,
# datetime64, or timedelta64 dtypes.  Those types are NumPy-only (see numpy.py).
# CuPy does not support bfloat16 natively, so BF16 is omitted (same as NumPy).
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

from ._array_types import make_array_like_type as _make_array_like_type
from ._array_types import make_array_type
from ._dtypes import (
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

# ---------------------------------------------------------------------------
# Backend-specific conversion (CuPy arrays, ndarray, scalars, sequences)
# ---------------------------------------------------------------------------


def _cupy_asarray(obj: object) -> tp.Any:
  import cupy as cp

  return cp.asarray(obj)


# Backend-scoped fast-path trust: only np.ndarray and cupy.ndarray skip
# conversion.  Foreign-backend arrays (e.g. torch.Tensor) fall through
# to the slow path where cp.asarray() verifies actual convertibility.
_CUPY_TRUSTED: tuple[type, ...] = (_np.ndarray, CuPyArray)


def make_array_like_type(
  dtype_spec: object,
  *,
  casting: str = "same_kind",
  name: str = "ArrayLike",
  asarray: object | None = _cupy_asarray,
  trusted_types: object | None = _CUPY_TRUSTED,
) -> tp.Any:
  """CuPy-aware version of :func:`shapix.make_array_like_type`.

  Defaults to ``cp.asarray`` for the slow path, so CuPy arrays, NumPy arrays,
  Python scalars, and nested sequences are accepted.
  The fast path only trusts ``np.ndarray`` and ``cupy.ndarray``; other
  backend arrays (e.g. ``torch.Tensor``) go through the slow path.
  """
  return _make_array_like_type(
    dtype_spec,  # type: ignore[arg-type]
    casting=casting,
    name=name,
    asarray=asarray,  # type: ignore[arg-type]
    trusted_types=trusted_types,  # type: ignore[arg-type]
  )


# ---------------------------------------------------------------------------
# Array types (shape-checked via beartype Is[])
# ---------------------------------------------------------------------------

if tp.TYPE_CHECKING:
  _Dims = tp.TypeVarTuple("_Dims")

  Bool = tp.TypeAliasType("Bool", CuPyArray, type_params=(_Dims,))

  I8 = tp.TypeAliasType("I8", CuPyArray, type_params=(_Dims,))
  I16 = tp.TypeAliasType("I16", CuPyArray, type_params=(_Dims,))
  I32 = tp.TypeAliasType("I32", CuPyArray, type_params=(_Dims,))
  I64 = tp.TypeAliasType("I64", CuPyArray, type_params=(_Dims,))

  U8 = tp.TypeAliasType("U8", CuPyArray, type_params=(_Dims,))
  U16 = tp.TypeAliasType("U16", CuPyArray, type_params=(_Dims,))
  U32 = tp.TypeAliasType("U32", CuPyArray, type_params=(_Dims,))
  U64 = tp.TypeAliasType("U64", CuPyArray, type_params=(_Dims,))

  F16 = tp.TypeAliasType("F16", CuPyArray, type_params=(_Dims,))
  F32 = tp.TypeAliasType("F32", CuPyArray, type_params=(_Dims,))
  F64 = tp.TypeAliasType("F64", CuPyArray, type_params=(_Dims,))

  C64 = tp.TypeAliasType("C64", CuPyArray, type_params=(_Dims,))
  C128 = tp.TypeAliasType("C128", CuPyArray, type_params=(_Dims,))

  Int = tp.TypeAliasType("Int", CuPyArray, type_params=(_Dims,))
  UInt = tp.TypeAliasType("UInt", CuPyArray, type_params=(_Dims,))
  Integer = tp.TypeAliasType("Integer", CuPyArray, type_params=(_Dims,))
  Float = tp.TypeAliasType("Float", CuPyArray, type_params=(_Dims,))
  Real = tp.TypeAliasType("Real", CuPyArray, type_params=(_Dims,))
  Complex = tp.TypeAliasType("Complex", CuPyArray, type_params=(_Dims,))
  Inexact = tp.TypeAliasType("Inexact", CuPyArray, type_params=(_Dims,))
  Num = tp.TypeAliasType("Num", CuPyArray, type_params=(_Dims,))
  Shaped = tp.TypeAliasType("Shaped", CuPyArray, type_params=(_Dims,))

else:
  Bool = make_array_type(CuPyArray, BOOL)

  I8 = make_array_type(CuPyArray, INT8)
  I16 = make_array_type(CuPyArray, INT16)
  I32 = make_array_type(CuPyArray, INT32)
  I64 = make_array_type(CuPyArray, INT64)

  U8 = make_array_type(CuPyArray, UINT8)
  U16 = make_array_type(CuPyArray, UINT16)
  U32 = make_array_type(CuPyArray, UINT32)
  U64 = make_array_type(CuPyArray, UINT64)

  F16 = make_array_type(CuPyArray, FLOAT16)
  F32 = make_array_type(CuPyArray, FLOAT32)
  F64 = make_array_type(CuPyArray, FLOAT64)

  C64 = make_array_type(CuPyArray, COMPLEX64)
  C128 = make_array_type(CuPyArray, COMPLEX128)

  Int = make_array_type(CuPyArray, INT)
  UInt = make_array_type(CuPyArray, UINT)
  Integer = make_array_type(CuPyArray, INTEGER)
  Float = make_array_type(CuPyArray, FLOAT)
  Real = make_array_type(CuPyArray, REAL)
  Complex = make_array_type(CuPyArray, COMPLEX)
  Inexact = make_array_type(CuPyArray, INEXACT)
  Num = make_array_type(CuPyArray, NUM)
  Shaped = make_array_type(CuPyArray, SHAPED)

# ---------------------------------------------------------------------------
# Like types — runtime: scalar | array | nested sequences; static: CuPyArray
# ---------------------------------------------------------------------------

if tp.TYPE_CHECKING:
  BoolLike = tp.TypeAliasType("BoolLike", CuPyArray, type_params=(_Dims,))

  I8Like = tp.TypeAliasType("I8Like", CuPyArray, type_params=(_Dims,))
  I16Like = tp.TypeAliasType("I16Like", CuPyArray, type_params=(_Dims,))
  I32Like = tp.TypeAliasType("I32Like", CuPyArray, type_params=(_Dims,))
  I64Like = tp.TypeAliasType("I64Like", CuPyArray, type_params=(_Dims,))

  U8Like = tp.TypeAliasType("U8Like", CuPyArray, type_params=(_Dims,))
  U16Like = tp.TypeAliasType("U16Like", CuPyArray, type_params=(_Dims,))
  U32Like = tp.TypeAliasType("U32Like", CuPyArray, type_params=(_Dims,))
  U64Like = tp.TypeAliasType("U64Like", CuPyArray, type_params=(_Dims,))

  F16Like = tp.TypeAliasType("F16Like", CuPyArray, type_params=(_Dims,))
  F32Like = tp.TypeAliasType("F32Like", CuPyArray, type_params=(_Dims,))
  F64Like = tp.TypeAliasType("F64Like", CuPyArray, type_params=(_Dims,))

  C64Like = tp.TypeAliasType("C64Like", CuPyArray, type_params=(_Dims,))
  C128Like = tp.TypeAliasType("C128Like", CuPyArray, type_params=(_Dims,))

  IntLike = tp.TypeAliasType("IntLike", CuPyArray, type_params=(_Dims,))
  UIntLike = tp.TypeAliasType("UIntLike", CuPyArray, type_params=(_Dims,))
  IntegerLike = tp.TypeAliasType("IntegerLike", CuPyArray, type_params=(_Dims,))
  FloatLike = tp.TypeAliasType("FloatLike", CuPyArray, type_params=(_Dims,))
  RealLike = tp.TypeAliasType("RealLike", CuPyArray, type_params=(_Dims,))
  ComplexLike = tp.TypeAliasType("ComplexLike", CuPyArray, type_params=(_Dims,))
  InexactLike = tp.TypeAliasType("InexactLike", CuPyArray, type_params=(_Dims,))
  NumLike = tp.TypeAliasType("NumLike", CuPyArray, type_params=(_Dims,))
  ShapedLike = tp.TypeAliasType("ShapedLike", CuPyArray, type_params=(_Dims,))

else:
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
