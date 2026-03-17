"""Dtype specifications and matching for numpy, JAX, and PyTorch arrays.

Each :class:`DtypeSpec` describes a set of allowed dtype strings, with optional
byte-order constraints. Pre-defined specs cover individual dtypes
(``FLOAT32``, ``INT64``, …), category groups (``FLOAT``, ``INTEGER``, …), and
endianness variants (``FLOAT32_LE``, ``INT64_BE``, …).

The :func:`extract_dtype_str` function normalises dtype extraction across
numpy, JAX, PyTorch, and TensorFlow arrays into a canonical string like
``"float32"`` or ``"int64"``.
"""

from __future__ import annotations

import typing as tp
from dataclasses import dataclass, field

__all__ = [
  # Core
  "DtypeSpec",
  "extract_dtype_str",
  # Individual types
  "BOOL",
  "INT8",
  "INT16",
  "INT32",
  "INT64",
  "UINT8",
  "UINT16",
  "UINT32",
  "UINT64",
  "BFLOAT16",
  "FLOAT16",
  "FLOAT32",
  "FLOAT64",
  "FLOAT128",
  "COMPLEX64",
  "COMPLEX128",
  "COMPLEX256",
  "VOID",
  "STRING",
  "BYTES",
  "OBJECT",
  "DATETIME64",
  "TIMEDELTA64",
  # Category groups
  "INT",
  "UINT",
  "INTEGER",
  "FLOAT",
  "REAL",
  "COMPLEX",
  "INEXACT",
  "NUM",
  "SHAPED",
  # Endianness — signed integers
  "INT16_LE",
  "INT16_BE",
  "INT16_N",
  "INT32_LE",
  "INT32_BE",
  "INT32_N",
  "INT64_LE",
  "INT64_BE",
  "INT64_N",
  # Endianness — unsigned integers
  "UINT16_LE",
  "UINT16_BE",
  "UINT16_N",
  "UINT32_LE",
  "UINT32_BE",
  "UINT32_N",
  "UINT64_LE",
  "UINT64_BE",
  "UINT64_N",
  # Endianness — floating point
  "BFLOAT16_LE",
  "BFLOAT16_BE",
  "BFLOAT16_N",
  "FLOAT16_LE",
  "FLOAT16_BE",
  "FLOAT16_N",
  "FLOAT32_LE",
  "FLOAT32_BE",
  "FLOAT32_N",
  "FLOAT64_LE",
  "FLOAT64_BE",
  "FLOAT64_N",
  "FLOAT128_LE",
  "FLOAT128_BE",
  "FLOAT128_N",
  # Endianness — complex
  "COMPLEX64_LE",
  "COMPLEX64_BE",
  "COMPLEX64_N",
  "COMPLEX128_LE",
  "COMPLEX128_BE",
  "COMPLEX128_N",
  "COMPLEX256_LE",
  "COMPLEX256_BE",
  "COMPLEX256_N",
  # Endianness — category groups
  "INT_LE",
  "INT_BE",
  "INT_N",
  "UINT_LE",
  "UINT_BE",
  "UINT_N",
  "INTEGER_LE",
  "INTEGER_BE",
  "INTEGER_N",
  "FLOAT_LE",
  "FLOAT_BE",
  "FLOAT_N",
  "REAL_LE",
  "REAL_BE",
  "REAL_N",
  "COMPLEX_LE",
  "COMPLEX_BE",
  "COMPLEX_N",
  "INEXACT_LE",
  "INEXACT_BE",
  "INEXACT_N",
  "NUM_LE",
  "NUM_BE",
  "NUM_N",
  "SHAPED_LE",
  "SHAPED_BE",
  "SHAPED_N",
]

# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DtypeSpec:
  """Describes a set of allowed dtypes with optional byte-order constraint.

  Parameters
  ----------
  name:
      Human-readable label used in error messages (e.g. ``"Float32"``).
  allowed:
      Frozenset of canonical dtype strings that this spec accepts
      (e.g. ``frozenset({"float32"})``).
  byteorder:
      Byte-order constraint: ``"any"`` (default), ``"little"``, ``"big"``,
      or ``"native"``.
  _structured:
      (Internal) A numpy dtype object for exact structured dtype matching.
      Use :meth:`structured` to create structured specs.

  Example::

      SMALL_FLOAT = DtypeSpec("SmallFloat", frozenset({"float16", "float32"}))
      SMALL_FLOAT.matches(np.zeros(1, dtype=np.float32))  # True
      SMALL_FLOAT.matches(np.zeros(1, dtype=np.float64))  # False

      INT32_LE = DtypeSpec("Int32LE", frozenset({"int32"}), byteorder="little")
      INT32_LE.matches(np.zeros(1, dtype="<i4"))  # True
      INT32_LE.matches(np.zeros(1, dtype=">i4"))  # False
  """

  name: str
  allowed: frozenset[str]
  byteorder: str = "any"
  _structured: object = field(default=None, repr=False, compare=False)

  _VALID_BYTEORDERS: tp.ClassVar[frozenset[str]] = frozenset({
    "any",
    "little",
    "big",
    "native",
  })

  def __post_init__(self) -> None:
    if self.byteorder not in self._VALID_BYTEORDERS:
      msg = f"Invalid byteorder {self.byteorder!r}, must be one of {sorted(self._VALID_BYTEORDERS)}"
      raise ValueError(msg)

  def matches(self, obj: object) -> bool:
    """Return ``True`` if *obj*'s dtype matches this spec."""
    dtype_str = extract_dtype_str(obj)
    if not dtype_str:
      return False

    # Wildcard: any dtype passes (used by SHAPED)
    if "*" in self.allowed:
      return self._check_byteorder(obj)

    if dtype_str not in self.allowed:
      return False

    # Structured dtype: exact layout comparison
    if self._structured is not None:
      obj_dtype = getattr(obj, "dtype", None)
      if obj_dtype is None or obj_dtype != self._structured:
        return False

    return self._check_byteorder(obj)

  def _check_byteorder(self, obj: object) -> bool:
    if self.byteorder == "any":
      return True

    dtype = getattr(obj, "dtype", None)
    if dtype is None:
      return True  # scalars have no byte order

    # dtype.str always resolves to '<', '>', or '|' (never '=')
    dt_str: str = getattr(dtype, "str", "")
    if not dt_str:
      return True

    bo = dt_str[0]
    if bo == "|":
      return True  # single-byte / not applicable

    if self.byteorder == "native":
      return bool(getattr(dtype, "isnative", True))
    if self.byteorder == "little":
      return bo == "<"
    if self.byteorder == "big":
      return bo == ">"
    return True

  @staticmethod
  def structured(dtype: object) -> DtypeSpec:
    """Create a DtypeSpec matching a specific structured numpy dtype.

    Example::

        import numpy as np

        point_dt = np.dtype([("x", np.float32), ("y", np.float32)])
        POINT = DtypeSpec.structured(point_dt)
    """
    import numpy as np

    dt = np.dtype(dtype)  # type: ignore[arg-type]
    return DtypeSpec(
      name=f"Structured({dt})", allowed=frozenset({"void"}), _structured=dt
    )


def extract_dtype_str(obj: object) -> str:
  """Extract a canonical dtype string from a numpy / JAX / PyTorch array.

  Returns a short string like ``"float32"``, ``"int64"``, ``"bool"``, etc.
  Returns ``""`` if the object has no ``dtype`` attribute.
  """
  dtype = getattr(obj, "dtype", None)
  if dtype is None:
    return ""

  # NumPy / JAX: dtype.name gives canonical names like "float32", "float128",
  # "complex256", "datetime64", "str", ...
  dtype_name = getattr(dtype, "name", None)
  if isinstance(dtype_name, str) and dtype_name:
    if dtype_name.startswith("void"):
      return "void"
    if dtype_name.startswith("bytes"):
      return "bytes"
    if dtype_name.startswith("str"):
      return "str"
    return dtype_name

  # NumPy / JAX: dtype.type.__name__ (e.g. "float32")
  dtype_type = getattr(dtype, "type", None)
  if dtype_type is not None:
    name = getattr(dtype_type, "__name__", None)
    if name is not None:
      return name

  # TensorFlow: dtype.as_numpy_dtype.__name__
  as_numpy = getattr(dtype, "as_numpy_dtype", None)
  if as_numpy is not None:
    return getattr(as_numpy, "__name__", str(dtype))

  # PyTorch / fallback: repr(dtype) → "torch.float32" → "float32"
  s = dtype if isinstance(dtype, str) else repr(dtype)
  _, _, tail = s.rpartition(".")
  return tail or s


# ---------------------------------------------------------------------------
# Endianness helpers
# ---------------------------------------------------------------------------


def _le(spec: DtypeSpec) -> DtypeSpec:
  """Create a little-endian variant of *spec*."""
  return DtypeSpec(spec.name + "LE", spec.allowed, byteorder="little")


def _be(spec: DtypeSpec) -> DtypeSpec:
  """Create a big-endian variant of *spec*."""
  return DtypeSpec(spec.name + "BE", spec.allowed, byteorder="big")


def _native(spec: DtypeSpec) -> DtypeSpec:
  """Create a native-byte-order variant of *spec*."""
  return DtypeSpec(spec.name + "N", spec.allowed, byteorder="native")


# ---------------------------------------------------------------------------
# Pre-defined dtype specs — individual types
# ---------------------------------------------------------------------------

# Boolean (single-byte — no endianness)
BOOL = DtypeSpec("Bool", frozenset({"bool_", "bool"}))

# Signed integers
INT8 = DtypeSpec("Int8", frozenset({"int8"}))  # single-byte
INT16 = DtypeSpec("Int16", frozenset({"int16"}))
INT32 = DtypeSpec("Int32", frozenset({"int32"}))
INT64 = DtypeSpec("Int64", frozenset({"int64"}))

# Unsigned integers
UINT8 = DtypeSpec("UInt8", frozenset({"uint8"}))  # single-byte
UINT16 = DtypeSpec("UInt16", frozenset({"uint16"}))
UINT32 = DtypeSpec("UInt32", frozenset({"uint32"}))
UINT64 = DtypeSpec("UInt64", frozenset({"uint64"}))

# Floating point
BFLOAT16 = DtypeSpec("BFloat16", frozenset({"bfloat16"}))
FLOAT16 = DtypeSpec("Float16", frozenset({"float16"}))
FLOAT32 = DtypeSpec("Float32", frozenset({"float32"}))
FLOAT64 = DtypeSpec("Float64", frozenset({"float64"}))
FLOAT128 = DtypeSpec("Float128", frozenset({"float128", "longdouble"}))

# Complex
COMPLEX64 = DtypeSpec("Complex64", frozenset({"complex64"}))
COMPLEX128 = DtypeSpec("Complex128", frozenset({"complex128"}))
COMPLEX256 = DtypeSpec("Complex256", frozenset({"complex256", "clongdouble"}))

# Void / structured / string / object / datetime
VOID = DtypeSpec("Void", frozenset({"void"}))
STRING = DtypeSpec("Str", frozenset({"str_", "str"}))
BYTES = DtypeSpec("Bytes", frozenset({"bytes_", "bytes"}))
OBJECT = DtypeSpec("Obj", frozenset({"object_", "object"}))
DATETIME64 = DtypeSpec("DateTime64", frozenset({"datetime64"}))
TIMEDELTA64 = DtypeSpec("TimeDelta64", frozenset({"timedelta64"}))

# ---------------------------------------------------------------------------
# Endianness variants — individual types (multi-byte only)
# ---------------------------------------------------------------------------

# Signed integers
INT16_LE = _le(INT16)
INT16_BE = _be(INT16)
INT16_N = _native(INT16)
INT32_LE = _le(INT32)
INT32_BE = _be(INT32)
INT32_N = _native(INT32)
INT64_LE = _le(INT64)
INT64_BE = _be(INT64)
INT64_N = _native(INT64)

# Unsigned integers
UINT16_LE = _le(UINT16)
UINT16_BE = _be(UINT16)
UINT16_N = _native(UINT16)
UINT32_LE = _le(UINT32)
UINT32_BE = _be(UINT32)
UINT32_N = _native(UINT32)
UINT64_LE = _le(UINT64)
UINT64_BE = _be(UINT64)
UINT64_N = _native(UINT64)

# Floating point
BFLOAT16_LE = _le(BFLOAT16)
BFLOAT16_BE = _be(BFLOAT16)
BFLOAT16_N = _native(BFLOAT16)
FLOAT16_LE = _le(FLOAT16)
FLOAT16_BE = _be(FLOAT16)
FLOAT16_N = _native(FLOAT16)
FLOAT32_LE = _le(FLOAT32)
FLOAT32_BE = _be(FLOAT32)
FLOAT32_N = _native(FLOAT32)
FLOAT64_LE = _le(FLOAT64)
FLOAT64_BE = _be(FLOAT64)
FLOAT64_N = _native(FLOAT64)
FLOAT128_LE = _le(FLOAT128)
FLOAT128_BE = _be(FLOAT128)
FLOAT128_N = _native(FLOAT128)

# Complex
COMPLEX64_LE = _le(COMPLEX64)
COMPLEX64_BE = _be(COMPLEX64)
COMPLEX64_N = _native(COMPLEX64)
COMPLEX128_LE = _le(COMPLEX128)
COMPLEX128_BE = _be(COMPLEX128)
COMPLEX128_N = _native(COMPLEX128)
COMPLEX256_LE = _le(COMPLEX256)
COMPLEX256_BE = _be(COMPLEX256)
COMPLEX256_N = _native(COMPLEX256)

# ---------------------------------------------------------------------------
# Pre-defined dtype specs — category groups
# ---------------------------------------------------------------------------

_INTS = INT8.allowed | INT16.allowed | INT32.allowed | INT64.allowed
_UINTS = UINT8.allowed | UINT16.allowed | UINT32.allowed | UINT64.allowed
_FLOATS = (
  BFLOAT16.allowed
  | FLOAT16.allowed
  | FLOAT32.allowed
  | FLOAT64.allowed
  | FLOAT128.allowed
)
_COMPLEXES = COMPLEX64.allowed | COMPLEX128.allowed | COMPLEX256.allowed

INT = DtypeSpec("Int", _INTS)
UINT = DtypeSpec("UInt", _UINTS)
INTEGER = DtypeSpec("Integer", _INTS | _UINTS)
FLOAT = DtypeSpec("Float", _FLOATS)
REAL = DtypeSpec("Real", _INTS | _UINTS | _FLOATS)
COMPLEX = DtypeSpec("Complex", _COMPLEXES)
INEXACT = DtypeSpec("Inexact", _FLOATS | _COMPLEXES)
NUM = DtypeSpec("Num", _INTS | _UINTS | _FLOATS | _COMPLEXES)
SHAPED = DtypeSpec("Shaped", frozenset({"*"}))

# ---------------------------------------------------------------------------
# Endianness variants — category groups
# ---------------------------------------------------------------------------

INT_LE = _le(INT)
INT_BE = _be(INT)
INT_N = _native(INT)
UINT_LE = _le(UINT)
UINT_BE = _be(UINT)
UINT_N = _native(UINT)
INTEGER_LE = _le(INTEGER)
INTEGER_BE = _be(INTEGER)
INTEGER_N = _native(INTEGER)
FLOAT_LE = _le(FLOAT)
FLOAT_BE = _be(FLOAT)
FLOAT_N = _native(FLOAT)
REAL_LE = _le(REAL)
REAL_BE = _be(REAL)
REAL_N = _native(REAL)
COMPLEX_LE = _le(COMPLEX)
COMPLEX_BE = _be(COMPLEX)
COMPLEX_N = _native(COMPLEX)
INEXACT_LE = _le(INEXACT)
INEXACT_BE = _be(INEXACT)
INEXACT_N = _native(INEXACT)
NUM_LE = _le(NUM)
NUM_BE = _be(NUM)
NUM_N = _native(NUM)
SHAPED_LE = _le(SHAPED)
SHAPED_BE = _be(SHAPED)
SHAPED_N = _native(SHAPED)
