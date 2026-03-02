"""Dtype specifications and matching for numpy, JAX, and PyTorch arrays.

Each :class:`DtypeSpec` describes a set of allowed dtype strings. Pre-defined
specs cover individual dtypes (``FLOAT32``, ``INT64``, …) and category groups
(``FLOAT``, ``INTEGER``, ``NUM``, ``SHAPED``, …).

The :func:`extract_dtype_str` function normalises dtype extraction across
numpy, JAX, PyTorch, and TensorFlow arrays into a canonical string like
``"float32"`` or ``"int64"``.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "DtypeSpec",
    "extract_dtype_str",
    # Boolean
    "BOOL",
    # Signed integers
    "INT8",
    "INT16",
    "INT32",
    "INT64",
    # Unsigned integers
    "UINT8",
    "UINT16",
    "UINT32",
    "UINT64",
    # Floats
    "BFLOAT16",
    "FLOAT16",
    "FLOAT32",
    "FLOAT64",
    # Complex
    "COMPLEX64",
    "COMPLEX128",
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
]


@dataclass(frozen=True, slots=True)
class DtypeSpec:
    """Describes a set of allowed dtypes by their canonical string names.

    Parameters
    ----------
    name:
        Human-readable label used in error messages (e.g. ``"Float32"``).
    allowed:
        Frozenset of canonical dtype strings that this spec accepts
        (e.g. ``frozenset({"float32"})``).

    Example::

        SMALL_FLOAT = DtypeSpec("SmallFloat", frozenset({"float16", "float32"}))
        SMALL_FLOAT.matches(np.zeros(1, dtype=np.float32))  # True
        SMALL_FLOAT.matches(np.zeros(1, dtype=np.float64))  # False
    """

    name: str
    allowed: frozenset[str]

    def matches(self, obj: object) -> bool:
        """Return ``True`` if *obj*'s dtype is in the allowed set."""
        return extract_dtype_str(obj) in self.allowed


def extract_dtype_str(obj: object) -> str:
    """Extract a canonical dtype string from a numpy / JAX / PyTorch array.

    Returns a short string like ``"float32"``, ``"int64"``, ``"bool"``, etc.
    Returns ``""`` if the object has no ``dtype`` attribute.
    """
    dtype = getattr(obj, "dtype", None)
    if dtype is None:
        return ""

    # NumPy / JAX: dtype.type.__name__ (e.g. "float32")
    dtype_type = getattr(dtype, "type", None)
    if dtype_type is not None:
        name = getattr(dtype_type, "__name__", None)
        if name is not None:
            # NumPy structured arrays: dtype.type.__name__ == "void"
            if name == "void":
                return str(dtype)
            return name

    # TensorFlow: dtype.as_numpy_dtype.__name__
    as_numpy = getattr(dtype, "as_numpy_dtype", None)
    if as_numpy is not None:
        return getattr(as_numpy, "__name__", str(dtype))

    # PyTorch / fallback: repr(dtype) → "torch.float32" → "float32"
    s = str(dtype) if isinstance(dtype, str) else repr(dtype)
    _, _, tail = s.rpartition(".")
    return tail or s


# ---------------------------------------------------------------------------
# Pre-defined dtype specs
# ---------------------------------------------------------------------------

# Concrete types
BOOL = DtypeSpec("Bool", frozenset({"bool_", "bool"}))

INT8 = DtypeSpec("Int8", frozenset({"int8"}))
INT16 = DtypeSpec("Int16", frozenset({"int16"}))
INT32 = DtypeSpec("Int32", frozenset({"int32"}))
INT64 = DtypeSpec("Int64", frozenset({"int64"}))

UINT8 = DtypeSpec("UInt8", frozenset({"uint8"}))
UINT16 = DtypeSpec("UInt16", frozenset({"uint16"}))
UINT32 = DtypeSpec("UInt32", frozenset({"uint32"}))
UINT64 = DtypeSpec("UInt64", frozenset({"uint64"}))

BFLOAT16 = DtypeSpec("BFloat16", frozenset({"bfloat16"}))
FLOAT16 = DtypeSpec("Float16", frozenset({"float16"}))
FLOAT32 = DtypeSpec("Float32", frozenset({"float32"}))
FLOAT64 = DtypeSpec("Float64", frozenset({"float64"}))

COMPLEX64 = DtypeSpec("Complex64", frozenset({"complex64"}))
COMPLEX128 = DtypeSpec("Complex128", frozenset({"complex128"}))

# Category groups
_INTS = INT8.allowed | INT16.allowed | INT32.allowed | INT64.allowed
_UINTS = UINT8.allowed | UINT16.allowed | UINT32.allowed | UINT64.allowed
_FLOATS = BFLOAT16.allowed | FLOAT16.allowed | FLOAT32.allowed | FLOAT64.allowed
_COMPLEXES = COMPLEX64.allowed | COMPLEX128.allowed

INT = DtypeSpec("Int", _INTS)
UINT = DtypeSpec("UInt", _UINTS)
INTEGER = DtypeSpec("Integer", _INTS | _UINTS)
FLOAT = DtypeSpec("Float", _FLOATS)
REAL = DtypeSpec("Real", _INTS | _UINTS | _FLOATS)
COMPLEX = DtypeSpec("Complex", _COMPLEXES)
INEXACT = DtypeSpec("Inexact", _FLOATS | _COMPLEXES)
NUM = DtypeSpec("Num", _INTS | _UINTS | _FLOATS | _COMPLEXES)
SHAPED = DtypeSpec("Shaped", BOOL.allowed | NUM.allowed | frozenset({"str_", "bytes_", "void", "object_"}))
