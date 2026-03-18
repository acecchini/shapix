"""Tests for _dtypes.py — dtype extraction and spec matching across all backends."""

from __future__ import annotations

import numpy as np
import pytest

from shapix._dtypes import (
  BOOL,
  COMPLEX,
  COMPLEX64,
  COMPLEX128,
  COMPLEX256,
  FLOAT,
  FLOAT16,
  FLOAT32,
  FLOAT64,
  FLOAT128,
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
  DtypeSpec,
  extract_dtype_str,
)


# =====================================================================
# extract_dtype_str — NumPy
# =====================================================================


class TestExtractDtypeStrNumpy:
  @pytest.mark.parametrize(
    "np_dtype, expected",
    [
      (np.bool_, "bool"),
      (np.int8, "int8"),
      (np.int16, "int16"),
      (np.int32, "int32"),
      (np.int64, "int64"),
      (np.uint8, "uint8"),
      (np.uint16, "uint16"),
      (np.uint32, "uint32"),
      (np.uint64, "uint64"),
      (np.float16, "float16"),
      (np.float32, "float32"),
      (np.float64, "float64"),
      (np.complex64, "complex64"),
      (np.complex128, "complex128"),
    ],
  )
  def test_numpy_dtypes(self, np_dtype: type, expected: str) -> None:
    assert extract_dtype_str(np.zeros(2, dtype=np_dtype)) == expected

  def test_no_dtype_attr(self) -> None:
    assert extract_dtype_str(42) == ""

  def test_structured_dtype(self) -> None:
    dt = np.dtype([("x", np.float32), ("y", np.int32)])
    arr = np.zeros(2, dtype=dt)
    assert extract_dtype_str(arr) == "void"


# =====================================================================
# extract_dtype_str — PyTorch
# =====================================================================


class TestExtractDtypeStrTorch:
  @pytest.fixture(autouse=True)
  def _skip(self) -> None:
    pytest.importorskip("torch")

  @pytest.mark.parametrize(
    "torch_attr, expected",
    [
      ("bool", "bool"),
      ("int8", "int8"),
      ("int16", "int16"),
      ("int32", "int32"),
      ("int64", "int64"),
      ("uint8", "uint8"),
      ("float16", "float16"),
      ("float32", "float32"),
      ("float64", "float64"),
      ("bfloat16", "bfloat16"),
      ("complex64", "complex64"),
      ("complex128", "complex128"),
    ],
  )
  def test_torch_dtypes(self, torch_attr: str, expected: str) -> None:
    import torch

    dt = getattr(torch, torch_attr)
    assert extract_dtype_str(torch.ones(2, dtype=dt)) == expected


# =====================================================================
# extract_dtype_str — JAX
# =====================================================================


class TestExtractDtypeStrJax:
  @pytest.fixture(autouse=True)
  def _skip(self) -> None:
    pytest.importorskip("jax")

  @pytest.mark.parametrize(
    "jnp_attr, expected",
    [
      ("bool_", "bool"),
      ("int8", "int8"),
      ("int16", "int16"),
      ("int32", "int32"),
      ("uint8", "uint8"),
      ("uint16", "uint16"),
      ("uint32", "uint32"),
      ("float16", "float16"),
      ("float32", "float32"),
      ("bfloat16", "bfloat16"),
      ("complex64", "complex64"),
    ],
  )
  def test_jax_dtypes(self, jnp_attr: str, expected: str) -> None:
    import jax.numpy as jnp

    dt = getattr(jnp, jnp_attr)
    assert extract_dtype_str(jnp.ones(2, dtype=dt)) == expected


class TestExtractDtypeStrJax64:
  """64-bit JAX dtypes require x64 mode."""

  @pytest.fixture(autouse=True)
  def _enable_x64(self) -> None:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    yield  # type: ignore[func-returns-value]
    jax.config.update("jax_enable_x64", False)

  @pytest.mark.parametrize(
    "jnp_attr, expected",
    [
      ("int64", "int64"),
      ("uint64", "uint64"),
      ("float64", "float64"),
      ("complex128", "complex128"),
    ],
  )
  def test_jax_64bit_dtypes(self, jnp_attr: str, expected: str) -> None:
    import jax.numpy as jnp

    dt = getattr(jnp, jnp_attr)
    assert extract_dtype_str(jnp.ones(2, dtype=dt)) == expected


# =====================================================================
# DtypeSpec.matches — parametrized
# =====================================================================


class TestDtypeSpecMatches:
  @pytest.mark.parametrize(
    "spec, np_dtype, expected",
    [
      # Concrete types
      (FLOAT32, np.float32, True),
      (FLOAT32, np.float64, False),
      (FLOAT32, np.int32, False),
      (FLOAT64, np.float64, True),
      (FLOAT64, np.float32, False),
      (FLOAT16, np.float16, True),
      (INT32, np.int32, True),
      (INT32, np.float32, False),
      (INT64, np.int64, True),
      (INT64, np.uint8, False),
      (INT8, np.int8, True),
      (INT16, np.int16, True),
      (UINT8, np.uint8, True),
      (UINT8, np.int8, False),
      (UINT16, np.uint16, True),
      (UINT32, np.uint32, True),
      (UINT64, np.uint64, True),
      (BOOL, np.bool_, True),
      (BOOL, np.int32, False),
      (COMPLEX64, np.complex64, True),
      (COMPLEX128, np.complex128, True),
      (COMPLEX128, np.float64, False),
      # Category types
      (FLOAT, np.float32, True),
      (FLOAT, np.float64, True),
      (FLOAT, np.float16, True),
      (FLOAT, np.int32, False),
      (INT, np.int32, True),
      (INT, np.int64, True),
      (INT, np.uint8, False),
      (UINT, np.uint8, True),
      (UINT, np.int32, False),
      (INTEGER, np.int32, True),
      (INTEGER, np.uint8, True),
      (INTEGER, np.float32, False),
      (REAL, np.float32, True),
      (REAL, np.int64, True),
      (REAL, np.complex128, False),
      (COMPLEX, np.complex64, True),
      (COMPLEX, np.complex128, True),
      (COMPLEX, np.float64, False),
      (INEXACT, np.float32, True),
      (INEXACT, np.complex64, True),
      (INEXACT, np.int32, False),
      (NUM, np.float32, True),
      (NUM, np.int64, True),
      (NUM, np.complex128, True),
      (NUM, np.bool_, False),
      (SHAPED, np.float32, True),
      (SHAPED, np.bool_, True),
      (SHAPED, np.int32, True),
    ],
  )
  def test_matches(self, spec: DtypeSpec, np_dtype: type, expected: bool) -> None:
    assert spec.matches(np.zeros(2, dtype=np_dtype)) is expected


class TestCustomDtypeSpec:
  def test_custom_spec(self) -> None:
    spec = DtypeSpec("MyType", frozenset({"float32", "float64"}))
    assert spec.matches(np.zeros(2, dtype=np.float32))
    assert spec.matches(np.zeros(2, dtype=np.float64))
    assert not spec.matches(np.zeros(2, dtype=np.int32))


class TestShapedVoidAndStructured:
  def test_void_dtype_is_shaped(self) -> None:
    dt = np.dtype("V8")
    arr = np.zeros(2, dtype=dt)
    assert extract_dtype_str(arr) == "void"
    assert SHAPED.matches(arr)

  def test_structured_dtype_is_shaped(self) -> None:
    dt = np.dtype([("x", np.float32), ("y", np.int32)])
    arr = np.zeros(2, dtype=dt)
    assert extract_dtype_str(arr) == "void"
    assert SHAPED.matches(arr)

  def test_custom_structured_dtype_spec_is_exact(self) -> None:
    dt_xy = np.dtype([("x", np.float32), ("y", np.int32)])
    dt_xz = np.dtype([("x", np.float32), ("z", np.int32)])

    spec = DtypeSpec.structured(dt_xy)
    assert spec.matches(np.zeros(2, dtype=dt_xy))
    assert not spec.matches(np.zeros(2, dtype=dt_xz))


class TestDatetimeTimedelta:
  @pytest.mark.parametrize(
    "dtype_str, expected",
    [
      ("datetime64[ns]", "datetime64"),
      ("datetime64[D]", "datetime64"),
      ("datetime64[us]", "datetime64"),
      ("datetime64", "datetime64"),
      ("timedelta64[ms]", "timedelta64"),
      ("timedelta64[s]", "timedelta64"),
      ("timedelta64[ns]", "timedelta64"),
      ("timedelta64", "timedelta64"),
    ],
  )
  def test_extract_datetime_timedelta(self, dtype_str: str, expected: str) -> None:
    arr = np.zeros(2, dtype=dtype_str)
    assert extract_dtype_str(arr) == expected

  def test_datetime64_spec_matches_unit_qualified(self) -> None:
    from shapix._dtypes import DATETIME64

    assert DATETIME64.matches(np.zeros(2, dtype="datetime64[ns]"))
    assert DATETIME64.matches(np.zeros(2, dtype="datetime64[D]"))

  def test_timedelta64_spec_matches_unit_qualified(self) -> None:
    from shapix._dtypes import TIMEDELTA64

    assert TIMEDELTA64.matches(np.zeros(2, dtype="timedelta64[ms]"))
    assert TIMEDELTA64.matches(np.zeros(2, dtype="timedelta64[s]"))


class TestDtypeEdgeCases:
  def test_shaped_matches_bfloat16(self) -> None:
    torch = pytest.importorskip("torch")
    assert SHAPED.matches(torch.ones(2, dtype=torch.bfloat16))

  def test_matches_no_dtype_returns_false(self) -> None:
    assert SHAPED.matches(42) is False

  def test_custom_spec_empty_allowed_matches_nothing(self) -> None:
    spec = DtypeSpec("Empty", frozenset())
    assert not spec.matches(np.zeros(2, dtype=np.float32))

  def test_bool_is_not_num(self) -> None:
    assert not NUM.matches(np.zeros(2, dtype=np.bool_))

  def test_structured_spec_rejects_plain_void(self) -> None:
    dt_xy = np.dtype([("x", np.float32), ("y", np.int32)])
    spec = DtypeSpec.structured(dt_xy)
    plain_void = np.zeros(2, dtype="V8")
    assert not spec.matches(plain_void)

  def test_structured_spec_rejects_no_dtype(self) -> None:
    dt = np.dtype([("x", np.float32)])
    spec = DtypeSpec.structured(dt)
    assert not spec.matches(42)


# =====================================================================
# Endianness (byteorder) matching
# =====================================================================


class TestByteorderMatching:
  def test_le_accepts_little_endian(self) -> None:
    from shapix._dtypes import FLOAT32_LE

    arr = np.zeros(2, dtype="<f4")
    assert FLOAT32_LE.matches(arr)

  def test_le_rejects_big_endian(self) -> None:
    from shapix._dtypes import FLOAT32_LE

    arr = np.zeros(2, dtype=">f4")
    assert not FLOAT32_LE.matches(arr)

  def test_be_accepts_big_endian(self) -> None:
    from shapix._dtypes import FLOAT32_BE

    arr = np.zeros(2, dtype=">f4")
    assert FLOAT32_BE.matches(arr)

  def test_be_rejects_little_endian(self) -> None:
    from shapix._dtypes import FLOAT32_BE

    arr = np.zeros(2, dtype="<f4")
    assert not FLOAT32_BE.matches(arr)

  def test_native_accepts_native(self) -> None:
    from shapix._dtypes import FLOAT32_N

    arr = np.zeros(2, dtype=np.float32)
    assert FLOAT32_N.matches(arr)

  def test_any_byteorder_accepts_all(self) -> None:
    arr_le = np.zeros(2, dtype="<f4")
    arr_be = np.zeros(2, dtype=">f4")
    assert FLOAT32.matches(arr_le)
    assert FLOAT32.matches(arr_be)

  def test_single_byte_always_passes(self) -> None:
    from shapix._dtypes import INT8

    arr = np.zeros(2, dtype=np.int8)
    # Single-byte dtype has "|" byte order, should pass any constraint
    assert INT8.matches(arr)

  def test_le_category_group(self) -> None:
    from shapix._dtypes import FLOAT_LE

    arr_le = np.zeros(2, dtype="<f4")
    arr_be = np.zeros(2, dtype=">f4")
    assert FLOAT_LE.matches(arr_le)
    assert not FLOAT_LE.matches(arr_be)

  def test_shaped_le_accepts_any_dtype_le(self) -> None:
    from shapix._dtypes import SHAPED_LE

    arr_le = np.zeros(2, dtype="<f4")
    arr_be = np.zeros(2, dtype=">f4")
    assert SHAPED_LE.matches(arr_le)
    assert not SHAPED_LE.matches(arr_be)

  def test_byteorder_no_dtype_attr_passes(self) -> None:
    """Objects without .dtype always pass byteorder checks."""
    from shapix._dtypes import FLOAT32_LE

    # Manually check: no dtype attr → _check_byteorder returns True
    assert FLOAT32_LE._check_byteorder(42)

  def test_byteorder_empty_str_passes(self) -> None:
    """dtype with empty .str always passes."""
    from shapix._dtypes import FLOAT32_LE

    class FakeDtype:
      str = ""

    class FakeArr:
      dtype = FakeDtype()

    assert FLOAT32_LE._check_byteorder(FakeArr())


class TestExtendedPrecisionDtypes:
  @pytest.mark.skipif(
    np.dtype(np.longdouble).name not in FLOAT128.allowed,
    reason="platform does not expose float128/longdouble distinctly",
  )
  def test_float128_matches(self) -> None:
    arr = np.zeros(2, dtype=np.longdouble)
    assert extract_dtype_str(arr) in FLOAT128.allowed
    assert FLOAT128.matches(arr)

  @pytest.mark.skipif(
    np.dtype(np.clongdouble).name not in COMPLEX256.allowed,
    reason="platform does not expose complex256/clongdouble distinctly",
  )
  def test_complex256_matches(self) -> None:
    arr = np.zeros(2, dtype=np.clongdouble)
    assert extract_dtype_str(arr) in COMPLEX256.allowed
    assert COMPLEX256.matches(arr)
    assert COMPLEX.matches(arr)
