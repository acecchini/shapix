"""Targeted edge-path tests for low-frequency runtime branches."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from beartype import BeartypeConf

import shapix._memo as memo_mod
from shapix import N
from shapix._array_types import _ArrayChecker, _to_shape_spec, make_array_type
from shapix._dtypes import FLOAT32, extract_dtype_str
from shapix._memo import ShapeMemo, bindings_str, get_memo
from shapix._shape import FixedDim, NamedDim, VariadicDim, check_shape
from shapix._tree import _TreeFactory
from shapix.claw import shapix_this_package
from shapix.numpy import F32


class TestArrayFactoryEdges:
  def test_shape_checker_rejects_obj_with_dtype_but_no_shape(self) -> None:
    class DtypeOnly:
      dtype = np.dtype(np.float32)

    checker = _ArrayChecker(FLOAT32, (NamedDim("N"),))
    assert checker(DtypeOnly()) is False

  def test_array_factory_repr(self) -> None:
    factory = make_array_type(np.ndarray, FLOAT32)
    assert repr(factory) == "Float32Array"

  def test_to_shape_spec_rejects_unknown_token(self) -> None:
    class Token:
      def __str__(self) -> str:
        return "Token"

    with pytest.raises(TypeError, match="Invalid shape token"):
      _to_shape_spec((Token(),))


class TestByteorderEdgePaths:
  def test_single_byte_with_endianness_spec_passes(self) -> None:
    """Single-byte dtype with non-'any' byteorder spec should always pass (bo='|')."""
    from shapix._dtypes import DtypeSpec

    # Create a little-endian int8 spec — unusual but valid
    spec = DtypeSpec("I8LE", frozenset({"int8"}), byteorder="little")
    arr = np.zeros(2, dtype=np.int8)
    # int8 has "|" byte order, so _check_byteorder returns True
    assert spec.matches(arr)

  def test_multi_byte_little_endian_matches(self) -> None:
    """Multi-byte dtype with matching endianness spec should pass."""
    from shapix._dtypes import DtypeSpec

    spec = DtypeSpec("F32LE", frozenset({"float32"}), byteorder="little")
    arr = np.zeros(2, dtype="<f4")  # explicit little-endian float32
    assert spec.matches(arr)

  def test_multi_byte_big_endian_rejected(self) -> None:
    """Multi-byte dtype with mismatched endianness spec should fail."""
    from shapix._dtypes import DtypeSpec

    spec = DtypeSpec("F32LE", frozenset({"float32"}), byteorder="little")
    arr = np.zeros(2, dtype=">f4")  # explicit big-endian float32
    assert not spec.matches(arr)

  def test_extract_dtype_str_dtype_type_without_name(self) -> None:
    """dtype.type exists but __name__ is None should fall through."""
    from shapix._dtypes import extract_dtype_str

    class FakeDtypeType:
      __name__ = None  # type: ignore[assignment]

    class FakeDtype:
      type = FakeDtypeType

    class FakeArr:
      dtype = FakeDtype()

    # Should fall through to PyTorch/fallback path
    result = extract_dtype_str(FakeArr())
    assert isinstance(result, str)


class TestShapeEdgeBranches:
  def test_variadic_prefix_error_is_returned(self) -> None:
    memo = ShapeMemo()
    spec = (FixedDim(2), VariadicDim("B"), NamedDim("C"))
    err = check_shape((3, 5, 7), spec, memo)
    assert "expected 2 but got 3" in err

  def test_variadic_suffix_error_is_returned(self) -> None:
    memo = ShapeMemo()
    spec = (NamedDim("N"), VariadicDim("B"), FixedDim(7))
    err = check_shape((3, 5, 6), spec, memo)
    assert "expected 7 but got 6" in err

  def test_variadic_broadcast_failure_path(self) -> None:
    memo = ShapeMemo()
    spec = (VariadicDim("B", broadcastable=True), NamedDim("C"))

    assert check_shape((2, 3, 4), spec, memo) == ""
    err = check_shape((5, 4, 4), spec, memo)
    assert "cannot broadcast" in err


class TestDtypeEdgeBranches:
  def test_extract_dtype_str_tensorflow_style_dtype(self) -> None:
    class FakeDType:
      as_numpy_dtype = np.float32

    class FakeTensor:
      dtype = FakeDType()

    assert extract_dtype_str(FakeTensor()) == "float32"


class TestMemoEdgeBranches:
  def test_get_memo_falls_back_if_frame_lookup_fails(
    self, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    def _raise_value_error(_depth: int = 0) -> None:
      raise ValueError

    monkeypatch.setattr(memo_mod.sys, "_getframe", _raise_value_error)
    assert isinstance(get_memo(_depth=999), ShapeMemo)

  def test_bindings_str_formats_single_and_variadic_bindings(self) -> None:
    memo = ShapeMemo(single={"N": 3}, variadic={"B": (False, (2, 4))})
    formatted = bindings_str(memo)
    assert "N=3" in formatted
    assert "~B=(2, 4)" in formatted


class TestTreeFactoryEdgeBranches:
  def test_tuple_with_only_leaf_type(self) -> None:
    tree = _TreeFactory(object, name="Tree")
    hint = tree[F32[N],]
    assert hasattr(hint, "__metadata__")

  def test_tree_factory_repr(self) -> None:
    tree = _TreeFactory(object, name="MyTree")
    assert repr(tree) == "MyTree"

  def test_tree_factory_empty_tuple_raises(self) -> None:
    tree = _TreeFactory(object, name="Tree")
    import pytest

    with pytest.raises(TypeError, match="at least a leaf type"):
      tree[()]

  def test_tree_factory_single_item_not_tuple(self) -> None:
    tree = _TreeFactory(object, name="Tree")
    hint = tree[F32[N]]
    assert hasattr(hint, "__metadata__")

  def test_tree_single_ellipsis_raises(self) -> None:
    tree = _TreeFactory(object, name="Tree")
    import pytest

    with pytest.raises(TypeError, match="At least one structure name"):
      tree[F32[N], ...]

  def test_tree_both_ellipsis_raises(self) -> None:
    from shapix._tree import Structure

    tree = _TreeFactory(object, name="Tree")
    X = Structure("X")
    import pytest

    with pytest.raises(TypeError, match="Cannot have"):
      tree[F32[N], ..., X, ...]


class TestArrayFactoryShapeSpecEdges:
  def test_scalar_dim_to_shape_spec(self) -> None:
    from shapix._dimensions import Scalar

    specs = _to_shape_spec((Scalar,))
    assert specs == ()

  def test_mixed_type_shape_spec(self) -> None:
    from shapix._dimensions import Dimension
    from shapix._shape import ANONYMOUS_VARIADIC

    specs = _to_shape_spec((3, Dimension("N"), Ellipsis))
    assert specs[0] == FixedDim(3)
    assert specs[1] == NamedDim("N")
    assert specs[2] is ANONYMOUS_VARIADIC

  def test_memo_restored_on_failure(self) -> None:
    checker = _ArrayChecker(FLOAT32, (NamedDim("N"),))
    arr = np.ones((10,), dtype=np.float32)
    # Should fail because N=5 != 10, and memo should be restored
    from shapix._memo import pop_memo, push_memo

    push_memo_ref = push_memo()
    push_memo_ref.single["N"] = 5
    result = checker(arr)
    assert result is False
    pop_memo()


class TestArrayLikeCheckerEdges:
  def test_casting_no_uses_strict_match(self) -> None:
    """casting='no' should only accept exact dtype match."""
    from shapix._array_types import _ArrayLikeChecker

    checker = _ArrayLikeChecker(FLOAT32, (NamedDim("X"),), casting="no", name="F32Like")
    assert checker(np.ones(3, dtype=np.float32)) is True
    assert checker(np.ones(3, dtype=np.float64)) is False

  def test_wildcard_dtype_accepts_anything(self) -> None:
    """SHAPED's wildcard '*' in allowed should accept any dtype."""
    from shapix._array_types import _ArrayLikeChecker
    from shapix._dtypes import SHAPED

    checker = _ArrayLikeChecker(
      SHAPED, (NamedDim("X"),), casting="same_kind", name="ShapedLike"
    )
    assert checker(np.ones(3, dtype=np.float32)) is True
    assert checker(np.ones(3, dtype=np.int64)) is True
    assert checker(np.ones(3, dtype=np.bool_)) is True

  def test_asarray_failure_returns_false(self) -> None:
    """Objects that can't be converted to array should return False."""
    from shapix._array_types import _ArrayLikeChecker

    checker = _ArrayLikeChecker(
      FLOAT32, (NamedDim("X"),), casting="same_kind", name="F32Like"
    )
    # object() has no shape/dtype → takes slow path → np.asarray(object()) succeeds
    # but a custom __array__ that raises should trigger the except branch

    class Unconvertible:
      def __array__(self, *_a: object, **_kw: object) -> None:  # noqa: PLW3201
        msg = "nope"
        raise TypeError(msg)

    assert checker(Unconvertible()) is False

  def test_asarray_slow_path_bad_dtype(self) -> None:
    """Slow-path object whose dtype string is empty should fail."""
    from shapix._array_types import _ArrayLikeChecker

    # A plain list goes through slow path (no .shape/.dtype), converts to array
    # but complex128 can't cast to float32 under same_kind
    assert (
      _ArrayLikeChecker(
        FLOAT32, (NamedDim("X"),), casting="same_kind", name="F32Like"
      )([(1 + 2j)])
      is False
    )

  def test_arraylike_memo_restore_on_shape_failure(self) -> None:
    """ArrayLikeChecker should restore memo on shape mismatch."""
    from shapix._array_types import _ArrayLikeChecker
    from shapix._memo import pop_memo, push_memo

    checker = _ArrayLikeChecker(
      FLOAT32, (NamedDim("N"),), casting="same_kind", name="F32Like"
    )
    memo = push_memo()
    memo.single["N"] = 5

    # Should fail because N=5 but array has shape (10,)
    result = checker(np.ones(10, dtype=np.float32))
    assert result is False
    assert memo.single["N"] == 5  # memo restored
    pop_memo()

  def test_arraylike_dtype_no_source_returns_false(self) -> None:
    """Object with dtype but no extractable string should fail."""
    from shapix._array_types import _ArrayLikeChecker

    class WeirdDtype:
      pass

    class WeirdObj:
      dtype = WeirdDtype()
      shape = (3,)

    checker = _ArrayLikeChecker(
      FLOAT32, (NamedDim("X"),), casting="same_kind", name="F32Like"
    )
    assert checker(WeirdObj()) is False

  def test_arraylike_can_cast_type_error(self) -> None:
    """np.can_cast TypeError should be caught gracefully."""
    from shapix._array_types import _ArrayLikeChecker
    from shapix._dtypes import DtypeSpec

    # Create a spec with a bogus target dtype name
    bogus = DtypeSpec("Bogus", frozenset({"not_a_real_dtype"}))
    checker = _ArrayLikeChecker(
      bogus, (NamedDim("X"),), casting="same_kind", name="BogusLike"
    )
    assert checker(np.ones(3, dtype=np.float32)) is False

  def test_arraylike_repr(self) -> None:
    from shapix._array_types import _ArrayLikeChecker

    checker = _ArrayLikeChecker(
      FLOAT32, (NamedDim("N"), NamedDim("C")), casting="same_kind", name="F32Like"
    )
    assert repr(checker) == "F32Like[N, C]"

  def test_arraylike_factory_single_dim_subscript(self) -> None:
    """F32Like[N] (single dim, not tuple) should work."""
    from shapix._array_types import make_array_like_type

    factory = make_array_like_type(FLOAT32, name="F32Like")
    hint = factory[N]
    assert hasattr(hint, "__metadata__")

  def test_arraylike_factory_repr(self) -> None:
    from shapix._array_types import make_array_like_type

    factory = make_array_like_type(FLOAT32, name="F32Like")
    assert repr(factory) == "F32Like"

  def test_arraylike_fail_obj_replay(self) -> None:
    """Second call with same failing obj should replay failure."""
    from shapix._array_types import _ArrayLikeChecker

    checker = _ArrayLikeChecker(FLOAT32, (NamedDim("N"),), casting="no", name="F32Like")
    bad_obj = np.ones(3, dtype=np.int64)
    assert checker(bad_obj) is False  # first call fails (casting='no'), sets _fail_obj
    assert checker(bad_obj) is False  # replay

  def test_struct_checker_fail_obj_replay(self) -> None:
    """StructChecker should replay failure for same object."""
    checker = _ArrayChecker(FLOAT32, (NamedDim("N"),))
    bad = np.ones(3, dtype=np.int64)
    assert checker(bad) is False  # dtype mismatch, sets _fail_obj
    assert checker(bad) is False  # replay


class TestInputValidation:
  def test_invalid_casting_in_make_array_like_type(self) -> None:
    from shapix._array_types import make_array_like_type

    with pytest.raises(ValueError, match="Invalid casting"):
      make_array_like_type(FLOAT32, casting="bogus")

  def test_invalid_casting_in_make_scalar_like_type(self) -> None:
    from shapix.numpy import make_scalar_like_type

    with pytest.raises(ValueError, match="Invalid casting"):
      make_scalar_like_type(np.float32, casting="bogus")

  def test_invalid_byteorder_in_dtype_spec(self) -> None:
    from shapix._dtypes import DtypeSpec

    with pytest.raises(ValueError, match="Invalid byteorder"):
      DtypeSpec("Bad", frozenset({"float32"}), byteorder="wrong")

  def test_valid_castings_accepted(self) -> None:
    from shapix._array_types import make_array_like_type

    for casting in ("no", "equiv", "safe", "same_kind", "unsafe"):
      factory = make_array_like_type(FLOAT32, casting=casting)
      assert factory is not None

  def test_memo_snapshot_restore(self) -> None:
    memo = ShapeMemo(single={"N": 5}, variadic={"B": (False, (2, 3))})
    memo.structures["T"] = "spec"
    snap = memo.snapshot()

    memo.single["C"] = 10
    memo.variadic["X"] = (True, (1,))
    memo.structures["S"] = "other"

    memo.restore(snap)
    assert memo.single == {"N": 5}
    assert memo.variadic == {"B": (False, (2, 3))}
    assert memo.structures == {"T": "spec"}


class TestScalarLikeFactory:
  def test_make_scalar_like_type_exception_path(self) -> None:
    """Objects where np.asarray or np.can_cast raises should return False."""
    from shapix.numpy import make_scalar_like_type

    T = make_scalar_like_type(np.float32, casting="same_kind")

    class Unconvertible:
      def __array__(self, *_a: object, **_kw: object) -> None:  # noqa: PLW3201
        msg = "nope"
        raise TypeError(msg)

    from beartype.door import is_bearable

    assert not is_bearable(Unconvertible(), T)

  def test_make_scalar_like_type_exact_match(self) -> None:
    from beartype.door import is_bearable

    from shapix.numpy import make_scalar_like_type

    T = make_scalar_like_type(np.float32, casting="no")
    assert is_bearable(np.float32(1.0), T)
    assert not is_bearable(1.0, T)  # Python float != np.float32 under "no"


class TestVersionExport:
  def test_shapix_has_version(self) -> None:
    import shapix

    assert hasattr(shapix, "__version__")
    assert isinstance(shapix.__version__, str)
    assert len(shapix.__version__) > 0

  def test_root_import_does_not_require_numpy(self) -> None:
    script = """
import builtins
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path('.').resolve() / 'src'))
real_import = builtins.__import__

def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'numpy' or name.startswith('numpy.'):
        raise ModuleNotFoundError("No module named 'numpy'")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = fake_import
try:
    import shapix
    print(hasattr(shapix, 'Value'), hasattr(shapix, 'make_scalar_like_type'))
finally:
    builtins.__import__ = real_import
"""
    result = subprocess.run(
      [sys.executable, "-c", script],
      capture_output=True,
      text=True,
      cwd=Path(__file__).resolve().parents[1],
      check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "True False"


class TestVersionFallback:
  def test_version_fallback_when_metadata_missing(self) -> None:
    """__version__ should be '0+unknown' when package metadata is unavailable."""
    script = """
import builtins
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path('.').resolve() / 'src'))
real_import = builtins.__import__

def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == 'numpy' or name.startswith('numpy.'):
        raise ModuleNotFoundError("No module named 'numpy'")
    return real_import(name, globals, locals, fromlist, level)

builtins.__import__ = fake_import

import importlib.metadata
_real_version = importlib.metadata.version
def _fake_version(name):
    if name == 'shapix':
        raise importlib.metadata.PackageNotFoundError(name)
    return _real_version(name)

importlib.metadata.version = _fake_version

try:
    import shapix
    print(shapix.__version__)
finally:
    builtins.__import__ = real_import
    importlib.metadata.version = _real_version
"""
    result = subprocess.run(
      [sys.executable, "-c", script],
      capture_output=True,
      text=True,
      cwd=Path(__file__).resolve().parents[1],
      check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0+unknown"


class TestOptionalBackendImports:
  def test_cupy_import_error_is_clear_when_backend_missing(self) -> None:
    script = """
import importlib
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path('.').resolve() / 'src'))
real_import_module = importlib.import_module

def fake_import_module(name, package=None):
    if name == 'cupy' or name.startswith('cupy.'):
        raise ModuleNotFoundError("No module named 'cupy'")
    return real_import_module(name, package)

importlib.import_module = fake_import_module
try:
    import shapix.cupy
except ModuleNotFoundError as exc:
    print(exc)
finally:
    importlib.import_module = real_import_module
"""
    result = subprocess.run(
      [sys.executable, "-c", script],
      capture_output=True,
      text=True,
      cwd=Path(__file__).resolve().parents[1],
      check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "shapix.cupy requires 'cupy' at runtime." in result.stdout


class TestNewDimensionSymbols:
  def test_d_and_k_available(self) -> None:
    from shapix import D, K

    assert str(D) == "D"
    assert str(K) == "K"

  def test_d_and_k_in_annotations(self) -> None:
    from beartype import beartype

    from shapix import D, K
    from shapix.numpy import F32  # noqa: F401

    @beartype
    def f(x: F32[D, K]) -> F32[D, K]:  # type: ignore[valid-type]
      return x

    arr = np.ones((8, 4), dtype=np.float32)
    result = f(arr)
    assert result.shape == (8, 4)


class TestTrustedTypesParameter:
  """Test backend-scoped trusted_types on _ArrayLikeChecker."""

  def test_trusted_types_none_uses_global(self) -> None:
    """trusted_types=None falls back to global _is_trusted_array."""
    from shapix._array_types import _ArrayLikeChecker

    checker = _ArrayLikeChecker(
      FLOAT32, (NamedDim("X"),), casting="same_kind", name="F32Like"
    )
    assert checker._trusted_types is None
    assert checker(np.ones(3, dtype=np.float32)) is True

  def test_trusted_types_scoped(self) -> None:
    """trusted_types restricts fast path to specified types."""
    from shapix._array_types import _ArrayLikeChecker

    # Only trust np.ndarray
    checker = _ArrayLikeChecker(
      FLOAT32,
      (NamedDim("X"),),
      casting="same_kind",
      name="F32Like",
      trusted_types=(np.ndarray,),
    )
    assert checker._trusted_types == (np.ndarray,)
    # ndarray goes through fast path
    assert checker(np.ones(3, dtype=np.float32)) is True

  def test_factory_threads_trusted_types(self) -> None:
    """make_array_like_type passes trusted_types to factory."""
    from shapix._array_types import make_array_like_type

    factory = make_array_like_type(FLOAT32, name="F32Like", trusted_types=(np.ndarray,))
    assert factory._trusted_types == (np.ndarray,)


class TestTrustedArrayCache:
  def test_ndarray_is_trusted(self) -> None:
    from shapix._array_types import _is_trusted_array

    assert _is_trusted_array(np.ones(3))

  def test_plain_object_not_trusted(self) -> None:
    from shapix._array_types import _is_trusted_array

    assert not _is_trusted_array(object())

  def test_spoofed_not_trusted(self) -> None:
    from shapix._array_types import _is_trusted_array

    class Fake:
      shape = (3,)
      dtype = np.dtype(np.float32)

    assert not _is_trusted_array(Fake())


class TestClawWrapper:
  def test_shapix_this_package_delegates_to_beartype(
    self, monkeypatch: pytest.MonkeyPatch
  ) -> None:
    captured: dict[str, object] = {}

    def _fake_beartype_this_package(*, conf: object) -> None:
      captured["conf"] = conf

    import shapix.claw as claw_mod

    monkeypatch.setattr(claw_mod, "_beartype_this_package", _fake_beartype_this_package)
    conf = BeartypeConf()
    shapix_this_package(conf=conf)

    assert captured["conf"] is conf
