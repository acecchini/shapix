"""Targeted edge-path tests for low-frequency runtime branches."""

from __future__ import annotations

import numpy as np
import pytest

import shapix._memo as memo_mod
from beartype import BeartypeConf
from shapix import N
from shapix._array_types import _StructChecker, _to_shape_spec, make_array_type
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

    checker = _StructChecker(FLOAT32, (NamedDim("N"),))
    assert checker(DtypeOnly()) is False

  def test_array_factory_repr(self) -> None:
    factory = make_array_type(np.ndarray, FLOAT32)
    assert repr(factory) == "Float32Array"

  def test_to_shape_spec_non_dimension_falls_back_to_named(self) -> None:
    class Token:
      def __str__(self) -> str:
        return "Token"

    spec = _to_shape_spec((Token(),))
    assert spec == (NamedDim("Token"),)


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
  def test_get_memo_falls_back_if_frame_lookup_fails(self, monkeypatch) -> None:
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
    hint = tree[(F32[N],)]
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
    checker = _StructChecker(FLOAT32, (NamedDim("N"),))
    arr = np.ones((10,), dtype=np.float32)
    # Should fail because N=5 != 10, and memo should be restored
    from shapix._memo import push_memo, pop_memo

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
        raise TypeError("nope")

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
    from shapix._memo import push_memo, pop_memo

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
    checker = _StructChecker(FLOAT32, (NamedDim("N"),))
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


class TestClawWrapper:
  def test_shapix_this_package_delegates_to_beartype(self, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_beartype_this_package(*, conf: object) -> None:
      captured["conf"] = conf

    import shapix.claw as claw_mod

    monkeypatch.setattr(claw_mod, "_beartype_this_package", _fake_beartype_this_package)
    conf = BeartypeConf()
    shapix_this_package(conf=conf)

    assert captured["conf"] is conf
