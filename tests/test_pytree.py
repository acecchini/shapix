"""Comprehensive runtime tests for PyTree type annotations.

Covers dict/list/tuple trees, nested structures, cross-leaf dimension
consistency, structure binding, cross-argument matching, edge cases,
and decorator integration.
"""

from collections import namedtuple

import numpy as np
import pytest

optree = pytest.importorskip("optree")

from beartype import beartype
from beartype.door import is_bearable
from beartype.roar import BeartypeCallHintParamViolation

import shapix
from shapix import C, H, N, PyTree, S, T, W, __
from shapix.numpy import F32, F64, I64, Shaped


# =====================================================================
# Basic leaf type checking
# =====================================================================


class TestBasicPyTree:
  """PyTree[LeafType] validates all leaves match the type."""

  def test_dict_of_arrays(self) -> None:
    @beartype
    def f(x: PyTree[F32[N, C]]) -> PyTree[F32[N, C]]:
      return x

    data = {
      "a": np.ones((3, 4), dtype=np.float32),
      "b": np.ones((3, 4), dtype=np.float32),
    }
    result = f(data)
    assert result["a"].shape == (3, 4)

  def test_list_of_arrays(self) -> None:
    @beartype
    def f(x: PyTree[F32[N]]) -> PyTree[F32[N]]:
      return x

    data = [np.ones(5, dtype=np.float32), np.ones(5, dtype=np.float32)]
    result = f(data)
    assert len(result) == 2

  def test_tuple_of_arrays(self) -> None:
    @beartype
    def f(x: PyTree[F32[N]]) -> PyTree[F32[N]]:
      return x

    data = (np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32))
    result = f(data)
    assert len(result) == 2

  def test_nested_dict_list(self) -> None:
    @beartype
    def f(x: PyTree[F32[N, C]]) -> PyTree[F32[N, C]]:
      return x

    data = {
      "params": [np.ones((2, 3), dtype=np.float32), np.ones((2, 3), dtype=np.float32)],
      "state": {"count": np.ones((2, 3), dtype=np.float32)},
    }
    f(data)

  def test_single_leaf(self) -> None:
    """A bare array is a trivial pytree with one leaf."""

    @beartype
    def f(x: PyTree[F32[N]]) -> PyTree[F32[N]]:
      return x

    arr = np.ones(5, dtype=np.float32)
    result = f(arr)
    assert result.shape == (5,)


# =====================================================================
# Cross-leaf dimension consistency
# =====================================================================


class TestCrossLeafConsistency:
  """All leaves in a PyTree share the same dimension memo."""

  def test_consistent_shapes(self) -> None:
    @shapix.check
    @beartype
    def f(x: PyTree[F32[N, C]]) -> PyTree[F32[N, C]]:
      return x

    f({"a": np.ones((3, 4), dtype=np.float32), "b": np.ones((3, 4), dtype=np.float32)})

  def test_inconsistent_N_rejected(self) -> None:
    @shapix.check
    @beartype
    def f(x: PyTree[F32[N, C]]) -> PyTree[F32[N, C]]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f({
        "a": np.ones((3, 4), dtype=np.float32),
        "b": np.ones((5, 4), dtype=np.float32),  # N=5 vs N=3
      })

  def test_anonymous_dims_no_consistency(self) -> None:
    """Anonymous dims (__) should not enforce cross-leaf consistency."""

    @shapix.check
    @beartype
    def f(x: PyTree[F32[__, C]]) -> PyTree[F32[__, C]]:
      return x

    f({
      "a": np.ones((3, 4), dtype=np.float32),
      "b": np.ones((7, 4), dtype=np.float32),  # different first dim, OK
    })


# =====================================================================
# Structure binding
# =====================================================================


class TestStructureBinding:
  """PyTree[LeafType, T] enforces tree structure consistency."""

  def test_same_structure_passes(self) -> None:
    @shapix.check
    @beartype
    def f(x: PyTree[F32[N, C], T], y: PyTree[F32[N, C], T]) -> PyTree[F32[N, C]]:
      return x

    tree = {
      "a": np.ones((3, 4), dtype=np.float32),
      "b": np.ones((3, 4), dtype=np.float32),
    }
    f(tree, tree)

  def test_different_structure_rejected(self) -> None:
    @shapix.check
    @beartype
    def f(x: PyTree[F32[N], T], y: PyTree[F32[N], T]) -> PyTree[F32[N]]:
      return x

    x = {"a": np.ones(3, dtype=np.float32), "b": np.ones(3, dtype=np.float32)}
    y = [np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)]
    with pytest.raises(BeartypeCallHintParamViolation):
      f(x, y)

  def test_different_keys_rejected(self) -> None:
    @shapix.check
    @beartype
    def f(x: PyTree[F32[N], T], y: PyTree[F32[N], T]) -> PyTree[F32[N]]:
      return x

    x = {"a": np.ones(3, dtype=np.float32)}
    y = {"b": np.ones(3, dtype=np.float32)}
    with pytest.raises(BeartypeCallHintParamViolation):
      f(x, y)

  def test_independent_structure_names(self) -> None:
    """Different names bind independently."""

    @shapix.check
    @beartype
    def f(x: PyTree[F32[N], S], y: PyTree[F32[N], T]) -> PyTree[F32[N]]:
      return x

    x = {"a": np.ones(3, dtype=np.float32)}
    y = [np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)]
    f(x, y)  # OK — S and T are independent

  def test_list_structure_match(self) -> None:
    @shapix.check
    @beartype
    def f(x: PyTree[F32[N], T], y: PyTree[F32[N], T]) -> PyTree[F32[N]]:
      return x

    x = [np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)]
    y = [np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)]
    f(x, y)

  def test_nested_structure_match(self) -> None:
    @shapix.check
    @beartype
    def f(x: PyTree[F32[N], T], y: PyTree[F32[N], T]) -> PyTree[F32[N]]:
      return x

    tree = {
      "params": [np.ones(3, dtype=np.float32)],
      "state": np.ones(3, dtype=np.float32),
    }
    f(tree, tree)


# =====================================================================
# Cross-argument consistency
# =====================================================================


class TestCrossArgConsistency:
  """Dimension bindings are shared between PyTree and plain array args."""

  def test_pytree_and_plain_array(self) -> None:
    @shapix.check
    @beartype
    def f(params: PyTree[F32[N, C]], bias: F32[C]) -> PyTree[F32[N, C]]:
      return params

    params = {"w": np.ones((3, 4), dtype=np.float32)}
    bias = np.ones(4, dtype=np.float32)
    f(params, bias)

  def test_pytree_and_plain_array_mismatch(self) -> None:
    @shapix.check
    @beartype
    def f(params: PyTree[F32[N, C]], bias: F32[C]) -> PyTree[F32[N, C]]:
      return params

    params = {"w": np.ones((3, 4), dtype=np.float32)}
    bias = np.ones(5, dtype=np.float32)  # C=5 vs C=4
    with pytest.raises(BeartypeCallHintParamViolation):
      f(params, bias)

  def test_two_pytree_args(self) -> None:
    @shapix.check
    @beartype
    def f(x: PyTree[F32[N, C]], y: PyTree[F32[N, C]]) -> PyTree[F32[N, C]]:
      return x

    x = {"a": np.ones((3, 4), dtype=np.float32)}
    y = {"b": np.ones((3, 4), dtype=np.float32)}
    f(x, y)


# =====================================================================
# Dtype checking
# =====================================================================


class TestDtypeChecking:
  def test_wrong_dtype_rejected(self) -> None:
    @beartype
    def f(x: PyTree[F32[N]]) -> PyTree[F32[N]]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f({"a": np.ones(3, dtype=np.float64)})  # F64, not F32

  def test_multiple_dtypes_in_shaped(self) -> None:
    """Shaped accepts any numeric dtype."""

    @beartype
    def f(x: PyTree[Shaped[N]]) -> PyTree[Shaped[N]]:
      return x

    f({"ints": np.ones(5, dtype=np.int32), "floats": np.ones(5, dtype=np.float64)})

  def test_i64_pytree(self) -> None:
    @beartype
    def f(x: PyTree[I64[N]]) -> PyTree[I64[N]]:
      return x

    f([np.arange(5, dtype=np.int64)])


# =====================================================================
# Shape mismatch
# =====================================================================


class TestShapeMismatch:
  def test_wrong_rank_rejected(self) -> None:
    @beartype
    def f(x: PyTree[F32[N, C]]) -> PyTree[F32[N, C]]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f({"a": np.ones(3, dtype=np.float32)})  # 1D, not 2D


# =====================================================================
# Edge cases
# =====================================================================


class TestEdgeCases:
  def test_empty_dict(self) -> None:
    """Empty tree has no leaves to check — should pass."""

    @beartype
    def f(x: PyTree[F32[N]]) -> PyTree[F32[N]]:
      return x

    f({})

  def test_empty_list(self) -> None:
    @beartype
    def f(x: PyTree[F32[N]]) -> PyTree[F32[N]]:
      return x

    f([])

  def test_namedtuple_tree(self) -> None:
    Point = namedtuple("Point", ["x", "y"])

    @beartype
    def f(p: PyTree[F32[N]]) -> PyTree[F32[N]]:
      return p

    p = Point(x=np.ones(3, dtype=np.float32), y=np.ones(3, dtype=np.float32))
    result = f(p)
    assert len(optree.tree_leaves(result)) == 2

  def test_deeply_nested(self) -> None:
    @beartype
    def f(x: PyTree[F32[N]]) -> PyTree[F32[N]]:
      return x

    deep = {"a": {"b": {"c": [np.ones(4, dtype=np.float32)]}}}
    f(deep)

  def test_4d_leaves(self) -> None:
    @beartype
    def f(x: PyTree[F32[N, C, H, W]]) -> PyTree[F32[N, C, H, W]]:
      return x

    data = {
      "images": np.ones((2, 3, 28, 28), dtype=np.float32),
      "masks": np.ones((2, 3, 28, 28), dtype=np.float32),
    }
    f(data)


# =====================================================================
# Decorator integration
# =====================================================================


class TestDecoratorIntegration:
  def test_check_context_with_pytree(self) -> None:
    with shapix.check_context():
      tree = {"a": np.ones((3, 4), dtype=np.float32)}
      assert is_bearable(tree, PyTree[F32[N, C]])  # pyright: ignore[reportArgumentType]

  def test_check_context_rejects(self) -> None:
    with shapix.check_context():
      tree = {"a": np.ones(3, dtype=np.float64)}
      assert not is_bearable(tree, PyTree[F32[N]])  # pyright: ignore[reportArgumentType]

  def test_shapix_check_with_conf(self) -> None:
    from beartype import BeartypeConf

    @shapix.check(conf=BeartypeConf())
    def f(x: PyTree[F32[N, C]]) -> PyTree[F32[N, C]]:
      return x

    data = {"w": np.ones((3, 4), dtype=np.float32)}
    f(data)

  def test_sequential_calls_independent(self) -> None:
    @shapix.check
    @beartype
    def f(x: PyTree[F32[N]]) -> PyTree[F32[N]]:
      return x

    f({"a": np.ones(3, dtype=np.float32)})  # N=3
    f({"a": np.ones(7, dtype=np.float32)})  # N=7, independent


# =====================================================================
# Return type checking
# =====================================================================


class TestReturnChecking:
  def test_return_type_checked(self) -> None:
    @beartype
    def f(x: PyTree[F32[N]]) -> PyTree[F64[N]]:
      return x  # returns F32, not F64

    with pytest.raises(Exception):  # noqa: B017
      f({"a": np.ones(3, dtype=np.float32)})


# =====================================================================
# Prefix / suffix wildcards
# =====================================================================


class TestPrefixSuffix:
  def test_prefix_matching(self) -> None:
    """T[...] — top-level structure matches T, subtrees are arbitrary."""

    @shapix.check
    @beartype
    def f(x: PyTree[F32[N], T[...]], y: PyTree[F32[N], T[...]]) -> PyTree[F32[N]]:
      return x

    # Both have dict at top level with keys "a", "b"
    x = {"a": np.ones(3, dtype=np.float32), "b": np.ones(3, dtype=np.float32)}
    y = {"a": np.ones(3, dtype=np.float32), "b": np.ones(3, dtype=np.float32)}
    f(x, y)

  def test_suffix_matching(self) -> None:
    """PyTree[leaf, ..., T] — full structure matches T."""

    @shapix.check
    @beartype
    def f(x: PyTree[F32[N], ..., T], y: PyTree[F32[N], ..., T]) -> PyTree[F32[N]]:
      return x

    x = {"a": np.ones(3, dtype=np.float32)}
    y = {"a": np.ones(3, dtype=np.float32)}
    f(x, y)

  def test_suffix_different_structure_rejected(self) -> None:
    @shapix.check
    @beartype
    def f(x: PyTree[F32[N], ..., T], y: PyTree[F32[N], ..., T]) -> PyTree[F32[N]]:
      return x

    x = {"a": np.ones(3, dtype=np.float32)}
    y = [np.ones(3, dtype=np.float32)]
    with pytest.raises(BeartypeCallHintParamViolation):
      f(x, y)
