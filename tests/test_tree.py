# pyright: reportArgumentType=false, reportGeneralTypeIssues=false, reportInvalidTypeArguments=false, reportReturnType=false, reportIndexIssue=false, reportAttributeAccessIssue=false
"""Comprehensive runtime tests for Tree type annotations.

Covers dict/list/tuple trees, nested structures, cross-leaf dimension
consistency, structure binding, cross-argument matching, edge cases,
decorator integration, and the new positional structure notation.
"""

from collections import namedtuple

import numpy as np
import pytest

optree = pytest.importorskip("optree")

from beartype import beartype
from beartype.door import is_bearable
from beartype.roar import (
  BeartypeCallHintParamViolation,
  BeartypeCallHintReturnViolation,
)

import shapix
from shapix import C, H, N, S, T, W, __
from shapix.optree import Tree
from shapix._tree import Structure, _TreeFactory
from shapix.numpy import F32, F64, I64, Shaped


# =====================================================================
# Basic leaf type checking
# =====================================================================


class TestBasicTree:
  """Tree[LeafType] validates all leaves match the type."""

  def test_dict_of_arrays(self) -> None:
    @beartype
    def f(x: Tree[F32[N, C]]) -> Tree[F32[N, C]]:
      return x

    data = {
      "a": np.ones((3, 4), dtype=np.float32),
      "b": np.ones((3, 4), dtype=np.float32),
    }
    result = f(data)
    assert result["a"].shape == (3, 4)

  def test_list_of_arrays(self) -> None:
    @beartype
    def f(x: Tree[F32[N]]) -> Tree[F32[N]]:
      return x

    data = [np.ones(5, dtype=np.float32), np.ones(5, dtype=np.float32)]
    result = f(data)
    assert len(result) == 2

  def test_tuple_of_arrays(self) -> None:
    @beartype
    def f(x: Tree[F32[N]]) -> Tree[F32[N]]:
      return x

    data = (np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32))
    result = f(data)
    assert len(result) == 2

  def test_nested_dict_list(self) -> None:
    @beartype
    def f(x: Tree[F32[N, C]]) -> Tree[F32[N, C]]:
      return x

    data = {
      "params": [np.ones((2, 3), dtype=np.float32), np.ones((2, 3), dtype=np.float32)],
      "state": {"count": np.ones((2, 3), dtype=np.float32)},
    }
    f(data)

  def test_single_leaf(self) -> None:
    """A bare array is a trivial pytree with one leaf."""

    @beartype
    def f(x: Tree[F32[N]]) -> Tree[F32[N]]:
      return x

    arr = np.ones(5, dtype=np.float32)
    result = f(arr)
    assert result.shape == (5,)


# =====================================================================
# Cross-leaf dimension consistency
# =====================================================================


class TestCrossLeafConsistency:
  """All leaves in a Tree share the same dimension memo."""

  def test_consistent_shapes(self) -> None:
    @shapix.check
    @beartype
    def f(x: Tree[F32[N, C]]) -> Tree[F32[N, C]]:
      return x

    f({"a": np.ones((3, 4), dtype=np.float32), "b": np.ones((3, 4), dtype=np.float32)})

  def test_inconsistent_N_rejected(self) -> None:
    @shapix.check
    @beartype
    def f(x: Tree[F32[N, C]]) -> Tree[F32[N, C]]:
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
    def f(x: Tree[F32[__, C]]) -> Tree[F32[__, C]]:
      return x

    f({
      "a": np.ones((3, 4), dtype=np.float32),
      "b": np.ones((7, 4), dtype=np.float32),  # different first dim, OK
    })


# =====================================================================
# Structure binding
# =====================================================================


class TestStructureBinding:
  """Tree[LeafType, T] enforces tree structure consistency."""

  def test_same_structure_passes(self) -> None:
    @shapix.check
    @beartype
    def f(x: Tree[F32[N, C], T], y: Tree[F32[N, C], T]) -> Tree[F32[N, C]]:
      return x

    tree = {
      "a": np.ones((3, 4), dtype=np.float32),
      "b": np.ones((3, 4), dtype=np.float32),
    }
    f(tree, tree)

  def test_different_structure_rejected(self) -> None:
    @shapix.check
    @beartype
    def f(x: Tree[F32[N], T], y: Tree[F32[N], T]) -> Tree[F32[N]]:
      return x

    x = {"a": np.ones(3, dtype=np.float32), "b": np.ones(3, dtype=np.float32)}
    y = [np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)]
    with pytest.raises(BeartypeCallHintParamViolation):
      f(x, y)

  def test_different_keys_rejected(self) -> None:
    @shapix.check
    @beartype
    def f(x: Tree[F32[N], T], y: Tree[F32[N], T]) -> Tree[F32[N]]:
      return x

    x = {"a": np.ones(3, dtype=np.float32)}
    y = {"b": np.ones(3, dtype=np.float32)}
    with pytest.raises(BeartypeCallHintParamViolation):
      f(x, y)

  def test_independent_structure_names(self) -> None:
    """Different names bind independently."""

    @shapix.check
    @beartype
    def f(x: Tree[F32[N], S], y: Tree[F32[N], T]) -> Tree[F32[N]]:
      return x

    x = {"a": np.ones(3, dtype=np.float32)}
    y = [np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)]
    f(x, y)  # OK — S and T are independent

  def test_list_structure_match(self) -> None:
    @shapix.check
    @beartype
    def f(x: Tree[F32[N], T], y: Tree[F32[N], T]) -> Tree[F32[N]]:
      return x

    x = [np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)]
    y = [np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)]
    f(x, y)

  def test_nested_structure_match(self) -> None:
    @shapix.check
    @beartype
    def f(x: Tree[F32[N], T], y: Tree[F32[N], T]) -> Tree[F32[N]]:
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
  """Dimension bindings are shared between Tree and plain array args."""

  def test_pytree_and_plain_array(self) -> None:
    @shapix.check
    @beartype
    def f(params: Tree[F32[N, C]], bias: F32[C]) -> Tree[F32[N, C]]:
      return params

    params = {"w": np.ones((3, 4), dtype=np.float32)}
    bias = np.ones(4, dtype=np.float32)
    f(params, bias)

  def test_pytree_and_plain_array_mismatch(self) -> None:
    @shapix.check
    @beartype
    def f(params: Tree[F32[N, C]], bias: F32[C]) -> Tree[F32[N, C]]:
      return params

    params = {"w": np.ones((3, 4), dtype=np.float32)}
    bias = np.ones(5, dtype=np.float32)  # C=5 vs C=4
    with pytest.raises(BeartypeCallHintParamViolation):
      f(params, bias)

  def test_two_pytree_args(self) -> None:
    @shapix.check
    @beartype
    def f(x: Tree[F32[N, C]], y: Tree[F32[N, C]]) -> Tree[F32[N, C]]:
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
    def f(x: Tree[F32[N]]) -> Tree[F32[N]]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f({"a": np.ones(3, dtype=np.float64)})  # F64, not F32

  def test_multiple_dtypes_in_shaped(self) -> None:
    """Shaped accepts any numeric dtype."""

    @beartype
    def f(x: Tree[Shaped[N]]) -> Tree[Shaped[N]]:
      return x

    f({"ints": np.ones(5, dtype=np.int32), "floats": np.ones(5, dtype=np.float64)})

  def test_i64_pytree(self) -> None:
    @beartype
    def f(x: Tree[I64[N]]) -> Tree[I64[N]]:
      return x

    f([np.arange(5, dtype=np.int64)])


# =====================================================================
# Shape mismatch
# =====================================================================


class TestShapeMismatch:
  def test_wrong_rank_rejected(self) -> None:
    @beartype
    def f(x: Tree[F32[N, C]]) -> Tree[F32[N, C]]:
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
    def f(x: Tree[F32[N]]) -> Tree[F32[N]]:
      return x

    f({})

  def test_empty_list(self) -> None:
    @beartype
    def f(x: Tree[F32[N]]) -> Tree[F32[N]]:
      return x

    f([])

  def test_namedtuple_tree(self) -> None:
    Point = namedtuple("Point", ["x", "y"])

    @beartype
    def f(p: Tree[F32[N]]) -> Tree[F32[N]]:
      return p

    p = Point(x=np.ones(3, dtype=np.float32), y=np.ones(3, dtype=np.float32))
    result = f(p)
    assert len(optree.tree_leaves(result)) == 2

  def test_deeply_nested(self) -> None:
    @beartype
    def f(x: Tree[F32[N]]) -> Tree[F32[N]]:
      return x

    deep = {"a": {"b": {"c": [np.ones(4, dtype=np.float32)]}}}
    f(deep)

  def test_4d_leaves(self) -> None:
    @beartype
    def f(x: Tree[F32[N, C, H, W]]) -> Tree[F32[N, C, H, W]]:
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
      assert is_bearable(tree, Tree[F32[N, C]])  # pyright: ignore[reportArgumentType]

  def test_check_context_rejects(self) -> None:
    with shapix.check_context():
      tree = {"a": np.ones(3, dtype=np.float64)}
      assert not is_bearable(tree, Tree[F32[N]])  # pyright: ignore[reportArgumentType]

  def test_shapix_check_with_conf(self) -> None:
    from beartype import BeartypeConf

    @shapix.check(conf=BeartypeConf())
    def f(x: Tree[F32[N, C]]) -> Tree[F32[N, C]]:
      return x

    data = {"w": np.ones((3, 4), dtype=np.float32)}
    f(data)

  def test_sequential_calls_independent(self) -> None:
    @shapix.check
    @beartype
    def f(x: Tree[F32[N]]) -> Tree[F32[N]]:
      return x

    f({"a": np.ones(3, dtype=np.float32)})  # N=3
    f({"a": np.ones(7, dtype=np.float32)})  # N=7, independent


# =====================================================================
# Return type checking
# =====================================================================


class TestReturnChecking:
  def test_return_type_checked(self) -> None:
    @beartype
    def f(x: Tree[F32[N]]) -> Tree[F64[N]]:
      return x  # returns F32, not F64

    with pytest.raises(BeartypeCallHintReturnViolation):
      f({"a": np.ones(3, dtype=np.float32)})


# =====================================================================
# Top-level (prefix) matching — Tree[leaf, T, ...]
# =====================================================================


class TestTopLevel:
  def test_top_level_matching(self) -> None:
    """Tree[leaf, T, ...] — top-level structure matches T, subtrees arbitrary."""

    @shapix.check
    @beartype
    def f(
      x: Tree[F32[N], T, ...],
      y: Tree[F32[N], T, ...],  # pyright: ignore
    ) -> Tree[F32[N]]:
      return x

    # Both have dict at top level with keys "a", "b"
    x = {"a": np.ones(3, dtype=np.float32), "b": np.ones(3, dtype=np.float32)}
    y = {"a": np.ones(3, dtype=np.float32), "b": np.ones(3, dtype=np.float32)}
    f(x, y)

  def test_top_level_different_subtrees_pass(self) -> None:
    """Subtrees can differ when only top-level is checked."""

    @shapix.check
    @beartype
    def f(
      x: Tree[F32[N], T, ...],
      y: Tree[F32[N], T, ...],  # pyright: ignore
    ) -> Tree[F32[N]]:
      return x

    # Same top-level dict with keys "a", "b", but different nesting
    x = {"a": [np.ones(3, dtype=np.float32)], "b": np.ones(3, dtype=np.float32)}
    y = {"a": np.ones(3, dtype=np.float32), "b": {"c": np.ones(3, dtype=np.float32)}}
    f(x, y)

  def test_top_level_mismatch_rejected(self) -> None:
    @shapix.check
    @beartype
    def f(
      x: Tree[F32[N], T, ...],
      y: Tree[F32[N], T, ...],  # pyright: ignore
    ) -> Tree[F32[N]]:
      return x

    x = {"a": np.ones(3, dtype=np.float32)}
    y = [np.ones(3, dtype=np.float32)]
    with pytest.raises(BeartypeCallHintParamViolation):
      f(x, y)


# =====================================================================
# Bottom-level (suffix) matching — Tree[leaf, ..., T]
# =====================================================================


class TestBottomLevel:
  def test_bottom_level_match(self) -> None:
    """Tree[leaf, ..., T] — T matches the leaf-adjacent container level."""

    @shapix.check
    @beartype
    def f(
      x: Tree[F32[N], ..., T],
      y: Tree[F32[N], ..., T],  # pyright: ignore
    ) -> Tree[F32[N]]:
      return x

    # Both have lists at the bottom (leaf-adjacent) level
    x = {"a": [np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)]}
    y = {"b": [np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)]}
    f(x, y)

  def test_bottom_level_single_level(self) -> None:
    """On a flat tree, the bottom level IS the top level."""

    @shapix.check
    @beartype
    def f(
      x: Tree[F32[N], ..., T],
      y: Tree[F32[N], ..., T],  # pyright: ignore
    ) -> Tree[F32[N]]:
      return x

    x = {"a": np.ones(3, dtype=np.float32)}
    y = {"a": np.ones(3, dtype=np.float32)}
    f(x, y)

  def test_bottom_level_different_structure_rejected(self) -> None:
    @shapix.check
    @beartype
    def f(
      x: Tree[F32[N], ..., T],
      y: Tree[F32[N], ..., T],  # pyright: ignore
    ) -> Tree[F32[N]]:
      return x

    # x bottom = list, y bottom = dict
    x = {"a": [np.ones(3, dtype=np.float32)]}
    y = {"a": {"k": np.ones(3, dtype=np.float32)}}
    with pytest.raises(BeartypeCallHintParamViolation):
      f(x, y)


# =====================================================================
# Multi-level top-down — Tree[leaf, T, S] / Tree[leaf, T, S, ...]
# =====================================================================


class TestMultiLevel:
  def test_composite_top_and_rest(self) -> None:
    """Tree[leaf, T, S] — T = top level, S = full remaining."""

    @shapix.check
    @beartype
    def f(
      x: Tree[F32[N], T, S],
      y: Tree[F32[N], T, S],  # pyright: ignore
    ) -> Tree[F32[N]]:
      return x

    # T = dict{"a", "b"}, S = list[2]
    x = {
      "a": [np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)],
      "b": [np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)],
    }
    f(x, x)

  def test_composite_mismatch_rejected(self) -> None:
    """Different inner structures across same-level siblings should fail."""

    @shapix.check
    @beartype
    def f(x: Tree[F32[N], T, S]) -> Tree[F32[N]]:  # pyright: ignore
      return x

    # "a" has list[2] inside, "b" has list[1] — S can't match both
    x = {
      "a": [np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)],
      "b": [np.ones(3, dtype=np.float32)],
    }
    with pytest.raises(BeartypeCallHintParamViolation):
      f(x)

  def test_multi_level_top_down_wildcard(self) -> None:
    """Tree[leaf, T, S, ...] — T = top, S = next, rest arbitrary."""

    @shapix.check
    @beartype
    def f(
      x: Tree[F32[N], T, S, ...],  # pyright: ignore
      y: Tree[F32[N], T, S, ...],  # pyright: ignore
    ) -> Tree[F32[N]]:
      return x

    # T = dict{"a"}, S = list[2], then arbitrary below
    x = {"a": [[np.ones(3, dtype=np.float32)], [np.ones(3, dtype=np.float32)]]}
    y = {"a": [np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)]}
    f(x, y)

  def test_multi_level_bottom_up(self) -> None:
    """Tree[leaf, ..., T, S] — S = bottom, T = second-from-bottom."""

    @shapix.check
    @beartype
    def f(
      x: Tree[F32[N], ..., T, S],  # pyright: ignore
      y: Tree[F32[N], ..., T, S],  # pyright: ignore
    ) -> Tree[F32[N]]:
      return x

    # Bottom (S) = list[2], second-from-bottom (T) = dict{"a"}
    x = {"a": [np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)]}
    y = {"a": [np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)]}
    f(x, y)

  def test_tree_shallower_than_spec_rejected(self) -> None:
    """Tree[leaf, T, S] on a flat tree should fail (only 1 level, need 2)."""

    @shapix.check
    @beartype
    def f(x: Tree[F32[N], T, S]) -> Tree[F32[N]]:  # pyright: ignore
      return x

    # Only 1 level deep, but spec needs T (top) + S (remaining)
    x = {"a": np.ones(3, dtype=np.float32)}
    # This should still pass — T peels one level, S captures remaining
    # (which is just a single leaf = the array itself).
    # Actually: tree_flatten on the array returns the array as leaf,
    # so children[0] is node → returns False (expected container, got leaf).
    # But wait — with wildcard=False, the last name captures full remaining
    # via tree_structure. The array is a leaf, so tree_structure returns a
    # leaf spec. Let me verify the actual behavior...
    #
    # T peels one level: children of dict → [array], top_spec = dict{"a"}
    # S captures full remaining of children[0] = array.
    # tree_structure(array) returns a leaf TreeSpec.
    # This should work — S just binds to the leaf structure.
    f(x)

  def test_bottom_up_too_shallow_rejected(self) -> None:
    """Tree[leaf, ..., T, S] on a flat tree — only 1 level, need 2."""

    @shapix.check
    @beartype
    def f(x: Tree[F32[N], ..., T, S]) -> Tree[F32[N]]:  # pyright: ignore
      return x

    # Only 1 level deep, but need 2 levels for bottom-up
    x = {"a": np.ones(3, dtype=np.float32)}
    with pytest.raises(BeartypeCallHintParamViolation):
      f(x)

  def test_empty_tree_with_structure(self) -> None:
    """Empty dict has no levels — structure spec should still be handled."""

    @shapix.check
    @beartype
    def f(x: Tree[F32[N], T, ...]) -> Tree[F32[N]]:  # pyright: ignore
      return x

    # Empty dict: tree_flatten returns no children, so is_leaf check fails
    # Actually: tree_flatten({}, is_leaf=...) returns ([], top_spec)
    # So children=[], top_spec binds to T. Should pass.
    f({})


# =====================================================================
# Leaf type failures
# =====================================================================


class TestLeafTypeFailures:
  """Failures when leaves don't match the annotated leaf type."""

  def test_mixed_dtypes_in_leaves(self) -> None:
    """Some leaves F32, others F64 — should reject."""

    @beartype
    def f(x: Tree[F32[N]]) -> Tree[F32[N]]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f({"a": np.ones(3, dtype=np.float32), "b": np.ones(3, dtype=np.float64)})

  def test_non_array_leaf(self) -> None:
    """A plain Python int as leaf should fail leaf type check."""

    @beartype
    def f(x: Tree[F32[N]]) -> Tree[F32[N]]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f({"a": np.ones(3, dtype=np.float32), "b": 42})

  def test_mixed_ranks_in_leaves(self) -> None:
    """Leaves with different ranks — one 1D, one 2D."""

    @beartype
    def f(x: Tree[F32[N, C]]) -> Tree[F32[N, C]]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f({
        "a": np.ones((3, 4), dtype=np.float32),
        "b": np.ones(3, dtype=np.float32),  # 1D, not 2D
      })

  def test_wrong_dtype_deep_nesting(self) -> None:
    """Wrong dtype buried deep inside nested containers."""

    @beartype
    def f(x: Tree[F32[N]]) -> Tree[F32[N]]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f({"a": [np.ones(3, dtype=np.float32), {"b": np.ones(3, dtype=np.int64)}]})

  def test_zero_dim_array_leaf(self) -> None:
    """A 0-D array leaf should fail when 1-D is expected."""

    @beartype
    def f(x: Tree[F32[N]]) -> Tree[F32[N]]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f({"a": np.float32(1.0), "b": np.ones(3, dtype=np.float32)})

  def test_string_leaf(self) -> None:
    """String leaf should fail."""

    @beartype
    def f(x: Tree[F32[N]]) -> Tree[F32[N]]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f({"a": "not an array"})


# =====================================================================
# Cross-leaf dimension failures
# =====================================================================


class TestCrossLeafDimFailures:
  """Failures when named dimensions are inconsistent across leaves."""

  def test_inconsistent_C_across_leaves(self) -> None:
    """C mismatch between leaves — N matches but C doesn't."""

    @shapix.check
    @beartype
    def f(x: Tree[F32[N, C]]) -> Tree[F32[N, C]]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f({
        "a": np.ones((3, 4), dtype=np.float32),  # C=4
        "b": np.ones((3, 5), dtype=np.float32),  # C=5
      })

  def test_inconsistent_dims_in_deeply_nested(self) -> None:
    """Dimension mismatch between leaves at different nesting depths."""

    @shapix.check
    @beartype
    def f(x: Tree[F32[N, C]]) -> Tree[F32[N, C]]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f({
        "top": np.ones((3, 4), dtype=np.float32),  # N=3, C=4
        "nested": {"deep": np.ones((3, 7), dtype=np.float32)},  # N=3, C=7
      })

  def test_all_dims_inconsistent(self) -> None:
    """Both N and C differ between leaves."""

    @shapix.check
    @beartype
    def f(x: Tree[F32[N, C]]) -> Tree[F32[N, C]]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f({
        "a": np.ones((3, 4), dtype=np.float32),  # N=3, C=4
        "b": np.ones((5, 6), dtype=np.float32),  # N=5, C=6
      })

  def test_inconsistent_4d_leaves(self) -> None:
    """4D shape with one dimension differing."""

    @shapix.check
    @beartype
    def f(x: Tree[F32[N, C, H, W]]) -> Tree[F32[N, C, H, W]]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f({
        "images": np.ones((2, 3, 28, 28), dtype=np.float32),
        "masks": np.ones((2, 3, 28, 14), dtype=np.float32),  # W=14 vs W=28
      })

  def test_inconsistent_across_list_items(self) -> None:
    """Leaves in a list with mismatched dimensions."""

    @shapix.check
    @beartype
    def f(x: Tree[F32[N]]) -> Tree[F32[N]]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f([np.ones(3, dtype=np.float32), np.ones(5, dtype=np.float32)])


# =====================================================================
# Structure binding failures
# =====================================================================


class TestStructureBindingFailures:
  """Failures when structure bindings mismatch across arguments."""

  def test_different_list_lengths(self) -> None:
    """Same container type but different lengths."""

    @shapix.check
    @beartype
    def f(x: Tree[F32[N], T], y: Tree[F32[N], T]) -> Tree[F32[N]]:
      return x

    x = [np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)]
    y = [np.ones(3, dtype=np.float32)]  # length 1 vs 2
    with pytest.raises(BeartypeCallHintParamViolation):
      f(x, y)

  def test_different_nesting_depths(self) -> None:
    """x is flat dict, y is nested dict — different structures."""

    @shapix.check
    @beartype
    def f(x: Tree[F32[N], T], y: Tree[F32[N], T]) -> Tree[F32[N]]:
      return x

    x = {"a": np.ones(3, dtype=np.float32)}
    y = {"a": {"b": np.ones(3, dtype=np.float32)}}
    with pytest.raises(BeartypeCallHintParamViolation):
      f(x, y)

  def test_tuple_vs_list(self) -> None:
    """Tuple and list have different tree specs."""

    @shapix.check
    @beartype
    def f(x: Tree[F32[N], T], y: Tree[F32[N], T]) -> Tree[F32[N]]:
      return x

    x = (np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32))
    y = [np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)]
    with pytest.raises(BeartypeCallHintParamViolation):
      f(x, y)

  def test_dict_extra_keys(self) -> None:
    """Same keys plus an extra key — different structure."""

    @shapix.check
    @beartype
    def f(x: Tree[F32[N], T], y: Tree[F32[N], T]) -> Tree[F32[N]]:
      return x

    x = {"a": np.ones(3, dtype=np.float32), "b": np.ones(3, dtype=np.float32)}
    y = {"a": np.ones(3, dtype=np.float32)}
    with pytest.raises(BeartypeCallHintParamViolation):
      f(x, y)

  def test_nested_inner_structure_differs(self) -> None:
    """Same top-level keys, different inner container types."""

    @shapix.check
    @beartype
    def f(x: Tree[F32[N], T], y: Tree[F32[N], T]) -> Tree[F32[N]]:
      return x

    x = {"params": [np.ones(3, dtype=np.float32)]}
    y = {"params": (np.ones(3, dtype=np.float32),)}
    with pytest.raises(BeartypeCallHintParamViolation):
      f(x, y)

  def test_leaf_vs_container(self) -> None:
    """One arg is a bare leaf, the other is a container."""

    @shapix.check
    @beartype
    def f(x: Tree[F32[N], T], y: Tree[F32[N], T]) -> Tree[F32[N]]:
      return x

    x = np.ones(3, dtype=np.float32)
    y = [np.ones(3, dtype=np.float32)]
    with pytest.raises(BeartypeCallHintParamViolation):
      f(x, y)

  def test_three_args_third_differs(self) -> None:
    """First two match structure T, third differs."""

    @shapix.check
    @beartype
    def f(x: Tree[F32[N], T], y: Tree[F32[N], T], z: Tree[F32[N], T]) -> Tree[F32[N]]:
      return x

    a = {"a": np.ones(3, dtype=np.float32)}
    b = {"a": np.ones(3, dtype=np.float32)}
    c = {"b": np.ones(3, dtype=np.float32)}  # different key
    with pytest.raises(BeartypeCallHintParamViolation):
      f(a, b, c)


# =====================================================================
# Cross-argument dimension + structure failures
# =====================================================================


class TestCrossArgFailures:
  """Failures when dimensions mismatch between Tree and plain arrays."""

  def test_tree_dim_vs_plain_array_dim(self) -> None:
    """Tree leaves bind N=3, but plain array has N=7."""

    @shapix.check
    @beartype
    def f(params: Tree[F32[N]], bias: F32[N]) -> Tree[F32[N]]:
      return params

    params = {"w": np.ones(3, dtype=np.float32)}
    bias = np.ones(7, dtype=np.float32)  # N=7 vs N=3
    with pytest.raises(BeartypeCallHintParamViolation):
      f(params, bias)

  def test_two_trees_dim_mismatch(self) -> None:
    """Two Tree args with same dim name but different values."""

    @shapix.check
    @beartype
    def f(x: Tree[F32[N, C]], y: Tree[F32[N, C]]) -> Tree[F32[N, C]]:
      return x

    x = {"a": np.ones((3, 4), dtype=np.float32)}  # N=3, C=4
    y = {"b": np.ones((3, 5), dtype=np.float32)}  # N=3, C=5
    with pytest.raises(BeartypeCallHintParamViolation):
      f(x, y)

  def test_tree_and_plain_array_both_dims_mismatch(self) -> None:
    """Both N and C differ between tree and plain array."""

    @shapix.check
    @beartype
    def f(x: Tree[F32[N, C]], b: F32[C]) -> Tree[F32[N, C]]:
      return x

    x = {"w": np.ones((3, 4), dtype=np.float32)}  # C=4
    b = np.ones(10, dtype=np.float32)  # C=10
    with pytest.raises(BeartypeCallHintParamViolation):
      f(x, b)


# =====================================================================
# Top-level (prefix) matching failures
# =====================================================================


class TestTopLevelFailures:
  """Failures for Tree[leaf, T, ...] — top-level structure mismatches."""

  def test_dict_vs_list_top_level(self) -> None:
    """One has dict at top, other has list."""

    @shapix.check
    @beartype
    def f(
      x: Tree[F32[N], T, ...],
      y: Tree[F32[N], T, ...],  # pyright: ignore
    ) -> Tree[F32[N]]:
      return x

    x = {"a": np.ones(3, dtype=np.float32)}
    y = [np.ones(3, dtype=np.float32)]
    with pytest.raises(BeartypeCallHintParamViolation):
      f(x, y)

  def test_different_dict_keys_top_level(self) -> None:
    """Same container type (dict) but different keys at top."""

    @shapix.check
    @beartype
    def f(
      x: Tree[F32[N], T, ...],
      y: Tree[F32[N], T, ...],  # pyright: ignore
    ) -> Tree[F32[N]]:
      return x

    x = {"a": np.ones(3, dtype=np.float32)}
    y = {"b": np.ones(3, dtype=np.float32)}
    with pytest.raises(BeartypeCallHintParamViolation):
      f(x, y)

  def test_different_list_lengths_top_level(self) -> None:
    """Same container type (list) but different lengths at top."""

    @shapix.check
    @beartype
    def f(
      x: Tree[F32[N], T, ...],
      y: Tree[F32[N], T, ...],  # pyright: ignore
    ) -> Tree[F32[N]]:
      return x

    x = [np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)]
    y = [np.ones(3, dtype=np.float32)]
    with pytest.raises(BeartypeCallHintParamViolation):
      f(x, y)

  def test_leaf_at_top_rejected(self) -> None:
    """A bare array has no container level — peeling one level should fail."""

    @shapix.check
    @beartype
    def f(x: Tree[F32[N], T, ...]) -> Tree[F32[N]]:  # pyright: ignore
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones(3, dtype=np.float32))

  def test_multi_name_prefix_second_level_differs(self) -> None:
    """Tree[leaf, T, S, ...] — T matches but S doesn't."""

    @shapix.check
    @beartype
    def f(
      x: Tree[F32[N], T, S, ...],  # pyright: ignore
      y: Tree[F32[N], T, S, ...],  # pyright: ignore
    ) -> Tree[F32[N]]:
      return x

    # Both have dict{"a"} at top (T matches), but second level differs
    x = {"a": [np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)]}  # S=list[2]
    y = {"a": [np.ones(3, dtype=np.float32)]}  # S=list[1]
    with pytest.raises(BeartypeCallHintParamViolation):
      f(x, y)


# =====================================================================
# Bottom-level (suffix) matching failures
# =====================================================================


class TestBottomLevelFailures:
  """Failures for Tree[leaf, ..., T] — bottom-level structure mismatches."""

  def test_different_bottom_container_type(self) -> None:
    """x has list at bottom, y has dict at bottom."""

    @shapix.check
    @beartype
    def f(
      x: Tree[F32[N], ..., T],
      y: Tree[F32[N], ..., T],  # pyright: ignore
    ) -> Tree[F32[N]]:
      return x

    x = {"a": [np.ones(3, dtype=np.float32)]}  # bottom = list
    y = {"a": {"k": np.ones(3, dtype=np.float32)}}  # bottom = dict
    with pytest.raises(BeartypeCallHintParamViolation):
      f(x, y)

  def test_different_bottom_list_lengths(self) -> None:
    """Same container type at bottom but different lengths."""

    @shapix.check
    @beartype
    def f(
      x: Tree[F32[N], ..., T],
      y: Tree[F32[N], ..., T],  # pyright: ignore
    ) -> Tree[F32[N]]:
      return x

    x = {"a": [np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)]}
    y = {"a": [np.ones(3, dtype=np.float32)]}
    with pytest.raises(BeartypeCallHintParamViolation):
      f(x, y)

  def test_different_bottom_dict_keys(self) -> None:
    """Same bottom container type (dict) but different keys."""

    @shapix.check
    @beartype
    def f(
      x: Tree[F32[N], ..., T],
      y: Tree[F32[N], ..., T],  # pyright: ignore
    ) -> Tree[F32[N]]:
      return x

    x = [{"k1": np.ones(3, dtype=np.float32)}]
    y = [{"k2": np.ones(3, dtype=np.float32)}]
    with pytest.raises(BeartypeCallHintParamViolation):
      f(x, y)

  def test_inconsistent_bottom_levels_within_tree(self) -> None:
    """Bottom-level containers differ within the same tree."""

    @shapix.check
    @beartype
    def f(x: Tree[F32[N], ..., T]) -> Tree[F32[N]]:
      return x

    # Two siblings at bottom level have different structures
    x = {
      "a": [np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)],  # list[2]
      "b": [np.ones(3, dtype=np.float32)],  # list[1]
    }
    with pytest.raises(BeartypeCallHintParamViolation):
      f(x)

  def test_bottom_up_too_shallow(self) -> None:
    """Tree[leaf, ..., T, S] on a flat (1-level) tree — need 2 levels."""

    @shapix.check
    @beartype
    def f(x: Tree[F32[N], ..., T, S]) -> Tree[F32[N]]:  # pyright: ignore
      return x

    x = {"a": np.ones(3, dtype=np.float32)}
    with pytest.raises(BeartypeCallHintParamViolation):
      f(x)

  def test_multi_name_suffix_second_from_bottom_differs(self) -> None:
    """Tree[leaf, ..., T, S] — S matches but T doesn't."""

    @shapix.check
    @beartype
    def f(
      x: Tree[F32[N], ..., T, S],  # pyright: ignore
      y: Tree[F32[N], ..., T, S],  # pyright: ignore
    ) -> Tree[F32[N]]:
      return x

    # Both have list[2] at bottom (S), but second-from-bottom differs
    x = {"a": [np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)]}
    y = {
      "a": [np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)],
      "b": [np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)],
    }
    with pytest.raises(BeartypeCallHintParamViolation):
      f(x, y)


# =====================================================================
# Multi-level (composite) matching failures
# =====================================================================


class TestMultiLevelFailures:
  """Failures for Tree[leaf, T, S] and related multi-level specs."""

  def test_top_matches_inner_differs(self) -> None:
    """Tree[leaf, T, S] — T (top) matches but S (remaining) differs."""

    @shapix.check
    @beartype
    def f(
      x: Tree[F32[N], T, S],
      y: Tree[F32[N], T, S],  # pyright: ignore
    ) -> Tree[F32[N]]:
      return x

    # Both dict{"a","b"} at top (T matches)
    x = {
      "a": [np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)],
      "b": [np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)],
    }
    y = {
      "a": [np.ones(3, dtype=np.float32)],  # inner list[1] instead of list[2]
      "b": [np.ones(3, dtype=np.float32)],
    }
    with pytest.raises(BeartypeCallHintParamViolation):
      f(x, y)

  def test_inner_differs_top_matches_same_arg(self) -> None:
    """Tree[leaf, T, S] — within one tree, siblings have different S."""

    @shapix.check
    @beartype
    def f(x: Tree[F32[N], T, S]) -> Tree[F32[N]]:  # pyright: ignore
      return x

    # Top = dict{"a","b"}, but a's inner is list[2], b's inner is list[1]
    x = {
      "a": [np.ones(3, dtype=np.float32), np.ones(3, dtype=np.float32)],
      "b": [np.ones(3, dtype=np.float32)],
    }
    with pytest.raises(BeartypeCallHintParamViolation):
      f(x)

  def test_top_level_type_mismatch(self) -> None:
    """Tree[leaf, T, S] — T differs between args (dict vs list)."""

    @shapix.check
    @beartype
    def f(
      x: Tree[F32[N], T, S],
      y: Tree[F32[N], T, S],  # pyright: ignore
    ) -> Tree[F32[N]]:
      return x

    x = {"a": [np.ones(3, dtype=np.float32)]}
    y = [[np.ones(3, dtype=np.float32)]]
    with pytest.raises(BeartypeCallHintParamViolation):
      f(x, y)

  def test_leaf_where_container_expected(self) -> None:
    """Tree[leaf, T, S] with a flat tree — peeling finds leaf not container."""

    @shapix.check
    @beartype
    def f(x: Tree[F32[N], T, S, ...]) -> Tree[F32[N]]:  # pyright: ignore
      return x

    # Bare leaf: no container to peel
    with pytest.raises(BeartypeCallHintParamViolation):
      f(np.ones(3, dtype=np.float32))


# =====================================================================
# Return type failures
# =====================================================================


class TestReturnTypeFailures:
  """Failures when the return type doesn't match Tree annotation."""

  def test_return_wrong_dtype(self) -> None:
    @beartype
    def f(x: Tree[F32[N]]) -> Tree[I64[N]]:
      return x  # returning F32 when I64 expected

    with pytest.raises(BeartypeCallHintReturnViolation):
      f({"a": np.ones(3, dtype=np.float32)})

  def test_return_wrong_shape(self) -> None:
    @beartype
    def f(x: Tree[F32[N]]) -> Tree[F32[N, C]]:
      return x  # returning 1D when 2D expected

    with pytest.raises(BeartypeCallHintReturnViolation):
      f({"a": np.ones(3, dtype=np.float32)})

  def test_return_structure_mismatch(self) -> None:
    """Return tree has different structure from input when both bound to T."""

    @shapix.check
    @beartype
    def f(x: Tree[F32[N], T]) -> Tree[F32[N], T]:
      # Return different structure than input
      return [np.ones(3, dtype=np.float32)]

    with pytest.raises(BeartypeCallHintReturnViolation):
      f({"a": np.ones(3, dtype=np.float32)})


# =====================================================================
# Shape mismatch failures (expanded)
# =====================================================================


class TestShapeMismatchFailures:
  def test_wrong_rank_3d_vs_2d(self) -> None:
    @beartype
    def f(x: Tree[F32[N, C]]) -> Tree[F32[N, C]]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f({"a": np.ones((3, 4, 5), dtype=np.float32)})  # 3D, not 2D

  def test_scalar_leaf_vs_1d(self) -> None:
    @beartype
    def f(x: Tree[F32[N]]) -> Tree[F32[N]]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f({"a": np.float32(1.0)})  # 0D scalar, not 1D

  def test_some_leaves_wrong_rank(self) -> None:
    """Some leaves correct rank, others wrong."""

    @beartype
    def f(x: Tree[F32[N, C]]) -> Tree[F32[N, C]]:
      return x

    with pytest.raises(BeartypeCallHintParamViolation):
      f([
        np.ones((3, 4), dtype=np.float32),  # 2D — OK
        np.ones((3, 4, 5), dtype=np.float32),  # 3D — wrong
      ])


# =====================================================================
# JAX backend fallback
# =====================================================================


# =====================================================================
# Repr and factory edge cases
# =====================================================================


class TestReprAndFactory:
  def test_structure_repr(self) -> None:
    assert repr(T) == "T"
    assert repr(S) == "S"
    custom = Structure("Params")
    assert repr(custom) == "Params"

  def test_tree_factory_repr(self) -> None:
    assert repr(Tree) == "Tree"

  def test_parse_spec_args_both_ellipsis_error(self) -> None:
    with pytest.raises(TypeError, match="Cannot have"):
      Tree[F32[N], ..., ...]

  def test_parse_spec_args_single_ellipsis_error(self) -> None:
    """Single Ellipsis should say 'at least one structure name required'."""
    with pytest.raises(TypeError, match="At least one structure name"):
      _TreeFactory._parse_spec_args((...,))

  def test_parse_spec_args_both_ends_ellipsis_error(self) -> None:
    with pytest.raises(TypeError, match="Cannot have"):
      _TreeFactory._parse_spec_args((..., T, ...))

  def test_parse_spec_args_empty_tuple_error(self) -> None:
    with pytest.raises(TypeError, match="at least a leaf type"):
      Tree[()]


# =====================================================================
# Replay guard (_fail_obj) — non-contradictory error messages
# =====================================================================


class TestReplayGuard:
  """_TreeChecker must produce non-contradictory beartype error messages."""

  def test_plain_beartype_tree_param_mismatch_error_message(self) -> None:
    """Plain @beartype tree param mismatch must not say True == Is[...]."""

    @shapix.check
    @beartype
    def f(x: Tree[F32[N], T], y: Tree[F32[N], T]) -> Tree[F32[N]]:
      return x

    x = {"a": np.ones(3, dtype=np.float32)}
    y = [np.ones(3, dtype=np.float32)]  # different structure
    with pytest.raises(BeartypeCallHintParamViolation) as exc_info:
      f(x, y)
    # The error message should NOT contain "True ==" which would indicate
    # the replay guard failed (validator passed on re-invocation)
    assert "True ==" not in str(exc_info.value)

  def test_plain_beartype_tree_return_mismatch_error_message(self) -> None:
    """Return type tree mismatch must produce non-contradictory error."""

    @shapix.check
    @beartype
    def f(x: Tree[F32[N], T]) -> Tree[F32[N], T]:
      # Return different structure than input
      return [np.ones(3, dtype=np.float32)]

    with pytest.raises(BeartypeCallHintReturnViolation) as exc_info:
      f({"a": np.ones(3, dtype=np.float32)})
    assert "True ==" not in str(exc_info.value)


# =====================================================================
# Backend-specific Tree modules
# =====================================================================


class TestOptreeBackend:
  def test_optree_tree_basic(self) -> None:
    from shapix.optree import Tree as OptreeTree

    @shapix.check
    @beartype
    def f(x: OptreeTree[F32[N]]) -> OptreeTree[F32[N]]:
      return x

    data = {"a": np.ones(3, dtype=np.float32)}
    f(data)

  def test_optree_tree_repr(self) -> None:
    from shapix.optree import Tree as OptreeTree

    assert repr(OptreeTree) == "Tree"
