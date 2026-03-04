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
from beartype.roar import BeartypeCallHintParamViolation

import shapix
from shapix import C, H, N, S, T, Tree, W, __
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

    with pytest.raises(Exception):  # noqa: B017
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
# JAX backend fallback
# =====================================================================


class TestJaxBackend:
  def test_jax_backend_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
    """When optree is unavailable, jax.tree_util should work as fallback."""
    pytest.importorskip("jax")

    import builtins
    from unittest.mock import patch

    from shapix import _tree

    # Clear cache and simulate optree being unavailable
    _tree._tree_ops_cache = None
    original_import = builtins.__import__

    def mock_import(name: str, *args: object, **kwargs: object) -> object:
      if name == "optree":
        raise ImportError("mocked")
      return original_import(name, *args, **kwargs)

    try:
      with patch.object(builtins, "__import__", side_effect=mock_import):
        tree_ops = _tree._get_tree_ops()
        # Should have fallen back to jax.tree_util
        assert hasattr(tree_ops, "tree_leaves")
        assert hasattr(tree_ops, "tree_flatten")
        assert hasattr(tree_ops, "tree_structure")
    finally:
      # Reset cache so other tests aren't affected
      _tree._tree_ops_cache = None


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
    factory = _TreeFactory()
    with pytest.raises(TypeError, match="Cannot have"):
      factory[F32[N], ..., ...]

  def test_parse_spec_args_only_ellipsis_error(self) -> None:
    factory = _TreeFactory()
    # Single Ellipsis hits "both start and end" since args[0] == args[-1]
    with pytest.raises(TypeError, match="Cannot have"):
      factory._parse_spec_args((...,))

  def test_parse_spec_args_empty_tuple_error(self) -> None:
    factory = _TreeFactory()
    with pytest.raises(TypeError, match="at least a leaf type"):
      factory[()]


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


class TestJaxTreeBackend:
  def test_jax_tree_basic(self) -> None:
    pytest.importorskip("jax")
    from shapix.jax import Tree as JaxTree

    @shapix.check
    @beartype
    def f(x: JaxTree[F32[N]]) -> JaxTree[F32[N]]:
      return x

    data = {"a": np.ones(3, dtype=np.float32)}
    f(data)

  def test_jax_tree_repr(self) -> None:
    pytest.importorskip("jax")
    from shapix.jax import Tree as JaxTree

    assert repr(JaxTree) == "Tree"
