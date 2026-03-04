# pyright: reportMissingImports=false
"""Tree type annotations with runtime leaf-type and structure checking.

A tree (pytree) is a nested container structure (dicts, lists, tuples,
namedtuples, etc.) whose leaves are typed arrays. This module provides a
``Tree`` annotation that validates leaves and optionally enforces structure
consistency across arguments, following the same patterns as jaxtyping.

Requires ``optree`` or ``jax`` for tree traversal. Install with
``pip install optree`` or ``pip install jax``.

Usage patterns::

    from shapix import Tree, T, S, N, C
    from shapix.numpy import F32


    # Basic — all leaves must be F32[N, C]
    def f(x: Tree[F32[N, C]]) -> Tree[F32[N, C]]: ...


    # Structure binding — x and y must have identical tree structure
    def f(x: Tree[F32[N, C], T], y: Tree[F32[N, C], T]): ...


    # Top-level only — T matches top level, subtrees are arbitrary
    def f(x: Tree[F32[N, C], T, ...]): ...


    # Bottom-level — T matches the leaf-adjacent container level
    def f(x: Tree[F32[N, C], ..., T]): ...


    # Multi-level — T = top level, S = full remaining structure below
    def f(x: Tree[int, T], y: Tree[int, S], z: Tree[int, T, S]): ...
"""

from __future__ import annotations

import typing as tp
from collections.abc import Callable
from typing import Annotated

from beartype.vale import Is

__all__ = ["Tree", "Structure", "T", "S"]


_tree_ops_cache: tp.Any = None


def _get_tree_ops() -> tp.Any:
  """Get tree operations module (optree or jax.tree_util), auto-detecting."""
  global _tree_ops_cache  # noqa: PLW0603
  if _tree_ops_cache is not None:
    return _tree_ops_cache
  try:
    import optree

    _tree_ops_cache = optree
    return optree
  except ImportError:
    pass
  try:
    import jax.tree_util as jtu

    _tree_ops_cache = jtu
    return jtu
  except ImportError:
    pass
  msg = "Tree requires optree or jax. Install with: pip install optree"
  raise ImportError(msg)


# ---------------------------------------------------------------------------
# Structure symbol
# ---------------------------------------------------------------------------


class Structure(str):
  """Named tree structure symbol for Tree annotations.

  Create symbols and use them in Tree type annotations::

      from shapix import Structure, Tree, N, C
      from shapix.numpy import F32

      Params = Structure("Params")


      def f(x: Tree[F32[N, C], Params], y: Tree[F32[N, C], Params]): ...
  """

  def __repr__(self) -> str:
    return str(self)


# ---------------------------------------------------------------------------
# Tree checker (beartype validator)
# ---------------------------------------------------------------------------


class _TreeChecker:
  """Beartype validator for tree leaf types and structure consistency."""

  __slots__ = ("_leaf_type", "_structure_spec", "_get_ops", "_repr")

  def __init__(
    self,
    leaf_type: type,
    structure_spec: str | None = None,
    get_ops: Callable[[], tp.Any] = _get_tree_ops,
  ) -> None:
    self._leaf_type = leaf_type
    self._structure_spec = structure_spec
    self._get_ops = get_ops
    spec_str = f", {structure_spec!r}" if structure_spec else ""
    self._repr = f"Tree[{leaf_type!r}{spec_str}]"

  def __call__(self, obj: object) -> bool:
    tree_ops = self._get_ops()
    from beartype.door import is_bearable

    from ._memo import _get_explicit_stack, get_memo

    # Bridge memo context so is_bearable → _ShapeChecker → get_memo() works
    memo = get_memo(_depth=3)
    stack = _get_explicit_stack()
    stack.append(memo)
    try:
      return self._validate(obj, tree_ops, is_bearable, memo)
    finally:
      stack.pop()

  def _validate(
    self, obj: object, tree_ops: tp.Any, is_bearable: tp.Any, memo: tp.Any
  ) -> bool:
    leaf_type = self._leaf_type

    # Check leaves
    leaves = tree_ops.tree_leaves(obj)
    if not all(
      is_bearable(leaf, leaf_type)
      for leaf in leaves  # pyright: ignore[reportArgumentType]
    ):
      return False

    # Check structure if spec provided
    spec = self._structure_spec
    if spec is None:
      return True

    if spec.endswith(" ..."):
      names = spec[:-4].split()
      return self._check_top_down(names, obj, tree_ops, memo, wildcard=True)
    if spec.startswith("... "):
      names = spec[4:].split()
      return self._check_bottom_up(names, obj, tree_ops, memo)
    names = spec.split()
    if len(names) == 1:
      return self._bind_or_check(names[0], tree_ops.tree_structure(obj), memo)
    return self._check_top_down(names, obj, tree_ops, memo, wildcard=False)

  def _bind_or_check(self, name: str, tree_spec: object, memo: tp.Any) -> bool:
    """Bind a structure name or check it matches a previous binding."""
    if name not in memo.structures:
      memo.structures[name] = tree_spec
      return True
    return memo.structures[name] == tree_spec

  def _check_top_down(
    self,
    names: list[str],
    obj: object,
    tree_ops: tp.Any,
    memo: tp.Any,
    *,
    wildcard: bool,
  ) -> bool:
    """Check structure names top-down (outer to inner).

    Without wildcard: each name except the last captures ONE level;
    the last captures the FULL remaining structure.

    With wildcard: ALL names capture one level each; inner levels unchecked.
    """
    current_nodes = [obj]
    for i, name in enumerate(names):
      is_last = i == len(names) - 1
      if is_last and not wildcard:
        # Last name → capture full remaining structure of every node
        for node in current_nodes:
          spec = tree_ops.tree_structure(node)
          if not self._bind_or_check(name, spec, memo):
            return False
        return True
      # Peel one level
      next_nodes = []
      for node in current_nodes:
        children, top_spec = tree_ops.tree_flatten(
          node, is_leaf=lambda x, n=node: x is not n
        )
        if len(children) == 1 and children[0] is node:
          return False  # expected container, got leaf
        if not self._bind_or_check(name, top_spec, memo):
          return False
        next_nodes.extend(children)
      current_nodes = next_nodes
    return True

  def _check_bottom_up(
    self, names: list[str], obj: object, tree_ops: tp.Any, memo: tp.Any
  ) -> bool:
    """Check structure names bottom-up (inner to outer).

    Names match from the bottom: the last name matches the leaf-adjacent
    container level, the second-to-last matches the level above, etc.
    Outer levels beyond the names are unchecked.
    """
    levels = self._collect_levels(obj, tree_ops)
    if len(levels) < len(names):
      return False
    for i, name in enumerate(reversed(names)):
      level_specs = levels[len(levels) - 1 - i]
      first = level_specs[0]
      if not all(s == first for s in level_specs):
        return False
      if not self._bind_or_check(name, first, memo):
        return False
    return True

  @staticmethod
  def _collect_levels(obj: object, tree_ops: tp.Any) -> list[list[object]]:
    """Collect one-level structure specs at each depth of the tree."""
    levels: list[list[object]] = []
    current_nodes = [obj]
    while current_nodes:
      level_specs: list[object] = []
      next_nodes: list[object] = []
      for node in current_nodes:
        children, top_spec = tree_ops.tree_flatten(
          node, is_leaf=lambda x, n=node: x is not n
        )
        if len(children) == 1 and children[0] is node:
          continue  # leaf
        level_specs.append(top_spec)
        next_nodes.extend(children)
      if not level_specs:
        break
      levels.append(level_specs)
      current_nodes = next_nodes
    return levels

  def __repr__(self) -> str:
    return self._repr


# ---------------------------------------------------------------------------
# Tree factory
# ---------------------------------------------------------------------------


class _TreeFactory:
  """Subscriptable factory for Tree type annotations.

  Supports::

      Tree[LeafType]  # leaf checking only
      Tree[LeafType, T]  # full structure binding
      Tree[LeafType, T, ...]  # top-level only
      Tree[LeafType, ..., T]  # bottom-level only
      Tree[LeafType, T, S]  # T = top level, S = full remaining
      Tree[LeafType, T, S, ...]  # T = top, S = next, rest arbitrary
      Tree[LeafType, ..., T, S]  # S = bottom, T = second-from-bottom
  """

  __slots__ = ("_get_ops", "_name")

  def __init__(
    self, get_ops: Callable[[], tp.Any] = _get_tree_ops, *, name: str = "Tree"
  ) -> None:
    self._get_ops = get_ops
    self._name = name

  def __getitem__(self, item: object) -> type:
    if not isinstance(item, tuple):
      return self._make(item, None)

    if len(item) < 1:
      msg = f"{self._name} requires at least a leaf type: {self._name}[LeafType]"
      raise TypeError(msg)

    leaf_type = item[0]
    spec_args = item[1:]

    if not spec_args:
      return self._make(leaf_type, None)

    structure_spec = self._parse_spec_args(spec_args)
    return self._make(leaf_type, structure_spec)

  @staticmethod
  def _parse_spec_args(args: tuple[object, ...]) -> str:
    """Parse structure spec arguments into internal string form."""
    has_leading = args[0] is Ellipsis
    has_trailing = args[-1] is Ellipsis

    if has_leading and has_trailing:
      msg = "Cannot have ... at both start and end of structure spec"
      raise TypeError(msg)

    names = [str(a) for a in args if a is not Ellipsis]
    if not names:
      msg = "At least one structure name required"
      raise TypeError(msg)

    spec = " ".join(names)
    if has_trailing:
      spec += " ..."
    elif has_leading:
      spec = "... " + spec
    return spec

  def _make(self, leaf_type: object, structure_spec: str | None) -> type:
    checker = _TreeChecker(leaf_type, structure_spec, self._get_ops)  # type: ignore[arg-type]
    return Annotated[tp.Any, Is[checker]]  # type: ignore[return-value]

  def __repr__(self) -> str:
    return self._name


# ---------------------------------------------------------------------------
# Pre-defined structure symbols
# ---------------------------------------------------------------------------

if tp.TYPE_CHECKING:
  T: Structure
  S: Structure
else:
  T = Structure("T")
  S = Structure("S")

# ---------------------------------------------------------------------------
# Tree type alias (auto-detecting backend: optree → jax fallback)
# ---------------------------------------------------------------------------

if tp.TYPE_CHECKING:

  class Tree[_T]:
    """Static type stub — ``Tree[LeafType]`` for type checkers."""

    def __class_getitem__(cls, item: object) -> type: ...  # type: ignore[override]

else:
  Tree = _TreeFactory()
