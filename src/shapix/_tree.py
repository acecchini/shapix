# pyright: reportMissingImports=false
"""Tree type annotations with runtime leaf-type and structure checking.

A tree (pytree) is a nested container structure (dicts, lists, tuples,
namedtuples, etc.) whose leaves are typed arrays. This module provides a
``Tree`` annotation that validates leaves and optionally enforces structure
consistency across arguments, following the same patterns as jaxtyping.
At runtime ``Tree[...]`` produces a shapix runtime hint, so beartype can report
readable leaf and structure failures without changing the public syntax.

Requires ``optree`` or ``jax`` for tree traversal. Install with
``pip install optree`` or ``pip install jax``.

.. note::

   Structure arguments (``T``, ``S``, ``...``) are **runtime-only**.
   Type checkers see ``Tree`` as ``Tree[LeafType]`` (one type parameter)
   and cannot validate multi-arg structure syntax like ``Tree[F32[N], T]``.
   Leaf-only annotations such as ``Tree[F32[N, C]]`` are fully supported
   by all type checkers.

Import ``Tree`` from an explicit backend module::

    from shapix.optree import Tree  # backed by optree
    from shapix.jax import Tree  # backed by jax.tree_util

Usage patterns::

    from shapix import T, S, N, C
    from shapix.optree import Tree
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

from beartype.door import TypeHint

from ._memo import ShapeMemo, has_untagged_memo
from ._runtime_hints import (
  ReplayFailureState,
  ValidationFailure,
  get_runtime_validator,
  hint_label,
  make_runtime_hint,
)

__all__ = ["Structure", "T", "S"]


# ---------------------------------------------------------------------------
# Structure symbol
# ---------------------------------------------------------------------------


class Structure(str):
  """Named tree structure symbol for Tree annotations.

  Create symbols and use them in Tree type annotations::

      from shapix import Structure, N, C
      from shapix.optree import Tree
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

  __slots__ = ("_leaf_type", "_structure_spec", "_get_ops", "_repr", "_fail_state")

  def __init__(
    self,
    leaf_type: object,
    structure_spec: str | None = None,
    *,
    get_ops: Callable[[], tp.Any],
  ) -> None:
    self._leaf_type = leaf_type
    self._structure_spec = structure_spec
    self._get_ops = get_ops
    self._fail_state = ReplayFailureState()
    spec_str = f", {structure_spec}" if structure_spec else ""
    self._repr = f"Tree[{hint_label(leaf_type)}{spec_str}]"

  def __call__(self, obj: object) -> bool:
    return self.instancecheck(obj)

  def instancecheck(self, obj: object) -> bool:
    if self._fail_state.should_replay(obj):
      return False

    tree_ops = self._get_ops()
    from ._memo import get_memo, get_scope, pop_memo, push_memo

    # Bridge memo + runtime scope so leaf checks reuse the caller's bindings
    # and can resolve ``Value(...)`` expressions against the same parameters.
    memo = get_memo()
    scope = get_scope()
    snap = memo.snapshot()
    has_prior = any(snap)
    push_memo(memo, scope=scope)
    try:
      failure = self._validate(obj, tree_ops, memo)
    finally:
      pop_memo()

    if failure is None:
      self._fail_state.clear()
      return True

    memo.restore(snap)
    if has_prior and not has_untagged_memo():
      self._fail_state.record(obj, memo, failure)
    else:
      self._fail_state.clear()
    return False

  def instancecheck_str(self, obj: object) -> str:
    detail = self._fail_state.detail_for(obj)
    if detail is None:
      tree_ops = self._get_ops()
      from ._memo import get_memo, get_scope, pop_memo, push_memo

      memo = get_memo()
      scope = get_scope()
      snap = memo.snapshot()
      push_memo(memo, scope=scope)
      try:
        failure = self._validate(obj, tree_ops, memo)
      finally:
        pop_memo()
      memo.restore(snap)
      if failure is None:
        detail = ValidationFailure(f"unexpectedly accepted {obj!r} for {self!r}")
      else:
        detail = failure
    self._fail_state.clear()
    return detail.message

  def _validate(
    self, obj: object, tree_ops: tp.Any, memo: ShapeMemo
  ) -> ValidationFailure | None:
    if obj is None and self._structure_spec is not None:
      return ValidationFailure("tree structure checks require a non-None tree")

    leaf_type = self._leaf_type

    # Check leaves
    leaves = tree_ops.tree_leaves(obj)
    for leaf in leaves:  # pyright: ignore[reportArgumentType]
      failure = self._check_leaf(leaf, leaf_type)
      if failure is not None:
        return failure

    # Check structure if spec provided
    spec = self._structure_spec
    if spec is None:
      return None

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

  def _check_leaf(self, leaf: object, leaf_type: object) -> ValidationFailure | None:
    validator = get_runtime_validator(leaf_type)
    if validator is not None:
      if validator.instancecheck(leaf):
        return None
      return ValidationFailure(
        f"tree leaf {leaf!r} violates {hint_label(leaf_type)}: {validator.instancecheck_str(leaf)}"
      )

    from beartype.door import is_bearable

    if is_bearable(leaf, tp.cast(tp.Any, leaf_type)):
      return None
    return ValidationFailure(
      f"tree leaf {leaf!r} does not match {hint_label(leaf_type)}"
    )

  def _bind_or_check(
    self, name: str, tree_spec: object, memo: ShapeMemo
  ) -> ValidationFailure | None:
    """Bind a structure name or check it matches a previous binding."""
    if name not in memo.structures:
      memo.structures[name] = tree_spec
      return None

    existing = memo.structures[name]
    if existing == tree_spec:
      return None
    return ValidationFailure(
      f"tree structure '{name}' does not match existing binding {existing!r}; got {tree_spec!r}"
    )

  def _check_top_down(
    self,
    names: list[str],
    obj: object,
    tree_ops: tp.Any,
    memo: ShapeMemo,
    *,
    wildcard: bool,
  ) -> ValidationFailure | None:
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
          failure = self._bind_or_check(name, spec, memo)
          if failure is not None:
            return failure
        return None
      # Peel one level
      next_nodes = []
      for node in current_nodes:
        children, top_spec = tree_ops.tree_flatten(
          node, is_leaf=lambda x, n=node: x is not n
        )
        if len(children) == 1 and children[0] is node:
          return ValidationFailure(
            f"tree structure '{name}' expected a container level but found a leaf"
          )
        failure = self._bind_or_check(name, top_spec, memo)
        if failure is not None:
          return failure
        next_nodes.extend(children)
      current_nodes = next_nodes
    return None

  def _check_bottom_up(
    self, names: list[str], obj: object, tree_ops: tp.Any, memo: ShapeMemo
  ) -> ValidationFailure | None:
    """Check structure names bottom-up (inner to outer).

    Names match from the bottom: the last name matches the leaf-adjacent
    container level, the second-to-last matches the level above, etc.
    Outer levels beyond the names are unchecked.
    """
    levels = self._collect_levels(obj, tree_ops)
    if len(levels) < len(names):
      return ValidationFailure(
        f"tree structure spec '{' '.join(names)}' requires {len(names)} container levels but got {len(levels)}"
      )
    for i, name in enumerate(reversed(names)):
      level_specs = levels[len(levels) - 1 - i]
      first = level_specs[0]
      if not all(s == first for s in level_specs):
        return ValidationFailure(
          f"tree structure '{name}' has inconsistent sibling structures at one level"
        )
      failure = self._bind_or_check(name, first, memo)
      if failure is not None:
        return failure
    return None

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

  __slots__ = ("_get_ops", "_name", "_cache", "_module")

  def __init__(self, get_ops: Callable[[], tp.Any], *, name: str = "Tree") -> None:
    self._get_ops = get_ops
    self._name = name
    self._cache: dict[tuple[object, str | None], type] = {}
    self._module = getattr(get_ops, "__module__", __name__)

  def __getitem__(self, item: object) -> type:
    if not isinstance(item, tuple):
      self._validate_leaf_type(item, owner_name=self._name)
      return self._make(item, None)

    if len(item) < 1:
      msg = f"{self._name} requires at least a leaf type: {self._name}[LeafType]"
      raise TypeError(msg)

    leaf_type = item[0]
    self._validate_leaf_type(leaf_type, owner_name=self._name)
    spec_args = item[1:]

    if not spec_args:
      return self._make(leaf_type, None)

    structure_spec = self._parse_spec_args(spec_args, owner_name=self._name)
    return self._make(leaf_type, structure_spec)

  @staticmethod
  def _validate_leaf_type(leaf_type: object, *, owner_name: str = "Tree") -> None:
    try:
      TypeHint(tp.cast(tp.Any, leaf_type))
    except Exception as exc:  # noqa: BLE001
      msg = (
        f"{owner_name} leaf type must be a beartype-valid type hint, got {leaf_type!r}"
      )
      raise TypeError(msg) from exc

  @staticmethod
  def _normalize_structure_arg(arg: object, *, owner_name: str = "Tree") -> str:
    if not isinstance(arg, Structure):
      msg = (
        f"{owner_name} structure arguments must be Structure names or Ellipsis, "
        f"got {arg!r}"
      )
      raise TypeError(msg)

    name = str(arg)
    if not name or name == "..." or any(ch.isspace() for ch in name):
      msg = (
        f"{owner_name} structure names must be non-empty single names and "
        f"cannot be '...'; got {name!r}"
      )
      raise TypeError(msg)
    return name

  @staticmethod
  def _parse_spec_args(args: tuple[object, ...], *, owner_name: str = "Tree") -> str:
    """Parse structure spec arguments into internal string form."""
    has_leading = args[0] is Ellipsis
    has_trailing = args[-1] is Ellipsis

    if has_leading and has_trailing and len(args) > 1:
      msg = "Cannot have ... at both start and end of structure spec"
      raise TypeError(msg)

    names = [
      _TreeFactory._normalize_structure_arg(arg, owner_name=owner_name)
      for arg in args
      if arg is not Ellipsis
    ]
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
    key = (leaf_type, structure_spec)
    cached = self._cache.get(key)
    if cached is not None:
      return cached

    checker = _TreeChecker(leaf_type, structure_spec, get_ops=self._get_ops)
    hint = make_runtime_hint(repr(checker), checker, module=self._module, origin=object)
    self._cache[key] = hint
    return hint

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
