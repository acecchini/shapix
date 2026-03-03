# pyright: reportMissingImports=false
"""PyTree type annotations with runtime leaf-type and structure checking.

A PyTree (pytree) is a nested container structure (dicts, lists, tuples,
namedtuples, etc.) whose leaves are typed arrays. This module provides a
``PyTree`` annotation that validates leaves and optionally enforces structure
consistency across arguments, following the same patterns as jaxtyping.

Requires ``optree`` for tree traversal. Install with ``pip install optree``.

Usage patterns::

    from shapix import PyTree, T, S, N, C
    from shapix.numpy import F32


    # Basic — all leaves must be F32[N, C]
    def f(x: PyTree[F32[N, C]]) -> PyTree[F32[N, C]]: ...


    # Structure binding — x and y must have identical tree structure
    def f(x: PyTree[F32[N, C], T], y: PyTree[F32[N, C], T]): ...


    # Composite — z has structure S at the top, T inside each leaf of S
    def f(x: PyTree[int, T], y: PyTree[int, S], z: PyTree[int, S[T]]): ...


    # Prefix — top-level matches T, subtrees are arbitrary
    def f(x: PyTree[F32[N, C], T[...]]): ...


    # Suffix — inner structure matches T, outer is arbitrary
    def f(x: PyTree[F32[N, C], ..., T]): ...
"""

from __future__ import annotations

import typing as tp
from typing import Annotated

from beartype.vale import Is

__all__ = ["PyTree", "Structure", "T", "S"]


def _import_optree():  # noqa: ANN202
  """Lazy import of optree with clear error message."""
  try:
    import optree
  except ImportError as exc:
    raise ImportError(
      "PyTree requires optree. Install with: pip install optree"
    ) from exc
  return optree


# ---------------------------------------------------------------------------
# Structure spec types (results of Structure operator calls)
# ---------------------------------------------------------------------------


class _PrefixSpec:
  """Result of ``T[...]`` — match top-level structure only."""

  __slots__ = ("_name",)

  def __init__(self, name: str) -> None:
    self._name = name

  def __repr__(self) -> str:
    return f"{self._name}[...]"


class _CompositeSpec:
  """Result of ``S[T]`` — outer S, inner T."""

  __slots__ = ("_names",)

  def __init__(self, names: list[str]) -> None:
    self._names = names

  def __repr__(self) -> str:
    result = self._names[-1]
    for name in reversed(self._names[:-1]):
      result = f"{name}[{result}]"
    return result


# ---------------------------------------------------------------------------
# Structure symbol
# ---------------------------------------------------------------------------


class Structure(str):
  """Named tree structure symbol for PyTree annotations.

  Create symbols and use them in PyTree type annotations::

      from shapix import Structure, PyTree, N, C
      from shapix.numpy import F32

      Params = Structure("Params")


      def f(x: PyTree[F32[N, C], Params], y: PyTree[F32[N, C], Params]): ...

  Operators:

  - ``S[T]`` — composite: outer structure S, inner structure T.
  - ``T[...]`` — prefix: top-level matches T, subtrees arbitrary.
  """

  def __getitem__(self, item: object) -> _CompositeSpec | _PrefixSpec:  # type: ignore[override]
    if item is Ellipsis:
      return _PrefixSpec(str(self))
    if isinstance(item, Structure):
      return _CompositeSpec([str(self), str(item)])
    if isinstance(item, _CompositeSpec):
      return _CompositeSpec([str(self), *item._names])
    msg = (
      f"Structure subscript must be ... or another Structure, got {type(item).__name__}"
    )
    raise TypeError(msg)

  def __repr__(self) -> str:
    return str(self)


# ---------------------------------------------------------------------------
# PyTree checker (beartype validator)
# ---------------------------------------------------------------------------


class _PyTreeChecker:
  """Beartype validator for pytree leaf types and structure consistency."""

  __slots__ = ("_leaf_type", "_structure_spec", "_repr")

  def __init__(self, leaf_type: type, structure_spec: str | None = None) -> None:
    self._leaf_type = leaf_type
    self._structure_spec = structure_spec
    spec_str = f", {structure_spec!r}" if structure_spec else ""
    self._repr = f"PyTree[{leaf_type!r}{spec_str}]"

  def __call__(self, obj: object) -> bool:
    optree = _import_optree()
    from beartype.door import is_bearable

    from ._memo import _get_explicit_stack, get_memo

    # Bridge memo context so is_bearable → _ShapeChecker → get_memo() works
    memo = get_memo(_depth=3)
    stack = _get_explicit_stack()
    stack.append(memo)
    try:
      return self._validate(obj, optree, is_bearable, memo)
    finally:
      stack.pop()

  def _validate(
    self, obj: object, optree: tp.Any, is_bearable: tp.Any, memo: tp.Any
  ) -> bool:
    leaf_type = self._leaf_type

    # Check leaves
    leaves = optree.tree_leaves(obj)
    if not all(
      is_bearable(leaf, leaf_type)
      for leaf in leaves  # pyright: ignore[reportArgumentType]
    ):
      return False

    # Check structure if spec provided
    spec = self._structure_spec
    if spec is None:
      return True

    tree_spec = optree.tree_structure(obj)

    if spec.endswith(" ..."):
      return self._check_prefix(spec[:-4].strip(), obj, optree, memo)
    if spec.startswith("... "):
      return self._check_suffix(spec[4:].strip(), obj, optree, memo)
    if " " in spec:
      return self._check_composite(spec.split(), tree_spec, memo)
    return self._bind_or_check(spec, tree_spec, memo)

  def _bind_or_check(self, name: str, tree_spec: object, memo: tp.Any) -> bool:
    """Bind a structure name or check it matches a previous binding."""
    if name not in memo.structures:
      memo.structures[name] = tree_spec
      return True
    return memo.structures[name] == tree_spec

  def _check_composite(self, names: list[str], tree_spec: object, memo: tp.Any) -> bool:
    """Check composite structure 'S T' — outer structure S, inner structure T.

    For 'S T': the tree should have structure S at the outer level, and each
    leaf of S should be a subtree with structure T.
    """
    optree = _import_optree()

    if len(names) < 2:
      return self._bind_or_check(names[0], tree_spec, memo)

    # For 'S T': outer is S, inner is T
    outer_name = names[0]
    inner_names = names[1:]

    # If outer structure is already bound, use it to flatten one level
    if outer_name in memo.structures:
      # Flatten the tree one level using the known outer structure
      try:
        _leaves, _ = optree.tree_flatten(
          tree_spec,  # type: ignore[arg-type]
          is_leaf=lambda x: x is not tree_spec and not isinstance(x, type(tree_spec)),
        )
      except (TypeError, ValueError):
        return False
    else:
      # Can't verify composite without the outer structure being bound first
      memo.structures[outer_name] = tree_spec
      return True

    if len(inner_names) == 1:
      return self._bind_or_check(inner_names[0], tree_spec, memo)
    return self._check_composite(inner_names, tree_spec, memo)

  def _check_prefix(self, name: str, obj: object, optree: tp.Any, memo: tp.Any) -> bool:
    """'T ...' — top-level matches T, subtrees are arbitrary."""
    # Get the top-level structure (treating subtrees as leaves)
    _children, top_spec = optree.tree_flatten(obj, is_leaf=lambda x: x is not obj)
    return self._bind_or_check(name, top_spec, memo)

  def _check_suffix(self, name: str, obj: object, optree: tp.Any, memo: tp.Any) -> bool:
    """'... T' — inner structure matches T, outer is arbitrary."""
    # Get the full structure
    tree_spec = optree.tree_structure(obj)
    return self._bind_or_check(name, tree_spec, memo)

  def __repr__(self) -> str:
    return self._repr


# ---------------------------------------------------------------------------
# PyTree factory
# ---------------------------------------------------------------------------


class _PyTreeFactory:
  """Subscriptable factory for PyTree type annotations.

  Supports::

      PyTree[LeafType]  # leaf checking only
      PyTree[LeafType, T]  # structure binding
      PyTree[LeafType, S[T]]  # composite structure
      PyTree[LeafType, T[...]]  # prefix (top-level matches T)
      PyTree[LeafType, ..., T]  # suffix (full structure matches T)
  """

  def __getitem__(self, item: object) -> type:
    structure_spec: str | None = None

    if isinstance(item, tuple):
      if len(item) == 2:
        leaf_type, raw_spec = item
        structure_spec = self._resolve_spec(raw_spec)
      elif len(item) == 3 and item[1] is Ellipsis:
        # PyTree[leaf, ..., T] → suffix
        leaf_type = item[0]
        suffix_name = item[2]
        if isinstance(suffix_name, Structure):
          structure_spec = f"... {suffix_name}"
        elif isinstance(suffix_name, str):
          structure_spec = f"... {suffix_name}"
        else:
          msg = f"Suffix structure must be a Structure or str, got {type(suffix_name).__name__}"
          raise TypeError(msg)
      else:
        msg = (
          "PyTree takes 1–3 arguments: PyTree[LeafType], "
          "PyTree[LeafType, T], or PyTree[LeafType, ..., T]"
        )
        raise TypeError(msg)
    else:
      leaf_type = item

    checker = _PyTreeChecker(leaf_type, structure_spec)  # type: ignore[arg-type]
    return Annotated[tp.Any, Is[checker]]  # type: ignore[return-value]

  @staticmethod
  def _resolve_spec(raw: object) -> str:
    """Convert a symbolic or string spec to the internal string form."""
    if isinstance(raw, str):
      return raw
    if isinstance(raw, Structure):
      return str(raw)
    if isinstance(raw, _PrefixSpec):
      return f"{raw._name} ..."
    if isinstance(raw, _CompositeSpec):
      return " ".join(raw._names)
    msg = f"Structure spec must be a Structure, str, or operator result, got {type(raw).__name__}"
    raise TypeError(msg)

  def __repr__(self) -> str:
    return "PyTree"


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
# PyTree type alias
# ---------------------------------------------------------------------------

if tp.TYPE_CHECKING:
  _T = tp.TypeVar("_T")

  class PyTree(tp.Generic[_T]):
    """Static type stub — ``PyTree[LeafType]`` for type checkers."""

    def __class_getitem__(cls, item: object) -> type: ...  # type: ignore[override]

else:
  PyTree = _PyTreeFactory()
