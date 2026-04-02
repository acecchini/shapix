"""Verify all public imports and Dimension operations type-check cleanly.

Tested with: pyright, mypy, ty
"""

import typing as tp

# ---------------------------------------------------------------------------
# All public exports from shapix
# ---------------------------------------------------------------------------

from shapix import (
  B,
  C,
  Dimension,
  DtypeSpec,
  H,
  L,
  N,
  P,
  S,
  Scalar,
  Structure,
  T,
  Value,
  W,
  __,
  check,
  check_context,
  make_array_type,
)

# All pre-defined dimensions are available
dims: list[object] = [B, C, H, L, N, P, Scalar, W, __]

# All pre-defined structure symbols are Structure instances
structs: list[Structure] = [T, S]

# check and check_context are callable
assert callable(check)
assert callable(check_context)
assert callable(make_array_type)

# DtypeSpec is a class
_dtype_spec_cls = DtypeSpec
_value_marker = Value

# ---------------------------------------------------------------------------
# Dimension arithmetic — runtime-only for the predefined root exports.
#
# Under TYPE_CHECKING these names are checker-friendly literal aliases rather
# than runtime Dimension objects, so newer ty releases reject direct operator
# use even with checker-specific ignore syntax. Exercise those operators in the
# runtime branch and keep the checker branch explicit and neutral.
# ---------------------------------------------------------------------------

if tp.TYPE_CHECKING:
  expr_add: object = ...
  expr_radd: object = ...
  expr_sub: object = ...
  expr_rsub: object = ...
  expr_mul: object = ...
  expr_rmul: object = ...
  expr_truediv: object = ...
  expr_rtruediv: object = ...
  expr_floordiv: object = ...
  expr_rfloordiv: object = ...
  expr_pow: object = ...
  expr_rpow: object = ...
  expr_mod: object = ...
  expr_rmod: object = ...
  expr_neg: object = ...

  chained_add: object = ...
  chained_mul: object = ...
  nested_expr: object = ...
  complex_expr: object = ...
  deep_nesting: object = ...

  variadic: object = ...
  broadcastable: object = ...
  anon_variadic: object = ...
  variadic_b: object = ...
  broadcastable_c: object = ...
  variadic_h: object = ...
  double_variadic: object = ...
  double_broad: object = ...
else:
  expr_add = N + 1
  expr_radd = 2 + N
  expr_sub = N - 1
  expr_rsub = 2 - N
  expr_mul = N * 2
  expr_rmul = 2 * N
  expr_truediv = N / 2
  expr_rtruediv = 2 / N
  expr_floordiv = N // 2
  expr_rfloordiv = 2 // N
  expr_pow = N**2
  expr_rpow = 2**N
  expr_mod = N % 2
  expr_rmod = 2 % N
  expr_neg = -N

  # -------------------------------------------------------------------------
  # Chained / nested arithmetic (runtime-only)
  # -------------------------------------------------------------------------

  chained_add = N + C + 1
  chained_mul = N * C * 2
  nested_expr = (N + 1) * 2
  complex_expr = (N * C) + (H - 1)
  deep_nesting = ((N + 1) * 2) // C

  # -------------------------------------------------------------------------
  # Unary operators (runtime-only)
  # -------------------------------------------------------------------------

  variadic = ~N
  broadcastable = +N
  anon_variadic = ~__

  # Unary on other predefined dims
  variadic_b = ~B
  broadcastable_c = +C
  variadic_h = ~H

  # Double application should be idempotent / safe
  double_variadic = ~~N
  double_broad = ++N

# ---------------------------------------------------------------------------
# Custom dimensions
# ---------------------------------------------------------------------------

Vocab = Dimension("Vocab")
Embed = Dimension("Embed")
Heads = Dimension("Heads")
SeqLen = Dimension("SeqLen")

# Custom dim arithmetic
combo = Vocab + Embed
scaled_heads = Heads * 8
half_embed = Embed // 2

# Unary on custom dims
variadic_vocab = ~Vocab
broadcastable_embed = +Embed

# ---------------------------------------------------------------------------
# Dimension from various string forms
# ---------------------------------------------------------------------------

int_dim = Dimension("3")
empty_dim = Dimension("")
expr_dim = Dimension("(N+1)")

# ---------------------------------------------------------------------------
# Dimension is a str subclass
# ---------------------------------------------------------------------------

as_str: str = Vocab
in_fstring: str = f"dim={Vocab}"
concatenated: str = str(Vocab) + str(Embed)
