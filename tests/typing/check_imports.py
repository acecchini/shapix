"""Verify all public imports and Dimension operations type-check cleanly.

Tested with: pyright, mypy, ty
"""

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
  W,
  __,
  check,
  check_context,
  make_array_type,
)

# All pre-defined dimensions are Dimension instances
dims: list[Dimension] = [B, C, H, L, N, P, Scalar, W, __]

# All pre-defined structure symbols are Structure instances
structs: list[Structure] = [T, S]

# check and check_context are callable
assert callable(check)
assert callable(check_context)
assert callable(make_array_type)

# DtypeSpec is a class
_ = DtypeSpec

# ---------------------------------------------------------------------------
# Dimension arithmetic — all binary operators
# ---------------------------------------------------------------------------

expr_add: Dimension = N + 1
expr_radd: Dimension = 2 + N
expr_sub: Dimension = N - 1
expr_rsub: Dimension = 2 - N
expr_mul: Dimension = N * 2
expr_rmul: Dimension = 2 * N
expr_truediv: Dimension = N / 2
expr_rtruediv: Dimension = 2 / N
expr_floordiv: Dimension = N // 2
expr_rfloordiv: Dimension = 2 // N
expr_pow: Dimension = N**2
expr_rpow: Dimension = 2**N
expr_mod: Dimension = N % 2
expr_rmod: Dimension = 2 % N
expr_neg: Dimension = -N

# ---------------------------------------------------------------------------
# Chained / nested arithmetic
# ---------------------------------------------------------------------------

chained_add: Dimension = N + C + 1
chained_mul: Dimension = N * C * 2
nested_expr: Dimension = (N + 1) * 2
complex_expr: Dimension = (N * C) + (H - 1)
deep_nesting: Dimension = ((N + 1) * 2) // C

# ---------------------------------------------------------------------------
# Unary operators
# ---------------------------------------------------------------------------

variadic: Dimension = ~N
broadcastable: Dimension = +N
anon_variadic: Dimension = ~__

# Unary on other predefined dims
variadic_b: Dimension = ~B
broadcastable_c: Dimension = +C
variadic_h: Dimension = ~H

# Double application should be idempotent / safe
double_variadic: Dimension = ~~N
double_broad: Dimension = ++N

# ---------------------------------------------------------------------------
# Custom dimensions
# ---------------------------------------------------------------------------

Vocab = Dimension("Vocab")
Embed = Dimension("Embed")
Heads = Dimension("Heads")
SeqLen = Dimension("SeqLen")

# Custom dim arithmetic
combo: Dimension = Vocab + Embed
scaled_heads: Dimension = Heads * 8
half_embed: Dimension = Embed // 2

# Unary on custom dims
variadic_vocab: Dimension = ~Vocab
broadcastable_embed: Dimension = +Embed

# ---------------------------------------------------------------------------
# Dimension from various string forms
# ---------------------------------------------------------------------------

int_dim = Dimension("3")
empty_dim = Dimension("")
expr_dim = Dimension("(N+1)")

# ---------------------------------------------------------------------------
# Dimension is a str subclass
# ---------------------------------------------------------------------------

as_str: str = N
in_fstring: str = f"dim={N}"
concatenated: str = str(N) + str(C)
