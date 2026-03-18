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
# Dimension arithmetic — runtime-only (not representable as static types)
# These use type: ignore because Dimension arithmetic requires runtime objects
# ---------------------------------------------------------------------------

expr_add = N + 1  # type: ignore[operator]
expr_radd = 2 + N  # type: ignore[operator]
expr_sub = N - 1  # type: ignore[operator]
expr_rsub = 2 - N  # type: ignore[operator]
expr_mul = N * 2  # type: ignore[operator]
expr_rmul = 2 * N  # type: ignore[operator]
expr_truediv = N / 2  # type: ignore[operator]
expr_rtruediv = 2 / N  # type: ignore[operator]
expr_floordiv = N // 2  # type: ignore[operator]
expr_rfloordiv = 2 // N  # type: ignore[operator]
expr_pow = N**2  # type: ignore[operator]
expr_rpow = 2**N  # type: ignore[operator]
expr_mod = N % 2  # type: ignore[operator]
expr_rmod = 2 % N  # type: ignore[operator]
expr_neg = -N  # type: ignore[operator]

# ---------------------------------------------------------------------------
# Chained / nested arithmetic (runtime-only)
# ---------------------------------------------------------------------------

chained_add = N + C + 1  # type: ignore[operator]
chained_mul = N * C * 2  # type: ignore[operator]
nested_expr = (N + 1) * 2  # type: ignore[operator]
complex_expr = (N * C) + (H - 1)  # type: ignore[operator]
deep_nesting = ((N + 1) * 2) // C  # type: ignore[operator]

# ---------------------------------------------------------------------------
# Unary operators (runtime-only)
# ---------------------------------------------------------------------------

variadic = ~N  # type: ignore[operator]
broadcastable = +N  # type: ignore[operator]
anon_variadic = ~__  # type: ignore[operator]

# Unary on other predefined dims
variadic_b = ~B  # type: ignore[operator]
broadcastable_c = +C  # type: ignore[operator]
variadic_h = ~H  # type: ignore[operator]

# Double application should be idempotent / safe
double_variadic = ~~N  # type: ignore[operator]
double_broad = ++N  # type: ignore[operator]

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
