"""Verify basic imports and Dimension operations type-check cleanly."""

from shapix import B, C, Dimension, H, N, P, Scalar, T, W, __, check, check_context

# All pre-defined dimensions are Dimension instances
dims: list[Dimension] = [B, C, H, N, P, Scalar, T, W, __]

# check and check_context are callable
assert callable(check)
assert callable(check_context)

# Dimension arithmetic produces Dimension
expr_add: Dimension = N + 1
expr_radd: Dimension = 2 + N
expr_sub: Dimension = N - 1
expr_mul: Dimension = N * 2
expr_rmul: Dimension = 2 * N
expr_floordiv: Dimension = N // 2
expr_neg: Dimension = -N

# Unary operators
variadic: Dimension = ~N
broadcastable: Dimension = +N
anon_variadic: Dimension = ~__

# Custom dimensions
Vocab = Dimension("Vocab")
Embed = Dimension("Embed")
combo: Dimension = Vocab + Embed
