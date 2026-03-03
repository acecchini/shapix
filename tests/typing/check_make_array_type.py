"""Verify make_array_type and DtypeSpec work with type checkers.

Tested with: pyright, mypy, ty
"""

import numpy as np

from shapix import DtypeSpec, make_array_type

# ---------------------------------------------------------------------------
# DtypeSpec construction
# ---------------------------------------------------------------------------

float32_spec = DtypeSpec("Float32", frozenset({"float32"}))
int_spec = DtypeSpec("Int", frozenset({"int8", "int16", "int32", "int64"}))
multi_spec = DtypeSpec("Real", frozenset({"float32", "float64", "int32", "int64"}))

# Access DtypeSpec attributes
name: str = float32_spec.name
allowed: frozenset[str] = float32_spec.allowed

# ---------------------------------------------------------------------------
# make_array_type usage
# ---------------------------------------------------------------------------

MyF32 = make_array_type(np.ndarray, float32_spec)
MyInt = make_array_type(np.ndarray, int_spec)

# Repr should be available
repr_str: str = repr(MyF32)
