Shapix: Replace jaxtyping with native runtime shape checker                                  
                                                        
 Context

 Shapix currently wraps jaxtyping for runtime shape/dtype checking. jaxtyping has several
 pain points:
 - Requires its own @jaxtyped decorator that replaces/wraps beartype, preventing custom
 BeartypeConf
 - Uses string-based shape specs ("batch channels height width") instead of Python objects
 - Confuses static type checkers with metaclass magic
 - Heavy dependency for what is fundamentally simple logic (~2000 LOC for core checking)

 Goal: Replace jaxtyping with a lean, native runtime checker that:
 - Works with standard @beartype decorators and beartype.claw import hooks
 - Allows full BeartypeConf customization
 - Uses Annotated[T, Is[validator]] so beartype handles it natively
 - Supports cross-argument dimension consistency (e.g., "batch" same across all params)
 - Is compatible with static type checkers (pyright)

 Key Technical Insight

 beartype's Is[validator] call stack is deterministic:
 Frame 0: our validator
 Frame 1: beartype's _is_valid_bool wrapper
 Frame 2: beartype-generated wrapper function  <-- STABLE ANCHOR

 All parameter checks for one function call share the same Frame 2. This lets us implement
 frame-based memo management — cross-arg dimension consistency with JUST @beartype,
 no shapix-specific decorator required.

 Architecture

 src/shapix/
 ├── __init__.py           # Public API exports
 ├── _memo.py              # Frame-based memo management (thread-safe)
 ├── _shape.py             # Shape spec parsing and matching
 ├── _dtypes.py            # Dtype specifications and matching
 ├── _array_types.py       # Array type factory → Annotated[T, Is[checker]]
 ├── _dimensions.py        # Dimension symbols (evolve existing dimension.py)
 ├── _decorator.py         # Optional @shapix.check + context manager
 ├── numpy.py              # NumPy array types (rewritten, no jaxtyping)
 ├── jax.py                # JAX array types (rewritten)
 ├── torch.py              # PyTorch tensor types (new)
 └── claw.py               # Import hook wrapping beartype.claw (optional)

 Remove: src/shapix/jaxtyping/ directory (replaced by _array_types.py, _dtypes.py,
 _dimensions.py).

 Implementation Plan

 Step 1: _memo.py — Frame-based memo management

 Thread-local stack of dimension memos. Each memo tracks bound dimension names → sizes.

 import sys
 import threading
 from dataclasses import dataclass, field

 @dataclass
 class ShapeMemo:
     single: dict[str, int] = field(default_factory=dict)
     variadic: dict[str, tuple[int, ...]] = field(default_factory=dict)

 _local = threading.local()

 def get_memo() -> ShapeMemo:
     """Get memo for the current call context using frame identity."""
     if not hasattr(_local, 'stack'):
         _local.stack = []  # list of (frame_id, ShapeMemo)

     frame = sys._getframe(2)  # Up to beartype wrapper
     frame_id = id(frame)

     # Fast path: top of stack (same call, next param)
     if _local.stack and _local.stack[-1][0] == frame_id:
         return _local.stack[-1][1]

     # Check deeper (nested call returning to outer)
     for i in range(len(_local.stack) - 2, -1, -1):
         if _local.stack[i][0] == frame_id:
             del _local.stack[i+1:]
             return _local.stack[i][1]

     # New call — clean stale entries if stack is large
     if len(_local.stack) > 8:
         active = set()
         f = sys._getframe(0)
         while f:
             active.add(id(f))
             f = f.f_back
         _local.stack = [(fid, m) for fid, m in _local.stack if fid in active]

     memo = ShapeMemo()
     _local.stack.append((frame_id, memo))
     return memo

 # Explicit push/pop for @shapix.check decorator (guaranteed correctness)
 def push_memo() -> ShapeMemo: ...
 def pop_memo() -> None: ...
 def get_explicit_memo() -> ShapeMemo | None: ...

 The frame-based get_memo() is called from Is validators. It auto-detects call boundaries
 using the beartype wrapper frame identity. For users who want guaranteed behavior,
 push_memo/
 pop_memo provide an explicit stack (used by @shapix.check).

 Step 2: _dimensions.py — Dimension symbols

 Evolve existing dimension.py. Keep Dimension(str) with arithmetic, add dim type wrappers.

 class Dimension(str):
     """Named dimension symbol. Also serves as the dim spec for shape checking."""
     # Keep existing __add__, __sub__, __mul__, etc.

     @property
     def _dim_spec(self) -> '_AbstractDim':
         """Convert to internal dim spec for shape matching."""
         raw = str(self)
         if raw == '':
             return _ScalarSpec()
         if raw == '...':
             return _EllipsisSpec()
         if raw.startswith('*'):
             return _VariadicDim(raw[1:], broadcastable=False)
         if raw.startswith('#'):
             return _NamedDim(raw[1:], broadcastable=True)
         if raw == '_' or raw.startswith('_'):
             return _AnonymousDim()
         if raw.isdigit():
             return _FixedDim(int(raw))
         if raw.isidentifier():
             return _NamedDim(raw, broadcastable=False)
         return _SymbolicDim(raw)  # Arithmetic expressions like "(N+1)"

 Pre-defined symbols stay the same: N, B, C, H, W, sB, hN, _N, __, Scalar, Any, etc.

 Additionally, allow passing int literals directly: Float32Array[3, N, N] → _FixedDim(3).

 Step 3: _shape.py — Shape checking

 Dataclasses for dimension specs + matching logic.

 @dataclass(frozen=True)
 class _NamedDim:
     name: str
     broadcastable: bool

 @dataclass(frozen=True)
 class _FixedDim:
     size: int

 @dataclass(frozen=True)
 class _SymbolicDim:
     expr: str  # e.g. "(N+1)"

 @dataclass(frozen=True)
 class _VariadicDim:
     name: str
     broadcastable: bool

 class _AnonymousDim: ...      # Matches any single dim
 class _AnonymousVariadic: ...  # Matches any number of dims

 def check_shape(
     shape: tuple[int, ...],
     spec: tuple[_AbstractDim, ...],
     memo: ShapeMemo,
 ) -> str:
     """Check shape against spec, updating memo. Returns '' on success, error message on
 failure."""

 Core logic mirrors jaxtyping's _check_dims and _check_shape but cleaner:
 - Named dims: first occurrence binds to memo, subsequent must match
 - Fixed dims: must match exactly
 - Symbolic dims: evaluate expression against memo bindings
 - Variadic dims: match zero or more dims, bind to memo
 - Broadcasting: if broadcastable=True, size 1 always matches
 - Anonymous: always matches, no binding

 Step 4: _dtypes.py — Dtype specifications

 @dataclass(frozen=True)
 class DtypeSpec:
     name: str                    # Human-readable name, e.g. "Float32"
     allowed: frozenset[str]      # Allowed dtype strings, e.g. {"float32"}

     def matches(self, obj) -> bool:
         dtype_str = _extract_dtype_str(obj)  # Handles numpy, jax, torch
         return dtype_str in self.allowed

 # Pre-defined specs
 BOOL = DtypeSpec("Bool", frozenset({"bool", "bool_"}))
 INT8 = DtypeSpec("Int8", frozenset({"int8"}))
 # ... all numeric types ...
 FLOAT = DtypeSpec("Float", frozenset({"float16", "float32", "float64", "bfloat16"}))
 SHAPED = DtypeSpec("Shaped", _ALL_DTYPES)  # Any dtype

 def _extract_dtype_str(obj) -> str:
     """Extract dtype as string from numpy/jax/torch array."""
     # numpy/jax: obj.dtype.type.__name__  (with struct array handling)
     # torch: str(obj.dtype).split('.')[-1]
     # fallback: str(obj.dtype)

 Step 5: _array_types.py — Core type factory

 This is the heart of the library. Float32Array[N, C, H, W] returns
 Annotated[np.ndarray, Is[_ShapeChecker(...)]] at runtime.

 from typing import Annotated
 from beartype.vale import Is

 class _ShapeChecker:
     """Callable validator for beartype's Is[]. Checks dtype + shape."""

     __slots__ = ('_dtype_spec', '_shape_spec', '_array_type')

     def __init__(self, array_type: type, dtype_spec: DtypeSpec, shape_spec: tuple):
         self._array_type = array_type
         self._dtype_spec = dtype_spec
         self._shape_spec = shape_spec

     def __call__(self, obj) -> bool:
         # 1. Check dtype
         if not self._dtype_spec.matches(obj):
             return False
         # 2. Get memo (frame-based auto-detection)
         memo = get_memo()
         # 3. Check shape
         return check_shape(obj.shape, self._shape_spec, memo) == ''

     def __repr__(self) -> str:
         dims = ', '.join(str(d) for d in self._shape_spec)
         return f"shapix.{self._dtype_spec.name}[{dims}]"


 class _ArrayFactory:
     """Factory: Float32Array[N, C] → Annotated[ndarray, Is[checker]]"""

     def __init__(self, array_type: type, dtype_spec: DtypeSpec):
         self._array_type = array_type
         self._dtype_spec = dtype_spec

     def __class_getitem__(cls, args):
         # Called on the class itself, not instances
         raise TypeError("Use pre-built factories like Float32Array, not _ArrayFactory
 directly")

     def __getitem__(self, dims) -> type:
         if not isinstance(dims, tuple):
             dims = (dims,)

         # Convert each dim to its spec
         shape_spec = tuple(
             _FixedDim(d) if isinstance(d, int)
             else d._dim_spec if isinstance(d, Dimension)
             else d  # already a spec
             for d in dims
         )

         checker = _ShapeChecker(self._array_type, self._dtype_spec, shape_spec)
         return Annotated[self._array_type, Is[checker]]


 def make_array_type(array_type: type, dtype_spec: DtypeSpec) -> _ArrayFactory:
     """Create a subscriptable array type factory."""
     return _ArrayFactory(array_type, dtype_spec)

 Static type checker integration: Under TYPE_CHECKING, array types are generic NDArray
 aliases.
 At runtime, they're _ArrayFactory instances. The TYPE_CHECKING branch uses Python 3.12
 type statement with TypeVarTuple so pyright sees proper types:

 if TYPE_CHECKING:
     type Float32Array[*Dims] = NDArray[np.float32]  # pyright sees ndarray
 else:
     Float32Array = make_array_type(np.ndarray, FLOAT32)  # runtime: Is[checker]

 Step 6: _decorator.py — Optional explicit decorator + context manager

 For users who want guaranteed cross-arg consistency (no frame heuristics):

 import functools
 from beartype import beartype
 from beartype._conf.confmain import BeartypeConf

 def check(fn=None, *, conf: BeartypeConf | None = None):
     """Optional decorator for guaranteed cross-arg dimension consistency.

     Usage:
         @shapix.check           # memo only, pair with @beartype
         @beartype
         def f(...): ...

         @shapix.check(conf=BeartypeConf(...))  # memo + beartype combined
         def f(...): ...
     """
     def decorator(fn):
         inner = beartype(fn, conf=conf) if conf is not None else fn
         @functools.wraps(fn)
         def wrapper(*args, **kwargs):
             push_memo()
             try:
                 return inner(*args, **kwargs)
             finally:
                 pop_memo()
         return wrapper

     if fn is not None:
         return decorator(fn)
     return decorator


 class check_context:
     """Context manager for manual isinstance checks."""
     def __enter__(self):
         push_memo()
         return self
     def __exit__(self, *exc):
         pop_memo()

 Step 7: numpy.py, jax.py, torch.py — Library-specific types

 Rewrite numpy.py and jax.py to use our own factories instead of jaxtyping.
 Add new torch.py.

 Pattern (same for all three, varying the base array type):

 # numpy.py
 if TYPE_CHECKING:
     type Float32Array[*Dims] = NDArray[np.float32]
     # ... all dtype variants ...
 else:
     Float32Array = make_array_type(np.ndarray, FLOAT32)
     # ... all dtype variants ...

 Keep existing scalar types (Int8Like, Float32Like, etc.) using Annotated[T, Is[...]]
 with beartype validators — these are independent of jaxtyping and stay as-is.

 Keep existing ArrayLike recursive types (_ArrayLike1D, etc.) — these are pure type aliases.

 Step 8: claw.py — Import hook (optional convenience)

 Wraps beartype.claw to auto-add @shapix.check + @beartype:

 from beartype.claw import beartype_package, beartype_this_package
 from beartype._conf.confmain import BeartypeConf

 def shapix_this_package(*, conf: BeartypeConf = BeartypeConf()):
     """Auto-instrument the calling package with shapix + beartype checking."""
     # Use beartype.claw to add @beartype, and configure it to place
     # our decorator appropriately
     beartype_this_package(conf=conf)
     # Note: with frame-based memo, beartype.claw alone is sufficient.
     # This function exists for future extensibility and as a semantic entry point.

 Since the frame-based memo works with standard @beartype, users can just use
 beartype.claw.beartype_this_package() directly. shapix.claw is a thin wrapper
 for discoverability.

 Step 9: Update __init__.py — Clean public API

 from shapix._dimensions import (
     Dimension, Scalar, Any, B, N, P, L, C, H, W, T,
     sB, sN, sL, sC, hN, hL, hC, _B, _N, _L, _C, __,
 )
 from shapix._decorator import check, check_context
 from shapix._dtypes import DtypeSpec
 from shapix._array_types import make_array_type

 Step 10: Update pyproject.toml — Remove jaxtyping dependency

 - Remove jaxtyping from dependencies
 - Keep beartype, numpy as core deps
 - Keep jax, torch as optional deps

 Step 11: Tests

 Create tests/ with:
 - test_memo.py — Frame-based memo: single call, nested calls, sequential calls, threads
 - test_shape.py — Shape matching: named, fixed, symbolic, variadic, broadcast, anonymous
 - test_dtypes.py — Dtype extraction and matching for numpy/jax/torch
 - test_array_types.py — Float32Array[N, C] produces correct Annotated[..., Is[...]]
 - test_integration.py — End-to-end: @beartype + shaped annotations, cross-arg consistency
 - test_decorator.py — @shapix.check, check_context

 Verification

 1. Unit tests: uv run pytest tests/
 2. Type checking: uv run pyright src/shapix/ — verify no errors under TYPE_CHECKING
 3. Integration test: Decorate a function with @beartype, use Float32Array[N, C] annotations,
 verify cross-arg consistency (same N across params) works via frame-based memo
 4. Nested calls: Verify inner function gets its own memo, outer function's memo is preserved
 5. No jaxtyping import: Verify import jaxtyping is not used anywhere in src/

 Migration from current shapix

 - shapix.jaxtyping.ArrayType / ArrayLikeType → replaced by make_array_type()
 - shapix.jaxtyping.dimension.* → moved to shapix._dimensions (same symbols, same API)
 - shapix.jaxtyping.dtype.String / StructDtype → moved to shapix._dtypes
 - shapix.numpy.* / shapix.jax.* — same public API, new internals
 - Delete src/shapix/jaxtyping/ directory entirely