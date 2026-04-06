"""Validate tox environment names follow the factor scheme.

Usage: python tools/validate_tox_env.py <env_name>

Factor scheme:
  Runtime:    {python}-{beartype}-{backend}
  Typecheck:  {python}-{beartype}-type-{checker}

Exits non-zero on invalid env names.
"""

from __future__ import annotations

import sys

PYTHONS = {"py310", "py311", "py312", "py313"}
BEARTYPES = {"bt020", "bt021", "bt022"}
BACKENDS = {
  "numpy22",
  "numpy23",
  "numpy24",
  "jax05",
  "jax06",
  "jax07",
  "jax08",
  "jax09",
  "torch26",
  "torch27",
  "torch28",
  "torch29",
  "torch210",
  "optree014",
  "optree015",
  "optree016",
  "optree017",
  "optree018",
  "optree019",
}
CHECKERS = {
  "pyright1392",
  "pyright1400",
  "pyright1408",
  "mypy115",
  "mypy116",
  "mypy117",
  "mypy118",
  "mypy119",
  "ty",
}
SPECIAL = {"dev"}


def validate(env_name: str) -> str | None:
  """Return error message, or None if valid."""
  if env_name in SPECIAL:
    return None

  factors = env_name.split("-")

  # Classify each factor
  pythons = [f for f in factors if f in PYTHONS]
  beartypes = [f for f in factors if f in BEARTYPES]
  backends = [f for f in factors if f in BACKENDS]
  checkers = [f for f in factors if f in CHECKERS]
  is_type = "type" in factors

  # Collect known factors
  known = set(pythons + beartypes + backends + checkers)
  if is_type:
    known.add("type")
  unknown = [f for f in factors if f not in known]

  if unknown:
    return f"unknown factors: {', '.join(unknown)}"
  if len(pythons) != 1:
    return f"expected exactly 1 python factor, got {len(pythons)}: {pythons}"
  if len(beartypes) != 1:
    return f"expected exactly 1 beartype factor, got {len(beartypes)}: {beartypes}"

  if is_type:
    # Typecheck env: {python}-{beartype}-type-{checker}
    if len(checkers) != 1:
      return f"type env needs exactly 1 checker, got {len(checkers)}: {checkers}"
    if backends:
      return f"type env should not have backend factors: {backends}"
  else:
    # Runtime env: {python}-{beartype}-{backend}
    if len(backends) != 1:
      return f"runtime env needs exactly 1 backend, got {len(backends)}: {backends}"
    if checkers:
      return f"runtime env should not have checker factors: {checkers}"

  return None


def main() -> None:
  if len(sys.argv) != 2:
    sys.stderr.write(f"usage: {sys.argv[0]} <env_name>\n")
    sys.exit(2)

  env_name = sys.argv[1]
  error = validate(env_name)
  if error:
    sys.stderr.write(f"invalid tox env '{env_name}': {error}\n")
    sys.exit(1)


if __name__ == "__main__":
  main()
