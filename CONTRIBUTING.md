# Contributing

## Local development

```bash
uv sync                     # install all deps (dev + optional + static)
uv run pytest tests/ -n=auto  # run all tests with xdist
uv run pyright src/         # type-check source
uv run tox run -e dev       # locked env with all deps
```

Use `-n=auto` by default. Fall back to `-n0` only when debugging an xdist-specific
failure or when you need fully serial, deterministic reproduction of a narrow test.

## Tox environment naming

Environments follow a **factor-based** naming scheme. Factor deps are merged automatically.

### Runtime: `{python}-{beartype}-{backend}`

| Factor   | Values |
|----------|--------|
| Python   | `py312`, `py313`, `py314` |
| beartype | `bt020`, `bt021`, `bt022` |
| Backend  | `numpy22`..`numpy24`, `jax05`..`jax09`, `torch26`..`torch210`, `optree014`..`optree019` |

```bash
uv run tox run -e py312-bt022-numpy24    # numpy 2.4 on py3.12
uv run tox run -e py314-bt022-jax09      # jax 0.9 on py3.14
```

### Typecheck: `{python}-{beartype}-type-{checker}`

```bash
uv run tox run -e py312-bt022-type-pyright1408
uv run tox run -e py312-bt022-type-mypy119
uv run tox run -e py312-bt022-type-ty
```

The `type` factor installs all backends so all imports resolve during type checking.

## CI tiers

| Tier | Trigger | Runtime envs | Typecheck envs |
|------|---------|-------------|----------------|
| PR   | pull_request | py{312,314} × bt022 × latest backends | latest checkers |
| Push | push to main | py{312,313,314} × bt022 × latest backends | broader checker versions |
| Nightly | cron 04:00 UTC | py312 × all beartype × all versions | all checker versions |

## Adding a new backend version

1. Add the factor to `tox.toml` (e.g. `[env.numpy25]`)
2. Add the factor name to `tools/validate_tox_env.py` `BACKENDS` set
3. Add to CI matrix in `.github/workflows/ci.yml` and `nightly.yml`
