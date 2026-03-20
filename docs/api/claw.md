---
title: "shapix.claw"
description: Import hook for package-wide beartype instrumentation.
---

# `shapix.claw`

Thin wrapper around `beartype.claw` for package-wide instrumentation.

```python
from shapix.claw import shapix_this_package
```

---

## `shapix_this_package`

```python
def shapix_this_package(*, conf: BeartypeConf = BeartypeConf()) -> None:
    """Instrument the calling package with @beartype for runtime checking.

    This is a thin wrapper around beartype.claw.beartype_this_package
    that exists as a semantic entry point for shapix users.

    Parameters
    ----------
    conf
        Optional BeartypeConf for customizing the checking behavior.
    """
```

### Example

```python
# your_package/__init__.py
from shapix.claw import shapix_this_package
shapix_this_package()
```

All subsequently imported submodules in `your_package` will have `@beartype` applied automatically to every annotated function.

### With custom configuration

```python
from beartype import BeartypeConf
from shapix.claw import shapix_this_package

shapix_this_package(conf=BeartypeConf(
    is_color=False,
))
```

See [Import Hook](../features/claw.md) for detailed usage guide.
