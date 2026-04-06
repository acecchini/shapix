---
description: Thin wrapper around beartype.claw for package-wide instrumentation.
---

# `shapix.claw`

`shapix.claw` contains one public helper:

```python
from shapix.claw import shapix_this_package
```

## `shapix_this_package`

`shapix_this_package(*, conf: BeartypeConf = BeartypeConf()) -> None`

It forwards directly to `beartype.claw.beartype_this_package`, but gives shapix
users a semantic entry point that matches the rest of the library.

### Example

```python
# your_package/__init__.py
from shapix.claw import shapix_this_package

shapix_this_package()
```

All subsequently imported submodules in `your_package` are instrumented with
beartype, so shapix array annotations start working there automatically.

### Custom configuration

```python
from beartype import BeartypeConf
from shapix.claw import shapix_this_package

shapix_this_package(conf=BeartypeConf(
  is_color=False,
))
```
