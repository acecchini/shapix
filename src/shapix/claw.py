"""Import hook integration with ``beartype.claw``.

Since shapix's frame-based memo works automatically with ``@beartype``,
you can use ``beartype.claw`` directly. This module provides a convenience
wrapper for discoverability::

    # In your package's __init__.py:
    from shapix.claw import shapix_this_package

    shapix_this_package()

    # All subsequently imported submodules get @beartype automatically,
    # and shapix array annotations are checked with cross-arg consistency.
"""

from __future__ import annotations

from beartype import BeartypeConf
from beartype.claw import beartype_this_package as _beartype_this_package


def shapix_this_package(*, conf: BeartypeConf = BeartypeConf()) -> None:
  """Instrument the calling package with ``@beartype`` for runtime checking.

  This is a thin wrapper around ``beartype.claw.beartype_this_package``
  that exists as a semantic entry point for shapix users.
  """
  _beartype_this_package(conf=conf)
