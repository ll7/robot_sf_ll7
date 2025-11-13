"""Compatibility shim for archived classic interactions demo.

This module preserves the historic import path ``examples.classic_interactions_pygame``
by delegating to the archival location under ``examples._archived``.
"""

from importlib import import_module as _import_module

_ARCHIVED_MODULE = _import_module("examples._archived.classic_interactions_pygame")

globals().update(vars(_ARCHIVED_MODULE))

if hasattr(_ARCHIVED_MODULE, "__all__"):
    __all__ = tuple(_ARCHIVED_MODULE.__all__)  # type: ignore[attr-defined]
else:
    __all__ = tuple(name for name in globals() if not name.startswith("_"))

del _ARCHIVED_MODULE
del _import_module
