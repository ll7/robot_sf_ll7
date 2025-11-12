"""Helpers for loading and validating the examples manifest.

This package exposes data structures and utilities that power documentation
rendering and automated smoke tests for the ``examples/`` directory.

When this module is imported as the top-level ``examples`` package (because the
module search path happens to include ``robot_sf/`` ahead of the repository
root), we transparently delegate to the distributable examples package located
next to ``robot_sf``. This keeps the legacy ``examples.*`` imports working even
when ``robot_sf/examples`` would have shadowed them during test collection.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

if __name__ == "examples":
    _distribution_root = Path(__file__).resolve().parents[2]
    _shim_target = _distribution_root / "examples" / "__init__.py"
    if not _shim_target.exists():  # pragma: no cover - defensive guard
        raise ModuleNotFoundError("examples")
    _spec = importlib.util.spec_from_file_location("examples", _shim_target)
    if _spec is None or _spec.loader is None:  # pragma: no cover - defensive guard
        raise ModuleNotFoundError("examples")
    _module = importlib.util.module_from_spec(_spec)
    # Ensure the module gets registered under the canonical name before exec.
    sys.modules[__name__] = _module
    _spec.loader.exec_module(_module)
    globals().update(_module.__dict__)
    __all__ = tuple(getattr(_module, "__all__", ()))
else:
    from .manifest_loader import (
        ExampleCategory,
        ExampleManifest,
        ExampleScript,
        ManifestValidationError,
        load_manifest,
    )

    __all__ = [
        "ExampleCategory",
        "ExampleManifest",
        "ExampleScript",
        "ManifestValidationError",
        "load_manifest",
    ]
