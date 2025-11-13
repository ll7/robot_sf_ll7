"""Top-level package for distributable robot_sf examples and shims.

Legacy paths under ``examples.*`` continue working by registering aliases that
point to their new locations. This allows downstream consumers and tests to
keep the original import strings even after the directory reorganization.
"""

from __future__ import annotations

import importlib
import sys
from typing import Final

__all__: tuple[str, ...] = ()

_LEGACY_ALIASES: Final[dict[str, str]] = {
    "examples.classic_interactions_pygame": "examples._archived.classic_interactions_pygame",
}


def _register_legacy_aliases() -> None:
    """Register compatibility aliases for moved example modules."""

    for alias, target in _LEGACY_ALIASES.items():
        if alias in sys.modules:
            continue
        try:
            sys.modules[alias] = importlib.import_module(target)
        except ModuleNotFoundError:
            # Skip silently if the archival module is missing; import will fail
            # later with a normal ModuleNotFoundError referencing ``target``.
            continue


_register_legacy_aliases()
