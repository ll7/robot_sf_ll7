"""Scenario-domain benchmark helpers.

Submodules remain lazy so importing one scenario boundary helper does not pull in unrelated
benchmark or simulator dependencies. Legacy top-level module paths are retained as narrow
compatibility shims while callers migrate to this package.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [  # noqa: F822 - names are resolved lazily by __getattr__
    "scenario_coverage",
    "scenario_failure_cause",
    "scenario_schema",
    "scenario_staging",
]
_MODULE_NAMES = frozenset(__all__)


def __getattr__(name: str) -> Any:
    """Load a scenario helper submodule only when requested.

    Returns:
        The requested scenario helper module.

    Raises:
        AttributeError: If ``name`` is not a declared scenario helper.
    """

    if name not in _MODULE_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
