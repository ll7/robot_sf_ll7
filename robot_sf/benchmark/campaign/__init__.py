"""Campaign-domain admission, atlas, preflight, and logging helpers.

The package intentionally does not import its submodules eagerly. Campaign callers can import
the narrow helper they need without loading optional plotting or model dependencies, while the
legacy top-level module paths remain available through compatibility shims.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [  # noqa: F822 - names are resolved lazily by __getattr__
    "campaign_arm_admission",
    "campaign_atlas",
    "campaign_checkpoint_preflight",
    "campaign_logging",
    "campaign_runtime_preflight",
]
_MODULE_NAMES = tuple(__all__)


def __getattr__(name: str) -> Any:
    """Load a campaign submodule only when a caller requests it.

    Returns:
        The requested campaign submodule.

    Raises:
        AttributeError: If ``name`` is not a declared campaign submodule.
    """

    if name not in _MODULE_NAMES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f"{__name__}.{name}")
    globals()[name] = module
    return module
