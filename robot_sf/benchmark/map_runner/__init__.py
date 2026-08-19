"""Map-runner domain helpers for benchmark orchestration.

The package intentionally does not import its submodules eagerly. Map-runner callers can import
the narrow helper they need without loading optional plotting, model, or simulator dependencies,
while the legacy top-level module paths remain available through compatibility shims.

The bare-name module ``map_runner.py`` is unique among the moved modules: its legacy path
``robot_sf.benchmark.map_runner`` now resolves to this package (a package directory always
shadows a same-named module file), so attribute lookups that previously hit the module's public
surface are delegated to :mod:`robot_sf.benchmark.map_runner.map_runner`.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [  # noqa: F822 - names are resolved lazily by __getattr__
    "map_runner",
    "map_runner_batch_plan",
    "map_runner_batch_runner",
    "map_runner_batch_summary",
    "map_runner_env",
    "map_runner_episode",
    "map_runner_identity",
    "map_runner_jsonl",
    "map_runner_metrics",
    "map_runner_native_command",
    "map_runner_observations",
    "map_runner_provenance",
    "map_runner_static_deadlock",
    "map_runner_trace",
    "map_runner_view_integrity",
    "map_runner_worker",
]
_MODULE_NAMES = tuple(__all__)
_PUBLIC_DELEGATE = "map_runner"


def __getattr__(name: str) -> Any:
    """Load a map-runner submodule, or delegate to the core module surface.

    Args:
        name: Requested attribute name.

    Returns:
        The requested submodule or the matching public attribute of
        :mod:`robot_sf.benchmark.map_runner.map_runner`.

    Raises:
        AttributeError: If ``name`` is neither a declared submodule nor a public name of the
            core map-runner module.
    """

    if name in _MODULE_NAMES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    core = import_module(f"{__name__}.{_PUBLIC_DELEGATE}")
    value = getattr(core, name)
    globals()[name] = value
    return value
