"""Backend registry for simulator creation.

Provides register/get/list helpers and registers the default "fast-pysf" backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:  # avoid runtime imports
    from robot_sf.sim.facade import SimulatorFactory

_REGISTRY: dict[str, SimulatorFactory] = {}


def register_backend(key: str, factory: SimulatorFactory, *, override: bool = False) -> None:
    k = key.strip()
    if not k:
        raise ValueError("Backend key must be a non-empty string")
    if k in _REGISTRY and not override:
        raise KeyError(f"Backend '{k}' already registered; pass override=True to replace")
    _REGISTRY[k] = factory
    logger.info("Registered simulator backend: {}", k)


def get_backend(key: str) -> SimulatorFactory:
    try:
        return _REGISTRY[key]
    except KeyError as e:  # provide suggestions
        known = ", ".join(sorted(_REGISTRY.keys())) or "<none>"
        raise KeyError(f"Unknown backend '{key}'. Known: {known}") from e


def list_backends() -> list[str]:
    return sorted(_REGISTRY.keys())


# Default backend registration (fast-pysf)
# Register default backends on import
try:
    from robot_sf.sim.backends.fast_pysf_backend import fast_pysf_factory

    register_backend("fast-pysf", fast_pysf_factory, override=True)
except (ValueError, KeyError) as _e:  # pragma: no cover - defensive
    logger.warning("Failed to register default fast-pysf backend: {}", _e)

try:
    from robot_sf.sim.backends.dummy_backend import dummy_factory

    register_backend("dummy", dummy_factory, override=True)
except (ValueError, KeyError) as _e:  # pragma: no cover - defensive
    logger.warning("Failed to register dummy backend: {}", _e)
