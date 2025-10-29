"""Backend registry for simulator creation.

Provides register/get/list helpers and registers the default "fast-pysf" backend.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:  # avoid runtime imports
    from robot_sf.gym_env.env_config import EnvSettings
    from robot_sf.nav.map_config import MapDefinition
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


def _fast_pysf_factory(env_config: EnvSettings, map_def: MapDefinition, peds: bool):
    from robot_sf.sim.simulator import init_simulators

    return init_simulators(
        env_config, map_def, random_start_pos=True, peds_have_obstacle_forces=peds
    )[0]


# Register default on import
try:
    register_backend("fast-pysf", _fast_pysf_factory, override=True)
except (ValueError, KeyError) as _e:  # pragma: no cover - defensive
    logger.warning("Failed to register default fast-pysf backend: {}", _e)
