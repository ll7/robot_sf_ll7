"""
Simulator Facade: stable internal contract to construct simulator instances
from a backend key. Keeps env code decoupled from specific physics engines.

This module deliberately keeps the surface minimal. The returned simulator object
must satisfy the attributes used by existing env code (Simulator dataclass shape),
but the creation path is abstracted behind a registry.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # avoid runtime import cost
    from robot_sf.gym_env.env_config import EnvSettings
    from robot_sf.nav.map_config import MapDefinition

# Factory type for backends (env_config, map_def, peds_have_obstacle_forces) -> simulator-like
SimulatorFactory = Callable[["EnvSettings", "MapDefinition", bool], Any]
