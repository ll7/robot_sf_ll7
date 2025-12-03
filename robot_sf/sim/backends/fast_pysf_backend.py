"""Fast-PySF backend adapter.

Exposes a simulator factory compatible with the backend registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # for type hints only
    from robot_sf.gym_env.env_config import EnvSettings
    from robot_sf.nav.map_config import MapDefinition

from robot_sf.sim.simulator import init_simulators


def fast_pysf_factory(env_config: EnvSettings, map_def: MapDefinition, peds: bool):
    """Create a Fast-PySF simulator using ``init_simulators``.

    Args:
        env_config: Environment configuration passed to the simulator factory.
        map_def: Map definition describing geometry and obstacles.
        peds: Whether pedestrians exert obstacle forces (forwarded to simulator init).

    Returns:
        Simulator: Configured simulator instance.
    """
    return init_simulators(
        env_config, map_def, random_start_pos=True, peds_have_obstacle_forces=peds
    )[0]
