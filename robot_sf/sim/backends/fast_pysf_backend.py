"""Fast-PySF backend adapter.

Exposes a simulator factory compatible with the backend registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # for type hints only
    from robot_sf.gym_env.env_config import EnvSettings
    from robot_sf.nav.map_config import MapDefinition


def fast_pysf_factory(env_config: EnvSettings, map_def: MapDefinition, peds: bool):
    """Create a simulator instance using the existing init_simulators helper.

    Parameters
    - env_config: environment settings
    - map_def: selected map definition
    - peds: whether pedestrians perceive robot as obstacle (interaction forces)
    """
    from robot_sf.sim.simulator import init_simulators

    return init_simulators(
        env_config, map_def, random_start_pos=True, peds_have_obstacle_forces=peds
    )[0]
