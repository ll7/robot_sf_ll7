"""Reserved robot polygons stay excluded from crowd spawns and goals."""

import numpy as np

from robot_sf.ped_npc.ped_population import PedSpawnConfig, populate_simulation


def test_four_corner_reserved_zone_excludes_crowd_spawns_and_goals() -> None:
    """A caller-supplied rectangle must work without the three-corner map encoding."""
    np.random.seed(1001)
    states, _groups, behaviors = populate_simulation(
        tau=0.5,
        spawn_config=PedSpawnConfig(
            peds_per_area_m2=0.0,
            max_group_members=3,
            force_population_size=40,
            route_spawn_seed=1001,
        ),
        ped_routes=[],
        ped_crowded_zones=[((0.0, 0.0), (10.0, 0.0), (10.0, 10.0))],
        obstacle_polygons=[],
        reserved_zones=[((0.0, 0.0), (4.0, 0.0), (4.0, 10.0), (0.0, 10.0))],
        ped_radius=0.4,
        reserved_zone_radius=0.3,
    )

    assert states.num_peds == 40
    # The reserved rectangle spans the crowd's whole height; its inflated right
    # edge is x=4+0.4+0.3. Check samples and goals against that authored geometry.
    assert np.all(states.ped_positions[:, 0] > 4.7)
    assert all(states.goal_of(row)[0] > 4.7 for row in range(states.num_peds))
    assert behaviors
