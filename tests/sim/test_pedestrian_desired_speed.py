"""End-to-end tests for decoupled pedestrian desired speed (issue #4972).

Verifies the decoupling reaches ``peds.max_speeds`` through the real
``init_simulators`` construction path, and that the legacy default is preserved.
"""

from __future__ import annotations

import numpy as np

from robot_sf.common.types import Line2D, Rect
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition, MapDefinitionPool
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.sim.simulator import init_simulators


def _minimal_map() -> MapDefinition:
    """Build a compact map with one robot route and a pedestrian spawn zone."""
    width = 20.0
    height = 20.0
    spawn_zone: Rect = ((1.0, 1.0), (2.0, 1.0), (1.0, 2.0))
    goal_zone: Rect = ((16.0, 16.0), (17.0, 16.0), (16.0, 17.0))
    bounds: list[Line2D] = [
        ((0.0, 0.0), (width, 0.0)),
        ((width, 0.0), (width, height)),
        ((width, height), (0.0, height)),
        ((0.0, height), (0.0, 0.0)),
    ]
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(1.2, 1.2), (16.8, 16.8)],
        spawn_zone=spawn_zone,
        goal_zone=goal_zone,
    )
    return MapDefinition(
        width=width,
        height=height,
        obstacles=[],
        robot_spawn_zones=[spawn_zone],
        ped_spawn_zones=[spawn_zone],
        robot_goal_zones=[goal_zone],
        bounds=bounds,
        robot_routes=[route],
        ped_goal_zones=[goal_zone],
        ped_crowded_zones=[],
        ped_routes=[route],
        single_pedestrians=[],
    )


def _build_simulator(sim_config: SimulationSettings):
    """Construct a single robot simulator for the minimal map."""
    map_def = _minimal_map()
    config = RobotSimulationConfig(
        map_pool=MapDefinitionPool(map_defs={"test": map_def}),
        sim_config=sim_config,
    )
    return init_simulators(
        config,
        map_def,
        num_robots=1,
        random_start_pos=False,
        peds_have_obstacle_forces=True,
    )[0]


def test_default_simulator_keeps_legacy_slow_regime():
    """Default config must leave peds at the ~0.65 m/s spawn-coupled speed."""
    sim = _build_simulator(
        SimulationSettings(
            difficulty=0,
            ped_density_by_difficulty=[0.04],
            population_size=8,
        )
    )
    peds = sim.pysf_sim.peds
    # initial_speed=0.5 default, multiplier 1.3 -> 0.65 m/s for every pedestrian.
    np.testing.assert_allclose(peds.max_speeds, np.full(peds.size(), 0.65))


def test_typical_tier_decouples_desired_speed_from_spawn():
    """ped_speed_tier='typical' should drive peds toward ~1.3 m/s, not 0.65 m/s."""
    sim = _build_simulator(
        SimulationSettings(
            difficulty=0,
            ped_density_by_difficulty=[0.04],
            population_size=200,
            ped_speed_tier="typical",
            desired_speed_seed=123,
        )
    )
    peds = sim.pysf_sim.peds
    # Decoupled: the mean desired speed must be near 1.3, well above the 0.65 legacy.
    assert float(np.mean(peds.max_speeds)) > 1.0
    assert abs(float(np.mean(peds.max_speeds)) - 1.3) < 0.15
    # No pedestrian should be stuck at the legacy 0.65 m/s spawn-coupled value.
    assert not np.any(np.isclose(peds.max_speeds, 0.65))


def test_explicit_desired_speed_decouples_deterministically():
    """Explicit desired_speed_mean/std=0 pins every ped to the configured speed."""
    sim = _build_simulator(
        SimulationSettings(
            difficulty=0,
            ped_density_by_difficulty=[0.04],
            population_size=12,
            desired_speed_mean=1.5,
            desired_speed_std=0.0,
            desired_speed_seed=0,
        )
    )
    peds = sim.pysf_sim.peds
    np.testing.assert_allclose(peds.max_speeds, np.full(peds.size(), 1.5))
