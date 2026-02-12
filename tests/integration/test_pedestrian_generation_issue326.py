"""Integration regressions for issue #326 pedestrian generation behavior."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from robot_sf.common.types import Rect
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition, SinglePedestrianDefinition
from robot_sf.nav.obstacle import Obstacle
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.sim.sim_config import SimulationSettings
from robot_sf.sim.simulator import Simulator, init_simulators


def _build_simulator_for_map(svg_map: str) -> Simulator:
    """Create a deterministic simulator for an SVG map with route pedestrians enabled."""
    map_def = convert_map(svg_map)
    sim_config = SimulationSettings(
        difficulty=0,
        ped_density_by_difficulty=[0.06],
        route_spawn_distribution="spread",
        route_spawn_seed=21,
        route_spawn_jitter_frac=0.05,
    )
    cfg = RobotSimulationConfig(sim_config=sim_config)
    return init_simulators(
        cfg,
        map_def,
        num_robots=1,
        random_start_pos=False,
        peds_have_obstacle_forces=True,
    )[0]


@pytest.mark.parametrize(
    "svg_map",
    [
        "maps/svg_maps/classic_crossing.svg",
        "maps/svg_maps/classic_doorway.svg",
        "maps/svg_maps/classic_bottleneck.svg",
    ],
)
def test_issue_326_spawn_count_nonzero_on_representative_maps(svg_map: str) -> None:
    """Pedestrian routes on representative maps should produce a non-zero population."""
    sim = _build_simulator_for_map(svg_map)
    assert len(sim.map_def.ped_routes) > 0
    assert sim.ped_pos.shape[0] > 0


def test_issue_326_no_unexpected_vanish_or_nan_over_long_horizon() -> None:
    """Pedestrians should keep a stable count and finite positions during route following."""
    sim = _build_simulator_for_map("maps/svg_maps/debug_06.svg")
    initial_count = int(sim.ped_pos.shape[0])
    assert initial_count > 0
    x_margin = max(2.0, sim.config.ped_radius * 6)
    y_margin = x_margin
    actions = [(0.0, 0.0)]

    for _ in range(120):
        sim.step_once(actions)
        ped_pos = sim.ped_pos
        assert int(ped_pos.shape[0]) == initial_count
        assert np.isfinite(ped_pos).all()
        assert np.all(ped_pos[:, 0] >= -x_margin)
        assert np.all(ped_pos[:, 0] <= sim.map_def.width + x_margin)
        assert np.all(ped_pos[:, 1] >= -y_margin)
        assert np.all(ped_pos[:, 1] <= sim.map_def.height + y_margin)


def _build_obstacle_interaction_map() -> MapDefinition:
    """Build a compact map with one single pedestrian moving through a central obstacle."""
    spawn_zone: Rect = ((0.5, 0.5), (1.5, 0.5), (1.5, 1.5))
    goal_zone: Rect = ((8.5, 8.5), (9.5, 8.5), (9.5, 9.5))
    robot_route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(1.0, 1.0), (9.0, 9.0)],
        spawn_zone=spawn_zone,
        goal_zone=goal_zone,
    )
    obstacle = Obstacle([(4.0, 4.0), (6.0, 4.0), (6.0, 6.0), (4.0, 6.0)])
    single_ped = SinglePedestrianDefinition(
        id="issue_326_single",
        start=(2.0, 5.0),
        goal=(8.0, 5.0),
        speed_m_s=0.9,
    )
    return MapDefinition(
        width=10.0,
        height=10.0,
        obstacles=[obstacle],
        robot_spawn_zones=[spawn_zone],
        ped_spawn_zones=[spawn_zone],
        robot_goal_zones=[goal_zone],
        bounds=[
            ((0.0, 0.0), (10.0, 0.0)),
            ((10.0, 0.0), (10.0, 10.0)),
            ((10.0, 10.0), (0.0, 10.0)),
            ((0.0, 10.0), (0.0, 0.0)),
        ],
        robot_routes=[robot_route],
        ped_goal_zones=[goal_zone],
        ped_crowded_zones=[],
        ped_routes=[],
        single_pedestrians=[single_ped],
    )


def _count_steps_inside_obstacle(peds_have_obstacle_forces: bool) -> int:
    """Count timesteps where the single pedestrian enters the obstacle interior."""
    map_def = _build_obstacle_interaction_map()
    sim_config = SimulationSettings(
        difficulty=0, ped_density_by_difficulty=[0.0], max_peds_per_group=1
    )
    cfg = RobotSimulationConfig(
        sim_config=replace(sim_config),
        peds_have_static_obstacle_forces=peds_have_obstacle_forces,
    )
    sim = init_simulators(
        cfg,
        map_def,
        num_robots=1,
        random_start_pos=False,
        peds_have_obstacle_forces=peds_have_obstacle_forces,
    )[0]
    obstacle = map_def.obstacles[0]
    actions = [(0.0, 0.0)]
    inside_steps = 0
    for _ in range(160):
        sim.step_once(actions)
        ped = sim.ped_pos[0]
        if obstacle.contains_point((float(ped[0]), float(ped[1]))):
            inside_steps += 1
    return inside_steps


def test_issue_326_obstacle_forces_reduce_obstacle_interior_penetration() -> None:
    """Obstacle forces should reduce obstacle-interior penetration over repeated steps."""
    inside_without_forces = _count_steps_inside_obstacle(peds_have_obstacle_forces=False)
    inside_with_forces = _count_steps_inside_obstacle(peds_have_obstacle_forces=True)

    assert inside_without_forces > 0
    assert inside_with_forces < inside_without_forces
