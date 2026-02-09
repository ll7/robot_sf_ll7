"""Integration tests for multi-robot collision sensing wiring."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from robot_sf.common.types import Rect
from robot_sf.gym_env.env_util import create_spaces, init_collision_and_sensors
from robot_sf.gym_env.unified_config import RobotSimulationConfig
from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition
from robot_sf.robot.robot_state import RobotState


def _minimal_map_def() -> MapDefinition:
    """Create a map definition with valid routes for sensor/space initialization."""
    spawn_zone: Rect = ((1.0, 1.0), (2.0, 1.0), (2.0, 2.0))
    goal_zone: Rect = ((8.0, 8.0), (9.0, 8.0), (9.0, 9.0))
    bounds = [
        ((0.0, 0.0), (10.0, 0.0)),
        ((10.0, 0.0), (10.0, 10.0)),
        ((10.0, 10.0), (0.0, 10.0)),
        ((0.0, 10.0), (0.0, 0.0)),
    ]
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(1.5, 1.5), (8.5, 8.5)],
        spawn_zone=spawn_zone,
        goal_zone=goal_zone,
    )
    return MapDefinition(
        width=10.0,
        height=10.0,
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


class _FakeMultiRobotSim:
    """Minimal simulator surface used by env_util collision/sensor initialization."""

    def __init__(self, map_def: MapDefinition) -> None:
        self.map_def = map_def
        self.robots = [
            SimpleNamespace(pose=((1.0, 1.0), 0.0), current_speed=(0.0, 0.0)),
            SimpleNamespace(pose=((1.2, 1.0), 0.0), current_speed=(0.0, 0.0)),
        ]
        self._goal_pos = [(8.5, 8.5), (8.2, 8.4)]

    @property
    def robot_pos(self) -> list[tuple[float, float]]:
        return [robot.pose[0] for robot in self.robots]

    @property
    def goal_pos(self) -> list[tuple[float, float]]:
        return self._goal_pos

    @property
    def next_goal_pos(self) -> list[None]:
        return [None, None]

    @property
    def ped_pos(self) -> np.ndarray:
        return np.empty((0, 2), dtype=np.float64)

    def get_obstacle_lines(self) -> np.ndarray:
        return np.empty((0, 4), dtype=np.float64)


def test_multi_robot_collision_signal_and_dynamic_exclusion() -> None:
    """Two robots in close proximity should produce a robot-robot collision signal."""
    config = RobotSimulationConfig()
    map_def = _minimal_map_def()
    _action_space, _obs_space, orig_obs_space = create_spaces(config, map_def)
    sim = _FakeMultiRobotSim(map_def)
    occupancies, sensors = init_collision_and_sensors(sim, config, orig_obs_space)

    dynamic_for_robot_0 = occupancies[0].get_dynamic_objects()
    assert dynamic_for_robot_0 is not None
    assert len(dynamic_for_robot_0) == 1
    assert dynamic_for_robot_0[0][0] == sim.robot_pos[1]
    assert all(circle[0] != sim.robot_pos[0] for circle in dynamic_for_robot_0)

    state = RobotState(
        nav=SimpleNamespace(reached_waypoint=False, reached_destination=False),
        occupancy=occupancies[0],
        sensors=sensors[0],
        d_t=config.sim_config.time_per_step_in_secs,
        sim_time_limit=config.sim_config.sim_time_in_secs,
    )
    state.step()
    assert state.is_collision_with_robot is True
    assert state.meta_dict()["is_robot_collision"] is True
