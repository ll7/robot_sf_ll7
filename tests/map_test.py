"""Map occupancy and simulator obstacle collision tests."""

from math import dist, pi
from pathlib import Path

import numpy as np
import pytest

from robot_sf.gym_env.env_config import PedEnvSettings
from robot_sf.nav.occupancy import ContinuousOccupancy, EgoPedContinuousOccupancy
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.sim.simulator import init_ped_simulators

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEBUG_SVG_MAP = _REPO_ROOT / "maps" / "svg_maps" / "debug_06.svg"


def _debug_ped_simulator():
    """Build the debug SVG pedestrian simulator used by map obstacle tests."""
    env_config = PedEnvSettings()
    map_def = convert_map(str(_DEBUG_SVG_MAP))
    return init_ped_simulators(env_config, map_def)[0]


def test_create_map():
    """TODO docstring. Document this function."""
    _map = ContinuousOccupancy(
        10,
        10,
        lambda: None,
        lambda: None,
        lambda: np.array([[]]),
        lambda: np.array([[]]),
    )
    assert _map is not None


def test_is_in_bounds():
    """TODO docstring. Document this function."""
    _map = ContinuousOccupancy(
        10,
        10,
        lambda: None,
        lambda: None,
        lambda: np.array([[]]),
        lambda: np.array([[]]),
    )
    assert _map.is_in_bounds(0, 0)
    assert _map.is_in_bounds(10, 0)
    assert _map.is_in_bounds(0, 10)
    assert _map.is_in_bounds(10, 10)
    assert _map.is_in_bounds(0, 0)
    assert _map.is_in_bounds(5, 5)


def test_is_out_of_bounds():
    """TODO docstring. Document this function."""
    _map = ContinuousOccupancy(
        10,
        10,
        lambda: None,
        lambda: None,
        lambda: np.array([]),
        lambda: np.array([[]]),
    )
    assert not _map.is_in_bounds(10.0000001, 10)
    assert not _map.is_in_bounds(10, 10.0000001)
    assert not _map.is_in_bounds(-10.0000001, 10)
    assert not _map.is_in_bounds(10, -10.0000001)


def test_is_collision_with_obstacle_segment_fully_contained_inside_circle():
    """TODO docstring. Document this function."""
    obstacle_pos = np.random.uniform(-10, 10, size=(1, 4))
    robot_pos = (obstacle_pos[0, 0], obstacle_pos[0, 1])
    _map = ContinuousOccupancy(
        10,
        10,
        lambda: robot_pos,
        lambda: None,
        lambda: obstacle_pos,
        lambda: np.array([[20, 20]]),
        agent_radius=2,
    )
    assert _map.is_obstacle_collision


def test_is_collision_with_obstacle_segment_outside_circle():
    """TODO docstring. Document this function."""
    obstacle_pos = np.random.uniform(-10, 10, size=(1, 4))
    middle = np.squeeze((obstacle_pos[0::2] + obstacle_pos[0:2:]) / 2)
    robot_pos = (middle[0], middle[1])
    radius = dist(obstacle_pos[0, :2], obstacle_pos[0, 2:]) / 2.1
    _map = ContinuousOccupancy(
        10,
        10,
        lambda: robot_pos,
        lambda: None,
        lambda: obstacle_pos,
        lambda: np.array([[20, 20]]),
        radius,
    )
    assert _map.is_obstacle_collision


def test_is_collision_with_pedestrian():
    """TODO docstring. Document this function."""
    ped_pos = np.random.uniform(-10, 10, size=2)
    robot_pos = (ped_pos[0], ped_pos[1])
    _map = ContinuousOccupancy(
        40,
        40,
        lambda: robot_pos,
        lambda: None,
        lambda: np.array([[]]),
        lambda: np.array([ped_pos]),
        agent_radius=2,
    )
    assert _map.is_pedestrian_collision


def test_is_collision_with_agent():
    """TODO docstring. Document this function."""
    agent_pos = np.random.uniform(-10, 10, size=2)
    enemy_pos = (agent_pos[0], agent_pos[1])
    _map = EgoPedContinuousOccupancy(
        40,
        40,
        lambda: agent_pos,
        lambda: None,
        lambda: np.array([[]]),
        lambda: np.array([]),
        agent_radius=1,
        ped_radius=0.4,
        goal_radius=1.0,
        get_enemy_coords=lambda: enemy_pos,
        enemy_radius=1.0,
    )
    assert _map.is_agent_agent_collision


def test_proximity_point():
    """TODO docstring. Document this function."""
    fixed_point = (50, 50)
    lower_bound = 15
    upper_bound = 20
    env_config = PedEnvSettings()
    map_def = convert_map(str(_DEBUG_SVG_MAP))
    _sim = init_ped_simulators(env_config, map_def)[0]
    new_point = _sim.get_proximity_point(
        fixed_point,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )
    distance = np.linalg.norm(np.array(fixed_point) - np.array(new_point))
    assert lower_bound <= distance <= upper_bound
    # TODO: Add test to check with obstacle


def test_ped_simulator_detects_svg_obstacle_collision():
    """Map-derived obstacle lines should reject positions intersecting static obstacles."""
    sim = _debug_ped_simulator()
    obstacle_lines = sim.get_obstacle_lines()
    assert obstacle_lines.shape[0] > 0

    first_line = obstacle_lines[0]
    midpoint = (
        float((first_line[0] + first_line[2]) / 2.0),
        float((first_line[1] + first_line[3]) / 2.0),
    )

    assert sim.is_obstacle_collision(*midpoint)
    assert not sim.is_obstacle_collision(6.0, 1.0)


def test_proximity_point_resamples_when_candidate_hits_obstacle(monkeypatch):
    """Proximity sampling should skip obstacle-colliding candidates before returning."""
    sim = _debug_ped_simulator()
    fixed_point = (5.0, 1.0)
    draws = iter([pi / 2.0, 3.0, 0.0, 1.0])

    monkeypatch.setattr("robot_sf.sim.simulator.uniform", lambda _low, _high: next(draws))

    point = sim.get_proximity_point(fixed_point, lower_bound=1.0, upper_bound=3.0)

    assert point == pytest.approx((6.0, 1.0))
    assert not sim.is_obstacle_collision(*point)


def test_simulator_obstacle_lines_helpers():
    """Validate simulator obstacle helpers return stable shapes and types."""
    env_config = PedEnvSettings()
    map_def = convert_map(str(_DEBUG_SVG_MAP))
    sim = init_ped_simulators(env_config, map_def)[0]

    lines = sim.get_obstacle_lines()
    assert lines.ndim == 2
    assert lines.shape[1] == 4

    segments = sim.iter_obstacle_segments()
    assert len(segments) == lines.shape[0]
    for segment in segments:
        assert isinstance(segment, tuple)
        assert len(segment) == 2
        for point in segment:
            assert isinstance(point, tuple)
            assert len(point) == 2
