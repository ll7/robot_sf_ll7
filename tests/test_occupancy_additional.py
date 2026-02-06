"""Additional tests for occupancy collision helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from robot_sf.nav.global_route import GlobalRoute
from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.obstacle import Obstacle
from robot_sf.nav.occupancy import (
    ContinuousOccupancy,
    EgoPedContinuousOccupancy,
    check_quality_of_map_point,
    circle_collides_any,
    circle_collides_any_lines,
    is_circle_circle_intersection,
    is_circle_line_intersection,
)

if TYPE_CHECKING:
    from robot_sf.common.types import Rect


def _make_map(*, obstacles, bounds) -> MapDefinition:
    """Create a minimal map definition for occupancy tests."""
    width = 2.0
    height = 2.0
    spawn_zone: Rect = ((0.1, 0.1), (0.2, 0.1), (0.2, 0.2))
    goal_zone: Rect = ((1.6, 1.6), (1.8, 1.6), (1.8, 1.8))
    route = GlobalRoute(
        spawn_id=0,
        goal_id=0,
        waypoints=[(0.2, 0.2), (1.8, 1.8)],
        spawn_zone=spawn_zone,
        goal_zone=goal_zone,
    )
    return MapDefinition(
        width=width,
        height=height,
        obstacles=obstacles,
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


def test_continuous_occupancy_goal_and_bounds() -> None:
    """Validate goal proximity and in-bounds checks for continuous occupancy."""
    occ = ContinuousOccupancy(
        width=2.0,
        height=2.0,
        get_agent_coords=lambda: (1.0, 1.0),
        get_goal_coords=lambda: (1.0, 1.0),
        get_obstacle_coords=lambda: np.zeros((0, 4)),
        get_pedestrian_coords=lambda: np.zeros((0, 2)),
        agent_radius=0.2,
        goal_radius=0.3,
    )
    assert occ.is_robot_at_goal is True
    assert occ.is_in_bounds(1.0, 1.0) is True
    assert occ.is_in_bounds(-0.1, 1.0) is False


def test_ego_ped_distance_and_collision() -> None:
    """Compute ego-ped distance and collision against an enemy agent."""
    occ = EgoPedContinuousOccupancy(
        width=2.0,
        height=2.0,
        get_agent_coords=lambda: (0.5, 0.0),
        get_goal_coords=lambda: (1.5, 0.0),
        get_obstacle_coords=lambda: np.zeros((0, 4)),
        get_pedestrian_coords=lambda: np.zeros((0, 2)),
        agent_radius=1.0,
        ped_radius=0.4,
        goal_radius=0.2,
        get_enemy_coords=lambda: (0.0, 0.0),
        enemy_radius=1.0,
    )
    assert occ.enemy_coords == (0.0, 0.0)
    assert occ.distance_to_robot == 0.5
    assert occ.is_agent_agent_collision is True


def test_circle_intersection_helpers_cover_branches() -> None:
    """Exercise circle-circle and circle-line intersection branches."""
    assert is_circle_circle_intersection(((0.0, 0.0), 1.0), ((1.5, 0.0), 0.6)) is True
    assert is_circle_circle_intersection(((0.0, 0.0), 0.5), ((2.0, 0.0), 0.4)) is False

    circle = ((0.0, 0.0), 1.0)
    assert is_circle_line_intersection(circle, ((0.5, 0.0), (2.0, 0.0))) is True
    assert is_circle_line_intersection(circle, ((-2.0, 0.0), (2.0, 0.0))) is True
    assert is_circle_line_intersection(circle, ((2.0, 2.0), (3.0, 3.0))) is False

    assert circle_collides_any(((0.0, 0.0), 0.5), [((2.0, 0.0), 0.4), ((0.4, 0.0), 0.2)]) is True
    assert circle_collides_any(((0.0, 0.0), 0.5), [((2.0, 0.0), 0.4)]) is False


def test_circle_collides_any_lines_accepts_array_and_flat_segments() -> None:
    """Support numpy arrays and flat tuples when checking line collisions."""
    circle = ((0.0, 0.0), 1.0)
    segments = np.array([[-2.0, 0.0, 2.0, 0.0]])
    assert circle_collides_any_lines(circle, segments) is True

    flat_segments = [(-2.0, 0.0, 2.0, 0.0)]
    assert circle_collides_any_lines(circle, flat_segments) is True

    invalid_segments = ["bad-segment"]
    assert circle_collides_any_lines(circle, invalid_segments) is False


def test_continuous_occupancy_dynamic_and_pedestrian_collision() -> None:
    """Check dynamic object and pedestrian collision helpers."""
    occ = ContinuousOccupancy(
        width=2.0,
        height=2.0,
        get_agent_coords=lambda: (0.0, 0.0),
        get_goal_coords=lambda: (1.0, 1.0),
        get_obstacle_coords=lambda: np.zeros((0, 4)),
        get_pedestrian_coords=lambda: np.array([[0.6, 0.0]]),
        get_dynamic_objects=lambda: [((0.3, 0.0), 0.4)],
        agent_radius=0.4,
        ped_radius=0.4,
        goal_radius=0.2,
    )
    assert occ.is_pedestrian_collision is True
    assert occ.is_dynamic_collision is True

    occ_no_dynamic = ContinuousOccupancy(
        width=2.0,
        height=2.0,
        get_agent_coords=lambda: (0.0, 0.0),
        get_goal_coords=lambda: (1.0, 1.0),
        get_obstacle_coords=lambda: np.zeros((0, 4)),
        get_pedestrian_coords=lambda: np.zeros((0, 2)),
        agent_radius=0.4,
        ped_radius=0.4,
        goal_radius=0.2,
    )
    assert occ_no_dynamic.is_dynamic_collision is False


def test_continuous_occupancy_obstacle_collision_detects_lines() -> None:
    """Obstacle collision uses obstacle line segments from the callback."""
    occ = ContinuousOccupancy(
        width=2.0,
        height=2.0,
        get_agent_coords=lambda: (0.0, 0.0),
        get_goal_coords=lambda: (1.0, 1.0),
        get_obstacle_coords=lambda: np.array([[-2.0, 0.0, 2.0, 0.0]]),
        get_pedestrian_coords=lambda: np.zeros((0, 2)),
        agent_radius=0.5,
        ped_radius=0.4,
        goal_radius=0.2,
    )
    assert occ.is_obstacle_collision is True


def test_check_quality_of_map_point_flags_obstacles() -> None:
    """Reject map points that intersect obstacle/boundary lines."""
    bounds = [
        ((0.0, 0.0), (2.0, 0.0)),
        ((2.0, 0.0), (2.0, 2.0)),
        ((2.0, 2.0), (0.0, 2.0)),
        ((0.0, 2.0), (0.0, 0.0)),
    ]
    obstacle = Obstacle([(0.8, 0.8), (1.2, 0.8), (1.2, 1.2), (0.8, 1.2)])
    map_def = _make_map(obstacles=[obstacle], bounds=bounds)
    assert check_quality_of_map_point(map_def, (1.0, 1.0), radius=0.3) is False
    assert check_quality_of_map_point(map_def, (0.2, 1.8), radius=0.05) is True
    assert check_quality_of_map_point(map_def, (3.0, 3.0), radius=0.1) is False
    assert check_quality_of_map_point(map_def, (0.1, 0.1), radius=0.0) is True
