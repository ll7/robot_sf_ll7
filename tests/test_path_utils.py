"""Tests for shortest-path utilities used by benchmark metrics."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import MultiPolygon, Polygon, box

from robot_sf.benchmark.path_utils import (
    compute_completion_reference_length,
    compute_shortest_path_length,
    remaining_route_length,
)
from robot_sf.nav.map_config import MapDefinition
from robot_sf.nav.obstacle import Obstacle


def _simple_map_def() -> MapDefinition:
    """Return a minimal MapDefinition with clear free space for planning."""
    width, height = 20.0, 20.0
    obstacles = []
    robot_spawn_zones = [((2, 2), (3, 2), (3, 3))]
    ped_spawn_zones = [((4, 4), (5, 4), (5, 5))]
    robot_goal_zones = [((17, 17), (18, 17), (18, 18))]
    bounds = [
        (0, width, 0, 0),
        (0, width, height, height),
        (0, 0, 0, height),
        (width, width, 0, height),
    ]
    ped_goal_zones = [((6, 6), (7, 6), (7, 7))]
    ped_crowded_zones: list = []
    robot_routes: list = []
    ped_routes: list = []
    single_pedestrians: list = []

    return MapDefinition(
        width,
        height,
        obstacles,
        robot_spawn_zones,
        ped_spawn_zones,
        robot_goal_zones,
        bounds,
        robot_routes,
        ped_goal_zones,
        ped_crowded_zones,
        ped_routes,
        single_pedestrians,
    )


def test_compute_shortest_path_length_returns_finite_on_clear_map() -> None:
    """Verify shortest-path length is finite on a simple clear map (metric stability)."""
    map_def = _simple_map_def()
    start = np.array([3.0, 3.0], dtype=float)
    goal = np.array([16.0, 16.0], dtype=float)

    length = compute_shortest_path_length(map_def, start, goal)

    assert np.isfinite(length)
    direct = float(np.linalg.norm(goal - start))
    assert length >= direct * 0.9
    assert length <= direct * 1.5


def test_remaining_route_arclength_preserves_retreat_and_duplicate_vertices() -> None:
    """A seven-metre L route has remaining lengths 6, 2, 3 after a retreat."""
    route = np.array([[0, 0], [3, 0], [3, 0], [3, 4]], dtype=float)
    positions = np.array([[1, 0], [3, 2], [3, 1]], dtype=float)
    np.testing.assert_allclose(remaining_route_length(positions, route), [6, 2, 3])
    np.testing.assert_array_equal(
        remaining_route_length(positions, np.array([[3, 4], [3, 4]])), [0, 0, 0]
    )


def test_zone_reference_uses_goal_set_and_reset_geometry() -> None:
    """Rectangle entry is three metres away, not five metres to the final waypoint."""
    map_def = _simple_map_def()
    # Three route-zone corners denote a parallelogram, not a triangle.
    zone = np.array([[5, 1], [7, 1], [7, 3]], dtype=float)
    kwargs = {
        "completion_policy": "goal_zone_entry_v1",
        "goal_zone": zone,
        "scenario_id": "dev-reference",
        "seed": 1001,
    }
    goal = np.array([7.0, 2.0])
    assert compute_completion_reference_length(map_def, np.array([2.0, 2.0]), goal, **kwargs) == 3
    assert compute_completion_reference_length(map_def, np.array([6.0, 2.0]), goal, **kwargs) == 0
    assert compute_completion_reference_length(map_def, np.array([3.0, 2.0]), goal, **kwargs) == 2
    kwargs["goal_zone"] = zone + [1, 0]
    assert compute_completion_reference_length(map_def, np.array([2.0, 2.0]), goal, **kwargs) == 4


def test_zone_reference_follows_obstacle_vertices_to_nearest_goal_edge() -> None:
    """The detour is sqrt(8) + 2 + sqrt(5), including compound obstacles."""
    map_def = _simple_map_def()
    map_def.obstacles = [Obstacle.from_geometry(MultiPolygon([box(4, 3, 6, 7), box(0, 0, 1, 1)]))]
    zone = np.array([[8, 4], [9, 4], [9, 6], [8, 6]], dtype=float)
    result = compute_completion_reference_length(
        map_def,
        np.array([2.0, 5.0]),
        np.array([9.0, 5.0]),
        completion_policy="goal_zone_entry_v1",
        goal_zone=zone,
        seed=1001,
    )
    # Shortest path: (2,5) -> (4,7) -> (6,7) -> (8,6).
    assert result == pytest.approx(8**0.5 + 2 + 5**0.5)


def test_zone_reference_respects_holes_and_unreachable_goal_sets() -> None:
    """Free-space inside a hole is usable, but a closed obstacle ring cannot be crossed."""
    map_def = _simple_map_def()
    ring = Polygon([(3, 3), (9, 3), (9, 9), (3, 9)], holes=[[(4, 4), (8, 4), (8, 8), (4, 8)]])
    map_def.obstacles = [Obstacle.from_geometry(ring)]
    zone = np.array([[5, 5], [6, 5], [6, 6], [5, 6]], dtype=float)
    kwargs = {"completion_policy": "goal_zone_entry_v1", "goal_zone": zone, "seed": 1001}
    assert compute_completion_reference_length(
        map_def,
        np.array([4.5, 4.5]),
        np.array([6.0, 6.0]),
        **kwargs,
    ) == pytest.approx(0.5**0.5)
    assert np.isnan(
        compute_completion_reference_length(
            map_def,
            np.array([2.0, 5.0]),
            np.array([6.0, 6.0]),
            **kwargs,
        )
    )


def test_zone_reference_refuses_invalid_or_blocked_completion_geometry() -> None:
    """Undefined completion references remain NaN rather than plausible point distances."""
    map_def = _simple_map_def()
    map_def.obstacles = [Obstacle.from_geometry(box(4, 4, 7, 7))]
    valid = np.array([[8, 4], [9, 4], [9, 6], [8, 6]], dtype=float)
    blocked = np.array([[5, 5], [6, 5], [6, 6], [5, 6]], dtype=float)
    crossed = np.array([[8, 4], [9, 6], [8, 6], [9, 4]], dtype=float)
    for map_arg, start, zone in (
        (None, [2, 2], valid),
        (map_def, [2, 2], None),
        (map_def, [np.nan, 2], valid),
        (map_def, [2, 2], valid * np.nan),
        (map_def, [2, 2], crossed),
        (map_def, [-1, 2], valid),
        (map_def, [5, 5], valid),
        (map_def, [2, 2], blocked),
    ):
        assert np.isnan(
            compute_completion_reference_length(
                map_arg,
                np.asarray(start, dtype=float),
                np.array([9.0, 5.0]),
                completion_policy="goal_zone_entry_v1",
                goal_zone=zone,
                seed=1001,
            )
        )
