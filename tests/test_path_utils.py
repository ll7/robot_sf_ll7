"""Tests for shortest-path utilities used by benchmark metrics."""

from __future__ import annotations

import numpy as np

from robot_sf.benchmark.path_utils import compute_shortest_path_length
from robot_sf.nav.map_config import MapDefinition


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
