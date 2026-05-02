"""Tests for the experimental occupancy-grid route planner."""

from __future__ import annotations

import numpy as np

from robot_sf.planner.grid_route import (
    GridRoutePlannerAdapter,
    GridRoutePlannerConfig,
    build_grid_route_config,
)


def _observation(
    *,
    goal: tuple[float, float] = (2.0, 0.0),
    heading: float = 0.0,
    radius: float = 0.3,
    occupied_cells: list[tuple[int, int]] | None = None,
) -> dict[str, object]:
    grid = np.zeros((3, 21, 21), dtype=float)
    for row, col in occupied_cells or []:
        grid[0, row, col] = 1.0
        grid[2, row, col] = 1.0
    return {
        "robot": {
            "position": [0.0, 0.0],
            "heading": [heading],
            "speed": [0.0],
            "radius": [radius],
        },
        "goal": {"current": list(goal), "next": list(goal)},
        "pedestrians": {"positions": [], "velocities": [], "count": [0], "radius": [0.3]},
        "occupancy_grid": grid,
        "occupancy_grid_meta": {
            "origin": [-2.0, -2.0],
            "resolution": [0.2],
            "size": [4.2, 4.2],
            "use_ego_frame": [0.0],
            "center_on_robot": [0.0],
            "channel_indices": [0, 1, 2],
            "robot_pose": [0.0, 0.0, heading],
        },
    }


def test_grid_route_build_config_defaults() -> None:
    """Config builder should parse defaults and selected overrides safely."""
    cfg = build_grid_route_config({"max_linear_speed": 0.7, "waypoint_lookahead_cells": 3})
    assert cfg.max_linear_speed == 0.7
    assert cfg.waypoint_lookahead_cells == 3
    assert build_grid_route_config(None).goal_tolerance == 0.25


def test_grid_route_returns_bounded_open_space_command() -> None:
    """Open-space commands should remain finite and within configured bounds."""
    planner = GridRoutePlannerAdapter(
        GridRoutePlannerConfig(max_linear_speed=0.6, max_angular_speed=0.8)
    )
    linear, angular = planner.plan(_observation())
    assert 0.0 <= linear <= 0.6
    assert abs(angular) <= 0.8


def test_grid_route_stops_at_goal() -> None:
    """Planner should stop when the current goal is already satisfied."""
    planner = GridRoutePlannerAdapter(GridRoutePlannerConfig(goal_tolerance=0.3))
    assert planner.plan(_observation(goal=(0.1, 0.0))) == (0.0, 0.0)


def test_grid_route_astar_detours_around_blocked_centerline() -> None:
    """A* routing should leave the centerline when the direct corridor is blocked."""
    planner = GridRoutePlannerAdapter(GridRoutePlannerConfig(obstacle_inflation_cells=0))
    observation = _observation(
        occupied_cells=[(10, 11), (10, 12), (10, 13), (9, 12), (11, 12)],
    )
    robot_pos, _heading, goal, radius = planner._extract_state(observation)
    grid, meta = planner._extract_grid_payload(observation) or (None, None)
    assert grid is not None
    assert meta is not None

    blocked = planner._blocked_grid(grid, meta, radius)
    assert blocked is not None
    start = planner._world_to_grid(robot_pos, meta, (blocked.shape[0], blocked.shape[1]))
    goal_cell = planner._world_to_grid(goal, meta, (blocked.shape[0], blocked.shape[1]))
    assert start is not None
    assert goal_cell is not None
    free_start = planner._nearest_free(blocked, start, 4)
    free_goal = planner._nearest_free(blocked, goal_cell, 4)
    assert free_start is not None
    assert free_goal is not None

    path = planner._astar(blocked, free_start, free_goal)

    assert path
    assert path[0] == free_start
    assert path[-1] == free_goal
    assert any(cell[0] != free_start[0] for cell in path[1:-1])


def test_grid_route_fails_safe_on_malformed_observation() -> None:
    """Malformed observations should fail safely to a stop."""
    planner = GridRoutePlannerAdapter()
    assert planner.plan({"robot": object()}) == (0.0, 0.0)


def test_clearance_map_free_cells_get_positive_distance() -> None:
    """Free cells adjacent to an obstacle should receive clearance ≥ 1."""
    planner = GridRoutePlannerAdapter()
    blocked = np.zeros((5, 5), dtype=bool)
    blocked[2, 2] = True
    cm = planner._compute_clearance_map(blocked)

    assert cm[2, 2] == 0.0, "obstacle cell must have clearance 0"
    # The four immediate 4-neighbours are at BFS distance 1
    for r, c in [(1, 2), (3, 2), (2, 1), (2, 3)]:
        assert cm[r, c] == 1.0
    # Corner cells are at BFS distance 2 (4-connected)
    assert cm[0, 0] >= 2.0


def test_clearance_map_all_free_grid() -> None:
    """A grid with no obstacles should leave all cells at +inf clearance."""
    planner = GridRoutePlannerAdapter()
    blocked = np.zeros((4, 4), dtype=bool)
    cm = planner._compute_clearance_map(blocked)
    assert np.all(np.isinf(cm))


def test_astar_with_clearance_prefers_corridor_centre() -> None:
    """With clearance penalty, A* should route through the centre of a corridor.

    Setup: 3-cell-wide horizontal corridor (rows 1-3 of a 5x10 grid, all free;
    rows 0 and 4 are walls).  A star from (2,0) to (2,9) should stay at row 2
    (the centre) rather than drifting to row 1 or 3.
    """
    planner = GridRoutePlannerAdapter(GridRoutePlannerConfig(clearance_penalty_weight=1.0))
    blocked = np.ones((5, 10), dtype=bool)
    blocked[1:4, :] = False  # rows 1, 2, 3 are free

    clearance_map = planner._compute_clearance_map(blocked)
    path = planner._astar(blocked, (2, 0), (2, 9), clearance_map=clearance_map)

    assert path, "path must exist through the corridor"
    assert path[0] == (2, 0)
    assert path[-1] == (2, 9)
    # Every step in the path should stay at the centre row (row 2)
    assert all(r == 2 for r, _c in path), (
        "clearance-penalised A* should route through corridor centre"
    )


def test_astar_without_clearance_may_not_centre() -> None:
    """Without clearance penalty, A* is not required to stay centred."""
    planner = GridRoutePlannerAdapter(GridRoutePlannerConfig(clearance_penalty_weight=0.0))
    blocked = np.ones((5, 10), dtype=bool)
    blocked[1:4, :] = False

    path = planner._astar(blocked, (2, 0), (2, 9))
    assert path, "path must still exist without clearance penalty"


def test_astar_narrow_passage_routes_through_gap() -> None:
    """Narrow passages should route through the single available opening."""
    planner = GridRoutePlannerAdapter(GridRoutePlannerConfig(clearance_penalty_weight=0.5))
    blocked = np.ones((7, 14), dtype=bool)
    blocked[1:6, :] = False
    blocked[3, :] = True
    blocked[3, 7] = False

    clearance_map = planner._compute_clearance_map(blocked)
    path = planner._astar(blocked, (1, 0), (5, 13), clearance_map=clearance_map)

    assert path, "path must exist through the narrow passage"
    assert path[0] == (1, 0)
    assert path[-1] == (5, 13)
    assert (3, 7) in path, "Path must go through the narrow opening in the wall"


def test_build_grid_route_config_clearance_penalty() -> None:
    """build_grid_route_config should round-trip clearance_penalty_weight."""
    cfg = build_grid_route_config({"clearance_penalty_weight": 0.75})
    assert cfg.clearance_penalty_weight == 0.75
    assert build_grid_route_config(None).clearance_penalty_weight == 0.5
