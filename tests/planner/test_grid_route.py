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
