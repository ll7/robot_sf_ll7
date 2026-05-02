"""Tests for the native testing-only safety-barrier planner."""

from __future__ import annotations

import numpy as np

from robot_sf.planner.safety_barrier import (
    SafetyBarrierPlannerAdapter,
    SafetyBarrierPlannerConfig,
    build_safety_barrier_config,
)


def _observation(
    *,
    goal: tuple[float, float] = (2.0, 0.0),
    heading: float = 0.0,
    speed: float = 0.0,
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
            "speed": [speed],
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


def test_safety_barrier_build_config_defaults() -> None:
    """Config builder should parse the planner defaults and overrides safely."""
    cfg = build_safety_barrier_config({"max_linear_speed": 0.7, "turn_away_gain": 1.8})
    assert cfg.max_linear_speed == 0.7
    assert cfg.turn_away_gain == 1.8
    assert build_safety_barrier_config(None).goal_tolerance == 0.25


def test_safety_barrier_returns_bounded_goal_directed_command() -> None:
    """Open-space commands should stay finite and within configured limits."""
    planner = SafetyBarrierPlannerAdapter(
        SafetyBarrierPlannerConfig(max_linear_speed=0.6, max_angular_speed=0.8)
    )
    linear, angular = planner.plan(_observation())
    assert 0.0 <= linear <= 0.6
    assert abs(angular) <= 0.8


def test_safety_barrier_stops_at_goal() -> None:
    """Planner should stop once the goal is already inside tolerance."""
    planner = SafetyBarrierPlannerAdapter(SafetyBarrierPlannerConfig(goal_tolerance=0.3))
    assert planner.plan(_observation(goal=(0.1, 0.0))) == (0.0, 0.0)


def test_safety_barrier_keeps_current_goal_until_waypoint_is_reached() -> None:
    """Planner should not switch to the next-goal sentinel before reaching the current waypoint."""
    planner = SafetyBarrierPlannerAdapter(SafetyBarrierPlannerConfig(goal_tolerance=0.25))
    observation = _observation(goal=(2.0, 0.0))
    observation["goal"] = {"current": [2.0, 0.0], "next": [0.0, 0.0]}

    linear, angular = planner.plan(observation)

    assert linear > 0.0
    assert abs(angular) < 0.2


def test_safety_barrier_advances_to_next_goal_after_current_waypoint() -> None:
    """Planner should use the next waypoint once the current waypoint is already satisfied."""
    planner = SafetyBarrierPlannerAdapter(SafetyBarrierPlannerConfig(goal_tolerance=0.25))
    observation = _observation(goal=(0.1, 0.0))
    observation["goal"] = {"current": [0.1, 0.0], "next": [2.0, 0.0]}

    linear, angular = planner.plan(observation)

    assert linear > 0.0
    assert abs(angular) < 0.2


def test_safety_barrier_stops_and_turns_on_immediate_obstacle() -> None:
    """Immediate frontal occupancy should collapse forward speed and trigger recovery turn."""
    planner = SafetyBarrierPlannerAdapter()
    linear, angular = planner.plan(_observation(occupied_cells=[(10, 11), (10, 12), (10, 13)]))
    assert linear == 0.0
    assert abs(angular) > 0.0


def test_safety_barrier_turns_away_from_more_blocked_side() -> None:
    """Asymmetric occupancy should bias the turn direction away from the blocked side."""
    planner = SafetyBarrierPlannerAdapter()
    linear, angular = planner.plan(
        _observation(occupied_cells=[(11, 11), (12, 11), (13, 12), (13, 13), (14, 13)])
    )
    assert linear >= 0.0
    assert angular < 0.0


def test_safety_barrier_reacts_to_shallow_front_right_obstacle() -> None:
    """A shallow front-right obstacle should trigger an earlier left-turn bias."""
    planner = SafetyBarrierPlannerAdapter()
    linear, angular = planner.plan(_observation(occupied_cells=[(8, 13), (9, 13), (10, 14)]))

    assert linear >= 0.0
    assert angular > 0.0


def test_safety_barrier_fails_safe_on_malformed_observation() -> None:
    """Malformed observations should not raise and should fail safely to stop."""
    planner = SafetyBarrierPlannerAdapter()
    assert planner.plan({"robot": object()}) == (0.0, 0.0)
