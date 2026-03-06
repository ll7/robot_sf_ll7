"""Tests for guarded PPO safety veto behavior."""

from __future__ import annotations

import numpy as np

from robot_sf.planner.guarded_ppo import (
    GuardedPPOAdapter,
    build_guarded_ppo_config,
    build_guarded_ppo_fallback,
)


def _obs(
    *,
    robot=(0.0, 0.0),
    heading=0.0,
    goal=(2.0, 0.0),
    ped_positions=None,
    ped_velocities=None,
) -> dict[str, object]:
    ped_positions = [] if ped_positions is None else ped_positions
    ped_velocities = [] if ped_velocities is None else ped_velocities
    return {
        "robot": {
            "position": np.asarray(robot, dtype=float),
            "heading": np.asarray([heading], dtype=float),
            "speed": np.asarray([0.2], dtype=float),
        },
        "goal": {
            "current": np.asarray(goal, dtype=float),
            "next": np.asarray(goal, dtype=float),
        },
        "pedestrians": {
            "positions": np.asarray(ped_positions, dtype=float),
            "velocities": np.asarray(ped_velocities, dtype=float),
        },
    }


class _FallbackAdapter:
    def __init__(self, command: tuple[float, float]) -> None:
        self.command = command

    def plan(self, observation: dict[str, object]) -> tuple[float, float]:
        del observation
        return self.command


def test_guarded_ppo_keeps_safe_ppo_command() -> None:
    """Safe PPO commands should pass through unchanged."""
    guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config({"guard_near_field_distance": 2.5}),
        fallback_adapter=_FallbackAdapter((0.0, 1.0)),
    )
    command, decision = guard.choose_command(_obs(ped_positions=[(2.0, 1.0)]), (0.4, 0.0))
    assert command == (0.4, 0.0)
    assert decision in {"ppo_clear", "ppo_safe"}


def test_guarded_ppo_uses_fallback_when_ppo_is_unsafe() -> None:
    """Unsafe PPO commands should be replaced by a safe fallback when available."""
    guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config(
            {
                "guard_near_field_distance": 2.5,
                "guard_hard_ped_clearance": 0.45,
                "guard_first_step_ped_clearance": 0.55,
            }
        ),
        fallback_adapter=_FallbackAdapter((0.0, 1.0)),
    )
    command, decision = guard.choose_command(
        _obs(ped_positions=[(0.58, 0.0)], ped_velocities=[(0.0, 0.0)]),
        (0.6, 0.0),
    )
    assert command == (0.0, 1.0)
    assert decision == "fallback_safe"


def test_guarded_ppo_falls_back_to_stop_when_no_safe_motion_exists() -> None:
    """Guard should stop when PPO and fallback are both unsafe."""
    guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config(
            {
                "guard_near_field_distance": 2.5,
                "guard_hard_ped_clearance": 0.45,
                "guard_first_step_ped_clearance": 0.55,
            }
        ),
        fallback_adapter=_FallbackAdapter((0.6, 0.0)),
    )
    command, decision = guard.choose_command(
        _obs(ped_positions=[(0.58, 0.0)], ped_velocities=[(0.0, 0.0)]),
        (0.6, 0.0),
    )
    assert command == (0.0, 0.0)
    assert decision == "stop_safe"


def test_guarded_ppo_goal_and_clear_branches() -> None:
    """Guard should short-circuit for reached goals and clear near-field scenes."""
    guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config({"goal_tolerance": 0.3, "guard_near_field_distance": 0.5}),
        fallback_adapter=_FallbackAdapter((0.1, 0.2)),
    )
    command, decision = guard.choose_command(_obs(goal=(0.1, 0.0)), (0.3, 0.1))
    assert command == (0.0, 0.0)
    assert decision == "goal_reached"

    command, decision = guard.choose_command(_obs(ped_positions=[(2.0, 2.0)]), (0.3, 0.1))
    assert command == (0.3, 0.1)
    assert decision == "ppo_clear"


def test_guarded_ppo_best_effort_prefers_fallback_when_clearer() -> None:
    """When nothing is safe, the guard should prefer fallback if it has more clearance."""
    guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config({"guard_near_field_distance": 2.5}),
        fallback_adapter=_FallbackAdapter((0.0, 1.0)),
    )
    evaluations = iter(
        [
            {"safe": False, "min_ped_clear": 0.2},
            {"safe": False, "min_ped_clear": 0.8},
            {"safe": False, "min_ped_clear": 0.5},
        ]
    )
    guard._evaluate_command = lambda observation, command: next(evaluations)  # type: ignore[method-assign]
    command, decision = guard.choose_command(
        _obs(ped_positions=[(0.58, 0.0)], ped_velocities=[(0.0, 0.0)]),
        (0.6, 0.0),
    )
    assert command == (0.0, 1.0)
    assert decision == "fallback_best_effort"


def test_guarded_ppo_handles_malformed_pedestrian_payloads_and_config_builders() -> None:
    """Malformed pedestrian arrays should be sanitized and builder helpers should default cleanly."""
    guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config(None),
        fallback_adapter=_FallbackAdapter((0.0, 0.0)),
    )
    robot_pos, heading, goal, ped_pos, ped_vel = guard._extract_state(
        {
            "robot": {"position": [0.0, 0.0], "heading": [0.0]},
            "goal": {"current": [1.0, 0.0], "next": [1.0, 0.0]},
            "pedestrians": {"positions": [1.0, 2.0, 3.0], "velocities": [0.1]},
        }
    )
    assert robot_pos.tolist() == [0.0, 0.0]
    assert heading == 0.0
    assert goal.tolist() == [1.0, 0.0]
    assert ped_pos.shape == (0, 2)
    assert ped_vel.shape == (0, 2)

    fallback = build_guarded_ppo_fallback(None)
    assert fallback is not None


def test_guarded_ppo_reshapes_flattened_pedestrian_payloads() -> None:
    """Flattened compatibility payloads should be reshaped using pedestrian count."""
    guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config(None),
        fallback_adapter=_FallbackAdapter((0.0, 0.0)),
    )
    _robot_pos, _heading, _goal, ped_pos, ped_vel = guard._extract_state(
        {
            "robot": {"position": [0.0, 0.0], "heading": [0.0]},
            "goal": {"current": [1.0, 0.0], "next": [1.0, 0.0]},
            "pedestrians": {
                "count": 2,
                "positions": [1.0, 2.0, 3.0, 4.0],
                "velocities": [0.1, 0.2, 0.3, 0.4],
            },
        }
    )
    assert ped_pos.shape == (2, 2)
    assert ped_vel.shape == (2, 2)
    assert ped_pos.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_guarded_ppo_clear_path_still_checks_obstacle_safety() -> None:
    """Clear-path PPO should not bypass obstacle safety evaluation."""
    guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config({"guard_near_field_distance": 0.5}),
        fallback_adapter=_FallbackAdapter((0.0, 0.0)),
    )
    evaluations = iter(
        [
            {"safe": False, "min_ped_clear": float("inf")},
            {"safe": True, "min_ped_clear": float("inf")},
        ]
    )
    guard._evaluate_command = lambda observation, command: next(evaluations)  # type: ignore[method-assign]
    command, decision = guard.choose_command(_obs(ped_positions=[(2.0, 2.0)]), (0.3, 0.1))
    assert command == (0.0, 0.0)
    assert decision == "fallback_safe"


def test_guarded_ppo_obstacle_clearance_helper_branches() -> None:
    """Obstacle clearance helper should handle invalid payloads and distance queries."""
    guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config(
            {"guard_obstacle_threshold": 0.5, "guard_obstacle_search_cells": 2}
        ),
        fallback_adapter=_FallbackAdapter((0.0, 0.0)),
    )
    point = np.asarray([0.0, 0.0], dtype=float)

    assert guard._min_obstacle_clearance(point, {}) == float("inf")

    grid = np.zeros((1, 5, 5), dtype=float)
    meta = {"resolution": [0.5]}
    guard._extract_grid_payload = lambda observation: (grid, meta)  # type: ignore[method-assign]

    guard._preferred_channel = lambda meta: 2  # type: ignore[method-assign]
    assert guard._min_obstacle_clearance(point, {}) == float("inf")

    guard._preferred_channel = lambda meta: 0  # type: ignore[method-assign]
    guard._world_to_grid = lambda point, meta, grid_shape: None  # type: ignore[method-assign]
    assert guard._min_obstacle_clearance(point, {}) == 0.0

    guard._world_to_grid = lambda point, meta, grid_shape: (2, 2)  # type: ignore[method-assign]
    grid[0, 2, 2] = 1.0
    assert guard._min_obstacle_clearance(point, {}) == 0.0

    grid.fill(0.0)
    assert guard._min_obstacle_clearance(point, {}) == float("inf")

    grid[0, 1, 4] = 1.0
    clearance = guard._min_obstacle_clearance(point, {})
    assert 1.0 < clearance < 1.2


def test_guarded_ppo_no_peds_and_stop_best_effort_branch() -> None:
    """No-ped scenes should pass through PPO, and unsafe tie cases should stop."""
    clear_guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config({"guard_near_field_distance": 0.5}),
        fallback_adapter=_FallbackAdapter((0.2, 0.3)),
    )
    command, decision = clear_guard.choose_command(
        _obs(ped_positions=[], ped_velocities=[]), (0.4, 0.1)
    )
    assert command == (0.4, 0.1)
    assert decision == "ppo_clear"

    blocked_guard = GuardedPPOAdapter(
        config=build_guarded_ppo_config({"guard_near_field_distance": 2.5}),
        fallback_adapter=_FallbackAdapter((0.0, 1.0)),
    )
    evaluations = iter(
        [
            {"safe": False, "min_ped_clear": 0.6},
            {"safe": False, "min_ped_clear": 0.5},
            {"safe": False, "min_ped_clear": 0.7},
        ]
    )
    blocked_guard._evaluate_command = lambda observation, command: next(evaluations)  # type: ignore[method-assign]
    command, decision = blocked_guard.choose_command(
        _obs(ped_positions=[(0.58, 0.0)], ped_velocities=[(0.0, 0.0)]),
        (0.6, 0.0),
    )
    assert command == (0.0, 0.0)
    assert decision == "stop_best_effort"
