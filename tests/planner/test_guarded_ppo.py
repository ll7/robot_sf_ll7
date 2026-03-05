"""Tests for guarded PPO safety veto behavior."""

from __future__ import annotations

import numpy as np

from robot_sf.planner.guarded_ppo import GuardedPPOAdapter, build_guarded_ppo_config


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
