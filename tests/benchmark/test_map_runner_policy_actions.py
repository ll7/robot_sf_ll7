"""Behavioral contracts for shared map-runner policy action conversion."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.map_runner_policies.map_runner_policy_actions import (
    ppo_action_to_unicycle,
)
from robot_sf.planner.kinematics_model import DifferentialDriveKinematicsModel


@pytest.mark.parametrize(
    ("action", "cfg", "field"),
    (
        ({"v": float("nan"), "omega": 0.0}, {}, "v"),
        ({"v": 0.0, "omega": float("inf")}, {}, "omega"),
        ({"vx": float("nan"), "vy": 0.0}, {}, "vx"),
        ({"vx": 0.0, "vy": float("-inf")}, {}, "vy"),
        ({"vx": 1.0, "vy": 0.0}, {"omega_max": float("nan")}, "omega_max"),
        ({"vx": 1.0, "vy": 0.0}, {"omega_kp": float("inf")}, "omega_kp"),
    ),
)
def test_ppo_action_to_unicycle_rejects_non_finite_inputs(
    action: dict[str, float],
    cfg: dict[str, float],
    field: str,
) -> None:
    """Non-finite policy values fail before projection, clipping, or heading math."""
    with pytest.raises(ValueError, match=f"{field} must be finite"):
        ppo_action_to_unicycle(
            action,
            {"robot": {"heading": [0.0]}},
            cfg,
            kinematics_model=DifferentialDriveKinematicsModel(
                max_linear_speed=2.0,
                max_angular_speed=2.0,
            ),
        )
