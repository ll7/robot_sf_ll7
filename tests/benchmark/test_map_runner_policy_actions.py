"""Behavioral contracts for shared map-runner policy action conversion."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.map_runner_policies.map_runner_policy_actions import (
    ppo_action_to_unicycle,
    validate_ppo_policy_action,
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


@pytest.mark.parametrize(
    ("action", "cfg", "message"),
    (
        (
            {"v": 2.01, "omega": 0.0},
            {"action_space": "unicycle", "v_max": 2.0, "omega_max": 1.0},
            "field 'v'.*outside",
        ),
        (
            {"v": 1.0, "omega": -1.01},
            {"action_space": "unicycle", "v_max": 2.0, "omega_max": 1.0},
            "field 'omega'.*outside",
        ),
        (
            {"v": -0.01, "omega": 0.0},
            {"action_space": "unicycle", "v_max": 2.0, "omega_max": 1.0},
            "field 'v'.*outside",
        ),
        (
            {"vx": 1.2, "vy": 1.2},
            {"action_space": "velocity", "v_max": 1.5, "omega_max": 1.0},
            "speed=.*exceeds",
        ),
    ),
)
def test_validate_ppo_policy_action_rejects_out_of_bounds_values(
    action: dict[str, float], cfg: dict[str, float | str], message: str
) -> None:
    """Policy-space violations fail before the benchmark kinematics projection."""
    with pytest.raises(ValueError, match=message):
        validate_ppo_policy_action(action, cfg)


def test_validate_ppo_policy_action_records_declared_bounds() -> None:
    """The validator exposes a stable local action contract for trace provenance."""
    contract = validate_ppo_policy_action(
        {"v": 0.5, "omega": -0.25},
        {"action_space": "unicycle", "v_max": 2.0, "omega_max": 1.0},
    )

    assert contract == {
        "status": "valid",
        "action_space": "unicycle",
        "output_keys": ["v", "omega"],
        "bounds": {"v": [0.0, 2.0], "omega": [-1.0, 1.0]},
        "source": "PPOPlannerConfig",
    }
