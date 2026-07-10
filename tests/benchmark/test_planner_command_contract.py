"""Tests for benchmark planner command-space helper contracts."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.planner_command_contract import (
    PlannerContractValidationError,
    default_robot_command_space,
    init_feasibility_metadata,
    planner_kinematics_compatibility,
    project_with_feasibility,
    validate_planner_contract,
)
from robot_sf.planner.kinematics_model import DifferentialDriveKinematicsModel


def test_default_robot_command_space_respects_holonomic_command_mode() -> None:
    """Command-space labels should preserve benchmark action-mode metadata."""
    assert default_robot_command_space("differential_drive", {}) == "unicycle_vw"
    assert default_robot_command_space("holonomic", {}) == "holonomic_vxy_world"
    assert (
        default_robot_command_space(
            "holonomic",
            {"command_mode": "unicycle"},
        )
        == "unicycle_vw"
    )
    assert (
        default_robot_command_space(
            "holonomic",
            {"command_mode": "unicycle"},
            robot_command_mode="vx_vy",
        )
        == "holonomic_vxy_world"
    )


def test_project_with_feasibility_tracks_projection_counters() -> None:
    """Projection metadata should remain stable outside the large map-runner module."""
    meta: dict[str, object] = {}
    init_feasibility_metadata(meta)
    model = DifferentialDriveKinematicsModel(max_linear_speed=1.0, max_angular_speed=0.5)

    projected = project_with_feasibility(
        model=model,
        command=(2.0, 1.0),
        meta=meta,
    )

    assert projected == pytest.approx((1.0, 0.5))
    feasibility = meta["kinematics_feasibility"]
    assert isinstance(feasibility, dict)
    assert feasibility["commands_evaluated"] == 1
    assert feasibility["infeasible_native_count"] == 1
    assert feasibility["projected_count"] == 1
    assert feasibility["_max_abs_delta_linear"] == pytest.approx(1.0)
    assert feasibility["_max_abs_delta_angular"] == pytest.approx(0.5)


def test_planner_kinematics_compatibility_blocks_known_invalid_pairs() -> None:
    """Compatibility guard should fail closed for known unsupported planner/action pairs."""
    compatible, reason = planner_kinematics_compatibility(
        algo="rvo",
        robot_kinematics="holonomic",
        algo_config={},
    )
    assert compatible is False
    assert reason is not None and "disabled" in reason

    compatible, reason = planner_kinematics_compatibility(
        algo="ppo",
        robot_kinematics="holonomic",
        algo_config={"obs_mode": "image"},
    )
    assert compatible is False
    assert reason is not None and "non-image" in reason

    compatible, reason = planner_kinematics_compatibility(
        algo="dwa",
        robot_kinematics="holonomic",
        algo_config={},
    )
    assert compatible is False
    assert reason is not None and "unicycle" in reason

    assert planner_kinematics_compatibility(
        algo="ppo",
        robot_kinematics="differential_drive",
        algo_config={"obs_mode": "image"},
    ) == (True, None)


def test_validate_planner_contract_names_incompatible_action_fixture() -> None:
    """Contract validation errors should name planner, kinematics, and remediation context."""
    with pytest.raises(PlannerContractValidationError) as excinfo:
        validate_planner_contract(
            algo="rvo",
            robot_kinematics="holonomic",
            algo_config={},
            observation_mode="socnav_state",
        )

    message = str(excinfo.value)
    assert "planner 'rvo'" in message
    assert "holonomic" in message
    assert "contract mismatch" in message


def test_validate_planner_contract_returns_active_robot_kinematics() -> None:
    """Returned contracts should describe the validated runtime kinematics."""
    contract = validate_planner_contract(
        algo="orca",
        robot_kinematics="differential_drive",
        algo_config={},
        observation_mode="socnav_state",
    )

    action = contract["action_contract"]
    assert action["active_robot_kinematics"] == "differential_drive"
    assert action["compatible_robot_kinematics"] == ["differential_drive"]
