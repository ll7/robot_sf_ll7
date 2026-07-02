"""Map-runner wiring tests for the issue #4010 diffusion-policy planner."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from robot_sf.benchmark.map_runner import _build_policy


def _map_runner_obs() -> dict[str, object]:
    return {
        "robot": {
            "position": [0.0, 0.0],
            "velocity": [0.0, 0.0],
            "goal": [2.0, 0.0],
            "heading": [0.0],
            "radius": [0.3],
        },
        "pedestrians": {
            "positions": [[0.7, 0.1], [1.1, -0.2]],
            "velocities": [[-0.1, 0.0], [0.0, 0.1]],
            "radii": [0.25, 0.25],
            "count": [2],
        },
        "dt": [0.1],
    }


def test_build_policy_registers_diffusion_policy_with_diagnostic_metadata() -> None:
    """Registry construction returns callable policy and diagnostic-only metadata."""
    policy, meta = _build_policy(
        "diffusion_policy",
        {
            "allow_untrained_smoke": True,
            "deterministic": True,
            "seed": 4010,
            "max_linear_speed": 0.6,
            "max_angular_speed": 0.5,
            "num_action_samples": 4,
            "denoising_steps": 3,
        },
        robot_kinematics="differential_drive",
    )
    command = policy(_map_runner_obs())

    assert callable(policy)
    assert 0.0 <= command[0] <= 0.6
    assert -0.5 <= command[1] <= 0.5
    assert meta["algorithm"] == "diffusion_policy"
    assert meta["planner_kinematics"]["execution_mode"] == "adapter"
    assert meta["diffusion_policy"]["evidence_tier"] == "diagnostic-only"
    assert meta["planner_contract"]["action_contract"]["command_space"] == "unicycle_vw"
    assert meta["planner_kinematics"]["limitations"] == (
        "diagnostic_only_untrained_smoke_not_benchmark_evidence"
    )


def test_build_policy_projects_command_through_feasibility_contract() -> None:
    """Adapter output is still constrained by the benchmark command contract."""
    policy, meta = _build_policy(
        "colson_style_diffusion",
        {
            "allow_untrained_smoke": True,
            "deterministic": True,
            "max_linear_speed": 5.0,
            "max_angular_speed": 5.0,
            "num_action_samples": 2,
        },
        robot_kinematics="differential_drive",
    )
    linear, angular = policy(_map_runner_obs())

    assert 0.0 <= linear <= 5.0
    assert abs(angular) <= 5.0
    assert meta["kinematics_feasibility"]["commands_evaluated"] == 1
    assert meta["kinematics_feasibility"]["projected_count"] >= 0


def test_build_policy_attaches_reset_close_and_stats_hooks() -> None:
    """Map-runner wrapper exposes adapter lifecycle hooks."""
    policy, _meta = _build_policy(
        "diffusion_local_policy",
        {"allow_untrained_smoke": True, "seed": 1},
    )
    assert hasattr(policy, "_planner_reset")
    assert hasattr(policy, "_planner_close")
    assert hasattr(policy, "_planner_stats")
    policy._planner_reset(seed=1)
    policy(_map_runner_obs())
    stats = policy._planner_stats()
    policy._planner_close()

    assert stats["diffusion_policy"]["status"] == "ok"
    assert stats["diffusion_policy"]["evidence_tier"] == "diagnostic-only"
