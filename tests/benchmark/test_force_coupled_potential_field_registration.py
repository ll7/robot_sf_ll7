"""Canonical registration tests for the issue #7889 experimental planner."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.benchmark.algorithm_metadata import enrich_algorithm_metadata
from robot_sf.benchmark.algorithm_readiness import (
    get_algorithm_readiness,
    require_algorithm_allowed,
)
from robot_sf.benchmark.map_runner.map_runner import _build_policy
from robot_sf.benchmark.policy_builders import build_registered_adapter_policy_spec


def _observation() -> dict:
    grid = np.zeros((3, 9, 9), dtype=np.float32)
    grid[0, 4, 5] = 1.0
    return {
        "robot": {"position": [0.0, 0.0], "heading": [0.0]},
        "goal": {"current": [4.0, 0.0]},
        "pedestrians": {"positions": [[1.0, 0.0]], "count": [1]},
        "occupancy_grid": grid,
        "occupancy_grid_meta_origin": [-4.5, -4.5],
        "occupancy_grid_meta_resolution": [1.0],
        "occupancy_grid_meta_size": [9.0, 9.0],
        "occupancy_grid_meta_use_ego_frame": [1.0],
        "occupancy_grid_meta_center_on_robot": [1.0],
        "occupancy_grid_meta_channel_indices": [0, 1, -1, 2],
        "occupancy_grid_meta_robot_pose": [0.0, 0.0, 0.0],
    }


def test_registered_builder_is_opt_in_and_map_runner_resolvable() -> None:
    with pytest.raises(ValueError, match="allow_testing_algorithms"):
        build_registered_adapter_policy_spec("force_coupled_potential_field", {})

    config = {"allow_testing_algorithms": True, "max_linear_speed": 0.8}
    spec = build_registered_adapter_policy_spec("force_coupled_potential_field", config)
    assert spec is not None
    assert spec.adapter_name == "ForceCoupledPotentialFieldPlanner"

    policy, meta = _build_policy("force_coupled_potential_field", config)
    command = policy(_observation())
    assert np.all(np.isfinite(command))
    assert meta["algorithm"] == "force_coupled_potential_field"
    assert meta["planner_kinematics"]["testing_only_adapter"] is True
    diagnostics = policy._planner_stats()
    assert diagnostics["status"] == "ok"
    assert diagnostics["missing_inputs"] == []
    assert diagnostics["obstacle_repulsive_force"][0] < 0.0


def test_readiness_and_method_metadata_preserve_smoke_only_boundary() -> None:
    readiness = get_algorithm_readiness("force_coupled_potential_field")
    assert readiness is not None
    assert readiness.tier == "experimental"
    assert readiness.requires_explicit_opt_in is True
    with pytest.raises(ValueError, match="allow_testing_algorithms"):
        require_algorithm_allowed(
            algo="force_coupled_potential_field",
            benchmark_profile="experimental",
            ppo_paper_ready=False,
        )

    metadata = enrich_algorithm_metadata(
        algo="force_coupled_potential_field",
        metadata={"status": "ok"},
        execution_mode="adapter",
        robot_kinematics="differential_drive",
    )
    assert metadata["baseline_category"] == "classical"
    assert metadata["policy_semantics"] == ("clean_room_force_coupled_potential_field_experimental")
    assert metadata["observation_spec"]["default_mode"] == "socnav_state"
    assert metadata["planner_kinematics"]["prototype_only"] is True
