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
    return {
        "robot": {"position": [0.0, 0.0], "heading": [0.0]},
        "goal": {"current": [4.0, 0.0]},
        "obstacles": {"positions": [[1.5, 0.5]]},
        "pedestrians": {"positions": [[1.0, 0.0]], "count": [1]},
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
    assert policy._planner_stats()["status"] == "ok"


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
