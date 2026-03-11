"""Tests for benchmark algorithm metadata enrichment contracts."""

from __future__ import annotations

from robot_sf.benchmark.algorithm_metadata import (
    canonical_algorithm_name,
    enrich_algorithm_metadata,
    infer_execution_mode_from_counts,
)


def test_canonical_algorithm_name_resolves_aliases() -> None:
    """Alias names should resolve to canonical benchmark algorithm names."""
    assert canonical_algorithm_name("simple_policy") == "goal"
    assert canonical_algorithm_name("sf") == "social_force"
    assert canonical_algorithm_name("unknown_algo") == "unknown_algo"


def test_random_baseline_metadata_marks_stochastic_reference() -> None:
    """Random baseline metadata should expose diagnostic stochastic semantics."""
    meta = enrich_algorithm_metadata(algo="random", metadata={"status": "ok"})
    assert meta["baseline_category"] == "diagnostic"
    assert meta["policy_semantics"] == "stochastic_uniform_action_reference"
    assert meta["stochastic_reference"] is True
    assert meta["distinct_from_goal_baseline"] is True


def test_planner_kinematics_and_adapter_impact_fields() -> None:
    """PPO metadata should include mixed command compatibility and impact scaffold."""
    meta = enrich_algorithm_metadata(
        algo="ppo",
        metadata={"status": "ok"},
        robot_kinematics="differential_drive",
        adapter_impact_requested=True,
    )
    planner = meta["planner_kinematics"]
    impact = meta["adapter_impact"]
    assert planner["planner_command_space"] == "mixed_vw_or_vxy"
    assert planner["supports_native_commands"] is True
    assert planner["supports_adapter_commands"] is True
    assert planner["robot_kinematics"] == "differential_drive"
    assert impact["requested"] is True
    assert impact["native_steps"] == 0
    assert impact["adapted_steps"] == 0


def test_infer_execution_mode_from_counts() -> None:
    """Execution mode inference should reflect observed native/adapted step counts."""
    assert infer_execution_mode_from_counts(native_steps=3, adapted_steps=0) == "native"
    assert infer_execution_mode_from_counts(native_steps=0, adapted_steps=3) == "adapter"
    assert infer_execution_mode_from_counts(native_steps=3, adapted_steps=2) == "mixed"
    assert infer_execution_mode_from_counts(native_steps=0, adapted_steps=0) == "unknown"
