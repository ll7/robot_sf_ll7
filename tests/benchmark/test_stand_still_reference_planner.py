"""Contracts for the native stand-still reference planner."""

from __future__ import annotations

from robot_sf.benchmark.algorithm_metadata import (
    canonical_algorithm_name,
    enrich_algorithm_metadata,
    observation_spec_for_algorithm,
    resolve_observation_mode,
)
from robot_sf.benchmark.algorithm_readiness import (
    get_algorithm_readiness,
    paper_baseline_algorithms,
    require_algorithm_allowed,
)
from robot_sf.benchmark.map_runner.map_runner import _build_policy, _preflight_policy


def test_stand_still_emits_zero_without_reading_observation() -> None:
    """The policy is invariant to missing, malformed, or changing observations."""
    policy, metadata = _build_policy("stand_still", {"max_speed": 9.0})

    assert policy({}) == (0.0, 0.0)
    assert policy({"robot": None, "goal": [100.0, -4.0], "pedestrians": object()}) == (
        0.0,
        0.0,
    )
    assert policy(None) == (0.0, 0.0)

    assert metadata["algorithm"] == "stand_still"
    assert metadata["canonical_algorithm"] == "stand_still"
    assert metadata["baseline_category"] == "classical"
    assert metadata["policy_semantics"] == "constant_zero_velocity_reference"
    assert metadata["planner_kinematics"]["execution_mode"] == "native"
    assert metadata["planner_kinematics"]["supports_native_commands"] is True
    assert metadata["planner_kinematics"]["supports_adapter_commands"] is False
    assert metadata["planner_contract"]["planner_id"] == "stand_still"


def test_stand_still_readiness_and_observation_contract() -> None:
    """Map-runner config validation accepts this baseline and records ignored inputs."""
    readiness = get_algorithm_readiness("stand_still")
    assert readiness is not None
    assert readiness.canonical_name == "stand_still"
    assert readiness.tier == "baseline-ready"
    assert canonical_algorithm_name("stand_still") == "stand_still"
    assert (
        require_algorithm_allowed(
            algo="stand_still",
            benchmark_profile="baseline-safe",
            ppo_paper_ready=False,
        )
        == readiness
    )
    effective_config, preflight = _preflight_policy(
        algo="stand_still",
        algo_config={},
        benchmark_profile="baseline-safe",
        missing_prereq_policy="fail-fast",
    )
    assert effective_config == {}
    assert preflight["status"] == "ok"

    observation_spec = observation_spec_for_algorithm("stand_still")
    assert observation_spec["inputs"] == []
    assert observation_spec["default_mode"] == "socnav_state"
    assert resolve_observation_mode("stand_still", "sensor_fusion_state") == "sensor_fusion_state"
    metadata = enrich_algorithm_metadata(algo="stand_still")
    assert metadata["planner_contract"]["observation_contract"]["required_inputs"] == []
    assert "stand_still" not in paper_baseline_algorithms()
