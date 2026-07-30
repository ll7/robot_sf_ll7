"""Tests for TypedDict episode payload types introduced by issue #6470.

Verifies structural compatibility, key access patterns, and round-trip
serialization of the TypedDicts used in map_runner_episode.py.
"""

from __future__ import annotations

from robot_sf.benchmark.types import (
    AdapterImpact,
    AlgoMeta,
    EpisodeRecordDict,
    NoiseSpec,
    OutcomePayload,
    PlannerDecisionTraceEntry,
    TrackingPrecisionSpec,
    TrackingPrecisionSpeedContract,
)


def test_noise_spec_structural_compatibility() -> None:
    """NoiseSpec accepts all documented fields."""
    spec: NoiseSpec = {
        "enabled": True,
        "profile": "gaussian",
        "seed": 42,
        "pose_noise_std_m": 0.05,
        "heading_noise_std_rad": 0.01,
        "lidar_dropout_prob": 0.0,
        "lidar_dropout_value": 0.0,
        "pedestrian_position_noise_std_m": 0.1,
        "pedestrian_false_negative_prob": 0.0,
        "pedestrian_occlusion_max_range_m": None,
        "observation_delay_steps": 0,
        "pedestrian_false_positive_prob": 0.0,
        "pedestrian_false_positive_radius_m": 4.0,
        "pedestrian_false_positive_radius": 0.35,
        "interpretation": "non_calibrated_benchmark_robustness_noise_not_a_real_sensor_model",
    }
    assert spec["enabled"] is True
    assert spec["profile"] == "gaussian"
    assert spec["seed"] == 42


def test_noise_spec_optional_keys() -> None:
    """NoiseSpec with total=False tolerates partial construction."""
    spec: NoiseSpec = {"enabled": True}
    assert spec.get("profile") is None
    assert spec.get("nonexistent", "fallback") == "fallback"


def test_tracking_precision_spec_structural() -> None:
    """TrackingPrecisionSpec accepts nested speed_contract."""
    spec: TrackingPrecisionSpec = {
        "enabled": True,
        "target_motp_m": 0.5,
        "speed_contract": {
            "threshold_m": 2.5,
            "default_speed": 2.0,
            "defensive_speed": 0.5,
            "mode": "diagnostic",
        },
        "seed_salt": 0,
        "schema_version": "tracking_precision_contract.v1",
        "interpretation": "internal_non_hardware_tracking_precision_proxy_not_a_real_sensor_model",
    }
    assert spec["enabled"] is True
    assert spec["speed_contract"]["mode"] == "diagnostic"


def test_tracking_precision_speed_contract_optional() -> None:
    """TrackingPrecisionSpeedContract tolerates partial construction."""
    contract: TrackingPrecisionSpeedContract = {"threshold_m": 3.0}
    assert contract["threshold_m"] == 3.0
    assert contract.get("mode") is None


def test_algo_meta_structural_compatibility() -> None:
    """AlgoMeta accepts standard algorithm metadata fields."""
    meta: AlgoMeta = {
        "algorithm": "orca",
        "canonical_algorithm": "orca",
        "baseline_category": "classical",
        "policy_semantics": "orca_adapter",
        "status": "ok",
        "adapter_impact": {
            "requested": True,
            "native_steps": 10,
            "adapted_steps": 90,
            "status": "complete",
            "execution_mode": "mixed",
            "adapter_fraction": 0.9,
        },
        "tracking_precision": {"contract_honored": True, "step_count": 100},
    }
    assert meta["algorithm"] == "orca"
    assert meta["adapter_impact"]["adapter_fraction"] == 0.9
    assert meta["tracking_precision"]["contract_honored"] is True


def test_algo_meta_partial_dict() -> None:
    """AlgoMeta with total=False tolerates minimal construction."""
    meta: AlgoMeta = {"algorithm": "goal"}
    assert meta["algorithm"] == "goal"
    assert meta.get("adapter_impact") is None


def test_planner_decision_trace_entry() -> None:
    """PlannerDecisionTraceEntry accepts standard fields."""
    entry: PlannerDecisionTraceEntry = {
        "step": 5,
        "selected_source": "hybrid",
        "selected_command": [1.0, 0.0],
        "selected_score": 0.95,
        "distance_to_goal_m": 3.0,
    }
    assert entry["step"] == 5
    assert entry["selected_score"] == 0.95
    assert entry.get("static_recenter") is None


def test_planner_decision_trace_entry_with_topology() -> None:
    """PlannerDecisionTraceEntry carries optional topology fields."""
    entry: PlannerDecisionTraceEntry = {
        "step": 3,
        "selected_source": "topology",
        "selected_command": [0.5, 0.1],
        "selected_score": None,
        "topology_guided": {"status": "ok", "hypothesis_count": 3},
        "topology_lane_status": "active",
        "topology_fallback_reason": "no_candidate",
    }
    assert entry["topology_lane_status"] == "active"
    assert entry["topology_guided"]["hypothesis_count"] == 3


def test_outcome_payload() -> None:
    """OutcomePayload carries episode outcome flags."""
    outcome: OutcomePayload = {
        "route_complete": True,
        "collision_event": False,
        "timeout_event": False,
    }
    assert outcome["route_complete"] is True
    assert outcome["collision_event"] is False


def test_adapter_impact() -> None:
    """AdapterImpact carries adapter counters."""
    impact: AdapterImpact = {
        "requested": True,
        "native_steps": 0,
        "adapted_steps": 100,
        "status": "complete",
        "execution_mode": "adapter",
        "adapter_fraction": 1.0,
    }
    assert impact["execution_mode"] == "adapter"
    assert impact["adapter_fraction"] == 1.0


def test_episode_record_dict_structural() -> None:
    """EpisodeRecordDict accepts a minimal valid record shape."""
    record: EpisodeRecordDict = {
        "version": "v1",
        "episode_id": "test-scenario--42--abc123",
        "scenario_id": "test-scenario",
        "seed": 42,
        "scenario_params": {"id": "test-scenario", "algo": "orca"},
        "metrics": {"success": 1.0, "collisions": 0.0},
        "algorithm_metadata": {"algorithm": "orca"},
        "algo": "orca",
        "observation_mode": "lidar",
        "observation_level": "full",
        "outcome": {"route_complete": True, "collision_event": False, "timeout_event": False},
        "observation_noise": {"enabled": False},
        "tracking_precision": {"enabled": False},
    }
    assert record["episode_id"] == "test-scenario--42--abc123"
    assert record["metrics"]["success"] == 1.0
    assert record["observation_noise"]["enabled"] is False


def test_episode_record_carries_typed_metadata() -> None:
    """EpisodeRecordDict.algorithm_metadata is typed as AlgoMeta."""
    record: EpisodeRecordDict = {
        "version": "v1",
        "episode_id": "ep-1",
        "scenario_id": "sc-1",
        "seed": 0,
        "scenario_params": {},
        "metrics": {},
        "algorithm_metadata": {
            "algorithm": "sf",
            "canonical_algorithm": "social_force",
            "adapter_impact": {"requested": False},
        },
        "algo": "sf",
        "observation_mode": "lidar",
        "observation_level": "full",
        "outcome": {"route_complete": False, "collision_event": False, "timeout_event": False},
        "observation_noise": {},
        "tracking_precision": {},
    }
    algo_meta: AlgoMeta = record["algorithm_metadata"]
    assert algo_meta["algorithm"] == "sf"
    assert algo_meta["adapter_impact"]["requested"] is False


def test_episode_record_partial_defaults() -> None:
    """EpisodeRecordDict with total=False tolerates missing optional fields."""
    record: EpisodeRecordDict = {
        "version": "v1",
        "episode_id": "ep-minimal",
        "scenario_id": "sc-minimal",
        "seed": 1,
        "scenario_params": {},
        "metrics": {},
        "algo": "goal",
        "observation_mode": "lidar",
        "observation_level": "full",
        "outcome": {"route_complete": False, "collision_event": False, "timeout_event": False},
        "observation_noise": {},
        "tracking_precision": {},
    }
    assert record.get("benchmark_track") is None
    assert record.get("integrity") is None
