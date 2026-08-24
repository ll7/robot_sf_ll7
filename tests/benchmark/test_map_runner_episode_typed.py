"""Tests for TypedDict episode payload types introduced by issue #6470.

Verifies structural compatibility, key access patterns, and round-trip
serialization of the TypedDicts used in map_runner_episode.py.
"""

from __future__ import annotations

import ast
import inspect
import json
import typing
from collections.abc import Callable
from multiprocessing.context import BaseContext

import numpy as np

from robot_sf.benchmark.observation_noise import normalize_observation_noise_spec
from robot_sf.benchmark.tracking_precision_contract import normalize_tracking_precision_spec
from robot_sf.benchmark.types import (
    AdapterImpact,
    AlgoMeta,
    EpisodeRecordDict,
    MapBatchConfig,
    NoiseConfig,
    NoiseSpec,
    OutcomePayload,
    PlannerDecisionTrace,
    PlannerDecisionTraceEntry,
    PlannerDynamicWindow,
    PlannerRuntime,
    PlannerTargetGoal,
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


def test_normalized_specs_match_typed_dict_keys_and_serialize() -> None:
    """Canonical normalizers stay aligned with their TypedDict payload contracts."""
    noise_spec = normalize_observation_noise_spec(None)
    assert set(noise_spec) == set(typing.get_type_hints(NoiseSpec))

    tracking_spec = normalize_tracking_precision_spec(None)
    assert set(tracking_spec) == set(typing.get_type_hints(TrackingPrecisionSpec))
    assert set(tracking_spec["speed_contract"]) == set(
        typing.get_type_hints(TrackingPrecisionSpeedContract)
    )
    json.dumps(noise_spec)
    json.dumps(tracking_spec)


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
        "fallback_reason": "",
        "benchmark_track": {"benchmark_track": "oracle"},
        "config": {"horizon": 120},
        "config_hash": "abc123",
        "kinematics_feasibility": {"status": "available"},
        "safety_shield_contract": {"enabled": False},
        "fallback_or_degraded": False,
        "_native_run_state": {
            "deadlock_field": {},
            "planner_diagnostics": {"fallback_count": 0},
        },
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
    assert meta["fallback_or_degraded"] is False
    assert meta["_native_run_state"]["deadlock_field"] == {}
    assert {"fallback_or_degraded", "_native_run_state"} <= set(typing.get_type_hints(AlgoMeta))


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
        "planner_mode": "NORMAL",
        "rejection_counts": {},
        "nearest_static_obstacle_distance_m": 1.25,
    }
    assert entry["step"] == 5
    assert entry["selected_score"] == 0.95
    assert entry["planner_mode"] == "NORMAL"
    assert entry["nearest_static_obstacle_distance_m"] == 1.25
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


def test_planner_decision_trace_entry_with_dwa_payload() -> None:
    """PlannerDecisionTraceEntry matches the serialized DWA diagnostic shape."""
    target_goal: PlannerTargetGoal = {"kind": "next", "x": 5.0, "y": 3.0}
    dynamic_window: PlannerDynamicWindow = {
        "v_min": 0.0,
        "v_max": 1.0,
        "w_min": -0.3,
        "w_max": 0.3,
    }
    entry: PlannerDecisionTraceEntry = {
        "step": 1,
        "selected_command": [0.5, 0.1],
        "selected_score": 2.5,
        "feasible_score_min": 1.0,
        "feasible_score_max": 2.5,
        "dynamic_window": dynamic_window,
        "target_goal": target_goal,
        "global_route_probe_activated": True,
    }
    assert entry["target_goal"]["kind"] == "next"
    assert entry["dynamic_window"]["v_max"] == 1.0


def test_planner_decision_trace_envelope() -> None:
    """PlannerDecisionTrace types the episode-level trace without changing its dict shape."""
    trace: PlannerDecisionTrace = {
        "schema_version": "planner-decision-trace.v1",
        "dt": 0.1,
        "initial_goal_distance_m": 5.0,
        "steps": [
            {
                "step": 0,
                "selected_command": [1.0, 0.0],
            }
        ],
    }
    assert trace["schema_version"] == "planner-decision-trace.v1"
    assert trace["steps"][0]["step"] == 0


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
        "metrics": {
            "success": True,
            "collisions": 0,
            "force_quantiles": {"q50": 0.1, "q90": 0.2, "q95": 0.3},
        },
        "algorithm_metadata": {"algorithm": "orca"},
        "algo": "orca",
        "observation_mode": "lidar",
        "observation_level": "full",
        "outcome": {"route_complete": True, "collision_event": False, "timeout_event": False},
        "observation_noise": {"enabled": False},
        "tracking_precision": {"enabled": False},
    }
    assert record["episode_id"] == "test-scenario--42--abc123"
    assert record["metrics"]["success"] is True
    assert record["metrics"]["force_quantiles"]["q50"] == 0.1
    assert record["observation_noise"]["enabled"] is False
    assert json.loads(json.dumps(record)) == record


def test_episode_record_schema_extensions_are_typed() -> None:
    """EpisodeRecordDict includes optional schema-level runner extensions."""
    record: EpisodeRecordDict = {
        "notes": "diagnostic run",
        "tags": ["smoke"],
        "identity": {"robot": "r1"},
        "video": {
            "path": "output/recordings/episode.mp4",
            "format": "mp4",
            "filesize_bytes": 1,
            "frames": 0,
            "renderer": "none",
        },
    }

    assert record["tags"] == ["smoke"]
    assert record["identity"]["robot"] == "r1"
    assert json.loads(json.dumps(record)) == record


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


def test_benchmark_orchestration_annotations_resolve_at_runtime() -> None:
    """Typed orchestration boundaries remain inspectable by runtime tooling."""
    from robot_sf.benchmark import map_runner_episode
    from robot_sf.benchmark.map_runner import map_runner

    batch_hints = typing.get_type_hints(map_runner.run_map_batch)
    loop_hints = typing.get_type_hints(map_runner_episode._run_episode_step_loop)
    typing.get_type_hints(map_runner_episode._EpisodeRunContext)
    typing.get_type_hints(map_runner_episode._prepare_policy_and_observation_contract)
    config_hints = typing.get_type_hints(MapBatchConfig)
    noise_hints = typing.get_type_hints(NoiseConfig)
    planner_hints = typing.get_type_hints(PlannerRuntime)

    assert batch_hints["batch_config"] == MapBatchConfig | None
    assert batch_hints["multiprocessing_context"] == BaseContext | None
    assert loop_hints["planner_runtime"] is PlannerRuntime
    assert loop_hints["noise"] is NoiseConfig
    assert config_hints["multiprocessing_context"] == BaseContext | None
    assert noise_hints["rng"] is np.random.Generator
    assert planner_hints["policy_fn"] == Callable[..., typing.Any]


def test_orchestration_boundaries_do_not_expose_any_annotations() -> None:
    """Issue #6461 boundaries must not regress to nested ``Any`` annotations."""
    from robot_sf.benchmark import map_runner_episode
    from robot_sf.benchmark.map_runner import map_runner

    for function in (map_runner.run_map_batch, map_runner_episode._run_episode_step_loop):
        definition = ast.parse(inspect.getsource(function)).body[0]
        assert isinstance(definition, ast.FunctionDef)
        annotations = [
            argument.annotation
            for argument in (
                *definition.args.posonlyargs,
                *definition.args.args,
                *definition.args.kwonlyargs,
            )
            if argument.annotation is not None
        ]
        if definition.returns is not None:
            annotations.append(definition.returns)
        assert not any(
            isinstance(node, ast.Name) and node.id == "Any"
            for annotation in annotations
            for node in ast.walk(annotation)
        )


def test_episode_boundary_annotations_resolve_at_runtime() -> None:
    """Episode-boundary annotations remain usable by runtime schema tooling."""
    from robot_sf.benchmark import (
        map_runner,
        map_runner_episode,
        map_runner_static_deadlock,
        map_runner_worker,
    )

    assert typing.get_type_hints(map_runner._run_map_episode)["return"] is EpisodeRecordDict
    assert typing.get_type_hints(map_runner_episode.run_map_episode)["return"] is EpisodeRecordDict
    assert typing.get_type_hints(map_runner_worker.execute_map_job)["return"] is EpisodeRecordDict
    assert (
        typing.get_type_hints(map_runner_static_deadlock.static_deadlock_trace_fields)[
            "planner_decision_trace"
        ]
        == list[PlannerDecisionTraceEntry]
    )
