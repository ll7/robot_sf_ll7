#!/usr/bin/env python3
"""Run the deterministic four-arm smoke for issue #8072.

The report is deliberately diagnostic-only.  It exercises the H=1 heading
baseline, H=2 and H=3 observation reconstruction, the #8073 candidate-provider
envelope, and the evaluator-only #8065 oracle-component upper bound on one
synthetic transition fixture.  It does not run a simulator campaign or make a
prediction-quality claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from robot_sf.prediction import (
    ControllerMutationFlags,
    DynamicsParameters,
    ForceComponents,
    ForceTimeRobotState,
    GoalCandidateProvider,
    GoalCandidateProviderConfig,
    GoalCandidateSource,
    GoalChangeKind,
    GoalForceEstimate,
    GoalForceEstimatorMode,
    GoalForceInverseConfig,
    GoalForceInverseEstimator,
    GoalForceObservation,
    ObservableForceComponent,
    OracleTransitionTraceV1,
    PublicGoalCandidateRecord,
    SpeedCap,
    SpeedCapStatus,
    TransitionBoundary,
    TransitionBoundaryKind,
    stable_config_hash,
)
from robot_sf.prediction._contract_utils import reject_unknown_keys, stable_digest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "robot_sf.goal_force_inverse_smoke.v1"
REPORT_SCHEMA_VERSION = "robot_sf.goal_force_inverse_smoke_report.v1"
CLAIM_BOUNDARY = "implementation_integrity_smoke"
COMPONENT_TYPES = (
    "social",
    "obstacle",
    "pedestrian_robot",
    "group",
    "adversarial",
)
SOURCE_FILES = (
    "robot_sf/prediction/goal_force_inverse_dynamics.py",
    "robot_sf/prediction/goal_candidate_provider.py",
    "robot_sf/prediction/oracle_transition_trace.py",
    "scripts/validation/run_goal_force_inverse_smoke.py",
)
ARM_NAMES = (
    "h1_heading_baseline",
    "h2_observation_reconstructed",
    "h3_observation_reconstructed",
    "oracle_component_upper_bound",
)
CONFIG_KEYS = frozenset(
    {
        "schema_version",
        "issue",
        "seed",
        "fixture_id",
        "track_id",
        "tracking_epoch_id",
        "timestamps_s",
        "positions_xy",
        "velocities_xy",
        "true_goal_force_xy",
        "preferred_speed_mps",
        "relaxation_time_s",
        "desired_force_factor",
        "max_speed_mps",
        "known_force_components",
        "candidate_records",
        "estimator",
    }
)
COMPONENT_KEYS = frozenset({"component_id", "component_type", "force_xy"})
RECORD_KEYS = frozenset({"source", "source_id", "position"})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return dict(value)


def _xy(value: Any, field_name: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or len(value) != 2:
        raise ValueError(f"{field_name} must contain exactly two values")
    result = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _positive(value: Any, field_name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{field_name} must be finite and positive")
    return result


def _load_config(config_path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"cannot read smoke config {config_path}: {exc}") from exc
    config = _mapping(payload, "smoke config")
    reject_unknown_keys(config, set(CONFIG_KEYS), "goal_force_inverse_smoke")
    _require(config.get("schema_version") == SCHEMA_VERSION, "smoke config schema_version drifted")
    _require(config.get("issue") == 8072, "smoke config issue must be 8072")
    _require(type(config.get("seed")) is int, "smoke config seed must be an integer")
    for field_name in (
        "fixture_id",
        "track_id",
        "tracking_epoch_id",
    ):
        _require(isinstance(config.get(field_name), str), f"{field_name} must be text")
    for field_name in ("timestamps_s", "positions_xy", "velocities_xy"):
        _require(
            isinstance(config.get(field_name), Sequence)
            and not isinstance(config.get(field_name), (str, bytes)),
            f"{field_name} must be an array",
        )
    lengths = {
        len(config[field_name]) for field_name in ("timestamps_s", "positions_xy", "velocities_xy")
    }
    _require(lengths == {3}, "the smoke fixture must contain exactly three snapshots")
    timestamps = [float(value) for value in config["timestamps_s"]]
    _require(all(math.isfinite(value) for value in timestamps), "timestamps must be finite")
    _require(timestamps == sorted(timestamps), "timestamps must be ordered")
    _require(len(set(timestamps)) == len(timestamps), "timestamps must be distinct")
    config["positions_xy"] = [_xy(value, "positions_xy[]") for value in config["positions_xy"]]
    config["velocities_xy"] = [_xy(value, "velocities_xy[]") for value in config["velocities_xy"]]
    config["true_goal_force_xy"] = _xy(config["true_goal_force_xy"], "true_goal_force_xy")
    for field_name in (
        "preferred_speed_mps",
        "relaxation_time_s",
        "desired_force_factor",
        "max_speed_mps",
    ):
        config[field_name] = _positive(config[field_name], field_name)

    raw_components = config["known_force_components"]
    _require(
        isinstance(raw_components, Sequence) and not isinstance(raw_components, (str, bytes)),
        "known_force_components must be an array",
    )
    components: list[dict[str, Any]] = []
    for index, raw_component in enumerate(raw_components):
        component = _mapping(raw_component, f"known_force_components[{index}]")
        reject_unknown_keys(component, set(COMPONENT_KEYS), f"known_force_components[{index}]")
        _require(set(component) == COMPONENT_KEYS, f"known_force_components[{index}] is incomplete")
        component["component_id"] = str(component["component_id"])
        component["component_type"] = str(component["component_type"])
        component["force_xy"] = _xy(component["force_xy"], f"component[{index}].force_xy")
        components.append(component)
    _require(
        tuple(component["component_type"] for component in components) == COMPONENT_TYPES,
        "known_force_components must provide the canonical ordered roster",
    )
    _require(
        len({component["component_id"] for component in components}) == len(components),
        "known_force_components IDs must be unique",
    )
    config["known_force_components"] = components

    raw_records = config["candidate_records"]
    _require(
        isinstance(raw_records, Sequence) and not isinstance(raw_records, (str, bytes)),
        "candidate_records must be an array",
    )
    records: list[dict[str, Any]] = []
    for index, raw_record in enumerate(raw_records):
        record = _mapping(raw_record, f"candidate_records[{index}]")
        reject_unknown_keys(record, set(RECORD_KEYS), f"candidate_records[{index}]")
        _require(set(record) == RECORD_KEYS, f"candidate_records[{index}] is incomplete")
        record["source"] = str(record["source"])
        record["source_id"] = str(record["source_id"])
        record["position"] = _xy(record["position"], f"candidate_records[{index}].position")
        records.append(record)
    config["candidate_records"] = records
    config["estimator"] = _mapping(config["estimator"], "estimator")
    return config


def _source_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative_path in SOURCE_FILES:
        path = REPO_ROOT / relative_path
        try:
            hashes[relative_path] = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError(f"cannot hash source file {relative_path}: {exc}") from exc
    return hashes


def _observations(config: Mapping[str, Any]) -> tuple[GoalForceObservation, ...]:
    return tuple(
        GoalForceObservation(
            track_id=str(config["track_id"]),
            tracking_epoch_id=str(config["tracking_epoch_id"]),
            timestamp_s=float(timestamp),
            step_index=index,
            position_xy=position,
            velocity_xy=velocity,
        )
        for index, (timestamp, position, velocity) in enumerate(
            zip(
                config["timestamps_s"], config["positions_xy"], config["velocities_xy"], strict=True
            )
        )
    )


def _components(
    config: Mapping[str, Any],
    component_config_hash: str,
) -> tuple[ObservableForceComponent, ...]:
    return tuple(
        ObservableForceComponent(
            component_id=str(value["component_id"]),
            component_type=str(value["component_type"]),
            force_xy=value["force_xy"],
            config_hash=component_config_hash,
        )
        for value in config["known_force_components"]
    )


def _candidate_generation(config: Mapping[str, Any], latest: GoalForceObservation) -> Any:
    provider = GoalCandidateProvider(
        GoalCandidateProviderConfig(
            enabled_sources=(GoalCandidateSource.MAP_DESTINATION_ZONE,),
            unknown_enabled=True,
            final_destination_cap=8,
        )
    )
    records = tuple(
        PublicGoalCandidateRecord(
            source=str(value["source"]),
            source_id=str(value["source_id"]),
            position=value["position"],
        )
        for value in config["candidate_records"]
    )
    return provider.generate(records, observed_position_global=latest.position_xy)


def _vector_sum(values: Sequence[tuple[float, float]]) -> tuple[float, float]:
    return (sum(value[0] for value in values), sum(value[1] for value in values))


def _oracle_trace(
    config: Mapping[str, Any],
    observations: Sequence[GoalForceObservation],
) -> OracleTransitionTraceV1:
    known_by_type = {
        str(value["component_type"]): value["force_xy"]
        for value in config["known_force_components"]
    }
    known_total = _vector_sum(tuple(known_by_type.values()))
    goal_force = config["true_goal_force_xy"]
    registry_total = _vector_sum((goal_force, known_total))
    pre = observations[0]
    post = observations[1]
    if pre.position_xy is None or pre.velocity_xy is None:
        raise ValueError("smoke pre-transition observation must be visible")
    if post.position_xy is None or post.velocity_xy is None:
        raise ValueError("smoke post-transition observation must be visible")
    return OracleTransitionTraceV1(
        episode_id="goal-force-smoke",
        transition_id="goal-force-smoke:t0",
        transition_step_index=pre.step_index,
        simulator_pedestrian_id="sim-ped-1",
        actor_track_id=pre.track_id,
        actor_tracking_epoch_id=pre.tracking_epoch_id,
        backend="synthetic_fixture",
        pre_behavior=TransitionBoundary(
            boundary=TransitionBoundaryKind.PRE_BEHAVIOR,
            timestamp_s=pre.timestamp_s,
            step_index=pre.step_index,
            position_xy=pre.position_xy,
            velocity_xy=pre.velocity_xy,
            active_goal_xy=(10.0, 0.0),
            route_waypoint_index=None,
            goal_threshold_reached=False,
        ),
        post_behavior_pre_force=TransitionBoundary(
            boundary=TransitionBoundaryKind.POST_BEHAVIOR_PRE_FORCE,
            timestamp_s=pre.timestamp_s,
            step_index=pre.step_index,
            force_time_robot_state=ForceTimeRobotState(),
            mutation_flags=ControllerMutationFlags(),
            position_xy=pre.position_xy,
            velocity_xy=pre.velocity_xy,
            active_goal_xy=(10.0, 0.0),
            route_waypoint_index=None,
            goal_threshold_reached=False,
        ),
        post_integration=TransitionBoundary(
            boundary=TransitionBoundaryKind.POST_INTEGRATION,
            timestamp_s=post.timestamp_s,
            step_index=post.step_index,
            position_xy=post.position_xy,
            velocity_xy=post.velocity_xy,
            active_goal_xy=(10.0, 0.0),
            route_waypoint_index=None,
            goal_threshold_reached=False,
        ),
        force_components=ForceComponents(
            social_force_xy=known_by_type["social"],
            goal_force_xy=goal_force,
            obstacle_force_xy=known_by_type["obstacle"],
            pedestrian_robot_force_xy=known_by_type["pedestrian_robot"],
            adversarial_force_xy=known_by_type["adversarial"],
            group_force_xy=known_by_type["group"],
            registry_total_force_xy=registry_total,
            final_pre_cap_force_xy=registry_total,
            uncapped_velocity_xy=post.velocity_xy,
            applied_velocity_xy=post.velocity_xy,
        ),
        dynamics=DynamicsParameters(
            preferred_speed_mps=config["preferred_speed_mps"],
            relaxation_time_s=config["relaxation_time_s"],
            desired_force_factor=config["desired_force_factor"],
            goal_threshold_m=0.2,
            goal_threshold_reached=False,
        ),
        speed_cap=SpeedCap(
            status=SpeedCapStatus.NOT_APPLIED,
            max_speed_mps=config["max_speed_mps"],
            uncapped_speed_mps=math.hypot(*post.velocity_xy),
            applied_speed_mps=math.hypot(*post.velocity_xy),
        ),
        goal_change_kind=GoalChangeKind.NONE,
        exact_inverse_eligible=True,
        exact_inverse_reasons=(),
    )


def _angle_error(left: tuple[float, float], right: tuple[float, float]) -> float | None:
    left_norm = math.hypot(*left)
    right_norm = math.hypot(*right)
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    cosine = (left[0] * right[0] + left[1] * right[1]) / (left_norm * right_norm)
    return math.acos(max(-1.0, min(1.0, cosine)))


def _coverage_95(
    estimate: GoalForceEstimate,
    true_force: tuple[float, float],
) -> bool | None:
    if estimate.force_estimate is None:
        return None
    covariance = estimate.force_estimate.covariance_xy
    determinant = covariance[0][0] * covariance[1][1] - covariance[0][1] ** 2
    if determinant <= 0.0:
        return False
    error = (
        estimate.force_estimate.mean_xy[0] - true_force[0],
        estimate.force_estimate.mean_xy[1] - true_force[1],
    )
    mahalanobis = (
        covariance[1][1] * error[0] ** 2
        - 2.0 * covariance[0][1] * error[0] * error[1]
        + covariance[0][0] * error[1] ** 2
    ) / determinant
    return mahalanobis <= 5.991464547107979


def _metrics(
    estimate: GoalForceEstimate,
    true_force: tuple[float, float],
) -> dict[str, Any]:
    if estimate.force_estimate is None:
        return {
            "force_mae": None,
            "force_rmse": None,
            "direction_error_rad": None,
            "covariance_coverage_95": None,
        }
    difference = (
        estimate.force_estimate.mean_xy[0] - true_force[0],
        estimate.force_estimate.mean_xy[1] - true_force[1],
    )
    return {
        "force_mae": (abs(difference[0]) + abs(difference[1])) / 2.0,
        "force_rmse": math.sqrt((difference[0] ** 2 + difference[1] ** 2) / 2.0),
        "direction_error_rad": _angle_error(estimate.force_estimate.mean_xy, true_force),
        "covariance_coverage_95": _coverage_95(estimate, true_force),
    }


def _run_actor_arm(
    name: str,
    base_config: GoalForceInverseConfig,
    history: Sequence[GoalForceObservation],
    components: Sequence[Any],
    candidate_generation: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    history_length = {
        "h1_heading_baseline": 1,
        "h2_observation_reconstructed": 2,
        "h3_observation_reconstructed": 3,
    }[name]
    estimator = GoalForceInverseEstimator(replace(base_config, history_length=history_length))
    started = time.perf_counter()
    estimate = estimator.estimate(
        history,
        known_force_components=components,
        candidate_set=candidate_generation,
        preferred_speed_mps=config["preferred_speed_mps"],
        relaxation_time_s=config["relaxation_time_s"],
        desired_force_factor=config["desired_force_factor"],
        max_speed_mps=config["max_speed_mps"],
    )
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "history_length": history_length,
        "history_age_s": history[-1].timestamp_s - history[-history_length].timestamp_s,
        "mode": estimate.mode.value,
        "information_source": "actor_observation_only",
        "inferred_preferred_speed_mps": estimate.inferred_preferred_speed_mps,
        "component_availability": (
            estimate.reconstruction.to_dict()
            if estimate.reconstruction is not None
            else [item.to_dict() for item in estimate.component_diagnostics]
        ),
        "saturation_status": estimate.speed_cap_status.value,
        "censoring_state": estimate.censoring_state.value,
        "arrival_probability": estimate.arrival_probability,
        "braking_probability": estimate.braking_probability,
        "runtime_ms": elapsed_ms,
        "metrics": _metrics(estimate, config["true_goal_force_xy"]),
        "estimate": estimate.to_dict(),
        "estimate_digest": estimate.content_digest,
    }


def _run_oracle_arm(
    estimator: GoalForceInverseEstimator,
    trace: OracleTransitionTraceV1,
    true_force: tuple[float, float],
) -> dict[str, Any]:
    started = time.perf_counter()
    estimate = estimator.estimate_from_oracle_trace(trace)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "history_length": 2,
        "history_age_s": trace.post_integration.timestamp_s
        - trace.post_behavior_pre_force.timestamp_s,
        "mode": estimate.mode.value,
        "information_source": "evaluator_oracle_component_upper_bound",
        "inferred_preferred_speed_mps": estimate.inferred_preferred_speed_mps,
        "component_availability": [item.to_dict() for item in estimate.component_diagnostics],
        "saturation_status": estimate.speed_cap_status.value,
        "censoring_state": estimate.censoring_state.value,
        "arrival_probability": estimate.arrival_probability,
        "braking_probability": estimate.braking_probability,
        "runtime_ms": elapsed_ms,
        "metrics": _metrics(estimate, true_force),
        "estimate": estimate.to_dict(),
        "estimate_digest": estimate.content_digest,
    }


def _leakage_canary(
    base_config: GoalForceInverseConfig,
    history: Sequence[GoalForceObservation],
    components: Sequence[Any],
    candidate_generation: Any,
    config: Mapping[str, Any],
    trace: OracleTransitionTraceV1,
) -> dict[str, Any]:
    estimator = GoalForceInverseEstimator(replace(base_config, history_length=2))
    kwargs = {
        "known_force_components": components,
        "candidate_set": candidate_generation,
        "preferred_speed_mps": config["preferred_speed_mps"],
        "relaxation_time_s": config["relaxation_time_s"],
        "desired_force_factor": config["desired_force_factor"],
        "max_speed_mps": config["max_speed_mps"],
    }
    before = estimator.estimate(history, **kwargs)
    randomized_goal = (123.0, -456.0)
    randomized_pre = replace(trace.pre_behavior, active_goal_xy=randomized_goal)
    randomized_post_behavior = replace(
        trace.post_behavior_pre_force, active_goal_xy=randomized_goal
    )
    randomized_post = replace(trace.post_integration, active_goal_xy=randomized_goal)
    randomized_trace = replace(
        trace,
        pre_behavior=randomized_pre,
        post_behavior_pre_force=randomized_post_behavior,
        post_integration=randomized_post,
    )
    # The mutated oracle trace is intentionally not an input to the actor arm.
    after = estimator.estimate(history, **kwargs)
    return {
        "status": "pass" if before.content_digest == after.content_digest else "fail",
        "actor_digest_before": before.content_digest,
        "actor_digest_after": after.content_digest,
        "randomized_oracle_trace_digest": GoalForceInverseEstimator(
            replace(base_config, history_length=2)
        )
        .estimate_from_oracle_trace(randomized_trace)
        .content_digest,
        "actor_input_contract": "oracle_goal_route_and_trace_are_not_actor_inputs",
    }


def build_report(config_path: Path) -> dict[str, Any]:
    """Build the compact diagnostic report from a strict YAML fixture."""
    config = _load_config(config_path)
    observations = _observations(config)
    base_config = GoalForceInverseConfig.from_mapping(config["estimator"])
    components = _components(config, base_config.config_hash)
    candidate_generation = _candidate_generation(config, observations[-1])
    _require(base_config.enabled, "smoke estimator config must be enabled")
    mode = (
        base_config.mode.value
        if isinstance(base_config.mode, GoalForceEstimatorMode)
        else base_config.mode
    )
    _require(
        mode == "actor_observation_only", "smoke actor arms require actor_observation_only mode"
    )
    trace = _oracle_trace(config, observations)
    arms = {
        name: _run_actor_arm(
            name,
            base_config,
            observations,
            components,
            candidate_generation,
            config,
        )
        for name in ARM_NAMES[:3]
    }
    arms[ARM_NAMES[3]] = _run_oracle_arm(
        GoalForceInverseEstimator(base_config), trace, config["true_goal_force_xy"]
    )
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "issue": 8072,
        "status": "diagnostic_only_valid",
        "claim_boundary": CLAIM_BOUNDARY,
        "fixture_id": config["fixture_id"],
        "seed": config["seed"],
        "track_id": config["track_id"],
        "tracking_epoch_id": config["tracking_epoch_id"],
        "source_hashes": _source_hashes(),
        "smoke_config_digest": stable_config_hash(config),
        "estimator_config": base_config.to_dict(),
        "estimator_config_hash": base_config.config_hash,
        "candidate_generation": candidate_generation.to_dict(),
        "oracle_trace_digest": trace.content_digest,
        "evaluation_truth": {"true_goal_force_xy": list(config["true_goal_force_xy"])},
        "leakage_canary": _leakage_canary(
            base_config,
            observations,
            components,
            candidate_generation,
            config,
            trace,
        ),
        "arms": arms,
        "notes": [
            "Synthetic fixture only; no simulator campaign was run.",
            "Force errors and covariance coverage are diagnostic outputs, not benchmark evidence.",
            "Oracle and actor arms use separate information sources and denominators.",
        ],
    }
    report["report_digest"] = stable_digest(report)
    return report


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the smoke command parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Tracked YAML fixture config.")
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build and emit the diagnostic report."""
    args = build_arg_parser().parse_args(argv)
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    report = build_report(config_path)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        output_path = args.output if args.output.is_absolute() else REPO_ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        print(json.dumps({"status": "valid", "output": str(output_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
