"""Diagnostic comparator for CALF/LegNav-inspired local-navigation evaluation.

The comparator keeps a Robot SF learned-policy smoke under two paired
observation contracts: ideal state and perception-limited state. It reuses
the existing step-diagnostics trace rather than claiming to execute the
external CALF policy or LegNav simulator. All metrics are therefore local
trace metrics with explicit proxy and unavailable statuses.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

CALF_LEGNAV_COMPARATOR_SCHEMA_VERSION = "calf_legnav_comparator.v1"
CALF_LEGNAV_SOURCE_URL = "https://arxiv.org/abs/2607.27922"
CALF_LEGNAV_SOURCE_TITLE = "Learning Social Robot Navigation By Sensing Human Legs"
CALF_LEGNAV_CLAIM_BOUNDARY = (
    "Diagnostic paired Robot SF policy smoke only; this does not execute CALF, LegNav, "
    "a TurtleBot 4, or a calibrated leg sensor and is not benchmark, safety, or transfer evidence."
)
CALF_LEGNAV_EVIDENCE_STATUS = "diagnostic-only"
CONDITION_IDEAL = "perfect_perception"
CONDITION_SENSOR = "sensor_limited"
CONDITIONS = (CONDITION_IDEAL, CONDITION_SENSOR)


def canonical_config_digest(config: Mapping[str, Any]) -> str:
    """Return a SHA-256 digest for a JSON-compatible comparator config."""
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any) -> Mapping[str, Any]:
    """Return a mapping value or an empty mapping for malformed optional fields."""
    return value if isinstance(value, Mapping) else {}


def _finite_number(value: Any) -> float | None:
    """Return a finite float, preserving unavailable values as ``None``."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _trace_rows(trace: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return trace rows while rejecting a missing or malformed step list."""
    rows = trace.get("steps")
    if not isinstance(rows, list):
        raise ValueError("trace.steps must be a list")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("trace.steps must contain only mapping rows")
    return list(rows)


def _metric(
    *,
    value: float | None,
    status: str,
    units: str,
    source: str,
    mapping_class: str,
    reason: str | None = None,
) -> dict[str, Any]:
    """Build one metric row with explicit missing-data semantics.

    Returns:
        A schema-shaped metric mapping.
    """
    row: dict[str, Any] = {
        "value": value,
        "status": status,
        "units": units,
        "source": source,
        "mapping_class": mapping_class,
    }
    if reason is not None:
        row["reason"] = reason
    return row


def _distance_values(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    """Extract finite ground-truth robot-human distances from trace rows.

    Returns:
        Finite distances in trace order.
    """
    values: list[float] = []
    for row in rows:
        for key in ("min_robot_ped_distance", "post_step_min_robot_ped_distance"):
            value = _finite_number(row.get(key))
            if value is not None:
                values.append(value)
    return values


def _action_values(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Extract finite two-channel actions, preferring environment actions.

    Returns:
        An ``(n, 2)`` array, or an empty array when no action is usable.
    """
    actions: list[list[float]] = []
    for row in rows:
        raw = row.get("env_action", row.get("policy_command"))
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            values = [_finite_number(item) for item in raw]
            if len(values) >= 2 and values[0] is not None and values[1] is not None:
                actions.append([values[0], values[1]])
    return np.asarray(actions, dtype=float).reshape(-1, 2) if actions else np.empty((0, 2))


def _observation_contract(
    trace: Mapping[str, Any], *, expected_condition: str | None = None
) -> dict[str, Any]:
    """Summarize the observed evidence class and perturbation profile.

    When ``expected_condition`` is supplied, the observed evidence class must
    match the paired slot the trace was assigned to. A mismatch (for example a
    ``perfect_perception`` slot whose trace is actually ``perception_limited``)
    fails closed instead of silently reporting a mislabelled pair.

    Returns:
        A condition and observation-provenance mapping.
    """
    rows = _trace_rows(trace)
    classes = {
        str(_mapping(row.get("observed_observation")).get("evidence_class"))
        for row in rows
        if _mapping(row.get("observed_observation")).get("evidence_class") is not None
    }
    profiles = sorted(
        {
            str(_mapping(row.get("observed_observation")).get("noise_profile"))
            for row in rows
            if _mapping(row.get("observed_observation")).get("noise_profile") is not None
        }
    )
    if classes == {"ideal_state"}:
        status = "available"
        condition = CONDITION_IDEAL
    elif classes == {"perception_limited"}:
        status = "available"
        condition = CONDITION_SENSOR
    else:
        status = "unavailable"
        condition = "unknown"
    mismatch_reason: str | None = None
    if expected_condition is not None and condition != expected_condition:
        status = "unavailable"
        mismatch_reason = (
            f"observed observation contract {condition!r} does not match the paired "
            f"slot {expected_condition!r}; the pair is not a valid contrast"
        )
    config = _mapping(trace.get("observation_perturbation_config"))
    return {
        "condition": condition,
        "expected_condition": expected_condition,
        "condition_binding": (
            "unavailable"
            if mismatch_reason
            else ("matched" if expected_condition is not None else "not_checked")
        ),
        **({"reason": mismatch_reason} if mismatch_reason else {}),
        "status": status,
        "evidence_classes": sorted(classes),
        "noise_profiles": profiles,
        "config": dict(config),
        "observed_actor_count": {
            "min": min(
                (
                    int(_mapping(row.get("observation_perturbation")).get("observed_actor_count"))
                    for row in rows
                    if _finite_number(
                        _mapping(row.get("observation_perturbation")).get("observed_actor_count")
                    )
                    is not None
                ),
                default=None,
            ),
            "max": max(
                (
                    int(_mapping(row.get("observation_perturbation")).get("observed_actor_count"))
                    for row in rows
                    if _finite_number(
                        _mapping(row.get("observation_perturbation")).get("observed_actor_count")
                    )
                    is not None
                ),
                default=None,
            ),
        },
    }


def _execution_block(trace: Mapping[str, Any]) -> dict[str, Any]:
    """Return explicit execution and fallback/degraded state from one trace."""
    fallback = _mapping(trace.get("fallback_degraded_status"))
    reported = fallback.get("reported_fallback_or_degraded")
    if reported is True:
        status = "blocked"
        reason = "trace reports fallback_or_degraded execution"
    elif reported is False:
        status = "available"
        reason = None
    else:
        status = "blocked"
        reason = "trace lacks an explicit fallback_or_degraded verdict"
    return {
        "status": status,
        "reason": reason,
        "planner_execution_mode": trace.get("planner_execution_mode"),
        "fallback_degraded_status": dict(fallback),
    }


def _condition_metrics(
    trace: Mapping[str, Any],
    *,
    personal_space_radius_m: float,
    dt_s: float,
) -> dict[str, dict[str, Any]]:
    """Compute paired local metrics from one diagnostic trace.

    Returns:
        Metric mappings keyed by the stable report metric name.
    """
    rows = _trace_rows(trace)
    execution = _execution_block(trace)
    execution_status = str(execution["status"])
    unavailable_reason = execution.get("reason")

    if not rows:
        missing_reason = "trace has no executed steps"
        return {
            name: _metric(
                value=None,
                status="unavailable",
                units=units,
                source=source,
                mapping_class=mapping,
                reason=missing_reason,
            )
            for name, units, source, mapping in (
                (
                    "success_rate",
                    "fraction",
                    "trace.done_info.success",
                    "exact_local",
                ),
                (
                    "collision_rate",
                    "fraction",
                    "trace.collision_flags",
                    "exact_local",
                ),
                (
                    "minimum_human_distance_m",
                    "m",
                    "trace.ground_truth_simulator_distance",
                    "qualified_proxy",
                ),
                (
                    "personal_space_compliance_rate",
                    "fraction",
                    "trace.ground_truth_simulator_distance",
                    "qualified_proxy",
                ),
                (
                    "angular_jerk_rad_s3",
                    "rad/s^3",
                    "trace.env_action",
                    "qualified_proxy",
                ),
                (
                    "action_smoothness_l2",
                    "action_units/step",
                    "trace.env_action",
                    "exact_local",
                ),
                (
                    "timeout_rate",
                    "fraction",
                    "trace.done_info.truncated",
                    "exact_local",
                ),
            )
        }

    def row(
        *,
        value: float | None,
        units: str,
        source: str,
        mapping: str,
        reason: str | None = unavailable_reason,
    ) -> dict[str, Any]:
        """Apply common execution status to a computed row.

        Returns:
            A metric row with blocked or unavailable semantics applied.
        """
        if execution_status == "blocked":
            return _metric(
                value=None,
                status="blocked",
                units=units,
                source=source,
                mapping_class=mapping,
                reason=reason,
            )
        return _metric(
            value=value,
            status="available" if value is not None else "unavailable",
            units=units,
            source=source,
            mapping_class=mapping,
            reason=None if value is not None else (reason or "required trace field unavailable"),
        )

    success_values = [bool(row_item.get("is_success")) for row_item in rows]
    done_info = _mapping(trace.get("done_info"))
    success = bool(done_info.get("success")) or any(success_values)
    collision = any(
        bool(row_item.get(key))
        for row_item in rows
        for key in ("is_pedestrian_collision", "is_obstacle_collision", "is_robot_collision")
    )
    horizon = _finite_number(trace.get("horizon"))
    horizon_reached = horizon is not None and horizon > 0.0 and len(rows) >= int(horizon)
    truncated = (
        bool(done_info.get("truncated"))
        or any(bool(row_item.get("truncated")) for row_item in rows)
        or (horizon_reached and not success)
    )
    distances = _distance_values(rows)
    actions = _action_values(rows)
    metric_rows = {
        "success_rate": row(
            value=float(success),
            units="fraction",
            source="trace.done_info.success|trace.is_success",
            mapping="exact_local",
        ),
        "collision_rate": row(
            value=float(collision),
            units="fraction",
            source="trace.collision_flags",
            mapping="exact_local",
        ),
        "minimum_human_distance_m": row(
            value=min(distances) if distances else None,
            units="m",
            source="trace.ground_truth_simulator_distance",
            mapping="qualified_proxy",
            reason="ground-truth simulator distance is not present",
        ),
        "personal_space_compliance_rate": row(
            value=(
                float(np.mean(np.asarray(distances) >= personal_space_radius_m))
                if distances
                else None
            ),
            units="fraction",
            source="trace.ground_truth_simulator_distance",
            mapping="qualified_proxy",
            reason="ground-truth simulator distance is not present",
        ),
        "angular_jerk_rad_s3": row(
            value=(
                float(np.mean(np.abs(np.diff(actions[:, 1], n=2))) / dt_s**2)
                if actions.shape[0] >= 3
                else None
            ),
            units="rad/s^3",
            source="trace.env_action[1]",
            mapping="qualified_proxy",
            reason="at least three finite action rows are required",
        ),
        "action_smoothness_l2": row(
            value=(
                float(np.mean(np.linalg.norm(np.diff(actions, axis=0), axis=1)))
                if actions.shape[0] >= 2
                else None
            ),
            units="action_units/step",
            source="trace.env_action",
            mapping="exact_local",
            reason="at least two finite action rows are required",
        ),
        "timeout_rate": row(
            value=float(truncated and not success),
            units="fraction",
            source="trace.done_info.truncated|trace.truncated|trace.horizon",
            mapping="exact_local",
        ),
    }
    return metric_rows


def _unsupported_fields() -> list[dict[str, str]]:
    """Return CALF/LegNav fields not reproduced by this Robot SF diagnostic."""
    return [
        {
            "field": "leg_lidar_segmentation_and_gait_state",
            "status": "unavailable",
            "reason": "Robot SF trace exposes simulator pedestrian state, not leg-segmented scans.",
        },
        {
            "field": "legnav_pedestrian_gait_model",
            "status": "unavailable",
            "reason": "The external LegNav pedestrian gait model is not hydrated here.",
        },
        {
            "field": "turtlebot_4_zero_shot_deployment",
            "status": "unavailable",
            "reason": "No TurtleBot 4 hardware, deployment checkpoint, or runtime is in scope.",
        },
        {
            "field": "source_policy_training_algorithm_and_bounds",
            "status": "unavailable",
            "reason": "The comparator evaluates a Robot SF PPO candidate; CALF training settings are not imported.",
        },
        {
            "field": "social_force_hsfm_calibration_or_parity",
            "status": "unavailable",
            "reason": "No Social Force or HSFM calibration is established for this paired diagnostic.",
        },
    ]


def _condition_report(
    trace: Mapping[str, Any],
    *,
    expected_condition: str,
    personal_space_radius_m: float,
    dt_s: float,
) -> dict[str, Any]:
    """Build the provenance and metrics block for one condition trace.

    Returns:
        A schema-shaped condition report.
    """
    observation = _observation_contract(trace, expected_condition=expected_condition)
    execution = _execution_block(trace)
    return {
        "status": (
            "available"
            if observation["status"] == "available" and execution["status"] == "available"
            else "blocked"
        ),
        "scenario_id": trace.get("scenario_id"),
        "seed": trace.get("seed"),
        "candidate": trace.get("candidate"),
        "algo": trace.get("algo"),
        "observation_contract": observation,
        "execution": execution,
        "metrics": _condition_metrics(
            trace,
            personal_space_radius_m=personal_space_radius_m,
            dt_s=dt_s,
        ),
    }


def build_calf_legnav_comparator_report(
    perfect_trace: Mapping[str, Any],
    sensor_trace: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    input_refs: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed paired CALF/LegNav-inspired diagnostic report.

    The two traces must use the same candidate, scenario, and seed. The
    comparator intentionally reports a one-episode paired smoke, not an
    aggregate benchmark result or a source-policy reproduction.

    Returns:
        A schema-shaped diagnostic report with explicit proxy and unavailable fields.
    """
    identity_fields = ("candidate", "scenario_id", "seed", "horizon")
    mismatches = [
        field for field in identity_fields if perfect_trace.get(field) != sensor_trace.get(field)
    ]
    if mismatches:
        raise ValueError("paired traces disagree on: " + ", ".join(mismatches))

    config_identity = {
        "candidate": "candidate",
        "scenario_name": "scenario_id",
        "seed": "seed",
        "horizon": "horizon",
    }
    config_mismatches = [
        config_field
        for config_field, trace_field in config_identity.items()
        if config.get(config_field) != perfect_trace.get(trace_field)
    ]
    if config_mismatches:
        raise ValueError("trace identity disagrees with config: " + ", ".join(config_mismatches))

    personal_space_radius_m = _finite_number(config.get("personal_space_radius_m"))
    if personal_space_radius_m is None or personal_space_radius_m <= 0.0:
        raise ValueError("personal_space_radius_m must be a positive finite number")
    dt_s = _finite_number(config.get("dt_s", 0.1))
    if dt_s is None or dt_s <= 0.0:
        raise ValueError("dt_s must be a positive finite number")

    conditions = {
        CONDITION_IDEAL: _condition_report(
            perfect_trace,
            expected_condition=CONDITION_IDEAL,
            personal_space_radius_m=personal_space_radius_m,
            dt_s=dt_s,
        ),
        CONDITION_SENSOR: _condition_report(
            sensor_trace,
            expected_condition=CONDITION_SENSOR,
            personal_space_radius_m=personal_space_radius_m,
            dt_s=dt_s,
        ),
    }
    metric_names = sorted(
        set(conditions[CONDITION_IDEAL]["metrics"]) | set(conditions[CONDITION_SENSOR]["metrics"])
    )
    paired_metrics: dict[str, Any] = {}
    for name in metric_names:
        left = conditions[CONDITION_IDEAL]["metrics"][name]
        right = conditions[CONDITION_SENSOR]["metrics"][name]
        left_value = _finite_number(left.get("value"))
        right_value = _finite_number(right.get("value"))
        paired_metrics[name] = {
            "perfect_perception": left,
            "sensor_limited": right,
            "sensor_minus_perfect": (
                right_value - left_value
                if left_value is not None and right_value is not None
                else None
            ),
            "delta_status": (
                "available" if left_value is not None and right_value is not None else "unavailable"
            ),
        }

    trace_status = [conditions[name]["status"] for name in CONDITIONS]
    top_status = "available" if all(status == "available" for status in trace_status) else "blocked"
    return {
        "schema_version": CALF_LEGNAV_COMPARATOR_SCHEMA_VERSION,
        "issue": 7318,
        "status": top_status,
        "evidence_status": CALF_LEGNAV_EVIDENCE_STATUS,
        "claim_boundary": CALF_LEGNAV_CLAIM_BOUNDARY,
        "method_card": {
            "source_title": CALF_LEGNAV_SOURCE_TITLE,
            "source_url": CALF_LEGNAV_SOURCE_URL,
            "source_method": {
                "observation": "2D LiDAR leg patterns; exact source sensor dimensions are not imported.",
                "temporal_features": "Source temporal-feature details are not imported into the Robot SF trace adapter.",
                "architecture": "CALF combines convolution, attention, and multilayer perceptron components.",
                "action_bounds": "Source action bounds are not imported; local bounds remain candidate-specific.",
                "training": "Deep reinforcement learning in the external LegNav simulator; exact recipe not reproduced.",
                "simulator": "LegNav is a lightweight 2D simulator with LiDAR ray tracing and a pedestrian gait model.",
                "social_force_hsfm": "No Social Force or HSFM calibration/parity is assumed by this comparator.",
                "deployment": "The source reports zero-shot TurtleBot 4 deployment; this comparator does not reproduce it.",
            },
            "local_transfer_question": (
                "Which observation and evaluation fields can be compared in Robot SF without treating "
                "sensor, embodiment, or simulator differences as validated transfer?"
            ),
        },
        "provenance": {
            "candidate": perfect_trace.get("candidate"),
            "algo": perfect_trace.get("algo"),
            "scenario_id": perfect_trace.get("scenario_id"),
            "seed": perfect_trace.get("seed"),
            "horizon": perfect_trace.get("horizon"),
            "config_digest": canonical_config_digest(config),
            "input_refs": dict(input_refs or {}),
            "paired_episode_count": 1,
            "uncertainty": {
                "status": "unavailable",
                "reason": "One paired episode is a deterministic smoke and cannot estimate variance or confidence intervals.",
            },
        },
        "conditions": conditions,
        "paired_metrics": paired_metrics,
        "unsupported_fields": _unsupported_fields(),
        "zero_shot_transfer": {
            "status": "unavailable",
            "reason": "No CALF checkpoint, LegNav simulator, leg-sensor contract, or TurtleBot 4 runtime is hydrated.",
        },
        "comparison_interpretation": {
            "status": CALF_LEGNAV_EVIDENCE_STATUS,
            "result": "paired_observation_contract_smoke",
            "not_claims": [
                "CALF policy reproduction",
                "LegNav simulator parity",
                "sensor realism or calibration",
                "Social Force or HSFM parity",
                "universal policy or planner ranking",
                "real-world safety or zero-shot transfer",
            ],
        },
    }


__all__ = [
    "CALF_LEGNAV_CLAIM_BOUNDARY",
    "CALF_LEGNAV_COMPARATOR_SCHEMA_VERSION",
    "CALF_LEGNAV_SOURCE_TITLE",
    "CALF_LEGNAV_SOURCE_URL",
    "CONDITION_IDEAL",
    "CONDITION_SENSOR",
    "build_calf_legnav_comparator_report",
    "canonical_config_digest",
]
