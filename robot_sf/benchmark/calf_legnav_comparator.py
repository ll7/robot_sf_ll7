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
from numbers import Integral, Real
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
    encoded = json.dumps(
        config,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping(value: Any) -> Mapping[str, Any]:
    """Return a mapping value or an empty mapping for malformed optional fields."""
    return value if isinstance(value, Mapping) else {}


def _finite_number(value: Any) -> float | None:
    """Return a finite float, preserving unavailable values as ``None``."""
    if value is None or isinstance(value, bool) or not isinstance(value, Real):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _integer_value(value: Any) -> int | None:
    """Return a strict integer value, rejecting booleans, floats, and strings."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        return None
    return int(value)


def _trace_rows(trace: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return trace rows while rejecting a missing or malformed step list."""
    rows = trace.get("steps")
    if not isinstance(rows, list):
        raise ValueError("trace.steps must be a list")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError("trace.steps must contain only mapping rows")
    return list(rows)


def _trace_completion_reason(trace: Mapping[str, Any]) -> str | None:
    """Return a blocker when a trace is incomplete or has invalid step identity."""
    rows = _trace_rows(trace)
    horizon = _integer_value(trace.get("horizon"))
    if horizon is None or horizon < 1:
        return "trace.horizon must be a positive integer"
    steps = [row.get("step") for row in rows]
    if any(_integer_value(step) is None for step in steps):
        return "trace steps must expose integer step identities"
    if [int(step) for step in steps] != list(range(len(rows))):
        return "trace step identities must be contiguous from zero"
    if len(rows) > horizon:
        return "trace contains more steps than its declared horizon"
    if len(rows) == horizon:
        return None
    done_info = trace.get("done_info")
    if not isinstance(done_info, Mapping) or not any(
        done_info.get(field) is True for field in ("success", "terminated", "truncated")
    ):
        return "trace ended before its horizon without a terminal done_info verdict"
    return None


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


def _distance_values(rows: Sequence[Mapping[str, Any]]) -> tuple[list[float], bool]:
    """Extract one conservative ground-truth distance per executed action.

    The diagnostics producer records the distance before and after each action.
    Those observations overlap between adjacent rows, so treating both fields
    as independent samples would double-weight interior states in compliance
    rates. The per-action representative is the minimum finite distance in the
    interval; if only one field is present, that field is used.

    Returns:
        One finite distance per row that contains at least one distance.
    """
    values: list[float] = []
    malformed = False
    for row in rows:
        row_values: list[float] = []
        for key in ("min_robot_ped_distance", "post_step_min_robot_ped_distance"):
            if key not in row or row[key] is None:
                continue
            value = _finite_number(row[key])
            if value is None or value < 0.0:
                malformed = True
            else:
                row_values.append(value)
        if row_values:
            values.append(min(row_values))
    return values, malformed


def _boolean_values(rows: Sequence[Mapping[str, Any]], field: str) -> tuple[list[bool], bool]:
    """Return boolean field values and whether any present value is malformed."""
    values: list[bool] = []
    malformed = False
    for row in rows:
        if field not in row:
            continue
        value = row[field]
        if isinstance(value, bool):
            values.append(value)
        else:
            malformed = True
    return values, malformed


def _optional_boolean(mapping: Mapping[str, Any], field: str) -> tuple[bool | None, bool]:
    """Return one optional boolean and whether a present value is malformed."""
    if field not in mapping:
        return None, False
    value = mapping[field]
    return (value, False) if isinstance(value, bool) else (None, True)


def _action_values(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, bool]:
    """Extract finite two-channel actions, preferring environment actions.

    Returns:
        An ``(n, 2)`` array, or an empty array when no action is usable, and a
        malformed-input flag.
    """
    actions: list[list[float]] = []
    malformed = False
    for row in rows:
        raw = row.get("env_action")
        if raw is None:
            raw = row.get("policy_command")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            values = [_finite_number(item) for item in raw]
            if len(values) >= 2 and values[0] is not None and values[1] is not None:
                actions.append([values[0], values[1]])
            else:
                malformed = True
        else:
            malformed = True
    return (
        np.asarray(actions, dtype=float).reshape(-1, 2) if actions else np.empty((0, 2)),
        malformed,
    )


def _observed_actor_counts(rows: Sequence[Mapping[str, Any]]) -> tuple[list[int], bool]:
    """Extract non-negative observed actor counts and flag malformed metadata.

    Returns:
        Valid counts and whether any row contained missing or malformed count metadata.
    """
    counts: list[int] = []
    malformed = False
    for row in rows:
        perturbation = row.get("observation_perturbation")
        if not isinstance(perturbation, Mapping) or "observed_actor_count" not in perturbation:
            malformed = True
            continue
        count = _integer_value(perturbation["observed_actor_count"])
        if count is None or count < 0:
            malformed = True
            continue
        counts.append(count)
    return counts, malformed


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
    classes: set[str] = set()
    profiles: set[str] = set()
    malformed_rows = False
    for row in rows:
        observed = row.get("observed_observation")
        if not isinstance(observed, Mapping):
            malformed_rows = True
            continue
        evidence_class = observed.get("evidence_class")
        noise_profile = observed.get("noise_profile")
        if not isinstance(evidence_class, str) or not evidence_class.strip():
            malformed_rows = True
        else:
            classes.add(evidence_class)
        if not isinstance(noise_profile, str) or not noise_profile.strip():
            malformed_rows = True
        else:
            profiles.add(noise_profile)
    if classes == {"ideal_state"}:
        condition = CONDITION_IDEAL
    elif classes == {"perception_limited"}:
        condition = CONDITION_SENSOR
    else:
        condition = "unknown"
    status = "available" if not malformed_rows and condition != "unknown" else "unavailable"
    mismatch_reason: str | None = None
    if expected_condition is not None and condition != expected_condition:
        mismatch_reason = (
            f"observed observation contract {condition!r} does not match the paired "
            f"slot {expected_condition!r}; the pair is not a valid contrast"
        )
        status = "unavailable"
    actor_counts, malformed_actor_counts = _observed_actor_counts(rows)
    malformed_reasons = []
    if malformed_rows:
        malformed_reasons.append(
            "each trace row must expose a typed observed observation evidence class and noise profile"
        )
    if malformed_actor_counts:
        malformed_reasons.append(
            "each trace row must expose a non-negative integer observed_actor_count"
        )
        status = "unavailable"
    malformed_reason = "; ".join(malformed_reasons) or None
    config = _mapping(trace.get("observation_perturbation_config"))
    return {
        "condition": condition,
        "expected_condition": expected_condition,
        "condition_binding": (
            "unavailable"
            if mismatch_reason or malformed_reason
            else ("matched" if expected_condition is not None else "not_checked")
        ),
        **(
            {"reason": mismatch_reason or malformed_reason}
            if mismatch_reason or malformed_reason
            else {}
        ),
        "status": status,
        "evidence_classes": sorted(classes),
        "noise_profiles": sorted(profiles),
        "config": dict(config),
        "observed_actor_count": {
            "min": min(actor_counts, default=None),
            "max": max(actor_counts, default=None),
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
        completion_reason = _trace_completion_reason(trace)
        execution_mode = trace.get("planner_execution_mode")
        if completion_reason is not None:
            status = "blocked"
            reason = completion_reason
        elif execution_mode not in {"native_env_action", "command_adapter", "mixed"}:
            status = "blocked"
            reason = "trace lacks a recognized planner execution mode"
        else:
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

    success_values, malformed_row_success = _boolean_values(rows, "is_success")
    done_info = _mapping(trace.get("done_info"))
    done_success, malformed_done_success = _optional_boolean(done_info, "success")
    success_present = done_success is not None or bool(success_values)
    success = (
        None
        if malformed_row_success or malformed_done_success or not success_present
        else float(bool(done_success) or any(success_values))
    )
    collision_values: list[bool] = []
    malformed_collision = False
    for key in ("is_pedestrian_collision", "is_obstacle_collision", "is_robot_collision"):
        values, malformed = _boolean_values(rows, key)
        collision_values.extend(values)
        malformed_collision = malformed_collision or malformed
    collision = (
        None if malformed_collision or not collision_values else float(any(collision_values))
    )
    horizon = _integer_value(trace.get("horizon"))
    horizon_reached = horizon is not None and horizon > 0 and len(rows) >= horizon
    row_truncated_values, malformed_row_truncated = _boolean_values(rows, "truncated")
    done_truncated, malformed_done_truncated = _optional_boolean(done_info, "truncated")
    truncated = (
        None
        if success is None or malformed_row_truncated or malformed_done_truncated
        else float(
            bool(done_truncated)
            or any(row_truncated_values)
            or (horizon_reached and not bool(success))
        )
    )
    distances, malformed_distances = _distance_values(rows)
    actions, malformed_actions = _action_values(rows)
    boolean_reason = "outcome flags must be booleans when present"
    distance_reason = (
        "distance fields must be finite non-negative numbers when present"
        if malformed_distances
        else "ground-truth simulator distance is not present"
    )
    action_reason = (
        "action fields must contain at least two finite numeric channels"
        if malformed_actions
        else "at least three finite action rows are required"
    )
    metric_rows = {
        "success_rate": row(
            value=success,
            units="fraction",
            source="trace.done_info.success|trace.is_success",
            mapping="exact_local",
            reason=boolean_reason if success is None else None,
        ),
        "collision_rate": row(
            value=collision,
            units="fraction",
            source="trace.collision_flags",
            mapping="exact_local",
            reason=boolean_reason if collision is None else None,
        ),
        "minimum_human_distance_m": row(
            value=min(distances) if distances and not malformed_distances else None,
            units="m",
            source="trace.ground_truth_simulator_distance",
            mapping="qualified_proxy",
            reason=distance_reason,
        ),
        "personal_space_compliance_rate": row(
            value=(
                float(np.mean(np.asarray(distances) >= personal_space_radius_m))
                if distances and not malformed_distances
                else None
            ),
            units="fraction",
            source="trace.ground_truth_simulator_distance",
            mapping="qualified_proxy",
            reason=distance_reason,
        ),
        "angular_jerk_rad_s3": row(
            value=(
                float(np.mean(np.abs(np.diff(actions[:, 1], n=2))) / dt_s**2)
                if actions.shape[0] >= 3 and not malformed_actions
                else None
            ),
            units="rad/s^3",
            source="trace.env_action[1]",
            mapping="qualified_proxy",
            reason=action_reason,
        ),
        "action_smoothness_l2": row(
            value=(
                float(np.mean(np.linalg.norm(np.diff(actions, axis=0), axis=1)))
                if actions.shape[0] >= 2 and not malformed_actions
                else None
            ),
            units="action_units/step",
            source="trace.env_action",
            mapping="exact_local",
            reason=(
                "action fields must contain at least two finite numeric channels"
                if malformed_actions
                else "at least two finite action rows are required"
            ),
        ),
        "timeout_rate": row(
            value=(
                float(bool(truncated) and not bool(success))
                if truncated is not None and success is not None
                else None
            ),
            units="fraction",
            source="trace.done_info.truncated|trace.truncated|trace.horizon",
            mapping="exact_local",
            reason=boolean_reason if truncated is None else None,
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
