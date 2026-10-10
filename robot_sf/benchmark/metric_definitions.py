"""Metric meaning versions, independent of the episode JSON envelope version.

Unmarked historical rows/assets use v1 (including published release 0.0.7).
Version v2 fixes issue #10007 F4/F5/F7/F8 and D-055 curvature; it requires fresh normalization.
"""

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

from robot_sf.benchmark import constants
from robot_sf.benchmark.robot_force_contract import declared_force_source_contract

LEGACY_METRIC_SCHEMA_VERSION = "robot-sf-metrics.v1"
METRIC_SCHEMA_VERSION = "robot-sf-metrics.v2"
CHANGED_METRICS = frozenset(
    {
        "path_efficiency",
        "path_efficiency_reference_violation",
        "path_length",
        "success_path_length",
        "socnavbench_path_length",
        "socnavbench_path_length_ratio",
        "socnavbench_path_irregularity",
        "failure_to_progress",
        "deadlock",
        "deadlock_stall",
        "jerk_mean",
        "curvature_mean",
        "time_to_goal",
        "time_to_goal_norm",
        "time_to_goal_norm_success_only",
        "time_to_goal_ideal_ratio",
        "aggregated_time",
        "snqi",
        "snqi_v1",
        "snqi_v2",
        "snqi_v2_terms",
        "social_mini_game",
        "shortest_path_len",
    }
)


def metric_schema_version(value: Mapping[str, Any]) -> str:
    """Resolve row, metric mapping, or anchor document version; reject contradictions.

    Returns:
        Known version slug, defaulting to the historical definition for unmarked inputs."""
    versions = [value["metric_schema_version"]] if "metric_schema_version" in value else []
    for key in ("metrics", "_metadata"):
        child = value.get(key)
        if isinstance(child, Mapping) and "metric_schema_version" in child:
            versions.append(child["metric_schema_version"])
    v2_only = _carries_v2_only_fields(value)
    if not versions:
        if v2_only:
            raise ValueError("missing metric_schema_version on v2-only fields")
        return LEGACY_METRIC_SCHEMA_VERSION
    if v2_only and LEGACY_METRIC_SCHEMA_VERSION in versions:
        raise ValueError("incompatible metric definitions: v2-only fields marked v1")
    if any(v not in (LEGACY_METRIC_SCHEMA_VERSION, METRIC_SCHEMA_VERSION) for v in versions):
        raise ValueError(f"unsupported metric_schema_version: {versions}")
    if len(set(versions)) != 1:
        raise ValueError(f"contradictory metric_schema_version: {versions}")
    return versions[0]


def require_uniform_metric_schema(records: Iterable[Mapping[str, Any]]) -> str:
    """Reject pooled changed definitions before aggregation or calibration.

    Returns:
        Common schema version, or the legacy default for an empty collection."""
    records = list(records)
    require_uniform_trace_schema(records)
    versions = {metric_schema_version(record) for record in records}
    if len(versions) > 1:
        raise ValueError(
            f"incompatible metric definitions: {sorted(versions)}; recompute from traces"
        )
    return next(iter(versions), LEGACY_METRIC_SCHEMA_VERSION)


def require_anchor_compatibility(metrics: Mapping[str, Any], anchors: Mapping[str, Any]) -> None:
    """Never normalize current physical jerk/time with historical anchors silently."""
    observed, expected = metric_schema_version(metrics), metric_schema_version(anchors)
    if observed != expected:
        raise ValueError(
            f"incompatible metric definitions: episodes={observed}, anchors={expected}; "
            "derive new anchors from matching-schema episodes (do not relabel old assets)"
        )


def changed_metric_field(field: str) -> bool:
    """Recognize flattened metric fields whose definition or composite inputs changed.

    Returns:
        Whether a field depends on one of the corrected definitions."""
    return field.startswith("metrics.") and field.split(".")[1] in CHANGED_METRICS


SIMULATION_TRACE_SCHEMAS = frozenset({"simulation-step-trace.v1", "simulation-step-trace.v2"})
NATIVE_TRACE_SCHEMAS = frozenset({"paired_effect_native_trace.v1", "paired_effect_native_trace.v2"})


def _carries_v2_only_fields(value: Mapping[str, Any]) -> bool:
    """Recognize exclusive v2 evidence without guessing the units of legacy scalars.

    Returns:
        Whether this input carries fields exclusive to the current definitions.
    """
    if "path_efficiency_reference_violation" in value:
        return True
    deadlock = value.get("deadlock_stall")
    if isinstance(deadlock, Mapping) and deadlock.get("schema_version") == "deadlock-stall.v2":
        return True
    for key in ("simulation_step_trace", "paired_effect_native_trace"):
        trace = value.get(key)
        if isinstance(trace, Mapping) and str(trace.get("schema_version", "")).endswith(".v2"):
            return True
    return any(
        _carries_v2_only_fields(child)
        for key in ("metrics", "_metadata", "algorithm_metadata")
        if isinstance((child := value.get(key)), Mapping)
    )


def require_uniform_trace_schema(records: Iterable[Mapping[str, Any]]) -> None:
    """Refuse pooling trace references with different meanings, including within rows."""
    versions = set()
    for row in records:
        metadata = row.get("algorithm_metadata", {})
        if not isinstance(metadata, Mapping):
            continue
        for key, supported in (
            ("simulation_step_trace", SIMULATION_TRACE_SCHEMAS),
            ("paired_effect_native_trace", NATIVE_TRACE_SCHEMAS),
        ):
            trace = metadata.get(key)
            if not isinstance(trace, Mapping):
                continue
            schema = trace.get("schema_version")
            if schema not in supported:
                raise ValueError(f"unsupported trace schema: {schema}")
            versions.add(schema.rsplit(".", 1)[-1])
    if len(versions) > 1:
        raise ValueError("trace_schema_version_mismatch: never pool v1/v2 traces")


# This registry is the calibration contract. Definition changes must update it and
# the fixed-trace canary together; a schema label alone cannot establish identity.
def snqi_v2_source_definitions() -> dict[str, Any]:
    """Resolve source meanings from the same constants consumed by metric producers.

    Returns:
        A fresh JSON-compatible registry without paths or import-time threshold snapshots.
    """
    return {
        "success": {
            "formula": "reached_goal_step < horizon and total_collision_count == 0",
            "units": "binary",
            "alignment": "episode termination",
            "reduction": "one episode indicator",
        },
        "total_collision_count": {
            "formula": "ped_collision_count + obstacle_collision_count + agent_collision_count",
            "units": "collision timesteps",
            "alignment": "post-step footprint samples",
            "thresholds": {
                "pedestrian_surface_clearance_m": 0.0,
                "wall_agent_center_distance_m": constants.COLLISION_DIST,
            },
            "reduction": "sum counts; scoring uses count > 0",
        },
        "time_to_goal_ideal_ratio": {
            "formula": "elapsed_goal_time / (shortest_path_len / robot_max_speed)",
            "units": "dimensionless",
            "alignment": "reset included: goal_step*dt; otherwise (goal_step+1)*dt",
            "thresholds": "successful episode; positive finite ideal time and physical speed cap",
            "reduction": "success only; scoring clips (ratio-1)/2 to [0,1], failures contribute zero",
        },
        "near_misses": {
            "formula": "count steps with 0 <= min pedestrian surface clearance < NEAR_MISS_DIST m",
            "units": "timesteps",
            "alignment": "post-step robot/pedestrian footprints",
            "thresholds": {"clearance_m": constants.NEAR_MISS_DIST},
            "reduction": "minimum over present pedestrians then count; scoring clips count/steps/0.25",
        },
        "jerk_mean": {
            "formula": "mean norm((a[t+1]-a[t])/dt) for first T-2 acceleration differences",
            "units": "m/s^3",
            "alignment": "post-step recorded robot acceleration, last difference excluded",
            "thresholds": "T < 3 returns zero; invalid dt returns NaN",
            "reduction": "arithmetic mean over T-2; calibration episode p95 with linear interpolation",
        },
        "curvature_mean": {
            "formula": "sum abs wrapped consecutive displacement turns / max(counted path length, CURVATURE_LENGTH_FLOOR_M)",
            "units": "rad/m",
            "alignment": "positions including reset pose when supplied; bridge stops",
            "thresholds": {
                "minimum_displacement_m": constants.CURVATURE_MIN_DISPLACEMENT_M,
                "length_floor_m": constants.CURVATURE_LENGTH_FLOOR_M,
            },
            "reduction": "finite displacements only; fewer than two returns zero; episode linear p95",
        },
        "robot_force_impulse_total": {
            "formula": "dt * sum over steps and present pedestrians of norm(recorded force)",
            "units": "m/s (model acceleration impulse)",
            "alignment": "recorded pre-integration force samples",
            "thresholds": "zero invalid present samples; absent slots contribute zero",
            "reduction": "episode sum, calibration episode linear p95",
            "kernel": "multiplier * delta/distance^4; active if distance <= activation+robot_radius+ped_radius; per-pedestrian response multipliers when declared",
        },
        "robot_force_pp_equiv_impulse_total": {
            "formula": "dt * sum norm(counterfactual pedestrian-pair force)",
            "units": "m/s (model acceleration impulse)",
            "alignment": "pre-integration force-input geometry; forward velocity difference at first sample, backward thereafter",
            "thresholds": "selected iff abs(Spearman(raw F, clipped N)) >= 0.90; zero invalid present samples",
            "reduction": "episode sum, calibration episode linear p95",
            "kernel": "effective distance=max(0,distance-robot_radius+ped_radius); interaction=lambda*relative_velocity+direction; B=gamma*norm(interaction)+1e-8; theta=angle(interaction)-angle(direction); along=exp(-distance/B-(n_prime*B*theta)^2); lateral=-sign(theta)*exp(-distance/B-(n*B*theta)^2), sign(0)=1; force=factor*(unit*along+normal*lateral) within activation_threshold",
        },
    }


# Public registry snapshot; the digest resolves thresholds afresh from the producer constants.
SNQI_V2_SOURCE_DEFINITIONS = snqi_v2_source_definitions()


def metric_definitions_sha256() -> str:
    """Hash canonical SNQI source meanings and both recorded force reference contracts.

    Returns:
        SHA256 of sorted compact UTF-8 JSON, independent of paths or formatting.
    """
    document = {
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "sources": snqi_v2_source_definitions(),
        "force_contracts": {
            source: declared_force_source_contract(source)
            for source in ("robot_force_impulse_total", "robot_force_pp_equiv_impulse_total")
        },
    }
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def require_definitions_digest(metrics: Mapping[str, Any], expected: str | None) -> None:
    """Require identical source meanings for bound anchors; retain explicit old-asset compatibility."""
    if expected is None:
        return  # Historical assets lack this binding and rely on procedural source-drift checks.
    if metrics.get("metric_definitions_sha256") != expected:
        raise ValueError("SNQI-v2 metric definitions digest mismatch or absent on episode")


def calibration_definitions_digest(
    episodes: Iterable[Mapping[str, Any]], *, allow_historical_unbound: bool = False
) -> str | None:
    """Validate producer binding before stamping anchors; never relabel unbound old rows.

    Returns:
        Current digest for fully bound inputs, or None for wholly unbound historical inputs.
    """
    rows = list(episodes)
    if all("metric_definitions_sha256" not in row.get("metrics", {}) for row in rows):
        if not allow_historical_unbound and any(
            metric_schema_version(row) == METRIC_SCHEMA_VERSION for row in rows
        ):
            raise ValueError(
                "SNQI-v2 current-schema calibration requires definitions digest; "
                "historical reconstruction requires allow_historical_unbound=True"
            )
        return None
    expected = metric_definitions_sha256()
    for row in rows:
        require_definitions_digest(row.get("metrics", {}), expected)
        if metric_schema_version(row) != METRIC_SCHEMA_VERSION:
            raise ValueError("SNQI-v2 definitions digest requires current metric schema")
    return expected
