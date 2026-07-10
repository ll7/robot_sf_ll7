"""Config and analysis helpers for the issue #5088 braking-authority smoke."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from robot_sf.benchmark.metrics import EpisodeData, time_to_collision_min
from robot_sf.training.scenario_loader import load_scenarios

CONFIG_SCHEMA = "braking-authority-sensitivity-smoke.v1"
REPORT_SCHEMA = "braking-authority-sensitivity-report.v1"
NON_EVIDENCE_READINESS = frozenset({"fallback", "degraded"})
NON_EVIDENCE_AVAILABILITY = frozenset({"partial-failure", "failed", "not_available"})
EVIDENCE_EXECUTION_MODES = frozenset({"native", "adapter", "mixed"})
EVIDENCE_READINESS = frozenset({"native", "adapter"})


def load_smoke_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the config-first braking-authority smoke contract.

    Returns:
        Normalized smoke configuration.
    """
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"smoke config must be a mapping: {config_path}")
    return validate_smoke_config(payload)


def validate_smoke_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a normalized smoke config or fail closed on incomplete controls.

    Returns:
        Normalized smoke configuration.
    """
    config = deepcopy(dict(payload))
    _validate_top_level(config)
    _validate_scenario(config)
    config["seeds"] = _validate_seeds(config.get("seeds"))
    _validate_planner(config)
    _validate_run(config)
    _validate_arms(config)
    _validate_signal(config)
    return config


def _validate_top_level(config: Mapping[str, Any]) -> None:
    """Validate schema, issue, and claim-boundary identity."""
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA!r}")
    if int(config.get("issue", 0)) != 5088:
        raise ValueError("issue must be 5088")
    if "targeted-smoke" not in str(config.get("claim_boundary", "")):
        raise ValueError("claim_boundary must explicitly stay targeted-smoke evidence")


def _validate_scenario(config: Mapping[str, Any]) -> None:
    """Require one named committed scenario path."""
    scenario = _mapping(config, "scenario")
    if not str(scenario.get("path", "")).strip() or not str(scenario.get("name", "")).strip():
        raise ValueError("scenario.path and scenario.name are required")


def _validate_seeds(seeds: Any) -> list[int]:
    """Normalize and validate the fixed seed set.

    Returns:
        Unique integer seeds in configured order.
    """
    if not isinstance(seeds, list) or not seeds:
        raise ValueError("seeds must be a non-empty fixed list")
    normalized_seeds = [int(seed) for seed in seeds]
    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("seeds must be unique")
    return normalized_seeds


def _validate_planner(config: Mapping[str, Any]) -> None:
    """Require a recognized benchmark profile and named planner."""
    planner = _mapping(config, "planner")
    if not str(planner.get("algo", "")).strip():
        raise ValueError("planner.algo is required")
    if planner.get("benchmark_profile") not in {
        "baseline-safe",
        "paper-baseline",
        "experimental",
    }:
        raise ValueError("planner.benchmark_profile is invalid")


def _validate_run(config: Mapping[str, Any]) -> None:
    """Require deterministic single-worker execution and TTC traces."""
    run = _mapping(config, "run")
    if int(run.get("workers", 0)) != 1:
        raise ValueError("run.workers must be 1 for deterministic targeted smoke")
    if not bool(run.get("record_simulation_step_trace", False)):
        raise ValueError("run.record_simulation_step_trace must be true for TTC provenance")
    if float(run.get("dt", 0.0)) <= 0.0 or int(run.get("horizon", 0)) <= 0:
        raise ValueError("run.dt and run.horizon must be positive")


def _validate_arms(config: Mapping[str, Any]) -> None:
    """Require unique positive braking-authority arms."""
    arms = config.get("arms")
    if not isinstance(arms, list) or len(arms) < 2:
        raise ValueError("arms must contain at least two braking-authority values")
    keys: set[str] = set()
    values: set[float] = set()
    for arm in arms:
        if not isinstance(arm, dict):
            raise ValueError("each arm must be a mapping")
        key = str(arm.get("key", "")).strip()
        value = _positive_finite(arm.get("max_linear_decel_m_s2"), field=key or "arm")
        if not key or key in keys:
            raise ValueError("arm keys must be non-empty and unique")
        keys.add(key)
        values.add(value)
        arm["max_linear_decel_m_s2"] = value
    if len(values) < 2:
        raise ValueError("arms must contain at least two distinct braking-authority values")


def _validate_signal(config: Mapping[str, Any]) -> None:
    """Validate the bounded metric-sensitivity decision rule."""
    signal = _mapping(config, "signal")
    metrics = signal.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        raise ValueError("signal.metrics must be a non-empty list")
    allowed_metrics = {"near_misses", "min_clearance", "time_to_collision_min"}
    if not set(metrics).issubset(allowed_metrics):
        raise ValueError(f"signal.metrics must be drawn from {sorted(allowed_metrics)}")
    tolerance = float(signal.get("absolute_tolerance", 0.0))
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("signal.absolute_tolerance must be finite and non-negative")
    signal["absolute_tolerance"] = tolerance
    minimum_valid_seeds = int(signal.get("minimum_valid_seeds", 0))
    if minimum_valid_seeds <= 0 or minimum_valid_seeds > len(config["seeds"]):
        raise ValueError("signal.minimum_valid_seeds must be within the fixed seed count")
    signal["minimum_valid_seeds"] = minimum_valid_seeds


def load_selected_scenario(config: Mapping[str, Any], *, repo_root: Path) -> dict[str, Any]:
    """Load the single committed source scenario named by the smoke config.

    Returns:
        Selected scenario mapping.
    """
    scenario_spec = _mapping(config, "scenario")
    scenario_path = repo_root / str(scenario_spec["path"])
    scenario_name = str(scenario_spec["name"])
    matches = [
        dict(scenario)
        for scenario in load_scenarios(scenario_path)
        if str(scenario.get("name", scenario.get("scenario_id", ""))) == scenario_name
    ]
    if len(matches) != 1:
        raise ValueError(
            f"scenario {scenario_name!r} must resolve exactly once in {scenario_path}; "
            f"found {len(matches)}"
        )
    return matches[0]


def materialize_arm_scenario(
    source: Mapping[str, Any],
    *,
    arm: Mapping[str, Any],
    seeds: Sequence[int],
) -> dict[str, Any]:
    """Copy one scenario and vary only its differential-drive braking authority.

    Returns:
        Materialized scenario arm.
    """
    scenario = deepcopy(dict(source))
    robot_config = deepcopy(dict(scenario.get("robot_config") or {}))
    robot_type = str(robot_config.get("type", robot_config.get("model", "differential_drive")))
    if robot_type not in {"differential_drive", "differential", "diff_drive"}:
        raise ValueError("issue #5088 smoke requires a differential-drive source scenario")
    robot_config["type"] = "differential_drive"
    robot_config["max_linear_decel"] = _positive_finite(
        arm.get("max_linear_decel_m_s2"), field=str(arm.get("key", "arm"))
    )
    scenario["robot_config"] = robot_config
    scenario["seeds"] = [int(seed) for seed in seeds]
    metadata = deepcopy(dict(scenario.get("metadata") or {}))
    metadata["braking_authority_sensitivity_arm"] = str(arm["key"])
    metadata["source_issue"] = 5088
    scenario["metadata"] = metadata
    return scenario


def analyze_smoke_results(
    config: Mapping[str, Any],
    *,
    arm_records: Mapping[str, Sequence[Mapping[str, Any]]],
    arm_summaries: Mapping[str, Mapping[str, Any]],
    config_path: str,
    run_commit: str,
    raw_artifact_root: str,
    raw_artifacts: Mapping[str, Mapping[str, Any]],
    reproduction_command: str,
) -> dict[str, Any]:
    """Build a fail-closed compact sensitivity report from completed arm runs.

    Returns:
        Compact sensitivity report.
    """
    validated = validate_smoke_config(config)
    expected_seeds = sorted(int(seed) for seed in validated["seeds"])
    metric_names = [str(metric) for metric in validated["signal"]["metrics"]]
    arm_rows: list[dict[str, Any]] = []
    per_arm_means: dict[str, dict[str, float]] = {}
    for arm in validated["arms"]:
        key = str(arm["key"])
        records = list(arm_records.get(key, ()))
        if sorted(int(record["seed"]) for record in records) != expected_seeds:
            raise ValueError(f"arm {key!r} does not contain the exact fixed seed set")
        availability = _validated_availability(arm_summaries.get(key, {}), arm_key=key)
        episodes = [_episode_metric_row(record) for record in records]
        metric_values = {
            metric: [
                float(episode[metric]) for episode in episodes if episode.get(metric) is not None
            ]
            for metric in metric_names
        }
        means = {
            metric: _finite_mean(values, field=metric) for metric, values in metric_values.items()
        }
        minimum_valid_seeds = int(validated["signal"]["minimum_valid_seeds"])
        insufficient_metrics = [
            metric for metric, values in metric_values.items() if len(values) < minimum_valid_seeds
        ]
        if insufficient_metrics:
            raise ValueError(
                f"arm {key!r} has fewer than {minimum_valid_seeds} valid seeds for "
                f"{', '.join(insufficient_metrics)}"
            )
        per_arm_means[key] = means
        outcome_counts: dict[str, int] = {}
        for episode in episodes:
            status = str(episode["status"])
            outcome_counts[status] = outcome_counts.get(status, 0) + 1
        arm_rows.append(
            {
                "key": key,
                "max_linear_decel_m_s2": float(arm["max_linear_decel_m_s2"]),
                "stopping_distance_at_max_speed_m": _stopping_distance_from_summary(
                    arm_summaries[key],
                    arm_key=key,
                    expected_decel=float(arm["max_linear_decel_m_s2"]),
                ),
                "execution_mode": availability["execution_mode"],
                "readiness_status": availability["readiness_status"],
                "availability_status": availability["availability_status"],
                "benchmark_success": availability["benchmark_success"],
                "episode_count": len(episodes),
                "outcome_counts": dict(sorted(outcome_counts.items())),
                "metric_valid_counts": {
                    metric: len(values) for metric, values in metric_values.items()
                },
                "metric_means": means,
                "episodes": episodes,
            }
        )

    ordered = sorted(validated["arms"], key=lambda arm: float(arm["max_linear_decel_m_s2"]))
    comparator_key = str(ordered[0]["key"])
    stronger_key = str(ordered[-1]["key"])
    deltas = {
        metric: per_arm_means[stronger_key][metric] - per_arm_means[comparator_key][metric]
        for metric in metric_names
    }
    tolerance = float(validated["signal"]["absolute_tolerance"])
    activated_metrics = [metric for metric, delta in deltas.items() if abs(delta) > tolerance]
    episode_by_arm_seed = {
        arm["key"]: {episode["seed"]: episode for episode in arm["episodes"]} for arm in arm_rows
    }
    comparable_seed_counts: dict[str, int] = {}
    changed_seed_counts: dict[str, int] = {}
    for metric in metric_names:
        pairs = [
            (
                episode_by_arm_seed[comparator_key][seed].get(metric),
                episode_by_arm_seed[stronger_key][seed].get(metric),
            )
            for seed in expected_seeds
        ]
        comparable = [
            (float(left), float(right))
            for left, right in pairs
            if left is not None and right is not None
        ]
        comparable_seed_counts[metric] = len(comparable)
        changed_seed_counts[metric] = sum(
            abs(right - left) > tolerance for left, right in comparable
        )
    signal_activated = bool(activated_metrics)
    return {
        "schema_version": REPORT_SCHEMA,
        "issue": 5088,
        "status": "signal_activated" if signal_activated else "no_observed_metric_signal",
        "evidence_tier": "targeted-smoke",
        "claim_boundary": str(validated["claim_boundary"]),
        "config_path": config_path,
        "config_sha256": _sha256(Path(config_path)),
        "scenario_path": str(validated["scenario"]["path"]),
        "scenario_name": str(validated["scenario"]["name"]),
        "seeds": expected_seeds,
        "planner": deepcopy(dict(validated["planner"])),
        "run_commit": run_commit,
        "reproduction_command": reproduction_command,
        "raw_artifact_root": raw_artifact_root,
        "raw_artifact_classification": "local-scratch; not durable evidence",
        "raw_artifacts": deepcopy(dict(raw_artifacts)),
        "comparison": {
            "comparator_arm": comparator_key,
            "stronger_braking_arm": stronger_key,
            "metric_deltas_stronger_minus_comparator": deltas,
            "absolute_tolerance": tolerance,
            "activated_metrics": activated_metrics,
            "comparable_seed_counts": comparable_seed_counts,
            "changed_seed_counts": changed_seed_counts,
            "signal_activated": signal_activated,
        },
        "arms": arm_rows,
        "fallback_degraded_exclusion": (
            "fallback, degraded, partial-failure, failed, and not_available runs fail closed"
        ),
    }


def format_report_markdown(report: Mapping[str, Any]) -> str:
    """Format the compact tracked report with its claim boundary first.

    Returns:
        Markdown rendering of the report.
    """
    comparison = _mapping(report, "comparison")
    lines = [
        "# Issue #5088 Braking-Authority Sensitivity Smoke",
        "",
        f"- Status: `{report['status']}`",
        f"- Evidence tier: `{report['evidence_tier']}`",
        f"- Claim boundary: {report['claim_boundary']}",
        f"- Run commit: `{report['run_commit']}`",
        f"- Config: `{report['config_path']}`",
        f"- Scenario: `{report['scenario_path']}` (`{report['scenario_name']}`)",
        f"- Seeds: `{', '.join(str(seed) for seed in report['seeds'])}`",
        f"- Planner: `{report['planner']['algo']}`",
        "",
        "## Result",
        "",
        f"Metric-sensitivity signal activated: `{comparison['signal_activated']}`.",
        f"Activated metrics: `{', '.join(comparison['activated_metrics']) or 'none'}`.",
        "Changed-seed counts: "
        + ", ".join(
            f"`{metric}={comparison['changed_seed_counts'][metric]}/"
            f"{comparison['comparable_seed_counts'][metric]}`"
            for metric in comparison["changed_seed_counts"]
        )
        + ".",
        "",
        "| Arm | Braking (m/s^2) | Stop distance at max speed (m) | Mode | Readiness | Near misses | Min clearance (m) | Min TTC (s) | TTC valid |",
        "| --- | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for arm in report["arms"]:
        means = arm["metric_means"]
        lines.append(
            f"| `{arm['key']}` | {arm['max_linear_decel_m_s2']:.3f} | "
            f"{arm['stopping_distance_at_max_speed_m']:.3f} | {arm['execution_mode']} | "
            f"{arm['readiness_status']} | {means['near_misses']:.6f} | "
            f"{means['min_clearance']:.6f} | {means['time_to_collision_min']:.6f} | "
            f"{arm['metric_valid_counts']['time_to_collision_min']}/{arm['episode_count']} |"
        )
    lines.extend(
        [
            "",
            "## Reproduce",
            "",
            "```bash",
            str(report["reproduction_command"]),
            "```",
            "",
            "Replace `<fresh-artifact-dir>` with an empty local scratch directory. Raw episode "
            "JSONL remains disposable and untracked; `report.json` plus this README are the "
            "tracked compact evidence. Fallback/degraded execution is excluded, and this smoke "
            "makes no calibrated safety or paper-facing claim.",
            "",
        ]
    )
    return "\n".join(lines)


def git_head(repo_root: Path) -> str:
    """Return the checked-out commit used to execute the smoke."""
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()


def _episode_metric_row(record: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _mapping(record, "metrics")
    ttc, ttc_status = _ttc_from_record(record)
    return {
        "seed": int(record["seed"]),
        "status": str(record.get("status", "unknown")),
        "near_misses": _finite_float(metrics.get("near_misses"), field="near_misses"),
        "min_clearance": _finite_float(metrics.get("min_clearance"), field="min_clearance"),
        "time_to_collision_min": ttc,
        "time_to_collision_min_status": ttc_status,
    }


def _ttc_from_record(record: Mapping[str, Any]) -> tuple[float | None, str]:
    """Compute TTC from a trace, preserving unsupported inputs as unavailable.

    Returns:
        TTC in seconds plus an availability status.
    """
    algorithm_metadata = _mapping(record, "algorithm_metadata")
    trace = _mapping(algorithm_metadata, "simulation_step_trace")
    steps = trace.get("steps")
    if not isinstance(steps, list) or len(steps) < 2:
        return None, "unsupported_trace_too_short"
    robot_pos = np.asarray([_mapping(step, "robot")["position"] for step in steps], dtype=float)
    robot_vel = np.asarray([_mapping(step, "robot")["velocity"] for step in steps], dtype=float)
    ped_counts = {len(step.get("pedestrians", [])) for step in steps if isinstance(step, Mapping)}
    if len(ped_counts) != 1 or not ped_counts or next(iter(ped_counts)) == 0:
        raise ValueError("TTC trace requires a stable non-zero pedestrian count")
    peds_pos = np.asarray(
        [[pedestrian["position"] for pedestrian in step["pedestrians"]] for step in steps],
        dtype=float,
    )
    episode = EpisodeData(
        robot_pos=robot_pos,
        robot_vel=robot_vel,
        robot_acc=np.zeros_like(robot_pos),
        peds_pos=peds_pos,
        ped_forces=np.zeros_like(peds_pos),
        goal=np.zeros(2, dtype=float),
        dt=_positive_finite(trace.get("dt"), field="simulation_step_trace.dt"),
    )
    ttc = float(time_to_collision_min(episode))
    if not math.isfinite(ttc):
        return None, "no_approaching_pairs"
    return ttc, "available"


def _validated_availability(summary: Mapping[str, Any], *, arm_key: str) -> dict[str, Any]:
    availability = _mapping(summary, "benchmark_availability")
    execution_mode = str(availability.get("execution_mode", "unknown"))
    readiness = str(availability.get("readiness_status", "unknown"))
    status = str(availability.get("availability_status", "unknown"))
    if readiness in NON_EVIDENCE_READINESS or status in NON_EVIDENCE_AVAILABILITY:
        raise ValueError(f"arm {arm_key!r} is non-evidence: readiness={readiness}, status={status}")
    if not bool(availability.get("benchmark_success", False)):
        raise ValueError(f"arm {arm_key!r} did not satisfy benchmark-success runtime integrity")
    if (
        execution_mode not in EVIDENCE_EXECUTION_MODES
        or readiness not in EVIDENCE_READINESS
        or status != "available"
    ):
        raise ValueError(
            f"arm {arm_key!r} has unsupported runtime classification: "
            f"execution={execution_mode}, readiness={readiness}, availability={status}"
        )
    return {
        "execution_mode": execution_mode,
        "readiness_status": readiness,
        "availability_status": status,
        "benchmark_success": True,
    }


def _stopping_distance_from_summary(
    summary: Mapping[str, Any],
    *,
    arm_key: str,
    expected_decel: float,
) -> float:
    """Validate and return the arm's emitted stopping-distance provenance.

    Returns:
        Stopping distance at the configured peak speed in meters.
    """
    provenance = _mapping(summary, "provenance")
    identity = _mapping(provenance, "config_identity")
    metric_config = _mapping(identity, "metric_affecting_config")
    envelope = _mapping(metric_config, "actuation_envelope")
    observed_decel = _finite_float(
        envelope.get("max_braking_decel_m_s2"), field=f"{arm_key}.max_braking_decel"
    )
    if not math.isclose(observed_decel, expected_decel, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            f"arm {arm_key!r} braking provenance {observed_decel} "
            f"does not match configured {expected_decel}"
        )
    return _finite_float(
        envelope.get("stopping_distance_envelope_m"), field=f"{arm_key}.stopping_distance"
    )


def _mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return dict(value)


def _positive_finite(value: Any, *, field: str) -> float:
    parsed = _finite_float(value, field=field)
    if parsed <= 0.0:
        raise ValueError(f"{field} must be positive")
    return parsed


def _finite_float(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric") from exc
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def _finite_mean(values: Sequence[float], *, field: str) -> float:
    if not values:
        raise ValueError(f"{field} has no values")
    return _finite_float(sum(values) / len(values), field=field)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_report(report: Mapping[str, Any], output_dir: Path) -> None:
    """Write the compact local report and its checksum."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_dir / "README.md").write_text(format_report_markdown(report), encoding="utf-8")
    checksum_lines = [
        f"{_sha256(path)}  {path.name}" for path in (output_dir / "README.md", report_path)
    ]
    (output_dir / "checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")


__all__ = [
    "CONFIG_SCHEMA",
    "REPORT_SCHEMA",
    "analyze_smoke_results",
    "format_report_markdown",
    "git_head",
    "load_selected_scenario",
    "load_smoke_config",
    "materialize_arm_scenario",
    "validate_smoke_config",
    "write_report",
]
