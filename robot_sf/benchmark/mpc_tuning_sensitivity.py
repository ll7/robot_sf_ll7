"""Validation and reporting helpers for issue #5579's MPC tuning study.

The module owns the bounded, config-first analysis contract. Episode execution remains in the
issue runner and is optional for CPU validation; this module never submits work or upgrades a
diagnostic result to benchmark evidence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from robot_sf.planner.prediction_mpc import build_prediction_mpc_config
from robot_sf.training.scenario_loader import load_scenarios

CONFIG_SCHEMA = "issue_5579_mpc_tuning_sensitivity.v1"
REPORT_SCHEMA = "issue_5579_mpc_tuning_sensitivity_report.v1"
TARGET_ARM_KEYS = ("prediction_mpc", "prediction_mpc_cbf")
TOP_PARAMETERS = ("max_linear_speed", "horizon_steps", "pedestrian_safety_margin")
VALID_EXECUTION_MODES = frozenset({"native", "adapter", "mixed"})
VALID_READINESS_STATUSES = frozenset({"native", "adapter"})


def load_sensitivity_config(path: str | Path, *, repo_root: Path | None = None) -> dict[str, Any]:
    """Load and fail closed on an issue #5579 sensitivity config.

    Returns:
        Validated sensitivity configuration.
    """
    config_path = Path(path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"sensitivity config must be a mapping: {config_path}")
    return validate_sensitivity_config(payload, repo_root=repo_root or Path.cwd())


def validate_sensitivity_config(
    payload: Mapping[str, Any], *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Validate the bounded search, paired scenario scope, and arm provenance.

    Returns:
        A deep-copied validated configuration mapping.
    """
    config = deepcopy(dict(payload))
    root = (repo_root or Path.cwd()).resolve()
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise ValueError(f"schema_version must be {CONFIG_SCHEMA!r}")
    if int(config.get("issue", 0)) != 5579:
        raise ValueError("issue must be 5579")
    claim_boundary = str(config.get("claim_boundary", ""))
    normalized_claim_boundary = claim_boundary.lower()
    if (
        "diagnostic" not in normalized_claim_boundary
        or "benchmark ranking" not in normalized_claim_boundary
    ):
        raise ValueError("claim_boundary must retain the diagnostic/no-ranking boundary")
    _validate_execution_boundary(config.get("execution_boundary"))
    _validate_scenario_scope(config.get("scenario_scope"), repo_root=root)
    _validate_arm_list(config.get("target_arms"), expected=TARGET_ARM_KEYS, repo_root=root)
    _validate_arm_list(
        config.get("incumbent_arms"),
        expected=(
            "scenario_adaptive_hybrid_orca_v1",
            "scenario_adaptive_hybrid_orca_v2_collision_guard",
            "hybrid_rule_v3_fast_progress_static_escape",
            "hybrid_rule_v3_fast_progress_static_escape_continuous",
        ),
        repo_root=root,
    )
    _validate_search(config.get("search"))
    _validate_comparison(config.get("comparison"))
    return config


def selected_scenarios(config: Mapping[str, Any], *, repo_root: Path) -> list[dict[str, Any]]:
    """Return the exact fixed scenario subset with paired seeds materialized."""
    scope = _mapping(config.get("scenario_scope"), "scenario_scope")
    source = _repo_path(str(scope["source_matrix"]), repo_root)
    rows = load_scenarios(source, base_dir=source)
    by_name = {str(row.get("name")): dict(row) for row in rows}
    scenario_ids = [str(value) for value in scope["scenario_ids"]]
    missing = [name for name in scenario_ids if name not in by_name]
    if missing:
        raise ValueError(f"selected scenarios are absent from source matrix: {missing}")
    seeds = [int(seed) for seed in scope["seeds"]]
    selected: list[dict[str, Any]] = []
    for scenario_id in scenario_ids:
        scenario = deepcopy(by_name[scenario_id])
        scenario["seeds"] = list(seeds)
        selected.append(scenario)
    return selected


def build_candidate_plan(config: Mapping[str, Any], *, repo_root: Path) -> list[dict[str, Any]]:
    """Build deterministic target-candidate and incumbent execution rows.

    Returns:
        Ordered target and incumbent execution entries.
    """
    validated = validate_sensitivity_config(config, repo_root=repo_root)
    plan: list[dict[str, Any]] = []
    for arm in validated["target_arms"]:
        base = _load_yaml_mapping(_repo_path(str(arm["algo_config_path"]), repo_root))
        for point in validated["search"]["candidate_points"]:
            effective = deepcopy(base)
            effective.update(point["overrides"])
            build_prediction_mpc_config(effective)
            plan.append(
                {
                    "arm_key": str(arm["key"]),
                    "algo": str(arm["algo"]),
                    "candidate_id": str(point["id"]),
                    "target": True,
                    "overrides": deepcopy(dict(point["overrides"])),
                    "effective_config": effective,
                    "config_sha256_16": config_hash(effective),
                    "algo_config_path": str(arm["algo_config_path"]),
                }
            )
    for arm in validated["incumbent_arms"]:
        config_path = _repo_path(str(arm["algo_config_path"]), repo_root)
        effective = _load_yaml_mapping(config_path)
        plan.append(
            {
                "arm_key": str(arm["key"]),
                "algo": str(arm["algo"]),
                "candidate_id": "incumbent",
                "target": False,
                "overrides": {},
                "effective_config": effective,
                "config_sha256_16": config_hash(effective),
                "algo_config_path": str(arm["algo_config_path"]),
            }
        )
    return plan


def config_hash(config: Mapping[str, Any]) -> str:
    """Return the stable short hash used in compact sensitivity tables."""
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def normalize_episode_record(
    record: Mapping[str, Any], *, arm_key: str, candidate_id: str
) -> dict[str, Any]:
    """Normalize one runner row while preserving explicit availability provenance.

    Returns:
        A typed episode row suitable for the sensitivity analyzer.
    """
    outcome = record.get("outcome")
    if not isinstance(outcome, Mapping):
        raise ValueError("episode row must contain an outcome mapping")
    required = ("route_complete", "collision_event")
    missing = [field for field in required if field not in outcome]
    if missing:
        raise ValueError(f"episode outcome is missing explicit fields: {missing}")
    availability = record.get("sensitivity_availability")
    if not isinstance(availability, Mapping):
        availability = record.get("benchmark_availability")
    if not isinstance(availability, Mapping):
        raise ValueError("episode row is missing sensitivity_availability provenance")
    availability_fields = (
        "execution_mode",
        "readiness_status",
        "availability_status",
        "benchmark_success",
    )
    missing_availability = [field for field in availability_fields if field not in availability]
    if missing_availability:
        raise ValueError(f"episode availability is missing explicit fields: {missing_availability}")
    return {
        "arm_key": arm_key,
        "candidate_id": candidate_id,
        "scenario_id": str(record.get("scenario_id", "")),
        "seed": _int_field(record.get("seed"), field="seed"),
        "route_complete": _bool_field(outcome["route_complete"], field="route_complete"),
        "collision_event": _bool_field(outcome["collision_event"], field="collision_event"),
        "success": _bool_field(outcome["route_complete"], field="route_complete")
        and not _bool_field(outcome["collision_event"], field="collision_event"),
        "status": str(record.get("status", "")),
        "execution_mode": str(availability.get("execution_mode", "")),
        "readiness_status": str(availability.get("readiness_status", "")),
        "availability_status": str(availability.get("availability_status", "")),
        "benchmark_success": _bool_field(
            availability["benchmark_success"], field="benchmark_success"
        ),
    }


def analyze_results(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
    config_path: str,
    run_commit: str,
    reproduction_command: str,
    raw_artifact_root: str,
) -> dict[str, Any]:
    """Build a fail-closed best-of-20 report from normalized episode rows.

    Returns:
        Diagnostic report with candidate-level rows and the preregistered read.
    """
    validated = validate_sensitivity_config(config, repo_root=repo_root)
    plan = build_candidate_plan(validated, repo_root=repo_root)
    scenario_scope = validated["scenario_scope"]
    expected_keys = {
        (entry["arm_key"], entry["candidate_id"], scenario_id, int(seed))
        for entry in plan
        for scenario_id in scenario_scope["scenario_ids"]
        for seed in scenario_scope["seeds"]
    }
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    seen: set[tuple[str, str, str, int]] = set()
    for raw_row in rows:
        row = dict(raw_row)
        key = (
            str(row.get("arm_key", "")),
            str(row.get("candidate_id", "")),
            str(row.get("scenario_id", "")),
            _int_field(row.get("seed"), field="seed"),
        )
        if key not in expected_keys:
            raise ValueError(f"unexpected sensitivity row key: {key}")
        if key in seen:
            raise ValueError(f"duplicate sensitivity row key: {key}")
        seen.add(key)
        grouped.setdefault((key[0], key[1]), []).append(row)
    missing = sorted(expected_keys - seen)
    if missing:
        raise ValueError(f"sensitivity results are missing {len(missing)} expected rows")

    plan_by_key = {(entry["arm_key"], entry["candidate_id"]): entry for entry in plan}
    candidate_rows: list[dict[str, Any]] = []
    for group_key, entry in plan_by_key.items():
        group_rows = grouped[group_key]
        eligible_rows = [row for row in group_rows if _eligible(row)]
        excluded_rows = [row for row in group_rows if not _eligible(row)]
        success_count = sum(row.get("success") is True for row in eligible_rows)
        candidate_rows.append(
            {
                "arm_key": group_key[0],
                "candidate_id": group_key[1],
                "target": bool(entry["target"]),
                "config_sha256_16": entry["config_sha256_16"],
                "overrides": deepcopy(entry["overrides"]),
                "episodes": len(group_rows),
                "eligible_episodes": len(eligible_rows),
                "excluded_episodes": len(excluded_rows),
                "successes": success_count,
                "success_rate": (success_count / len(eligible_rows) if eligible_rows else None),
                "status": "eligible" if not excluded_rows else "excluded",
            }
        )

    all_rows_eligible = all(row["excluded_episodes"] == 0 for row in candidate_rows)
    target_summary = _summarize_targets(candidate_rows)
    incumbent_summary = _summarize_incumbents(candidate_rows)
    read = _build_read(target_summary, incumbent_summary, all_rows_eligible)
    return {
        "schema_version": REPORT_SCHEMA,
        "issue": 5579,
        "study_id": str(validated["study_id"]),
        "status": "complete_diagnostic" if all_rows_eligible else "blocked",
        "evidence_tier": "diagnostic-only",
        "benchmark_evidence": False,
        "claim_boundary": str(validated["claim_boundary"]),
        "config_path": config_path,
        "config_sha256": _config_sha256(config_path, repo_root=repo_root),
        "run_commit": run_commit,
        "reproduction_command": reproduction_command,
        "raw_artifact_root": raw_artifact_root,
        "scenario_scope": deepcopy(dict(scenario_scope)),
        "candidate_count": int(validated["search"]["candidate_count"]),
        "target_arm_count": len(validated["target_arms"]),
        "total_episode_rows": len(rows),
        "eligible_episode_rows": sum(row["eligible_episodes"] for row in candidate_rows),
        "excluded_episode_rows": sum(row["excluded_episodes"] for row in candidate_rows),
        "candidate_rows": candidate_rows,
        "target_summary": target_summary,
        "incumbent_summary": incumbent_summary,
        "read": read,
        "fallback_degraded_exclusion": (
            "Rows are eligible only with explicit native/adapter/mixed execution, native/adapter "
            "readiness, available status, and benchmark_success=true."
        ),
    }


def write_report(report: Mapping[str, Any], output_dir: str | Path) -> dict[str, str]:
    """Write compact JSON, Markdown, and candidate-level CSV report artifacts.

    Returns:
        Paths for the generated JSON, Markdown, and CSV artifacts.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "sensitivity_report.json"
    markdown_path = out_dir / "sensitivity_report.md"
    csv_path = out_dir / "sensitivity_candidate_rows.csv"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown_path.write_text(format_report_markdown(report), encoding="utf-8")
    rows = list(report.get("candidate_rows", []))
    if not rows:
        raise ValueError("cannot write an empty sensitivity candidate table")
    fields = (
        "arm_key",
        "candidate_id",
        "target",
        "config_sha256_16",
        "episodes",
        "eligible_episodes",
        "excluded_episodes",
        "successes",
        "success_rate",
        "status",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("# AI-GENERATED NEEDS-REVIEW\n")
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})
    return {"json": str(json_path), "markdown": str(markdown_path), "candidate_csv": str(csv_path)}


def format_report_markdown(report: Mapping[str, Any]) -> str:
    """Render the claim boundary before any diagnostic read.

    Returns:
        Markdown representation of the compact sensitivity report.
    """
    lines = [
        "# Issue #5579 MPC Tuning-Budget Sensitivity",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Evidence tier: `{report.get('evidence_tier')}`",
        f"- Claim boundary: {report.get('claim_boundary')}",
        f"- Run commit: `{report.get('run_commit')}`",
        f"- Config: `{report.get('config_path')}`",
        "",
        "## Best-found target configurations",
        "",
        "| Arm | Best candidate | Success rate | Eligible episodes | Excluded episodes |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for arm in report.get("target_summary", []):
        best = arm.get("best_candidate") or {}
        lines.append(
            f"| `{arm['arm_key']}` | `{best.get('candidate_id', 'NA')}` | "
            f"{_format_rate(best.get('success_rate'))} | {best.get('eligible_episodes', 0)} | "
            f"{best.get('excluded_episodes', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Incumbent hybrid band",
            "",
            "| Arm | Success rate | Eligible episodes | Excluded episodes |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for arm in report.get("incumbent_summary", []):
        lines.append(
            f"| `{arm['arm_key']}` | {_format_rate(arm.get('success_rate'))} | "
            f"{arm.get('eligible_episodes', 0)} | {arm.get('excluded_episodes', 0)} |"
        )
    lines.extend(
        [
            "",
            "## Pre-registered read",
            "",
            f"- Decision: `{report.get('read', {}).get('decision')}`",
            f"- Detail: {report.get('read', {}).get('detail')}",
            "",
            "Fallback, degraded, failed, and unavailable rows are never treated as success evidence.",
            "This diagnostic does not change benchmark metrics, roster status, or paper-facing claims.",
            "",
        ]
    )
    return "\n".join(lines)


def _validate_execution_boundary(value: Any) -> None:
    boundary = _mapping(value, "execution_boundary")
    for field in (
        "full_benchmark_campaign_run_in_this_pr",
        "slurm_or_gpu_submission_in_this_pr",
        "paper_or_dissertation_claim_edit_in_this_pr",
    ):
        if boundary.get(field) is not False:
            raise ValueError(f"execution_boundary.{field} must be false")


def _validate_scenario_scope(value: Any, *, repo_root: Path) -> None:
    scope = _mapping(value, "scenario_scope")
    source = _repo_path(str(scope.get("source_matrix", "")), repo_root)
    if not source.is_file():
        raise ValueError(f"scenario_scope.source_matrix does not exist: {source}")
    scenario_ids = scope.get("scenario_ids")
    if not isinstance(scenario_ids, list) or len(scenario_ids) != 3:
        raise ValueError("scenario_scope.scenario_ids must contain exactly three scenarios")
    if len(set(scenario_ids)) != len(scenario_ids) or not all(
        isinstance(item, str) and item.strip() for item in scenario_ids
    ):
        raise ValueError("scenario_scope.scenario_ids must be unique non-empty strings")
    seeds = scope.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 3:
        raise ValueError("scenario_scope.seeds must contain exactly three fixed seeds")
    if len(set(seeds)) != len(seeds) or not all(_is_int(item) for item in seeds):
        raise ValueError("scenario_scope.seeds must be unique integers")
    if int(scope.get("workers", 0)) != 1:
        raise ValueError("scenario_scope.workers must be 1")
    if int(scope.get("horizon", 0)) <= 0 or float(scope.get("dt", 0.0)) <= 0.0:
        raise ValueError("scenario_scope.horizon and dt must be positive")


def _validate_arm_list(value: Any, *, expected: Sequence[str], repo_root: Path) -> None:
    if not isinstance(value, list) or len(value) != len(expected):
        raise ValueError(f"arm list must contain exactly {len(expected)} entries")
    keys: list[str] = []
    for arm in value:
        key = _validate_arm(arm, repo_root=repo_root)
        if key in keys:
            raise ValueError(f"duplicate arm key: {key}")
        keys.append(key)
    if tuple(keys) != tuple(expected):
        raise ValueError(f"arm keys must be {list(expected)} in declared order")


def _validate_arm(value: Any, *, repo_root: Path) -> str:
    if not isinstance(value, Mapping):
        raise ValueError("each arm must be a mapping")
    key = str(value.get("key", "")).strip()
    algo = str(value.get("algo", "")).strip()
    path_value = str(value.get("algo_config_path", "")).strip()
    if not key or not algo or not path_value:
        raise ValueError("each arm needs non-empty key, algo, and algo_config_path")
    config_path = _repo_path(path_value, repo_root)
    if not config_path.is_file():
        raise ValueError(f"arm config does not exist: {path_value}")
    if key in TARGET_ARM_KEYS:
        algorithm_config = _load_yaml_mapping(config_path)
        if algorithm_config.get("predictor_backend") != "constant_velocity":
            raise ValueError(f"target arm {key} must use constant_velocity prediction")
        if algorithm_config.get("allow_predictor_fallback") is not False:
            raise ValueError(f"target arm {key} must disable predictor fallback")
        build_prediction_mpc_config(algorithm_config)
    return key


def _validate_search(value: Any) -> None:
    search = _mapping(value, "search")
    if search.get("design") != "bounded_grid_subset":
        raise ValueError("search.design must be 'bounded_grid_subset'")
    candidate_count = int(search.get("candidate_count", 0))
    if candidate_count <= 0 or candidate_count > 20:
        raise ValueError("search.candidate_count must be in [1, 20]")
    if tuple(search.get("top_parameters", ())) != TOP_PARAMETERS:
        raise ValueError(f"search.top_parameters must be {list(TOP_PARAMETERS)}")
    levels = _validate_parameter_levels(search.get("parameter_levels"))
    _validate_candidate_points(search.get("candidate_points"), candidate_count, levels)


def _validate_parameter_levels(value: Any) -> Mapping[str, Any]:
    levels = _mapping(value, "search.parameter_levels")
    for parameter in TOP_PARAMETERS:
        values = levels.get(parameter)
        if not isinstance(values, list) or len(values) < 2:
            raise ValueError(f"parameter_levels.{parameter} must contain at least two values")
        if not all(_finite_number(item) for item in values):
            raise ValueError(f"parameter_levels.{parameter} must contain finite numbers")
    return levels


def _validate_candidate_points(value: Any, candidate_count: int, levels: Mapping[str, Any]) -> None:
    points = value
    if not isinstance(points, list) or len(points) != candidate_count:
        raise ValueError("candidate_points must match candidate_count")
    ids: set[str] = set()
    override_signatures: set[str] = set()
    incumbent_count = 0
    for point in points:
        if not isinstance(point, Mapping):
            raise ValueError("each candidate point must be a mapping")
        point_id = str(point.get("id", "")).strip()
        overrides = point.get("overrides")
        if not point_id or point_id in ids or not isinstance(overrides, Mapping):
            raise ValueError("candidate point ids must be unique and overrides must be mappings")
        ids.add(point_id)
        if not overrides:
            incumbent_count += 1
        unknown = sorted(set(overrides) - set(TOP_PARAMETERS))
        if unknown:
            raise ValueError(f"candidate {point_id} varies unsupported parameters: {unknown}")
        _validate_candidate_levels(point_id, overrides, levels)
        signature = json.dumps(dict(overrides), sort_keys=True, separators=(",", ":"))
        if signature in override_signatures:
            raise ValueError(f"candidate points repeat override combination: {point_id}")
        override_signatures.add(signature)
    if incumbent_count != 1:
        raise ValueError("search must contain exactly one incumbent candidate point")


def _validate_candidate_levels(
    point_id: str, overrides: Mapping[str, Any], levels: Mapping[str, Any]
) -> None:
    for parameter, override_value in overrides.items():
        if override_value not in levels[parameter]:
            raise ValueError(f"candidate {point_id} uses undeclared level for {parameter}")


def _validate_comparison(value: Any) -> None:
    comparison = _mapping(value, "comparison")
    if comparison.get("primary_metric") != "route_complete_and_collision_free":
        raise ValueError("comparison.primary_metric has drifted")
    if comparison.get("higher_is_better") is not True:
        raise ValueError("comparison.higher_is_better must be true")
    if comparison.get("fallback_degraded_policy") != "exclude_from_success_evidence":
        raise ValueError("comparison must exclude fallback/degraded rows")
    _mapping(comparison.get("hybrid_band_read"), "comparison.hybrid_band_read")


def _summarize_targets(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    summaries = []
    for arm_key in TARGET_ARM_KEYS:
        candidates = [dict(row) for row in rows if row["arm_key"] == arm_key]
        eligible = [row for row in candidates if row["success_rate"] is not None]
        ordered = sorted(
            eligible,
            key=lambda row: (-float(row["success_rate"]), str(row["candidate_id"])),
        )
        summaries.append(
            {
                "arm_key": arm_key,
                "candidate_count": len(candidates),
                "best_candidate": ordered[0] if ordered else None,
                "candidates": candidates,
            }
        )
    return summaries


def _summarize_incumbents(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "arm_key": str(row["arm_key"]),
            "success_rate": row["success_rate"],
            "eligible_episodes": row["eligible_episodes"],
            "excluded_episodes": row["excluded_episodes"],
            "status": row["status"],
        }
        for row in rows
        if not row["target"]
    ]


def _build_read(
    targets: Sequence[Mapping[str, Any]],
    incumbents: Sequence[Mapping[str, Any]],
    all_rows_eligible: bool,
) -> dict[str, Any]:
    target_rates = [
        float(summary["best_candidate"]["success_rate"])
        for summary in targets
        if summary.get("best_candidate") is not None
    ]
    incumbent_rates = [
        float(summary["success_rate"])
        for summary in incumbents
        if summary.get("success_rate") is not None
    ]
    if not all_rows_eligible or len(target_rates) != len(TARGET_ARM_KEYS) or not incumbent_rates:
        return {
            "decision": "blocked",
            "detail": "Complete native/adapter rows are required before the pre-registered read.",
        }
    if max(target_rates) < min(incumbent_rates):
        decision = "structural_reading_strengthens_on_tested_slice"
        detail = "Both best-of-20 MPC rates remain below every incumbent hybrid rate."
    elif min(target_rates) >= max(incumbent_rates):
        decision = "budget_bound_reading_supported_on_tested_slice"
        detail = "Both best-of-20 MPC rates meet or exceed every incumbent hybrid rate."
    else:
        decision = "mixed_or_inconclusive"
        detail = "The best-of-20 MPC rates overlap the incumbent hybrid band."
    return {
        "decision": decision,
        "detail": detail,
        "best_mpc_rates": target_rates,
        "incumbent_rates": incumbent_rates,
        "incumbent_band": {"minimum": min(incumbent_rates), "maximum": max(incumbent_rates)},
    }


def _eligible(row: Mapping[str, Any]) -> bool:
    return (
        str(row.get("execution_mode", "")).strip().lower() in VALID_EXECUTION_MODES
        and str(row.get("readiness_status", "")).strip().lower() in VALID_READINESS_STATUSES
        and str(row.get("availability_status", "")).strip().lower() == "available"
        and row.get("benchmark_success") is True
    )


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _repo_path(value: str, repo_root: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_sha256(config_path: str, *, repo_root: Path) -> str | None:
    path = Path(config_path)
    if not path.is_absolute():
        path = repo_root / path
    return _sha256(path) if path.is_file() else None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _int_field(value: Any, *, field: str) -> int:
    if not _is_int(value):
        raise ValueError(f"{field} must be an integer")
    return int(value)


def _bool_field(value: Any, *, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _format_rate(value: Any) -> str:
    return "NA" if value is None else f"{float(value):.6f}"


__all__ = [
    "CONFIG_SCHEMA",
    "REPORT_SCHEMA",
    "TARGET_ARM_KEYS",
    "TOP_PARAMETERS",
    "analyze_results",
    "build_candidate_plan",
    "config_hash",
    "format_report_markdown",
    "load_sensitivity_config",
    "normalize_episode_record",
    "selected_scenarios",
    "validate_sensitivity_config",
    "write_report",
]
