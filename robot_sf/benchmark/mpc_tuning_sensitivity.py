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
import random
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from robot_sf.benchmark.algorithm_metadata import enrich_algorithm_metadata
from robot_sf.planner.prediction_mpc import build_prediction_mpc_config
from robot_sf.training.scenario_loader import load_scenarios

CONFIG_SCHEMA = "issue_5579_mpc_tuning_sensitivity.v2"
REPORT_SCHEMA = "issue_5579_mpc_tuning_sensitivity_report.v1"
SELECTION_SCHEMA = "issue_5579_mpc_tuning_selection.v1"
INFERENCE_SCHEMA = "issue_5579_mpc_tuning_held_out_inference.v1"
SELECTION_RULE = "highest eligible route-complete collision-free rate; candidate_id tie-break"
STUDY_ID = "issue_5579_mpc_tuning_budget_sensitivity_v2"
TARGET_ARM_KEYS = ("prediction_mpc", "prediction_mpc_cbf")
INCUMBENT_ARM_KEYS = (
    "scenario_adaptive_hybrid_orca_v1",
    "scenario_adaptive_hybrid_orca_v2_collision_guard",
    "hybrid_rule_v3_fast_progress_static_escape",
    "hybrid_rule_v3_fast_progress_static_escape_continuous",
)
TUNING_SCENARIO_IDS = (
    "classic_bottleneck_medium",
    "classic_cross_trap_high",
    "francis2023_intersection_wait",
)
TOP_PARAMETERS = ("max_linear_speed", "horizon_steps", "pedestrian_safety_margin")
VALID_EXECUTION_MODES = frozenset({"native", "adapter", "mixed"})
VALID_READINESS_STATUSES = frozenset({"native", "adapter"})
NATIVE_SOLVER_PLANNER = "PredictionMPCPlannerAdapter"

# The 2026-08-03 #5579 freeze requires "native solver execution" for the canary. That
# phrase names the solver that produced the command, not the benchmark command-space
# `execution_mode` field: the canonical `prediction_mpc` planner is registry-declared as
# an adapter-projected unicycle_vw planner, so a gate bound to `execution_mode: native`
# is unsatisfiable by construction. The packet therefore declares the reachable solver
# contract explicitly and the validator cross-checks it against the runtime planner
# registry, so the campaign gate cannot silently become impossible to pass.
SOLVER_EXECUTION_IDENTITY_FIELDS = (
    "solver_execution_mode",
    "solver_planner_adapter",
    "planner_execution_mode",
    "supports_native_commands",
    "benchmark_execution_mode",
    "benchmark_readiness_status",
)
SOLVER_EXECUTION_REQUIRED_FLAGS = (
    "require_valid_provenance",
    "require_finite_commands",
    "require_solver_update",
    "require_control_update",
    "forbid_solver_failure",
    "forbid_fallback",
)
DEFAULT_SOLVER_EXECUTION: dict[str, Any] = {
    "solver_execution_mode": "prediction_mpc_native_solver",
    "solver_planner_adapter": NATIVE_SOLVER_PLANNER,
    "planner_execution_mode": "adapter",
    "supports_native_commands": False,
    "benchmark_execution_mode": "adapter",
    "benchmark_readiness_status": "adapter",
    **dict.fromkeys(SOLVER_EXECUTION_REQUIRED_FLAGS, True),
}


def solver_execution_contract(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return the declared canary solver-execution contract for a packet.

    Returns:
        The packet's ``canary.solver_execution`` block, or the frozen default.
    """
    if isinstance(config, Mapping):
        canary = config.get("canary")
        if isinstance(canary, Mapping):
            declared = canary.get("solver_execution")
            if isinstance(declared, Mapping):
                return dict(declared)
    return dict(DEFAULT_SOLVER_EXECUTION)


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
    if config.get("study_id") != STUDY_ID:
        raise ValueError(f"study_id must be {STUDY_ID!r}")
    claim_boundary = str(config.get("claim_boundary", ""))
    normalized_claim_boundary = claim_boundary.lower()
    if (
        "diagnostic" not in normalized_claim_boundary
        or "benchmark ranking" not in normalized_claim_boundary
    ):
        raise ValueError("claim_boundary must retain the diagnostic/no-ranking boundary")
    _validate_execution_boundary(config.get("execution_boundary"))
    _validate_scenario_scope(config.get("scenario_scope"), repo_root=root)
    for required_section in ("tuning_scope", "held_out_scope", "canary", "inference"):
        if required_section not in config:
            raise ValueError(f"{required_section} section is required")
    _validate_tuning_scope(
        config.get("tuning_scope"), repo_root=root, scenario_scope=config["scenario_scope"]
    )
    _validate_held_out_scope(
        config.get("held_out_scope"),
        repo_root=root,
        scenario_scope=config["scenario_scope"],
        tuning_scope=config["tuning_scope"],
    )
    _validate_arm_list(config.get("target_arms"), expected=TARGET_ARM_KEYS, repo_root=root)
    _validate_arm_list(
        config.get("incumbent_arms"),
        expected=INCUMBENT_ARM_KEYS,
        repo_root=root,
    )
    _validate_canary(config.get("canary"), target_arms=config["target_arms"])
    _validate_search(config.get("search"))
    _validate_comparison(config.get("comparison"))
    _validate_inference(config.get("inference"))
    return config


def selected_scenarios(
    config: Mapping[str, Any], *, repo_root: Path, scope_name: str = "scenario_scope"
) -> list[dict[str, Any]]:
    """Return one declared fixed scenario scope with paired seeds materialized."""
    scope = _mapping(config.get(scope_name), scope_name)
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


def build_candidate_plan(
    config: Mapping[str, Any],
    *,
    repo_root: Path,
    target_candidate_ids: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Build deterministic target-candidate and incumbent execution rows.

    Returns:
        Ordered target and incumbent execution entries.
    """
    validated = validate_sensitivity_config(config, repo_root=repo_root)
    selected = _validate_target_candidate_ids(target_candidate_ids, validated)
    plan: list[dict[str, Any]] = []
    for arm in validated["target_arms"]:
        base = _load_yaml_mapping(_repo_path(str(arm["algo_config_path"]), repo_root))
        points = validated["search"]["candidate_points"]
        if selected is not None:
            selected_id = selected[str(arm["key"])]
            points = [point for point in points if str(point["id"]) == selected_id]
            if len(points) != 1:  # pragma: no cover - guarded by the helper above.
                raise ValueError(f"selected candidate is not declared for arm {arm['key']}")
        for point in points:
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
    record: Mapping[str, Any],
    *,
    arm_key: str,
    candidate_id: str,
    expected_config_hash: str | None = None,
    solver_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize one runner row while preserving explicit availability provenance.

    Returns:
        A typed episode row suitable for the sensitivity analyzer.
    """
    contract = (
        dict(solver_contract) if solver_contract is not None else dict(DEFAULT_SOLVER_EXECUTION)
    )
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
    algorithm_metadata = record.get("algorithm_metadata")
    planner_runtime = (
        algorithm_metadata.get("planner_runtime")
        if isinstance(algorithm_metadata, Mapping)
        else None
    )
    solver_evidence = _native_solver_evidence(
        algorithm_metadata,
        expected_config_hash=expected_config_hash,
        contract=contract,
    )
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
        "planner_runtime_status": _planner_runtime_status(planner_runtime),
        "solver_execution_mode": solver_evidence["solver_execution_mode"],
        "valid_solver_provenance": solver_evidence["valid_solver_provenance"],
        "finite_commands": solver_evidence["finite_commands"],
        "solver_successes": solver_evidence["solver_successes"],
        "solver_failures": solver_evidence["solver_failures"],
        "fallback_stop_count": solver_evidence["fallback_stop_count"],
        "control_updates": solver_evidence["control_updates"],
        "native_solver_eligible": solver_evidence["native_solver_eligible"],
        "native_solver_exclusion_reasons": solver_evidence["exclusion_reasons"],
    }


def analyze_results(  # noqa: PLR0913
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path,
    config_path: str,
    run_commit: str,
    reproduction_command: str,
    raw_artifact_root: str,
    scope_name: str = "scenario_scope",
    target_candidate_ids: Mapping[str, str] | None = None,
    selection_artifact: str | None = None,
) -> dict[str, Any]:
    """Build a fail-closed report for one declared scope from normalized episode rows.

    Returns:
        Diagnostic report with candidate-level rows and the preregistered read.
    """
    validated = validate_sensitivity_config(config, repo_root=repo_root)
    if scope_name == "held_out_scope" and target_candidate_ids is None:
        raise ValueError(
            "held_out_scope requires tuning-selected target candidates; "
            "do not re-select candidates from held-out outcomes"
        )
    selected = _validate_target_candidate_ids(target_candidate_ids, validated)
    plan = build_candidate_plan(
        validated,
        repo_root=repo_root,
        target_candidate_ids=selected,
    )
    scenario_scope = _mapping(validated.get(scope_name), scope_name)
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

    solver_contract = solver_execution_contract(validated)
    plan_by_key = {(entry["arm_key"], entry["candidate_id"]): entry for entry in plan}
    candidate_rows: list[dict[str, Any]] = []
    for group_key, entry in plan_by_key.items():
        group_rows = grouped[group_key]
        eligible_rows = [row for row in group_rows if _eligible(row, solver_contract)]
        excluded_rows = [row for row in group_rows if not _eligible(row, solver_contract)]
        exclusion_reasons = sorted(
            {
                reason
                for row in excluded_rows
                for reason in _eligibility_reasons(row, solver_contract)
            }
        )
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
                "exclusion_reasons": exclusion_reasons,
            }
        )

    all_rows_eligible = all(row["excluded_episodes"] == 0 for row in candidate_rows)
    target_summary = _summarize_targets(candidate_rows)
    incumbent_summary = _summarize_incumbents(candidate_rows)
    selection_mode = "fixed_from_tuning" if selected is not None else "tuning_search"
    inference = _build_inference(
        validated,
        grouped,
        target_summary,
        scenario_scope=scenario_scope,
        scope_name=scope_name,
        all_rows_eligible=all_rows_eligible,
        solver_contract=solver_contract,
    )
    read = _build_read(
        target_summary,
        incumbent_summary,
        all_rows_eligible,
        selection_mode=selection_mode,
        inference=inference,
    )
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
        "execution_scope_name": scope_name,
        "selection_mode": selection_mode,
        "selected_target_candidates": deepcopy(dict(selected or {})),
        "selection_artifact": selection_artifact,
        "scenario_scope": deepcopy(dict(scenario_scope)),
        "candidate_count": int(validated["search"]["candidate_count"]),
        "executed_candidate_count": len(plan),
        "target_arm_count": len(validated["target_arms"]),
        "total_episode_rows": len(rows),
        "eligible_episode_rows": sum(row["eligible_episodes"] for row in candidate_rows),
        "excluded_episode_rows": sum(row["excluded_episodes"] for row in candidate_rows),
        "candidate_rows": candidate_rows,
        "target_summary": target_summary,
        "incumbent_summary": incumbent_summary,
        "inference": inference,
        "read": read,
        "fallback_degraded_exclusion": (
            "Rows are eligible only with explicit native/adapter/mixed execution, native/adapter "
            "readiness, available status, benchmark_success=true, and planner_runtime_status="
            "eligible; missing, fallback, solver-failure, or malformed runtime diagnostics block "
            "the read."
        ),
    }


def write_tuning_selection(
    report: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    output_path: str | Path,
    config_path: str | Path,
    repo_root: Path,
    source_report: str | Path,
) -> dict[str, Any]:
    """Persist the tuning-selected target candidates for the held-out phase.

    The selection is deliberately a separate artifact so the held-out runner cannot silently
    re-select a target candidate from held-out outcomes. It is bound to the tuning report
    that produced it -- both by content digest and by the persisted report file -- so a
    post-tuning edit of the selected candidate cannot survive ``load_tuning_selection``.

    Returns:
        The validated selection payload written to ``output_path``.
    """
    validated = validate_sensitivity_config(config, repo_root=repo_root)
    selected = _derive_tuning_selection(report, validated)

    config_sha256 = _config_sha256(str(config_path), repo_root=repo_root)
    if config_sha256 is None:
        raise ValueError(f"selection config does not exist: {config_path}")
    source_report_run_commit = str(report.get("run_commit") or "").strip()
    if not source_report_run_commit:
        raise ValueError("tuning selection source report must record a run commit")
    report_path = _repo_path(str(source_report), repo_root)
    if not report_path.is_file():
        raise ValueError(f"tuning selection source report does not exist: {source_report}")
    payload: dict[str, Any] = {
        "schema_version": SELECTION_SCHEMA,
        "study_id": str(validated["study_id"]),
        "selection_scope": "tuning_scope",
        "selection_status": "complete",
        "selection_rule": SELECTION_RULE,
        "config_sha256": config_sha256,
        "selected_target_candidates": selected,
        "source_report": str(source_report),
        "source_report_sha256": _sha256(report_path),
        "source_report_run_commit": source_report_run_commit,
        "selection_input_digest": _selection_input_digest(report),
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _derive_tuning_selection(
    report: Mapping[str, Any], validated: Mapping[str, Any]
) -> dict[str, str]:
    """Re-derive the declared selection rule from one completed tuning report.

    Returns:
        The winning candidate ID for each target arm.
    """
    if report.get("schema_version") != REPORT_SCHEMA:
        raise ValueError(f"tuning selection source report must be {REPORT_SCHEMA!r}")
    if report.get("study_id") != validated["study_id"]:
        raise ValueError("tuning selection source report belongs to a different study")
    if report.get("execution_scope_name") != "tuning_scope":
        raise ValueError("tuning selection must be created from the tuning_scope report")
    if report.get("status") != "complete_diagnostic":
        raise ValueError("cannot create a held-out selection from a blocked tuning report")
    summaries = {
        str(summary.get("arm_key")): summary
        for summary in report.get("target_summary", [])
        if isinstance(summary, Mapping)
    }
    selected: dict[str, str] = {}
    for arm_key in TARGET_ARM_KEYS:
        best = summaries.get(arm_key, {}).get("best_candidate")
        if not isinstance(best, Mapping) or not str(best.get("candidate_id", "")).strip():
            raise ValueError(f"tuning report has no eligible selection for target arm {arm_key}")
        selected[arm_key] = str(best["candidate_id"])
    return dict(_validate_target_candidate_ids(selected, validated) or {})


def _selection_input_digest(report: Mapping[str, Any]) -> str:
    """Return the digest of the exact report content the selection rule consumed.

    Returns:
        A ``sha256`` digest over the report identity and the target candidate table.
    """
    payload = {
        "schema_version": report.get("schema_version"),
        "study_id": report.get("study_id"),
        "status": report.get("status"),
        "execution_scope_name": report.get("execution_scope_name"),
        "config_sha256": report.get("config_sha256"),
        "run_commit": report.get("run_commit"),
        "target_summary": report.get("target_summary"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_tuning_selection(
    path: str | Path,
    config: Mapping[str, Any],
    *,
    config_path: str | Path,
    repo_root: Path,
) -> dict[str, str]:
    """Load and validate the tuning selection required by held-out execution.

    Returns:
        One declared candidate ID for each target arm.
    """
    selection_path = Path(path)
    if not selection_path.is_file():
        raise ValueError(f"held-out phase requires a tuning selection artifact: {selection_path}")
    try:
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid tuning selection artifact: {selection_path}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("tuning selection artifact must be a mapping")
    if payload.get("schema_version") != SELECTION_SCHEMA:
        raise ValueError(f"selection schema_version must be {SELECTION_SCHEMA!r}")
    if payload.get("study_id") != STUDY_ID:
        raise ValueError(f"selection study_id must be {STUDY_ID!r}")
    if payload.get("selection_scope") != "tuning_scope":
        raise ValueError("selection must be derived from tuning_scope")
    if payload.get("selection_status") != "complete":
        raise ValueError("selection status must be complete")
    if payload.get("selection_rule") != SELECTION_RULE:
        raise ValueError("tuning selection rule does not match the frozen packet rule")
    expected_config_sha256 = _config_sha256(str(config_path), repo_root=repo_root)
    if expected_config_sha256 is None or payload.get("config_sha256") != expected_config_sha256:
        raise ValueError("tuning selection config provenance does not match the active config")
    validated = validate_sensitivity_config(config, repo_root=repo_root)
    selected = payload.get("selected_target_candidates")
    normalized = dict(_validate_target_candidate_ids(selected, validated) or {})
    _verify_selection_source_report(
        payload,
        validated,
        repo_root=repo_root,
        expected_config_sha256=expected_config_sha256,
        selected=normalized,
    )
    return normalized


def _verify_selection_source_report(
    payload: Mapping[str, Any],
    validated: Mapping[str, Any],
    *,
    repo_root: Path,
    expected_config_sha256: str,
    selected: Mapping[str, str],
) -> None:
    """Bind the selection artifact to the tuning report that produced it.

    The config digest alone cannot detect a post-tuning edit of the selected candidate,
    because every declared candidate belongs to the same config. The winners are therefore
    re-derived from the referenced tuning report and compared to the recorded selection.
    """
    source_report = str(payload.get("source_report") or "").strip()
    if not source_report:
        raise ValueError("tuning selection must record the source tuning report")
    report_path = _repo_path(source_report, repo_root)
    if not report_path.is_file():
        raise ValueError(f"tuning selection source report is missing: {source_report}")
    if payload.get("source_report_sha256") != _sha256(report_path):
        raise ValueError("tuning selection source report digest does not match the artifact")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid tuning selection source report: {source_report}") from exc
    if not isinstance(report, Mapping):
        raise ValueError("tuning selection source report must be a mapping")
    if report.get("config_sha256") != expected_config_sha256:
        raise ValueError("tuning selection source report was produced by a different config")
    _validate_selection_source_report_run_commit(payload, report)
    if payload.get("selection_input_digest") != _selection_input_digest(report):
        raise ValueError("tuning selection input digest does not match the source report")
    rederived = _derive_tuning_selection(report, validated)
    if rederived != dict(selected):
        raise ValueError(
            "tuning selection does not match the source report selection rule: "
            f"recorded={dict(selected)} rederived={rederived}"
        )


def _validate_selection_source_report_run_commit(
    payload: Mapping[str, Any], report: Mapping[str, Any]
) -> None:
    """Require the selection artifact to bind the source report's run commit."""
    report_run_commit = str(report.get("run_commit") or "").strip()
    if not report_run_commit:
        raise ValueError("tuning selection source report must record a run commit")
    if payload.get("source_report_run_commit") != report_run_commit:
        raise ValueError("tuning selection source report run commit does not match the artifact")


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
    selection_mode = str(report.get("selection_mode", "tuning_search"))
    if selection_mode == "fixed_from_tuning":
        target_heading = "## Tuning-selected target configurations"
        target_selection = "fixed from the tuning-scope selection artifact"
        candidate_heading = "Selected candidate"
    else:
        target_heading = "## Best-found target configurations"
        target_selection = "selected within the tuning scope"
        candidate_heading = "Best candidate"
    lines = [
        "# Issue #5579 MPC Tuning-Budget Sensitivity",
        "",
        f"- Status: `{report.get('status')}`",
        f"- Evidence tier: `{report.get('evidence_tier')}`",
        f"- Claim boundary: {report.get('claim_boundary')}",
        f"- Run commit: `{report.get('run_commit')}`",
        f"- Config: `{report.get('config_path')}`",
        f"- Target selection: {target_selection}",
        "",
        target_heading,
        "",
        f"| Arm | {candidate_heading} | Success rate | Eligible episodes | Excluded episodes |",
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
    lines.extend(_inference_markdown_lines(report.get("inference")))
    lines.extend(
        [
            "",
            "## Pre-registered read",
            "",
            f"- Decision: `{report.get('read', {}).get('decision')}`",
            f"- Detail: {report.get('read', {}).get('detail')}",
            f"- Inference decision: `{report.get('read', {}).get('inference_decision')}`",
            f"- Inference detail: {report.get('read', {}).get('inference_detail')}",
            "",
            "Fallback, degraded, failed, and unavailable rows are never treated as success evidence.",
            "This diagnostic does not change benchmark metrics, roster status, or paper-facing claims.",
            "",
        ]
    )
    return "\n".join(lines)


def _inference_markdown_lines(inference: Any) -> list[str]:
    """Render the frozen held-out paired-contrast table.

    Returns:
        Markdown lines describing the paired deltas, intervals, and Holm decisions.
    """
    if not isinstance(inference, Mapping):
        return []
    lines = [
        "",
        "## Pre-registered held-out inference",
        "",
        f"- Status: `{inference.get('status')}`",
        f"- Detail: {inference.get('detail')}",
    ]
    contrasts = list(inference.get("contrasts") or [])
    if not contrasts:
        return lines
    bootstrap = inference.get("bootstrap") or {}
    multiplicity = inference.get("multiplicity") or {}
    lines.extend(
        [
            f"- Resampling: `{inference.get('resampling_unit')}`, "
            f"{bootstrap.get('replicates')} replicates, "
            f"{bootstrap.get('confidence_level')} interval, seed `{bootstrap.get('seed')}`",
            f"- Multiplicity: `{multiplicity.get('method')}` at "
            f"familywise alpha {multiplicity.get('familywise_alpha')}",
            "",
            "| Target arm | Candidate | Incumbent arm | Paired delta | 95% CI | Holm p | Reject |",
            "| --- | --- | --- | ---: | ---: | ---: | :--: |",
        ]
    )
    for contrast in contrasts:
        lines.append(
            f"| `{contrast['target_arm']}` | `{contrast['target_candidate_id']}` | "
            f"`{contrast['incumbent_arm']}` | {contrast['paired_delta']:.4f} | "
            f"[{contrast['ci_lower']:.4f}, {contrast['ci_upper']:.4f}] | "
            f"{contrast['holm_adjusted_p_value']:.4f} | "
            f"{'yes' if contrast['holm_significant'] else 'no'} |"
        )
    return lines


def _validate_execution_boundary(value: Any) -> None:
    """Require every execution-boundary flag to be false for this diagnostic slice."""
    boundary = _mapping(value, "execution_boundary")
    for field in (
        "full_benchmark_campaign_run_in_this_pr",
        "slurm_or_gpu_submission_in_this_pr",
        "paper_or_dissertation_claim_edit_in_this_pr",
    ):
        if boundary.get(field) is not False:
            raise ValueError(f"execution_boundary.{field} must be false")


def _validate_scenario_scope(value: Any, *, repo_root: Path) -> None:
    """Validate the paired three-scenario, three-seed scope and execution limits."""
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
    if tuple(scenario_ids) != TUNING_SCENARIO_IDS:
        raise ValueError(
            "scenario_scope.scenario_ids must match the frozen 2026-08-03 tuning scenarios"
        )
    seeds = scope.get("seeds")
    if not isinstance(seeds, list) or len(seeds) != 3:
        raise ValueError("scenario_scope.seeds must contain exactly three fixed seeds")
    if len(set(seeds)) != len(seeds) or not all(_is_int(item) for item in seeds):
        raise ValueError("scenario_scope.seeds must be unique integers")
    if int(scope.get("workers", 0)) != 1:
        raise ValueError("scenario_scope.workers must be 1")
    if int(scope.get("horizon", 0)) <= 0 or float(scope.get("dt", 0.0)) <= 0.0:
        raise ValueError("scenario_scope.horizon and dt must be positive")


def compute_scenario_list_hash(scenario_ids: Sequence[str]) -> str:
    """Return deterministic SHA-256 hex digest of sorted scenario IDs."""
    sorted_ids = sorted(str(name) for name in scenario_ids)
    encoded = json.dumps(sorted_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_scenario_ids(source: Path) -> list[str]:
    """Return sorted unique scenario IDs from a declared source matrix."""
    rows = load_scenarios(source, base_dir=source)
    scenario_ids = [str(row.get("name", "")) for row in rows]
    if not scenario_ids or any(not scenario_id for scenario_id in scenario_ids):
        raise ValueError(f"scenario source has missing scenario IDs: {source}")
    if len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError(f"scenario source has duplicate scenario IDs: {source}")
    return sorted(scenario_ids)


def _validate_tuning_scope(
    value: Any, *, repo_root: Path, scenario_scope: Mapping[str, Any]
) -> None:
    """Validate the frozen tuning scope against the primary three-scenario scope."""
    scope = _mapping(value, "tuning_scope")
    source = _repo_path(str(scope.get("source_matrix", "")), repo_root)
    if not source.is_file():
        raise ValueError(f"tuning_scope.source_matrix does not exist: {source}")
    scenario_source = _repo_path(str(scenario_scope.get("source_matrix", "")), repo_root)
    if source.resolve() != scenario_source.resolve():
        raise ValueError("tuning_scope.source_matrix must match scenario_scope.source_matrix")
    scenario_ids = scope.get("scenario_ids")
    if not isinstance(scenario_ids, list) or len(scenario_ids) != 3:
        raise ValueError("tuning_scope.scenario_ids must contain exactly three scenarios")
    if len(set(scenario_ids)) != len(scenario_ids) or not all(
        isinstance(item, str) and item.strip() for item in scenario_ids
    ):
        raise ValueError("tuning_scope.scenario_ids must be unique non-empty strings")
    if scenario_ids != scenario_scope.get("scenario_ids"):
        raise ValueError("tuning_scope.scenario_ids must exactly match scenario_scope.scenario_ids")
    seeds = scope.get("seeds")
    if seeds != [101, 102, 103]:
        raise ValueError("tuning_scope.seeds must be [101, 102, 103]")
    if seeds != scenario_scope.get("seeds"):
        raise ValueError("tuning_scope.seeds must exactly match scenario_scope.seeds")
    _validate_scope_execution_settings(scope, scenario_scope, scope_name="tuning_scope")
    source_ids = _source_scenario_ids(source)
    missing = sorted(set(scenario_ids) - set(source_ids))
    if missing:
        raise ValueError(f"tuning_scope.scenario_ids are absent from source matrix: {missing}")
    scenario_list_hash = str(scope.get("scenario_list_hash", "")).strip()
    expected_hash = compute_scenario_list_hash(scenario_ids)
    if scenario_list_hash != expected_hash:
        raise ValueError(
            f"tuning_scope.scenario_list_hash ({scenario_list_hash!r}) does not match "
            f"expected SHA-256 hash of sorted scenario_ids ({expected_hash!r})"
        )


def _validate_held_out_scope(
    value: Any,
    *,
    repo_root: Path,
    scenario_scope: Mapping[str, Any],
    tuning_scope: Mapping[str, Any],
) -> None:
    """Validate the literal held-out matrix and its frozen exclusion/hash contract."""
    scope = _mapping(value, "held_out_scope")
    source = _repo_path(str(scope.get("source_matrix", "")), repo_root)
    if not source.is_file():
        raise ValueError(f"held_out_scope.source_matrix does not exist: {source}")
    scenario_source = _repo_path(str(scenario_scope.get("source_matrix", "")), repo_root)
    if source.resolve() != scenario_source.resolve():
        raise ValueError("held_out_scope.source_matrix must match scenario_scope.source_matrix")
    if str(scope.get("seed_set", "")) != "paper_eval_s10":
        raise ValueError("held_out_scope.seed_set must be 'paper_eval_s10'")
    seeds = scope.get("seeds")
    expected_seeds = list(range(111, 121))
    if seeds != expected_seeds:
        raise ValueError(f"held_out_scope.seeds must be {expected_seeds}")
    _validate_scope_execution_settings(scope, scenario_scope, scope_name="held_out_scope")
    _validate_held_out_matrix(scope, source, tuning_scope)


def _validate_scope_execution_settings(
    scope: Mapping[str, Any], scenario_scope: Mapping[str, Any], *, scope_name: str
) -> None:
    """Require a phase's execution settings to match the declared tuning settings."""
    if int(scope.get("horizon", 0)) != int(scenario_scope.get("horizon", 0)):
        raise ValueError(f"{scope_name}.horizon must match scenario_scope.horizon")
    if float(scope.get("dt", 0.0)) != float(scenario_scope.get("dt", 0.0)):
        raise ValueError(f"{scope_name}.dt must match scenario_scope.dt")
    if int(scope.get("workers", 0)) != int(scenario_scope.get("workers", 0)):
        raise ValueError(f"{scope_name}.workers must match scenario_scope.workers")


def _validate_held_out_matrix(
    scope: Mapping[str, Any], source: Path, tuning_scope: Mapping[str, Any]
) -> None:
    """Require the held-out IDs and hash to equal the source minus tuning IDs."""
    excluded = scope.get("excluded_scenarios")
    expected_excluded = list(tuning_scope.get("scenario_ids", []))
    if excluded != expected_excluded:
        raise ValueError(
            "held_out_scope.excluded_scenarios must exactly match tuning_scope.scenario_ids"
        )
    source_ids = _source_scenario_ids(source)
    expected_held_out = sorted(set(source_ids) - set(expected_excluded))
    scenario_ids = scope.get("scenario_ids")
    if scenario_ids != expected_held_out:
        raise ValueError(
            "held_out_scope.scenario_ids must exactly match the source matrix minus "
            "tuning_scope.scenario_ids"
        )
    scenario_list_hash = str(scope.get("scenario_list_hash", "")).strip()
    expected_hash = compute_scenario_list_hash(expected_held_out)
    if scenario_list_hash != expected_hash:
        raise ValueError(
            f"held_out_scope.scenario_list_hash ({scenario_list_hash!r}) does not match "
            f"the frozen eligible matrix hash ({expected_hash!r})"
        )


def _validate_canary(value: Any, *, target_arms: Any) -> None:
    """Validate canary section specifying 6/6 eligibility at seed 101."""
    canary = _mapping(value, "canary")
    if canary.get("seed") != 101:
        raise ValueError("canary.seed must be 101")
    if canary.get("required_eligible_episodes") != 6:
        raise ValueError("canary.required_eligible_episodes must be 6")
    if canary.get("target_eligible_ratio") != "6/6":
        raise ValueError("canary.target_eligible_ratio must be '6/6'")
    if canary.get("stop_on_ineligible") is not True:
        raise ValueError("canary.stop_on_ineligible must be true")
    _validate_solver_execution(canary.get("solver_execution"), target_arms=target_arms)


def _validate_solver_execution(value: Any, *, target_arms: Any) -> None:
    """Validate the declared solver contract and prove the canary gate is reachable.

    A canary that no runtime row can satisfy is not a stop rule, it is a permanent
    block. The declared planner identity is therefore checked against the runtime
    algorithm-metadata registry for every declared target arm.

    The ``require_*``/``forbid_*`` flags are freeze declarations, not switches: the
    predicates always enforce them and this validator rejects any packet that tries to
    declare a weaker canary than the 2026-08-03 #5579 contract.
    """
    contract = _mapping(value, "canary.solver_execution")
    missing = [field for field in SOLVER_EXECUTION_IDENTITY_FIELDS if field not in contract]
    if missing:
        raise ValueError(f"canary.solver_execution is missing declared fields: {missing}")
    if not str(contract["solver_execution_mode"]).strip():
        raise ValueError("canary.solver_execution.solver_execution_mode must be a non-empty token")
    for flag in SOLVER_EXECUTION_REQUIRED_FLAGS:
        if contract.get(flag) is not True:
            raise ValueError(f"canary.solver_execution.{flag} must be true")
    if str(contract["benchmark_execution_mode"]).strip().lower() not in VALID_EXECUTION_MODES:
        raise ValueError("canary.solver_execution.benchmark_execution_mode is not a valid mode")
    if str(contract["benchmark_readiness_status"]).strip().lower() not in VALID_READINESS_STATUSES:
        raise ValueError("canary.solver_execution.benchmark_readiness_status is not a valid status")
    _validate_solver_execution_reachable(contract, target_arms=target_arms)


def _validate_solver_execution_reachable(contract: Mapping[str, Any], *, target_arms: Any) -> None:
    """Require the declared solver identity to match the runtime planner registry."""
    if not isinstance(target_arms, Sequence) or isinstance(target_arms, str):
        raise ValueError("target_arms must be declared before canary.solver_execution")
    for arm in target_arms:
        arm_map = _mapping(arm, "target_arms[]")
        reachable = _registry_planner_kinematics(str(arm_map.get("algo", "")))
        for field, declared_key in (
            ("adapter_name", "solver_planner_adapter"),
            ("execution_mode", "planner_execution_mode"),
            ("supports_native_commands", "supports_native_commands"),
        ):
            if reachable.get(field) != contract[declared_key]:
                raise ValueError(
                    "canary.solver_execution is unreachable for target arm "
                    f"{arm_map.get('key')!r}: runtime planner_kinematics.{field}="
                    f"{reachable.get(field)!r} but the packet declares {declared_key}="
                    f"{contract[declared_key]!r}"
                )


def _registry_planner_kinematics(algo: str) -> dict[str, Any]:
    """Return the runtime planner kinematics the benchmark runner emits for one algorithm.

    Returns:
        The enriched ``planner_kinematics`` block produced by the shared adapter path.
    """
    if not algo:
        raise ValueError("target arm must declare an algo key")
    metadata = enrich_algorithm_metadata(
        algo=algo,
        metadata={"status": "ok"},
        execution_mode="adapter",
        robot_kinematics="differential_drive",
    )
    kinematics = metadata.get("planner_kinematics")
    if not isinstance(kinematics, Mapping):
        raise ValueError(f"algorithm {algo!r} declares no planner kinematics contract")
    return dict(kinematics)


def _validate_inference(value: Any) -> None:
    """Validate the paired held-out bootstrap and eight-contrast correction contract."""
    inference = _mapping(value, "inference")
    if inference.get("inference_population") != "fixed_declared_held_out_suite":
        raise ValueError("inference.inference_population must be fixed_declared_held_out_suite")
    if inference.get("estimand") != "paired_delta":
        raise ValueError("inference.estimand must be paired_delta")
    if inference.get("primary_metric") != "route_complete_and_collision_free":
        raise ValueError("inference.primary_metric must match comparison.primary_metric")
    if inference.get("resampling_unit") != "paired_seed_block":
        raise ValueError("inference.resampling_unit must be paired_seed_block")
    _validate_bootstrap(inference.get("bootstrap"))
    _validate_multiplicity(inference.get("multiplicity"))


def _validate_bootstrap(value: Any) -> None:
    """Validate the paired seed-block percentile bootstrap settings."""
    bootstrap = _mapping(value, "inference.bootstrap")
    if bootstrap.get("method") != "paired_seed_block_percentile_bootstrap":
        raise ValueError(
            "inference.bootstrap.method must be paired_seed_block_percentile_bootstrap"
        )
    if bootstrap.get("confidence_level") != 0.95:
        raise ValueError("inference.bootstrap.confidence_level must be 0.95")
    if not _is_int(bootstrap.get("replicates")) or int(bootstrap["replicates"]) <= 0:
        raise ValueError("inference.bootstrap.replicates must be a positive integer")
    if not _is_int(bootstrap.get("seed")):
        raise ValueError("inference.bootstrap.seed must be a frozen integer for reproducibility")


def _validate_multiplicity(value: Any) -> None:
    """Validate Holm-Bonferroni correction across the eight declared contrasts."""
    multiplicity = _mapping(value, "inference.multiplicity")
    if multiplicity.get("method") != "holm_bonferroni":
        raise ValueError("inference.multiplicity.method must be holm_bonferroni")
    if multiplicity.get("family") != "target_arm_vs_incumbent_arm":
        raise ValueError("inference.multiplicity.family must be target_arm_vs_incumbent_arm")
    if multiplicity.get("familywise_alpha") != 0.05:
        raise ValueError("inference.multiplicity.familywise_alpha must be 0.05")
    if multiplicity.get("contrast_count") != 8:
        raise ValueError("inference.multiplicity.contrast_count must be 8")
    contrasts = multiplicity.get("contrasts")
    expected_contrasts = [
        {"target_arm": target_arm, "incumbent_arm": incumbent_arm}
        for target_arm in TARGET_ARM_KEYS
        for incumbent_arm in INCUMBENT_ARM_KEYS
    ]
    if contrasts != expected_contrasts:
        raise ValueError(
            "inference.multiplicity.contrasts must declare each target/incumbent pair exactly once"
        )


def _validate_arm_list(value: Any, *, expected: Sequence[str], repo_root: Path) -> None:
    """Validate an arm list matches the expected keys in declared order."""
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
    """Validate one arm and its config, enforcing target-arm prediction constraints.

    Returns:
        The validated arm's key string.
    """
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
        if algorithm_config.get("fallback_to_stop") is not False:
            raise ValueError(f"target arm {key} must disable solver fallback_to_stop")
        build_prediction_mpc_config(algorithm_config)
    return key


def _validate_search(value: Any) -> None:
    """Validate the bounded grid-subset search design, parameters, and candidate points."""
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
    """Validate that each top parameter declares at least two finite levels.

    Returns:
        The validated parameter-levels mapping keyed by parameter name.
    """
    levels = _mapping(value, "search.parameter_levels")
    for parameter in TOP_PARAMETERS:
        values = levels.get(parameter)
        if not isinstance(values, list) or len(values) < 2:
            raise ValueError(f"parameter_levels.{parameter} must contain at least two values")
        if not all(_finite_number(item) for item in values):
            raise ValueError(f"parameter_levels.{parameter} must contain finite numbers")
    return levels


def _validate_candidate_points(value: Any, candidate_count: int, levels: Mapping[str, Any]) -> None:
    """Validate unique, non-repeating candidate points with exactly one incumbent."""
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


def _validate_target_candidate_ids(value: Any, config: Mapping[str, Any]) -> dict[str, str] | None:
    """Validate one tuning-selected candidate ID for each target arm.

    Returns:
        A normalized target-arm-to-candidate mapping, or ``None`` when no selection was supplied.
    """
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError("target_candidate_ids must be a mapping")
    normalized: dict[str, str] = {}
    for raw_arm, raw_candidate in value.items():
        if not isinstance(raw_arm, str) or not isinstance(raw_candidate, str):
            raise ValueError("target_candidate_ids keys and values must be strings")
        normalized[raw_arm.strip()] = raw_candidate.strip()
    if set(normalized) != set(TARGET_ARM_KEYS) or any(
        not arm or not candidate for arm, candidate in normalized.items()
    ):
        raise ValueError(
            "target_candidate_ids must contain exactly one non-empty candidate for each target arm"
        )
    candidate_ids = {
        str(point["id"])
        for point in _mapping(config.get("search"), "search").get("candidate_points", [])
        if isinstance(point, Mapping) and "id" in point
    }
    invalid = sorted(set(normalized.values()) - candidate_ids)
    if invalid:
        raise ValueError(f"target_candidate_ids contain undeclared candidates: {invalid}")
    return normalized


def _validate_candidate_levels(
    point_id: str, overrides: Mapping[str, Any], levels: Mapping[str, Any]
) -> None:
    """Require every candidate override value to be a declared level for its parameter."""
    for parameter, override_value in overrides.items():
        if override_value not in levels[parameter]:
            raise ValueError(f"candidate {point_id} uses undeclared level for {parameter}")


def _validate_comparison(value: Any) -> None:
    """Validate the preregistered comparison metric and fallback-exclusion policy."""
    comparison = _mapping(value, "comparison")
    if comparison.get("primary_metric") != "route_complete_and_collision_free":
        raise ValueError("comparison.primary_metric has drifted")
    if comparison.get("higher_is_better") is not True:
        raise ValueError("comparison.higher_is_better must be true")
    if comparison.get("fallback_degraded_policy") != "exclude_from_success_evidence":
        raise ValueError("comparison must exclude fallback/degraded rows")
    _mapping(comparison.get("hybrid_band_read"), "comparison.hybrid_band_read")


def _summarize_targets(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize each target arm's best candidate by success rate.

    Returns:
        A list of per-target-arm summaries including the best candidate by success rate.
    """
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
    """Summarize the incumbent hybrid arms' success rates and episode counts.

    Returns:
        A list of incumbent-arm summaries with success rates and episode counts.
    """
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


def _build_inference(
    config: Mapping[str, Any],
    grouped: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    target_summary: Sequence[Mapping[str, Any]],
    *,
    scenario_scope: Mapping[str, Any],
    scope_name: str,
    all_rows_eligible: bool,
    solver_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Produce the frozen paired-bootstrap / Holm-Bonferroni held-out inference block.

    The estimand is the paired difference in collision-free route completion between each
    target arm and each incumbent arm on the declared held-out suite. The resampling unit
    is the paired seed block: one bootstrap replicate resamples whole seeds with
    replacement and reuses the same resampled seeds for every arm and every contrast, so
    the pairing declared by the packet is preserved.

    Returns:
        The inference block recorded in the report and consumed by the pre-registered read.
    """
    inference_config = _mapping(config.get("inference"), "inference")
    bootstrap_config = _mapping(inference_config.get("bootstrap"), "inference.bootstrap")
    multiplicity_config = _mapping(inference_config.get("multiplicity"), "inference.multiplicity")
    header: dict[str, Any] = {
        "schema_version": INFERENCE_SCHEMA,
        "inference_population": str(inference_config["inference_population"]),
        "estimand": str(inference_config["estimand"]),
        "primary_metric": str(inference_config["primary_metric"]),
        "resampling_unit": str(inference_config["resampling_unit"]),
        "bootstrap": {
            "method": str(bootstrap_config["method"]),
            "confidence_level": float(bootstrap_config["confidence_level"]),
            "replicates": int(bootstrap_config["replicates"]),
            "seed": int(bootstrap_config["seed"]),
        },
        "multiplicity": {
            "method": str(multiplicity_config["method"]),
            "family": str(multiplicity_config["family"]),
            "familywise_alpha": float(multiplicity_config["familywise_alpha"]),
            "contrast_count": int(multiplicity_config["contrast_count"]),
        },
        "contrasts": [],
    }
    if scope_name != "held_out_scope":
        return {
            **header,
            "status": "not_applicable",
            "detail": (
                "The frozen inference contract is defined on the declared held-out suite; "
                f"this report executed {scope_name}."
            ),
        }
    if not all_rows_eligible:
        return {
            **header,
            "status": "blocked",
            "detail": (
                "Complete solver-valid rows are required before the frozen paired bootstrap "
                "and Holm-Bonferroni correction can be computed."
            ),
        }

    seeds = [int(seed) for seed in scenario_scope["seeds"]]
    scenario_ids = [str(scenario_id) for scenario_id in scenario_scope["scenario_ids"]]
    selected_candidates = {
        str(summary["arm_key"]): str(summary["best_candidate"]["candidate_id"])
        for summary in target_summary
        if summary.get("best_candidate") is not None
    }
    arm_seed_successes: dict[str, dict[int, int]] = {}
    for (arm_key, candidate_id), rows in grouped.items():
        if arm_key in TARGET_ARM_KEYS and selected_candidates.get(arm_key) != candidate_id:
            continue
        arm_seed_successes[arm_key] = _seed_block_successes(
            rows, seeds=seeds, scenario_count=len(scenario_ids), solver_contract=solver_contract
        )
    missing_arms = [
        arm_key
        for arm_key in (*TARGET_ARM_KEYS, *INCUMBENT_ARM_KEYS)
        if arm_key not in arm_seed_successes
    ]
    if missing_arms:
        return {
            **header,
            "status": "blocked",
            "detail": f"paired inference is missing complete arms: {missing_arms}",
        }

    paired_units = len(seeds) * len(scenario_ids)
    replicates = int(bootstrap_config["replicates"])
    resampled_seed_blocks = _resample_seed_blocks(
        seeds, replicates=replicates, seed=int(bootstrap_config["seed"])
    )
    alpha = 1.0 - float(bootstrap_config["confidence_level"])
    contrasts: list[dict[str, Any]] = []
    for declared in multiplicity_config["contrasts"]:
        target_arm = str(declared["target_arm"])
        incumbent_arm = str(declared["incumbent_arm"])
        target_blocks = arm_seed_successes[target_arm]
        incumbent_blocks = arm_seed_successes[incumbent_arm]
        target_rate = sum(target_blocks.values()) / paired_units
        incumbent_rate = sum(incumbent_blocks.values()) / paired_units
        deltas = [
            (
                sum(target_blocks[seed] for seed in block)
                - sum(incumbent_blocks[seed] for seed in block)
            )
            / paired_units
            for block in resampled_seed_blocks
        ]
        lower, upper = _percentile_interval(deltas, alpha=alpha)
        contrasts.append(
            {
                "target_arm": target_arm,
                "target_candidate_id": selected_candidates[target_arm],
                "incumbent_arm": incumbent_arm,
                "paired_units": paired_units,
                "seed_blocks": len(seeds),
                "target_rate": target_rate,
                "incumbent_rate": incumbent_rate,
                "paired_delta": target_rate - incumbent_rate,
                "ci_lower": lower,
                "ci_upper": upper,
                "p_value": _bootstrap_two_sided_p_value(deltas),
            }
        )
    _apply_holm_bonferroni(contrasts, alpha=float(multiplicity_config["familywise_alpha"]))
    return {
        **header,
        "status": "complete",
        "detail": (
            "Paired seed-block percentile bootstrap with Holm-Bonferroni correction over the "
            f"{len(contrasts)} declared target-versus-incumbent contrasts."
        ),
        "paired_units": paired_units,
        "seed_blocks": seeds,
        "selected_target_candidates": dict(selected_candidates),
        "contrasts": contrasts,
        "significant_contrasts": [
            f"{contrast['target_arm']} vs {contrast['incumbent_arm']}"
            for contrast in contrasts
            if contrast["holm_significant"]
        ],
    }


def _seed_block_successes(
    rows: Sequence[Mapping[str, Any]],
    *,
    seeds: Sequence[int],
    scenario_count: int,
    solver_contract: Mapping[str, Any],
) -> dict[int, int]:
    """Return eligible successes per seed block for one arm.

    Returns:
        A per-seed success count covering the full declared scenario slice.

    Raises:
        ValueError: If a seed block is not fully covered by eligible rows.
    """
    counts = {int(seed): 0 for seed in seeds}
    totals = {int(seed): 0 for seed in seeds}
    for row in rows:
        if not _eligible(row, solver_contract):
            continue
        seed = int(row["seed"])
        if seed not in counts:
            continue
        totals[seed] += 1
        counts[seed] += 1 if row.get("success") is True else 0
    incomplete = [seed for seed in totals if totals[seed] != scenario_count]
    if incomplete:
        raise ValueError(f"paired seed blocks are incomplete for seeds: {sorted(incomplete)}")
    return {int(seed): int(counts[int(seed)]) for seed in seeds}


def _resample_seed_blocks(seeds: Sequence[int], *, replicates: int, seed: int) -> list[list[int]]:
    """Return the deterministic paired seed-block bootstrap resamples.

    Returns:
        One resampled seed list per bootstrap replicate.
    """
    rng = random.Random(seed)
    pool = [int(value) for value in seeds]
    return [[rng.choice(pool) for _ in pool] for _ in range(replicates)]


def _percentile_interval(values: Sequence[float], *, alpha: float) -> tuple[float, float]:
    """Return the two-sided percentile interval of a bootstrap distribution.

    Returns:
        Lower and upper percentile bounds.
    """
    ordered = sorted(values)
    if not ordered:
        return (float("nan"), float("nan"))
    return (
        _percentile(ordered, alpha / 2.0),
        _percentile(ordered, 1.0 - alpha / 2.0),
    )


def _percentile(ordered: Sequence[float], quantile: float) -> float:
    """Return a linear-interpolated percentile of a sorted sequence.

    Returns:
        The interpolated percentile value.
    """
    if len(ordered) == 1:
        return float(ordered[0])
    position = quantile * (len(ordered) - 1)
    lower_index = math.floor(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    weight = position - lower_index
    return float(ordered[lower_index] * (1.0 - weight) + ordered[upper_index] * weight)


def _bootstrap_two_sided_p_value(deltas: Sequence[float]) -> float:
    """Return the two-sided achieved significance level for a zero paired delta.

    Returns:
        A p-value in ``[1 / (replicates + 1), 1.0]``.
    """
    replicates = len(deltas)
    if replicates == 0:
        return 1.0
    at_or_below = sum(1 for delta in deltas if delta <= 0.0)
    at_or_above = sum(1 for delta in deltas if delta >= 0.0)
    tail = min(at_or_below, at_or_above) / replicates
    return float(min(1.0, max(2.0 * tail, 1.0 / (replicates + 1))))


def _apply_holm_bonferroni(contrasts: list[dict[str, Any]], *, alpha: float) -> None:
    """Annotate each contrast with its Holm-Bonferroni rank, adjusted p-value, and decision."""
    family_size = len(contrasts)
    ordered = sorted(range(family_size), key=lambda index: contrasts[index]["p_value"])
    running_max = 0.0
    for rank, index in enumerate(ordered, start=1):
        adjusted = min(1.0, (family_size - rank + 1) * float(contrasts[index]["p_value"]))
        running_max = max(running_max, adjusted)
        contrasts[index]["holm_rank"] = rank
        contrasts[index]["holm_family_size"] = family_size
        contrasts[index]["holm_adjusted_p_value"] = running_max
        contrasts[index]["holm_significant"] = running_max <= alpha


def _build_read(
    targets: Sequence[Mapping[str, Any]],
    incumbents: Sequence[Mapping[str, Any]],
    all_rows_eligible: bool,
    *,
    selection_mode: str,
    inference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Produce the preregistered decision comparing target MPC rates to the incumbent band.

    Returns:
        A decision mapping describing whether the MPC read is blocked, supported, or mixed.
    """
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
            "detail": (
                "Complete native solver/provenance-valid rows are required before the "
                "pre-registered read."
            ),
            **_inference_read(inference),
        }
    if max(target_rates) < min(incumbent_rates):
        decision = "structural_reading_strengthens_on_tested_slice"
        detail = (
            "Both tuning-selected MPC rates remain below every incumbent hybrid rate."
            if selection_mode == "fixed_from_tuning"
            else "Both best-of-20 MPC rates remain below every incumbent hybrid rate."
        )
    elif min(target_rates) >= max(incumbent_rates):
        decision = "budget_bound_reading_supported_on_tested_slice"
        detail = (
            "Both tuning-selected MPC rates meet or exceed every incumbent hybrid rate."
            if selection_mode == "fixed_from_tuning"
            else "Both best-of-20 MPC rates meet or exceed every incumbent hybrid rate."
        )
    else:
        decision = "mixed_or_inconclusive"
        detail = (
            "The tuning-selected MPC rates overlap the incumbent hybrid band."
            if selection_mode == "fixed_from_tuning"
            else "The best-of-20 MPC rates overlap the incumbent hybrid band."
        )
    read = {
        "decision": decision,
        "detail": detail,
        "selection_mode": selection_mode,
        "best_mpc_rates": target_rates,
        "incumbent_rates": incumbent_rates,
        "incumbent_band": {"minimum": min(incumbent_rates), "maximum": max(incumbent_rates)},
    }
    read.update(_inference_read(inference))
    return read


def _inference_read(inference: Mapping[str, Any] | None) -> dict[str, Any]:
    """Fold the frozen paired-bootstrap/Holm result into the pre-registered read.

    Returns:
        The inference-derived fields of the pre-registered read.
    """
    status = str((inference or {}).get("status", "missing"))
    if status != "complete":
        return {
            "inference_status": status,
            "inference_decision": "not_established",
            "inference_detail": str(
                (inference or {}).get("detail", "no frozen paired-bootstrap inference is available")
            ),
        }
    contrasts = list(inference.get("contrasts") or [])
    significant = [contrast for contrast in contrasts if contrast.get("holm_significant")]
    favoring_target = [
        contrast for contrast in significant if float(contrast["paired_delta"]) > 0.0
    ]
    favoring_incumbent = [
        contrast for contrast in significant if float(contrast["paired_delta"]) < 0.0
    ]
    if not significant:
        decision = "no_contrast_significant_after_holm"
    elif favoring_target and not favoring_incumbent:
        decision = "target_advantage_supported_on_declared_contrasts"
    elif favoring_incumbent and not favoring_target:
        decision = "incumbent_advantage_supported_on_declared_contrasts"
    else:
        decision = "mixed_significant_contrasts"
    return {
        "inference_status": status,
        "inference_decision": decision,
        "inference_detail": (
            f"{len(significant)}/{len(contrasts)} declared contrasts reject the zero paired "
            "delta after Holm-Bonferroni correction of the paired seed-block bootstrap."
        ),
        "inference_significant_contrasts": [
            f"{contrast['target_arm']} vs {contrast['incumbent_arm']}" for contrast in significant
        ],
        "inference_paired_deltas": [
            {
                "target_arm": contrast["target_arm"],
                "incumbent_arm": contrast["incumbent_arm"],
                "paired_delta": contrast["paired_delta"],
                "ci_lower": contrast["ci_lower"],
                "ci_upper": contrast["ci_upper"],
                "holm_adjusted_p_value": contrast["holm_adjusted_p_value"],
            }
            for contrast in contrasts
        ],
    }


def _native_solver_evidence(
    value: Any,
    *,
    expected_config_hash: str | None,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract strict native-solver evidence from one raw algorithm metadata mapping.

    Returns:
        A normalized strict-evidence mapping used by the canary gate.
    """
    if not isinstance(value, Mapping):
        return _missing_solver_evidence()
    kinematics = value.get("planner_kinematics")
    runtime = value.get("planner_runtime")
    kinematics = kinematics if isinstance(kinematics, Mapping) else {}
    runtime = runtime if isinstance(runtime, Mapping) else {}
    solver_execution_mode, identity_reasons = _solver_identity(value, kinematics, runtime, contract)
    valid_provenance, provenance_reasons = _solver_provenance(value, expected_config_hash)
    runtime_evidence = _solver_runtime_evidence(runtime)
    reasons = identity_reasons + provenance_reasons + runtime_evidence["exclusion_reasons"]
    return {
        "solver_execution_mode": solver_execution_mode,
        "valid_solver_provenance": valid_provenance,
        "finite_commands": runtime_evidence["finite_commands"],
        "solver_successes": runtime_evidence["solver_successes"],
        "solver_failures": runtime_evidence["solver_failures"],
        "fallback_stop_count": runtime_evidence["fallback_stop_count"],
        "control_updates": runtime_evidence["control_updates"],
        "native_solver_eligible": not reasons,
        "exclusion_reasons": sorted(set(reasons)),
    }


def _missing_solver_evidence() -> dict[str, Any]:
    """Return the fail-closed evidence shape for missing algorithm metadata."""
    return {
        "solver_execution_mode": "unknown",
        "valid_solver_provenance": False,
        "finite_commands": False,
        "solver_successes": None,
        "solver_failures": None,
        "fallback_stop_count": None,
        "control_updates": None,
        "native_solver_eligible": False,
        "exclusion_reasons": ["algorithm_metadata_missing"],
    }


def _solver_identity(
    value: Mapping[str, Any],
    kinematics: Mapping[str, Any],
    runtime: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> tuple[str, list[str]]:
    """Validate the declared planner identity against the packet solver contract.

    Returns:
        The normalized solver mode and any identity exclusion reasons.
    """
    expected_adapter = str(contract["solver_planner_adapter"])
    expected_planner_mode = str(contract["planner_execution_mode"]).strip().lower()
    expected_solver_mode = str(contract["solver_execution_mode"]).strip().lower()
    adapter_name = str(kinematics.get("adapter_name", "")).strip()
    planner_execution_mode = str(kinematics.get("execution_mode", "")).strip().lower()
    identity_matches = (
        adapter_name == expected_adapter
        and planner_execution_mode == expected_planner_mode
        and kinematics.get("supports_native_commands") is contract["supports_native_commands"]
    )
    solver_mode = (
        str(
            runtime.get("solver_execution_mode")
            or kinematics.get("solver_execution_mode")
            or value.get("solver_execution_mode")
            or ""
        )
        .strip()
        .lower()
    )
    if not solver_mode and identity_matches:
        solver_mode = expected_solver_mode
    reasons: list[str] = []
    if solver_mode != expected_solver_mode:
        reasons.append("native_solver_execution_missing")
    if adapter_name != expected_adapter:
        reasons.append("unexpected_solver_planner")
    if kinematics.get("supports_native_commands") is not contract["supports_native_commands"]:
        reasons.append("solver_command_support_mismatch")
    if planner_execution_mode != expected_planner_mode:
        reasons.append("planner_execution_mode_mismatch")
    return solver_mode or "unknown", reasons


def _solver_provenance(
    value: Mapping[str, Any], expected_config_hash: str | None
) -> tuple[bool, list[str]]:
    """Validate effective algorithm config identity and the expected candidate hash.

    Returns:
        A validity flag and any provenance exclusion reasons.
    """
    effective_config = value.get("config")
    metadata_hash = value.get("config_hash")
    valid = (
        isinstance(effective_config, Mapping)
        and isinstance(metadata_hash, str)
        and bool(metadata_hash)
        and config_hash(effective_config) == metadata_hash
    )
    if expected_config_hash is not None and metadata_hash != expected_config_hash:
        valid = False
    return valid, [] if valid else ["solver_provenance_invalid"]


def _solver_runtime_evidence(runtime: Mapping[str, Any]) -> dict[str, Any]:
    """Validate solver counters, finite commands, and a successful control update.

    Returns:
        Normalized counters, command evidence, and exclusion reasons.
    """
    solver_successes = _optional_counter(runtime.get("solver_successes"))
    solver_failures = _optional_counter(runtime.get("solver_failures"))
    fallback_stop_count = _optional_counter(runtime.get("fallback_stop_count"))
    control_updates = _control_update_count(runtime)
    finite_commands = _finite_command_evidence(runtime)
    reasons = _solver_counter_reasons(
        solver_successes, solver_failures, fallback_stop_count, runtime
    )
    if not finite_commands:
        reasons.append("commands_not_finite")
    if control_updates is None or control_updates < 1:
        reasons.append("control_update_missing")
    return {
        "finite_commands": finite_commands,
        "solver_successes": solver_successes,
        "solver_failures": solver_failures,
        "fallback_stop_count": fallback_stop_count,
        "control_updates": control_updates,
        "exclusion_reasons": reasons,
    }


def _solver_counter_reasons(
    solver_successes: int | None,
    solver_failures: int | None,
    fallback_stop_count: int | None,
    runtime: Mapping[str, Any],
) -> list[str]:
    """Return counter and fallback exclusion reasons for strict solver evidence."""
    reasons: list[str] = []
    if solver_successes is None:
        reasons.append("solver_successes_missing")
    elif solver_successes < 1:
        reasons.append("solver_update_missing")
    if solver_failures is None:
        reasons.append("solver_failures_missing")
    elif solver_failures > 0:
        reasons.append("solver_failure")
    if fallback_stop_count is None:
        reasons.append("fallback_stop_count_missing")
    elif fallback_stop_count > 0:
        reasons.append("fallback")
    fallback_count = _optional_counter(runtime.get("fallback_count"))
    if fallback_count is not None and fallback_count > 0:
        reasons.append("fallback")
    if runtime.get("fallback_triggered") is True:
        reasons.append("fallback")
    return reasons


def _finite_command_evidence(runtime: Mapping[str, Any]) -> bool:
    """Require finite linear and angular command summaries.

    Returns:
        True when both command summaries are finite and no explicit false marker is present.
    """
    finite = all(
        _finite_number(runtime.get(field)) for field in ("mean_abs_linear", "mean_abs_angular")
    )
    return finite and runtime.get("commands_finite") is not False


def _control_update_count(runtime: Mapping[str, Any]) -> int | None:
    """Return successful control updates, using the runner's nonzero-command counter fallback."""
    count = _optional_counter(
        runtime.get("successful_control_updates", runtime.get("control_updates"))
    )
    return count if count is not None else _optional_counter(runtime.get("nonzero_command_count"))


def validate_canary_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    scenario_ids: Sequence[str],
    seed: int,
    required_eligible: int,
    target_arm_keys: Sequence[str] = TARGET_ARM_KEYS,
    candidate_id: str = "incumbent",
    solver_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the exact native-solver canary gate to normalized rows.

    Every row must match the packet's declared solver-execution contract and carry valid
    effective-config provenance, finite commands, and a successful solver/control update,
    with no solver failure or fallback. The expected two-arm by three-scenario key set is
    closed before any production phase can proceed.

    Returns:
        A status mapping containing exact-key, eligibility, and exclusion diagnostics.
    """
    contract = (
        dict(solver_contract) if solver_contract is not None else dict(DEFAULT_SOLVER_EXECUTION)
    )
    expected_keys = {
        (str(arm_key), str(candidate_id), str(scenario_id), int(seed))
        for arm_key in target_arm_keys
        for scenario_id in scenario_ids
    }
    seen: set[tuple[str, str, str, int]] = set()
    duplicate_keys: list[tuple[str, str, str, int]] = []
    unexpected_keys: list[tuple[str, str, str, int]] = []
    invalid_rows: list[dict[str, Any]] = []
    eligible_keys: set[tuple[str, str, str, int]] = set()
    for raw_row in rows:
        row = dict(raw_row)
        key = (
            str(row.get("arm_key", "")),
            str(row.get("candidate_id", "")),
            str(row.get("scenario_id", "")),
            int(row["seed"]) if _is_int(row.get("seed")) else -1,
        )
        if key in seen:
            duplicate_keys.append(key)
            continue
        seen.add(key)
        if key not in expected_keys:
            unexpected_keys.append(key)
            continue
        reasons = _canary_exclusion_reasons(row, contract)
        if reasons:
            invalid_rows.append({"key": key, "reasons": reasons})
        else:
            eligible_keys.add(key)

    missing_keys = sorted(expected_keys - seen)
    expected_count_mismatch = len(expected_keys) != int(required_eligible)
    status = (
        "ok"
        if not expected_count_mismatch
        and len(rows) == len(expected_keys)
        and not duplicate_keys
        and not unexpected_keys
        and not missing_keys
        and len(eligible_keys) == int(required_eligible)
        else "failed"
    )
    return {
        "status": status,
        "eligible_episodes": len(eligible_keys),
        "total_episodes": len(rows),
        "required_eligible": int(required_eligible),
        "expected_episodes": len(expected_keys),
        "missing_keys": [list(key) for key in missing_keys],
        "duplicate_keys": [list(key) for key in sorted(set(duplicate_keys))],
        "unexpected_keys": [list(key) for key in sorted(set(unexpected_keys))],
        "invalid_rows": invalid_rows,
        "target_eligible_ratio": f"{len(eligible_keys)}/{len(rows)}",
        "required_solver_execution_mode": str(contract["solver_execution_mode"]),
        "required_planner_adapter": str(contract["solver_planner_adapter"]),
    }


def _canary_exclusion_reasons(row: Mapping[str, Any], contract: Mapping[str, Any]) -> list[str]:
    """Return every strict canary predicate that a row fails."""
    reasons = _canary_availability_reasons(row, contract) + _canary_solver_reasons(row, contract)
    if row.get("native_solver_exclusion_reasons"):
        reasons.extend(str(reason) for reason in row["native_solver_exclusion_reasons"])
    return sorted(set(reasons))


def _canary_availability_reasons(row: Mapping[str, Any], contract: Mapping[str, Any]) -> list[str]:
    """Return outer benchmark-execution and availability canary failures."""
    reasons: list[str] = []
    if not _eligible(row, contract):
        reasons.append("availability_or_runtime_ineligible")
    expected_mode = str(contract["benchmark_execution_mode"]).strip().lower()
    expected_readiness = str(contract["benchmark_readiness_status"]).strip().lower()
    if str(row.get("execution_mode", "")).strip().lower() != expected_mode:
        reasons.append("execution_mode_mismatch")
    if str(row.get("readiness_status", "")).strip().lower() != expected_readiness:
        reasons.append("readiness_status_mismatch")
    return reasons


def _canary_solver_reasons(row: Mapping[str, Any], contract: Mapping[str, Any]) -> list[str]:
    """Return native solver, provenance, finite-command, and update failures."""
    return _native_solver_identity_reasons(row, contract) + _native_solver_runtime_reasons(row)


def _native_solver_identity_reasons(
    row: Mapping[str, Any], contract: Mapping[str, Any]
) -> list[str]:
    """Return native solver mode and provenance failures."""
    reasons: list[str] = []
    if row.get("native_solver_eligible") is not True:
        reasons.append("native_solver_evidence_ineligible")
    if row.get("solver_execution_mode") != str(contract["solver_execution_mode"]).strip().lower():
        reasons.append("native_solver_execution_missing")
    if row.get("valid_solver_provenance") is not True:
        reasons.append("solver_provenance_invalid")
    if row.get("finite_commands") is not True:
        reasons.append("commands_not_finite")
    return reasons


def _native_solver_runtime_reasons(row: Mapping[str, Any]) -> list[str]:
    """Return solver counter and control-update failures."""
    reasons: list[str] = []
    if not _is_int(row.get("solver_successes")) or int(row["solver_successes"]) < 1:
        reasons.append("solver_update_missing")
    if not _is_int(row.get("control_updates")) or int(row["control_updates"]) < 1:
        reasons.append("control_update_missing")
    if not _is_int(row.get("solver_failures")):
        reasons.append("solver_failures_missing")
    elif int(row["solver_failures"]) < 0:
        reasons.append("solver_failures_invalid")
    elif int(row["solver_failures"]) > 0:
        reasons.append("solver_failure")
    if not _is_int(row.get("fallback_stop_count")):
        reasons.append("fallback_stop_count_missing")
    elif int(row["fallback_stop_count"]) < 0:
        reasons.append("fallback_stop_count_invalid")
    elif int(row["fallback_stop_count"]) > 0:
        reasons.append("fallback")
    return reasons


def _eligibility_reasons(
    row: Mapping[str, Any], contract: Mapping[str, Any] | None = None
) -> list[str]:
    """Return the concrete fail-closed reasons for one analysis row.

    Returns:
        Sorted, de-duplicated reasons matching the predicates used by ``_eligible``.
    """
    solver_contract = dict(contract) if contract is not None else dict(DEFAULT_SOLVER_EXECUTION)
    reasons: list[str] = []
    if str(row.get("execution_mode", "")).strip().lower() not in VALID_EXECUTION_MODES:
        reasons.append("execution_mode_not_supported")
    if str(row.get("readiness_status", "")).strip().lower() not in VALID_READINESS_STATUSES:
        reasons.append("readiness_status_not_supported")
    if str(row.get("availability_status", "")).strip().lower() != "available":
        reasons.append("availability_not_available")
    if row.get("benchmark_success") is not True:
        reasons.append("benchmark_success_false")
    runtime_status = str(row.get("planner_runtime_status", "missing"))
    if runtime_status != "eligible":
        reasons.append(runtime_status)

    if str(row.get("arm_key", "")) in TARGET_ARM_KEYS:
        reasons.extend(_native_solver_identity_reasons(row, solver_contract))
        reasons.extend(_native_solver_runtime_reasons(row))
        reasons.extend(str(reason) for reason in row.get("native_solver_exclusion_reasons", ()))
    return sorted(set(reasons))


def _eligible(row: Mapping[str, Any], contract: Mapping[str, Any] | None = None) -> bool:
    """Report whether a row meets every fail-closed eligibility requirement.

    The shared availability/runtime contract applies to every arm. The strict
    native-solver, provenance, finite-command, and control-update evidence is
    required only for the prediction-aware MPC target arms: incumbents are frozen
    hybrid-rule arms that legitimately execute via their declared adapter, so they
    cannot carry ``PredictionMPCPlannerAdapter`` solver evidence and are gated by
    the outer availability/runtime predicates alone.

    Returns:
        True when the row satisfies every fail-closed eligibility requirement.
    """
    return not _eligibility_reasons(row, contract)


def _planner_runtime_status(value: Any) -> str:
    """Classify planner runtime diagnostics for fail-closed episode eligibility.

    Returns:
        ``eligible`` when no known fallback/failure counter is active; otherwise
        a status that keeps the row out of the preregistered read.
    """
    if not isinstance(value, Mapping) or not value:
        return "missing"
    for field in ("solver_failures", "fallback_stop_count", "fallback_count"):
        if field not in value:
            continue
        count = value[field]
        if not _is_int(count) or count < 0:
            return "invalid"
        if count > 0:
            return "solver_failure" if field == "solver_failures" else "fallback"
    if value.get("fallback_triggered") is True:
        return "fallback"
    checkpoint = value.get("checkpoint_provenance")
    if isinstance(checkpoint, Mapping) and checkpoint.get("fallback_triggered") is True:
        return "fallback"
    return "eligible"


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    """Return a value as a mapping, raising a labeled error otherwise."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping")
    return value


def _repo_path(value: str, repo_root: Path) -> Path:
    """Resolve a possibly relative path against the repository root.

    Returns:
        The resolved absolute path relative to the repository root.
    """
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    """Load a YAML file and require it to contain a mapping.

    Returns:
        The parsed YAML content as a dictionary.
    """
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"YAML config must be a mapping: {path}")
    return payload


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_sha256(config_path: str, *, repo_root: Path) -> str | None:
    """Return a config file's SHA-256 digest, or None when the file is absent."""
    path = Path(config_path)
    if not path.is_absolute():
        path = repo_root / path
    return _sha256(path) if path.is_file() else None


def _is_int(value: Any) -> bool:
    """Report whether a value is an integer, excluding booleans.

    Returns:
        True when the value is an integer but not a boolean.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def _optional_counter(value: Any) -> int | None:
    """Return a non-negative integer counter, or None when evidence is absent/invalid."""
    if not _is_int(value) or int(value) < 0:
        return None
    return int(value)


def _finite_number(value: Any) -> bool:
    """Report whether a value is a finite integer or float, excluding booleans.

    Returns:
        True when the value is a finite integer or float but not a boolean.
    """
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _int_field(value: Any, *, field: str) -> int:
    """Return a value as an integer, raising a labeled error otherwise."""
    if not _is_int(value):
        raise ValueError(f"{field} must be an integer")
    return int(value)


def _bool_field(value: Any, *, field: str) -> bool:
    """Return a value as a boolean, raising a labeled error otherwise."""
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be a boolean")
    return value


def _format_rate(value: Any) -> str:
    """Format a success rate for display, mapping None to NA.

    Returns:
        The formatted rate string, or ``NA`` when the value is None.
    """
    return "NA" if value is None else f"{float(value):.6f}"


__all__ = [
    "CONFIG_SCHEMA",
    "INCUMBENT_ARM_KEYS",
    "NATIVE_SOLVER_PLANNER",
    "REPORT_SCHEMA",
    "SELECTION_SCHEMA",
    "TARGET_ARM_KEYS",
    "TOP_PARAMETERS",
    "TUNING_SCENARIO_IDS",
    "analyze_results",
    "build_candidate_plan",
    "compute_scenario_list_hash",
    "config_hash",
    "format_report_markdown",
    "load_sensitivity_config",
    "load_tuning_selection",
    "normalize_episode_record",
    "selected_scenarios",
    "validate_canary_rows",
    "validate_sensitivity_config",
    "write_report",
    "write_tuning_selection",
]
