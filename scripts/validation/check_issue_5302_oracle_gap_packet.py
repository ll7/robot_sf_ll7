#!/usr/bin/env python3
"""Fail-closed checker for the issue #5302 oracle-gap analysis packet.

The checker validates the pre-registration contract only. It does not run a
benchmark, submit compute, calculate a planner ranking, or promote a claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

DEFAULT_PACKET = Path("configs/analysis/issue_5302_oracle_gap_packet.yaml")
SCHEMA_VERSION = "issue_5302_oracle_gap_analysis_packet.v1"
EXPECTED_PLANNERS = (
    "orca",
    "ppo",
    "prediction_planner",
    "scenario_adaptive_hybrid_orca_v1",
    "prediction_mpc",
)
REQUIRED_CEILINGS = (
    "best_fixed_planner",
    "best_planner_per_scenario_family",
    "best_planner_per_scenario_cell",
    "hindsight_per_episode_oracle",
)
REQUIRED_METRICS = {
    "collision_rate",
    "severe_intrusion_rate",
    "completion_rate",
    "timeout_rate",
    "tail_clearance",
    "worst_family_performance",
    "jerk",
    "pedestrian_disturbance",
    "compute_time_ms_p50",
    "compute_time_ms_p95",
    "compute_time_ms_p99",
}
FORBIDDEN_TRANSIENT_KEYS = {
    "target_host",
    "slurm_job_id",
    "queue",
    "packet_lineage",
    "route_command",
}


class PacketError(ValueError):
    """Raised when the packet is unsafe or incomplete."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PacketError(message)


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    _require(isinstance(value, dict), f"{key} must be a mapping")
    return value


def _repo_relative_path(value: Any, key: str) -> str:
    _require(isinstance(value, str) and value.strip(), f"{key} must be a non-empty path")
    path = PurePosixPath(value)
    _require(not path.is_absolute() and ".." not in path.parts, f"{key} must be repo-relative")
    return value


def _walk_for_forbidden_keys(value: Any, path: str = "packet") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _require(
                key not in FORBIDDEN_TRANSIENT_KEYS, f"{path}.{key} is transient routing state"
            )
            _walk_for_forbidden_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_for_forbidden_keys(child, f"{path}[{index}]")


def load_packet(path: Path) -> dict[str, Any]:
    """Load a YAML packet and require a mapping root."""
    _require(path.is_file(), f"packet not found: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "packet must be a YAML mapping")
    return payload


def validate_packet(  # noqa: PLR0915
    packet: dict[str, Any], *, repo_root: Path | None = None
) -> dict[str, Any]:
    """Validate the issue #5302 contract and return a compact summary."""
    root = repo_root or Path(__file__).resolve().parents[2]
    _walk_for_forbidden_keys(packet)
    _require(packet.get("schema_version") == SCHEMA_VERSION, "schema_version mismatch")
    _require(packet.get("issue") == 5302, "issue must be 5302")

    claim_boundary = str(packet.get("claim_boundary", "")).lower()
    for phrase in ("does not run", "does not submit", "paper/dissertation"):
        _require(phrase in claim_boundary, f"claim_boundary must mention {phrase}")

    execution = _mapping(packet, "execution_boundary")
    for key in (
        "run_benchmark",
        "compute_submit_authorized",
        "paper_claim_edits",
        "selector_training",
    ):
        _require(execution.get(key) is False, f"execution_boundary.{key} must be false")
    _require(
        execution.get("fallback_or_degraded_success_allowed") is False,
        "fallback_or_degraded_success_allowed must be false",
    )

    roster = _mapping(packet, "planner_roster")
    rows = roster.get("required")
    _require(isinstance(rows, list), "planner_roster.required must be a list")
    planner_ids = tuple(str(row.get("planner_id")) for row in rows if isinstance(row, dict))
    _require(
        planner_ids == EXPECTED_PLANNERS,
        "planner roster must match the approved five-planner order",
    )
    _require(all(isinstance(row, dict) for row in rows), "planner roster rows must be mappings")
    for row in rows:
        if row["planner_id"] == "orca":
            _require(row.get("config_path") is None, "orca must use its canonical baseline config")
        else:
            config_path = _repo_relative_path(
                row.get("config_path"), f"{row['planner_id']}.config_path"
            )
            _require((root / config_path).is_file(), f"planner config missing: {config_path}")
    gated = roster.get("gated_or_deferred")
    _require(
        isinstance(gated, list)
        and {row.get("planner_id") for row in gated} == {"risk_dwa", "sipp_lattice"},
        "gated roster must name risk_dwa and sipp_lattice",
    )

    inputs = _mapping(packet, "input_contract")
    required_fields = inputs.get("required_row_fields")
    _require(isinstance(required_fields, list), "input_contract.required_row_fields must be a list")
    required_field_set = {str(field) for field in required_fields}
    _require(
        {
            "episode_id",
            "scenario_family",
            "scenario_cell",
            "split",
            "planner_id",
            "selection_score",
            "row_status",
            "execution_mode",
            "config_hash",
            "repo_commit",
        }
        <= required_field_set,
        "input row identity/provenance fields are incomplete",
    )
    split = _mapping(inputs, "split_contract")
    _require(
        split.get("allowed_values") == ["selection", "evaluation"],
        "split values must be selection/evaluation",
    )
    for key in (
        "selection_and_evaluation_family_sets_must_be_disjoint",
        "scenario_cell_is_frozen_before_execution",
        "same_episode_set_across_planners",
    ):
        _require(split.get(key) is True, f"split_contract.{key} must be true")
    _require(
        split.get("required_planner_count_per_episode") == 5,
        "five planner rows per episode are required",
    )
    provenance = _mapping(inputs, "provenance")
    provenance_paths = provenance.get("required_paths")
    _require(
        isinstance(provenance_paths, list) and provenance_paths, "provenance paths are required"
    )
    for path in provenance_paths:
        rel = _repo_relative_path(path, "input_contract.provenance.required_paths")
        _require((root / rel).exists(), f"provenance path missing: {rel}")

    ceilings = _mapping(packet, "ceilings")
    _require(
        ceilings.get("score_source") == "selection_score_from_frozen_metric_layer",
        "score must come from frozen metric layer",
    )
    constraints = ceilings.get("constraints")
    _require(
        constraints == ["collision_rate", "severe_intrusion_rate"],
        "safety constraints must be collision and severe intrusion",
    )
    estimands = ceilings.get("estimands")
    _require(isinstance(estimands, list), "ceilings.estimands must be a list")
    _require(
        tuple(str(row.get("id")) for row in estimands) == REQUIRED_CEILINGS,
        "four ceiling estimands are incomplete or reordered",
    )
    _require(
        all(isinstance(row, dict) and "attainable_online" in row for row in estimands),
        "ceiling attainability must be explicit",
    )
    gaps = _mapping(ceilings, "gap_definitions")
    for key in ("family_gap", "cell_gap", "oracle_gap"):
        _require(
            key in gaps and "best_fixed_planner" in str(gaps[key]),
            f"gap definition missing fixed-planner baseline: {key}",
        )
    uncertainty = _mapping(ceilings, "uncertainty")
    _require(
        uncertainty.get("method") == "hierarchical_bootstrap", "hierarchical bootstrap is required"
    )
    _require(
        uncertainty.get("resampling_order") == "scenario_family_then_scenario_cell_then_episode",
        "bootstrap must resample family, cell, then episode",
    )
    _require(uncertainty.get("confidence") == 0.95, "bootstrap confidence must be 0.95")

    metrics = _mapping(packet, "metrics")
    metric_values = {str(metric) for metric in metrics.get("required", [])}
    _require(REQUIRED_METRICS <= metric_values, "required report metrics are incomplete")
    _require(
        metrics.get("metric_semantics") == "frozen_existing_metric_layer_only",
        "metric semantics must remain frozen",
    )

    row_policy = _mapping(packet, "row_status_policy")
    _require(
        row_policy.get("successful_evidence_values") == ["successful_evidence"],
        "only successful_evidence may count",
    )
    _require(row_policy.get("eligible_execution_modes") == ["native"], "only native rows may count")
    _require(
        "fallback" in row_policy.get("visible_but_ineligible_values", []),
        "fallback rows must remain visible and ineligible",
    )
    _require(
        "blocks the report" in str(row_policy.get("exclusion_rule")),
        "row exclusion rule must fail closed",
    )

    outputs = _mapping(packet, "outputs")
    local_root = str(outputs.get("local_root", ""))
    _require(local_root.startswith("output/"), "outputs.local_root must be under output/")
    _require(outputs.get("disposable") is True, "local outputs must be disposable")
    required_outputs = outputs.get("required_paths")
    _require(
        isinstance(required_outputs, list) and len(required_outputs) >= 8,
        "output contract is incomplete",
    )
    durable = _mapping(outputs, "durable_evidence")
    durable_path = _repo_relative_path(durable.get("path"), "outputs.durable_evidence.path")
    _require(
        durable_path.startswith("docs/context/evidence/"),
        "durable evidence must use context evidence",
    )
    _require(
        durable.get("required_before_claim") is True, "durable evidence is required before claim"
    )

    validation = _mapping(packet, "validation")
    no_submit = validation.get("no_submit_commands")
    _require(
        isinstance(no_submit, list)
        and all(
            "check_issue_5302_oracle_gap_packet.py" in str(command)
            or "test_check_issue_5302_oracle_gap_packet.py" in str(command)
            for command in no_submit
        ),
        "validation must contain only checker/test commands",
    )
    _require(
        validation.get("campaign_execution_allowed_in_this_pr") is False,
        "campaign execution must be disabled in this PR",
    )

    readiness = _mapping(packet, "readiness_decision")
    _require(
        readiness.get("status") == "blocked_pending_native_campaign_bundle",
        "readiness status must remain blocked pending native data",
    )
    for key in (
        "benchmark_campaign_run",
        "compute_submit_authorized",
        "ranking_claim_promotion",
        "selector_training",
        "paper_claim_edits",
        "fallback_degraded_success_allowed",
    ):
        _require(readiness.get(key) is False, f"readiness_decision.{key} must be false")

    return {
        "status": "ok",
        "issue": 5302,
        "schema_version": SCHEMA_VERSION,
        "planner_count": len(planner_ids),
        "ceiling_count": len(estimands),
        "required_metric_count": len(metric_values),
        "durable_evidence_path": durable_path,
        "campaign_execution_allowed": False,
    }


def main(argv: list[str] | None = None) -> int:
    """Validate the shipped packet and emit JSON or a short status line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    try:
        result = validate_packet(load_packet(args.packet))
    except (OSError, PacketError, yaml.YAMLError) as exc:
        if args.as_json:
            print(json.dumps({"status": "not_ready", "error": str(exc)}))
        else:
            print(f"not_ready: {exc}")
        return 1
    if args.as_json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(
            f"ready: issue #5302 packet ({result['planner_count']} planners, {result['ceiling_count']} ceilings)"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
