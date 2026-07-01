#!/usr/bin/env python3
"""Validate the issue #3810 long-horizon SNQI no-submit launch packet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from issue_3810_readiness_refresh import validate_readiness_refresh  # noqa: E402

DEFAULT_PACKET = Path("configs/benchmarks/issue_3810_long_horizon_snqi_launch_packet.yaml")
EXPECTED_TARGET_HOST = "imech156-u"
REQUIRED_OUTPUTS = {
    "preflight/private_ops_route_dry_run.json",
    "reports/snqi_recalibration_inputs.json",
    "reports/snqi_recalibration_bundle.json",
    "reports/horizon_sensitivity_report.json",
    "reports/interaction_exposure_diagnostics.json",
}
REQUIRED_VALIDATION_COMMAND_MARKERS = {
    "run_research_campaign_manifest.py",
    "check_issue_3810_long_horizon_launch_packet.py",
    "test_check_issue_3810_long_horizon_launch_packet.py -q",
}
REQUIRED_ROUTE_FIELDS = {
    "private_ops_repo",
    "queue_summary_command",
    "route_command",
    "submit_wrapper",
}
REQUIRED_REPORT_FIELDS = {
    "timeout_rate",
    "max_steps_share",
    "success_ranking",
    "snqi_ranking",
    "rank_inversion_persistence",
    "comparison_scope_caveat",
}
REQUIRED_EXPOSURE_FIELDS = {
    "interaction_exposure_share",
    "robot_motion_share_before_first_clearance",
    "first_clearance_step",
    "low_exposure_success",
}
REQUIRED_INTERPRETATION_GATE_EVIDENCE = {
    "private_ops_route_dry_run",
    "retained_compact_review_bundle",
    "external_artifact_pointer",
    "snqi_recalibration_bundle",
    "horizon_sensitivity_report",
    "interaction_exposure_diagnostics",
}
REQUIRED_INTERPRETATION_GATE_BLOCKERS = {
    "slurm_job_13251_running",
    "retention_paths_missing",
    "route_config_provenance_incomplete",
    "snqi_recalibration_inputs_missing",
    "interaction_exposure_diagnostics_missing",
}


class PacketError(ValueError):
    """Raised when the launch packet would not fail closed."""


def _load_packet(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PacketError("packet must be a YAML mapping")
    return payload


def _path_is_repo_relative(value: str) -> bool:
    # A blank value normalizes to "." via PurePosixPath, which would wrongly pass
    # as repo-relative; treat empty/whitespace as not a valid repo path.
    if not value or not value.strip():
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and ".." not in path.parts


def _require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise PacketError(f"{key} must be a mapping")
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PacketError(message)


def validate_packet(packet: dict[str, Any]) -> dict[str, Any]:  # noqa: PLR0915
    """Return a compact validation summary for a fail-closed packet."""
    campaign = _require_mapping(packet, "campaign")
    scenario_suite = _require_mapping(packet, "scenario_suite")
    seed_policy = _require_mapping(packet, "seed_policy")
    planners = _require_mapping(packet, "planners")
    metrics = _require_mapping(packet, "metrics")
    row_status_policy = _require_mapping(packet, "row_status_policy")
    outputs = _require_mapping(packet, "outputs")
    durable_evidence = _require_mapping(packet, "durable_evidence")
    validation = _require_mapping(packet, "validation")
    launch_packet = _require_mapping(packet, "launch_packet")

    _require(packet.get("schema_version") == "research-campaign-manifest.v0.1", "unexpected schema")
    _require(campaign.get("parent_issue") == 3810, "campaign.parent_issue must be 3810")
    _require(bool(str(campaign.get("id") or "").strip()), "campaign.id required")
    _require(
        campaign.get("evidence_tier") == "launch-packet-only", "evidence tier must be launch-only"
    )
    claim_boundary = str(campaign.get("claim_boundary", ""))
    _require("does not run a benchmark" in claim_boundary, "claim boundary must forbid run claims")
    _require(
        "horizon alone" in claim_boundary or "horizon all change" in claim_boundary,
        "claim boundary must reject horizon-only claims",
    )

    _require(scenario_suite.get("max_episode_steps") == 600, "max_episode_steps must be 600")
    _require(scenario_suite.get("scope") == "all_scenarios_in_matrix", "scenario scope must be all")
    matrix_path = str(scenario_suite.get("matrix_path", ""))
    _require(_path_is_repo_relative(matrix_path), "scenario matrix must be repo-relative")
    _require(
        scenario_suite.get("suite_hash_policy") == "record_sha256_before_run",
        "suite hash policy must record sha256 before run",
    )

    _require(seed_policy.get("seed_set") == "paper_eval_s30", "seed_set must be paper_eval_s30")
    seeds = seed_policy.get("seeds")
    _require(
        isinstance(seeds, list) and len(seeds) == 30, "packet must include exactly 30 S30 seeds"
    )
    _require(len(set(seeds)) == 30, "seed list must not contain duplicates")

    planner_rows = planners.get("rows")
    _require(isinstance(planner_rows, list) and len(planner_rows) >= 10, "planner roster too small")
    expected_availability = {
        str(row.get("expected_availability")) for row in planner_rows if isinstance(row, dict)
    }
    _require(
        "diagnostic_only_candidate" in expected_availability, "candidate rows must be explicit"
    )
    _require("dependency_gated" in expected_availability, "dependency-gated rows must be explicit")

    metric_ids = set(metrics.get("ids") or [])
    _require("snqi" in metric_ids, "SNQI metric required")
    _require("interaction_exposure_share" in metric_ids, "interaction exposure metric required")
    _require("timeout_rate" in metric_ids, "timeout metric required")
    _require("max_steps_share" in metric_ids, "max-steps share metric required")

    success_values = set(row_status_policy.get("success_values") or [])
    fail_closed_values = set(row_status_policy.get("fail_closed_values") or [])
    _require(success_values == {"successful_evidence"}, "only successful_evidence may count")
    _require(
        {"not_available", "failed", "blocked"} <= fail_closed_values, "fail-closed statuses missing"
    )
    fallback_policy = str(row_status_policy.get("fallback_policy", ""))
    _require("cannot count" in fallback_policy, "fallback policy must exclude weak rows")

    local_root = str(outputs.get("local_root", ""))
    _require(local_root.startswith("output/"), "outputs.local_root must be under output/")
    _require(outputs.get("disposable") is True, "outputs must be disposable local output")
    required_paths = set(outputs.get("required_paths") or [])
    _require(REQUIRED_OUTPUTS <= required_paths, "required outputs missing issue #3810 artifacts")
    validation_commands = validation.get("commands")
    _require(isinstance(validation_commands, list), "validation.commands must be a list")
    validation_command_blob = "\n".join(str(command) for command in validation_commands)
    for marker in REQUIRED_VALIDATION_COMMAND_MARKERS:
        _require(
            marker in validation_command_blob,
            f"validation.commands missing required marker: {marker}",
        )
    _require(validation.get("dry_run") == "required", "validation.dry_run must be required")

    durable_plan = _require_mapping(durable_evidence, "plan")
    durable_path = str(durable_plan.get("path", ""))
    _require(
        durable_plan.get("required_before_claim") is True, "durable evidence required before claim"
    )
    _require(
        durable_path.startswith("docs/context/evidence/"), "durable path must be context evidence"
    )
    retention_paths = _require_mapping(durable_evidence, "retention_paths")
    compact_review_bundle = str(retention_paths.get("compact_review_bundle", ""))
    external_artifact_pointer = str(retention_paths.get("external_artifact_pointer", ""))
    private_ops_ledger = str(retention_paths.get("private_ops_ledger", ""))
    local_output_boundary = str(durable_evidence.get("local_output_boundary", ""))
    _require(
        compact_review_bundle == durable_path,
        "compact review bundle must match durable evidence path",
    )
    _require(
        external_artifact_pointer == "wandb-or-release-artifact-required-before-claim",
        "external artifact pointer policy missing",
    )
    _require(
        private_ops_ledger.endswith("robot_sf_ll7-private-ops/ops/jobs/jobs.yaml"),
        "private-ops ledger path missing",
    )
    _require(
        "raw episode JSONL" in local_output_boundary, "local output boundary must mention raw JSONL"
    )
    _require(
        "out of git" in local_output_boundary or "out git" in local_output_boundary,
        "local output boundary must keep raw outputs out of git",
    )

    _require(
        launch_packet.get("decision") == "blocked_pending_job_13251_retention_and_analysis",
        "decision must stay blocked pending job 13251 retention and analysis",
    )
    _require(
        launch_packet.get("compute_submit_authorized") is False, "compute submit must be false"
    )
    _require(
        launch_packet.get("slurm_job_id") == "not_submitted", "slurm job must be not_submitted"
    )
    _require(
        launch_packet.get("target_host") == EXPECTED_TARGET_HOST,
        f"target host must be {EXPECTED_TARGET_HOST}",
    )
    blocking_jobs = launch_packet.get("blocking_jobs")
    _require(isinstance(blocking_jobs, list), "blocking_jobs must be a list")
    _require(13251 in blocking_jobs, "job 13251 must block interpretation until retained")
    active_submission = _require_mapping(launch_packet, "active_slurm_submission")
    _require(
        active_submission.get("status") == "submitted_not_benchmark_evidence",
        "active submission must be submitted_not_benchmark_evidence",
    )
    _require(active_submission.get("slurm_job_id") == 13251, "active Slurm job must be 13251")
    _require(
        active_submission.get("cluster_route") == "imech192:l40s",
        "active Slurm route must be imech192:l40s",
    )
    _require(
        active_submission.get("public_commit") == "c49a78e6622eb305a3438d9a6f11a43edb385ff4",
        "active submission public commit mismatch",
    )
    active_submission_boundary_value = active_submission.get("boundary")
    active_submission_boundary = (
        str(active_submission_boundary_value)
        if active_submission_boundary_value is not None
        else ""
    )
    _require(
        active_submission_boundary.startswith("Submission evidence only"),
        "active submission boundary must forbid benchmark evidence",
    )

    readiness_refresh = validate_readiness_refresh(
        launch_packet,
        EXPECTED_TARGET_HOST,
        require=_require,
        require_mapping=_require_mapping,
        require_commands=True,
        require_head_ref=True,
    )
    readiness_decision = str(readiness_refresh.get("decision", ""))
    _require(
        "not a local submit action" in readiness_decision, "readiness refresh must stay no-submit"
    )

    live_issue_state = _require_mapping(launch_packet, "live_issue_state")
    _require(
        live_issue_state.get("checked_date") == "2026-07-01",
        "live issue state date must be 2026-07-01",
    )
    _require(
        live_issue_state.get("required_label") == "state:running",
        "live issue state must record state:running blocker",
    )
    _require(
        live_issue_state.get("submit_blocker") is True,
        "live issue state must block submit while running",
    )
    _require(
        "job 13251" in str(live_issue_state.get("resolution_required_before_submit", "")),
        "live issue state must record submitted job 13251",
    )

    ledger = _require_mapping(launch_packet, "ledger_reconciliation")
    _require(
        ledger.get("job_13175_state") == "superseded_by_submitted_job_13251",
        "job 13175 reconciliation must be superseded by submitted job 13251",
    )
    _require(
        ledger.get("issue_3810_duplicate_status") == "no_duplicate_reported_in_submission_comment",
        "issue #3810 duplicate status must reflect submission comment",
    )
    running_related_jobs = ledger.get("running_related_jobs")
    _require(
        isinstance(running_related_jobs, list),
        "running_related_jobs must be listed",
    )

    go_no_go = _require_mapping(launch_packet, "go_no_go")
    _require(
        go_no_go.get("recommendation") == "blocked_pending_job_13251_retention_and_analysis",
        "go/no-go recommendation must remain blocked pending job 13251 retention",
    )
    _require(
        go_no_go.get("local_submission_status") == "no_submit_current_machine",
        "local submission status must remain no-submit",
    )
    exact_cmd = go_no_go.get("exact_local_decision_command")
    _require(
        "check_issue_3810_long_horizon_launch_packet.py"
        in (str(exact_cmd) if exact_cmd is not None else ""),
        "exact local decision command missing",
    )
    slurm_status = go_no_go.get("slurm_command_status")
    _require(
        "does not authorize" in (str(slurm_status) if slurm_status is not None else ""),
        "slurm command status must forbid additional submission",
    )
    _require(
        "job 13251" in (str(slurm_status) if slurm_status is not None else ""),
        "go/no-go must mention job 13251",
    )
    dry_run = _require_mapping(go_no_go, "private_ops_dry_run")
    _require(
        dry_run.get("required_before_submit") is False,
        "private-ops dry run must be superseded by live submission",
    )
    _require(
        dry_run.get("target_host") == EXPECTED_TARGET_HOST,
        "private-ops dry run host mismatch",
    )
    _require(
        dry_run.get("current_public_status") == "superseded_by_live_submission",
        "private-ops dry run must be superseded_by_live_submission",
    )
    dry_run_fields = set(dry_run.get("required_fields") or [])
    _require(
        {
            "target_host",
            "queue_summary_timestamp",
            "duplicate_status",
            "live_issue_state",
            "job_13175_state",
            "route_id",
            "submit_wrapper_supports_target_host",
            "owning_worktree",
            "owning_worktree_clean",
            "decision",
        }
        <= dry_run_fields,
        "private-ops dry run fields missing",
    )
    _require(
        "No additional submission" in str(dry_run.get("decision_policy", "")),
        "private-ops dry run must forbid additional submission",
    )

    interpretation_gate = _require_mapping(launch_packet, "interpretation_gate")
    _require(
        interpretation_gate.get("status") == "blocked_pending_active_run_retention_reconciliation",
        "interpretation gate must stay blocked pending active-run reconciliation",
    )
    _require(
        interpretation_gate.get("required_before_claim") is True,
        "interpretation gate must be required before claims",
    )
    _require(
        interpretation_gate.get("required_before_report_publication") is True,
        "interpretation gate must be required before report publication",
    )
    _require(
        interpretation_gate.get("active_run_state") == "state:running",
        "interpretation gate must record active run state",
    )
    gate_blockers = set(interpretation_gate.get("blockers") or [])
    _require(
        REQUIRED_INTERPRETATION_GATE_BLOCKERS <= gate_blockers,
        "interpretation gate blockers missing",
    )
    gate_evidence = set(interpretation_gate.get("required_evidence") or [])
    _require(
        REQUIRED_INTERPRETATION_GATE_EVIDENCE <= gate_evidence,
        "interpretation gate evidence missing",
    )
    gate_policy = str(interpretation_gate.get("decision_policy", ""))
    _require(
        "13251" in gate_policy
        and "not permission" in gate_policy
        and "promote benchmark claims" in gate_policy,
        "interpretation gate must block claim promotion",
    )

    route = _require_mapping(launch_packet, "route")
    route_fields = set(route.keys())
    _require(REQUIRED_ROUTE_FIELDS <= route_fields, "route missing required fields")
    _require(
        str(route.get("private_ops_repo", "")) == route.get("private_ops_repo"),
        "route.private_ops_repo must be present",
    )
    _require(
        str(route.get("submit_policy")) == "no_submit_until_decision_packet_go",
        "submit policy missing",
    )
    _require(
        str(route.get("submit_wrapper", "")).endswith("submit_and_record.sh"),
        "submit wrapper missing",
    )
    for field in ("private_ops_repo", "queue_summary_command", "route_command", "submit_wrapper"):
        route_path = str(route.get(field, ""))
        _require(route_path, f"route.{field} is required")
        _require(PurePosixPath(route_path).is_absolute(), f"route.{field} must be absolute")
        if field == "private_ops_repo":
            _require(
                route_path.endswith("robot_sf_ll7-private-ops"),
                "route.private_ops_repo must point to private-ops checkout",
            )
            continue
        _require(
            "private-ops" in route_path,
            f"route.{field} must use private-ops script path",
        )

    preflight = _require_mapping(launch_packet, "preflight")
    _require("private_ops_dry_run" in preflight, "private-ops dry-run command missing")
    _require(
        "must not be invoked" in str(preflight.get("no_submit_guard", "")),
        "no-submit guard missing",
    )
    _require(
        "public_manifest_check" in preflight,
        "preflight missing public manifest check",
    )
    _require(
        "public_packet_check" in preflight,
        "preflight missing public packet check",
    )
    _require(
        "run_research_campaign_manifest.py" in str(preflight.get("public_manifest_check", "")),
        "preflight public manifest check command missing",
    )
    _require(
        "check_issue_3810_long_horizon_launch_packet.py"
        in str(preflight.get("public_packet_check", "")),
        "preflight public packet check command missing",
    )

    config_patch = _require_mapping(launch_packet, "config_patch")
    overrides = _require_mapping(config_patch, "required_overrides")
    _require(overrides.get("horizon") == 600, "config patch must set horizon 600")
    _require(overrides.get("stop_on_failure") is False, "long run must keep collecting rows")

    snqi = _require_mapping(launch_packet, "snqi_recalibration")
    snqi_inputs = _require_mapping(snqi, "inputs")
    _require("campaign_table" in snqi_inputs, "SNQI campaign_table input missing")
    _require("episode_records" in snqi_inputs, "SNQI episode_records input missing")
    snqi_scripts = snqi.get("source_scripts") or snqi_inputs.get("source_scripts")
    _require(isinstance(snqi_scripts, list), "SNQI source_scripts must be a list")
    for required_script in (
        "scripts/recompute_snqi_weights.py",
        "scripts/snqi_sensitivity_analysis.py",
        "scripts/validate_snqi_scripts.py",
    ):
        _require(required_script in snqi_scripts, f"SNQI source script missing: {required_script}")
    _require(
        str(snqi.get("checksum_policy")) == "checksum_pin_before_claim",
        "SNQI checksum policy missing",
    )
    _require(
        str(snqi.get("comparison_rule"))
        == "within_regime_only_no_cross_regime_absolute_snqi_claim",
        "SNQI comparison rule must forbid cross-regime absolute claims",
    )

    report = _require_mapping(launch_packet, "horizon_sensitivity_report")
    _require(
        "build_horizon_timestep_denominator_report.py" in str(report.get("command", "")),
        "horizon report command missing build_horizon_timestep_denominator_report.py",
    )
    _require(
        "--long-campaign" in str(report.get("command", "")),
        "horizon report command missing --long-campaign",
    )
    _require(
        "--short-campaign-reference" in str(report.get("command", "")),
        "horizon report command missing --short-campaign-reference",
    )
    report_fields = set(report.get("required_fields") or [])
    _require(REQUIRED_REPORT_FIELDS <= report_fields, "horizon report fields missing")
    _require(
        "not_horizon_only" in str(report.get("required_caveat", "")), "comparison caveat missing"
    )

    exposure = _require_mapping(launch_packet, "wait_it_out_guard")
    _require(
        exposure.get("diagnostic_name") == "interaction_exposure",
        "exposure diagnostic name incorrect",
    )
    exposure_fields = set(exposure.get("required_episode_fields") or [])
    _require(REQUIRED_EXPOSURE_FIELDS <= exposure_fields, "interaction exposure fields missing")
    _require(
        exposure.get("low_exposure_success_status") == "diagnostic_only",
        "low exposure must be diagnostic",
    )
    _require(
        "#3813" in str(exposure.get("diagnostic_contract", "")), "sustained-flow follow-up missing"
    )

    return {
        "ok": True,
        "issue": 3810,
        "campaign_id": campaign["id"],
        "max_episode_steps": scenario_suite["max_episode_steps"],
        "seed_count": len(seeds),
        "planner_count": len(planner_rows),
        "compute_submit_authorized": launch_packet["compute_submit_authorized"],
        "slurm_job_id": launch_packet["slurm_job_id"],
        "target_host": launch_packet["target_host"],
        "blocking_jobs": blocking_jobs,
        "active_slurm_job_id": active_submission["slurm_job_id"],
        "job_13175_state": ledger["job_13175_state"],
        "issue_3810_duplicate_status": ledger["issue_3810_duplicate_status"],
        "live_issue_state": live_issue_state["required_label"],
        "go_no_go": go_no_go["recommendation"],
        "private_ops_dry_run": dry_run["current_public_status"],
        "interpretation_gate": interpretation_gate["status"],
    }


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--json", action="store_true", help="Emit JSON summary.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run the command-line validator."""
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    try:
        summary = validate_packet(_load_packet(args.packet))
    except (OSError, PacketError, yaml.YAMLError) as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2, sort_keys=True))
        else:
            print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            "issue #3810 launch packet ok: "
            f"{summary['planner_count']} planners, {summary['seed_count']} seeds, "
            f"horizon {summary['max_episode_steps']}, no submit"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
