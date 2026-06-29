#!/usr/bin/env python3
"""Validate the issue #3810 long-horizon SNQI no-submit launch packet."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

DEFAULT_PACKET = Path("configs/benchmarks/issue_3810_long_horizon_snqi_launch_packet.yaml")
REQUIRED_OUTPUTS = {
    "preflight/private_ops_route_dry_run.json",
    "reports/snqi_recalibration_inputs.json",
    "reports/snqi_recalibration_bundle.json",
    "reports/horizon_sensitivity_report.json",
    "reports/interaction_exposure_diagnostics.json",
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
    _require(isinstance(seeds, list) and len(seeds) >= 30, "packet must include all S30 seeds")

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

    durable_plan = _require_mapping(durable_evidence, "plan")
    durable_path = str(durable_plan.get("path", ""))
    _require(
        durable_plan.get("required_before_claim") is True, "durable evidence required before claim"
    )
    _require(
        durable_path.startswith("docs/context/evidence/"), "durable path must be context evidence"
    )

    _require(
        launch_packet.get("decision") == "blocked_until_private_slurm_go",
        "decision must block submit",
    )
    _require(
        launch_packet.get("compute_submit_authorized") is False, "compute submit must be false"
    )
    _require(
        launch_packet.get("slurm_job_id") == "not_submitted", "slurm job must be not_submitted"
    )
    _require(13175 in (launch_packet.get("blocking_jobs") or []), "job 13175 blocker missing")

    route = _require_mapping(launch_packet, "route")
    _require(
        str(route.get("submit_policy")) == "no_submit_until_decision_packet_go",
        "submit policy missing",
    )
    _require(
        str(route.get("submit_wrapper", "")).endswith("submit_and_record.sh"),
        "submit wrapper missing",
    )

    preflight = _require_mapping(launch_packet, "preflight")
    _require("private_ops_dry_run" in preflight, "private-ops dry-run command missing")
    _require(
        "must not be invoked" in str(preflight.get("no_submit_guard", "")),
        "no-submit guard missing",
    )

    config_patch = _require_mapping(launch_packet, "config_patch")
    overrides = _require_mapping(config_patch, "required_overrides")
    _require(overrides.get("horizon") == 600, "config patch must set horizon 600")
    _require(overrides.get("stop_on_failure") is False, "long run must keep collecting rows")

    snqi = _require_mapping(launch_packet, "snqi_recalibration")
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
    report_fields = set(report.get("required_fields") or [])
    _require(REQUIRED_REPORT_FIELDS <= report_fields, "horizon report fields missing")
    _require(
        "not_horizon_only" in str(report.get("required_caveat", "")), "comparison caveat missing"
    )

    exposure = _require_mapping(launch_packet, "wait_it_out_guard")
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
        "blocking_jobs": launch_packet["blocking_jobs"],
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
    except (OSError, PacketError, yaml.YAMLError, KeyError, TypeError, AttributeError) as exc:
        # KeyError/TypeError/AttributeError backstop: a malformed packet must
        # always fail closed with a clear error, never an unhandled traceback.
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
