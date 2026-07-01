#!/usr/bin/env python3
"""Validate issue #3810 long-horizon analysis + retention contract."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from issue_3810_readiness_refresh import validate_readiness_refresh  # noqa: E402

DEFAULT_PACKET = Path("configs/benchmarks/issue_3810_long_horizon_snqi_launch_packet.yaml")
EXPECTED_TARGET_HOST = "imech039"

REQUIRED_PRE_FLIGHT_MARKERS = {
    "ROBOT_SF_PRIVATE_OPS",
    "run_preflight.sh",
    "run_research_campaign_manifest.py",
    "check_issue_3810_long_horizon_launch_packet.py",
}

REQUIRED_OUTPUTS = {
    "preflight/private_ops_route_dry_run.json",
    "preflight/validate_config.json",
    "preflight/preview_scenarios.json",
    "preflight/planner_availability.json",
    "reports/campaign_summary.json",
    "reports/campaign_table.csv",
    "reports/campaign_table.md",
    "reports/snqi_recalibration_inputs.json",
    "reports/snqi_recalibration_bundle.json",
    "reports/horizon_sensitivity_report.json",
    "reports/horizon_sensitivity_report.md",
    "reports/interaction_exposure_diagnostics.json",
    "reports/interaction_exposure_diagnostics.md",
    "episodes/",
}

REQUIRED_ROUTE_FIELDS = {
    "private_ops_repo",
    "queue_summary_command",
    "route_command",
    "submit_wrapper",
    "target_host",
}

REQUIRED_HORIZON_FIELDS = {
    "timeout_rate",
    "max_steps_share",
    "success_ranking",
    "snqi_ranking",
    "rank_inversion_persistence",
    "comparison_scope_caveat",
}

REQUIRED_WAIT_FIELDS = {
    "interaction_exposure_share",
    "robot_motion_share_before_first_clearance",
    "first_clearance_step",
    "low_exposure_success",
}


class PacketError(ValueError):
    """Raised when the issue #3810 analysis packet violates the contract."""

    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PacketError(message)


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    _require(isinstance(value, Mapping), f"{key} must be a mapping")
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), "packet must be YAML mapping")
    return payload


def validate_packet(packet: Mapping[str, Any], *, issue: int = 3810) -> dict[str, Any]:
    """Validate the issue #3810 analysis, retention, and no-submit guard contract."""

    campaign = _require_mapping(packet, "campaign")
    _require(campaign.get("parent_issue") == issue, "packet parent issue mismatch")

    launch_packet = _require_mapping(packet, "launch_packet")
    validate_readiness_refresh(
        launch_packet,
        EXPECTED_TARGET_HOST,
        require=_require,
        require_mapping=_require_mapping,
    )

    analysis = _require_mapping(packet, "analysis_and_retention_packet")
    _require(analysis.get("enabled") is True, "analysis_and_retention_packet.enabled must be true")

    route = _require_mapping(analysis, "route")
    _require(REQUIRED_ROUTE_FIELDS <= set(route), "route field set incomplete")
    _require(
        route.get("target_host") == EXPECTED_TARGET_HOST,
        f"target host must be {EXPECTED_TARGET_HOST}",
    )

    preflight = _require_mapping(analysis, "preflight")
    _require(preflight.get("required") is True, "preflight.required must be true")
    preflight_blob = "\n".join(str(v) for v in preflight.values())
    for marker in REQUIRED_PRE_FLIGHT_MARKERS:
        _require(marker in preflight_blob, f"preflight missing marker: {marker}")
    queue_command = str(preflight.get("queue_and_duplicate_check_command", ""))
    duplicate_command = str(preflight.get("duplicate_check_command", ""))
    _require(
        "queue_summary.sh" in queue_command and "--limit" in queue_command,
        "queue duplicate check must use current queue_summary interface",
    )
    _require(
        "queue_summary.sh" in duplicate_command and "--limit" in duplicate_command,
        "duplicate check must use current queue_summary interface",
    )
    stale_flags = {"--json", "--target-host", "--issue"}
    stale_command_blob = f"{queue_command}\n{duplicate_command}"
    _require(
        not any(flag in stale_command_blob for flag in stale_flags),
        "preflight queue commands must not use stale private-ops flags",
    )

    expected = _require_mapping(analysis, "expected_outputs")
    _require(
        expected.get("local_root") == "output/benchmarks/issue_3810_comprehensive_h600_snqi",
        "expected_outputs.local_root mismatch",
    )
    required_paths = set(expected.get("required_paths") or [])
    _require(REQUIRED_OUTPUTS <= required_paths, "required outputs missing")

    retention = _require_mapping(analysis, "retention")
    compact_review_bundle = str(retention.get("compact_review_bundle", ""))
    _require(
        compact_review_bundle.startswith("docs/context/evidence/"),
        "compact review path must be durable",
    )
    durable_manifests = set(retention.get("durable_manifests") or [])
    _require(
        compact_review_bundle in durable_manifests, "durable manifests must include compact bundle"
    )
    _require(
        f"{compact_review_bundle.rstrip('/')}/reports" in durable_manifests,
        "durable manifests must include reports directory",
    )
    _require(
        f"{compact_review_bundle.rstrip('/')}/metadata.json" in durable_manifests,
        "durable manifests must include metadata.json",
    )

    execution_contract = _require_mapping(analysis, "execution_contract")
    _require(
        execution_contract.get("route_preflight_required") is True,
        "execution_contract.route_preflight_required must be true",
    )
    _require(
        execution_contract.get("launch_only") is False,
        "execution_contract.launch_only must be false",
    )
    _require(
        execution_contract.get("local_submission_allowed") is False,
        "execution_contract.local_submission_allowed must be false",
    )
    _require(
        execution_contract.get("decision_gate", {}).get("status")
        == "blocked_pending_submit_host_route_and_reconciliation",
        "execution contract must stay blocked until host/route reconciliation",
    )
    decision_reason = str(execution_contract.get("decision_gate", {}).get("reason", ""))
    _require("Issue remains open" in decision_reason, "decision gate must keep issue open")
    _require("job 13175" in decision_reason, "decision gate must require job 13175 reconciliation")

    snqi = _require_mapping(analysis, "snqi_recalibration_inputs")
    for field in (
        "weights_path",
        "baseline_path",
        "campaign_table_path",
        "episode_records_glob",
    ):
        _require(str(snqi.get(field, "")) != "", f"snqi_recalibration_inputs.{field} required")

    required_scripts = set(snqi.get("required_scripts") or [])
    for script in (
        "scripts/recompute_snqi_weights.py",
        "scripts/snqi_sensitivity_analysis.py",
        "scripts/validate_snqi_scripts.py",
    ):
        _require(script in required_scripts, f"missing snqi script: {script}")

    horizon = _require_mapping(analysis, "horizon_sensitivity_report")
    command = str(horizon.get("command", ""))
    _require(
        "build_horizon_timestep_denominator_report.py" in command,
        "horizon command missing report builder",
    )
    _require(
        horizon.get("required_scope_caveat")
        == "prior_scoped_short_horizon_vs_new_comprehensive_long_horizon_not_horizon_only",
        "horizon scope caveat mismatch",
    )
    _require(
        REQUIRED_HORIZON_FIELDS <= set(horizon.get("required_fields") or []),
        "horizon required fields missing",
    )

    wait_guard = _require_mapping(analysis, "wait_it_out_guard")
    required_wait = set(wait_guard.get("required_episode_fields") or [])
    _require(REQUIRED_WAIT_FIELDS <= required_wait, "wait-it-out episode fields missing")
    _require(
        wait_guard.get("low_exposure_success_status") == "diagnostic_only",
        "wait-it-out low exposure must remain diagnostic_only",
    )

    return {
        "ok": True,
        "issue": issue,
        "target_host": route["target_host"],
        "route_contract": execution_contract.get("decision_gate", {}).get("status"),
        "retention_bundle": compact_review_bundle,
    }


def main() -> int:
    """Run the issue #3810 analysis packet validator CLI."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    packet = _load_yaml(args.packet)
    summary = validate_packet(packet)
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    else:
        print("issue_3810 analysis+retention packet checks ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
