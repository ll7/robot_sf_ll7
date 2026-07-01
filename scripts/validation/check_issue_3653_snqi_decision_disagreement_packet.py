#!/usr/bin/env python3
"""Validate the issue #3653 SNQI decision-disagreement application packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PACKET = Path("configs/benchmarks/issue_3653_snqi_decision_disagreement_packet.yaml")
SCHEMA_VERSION = "issue-3653-snqi-decision-disagreement-application-packet.v1"
EXPECTED_STATUS = "blocked_missing_valid_campaign_episodes"
EXPECTED_INPUT_STATUS = "blocked_until_episode_jsonl_promoted_or_hydratable"
EXPECTED_WEIGHT_HASH = "71a67c3c02faff166f8c96bef8bcf898533981ca2b2c4493829988520fb1aeb2"
EXPECTED_BASELINE_HASH = "329ca5766491e1587979d0a435c7ba676e148ccdff97040a36546bbb9414035a"
REQUIRED_ARTIFACTS = {
    "snqi_scalarization_sensitivity.json",
    "snqi_scalarization_sensitivity_planner_rows.csv",
    "snqi_scalarization_sensitivity_decision_disagreement.csv",
    "snqi_scalarization_sensitivity.md",
    "snqi_scalarization_sensitivity_pareto.svg",
}
REQUIRED_REPORT_FIELDS = {
    "decision_disagreement",
    "planner_rows",
    "term_dominance",
    "weight_zero_ablation",
    "weight_sweep",
    "pareto_front",
}


class PacketError(ValueError):
    """Raised when the packet fails closed before empirical claim use."""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PacketError(message)


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    _require(isinstance(value, Mapping), f"{key} must be a mapping")
    return value


def _require_sequence(payload: Mapping[str, Any], key: str) -> Sequence[Any]:
    value = payload.get(key)
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)), f"{key} must be a list"
    )
    return value


def _repo_relative_path(value: Any, key: str) -> Path:
    _require(isinstance(value, str), f"{key} must be a string path")
    _require(value.strip() != "", f"{key} must be non-empty")
    path = Path(value)
    _require(not path.is_absolute(), f"{key} must be repo-relative")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_packet(path: Path) -> dict[str, Any]:
    """Load a YAML packet and require a top-level mapping."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise PacketError("packet must be a YAML mapping")
    return payload


def validate_packet(packet: Mapping[str, Any], *, repo_root: Path | None = None) -> dict[str, Any]:
    """Validate the no-submit #3653 empirical-application packet."""

    repo_root = repo_root or _repo_root()
    _require(packet.get("schema_version") == SCHEMA_VERSION, "schema_version mismatch")
    _require(packet.get("issue") == 3653, "issue must be 3653")

    claim_boundary = str(packet.get("claim_boundary", "")).lower()
    for phrase in (
        "full benchmark campaign",
        "slurm/gpu",
        "paper/dissertation",
        "primary safety ranking",
    ):
        _require(phrase in claim_boundary, f"claim_boundary must mention {phrase}")

    execution = _require_mapping(packet, "execution_boundary")
    _require(
        execution.get("run_benchmark") is False, "execution_boundary.run_benchmark must be false"
    )
    _require(
        execution.get("compute_submit_authorized") is False,
        "execution_boundary.compute_submit_authorized must be false",
    )
    _require(
        execution.get("local_submission_allowed") is False,
        "execution_boundary.local_submission_allowed must be false",
    )
    _require(execution.get("slurm_job_id") == "not_submitted", "slurm_job_id must be not_submitted")
    _require(execution.get("target_host") == "imech039", "target_host must be imech039")
    _require(execution.get("current_status") == EXPECTED_STATUS, "current_status must stay blocked")

    campaign = _require_mapping(packet, "source_campaign")
    _require(campaign.get("parent_issue") == 1554, "source campaign must reference issue #1554")
    _require(campaign.get("source_job_id") == 13175, "source campaign must reference job 13175")
    _require(campaign.get("scenario_horizon") == "h500", "scenario horizon must be h500")
    _require(campaign.get("seed_budget") == "S20", "seed budget must be S20")
    _require(int(campaign.get("planner_count", 0)) == 9, "planner_count must be 9")
    _require(
        int(campaign.get("expected_episode_count", 0)) == 8640,
        "expected_episode_count must be 8640",
    )
    _require(
        campaign.get("provenance_status") == "needs_valid_campaign_artifact",
        "source campaign must remain artifact-blocked",
    )

    inputs = _require_mapping(packet, "inputs")
    episodes = _repo_relative_path(inputs.get("episodes_jsonl"), "inputs.episodes_jsonl")
    baseline = _repo_relative_path(inputs.get("baseline_path"), "inputs.baseline_path")
    weights = _repo_relative_path(inputs.get("weights_path"), "inputs.weights_path")
    _require(
        inputs.get("input_status") == EXPECTED_INPUT_STATUS, "inputs.input_status must be blocked"
    )
    _require(inputs.get("weights_sha256") == EXPECTED_WEIGHT_HASH, "weights_sha256 mismatch")
    _require(inputs.get("baseline_sha256") == EXPECTED_BASELINE_HASH, "baseline_sha256 mismatch")
    _require((repo_root / weights).is_file(), "weights_path must exist")
    _require((repo_root / baseline).is_file(), "baseline_path must exist")
    _require(_sha256(repo_root / weights) == EXPECTED_WEIGHT_HASH, "weights file hash mismatch")
    _require(_sha256(repo_root / baseline) == EXPECTED_BASELINE_HASH, "baseline file hash mismatch")

    export = _require_mapping(packet, "export")
    _require(
        export.get("script") == "scripts/benchmark/snqi_scalarization_sensitivity_export.py",
        "export script mismatch",
    )
    _require((repo_root / str(export["script"])).is_file(), "export script must exist")
    command = list(_require_sequence(export, "command_template"))
    command_text = " ".join(str(part) for part in command)
    for required in (str(episodes), str(baseline), str(weights), "--preflight-out"):
        _require(required in command_text, f"command_template missing {required}")

    artifacts = {str(item) for item in _require_sequence(export, "required_artifacts")}
    _require(artifacts == REQUIRED_ARTIFACTS, "required_artifacts mismatch")

    success = _require_mapping(packet, "success_criteria")
    _require(success.get("preflight_status") == "ready", "success preflight_status must be ready")
    _require(int(success.get("min_planners", 0)) >= 2, "success min_planners must be at least 2")
    _require(
        {str(item) for item in _require_sequence(success, "required_report_fields")}
        == REQUIRED_REPORT_FIELDS,
        "required_report_fields mismatch",
    )
    _require(
        success.get("blocked_status_when_inputs_missing") == EXPECTED_STATUS,
        "blocked status must match execution current_status",
    )

    return {
        "status": "ok",
        "issue": 3653,
        "current_status": execution["current_status"],
        "target_host": execution["target_host"],
        "source_job_id": campaign["source_job_id"],
        "expected_episode_count": campaign["expected_episode_count"],
        "weights_sha256": inputs["weights_sha256"],
        "baseline_sha256": inputs["baseline_sha256"],
        "episodes_jsonl": str(episodes),
        "artifact_count": len(artifacts),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the packet checker CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable summary.")
    args = parser.parse_args(argv)

    try:
        summary = validate_packet(load_packet(args.packet))
    except PacketError as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(
            "OK: issue #3653 SNQI decision-disagreement packet is "
            f"{summary['current_status']} for job {summary['source_job_id']}"
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
