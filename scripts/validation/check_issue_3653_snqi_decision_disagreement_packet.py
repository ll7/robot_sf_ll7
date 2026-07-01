#!/usr/bin/env python3
"""Validate the issue #3653 SNQI decision-disagreement application packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
EXPECTED_EVIDENCE_PACKET_HASH = "7da1d5607536bc35d82d482029e150c3cf6442f586ac05e06c925fa9d9e2850c"
EXPECTED_ARTIFACT_ROOT = "output/issue1554-s20-h500-l40s-mem180/13175"
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


def _validate_evidence_packet(campaign: Mapping[str, Any], *, repo_root: Path) -> Path:
    evidence_packet_path = _repo_relative_path(
        campaign.get("evidence_packet"), "source_campaign.evidence_packet"
    )
    _require(
        campaign.get("evidence_packet_sha256") == EXPECTED_EVIDENCE_PACKET_HASH,
        "evidence_packet_sha256 mismatch",
    )
    evidence_packet_file = repo_root / evidence_packet_path
    _require(evidence_packet_file.is_file(), "evidence_packet must exist")
    _require(
        _sha256(evidence_packet_file) == EXPECTED_EVIDENCE_PACKET_HASH,
        "evidence packet hash mismatch",
    )

    evidence_packet = json.loads(evidence_packet_file.read_text(encoding="utf-8"))
    _require(evidence_packet.get("job_id") == "13175", "evidence packet job_id mismatch")
    _require(evidence_packet.get("status") == "diagnostic_only", "evidence packet status mismatch")
    _require(
        evidence_packet.get("artifact_root") == EXPECTED_ARTIFACT_ROOT,
        "evidence packet artifact_root mismatch",
    )

    coverage = _require_mapping(evidence_packet, "coverage_snapshot")
    packet_campaign = _require_mapping(evidence_packet, "campaign")
    _require(
        packet_campaign.get("scenario_matrix")
        == "configs/scenarios/classic_interactions_francis2023.yaml",
        "evidence packet scenario_matrix mismatch",
    )
    _require(
        packet_campaign.get("seed_set") == "paper_eval_s20",
        "evidence packet seed_set mismatch",
    )
    _require(int(coverage.get("planner_count", 0)) == 9, "evidence packet planner_count mismatch")
    _require(int(coverage.get("episode_rows", 0)) == 8640, "evidence packet episode_rows mismatch")
    return evidence_packet_path


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
        campaign.get("provenance_status") == "needs_raw_episode_artifact",
        "source campaign must remain raw-episode-artifact-blocked",
    )
    _require(campaign.get("artifact_root") == EXPECTED_ARTIFACT_ROOT, "artifact_root mismatch")
    evidence_packet_path = _validate_evidence_packet(campaign, repo_root=repo_root)

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
        "evidence_packet": str(evidence_packet_path),
        "evidence_packet_sha256": campaign["evidence_packet_sha256"],
        "episodes_jsonl": str(episodes),
        "artifact_count": len(artifacts),
    }


def _command_template_value(export: Mapping[str, Any], flag: str) -> str | None:
    """Return the argument value that follows a flag in the packet command template."""

    template = export.get("command_template")
    if not isinstance(template, Sequence) or isinstance(template, (str, bytes)):
        return None
    for index, item in enumerate(template):
        if item == flag and index + 1 < len(template):
            value = template[index + 1]
            if isinstance(value, str):
                return value
    return None


def _packet_export_output_dir(packet: Mapping[str, Any]) -> Path:
    export = _require_mapping(packet, "export")
    value = _command_template_value(export, "--output-dir")
    _require(value is not None, "export.command_template must include --output-dir")
    return _repo_relative_path(value, "export.command_template.--output-dir")


def _packet_preflight_output_path(packet: Mapping[str, Any]) -> Path | None:
    export = _require_mapping(packet, "export")
    value = _command_template_value(export, "--preflight-out")
    if value is None:
        return None
    return _repo_relative_path(value, "export.command_template.--preflight-out")


def _require_export_report_fields(packet: Mapping[str, Any], report: Mapping[str, Any]) -> None:
    success = _require_mapping(packet, "success_criteria")
    required = {str(item) for item in _require_sequence(success, "required_report_fields")}
    missing = sorted(field for field in required if field not in report)
    if missing:
        raise PacketError(f"export report missing required fields: {missing}")
    _require_populated_export_report(report)


def _require_populated_export_report(report: Mapping[str, Any]) -> None:
    """Require usable Pareto and decision-disagreement payloads before export success."""

    _require_populated_pareto_front(report)
    _require_populated_decision_disagreement(report)


def _require_populated_pareto_front(report: Mapping[str, Any]) -> None:
    """Require non-empty finite Pareto-front points."""

    pareto = _require_mapping(report, "pareto_front")
    points = pareto.get("points")
    if not isinstance(points, Sequence) or isinstance(points, (str, bytes)) or not points:
        raise PacketError("export report pareto_front.points must be a non-empty sequence")
    for index, point in enumerate(points, start=1):
        if not isinstance(point, Mapping):
            raise PacketError(f"export report pareto_front point {index} must be a mapping")
        for field in ("planner", "constraints_first_score", "snqi_mean"):
            if field not in point:
                raise PacketError(
                    f"export report pareto_front point {index} missing field {field!r}"
                )
        for field in ("constraints_first_score", "snqi_mean"):
            try:
                value = float(point[field])
            except (TypeError, ValueError) as exc:
                raise PacketError(
                    f"export report pareto_front point {index} non-numeric field {field!r}"
                ) from exc
            if not math.isfinite(value):
                raise PacketError(
                    f"export report pareto_front point {index} non-finite field {field!r}"
                )


def _require_populated_decision_disagreement(report: Mapping[str, Any]) -> None:
    """Require usable decision-disagreement summary values."""

    disagreement = _require_mapping(report, "decision_disagreement")
    for field in (
        "snqi_winner",
        "constraints_first_winner",
        "winner_disagreement",
        "pairwise_disagreement_rate",
        "pairwise_reversal_count",
    ):
        if field not in disagreement:
            raise PacketError(f"export report decision_disagreement missing field {field!r}")
    for field in ("snqi_winner", "constraints_first_winner"):
        if not isinstance(disagreement[field], str) or not disagreement[field].strip():
            raise PacketError(
                f"export report decision_disagreement field {field!r} must be non-empty string"
            )
    if not isinstance(disagreement["winner_disagreement"], bool):
        raise PacketError(
            "export report decision_disagreement field 'winner_disagreement' must be boolean"
        )
    try:
        disagreement_rate = float(disagreement["pairwise_disagreement_rate"])
        reversal_count_raw = float(disagreement["pairwise_reversal_count"])
    except (TypeError, ValueError) as exc:
        raise PacketError("export report decision_disagreement numeric fields are invalid") from exc
    if not math.isfinite(disagreement_rate) or not 0.0 <= disagreement_rate <= 1.0:
        raise PacketError("export report decision_disagreement rate must be finite in [0, 1]")
    if not math.isfinite(reversal_count_raw) or not reversal_count_raw.is_integer():
        raise PacketError(
            "export report decision_disagreement reversal count must be finite integer"
        )
    reversal_count = int(reversal_count_raw)
    if reversal_count < 0:
        raise PacketError("export report decision_disagreement reversal count must be non-negative")


def export_if_ready(
    packet: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the packet's SNQI export only when the campaign input is present and ready."""

    from robot_sf.benchmark.snqi_scalarization_sensitivity import (
        SENSITIVITY_PREFLIGHT_READY,
        build_scalarization_sensitivity_report,
        classify_scalarization_sensitivity_inputs,
        load_baseline_mapping,
        load_jsonl,
        load_weight_mapping,
        write_diagnostic_artifacts,
    )

    repo_root = repo_root or _repo_root()
    summary = validate_packet(packet, repo_root=repo_root)
    inputs = _require_mapping(packet, "inputs")
    episodes_rel = _repo_relative_path(inputs.get("episodes_jsonl"), "inputs.episodes_jsonl")
    episodes = repo_root / episodes_rel
    if not episodes.is_file():
        raise PacketError(f"episodes_jsonl missing: {episodes_rel} (status: {EXPECTED_STATUS})")

    weights = load_weight_mapping(
        repo_root / _repo_relative_path(inputs.get("weights_path"), "inputs.weights_path")
    )
    baseline = load_baseline_mapping(
        repo_root / _repo_relative_path(inputs.get("baseline_path"), "inputs.baseline_path")
    )
    records = load_jsonl(episodes)
    if len(records) != int(summary["expected_episode_count"]):
        raise PacketError(
            "episodes_jsonl row count mismatch: "
            f"expected {summary['expected_episode_count']}, found {len(records)}"
        )
    preflight = classify_scalarization_sensitivity_inputs(
        records,
        weights=weights,
        baseline=baseline,
    )
    if preflight["status"] != SENSITIVITY_PREFLIGHT_READY:
        raise PacketError(
            "SNQI scalarization-sensitivity preflight not ready: "
            f"{preflight['status']} issues={preflight['issues']}"
        )

    resolved_output_dir = output_dir or (repo_root / _packet_export_output_dir(packet))
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    preflight_rel = _packet_preflight_output_path(packet)
    preflight_output = (
        resolved_output_dir / "preflight.json"
        if output_dir is not None
        else repo_root / preflight_rel
        if preflight_rel is not None
        else None
    )
    if preflight_output is not None:
        preflight_output.parent.mkdir(parents=True, exist_ok=True)
        preflight_output.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")

    report = build_scalarization_sensitivity_report(records, weights=weights, baseline=baseline)
    _require_export_report_fields(packet, report)
    artifacts = write_diagnostic_artifacts(report, resolved_output_dir)
    artifact_paths = {
        "json": artifacts.json_path,
        "csv": artifacts.csv_path,
        "decision_disagreement_csv": artifacts.decision_disagreement_csv_path,
        "markdown": artifacts.markdown_path,
        "svg": artifacts.svg_path,
    }
    required = {
        str(item)
        for item in _require_sequence(_require_mapping(packet, "export"), "required_artifacts")
    }
    written = {path.name for path in artifact_paths.values()}
    missing = sorted(required - written)
    if missing:
        raise PacketError(f"export missing required artifacts: {missing}")

    return {
        **summary,
        "status": "exported",
        "preflight_status": preflight["status"],
        "output_dir": str(resolved_output_dir),
        "artifacts": {key: str(path) for key, path in artifact_paths.items()},
        "decision_disagreement_rate": report["summary"]["decision_disagreement_rate"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the packet checker CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable summary.")
    parser.add_argument(
        "--export-if-ready",
        action="store_true",
        help="Run the diagnostic export when packet inputs are present and preflight-ready.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override the packet output directory for --export-if-ready.",
    )
    args = parser.parse_args(argv)

    try:
        packet = load_packet(args.packet)
        if args.export_if_ready:
            summary = export_if_ready(packet, output_dir=args.output_dir)
        else:
            summary = validate_packet(packet)
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
