"""Tests for predictive retraining readiness decision packets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from robot_sf.training.predictive_retraining_readiness import (
    DECISION_BLOCKED,
    PredictiveRetrainingReadinessError,
    evaluate_retraining_readiness,
    load_readiness_packet,
)
from scripts.training import check_predictive_retraining_readiness as readiness_cli


def _write_packet(tmp_path: Path, payload: dict[str, object]) -> Path:
    packet = tmp_path / "packet.yaml"
    packet.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return packet


def _complete_payload(tmp_path: Path) -> dict[str, object]:
    weighting_spec = tmp_path / "weighting.yaml"
    pipeline_config = tmp_path / "pipeline.yaml"
    weighting_spec.write_text("profile_id: test\n", encoding="utf-8")
    pipeline_config.write_text("experiment:\n  run_id: test\n", encoding="utf-8")
    return {
        "schema_version": "predictive-retraining-readiness.v1",
        "issue": "#3214",
        "candidate_id": "crossing_conflict_weighted_predictive_retraining",
        "claim_boundary": (
            "No Slurm submission, no full benchmark campaign run, and no paper claim edit."
        ),
        "inputs": {
            "weighting_spec": str(weighting_spec),
            "pipeline_config": str(pipeline_config),
        },
        "prior_result": {
            "status": "verified_negative",
            "evidence_status": "diagnostic negative result only; not paper claim",
            "source_issue": "#3254",
        },
        "launch_decision": {
            "state": "blocked_until_control_law_change",
            "compute_submit_allowed": False,
            "fail_closed_on_missing": True,
            "required_before_rerun": [
                "control_law_change_config",
                "checkpoint_provenance_plan",
                "hard_seed_evaluation_plan",
            ],
        },
    }


def test_default_issue_3214_packet_is_complete_but_launch_blocked() -> None:
    """Checked-in packet is complete but must not authorize blind retraining."""

    repo = Path(__file__).resolve().parents[2]
    report = evaluate_retraining_readiness(repo_root=repo)

    assert report["issue"] == "#3214"
    assert report["packet_complete"] is True
    assert report["launch_ready"] is False
    assert report["decision"] == DECISION_BLOCKED
    assert report["prior_result_status"] == "verified_negative"
    assert report["compute_submit_allowed"] is False
    assert report["blockers"] == []


def test_complete_negative_result_packet_blocks_without_missing_prerequisites(
    tmp_path: Path,
) -> None:
    """A complete packet reports the negative-result decision instead of missing inputs."""

    packet = _write_packet(tmp_path, _complete_payload(tmp_path))

    report = evaluate_retraining_readiness(packet, repo_root=tmp_path)

    assert report["packet_complete"] is True
    assert report["launch_ready"] is False
    assert report["decision"] == DECISION_BLOCKED
    assert report["blockers"] == []


def test_missing_retraining_prerequisites_fail_closed(tmp_path: Path) -> None:
    """Missing control-law/provenance prerequisites keep packet incomplete."""

    payload = _complete_payload(tmp_path)
    payload["launch_decision"] = {
        "state": "blocked_until_control_law_change",
        "compute_submit_allowed": False,
        "fail_closed_on_missing": True,
        "required_before_rerun": ["control_law_change_config"],
    }
    packet = _write_packet(tmp_path, payload)

    report = evaluate_retraining_readiness(packet, repo_root=tmp_path)

    assert report["packet_complete"] is False
    assert report["launch_ready"] is False
    assert any("checkpoint_provenance_plan" in blocker for blocker in report["blockers"])
    assert any("hard_seed_evaluation_plan" in blocker for blocker in report["blockers"])


def test_missing_input_path_fails_closed(tmp_path: Path) -> None:
    """Missing public config inputs are surfaced as blockers, not launch-ready state."""

    payload = _complete_payload(tmp_path)
    payload["inputs"] = {
        "weighting_spec": str(tmp_path / "missing_weighting.yaml"),
        "pipeline_config": str(tmp_path / "missing_pipeline.yaml"),
    }
    packet = _write_packet(tmp_path, payload)

    report = evaluate_retraining_readiness(packet, repo_root=tmp_path)

    assert report["packet_complete"] is False
    assert report["launch_ready"] is False
    assert any("inputs.weighting_spec does not exist" in blocker for blocker in report["blockers"])
    assert any("inputs.pipeline_config does not exist" in blocker for blocker in report["blockers"])


def test_cli_require_launch_ready_exits_blocked(monkeypatch, tmp_path: Path, capsys) -> None:
    """CLI can be used as a fail-closed launch gate when compute authorization exists later."""

    packet = _write_packet(tmp_path, _complete_payload(tmp_path))
    monkeypatch.setattr(
        readiness_cli,
        "_parse_args",
        lambda: readiness_cli.argparse.Namespace(
            packet=packet,
            repo_root=tmp_path,
            require_launch_ready=True,
        ),
    )

    assert readiness_cli.main() == 3
    report = json.loads(capsys.readouterr().out)
    assert report["decision"] == DECISION_BLOCKED
    assert report["launch_ready"] is False


def test_load_readiness_packet_rejects_non_mapping(tmp_path: Path) -> None:
    """Malformed packet YAML remains operator error."""

    packet = tmp_path / "bad.yaml"
    packet.write_text("- not\n- mapping\n", encoding="utf-8")

    with pytest.raises(PredictiveRetrainingReadinessError, match="must be a mapping"):
        load_readiness_packet(packet)
