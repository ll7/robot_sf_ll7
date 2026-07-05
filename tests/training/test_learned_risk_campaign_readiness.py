"""Tests for the learned-risk model v1 campaign-readiness aggregator (issue #1472).

These tests exercise the fail-closed campaign gate: it aggregates the launch-packet
and durable trace-manifest owners and decides ``campaign_launch_ready`` only when
both pass. Synthetic inputs cover ready, each-half-blocked, structurally invalid,
and missing-file cases; one test pins the honest current blocked state of the
checked-in #1472 campaign inputs.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from robot_sf.training.learned_risk_campaign_readiness import (
    CAMPAIGN_BLOCKED,
    CAMPAIGN_READY,
    DEFAULT_LAUNCH_PACKET,
    DEFAULT_TRACE_MANIFEST,
    LearnedRiskCampaignReadinessError,
    evaluate_campaign_readiness,
)
from scripts.validation.check_learned_risk_campaign_readiness import main as readiness_cli_main

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_yaml(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _valid_launch_packet(tmp_path: Path) -> Path:
    """Write a launch packet that passes the launch-packet owner.

    Repo-relative metadata paths (candidate config, source report, slurm handoff)
    resolve against ``_REPO_ROOT``; the trace/baseline fixtures are absolute tmp
    files so checksums match regardless of the resolved repo root.
    """
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        '{"scenario_id":"demo","seed":1,"candidate_id":"baseline",'
        '"termination_reason":"success","metrics":{},'
        '"trajectory_features":{},"labels":{"collision":false,'
        '"near_miss":false,"low_progress":false}}\n',
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.json"
    baseline.write_text('{"candidate_id":"hybrid_rule_v3_static_margin0_waypoint2"}\n')
    packet = {
        "schema_version": "learned-risk-launch-packet.v1",
        "candidate_id": "learned_risk_model_v1",
        "generating_commit": "e14e2f8bc2058d9f0e071219629915dd5b5dd5a8",
        "slurm_handoff": "docs/context/policy_search/SLURM/001_learned_risk_model_v1.md",
        "slurm_execution": {
            "entrypoint": "scripts/training/train_learned_risk_model.py",
            "config": "configs/training/learned_risk_model_v1.yaml",
            "command": [
                "uv",
                "run",
                "python",
                "scripts/training/train_learned_risk_model.py",
                "--config",
                "configs/training/learned_risk_model_v1.yaml",
            ],
            "expected_output_root": (
                "wandb-artifact://robot-sf/learned-risk/learned_risk_model_v1:pending"
            ),
            "expected_log_path": "slurm://learned-risk-model-v1/logs/%j.out",
            "status_artifact_path": (
                "wandb-artifact://robot-sf/learned-risk/learned_risk_model_v1_status:pending"
            ),
        },
        "trace_input_contract": {
            "required_episode_fields": [
                "scenario_id",
                "seed",
                "termination_reason",
                "metrics",
                "trajectory_features",
                "labels",
            ],
            "feature_inputs": ["metrics.min_distance"],
            "label_targets": ["collision", "near_miss", "low_progress"],
            "missing_required_fields_behavior": "fail_closed",
            "trace_fixture_paths": [str(trace)],
            "checksums": {str(trace): hashlib.sha256(trace.read_bytes()).hexdigest()},
        },
        "baseline_comparison": {
            "candidate_id": "hybrid_rule_v3_static_margin0_waypoint2",
            "candidate_config": "configs/policy_search/candidates/"
            "hybrid_rule_v3_static_margin0_waypoint2.yaml",
            "scenario_slices": ["stress_slice", "full_matrix"],
            "seeds": [111, 112, 113],
            "summary_artifacts": [
                str(baseline),
                "wandb-artifact://robot-sf/policy-search/baseline:v7",
            ],
            "source_report": "docs/context/policy_search/reports/"
            "2026-04-30_best_non_learning_local_policy_report.md",
            "checksums": {str(baseline): hashlib.sha256(baseline.read_bytes()).hexdigest()},
        },
        "safety_policy": {
            "hard_guards_authoritative": True,
            "learned_output_role": "auxiliary_cost_only",
            "required_diagnostics": [
                "learned_risk_score",
                "hard_guard_decision",
                "auxiliary_cost_weight",
            ],
        },
        "execution_boundary": {
            "full_training_in_this_issue": False,
            "submit_slurm_from_this_issue": False,
            "local_preflight_command": "uv run python scripts/validation/"
            "validate_learned_risk_launch_packet.py --config packet.yaml --json",
            "slurm_command_shape": "submit follow-up only",
        },
    }
    return _write_yaml(tmp_path / "launch_packet.yaml", packet)


def _ready_trace_manifest(tmp_path: Path) -> Path:
    """Write a contract-complete durable trace manifest (decides ready)."""
    stress = "wandb-artifact://robot-sf/learned-risk/traces_stress_slice:v3"
    full = "wandb-artifact://robot-sf/learned-risk/traces_full_matrix:v3"
    baseline = "wandb-artifact://robot-sf/policy-search/baseline:v7"
    manifest = {
        "schema_version": "learned-risk-trace-manifest.v1",
        "source_issue": 2312,
        "parent_issue": 1472,
        "candidate_id": "learned_risk_model_v1",
        "trace_schema_version": "mechanism_trace.v1",
        "baseline_artifact_uri": baseline,
        "trace_artifacts": [stress, full],
        "split_ids": ["stress_slice", "full_matrix"],
        "required_episode_fields": [
            "scenario_id",
            "seed",
            "candidate_id",
            "termination_reason",
            "metrics",
            "trajectory_features",
            "labels",
        ],
        "label_availability": {
            "collision": "present",
            "near_miss": "present",
            "low_progress": "present",
        },
        "checksums": {stress: "a" * 64, full: "b" * 64, baseline: "c" * 64},
        "retrieval_status": "available",
    }
    return _write_yaml(tmp_path / "trace_manifest.yaml", manifest)


def _gate(report: dict[str, object], name: str) -> dict[str, object]:
    return next(gate for gate in report["gates"] if gate["name"] == name)


def test_checked_in_campaign_reports_blocked() -> None:
    """The committed #1472 inputs are valid but honestly blocked (fail-closed)."""
    report = evaluate_campaign_readiness(
        DEFAULT_LAUNCH_PACKET,
        DEFAULT_TRACE_MANIFEST,
        repo_root=_REPO_ROOT,
    )

    assert report["campaign_state"] == CAMPAIGN_BLOCKED
    assert report["campaign_ready"] is False
    assert report["issue"] == 1472
    assert (
        report["launch_packet"]
        == "configs/training/learned_risk_model_issue_1395_launch_packet.yaml"
    )
    assert (
        report["trace_manifest"] == "configs/training/learned_risk_trace_manifest_issue_2312.yaml"
    )
    # The launch packet is internally valid; the campaign is blocked on the
    # unresolved durable trace/baseline artifacts only.
    assert _gate(report, "launch_packet")["status"] == "passed"
    assert _gate(report, "trace_manifest")["status"] == "blocked"
    assert report["blocking_gates"] == ["trace_manifest"]
    assert any("trace_manifest" in blocker for blocker in report["blockers"])


def test_ready_campaign_decides_ready(tmp_path: Path) -> None:
    """Both gates passing yields a single campaign_launch_ready decision."""
    report = evaluate_campaign_readiness(
        _valid_launch_packet(tmp_path),
        _ready_trace_manifest(tmp_path),
        repo_root=_REPO_ROOT,
    )

    assert report["campaign_state"] == CAMPAIGN_READY
    assert report["campaign_ready"] is True
    assert report["blocking_gates"] == []
    assert report["blockers"] == []


def test_blocked_trace_manifest_blocks_campaign(tmp_path: Path) -> None:
    """A valid launch packet cannot unblock a pending durable trace manifest."""
    report = evaluate_campaign_readiness(
        _valid_launch_packet(tmp_path),
        DEFAULT_TRACE_MANIFEST,
        repo_root=_REPO_ROOT,
    )

    assert report["campaign_state"] == CAMPAIGN_BLOCKED
    assert _gate(report, "launch_packet")["status"] == "passed"
    assert report["blocking_gates"] == ["trace_manifest"]


def test_invalid_launch_packet_blocks_campaign(tmp_path: Path) -> None:
    """A launch packet that weakens hard guards fails closed at the campaign gate."""
    packet_path = _valid_launch_packet(tmp_path)
    packet = yaml.safe_load(packet_path.read_text(encoding="utf-8"))
    broken = copy.deepcopy(packet)
    broken["safety_policy"]["hard_guards_authoritative"] = False
    _write_yaml(packet_path, broken)

    report = evaluate_campaign_readiness(
        packet_path,
        _ready_trace_manifest(tmp_path),
        repo_root=_REPO_ROOT,
    )

    assert report["campaign_state"] == CAMPAIGN_BLOCKED
    assert _gate(report, "launch_packet")["status"] == "blocked"
    assert "launch_packet" in report["blocking_gates"]
    assert any("hard_guards_authoritative" in blocker for blocker in report["blockers"])


def test_structurally_invalid_manifest_blocks_without_raising(tmp_path: Path) -> None:
    """A malformed manifest is a fail-closed blocked gate, not an aggregator crash."""
    bad_manifest = _write_yaml(tmp_path / "bad_manifest.yaml", {"schema_version": "wrong"})

    report = evaluate_campaign_readiness(
        _valid_launch_packet(tmp_path),
        bad_manifest,
        repo_root=_REPO_ROOT,
    )

    assert report["campaign_state"] == CAMPAIGN_BLOCKED
    assert _gate(report, "trace_manifest")["status"] == "blocked"
    assert _gate(report, "trace_manifest")["blockers"]


def test_missing_config_file_raises(tmp_path: Path) -> None:
    """A config path that is not a file is an operator error, not a blocked gate."""
    with pytest.raises(LearnedRiskCampaignReadinessError, match="trace manifest is not a file"):
        evaluate_campaign_readiness(
            _valid_launch_packet(tmp_path),
            tmp_path / "does_not_exist.yaml",
            repo_root=_REPO_ROOT,
        )


def test_cli_blocked_exit_code() -> None:
    """The CLI returns exit 3 for the checked-in (blocked) campaign state."""
    exit_code = readiness_cli_main(["--repo-root", str(_REPO_ROOT)])
    assert exit_code == 3


def test_cli_ready_exit_code(tmp_path: Path) -> None:
    """The CLI returns exit 0 when both gates pass."""
    exit_code = readiness_cli_main(
        [
            "--launch-packet",
            str(_valid_launch_packet(tmp_path)),
            "--trace-manifest",
            str(_ready_trace_manifest(tmp_path)),
            "--repo-root",
            str(_REPO_ROOT),
            "--json",
        ]
    )
    assert exit_code == 0


def test_cli_writes_ready_json_path(tmp_path: Path) -> None:
    """The CLI can write a ready report to a requested JSON path."""
    report_path = tmp_path / "reports" / "ready.json"
    exit_code = readiness_cli_main(
        [
            "--launch-packet",
            str(_valid_launch_packet(tmp_path)),
            "--trace-manifest",
            str(_ready_trace_manifest(tmp_path)),
            "--repo-root",
            str(_REPO_ROOT),
            "--json",
            str(report_path),
        ]
    )

    assert exit_code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["campaign_state"] == CAMPAIGN_READY
    assert report["campaign_ready"] is True


def test_cli_writes_blocked_json_path(tmp_path: Path) -> None:
    """The CLI preserves blocked exit code when writing report JSON."""
    report_path = tmp_path / "blocked.json"
    exit_code = readiness_cli_main(
        [
            "--repo-root",
            str(_REPO_ROOT),
            "--json",
            str(report_path),
        ]
    )

    assert exit_code == 3
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["campaign_state"] == CAMPAIGN_BLOCKED
    assert report["campaign_ready"] is False
    assert report["blocking_gates"] == ["trace_manifest"]
    assert any(":pending" in blocker for blocker in report["blockers"])
    assert any("retrieval_status" in blocker for blocker in report["blockers"])


def test_cli_missing_file_exit_code(tmp_path: Path) -> None:
    """The CLI returns exit 2 when an input file is missing."""
    exit_code = readiness_cli_main(
        [
            "--launch-packet",
            str(_valid_launch_packet(tmp_path)),
            "--trace-manifest",
            str(tmp_path / "missing.yaml"),
            "--repo-root",
            str(_REPO_ROOT),
        ]
    )
    assert exit_code == 2
