"""Tests for learned-risk launch-packet validation."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import pytest
import yaml

from robot_sf.training.learned_risk_launch_packet import (
    LearnedRiskLaunchPacketError,
    validate_launch_packet,
)
from scripts.validation.validate_learned_risk_launch_packet import main as validate_cli_main

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_packet(tmp_path: Path, packet: dict[str, object]) -> Path:
    path = tmp_path / "packet.yaml"
    path.write_text(yaml.safe_dump(packet, sort_keys=False), encoding="utf-8")
    return path


def _valid_packet(tmp_path: Path) -> dict[str, object]:
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
    return {
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
            "feature_inputs": ["metrics.min_distance", "trajectory_features.route_progress_delta"],
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
                "wandb-artifact://robot-sf/policy-search/baseline:pending",
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


def test_issue_1395_launch_packet_validates() -> None:
    """The checked-in #1395 launch packet should pass the local preflight."""
    report = validate_launch_packet(
        _REPO_ROOT / "configs/training/learned_risk_model_issue_1395_launch_packet.yaml"
    )

    assert report["status"] == "valid"
    assert report["candidate_id"] == "learned_risk_model_v1"
    assert report["baseline_comparison"]["candidate_id"] == (
        "hybrid_rule_v3_static_margin0_waypoint2"
    )
    assert report["slurm_execution"]["entrypoint"] == (
        "scripts/training/train_learned_risk_model.py"
    )
    assert report["slurm_execution"]["expected_log_path"].startswith("slurm://")


def test_validate_launch_packet_rejects_missing_slurm_execution(tmp_path: Path) -> None:
    """Launch packets must name the Slurm execution/log artifact contract."""
    packet = _valid_packet(tmp_path)
    broken = copy.deepcopy(packet)
    broken.pop("slurm_execution")

    with pytest.raises(LearnedRiskLaunchPacketError, match="slurm_execution"):
        validate_launch_packet(_write_packet(tmp_path, broken))


def test_validate_launch_packet_rejects_local_slurm_execution_output(tmp_path: Path) -> None:
    """Slurm execution outputs must point to durable or scheduler-managed locations."""
    packet = _valid_packet(tmp_path)
    broken = copy.deepcopy(packet)
    broken["slurm_execution"]["expected_output_root"] = "output/learned-risk/local"

    with pytest.raises(LearnedRiskLaunchPacketError, match="worktree-local output"):
        validate_launch_packet(_write_packet(tmp_path, broken))


def test_validate_launch_packet_rejects_missing_trace_fields(tmp_path: Path) -> None:
    """Trace fixtures must expose the declared fail-closed fields."""
    packet = _valid_packet(tmp_path)
    trace = tmp_path / "broken_trace.jsonl"
    trace.write_text('{"scenario_id":"demo","seed":1}\n', encoding="utf-8")
    broken = copy.deepcopy(packet)
    broken["trace_input_contract"]["trace_fixture_paths"] = [str(trace)]
    broken["trace_input_contract"]["checksums"] = {
        str(trace): hashlib.sha256(trace.read_bytes()).hexdigest()
    }

    with pytest.raises(LearnedRiskLaunchPacketError, match="missing required fields"):
        validate_launch_packet(_write_packet(tmp_path, broken))


def test_validate_launch_packet_rejects_non_authoritative_hard_guards(tmp_path: Path) -> None:
    """Learned risk may not replace hard safety guards."""
    packet = _valid_packet(tmp_path)
    broken = copy.deepcopy(packet)
    broken["safety_policy"]["hard_guards_authoritative"] = False

    with pytest.raises(LearnedRiskLaunchPacketError, match="hard_guards_authoritative"):
        validate_launch_packet(_write_packet(tmp_path, broken))


def test_validate_launch_packet_rejects_output_artifacts(tmp_path: Path) -> None:
    """Launch packets must not freeze worktree-local output artifacts."""
    packet = _valid_packet(tmp_path)
    broken = copy.deepcopy(packet)
    broken["baseline_comparison"]["summary_artifacts"] = ["output/local-summary.json"]
    broken["baseline_comparison"]["checksums"] = {"output/local-summary.json": "0" * 64}

    with pytest.raises(LearnedRiskLaunchPacketError, match="worktree-local output"):
        validate_launch_packet(_write_packet(tmp_path, broken))


def test_validate_launch_packet_rejects_invalid_trace_json(tmp_path: Path) -> None:
    """Invalid JSON in trace fixtures raises ValueError with file path and line number."""
    packet = _valid_packet(tmp_path)
    trace = tmp_path / "bad_trace.jsonl"
    trace.write_text(
        '{"scenario_id":"ok","seed":1,"termination_reason":"ok","metrics":{},"trajectory_features":{},"labels":{"collision":false,"near_miss":false,"low_progress":false}}\n'
        "{invalid json here}\n",
        encoding="utf-8",
    )
    broken = copy.deepcopy(packet)
    broken["trace_input_contract"]["trace_fixture_paths"] = [str(trace)]
    broken["trace_input_contract"]["checksums"] = {
        str(trace): hashlib.sha256(trace.read_bytes()).hexdigest()
    }

    with pytest.raises(ValueError, match="bad_trace.jsonl:2: invalid JSON"):
        validate_launch_packet(_write_packet(tmp_path, broken))


def test_validate_launch_packet_cli_reports_json() -> None:
    """The CLI should expose a machine-readable valid report."""
    exit_code = validate_cli_main(
        [
            "--config",
            str(_REPO_ROOT / "configs/training/learned_risk_model_issue_1395_launch_packet.yaml"),
            "--json",
        ]
    )

    assert exit_code == 0
