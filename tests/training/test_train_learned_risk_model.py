"""Tests for the learned-risk model v1 trainer entrypoint (issue #4617)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from robot_sf.training.learned_risk_trainer import (
    CANDIDATE_ID,
    CLAIM_BOUNDARY,
    STATE_BLOCKED_TRACE_MANIFEST,
    STATE_SMOKE_COMPLETED,
    TRAINING_SCHEMA_VERSION,
    LearnedRiskTrainerError,
    build_matrix,
    load_training_config,
    resolve_feature,
    synthesize_smoke_records,
)
from scripts.training.train_learned_risk_model import build_arg_parser
from scripts.training.train_learned_risk_model import main as trainer_main

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO_ROOT / "configs/training/learned_risk_model_v1.yaml"
_LAUNCH_PACKET_PATH = (
    _REPO_ROOT / "configs/training/learned_risk_model_issue_1395_launch_packet.yaml"
)


def test_config_file_exists_and_matches_launch_packet() -> None:
    """The shipped config exists and agrees with the #1472 launch-packet contract."""
    assert _CONFIG_PATH.is_file()
    config = load_training_config(_CONFIG_PATH)
    assert config["schema_version"] == TRAINING_SCHEMA_VERSION
    assert config["candidate_id"] == CANDIDATE_ID

    packet = yaml.safe_load(_LAUNCH_PACKET_PATH.read_text(encoding="utf-8"))
    contract = packet["trace_input_contract"]
    assert config["labels"] == contract["label_targets"]
    assert config["features"] == contract["feature_inputs"]
    assert config["launch_packet"].endswith("learned_risk_model_issue_1395_launch_packet.yaml")


def test_entrypoint_help_works() -> None:
    """The entrypoint exposes a parseable CLI contract (importable + --help)."""
    parser = build_arg_parser()
    args = parser.parse_args(["--config", "x.yaml", "--smoke"])
    assert args.smoke is True
    assert args.config == Path("x.yaml")
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])
    assert excinfo.value.code == 0


def test_smoke_run_writes_status_artifact(tmp_path: Path) -> None:
    """A synthetic CPU smoke run produces a well-formed status artifact and exits 0."""
    status_out = tmp_path / "smoke" / "status.json"
    exit_code = trainer_main(
        [
            "--config",
            str(_CONFIG_PATH),
            "--smoke",
            "--output-root",
            str(tmp_path / "smoke"),
            "--status-out",
            str(status_out),
            "--repo-root",
            str(_REPO_ROOT),
        ]
    )
    assert exit_code == 0
    assert status_out.is_file()
    status = json.loads(status_out.read_text(encoding="utf-8"))

    assert status["candidate_id"] == CANDIDATE_ID
    assert status["mode"] == "smoke"
    assert status["training_state"] == STATE_SMOKE_COMPLETED
    assert status["labels"] == ["collision", "near_miss", "low_progress"]
    assert status["claim_boundary"] == CLAIM_BOUNDARY
    assert status["slurm_submission"] is False
    assert status["row_count"] > 0
    # Every label head reports the required diagnostics keys.
    for label in status["labels"]:
        diag = status["diagnostics"][label]
        for key in ("auroc", "auprc", "brier", "false_negative_rate"):
            assert key in diag


def test_real_mode_blocks_when_trace_manifest_not_ready(tmp_path: Path) -> None:
    """Real mode fails closed (exit 3) while durable trace inputs stay unresolved."""
    status_out = tmp_path / "real" / "status.json"
    exit_code = trainer_main(
        [
            "--config",
            str(_CONFIG_PATH),
            "--output-root",
            str(tmp_path / "real"),
            "--status-out",
            str(status_out),
            "--repo-root",
            str(_REPO_ROOT),
        ]
    )
    assert exit_code == 3
    status = json.loads(status_out.read_text(encoding="utf-8"))
    assert status["mode"] == "real"
    assert status["training_state"] == STATE_BLOCKED_TRACE_MANIFEST
    assert status["slurm_submission"] is False
    assert status["blockers"], "blocked real mode must record the campaign blockers"


def test_config_declared_paths_are_not_worktree_output() -> None:
    """The config's declared output/status paths honour the no-`output/` contract."""
    config = load_training_config(_CONFIG_PATH)
    assert "output" not in Path(config["output_root"]).parts
    assert "output" not in Path(config["status_artifact_path"]).parts


def test_config_rejects_output_declared_paths(tmp_path: Path) -> None:
    """A config that declares a worktree-local `output/` artifact path fails closed."""
    config = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    config["output_root"] = "output/learned_risk_model_v1"
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(LearnedRiskTrainerError, match="output"):
        load_training_config(bad)


def test_config_rejects_label_mismatch(tmp_path: Path) -> None:
    """A config whose labels disagree with the launch-packet order fails closed."""
    config = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    config["labels"] = ["collision", "near_miss"]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(LearnedRiskTrainerError, match="labels"):
        load_training_config(bad)


def test_missing_required_trace_field_fails_closed() -> None:
    """Trace rows missing a required episode field fail closed with a clear error."""
    config = load_training_config(_CONFIG_PATH)
    records = synthesize_smoke_records(config)
    del records[0]["metrics"]
    with pytest.raises(LearnedRiskTrainerError, match="missing required fields"):
        build_matrix(records, list(config["features"]), list(config["labels"]))


def test_resolve_feature_dotted_and_plain() -> None:
    """Feature resolution reads dotted paths from root and plain names from features."""
    record = {
        "trajectory_features": {"min_rollout_clearance_m": 0.7},
        "metrics": {"min_distance": 1.4},
    }
    assert resolve_feature(record, "min_rollout_clearance_m") == pytest.approx(0.7)
    assert resolve_feature(record, "metrics.min_distance") == pytest.approx(1.4)
    with pytest.raises(LearnedRiskTrainerError, match="missing"):
        resolve_feature(record, "metrics.absent")


def test_smoke_run_via_fixture_jsonl(tmp_path: Path) -> None:
    """Smoke mode can consume a JSONL trace fixture instead of synthetic rows."""
    config = load_training_config(_CONFIG_PATH)
    records = synthesize_smoke_records(config)
    fixture = tmp_path / "traces.jsonl"
    fixture.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")
    status_out = tmp_path / "status.json"
    exit_code = trainer_main(
        [
            "--config",
            str(_CONFIG_PATH),
            "--smoke",
            "--trace-fixture",
            str(fixture),
            "--status-out",
            str(status_out),
            "--repo-root",
            str(_REPO_ROOT),
        ]
    )
    assert exit_code == 0
    status = json.loads(status_out.read_text(encoding="utf-8"))
    assert status["row_count"] == len(records)
