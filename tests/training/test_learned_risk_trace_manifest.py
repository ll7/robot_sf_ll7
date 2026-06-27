"""Tests for durable learned-risk trace-manifest validation.

These tests use synthetic manifests to exercise the fail-closed decision boundary
(structurally invalid -> raise; well-formed-but-unresolved -> blocked;
contract-complete -> ready). They also assert the checked-in #2312 manifest is
structurally valid and honestly reports the current blocked state.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_sf.training.learned_risk_trace_manifest import (
    DECISION_BLOCKED,
    DECISION_READY,
    LearnedRiskTraceManifestError,
    validate_trace_manifest,
)
from scripts.validation.validate_learned_risk_trace_manifest import main as validate_cli_main

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHECKED_IN_MANIFEST = _REPO_ROOT / "configs/training/learned_risk_trace_manifest_issue_2312.yaml"


def _write_manifest(tmp_path: Path, manifest: dict[str, object]) -> Path:
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


def _ready_manifest() -> dict[str, object]:
    """A fully resolvable manifest that should decide ready_for_training_handoff."""
    stress = "wandb-artifact://robot-sf/learned-risk/traces_stress_slice:v3"
    full = "wandb-artifact://robot-sf/learned-risk/traces_full_matrix:v3"
    baseline = "wandb-artifact://robot-sf/policy-search/baseline:v7"
    return {
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


def test_checked_in_manifest_reports_blocked() -> None:
    """The committed #2312 manifest is valid and honestly blocked (fail-closed)."""
    report = validate_trace_manifest(_CHECKED_IN_MANIFEST, repo_root=_REPO_ROOT)

    assert report["status"] == "ok"
    assert report["training_ready"] is False
    assert report["training_readiness_decision"] == DECISION_BLOCKED
    assert report["parent_issue"] == 1472
    # The pending baseline alias must be surfaced as a concrete blocker reason.
    assert any("baseline_artifact_uri" in blocker for blocker in report["blockers"])


def test_ready_manifest_decides_ready(tmp_path: Path) -> None:
    """A contract-complete manifest yields ready_for_training_handoff."""
    report = validate_trace_manifest(_write_manifest(tmp_path, _ready_manifest()))

    assert report["training_ready"] is True
    assert report["training_readiness_decision"] == DECISION_READY
    assert report["blockers"] == []
    assert report["label_availability"] == {
        "collision": "present",
        "near_miss": "present",
        "low_progress": "present",
    }


def test_pending_baseline_alias_is_blocked(tmp_path: Path) -> None:
    """A :pending baseline alias must never read as training-ready."""
    manifest = _ready_manifest()
    manifest["baseline_artifact_uri"] = "wandb-artifact://robot-sf/policy-search/baseline:pending"

    report = validate_trace_manifest(_write_manifest(tmp_path, manifest))

    assert report["training_readiness_decision"] == DECISION_BLOCKED
    assert any("unresolved placeholder" in blocker for blocker in report["blockers"])


def test_absent_label_is_blocked(tmp_path: Path) -> None:
    """Every required learned-risk label must be present to be ready."""
    manifest = _ready_manifest()
    manifest["label_availability"]["near_miss"] = "absent"

    report = validate_trace_manifest(_write_manifest(tmp_path, manifest))

    assert report["training_readiness_decision"] == DECISION_BLOCKED
    assert any("near_miss" in blocker for blocker in report["blockers"])


def test_concrete_trace_without_checksum_is_blocked(tmp_path: Path) -> None:
    """A concrete durable trace URI without a recorded digest fails closed."""
    manifest = _ready_manifest()
    manifest["checksums"] = {}

    report = validate_trace_manifest(_write_manifest(tmp_path, manifest))

    assert report["training_readiness_decision"] == DECISION_BLOCKED
    assert any("SHA-256" in blocker for blocker in report["blockers"])


def test_non_durable_trace_uri_is_blocked(tmp_path: Path) -> None:
    """Trace artifacts must use a durable URI scheme, not a bare relative path."""
    manifest = _ready_manifest()
    manifest["trace_artifacts"] = ["configs/training/local_only.jsonl"]

    report = validate_trace_manifest(_write_manifest(tmp_path, manifest))

    assert report["training_readiness_decision"] == DECISION_BLOCKED
    assert any("durable URI scheme" in blocker for blocker in report["blockers"])


def test_missing_required_split_is_blocked(tmp_path: Path) -> None:
    """Dropping a required scenario slice is a fail-closed blocker."""
    manifest = _ready_manifest()
    manifest["split_ids"] = ["stress_slice"]

    report = validate_trace_manifest(_write_manifest(tmp_path, manifest))

    assert report["training_readiness_decision"] == DECISION_BLOCKED
    assert any("full_matrix" in blocker for blocker in report["blockers"])


def test_baseline_without_checksum_is_blocked(tmp_path: Path) -> None:
    """A concrete baseline URI without a recorded digest fails closed, like traces."""
    manifest = _ready_manifest()
    baseline = manifest["baseline_artifact_uri"]
    del manifest["checksums"][baseline]

    report = validate_trace_manifest(_write_manifest(tmp_path, manifest))

    assert report["training_readiness_decision"] == DECISION_BLOCKED
    assert any("SHA-256" in blocker and baseline in blocker for blocker in report["blockers"])


def test_durable_uri_with_output_segment_is_not_a_hard_error(tmp_path: Path) -> None:
    """A durable URI carrying an 'output' path segment is accepted, not crashed.

    Only worktree-local (non-durable) pointers must raise; a durable scheme may
    legitimately include an 'output' segment.
    """
    durable_output = "wandb-artifact://robot-sf/learned-risk/output/traces:v1"
    manifest = _ready_manifest()
    manifest["trace_artifacts"] = [durable_output]
    manifest["checksums"] = {
        durable_output: "d" * 64,
        manifest["baseline_artifact_uri"]: "c" * 64,
    }

    report = validate_trace_manifest(_write_manifest(tmp_path, manifest))

    assert report["training_readiness_decision"] == DECISION_READY


def test_malformed_yaml_raises_structural_error(tmp_path: Path) -> None:
    """Invalid YAML is a structural error (CLI exit 2), not a raw traceback."""
    path = tmp_path / "manifest.yaml"
    path.write_text("schema_version: [unterminated\n", encoding="utf-8")

    with pytest.raises(LearnedRiskTraceManifestError, match="valid YAML"):
        validate_trace_manifest(path)


def test_non_list_split_ids_fails_closed(tmp_path: Path) -> None:
    """A non-list split_ids value fails closed with a blocker, never a crash."""
    manifest = _ready_manifest()
    manifest["split_ids"] = 5

    report = validate_trace_manifest(_write_manifest(tmp_path, manifest))

    assert report["training_readiness_decision"] == DECISION_BLOCKED
    assert report["split_ids"] == []
    assert any("split_ids" in blocker for blocker in report["blockers"])


def test_wrong_schema_version_raises(tmp_path: Path) -> None:
    """A wrong schema version is a structural error, not a blocked decision."""
    manifest = _ready_manifest()
    manifest["schema_version"] = "learned-risk-trace-manifest.v0"

    with pytest.raises(LearnedRiskTraceManifestError, match="schema_version"):
        validate_trace_manifest(_write_manifest(tmp_path, manifest))


def test_output_local_artifact_raises(tmp_path: Path) -> None:
    """Worktree-local output/ pointers are rejected as a hard error."""
    manifest = _ready_manifest()
    manifest["trace_artifacts"] = ["output/local-trace.jsonl"]

    with pytest.raises(LearnedRiskTraceManifestError, match="worktree-local output"):
        validate_trace_manifest(_write_manifest(tmp_path, manifest))


def test_local_fixture_checksum_mismatch_is_blocked(tmp_path: Path) -> None:
    """A tracked fixture snapshot with a wrong checksum fails closed."""
    fixture = tmp_path / "fixture.jsonl"
    fixture.write_text('{"scenario_id":"demo"}\n', encoding="utf-8")
    manifest = _ready_manifest()
    manifest["local_fixtures"] = [str(fixture)]
    manifest["checksums"][str(fixture)] = "0" * 64

    report = validate_trace_manifest(_write_manifest(tmp_path, manifest))

    assert report["training_readiness_decision"] == DECISION_BLOCKED
    assert any("checksum mismatch" in blocker for blocker in report["blockers"])


def test_cli_blocked_manifest_returns_exit_three() -> None:
    """The CLI returns the blocked exit code (3) for the checked-in manifest."""
    exit_code = validate_cli_main(
        ["--config", str(_CHECKED_IN_MANIFEST), "--repo-root", str(_REPO_ROOT), "--json"]
    )

    assert exit_code == 3


def test_cli_ready_manifest_returns_exit_zero(tmp_path: Path) -> None:
    """The CLI returns success (0) for a contract-complete manifest."""
    path = _write_manifest(tmp_path, _ready_manifest())

    exit_code = validate_cli_main(["--config", str(path), "--json"])

    assert exit_code == 0
