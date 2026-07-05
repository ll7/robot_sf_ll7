"""Tests for the issue #1428 ORCA-residual behavior-cloning lineage packet."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_sf.training.orca_residual_lineage_packet import (
    OrcaResidualLineagePacketError,
    validate_launch_packet,
    validate_smoke_nominal_gate,
)

REPO_ROOT = Path(__file__).parents[2]
CONFIG = REPO_ROOT / "configs/training/orca_residual/orca_residual_bc_issue_1428.yaml"


def test_checked_in_packet_is_valid() -> None:
    """The versioned packet should satisfy the fail-closed local preflight."""
    report = validate_launch_packet(CONFIG, repo_root=REPO_ROOT)

    assert report["status"] == "valid"
    assert report["objective"]["method"] == "behavior_cloning_residual"
    assert report["objective"]["target"] == "progress_probe_bounded_policy_action_minus_orca_action"
    assert report["objective"]["revision_id"] == "orca_residual_progress_probe_v1"
    assert report["residual_bounds"]["linear_delta"] == 0.35
    assert report["residual_bounds"]["angular_delta"] == 0.35


def test_packet_allows_legacy_residual_objective_target(tmp_path: Path) -> None:
    """Existing lineage packets keep validating while v1 records the progress-probe revision."""
    packet = _load_packet()
    packet["objective"]["target"] = "bounded_policy_action_minus_orca_action"
    packet["objective"].pop("revision_id", None)
    packet["objective"].pop("revision_reason", None)
    legacy_config = tmp_path / "legacy_objective.yaml"
    legacy_config.write_text(yaml.safe_dump(packet), encoding="utf-8")

    report = validate_launch_packet(legacy_config, repo_root=REPO_ROOT)

    assert report["status"] == "valid"
    assert report["objective"]["target"] == "bounded_policy_action_minus_orca_action"


def test_packet_rejects_unknown_objective_target(tmp_path: Path) -> None:
    """The objective label is allowlisted so ambiguous launch packets fail closed."""
    packet = _load_packet()
    packet["objective"]["target"] = "unbounded_orca_residual"
    bad_config = tmp_path / "bad_objective.yaml"
    bad_config.write_text(yaml.safe_dump(packet), encoding="utf-8")

    with pytest.raises(OrcaResidualLineagePacketError, match="objective.target"):
        validate_launch_packet(bad_config, repo_root=REPO_ROOT)


def test_packet_rejects_privileged_observation_features(tmp_path: Path) -> None:
    """Scenario-future or benchmark-only fields must not enter the residual policy."""
    packet = _load_packet()
    packet["observation_contract"]["privileged_features_allowed"] = True
    bad_config = tmp_path / "bad_privileged.yaml"
    bad_config.write_text(yaml.safe_dump(packet), encoding="utf-8")

    with pytest.raises(OrcaResidualLineagePacketError, match="privileged_features_allowed"):
        validate_launch_packet(bad_config, repo_root=REPO_ROOT)


def test_packet_requires_residual_bounds_to_match_candidate(tmp_path: Path) -> None:
    """The training packet bounds must stay aligned with the runtime candidate surface."""
    packet = _load_packet()
    packet["residual_bounds"]["linear_delta"] = 0.5
    bad_config = tmp_path / "bad_bound.yaml"
    bad_config.write_text(yaml.safe_dump(packet), encoding="utf-8")

    with pytest.raises(OrcaResidualLineagePacketError, match="linear_delta"):
        validate_launch_packet(bad_config, repo_root=REPO_ROOT)


def test_packet_rejects_worktree_local_output_artifact(tmp_path: Path) -> None:
    """Durable lineage must not depend on worktree-local output paths."""
    packet = _load_packet()
    packet["expected_outputs"]["checkpoint_pointer"] = "output/tmp/checkpoint.zip"
    bad_config = tmp_path / "bad_output.yaml"
    bad_config.write_text(yaml.safe_dump(packet), encoding="utf-8")

    with pytest.raises(OrcaResidualLineagePacketError, match="worktree-local output"):
        validate_launch_packet(bad_config, repo_root=REPO_ROOT)


def test_packet_allows_non_output_sibling_artifact_by_path(tmp_path: Path) -> None:
    """Output checks must use path parents, not fragile substring matching."""
    packet = _load_packet()
    sibling = tmp_path / "not-output" / "checkpoint.zip"
    sibling.parent.mkdir()
    sibling.write_text("fixture", encoding="utf-8")
    packet["expected_outputs"]["checkpoint_pointer"] = str(sibling)
    bad_config = tmp_path / "bad_sibling.yaml"
    bad_config.write_text(yaml.safe_dump(packet), encoding="utf-8")

    report = validate_launch_packet(bad_config, repo_root=REPO_ROOT)

    assert report["status"] == "valid"


def test_packet_rejects_directory_artifact_path(tmp_path: Path) -> None:
    """Directory-valued artifact pointers should fail closed before hashing."""
    packet = _load_packet()
    directory_artifact = tmp_path / "checkpoint_dir"
    directory_artifact.mkdir()
    packet["expected_outputs"]["checkpoint_pointer"] = str(directory_artifact)
    bad_config = tmp_path / "bad_directory.yaml"
    bad_config.write_text(yaml.safe_dump(packet), encoding="utf-8")

    with pytest.raises(
        OrcaResidualLineagePacketError, match="path does not exist or is a directory"
    ):
        validate_launch_packet(bad_config, repo_root=REPO_ROOT)


def test_packet_requires_guard_and_residual_diagnostics(tmp_path: Path) -> None:
    """Contribution, clipping, guard-veto, and degraded-status diagnostics are mandatory."""
    packet = _load_packet()
    packet["diagnostics"]["required_fields"].remove("guard_veto_rate")
    bad_config = tmp_path / "bad_diagnostics.yaml"
    bad_config.write_text(yaml.safe_dump(packet), encoding="utf-8")

    with pytest.raises(OrcaResidualLineagePacketError, match="guard_veto_rate"):
        validate_launch_packet(bad_config, repo_root=REPO_ROOT)


def test_packet_keeps_slurm_execution_deferred(tmp_path: Path) -> None:
    """The local lineage packet must not claim to submit the bounded Slurm job."""
    packet = _load_packet()
    packet["execution_boundary"]["submit_slurm_from_this_issue"] = True
    bad_config = tmp_path / "bad_slurm.yaml"
    bad_config.write_text(yaml.safe_dump(packet), encoding="utf-8")

    with pytest.raises(OrcaResidualLineagePacketError, match="submit_slurm_from_this_issue"):
        validate_launch_packet(bad_config, repo_root=REPO_ROOT)


def test_smoke_nominal_gate_accepts_complete_clear_summary() -> None:
    """A complete smoke summary may unlock nominal escalation."""
    report = validate_smoke_nominal_gate(_passing_smoke_summary())

    assert report["status"] == "valid"
    assert report["gate"] == "issue_1475_smoke_to_nominal"
    assert report["nominal_escalation_allowed"] is True


def test_smoke_nominal_gate_requires_issue_1475_evidence_fields() -> None:
    """Missing smoke telemetry keeps nominal blocked fail-closed."""
    summary = _passing_smoke_summary()
    summary.pop("guard_veto_rate")

    with pytest.raises(OrcaResidualLineagePacketError, match="guard_veto_rate"):
        validate_smoke_nominal_gate(summary)


def test_smoke_nominal_gate_rejects_degraded_or_pending_artifacts() -> None:
    """Fallback/degraded execution and pending artifacts are not success evidence."""
    summary = _passing_smoke_summary()
    summary["fallback_degraded_status"] = "degraded"
    summary["artifact_pointer_status"] = "pending"
    summary["nominal_escalation_allowed"] = True

    with pytest.raises(OrcaResidualLineagePacketError) as exc_info:
        validate_smoke_nominal_gate(summary)
    message = str(exc_info.value)
    assert "fallback_degraded_status" in message
    assert "artifact_pointer_status" in message


def _passing_smoke_summary() -> dict:
    return {
        "success_rate": 0.80,
        "collision_rate": 0.02,
        "residual_clipping_rate": 0.10,
        "guard_veto_rate": 0.0,
        "fallback_degraded_status": "clear",
        "artifact_pointer_status": "durable",
        "nominal_escalation_allowed": True,
    }


def _load_packet() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
