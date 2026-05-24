"""Tests for the issue #1428 ORCA-residual behavior-cloning lineage packet."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_sf.training.orca_residual_lineage_packet import (
    OrcaResidualLineagePacketError,
    validate_launch_packet,
)

CONFIG = Path("configs/training/orca_residual/orca_residual_bc_issue_1428.yaml")


def test_checked_in_packet_is_valid() -> None:
    """The versioned packet should satisfy the fail-closed local preflight."""
    report = validate_launch_packet(CONFIG)

    assert report["status"] == "valid"
    assert report["objective"]["method"] == "behavior_cloning_residual"
    assert report["residual_bounds"]["linear_delta"] == 0.25
    assert report["residual_bounds"]["angular_delta"] == 0.35


def test_packet_rejects_privileged_observation_features(tmp_path: Path) -> None:
    """Scenario-future or benchmark-only fields must not enter the residual policy."""
    packet = _load_packet()
    packet["observation_contract"]["privileged_features_allowed"] = True
    bad_config = tmp_path / "bad_privileged.yaml"
    bad_config.write_text(yaml.safe_dump(packet), encoding="utf-8")

    with pytest.raises(OrcaResidualLineagePacketError, match="privileged_features_allowed"):
        validate_launch_packet(bad_config)


def test_packet_requires_residual_bounds_to_match_candidate(tmp_path: Path) -> None:
    """The training packet bounds must stay aligned with the runtime candidate surface."""
    packet = _load_packet()
    packet["residual_bounds"]["linear_delta"] = 0.5
    bad_config = tmp_path / "bad_bound.yaml"
    bad_config.write_text(yaml.safe_dump(packet), encoding="utf-8")

    with pytest.raises(OrcaResidualLineagePacketError, match="linear_delta"):
        validate_launch_packet(bad_config)


def test_packet_rejects_worktree_local_output_artifact(tmp_path: Path) -> None:
    """Durable lineage must not depend on worktree-local output paths."""
    packet = _load_packet()
    packet["expected_outputs"]["checkpoint_pointer"] = "output/tmp/checkpoint.zip"
    bad_config = tmp_path / "bad_output.yaml"
    bad_config.write_text(yaml.safe_dump(packet), encoding="utf-8")

    with pytest.raises(OrcaResidualLineagePacketError, match="worktree-local output"):
        validate_launch_packet(bad_config)


def test_packet_requires_guard_and_residual_diagnostics(tmp_path: Path) -> None:
    """Contribution, clipping, guard-veto, and degraded-status diagnostics are mandatory."""
    packet = _load_packet()
    packet["diagnostics"]["required_fields"].remove("guard_veto_rate")
    bad_config = tmp_path / "bad_diagnostics.yaml"
    bad_config.write_text(yaml.safe_dump(packet), encoding="utf-8")

    with pytest.raises(OrcaResidualLineagePacketError, match="guard_veto_rate"):
        validate_launch_packet(bad_config)


def test_packet_keeps_slurm_execution_deferred(tmp_path: Path) -> None:
    """The local lineage packet must not claim to submit the bounded Slurm job."""
    packet = _load_packet()
    packet["execution_boundary"]["submit_slurm_from_this_issue"] = True
    bad_config = tmp_path / "bad_slurm.yaml"
    bad_config.write_text(yaml.safe_dump(packet), encoding="utf-8")

    with pytest.raises(OrcaResidualLineagePacketError, match="submit_slurm_from_this_issue"):
        validate_launch_packet(bad_config)


def _load_packet() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
