"""Tests for the issue #4013 real-trajectory readiness checker."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

import yaml

from scripts.training.check_issue_4013_real_trajectory_readiness import (
    build_report,
    render_markdown,
)

if TYPE_CHECKING:
    from pathlib import Path


def _manifest() -> dict:
    return {
        "schema": "robot_sf_real_trajectory_ingestion_manifest.v1",
        "dataset_id": "issue_4013_test",
        "title": "Issue 4013 test trajectories",
        "source": {
            "url": "https://example.org",
            "version": "test",
            "citation": "Test citation",
            "access_date": "2026-07-06",
        },
        "license": {
            "name": "test",
            "url": None,
            "posture": "bring-your-own",
            "supplier_acknowledgment": True,
            "redistribution": False,
        },
        "retrieval": {
            "instructions": "Stage locally.",
            "download_url": None,
            "fail_closed": True,
        },
        "checksums": {
            "algorithm": "SHA-256",
            "tree_sha256": None,
            "expected_tree_sha256": None,
        },
        "conversion": {
            "frame_rate_hz": 2.5,
            "length_unit": "meters",
            "coordinate_frame": "world_meters_xy",
            "timestamp_field": "frame",
            "agent_id_field": "pedestrian_id",
            "position_fields": ["x", "y"],
            "map_context_field": "scene",
            "missing_data_behavior": "fail_closed",
        },
        "splits": {"naming": "scene", "members": ["eth"]},
        "staging": {
            "staging_dir": "${ROBOT_SF_EXTERNAL_DATA_ROOT}/issue_4013_test",
            "local_only_raw": True,
            "durable_storage_target": "local-only-byo",
        },
        "privacy": {"pii_reviewed": True, "notes": "Test manifest only."},
        "availability": "missing",
        "benchmark_eligibility": "diagnostic_only",
        "related_issues": [4013],
    }


def _write_manifest(tmp_path: Path, manifest: dict) -> Path:
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


def test_missing_real_dataset_blocks_phase3_without_contract_error(tmp_path: Path) -> None:
    """Missing validated real data blocks Phase 3 without failing manifest structure."""
    report = build_report(_write_manifest(tmp_path, _manifest()))

    assert report["status"] == "blocked_real_trajectory_data_unavailable"
    assert report["blockers"][0]["code"] == "real_trajectory.availability_not_validated"
    assert report["preflight_issues"] == []


def test_validated_research_manifest_is_ready_for_real_trajectory_training(tmp_path: Path) -> None:
    """Checksum-validated research-eligible manifests unlock real-trajectory training."""
    manifest = deepcopy(_manifest())
    manifest["availability"] = "validated"
    manifest["benchmark_eligibility"] = "research_only"
    manifest["checksums"]["tree_sha256"] = "a" * 64
    manifest["checksums"]["expected_tree_sha256"] = "a" * 64

    report = build_report(_write_manifest(tmp_path, manifest))

    assert report["status"] == "ready_for_real_trajectory_training"
    assert report["blockers"] == []


def test_manifest_contract_error_blocks_before_training(tmp_path: Path) -> None:
    """Manifest contract violations block before any training/evaluation can run."""
    manifest = _manifest()
    manifest["license"]["supplier_acknowledgment"] = False

    report = build_report(_write_manifest(tmp_path, manifest))

    assert report["status"] == "blocked_manifest_contract"
    assert report["blockers"][0]["code"] == "license.acknowledgment_missing"


def test_markdown_contains_acceptance_evidence_and_claim_boundary(tmp_path: Path) -> None:
    """Markdown report includes the closure-audit evidence and claim boundary."""
    report = build_report(_write_manifest(tmp_path, _manifest()))
    markdown = render_markdown(report)

    assert "Issue #4013 Real-Trajectory Readiness" in markdown
    assert "No raw data is staged" in markdown
    assert "comparison against cv_prediction_mpc" in markdown
