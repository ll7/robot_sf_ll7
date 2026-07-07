"""Tests for issue #4013 real-trajectory readiness reporting."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import yaml

from robot_sf.data_ingestion.real_trajectory_contract import (
    _staging_tree_sha256,
    build_staging_tree_report,
)
from scripts.training.check_issue_4013_real_trajectory_readiness import build_report
from tests.data.test_real_trajectory_contract import valid_manifest as _valid_manifest_fixture

if TYPE_CHECKING:
    from pathlib import Path


def _manifest() -> dict:
    manifest = _valid_manifest_fixture.__wrapped__()
    manifest["availability"] = "validated"
    manifest["benchmark_eligibility"] = "research_only"
    return manifest


def _write_manifest(tmp_path: Path, manifest: dict) -> Path:
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest_path


@pytest.mark.parametrize(
    ("staging_dir", "reason"),
    [
        ("${ROBOT_SF_EXTERNAL_DATA_ROOT}/synthetic_byo", "environment variable unresolved"),
        ("output/external_data/missing_synthetic_byo", "staging directory missing"),
    ],
)
def test_readiness_report_records_unavailable_staging_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    staging_dir: str | None,
    reason: str,
) -> None:
    """The issue #4013 readiness report exposes staging-tree blockers."""
    monkeypatch.delenv("ROBOT_SF_EXTERNAL_DATA_ROOT", raising=False)
    manifest = _manifest()
    manifest["staging"]["staging_dir"] = staging_dir

    report = build_report(_write_manifest(tmp_path, manifest))

    assert report["staging_tree"]["available"] is False
    assert report["staging_tree"]["reason"] == reason


def test_readiness_report_records_available_staging_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A checksum-pinned local tree is surfaced in the readiness report."""
    data_root = tmp_path / "external_data"
    staging_dir = data_root / "synthetic_byo"
    staging_dir.mkdir(parents=True)
    (staging_dir / "trajectories.csv").write_text("frame,ped_id,x,y\n0,1,0,0\n", encoding="utf-8")
    monkeypatch.setenv("ROBOT_SF_EXTERNAL_DATA_ROOT", str(data_root))
    manifest = _manifest()
    manifest["staging"]["staging_dir"] = "${ROBOT_SF_EXTERNAL_DATA_ROOT}/synthetic_byo"
    tree_sha256 = _staging_tree_sha256(staging_dir)
    manifest["checksums"]["tree_sha256"] = tree_sha256
    manifest["checksums"]["expected_tree_sha256"] = tree_sha256

    report = build_report(_write_manifest(tmp_path, manifest))

    assert report["status"] == "ready_for_real_trajectory_training"
    assert report["staging_tree"] == {
        "available": True,
        "staging_dir": str(staging_dir),
        "file_count": 1,
        "tree_sha256": tree_sha256,
    }


def test_staging_tree_helper_reports_schema_invalid_missing_staging_dir() -> None:
    """The shared helper can still describe a schema-invalid missing staging field."""
    manifest = _manifest()
    manifest["staging"].pop("staging_dir")

    assert build_staging_tree_report(manifest) == {
        "available": False,
        "reason": "manifest.staging.staging_dir missing",
    }


def test_staging_tree_helper_reports_empty_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The shared helper reports an empty local staging tree as unavailable."""
    data_root = tmp_path / "external_data"
    staging_dir = data_root / "synthetic_byo"
    staging_dir.mkdir(parents=True)
    monkeypatch.setenv("ROBOT_SF_EXTERNAL_DATA_ROOT", str(data_root))
    manifest = _manifest()
    manifest["staging"]["staging_dir"] = "${ROBOT_SF_EXTERNAL_DATA_ROOT}/synthetic_byo"

    assert build_staging_tree_report(manifest) == {
        "available": False,
        "staging_dir": str(staging_dir),
        "file_count": 0,
        "reason": "staging directory empty",
    }
