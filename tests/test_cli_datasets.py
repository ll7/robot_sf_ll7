"""Tests for the ``robot-sf datasets`` list/verify/prepare UX (issue #5797)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from robot_sf import cli_datasets
from robot_sf.cli import main
from scripts.tools.manage_external_data import _tree_checksum

if TYPE_CHECKING:  # pragma: no cover - static typing only
    from pathlib import Path


def _stage_sdd_layout(root: Path) -> tuple[Path, Path]:
    """Stage a minimal SDD layout under a temp shared data root.

    Returns the staged annotations file and the asset root used.
    """
    asset_root = root / "sdd"
    asset_root.mkdir(parents=True)
    annotations = asset_root / "annotations.txt"
    annotations.write_text("frame rows\n", encoding="utf-8")
    return annotations, asset_root


def _write_provenance_manifest(manifest_dir: Path, asset_id: str, tree_sha256: str) -> Path:
    """Write a minimal provenance manifest pinning a tree checksum."""
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / f"{asset_id}.provenance.json"
    manifest_path.write_text(
        json.dumps({"asset_id": asset_id, "tree_sha256": tree_sha256}), encoding="utf-8"
    )
    return manifest_path


def test_datasets_list_includes_real_registry_assets(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``datasets list`` should surface the real external-data registry assets."""
    rc = main(["datasets", "list"])

    assert rc == 0
    rows = cli_datasets.list_datasets()
    asset_ids = {row["asset_id"] for row in rows}
    # The canonical restricted datasets must be represented.
    assert {"sdd", "eth-ucy"}.issubset(asset_ids)
    out = capsys.readouterr().out
    assert "eth-ucy" in out


def test_datasets_verify_passing_checksum(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``verify`` should PASS when a staged layout matches a pinned tree checksum."""
    shared_root = tmp_path / "shared"
    _stage_sdd_layout(shared_root)
    monkeypatch.setenv("ROBOT_SF_EXTERNAL_DATA_ROOT", str(shared_root))

    manifest_dir = tmp_path / "manifests"
    # Compute the canonical tree checksum for the staged files.
    report = cli_datasets.verify_datasets(asset_ids=["sdd"])
    # No manifest yet: available layout, no pinned checksum.
    assert report["results"][0]["layout_ok"] is True
    assert report["results"][0]["checksum_status"] == "no_pinned_checksum"

    staged_path = tmp_path / "shared" / "sdd"
    matched = sorted(staged_path.rglob("*"))
    matched = [p for p in matched if p.is_file()]
    checksum = _tree_checksum(staged_path, matched)
    _write_provenance_manifest(manifest_dir, "sdd", checksum["tree_sha256"])
    monkeypatch.setattr(cli_datasets, "DEFAULT_MANIFEST_DIR", manifest_dir)

    report = cli_datasets.verify_datasets(asset_ids=["sdd"])

    result = report["results"][0]
    assert result["layout_ok"] is True
    assert result["pinned_checksum"] is True
    assert result["checksum_status"] == "checksum_ok"
    assert report["ok"] is True


def test_datasets_verify_corrupted_checksum_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``verify`` should FAIL when a staged layout's recomputed checksum differs."""
    shared_root = tmp_path / "shared"
    _stage_sdd_layout(shared_root)
    monkeypatch.setenv("ROBOT_SF_EXTERNAL_DATA_ROOT", str(shared_root))

    manifest_dir = tmp_path / "manifests"
    # Pin a deliberately-wrong checksum.
    _write_provenance_manifest(manifest_dir, "sdd", "0" * 64)
    monkeypatch.setattr(cli_datasets, "DEFAULT_MANIFEST_DIR", manifest_dir)

    report = cli_datasets.verify_datasets(asset_ids=["sdd"])

    result = report["results"][0]
    assert result["layout_ok"] is True
    assert result["pinned_checksum"] is True
    assert result["checksum_status"] == "checksum_mismatch"
    assert result["expected_tree_sha256"] == "0" * 64
    assert result["observed_tree_sha256"] != "0" * 64
    assert report["ok"] is False


def test_datasets_prepare_restricted_prints_instructions_and_verifies_layout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A restricted dataset must print acquisition instructions and verify layout WITHOUT download."""
    # Stage a valid layout for a license-restricted dataset under a temp root.
    shared_root = tmp_path / "shared"
    _stage_sdd_layout(shared_root)
    monkeypatch.setenv("ROBOT_SF_EXTERNAL_DATA_ROOT", str(shared_root))

    rc = main(["datasets", "prepare", "sdd"])

    assert rc == 0
    out = capsys.readouterr().out
    # Exact acquisition instructions for the license-restricted asset.
    assert "https://cvgl.stanford.edu/projects/uav_data/" in out
    assert "annotations.txt" in out
    assert "manual/license-gated" in out
    # Local layout verified without downloading.
    assert "available" in out
    assert "Local verification (no download)" in out


def test_datasets_prepare_unknown_asset_errors(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An unknown asset id should raise a user-facing error, not a traceback."""
    from scripts.tools.manage_external_data import ExternalDataError

    with pytest.raises(ExternalDataError):
        cli_datasets.prepare_dataset("does-not-exist")
