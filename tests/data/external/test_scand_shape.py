"""Skip-if-absent shape-contract tests for SCAND external data.

These tests never require the terms-gated SCAND bytes. Exactly one test path
depends on locally staged real data and skips when it is absent; every other test
builds a synthetic recording-style layout under ``tmp_path``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from robot_sf.data.external import scand

if TYPE_CHECKING:
    from pathlib import Path


def _stage_minimal_dataset(root: Path) -> None:
    """Create a tiny documented SCAND layout satisfying the registry groups.

    Writes one non-empty ROS bag demonstration, one rectangular CSV export, one
    valid JSON metadata file, and a local terms/README copy. A nested demo
    subdirectory exercises recursive discovery.
    """

    demo_dir = root / "demonstrations"
    demo_dir.mkdir(parents=True, exist_ok=True)
    (demo_dir / "spot_demo.bag").write_bytes(b"#ROSBAG V2.0\n\x00\x01")
    (root / "commands.csv").write_text("t,v,omega\n0.0,0.5,0.0\n0.1,0.5,0.1\n", encoding="utf-8")
    (root / "meta.json").write_text('{"platform": "spot", "hours": 8.7}', encoding="utf-8")
    (root / "LICENSE.txt").write_text("Local copy of SCAND dataset terms.\n", encoding="utf-8")


def test_scand_absent_data_skips(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """External clones without staged SCAND bytes skip instead of failing."""

    monkeypatch.setenv(scand.EXTERNAL_DATA_ROOT_ENV, str(tmp_path))
    if not scand.is_available():
        pytest.skip("external dataset not staged")
    pytest.fail("temporary empty external-data root unexpectedly satisfied SCAND contract")


def test_scand_shape_contract_with_synthetic_layout(tmp_path: Path) -> None:
    """A complete synthetic layout resolves and produces recording shape metadata."""

    root = tmp_path / "scand_demos"
    _stage_minimal_dataset(root)

    assert scand.is_available(root)
    paths = scand.require_available(root)
    assert len(paths.bag_files) == 1
    assert len(paths.csv_files) == 1
    assert len(paths.json_files) == 1
    assert len(paths.license_or_readme) == 1

    contract = scand.load_shape_contract(root)
    assert contract["asset_id"] == "scand-demos"
    assert contract["docs_path"] == "docs/datasets/scand.md"
    assert contract["recording_counts"] == {"bag": 1, "csv": 1, "json": 1}
    assert contract["csv_files"]["commands.csv"] == {"row_count": 3, "column_count": 3}
    assert contract["json_files"]["meta.json"]["top_level_type"] == "dict"
    assert contract["bag_files"]["demonstrations/spot_demo.bag"]["size_bytes"] > 0
    assert contract["license_or_readme"] == ["LICENSE.txt"]


def test_scand_missing_license_is_unavailable(tmp_path: Path) -> None:
    """A layout without a license/readme copy is unavailable and raises with docs pointer."""

    root = tmp_path / "scand_demos"
    _stage_minimal_dataset(root)
    (root / "LICENSE.txt").unlink()

    assert not scand.is_available(root)
    with pytest.raises(scand.ScandDataError, match="docs/datasets/scand.md"):
        scand.require_available(root)


def test_scand_missing_recording_is_unavailable(tmp_path: Path) -> None:
    """A layout without any recording/export file is unavailable."""

    root = tmp_path / "scand_demos"
    root.mkdir(parents=True)
    (root / "LICENSE.txt").write_text("terms only\n", encoding="utf-8")

    assert not scand.is_available(root)
    with pytest.raises(scand.ScandDataError):
        scand.require_available(root)


def test_scand_absent_root_error_names_docs(tmp_path: Path) -> None:
    """An absent dataset root fails closed with an actionable docs pointer."""

    root = tmp_path / "scand_demos"
    assert not scand.is_available(root)
    with pytest.raises(scand.ScandDataError, match="docs/datasets/scand.md"):
        scand.require_available(root)


def test_scand_empty_csv_fails_closed(tmp_path: Path) -> None:
    """An empty staged CSV export fails closed with an actionable error."""

    root = tmp_path / "scand_demos"
    _stage_minimal_dataset(root)
    (root / "commands.csv").write_text("", encoding="utf-8")

    with pytest.raises(scand.ScandDataError, match="no non-empty data rows"):
        scand.load_shape_contract(root)


def test_scand_ragged_csv_fails_closed(tmp_path: Path) -> None:
    """A non-rectangular CSV export fails closed with an actionable error."""

    root = tmp_path / "scand_demos"
    _stage_minimal_dataset(root)
    (root / "commands.csv").write_text("t,v,omega\n0.0,0.5\n", encoding="utf-8")

    with pytest.raises(scand.ScandDataError, match="not rectangular"):
        scand.load_shape_contract(root)


def test_scand_malformed_json_fails_closed(tmp_path: Path) -> None:
    """A malformed JSON export fails closed with an actionable error."""

    root = tmp_path / "scand_demos"
    _stage_minimal_dataset(root)
    (root / "meta.json").write_text("{not valid json", encoding="utf-8")

    with pytest.raises(scand.ScandDataError, match="not valid JSON"):
        scand.load_shape_contract(root)


def test_scand_empty_bag_fails_closed(tmp_path: Path) -> None:
    """A zero-byte ROS bag fails closed with an actionable error."""

    root = tmp_path / "scand_demos"
    _stage_minimal_dataset(root)
    (root / "demonstrations" / "spot_demo.bag").write_bytes(b"")

    with pytest.raises(scand.ScandDataError, match="empty"):
        scand.load_shape_contract(root)
