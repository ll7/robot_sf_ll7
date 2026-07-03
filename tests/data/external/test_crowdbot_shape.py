"""Skip-if-absent shape-contract tests for CrowdBot external data.

These tests never require the research-access-gated CrowdBot bytes. Exactly one
test path depends on locally staged real data and skips when it is absent; every
other test builds a synthetic recording-style layout under ``tmp_path``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from robot_sf.data.external import crowdbot

if TYPE_CHECKING:
    from pathlib import Path


def _stage_minimal_dataset(root: Path) -> None:
    """Create a tiny documented CrowdBot layout satisfying the registry groups.

    Writes one non-empty ROS bag, one rectangular CSV export, one valid JSON
    metadata file, and a local terms/README copy.
    """

    root.mkdir(parents=True, exist_ok=True)
    (root / "qolo_run.bag").write_bytes(b"#ROSBAG V2.0\n\x00\x01")
    (root / "tracks.csv").write_text("t,id,x,y\n0.0,1,2.0,3.0\n0.1,1,2.1,3.1\n", encoding="utf-8")
    (root / "meta.json").write_text('{"robot": "qolo", "site": "lausanne"}', encoding="utf-8")
    (root / "README.md").write_text(
        "Local copy of CrowdBot research-use terms.\n", encoding="utf-8"
    )


def test_crowdbot_absent_data_skips(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """External clones without staged CrowdBot bytes skip instead of failing."""

    monkeypatch.setenv(crowdbot.EXTERNAL_DATA_ROOT_ENV, str(tmp_path))
    if not crowdbot.is_available():
        pytest.skip("external dataset not staged")
    pytest.fail("temporary empty external-data root unexpectedly satisfied CrowdBot contract")


def test_crowdbot_shape_contract_with_synthetic_layout(tmp_path: Path) -> None:
    """A complete synthetic layout resolves and produces recording shape metadata."""

    root = tmp_path / "crowdbot"
    _stage_minimal_dataset(root)

    assert crowdbot.is_available(root)
    paths = crowdbot.require_available(root)
    assert len(paths.bag_files) == 1
    assert len(paths.csv_files) == 1
    assert len(paths.json_files) == 1
    assert len(paths.license_or_readme) == 1

    contract = crowdbot.load_shape_contract(root)
    assert contract["asset_id"] == "crowdbot"
    assert contract["docs_path"] == "docs/datasets/crowdbot.md"
    assert contract["recording_counts"] == {"bag": 1, "csv": 1, "json": 1}
    assert contract["csv_files"]["tracks.csv"] == {"row_count": 3, "column_count": 4}
    assert contract["json_files"]["meta.json"]["top_level_type"] == "dict"
    assert contract["bag_files"]["qolo_run.bag"]["size_bytes"] > 0
    assert contract["license_or_readme"] == ["README.md"]


def test_crowdbot_missing_license_is_unavailable(tmp_path: Path) -> None:
    """A layout without a license/readme copy is unavailable and raises with docs pointer."""

    root = tmp_path / "crowdbot"
    _stage_minimal_dataset(root)
    (root / "README.md").unlink()

    assert not crowdbot.is_available(root)
    with pytest.raises(crowdbot.CrowdBotDataError, match="docs/datasets/crowdbot.md"):
        crowdbot.require_available(root)


def test_crowdbot_missing_recording_is_unavailable(tmp_path: Path) -> None:
    """A layout without any recording/export file is unavailable."""

    root = tmp_path / "crowdbot"
    root.mkdir(parents=True)
    (root / "README.md").write_text("terms only\n", encoding="utf-8")

    assert not crowdbot.is_available(root)
    with pytest.raises(crowdbot.CrowdBotDataError):
        crowdbot.require_available(root)


def test_crowdbot_absent_root_error_names_docs(tmp_path: Path) -> None:
    """An absent dataset root fails closed with an actionable docs pointer."""

    root = tmp_path / "crowdbot"
    assert not crowdbot.is_available(root)
    with pytest.raises(crowdbot.CrowdBotDataError, match="docs/datasets/crowdbot.md"):
        crowdbot.require_available(root)


def test_crowdbot_empty_csv_fails_closed(tmp_path: Path) -> None:
    """An empty staged CSV export fails closed with an actionable error."""

    root = tmp_path / "crowdbot"
    _stage_minimal_dataset(root)
    (root / "tracks.csv").write_text("", encoding="utf-8")

    with pytest.raises(crowdbot.CrowdBotDataError, match="no non-empty data rows"):
        crowdbot.load_shape_contract(root)


def test_crowdbot_ragged_csv_fails_closed(tmp_path: Path) -> None:
    """A non-rectangular CSV export fails closed with an actionable error."""

    root = tmp_path / "crowdbot"
    _stage_minimal_dataset(root)
    (root / "tracks.csv").write_text("t,id,x,y\n0.0,1,2.0\n", encoding="utf-8")

    with pytest.raises(crowdbot.CrowdBotDataError, match="not rectangular"):
        crowdbot.load_shape_contract(root)


def test_crowdbot_malformed_json_fails_closed(tmp_path: Path) -> None:
    """A malformed JSON export fails closed with an actionable error."""

    root = tmp_path / "crowdbot"
    _stage_minimal_dataset(root)
    (root / "meta.json").write_text("{not valid json", encoding="utf-8")

    with pytest.raises(crowdbot.CrowdBotDataError, match="not valid JSON"):
        crowdbot.load_shape_contract(root)


def test_crowdbot_empty_bag_fails_closed(tmp_path: Path) -> None:
    """A zero-byte ROS bag fails closed with an actionable error."""

    root = tmp_path / "crowdbot"
    _stage_minimal_dataset(root)
    (root / "qolo_run.bag").write_bytes(b"")

    with pytest.raises(crowdbot.CrowdBotDataError, match="empty"):
        crowdbot.load_shape_contract(root)
