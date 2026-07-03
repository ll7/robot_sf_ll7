"""Skip-if-absent shape-contract tests for inD external data.

These tests never require the request-gated inD bytes. Exactly one test path
depends on locally staged real data and skips when it is absent; every other test
builds a synthetic inD-shaped layout under ``tmp_path``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from robot_sf.data.external import ind

if TYPE_CHECKING:
    from pathlib import Path


def _write(path: Path, text: str) -> None:
    """Create parent directories and write a small text fixture file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


# Minimal well-formed inD per-recording CSV blocks. Column names match the stable
# published inD schema; only a defensible header subset is included so the fixture
# stays small while still carrying every required column.
_TRACKS_CSV = (
    "recordingId,trackId,frame,trackLifetime,xCenter,yCenter,heading\n"
    "0,1,0,1,12.5,-3.0,45.0\n"
    "0,1,1,2,12.6,-2.9,45.1\n"
    "0,2,0,1,5.0,8.0,90.0\n"
)
_TRACKS_META_CSV = (
    "recordingId,trackId,initialFrame,finalFrame,numFrames,width,length,class\n"
    "0,1,0,1,2,0.6,0.6,pedestrian\n"
    "0,2,0,0,1,1.8,4.5,car\n"
)
_RECORDING_META_CSV = (
    "recordingId,locationId,frameRate,speedLimit,weekday,numTracks\n0,1,25.0,13.89,Monday,2\n"
)


def _stage_recording(root: Path, recording_id: str = "00") -> None:
    """Create a tiny complete inD recording group under ``root``."""

    _write(root / f"{recording_id}_tracks.csv", _TRACKS_CSV)
    _write(root / f"{recording_id}_tracksMeta.csv", _TRACKS_META_CSV)
    _write(root / f"{recording_id}_recordingMeta.csv", _RECORDING_META_CSV)
    # A tiny non-empty PNG-named placeholder; the contract only checks presence.
    _write(root / f"{recording_id}_background.png", "PNG-placeholder")
    _write(root / "README.md", "Local copy of the upstream inD non-commercial terms.\n")


def test_ind_absent_data_skips(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """External clones without staged inD bytes skip instead of failing."""

    monkeypatch.setenv(ind.EXTERNAL_DATA_ROOT_ENV, str(tmp_path))
    if not ind.is_available():
        pytest.skip("external dataset not staged")
    pytest.fail("temporary empty external-data root unexpectedly satisfied inD contract")


def test_ind_shape_contract_with_synthetic_layout(tmp_path: Path) -> None:
    """A complete synthetic recording resolves and produces shape metadata."""

    root = tmp_path / "ind"
    _stage_recording(root)

    assert ind.is_available(root)
    dataset = ind.require_available(root)
    assert {rec.recording_id for rec in dataset.recordings} == {"00"}

    contract = ind.load_shape_contract(root)
    assert contract["asset_id"] == "ind-crossings"
    assert contract["docs_path"] == "docs/datasets/ind.md"

    recording = contract["recordings"]["00"]
    assert recording["tracks"]["row_count"] == 3
    assert recording["tracks"]["column_count"] == 7
    assert recording["tracks"]["path"] == "00_tracks.csv"
    assert recording["tracks_meta"]["row_count"] == 2
    assert recording["tracks_meta"]["column_count"] == 8
    assert recording["recording_meta"]["row_count"] == 1
    assert recording["background"]["path"] == "00_background.png"


def test_ind_resolves_multiple_recordings(tmp_path: Path) -> None:
    """Every complete recording group is resolved, including nested directories."""

    root = tmp_path / "ind"
    _stage_recording(root, "00")
    _stage_recording(root / "recordings" / "07", "07")

    contract = ind.load_shape_contract(root)
    assert set(contract["recordings"]) == {"00", "07"}
    assert contract["recordings"]["07"]["tracks"]["path"] == "recordings/07/07_tracks.csv"


def test_ind_background_fallback_png(tmp_path: Path) -> None:
    """A background staged without the ``_background`` suffix still resolves."""

    root = tmp_path / "ind"
    _stage_recording(root)
    (root / "00_background.png").unlink()
    _write(root / "00.png", "PNG-placeholder")

    assert ind.is_available(root)
    contract = ind.load_shape_contract(root)
    assert contract["recordings"]["00"]["background"]["path"] == "00.png"


def test_ind_missing_role_is_unavailable(tmp_path: Path) -> None:
    """A recording missing a CSV role is unavailable and raises with a pointer."""

    root = tmp_path / "ind"
    _stage_recording(root)
    (root / "00_tracksMeta.csv").unlink()

    assert not ind.is_available(root)
    with pytest.raises(ind.IndDataError, match="tracksMeta"):
        ind.require_available(root)


def test_ind_lone_tracks_csv_is_unavailable(tmp_path: Path) -> None:
    """A lone stray tracks CSV without its group does not satisfy the contract."""

    root = tmp_path / "ind"
    _write(root / "00_tracks.csv", _TRACKS_CSV)

    assert not ind.is_available(root)
    with pytest.raises(ind.IndDataError, match="docs/datasets/ind.md"):
        ind.load_shape_contract(root)


def test_ind_absent_root_raises(tmp_path: Path) -> None:
    """A non-existent dataset root raises an actionable acquisition error."""

    with pytest.raises(ind.IndDataError, match="is not staged"):
        ind.require_available(tmp_path / "missing")


def test_ind_empty_csv_fails_closed(tmp_path: Path) -> None:
    """An empty staged CSV fails closed with an actionable error."""

    root = tmp_path / "ind"
    _stage_recording(root)
    _write(root / "00_tracks.csv", "")

    with pytest.raises(ind.IndDataError, match="is empty"):
        ind.load_shape_contract(root)


def test_ind_header_only_csv_fails_closed(tmp_path: Path) -> None:
    """A CSV with a header but no data rows fails closed."""

    root = tmp_path / "ind"
    _stage_recording(root)
    _write(root / "00_tracks.csv", "recordingId,trackId,frame,xCenter,yCenter\n")

    with pytest.raises(ind.IndDataError, match="no data rows"):
        ind.load_shape_contract(root)


def test_ind_non_rectangular_csv_fails_closed(tmp_path: Path) -> None:
    """A non-rectangular CSV fails closed."""

    root = tmp_path / "ind"
    _stage_recording(root)
    _write(
        root / "00_tracks.csv",
        "recordingId,trackId,frame,xCenter,yCenter\n0,1,0,1.0,2.0\n0,1,1\n",
    )

    with pytest.raises(ind.IndDataError, match="not rectangular"):
        ind.load_shape_contract(root)


def test_ind_missing_expected_column_fails_closed(tmp_path: Path) -> None:
    """A CSV without inD's documented columns fails closed as not inD-shaped."""

    root = tmp_path / "ind"
    _stage_recording(root)
    _write(root / "00_tracks.csv", "a,b,c,d\n1,2,3,4\n")

    with pytest.raises(ind.IndDataError, match="missing expected column"):
        ind.load_shape_contract(root)


def test_ind_non_numeric_coordinate_fails_closed(tmp_path: Path) -> None:
    """A non-numeric coordinate value in the tracks table fails closed."""

    root = tmp_path / "ind"
    _stage_recording(root)
    _write(
        root / "00_tracks.csv",
        "recordingId,trackId,frame,xCenter,yCenter\n0,1,0,x,2.0\n",
    )

    with pytest.raises(ind.IndDataError, match="non-numeric"):
        ind.load_shape_contract(root)


def test_ind_non_finite_coordinate_fails_closed(tmp_path: Path) -> None:
    """A non-finite coordinate value in the tracks table fails closed."""

    root = tmp_path / "ind"
    _stage_recording(root)
    _write(
        root / "00_tracks.csv",
        "recordingId,trackId,frame,xCenter,yCenter\n0,1,0,inf,2.0\n",
    )

    with pytest.raises(ind.IndDataError, match="non-finite"):
        ind.load_shape_contract(root)
