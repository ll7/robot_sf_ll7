"""Synthetic contract tests for the fail-closed SDD trajectory parser."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from robot_sf.data.external.sdd_trajectories import (
    SddTrajectoryDataError,
    SddTrajectoryTrack,
    load_sdd_track_set,
)

if TYPE_CHECKING:
    from pathlib import Path


_VALID_ROWS = """
2 0 10 2 14 3 0 1 0 Pedestrian
1 10 20 14 24 2 0 0 1 "Pedestrian"
2 0 0 2 2 1 0 0 1 "Pedestrian"
1 10 10 14 14 1 0 1 0 Pedestrian
2 0 20 2 24 2 1 1 1 Pedestrian
9 0 0 2 2 1 0 0 0 Car
9 0 2 2 4 2 0 0 0 Car
"""


def _write_annotations(tmp_path: Path, text: str = _VALID_ROWS) -> Path:
    path = tmp_path / "annotations.txt"
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def _load(path: Path):
    return load_sdd_track_set(
        path,
        scene="synthetic-campus",
        split="train",
        frame_rate_hz=2.0,
        meters_per_pixel=0.5,
        image_height_px=100.0,
    )


def test_parser_accepts_quoted_and_unquoted_labels_and_converts_centers(tmp_path: Path) -> None:
    """Valid rows are sorted, filtered, and converted with explicit parameters."""

    track_set = _load(_write_annotations(tmp_path))

    assert track_set.schema_version == "sdd_trajectory.v1"
    assert track_set.scene_split == "synthetic-campus/train"
    assert [track.track_id for track in track_set.tracks] == [1, 2]
    track_one, track_two = track_set.tracks
    assert np.allclose(track_one.time_s, [0.5, 1.0])
    assert np.allclose(track_one.positions, [[6.0, 44.0], [6.0, 39.0]])
    assert np.allclose(track_two.time_s, [0.5, 1.5])
    assert np.allclose(track_two.positions, [[0.5, 49.5], [0.5, 44.0]])
    assert track_one.occluded_count == 1
    assert track_one.generated_count == 1
    assert track_two.occluded_count == 1
    assert track_two.generated_count == 1
    assert track_one.metadata["source_track_id"] == 1
    assert all(track.time_s.flags.writeable is False for track in track_set.tracks)
    assert all(track.positions.flags.writeable is False for track in track_set.tracks)


def test_record_arrays_are_copied_and_immutable() -> None:
    """Frozen records also protect numpy payloads from caller mutation."""

    times = np.asarray([0.0, 1.0])
    positions = np.asarray([[1.0, 2.0], [2.0, 3.0]])
    track = SddTrajectoryTrack(1, times, positions, 0, 0, "Pedestrian")
    times[0] = 99.0
    positions[0, 0] = 99.0

    assert track.time_s[0] == 0.0
    assert track.positions[0, 0] == 1.0
    with pytest.raises(ValueError):
        track.positions[0, 0] = 7.0


@pytest.mark.parametrize(
    ("value_name", "value"),
    [
        ("frame_rate_hz", 0.0),
        ("frame_rate_hz", -1.0),
        ("frame_rate_hz", float("nan")),
        ("frame_rate_hz", True),
        ("meters_per_pixel", 0.0),
        ("meters_per_pixel", -1.0),
        ("meters_per_pixel", float("inf")),
        ("meters_per_pixel", False),
    ],
)
def test_loader_rejects_invalid_explicit_conversion_parameters(
    tmp_path: Path, value_name: str, value: object
) -> None:
    """Rate and scale cannot be inferred or silently coerced from invalid values."""

    kwargs = {
        "scene": "scene",
        "split": "split",
        "frame_rate_hz": 2.0,
        "meters_per_pixel": 0.5,
    }
    kwargs[value_name] = value
    with pytest.raises(ValueError, match=value_name):
        load_sdd_track_set(_write_annotations(tmp_path), **kwargs)


@pytest.mark.parametrize(
    "text",
    [
        "1 0 0 1 1 0 0 0 0",
        "1 nope 0 1 1 0 0 0 0 Pedestrian",
        "1 nan 0 1 1 0 0 0 0 Pedestrian\n1 0 0 1 1 1 0 0 0 Pedestrian",
        "0 0 0 1 1 0 0 0 0 Pedestrian\n0 0 0 1 1 1 0 0 0 Pedestrian",
        "1 0 0 1 1 -1 0 0 0 Pedestrian\n1 0 0 1 1 1 0 0 0 Pedestrian",
    ],
)
def test_parser_rejects_malformed_nonfinite_and_invalid_identities(
    tmp_path: Path, text: str
) -> None:
    """Malformed fields, coordinates, and identities fail closed."""

    with pytest.raises((SddTrajectoryDataError, ValueError)):
        _load(_write_annotations(tmp_path, text))


def test_parser_rejects_duplicate_conflicting_frames(tmp_path: Path) -> None:
    """One track/frame identity cannot resolve to two different positions."""

    text = """
    1 0 0 2 2 1 0 0 0 Pedestrian
    1 10 0 12 2 1 0 0 0 Pedestrian
    1 0 0 2 2 2 0 0 0 Pedestrian
    """
    with pytest.raises(SddTrajectoryDataError, match="duplicate"):
        _load(_write_annotations(tmp_path, text))


@pytest.mark.parametrize(
    "text",
    [
        "1 0 0 1 1 1 0 0 0 Car\n1 0 0 1 1 2 0 0 0 Car",
        "1 0 0 1 1 1 1 0 0 Pedestrian\n1 0 0 1 1 2 1 0 0 Pedestrian",
        "1 0 0 1 1 1 0 0 0 Pedestrian",
    ],
)
def test_parser_rejects_empty_or_short_usable_input(tmp_path: Path, text: str) -> None:
    """Filtering never turns absent or one-point data into a valid track set."""

    with pytest.raises(SddTrajectoryDataError, match="usable|fewer"):
        _load(_write_annotations(tmp_path, text))


def test_parser_rejects_missing_file(tmp_path: Path) -> None:
    """A missing explicit annotation path is an input error, not valid data."""

    with pytest.raises(SddTrajectoryDataError, match="unavailable"):
        _load(tmp_path / "missing-annotations.txt")


def test_custom_accepted_labels_are_explicit(tmp_path: Path) -> None:
    """Non-default semantic labels can be selected without heuristic inference."""

    text = """
    4 0 0 2 2 0 0 0 0 Cyclist
    4 2 0 4 2 1 0 0 0 Cyclist
    5 0 0 2 2 0 0 0 0 Pedestrian
    5 2 0 4 2 1 0 0 0 Pedestrian
    """
    track_set = load_sdd_track_set(
        _write_annotations(tmp_path, text),
        scene="scene",
        split="split",
        frame_rate_hz=1.0,
        meters_per_pixel=1.0,
        accepted_labels=("Cyclist",),
    )

    assert [track.track_id for track in track_set.tracks] == [4]
    assert track_set.tracks[0].label == "Cyclist"
