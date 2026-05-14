"""Tests for video contact-sheet generation from episode frame artifacts."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from PIL import Image

from scripts.generate_video_contact_sheet import generate_contact_sheet

if TYPE_CHECKING:
    from pathlib import Path


def _write_rgb(path: Path, color: tuple[int, int, int]) -> None:
    """Write a tiny RGB fixture image."""
    Image.new("RGB", (8, 6), color=color).save(path)


def test_generate_contact_sheet_from_episode_frame_paths(tmp_path: Path) -> None:
    """Verify JSONL frame-path metadata can produce a deterministic sheet artifact."""
    frame_a = tmp_path / "episode_a_start.png"
    frame_b = tmp_path / "episode_a_mid.png"
    frame_c = tmp_path / "episode_b_start.png"
    _write_rgb(frame_a, (255, 0, 0))
    _write_rgb(frame_b, (0, 255, 0))
    _write_rgb(frame_c, (0, 0, 255))

    episodes_jsonl = tmp_path / "episodes.jsonl"
    rows = [
        {
            "episode_id": "episode-a",
            "scenario_id": "crossing",
            "seed": 1,
            "frame_paths": [str(frame_a), str(frame_b)],
        },
        {
            "episode_id": "episode-b",
            "scenario_id": "head_on",
            "seed": 2,
            "video": {"frame_paths": [str(frame_c)]},
        },
    ]
    episodes_jsonl.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "contact_sheet.png"

    result = generate_contact_sheet(episodes_jsonl, output_path)

    assert result == output_path
    assert output_path.exists()
    with Image.open(output_path) as sheet:
        assert sheet.size == (24, 6)
        assert sheet.getpixel((1, 1)) == (255, 0, 0)
        assert sheet.getpixel((9, 1)) == (0, 255, 0)
        assert sheet.getpixel((17, 1)) == (0, 0, 255)


def test_generate_contact_sheet_fails_when_no_frame_sources(tmp_path: Path) -> None:
    """Verify missing frame artifacts fail clearly instead of writing an empty gallery."""
    episodes_jsonl = tmp_path / "episodes.jsonl"
    episodes_jsonl.write_text(
        json.dumps({"episode_id": "episode-a", "video": {"path": "episode-a.mp4"}}) + "\n",
        encoding="utf-8",
    )

    output_path = tmp_path / "contact_sheet.png"

    try:
        generate_contact_sheet(episodes_jsonl, output_path)
    except ValueError as exc:
        assert "No frame image paths" in str(exc)
    else:  # pragma: no cover - makes the assertion message clearer on failure
        raise AssertionError("generate_contact_sheet should require frame image paths")


def test_generate_contact_sheet_reports_invalid_json_line_numbers(tmp_path: Path) -> None:
    """Malformed JSONL rows should fail with their line number for easier debugging."""
    episodes_jsonl = tmp_path / "episodes.jsonl"
    episodes_jsonl.write_text("{bad json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"Episode row 1 is not valid JSON"):
        generate_contact_sheet(episodes_jsonl, tmp_path / "contact_sheet.png")


def test_generate_contact_sheet_rejects_directory_frame_paths(tmp_path: Path) -> None:
    """Directory-valued frame paths should fail closed instead of reaching Pillow open calls."""
    frame_dir = tmp_path / "frames"
    frame_dir.mkdir()
    episodes_jsonl = tmp_path / "episodes.jsonl"
    episodes_jsonl.write_text(
        json.dumps({"episode_id": "episode-a", "frame_paths": [str(frame_dir)]}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="Frame image not found"):
        generate_contact_sheet(episodes_jsonl, tmp_path / "contact_sheet.png")
