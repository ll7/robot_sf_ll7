"""Fast unit tests for Full Classic skipped video artifact construction."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.full_classic import visuals
from robot_sf.benchmark.full_classic.visual_constants import (
    NOTE_DISABLED,
    NOTE_SMOKE_MODE,
    RENDERER_SIM_VIEW,
    RENDERER_SYNTHETIC,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("cfg", "expected_note"),
    [
        (SimpleNamespace(smoke=False, disable_videos=True, video_renderer="auto"), NOTE_DISABLED),
        (SimpleNamespace(smoke=True, disable_videos=False, video_renderer="auto"), NOTE_SMOKE_MODE),
    ],
    ids=["disabled", "smoke"],
)
def test_build_video_artifacts_skip_branches(
    tmp_path: Path, cfg: SimpleNamespace, expected_note: str
) -> None:
    """Both skip branches emit one deterministic artifact per selected record."""
    videos_dir = tmp_path / "videos"
    records = [{"episode_id": "episode-1", "scenario_id": "scenario-1"}]

    artifacts = visuals._build_video_artifacts(cfg, records, videos_dir, {})

    assert artifacts
    expected_renderer = RENDERER_SIM_VIEW if visuals._SIM_VIEW_AVAILABLE else RENDERER_SYNTHETIC
    for artifact in artifacts:
        assert artifact.status == "skipped"
        assert artifact.note == expected_note
        assert artifact.artifact_id == f"video_{artifact.episode_id}"
        assert artifact.scenario_id
        assert artifact.episode_id
        assert artifact.renderer == expected_renderer
        assert artifact.path_mp4 == str(videos_dir / f"{artifact.episode_id}.mp4")
    assert not list(videos_dir.rglob("*.mp4"))
