"""Contract tests for skipped Full Classic video artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.benchmark.full_classic import visuals
from robot_sf.benchmark.full_classic.orchestrator import run_full_benchmark
from robot_sf.benchmark.full_classic.visual_constants import (
    NOTE_DISABLED,
    NOTE_SMOKE_MODE,
    RENDERER_SIM_VIEW,
    RENDERER_SYNTHETIC,
)


class _Cfg:
    """Minimal config shim for exercising Full Classic artifact generation."""

    def __init__(self, tmp_path: Path, *, smoke: bool = False, disable_videos: bool = False):
        """Populate the subset of orchestrator settings used by these tests."""

        self.output_root = str(tmp_path)
        self.scenario_matrix_path = "configs/scenarios/classic_interactions.yaml"
        self.initial_episodes = 1
        self.max_episodes = 1
        self.batch_size = 1
        self.algo = "ppo"
        self.workers = 1
        self.master_seed = 123
        self.smoke = smoke
        self.disable_videos = disable_videos
        self.max_videos = 1
        self.capture_replay = True
        self.smoke_limit_jobs = True
        self.smoke_episodes = 1
        self.smoke_horizon_cap = 5
        self.target_collision_half_width = 0.05
        self.target_success_half_width = 0.05
        self.target_snqi_half_width = 0.05


def _read_json(path: Path) -> object:
    """Read one JSON manifest from disk."""
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_skipped_manifest(data: object, cfg: _Cfg, expected_note: str) -> None:
    """Validate a serialized skip manifest and its planned output paths."""
    assert isinstance(data, list)
    assert data, "the integration path must write a non-empty video manifest"
    expected_renderer = RENDERER_SIM_VIEW if visuals._SIM_VIEW_AVAILABLE else RENDERER_SYNTHETIC
    videos_dir = Path(cfg.output_root) / "videos"
    for artifact in data:
        assert artifact["status"] == "skipped"
        assert artifact["note"] == expected_note
        assert artifact["artifact_id"] == f"video_{artifact['episode_id']}"
        assert artifact["scenario_id"]
        assert artifact["episode_id"]
        assert artifact["renderer"] == expected_renderer
        assert artifact["path_mp4"] == str(videos_dir / f"{artifact['episode_id']}.mp4")


@pytest.mark.slow
def test_visuals_manifest_videos_disabled(tmp_path: Path) -> None:
    """Explicitly disabled videos still produce a truthful non-evidence manifest."""
    cfg = _Cfg(tmp_path / "run_disabled", smoke=True, disable_videos=True)
    run_full_benchmark(cfg)
    reports = Path(cfg.output_root) / "reports"
    video_manifest = reports / "video_artifacts.json"

    assert reports.exists()
    assert video_manifest.exists()
    _assert_skipped_manifest(_read_json(video_manifest), cfg, NOTE_DISABLED)
    assert not list((Path(cfg.output_root) / "videos").rglob("*.mp4"))


@pytest.mark.slow
def test_visuals_manifest_smoke_mode(tmp_path: Path) -> None:
    """Smoke mode still produces a truthful non-evidence manifest."""
    cfg = _Cfg(tmp_path / "run_smoke", smoke=True)
    run_full_benchmark(cfg)
    reports = Path(cfg.output_root) / "reports"
    video_manifest = reports / "video_artifacts.json"

    assert reports.exists()
    assert video_manifest.exists()
    _assert_skipped_manifest(_read_json(video_manifest), cfg, NOTE_SMOKE_MODE)
    assert not list((Path(cfg.output_root) / "videos").rglob("*.mp4"))
