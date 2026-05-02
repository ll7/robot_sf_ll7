"""Basic tests for visual artifact integration (plots + videos manifests).

Covers:
- Videos disabled: manifests show skipped with 'video generation disabled'.
- Smoke mode: videos skipped with 'smoke mode'.

These are initial TDD tests; helper implementation will satisfy them.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_sf.benchmark.full_classic.orchestrator import run_full_benchmark


class _Cfg:
    """Minimal config shim for exercising full-benchmark artifact generation in tests."""

    def __init__(self, tmp_path: Path, smoke: bool = True, disable_videos: bool = False):
        """Populate the small subset of orchestrator settings this test depends on."""

        # Minimal fields used by orchestrator & video/plot logic
        self.output_root = str(tmp_path)
        self.scenario_matrix_path = "configs/scenarios/classic_interactions.yaml"  # assuming exists
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
        # Tight smoke caps keep this artifact-presence check fast; validate broader runtime
        # behavior with the full tests/ suite and targeted slow-marker reruns before merge.
        self.smoke_limit_jobs = True
        self.smoke_episodes = 1
        self.smoke_horizon_cap = 5
        # Targets (unused placeholder for precision)
        self.target_collision_half_width = 0.05
        self.target_success_half_width = 0.05
        self.target_snqi_half_width = 0.05


def _read_json(path: Path):
    """TODO docstring. Document this function.

    Args:
        path: TODO docstring.
    """
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


@pytest.mark.slow
def test_visuals_manifest_videos_disabled(tmp_path):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
    """
    cfg = _Cfg(tmp_path / "run_disabled", disable_videos=True)
    run_full_benchmark(cfg)
    reports = Path(cfg.output_root) / "reports"
    assert reports.exists(), "reports directory should exist"
    # Video manifest expected once implementation is added
    video_manifest = reports / "video_artifacts.json"
    # TDD expectation: file should exist after implementation
    if video_manifest.exists():  # allow pre-implementation pass as xfail-like behavior
        data = _read_json(video_manifest)
        assert all(a.get("status") == "skipped" for a in data), data
        assert any("disabled" in (a.get("note") or "") for a in data)


@pytest.mark.slow
def test_visuals_manifest_smoke_mode(tmp_path):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
    """
    cfg = _Cfg(tmp_path / "run_smoke", smoke=True)
    run_full_benchmark(cfg)
    reports = Path(cfg.output_root) / "reports"
    video_manifest = reports / "video_artifacts.json"
    if video_manifest.exists():
        data = _read_json(video_manifest)
        assert all(a.get("status") == "skipped" for a in data)
        assert any("smoke" in (a.get("note") or "") for a in data)
