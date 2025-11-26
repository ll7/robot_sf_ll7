"""Tests for video_renderer toggle (auto|synthetic|sim-view).

Focus:
 - synthetic: always produce synthetic renderer artifacts (no reclassification)
 - sim-view: if sim view unavailable or replay missing -> skipped with sim-view renderer
 - auto: baseline (covered elsewhere) still works

These tests rely on existing lightweight orchestrator synthetic episodes.
"""

from __future__ import annotations

import json
from pathlib import Path

from robot_sf.benchmark.full_classic.orchestrator import run_full_benchmark
from tests.perf_utils.minimal_matrix import write_minimal_matrix


class _Cfg:
    def __init__(
        self,
        tmp_path: Path,
        video_renderer: str,
        capture_replay: bool = True,
        scenario_path: Path | None = None,
    ):
        tmp_path.mkdir(parents=True, exist_ok=True)
        self.output_root = str(tmp_path)
        scenario_file = scenario_path or write_minimal_matrix(tmp_path)
        self.scenario_matrix_path = str(scenario_file)
        self.initial_episodes = 1
        self.max_episodes = 1
        self.batch_size = 1
        self.algo = "ppo"
        self.workers = 1
        self.master_seed = 123
        self.smoke = False
        self.disable_videos = False
        self.max_videos = 1
        self.capture_replay = capture_replay
        self.video_renderer = video_renderer
        # Targets placeholders
        self.target_collision_half_width = 0.05
        self.target_success_half_width = 0.05
        self.target_snqi_half_width = 0.05


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_video_renderer_forced_synthetic(tmp_path):
    cfg = _Cfg(tmp_path / "synthetic", video_renderer="synthetic", capture_replay=True)
    run_full_benchmark(cfg)
    manifest = Path(cfg.output_root) / "reports" / "video_artifacts.json"
    assert manifest.exists(), "video_artifacts.json should exist"
    data = _read(manifest)
    assert data, "expected at least one artifact"
    for a in data:
        # Forced synthetic path should not reclassify to simulation_view
        assert a["renderer"] == "synthetic", a


def test_video_renderer_forced_sim_view_missing(tmp_path):
    # Force sim-view but disable replay capture so it cannot render; expect skipped simulation_view artifacts
    cfg = _Cfg(tmp_path / "simview_missing", video_renderer="sim-view", capture_replay=False)
    run_full_benchmark(cfg)
    manifest = Path(cfg.output_root) / "reports" / "video_artifacts.json"
    assert manifest.exists(), "video_artifacts.json should exist"
    data = _read(manifest)
    assert data, "expected artifacts"
    for a in data:
        assert a["renderer"] == "simulation_view", a
        assert a["status"] == "skipped", a
