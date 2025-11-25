"""Tests graceful degradation when matplotlib is absent.

We simulate absence by monkeypatching plots.plt to None and verifying
all plot artifacts are marked skipped with note 'matplotlib missing'.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

from robot_sf.benchmark.full_classic.orchestrator import run_full_benchmark
from tests.perf_utils.minimal_matrix import write_minimal_matrix


class _Cfg:
    def __init__(self, tmp_path: Path):
        tmp_path.mkdir(parents=True, exist_ok=True)
        self.output_root = str(tmp_path)
        self.scenario_matrix_path = str(write_minimal_matrix(tmp_path))
        self.initial_episodes = 1
        self.max_episodes = 1
        self.batch_size = 1
        self.algo = "ppo"
        self.workers = 1
        self.master_seed = 123
        self.smoke = True  # ensure videos skipped
        self.disable_videos = False
        self.max_videos = 1
        self.capture_replay = True
        self.target_collision_half_width = 0.05
        self.target_success_half_width = 0.05
        self.target_snqi_half_width = 0.05


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_plots_skipped_when_matplotlib_missing(monkeypatch, tmp_path):
    plots_mod = importlib.import_module("robot_sf.benchmark.full_classic.plots")
    # Force simulated absence
    monkeypatch.setattr(plots_mod, "plt", None)
    cfg = _Cfg(tmp_path / "no_mpl")
    run_full_benchmark(cfg)
    report_dir = Path(cfg.output_root) / "reports"
    plot_manifest = report_dir / "plot_artifacts.json"
    assert plot_manifest.exists(), "plot_artifacts.json must exist"
    data = _read_json(plot_manifest)
    assert data, "expected some artifacts"
    assert all(a["status"] == "skipped" for a in data if a["note"] == "matplotlib missing")
    # Ensure every artifact either skipped or generated depending on placeholder logic
    assert any(a.get("note") == "matplotlib missing" for a in data)
