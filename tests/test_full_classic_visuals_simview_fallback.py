"""Test SimulationView fallback path.

We force the SimulationView attempt to return an empty list (simulating unavailable
primary renderer) and stub the synthetic video generator to return a fabricated
artifact list. This isolates fallback selection logic from actual video IO.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

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
        self.smoke = False
        self.disable_videos = False
        self.max_videos = 1
        self.capture_replay = True
        self.target_collision_half_width = 0.05
        self.target_success_half_width = 0.05
        self.target_snqi_half_width = 0.05


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_simulation_view_empty_list_triggers_fallback(monkeypatch, tmp_path):
    visuals_mod = importlib.import_module("robot_sf.benchmark.full_classic.visuals")
    videos_mod = importlib.import_module("robot_sf.benchmark.full_classic.videos")

    def _no_videos(*_a, **_k):  # simulate SimulationView returning no artifacts
        return []

    def _stub_generate(records, out_dir, _cfg):  # fabricate 1 synthetic artifact
        rec = records[0] if records else {"episode_id": "ep0", "scenario_id": "sc0"}
        return [
            SimpleNamespace(
                artifact_id=f"video_{rec.get('episode_id')}",
                scenario_id=rec.get("scenario_id"),
                episode_id=rec.get("episode_id"),
                path_mp4=str(Path(out_dir) / f"{rec.get('episode_id')}.mp4"),
                status="generated",
                note="stub",
            ),
        ]

    monkeypatch.setattr(visuals_mod, "_attempt_sim_view_videos", _no_videos)
    monkeypatch.setattr(videos_mod, "generate_videos", _stub_generate)
    cfg = _Cfg(tmp_path / "simfail")
    run_full_benchmark(cfg)
    vids = _read_json(Path(cfg.output_root) / "reports" / "video_artifacts.json")
    assert vids, "Expected at least one video artifact"
    assert all(v["renderer"] == "synthetic" for v in vids)
    assert any(v.get("note") == "stub" for v in vids)
