"""Edge case tests for visual artifact generation.

Focus:
- Deterministic ordering (first N episodes)
- Disable vs smoke distinction in notes
- Renderer field presence
"""

from __future__ import annotations

import json
from pathlib import Path

from robot_sf.benchmark.full_classic.orchestrator import run_full_benchmark


class _Cfg:
    def __init__(self, tmp_path: Path, **over):
        self.output_root = str(tmp_path)
        self.scenario_matrix_path = "configs/scenarios/classic_interactions.yaml"
        self.initial_episodes = 1
        self.max_episodes = 3
        self.batch_size = 3
        self.algo = "ppo"
        self.workers = 1
        self.master_seed = 123
        self.smoke = over.get("smoke", False)
        self.disable_videos = over.get("disable_videos", False)
        self.max_videos = over.get("max_videos", 2)
        self.target_collision_half_width = 0.05
        self.target_success_half_width = 0.05
        self.target_snqi_half_width = 0.05


def _read(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_deterministic_video_selection(tmp_path):
    cfg1 = _Cfg(tmp_path / "sel1")
    cfg2 = _Cfg(tmp_path / "sel2")
    run_full_benchmark(cfg1)
    run_full_benchmark(cfg2)
    v1 = _read(Path(cfg1.output_root) / "reports" / "video_artifacts.json")
    v2 = _read(Path(cfg2.output_root) / "reports" / "video_artifacts.json")
    assert [a["episode_id"] for a in v1] == [a["episode_id"] for a in v2]


def test_renderer_field_present(tmp_path):
    cfg = _Cfg(tmp_path / "renderer")
    run_full_benchmark(cfg)
    vids = _read(Path(cfg.output_root) / "reports" / "video_artifacts.json")
    assert all("renderer" in a for a in vids)


def test_disable_vs_smoke_notes(tmp_path):
    cfg_disable = _Cfg(tmp_path / "disa", disable_videos=True)
    cfg_smoke = _Cfg(tmp_path / "smok", smoke=True)
    run_full_benchmark(cfg_disable)
    run_full_benchmark(cfg_smoke)
    v_disable = _read(Path(cfg_disable.output_root) / "reports" / "video_artifacts.json")
    v_smoke = _read(Path(cfg_smoke.output_root) / "reports" / "video_artifacts.json")
    assert any("disabled" in (a.get("note") or "") for a in v_disable)
    assert any("smoke" in (a.get("note") or "") for a in v_smoke)
