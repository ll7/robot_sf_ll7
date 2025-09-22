"""Test synthetic fallback path (T051).

Forces SimulationView unavailable and ensures auto mode produces synthetic video artifact.
"""

from __future__ import annotations

from pathlib import Path

from robot_sf.benchmark.full_classic import visuals as visuals_mod
from robot_sf.benchmark.full_classic.visual_constants import RENDERER_SYNTHETIC
from robot_sf.benchmark.full_classic.visuals import generate_visual_artifacts


class Cfg:
    smoke = False
    disable_videos = False
    max_videos = 1
    capture_replay = False  # avoid reclassification to sim-view insufficient replay
    video_renderer = "auto"


def test_synthetic_fallback_when_sim_view_unavailable(tmp_path: Path):
    # Force SimulationView unavailable
    visuals_mod._SIM_VIEW_AVAILABLE = False  # noqa: SLF001  # type: ignore[attr-defined]
    records = [
        {"episode_id": "ep1", "scenario_id": "sc1"},
    ]
    groups: list = []
    out = generate_visual_artifacts(tmp_path, Cfg, groups, records)
    vids = out["videos"]
    assert len(vids) == 1
    v0 = vids[0]
    renderer = v0.get("renderer") if isinstance(v0, dict) else getattr(v0, "renderer", None)
    assert renderer == RENDERER_SYNTHETIC, v0
