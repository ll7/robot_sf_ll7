"""Integration test: moviepy-missing skip (T052).

Monkeypatch moviepy readiness false so SimulationView attempt yields skipped artifacts
with NOTE_MOVIEPY_MISSING when forced sim-view.
"""

from __future__ import annotations

from pathlib import Path

from robot_sf.benchmark.full_classic import visuals as visuals_mod
from robot_sf.benchmark.full_classic.visual_constants import (
    NOTE_MOVIEPY_MISSING,
    RENDERER_SIM_VIEW,
)
from robot_sf.benchmark.full_classic.visuals import generate_visual_artifacts


class Cfg:
    smoke = False
    disable_videos = False
    max_videos = 1
    capture_replay = True
    video_renderer = "sim-view"


def test_moviepy_missing_sim_view_forced(tmp_path: Path, monkeypatch):
    # Pretend SimulationView available
    visuals_mod._SIM_VIEW_AVAILABLE = True  # noqa: SLF001  # type: ignore[attr-defined]
    monkeypatch.setattr(visuals_mod, "simulation_view_ready", lambda: True)
    # Force moviepy missing
    monkeypatch.setattr(visuals_mod, "moviepy_ready", lambda: False)
    # Avoid real SimulationView rendering by forcing the attempt to return empty list
    monkeypatch.setattr(visuals_mod, "_attempt_sim_view_videos", lambda *a, **k: [])
    # Provide minimal replay (2 steps) so insufficient replay is not cause
    records = [
        {
            "episode_id": "ep1",
            "scenario_id": "sc1",
            "replay_steps": [
                (0.0, 0.0, 0.0, 0.0),
                (0.1, 0.1, 0.0, 0.0),
            ],
        }
    ]
    groups: list = []
    out = generate_visual_artifacts(tmp_path, Cfg, groups, records)
    vids = out["videos"]
    assert len(vids) == 1
    v0 = vids[0]
    renderer = v0.get("renderer") if isinstance(v0, dict) else getattr(v0, "renderer", None)
    note = v0.get("note") if isinstance(v0, dict) else getattr(v0, "note", None)
    status = v0.get("status") if isinstance(v0, dict) else getattr(v0, "status", None)
    assert renderer == RENDERER_SIM_VIEW
    assert note == NOTE_MOVIEPY_MISSING
    assert status == "skipped"
