"""Test SimulationView success path (T050).

This test simulates a successful SimulationView rendering + encoding path
by monkeypatching dependencies:
  - Forces `_SIM_VIEW_AVAILABLE` True and `simulation_view_ready()` True.
  - Forces `moviepy_ready()` True so encoding is attempted.
  - Replaces `_attempt_sim_view_videos` with a lightweight stub that returns
    a VideoArtifact-like dict marked success (avoids real frame generation / moviepy).

We assert:
  - Renderer recorded as sim-view
  - Status == success
  - encode_time_s present
  - performance manifest includes first_video_time_s and (optionally) render split key.

Rationale:
Running the real SimulationView + moviepy stack in CI can be flaky or slow.
This stub ensures orchestration logic for success path is covered without heavy deps.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

from robot_sf.benchmark.full_classic import visuals as visuals_mod
from robot_sf.benchmark.full_classic.visual_constants import RENDERER_SIM_VIEW
from robot_sf.benchmark.full_classic.visuals import generate_visual_artifacts

if TYPE_CHECKING:
    from pathlib import Path


class Cfg:
    smoke = False
    disable_videos = False
    max_videos = 1
    capture_replay = True
    video_renderer = "sim-view"
    video_fps = 5


def test_simulation_view_success_stub(tmp_path: Path, monkeypatch):
    # Force SimulationView + moviepy readiness
    visuals_mod._SIM_VIEW_AVAILABLE = True  # type: ignore[attr-defined]
    monkeypatch.setattr(visuals_mod, "simulation_view_ready", lambda: True)
    monkeypatch.setattr(visuals_mod, "moviepy_ready", lambda: True)

    # Provide minimal valid replay (>=2 steps)
    records = [
        {
            "episode_id": "ep1",
            "scenario_id": "sc1",
            "replay_steps": [
                (0.0, 0.0, 0.0, 0.0),
                (0.1, 0.1, 0.0, 0.0),
                (0.2, 0.2, 0.0, 0.0),
            ],
        },
    ]

    # Build a stub success artifact (mirrors fields from VideoArtifact JSON output)
    stub_artifact = SimpleNamespace(
        artifact_id="video_ep1",
        scenario_id="sc1",
        episode_id="ep1",
        path_mp4=str(tmp_path / "videos" / "video_ep1.mp4"),
        status="success",
        renderer=RENDERER_SIM_VIEW,
        note=None,
        encode_time_s=0.1234,
        peak_rss_mb=42.0,
    )

    def _stub_attempt(records, out_dir, cfg, replay_map):
        # Return list with our success artifact
        return [stub_artifact]

    monkeypatch.setattr(visuals_mod, "_attempt_sim_view_videos", _stub_attempt)

    groups: list = []
    out = generate_visual_artifacts(tmp_path, Cfg, groups, records)
    vids = out["videos"]
    assert len(vids) == 1
    v0 = vids[0]
    # Manifest returns list of dicts (after serialization path) so support both
    renderer = v0.get("renderer") if isinstance(v0, dict) else getattr(v0, "renderer", None)
    status = v0.get("status") if isinstance(v0, dict) else getattr(v0, "status", None)
    encode_time = (
        v0.get("encode_time_s") if isinstance(v0, dict) else getattr(v0, "encode_time_s", None)
    )
    assert renderer == RENDERER_SIM_VIEW
    assert status == "success"
    assert encode_time and encode_time > 0

    perf = out["performance"]
    assert perf.get("first_video_time_s") is not None
    # first_video_render_time_s may be None with stub, but presence of key expected
    assert "first_video_render_time_s" in perf
