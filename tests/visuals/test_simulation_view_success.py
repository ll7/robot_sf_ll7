"""SimulationView success path coverage (T050/T050B).

Two tests live here:
  - Stubbed `_attempt_sim_view_videos` to cover orchestration fields without heavy deps.
  - Patched real encode path to exercise replay adapter → frame gen → encode result wiring.

Rationale: CI may lack pygame/moviepy; these tests still verify success-path metadata and
perf reporting without needing full rendering stacks.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

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


def test_simulation_view_encode_path(tmp_path: Path, monkeypatch):
    """Exercise real sim-view encode path (adapter -> frames -> encode)."""

    class AutoCfg(Cfg):
        video_renderer = "auto"

    visuals_mod._SIM_VIEW_AVAILABLE = True  # type: ignore[attr-defined]
    monkeypatch.setattr(visuals_mod, "simulation_view_ready", lambda: True)
    monkeypatch.setattr(visuals_mod, "moviepy_ready", lambda: True)
    # Avoid heavy plotting dependency; plots are not under test here.
    monkeypatch.setattr(visuals_mod, "generate_plots", lambda *_a, **_k: [])

    replay_ep = SimpleNamespace(steps=[0, 1, 2], dt=0.1)
    replay_map = {"ep1": replay_ep}
    monkeypatch.setattr(visuals_mod, "extract_replay_episodes", lambda _recs: replay_map)
    monkeypatch.setattr(visuals_mod, "validate_replay_episode", lambda _ep, min_length=2: True)

    frames = [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(3)]

    def fake_generate_frames(ep, *, fps: int = 10, max_frames=None):
        assert ep is replay_ep
        yield from frames

    class DummyEncodeResult:
        def __init__(self):
            self.status = "success"
            self.note = None
            self.encode_time_s = 0.0123
            self.peak_rss_mb = 5.0

    def fake_encode_frames(frame_iter, path, *, fps: int = 10, sample_memory: bool = False, **_kw):
        list(frame_iter)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"ok")
        return DummyEncodeResult()

    monkeypatch.setattr(visuals_mod, "generate_frames", fake_generate_frames)
    monkeypatch.setattr(visuals_mod, "encode_frames", fake_encode_frames)

    records = [
        {
            "episode_id": "ep1",
            "scenario_id": "sc1",
            "replay_steps": [
                (0.0, 0.0, 0.0, 0.0),
                (0.1, 0.1, 0.0, 0.0),
            ],
        },
    ]
    groups: list = []
    out = generate_visual_artifacts(tmp_path, AutoCfg(), groups, records)
    vids = out["videos"]
    assert len(vids) == 1
    v0 = vids[0]
    renderer = v0.get("renderer") if isinstance(v0, dict) else getattr(v0, "renderer", None)
    status = v0.get("status") if isinstance(v0, dict) else getattr(v0, "status", None)
    encode_time = (
        v0.get("encode_time_s") if isinstance(v0, dict) else getattr(v0, "encode_time_s", None)
    )
    assert renderer == RENDERER_SIM_VIEW
    assert status == "success"
    assert encode_time and encode_time > 0

    perf = out["performance"]
    assert perf.get("first_video_time_s") == encode_time
    assert perf.get("videos_time_s") >= 0.0
