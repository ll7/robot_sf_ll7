from pathlib import Path

import pytest

from robot_sf.benchmark.full_classic import visuals as visuals_mod
from robot_sf.benchmark.full_classic.visual_constants import NOTE_MOVIEPY_MISSING, RENDERER_SIM_VIEW


class DummyEnc:
    def __init__(self, status: str, note: str | None):
        self.status = status
        self.note = note
        self.encode_time_s = None
        self.peak_rss_mb = None


def fake_generate_frames(_ep, _fps: int, _max_frames=None):  # noqa: D401
    # Yield minimal frames; encoder result will mark skipped/moviepy-missing.
    for _ in range(2):
        yield None


def fake_encode_frames(_frame_iter, _path, _fps: int, _sample_memory: bool):  # noqa: D401
    return DummyEnc(status="skipped", note=NOTE_MOVIEPY_MISSING)


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    # Force simulation view available and replay capture active
    monkeypatch.setattr(visuals_mod, "_SIM_VIEW_AVAILABLE", True)
    monkeypatch.setattr(visuals_mod, "simulation_view_ready", lambda: True)
    monkeypatch.setattr(visuals_mod, "generate_frames", fake_generate_frames)
    # encode_frames returns a skipped/moviepy-missing result per episode
    monkeypatch.setattr(visuals_mod, "encode_frames", fake_encode_frames)
    # Provide replay extraction that ensures presence of episode entries
    monkeypatch.setattr(
        visuals_mod,
        "extract_replay_episodes",
        lambda recs: {r["episode_id"]: {"steps": [0, 1, 2]} for r in recs},
    )


class Cfg:
    capture_replay = True
    video_fps = 5
    smoke = False
    sim_view_max_frames = 0
    disable_videos = False
    video_renderer = "sim-view"
    max_videos = 2


def test_moviepy_missing_yields_skipped_artifact(tmp_path: Path, monkeypatch):
    # Provide replay episodes; we bypass validation by monkeypatching validate_replay_episode
    records = [
        {"episode_id": "ep1", "scenario_id": "sc1"},
        {"episode_id": "ep2", "scenario_id": "sc1"},
    ]
    # Force replay validation to always succeed so we exercise encode skip path.
    monkeypatch.setattr(visuals_mod, "validate_replay_episode", lambda _ep, min_length=2: True)
    root = tmp_path
    groups = []
    out = visuals_mod.generate_visual_artifacts(root, Cfg(), groups, records)
    artifacts = out["videos"]
    assert len(artifacts) == 2, artifacts
    # Regression guarantee: no episode silently omitted when encode path is unavailable
    ep_ids = {a.episode_id for a in artifacts}
    assert ep_ids == {"ep1", "ep2"}
    for a in artifacts:
        assert a.status in {"skipped", "failed"}
        assert a.renderer == RENDERER_SIM_VIEW
        assert a.note is not None
