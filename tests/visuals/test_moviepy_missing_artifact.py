"""Assert encode-level moviepy absence keeps per-episode artifacts.

SimulationView availability, replay extraction, frame generation, and encoding
are stubbed so every episode reaches the encode skip path. Both episodes still
produce SimulationView-classified artifacts with a non-null note, and the
performance summary reports zero successful videos.
"""

from pathlib import Path

import pytest

from robot_sf.benchmark.full_classic import visuals as visuals_mod
from robot_sf.benchmark.full_classic.visual_constants import NOTE_MOVIEPY_MISSING, RENDERER_SIM_VIEW


class DummyEnc:
    """Encode-result stand-in exposing status, note, and optional metric fields."""

    def __init__(self, status: str, note: str | None):
        """Store the encode status and note; leave metrics unset.

        Args:
            status: Encode status string.
            note: Optional note describing the status.
        """
        self.status = status
        self.note = note
        self.encode_time_s = None
        self.peak_rss_mb = None


def fake_generate_frames(_ep, *, fps: int = 10, max_frames=None, **_kwargs):
    # Yield minimal frames; encoder result will mark skipped/moviepy-missing.
    """Yield two placeholder frames so the stubbed encoder is reached.

    Args:
        _ep: Replay episode accepted and ignored.
        fps: Frames-per-second option accepted and ignored.
        max_frames: Optional frame cap accepted and ignored.
        _kwargs: Additional keyword arguments accepted and ignored.
    """
    for _ in range(2):
        yield None


def fake_encode_frames(_frame_iter, _path, *, fps: int = 10, sample_memory: bool = False, **_kw):
    """Return a skipped encode result carrying the moviepy-missing note.

    Args:
        _frame_iter: Frame iterator accepted and ignored.
        _path: Output path accepted and ignored.
        fps: Frames-per-second option accepted and ignored.
        sample_memory: Memory-sampling flag accepted and ignored.
        _kw: Additional keyword arguments accepted and ignored.
    """
    return DummyEnc(status="skipped", note=NOTE_MOVIEPY_MISSING)


@pytest.fixture(autouse=True)
def patch_dependencies(monkeypatch):
    # Force simulation view available and replay capture active
    """Mark SimulationView available and stub frame, encode, and replay helpers.

    Autouse fixture that routes every episode through the encode path and returns
    a skipped/moviepy-missing encode result.

    Args:
        monkeypatch: Pytest fixture used to patch the visuals module helpers.
    """
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
    """Config stub forcing SimulationView rendering with two videos and no frame cap."""

    capture_replay = True
    video_fps = 5
    smoke = False
    sim_view_max_frames = 0
    disable_videos = False
    video_renderer = "sim-view"
    max_videos = 2


def test_moviepy_missing_yields_skipped_artifact(tmp_path: Path, monkeypatch):
    # Provide replay episodes; we bypass validation by monkeypatching validate_replay_episode
    """Assert skipped encodes still yield one artifact per episode.

    Replay validation is forced true so the encode skip path is reached; both
    episodes produce SimulationView-classified artifacts with skipped or failed
    status and a note, and performance reports zero successful videos with a
    no-successful-videos note.

    Args:
        tmp_path: Directory receiving the generated visual artifacts.
        monkeypatch: Pytest fixture used to force replay validation to pass.
    """
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

    perf = out["performance"]
    assert perf.get("video_success_count") == 0
    note = perf.get("video_status_note")
    assert isinstance(note, str) and note.startswith("no-successful-videos")
