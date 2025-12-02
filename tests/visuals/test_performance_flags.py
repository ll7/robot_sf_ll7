"""Test performance budget flags (T053).

Monkeypatch time.perf_counter to inflate plots or video durations to trigger over_budget flags.
"""

from __future__ import annotations

import time as _time
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.full_classic import visuals as visuals_mod
from robot_sf.benchmark.full_classic.visuals import generate_visual_artifacts

if TYPE_CHECKING:
    from pathlib import Path


class Cfg:
    """Cfg class."""

    smoke = False
    disable_videos = False
    max_videos = 1
    capture_replay = True
    video_renderer = "synthetic"  # simplifies path


def test_performance_flags_over_budget(tmp_path: Path, monkeypatch):
    """Test performance flags over budget.

    Args:
        tmp_path: Auto-generated placeholder description.
        monkeypatch: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    # Provide simple record with minimal replay (synthetic path ignores replay)
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

    # Monkeypatch build to simulate long encode time & over-budget plots
    class FakeVideoArtifact:
        """FakeVideoArtifact class."""

        def __init__(self):
            """Init.

            Returns:
                Any: Auto-generated placeholder description.
            """
            self.artifact_id = "video_ep1"
            self.scenario_id = "sc1"
            self.episode_id = "ep1"
            self.path_mp4 = str(tmp_path / "videos" / "video_ep1.mp4")
            self.status = "success"
            self.renderer = "synthetic"
            self.note = None
            self.encode_time_s = 6.2  # >5 threshold
            self.peak_rss_mb = 30.0

    def fake_build(_cfg, _recs, _vdir, _rmap):
        """Fake build.

        Args:
            _cfg: Auto-generated placeholder description.
            _recs: Auto-generated placeholder description.
            _vdir: Auto-generated placeholder description.
            _rmap: Auto-generated placeholder description.

        Returns:
            Any: Auto-generated placeholder description.
        """
        return [FakeVideoArtifact()]

    monkeypatch.setattr(visuals_mod, "_build_video_artifacts", fake_build)

    base = _time.perf_counter()
    timeline = [base, base + 3.1, base + 3.2, base + 3.25]

    def fake_perf_counter2():
        """Fake perf counter2.

        Returns:
            Any: Auto-generated placeholder description.
        """
        if len(timeline) == 1:
            return timeline[0]
        return timeline.pop(0)

    monkeypatch.setattr("time.perf_counter", fake_perf_counter2)

    out = generate_visual_artifacts(tmp_path, Cfg, groups, records)
    perf = out["performance"]
    assert perf["plots_over_budget"] is True
    assert perf["video_over_budget"] is True


@pytest.mark.parametrize("peak", [101.0, 150.0])
def test_memory_over_budget_flag(tmp_path: Path, monkeypatch, peak):
    """Test memory over budget flag.

    Args:
        tmp_path: Auto-generated placeholder description.
        monkeypatch: Auto-generated placeholder description.
        peak: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """

    # Monkeypatch encoding to inject a synthetic success with high peak memory
    class FakeVideoArtifact:
        """FakeVideoArtifact class."""

        def __init__(self):
            """Init.

            Returns:
                Any: Auto-generated placeholder description.
            """
            self.artifact_id = "video_ep1"
            self.scenario_id = "sc1"
            self.episode_id = "ep1"
            self.path_mp4 = str(tmp_path / "videos" / "video_ep1.mp4")
            self.status = "success"
            self.renderer = "synthetic"
            self.note = None
            self.encode_time_s = 0.1
            self.peak_rss_mb = peak

    def fake_build_video(_cfg, _records, _videos_dir, _replay_map):
        """Fake build video.

        Args:
            _cfg: Auto-generated placeholder description.
            _records: Auto-generated placeholder description.
            _videos_dir: Auto-generated placeholder description.
            _replay_map: Auto-generated placeholder description.

        Returns:
            Any: Auto-generated placeholder description.
        """
        return [FakeVideoArtifact()]

    monkeypatch.setattr(visuals_mod, "_build_video_artifacts", fake_build_video)

    records = [
        {"episode_id": "ep1", "scenario_id": "sc1"},
    ]
    groups: list = []
    out = generate_visual_artifacts(tmp_path, Cfg, groups, records)
    perf = out["performance"]
    assert perf["memory_over_budget"] is True
