"""Characterization tests for Full Classic visual-artifact generation."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

# ``visuals`` imports the SimulationView stack, whose optional pygame dependency
# is not installed in the core test lane.  Stub it only while importing the
# unit under test; the exercised paths below replace all renderer interactions.
_injected_pygame = "pygame" not in sys.modules
if _injected_pygame:
    sys.modules["pygame"] = MagicMock()
from robot_sf.benchmark.full_classic import visuals  # noqa: E402
from robot_sf.benchmark.full_classic.visual_constants import (  # noqa: E402
    NOTE_SMOKE_MODE,
    RENDERER_SIM_VIEW,
)

if _injected_pygame:
    del sys.modules["pygame"]


def _config(**overrides: object) -> SimpleNamespace:
    """Build the small configuration surface used by visual generation."""
    values: dict[str, object] = {
        "smoke": False,
        "capture_replay": False,
        "disable_videos": False,
        "max_videos": 1,
        "video_renderer": "auto",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_smoke_path_writes_skipped_fallback_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Smoke mode keeps a useful non-evidence manifest when producers are empty."""
    monkeypatch.setattr(visuals, "generate_plots", lambda *_args: [])
    monkeypatch.setattr(visuals, "_build_video_artifacts", lambda *_args: [])
    monkeypatch.setattr(visuals, "_SIM_VIEW_AVAILABLE", False)

    result = visuals.generate_visual_artifacts(
        tmp_path,
        _config(smoke=True),
        groups={},
        records=[{"episode_id": "episode-1", "scenario_id": "scenario-1"}],
    )

    assert result["plots"] == [
        {
            "kind": "diagnostic_unavailable",
            "path_pdf": "",
            "status": "skipped",
            "note": "plots-unavailable",
        }
    ]
    assert result["videos"][0].status == "skipped"
    assert result["videos"][0].note == NOTE_SMOKE_MODE
    assert result["performance"]["videos_time_s"] == 0.0
    assert result["performance"]["video_success_count"] == 0

    reports_dir = tmp_path / "reports"
    video_manifest = json.loads((reports_dir / "video_artifacts.json").read_text())
    performance_manifest = json.loads((reports_dir / "performance_visuals.json").read_text())
    assert video_manifest[0]["episode_id"] == "episode-1"
    assert video_manifest[0]["note"] == NOTE_SMOKE_MODE
    assert performance_manifest == result["performance"]


def test_normal_path_preserves_video_performance_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Normal mode serializes timing, renderer, and budget fields unchanged."""
    times = iter([10.0, 10.5, 11.0, 11.75])
    monkeypatch.setattr(visuals, "time", SimpleNamespace(perf_counter=lambda: next(times)))
    monkeypatch.setattr(
        visuals,
        "generate_plots",
        lambda *_args: [
            SimpleNamespace(
                kind="summary", path_pdf="plots/summary.pdf", status="success", note=None
            )
        ],
    )
    monkeypatch.setattr(
        visuals,
        "_build_video_artifacts",
        lambda *_args: [
            visuals.VideoArtifact(
                artifact_id="video_episode-1",
                scenario_id="scenario-1",
                episode_id="episode-1",
                path_mp4="videos/video_episode-1.mp4",
                status="success",
                renderer=RENDERER_SIM_VIEW,
                encode_time_s=0.5,
                peak_rss_mb=80.0,
            )
        ],
    )

    result = visuals.generate_visual_artifacts(
        tmp_path,
        _config(),
        groups={},
        records=[{"episode_id": "episode-1", "scenario_id": "scenario-1"}],
    )

    assert result["performance"] == {
        "plots_time_s": 0.5,
        "videos_time_s": 0.75,
        "first_video_time_s": 0.5,
        "first_video_render_time_s": 0.25,
        "first_video_peak_rss_mb": 80.0,
        "plots_over_budget": False,
        "video_over_budget": False,
        "memory_over_budget": False,
        "plots_runtime_sec": 0.5,
        "videos_runtime_sec": 0.75,
        "first_video_encode_time_s": 0.5,
        "video_success_count": 1,
        "video_status_note": None,
    }

    reports_dir = tmp_path / "reports"
    assert json.loads((reports_dir / "plot_artifacts.json").read_text()) == result["plots"]
    assert json.loads((reports_dir / "video_artifacts.json").read_text()) == [
        {
            "artifact_id": "video_episode-1",
            "scenario_id": "scenario-1",
            "episode_id": "episode-1",
            "path_mp4": "videos/video_episode-1.mp4",
            "status": "success",
            "renderer": RENDERER_SIM_VIEW,
            "note": None,
            "encode_time_s": 0.5,
            "peak_rss_mb": 80.0,
        }
    ]
