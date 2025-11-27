"""Ensure auto mode falls back to synthetic rendering when sim-view fails."""

from __future__ import annotations

from robot_sf.benchmark.full_classic import visuals as visuals_mod
from robot_sf.benchmark.full_classic.visuals import (
    RENDERER_SIM_VIEW,
    RENDERER_SYNTHETIC,
    VideoArtifact,
    generate_visual_artifacts,
)


class Cfg:
    smoke = False
    disable_videos = False
    max_videos = 1
    capture_replay = True
    video_renderer = "auto"


def test_sim_view_failure_triggers_synthetic_fallback(tmp_path, monkeypatch):
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

    def fake_attempt_sim_view(_records, _videos_dir, _cfg, _replay_map):
        return [
            VideoArtifact(
                artifact_id="video_ep1",
                scenario_id="sc1",
                episode_id="ep1",
                path_mp4=str(tmp_path / "videos" / "video_ep1.mp4"),
                status="failed",
                renderer=RENDERER_SIM_VIEW,
                note="render-error:IndexError",
                encode_time_s=None,
                peak_rss_mb=None,
            ),
        ]

    def fake_synthetic(_records, _videos_dir, _cfg):
        return [
            VideoArtifact(
                artifact_id="video_ep1",
                scenario_id="sc1",
                episode_id="ep1",
                path_mp4=str(tmp_path / "videos" / "video_ep1.mp4"),
                status="success",
                renderer=RENDERER_SYNTHETIC,
                note=None,
                encode_time_s=0.2,
                peak_rss_mb=8.0,
            ),
        ]

    monkeypatch.setattr(visuals_mod, "_attempt_sim_view_videos", fake_attempt_sim_view)
    monkeypatch.setattr(visuals_mod, "_synthetic_fallback_videos", fake_synthetic)

    out = generate_visual_artifacts(tmp_path, Cfg, groups, records)
    videos = out["videos"]
    assert len(videos) == 1
    artifact = videos[0]
    assert artifact.renderer == RENDERER_SYNTHETIC
    assert artifact.status == "success"
    assert artifact.note and "fallback-from-sim-view" in artifact.note

    perf = out["performance"]
    assert perf["video_status_note"] == "fallback-from-sim-view"
