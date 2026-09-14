"""Tests for deterministic presentation-video curation."""

from __future__ import annotations

from pathlib import Path

from scripts.tools.prepare_presentation_video_pack import (
    Candidate,
    VideoPackError,
    _candidate_from_row,
    _portable_qa,
    _qa_video,
    _resolve_video_path,
    select_candidates,
)


def _candidate(
    tmp_path: Path,
    *,
    episode_id: str,
    scenario_id: str,
    outcome: str,
    score: float,
) -> Candidate:
    row = {
        "episode_id": episode_id,
        "scenario_id": scenario_id,
        "seed": 111,
        "algo": "ppo",
        "status": outcome,
        "steps": 100,
        "metrics": {"near_misses": score},
        "video": {"path": str(tmp_path / f"{episode_id}.mp4")},
    }
    return _candidate_from_row(row, Path(row["video"]["path"]), 1)


def test_selection_prefers_two_successes_and_one_collision_with_novel_scenarios(
    tmp_path: Path,
) -> None:
    """The default three-clip pack should tell a balanced visual story."""
    candidates = [
        _candidate(
            tmp_path,
            episode_id="success-a",
            scenario_id="classic_crossing",
            outcome="success",
            score=20,
        ),
        _candidate(
            tmp_path,
            episode_id="success-b",
            scenario_id="classic_doorway",
            outcome="success",
            score=18,
        ),
        _candidate(
            tmp_path,
            episode_id="success-c",
            scenario_id="classic_crossing",
            outcome="success",
            score=50,
        ),
        _candidate(
            tmp_path,
            episode_id="collision-a",
            scenario_id="classic_bottleneck",
            outcome="collision",
            score=2,
        ),
    ]

    selected = select_candidates(candidates, limit=3)

    assert [candidate.outcome for candidate in selected] == ["success", "success", "collision"]
    assert len({candidate.scenario_id for candidate in selected}) == 3
    assert selected[0].episode_id == "success-c"


def test_video_path_fallback_matches_scenario_seed_and_outcome(tmp_path: Path) -> None:
    """Legacy recording directories can be resolved from episode identity."""
    videos = tmp_path / "videos"
    videos.mkdir()
    expected = videos / "classic_doorway_seed112_ppo_collision.mp4"
    expected.write_bytes(b"fixture")
    row = {
        "scenario_id": "classic_doorway",
        "seed": 112,
        "status": "collision",
        "video": {},
    }

    assert _resolve_video_path(row, tmp_path / "episodes.jsonl", videos) == expected.resolve()


def test_video_url_reference_is_rejected_without_network_access(tmp_path: Path) -> None:
    """The local-only pack must reject URI references before any media command runs."""
    row = {"video": {"path": "https://example.invalid/clip.mp4"}}

    try:
        _resolve_video_path(row, tmp_path / "episodes.jsonl", tmp_path)
    except VideoPackError as exc:
        assert "URL video references" in str(exc)
    else:
        raise AssertionError("URL video reference was accepted")


def test_qa_paths_are_portable_and_path_free(tmp_path: Path) -> None:
    """Manifest QA retains relative artifact names without machine-specific directories."""
    qa = {
        "path": str(tmp_path / "source.mp4"),
        "samples": [{"path": str(tmp_path / "stills" / "middle.png")}],
    }

    portable = _portable_qa(qa, tmp_path)

    assert portable == {"path": "source.mp4", "samples": [{"path": "stills/middle.png"}]}


def test_selection_limit_one_uses_global_interest_order(tmp_path: Path) -> None:
    """A single requested clip should pick the highest-interest episode."""
    candidates = [
        _candidate(
            tmp_path, episode_id="success", scenario_id="success_case", outcome="success", score=1
        ),
        _candidate(
            tmp_path,
            episode_id="collision",
            scenario_id="collision_case",
            outcome="collision",
            score=1,
        ),
    ]

    selected = select_candidates(candidates, limit=1)

    assert [candidate.episode_id for candidate in selected] == ["collision"]


def test_video_qa_rejects_all_blank_samples(tmp_path: Path, monkeypatch) -> None:
    """A decodable but blank video must not enter a presentation pack."""
    import scripts.tools.prepare_presentation_video_pack as pack

    source = tmp_path / "blank.mp4"
    source.write_bytes(b"fixture")
    monkeypatch.setattr(
        pack,
        "_probe_video",
        lambda path, ffprobe: {
            "codec": "h264",
            "width": 1280,
            "height": 720,
            "frame_count": 20,
            "frame_rate": 10.0,
            "duration_s": 2.0,
            "size_bytes": path.stat().st_size,
        },
    )
    monkeypatch.setattr(pack, "_run_command", lambda command: (True, ""))
    monkeypatch.setattr(
        pack,
        "_capture_samples",
        lambda *args, **kwargs: [
            {"label": "start", "status": "pass", "nonblack_ratio": 0.0},
            {"label": "middle", "status": "pass", "nonblack_ratio": 0.0},
            {"label": "end", "status": "pass", "nonblack_ratio": 0.0},
        ],
    )

    result = _qa_video(source, ffmpeg="ffmpeg", ffprobe="ffprobe", min_duration=1.0)

    assert result["status"] == "failed"
    assert "all_sampled_frames_are_blank_or_dark" in result["reasons"]
    assert "insufficient_visible_samples<2_of_3" in result["reasons"]
