"""Tests for deterministic presentation-video curation."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from scripts.tools.prepare_presentation_video_pack import (
    Candidate,
    VideoPackError,
    _candidate_from_row,
    _frame_content_stats,
    _polish_video,
    _portable_qa,
    _qa_video,
    _resolve_video_path,
    prepare_pack,
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


def test_video_path_fallback_refuses_ambiguous_planner_matches(tmp_path: Path) -> None:
    """Filename fallback must not choose a planner when the episode is ambiguous."""
    videos = tmp_path / "videos"
    videos.mkdir()
    (videos / "classic_doorway_seed112_ppo_collision.mp4").write_bytes(b"ppo")
    (videos / "classic_doorway_seed112_socnav_orca_collision.mp4").write_bytes(b"orca")
    row = {
        "scenario_id": "classic_doorway",
        "seed": 112,
        "status": "collision",
        "video": {},
    }

    assert _resolve_video_path(row, tmp_path / "episodes.jsonl", videos) is None

    row["algo"] = "ppo"
    assert (
        _resolve_video_path(row, tmp_path / "episodes.jsonl", videos)
        == (videos / "classic_doorway_seed112_ppo_collision.mp4").resolve()
    )


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


def test_frame_stats_do_not_count_aspect_ratio_padding_as_visible_content(tmp_path: Path) -> None:
    """Blank letterboxed content must fail instead of passing on the pad color."""
    frame = Image.new("RGB", (640, 360), color=(16, 24, 32))
    frame.save(tmp_path / "letterboxed.png")

    assert _frame_content_stats(tmp_path / "letterboxed.png")["nonblack_ratio"] == 0.0


@pytest.mark.parametrize("suffix", [".mov", ".webm", ".mkv"])
def test_no_polish_rejects_non_mp4_sources(tmp_path: Path, suffix: str) -> None:
    """Never relabel non-MP4 source bytes as an MP4 presentation artifact."""
    source = tmp_path / f"source{suffix}"
    source.write_bytes(b"source bytes")
    candidate = replace(
        _candidate(
            tmp_path,
            episode_id="non-mp4",
            scenario_id="fixture",
            outcome="success",
            score=1,
        ),
        source_path=source,
    )
    destination = tmp_path / "clips" / "01_fixture.mp4"

    with pytest.raises(VideoPackError, match=r"--no-polish requires an \.mp4 source"):
        _polish_video(candidate, destination, "ffmpeg", no_polish=True)

    assert not destination.exists()


def test_no_polish_rejects_mp4_suffix_with_non_mp4_container(tmp_path: Path) -> None:
    """An MP4 filename is insufficient when ffprobe identifies another container."""
    source = tmp_path / "source.mp4"
    source.write_bytes(b"webm container bytes")
    candidate = _candidate(
        tmp_path,
        episode_id="mislabeled",
        scenario_id="fixture",
        outcome="success",
        score=1,
    )
    candidate = replace(candidate, source_path=source)
    destination = tmp_path / "clips" / "01_fixture.mp4"

    with pytest.raises(VideoPackError, match="verify an MP4 container"):
        _polish_video(
            candidate,
            destination,
            "ffmpeg",
            no_polish=True,
            source_probe={"codec": "vp9", "format_name": "matroska,webm"},
        )

    assert not destination.exists()


def test_output_guard_does_not_depend_on_process_cwd(tmp_path: Path, monkeypatch) -> None:
    """An unignored output inside the checkout is rejected from another cwd."""
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text("", encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[2]
    output = repo_root / "presentation-pack-unignored-test-fixture"
    monkeypatch.chdir(tmp_path)

    with pytest.raises(VideoPackError, match="non-ignored repository path"):
        prepare_pack(episodes, output_dir=output)

    assert not output.exists()


@pytest.mark.parametrize("duration", [math.nan, math.inf, -math.inf])
def test_prepare_pack_rejects_nonfinite_min_duration(tmp_path: Path, duration: float) -> None:
    """Duration constraints must be finite before any output directory is created."""
    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="finite and non-negative"):
        prepare_pack(episodes, output_dir=tmp_path / "pack", min_duration=duration)

    assert not (tmp_path / "pack").exists()


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="requires local ffmpeg and ffprobe binaries",
)
def test_prepare_pack_cli_runs_real_local_subprocess_path(tmp_path: Path) -> None:
    """Exercise the production CLI with local synthetic media and no network access."""
    ffmpeg = shutil.which("ffmpeg")
    assert ffmpeg is not None
    source = tmp_path / "native-fixture.mp4"
    media_result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=0x3498db:s=320x180:r=10:d=2",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-y",
            str(source),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert media_result.returncode == 0, media_result.stderr

    episodes = tmp_path / "episodes.jsonl"
    episodes.write_text(
        json.dumps(
            {
                "episode_id": "local-fixture",
                "scenario_id": "classic_fixture",
                "seed": 111,
                "algo": "ppo",
                "git_hash": "source-commit",
                "config_hash": "source-config",
                "status": "success",
                "steps": 3,
                "metrics": {"near_misses": 1},
                "video": {
                    "path": str(source),
                    "renderer": "native",
                    "frames": 3,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "presentation-pack"
    script = (
        Path(__file__).resolve().parents[2] / "scripts/tools/prepare_presentation_video_pack.py"
    )
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--episodes",
            str(episodes),
            "--output",
            str(output),
            "--max-clips",
            "1",
            "--min-duration",
            "0.5",
            "--no-polish",
        ],
        cwd=script.parents[2],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout.strip().splitlines()[-1])
    manifest_path = Path(summary["manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert summary["status"] == "ready"
    assert manifest["status"] == "ready"
    assert manifest["artifact_policy"] == {
        "claim_scope": "presentation_only_not_benchmark_evidence",
        "media_disposition": "local_presentation_only",
        "output_dir_git_ignored": False,
        "videos_tracked": False,
    }
    assert manifest["source"]["rights"]["status"] == "redistribution-unknown"
    assert manifest["source"]["rights"]["basis"] == "local-only-byo"
    assert (
        manifest["source"]["episodes_file_sha256"]
        == hashlib.sha256(episodes.read_bytes()).hexdigest()
    )
    assert manifest["source"]["source_git_hash_status"] == "complete"
    assert manifest["invocation"][0].endswith("prepare_presentation_video_pack.py")
    assert "--no-polish" in manifest["invocation"]
    clip = manifest["clips"][0]
    assert clip["source_file"] == source.name
    assert clip["source_git_hash"] == "source-commit"
    assert clip["source_config_hash"] == "source-config"
    assert clip["encoding"] == {"overlay": False, "status": "copied"}
    presentation_path = output / clip["presentation_path"]
    assert presentation_path.read_bytes() == source.read_bytes()
    assert (output / manifest["contact_sheet"]).is_file()
