"""Behavior locks for benchmark environment flags."""

import json
from pathlib import Path

import pytest

from robot_sf.benchmark.runner import _finalize_batch
from robot_sf.benchmark.visualization import _should_use_plot_subprocess


def test_plot_subprocess_env_flag_overrides_auto_heuristic(monkeypatch) -> None:
    """Explicit visualization subprocess settings take precedence over file heuristics."""
    monkeypatch.setenv("ROBOT_SF_VISUALIZATION_SUBPROCESS", "0")
    assert not _should_use_plot_subprocess()

    monkeypatch.setenv("ROBOT_SF_VISUALIZATION_SUBPROCESS", "1")
    assert _should_use_plot_subprocess()


def test_plot_subprocess_unset_env_uses_size_and_filter_heuristic(
    monkeypatch, tmp_path: Path
) -> None:
    """Unset visualization flag keeps small filtered files in-process and other cases isolated."""
    monkeypatch.delenv("ROBOT_SF_VISUALIZATION_SUBPROCESS", raising=False)
    small_path = tmp_path / "small.jsonl"
    small_path.write_bytes(b"x" * 5_000_000)
    large_path = tmp_path / "large.jsonl"
    large_path.write_bytes(b"x" * 5_000_001)

    assert not _should_use_plot_subprocess(
        episodes_path=str(small_path), scenario_filter="scenario-a"
    )
    assert _should_use_plot_subprocess(episodes_path=str(large_path), scenario_filter="scenario-a")
    assert _should_use_plot_subprocess(episodes_path=str(tmp_path / "missing.jsonl"))


def _write_video_episodes(out_path: Path) -> None:
    records = [
        {"video": {"frames": 10, "encode_seconds": 0.5, "overhead_ratio": 1.2}},
        {"video": {"frames": 30, "encode_seconds": 1.5, "overhead_ratio": 1.4}},
    ]
    out_path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")


def test_perf_snapshot_env_flag_writes_video_summary(monkeypatch, tmp_path: Path) -> None:
    """Enabled video snapshot flag writes the documented summary fields."""
    out_path = tmp_path / "episodes.jsonl"
    _write_video_episodes(out_path)
    monkeypatch.setenv("ROBOT_SF_VIDEO_PERF_SNAPSHOT", "1")

    _finalize_batch(out_path, wrote=0, resume=False)

    snapshot_path = tmp_path / "videos" / "perf_snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert set(snapshot) == {
        "episodes",
        "total_frames",
        "total_encode_seconds",
        "encode_ms_per_frame",
        "mean_overhead_ratio",
        "environment",
    }
    assert snapshot["episodes"] == 2
    assert snapshot["total_frames"] == 40
    assert snapshot["total_encode_seconds"] == 2.0
    assert snapshot["encode_ms_per_frame"] == 50.0
    assert snapshot["mean_overhead_ratio"] == pytest.approx(1.3)
    assert set(snapshot["environment"]) == {"os", "python", "processor"}


def test_perf_snapshot_unset_env_writes_nothing(monkeypatch, tmp_path: Path) -> None:
    """Unset video snapshot flag leaves no performance artifact behind."""
    out_path = tmp_path / "episodes.jsonl"
    _write_video_episodes(out_path)
    monkeypatch.delenv("ROBOT_SF_VIDEO_PERF_SNAPSHOT", raising=False)

    _finalize_batch(out_path, wrote=0, resume=False)

    assert not (tmp_path / "videos" / "perf_snapshot.json").exists()
