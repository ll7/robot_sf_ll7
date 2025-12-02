"""Module test_video_performance_budget auto-generated docstring."""

from __future__ import annotations

import importlib.util
import json

import pytest

try:
    from loguru import logger
except ImportError:  # pragma: no cover - loguru is a core dependency
    logger = None  # type: ignore[assignment]

from typing import TYPE_CHECKING

from robot_sf.benchmark.runner import run_batch

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_PATH = "docs/dev/issues/social-navigation-benchmark/episode_schema.json"


def _write_minimal_matrix(path: Path) -> None:
    """Write minimal matrix.

    Args:
        path: Auto-generated placeholder description.

    Returns:
        None: Auto-generated placeholder description.
    """
    scenarios = [
        {
            "id": "perf-video-uni-low-open",
            "density": "low",
            "flow": "uni",
            "obstacle": "open",
            "groups": 0.0,
            "speed_var": "low",
            "goal_topology": "point",
            "robot_context": "embedded",
            "repeats": 1,
        },
    ]
    import yaml  # type: ignore

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(scenarios, f)


moviepy_spec = importlib.util.find_spec("moviepy")


@pytest.mark.skipif(moviepy_spec is None, reason="moviepy/ffmpeg not available")
@pytest.mark.skipif(logger is None, reason="loguru not available")
def test_video_perf_soft_warn(tmp_path: Path, monkeypatch):
    """Test video perf soft warn.

    Args:
        tmp_path: Auto-generated placeholder description.
        monkeypatch: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    # Force soft breach but not hard; do not enforce
    monkeypatch.setenv("ROBOT_SF_VIDEO_OVERHEAD_SOFT", "0.0")
    monkeypatch.setenv("ROBOT_SF_VIDEO_OVERHEAD_HARD", "1.0")
    monkeypatch.delenv("ROBOT_SF_PERF_ENFORCE", raising=False)

    matrix_path = tmp_path / "matrix.yaml"
    out_jsonl = tmp_path / "episodes.jsonl"
    _write_minimal_matrix(matrix_path)

    captured: list[str] = []
    token = logger.add(
        lambda message: captured.append(str(message)),
        level="WARNING",
        format="{message}",
    )

    try:
        summary = run_batch(
            scenarios_or_path=str(matrix_path),
            out_path=str(out_jsonl),
            schema_path=SCHEMA_PATH,
            base_seed=0,
            horizon=8,
            dt=0.1,
            video_enabled=True,
            video_renderer="synthetic",
            workers=1,
            resume=False,
        )
    finally:
        logger.remove(token)

    assert summary["written"] == 1
    # Check JSONL video manifest has perf keys
    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    vid = rec.get("video", {})
    assert "encode_seconds" in vid and "overhead_ratio" in vid
    assert float(vid["encode_seconds"]) >= 0.0
    assert float(vid["overhead_ratio"]) >= 0.0
    captured_text = "\n".join(captured)
    assert "Video overhead soft breach" in captured_text


@pytest.mark.skipif(moviepy_spec is None, reason="moviepy/ffmpeg not available")
def test_video_perf_hard_enforce_fails(tmp_path: Path, monkeypatch):
    """Test video perf hard enforce fails.

    Args:
        tmp_path: Auto-generated placeholder description.
        monkeypatch: Auto-generated placeholder description.

    Returns:
        Any: Auto-generated placeholder description.
    """
    # Force hard breach and enforce; expect batch to record a failure
    monkeypatch.setenv("ROBOT_SF_VIDEO_OVERHEAD_SOFT", "0.0")
    monkeypatch.setenv("ROBOT_SF_VIDEO_OVERHEAD_HARD", "0.0")
    monkeypatch.setenv("ROBOT_SF_PERF_ENFORCE", "1")

    matrix_path = tmp_path / "matrix.yaml"
    out_jsonl = tmp_path / "episodes.jsonl"
    _write_minimal_matrix(matrix_path)

    summary = run_batch(
        scenarios_or_path=str(matrix_path),
        out_path=str(out_jsonl),
        schema_path=SCHEMA_PATH,
        base_seed=0,
        horizon=5,
        dt=0.1,
        video_enabled=True,
        video_renderer="synthetic",
        workers=1,
        resume=False,
    )
    # With enforcement, the episode should fail and not be written
    assert summary["written"] == 0
    assert len(summary["failures"]) == 1
