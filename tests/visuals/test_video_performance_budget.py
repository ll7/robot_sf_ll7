"""TODO docstring. Document this module."""

from __future__ import annotations

import importlib.util
import json
from typing import TYPE_CHECKING

import pytest

from robot_sf.benchmark.runner import run_batch

if TYPE_CHECKING:
    from pathlib import Path

SCHEMA_PATH = "robot_sf/benchmark/schemas/episode.schema.v1.json"


def _write_minimal_matrix(path: Path) -> None:
    """TODO docstring. Document this function.

    Args:
        path: TODO docstring.
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
def test_video_perf_soft_warn(tmp_path: Path, monkeypatch):
    # Force soft breach but not hard; do not enforce
    """Video perf budget should warn, not fail, when only the soft threshold is exceeded.

    Args:
        tmp_path: Temporary artifact directory for the matrix and JSONL output.
        monkeypatch: Pytest environment patch helper used to force the overhead ratio.
    """
    monkeypatch.setenv("ROBOT_SF_VIDEO_OVERHEAD_SOFT", "0.0")
    monkeypatch.setenv("ROBOT_SF_VIDEO_OVERHEAD_HARD", "1.0")
    monkeypatch.delenv("ROBOT_SF_PERF_ENFORCE", raising=False)
    # Force a known overhead_ratio so the soft-breach condition fires regardless of
    # CI runner speed — the test only cares that the warning is emitted, not the
    # actual timing.
    monkeypatch.setenv("ROBOT_SF_TEST_OVERRIDE_OVERHEAD_RATIO", "0.5")

    matrix_path = tmp_path / "matrix.yaml"
    out_jsonl = tmp_path / "episodes.jsonl"
    _write_minimal_matrix(matrix_path)

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

    assert summary["written"] == 1
    # Check JSONL video manifest has perf keys
    lines = out_jsonl.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    rec = json.loads(lines[0])
    vid = rec.get("video", {})
    assert "encode_seconds" in vid and "overhead_ratio" in vid
    assert float(vid["encode_seconds"]) >= 0.0
    assert float(vid["overhead_ratio"]) >= 0.0
    assert vid["overhead_budget_status"] == "soft_breach"
    assert vid["overhead_budget_enforced"] is False
    assert float(vid["overhead_soft_threshold"]) == pytest.approx(0.0)
    assert float(vid["overhead_hard_threshold"]) == pytest.approx(1.0)


@pytest.mark.skipif(moviepy_spec is None, reason="moviepy/ffmpeg not available")
def test_video_perf_hard_enforce_fails(tmp_path: Path, monkeypatch):
    # Force hard breach and enforce; expect batch to record a failure
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
        monkeypatch: TODO docstring.
    """
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
