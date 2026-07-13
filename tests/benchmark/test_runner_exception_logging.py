"""Regression tests for benchmark runner exception visibility."""

from __future__ import annotations

import numpy as np
from loguru import logger

from robot_sf.benchmark import runner


def test_run_batch_sequential_worker_failure_logs_warning(tmp_path, monkeypatch) -> None:
    """Worker exceptions in the serial batch path should be logged before being summarized."""
    captured: list = []

    def capture_message(message):
        """Capture warning events emitted by runner."""
        captured.append(message)

    def fake_run_job(job):
        """Fail worker execution to exercise exception logging."""
        del job
        raise RuntimeError("forced serial worker failure")

    handle = logger.add(capture_message, level="WARNING")
    monkeypatch.setattr(runner, "_run_job_worker", fake_run_job)
    try:
        wrote, failures, abort_metadata = runner._run_batch_sequential(
            [({"id": "scenario-1"}, 42)],
            out_path=tmp_path / "episodes.jsonl",
            schema={},
            fixed_params={},
            progress_cb=None,
            fail_fast=False,
        )
    finally:
        logger.remove(handle)

    assert wrote == 0
    assert len(failures) == 1
    assert failures[0]["scenario_id"] == "scenario-1"
    assert abort_metadata is None
    assert any(
        "Benchmark batch job failed in serial execution" in msg.record["message"]
        for msg in captured
    )


def test_maybe_encode_video_logs_nonfatal_errors(tmp_path, monkeypatch) -> None:
    """Video helper failures should be logged but not raised."""
    captured: list = []

    def capture_message(message):
        """Capture warning events emitted by runner."""
        captured.append(message)

    def fake_encode(*args, **kwargs):
        """Force a nonfatal encode failure."""
        del args, kwargs
        raise TypeError("forced encode failure")

    handle = logger.add(capture_message, level="WARNING")
    monkeypatch.setattr(runner, "_try_encode_synthetic_video", fake_encode)
    try:
        runner._maybe_encode_video(
            record={
                "episode_id": "episode-1",
                "scenario_id": "scenario-1",
                "seed": 42,
            },
            robot_pos_traj=[np.array([0.0, 0.0])],
            videos_dir=str(tmp_path),
            video_enabled=True,
            video_renderer="synthetic",
            perf_start=0.0,
        )
    finally:
        logger.remove(handle)

    assert any(
        "Synthetic video encoding failure for episode_id=episode-1 scenario_id=scenario-1 "
        "renderer=synthetic; continuing benchmark run." in msg.record["message"]
        for msg in captured
    )
