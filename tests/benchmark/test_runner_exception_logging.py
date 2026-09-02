"""Regression tests for benchmark runner exception visibility."""

from __future__ import annotations

from concurrent.futures import TimeoutError as FuturesTimeoutError
from unittest.mock import Mock

import numpy as np
import pytest
from loguru import logger

from robot_sf.benchmark import runner


@pytest.mark.base_sensitive
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


def test_policy_step_fallback_updates_returned_metadata(monkeypatch) -> None:
    """Planner-step fallback status must remain visible in returned metadata."""
    planner = Mock()
    planner.get_metadata.return_value = {"algorithm": "random", "status": "ok"}
    step_runner = Mock()
    step_runner.step.side_effect = ValueError("forced planner failure")

    monkeypatch.setattr(
        runner,
        "_load_baseline_planner",
        lambda *_args: (planner, object, {}),
    )
    monkeypatch.setattr(runner, "_build_observation", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_PlannerStepProcess", lambda *_args, **_kwargs: step_runner)

    policy, metadata = runner._create_robot_policy("random", None, seed=42)
    try:
        velocity = policy(
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.empty((0, 2)),
            0.1,
        )
    finally:
        runner._close_robot_policy(policy)

    assert velocity == pytest.approx(np.array([0.0, 0.0]))
    assert metadata["status"] == "policy_step_error_fallback"
    assert metadata["fallback_reason"] == "policy_step_error"


def test_policy_step_isolation_error_preserves_classification() -> None:
    """Worker-lifecycle errors remain identifiable after the fallback path."""
    step_runner = Mock()
    step_runner.step.side_effect = RuntimeError(
        "planner step worker exited before returning an action"
    )
    timeout_metadata = {
        "step_timeouts": 0,
        "worker_errors": 0,
        "step_retries": 0,
        "fallback_actions": 0,
    }
    metadata = {}

    action = runner._step_planner_with_retry(
        step_runner,
        None,
        "ppo",
        timeout_metadata,
        metadata,
        retry_budget=0,
    )

    assert action == {"vx": 0.0, "vy": 0.0}
    assert metadata["status"] == "policy_step_isolation_unavailable"
    assert metadata["fallback_reason"] == "planner step worker exited before returning an action"


def test_action_to_velocity_covers_action_shapes_and_validation() -> None:
    """Extracted action conversion must clamp, align, and reject malformed actions."""
    zero = np.array([0.0, 0.0])
    goal = np.array([1.0, 0.0])

    assert runner._action_to_velocity(
        {"vx": 3.0, "vy": 4.0}, zero, zero, goal, "random"
    ) == pytest.approx(np.array([1.2, 1.6]))
    assert runner._action_to_velocity(
        {"v": 1.0, "omega": 0.0}, zero, zero, zero, "random"
    ) == pytest.approx(zero)
    with pytest.raises(ValueError, match="Invalid action format from random"):
        runner._action_to_velocity({"invalid": 1.0}, zero, zero, goal, "random")


@pytest.mark.parametrize("failure", [FuturesTimeoutError(), RuntimeError("transient")])
def test_step_planner_retry_recovers_transient_failures(failure) -> None:
    """Transient timeout and worker errors should consume the retry budget once."""
    step_runner = Mock()
    step_runner.step.side_effect = [failure, {"vx": 1.0, "vy": 0.0}]
    timeout_metadata = {
        "step_timeouts": 0,
        "worker_errors": 0,
        "step_retries": 0,
        "fallback_actions": 0,
    }

    action = runner._step_planner_with_retry(
        step_runner,
        None,
        "orca",
        timeout_metadata,
        {},
        retry_budget=1,
    )

    assert action == {"vx": 1.0, "vy": 0.0}
    assert timeout_metadata["step_retries"] == 1
    assert timeout_metadata["fallback_actions"] == 0


def test_step_planner_timeout_falls_back_without_retry() -> None:
    """A timeout with no retry budget should return an explicit zero action."""
    step_runner = Mock()
    step_runner.step.side_effect = FuturesTimeoutError()
    timeout_metadata = {
        "step_timeouts": 0,
        "worker_errors": 0,
        "step_retries": 0,
        "fallback_actions": 0,
    }
    metadata = {}

    action = runner._step_planner_with_retry(
        step_runner,
        None,
        "random",
        timeout_metadata,
        metadata,
        retry_budget=0,
    )

    assert action == {"vx": 0.0, "vy": 0.0}
    assert metadata == {
        "status": "policy_step_timeout_fallback",
        "fallback_reason": "policy_step_timeout",
    }
    assert timeout_metadata["fallback_actions"] == 1


def test_policy_builder_records_unavailable_step_isolation(monkeypatch) -> None:
    """Isolation setup failures should be surfaced through policy metadata."""
    planner = Mock()
    planner.get_metadata.return_value = {"algorithm": "random", "status": "ok"}

    monkeypatch.setattr(
        runner,
        "_load_baseline_planner",
        lambda *_args: (planner, object, {}),
    )
    monkeypatch.setattr(
        runner,
        "_PlannerStepProcess",
        Mock(side_effect=RuntimeError("forced isolation failure")),
    )
    monkeypatch.setattr(runner, "_build_observation", lambda *_args, **_kwargs: None)

    policy, metadata = runner._create_robot_policy("random", None, seed=42)
    try:
        velocity = policy(
            np.array([0.0, 0.0]),
            np.array([0.0, 0.0]),
            np.array([1.0, 0.0]),
            np.empty((0, 2)),
            0.1,
        )
    finally:
        runner._close_robot_policy(policy)

    assert velocity == pytest.approx(np.array([0.0, 0.0]))
    assert metadata["status"] == "policy_step_isolation_unavailable"
    assert metadata["fallback_reason"] == "policy step isolation unavailable"
    assert metadata["policy_step_timeout"]["isolation"] == "unavailable"


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
