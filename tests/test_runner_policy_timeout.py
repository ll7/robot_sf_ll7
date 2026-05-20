"""Regression tests for benchmark runner planner-step timeouts."""

from __future__ import annotations

import multiprocessing as mp
import time
from typing import Any

import numpy as np
import pytest

from robot_sf import baselines
from robot_sf.benchmark import runner


class _SlowPlanner:
    """Planner stub whose step call exceeds the benchmark timeout budget."""

    def __init__(self, _config: dict[str, Any], *, seed: int) -> None:
        self.seed = seed

    def step(self, _obs: Any) -> dict[str, float]:
        time.sleep(0.75)
        return {"vx": 1.0, "vy": 0.0}

    def get_metadata(self) -> dict[str, Any]:
        return {"algorithm": "random", "status": "ok", "seed": self.seed}


class _EOFConn:
    """Pipe stand-in that simulates a worker closing before sending a payload."""

    def send(self, _payload: Any) -> None:
        """Accept parent commands without raising."""

    def poll(self, _timeout: float) -> bool:
        """Report a ready pipe so the parent immediately calls recv."""
        return True

    def recv(self) -> object:
        """Simulate the child process disappearing before a response is available."""
        raise EOFError

    def close(self) -> None:
        """Close hook used by the runner cleanup path."""


class _AliveProcess:
    """Process stand-in that keeps _PlannerStepProcess on the recv path."""

    def is_alive(self) -> bool:
        """Return true so the runner treats the worker as active before recv."""
        return True

    def join(self, timeout: float | None = None) -> None:
        """Join hook used by the runner cleanup path."""

    def terminate(self) -> None:
        """Terminate hook used by the runner cleanup path."""

    def kill(self) -> None:
        """Kill hook used by the runner cleanup path."""


@pytest.mark.skipif("fork" not in mp.get_all_start_methods(), reason="requires fork isolation")
def test_planner_step_timeout_fails_fast_and_reports_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A slow planner step should not block until the worker completes."""
    monkeypatch.setitem(baselines.BASELINES, "random", _SlowPlanner)
    monkeypatch.setattr(runner, "POLICY_STEP_TIMEOUT_SECS", 0.05)
    policy, metadata = runner._create_robot_policy("random", None, seed=123)

    start = time.monotonic()
    velocity = policy(
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.empty((0, 2)),
        0.1,
    )
    elapsed = time.monotonic() - start

    assert elapsed < 0.5
    assert velocity == pytest.approx(np.array([0.0, 0.0]))
    assert metadata["status"] == "policy_step_timeout_fallback"
    timeout_metadata = metadata["policy_step_timeout"]
    assert timeout_metadata["isolation"] == "process"
    assert timeout_metadata["step_timeout_s"] == 0.05
    assert timeout_metadata["step_timeouts"] == 1
    assert timeout_metadata["fallback_actions"] == 1


@pytest.mark.skipif("fork" not in mp.get_all_start_methods(), reason="requires fork isolation")
def test_planner_step_process_reports_eof_between_poll_and_recv() -> None:
    """A closed worker pipe should fail closed instead of leaking EOFError."""
    step_process = runner._PlannerStepProcess(object(), timeout_s=0.5)
    step_process._conn = _EOFConn()
    step_process._process = _AliveProcess()  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="exited before returning an action"):
        step_process.step({"obs": "value"})
