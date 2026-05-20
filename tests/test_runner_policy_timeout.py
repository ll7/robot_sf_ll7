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
