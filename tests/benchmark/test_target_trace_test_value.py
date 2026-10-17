"""Guard runtime sensitivity of the sentinel regression (a testing-only RV4 fix)."""

import runpy
from pathlib import Path

import pytest

from robot_sf.benchmark.map_runner import map_runner_episode


def test_sentinel_regression_rejects_missing_live_telemetry(monkeypatch):
    """The target regression must reject ablated live telemetry.

    Execute the regression in a fresh namespace and restore the decoder with
    monkeypatch. The reviewed local selector test passed this ablation; the
    replacement uses the unchanged real canary and planner configs.
    """
    namespace = runpy.run_path(str(Path("tests/benchmark/test_planner_target_trace.py")))
    regression = namespace["test_historical_selector_yields_goal_next_for_sentinel_next"]
    monkeypatch.setattr(map_runner_episode, "_planner_target_xy_from_stats", lambda payload: None)
    with pytest.raises(AssertionError, match="missing or incorrect live planner target telemetry"):
        regression()
