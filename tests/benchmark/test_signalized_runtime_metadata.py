"""Runtime signal-state metadata propagation tests."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from robot_sf.benchmark.full_classic.orchestrator import (
    _compute_episode_metrics,
    _episode_metadata_for_metrics,
)
from robot_sf.benchmark.metrics import (
    ROLLOVER_CRITICAL_EVENT,
    EpisodeData,
    compute_all_metrics,
    evaluate_stability_margin,
)


def _scenario_with_signal_state(signal_state: dict) -> SimpleNamespace:
    """Return a minimal scenario descriptor with raw signal-state metadata."""
    return SimpleNamespace(
        raw={
            "metadata": {
                "signal_state": signal_state,
            },
        },
        map_path="",
    )


def _scenario_with_rollover_stability(config: dict) -> SimpleNamespace:
    """Return a minimal scenario descriptor with TWV rollover instrumentation metadata."""
    return SimpleNamespace(raw={"metadata": {"rollover_stability": config}}, map_path="")


def _rollover_episode(*, steps: int = 4) -> EpisodeData:
    """Return a minimal no-pedestrian episode for rollover metric tests."""
    return EpisodeData(
        robot_pos=np.zeros((steps, 2), dtype=float),
        robot_vel=np.zeros((steps, 2), dtype=float),
        robot_acc=np.zeros((steps, 2), dtype=float),
        peds_pos=np.zeros((steps, 0, 2), dtype=float),
        ped_forces=np.zeros((steps, 0, 2), dtype=float),
        goal=np.array([1.0, 0.0], dtype=float),
        dt=1.0,
        reached_goal_step=None,
    )


def test_rollover_stability_margin_handles_invalid_inputs() -> None:
    """Invalid TWV geometry should fail closed, while invalid samples return NaN."""
    assert np.isnan(evaluate_stability_margin(float("nan"), 1.0))
    with pytest.raises(ValueError, match="t_w"):
        evaluate_stability_margin(1.0, 1.0, t_w=0.0)


def test_rollover_metrics_emit_critical_event_from_explicit_yaw_rate() -> None:
    """Enabled TWV metadata should produce campaign-table rollover counters."""
    episode = _rollover_episode()
    episode.robot_vel[:, 0] = 2.0
    episode.episode_metadata = {"rollover_stability": {"enabled": True, "yaw_rate": 3.0}}

    metrics = compute_all_metrics(episode, horizon=10, shortest_path_len=0.0)

    assert metrics["rollover_critical"] == 1.0
    assert metrics["rollover_critical_count"] == 4.0
    assert metrics["rollover_event"] == ROLLOVER_CRITICAL_EVENT


def test_rollover_metrics_can_estimate_yaw_rate_from_velocity_heading() -> None:
    """Absent explicit yaw rate, velocity heading changes should still feed the proxy."""
    episode = _rollover_episode(steps=3)
    episode.robot_vel[:] = np.array([[2.0, 0.0], [0.0, 2.0], [0.0, 2.0]], dtype=float)
    episode.episode_metadata = {"rollover_stability": {"enabled": True}}

    metrics = compute_all_metrics(episode, horizon=10, shortest_path_len=0.0)

    assert metrics["rollover_lateral_accel_abs_max"] == pytest.approx(np.pi)
    assert metrics["rollover_critical_count"] >= 1


def test_rollover_metrics_report_empty_enabled_episode() -> None:
    """Enabled instrumentation should return explicit empty counters for empty trajectories."""
    episode = _rollover_episode(steps=0)
    episode.episode_metadata = {"rollover_stability": {"enabled": True, "yaw_rate": 3.0}}

    metrics = compute_all_metrics(episode, horizon=10, shortest_path_len=0.0)

    assert metrics["rollover_stability_enabled"] == 1.0
    assert metrics["rollover_critical_count"] == 0.0
    assert np.isnan(metrics["rollover_min_stability_margin"])


def test_runtime_metadata_carries_enabled_rollover_stability_config() -> None:
    """Full-classic metrics should receive opt-in TWV rollover instrumentation metadata."""
    config = {"enabled": True, "yaw_rate": 3.0, "t_w": 0.8}

    metadata = _episode_metadata_for_metrics(_scenario_with_rollover_stability(config))

    assert metadata == {"rollover_stability": config}


def test_runtime_metadata_omits_disabled_rollover_stability_config() -> None:
    """Default-disabled TWV rollover metadata should not alter benchmark metrics."""
    metadata = _episode_metadata_for_metrics(
        _scenario_with_rollover_stability({"enabled": False, "yaw_rate": 3.0})
    )

    assert metadata is None


def test_runtime_signal_metadata_keeps_proxy_rows_excluded() -> None:
    """Proxy signal metadata should remain excluded from metric denominators."""
    scenario = _scenario_with_signal_state(
        {
            "schema_version": "signal-state-proxy.v1",
            "status": "proxy_diagnostic_only",
            "observation_mode": "trace_metadata_only",
            "planner_observable": False,
            "benchmark_evidence": False,
        }
    )

    metadata = _episode_metadata_for_metrics(scenario)

    assert metadata == {
        "signal_state": {
            "contract_state": "proxy_diagnostic",
            "benchmark_evidence": False,
        }
    }


def test_runtime_signal_metadata_promotes_explicit_observable_fields() -> None:
    """Observable signal metadata should expose metric fields to EpisodeData."""
    signal_state = {
        "schema_version": "signal-state-observable.v1",
        "status": "planner_observable_signal_state",
        "observation_mode": "planner_observable",
        "planner_observable": True,
        "benchmark_evidence": True,
        "timeline": [{"state": "red", "duration": 1.0}],
        "stop_line": [[0.0, -1.0], [0.0, 1.0]],
        "crosswalk_polygon": [[0.0, -1.0], [2.0, -1.0], [2.0, 1.0], [0.0, 1.0]],
    }

    metadata = _episode_metadata_for_metrics(_scenario_with_signal_state(signal_state))

    assert metadata is not None
    assert metadata["signal_state"]["contract_state"] == "planner_observable"
    assert metadata["signal_state"]["benchmark_evidence"] is True
    assert metadata["signal_state"]["timeline"] == signal_state["timeline"]
    assert metadata["signal_state"]["stop_line"] == signal_state["stop_line"]
    assert metadata["signal_state"]["crosswalk_polygon"] == signal_state["crosswalk_polygon"]


def test_real_episode_metrics_include_observable_signal_denominator() -> None:
    """The full-classic metrics bridge should produce runtime signal denominators."""
    scenario = _scenario_with_signal_state(
        {
            "schema_version": "signal-state-observable.v1",
            "status": "planner_observable_signal_state",
            "observation_mode": "planner_observable",
            "planner_observable": True,
            "benchmark_evidence": True,
            "timeline": [
                {"state": "red", "duration": 0.1},
                {"state": "green", "duration": 1.0},
            ],
            "stop_line": [[0.0, -1.0], [0.0, 1.0]],
            "crosswalk_polygon": [[0.0, -1.0], [2.0, -1.0], [2.0, 1.0], [0.0, 1.0]],
        }
    )
    metrics = _compute_episode_metrics(
        SimpleNamespace(job_id="job", scenario_id="signalized_observable_smoke", seed=2799),
        scenario,
        SimpleNamespace(snqi_weights_path=None),
        robot_pos=np.array([[-1.0, 0.0], [-0.5, 0.0], [0.5, 0.0], [1.5, 0.0]]),
        robot_vel=np.zeros((4, 2)),
        robot_acc=np.zeros((4, 2)),
        ped_pos=np.zeros((4, 0, 2)),
        ped_forces=np.zeros((4, 0, 2)),
        dt=0.1,
        reached_goal_step=3,
        goal=np.array([1.5, 0.0]),
        horizon=4,
        robot_radius=0.3,
        ped_radius=0.3,
    )

    assert metrics["signal_metrics_denominator"] == 1
    assert metrics["signal_unavailable_exclusion_count"] == 0
    assert metrics["signal_stop_line_crossings_under_red"] == 0
    assert metrics["signal_metrics_evidence"] == {
        "state": "planner_observable",
        "exclusion_reason": "",
    }
