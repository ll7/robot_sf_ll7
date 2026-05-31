"""Tests for scenario perturbation trace-response aggregation."""

from __future__ import annotations

import pytest

from scripts.validation.run_scenario_perturbation_trace_response import (
    TraceRun,
    build_trace_pair_summary,
    closest_approach_slice,
    summarize_trace_pairs,
)


def _frame(step: int, *, distance: float, clearance: float, goal_distance: float) -> dict:
    """Return a compact trace frame for aggregation tests."""
    return {
        "step": step,
        "time_s": step * 0.1,
        "goal_distance_m": goal_distance,
        "progress_m": 10.0 - goal_distance,
        "closest_pedestrian": {
            "pedestrian_index": 0,
            "pedestrian_position": [float(step), 0.0],
            "center_distance_m": distance,
            "clearance_m": clearance,
        },
    }


def test_closest_approach_slice_returns_window_around_minimum() -> None:
    """Trace slices should center on the minimum robot-pedestrian distance frame."""
    frames = [
        _frame(0, distance=2.0, clearance=0.6, goal_distance=9.0),
        _frame(1, distance=1.0, clearance=-0.4, goal_distance=8.5),
        _frame(2, distance=1.5, clearance=0.1, goal_distance=8.0),
    ]

    result = closest_approach_slice(frames, window=1)

    assert result["status"] == "ok"
    assert result["closest"]["step"] == 1
    assert [frame["step"] for frame in result["frames"]] == [0, 1, 2]


def test_build_trace_pair_summary_reports_closest_approach_deltas() -> None:
    """No-op and perturbed traces should compare closest-approach timing and clearance."""
    noop = TraceRun(
        planner="goal",
        scenario_id="demo_noop",
        source_scenario_id="demo",
        variant_id="demo_noop",
        family="noop",
        seed=111,
        termination_reason="max_steps",
        frames=[
            _frame(0, distance=2.0, clearance=0.6, goal_distance=9.0),
            _frame(1, distance=1.0, clearance=-0.4, goal_distance=8.5),
        ],
    )
    perturbed = TraceRun(
        planner="goal",
        scenario_id="demo_ped_offset",
        source_scenario_id="demo",
        variant_id="demo_ped_offset",
        family="pedestrian_route_offset",
        seed=111,
        termination_reason="max_steps",
        frames=[
            _frame(0, distance=2.5, clearance=1.1, goal_distance=9.2),
            _frame(1, distance=1.4, clearance=0.0, goal_distance=8.8),
        ],
    )

    row = build_trace_pair_summary(
        planner="goal",
        seed=111,
        noop=noop,
        perturbed=perturbed,
        window=0,
    )

    assert row["pair_status"] == "completed"
    assert row["closest_approach_delta"]["center_distance_m"] == pytest.approx(0.4)
    assert row["closest_approach_delta"]["clearance_m"] == pytest.approx(0.4)
    assert row["closest_approach_delta"]["goal_distance_m"] == pytest.approx(0.3)
    assert row["closest_approach_delta"]["progress_m"] == pytest.approx(-0.3)


def test_summarize_trace_pairs_groups_by_planner_without_recursive_summary() -> None:
    """Planner grouping should not recursively nest planner summaries."""
    rows = [
        {
            "planner": "goal",
            "pair_status": "completed",
            "closest_approach_delta": {"clearance_m": 0.4},
        },
        {
            "planner": "orca",
            "pair_status": "excluded",
            "closest_approach_delta": {"clearance_m": 0.1},
        },
    ]

    summary = summarize_trace_pairs(rows)

    assert summary["pairs"] == 2
    assert summary["status_counts"] == {"completed": 1, "excluded": 1}
    assert summary["mean_closest_approach_deltas_completed_pairs"]["clearance_m"] == pytest.approx(
        0.4
    )
    assert summary["by_planner"]["goal"]["pairs"] == 1
    assert "by_planner" not in summary["by_planner"]["goal"]
