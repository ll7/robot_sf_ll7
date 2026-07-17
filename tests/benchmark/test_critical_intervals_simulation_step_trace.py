"""Regression tests for ``simulation-step-trace.v1`` critical-interval adaptation."""

from __future__ import annotations

import pytest

from robot_sf.benchmark.critical_intervals import (
    ANCHOR_SOURCE_STEP_EVENT,
    STEP_NEAR_MISS_EVENTS_FIELD,
    adapt_simulation_step_trace,
    extract_critical_intervals,
    summarize_interval_metrics,
)


def _nested_trace() -> dict[str, object]:
    """Return a tiny current re-export trace with pedestrian order changes."""
    steps: list[dict[str, object]] = []
    for index in range(3):
        pedestrians = [
            {
                "id": 7,
                "position": [4.0, 0.0],
                "velocity": [0.0, 0.0],
            },
            {
                "id": 2,
                "position": [10.0, 0.0],
                "velocity": [0.0, 0.0],
            },
        ]
        if index == 1:
            pedestrians.reverse()
        steps.append(
            {
                "step": index,
                "time_s": (index + 1) * 0.1,
                "robot": {
                    "position": [float(index), 0.0],
                    "velocity": [1.0, 0.0],
                },
                "pedestrians": pedestrians,
            }
        )
    return {
        "schema_version": "simulation-step-trace.v1",
        "dt": 0.1,
        "steps": steps,
    }


def test_adapt_simulation_step_trace_converts_nested_rows_by_pedestrian_id() -> None:
    """The adapter should make current re-export rows columnar without order drift."""
    adapted = adapt_simulation_step_trace(_nested_trace())

    assert adapted["robot_pos"] == [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    assert adapted["robot_vel"] == [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    assert adapted["pedestrian_ids"] == [7, 2]
    assert adapted["peds_pos"][1] == [[4.0, 0.0], [10.0, 0.0]]
    assert adapted["ped_vel"][1] == [[0.0, 0.0], [0.0, 0.0]]


def test_nested_simulation_step_trace_works_end_to_end() -> None:
    """Public extraction and summary APIs should adapt the current schema automatically."""
    trace = _nested_trace()
    config = {
        "critical_intervals": {
            "closest_approach": {
                "enabled": True,
                "before_s": 0.1,
                "after_s": 0.1,
            }
        }
    }

    intervals = extract_critical_intervals(trace, config)
    report = summarize_interval_metrics(trace, intervals)

    assert intervals[0].status == "available"
    assert intervals[0].anchor_step == 2
    assert report.whole_run["n_steps"] == 3
    assert report.whole_run["min_distance_m"] == pytest.approx(2.0)


def test_adapter_preserves_legacy_columnar_trace() -> None:
    """Existing columnar callers should retain their exact trace mapping."""
    trace = {
        "robot_pos": [[0.0, 0.0]],
        "peds_pos": [[[1.0, 0.0]]],
        "robot_vel": [[0.0, 0.0]],
        "dt": 0.1,
    }

    assert adapt_simulation_step_trace(trace) is trace


def test_adapter_rejects_changing_pedestrian_id_sets() -> None:
    """The thin adapter should fail closed instead of imputing missing actors."""
    trace = _nested_trace()
    steps = trace["steps"]
    assert isinstance(steps, list)
    step = steps[-1]
    assert isinstance(step, dict)
    step["pedestrians"] = []

    with pytest.raises(ValueError, match="stable pedestrian IDs"):
        adapt_simulation_step_trace(trace)


def test_adapter_rejects_unknown_nested_trace_schema() -> None:
    """Nested rows from an unknown schema must not be silently reinterpreted."""
    trace = _nested_trace()
    trace["schema_version"] = "simulation-step-trace.v2"

    with pytest.raises(ValueError, match="Unknown schema_version"):
        adapt_simulation_step_trace(trace)


def test_real_trace_step_near_miss_events_anchor() -> None:
    """Issue #5884: verify that a real trace adapted to columnar can anchor from step_near_miss_events."""
    import json
    from pathlib import Path

    evidence_path = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "context"
        / "evidence"
        / "issue_4891_head_on_corridor_exemplars_2026-07"
        / "social_force"
        / "classic_head_on_corridor_medium_seed24_worst"
        / "trace_series.json"
    )

    with open(evidence_path, encoding="utf-8") as f:
        payload = json.load(f)

    # Wrap the real frames into the simulation-step-trace.v1 schema format.
    trace = {
        "schema_version": "simulation-step-trace.v1",
        "dt": 0.1,
        "steps": payload["frames"],
    }

    # Verify that the adapter can process it cleanly.
    adapted = adapt_simulation_step_trace(trace)
    T = len(adapted["robot_pos"])
    assert T > 0

    # Inject step_near_miss_events: let's put one at step 5
    events = [None] * T
    events[5] = {
        "step": 5,
        "metric": "surface_clearance",
        "near_miss": True,
    }
    trace[STEP_NEAR_MISS_EVENTS_FIELD] = events

    # Configure the collision_or_near_miss anchor.
    config = {
        "critical_intervals": {
            "collision_or_near_miss": {
                "enabled": True,
                "before_s": 0.5,
                "after_s": 0.5,
            }
        }
    }

    intervals = extract_critical_intervals(trace, config)
    assert len(intervals) == 1
    iv = intervals[0]
    assert iv.status == "available"
    assert iv.anchor_step == 5
    assert iv.source == ANCHOR_SOURCE_STEP_EVENT
