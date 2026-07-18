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


def test_trace_xy_validation_errors() -> None:
    """Verify that _trace_xy rejects boolean, non-numeric, infinite, and incorrect length vectors."""
    # Test boolean in coordinates (line 255)
    with pytest.raises(ValueError, match="must be a finite numeric"):
        adapt_simulation_step_trace(
            {
                "schema_version": "simulation-step-trace.v1",
                "steps": [
                    {"robot": {"position": [True, 2.0], "velocity": [0.0, 0.0]}, "pedestrians": []}
                ],
            }
        )

    # Test non-numeric values (line 258-259)
    with pytest.raises(ValueError, match="must be a numeric"):
        adapt_simulation_step_trace(
            {
                "schema_version": "simulation-step-trace.v1",
                "steps": [
                    {"robot": {"position": ["abc", 2.0], "velocity": [0.0, 0.0]}, "pedestrians": []}
                ],
            }
        )

    # Test infinite values (line 262)
    with pytest.raises(ValueError, match="must be a finite numeric"):
        adapt_simulation_step_trace(
            {
                "schema_version": "simulation-step-trace.v1",
                "steps": [
                    {
                        "robot": {"position": [float("inf"), 2.0], "velocity": [0.0, 0.0]},
                        "pedestrians": [],
                    }
                ],
            }
        )

    # Test array fallback exceptions (line 265-270)
    with pytest.raises(ValueError, match="must be a finite numeric"):
        adapt_simulation_step_trace(
            {
                "schema_version": "simulation-step-trace.v1",
                "steps": [
                    {
                        "robot": {"position": [1.0, 2.0, 3.0], "velocity": [0.0, 0.0]},
                        "pedestrians": [],
                    }
                ],
            }
        )


def test_pedestrian_key_validation() -> None:
    """Verify that _pedestrian_key handles missing IDs and unhashable IDs."""
    # Test id fallback to index when "id" is absent (line 276)
    adapted = adapt_simulation_step_trace(
        {
            "schema_version": "simulation-step-trace.v1",
            "steps": [
                {
                    "robot": {"position": [0.0, 0.0], "velocity": [0.0, 0.0]},
                    "pedestrians": [{"position": [1.0, 1.0], "velocity": [0.0, 0.0]}],
                }
            ],
        }
    )
    assert adapted["pedestrian_ids"] == [0]

    # Test unhashable id (line 280-281)
    with pytest.raises(ValueError, match="must be hashable"):
        adapt_simulation_step_trace(
            {
                "schema_version": "simulation-step-trace.v1",
                "steps": [
                    {
                        "robot": {"position": [0.0, 0.0], "velocity": [0.0, 0.0]},
                        "pedestrians": [{"id": [1, 2], "position": [1.0, 1.0]}],
                    }
                ],
            }
        )


def test_adapter_various_validation_cases() -> None:
    """Verify other schema verification paths in adapt_simulation_step_trace."""
    # Test steps is not a list (line 322)
    with pytest.raises(ValueError, match="requires a list-valued steps field"):
        adapt_simulation_step_trace(
            {"schema_version": "simulation-step-trace.v1", "steps": "not-a-list"}
        )

    # Test empty steps list (line 328-329)
    adapted = adapt_simulation_step_trace(
        {"schema_version": "simulation-step-trace.v1", "steps": []}
    )
    assert adapted["robot_pos"] == []
    assert adapted["peds_pos"] == []

    # Test simulation step 0 is not a mapping (line 333)
    with pytest.raises(ValueError, match="step 0 must be a mapping"):
        adapt_simulation_step_trace(
            {"schema_version": "simulation-step-trace.v1", "steps": ["not-a-dict"]}
        )

    # Test simulation step 0 pedestrians is not a list (line 336)
    with pytest.raises(ValueError, match="step 0 pedestrians must be a list"):
        adapt_simulation_step_trace(
            {
                "schema_version": "simulation-step-trace.v1",
                "steps": [{"robot": {"position": [0.0, 0.0]}, "pedestrians": "not-a-list"}],
            }
        )

    # Test simulation step 0 pedestrian is not a mapping (line 342)
    with pytest.raises(ValueError, match="step 0 pedestrian 0 must be a mapping"):
        adapt_simulation_step_trace(
            {
                "schema_version": "simulation-step-trace.v1",
                "steps": [{"robot": {"position": [0.0, 0.0]}, "pedestrians": ["not-a-dict"]}],
            }
        )

    # Test duplicate pedestrian id in step 0 (line 345)
    with pytest.raises(ValueError, match="contains duplicate pedestrian ID"):
        adapt_simulation_step_trace(
            {
                "schema_version": "simulation-step-trace.v1",
                "steps": [
                    {
                        "robot": {"position": [0.0, 0.0]},
                        "pedestrians": [
                            {"id": 1, "position": [1.0, 1.0]},
                            {"id": 1, "position": [2.0, 2.0]},
                        ],
                    }
                ],
            }
        )

    # Test step_index step is not a mapping (line 358)
    with pytest.raises(ValueError, match="step 1 must be a mapping"):
        adapt_simulation_step_trace(
            {
                "schema_version": "simulation-step-trace.v1",
                "steps": [{"robot": {"position": [0.0, 0.0]}, "pedestrians": []}, "not-a-dict"],
            }
        )

    # Test step_index step robot is not a mapping (line 361)
    with pytest.raises(ValueError, match="step 1 robot must be a mapping"):
        adapt_simulation_step_trace(
            {
                "schema_version": "simulation-step-trace.v1",
                "steps": [
                    {"robot": {"position": [0.0, 0.0]}, "pedestrians": []},
                    {"robot": "not-a-dict"},
                ],
            }
        )

    # Test missing robot velocity (line 369)
    adapted = adapt_simulation_step_trace(
        {
            "schema_version": "simulation-step-trace.v1",
            "steps": [{"robot": {"position": [0.0, 0.0], "velocity": None}, "pedestrians": []}],
        }
    )
    assert "robot_vel" not in adapted

    # Test step_index step pedestrians is not a list (line 380)
    with pytest.raises(ValueError, match="step 1 pedestrians must be a list"):
        adapt_simulation_step_trace(
            {
                "schema_version": "simulation-step-trace.v1",
                "steps": [
                    {"robot": {"position": [0.0, 0.0]}, "pedestrians": []},
                    {"robot": {"position": [0.0, 0.0]}, "pedestrians": "not-a-list"},
                ],
            }
        )

    # Test step_index step pedestrian is not a mapping (line 384)
    with pytest.raises(ValueError, match="step 1 pedestrian 0 must be a mapping"):
        adapt_simulation_step_trace(
            {
                "schema_version": "simulation-step-trace.v1",
                "steps": [
                    {
                        "robot": {"position": [0.0, 0.0]},
                        "pedestrians": [{"id": 1, "position": [1.0, 1.0]}],
                    },
                    {"robot": {"position": [0.0, 0.0]}, "pedestrians": ["not-a-dict"]},
                ],
            }
        )

    # Test duplicate pedestrian id in step_index (line 389)
    with pytest.raises(ValueError, match="step 1 contains duplicate pedestrian ID"):
        adapt_simulation_step_trace(
            {
                "schema_version": "simulation-step-trace.v1",
                "steps": [
                    {
                        "robot": {"position": [0.0, 0.0]},
                        "pedestrians": [{"id": 1, "position": [1.0, 1.0]}],
                    },
                    {
                        "robot": {"position": [0.0, 0.0]},
                        "pedestrians": [
                            {"id": 1, "position": [1.0, 1.0]},
                            {"id": 1, "position": [2.0, 2.0]},
                        ],
                    },
                ],
            }
        )

    # Test missing pedestrian velocity (line 410)
    adapted = adapt_simulation_step_trace(
        {
            "schema_version": "simulation-step-trace.v1",
            "steps": [
                {
                    "robot": {"position": [0.0, 0.0]},
                    "pedestrians": [{"id": 1, "position": [1.0, 1.0], "velocity": None}],
                }
            ],
        }
    )
    assert "ped_vel" not in adapted


def test_load_config_defaults_and_errors() -> None:
    """Verify load_config defaults and anchor validation errors."""
    # Test line 222: config_dict is None, path is None -> cfg = {}
    from robot_sf.benchmark.critical_intervals import load_config

    assert load_config(None, config_dict=None) == {}

    # Test line 232: anchor spec is not a dict
    with pytest.raises(ValueError, match="must map to a dict"):
        load_config(config_dict={"critical_intervals": {"collision_or_near_miss": "not-a-dict"}})


def test_adapt_trace_other_cases() -> None:
    """Verify numpy array inputs, conversion failures, and schema version bypass."""
    # Test line 270: array fallback success
    import numpy as np

    adapted = adapt_simulation_step_trace(
        {
            "schema_version": "simulation-step-trace.v1",
            "steps": [
                {
                    "robot": {"position": np.array([1.0, 2.0]), "velocity": np.array([0.0, 0.0])},
                    "pedestrians": [],
                }
            ],
        }
    )
    assert adapted["robot_pos"] == [[1.0, 2.0]]

    # Test line 266-267: np.asarray failure
    with pytest.raises(ValueError, match="must be a numeric"):
        adapt_simulation_step_trace(
            {
                "schema_version": "simulation-step-trace.v1",
                "steps": [
                    {
                        "robot": {"position": {"x": 1, "y": 2}, "velocity": [0.0, 0.0]},
                        "pedestrians": [],
                    }
                ],
            }
        )

    # Test line 320: schema version mismatch and steps is not a list
    trace = {"schema_version": "unknown"}
    assert adapt_simulation_step_trace(trace) is trace


def test_get_trace_arrays_edge_cases() -> None:
    """Verify _get_trace_arrays edge cases for empty, 1D, 2D input arrays and invalid dt values."""
    from robot_sf.benchmark.critical_intervals import _get_trace_arrays

    # Test line 445: empty robot_pos
    robot_pos, peds_pos, dt = _get_trace_arrays({"robot_pos": []})
    assert robot_pos.shape == (0, 2)

    # Test line 447: 1D robot_pos
    robot_pos, peds_pos, dt = _get_trace_arrays({"robot_pos": [1.0, 2.0]})
    assert robot_pos.shape == (1, 2)

    # Test line 455: 2D peds_pos
    robot_pos, peds_pos, dt = _get_trace_arrays(
        {"robot_pos": [[0.0, 0.0]], "peds_pos": [[1.0, 2.0]]}
    )
    assert peds_pos.shape == (1, 1, 2)

    # Test line 459: invalid dt
    robot_pos, peds_pos, dt = _get_trace_arrays({"dt": -1.0})
    assert dt == 0.1
