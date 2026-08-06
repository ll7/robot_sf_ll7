"""Guards for the process-trace diagnostic boundary."""

from __future__ import annotations

from robot_sf.analysis_workbench.interaction_coordinates import (
    build_worked_example_process_trace_from_export,
)
from robot_sf.analysis_workbench.simulation_trace_export import simulation_trace_export_from_dict


def test_process_trace_does_not_emit_canonical_metric_replacements() -> None:
    """Process traces are diagnostics, not new benchmark metric rows."""

    trace = simulation_trace_export_from_dict(
        {
            "schema_version": "simulation_trace_export.v1",
            "trace_id": "metric-boundary",
            "source": {
                "scenario_id": "metric-boundary",
                "seed": 1,
                "planner_id": "goal",
                "episode_id": "metric-boundary-episode",
                "generated_by": "unit-test fixture",
            },
            "evidence_boundary": "analysis_workbench_only",
            "coordinate_frame": "world",
            "units": {"position": "m", "heading": "rad", "time": "s", "velocity": "m/s"},
            "frames": [
                {
                    "step": 0,
                    "time_s": 0.0,
                    "robot": {
                        "position": [0.0, 0.0],
                        "heading": 0.0,
                        "velocity": [0.0, 0.0],
                        "radius": 0.25,
                    },
                    "pedestrians": [
                        {
                            "id": "ped-a",
                            "position": [1.0, 0.0],
                            "velocity": [0.0, 0.0],
                            "radius": 0.25,
                        }
                    ],
                    "planner": {
                        "selected_action": {"linear_velocity": 0.0, "angular_velocity": 0.0},
                        "event": "step",
                    },
                }
            ],
        }
    )

    payload = build_worked_example_process_trace_from_export(trace, focal_actor_id="ped-a")

    assert "metrics" not in payload
    assert "snqi" not in payload
    assert payload["claim_boundary"].startswith("Diagnostic renderer-neutral process quantities")
