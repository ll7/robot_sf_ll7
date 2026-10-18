"""Prediction execution diagnostics across the adapter binding lifecycle."""

from types import SimpleNamespace

import pytest

from robot_sf.planner.socnav_prediction import PredictionPlannerAdapter
from robot_sf.robot.differential_drive import DifferentialDriveSettings


@pytest.mark.parametrize("initialized", [False, True])
def test_prediction_diagnostics_unbound(initialized):
    """Protocol probes and constructed adapters report the same unbound state."""
    adapter = (
        PredictionPlannerAdapter()
        if initialized
        else PredictionPlannerAdapter.__new__(PredictionPlannerAdapter)
    )
    assert adapter.diagnostics() == {
        "planner_type": "PredictionPlannerAdapter",
        "prediction_execution_contract": "unbound_command_rollout_v1",
        "static_geometry_bound": False,
    }


def test_prediction_diagnostics_tracks_environment_binding():
    """Native execution and static geometry labels follow binding and rebinding."""
    adapter = PredictionPlannerAdapter()
    adapter.bind_env(
        SimpleNamespace(
            simulator=SimpleNamespace(
                robots=[SimpleNamespace(config=DifferentialDriveSettings())],
                iter_obstacle_segments=lambda: [((0.0, 0.0), (1.0, 0.0))],
            )
        )
    )
    assert adapter.diagnostics() == {
        "planner_type": "PredictionPlannerAdapter",
        "prediction_execution_contract": "native_motion_static_footprint_v2",
        "static_geometry_bound": True,
    }
    adapter.bind_env(SimpleNamespace())
    assert adapter.diagnostics() == {
        "planner_type": "PredictionPlannerAdapter",
        "prediction_execution_contract": "unbound_command_rollout_v1",
        "static_geometry_bound": False,
    }
