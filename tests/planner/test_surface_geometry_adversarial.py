"""Adversarial continuous-point and metre-window geometry regressions."""

import numpy as np
import pytest

from robot_sf.planner.goal_target import ACTIVE_WAYPOINT_V2, select_goal_target
from robot_sf.planner.guarded_ppo import GuardedPPOAdapter, build_guarded_ppo_config
from robot_sf.planner.predictive_mppi import PredictiveMPPIAdapter, build_predictive_mppi_config
from robot_sf.planner.risk_dwa import RiskDWAPlannerAdapter, build_risk_dwa_config


@pytest.mark.parametrize(
    ("robot", "current", "next_waypoint"),
    [
        ((5.0, 5.0), (8.0, 5.0), (0.0, 0.0)),  # last waypoint
        ((8.0, 5.0), (8.0, 5.0), (8.0, 8.0)),  # exactly at active waypoint
        ((0.0, 0.0), (0.0, 0.0), (0.0, 0.0)),  # one-waypoint route at origin
    ],
)
def test_active_waypoint_boundary(robot, current, next_waypoint):
    actual = select_goal_target(
        np.asarray(robot),
        np.asarray(current),
        np.asarray(next_waypoint),
        version=ACTIVE_WAYPOINT_V2,
    )
    np.testing.assert_array_equal(actual, current)


@pytest.mark.parametrize("arm", ["risk_dwa", "predictive_mppi", "guarded_ppo"])
def test_surface_guard_sees_cell_inside_1_3_m_safety_boundary(arm):
    # Default 0.1 m grid. Cell 13 columns away begins 1.25 m from robot;
    # with a 1.0 m body, physical clearance is 0.25 m, below 0.30 m guard.
    grid = np.zeros((1, 51, 51), dtype=float)
    grid[0, 25, 38] = 1.0
    meta = {"resolution": [0.1], "channel_indices": [0, -1, -1, 0]}
    if arm == "risk_dwa":
        planner = RiskDWAPlannerAdapter(
            build_risk_dwa_config(
                {
                    "clearance_model": "surface_v2",
                    "robot_radius_m": 1.0,
                    "pedestrian_radius_m": 0.4,
                    "hard_obstacle_clearance": 0.30,
                }
            )
        )
    elif arm == "predictive_mppi":
        planner = PredictiveMPPIAdapter(
            build_predictive_mppi_config(
                {
                    "clearance_model": "surface_v2",
                    "predictive_clearance_model": "surface_v2",
                    "predictive_robot_radius": 1.0,
                    "predictive_pedestrian_radius": 0.4,
                }
            ),
            allow_fallback=True,
        )
    else:
        planner = GuardedPPOAdapter(
            build_guarded_ppo_config(
                {
                    "guard_clearance_model": "surface_v2",
                    "guard_robot_radius_m": 1.0,
                    "guard_pedestrian_radius_m": 0.4,
                }
            )
        )
    planner._world_to_grid = lambda *args, **kwargs: (25, 25)
    actual = planner._min_obstacle_clearance(np.asarray([2.5, 2.5]), grid_payload=(grid, meta))
    assert actual == pytest.approx(0.25)


@pytest.mark.parametrize("arm", ["risk_dwa", "predictive_mppi", "guarded_ppo"])
def test_release_grid_clearance_uses_true_robot_position_inside_cell(arm):
    # Map runner uses 0.2 m occupancy cells. The robot is near the top-right
    # edge of cell (12,12); the obstacle occupies cell (19,14).
    point = np.asarray([2.59, 2.59])
    grid = np.zeros((1, 51, 51), dtype=float)
    grid[0, 19, 14] = 1.0
    meta = {
        "resolution": [0.2],
        "origin": [0.0, 0.0],
        "size": [10.2, 10.2],
        "channel_indices": [0, -1, -1, 0],
        "use_ego_frame": [0.0],
    }
    if arm == "risk_dwa":
        planner = RiskDWAPlannerAdapter(
            build_risk_dwa_config(
                {
                    "clearance_model": "surface_v2",
                    "robot_radius_m": 1.0,
                    "pedestrian_radius_m": 0.4,
                    "hard_obstacle_clearance": 0.30,
                }
            )
        )
    elif arm == "predictive_mppi":
        planner = PredictiveMPPIAdapter(
            build_predictive_mppi_config(
                {
                    "clearance_model": "surface_v2",
                    "predictive_clearance_model": "surface_v2",
                    "predictive_robot_radius": 1.0,
                    "predictive_pedestrian_radius": 0.4,
                }
            ),
            allow_fallback=True,
        )
    else:
        planner = GuardedPPOAdapter(
            build_guarded_ppo_config(
                {
                    "guard_clearance_model": "surface_v2",
                    "guard_robot_radius_m": 1.0,
                    "guard_pedestrian_radius_m": 0.4,
                }
            )
        )
    assert planner._world_to_grid(point, meta, grid_shape=(51, 51)) == (12, 12)
    actual = planner._min_obstacle_clearance(point, grid_payload=(grid, meta))
    hand_distance = np.hypot(2.8 - 2.59, 3.8 - 2.59) - 1.0
    assert hand_distance < 0.30
    assert actual == pytest.approx(hand_distance)


@pytest.mark.parametrize("arm", ["risk_dwa", "predictive_mppi", "guarded_ppo"])
def test_grid_upper_edge_is_clamped_and_outside_is_occupied_characterization(arm):
    """Pin the existing discontinuity; this characterization claims no boundary repair."""
    if arm == "risk_dwa":
        planner = RiskDWAPlannerAdapter(build_risk_dwa_config({"clearance_model": "surface_v2"}))
    elif arm == "predictive_mppi":
        planner = PredictiveMPPIAdapter(
            build_predictive_mppi_config(
                {
                    "clearance_model": "surface_v2",
                    "predictive_robot_radius": 1.0,
                    "predictive_pedestrian_radius": 0.4,
                }
            )
        )
    else:
        planner = GuardedPPOAdapter(
            build_guarded_ppo_config(
                {
                    "guard_clearance_model": "surface_v2",
                    "guard_robot_radius_m": 1.0,
                    "guard_pedestrian_radius_m": 0.4,
                }
            )
        )
    grid = np.zeros((1, 2, 2))
    meta = {
        "resolution": [0.2],
        "origin": [0.0, 0.0],
        "size": [0.4, 0.4],
        "channel_indices": [0, -1, -1, 0],
        "use_ego_frame": [0.0],
    }
    edge = np.asarray([0.4, 0.2])
    outside = np.asarray([0.4 + 1e-9, 0.2])
    assert planner._world_to_grid(edge, meta, (2, 2)) == (1, 1)
    assert planner._world_to_grid(outside, meta, (2, 2)) is None
    assert planner._grid_value(edge, grid, meta, 0) == 0.0
    assert planner._grid_value(outside, grid, meta, 0) == 1.0
    assert planner._min_obstacle_clearance(edge, grid_payload=(grid, meta)) == float("inf")
    assert planner._min_obstacle_clearance(outside, grid_payload=(grid, meta)) == -1.0
    grid[0, 1, 1] = 1.0
    assert planner._grid_value(edge, grid, meta, 0) == 1.0
    assert planner._min_obstacle_clearance(edge, grid_payload=(grid, meta)) == -1.0
