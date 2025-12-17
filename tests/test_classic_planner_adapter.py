"""Tests for classic planner integration adapters."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.nav.navigation import sample_route
from robot_sf.nav.svg_map_parser import convert_map
from robot_sf.planner.classic_global_planner import ClassicPlannerConfig
from robot_sf.planner.classic_planner_adapter import (
    PlannerActionAdapter,
    attach_classic_global_planner,
)
from robot_sf.robot.bicycle_drive import BicycleDriveRobot, BicycleDriveSettings
from robot_sf.robot.differential_drive import DifferentialDriveRobot, DifferentialDriveSettings


def _make_basic_map(tmp_path):
    svg = tmp_path / "planner_adapter_basic.svg"
    svg.write_text(
        """
<svg xmlns="http://www.w3.org/2000/svg" xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" width="6" height="3">
  <rect inkscape:label="robot_spawn_zone" x="0.2" y="0.2" width="0.5" height="0.5" />
  <rect inkscape:label="robot_goal_zone" x="4.8" y="0.2" width="0.5" height="0.5" />
  <path inkscape:label="robot_route_0_0" d="M 0.2 0.2 L 5.0 0.2" />
</svg>
        """.strip()
    )
    return convert_map(str(svg))


def test_attach_classic_global_planner_routes_via_planner(tmp_path):
    """sample_route should call the attached classic planner when enabled."""
    map_def = _make_basic_map(tmp_path)
    planner = attach_classic_global_planner(map_def, ClassicPlannerConfig(inflate_radius_cells=0))

    route = sample_route(map_def, spawn_id=0)

    assert map_def._use_planner is True
    assert route, "Planner-backed route should not be empty"
    assert planner is map_def._global_planner
    assert len(route) >= 2  # spawn + goal from planned path


def test_planner_action_adapter_bicycle_conversion(tmp_path):
    """Bicycle action adapter should compute accel/steer within limits."""
    robot = BicycleDriveRobot(BicycleDriveSettings(max_accel=1.0, wheelbase=1.0, max_steer=0.5))
    adapter = PlannerActionAdapter(
        robot=robot,
        action_space=robot.action_space,
        time_step=0.1,
    )

    action = adapter.from_velocity_command((1.0, 0.4))

    # Accel should saturate at max_accel and steer within max_steer
    assert action.shape == (2,)
    assert action[0] == pytest.approx(robot.config.max_accel)
    expected_steer = np.clip(np.arctan(0.4 * robot.config.wheelbase / 1.0), -0.5, 0.5)
    assert action[1] == pytest.approx(expected_steer)
    assert np.all(action <= robot.action_space.high)
    assert np.all(action >= robot.action_space.low)


def test_planner_action_adapter_differential_conversion():
    """Differential action adapter should return velocity deltas clipped to space."""
    robot = DifferentialDriveRobot(
        DifferentialDriveSettings(max_linear_speed=1.0, max_angular_speed=0.5)
    )
    robot.state.velocity = (0.3, 0.1)
    adapter = PlannerActionAdapter(
        robot=robot,
        action_space=robot.action_space,
        time_step=0.1,
    )

    action = adapter.from_velocity_command((0.6, -0.4))

    assert action.shape == (2,)
    # Linear delta respects current speed; angular delta should clip to bounds
    assert action[0] == pytest.approx(0.3)
    assert action[1] == pytest.approx(robot.action_space.low[1])
    assert np.all(action <= robot.action_space.high)
    assert np.all(action >= robot.action_space.low)
