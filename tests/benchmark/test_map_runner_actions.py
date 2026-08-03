"""Tests for robot_sf.benchmark.map_runner_actions — action and kinematics helpers."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.benchmark.map_runner_actions import (
    DEFAULT_KINEMATICS,
    command_xy_payload,
    robot_kinematics_label,
    robot_max_speed,
    scenario_robot_kinematics_label,
    stack_ped_positions,
    vel_and_acc,
)


class TestRobotKinematicsLabel:
    """Tests for robot_kinematics_label config inference."""

    def test_no_robot_config_returns_default(self) -> None:
        """A config without robot_config must return the default kinematics."""

        class BareConfig:
            pass

        assert robot_kinematics_label(BareConfig()) == DEFAULT_KINEMATICS

    def test_differential_drive_detected(self) -> None:
        """A DifferentialDriveConfig class name must map to differential_drive."""

        class DifferentialDriveRobotConfig:
            pass

        class Cfg:
            robot_config = DifferentialDriveRobotConfig()

        assert robot_kinematics_label(Cfg()) == "differential_drive"

    def test_bicycle_drive_detected(self) -> None:
        """A BicycleModelConfig class name must map to bicycle_drive."""

        class BicycleModelRobotConfig:
            pass

        class Cfg:
            robot_config = BicycleModelRobotConfig()

        assert robot_kinematics_label(Cfg()) == "bicycle_drive"

    def test_holonomic_detected(self) -> None:
        """A HolonomicRobotConfig class name must map to holonomic."""

        class HolonomicRobotConfig:
            pass

        class Cfg:
            robot_config = HolonomicRobotConfig()

        assert robot_kinematics_label(Cfg()) == "holonomic"

    def test_omni_detected_as_holonomic(self) -> None:
        """An OmniDirectionalConfig class name must map to holonomic."""

        class OmniDirectionalRobotConfig:
            pass

        class Cfg:
            robot_config = OmniDirectionalRobotConfig()

        assert robot_kinematics_label(Cfg()) == "holonomic"


class TestRobotMaxSpeed:
    """Tests for robot_max_speed extraction."""

    def test_no_robot_config_returns_none(self) -> None:
        """A config without robot_config must return None."""

        class BareConfig:
            pass

        assert robot_max_speed(BareConfig()) is None

    def test_max_linear_speed_extracted(self) -> None:
        """max_linear_speed must be extracted as a positive float."""

        class RobotCfg:
            max_linear_speed = 1.5

        class Cfg:
            robot_config = RobotCfg()

        assert robot_max_speed(Cfg()) == pytest.approx(1.5)

    def test_max_velocity_fallback(self) -> None:
        """max_velocity must be used when max_linear_speed is absent."""

        class RobotCfg:
            max_velocity = 2.0

        class Cfg:
            robot_config = RobotCfg()

        assert robot_max_speed(Cfg()) == pytest.approx(2.0)

    def test_zero_speed_returns_none(self) -> None:
        """A zero speed must return None (not positive)."""

        class RobotCfg:
            max_linear_speed = 0.0

        class Cfg:
            robot_config = RobotCfg()

        assert robot_max_speed(Cfg()) is None

    def test_negative_speed_returns_none(self) -> None:
        """A negative speed must return None."""

        class RobotCfg:
            max_linear_speed = -1.0

        class Cfg:
            robot_config = RobotCfg()

        assert robot_max_speed(Cfg()) is None


class TestScenarioRobotKinematicsLabel:
    """Tests for scenario_robot_kinematics_label."""

    def test_no_robot_config_returns_default(self) -> None:
        """A scenario without robot_config must return the default."""
        assert scenario_robot_kinematics_label({}) == DEFAULT_KINEMATICS

    def test_bicycle_type_detected(self) -> None:
        """A bicycle type must map to bicycle_drive."""
        scenario = {"robot_config": {"type": "bicycle_model"}}
        assert scenario_robot_kinematics_label(scenario) == "bicycle_drive"

    def test_holonomic_type_detected(self) -> None:
        """A holonomic type must map to holonomic."""
        scenario = {"robot_config": {"type": "holonomic"}}
        assert scenario_robot_kinematics_label(scenario) == "holonomic"

    def test_omni_type_detected(self) -> None:
        """An omni type must map to holonomic."""
        scenario = {"robot_config": {"type": "omnidirectional"}}
        assert scenario_robot_kinematics_label(scenario) == "holonomic"

    def test_differential_type_detected(self) -> None:
        """A differential type must map to differential_drive."""
        scenario = {"robot_config": {"type": "differential_drive"}}
        assert scenario_robot_kinematics_label(scenario) == "differential_drive"

    def test_empty_type_returns_default(self) -> None:
        """An empty type string must return the default."""
        scenario = {"robot_config": {"type": ""}}
        assert scenario_robot_kinematics_label(scenario) == DEFAULT_KINEMATICS

    def test_model_key_fallback(self) -> None:
        """The model key must be used when type is absent."""
        scenario = {"robot_config": {"model": "bicycle_v2"}}
        assert scenario_robot_kinematics_label(scenario) == "bicycle_drive"


class TestVelAndAcc:
    """Tests for vel_and_acc finite-difference computation."""

    def test_constant_velocity(self) -> None:
        """Constant-velocity trajectory must have near-zero acceleration."""
        dt = 0.1
        positions = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
        vel, acc = vel_and_acc(positions, dt)
        assert vel.shape == positions.shape
        assert acc.shape == positions.shape
        np.testing.assert_allclose(vel[:, 0], 10.0, atol=1e-10)
        np.testing.assert_allclose(acc, 0.0, atol=1e-10)

    def test_single_point_returns_zeros(self) -> None:
        """A single-point trajectory must return zero velocity and acceleration."""
        positions = np.array([[1.0, 2.0]])
        vel, acc = vel_and_acc(positions, 0.1)
        np.testing.assert_array_equal(vel, np.zeros_like(positions))
        np.testing.assert_array_equal(acc, np.zeros_like(positions))

    def test_output_shape_matches_input(self) -> None:
        """Output shapes must match the input positions shape."""
        positions = np.random.default_rng(42).standard_normal((10, 2))
        vel, acc = vel_and_acc(positions, 0.05)
        assert vel.shape == (10, 2)
        assert acc.shape == (10, 2)


class TestStackPedPositions:
    """Tests for stack_ped_positions padding."""

    def test_empty_trajectory(self) -> None:
        """An empty trajectory must return shape (0, 0, 2)."""
        result = stack_ped_positions([])
        assert result.shape == (0, 0, 2)

    def test_uniform_shapes_stacked(self) -> None:
        """Uniform pedestrian counts must stack without padding."""
        traj = [np.array([[1.0, 2.0], [3.0, 4.0]]) for _ in range(5)]
        result = stack_ped_positions(traj)
        assert result.shape == (5, 2, 2)

    def test_variable_shapes_padded(self) -> None:
        """Variable pedestrian counts must be padded to the maximum."""
        traj = [
            np.array([[1.0, 2.0]]),
            np.array([[3.0, 4.0], [5.0, 6.0]]),
            np.array([[7.0, 8.0]]),
        ]
        result = stack_ped_positions(traj)
        assert result.shape == (3, 2, 2)
        assert np.isnan(result[0, 1, 0])

    def test_empty_arrays_handled(self) -> None:
        """Empty pedestrian arrays must be handled without error."""
        traj = [np.zeros((0, 2)), np.array([[1.0, 2.0]])]
        result = stack_ped_positions(traj)
        assert result.shape == (2, 1, 2)


class TestCommandXyPayload:
    """Tests for command_xy_payload extraction."""

    def test_tuple_command(self) -> None:
        """A tuple command must produce a two-element array."""
        result = command_xy_payload((1.5, -0.5))
        np.testing.assert_array_equal(result, [1.5, -0.5])

    def test_dict_command(self) -> None:
        """A dict command with vx/vy must produce a two-element array."""
        result = command_xy_payload({"vx": 2.0, "vy": -1.0})
        np.testing.assert_array_equal(result, [2.0, -1.0])

    def test_dict_command_defaults_to_zero(self) -> None:
        """A dict command without vx/vy must default to zeros."""
        result = command_xy_payload({})
        np.testing.assert_array_equal(result, [0.0, 0.0])
