"""Tests for the VisualizableSimState class."""

from unittest.mock import patch

import numpy as np
import pytest

from robot_sf.render.sim_view import VisualizableSimState


class TestVisualizableSimState:
    # Remove the __init__ method entirely

    @pytest.fixture
    def setup_state_data(self):
        """Fixture to provide test data for VisualizableSimState."""
        return {
            "timestep": 0,
            "robot_pose": ((0.0, 0.0), 0.0),  # (position, orientation)
            "pedestrian_positions": np.array([[1.0, 1.0], [2.0, 2.0]]),
            "ray_vecs": np.array([[[0.0, 0.0], [1.0, 1.0]], [[0.0, 0.0], [2.0, 0.0]]]),
            "ped_actions": np.array([[[1.0, 1.0], [1.5, 1.5]], [[2.0, 2.0], [2.5, 2.5]]]),
            "robot_action": None,
        }

    @patch("robot_sf.render.sim_view.logger")
    def test_post_init_none_time_per_step(self, mock_logger, setup_state_data):
        """Test that __post_init__ warns and
        sets default value when time_per_step_in_secs is None."""
        data = setup_state_data
        state = VisualizableSimState(
            timestep=data["timestep"],
            robot_action=data["robot_action"],
            robot_pose=data["robot_pose"],
            pedestrian_positions=data["pedestrian_positions"],
            ray_vecs=data["ray_vecs"],
            ped_actions=data["ped_actions"],
            time_per_step_in_secs=None,
        )

        # Check warning was logged
        mock_logger.warning.assert_called_once_with(
            "time_per_step_in_secs is None, defaulting to 0.1s.",
        )

        # Check default value was set
        assert state.time_per_step_in_secs == 0.1

    @patch("robot_sf.render.sim_view.logger")
    def test_post_init_with_value(self, mock_logger, setup_state_data):
        """Test that __post_init__ doesn't modify time_per_step_in_secs when it has a value."""
        data = setup_state_data
        state = VisualizableSimState(
            timestep=data["timestep"],
            robot_action=data["robot_action"],
            robot_pose=data["robot_pose"],
            pedestrian_positions=data["pedestrian_positions"],
            ray_vecs=data["ray_vecs"],
            ped_actions=data["ped_actions"],
            time_per_step_in_secs=0.2,  # Different from default
        )

        # Check no warning was logged
        mock_logger.warning.assert_not_called()

        # Check value wasn't modified
        assert state.time_per_step_in_secs == 0.2

    def test_post_init_with_default_value(self, setup_state_data):
        """Test the behavior when time_per_step_in_secs uses its default."""
        # Create state with default value
        data = setup_state_data
        state = VisualizableSimState(
            timestep=data["timestep"],
            robot_action=data["robot_action"],
            robot_pose=data["robot_pose"],
            pedestrian_positions=data["pedestrian_positions"],
            ray_vecs=data["ray_vecs"],
            ped_actions=data["ped_actions"],
        )

        # Default is None, which should be converted to 0.1
        assert state.time_per_step_in_secs == 0.1

    def test_all_attributes_initialized(self, setup_state_data):
        """Test that all attributes are properly initialized."""
        data = setup_state_data
        state = VisualizableSimState(
            timestep=data["timestep"],
            robot_action=data["robot_action"],
            robot_pose=data["robot_pose"],
            pedestrian_positions=data["pedestrian_positions"],
            ray_vecs=data["ray_vecs"],
            ped_actions=data["ped_actions"],
            time_per_step_in_secs=0.1,
            ego_ped_pose=((3.0, 3.0), 0.0),
            ego_ped_ray_vecs=np.array([[[3.0, 3.0], [4.0, 4.0]]]),
            ego_ped_action=None,
        )

        # Check all attributes are properly set
        assert state.timestep == data["timestep"]
        assert state.robot_action == data["robot_action"]
        assert state.robot_pose == data["robot_pose"]
        assert np.array_equal(state.pedestrian_positions, data["pedestrian_positions"])
        assert np.array_equal(state.ray_vecs, data["ray_vecs"])
        assert np.array_equal(state.ped_actions, data["ped_actions"])
        assert state.time_per_step_in_secs == 0.1
        assert state.ego_ped_pose == ((3.0, 3.0), 0.0)
        assert np.array_equal(state.ego_ped_ray_vecs, np.array([[[3.0, 3.0], [4.0, 4.0]]]))
        assert state.ego_ped_action is None
