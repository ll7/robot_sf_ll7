import os

from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.nav.map_config import MapDefinition
from robot_sf.render.playback_recording import load_states
from robot_sf.render.sim_view import VisualizableSimState


def test_recording():
    """
    Test the state recording and saving functionality of the environment.

    This test:
    1. Creates an environment with recording enabled
    2. Runs a simulation for a few steps
    3. Verifies that states are recorded and saved correctly
    4. Checks that the saved file has the proper format and content
    """
    env = RobotEnv(recording_enabled=True)
    env.reset()

    # Run the simulation for a few timesteps
    for _ in range(10):
        action = env.action_space.sample()  # random action
        env.step(action)

    # Save the recording by calling reset
    filepath = env.save_recording()

    # Ensure a file was created
    assert filepath is not None
    assert os.path.exists(filepath)

    # Load the recording using the improved load_states function
    result = load_states(filepath, return_dict=True)

    # Check that we got a dictionary with all expected keys
    assert isinstance(result, dict)
    assert "states" in result
    assert "map_def" in result
    assert "metadata" in result
    assert "rewards" in result

    # Check that the recording has the correct length
    assert len(result["states"]) == 10

    # Check that the recorded states are instances of VisualizableSimState
    assert all(isinstance(state, VisualizableSimState) for state in result["states"])

    # Check that the map definition is an instance of MapDefinition
    assert isinstance(result["map_def"], MapDefinition)

    # Check that metadata exists and has expected fields
    assert isinstance(result["metadata"], dict)
    assert "timestamp" in result["metadata"]
    assert "num_states" in result["metadata"]
    assert "environment_type" in result["metadata"]
    assert result["metadata"]["num_states"] == 10
    assert result["metadata"]["environment_type"] == "RobotEnv"

    # Clean up
    if os.path.exists(filepath):
        os.remove(filepath)
