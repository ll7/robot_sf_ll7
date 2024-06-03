import os
import pickle

from robot_sf.gym_env.robot_env import RobotEnv, VisualizableSimState

def test_recording():
    env = RobotEnv(
        recording_enabled=True
    )
    env.reset()

    # Run the simulation for a few timesteps
    for _ in range(10):
        action = env.action_space.sample()  # replace with your action sampling logic
        env.step(action)

    # Save the recording
    env.reset()

    # Check that the file was created
    filename = max(
        os.listdir('recordings'), key=lambda x: os.path.getctime(os.path.join('recordings', x)))
    assert os.path.exists(os.path.join('recordings', filename))

    # Load the recording
    with open(os.path.join('recordings', filename), 'rb') as f:
        recorded_states = pickle.load(f)

    # Check that the recording has the correct length
    assert len(recorded_states) == 10

    # Check that the recorded states are instances of VisualizableSimState
    assert all(isinstance(state, VisualizableSimState) for state in recorded_states)
