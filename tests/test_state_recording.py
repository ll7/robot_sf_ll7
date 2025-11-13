"""Validate that RobotEnv stores pickle recordings in the canonical artifact tree."""

import pickle
from pathlib import Path

from robot_sf.common.artifact_paths import resolve_artifact_path
from robot_sf.gym_env.robot_env import RobotEnv, VisualizableSimState
from robot_sf.nav.map_config import MapDefinition


def test_recording():
    recordings_dir = resolve_artifact_path(Path("recordings"))
    recordings_dir.mkdir(parents=True, exist_ok=True)
    existing_files = set(recordings_dir.glob("*.pkl"))

    env = RobotEnv(recording_enabled=True)
    env.reset()

    # Run the simulation for a few timesteps
    for _ in range(10):
        action = env.action_space.sample()  # replace with your action sampling logic
        env.step(action)

    # Save the recording
    env.reset()

    current_files = set(recordings_dir.glob("*.pkl"))
    new_files = current_files - existing_files
    assert new_files, "Recording file was not created"
    # Resolve deterministic ordering in case multiple artifacts exist
    recording_path = sorted(new_files)[-1]

    # Load the recording
    with recording_path.open("rb") as fh:
        recorded_states, map_def = pickle.load(fh)

    # Check that the recording has the correct length
    assert len(recorded_states) == 10

    # Check that the recorded states are instances of VisualizableSimState
    assert all(isinstance(state, VisualizableSimState) for state in recorded_states)

    # Check that the map definition is an instance of MapDefinition
    assert isinstance(map_def, MapDefinition)

    recording_path.unlink()
