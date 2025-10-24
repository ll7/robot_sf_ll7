"""Tests for JSONL recording and playback functionality."""

import json
import os
import tempfile
from pathlib import Path
from typing import cast

import pytest

from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.gym_env.robot_env import RobotEnv
from robot_sf.render.jsonl_playback import JSONLPlaybackLoader
from robot_sf.render.jsonl_recording import JSONLRecorder


def test_jsonl_recording_basic():
    """Test basic JSONL recording functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create environment with JSONL recording enabled
        env = cast(
            RobotEnv,
            make_robot_env(
                recording_enabled=True,
                use_jsonl_recording=True,
                recording_dir=temp_dir,
                suite_name="test_suite",
                scenario_name="test_scenario",
                algorithm_name="test_algo",
                recording_seed=42,
            ),
        )

        # Reset to start episode
        env.reset()

        # Run a few steps
        for _ in range(5):
            action = env.action_space.sample()
            env.step(action)

        # End the episode
        env.end_episode_recording()

        # Check that files were created
        temp_path = Path(temp_dir)
        jsonl_files = list(temp_path.glob("*.jsonl"))
        meta_files = list(temp_path.glob("*.meta.json"))

        assert len(jsonl_files) == 1, f"Expected 1 JSONL file, got {len(jsonl_files)}"
        assert len(meta_files) == 1, f"Expected 1 metadata file, got {len(meta_files)}"

        # Check file naming convention
        expected_pattern = "test_suite_test_scenario_test_algo_42_ep0000.jsonl"
        assert jsonl_files[0].name == expected_pattern

        # Check file contents
        with open(jsonl_files[0], encoding="utf-8") as f:
            lines = f.readlines()

        # Should have at least: episode_start + 5 steps + episode_end
        assert len(lines) >= 7, f"Expected at least 7 records, got {len(lines)}"

        # Check first record is episode_start
        first_record = json.loads(lines[0])
        assert first_record["event"] == "episode_start"
        assert first_record["episode_id"] == 0

        # Check last record is episode_end
        last_record = json.loads(lines[-1])
        assert last_record["event"] == "episode_end"

        # Check metadata file
        with open(meta_files[0], encoding="utf-8") as f:
            metadata = json.load(f)

        assert metadata["episode_id"] == 0
        assert metadata["algorithm"] == "test_algo"
        assert metadata["scenario"] == "test_scenario"
        assert metadata["seed"] == 42

        env.close_recorder()


def test_jsonl_playback_loading():
    """Test loading JSONL recordings for playback."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a simple recording
        recorder = JSONLRecorder(
            output_dir=temp_dir, suite="test", scenario="basic", algorithm="manual", seed=123
        )

        recorder.start_episode()

        # Create mock state data
        from robot_sf.render.sim_view import VisualizableSimState

        for i in range(3):
            # Create a simple state with required parameters
            import numpy as np

            state = VisualizableSimState(
                timestep=i,
                robot_action=None,  # Required parameter
                robot_pose=((i * 1.0, i * 0.5), i * 0.1),
                pedestrian_positions=np.array([(i * 0.5, i * 0.3)]),
                ray_vecs=np.array([]),  # Required parameter
                ped_actions=np.array([]),  # Required parameter
            )
            recorder.record_step(state)

        recorder.end_episode()

        # Test loading
        loader = JSONLPlaybackLoader()
        episode_path = Path(temp_dir) / (
            f"{recorder.suite}_{recorder.scenario}_{recorder.algorithm}_{recorder.seed}_ep0000.jsonl"
        )
        episode, _ = loader.load_single_episode(episode_path)

        # Check episode data
        assert episode.episode_id == 0
        assert len(episode.states) == 3
        assert episode.reset_points is not None

        # Check state data
        first_state = episode.states[0]
        assert first_state.timestep == 0  # Should preserve original timestep from state
        assert first_state.robot_pose is not None


def test_directory_batch_loading():
    """Test loading multiple episodes from a directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create multiple episode files
        for episode_id in range(3):
            recorder = JSONLRecorder(
                output_dir=temp_dir,
                suite="batch_test",
                scenario="multi",
                algorithm="auto",
                seed=456,
            )

            recorder.current_episode_id = episode_id
            recorder.start_episode()

            # Add a few steps to each episode
            import numpy as np

            from robot_sf.render.sim_view import VisualizableSimState

            for step in range(2):
                state = VisualizableSimState(
                    timestep=step,
                    robot_action=None,
                    robot_pose=((step * 1.0, step * 1.0), 0.0),
                    pedestrian_positions=np.array([]),
                    ray_vecs=np.array([]),
                    ped_actions=np.array([]),
                )
                recorder.record_step(state)

            recorder.end_episode()

        # Test batch loading
        loader = JSONLPlaybackLoader()
        batch = loader.load_directory(temp_dir)

        assert batch.total_episodes == 3
        assert batch.total_steps == 6  # 2 steps per episode * 3 episodes
        assert len(batch.episodes) == 3

        # Check episodes are in order
        for i, episode in enumerate(batch.episodes):
            assert episode.episode_id == i
            assert len(episode.states) == 2


def test_legacy_pickle_compatibility():
    """Test backward compatibility with existing pickle files."""
    # This test uses the existing test pickle file
    test_pickle_file = "test_pygame/recordings/2024-06-04_08-39-59.pkl"

    if not os.path.exists(test_pickle_file):
        pytest.skip("Test pickle file not available")

    # Test loading legacy file
    loader = JSONLPlaybackLoader()
    episode, _ = loader.load_single_episode(test_pickle_file)

    # Should successfully load as an episode
    assert episode is not None
    assert len(episode.states) > 0
    assert episode.metadata is not None
    assert episode.metadata["legacy"] is True

    # Should detect some reset points in multi-episode files
    assert episode.reset_points is not None


if __name__ == "__main__":
    pytest.main([__file__])
