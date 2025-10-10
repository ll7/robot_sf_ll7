"""Tests for enhanced interactive playback with episode boundaries."""

import tempfile

from robot_sf.render.interactive_playback import (
    create_interactive_playback_from_batch,
)
from robot_sf.render.jsonl_playback import JSONLPlaybackLoader
from robot_sf.render.jsonl_recording import JSONLRecorder


def test_episode_boundary_detection():
    """Test that episode boundaries are correctly detected."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create multiple episodes
        states_per_episode = [3, 2, 4]  # Different number of states per episode

        for ep_idx, num_states in enumerate(states_per_episode):
            recorder = JSONLRecorder(
                output_dir=temp_dir,
                suite="test",
                scenario="boundary",
                algorithm="test",
                seed=ep_idx,
            )

            recorder.current_episode_id = ep_idx
            recorder.start_episode()

            import numpy as np

            from robot_sf.render.sim_view import VisualizableSimState

            for step in range(num_states):
                state = VisualizableSimState(
                    timestep=step,
                    robot_action=None,
                    robot_pose=((step * 1.0, ep_idx * 5.0), 0.0),  # Different position per episode
                    pedestrian_positions=np.array([]),
                    ray_vecs=np.array([]),
                    ped_actions=np.array([]),
                )
                recorder.record_step(state)

            recorder.end_episode()

        # Load batch
        loader = JSONLPlaybackLoader()
        batch = loader.load_directory(temp_dir)

        # Create interactive playback
        player = create_interactive_playback_from_batch(batch)

        # Check episode boundaries
        expected_boundaries = [0, 3, 5]  # Start of each episode
        assert player.episode_boundaries == expected_boundaries

        # Check total states
        assert len(player.states) == sum(states_per_episode)


def test_trajectory_clearing_at_boundaries():
    """Test that trajectories are cleared at episode boundaries."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a simple batch with 2 episodes
        for ep_idx in range(2):
            recorder = JSONLRecorder(
                output_dir=temp_dir,
                suite="test",
                scenario="clearing",
                algorithm="test",
                seed=ep_idx,
            )

            recorder.current_episode_id = ep_idx
            recorder.start_episode()

            import numpy as np

            from robot_sf.render.sim_view import VisualizableSimState

            # Add 2 states per episode
            for step in range(2):
                state = VisualizableSimState(
                    timestep=step,
                    robot_action=None,
                    robot_pose=((step + ep_idx * 10, step + ep_idx * 10), 0.0),
                    pedestrian_positions=np.array([]),
                    ray_vecs=np.array([]),
                    ped_actions=np.array([]),
                )
                recorder.record_step(state)

            recorder.end_episode()

        # Load and create playback
        loader = JSONLPlaybackLoader()
        batch = loader.load_directory(temp_dir)
        player = create_interactive_playback_from_batch(batch)

        # Enable trajectories
        player.show_trajectories = True

        # Simulate playback through first episode
        player.current_frame = 0
        player._update_trajectories(player.states[0])
        assert len(player.robot_trajectory) == 1

        player.current_frame = 1
        player._update_trajectories(player.states[1])
        assert len(player.robot_trajectory) == 2

        # Move to second episode (should clear trajectories)
        player.current_frame = 2  # Start of second episode
        should_clear = player._should_clear_trajectories()
        assert should_clear is True

        # Update trajectories (should clear due to boundary)
        player._update_trajectories(player.states[2])
        assert len(player.robot_trajectory) == 1  # Should be cleared and restarted


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
