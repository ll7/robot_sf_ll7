"""Tests for enhanced interactive playback with episode boundaries."""

import tempfile

import numpy as np

from robot_sf.render.interactive_playback import (
    create_interactive_playback_from_batch,
)
from robot_sf.render.jsonl_playback import BatchPlayback, JSONLPlaybackLoader, PlaybackEpisode
from robot_sf.render.jsonl_recording import JSONLRecorder
from robot_sf.render.sim_view import VisualizableSimState


def _state(x: float, y: float, timestep: int = 0) -> VisualizableSimState:
    """Build a minimal playback state for tests."""

    return VisualizableSimState(
        timestep=timestep,
        robot_action=None,
        robot_pose=((x, y), 0.0),
        pedestrian_positions=np.array([]),
        ray_vecs=np.array([]),
        ped_actions=np.array([]),
    )


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

            for step in range(num_states):
                state = _state(step * 1.0, ep_idx * 5.0, timestep=step)
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

            # Add 2 states per episode
            for step in range(2):
                state = _state(step + ep_idx * 10, step + ep_idx * 10, timestep=step)
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


def test_batch_playback_merges_episode_telemetry_for_analyzer():
    """Batch playback should offset telemetry frames and expose metric filtering state."""
    episode_one = PlaybackEpisode(
        episode_id=0,
        states=[_state(0.0, 0.0, 0), _state(1.0, 0.0, 1)],
        telemetry_samples=[
            {
                "episode_id": 0,
                "frame_idx": 0,
                "reward_total": 1.0,
                "reward_terms": {"progress": 0.5},
                "step_metrics": {"near_misses": 0.0},
            }
        ],
    )
    episode_two = PlaybackEpisode(
        episode_id=1,
        states=[_state(5.0, 5.0, 0)],
        telemetry_samples=[
            {
                "episode_id": 1,
                "frame_idx": 0,
                "reward_total": 2.0,
                "reward_terms": {"progress": 1.0},
                "step_metrics": {"comfort_exposure": 0.2},
            }
        ],
    )
    batch = BatchPlayback(
        episodes=[episode_one, episode_two],
        map_def=JSONLPlaybackLoader._default_map_definition(),
        total_episodes=2,
        total_steps=3,
    )

    player = create_interactive_playback_from_batch(batch)

    assert player.telemetry_replay is not None
    assert [sample["frame_idx"] for sample in player.telemetry_replay.samples] == [0, 2]
    assert sorted(player.available_metric_keys) == ["comfort_exposure", "near_misses"]


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
