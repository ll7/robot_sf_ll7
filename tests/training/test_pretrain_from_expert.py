"""Unit tests for pretrain_from_expert utilities."""

from __future__ import annotations

import numpy as np
from gymnasium import spaces
from gymnasium.spaces.utils import flatten as flatten_space

from scripts.training.pretrain_from_expert import _convert_to_transitions


def _build_dataset(
    observations: list[list[dict[str, np.ndarray]]],
    *,
    actions_match_observations: bool = False,
) -> dict[str, object]:
    """Helper to assemble a minimal dataset structure accepted by converter."""

    dummy_positions = [
        [np.zeros(2, dtype=np.float32) for _ in range(len(episode))] for episode in observations
    ]
    dummy_actions = []
    for episode in observations:
        action_steps = len(episode) if actions_match_observations else max(len(episode) - 1, 1)
        dummy_actions.append([np.zeros(2, dtype=np.float32) for _ in range(action_steps)])

    return {
        "positions": dummy_positions,
        "actions": dummy_actions,
        "observations": observations,
        "episode_count": len(observations),
    }


def test_convert_to_transitions_uses_observation_space_flattening() -> None:
    """Dict observations should flatten via env observation space."""

    observations = [
        [
            {
                "robot": np.array([0.1, 0.2], dtype=np.float32),
                "goal": np.array([0.3, 0.4, 0.5], dtype=np.float32),
            },
            {
                "robot": np.array([0.6, 0.7], dtype=np.float32),
                "goal": np.array([0.8, 0.9, 1.0], dtype=np.float32),
            },
            {
                "robot": np.array([1.1, 1.2], dtype=np.float32),
                "goal": np.array([1.3, 1.4, 1.5], dtype=np.float32),
            },
        ]
    ]
    dataset = _build_dataset(observations)

    obs_space = spaces.Dict(
        {
            "robot": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "goal": spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32),
        }
    )

    trajectories = _convert_to_transitions(dataset, obs_space)

    assert len(trajectories) == 1
    assert trajectories[0].obs.shape == (3, 5)
    expected = flatten_space(obs_space, observations[0][0])
    np.testing.assert_allclose(trajectories[0].obs[0], expected)


def test_convert_to_transitions_handles_fallback_without_space() -> None:
    """Conversion should still work when no observation space is provided."""

    observations = [
        [
            {
                "robot": np.array([0.1, 0.2], dtype=np.float64),
                "goal": np.array([0.3, 0.4], dtype=np.float64),
            },
            {
                "robot": np.array([0.5, 0.6], dtype=np.float64),
                "goal": np.array([0.7, 0.8], dtype=np.float64),
            },
            {
                "robot": np.array([0.9, 1.0], dtype=np.float64),
                "goal": np.array([1.1, 1.2], dtype=np.float64),
            },
        ]
    ]
    dataset = _build_dataset(observations)

    trajectories = _convert_to_transitions(dataset)

    assert len(trajectories) == 1
    assert trajectories[0].obs.shape == (3, 4)


def test_convert_to_transitions_appends_terminal_observation_when_missing() -> None:
    """Datasets with len(obs) == len(actions) should be auto-padded."""

    observations = [
        [
            {
                "robot": np.array([0.1, 0.2], dtype=np.float32),
                "goal": np.array([0.3, 0.4], dtype=np.float32),
            },
            {
                "robot": np.array([0.5, 0.6], dtype=np.float32),
                "goal": np.array([0.7, 0.8], dtype=np.float32),
            },
        ]
    ]

    dataset = _build_dataset(observations, actions_match_observations=True)

    trajectories = _convert_to_transitions(dataset)

    assert len(trajectories) == 1
    assert trajectories[0].obs.shape[0] == trajectories[0].acts.shape[0] + 1
    np.testing.assert_allclose(trajectories[0].obs[-1], trajectories[0].obs[-2])
