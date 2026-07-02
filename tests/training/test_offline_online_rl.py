"""Tests for offline-to-online RL transition preflight."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from gymnasium import spaces as gym_spaces

from robot_sf.benchmark.rl_trajectory_dataset import (
    RLTrajectoryEpisode,
    compute_return_to_go,
    write_rl_trajectory_dataset,
)
from robot_sf.training.offline_online_rl import (
    load_offline_transition_batch,
    validate_batch_against_env_spaces,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_load_train_split_derives_next_observations_and_drops_terminal(tmp_path: Path) -> None:
    """Transition builder uses only train rows and never crosses episode boundaries."""

    dataset_path = tmp_path / "dataset.jsonl"
    write_rl_trajectory_dataset(
        [
            _episode("train", "ep-train", rewards=(1.0, 2.0, 3.0)),
            _episode("validation", "ep-val", rewards=(9.0, 9.0)),
        ],
        dataset_path,
    )

    batch = load_offline_transition_batch(dataset_path, split="train", min_transitions=2)

    assert batch.size == 2
    assert batch.episode_ids == ("ep-train", "ep-train")
    assert batch.rewards.tolist() == [1.0, 2.0]
    assert batch.preflight.dropped_terminal_transitions == 1
    assert np.allclose(batch.observations[0], np.array([0.0, 0.5], dtype=np.float32))
    assert np.allclose(batch.next_observations[1], np.array([2.0, 2.5], dtype=np.float32))


def test_load_rejects_empty_train_split(tmp_path: Path) -> None:
    """Offline preflight fails closed when the requested split is absent."""

    dataset_path = tmp_path / "dataset.jsonl"
    write_rl_trajectory_dataset([_episode("validation", "ep-val")], dataset_path)

    with pytest.raises(ValueError, match="no episodes"):
        load_offline_transition_batch(dataset_path, split="train")


def test_load_rejects_inconsistent_action_shapes(tmp_path: Path) -> None:
    """Offline action tensors must have one consistent Box-compatible shape."""

    dataset_path = tmp_path / "dataset.jsonl"
    episode = _episode("train", "ep-train", rewards=(1.0, 2.0, 3.0))
    bad_episode = RLTrajectoryEpisode(
        dataset_id=episode.dataset_id,
        episode_id=episode.episode_id,
        scenario_id=episode.scenario_id,
        seed=episode.seed,
        source_policy_id=episode.source_policy_id,
        split=episode.split,
        observations=episode.observations,
        actions=([0.0, 0.1], [0.2], [0.3, 0.4]),
        rewards=episode.rewards,
        return_to_go=episode.return_to_go,
        terminated=episode.terminated,
        truncated=episode.truncated,
        pedestrians=episode.pedestrians,
        robot_states=episode.robot_states,
        provenance=episode.provenance,
    )
    write_rl_trajectory_dataset([bad_episode], dataset_path)

    with pytest.raises(ValueError, match="action shapes"):
        load_offline_transition_batch(dataset_path)


def test_validate_batch_against_env_spaces_fails_closed_on_action_mismatch(tmp_path: Path) -> None:
    """Dataset actions outside the online SAC action space are rejected."""

    dataset_path = tmp_path / "dataset.jsonl"
    write_rl_trajectory_dataset([_episode("train", "ep-train")], dataset_path)
    batch = load_offline_transition_batch(dataset_path)

    preflight = validate_batch_against_env_spaces(
        batch,
        observation_space=gym_spaces.Box(-10.0, 10.0, shape=(2,), dtype=np.float32),
        action_space=gym_spaces.Box(-0.05, 0.05, shape=(2,), dtype=np.float32),
    )

    assert not preflight.ok
    assert "action at transition 0" in preflight.validation_errors[0]


def test_validate_batch_against_env_spaces_accepts_box_contract(tmp_path: Path) -> None:
    """Compatible Box observations and actions pass preflight."""

    dataset_path = tmp_path / "dataset.jsonl"
    write_rl_trajectory_dataset([_episode("train", "ep-train")], dataset_path)
    batch = load_offline_transition_batch(dataset_path)

    preflight = validate_batch_against_env_spaces(
        batch,
        observation_space=gym_spaces.Box(-10.0, 10.0, shape=(2,), dtype=np.float32),
        action_space=gym_spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
    )

    assert preflight.ok


def _episode(
    split: str,
    episode_id: str,
    *,
    rewards: tuple[float, ...] = (1.0, 2.0),
) -> RLTrajectoryEpisode:
    observations = tuple([float(index), float(index) + 0.5] for index in range(len(rewards)))
    actions = tuple([0.1 * (index + 1), -0.1] for index in range(len(rewards)))
    return RLTrajectoryEpisode(
        dataset_id="issue_4012_unit",
        episode_id=episode_id,
        scenario_id="classic",
        seed=4012,
        source_policy_id="unit",
        split=split,
        observations=observations,
        actions=actions,
        rewards=rewards,
        return_to_go=tuple(compute_return_to_go(rewards)),
        terminated=tuple(index == len(rewards) - 1 for index in range(len(rewards))),
        truncated=tuple(False for _ in rewards),
        pedestrians=tuple({} for _ in rewards),
        robot_states=tuple({} for _ in rewards),
        provenance={"test": "issue_4012"},
    )
