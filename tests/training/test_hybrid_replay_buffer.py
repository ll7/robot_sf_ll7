"""Tests for the issue #4012 hybrid replay sampler."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.training.hybrid_replay_buffer import HybridReplayBuffer
from robot_sf.training.offline_online_rl import OfflineDatasetPreflight, OfflineTransitionBatch


def test_offline_only_sampling_works_before_online_data_exists() -> None:
    """Offline replay updates can seed before online transitions arrive."""

    buffer = HybridReplayBuffer(offline_batch=_batch(4), offline_sample_fraction=1.0, seed=7)

    sample = buffer.sample(3)

    assert sample["actions"].shape == (3, 2)
    assert sample["sources"] == ("offline", "offline", "offline")


def test_offline_only_sampling_fails_closed_when_partition_too_small() -> None:
    """Sampler never silently under-fills or replaces tiny offline partitions."""

    buffer = HybridReplayBuffer(offline_batch=_batch(1), offline_sample_fraction=1.0, seed=7)

    with pytest.raises(ValueError, match="offline replay partition"):
        buffer.sample(4)


def test_terminal_truncated_offline_transition_survives_hybrid_sample() -> None:
    """Final outcome flags remain present through the hybrid replay sampler."""

    buffer = HybridReplayBuffer(
        offline_batch=_batch(1, dones=(True,), truncated=(True,)),
        offline_sample_fraction=1.0,
        seed=7,
    )

    sample = buffer.sample(1)

    assert sample["sources"] == ("offline",)
    assert sample["dones"].tolist() == [True]
    assert sample["truncated"].tolist() == [True]


def test_mixed_sampling_respects_offline_fraction() -> None:
    """Sampler draws both partitions according to the configured fraction."""

    buffer = HybridReplayBuffer(offline_batch=_batch(10), offline_sample_fraction=0.5, seed=7)
    for index in range(10):
        buffer.add_online(
            np.array([index], dtype=np.float32),
            np.array([index + 1], dtype=np.float32),
            np.array([0.0, 0.0], dtype=np.float32),
            0.0,
            False,
        )

    sample = buffer.sample(10)

    assert sample["sources"].count("offline") == 5
    assert sample["sources"].count("online") == 5
    assert sample["truncated"].tolist() == [False] * 10
    assert buffer.stats.offline_count == 10
    assert buffer.stats.online_count == 10


def test_empty_replay_fails_closed() -> None:
    """Cannot sample an empty hybrid replay buffer."""

    buffer = HybridReplayBuffer(seed=7)

    with pytest.raises(ValueError, match="empty"):
        buffer.sample(1)


def test_invalid_offline_fraction_fails_closed() -> None:
    """Invalid replay mix settings fail at construction."""

    with pytest.raises(ValueError, match="offline_sample_fraction"):
        HybridReplayBuffer(offline_sample_fraction=1.5)


def _batch(
    size: int,
    *,
    dones: tuple[bool, ...] | None = None,
    truncated: tuple[bool, ...] | None = None,
) -> OfflineTransitionBatch:
    preflight = OfflineDatasetPreflight(
        dataset_path=__file__,
        dataset_sha256="0" * 64,
        split="train",
        episode_count=1,
        accepted_transitions=size,
        dropped_terminal_transitions=0,
        action_shape=(2,),
        observation_contract="dataset_observation",
        action_contract="box_direct",
    )
    return OfflineTransitionBatch(
        observations=tuple(np.array([index], dtype=np.float32) for index in range(size)),
        next_observations=tuple(np.array([index + 1], dtype=np.float32) for index in range(size)),
        actions=np.zeros((size, 2), dtype=np.float32),
        rewards=np.zeros(size, dtype=np.float32),
        dones=np.asarray(dones if dones is not None else (False,) * size, dtype=bool),
        truncated=np.asarray(truncated if truncated is not None else (False,) * size, dtype=bool),
        episode_ids=tuple(f"ep-{index}" for index in range(size)),
        scenario_ids=tuple("classic" for _ in range(size)),
        seeds=np.zeros(size, dtype=int),
        preflight=preflight,
    )
