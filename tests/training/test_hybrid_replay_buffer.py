"""Tests for the issue #4012 hybrid replay sampler."""

from __future__ import annotations

import numpy as np
import pytest

from robot_sf.training.hybrid_replay_buffer import HybridReplayBuffer
from robot_sf.training.offline_online_rl import OfflineDatasetPreflight, OfflineTransitionBatch


def test_offline_only_sampling_works_before_online_data_exists() -> None:
    """Offline replay can seed updates before online transitions arrive."""

    buffer = HybridReplayBuffer(offline_batch=_batch(4), offline_sample_fraction=1.0, seed=7)

    sample = buffer.sample(3)

    assert sample["actions"].shape == (3, 2)
    assert sample["sources"] == ("offline", "offline", "offline")


def test_offline_only_sampling_with_replacement_keeps_requested_batch_size() -> None:
    """Replacement sampling should not cap batches at available offline rows."""

    buffer = HybridReplayBuffer(offline_batch=_batch(1), offline_sample_fraction=1.0, seed=7)

    sample = buffer.sample(4)

    assert sample["actions"].shape == (4, 2)
    assert sample["sources"] == ("offline", "offline", "offline", "offline")


def test_mixed_sampling_respects_offline_fraction() -> None:
    """Sampler draws from both partitions under a mixed fraction."""

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
    assert buffer.stats.offline_count == 10
    assert buffer.stats.online_count == 10


def test_invalid_offline_fraction_fails_closed() -> None:
    """Invalid replay mix settings fail at construction."""

    with pytest.raises(ValueError, match="offline_sample_fraction"):
        HybridReplayBuffer(offline_sample_fraction=1.5)


def _batch(size: int) -> OfflineTransitionBatch:
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
        dones=np.zeros(size, dtype=bool),
        truncated=np.zeros(size, dtype=bool),
        episode_ids=tuple(f"ep-{index}" for index in range(size)),
        scenario_ids=tuple("classic" for _ in range(size)),
        seeds=np.zeros(size, dtype=int),
        preflight=preflight,
    )
