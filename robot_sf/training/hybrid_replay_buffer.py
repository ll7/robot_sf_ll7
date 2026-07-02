"""Small deterministic hybrid offline/online replay sampler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from robot_sf.training.offline_online_rl import OfflineTransitionBatch


@dataclass(frozen=True, slots=True)
class HybridReplayStats:
    """Partition counts for hybrid replay smoke diagnostics."""

    offline_count: int
    online_count: int
    offline_sample_fraction: float


class HybridReplayBuffer:
    """Framework-agnostic sampler that keeps offline and online partitions separate."""

    def __init__(
        self,
        *,
        offline_batch: OfflineTransitionBatch | None = None,
        offline_sample_fraction: float = 0.5,
        seed: int | None = None,
    ) -> None:
        """Initialize replay partitions and deterministic sampler."""

        if not 0.0 <= offline_sample_fraction <= 1.0:
            raise ValueError("offline_sample_fraction must be between 0.0 and 1.0")
        self._offline = offline_batch
        self._offline_sample_fraction = float(offline_sample_fraction)
        self._rng = np.random.default_rng(seed)
        self._online_observations: list[object] = []
        self._online_next_observations: list[object] = []
        self._online_actions: list[np.ndarray] = []
        self._online_rewards: list[float] = []
        self._online_dones: list[bool] = []

    @property
    def stats(self) -> HybridReplayStats:
        """Return current partition sizes."""

        return HybridReplayStats(
            offline_count=0 if self._offline is None else self._offline.size,
            online_count=len(self._online_rewards),
            offline_sample_fraction=self._offline_sample_fraction,
        )

    def add_online(
        self,
        observation: object,
        next_observation: object,
        action: np.ndarray,
        reward: float,
        done: bool,
    ) -> None:
        """Append one online transition to the online partition."""

        self._online_observations.append(observation)
        self._online_next_observations.append(next_observation)
        self._online_actions.append(np.asarray(action, dtype=np.float32))
        self._online_rewards.append(float(reward))
        self._online_dones.append(bool(done))

    def sample(self, batch_size: int) -> dict[str, object]:
        """Sample a mixed batch and label transition source for diagnostics.

        Returns:
            Dictionary containing sampled transitions and `offline`/`online` source labels.
        """

        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        offline_available = 0 if self._offline is None else self._offline.size
        online_available = len(self._online_rewards)
        if offline_available + online_available == 0:
            raise ValueError("cannot sample from empty hybrid replay buffer")

        offline_target = round(batch_size * self._offline_sample_fraction)
        offline_count = min(offline_target, offline_available)
        online_count = min(batch_size - offline_count, online_available)
        if offline_count + online_count < batch_size:
            remaining = batch_size - offline_count - online_count
            offline_count += min(remaining, offline_available - offline_count)
        if offline_count + online_count < batch_size:
            remaining = batch_size - offline_count - online_count
            online_count += min(remaining, online_available - online_count)

        observations: list[object] = []
        next_observations: list[object] = []
        actions: list[np.ndarray] = []
        rewards: list[float] = []
        dones: list[bool] = []
        sources: list[str] = []

        if offline_count:
            assert self._offline is not None
            for index in self._rng.choice(offline_available, size=offline_count, replace=True):
                observations.append(self._offline.observations[int(index)])
                next_observations.append(self._offline.next_observations[int(index)])
                actions.append(self._offline.actions[int(index)])
                rewards.append(float(self._offline.rewards[int(index)]))
                dones.append(bool(self._offline.dones[int(index)]))
                sources.append("offline")

        if online_count:
            for index in self._rng.choice(online_available, size=online_count, replace=True):
                observations.append(self._online_observations[int(index)])
                next_observations.append(self._online_next_observations[int(index)])
                actions.append(self._online_actions[int(index)])
                rewards.append(self._online_rewards[int(index)])
                dones.append(self._online_dones[int(index)])
                sources.append("online")

        return {
            "observations": tuple(observations),
            "next_observations": tuple(next_observations),
            "actions": np.stack(actions).astype(np.float32, copy=False),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=bool),
            "sources": tuple(sources),
        }
