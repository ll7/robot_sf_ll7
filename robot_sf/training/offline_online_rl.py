"""Offline-to-online reinforcement learning helpers for SAC smoke lanes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces as gym_spaces

from robot_sf.benchmark.rl_trajectory_dataset import (
    RLTrajectoryEpisode,
    load_rl_trajectory_dataset,
    sha256_file,
)


@dataclass(frozen=True, slots=True)
class OfflineDatasetPreflight:
    """Compatibility and provenance summary for an offline transition batch."""

    dataset_path: Path
    dataset_sha256: str
    split: str
    episode_count: int
    accepted_transitions: int
    dropped_terminal_transitions: int
    action_shape: tuple[int, ...]
    observation_contract: str
    action_contract: str
    validation_errors: tuple[str, ...] = field(default_factory=tuple)

    @property
    def ok(self) -> bool:
        """Return true when the batch passed fail-closed preflight."""

        return not self.validation_errors


@dataclass(frozen=True, slots=True)
class OfflineTransitionBatch:
    """Train-split transition batch derived from `RLTrajectoryDataset.v1` episodes."""

    observations: tuple[Any, ...]
    next_observations: tuple[Any, ...]
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    truncated: np.ndarray
    episode_ids: tuple[str, ...]
    scenario_ids: tuple[str, ...]
    seeds: np.ndarray
    preflight: OfflineDatasetPreflight

    @property
    def size(self) -> int:
        """Return transition count."""

        return int(self.rewards.shape[0])


@dataclass(slots=True)
class _TransitionAccumulator:
    observations: list[Any] = field(default_factory=list)
    next_observations: list[Any] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    dones: list[bool] = field(default_factory=list)
    truncated: list[bool] = field(default_factory=list)
    episode_ids: list[str] = field(default_factory=list)
    scenario_ids: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=list)
    action_shape: tuple[int, ...] | None = None


def load_offline_transition_batch(
    dataset_path: Path | str,
    *,
    split: str = "train",
    min_transitions: int = 1,
    action_contract: str = "box_direct",
    observation_contract: str = "dataset_observation",
    skip_terminal_without_next_observation: bool = False,
) -> OfflineTransitionBatch:
    """Load train-split transitions from `RLTrajectoryDataset.v1`.

    The dataset is episode-major and has no explicit `next_observations` field, so the
    transition builder derives `t -> t + 1` only inside each episode and drops final rows that
    have no next observation. It never synthesizes cross-episode next observations.

    Returns:
        Offline transition batch with preflight provenance.
    """

    path = Path(dataset_path)
    episodes = [episode for episode in load_rl_trajectory_dataset(path) if episode.split == split]
    if not episodes:
        raise ValueError(f"offline dataset has no episodes for split {split!r}")

    transitions = _TransitionAccumulator()
    dropped_terminal = 0

    for episode in episodes:
        dropped_terminal += _append_episode_transitions(
            episode,
            transitions=transitions,
            skip_terminal_without_next_observation=skip_terminal_without_next_observation,
        )

    if len(transitions.rewards) < min_transitions:
        raise ValueError(
            "offline dataset split "
            f"{split!r} yielded {len(transitions.rewards)} transitions; required {min_transitions}"
        )

    rewards_array = np.asarray(transitions.rewards, dtype=np.float32)
    if not np.all(np.isfinite(rewards_array)):
        raise ValueError("offline dataset rewards must be finite")

    actions_array = np.stack(transitions.actions).astype(np.float32, copy=False)
    preflight = OfflineDatasetPreflight(
        dataset_path=path,
        dataset_sha256=sha256_file(path),
        split=split,
        episode_count=len(episodes),
        accepted_transitions=len(transitions.rewards),
        dropped_terminal_transitions=dropped_terminal,
        action_shape=tuple(actions_array.shape[1:]),
        observation_contract=observation_contract,
        action_contract=action_contract,
    )
    return OfflineTransitionBatch(
        observations=tuple(transitions.observations),
        next_observations=tuple(transitions.next_observations),
        actions=actions_array,
        rewards=rewards_array,
        dones=np.asarray(transitions.dones, dtype=bool),
        truncated=np.asarray(transitions.truncated, dtype=bool),
        episode_ids=tuple(transitions.episode_ids),
        scenario_ids=tuple(transitions.scenario_ids),
        seeds=np.asarray(transitions.seeds, dtype=int),
        preflight=preflight,
    )


def validate_batch_against_env_spaces(
    batch: OfflineTransitionBatch,
    *,
    observation_space: gym_spaces.Space[Any],
    action_space: gym_spaces.Space[Any],
) -> OfflineDatasetPreflight:
    """Validate offline observations and actions against online SAC environment spaces.

    Returns:
        Preflight summary with validation errors populated on incompatibility.
    """

    errors: list[str] = []
    for index, action in enumerate(batch.actions):
        if not action_space.contains(_coerce_for_space(action, action_space)):
            errors.append(f"action at transition {index} is outside online action_space")
            break

    for index, observation in enumerate(batch.observations):
        if not observation_space.contains(_coerce_for_space(observation, observation_space)):
            errors.append(f"observation at transition {index} is outside online observation_space")
            break

    for index, observation in enumerate(batch.next_observations):
        if not observation_space.contains(_coerce_for_space(observation, observation_space)):
            errors.append(
                f"next_observation at transition {index} is outside online observation_space"
            )
            break

    return OfflineDatasetPreflight(
        dataset_path=batch.preflight.dataset_path,
        dataset_sha256=batch.preflight.dataset_sha256,
        split=batch.preflight.split,
        episode_count=batch.preflight.episode_count,
        accepted_transitions=batch.preflight.accepted_transitions,
        dropped_terminal_transitions=batch.preflight.dropped_terminal_transitions,
        action_shape=batch.preflight.action_shape,
        observation_contract=batch.preflight.observation_contract,
        action_contract=batch.preflight.action_contract,
        validation_errors=tuple(errors),
    )


def seed_sb3_replay_buffer(model: Any, batch: OfflineTransitionBatch) -> None:
    """Seed an SB3 off-policy replay buffer with validated offline transitions."""

    for index in range(batch.size):
        info = {"source": "offline_rl_trajectory_dataset", "episode_id": batch.episode_ids[index]}
        if bool(batch.truncated[index]):
            info["TimeLimit.truncated"] = True
        model.replay_buffer.add(
            _with_vec_dim(batch.observations[index]),
            _with_vec_dim(batch.next_observations[index]),
            np.asarray(batch.actions[index], dtype=np.float32).reshape(1, -1),
            np.asarray([batch.rewards[index]], dtype=np.float32),
            np.asarray([batch.dones[index]], dtype=bool),
            [info],
        )


def _append_episode_transitions(
    episode: RLTrajectoryEpisode,
    *,
    transitions: _TransitionAccumulator,
    skip_terminal_without_next_observation: bool,
) -> int:
    """Append transitions from one episode.

    Returns:
        Number of dropped terminal rows without next observations.
    """

    dropped_terminal = 0
    for index, reward in enumerate(episode.rewards):
        if index + 1 >= len(episode.observations):
            terminal = bool(episode.terminated[index] or episode.truncated[index])
            if terminal:
                if skip_terminal_without_next_observation:
                    dropped_terminal += 1
                    continue
                next_observation = episode.observations[index]
            else:
                raise ValueError(
                    f"episode {episode.episode_id!r} step {index} has no next observation"
                )
        else:
            next_observation = episode.observations[index + 1]
        if index >= len(episode.actions):
            raise ValueError(f"episode {episode.episode_id!r} step {index} has no action")

        action = np.asarray(episode.actions[index], dtype=np.float32)
        if not np.all(np.isfinite(action)):
            raise ValueError(f"episode {episode.episode_id!r} step {index} action is not finite")
        if transitions.action_shape is None:
            transitions.action_shape = tuple(action.shape)
        if tuple(action.shape) != transitions.action_shape:
            raise ValueError(
                f"offline action shapes must be consistent; expected {transitions.action_shape}, "
                f"got {tuple(action.shape)}"
            )

        transitions.observations.append(episode.observations[index])
        transitions.next_observations.append(next_observation)
        transitions.actions.append(action)
        transitions.rewards.append(float(reward))
        transitions.dones.append(bool(episode.terminated[index] or episode.truncated[index]))
        transitions.truncated.append(bool(episode.truncated[index]))
        transitions.episode_ids.append(episode.episode_id)
        transitions.scenario_ids.append(episode.scenario_id)
        transitions.seeds.append(int(episode.seed))
    return dropped_terminal


def _coerce_for_space(value: Any, space: gym_spaces.Space[Any]) -> Any:
    """Coerce JSON-loaded arrays into a form `gymnasium.Space.contains` can check.

    Returns:
        Value converted for the provided space type.
    """

    if isinstance(space, gym_spaces.Dict):
        if not isinstance(value, Mapping):
            return value
        return {
            key: _coerce_for_space(value[key], subspace)
            for key, subspace in space.spaces.items()
            if key in value
        }
    if isinstance(space, gym_spaces.Box):
        return np.asarray(value, dtype=space.dtype)
    return value


def _with_vec_dim(value: Any) -> Any:
    """Add SB3 vector-env batch dimension to Box or Dict observations.

    Returns:
        Observation value with a leading vectorized-env dimension.
    """

    if isinstance(value, Mapping):
        return {str(key): np.asarray(item)[None, ...] for key, item in value.items()}
    return np.asarray(value)[None, ...]
