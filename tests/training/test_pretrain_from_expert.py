"""Unit tests for pretrain_from_expert utilities."""

from __future__ import annotations

import builtins
import warnings
from types import SimpleNamespace

import numpy as np
import pytest
from gymnasium import spaces
from gymnasium.spaces.utils import flatten as flatten_space

from scripts.training.pretrain_from_expert import (
    ImitationDependencyWarning,
    _convert_to_transitions,
    _create_bc_trainer,
    _require_imitation_bc,
    _warn_imitation_dependency_mode,
)


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


def test_convert_to_transitions_repeats_single_frame_obs_to_match_stacked_space() -> None:
    """Single-frame saved observations should expand across stack dimension for BC."""

    observations = [
        [
            {
                "drive_state": np.array([0.1, 0.2, 0.3, 0.4, 0.5], dtype=np.float32),
                "rays": np.zeros(272, dtype=np.float32),
            },
            {
                "drive_state": np.array([0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32),
                "rays": np.ones(272, dtype=np.float32),
            },
        ]
    ]
    dataset = _build_dataset(observations)

    obs_space = spaces.Dict(
        {
            "drive_state": spaces.Box(low=-1.0, high=1.0, shape=(3, 5), dtype=np.float32),
            "rays": spaces.Box(low=0.0, high=1.0, shape=(3, 272), dtype=np.float32),
        }
    )

    trajectories = _convert_to_transitions(dataset, obs_space)

    assert len(trajectories) == 1
    assert trajectories[0].obs.shape == (2, 831)
    first_step = trajectories[0].obs[0]
    assert first_step.shape == (831,)


def test_warn_imitation_dependency_mode_emits_warning_when_training() -> None:
    """Warning should be emitted only when the imitation stack is unavailable."""
    try:
        import imitation  # noqa: F401
    except ImportError:
        with pytest.warns(ImitationDependencyWarning, match="--group imitation"):
            _warn_imitation_dependency_mode(dry_run=False)
    else:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            _warn_imitation_dependency_mode(dry_run=False)

        imitation_warnings = [
            warning
            for warning in captured
            if issubclass(warning.category, ImitationDependencyWarning)
        ]
        assert not imitation_warnings


def test_warn_imitation_dependency_mode_silent_for_dry_run() -> None:
    """Dry-run mode should not warn about imitation runtime dependency stack."""
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        _warn_imitation_dependency_mode(dry_run=True)

    assert len(captured) == 0


def test_require_imitation_bc_warns_and_raises_when_import_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Import failures should emit actionable warning before raising RuntimeError."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: object, **kwargs: object):
        if name == "imitation.algorithms":
            raise ImportError("forced missing imitation")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.warns(ImitationDependencyWarning, match="uv sync --group imitation"):
        with pytest.raises(RuntimeError, match="optional imitation stack"):
            _require_imitation_bc()


def test_create_bc_trainer_forces_cpu_and_passes_device_when_supported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BC trainer setup should keep PPO and BC on CPU to avoid mixed-device failures."""
    recorded: dict[str, object] = {}

    class _FakeBCModule:
        class BC:
            def __init__(
                self,
                *,
                observation_space,
                action_space,
                demonstrations,
                policy,
                batch_size,
                rng,
                device,
            ) -> None:
                recorded["bc_observation_space"] = observation_space
                recorded["bc_action_space"] = action_space
                recorded["bc_demonstrations"] = demonstrations
                recorded["bc_policy"] = policy
                recorded["bc_batch_size"] = batch_size
                recorded["bc_rng_type"] = type(rng).__name__
                recorded["bc_device"] = device

    class _FakePPO:
        def __init__(self, policy, env, learning_rate, verbose, device) -> None:
            recorded["ppo_policy_name"] = policy
            recorded["ppo_env"] = env
            recorded["ppo_learning_rate"] = learning_rate
            recorded["ppo_verbose"] = verbose
            recorded["ppo_device"] = device
            self.policy = "policy-on-cpu"

    monkeypatch.setattr(
        "scripts.training.pretrain_from_expert._require_imitation_bc",
        lambda: _FakeBCModule,
    )
    monkeypatch.setattr("scripts.training.pretrain_from_expert.PPO", _FakePPO)

    env = SimpleNamespace(
        observation_space=spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        action_space=spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
    )
    config = SimpleNamespace(
        learning_rate=3e-4,
        batch_size=8,
        random_seeds=(123,),
    )
    trajectories = [SimpleNamespace(obs=np.zeros((2, 4)), acts=np.zeros((1, 2)))]

    trainer, policy_model = _create_bc_trainer(env, trajectories, config)

    assert trainer is not None
    assert policy_model.policy == "policy-on-cpu"
    assert recorded["ppo_device"] == "cpu"
    assert recorded["bc_device"] == "cpu"
    assert recorded["bc_policy"] == "policy-on-cpu"
