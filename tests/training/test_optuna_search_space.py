"""Tests for config-driven Optuna expert PPO search spaces."""

from __future__ import annotations

from types import SimpleNamespace

import optuna
import pytest

from robot_sf.training.optuna_search_space import (
    apply_search_space_update,
    suggest_search_space,
    validate_search_space,
)


def test_search_space_samples_and_applies_allowlisted_fields() -> None:
    """Search-space updates PPO hyperparams and policy network architecture only."""

    search_space = {
        "ppo_hyperparams": {
            "learning_rate": {"type": "float", "low": 1e-5, "high": 3e-4, "log": True},
            "batch_size": {"type": "categorical", "values": [128, 256]},
        },
        "policy_net_arch": {"type": "categorical", "values": [[64, 64], [128, 128]]},
    }
    trial = optuna.trial.FixedTrial(
        {"learning_rate": 1e-4, "batch_size": 256, "policy_net_arch": "128,128"}
    )
    update = suggest_search_space(trial, search_space, max_batch_size=192)
    config = SimpleNamespace(ppo_hyperparams={"n_epochs": 5}, policy_net_arch=(64, 64))

    apply_search_space_update(config, update)

    assert config.ppo_hyperparams == {
        "n_epochs": 5,
        "learning_rate": pytest.approx(1e-4),
        "batch_size": 192,
    }
    assert config.policy_net_arch == (128, 128)


def test_search_space_rejects_unknown_sections() -> None:
    """Arbitrary nested config mutation is not accepted by the first HPO lane."""

    with pytest.raises(ValueError, match="Unsupported search_space section"):
        validate_search_space({"env_factory_kwargs": {"reward_name": {"type": "categorical"}}})
