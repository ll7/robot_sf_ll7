"""Regression tests for imitation environment contract helpers.

These tests verify fail-closed handling for training-config inputs and observation filtering
because the BC/PPO warm-start path depends on reconstructing the exact observation contract
instead of silently falling back to defaults.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from gymnasium import spaces

from scripts.training.imitation_env_contract import (
    load_training_env_overrides,
    make_training_contract_env,
    resolve_scenario_config_path,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_load_training_env_overrides_rejects_non_file_or_non_mapping(tmp_path: Path) -> None:
    """Provided training configs should fail closed for directories or non-mapping YAML."""
    config_dir = tmp_path / "config_dir"
    config_dir.mkdir()
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("- not-a-mapping\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="not a file"):
        load_training_env_overrides(config_dir)

    with pytest.raises(ValueError, match="training config must be a mapping"):
        load_training_env_overrides(invalid_yaml)


def test_resolve_scenario_config_path_rejects_invalid_training_config(tmp_path: Path) -> None:
    """Scenario resolution should not silently ignore invalid provided training configs."""
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("- not-a-mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="training config must be a mapping"):
        resolve_scenario_config_path(
            scenario_config_path=None,
            training_config_path=invalid_yaml,
        )


def test_make_training_contract_env_rejects_non_dict_observation_space(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Observation-key filtering must fail when the env cannot honor a dict-style contract."""

    class DummyEnv:
        observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=float)

    monkeypatch.setattr(
        "scripts.training.imitation_env_contract.make_robot_env",
        lambda config, seed=None: DummyEnv(),
    )

    with pytest.raises(ValueError, match="expected Dict observation space"):
        make_training_contract_env(
            training_config_path=None,
            scenario_config_path=None,
            observation_keys=["robot"],
        )
