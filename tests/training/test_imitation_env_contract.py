"""Regression tests for imitation environment contract helpers.

These tests verify fail-closed handling for training-config inputs and observation filtering
because the BC/PPO warm-start path depends on reconstructing the exact observation contract
instead of silently falling back to defaults.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from gymnasium import spaces

from scripts.training.imitation_env_contract import (
    load_training_env_factory_kwargs,
    load_training_env_overrides,
    make_training_contract_env,
    resolve_scenario_config_path,
)


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


def test_training_config_contract_resolves_base_inheritance() -> None:
    """Imitation consumers must receive the effective mapping from an inherited PPO config."""
    repo_root = Path(__file__).resolve().parents[2]
    training_config = (
        repo_root / "configs/training/ppo/expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml"
    ).resolve()
    expected_scenario = (
        repo_root / "configs/scenarios/classic_interactions_francis2023.yaml"
    ).resolve()

    overrides = load_training_env_overrides(training_config)
    factory_kwargs = load_training_env_factory_kwargs(training_config)
    robot_config = overrides["robot_config"]

    assert overrides["observation_mode"] == "socnav_struct"
    assert isinstance(robot_config, dict)
    assert robot_config["max_linear_speed"] == 3.0
    assert factory_kwargs["reward_name"] == "route_completion_v3"
    assert (
        resolve_scenario_config_path(
            scenario_config_path=None,
            training_config_path=training_config,
        )
        == expected_scenario
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


def test_make_training_contract_env_rejects_reserved_factory_kwargs(
    tmp_path: Path,
) -> None:
    """Factory kwargs should fail before Python raises duplicate-argument TypeError."""
    config_path = tmp_path / "training.yaml"
    config_path.write_text(
        "\n".join(
            [
                "env_factory_kwargs:",
                "  seed: 123",
                "  config: ignored",
                "",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="config, seed"):
        make_training_contract_env(
            training_config_path=config_path,
            scenario_config_path=None,
        )
