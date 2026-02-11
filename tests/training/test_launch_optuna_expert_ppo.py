"""Tests for Optuna launcher config parsing and CLI argument construction."""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from scripts.training.launch_optuna_expert_ppo import build_optuna_cli_args, load_launch_config


def _make_args(**overrides: object) -> argparse.Namespace:
    defaults = {
        "base_config": None,
        "trials": None,
        "metric": None,
        "objective_mode": None,
        "objective_window": None,
        "trial_timesteps": None,
        "eval_every": None,
        "eval_episodes": None,
        "study_name": None,
        "storage": None,
        "seed": None,
        "log_level": None,
        "disable_wandb": None,
        "deterministic": None,
        "dry_run": False,
        "config": Path("configs/training/ppo_imitation/optuna_expert_ppo.yaml"),
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_load_launch_config_rejects_unknown_keys(tmp_path: Path):
    """Unknown launcher keys should fail fast."""
    config_path = tmp_path / "optuna.yaml"
    config_path.write_text("base_config: expert_ppo.yaml\nunexpected_key: 1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown launcher config key"):
        load_launch_config(config_path)


def test_build_optuna_cli_args_resolves_relative_paths(tmp_path: Path):
    """Relative base config should resolve from launcher config location."""
    launch_dir = tmp_path / "configs"
    launch_dir.mkdir(parents=True)
    expert_config = launch_dir / "expert_ppo.yaml"
    expert_config.write_text("scenario_config: dummy.yaml\n", encoding="utf-8")
    launch_config = launch_dir / "optuna.yaml"
    launch_config.write_text("base_config: expert_ppo.yaml\n", encoding="utf-8")
    payload = load_launch_config(launch_config)

    cli_args = build_optuna_cli_args(
        launch_config_path=launch_config.resolve(),
        payload=payload,
        args=_make_args(),
    )

    assert cli_args[:2] == ["--config", str(expert_config.resolve())]


def test_build_optuna_cli_args_applies_cli_overrides(tmp_path: Path):
    """CLI values should override launcher defaults."""
    launch_dir = tmp_path / "configs"
    launch_dir.mkdir(parents=True)
    expert_config = launch_dir / "expert_ppo.yaml"
    expert_config.write_text("scenario_config: dummy.yaml\n", encoding="utf-8")
    launch_config = launch_dir / "optuna.yaml"
    launch_config.write_text("base_config: expert_ppo.yaml\n", encoding="utf-8")
    payload = {
        "base_config": "expert_ppo.yaml",
        "trials": 24,
        "objective_mode": "last_n_mean",
        "log_level": "INFO",
        "disable_wandb": True,
    }

    cli_args = build_optuna_cli_args(
        launch_config_path=launch_config.resolve(),
        payload=payload,
        args=_make_args(
            trials=4,
            objective_mode="auc",
            log_level="ERROR",
            disable_wandb=False,
            deterministic=True,
        ),
    )

    assert "--trials" in cli_args
    assert cli_args[cli_args.index("--trials") + 1] == "4"
    assert "--objective-mode" in cli_args
    assert cli_args[cli_args.index("--objective-mode") + 1] == "auc"
    assert "--log-level" in cli_args
    assert cli_args[cli_args.index("--log-level") + 1] == "ERROR"
    assert "--disable-wandb" not in cli_args
    assert "--deterministic" in cli_args


def test_build_optuna_cli_args_parses_string_booleans(tmp_path: Path):
    """Quoted YAML boolean strings should be interpreted safely."""
    launch_dir = tmp_path / "configs"
    launch_dir.mkdir(parents=True)
    expert_config = launch_dir / "expert_ppo.yaml"
    expert_config.write_text("scenario_config: dummy.yaml\n", encoding="utf-8")
    launch_config = launch_dir / "optuna.yaml"
    launch_config.write_text("base_config: expert_ppo.yaml\n", encoding="utf-8")

    cli_args = build_optuna_cli_args(
        launch_config_path=launch_config.resolve(),
        payload={
            "base_config": "expert_ppo.yaml",
            "disable_wandb": "false",
            "deterministic": "true",
        },
        args=_make_args(),
    )

    assert "--disable-wandb" not in cli_args
    assert "--deterministic" in cli_args
