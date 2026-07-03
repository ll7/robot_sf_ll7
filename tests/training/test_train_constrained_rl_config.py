"""Config validation tests for the issue #4017 constrained-RL training entry point."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.training.train_constrained_rl import load_constrained_rl_config

CONFIG_DIR = Path("configs/training/ppo")


def test_constrained_smoke_config_loads_constraint_specs() -> None:
    """The PPO-Lagrangian smoke config exposes the three initial safety budgets."""

    config = load_constrained_rl_config(CONFIG_DIR / "issue_4017_constrained_smoke.yaml")

    assert config.policy_id == "ppo_lagrangian_issue_4017_smoke"
    assert config.total_timesteps == 256
    assert config.device == "cpu"
    assert config.safety_constraints.enabled is True
    assert [spec.name for spec in config.safety_constraints.constraints] == [
        "collision_any",
        "near_miss",
        "comfort_exposure",
    ]


def test_unconstrained_smoke_config_matches_training_shape_without_constraints() -> None:
    """The matched baseline differs by policy id, output dir, and disabled constraints only."""

    constrained = load_constrained_rl_config(CONFIG_DIR / "issue_4017_constrained_smoke.yaml")
    baseline = load_constrained_rl_config(CONFIG_DIR / "issue_4017_unconstrained_smoke.yaml")

    assert baseline.safety_constraints.enabled is False
    assert baseline.safety_constraints.constraints == ()
    assert baseline.total_timesteps == constrained.total_timesteps
    assert baseline.seed == constrained.seed
    assert baseline.ppo_hyperparams == constrained.ppo_hyperparams
    assert baseline.env_overrides == constrained.env_overrides
    assert baseline.env_factory_kwargs == constrained.env_factory_kwargs


def test_unsupported_constraint_source_fails_closed(tmp_path: Path) -> None:
    """Unsupported safety-cost sources fail before any training launch."""

    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        """
policy_id: bad_constraint_source
algorithm: ppo
scenario_config: configs/scenarios/sets/classic_cross_trap_subset.yaml
total_timesteps: 256
seed: 4017
num_envs: 1
device: cpu
safety_constraints:
  enabled: true
  constraints:
    - name: unknown
      source_key: unknown_metric
      budget_per_episode: 0.0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported safety-cost source"):
        load_constrained_rl_config(config_path)


def test_disabled_constraints_reject_accidental_constraint_list(tmp_path: Path) -> None:
    """Baseline configs must not silently carry inactive constraint definitions."""

    config_path = tmp_path / "disabled_with_constraints.yaml"
    config_path.write_text(
        """
policy_id: disabled_with_constraints
algorithm: ppo
scenario_config: configs/scenarios/sets/classic_cross_trap_subset.yaml
total_timesteps: 256
seed: 4017
num_envs: 1
device: cpu
safety_constraints:
  enabled: false
  constraints:
    - name: collision_any
      source_key: collision_any
      budget_per_episode: 0.0
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Disabled safety_constraints"):
        load_constrained_rl_config(config_path)
