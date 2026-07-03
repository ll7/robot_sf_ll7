"""Config validation tests for issue #4016 QR-DQN trainer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from scripts.training.train_distributional_rl import load_distributional_rl_training_config

if TYPE_CHECKING:
    from pathlib import Path


def _write_config(tmp_path: Path, extra: str = "") -> Path:
    path = tmp_path / "qr_dqn.yaml"
    path.write_text(
        f"""
policy_id: qr_dqn_test
algorithm: qr_dqn
scenario_config: configs/scenarios/sets/classic_cross_trap_subset.yaml
total_timesteps: 8
seed: 4016
device: cpu
num_envs: 1
observation:
  synthetic_observation_dim: 4
action_lattice:
  linear_values: [0.0, 0.5]
  angular_values: [-0.5, 0.5]
  max_linear_speed: 0.5
  max_angular_speed: 0.5
critic:
  hidden_sizes: [8]
  num_quantiles: 4
risk_selection:
  objective: cvar_lower
  alpha: 0.5
dqn:
  batch_size: 2
  learning_starts: 2
output_dir: {tmp_path / "out"}
{extra}
""",
        encoding="utf-8",
    )
    return path


def test_distributional_rl_config_loads_smoke_yaml() -> None:
    """Repository smoke config stays loadable."""

    config = load_distributional_rl_training_config(
        "configs/training/distributional_rl/qr_dqn_issue_4016_smoke.yaml"
    )

    assert config.policy_id == "qr_dqn_issue_4016_smoke"
    assert config.algorithm == "qr_dqn"
    assert config.action_lattice.action_count == 25
    assert config.risk_selection.objective == "cvar_lower"


def test_distributional_rl_config_rejects_unknown_root_key(tmp_path: Path) -> None:
    """Unknown YAML root keys fail before training."""

    path = _write_config(tmp_path, extra="surprise: true\n")

    with pytest.raises(ValueError, match="unknown training config keys"):
        load_distributional_rl_training_config(path)


def test_distributional_rl_config_rejects_invalid_quantile_count(tmp_path: Path) -> None:
    """QR-DQN checkpoints require at least two quantiles."""

    path = _write_config(tmp_path)
    text = path.read_text(encoding="utf-8").replace("num_quantiles: 4", "num_quantiles: 1")
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="num_quantiles"):
        load_distributional_rl_training_config(path)


def test_distributional_rl_config_rejects_invalid_lattice(tmp_path: Path) -> None:
    """Action lattice validation is reused by the trainer."""

    path = _write_config(tmp_path)
    text = path.read_text(encoding="utf-8").replace(
        "max_linear_speed: 0.5", "max_linear_speed: 0.1"
    )
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="linear_values"):
        load_distributional_rl_training_config(path)
