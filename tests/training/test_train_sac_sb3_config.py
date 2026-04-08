"""Config-load and dry-run tests for the SAC training script (issue #790)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.training.train_sac_sb3 import (
    SACTrainingConfig,
    build_arg_parser,
    load_sac_training_config,
    run_sac_training,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GATE_CONFIG = Path("configs/training/sac/gate.yaml")
_FULL_CONFIG = Path("configs/training/sac/full.yaml")
_CLASSIC_SCENARIO = Path("configs/scenarios/classic_interactions.yaml").resolve()


def _minimal_config_yaml(tmp_path: Path, scenario_yaml: Path) -> Path:
    """Write the smallest valid SAC YAML config for testing.

    Args:
        tmp_path: Pytest temporary directory.
        scenario_yaml: Path to a scenario YAML file.

    Returns:
        Path: Path to the written config file.
    """
    content = f"""\
policy_id: sac_test
scenario_config: {scenario_yaml}
total_timesteps: 2000
seed: 0
sac_hyperparams:
  buffer_size: 2000
  learning_starts: 100
  batch_size: 64
env_factory_kwargs:
  reward_name: route_completion_v2
tracking:
  enabled: false
"""
    cfg = tmp_path / "sac_test.yaml"
    cfg.write_text(content, encoding="utf-8")
    return cfg


# ---------------------------------------------------------------------------
# Config-load tests
# ---------------------------------------------------------------------------


def test_gate_config_loads() -> None:
    """Gate config should load without error."""
    assert _GATE_CONFIG.exists(), f"Missing gate config: {_GATE_CONFIG}"
    config = load_sac_training_config(_GATE_CONFIG)
    assert config.policy_id
    assert config.total_timesteps == 50_000
    assert config.scenario_config.exists()
    assert config.output_dir == Path("output/models/sac").resolve()


def test_full_config_loads() -> None:
    """Full config should load without error."""
    assert _FULL_CONFIG.exists(), f"Missing full config: {_FULL_CONFIG}"
    config = load_sac_training_config(_FULL_CONFIG)
    assert config.policy_id
    assert config.total_timesteps == 3_000_000
    assert config.scenario_config.exists()
    assert config.output_dir == Path("output/models/sac").resolve()


def test_load_rejects_unknown_hyperparams(tmp_path: Path) -> None:
    """Config loader should raise on unrecognised SAC hyperparameter keys."""
    scenario_path = _CLASSIC_SCENARIO
    content = f"""\
policy_id: test
scenario_config: {scenario_path}
total_timesteps: 1000
sac_hyperparams:
  not_a_real_param: 99
"""
    cfg = tmp_path / "bad.yaml"
    cfg.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match="not_a_real_param"):
        load_sac_training_config(cfg)


def test_load_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    """Config loader should raise when YAML root is not a mapping."""
    cfg = tmp_path / "list.yaml"
    cfg.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_sac_training_config(cfg)


# ---------------------------------------------------------------------------
# Dry-run integration test
# ---------------------------------------------------------------------------


def test_dry_run_completes(tmp_path: Path) -> None:
    """Dry-run (1 000 steps) should complete and produce a checkpoint."""
    scenario_path = _CLASSIC_SCENARIO
    cfg_path = _minimal_config_yaml(tmp_path, scenario_path)
    config = load_sac_training_config(cfg_path)
    config.output_dir = tmp_path / "checkpoints"

    checkpoint = run_sac_training(config, dry_run=True)

    assert checkpoint.exists(), f"Checkpoint not found at {checkpoint}"
    assert checkpoint.suffix == ".zip"


def test_load_rejects_unknown_root_keys(tmp_path: Path) -> None:
    """Config loader should reject unknown top-level keys."""
    content = f"""\
policy_id: test
scenario_config: {_CLASSIC_SCENARIO}
total_timesteps: 1000
unexpected: true
"""
    cfg = tmp_path / "bad_root.yaml"
    cfg.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown root config keys"):
        load_sac_training_config(cfg)


# ---------------------------------------------------------------------------
# CLI argument parser tests
# ---------------------------------------------------------------------------


def test_arg_parser_requires_config() -> None:
    """CLI must require --config argument."""
    parser = build_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_arg_parser_accepts_dry_run() -> None:
    """CLI --dry-run flag should be parsed correctly."""
    parser = build_arg_parser()
    args = parser.parse_args(["--config", "some.yaml", "--dry-run"])
    assert args.dry_run is True


def test_arg_parser_default_log_level() -> None:
    """CLI default log level should be INFO."""
    parser = build_arg_parser()
    args = parser.parse_args(["--config", "some.yaml"])
    assert args.log_level == "INFO"


# ---------------------------------------------------------------------------
# SACTrainingConfig field validation
# ---------------------------------------------------------------------------


def test_config_dataclass_stores_seed() -> None:
    """SACTrainingConfig seed field should propagate to the model."""
    config = SACTrainingConfig(
        policy_id="test",
        scenario_config=Path("/dev/null"),
        total_timesteps=1000,
        seed=42,
    )
    assert config.seed == 42


def test_config_dataclass_seed_optional() -> None:
    """SACTrainingConfig seed field should default to None."""
    config = SACTrainingConfig(
        policy_id="test",
        scenario_config=Path("/dev/null"),
        total_timesteps=1000,
    )
    assert config.seed is None
