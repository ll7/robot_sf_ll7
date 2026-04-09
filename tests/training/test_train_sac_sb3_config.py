"""Config-load and dry-run tests for the SAC training script (issue #790)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces as gym_spaces

from robot_sf.training.scenario_loader import load_scenarios
from scripts.training.train_sac_sb3 import (
    SACTrainingConfig,
    _build_env,
    _relative_socnav_obs,
    build_arg_parser,
    load_sac_training_config,
    run_sac_training,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GATE_CONFIG = Path("configs/training/sac/gate.yaml")
_FULL_CONFIG = Path("configs/training/sac/full.yaml")
_SOCNAV_GATE_CONFIG = Path("configs/training/sac/gate_socnav_struct.yaml")
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


def test_socnav_gate_config_loads() -> None:
    """Benchmark-compatible gate config should load with socnav_struct override."""
    assert _SOCNAV_GATE_CONFIG.exists(), f"Missing gate config: {_SOCNAV_GATE_CONFIG}"
    config = load_sac_training_config(_SOCNAV_GATE_CONFIG)
    assert config.env_overrides == {"observation_mode": "socnav_struct"}
    assert config.action_semantics == "delta"
    assert config.relative_obs is True


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


def test_build_env_uses_expected_observation_and_action_contract() -> None:
    """Env construction should preserve the PPO-compatible observation/action contract."""
    config = load_sac_training_config(_GATE_CONFIG)
    scenario_definitions = load_scenarios(config.scenario_config)
    vec_env = _build_env(config, scenario_definitions=scenario_definitions)

    try:
        assert isinstance(vec_env.observation_space, gym_spaces.Dict)
        assert set(vec_env.observation_space.spaces) == {"drive_state", "rays"}
        assert isinstance(vec_env.action_space, gym_spaces.Box)
        assert vec_env.action_space.shape == (2,)
        assert tuple(vec_env.action_space.low.tolist()) == (0.0, -1.0)
        assert tuple(vec_env.action_space.high.tolist()) == (2.0, 1.0)
        assert getattr(vec_env.envs[0], "applied_seed", None) == config.seed
    finally:
        vec_env.close()


def test_build_env_applies_socnav_struct_override() -> None:
    """Env construction should support benchmark-compatible socnav_struct observations."""
    config = load_sac_training_config(_SOCNAV_GATE_CONFIG)
    scenario_definitions = load_scenarios(config.scenario_config)
    vec_env = _build_env(config, scenario_definitions=scenario_definitions)

    try:
        assert isinstance(vec_env.observation_space, gym_spaces.Dict)
        assert "robot_position" in vec_env.observation_space.spaces
        assert "goal_current" in vec_env.observation_space.spaces
        assert "pedestrians_positions" in vec_env.observation_space.spaces
        assert isinstance(vec_env.action_space, gym_spaces.Box)
        assert vec_env.action_space.shape == (2,)
    finally:
        vec_env.close()


def test_relative_socnav_obs_rebases_position_like_keys() -> None:
    """Relative observation preprocessing should remove absolute map offsets."""
    obs = {
        "robot_position": np.array([10.0, 5.0], dtype=np.float32),
        "goal_current": np.array([13.0, 7.0], dtype=np.float32),
        "goal_next": np.array([15.0, 9.0], dtype=np.float32),
        "pedestrians_positions": np.array([[11.0, 6.0], [0.0, 0.0]], dtype=np.float32),
        "robot_speed": np.array([0.2, 0.0], dtype=np.float32),
    }

    rel = _relative_socnav_obs(obs)

    assert rel["robot_position"] == pytest.approx(np.array([0.0, 0.0], dtype=np.float32))
    assert rel["goal_current"] == pytest.approx(np.array([3.0, 2.0], dtype=np.float32))
    assert rel["goal_next"] == pytest.approx(np.array([5.0, 4.0], dtype=np.float32))
    assert rel["pedestrians_positions"][0] == pytest.approx(np.array([1.0, 1.0], dtype=np.float32))
    assert rel["pedestrians_positions"][1] == pytest.approx(np.array([0.0, 0.0], dtype=np.float32))


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


def test_config_dataclass_device_defaults_to_auto() -> None:
    """SACTrainingConfig device field should default to 'auto'."""
    config = SACTrainingConfig(
        policy_id="test",
        scenario_config=Path("/dev/null"),
        total_timesteps=1000,
    )
    assert config.device == "auto"


def test_gate_v2_config_loads() -> None:
    """Benchmark-compatible v2 gate config should load with absolute action semantics."""
    cfg_path = Path("configs/training/sac/gate_socnav_struct_v2.yaml")
    assert cfg_path.exists(), f"Missing v2 config: {cfg_path}"
    config = load_sac_training_config(cfg_path)
    assert config.action_semantics == "absolute"
    assert config.relative_obs is True
    assert config.total_timesteps == 200_000


def test_config_action_semantics_defaults_to_delta() -> None:
    """SACTrainingConfig action_semantics should default to 'delta'."""
    config = SACTrainingConfig(
        policy_id="test",
        scenario_config=Path("/dev/null"),
        total_timesteps=1000,
    )
    assert config.action_semantics == "delta"


def test_config_relative_obs_defaults_to_true() -> None:
    """SACTrainingConfig should enable relative observations by default."""
    config = SACTrainingConfig(
        policy_id="test",
        scenario_config=Path("/dev/null"),
        total_timesteps=1000,
    )
    assert config.relative_obs is True


def test_config_device_loaded_from_yaml(tmp_path: Path) -> None:
    """Device field in YAML should propagate to SACTrainingConfig."""
    scenario_path = _CLASSIC_SCENARIO
    content = f"""\
policy_id: test
scenario_config: {scenario_path}
total_timesteps: 1000
device: cpu
"""
    cfg = tmp_path / "with_device.yaml"
    cfg.write_text(content, encoding="utf-8")
    config = load_sac_training_config(cfg)
    assert config.device == "cpu"
