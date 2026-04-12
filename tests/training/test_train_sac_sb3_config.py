"""Config-load and dry-run tests for the SAC training script (issue #790)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import gymnasium
import numpy as np
import pytest
from gymnasium import spaces as gym_spaces

from robot_sf.training.scenario_loader import load_scenarios
from scripts.training.train_sac_sb3 import (
    SACEvaluationConfig,
    SACTrainingConfig,
    _build_env,
    _default_eval_algo_config,
    _ego_socnav_obs,
    _PeriodicSACEvaluationCallback,
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
    assert config.output_dir == (_GATE_CONFIG.parent / "output/models/sac").resolve()


def test_full_config_loads() -> None:
    """Full config should load without error."""
    assert _FULL_CONFIG.exists(), f"Missing full config: {_FULL_CONFIG}"
    config = load_sac_training_config(_FULL_CONFIG)
    assert config.policy_id
    assert config.total_timesteps == 3_000_000
    assert config.scenario_config.exists()
    assert config.output_dir == (_FULL_CONFIG.parent / "output/models/sac").resolve()


def test_socnav_gate_config_loads() -> None:
    """Benchmark-compatible gate config should load with socnav_struct override."""
    assert _SOCNAV_GATE_CONFIG.exists(), f"Missing gate config: {_SOCNAV_GATE_CONFIG}"
    config = load_sac_training_config(_SOCNAV_GATE_CONFIG)
    assert config.env_overrides == {"observation_mode": "socnav_struct"}
    assert config.action_semantics == "delta"
    assert config.relative_obs is True
    assert config.num_envs == 1


def test_num_envs_config_loads(tmp_path: Path) -> None:
    """SAC config should parse num_envs from YAML."""
    content = f"""\
policy_id: sac_env_count_test
scenario_config: {_CLASSIC_SCENARIO}
total_timesteps: 1000
num_envs: 3
tracking:
  enabled: false
"""
    cfg = tmp_path / "num_envs.yaml"
    cfg.write_text(content, encoding="utf-8")

    config = load_sac_training_config(cfg)
    assert config.num_envs == 3


def test_num_envs_must_be_positive(tmp_path: Path) -> None:
    """SAC config should reject non-positive num_envs values."""
    content = f"""\
policy_id: sac_env_count_bad
scenario_config: {_CLASSIC_SCENARIO}
total_timesteps: 1000
num_envs: 0
tracking:
  enabled: false
"""
    cfg = tmp_path / "num_envs_bad.yaml"
    cfg.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="num_envs"):
        load_sac_training_config(cfg)


def test_scenario_sampling_config_loads(tmp_path: Path) -> None:
    """SAC config should accept weighted scenario sampling settings."""
    content = f"""\
policy_id: sac_sample_test
scenario_config: {_CLASSIC_SCENARIO}
total_timesteps: 1000
seed: 0
scenario_sampling:
  strategy: cycle
  include_scenarios: [head_on_interaction, narrow_passage]
  weights:
    head_on_interaction: 3.0
    narrow_passage: 1.0
sac_hyperparams:
  buffer_size: 2000
  learning_starts: 100
  batch_size: 64
env_factory_kwargs:
  reward_name: route_completion_v2
tracking:
  enabled: false
"""
    cfg = tmp_path / "sac_sample.yaml"
    cfg.write_text(content, encoding="utf-8")

    config = load_sac_training_config(cfg)
    assert config.scenario_sampling.strategy == "cycle"
    assert config.scenario_sampling.include_scenarios == ("head_on_interaction", "narrow_passage")
    assert config.scenario_sampling.weights == {
        "head_on_interaction": 3.0,
        "narrow_passage": 1.0,
    }


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


def test_build_env_switches_across_loaded_scenarios(monkeypatch: pytest.MonkeyPatch) -> None:
    """SAC training env should sample across the loaded scenario list per reset."""

    class _DummyEnv(gymnasium.Env):
        metadata: ClassVar[dict[str, object]] = {}

        def __init__(
            self, *, config: object, seed: int | None, scenario_name: str, **_: object
        ) -> None:
            super().__init__()
            self.observation_space = gym_spaces.Dict(
                {
                    "robot_position": gym_spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                    "goal_current": gym_spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                }
            )
            self.action_space = gym_spaces.Box(
                low=np.array([0.0, -1.0], dtype=np.float32),
                high=np.array([2.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
            self.applied_seed = seed
            self.scenario_name = scenario_name
            self.config = config

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            if seed is not None:
                self.applied_seed = seed
            obs = {
                "robot_position": np.zeros(2, dtype=np.float32),
                "goal_current": np.ones(2, dtype=np.float32),
            }
            return obs, {"scenario_name": self.scenario_name}

        def step(self, action: np.ndarray):
            obs = {
                "robot_position": np.zeros(2, dtype=np.float32),
                "goal_current": np.ones(2, dtype=np.float32),
            }
            return obs, 0.0, False, False, {"scenario_name": self.scenario_name}

    def _fake_build_robot_config_from_scenario(scenario: dict[str, object], *, scenario_path: Path):
        return SimpleNamespace(
            scenario_name=scenario["name"],
            scenario_path=scenario_path,
            observation_mode="socnav_struct",
        )

    monkeypatch.setattr(
        "scripts.training.train_sac_sb3.build_robot_config_from_scenario",
        _fake_build_robot_config_from_scenario,
    )
    monkeypatch.setattr("scripts.training.train_sac_sb3.make_robot_env", _DummyEnv)

    config = SACTrainingConfig(
        policy_id="test",
        scenario_config=Path("configs/scenarios/classic_interactions.yaml").resolve(),
        total_timesteps=1000,
        seed=7,
        relative_obs=False,
        obs_transform="none",
    )
    scenario_definitions = [{"name": "scenario_a"}, {"name": "scenario_b"}]
    vec_env = _build_env(config, scenario_definitions=scenario_definitions)

    try:
        seen: set[str] = set()
        for _ in range(8):
            vec_env.reset()
            active = vec_env.envs[0]
            seen.add(str(getattr(active, "scenario_id", "")))
        assert seen == {"scenario_a", "scenario_b"}
    finally:
        vec_env.close()


def test_build_env_uses_subproc_vec_env_for_multi_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SAC training env should use subprocess vectorization when num_envs > 1."""

    class _DummyEnv(gymnasium.Env):
        metadata: ClassVar[dict[str, object]] = {}

        def __init__(
            self, *, config: object, seed: int | None, scenario_name: str, **_: object
        ) -> None:
            super().__init__()
            self.observation_space = gym_spaces.Dict(
                {
                    "robot_position": gym_spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                    "goal_current": gym_spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
                }
            )
            self.action_space = gym_spaces.Box(
                low=np.array([0.0, -1.0], dtype=np.float32),
                high=np.array([2.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
            self.applied_seed = seed
            self.scenario_name = scenario_name
            self.config = config

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            if seed is not None:
                self.applied_seed = seed
            obs = {
                "robot_position": np.zeros(2, dtype=np.float32),
                "goal_current": np.ones(2, dtype=np.float32),
            }
            return obs, {"scenario_name": self.scenario_name}

        def step(self, action: np.ndarray):
            obs = {
                "robot_position": np.zeros(2, dtype=np.float32),
                "goal_current": np.ones(2, dtype=np.float32),
            }
            return obs, 0.0, False, False, {"scenario_name": self.scenario_name}

    class _FakeSubprocVecEnv:
        def __init__(self, env_fns: list[object]) -> None:
            self.envs = [fn() for fn in env_fns]
            self.observation_space = self.envs[0].observation_space
            self.action_space = self.envs[0].action_space

        def close(self) -> None:
            for env in self.envs:
                close = getattr(env, "close", None)
                if callable(close):
                    close()

    def _fake_build_robot_config_from_scenario(scenario: dict[str, object], *, scenario_path: Path):
        return SimpleNamespace(
            scenario_name=scenario["name"],
            scenario_path=scenario_path,
            observation_mode="socnav_struct",
        )

    monkeypatch.setattr(
        "scripts.training.train_sac_sb3.build_robot_config_from_scenario",
        _fake_build_robot_config_from_scenario,
    )
    monkeypatch.setattr("scripts.training.train_sac_sb3.make_robot_env", _DummyEnv)
    monkeypatch.setattr("scripts.training.train_sac_sb3.SubprocVecEnv", _FakeSubprocVecEnv)

    config = SACTrainingConfig(
        policy_id="test",
        scenario_config=Path("configs/scenarios/classic_interactions.yaml").resolve(),
        total_timesteps=1000,
        seed=11,
        relative_obs=False,
        obs_transform="none",
        num_envs=3,
    )
    scenario_definitions = [{"name": "scenario_a"}, {"name": "scenario_b"}]
    vec_env = _build_env(config, scenario_definitions=scenario_definitions)

    try:
        assert len(vec_env.envs) == 3
        assert [env.applied_seed for env in vec_env.envs] == [11, 12, 13]
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


def test_ego_socnav_obs_rotates_goal_and_pedestrians_into_robot_frame() -> None:
    """Ego transform should rotate translated position-like keys by robot heading."""
    obs = {
        "robot_position": np.array([10.0, 5.0], dtype=np.float32),
        "robot_heading": np.array([np.pi / 2], dtype=np.float32),
        "goal_current": np.array([13.0, 5.0], dtype=np.float32),
        "pedestrians_positions": np.array([[11.0, 5.0], [10.0, 7.0]], dtype=np.float32),
    }

    ego = _ego_socnav_obs(obs)

    assert np.allclose(ego["robot_position"], np.array([0.0, 0.0], dtype=np.float32), atol=1e-6)
    assert np.allclose(ego["goal_current"], np.array([0.0, -3.0], dtype=np.float32), atol=1e-6)
    assert np.allclose(
        ego["pedestrians_positions"][0],
        np.array([0.0, -1.0], dtype=np.float32),
        atol=1e-6,
    )
    assert np.allclose(
        ego["pedestrians_positions"][1],
        np.array([2.0, 0.0], dtype=np.float32),
        atol=1e-6,
    )


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


def test_config_dataclass_num_envs_defaults_to_one() -> None:
    """SACTrainingConfig num_envs field should default to one environment."""
    config = SACTrainingConfig(
        policy_id="test",
        scenario_config=Path("/dev/null"),
        total_timesteps=1000,
    )
    assert config.num_envs == 1


def test_gate_v2_config_loads() -> None:
    """Benchmark-compatible v2 gate config should load with absolute action semantics."""
    cfg_path = Path("configs/training/sac/gate_socnav_struct_v2.yaml")
    assert cfg_path.exists(), f"Missing v2 config: {cfg_path}"
    config = load_sac_training_config(cfg_path)
    assert config.action_semantics == "absolute"
    assert config.relative_obs is True
    assert config.total_timesteps == 200_000


def test_gate_ego_config_loads() -> None:
    """Ego SAC gate config should resolve the explicit obs transform."""
    cfg_path = Path("configs/training/sac/gate_socnav_struct_ego.yaml")
    assert cfg_path.exists(), f"Missing ego config: {cfg_path}"
    config = load_sac_training_config(cfg_path)
    assert config.obs_transform == "ego"
    assert config.action_semantics == "delta"


def test_gate_ego_multi_config_loads() -> None:
    """Multi-scenario ego SAC config should load with the expected transform."""
    cfg_path = Path("configs/training/sac/gate_socnav_struct_ego_multi_v1.yaml")
    assert cfg_path.exists(), f"Missing ego multi config: {cfg_path}"
    config = load_sac_training_config(cfg_path)
    assert config.obs_transform == "ego"
    assert config.action_semantics == "delta"


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


def test_eval_config_loads_and_resolves_relative_paths(tmp_path: Path) -> None:
    """SAC evaluation settings should parse and resolve relative paths."""
    scenario_matrix = Path("configs/scenarios/sets/ppo_full_maintained_eval_v1.yaml").resolve()
    algo_config = Path("configs/baselines/sac_gate_socnav_struct.yaml").resolve()
    content = f"""\
policy_id: test
scenario_config: {_CLASSIC_SCENARIO}
total_timesteps: 1000
evaluation:
  enabled: true
  frequency_steps: 250
  scenario_matrix: {scenario_matrix}
  algo_config: {algo_config}
  output_dir: output/tmp/sac_eval_custom
  tag_prefix: custom_eval
  horizon: 64
  dt: 0.2
  workers: 2
  min_success_rate: 0.4
  device: cpu
"""
    cfg = tmp_path / "with_eval.yaml"
    cfg.write_text(content, encoding="utf-8")
    config = load_sac_training_config(cfg)
    assert config.evaluation.enabled is True
    assert config.evaluation.frequency_steps == 250
    assert config.evaluation.scenario_matrix == scenario_matrix.resolve()
    assert config.evaluation.algo_config == algo_config.resolve()
    assert config.evaluation.output_dir == (cfg.parent / "output/tmp/sac_eval_custom").resolve()
    assert config.evaluation.tag_prefix == "custom_eval"
    assert config.evaluation.horizon == 64
    assert config.evaluation.dt == 0.2
    assert config.evaluation.workers == 2
    assert config.evaluation.min_success_rate == 0.4
    assert config.evaluation.device == "cpu"


def test_default_eval_algo_config_switches_for_ego_mode() -> None:
    """Periodic eval should use the ego-specific baseline config when needed."""
    config = SACTrainingConfig(
        policy_id="test",
        scenario_config=_CLASSIC_SCENARIO,
        total_timesteps=1000,
        obs_transform="ego",
    )
    assert _default_eval_algo_config(config) == Path(
        "configs/baselines/sac_gate_socnav_struct_ego.yaml"
    )


def test_periodic_eval_callback_runs_wrapper_and_logs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Periodic callback should save a checkpoint and call the reusable eval wrapper."""

    class _DummyModel:
        def __init__(self) -> None:
            self.saved_paths: list[Path] = []

        def save(self, path: str) -> None:
            self.saved_paths.append(Path(path))
            Path(path).write_text("stub", encoding="utf-8")

    class _DummyWandb:
        def __init__(self) -> None:
            self.logs: list[dict[str, float]] = []

        def log(self, payload, step=None):
            self.logs.append(dict(payload))

    eval_calls: list[dict[str, object]] = []

    def _fake_eval(**kwargs):
        eval_calls.append(kwargs)
        return {
            "success_rate": 0.75,
            "mean_min_distance": 0.8,
            "mean_avg_speed": 0.5,
            "gate_pass": True,
        }

    monkeypatch.setattr("scripts.validation.evaluate_sac.run_sac_evaluation", _fake_eval)

    training_config = SACTrainingConfig(
        policy_id="test",
        scenario_config=_CLASSIC_SCENARIO,
        total_timesteps=1000,
        output_dir=tmp_path / "models",
    )
    evaluation_config = SACEvaluationConfig(
        enabled=True,
        frequency_steps=10,
        scenario_matrix=Path("configs/scenarios/sets/ppo_full_maintained_eval_v1.yaml").resolve(),
        algo_config=Path("configs/baselines/sac_gate_socnav_struct.yaml").resolve(),
        output_dir=tmp_path / "eval",
        tag_prefix="periodic",
        horizon=20,
        dt=0.1,
        workers=1,
        min_success_rate=0.3,
        device="cpu",
    )
    callback = _PeriodicSACEvaluationCallback(
        training_config=training_config,
        evaluation_config=evaluation_config,
        wandb_run=_DummyWandb(),
    )
    callback.model = _DummyModel()
    callback.num_timesteps = 10

    assert callback._on_step() is True
    assert len(eval_calls) == 1
    assert eval_calls[0]["scenario_matrix"] == evaluation_config.scenario_matrix
    assert eval_calls[0]["tag"] == "periodic_00000010"
    assert callback.model.saved_paths[0].name == "test_00000010.zip"
    assert callback._wandb_run.logs[0]["sac/eval_success_rate"] == 0.75
