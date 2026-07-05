"""Contract tests for expert PPO training runtime helpers."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
from typing import TYPE_CHECKING

from robot_sf.feature_extractors.grid_socnav_extractor import GridSocNavExtractor
from robot_sf.training.imitation_config import (
    ConvergenceCriteria,
    EvaluationSchedule,
    ExpertTrainingConfig,
)
from robot_sf.training.ppo_policy import AsymmetricGridSocNavPolicy
from robot_sf.training.snqi_utils import default_training_snqi_context
from scripts.training import train_ppo

if TYPE_CHECKING:
    from pathlib import Path


def test_make_training_env_factory_is_pickleable_for_spawn(tmp_path: Path) -> None:
    """SubprocVecEnv spawn mode requires environment callables to be pickleable."""
    single_scenario_factory = train_ppo._make_training_env(
        123,
        scenario={"id": "spawn-smoke"},
        scenario_definitions=None,
        scenario_path=tmp_path / "scenarios.yaml",
        exclude_scenarios=(),
        suite_name="ppo_imitation",
        algorithm_name="pickle_contract",
        env_overrides={},
        env_factory_kwargs={},
        scenario_sampling={},
    )
    switching_factory = train_ppo._make_training_env(
        456,
        scenario=None,
        scenario_definitions=({"id": "spawn-a"}, {"id": "spawn-b"}),
        scenario_path=tmp_path / "scenarios.yaml",
        exclude_scenarios=("skip-me",),
        suite_name="ppo_imitation",
        algorithm_name="pickle_contract",
        env_overrides={},
        env_factory_kwargs={},
        scenario_sampling={"strategy": "round_robin"},
    )

    restored_single = pickle.loads(pickle.dumps(single_scenario_factory))
    restored_switching = pickle.loads(pickle.dumps(switching_factory))

    assert type(restored_single) is type(single_scenario_factory)
    assert restored_single.seed == 123
    assert type(restored_switching) is type(switching_factory)
    assert restored_switching.seed == 456


def test_warn_frequency_episodes_deprecated_warns_once(monkeypatch) -> None:
    """frequency_episodes deprecation warning should be emitted once per process."""
    calls: list[str] = []

    def _fake_warning(msg: str, *_args) -> None:
        """Record deprecation warning messages."""
        calls.append(msg)

    monkeypatch.setattr(train_ppo.logger, "warning", _fake_warning)
    monkeypatch.setattr(train_ppo, "_FREQUENCY_EPISODES_DEPRECATION_WARNED", False)

    train_ppo._warn_frequency_episodes_deprecated(10)
    train_ppo._warn_frequency_episodes_deprecated(20)

    assert len(calls) == 1
    assert "ignored" in calls[0]


def test_prepare_seed_state_relaxes_determinism_for_lightweight_cnn(monkeypatch, tmp_path) -> None:
    """lightweight_cnn should opt out of deterministic CUDA with an explicit warning."""
    warnings: list[str] = []
    seed_calls: list[tuple[int, bool]] = []

    def _fake_warning(message: str, *args) -> None:
        """Record formatted warning messages from seed preparation."""
        warnings.append(message.format(*args) if args else message)

    def _fake_set_global_seed(seed: int, deterministic: bool = True):
        """Record seed and determinism settings without touching global RNG state."""
        seed_calls.append((seed, deterministic))
        return object()

    monkeypatch.setattr(train_ppo.logger, "warning", _fake_warning)
    monkeypatch.setattr(train_ppo.common, "set_global_seed", _fake_set_global_seed)

    config = ExpertTrainingConfig.from_raw(
        scenario_config=tmp_path / "scenarios.yaml",
        seeds=(123,),
        total_timesteps=32_000,
        policy_id="ppo_lightweight_cnn_seed_test",
        convergence=ConvergenceCriteria(
            success_rate=0.9,
            collision_rate=0.05,
            plateau_window=1000,
        ),
        evaluation=EvaluationSchedule(
            frequency_episodes=0,
            evaluation_episodes=4,
            step_schedule=((None, 16_000),),
            randomize_seeds=False,
        ),
        feature_extractor="lightweight_cnn",
    )

    train_ppo._prepare_seed_state(config)

    assert seed_calls == [(123, False)]
    assert len(warnings) == 1
    assert "LIGHTWEIGHT_CNN DETerminism Override" in warnings[0]
    assert "not bitwise reproducible" in warnings[0]


def test_legacy_training_ppo_entrypoint_fails_with_migration_command() -> None:
    """Legacy PPO entrypoint should fail closed before users launch invalid runs."""
    result = subprocess.run(
        [sys.executable, "scripts/training_ppo.py"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "scripts/training/train_ppo.py" in result.stderr
    assert "--config" in result.stderr
    assert "docs/training/ppo_training_workflow.md" in result.stderr


def _capture_startup_summary(monkeypatch, tmp_path: Path) -> str:
    """Capture startup summary logs for focused contract assertions."""
    messages: list[str] = []

    def _fake_info(message: str, *args) -> None:
        """Record formatted startup summary info logs."""
        try:
            messages.append(message.format(*args) if args else message)
        except (IndexError, KeyError, ValueError):
            messages.append(f"{message} | args={args}")

    monkeypatch.setattr(train_ppo.logger, "info", _fake_info)
    monkeypatch.setattr(train_ppo, "_host_memory_gib", lambda: 64.0)
    monkeypatch.setattr(train_ppo, "_slurm_allocated_cpus", lambda: None)
    monkeypatch.setattr(train_ppo.os, "cpu_count", lambda: 12)

    config = ExpertTrainingConfig.from_raw(
        scenario_config=tmp_path / "scenarios.yaml",
        seeds=(123,),
        total_timesteps=120_000,
        policy_id="ppo_startup_summary_test",
        convergence=ConvergenceCriteria(
            success_rate=0.9,
            collision_rate=0.05,
            plateau_window=1000,
        ),
        evaluation=EvaluationSchedule(
            frequency_episodes=0,
            evaluation_episodes=4,
            step_schedule=((None, 60_000),),
            randomize_seeds=False,
        ),
        env_factory_kwargs={
            "reward_name": "route_completion_v3",
            "reward_curriculum": {
                "stages": [
                    {
                        "until_episodes": 1,
                        "reward_kwargs": {"weights": {"terminal_bonus": 1.0}},
                    },
                    {
                        "reward_kwargs": {"weights": {"terminal_bonus": 5.0}},
                    },
                ]
            },
        },
        scenario_sampling={"strategy": "random"},
        num_envs="auto_stable",
        worker_mode="subproc",
        resume_model_id="ppo_registry_source",
    )

    train_ppo._log_startup_summary(
        config=config,
        config_path=tmp_path / "train.yaml",
        num_envs=3,
        worker_mode="subproc",
    )

    return "\n".join(messages)


def test_log_startup_summary_reports_run_identity(monkeypatch, tmp_path: Path) -> None:
    """Startup summary should report policy, config, and training horizon identity."""
    summary = _capture_startup_summary(monkeypatch, tmp_path)

    assert "Training startup summary" in summary
    assert "policy_id=ppo_startup_summary_test" in summary
    assert "config_path=" in summary
    assert "total_timesteps=120000" in summary


def test_log_startup_summary_reports_training_settings(monkeypatch, tmp_path: Path) -> None:
    """Startup summary should report resolved reward, worker, and resume settings."""
    summary = _capture_startup_summary(monkeypatch, tmp_path)

    assert "reward_profile=curriculum[2 stages]: route_completion_v3" in summary
    assert "requested_num_envs=auto_stable" in summary
    assert "num_envs=3" in summary
    assert "worker_mode=subproc" in summary
    assert "model_id:ppo_registry_source" in summary


def test_log_startup_summary_reports_num_envs_resolution(monkeypatch, tmp_path: Path) -> None:
    """Startup summary should include the auto num-envs resolution explanation."""
    summary = _capture_startup_summary(monkeypatch, tmp_path)

    assert "num_envs resolution" in summary
    assert "mode=auto_stable" in summary


def test_subproc_worker_log_environment_is_warning_only_during_spawn(monkeypatch) -> None:
    """PPO subproc workers should inherit a quiet startup log level without muting parent logs."""
    monkeypatch.setenv("LOGURU_LEVEL", "INFO")

    with train_ppo._subproc_worker_log_environment("subproc"):
        assert train_ppo.os.environ["LOGURU_LEVEL"] == "WARNING"

    assert train_ppo.os.environ["LOGURU_LEVEL"] == "INFO"


def test_dummy_worker_log_environment_leaves_log_level_unchanged(monkeypatch) -> None:
    """Non-subproc PPO runs should not alter log-level inheritance."""
    monkeypatch.setenv("LOGURU_LEVEL", "INFO")

    with train_ppo._subproc_worker_log_environment("dummy"):
        assert train_ppo.os.environ["LOGURU_LEVEL"] == "INFO"

    assert train_ppo.os.environ["LOGURU_LEVEL"] == "INFO"


def test_init_training_model_quiets_loguru_while_spawning_subproc_workers(
    monkeypatch, tmp_path: Path
) -> None:
    """Subproc worker creation should inherit WARNING while parent logging is restored."""
    observed_levels: list[str | None] = []
    monkeypatch.setenv("LOGURU_LEVEL", "INFO")
    monkeypatch.setattr(train_ppo, "_resolve_num_envs", lambda _config: 2)
    monkeypatch.setattr(train_ppo, "_resolve_worker_mode", lambda _config, _num_envs: "subproc")

    def _fake_make_training_env(*args, **kwargs):
        """Return a placeholder env factory for vectorized env construction."""
        return object

    monkeypatch.setattr(train_ppo, "_make_training_env", _fake_make_training_env)
    monkeypatch.setattr(
        train_ppo,
        "_resolve_policy_selection",
        lambda _config: ("MlpPolicy", {}, "mlp"),
    )
    monkeypatch.setattr(train_ppo, "_resolve_resume_checkpoint", lambda **kwargs: None)
    monkeypatch.setattr(train_ppo, "_resolve_ppo_hyperparams", lambda _config: {})

    class _FakeSubprocVecEnv:
        """SubprocVecEnv stub that records inherited LOGURU level."""

        def __init__(self, env_fns, *, start_method: str | None = None):
            assert start_method == "spawn"
            observed_levels.append(train_ppo.os.environ.get("LOGURU_LEVEL"))
            self.env_fns = env_fns

    class _FakePPO:
        """PPO constructor stub that records initialization arguments."""

        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

    monkeypatch.setattr(train_ppo, "SubprocVecEnv", _FakeSubprocVecEnv)
    monkeypatch.setattr(train_ppo, "PPO", _FakePPO)

    config = ExpertTrainingConfig.from_raw(
        scenario_config=tmp_path / "scenarios.yaml",
        seeds=(123,),
        total_timesteps=32_000,
        policy_id="ppo_subproc_log_level_test",
        convergence=ConvergenceCriteria(
            success_rate=0.9,
            collision_rate=0.05,
            plateau_window=1000,
        ),
        evaluation=EvaluationSchedule(
            frequency_episodes=0,
            evaluation_episodes=4,
            step_schedule=((None, 16_000),),
            randomize_seeds=False,
        ),
        worker_mode="subproc",
    )

    train_ppo._init_training_model(
        config=config,
        scenario=None,
        scenario_definitions=({"id": "scenario_a"},),
        exclude_scenarios=(),
        run_id="run",
        tensorboard_log=None,
        resume_from=None,
    )

    assert observed_levels == ["WARNING"]
    assert train_ppo.os.environ["LOGURU_LEVEL"] == "INFO"


def test_resolve_policy_selection_uses_asymmetric_grid_socnav_policy(tmp_path: Path) -> None:
    """Asymmetric critic config should select the dedicated policy class."""
    config = ExpertTrainingConfig.from_raw(
        scenario_config=tmp_path / "scenarios.yaml",
        seeds=(123,),
        total_timesteps=120_000,
        policy_id="ppo_asymmetric_policy_test",
        convergence=ConvergenceCriteria(
            success_rate=0.9,
            collision_rate=0.05,
            plateau_window=1000,
        ),
        evaluation=EvaluationSchedule(
            frequency_episodes=0,
            evaluation_episodes=4,
            step_schedule=((None, 60_000),),
            randomize_seeds=False,
        ),
        feature_extractor="grid_socnav",
        env_overrides={
            "observation_mode": "socnav_struct",
            "use_occupancy_grid": True,
            "include_grid_in_observation": True,
        },
        env_factory_kwargs={
            "reward_name": "route_completion_v3",
            "asymmetric_critic": True,
        },
        scenario_sampling={"strategy": "random"},
        num_envs="auto_stable",
        worker_mode="subproc",
    )

    policy_cls, policy_kwargs, critic_profile = train_ppo._resolve_policy_selection(config)

    assert policy_cls is AsymmetricGridSocNavPolicy
    assert policy_kwargs["features_extractor_class"] is GridSocNavExtractor
    assert critic_profile == "asymmetric_grid_socnav"


def test_resolve_policy_selection_attention_head_sets_profile(tmp_path: Path) -> None:
    """Attention head config should set the attention_grid_socnav critic profile."""
    config = ExpertTrainingConfig.from_raw(
        scenario_config=tmp_path / "scenarios.yaml",
        seeds=(123,),
        total_timesteps=120_000,
        policy_id="ppo_attention_policy_test",
        convergence=ConvergenceCriteria(
            success_rate=0.9,
            collision_rate=0.05,
            plateau_window=1000,
        ),
        evaluation=EvaluationSchedule(
            frequency_episodes=0,
            evaluation_episodes=4,
            step_schedule=((None, 60_000),),
            randomize_seeds=False,
        ),
        feature_extractor="grid_socnav",
        feature_extractor_kwargs={"use_pedestrian_attention": True},
        env_overrides={
            "observation_mode": "socnav_struct",
            "use_occupancy_grid": True,
            "include_grid_in_observation": True,
        },
        env_factory_kwargs={
            "reward_name": "route_completion_v3",
        },
        scenario_sampling={"strategy": "random"},
        num_envs="auto_stable",
        worker_mode="subproc",
    )

    _policy_cls, policy_kwargs, critic_profile = train_ppo._resolve_policy_selection(config)

    assert critic_profile == "attention_grid_socnav"
    assert policy_kwargs["features_extractor_kwargs"].get("use_pedestrian_attention") is True


def test_resolve_policy_selection_attention_plus_asymmetric_sets_combined_profile(
    tmp_path: Path,
) -> None:
    """Attention + asymmetric critic together should set the combined profile."""
    config = ExpertTrainingConfig.from_raw(
        scenario_config=tmp_path / "scenarios.yaml",
        seeds=(123,),
        total_timesteps=120_000,
        policy_id="ppo_attention_asymmetric_policy_test",
        convergence=ConvergenceCriteria(
            success_rate=0.9,
            collision_rate=0.05,
            plateau_window=1000,
        ),
        evaluation=EvaluationSchedule(
            frequency_episodes=0,
            evaluation_episodes=4,
            step_schedule=((None, 60_000),),
            randomize_seeds=False,
        ),
        feature_extractor="grid_socnav",
        feature_extractor_kwargs={"use_pedestrian_attention": True},
        env_overrides={
            "observation_mode": "socnav_struct",
            "use_occupancy_grid": True,
            "include_grid_in_observation": True,
        },
        env_factory_kwargs={
            "reward_name": "route_completion_v3",
            "asymmetric_critic": True,
        },
        scenario_sampling={"strategy": "random"},
        num_envs="auto_stable",
        worker_mode="subproc",
    )

    policy_cls, _policy_kwargs, critic_profile = train_ppo._resolve_policy_selection(config)

    assert policy_cls is AsymmetricGridSocNavPolicy
    assert critic_profile == "asymmetric_attention_grid_socnav"


def test_write_perf_summary_writes_expected_keys(tmp_path: Path, monkeypatch) -> None:
    """Perf summary writer should produce machine-readable aggregate keys."""
    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(tmp_path))
    path = train_ppo._write_perf_summary(
        run_id="demo_run",
        startup_sec=1.2,
        per_checkpoint_perf=[
            {
                "eval_step": 100,
                "train_steps": 100,
                "num_envs": 2,
                "train_wall_sec": 2.0,
                "eval_wall_sec": 3.0,
                "train_env_steps_per_sec": 100.0,
            }
        ],
        total_wall_clock_sec=12.0,
        parameter_summary={
            "available": True,
            "policy_parameter_count": 1234,
            "policy_trainable_parameter_count": 1200,
            "model_parameter_count": None,
            "model_trainable_parameter_count": None,
        },
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "demo_run"
    assert payload["startup_sec"] == 1.2
    assert payload["total_wall_clock_sec"] == 12.0
    assert payload["train_env_steps_per_sec_mean"] == 100.0
    assert payload["eval_sec_per_checkpoint"] == 3.0
    assert payload["parameter_summary"] == {
        "available": True,
        "policy_parameter_count": 1234,
        "policy_trainable_parameter_count": 1200,
        "model_parameter_count": None,
        "model_trainable_parameter_count": None,
    }


def test_model_parameter_summary_counts_policy_parameters() -> None:
    """Parameter summary should count total and trainable policy parameters."""

    class _Parameter:
        def __init__(self, count: int, *, requires_grad: bool) -> None:
            self._count = count
            self.requires_grad = requires_grad

        def numel(self) -> int:
            return self._count

    class _Policy:
        def parameters(self) -> list[_Parameter]:
            return [
                _Parameter(3, requires_grad=True),
                _Parameter(5, requires_grad=False),
            ]

    class _Model:
        policy = _Policy()

    assert train_ppo._model_parameter_summary(_Model()) == {
        "available": True,
        "policy_parameter_count": 8,
        "policy_trainable_parameter_count": 3,
        "model_parameter_count": None,
        "model_trainable_parameter_count": None,
    }


def _capture_evaluate_policy_info_logs(monkeypatch, tmp_path: Path, *, episodes: int) -> list[str]:
    """Run a fake PPO evaluation and capture info logs."""
    messages: list[str] = []

    def _fake_info(message: str, *args) -> None:
        """Record formatted evaluation info logs."""
        messages.append(message.format(*args) if args else message)

    class _FakeState:
        """Environment state stub exposing max episode steps."""

        max_sim_steps = 2

    class _FakeEnv:
        """Evaluation environment stub with one successful step."""

        state = _FakeState()

        def reset(self):
            """Return an initial observation for evaluation."""
            return [0.0], {}

        def step(self, _action):
            """Return a successful terminal transition."""
            return [0.0], 1.0, True, False, {"success": True}

        def close(self) -> None:
            """Close the fake evaluation environment without side effects."""
            return None

    class _FakeModel:
        """Model stub returning a deterministic action."""

        def predict(self, obs, deterministic: bool):
            """Assert deterministic prediction and return a dummy action."""
            assert deterministic is True
            return 0, None

    config = ExpertTrainingConfig.from_raw(
        scenario_config=tmp_path / "scenarios.yaml",
        seeds=(123,),
        total_timesteps=120_000,
        policy_id="ppo_eval_progress_test",
        convergence=ConvergenceCriteria(
            success_rate=0.9,
            collision_rate=0.05,
            plateau_window=1000,
        ),
        evaluation=EvaluationSchedule(
            frequency_episodes=0,
            evaluation_episodes=episodes,
            step_schedule=((None, 60_000),),
            randomize_seeds=False,
        ),
    )

    monkeypatch.setattr(train_ppo.logger, "info", _fake_info)
    monkeypatch.setattr(
        train_ppo,
        "build_robot_config_from_scenario",
        lambda scenario, *, scenario_path: object(),
    )
    monkeypatch.setattr(train_ppo, "_apply_env_overrides", lambda env_config, overrides: None)
    monkeypatch.setattr(train_ppo, "make_robot_env", lambda **kwargs: _FakeEnv())

    train_ppo._evaluate_policy(
        _FakeModel(),
        config,
        scenario_definitions=({"id": "scenario_a", "name": "scenario_a"},),
        scenario_path=tmp_path / "scenarios.yaml",
        scenario_id=None,
        hold_out_scenarios=(),
        snqi_context=default_training_snqi_context(),
        eval_step=60_000,
    )

    return messages


def test_evaluate_policy_logs_compact_phase_progress(monkeypatch, tmp_path: Path) -> None:
    """Evaluation should emit sparse phase markers so long reset-heavy runs remain readable."""
    messages = _capture_evaluate_policy_info_logs(monkeypatch, tmp_path, episodes=12)

    progress_messages = [message for message in messages if "PPO evaluation progress" in message]
    assert any("PPO evaluation phase start step=60000 episodes=12" in msg for msg in messages)
    assert any("PPO evaluation phase complete step=60000 episodes=12" in msg for msg in messages)
    assert len(progress_messages) == 2
    assert progress_messages[0].startswith("PPO evaluation progress step=60000 episode=10/12")
    assert progress_messages[-1].startswith("PPO evaluation progress step=60000 episode=12/12")


def test_evaluate_policy_logs_single_progress_marker_for_ten_episodes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Exactly 10 evaluation episodes should log only the episode-10 final marker."""
    messages = _capture_evaluate_policy_info_logs(monkeypatch, tmp_path, episodes=10)

    progress_messages = [message for message in messages if "PPO evaluation progress" in message]
    assert progress_messages == [
        "PPO evaluation progress step=60000 episode=10/10 scenario=scenario_a steps=1 success=0.000"
    ]
