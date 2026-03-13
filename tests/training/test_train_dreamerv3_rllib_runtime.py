"""Runtime-path tests for DreamerV3 RLlib launcher orchestration."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.training.train_dreamerv3_rllib as dreamer


def _write_yaml(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


class _FakeRay:
    """Minimal ray facade used for run_training unit tests."""

    def __init__(self) -> None:
        self._initialized = False
        self.shutdown_called = False
        self.init_kwargs: dict[str, object] | None = None

    def init(self, **kwargs) -> None:
        self._initialized = True
        self.init_kwargs = dict(kwargs)

    def is_initialized(self) -> bool:
        return self._initialized

    def shutdown(self) -> None:
        self.shutdown_called = True
        self._initialized = False


class _FakeAlgo:
    """Simple algorithm stub exposing train/save/stop lifecycle hooks."""

    def __init__(self, *, fail_on_train: bool = False) -> None:
        self.fail_on_train = fail_on_train
        self.train_calls = 0
        self.stop_called = False

    def train(self) -> dict[str, object]:
        if self.fail_on_train:
            raise RuntimeError("train failed")
        self.train_calls += 1
        return {
            "episode_return_mean": float(self.train_calls),
            "episode_len_mean": 100.0 + self.train_calls,
            "num_env_steps_sampled_lifetime": self.train_calls * 10,
            "env_runner_results": {
                "sample_throughput": 50.0 + self.train_calls,
            },
            "timers": {
                "train_iteration_time_ms": 20.0 + self.train_calls,
            },
            "learners": {
                "default_policy": {
                    "world_model_loss": 1.5 + self.train_calls,
                }
            },
        }

    def compute_single_action(self, _obs, *, explore: bool = False):
        assert explore is False
        return 0.0

    def save_to_path(self, checkpoint_dir: str) -> str:
        path = Path(checkpoint_dir) / f"checkpoint_{self.train_calls}"
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def stop(self) -> None:
        self.stop_called = True


class _FakeAlgoNoCompletedEpisodes(_FakeAlgo):
    """Algorithm stub that mimics RLlib returning NaN means for empty-episode iterations."""

    def train(self) -> dict[str, object]:
        self.train_calls += 1
        return {
            "env_runner_results": {
                "episode_return_mean": float("nan"),
                "episode_len_mean": float("nan"),
                "num_episodes": 0,
                "num_env_steps_sampled_lifetime": self.train_calls * 10,
            }
        }


class _FrozenDateTime:
    """Deterministic datetime replacement for reproducible run directories."""

    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return datetime(2026, 2, 11, 12, 0, 0, tzinfo=UTC)
        return datetime(2026, 2, 11, 12, 0, 0, tzinfo=UTC)


class _EvalEnv:
    """Minimal env stub used to exercise Dreamer periodic evaluation integration."""

    def __init__(self) -> None:
        self.state = SimpleNamespace(max_sim_steps=5)
        self._steps = 0

    def reset(self):
        self._steps = 0
        return {"obs": 0.0}, {}

    def step(self, _action):
        self._steps += 1
        terminated = self._steps >= 1
        info = {
            "meta": {
                "is_route_complete": True,
                "is_pedestrian_collision": False,
                "is_robot_collision": False,
                "is_obstacle_collision": False,
                "is_timesteps_exceeded": False,
                "step_of_episode": self._steps,
                "max_sim_steps": 5,
            }
        }
        return {"obs": 0.0}, 1.0, terminated, False, info

    def close(self) -> None:
        return None


def _assert_run_training_artifacts(run_config: dreamer.DreamerRunConfig) -> None:
    """Validate the core summary/result artifacts emitted by the happy-path run."""
    run_dir = run_config.experiment.output_root / "smoke_20260211T120000Z"
    summary_path = run_dir / "run_summary.json"
    result_path = run_dir / "result.jsonl"
    checkpoint_dir = run_dir / "checkpoints"
    run_meta_path = run_dir / "run_meta.json"
    resolved_config_path = run_dir / "resolved_config.json"
    assert summary_path.exists()
    assert result_path.exists()
    assert checkpoint_dir.exists()
    assert run_meta_path.exists()
    assert resolved_config_path.exists()

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["run_id"] == "smoke"
    assert summary["train_iterations"] == 2
    assert summary["run_meta_path"] == str(run_meta_path)
    assert summary["resolved_config_path"] == str(resolved_config_path)
    assert summary["seed_report"]["seed"] == 11
    assert len(summary["history"]) == 2
    assert summary["history"][0]["episode_len_mean"] == 101.0
    assert summary["history"][1]["episode_len_mean"] == 102.0
    assert summary["history"][0]["observability"]["env_runner_results/sample_throughput"] == 51.0
    assert summary["history"][0]["observability"]["timers/train_iteration_time_ms"] == 21.0
    assert summary["history"][0]["observability"]["learners/default_policy/world_model_loss"] == 2.5
    assert summary["scenario_matrix_path"] is None
    assert summary["randomize_seeds"] is False
    assert summary["runtime_diagnostics"]["cfg_num_gpus"] == 1

    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    assert run_meta["status"] == "succeeded"
    assert run_meta["seed_report"]["seed"] == 11
    assert run_meta["artifacts"]["summary_path"] == str(summary_path)
    assert run_meta["runtime_diagnostics"]["cfg_num_gpus"] == 1

    resolved_config = json.loads(resolved_config_path.read_text(encoding="utf-8"))
    assert resolved_config["experiment"]["run_id"] == "smoke"

    result_lines = result_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(result_lines) == 2
    assert json.loads(result_lines[0])["iteration"] == 1
    assert json.loads(result_lines[0])["episode_len_mean"] == 101.0
    assert (
        json.loads(result_lines[0])["observability"]["env_runner_results/sample_throughput"] == 51.0
    )
    assert json.loads(result_lines[1])["iteration"] == 2
    assert json.loads(result_lines[1])["episode_len_mean"] == 102.0


def test_apply_config_method_warns_when_payload_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    """Missing AlgorithmConfig hooks should emit warnings instead of silent drops."""
    warnings: list[str] = []
    monkeypatch.setattr(
        dreamer.logger,
        "warning",
        lambda msg, *args: warnings.append(str(msg).format(*args)),
    )

    cfg = object()
    result = dreamer._apply_config_method(cfg, "resources", {"num_gpus": 1})

    assert result is cfg
    assert warnings
    assert "algorithm.resources" in warnings[0]


def test_apply_env_runner_settings_warns_when_payload_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing env_runner/rollout hooks should emit warnings."""
    warnings: list[str] = []
    monkeypatch.setattr(
        dreamer.logger,
        "warning",
        lambda msg, *args: warnings.append(str(msg).format(*args)),
    )

    cfg = object()
    result = dreamer._apply_env_runner_settings(cfg, {"num_env_runners": 4})

    assert result is cfg
    assert warnings
    assert "algorithm.env_runners payload" in warnings[0]


def test_run_training_writes_summary_and_result_jsonl(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """run_training should emit summary/result artifacts and clean up ray/algo lifecycle."""
    config_path = _write_yaml(
        tmp_path / "dreamer.yaml",
        f"""
experiment:
  run_id: smoke
  output_root: {tmp_path.as_posix()}/output
  train_iterations: 2
  checkpoint_every: 1
  seed: 11
  log_level: WARNING
ray:
  num_cpus: auto
  num_gpus: auto
algorithm:
  framework: torch
  env_runners:
    num_env_runners: auto
  resources:
    num_gpus: auto
""",
    )
    run_config = dreamer.load_run_config(config_path)
    fake_ray = _FakeRay()
    fake_algo = _FakeAlgo()
    detect_calls = {"count": 0}
    register_calls: list[str] = []

    monkeypatch.setattr(dreamer, "datetime", _FrozenDateTime)
    monkeypatch.setattr(
        dreamer,
        "_import_rllib",
        lambda: (
            fake_ray,
            object,
            lambda env_name, _creator: register_calls.append(env_name),
        ),
    )
    monkeypatch.setattr(
        dreamer,
        "detect_hardware_capacity",
        lambda **_kwargs: (
            detect_calls.__setitem__("count", detect_calls["count"] + 1)
            or SimpleNamespace(
                usable_cpus=8,
                visible_gpus=1,
                logical_cpus=8,
                allocated_cpus=8,
                allocated_gpus=1,
            )
        ),
    )
    monkeypatch.setattr(dreamer, "_init_wandb_tracking", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        dreamer,
        "_build_algorithm_config",
        lambda *_args, **_kwargs: SimpleNamespace(
            num_gpus=1,
            num_learners=0,
            num_gpus_per_learner=0,
            num_gpus_per_env_runner=0,
            num_env_runners=8,
            local_gpu_idx=0,
        ),
    )
    monkeypatch.setattr(dreamer, "_build_algorithm_instance", lambda _cfg: fake_algo)

    exit_code = dreamer.run_training(run_config)

    assert exit_code == 0
    assert detect_calls["count"] == 1  # one shared probe for ray+algorithm auto settings
    assert register_calls == ["robot_sf_dreamerv3_smoke"]
    assert fake_algo.stop_called is True
    assert fake_ray.shutdown_called is True
    assert fake_ray.init_kwargs is not None
    assert fake_ray.init_kwargs["num_cpus"] == 8
    assert fake_ray.init_kwargs["num_gpus"] == 1
    _assert_run_training_artifacts(run_config)


def test_build_runtime_diagnostics_reports_resolved_gpu_placement(tmp_path: Path) -> None:
    """Dry-run diagnostics should expose the resolved learner placement fields."""
    config_path = _write_yaml(
        tmp_path / "dreamer_diag.yaml",
        """
experiment:
    run_id: smoke_diag
ray:
    num_gpus: 1
algorithm:
    framework: torch
    env_runners:
        num_env_runners: 12
    resources:
        num_gpus: 1
    learners:
        num_learners: 0
        num_gpus_per_learner: 1
""",
    )
    run_config = dreamer.load_run_config(config_path)
    algo_config = SimpleNamespace(
        num_gpus=1,
        num_learners=0,
        num_gpus_per_learner=1,
        num_gpus_per_env_runner=0,
        num_env_runners=12,
        local_gpu_idx=0,
    )

    diagnostics = dreamer._build_runtime_diagnostics(
        run_config,
        algo_config=algo_config,
        capacity=None,
    )

    assert diagnostics["algo_learners"]["num_gpus_per_learner"] == 1
    assert diagnostics["cfg_num_gpus_per_learner"] == 1
    assert diagnostics["cfg_num_env_runners"] == 12


def test_run_training_cleans_up_ray_and_algo_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Lifecycle cleanup must happen even when algorithm training fails."""
    config_path = _write_yaml(
        tmp_path / "dreamer_fail.yaml",
        f"""
experiment:
  run_id: smoke_fail
  output_root: {tmp_path.as_posix()}/output
  train_iterations: 1
  checkpoint_every: 1
  seed: 11
  log_level: WARNING
algorithm:
  framework: torch
""",
    )
    run_config = dreamer.load_run_config(config_path)
    fake_ray = _FakeRay()
    fake_algo = _FakeAlgo(fail_on_train=True)

    monkeypatch.setattr(dreamer, "datetime", _FrozenDateTime)
    monkeypatch.setattr(
        dreamer,
        "_import_rllib",
        lambda: (fake_ray, object, lambda _env_name, _creator: None),
    )
    monkeypatch.setattr(dreamer, "_init_wandb_tracking", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dreamer, "_build_algorithm_config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(dreamer, "_build_algorithm_instance", lambda _cfg: fake_algo)

    with pytest.raises(RuntimeError, match="train failed"):
        dreamer.run_training(run_config)

    assert fake_algo.stop_called is True
    assert fake_ray.shutdown_called is True
    run_dir = run_config.experiment.output_root / "smoke_fail_20260211T120000Z"
    run_meta_path = run_dir / "run_meta.json"
    assert run_meta_path.exists()
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    assert run_meta["status"] == "failed"
    assert run_meta["error"]["type"] == "RuntimeError"


def test_run_training_records_null_reward_when_no_episodes_complete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Empty-episode RLlib iterations should serialize reward_mean as null instead of NaN."""
    config_path = _write_yaml(
        tmp_path / "dreamer_empty_eps.yaml",
        f"""
experiment:
    run_id: smoke_empty_eps
    output_root: {tmp_path.as_posix()}/output
    train_iterations: 1
    checkpoint_every: 1
    seed: 11
    log_level: WARNING
algorithm:
    framework: torch
""",
    )
    run_config = dreamer.load_run_config(config_path)
    fake_ray = _FakeRay()
    fake_algo = _FakeAlgoNoCompletedEpisodes()

    monkeypatch.setattr(dreamer, "datetime", _FrozenDateTime)
    monkeypatch.setattr(
        dreamer,
        "_import_rllib",
        lambda: (fake_ray, object, lambda _env_name, _creator: None),
    )
    monkeypatch.setattr(dreamer, "_init_wandb_tracking", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dreamer, "_build_algorithm_config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(dreamer, "_build_algorithm_instance", lambda _cfg: fake_algo)

    exit_code = dreamer.run_training(run_config)

    assert exit_code == 0
    run_dir = run_config.experiment.output_root / "smoke_empty_eps_20260211T120000Z"
    summary = json.loads((run_dir / "run_summary.json").read_text(encoding="utf-8"))
    result_lines = (run_dir / "result.jsonl").read_text(encoding="utf-8").strip().splitlines()
    result_record = json.loads(result_lines[0])

    assert summary["history"][0]["episodes_completed"] == 0
    assert summary["history"][0]["episode_len_mean"] is None
    assert summary["history"][0]["reward_mean"] is None
    assert result_record["episodes_completed"] == 0
    assert result_record["episode_len_mean"] is None
    assert result_record["reward_mean"] is None


def test_make_env_creator_preserves_grid_observation_with_scenario_matrix(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Scenario-matrix Dreamer env creation must not disable benchmark grid observations."""
    scenarios = tmp_path / "scenarios.yaml"
    scenarios.write_text("scenarios: []\n", encoding="utf-8")
    config_path = _write_yaml(
        tmp_path / "dreamer_grid_matrix.yaml",
        f"""
experiment:
    run_id: smoke
env:
    flatten_observation: true
    flatten_keys: null
    normalize_actions: true
    scenario_matrix:
        path: {scenarios.as_posix()}
        strategy: cycle
    config:
        observation_mode: socnav_struct
        use_image_obs: false
        use_occupancy_grid: true
        include_grid_in_observation: true
algorithm:
    framework: torch
""",
    )
    run_config = dreamer.load_run_config(config_path)
    captured: dict[str, object] = {}

    monkeypatch.setattr(dreamer, "load_scenarios", lambda *args, **kwargs: [{"name": "demo"}])
    monkeypatch.setattr(
        dreamer,
        "build_robot_config_from_scenario",
        lambda *_args, **_kwargs: dreamer.RobotSimulationConfig(),
    )

    class _SwitchingEnvStub:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.observation_space = "obs"
            self.action_space = "act"
            self.metadata = {}
            self.render_mode = None

        def reset(self, *args, **kwargs):  # pragma: no cover - passthrough stub
            return {}, {}

    monkeypatch.setattr(dreamer, "ScenarioSwitchingEnv", _SwitchingEnvStub)
    monkeypatch.setattr(
        dreamer,
        "wrap_for_dreamerv3",
        lambda env, **kwargs: {"env": env, "wrap_kwargs": kwargs},
    )

    creator = dreamer._make_env_creator(run_config)
    creator({"worker_index": 1})
    scenario_config = captured["config_builder"]({"name": "demo"})

    assert scenario_config.include_grid_in_observation is True
    assert scenario_config.use_occupancy_grid is True


def test_run_training_writes_periodic_eval_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Periodic Dreamer evaluation should emit summary and timeline artifacts."""
    config_path = _write_yaml(
        tmp_path / "dreamer_eval.yaml",
        f"""
experiment:
    run_id: smoke_eval
    output_root: {tmp_path.as_posix()}/output
    train_iterations: 1
    checkpoint_every: 1
    seed: 11
    log_level: WARNING
evaluation:
    enabled: true
    every_iterations: 1
    evaluation_episodes: 2
algorithm:
    framework: torch
""",
    )
    run_config = dreamer.load_run_config(config_path)
    fake_ray = _FakeRay()
    fake_algo = _FakeAlgo()

    monkeypatch.setattr(dreamer, "datetime", _FrozenDateTime)
    monkeypatch.setattr(
        dreamer,
        "_import_rllib",
        lambda: (fake_ray, object, lambda _env_name, _creator: None),
    )
    monkeypatch.setattr(dreamer, "_init_wandb_tracking", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dreamer, "_build_algorithm_config", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(dreamer, "_build_algorithm_instance", lambda _cfg: fake_algo)
    monkeypatch.setattr(
        dreamer,
        "_create_dreamer_eval_env_factory",
        lambda _cfg: lambda _episode_idx, _seed: (_EvalEnv(), f"scenario-{_episode_idx}"),
    )

    exit_code = dreamer.run_training(run_config)

    assert exit_code == 0
    run_dir = run_config.experiment.output_root / "smoke_eval_20260211T120000Z"
    eval_dir = run_dir / "evaluation"
    summary = json.loads((eval_dir / "summary_iter_00001.json").read_text(encoding="utf-8"))
    timeline_lines = (eval_dir / "eval_timeline.jsonl").read_text(encoding="utf-8").splitlines()
    assert summary["metric_means"]["success_rate"] == pytest.approx(1.0)
    assert summary["termination_reason_counts"]["success"] == 2
    assert len(timeline_lines) == 1


def test_make_env_creator_uses_scenario_switching_env_when_matrix_configured(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Scenario-matrix Dreamer configs should build via ScenarioSwitchingEnv."""
    scenarios = tmp_path / "scenarios.yaml"
    scenarios.write_text("scenarios: []\n", encoding="utf-8")
    config_path = _write_yaml(
        tmp_path / "dreamer_matrix.yaml",
        f"""
experiment:
  run_id: smoke
  output_root: {tmp_path.as_posix()}/output
  train_iterations: 1
  checkpoint_every: 1
  seed: 11
  log_level: WARNING
env:
  flatten_observation: true
  flatten_keys: [drive_state, rays]
  normalize_actions: true
  scenario_matrix:
    path: {scenarios.as_posix()}
    strategy: cycle
algorithm:
  framework: torch
""",
    )
    run_config = dreamer.load_run_config(config_path)
    captured: dict[str, object] = {}

    monkeypatch.setattr(dreamer, "load_scenarios", lambda *args, **kwargs: [{"name": "demo"}])

    class _SwitchingEnvStub:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)
            self.observation_space = "obs"
            self.action_space = "act"
            self.metadata = {}
            self.render_mode = None

        def reset(self, *args, **kwargs):  # pragma: no cover - passthrough stub
            return {}, {}

    monkeypatch.setattr(dreamer, "ScenarioSwitchingEnv", _SwitchingEnvStub)
    monkeypatch.setattr(
        dreamer,
        "wrap_for_dreamerv3",
        lambda env, **kwargs: {"env": env, "wrap_kwargs": kwargs},
    )

    creator = dreamer._make_env_creator(run_config)
    wrapped = creator({"worker_index": 2})

    assert isinstance(captured["scenario_sampler"], dreamer.ScenarioSampler)
    assert captured["scenario_path"] == scenarios.resolve()
    assert captured["seed"] == 13
    assert captured["suite_name"] == "dreamerv3_rllib"
    assert wrapped["wrap_kwargs"]["flatten_keys"] == ("drive_state", "rays")
