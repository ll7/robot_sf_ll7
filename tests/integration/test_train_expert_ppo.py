"""Integration and helper tests for the PPO training entrypoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from robot_sf import common
from robot_sf.training.imitation_config import (
    ConvergenceCriteria,
    EvaluationSchedule,
    ExpertTrainingConfig,
)
from scripts.training.train_ppo import (
    _BestCheckpointCandidate,
    _BestCheckpointTracker,
    _build_direct_wandb_training_payload,
    _DirectWandbMetricsCallback,
    _DirectWandbTrainingMetricsCallback,
    _extract_direct_wandb_train_metrics,
    _finalize_best_checkpoint,
    _persist_best_checkpoint_if_updated,
    _reapply_resumed_ppo_hyperparams,
    _resolve_resume_checkpoint,
    _update_wandb_best_checkpoint_summary,
    _upload_wandb_best_checkpoint_artifact,
    load_expert_training_config,
    run_expert_training,
)


def test_expert_training_dry_run(tmp_path, monkeypatch):
    """TODO docstring. Document this function.

    Args:
        tmp_path: TODO docstring.
        monkeypatch: TODO docstring.
    """
    monkeypatch.setenv("ROBOT_SF_ARTIFACT_ROOT", str(tmp_path))
    config_path = Path("configs/training/ppo_imitation/expert_ppo.yaml").resolve()
    config = load_expert_training_config(config_path)

    result = run_expert_training(config, config_path=config_path, dry_run=True)

    manifest_path = result.expert_manifest_path
    assert manifest_path.exists()

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["policy_id"] == config.policy_id
    assert set(payload["metrics"].keys()) >= {
        "success_rate",
        "collision_rate",
        "path_efficiency",
        "comfort_exposure",
        "snqi",
        "eval_episode_return",
        "eval_avg_step_reward",
    }
    checkpoint = result.checkpoint_path
    assert checkpoint.exists() and checkpoint.read_text(encoding="utf-8").startswith("dry-run")

    run_manifest_path = result.training_run_manifest_path
    assert run_manifest_path.exists()
    training_payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    assert training_payload["run_type"] == common.TrainingRunType.EXPERT_TRAINING.value
    assert isinstance(training_payload.get("eval_timeline_path"), str)
    assert training_payload["eval_timeline_path"].startswith(
        "benchmarks/ppo_imitation/eval_timeline/"
    )
    assert isinstance(training_payload.get("perf_summary_path"), str)
    assert training_payload["perf_summary_path"].startswith("benchmarks/ppo_imitation/perf/")
    notes = training_payload.get("notes", [])
    assert any(str(note).startswith("snqi_formula=") for note in notes)
    assert any(str(note).startswith("snqi_weights_source=") for note in notes)
    assert any(str(note).startswith("snqi_baseline_source=") for note in notes)

    log_dir = common.get_imitation_report_dir()
    assert any(log_dir.glob("episodes/*.jsonl"))
    assert any(log_dir.glob("eval_timeline/*.json"))
    assert any(log_dir.glob("eval_timeline/*.csv"))
    assert any(log_dir.glob("perf/*.json"))


def test_load_expert_training_config_supports_resume_and_scenario_sampling(tmp_path) -> None:
    """Loader should resolve warm-start checkpoints and weighted sampler config."""
    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    resume_path = resume_dir / "model.zip"
    resume_path.write_text("checkpoint", encoding="utf-8")

    scenario_config = Path("configs/scenarios/classic_interactions_francis2023.yaml").resolve()
    config_path = tmp_path / "warmstart.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "policy_id": "ppo_warmstart_test",
                "scenario_config": str(scenario_config),
                "seeds": [123],
                "randomize_seeds": True,
                "total_timesteps": 123456,
                "resume_from": "resume/model.zip",
                "scenario_sampling": {
                    "strategy": "random",
                    "profile_strategy": "cycle",
                    "weights": {
                        "classic_doorway_low": 3.0,
                        "classic_crossing_medium": 2.0,
                    },
                    "exclude_scenarios": ["francis2023_robot_crowding"],
                },
                "convergence": {
                    "success_rate": 0.9,
                    "collision_rate": 0.05,
                    "plateau_window": 1000,
                },
                "evaluation": {
                    "frequency_episodes": 10,
                    "evaluation_episodes": 4,
                    "hold_out_scenarios": [],
                    "step_schedule": [{"every_steps": 20000}],
                },
                "env_factory_kwargs": {
                    "reward_name": "route_completion_v3",
                    "reward_kwargs": {
                        "weights": {
                            "collision": -10.0,
                            "timeout": -4.0,
                        }
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_expert_training_config(config_path)

    assert config.resume_from == resume_path.resolve()
    assert config.scenario_sampling["strategy"] == "random"
    assert config.scenario_sampling["profile_strategy"] == "cycle"
    assert config.scenario_sampling["weights"] == {
        "classic_doorway_low": 3.0,
        "classic_crossing_medium": 2.0,
    }
    assert config.scenario_sampling["exclude_scenarios"] == ["francis2023_robot_crowding"]


def test_load_expert_training_config_defaults_randomize_seeds_to_false(tmp_path) -> None:
    """Omitted randomize_seeds should keep deterministic seed handling."""
    scenario_config = Path("configs/scenarios/classic_interactions_francis2023.yaml").resolve()
    config_path = tmp_path / "deterministic_default.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "policy_id": "ppo_deterministic_default_test",
                "scenario_config": str(scenario_config),
                "seeds": [123],
                "total_timesteps": 123456,
                "convergence": {
                    "success_rate": 0.9,
                    "collision_rate": 0.05,
                    "plateau_window": 1000,
                },
                "evaluation": {
                    "frequency_episodes": 10,
                    "evaluation_episodes": 4,
                    "hold_out_scenarios": [],
                    "step_schedule": [{"every_steps": 20000}],
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_expert_training_config(config_path)
    assert config.randomize_seeds is False


def test_load_expert_training_config_defaults_best_checkpoint_metric_to_success_rate(
    tmp_path,
) -> None:
    """Configs without an explicit best-checkpoint metric should now prefer success rate."""
    scenario_config = Path("configs/scenarios/classic_interactions_francis2023.yaml").resolve()
    config_path = tmp_path / "default_best_metric.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "policy_id": "ppo_default_best_metric_test",
                "scenario_config": str(scenario_config),
                "seeds": [123],
                "total_timesteps": 123456,
                "convergence": {
                    "success_rate": 0.9,
                    "collision_rate": 0.05,
                    "plateau_window": 1000,
                },
                "evaluation": {
                    "frequency_episodes": 10,
                    "evaluation_episodes": 4,
                    "hold_out_scenarios": [],
                    "step_schedule": [{"every_steps": 20000}],
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_expert_training_config(config_path)
    assert config.best_checkpoint_metric == "success_rate"


def test_load_expert_training_config_requires_step_schedule(tmp_path) -> None:
    """Configs without step_schedule should fail instead of silently changing cadence."""
    scenario_config = Path("configs/scenarios/classic_interactions_francis2023.yaml").resolve()
    config_path = tmp_path / "missing_schedule.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "policy_id": "ppo_missing_schedule_test",
                "scenario_config": str(scenario_config),
                "seeds": [123],
                "total_timesteps": 123456,
                "convergence": {
                    "success_rate": 0.9,
                    "collision_rate": 0.05,
                    "plateau_window": 1000,
                },
                "evaluation": {
                    "frequency_episodes": 10,
                    "evaluation_episodes": 4,
                    "hold_out_scenarios": [],
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="evaluation.step_schedule is required"):
        load_expert_training_config(config_path)


def test_load_expert_training_config_allows_missing_frequency_episodes(tmp_path) -> None:
    """Configs should load when evaluation uses only the step_schedule contract."""
    scenario_config = Path("configs/scenarios/classic_interactions_francis2023.yaml").resolve()
    config_path = tmp_path / "missing_frequency.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "policy_id": "ppo_missing_frequency_test",
                "scenario_config": str(scenario_config),
                "seeds": [123],
                "total_timesteps": 123456,
                "convergence": {
                    "success_rate": 0.9,
                    "collision_rate": 0.05,
                    "plateau_window": 1000,
                },
                "evaluation": {
                    "evaluation_episodes": 4,
                    "hold_out_scenarios": [],
                    "step_schedule": [{"every_steps": 20000}],
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_expert_training_config(config_path)

    assert config.evaluation.frequency_episodes == 0
    assert config.evaluation.step_schedule == ((None, 20000),)


def test_load_expert_training_config_supports_resume_model_id(tmp_path) -> None:
    """Loader should preserve portable registry-backed resume model ids."""
    scenario_config = Path("configs/scenarios/classic_interactions_francis2023.yaml").resolve()
    config_path = tmp_path / "warmstart_model_id.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "policy_id": "ppo_warmstart_model_id_test",
                "scenario_config": str(scenario_config),
                "seeds": [123],
                "randomize_seeds": True,
                "total_timesteps": 123456,
                "resume_model_id": "ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200",
                "convergence": {
                    "success_rate": 0.9,
                    "collision_rate": 0.05,
                    "plateau_window": 1000,
                },
                "evaluation": {
                    "frequency_episodes": 10,
                    "evaluation_episodes": 4,
                    "hold_out_scenarios": [],
                    "step_schedule": [{"every_steps": 20000}],
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_expert_training_config(config_path)

    assert config.resume_from is None
    assert config.resume_model_id == "ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200"


def test_load_expert_training_config_supports_resume_source_step(tmp_path) -> None:
    """Loader should preserve pinned source checkpoint steps for reproducible warm starts."""
    scenario_config = Path("configs/scenarios/classic_interactions_francis2023.yaml").resolve()
    config_path = tmp_path / "warmstart_model_id_step.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "policy_id": "ppo_warmstart_model_id_step_test",
                "scenario_config": str(scenario_config),
                "seeds": [123],
                "randomize_seeds": True,
                "total_timesteps": 123456,
                "resume_model_id": "ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200",
                "resume_source_step": 15240000,
                "convergence": {
                    "success_rate": 0.9,
                    "collision_rate": 0.05,
                    "plateau_window": 1000,
                },
                "evaluation": {
                    "frequency_episodes": 10,
                    "evaluation_episodes": 4,
                    "hold_out_scenarios": [],
                    "step_schedule": [{"every_steps": 20000}],
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_expert_training_config(config_path)

    assert config.resume_model_id == "ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200"
    assert config.resume_source_step == 15240000


def test_reapply_resumed_ppo_hyperparams_uses_yaml_values() -> None:
    """Warm-start runs should honor config PPO overrides after checkpoint load."""
    config = ExpertTrainingConfig(
        scenario_config=Path("configs/scenarios/classic_interactions_francis2023.yaml").resolve(),
        seeds=(123,),
        total_timesteps=1000,
        policy_id="ppo_resume_test",
        convergence=ConvergenceCriteria(
            success_rate=0.9,
            collision_rate=0.05,
            plateau_window=100,
        ),
        evaluation=EvaluationSchedule(
            frequency_episodes=10,
            evaluation_episodes=4,
            hold_out_scenarios=(),
            step_schedule=((1000, 1000),),
        ),
        ppo_hyperparams={
            "learning_rate": 7.5e-5,
            "batch_size": 128,
            "n_epochs": 6,
            "ent_coef": 0.005,
            "clip_range": 0.2,
            "target_kl": 0.03,
            "gamma": 0.98,
            "gae_lambda": 0.93,
            "vf_coef": 0.7,
            "max_grad_norm": 0.4,
        },
    )
    model = SimpleNamespace(
        learning_rate=1e-4,
        lr_schedule=lambda _: 1e-4,
        batch_size=256,
        n_epochs=4,
        ent_coef=0.01,
        clip_range=lambda _: 0.1,
        target_kl=0.02,
        gamma=0.99,
        gae_lambda=0.95,
        vf_coef=0.5,
        max_grad_norm=0.5,
        n_steps=2048,
        rollout_buffer=SimpleNamespace(gamma=0.99, gae_lambda=0.95),
    )

    _reapply_resumed_ppo_hyperparams(model, config)

    assert model.learning_rate == 7.5e-5
    assert model.lr_schedule(1.0) == 7.5e-5
    assert model.batch_size == 128
    assert model.n_epochs == 6
    assert model.ent_coef == 0.005
    assert model.clip_range(1.0) == 0.2
    assert model.target_kl == 0.03
    assert model.gamma == 0.98
    assert model.gae_lambda == 0.93
    assert model.rollout_buffer.gamma == 0.98
    assert model.rollout_buffer.gae_lambda == 0.93


def test_resolve_resume_checkpoint_prefers_model_registry(monkeypatch) -> None:
    """Registry-backed resume ids should resolve to a downloadable local path."""
    expected = Path("/tmp/downloaded/model.zip")
    called: dict[str, object] = {}

    def _fake_resolve(model_id: str, *, allow_download: bool = True):
        called["model_id"] = model_id
        called["allow_download"] = allow_download
        return expected

    monkeypatch.setattr("scripts.training.train_ppo.resolve_model_path", _fake_resolve)
    config = ExpertTrainingConfig(
        scenario_config=Path("configs/scenarios/classic_interactions_francis2023.yaml").resolve(),
        seeds=(123,),
        total_timesteps=1000,
        policy_id="ppo_resume_registry_test",
        convergence=ConvergenceCriteria(
            success_rate=0.9,
            collision_rate=0.05,
            plateau_window=100,
        ),
        evaluation=EvaluationSchedule(
            frequency_episodes=10,
            evaluation_episodes=4,
            hold_out_scenarios=(),
            step_schedule=((1000, 1000),),
        ),
        resume_model_id="ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200",
    )

    resolved = _resolve_resume_checkpoint(config=config, resume_from=None)

    assert resolved == expected
    assert called == {
        "model_id": "ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200",
        "allow_download": True,
    }


def test_extract_direct_wandb_train_metrics_filters_missing_and_non_numeric() -> None:
    """Direct W&B export should keep only finite scalar train metrics."""
    model = SimpleNamespace(
        logger=SimpleNamespace(
            name_to_value={
                "train/value_loss": 1.25,
                "train/policy_gradient_loss": "0.5",
                "train/entropy_loss": float("nan"),
                "train/ignored": 99.0,
            }
        )
    )

    assert _extract_direct_wandb_train_metrics(model) == {
        "train/value_loss": 1.25,
        "train/policy_gradient_loss": 0.5,
    }


def test_build_direct_wandb_training_payload_includes_rollout_and_time(monkeypatch) -> None:
    """Payload builder should expose rollout/time metrics before train-loss extraction."""
    model = SimpleNamespace(
        ep_info_buffer=[
            {"r": 10.0, "l": 100},
            {"r": 14.0, "l": 80},
        ],
    )
    monkeypatch.setattr("scripts.training.train_ppo._wandb_training_clock", lambda: 25.0)

    payload = _build_direct_wandb_training_payload(
        model=model,
        total_timesteps=15_400_000,
        rollout_iterations=3,
        start_timesteps=15_000_000,
        run_start_time=20.0,
    )

    assert payload == {
        "time/total_timesteps": 15_400_000,
        "time/iterations": 3,
        "time/fps": 80_000.0,
        "rollout/ep_rew_mean": 12.0,
        "rollout/ep_len_mean": 90.0,
    }


def test_direct_wandb_training_callback_logs_after_train(monkeypatch) -> None:
    """Callback should emit direct W&B metrics only after train-loss values are available."""
    logged: list[tuple[dict[str, float | int], int]] = []

    class _WandbRunStub:
        def log(self, payload: dict[str, float | int], *, step: int) -> None:
            logged.append((payload, step))

    callback = _DirectWandbTrainingMetricsCallback(
        wandb_run=_WandbRunStub(),
        start_timesteps=15_000_000,
        run_start_time=40.0,
    )
    callback.model = SimpleNamespace(
        num_timesteps=15_250_000,
        ep_info_buffer=[{"r": 8.0, "l": 60}],
        logger=SimpleNamespace(
            name_to_value={
                "train/value_loss": 0.75,
                "train/policy_gradient_loss": -0.1,
                "train/entropy_loss": -0.02,
            }
        ),
    )
    monkeypatch.setattr("scripts.training.train_ppo._wandb_training_clock", lambda: 42.0)

    callback.on_rollout_end()
    assert logged == []

    callback.log_after_train()

    assert logged == [
        (
            {
                "time/total_timesteps": 15_250_000,
                "time/iterations": 1,
                "time/fps": 125_000.0,
                "rollout/ep_rew_mean": 8.0,
                "rollout/ep_len_mean": 60.0,
                "train/value_loss": 0.75,
                "train/policy_gradient_loss": -0.1,
                "train/entropy_loss": -0.02,
            },
            15_250_000,
        )
    ]


def test_direct_wandb_metrics_callback_logs_core_training_series() -> None:
    """Direct W&B callback should mirror key SB3 metrics without waiting for eval checkpoints."""

    class _Run:
        def __init__(self) -> None:
            self.payloads: list[tuple[dict[str, float | int], int]] = []

        def log(self, payload, step):
            self.payloads.append((dict(payload), int(step)))

    run = _Run()
    callback = _DirectWandbMetricsCallback(run, log_every_steps=100)
    callback.model = SimpleNamespace(
        logger=SimpleNamespace(
            name_to_value={
                "rollout/ep_rew_mean": 12.5,
                "rollout/ep_len_mean": 90.0,
                "train/value_loss": 0.2,
                "time/fps": 430,
            }
        )
    )
    callback.num_timesteps = 150

    assert callback._on_step() is True
    assert len(run.payloads) == 1
    payload, step = run.payloads[0]
    assert step == 150
    assert payload["time/total_timesteps"] == 150
    assert payload["rollout/ep_rew_mean"] == 12.5
    assert payload["train/value_loss"] == 0.2


def test_finalize_best_checkpoint_writes_summary_sidecar(tmp_path) -> None:
    """Best-checkpoint finalization should persist a machine-readable summary file."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    source = checkpoint_dir / "ppo_test_step17000000.zip"
    source.write_text("checkpoint", encoding="utf-8")
    tracker = _BestCheckpointTracker(
        metric_name="success_rate",
        higher_is_better=True,
        convergence=ConvergenceCriteria(0.9, 0.1, 100),
    )
    tracker.best_overall = _BestCheckpointCandidate(
        eval_step=17_000_000,
        score=0.9667,
        metrics={"success_rate": 0.9667, "collision_rate": 0.0333, "snqi": 0.39},
        meets_convergence=True,
    )
    config = ExpertTrainingConfig.from_raw(
        scenario_config=Path("configs/scenarios/classic_interactions_francis2023.yaml").resolve(),
        seeds=(1,),
        total_timesteps=30_000_000,
        policy_id="ppo_test",
        convergence=ConvergenceCriteria(0.9, 0.1, 100),
        evaluation=EvaluationSchedule(
            frequency_episodes=0,
            evaluation_episodes=4,
            step_schedule=((None, 1_000_000),),
        ),
    )

    summary = _finalize_best_checkpoint(tracker, config=config, checkpoint_dir=checkpoint_dir)

    assert summary is not None
    assert summary.checkpoint_path.exists()
    assert summary.report_path is not None
    payload = json.loads(summary.report_path.read_text(encoding="utf-8"))
    assert payload["eval_step"] == 17_000_000
    assert payload["metric"] == "success_rate"
    assert payload["metrics"]["success_rate"] == pytest.approx(0.9667)


def test_update_wandb_best_checkpoint_summary_mirrors_metrics() -> None:
    """W&B summary should expose the selected best-checkpoint metadata."""
    run = SimpleNamespace(summary={})
    config = ExpertTrainingConfig.from_raw(
        scenario_config=Path("configs/scenarios/classic_interactions_francis2023.yaml").resolve(),
        seeds=(1,),
        total_timesteps=30_000_000,
        policy_id="ppo_test",
        convergence=ConvergenceCriteria(0.9, 0.1, 100),
        evaluation=EvaluationSchedule(
            frequency_episodes=0,
            evaluation_episodes=4,
            step_schedule=((None, 1_000_000),),
        ),
    )
    best = SimpleNamespace(
        metric="success_rate",
        value=0.9667,
        eval_step=17_000_000,
        checkpoint_path=Path("/tmp/model_best.zip"),
        report_path=Path("/tmp/model_best.summary.json"),
        meets_convergence=True,
        metrics={"success_rate": 0.9667, "collision_rate": 0.0333, "snqi": 0.39},
    )

    _update_wandb_best_checkpoint_summary(run, config=config, best_summary=best)

    assert run.summary["best/checkpoint_metric"] == "success_rate"
    assert run.summary["best/eval_step"] == 17_000_000
    assert run.summary["best/success_rate"] == pytest.approx(0.9667)
    assert run.summary["best/collision_rate"] == pytest.approx(0.0333)


def test_upload_wandb_best_checkpoint_artifact_logs_model_with_aliases(
    tmp_path, monkeypatch
) -> None:
    """Best checkpoint upload should publish a W&B model artifact with stable aliases."""

    class _Artifact:
        def __init__(self, name, artifact_type=None, metadata=None, **kwargs):
            if artifact_type is None:
                artifact_type = kwargs.get("type")
            self.name = name
            self.type = artifact_type
            self.metadata = metadata
            self.description = None
            self.files: list[tuple[str, str | None]] = []

        def add_file(self, path, name=None):
            self.files.append((str(path), name))

    class _Run:
        def __init__(self) -> None:
            self.logged: list[tuple[object, list[str] | None]] = []

        def log_artifact(self, artifact, aliases=None):
            self.logged.append((artifact, aliases))

    model_path = tmp_path / "model_best.zip"
    report_path = tmp_path / "model_best.summary.json"
    model_path.write_text("checkpoint", encoding="utf-8")
    report_path.write_text("{}", encoding="utf-8")
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Artifact=_Artifact))
    run = _Run()
    config = ExpertTrainingConfig.from_raw(
        scenario_config=Path("configs/scenarios/classic_interactions_francis2023.yaml").resolve(),
        seeds=(1,),
        total_timesteps=30_000_000,
        policy_id="ppo_test",
        convergence=ConvergenceCriteria(0.9, 0.1, 100),
        evaluation=EvaluationSchedule(
            frequency_episodes=0,
            evaluation_episodes=4,
            step_schedule=((None, 1_000_000),),
        ),
    )
    best = SimpleNamespace(
        metric="success_rate",
        value=0.9667,
        eval_step=17_000_000,
        checkpoint_path=model_path,
        report_path=report_path,
        meets_convergence=True,
        metrics={"success_rate": 0.9667, "collision_rate": 0.0333},
    )

    _upload_wandb_best_checkpoint_artifact(run, config=config, best_summary=best)

    assert len(run.logged) == 1
    artifact, aliases = run.logged[0]
    assert artifact.name == "ppo_test-best-success"
    assert aliases == ["best-success", "step-17000000"]
    assert (f"{model_path}", "model.zip") in artifact.files
    assert (f"{report_path}", "best_checkpoint_summary.json") in artifact.files


def test_persist_best_checkpoint_if_updated_uploads_immediately(tmp_path, monkeypatch) -> None:
    """Best checkpoints should be persisted as soon as a new best eval appears."""

    class _Artifact:
        def __init__(self, name, artifact_type=None, metadata=None, **kwargs):
            if artifact_type is None:
                artifact_type = kwargs.get("type")
            self.name = name
            self.type = artifact_type
            self.metadata = metadata
            self.description = None
            self.files: list[tuple[str, str | None]] = []

        def add_file(self, path, name=None):
            self.files.append((str(path), name))

    class _Run:
        def __init__(self) -> None:
            self.summary: dict[str, object] = {}
            self.logged: list[tuple[object, list[str] | None]] = []

        def log_artifact(self, artifact, aliases=None):
            self.logged.append((artifact, aliases))

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    source = checkpoint_dir / "ppo_test_step17000000.zip"
    source.write_text("checkpoint", encoding="utf-8")
    tracker = _BestCheckpointTracker(
        metric_name="success_rate",
        higher_is_better=True,
        convergence=ConvergenceCriteria(0.9, 0.1, 100),
    )
    tracker.best_overall = _BestCheckpointCandidate(
        eval_step=17_000_000,
        score=0.9667,
        metrics={"success_rate": 0.9667, "collision_rate": 0.0333, "snqi": 0.39},
        meets_convergence=True,
    )
    config = ExpertTrainingConfig.from_raw(
        scenario_config=Path("configs/scenarios/classic_interactions_francis2023.yaml").resolve(),
        seeds=(1,),
        total_timesteps=30_000_000,
        policy_id="ppo_test",
        convergence=ConvergenceCriteria(0.9, 0.1, 100),
        evaluation=EvaluationSchedule(
            frequency_episodes=0,
            evaluation_episodes=4,
            step_schedule=((None, 1_000_000),),
        ),
    )
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(Artifact=_Artifact))
    run = _Run()

    best, eval_step = _persist_best_checkpoint_if_updated(
        tracker,
        config=config,
        checkpoint_dir=checkpoint_dir,
        wandb_run=run,
        last_persisted_eval_step=None,
    )

    assert best is not None
    assert eval_step == 17_000_000
    assert best.checkpoint_path.exists()
    assert run.summary["best/eval_step"] == 17_000_000
    assert len(run.logged) == 1
    artifact, aliases = run.logged[0]
    assert artifact.name == "ppo_test-best-success"
    assert aliases == ["best-success", "step-17000000"]

    second_best, second_eval_step = _persist_best_checkpoint_if_updated(
        tracker,
        config=config,
        checkpoint_dir=checkpoint_dir,
        wandb_run=run,
        last_persisted_eval_step=eval_step,
    )

    assert second_best is None
    assert second_eval_step == 17_000_000
    assert len(run.logged) == 1
