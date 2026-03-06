"""TODO docstring. Document this module."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from robot_sf import common
from robot_sf.training.imitation_config import (
    ConvergenceCriteria,
    EvaluationSchedule,
    ExpertTrainingConfig,
)
from scripts.training.train_ppo import (
    _reapply_resumed_ppo_hyperparams,
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
