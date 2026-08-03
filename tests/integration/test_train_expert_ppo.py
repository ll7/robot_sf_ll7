"""Integration and helper tests for the PPO training entrypoint."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from robot_sf import common
from robot_sf.gym_env.environment_factory import make_robot_env
from robot_sf.training.imitation_config import (
    ConvergenceCriteria,
    EvaluationSchedule,
    ExpertTrainingConfig,
)
from robot_sf.training.scenario_loader import build_robot_config_from_scenario, load_scenarios
from scripts.training.train_ppo import (
    _apply_env_overrides,
    _BestCheckpointCandidate,
    _BestCheckpointTracker,
    _build_direct_wandb_training_payload,
    _describe_num_envs_resolution,
    _deterministic_eval_seed_for_episode,
    _DirectWandbMetricsCallback,
    _DirectWandbTrainingMetricsCallback,
    _extract_direct_wandb_train_metrics,
    _finalize_best_checkpoint,
    _load_expert_training_config_mapping,
    _parse_num_envs,
    _persist_best_checkpoint_if_updated,
    _reapply_resumed_ppo_hyperparams,
    _resolve_num_envs,
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
    assert isinstance(training_payload.get("eval_per_scenario_path"), str)
    assert training_payload["eval_per_scenario_path"].startswith(
        "benchmarks/ppo_imitation/eval_by_scenario/"
    )
    assert isinstance(training_payload.get("evaluation_scenario_config"), str)
    assert training_payload["evaluation_scenario_config"].startswith("configs/")
    notes = training_payload.get("notes", [])
    assert any(str(note).startswith("snqi_formula=") for note in notes)
    assert any(str(note).startswith("snqi_weights_source=") for note in notes)
    assert any(str(note).startswith("snqi_baseline_source=") for note in notes)
    # Issue #4967: artifacts must be self-describing with their resolved reward profile.
    assert any(str(note).startswith("reward_profile=route_completion_v2") for note in notes)
    assert any(str(note).startswith("reward_weights=") for note in notes)

    log_dir = common.get_imitation_report_dir()
    assert any(log_dir.glob("episodes/*.jsonl"))
    assert any(log_dir.glob("eval_timeline/*.json"))
    assert any(log_dir.glob("eval_timeline/*.csv"))
    assert any(log_dir.glob("eval_by_scenario/*.json"))
    assert any(log_dir.glob("eval_by_scenario/*.csv"))
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
                        "classic_cross_trap_medium": 2.0,
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
        "classic_cross_trap_medium": 2.0,
    }
    assert config.scenario_sampling["exclude_scenarios"] == ["francis2023_robot_crowding"]


def test_load_expert_training_config_preserves_reward_curriculum(tmp_path) -> None:
    """Loader should preserve staged reward curriculum config for factory wiring."""
    scenario_config = Path("configs/scenarios/classic_interactions_francis2023.yaml").resolve()
    config_path = tmp_path / "reward_curriculum.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "policy_id": "ppo_reward_curriculum_test",
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
                "env_factory_kwargs": {
                    "reward_name": "route_completion_v3",
                    "reward_curriculum": {
                        "stages": [
                            {
                                "until_episodes": 4,
                                "reward_kwargs": {
                                    "weights": {"terminal_bonus": 1.0},
                                },
                            },
                            {
                                "reward_kwargs": {
                                    "weights": {"terminal_bonus": 5.0},
                                },
                            },
                        ]
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_expert_training_config(config_path)

    curriculum = config.env_factory_kwargs["reward_curriculum"]
    assert curriculum["stages"][0]["until_episodes"] == 4
    assert curriculum["stages"][1]["reward_kwargs"]["weights"]["terminal_bonus"] == 5.0


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
    assert config.evaluation.randomize_seeds is False


def test_load_expert_training_config_allows_eval_seed_randomness_override(tmp_path) -> None:
    """Evaluation seed handling should be independently configurable."""
    scenario_config = Path("configs/scenarios/classic_interactions_francis2023.yaml").resolve()
    config_path = tmp_path / "eval_seed_override.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "policy_id": "ppo_eval_seed_override_test",
                "scenario_config": str(scenario_config),
                "seeds": [123, 231],
                "randomize_seeds": True,
                "total_timesteps": 123456,
                "convergence": {
                    "success_rate": 0.9,
                    "collision_rate": 0.05,
                    "plateau_window": 1000,
                },
                "evaluation": {
                    "frequency_episodes": 10,
                    "evaluation_episodes": 94,
                    "hold_out_scenarios": [],
                    "randomize_seeds": False,
                    "step_schedule": [{"every_steps": 20000}],
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_expert_training_config(config_path)
    assert config.randomize_seeds is True
    assert config.evaluation.randomize_seeds is False


def test_load_expert_training_config_supports_eval_scenario_config(tmp_path) -> None:
    """Evaluation surface overrides should resolve independently from training scenarios."""
    scenario_config = Path("configs/scenarios/classic_interactions_francis2023.yaml").resolve()
    eval_scenario_config = Path("configs/scenarios/sets/ppo_full_maintained_eval_v1.yaml").resolve()
    config_path = tmp_path / "eval_surface_override.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "policy_id": "ppo_eval_surface_override_test",
                "scenario_config": str(scenario_config),
                "seeds": [123, 231],
                "randomize_seeds": True,
                "total_timesteps": 123456,
                "convergence": {
                    "success_rate": 0.9,
                    "collision_rate": 0.05,
                    "plateau_window": 1000,
                },
                "evaluation": {
                    "frequency_episodes": 10,
                    "evaluation_episodes": 350,
                    "hold_out_scenarios": [],
                    "randomize_seeds": False,
                    "scenario_config": str(eval_scenario_config),
                    "step_schedule": [{"every_steps": 20000}],
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_expert_training_config(config_path)
    assert config.scenario_config == scenario_config
    assert config.evaluation.scenario_config == eval_scenario_config


def test_load_expert_training_config_inherits_eval_seed_randomness_by_default(
    tmp_path,
) -> None:
    """Evaluation randomness should inherit the legacy top-level flag when omitted."""
    scenario_config = Path("configs/scenarios/classic_interactions_francis2023.yaml").resolve()
    config_path = tmp_path / "eval_seed_inherit.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "policy_id": "ppo_eval_seed_inherit_test",
                "scenario_config": str(scenario_config),
                "seeds": [123, 231],
                "randomize_seeds": True,
                "total_timesteps": 123456,
                "convergence": {
                    "success_rate": 0.9,
                    "collision_rate": 0.05,
                    "plateau_window": 1000,
                },
                "evaluation": {
                    "frequency_episodes": 10,
                    "evaluation_episodes": 94,
                    "hold_out_scenarios": [],
                    "step_schedule": [{"every_steps": 20000}],
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_expert_training_config(config_path)
    assert config.randomize_seeds is True
    assert config.evaluation.randomize_seeds is True


@pytest.mark.parametrize(
    ("episode_idx", "expected_seed"),
    [
        (0, 11),
        (70, 11),
        (71, 17),
        (141, 17),
        (142, 29),
        (212, 29),
        (213, 11),
    ],
)
def test_deterministic_eval_seed_schedule_advances_per_full_scenario_cycle(
    episode_idx: int, expected_seed: int
) -> None:
    """Deterministic eval should cover all scenarios before advancing to the next seed block."""
    config = ExpertTrainingConfig(
        scenario_config=Path("configs/scenarios/classic_interactions_francis2023.yaml").resolve(),
        seeds=(11, 17, 29),
        total_timesteps=1000,
        policy_id="ppo_eval_seed_schedule_test",
        convergence=ConvergenceCriteria(
            success_rate=0.9,
            collision_rate=0.05,
            plateau_window=100,
        ),
        evaluation=EvaluationSchedule(
            frequency_episodes=0,
            evaluation_episodes=350,
            hold_out_scenarios=(),
            step_schedule=((None, 20_000),),
            randomize_seeds=False,
        ),
    )

    assert (
        _deterministic_eval_seed_for_episode(
            config,
            episode_idx=episode_idx,
            scenario_cycle_length=71,
        )
        == expected_seed
    )


def test_full_maintained_eval_manifest_loads_unique_scenarios() -> None:
    """The maintained eval surface should expose one unique row per maintained positive scenario."""
    scenarios = load_scenarios(Path("configs/scenarios/sets/ppo_full_maintained_eval_v1.yaml"))
    scenario_ids = [
        str(scenario.get("name") or scenario.get("scenario_id")) for scenario in scenarios
    ]

    assert len(scenarios) == 71
    assert len(set(scenario_ids)) == 71


def test_horizon100_eval_manifest_overrides_all_episode_limits() -> None:
    """The horizon-100 eval surface should keep membership but force a 100-step limit."""
    scenarios = load_scenarios(
        Path("configs/scenarios/sets/ppo_full_maintained_eval_v1_horizon100.yaml")
    )
    scenario_ids = [
        str(scenario.get("name") or scenario.get("scenario_id")) for scenario in scenarios
    ]
    step_limits = [
        int(scenario["simulation_config"]["max_episode_steps"]) for scenario in scenarios
    ]

    assert len(scenarios) == 71
    assert len(set(scenario_ids)) == 71
    assert len(step_limits) == 71
    assert set(step_limits) == {100}


def test_issue_857_horizon100_training_config_uses_horizon_matched_surface() -> None:
    """The horizon-matched training clone should train and evaluate on the 100-step surface."""
    config_path = Path(
        "configs/training/ppo/ablations/"
        "expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_horizon100.yaml"
    ).resolve()
    expected_surface = Path(
        "configs/scenarios/sets/ppo_full_maintained_eval_v1_horizon100.yaml"
    ).resolve()

    config = load_expert_training_config(config_path)

    assert config.scenario_config == expected_surface
    assert config.evaluation.scenario_config == expected_surface


def test_issue_857_horizon100_surface_truncates_empty_map_at_step_100() -> None:
    """The horizon-matched surface should time out a representative empty-map rollout at 100."""
    config_path = Path(
        "configs/training/ppo/ablations/"
        "expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_horizon100.yaml"
    ).resolve()
    config = load_expert_training_config(config_path)
    scenarios = load_scenarios(config.scenario_config)
    scenario = next(s for s in scenarios if s.get("name") == "empty_map_8_directions_east")
    env_config = build_robot_config_from_scenario(scenario, scenario_path=config.scenario_config)

    _apply_env_overrides(env_config, config.env_overrides)
    env = make_robot_env(
        config=env_config,
        seed=config.seeds[0],
        suite_name="issue857_smoke",
        scenario_name="empty_map_8_directions_east",
        algorithm_name=config.policy_id,
        **config.env_factory_kwargs,
    )

    try:
        env.reset()
        action = np.zeros_like(env.action_space.sample())
        step_count = 0
        terminated = False
        truncated = False
        info = {}
        while not (terminated or truncated):
            step_count += 1
            _obs, _reward, terminated, truncated, info = env.step(action)
            if step_count > 100:
                pytest.fail("Expected the horizon-100 surface to end by step 100.")
    finally:
        env.close()

    assert step_count == 100
    assert terminated is True
    assert truncated is False
    assert info["meta"]["is_timesteps_exceeded"] is True


def test_issue_708_predictive_foresight_override_enables_predictive_observation() -> None:
    """The issue-708 config should now expose predictive foresight features on reset."""
    config_path = Path(
        "configs/training/ppo/expert_ppo_issue_708_br06_v11_predictive_foresight_success_priority_from_scratch.yaml"
    ).resolve()
    config = load_expert_training_config(config_path)
    scenario = load_scenarios(config.scenario_config)[0]
    env_config = build_robot_config_from_scenario(scenario, scenario_path=config.scenario_config)

    assert getattr(env_config, "predictive_foresight_enabled", False) is False
    _apply_env_overrides(env_config, config.env_overrides)
    assert getattr(env_config, "predictive_foresight_enabled", False) is True

    env = make_robot_env(
        config=env_config,
        seed=config.seeds[0],
        suite_name="ppo_issue738_smoke",
        scenario_name="issue738_smoke",
        algorithm_name=config.policy_id,
        **config.env_factory_kwargs,
    )
    try:
        obs, _ = env.reset()
    finally:
        env.close()

    predictive_keys = sorted(key for key in obs if str(key).startswith("predictive_"))
    assert predictive_keys == [
        "predictive_crossing_count",
        "predictive_flow_alignment",
        "predictive_gap_scores",
        "predictive_min_clearance",
        "predictive_ttc_risk",
        "predictive_uncertainty",
    ]


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


def test_parse_num_envs_supports_host_aware_auto_modes() -> None:
    """Loader helper should normalize supported auto num_envs tokens."""

    assert _parse_num_envs(None) is None
    assert _parse_num_envs("auto") == "auto_throughput"
    assert _parse_num_envs("auto_throughput") == "auto_throughput"
    assert _parse_num_envs("auto_stable") == "auto_stable"
    assert _parse_num_envs(8) == 8


def test_load_expert_training_config_supports_auto_stable_num_envs(tmp_path) -> None:
    """Configs should preserve host-aware auto num_envs modes."""

    scenario_config = Path("configs/scenarios/classic_interactions_francis2023.yaml").resolve()
    config_path = tmp_path / "auto_stable.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "policy_id": "ppo_auto_stable_test",
                "scenario_config": str(scenario_config),
                "num_envs": "auto_stable",
                "worker_mode": "auto",
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
    assert config.num_envs == "auto_stable"


def test_load_expert_training_config_merges_base_config(tmp_path) -> None:
    """PPO configs may inherit common settings while preserving explicit variants."""
    scenario_config = Path("configs/scenarios/classic_interactions_francis2023.yaml").resolve()
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        yaml.safe_dump(
            {
                "policy_id": "base_policy",
                "scenario_config": str(scenario_config),
                "num_envs": 8,
                "worker_mode": "subproc",
                "seeds": [123],
                "randomize_seeds": True,
                "total_timesteps": 16000000,
                "feature_extractor": "grid_socnav",
                "env_overrides": {
                    "observation_mode": "socnav_struct",
                    "use_occupancy_grid": True,
                },
                "tracking": {
                    "wandb": {
                        "enabled": True,
                        "project": "robot_sf",
                        "group": "shared",
                        "tags": ["base"],
                    }
                },
                "convergence": {
                    "success_rate": 0.9,
                    "collision_rate": 0.05,
                    "plateau_window": 3000,
                },
                "evaluation": {
                    "evaluation_episodes": 20,
                    "hold_out_scenarios": [],
                    "step_schedule": [{"until_step": 16000000, "every_steps": 500000}],
                },
            }
        ),
        encoding="utf-8",
    )
    variant_path = tmp_path / "variant.yaml"
    variant_path.write_text(
        yaml.safe_dump(
            {
                "base_config": "base.yaml",
                "policy_id": "variant_policy",
                "num_envs": 14,
                "env_overrides": {"include_grid_in_observation": True},
                "tracking": {"wandb": {"tags": ["base", "num-envs-14"]}},
            }
        ),
        encoding="utf-8",
    )

    config = load_expert_training_config(variant_path)

    assert config.policy_id == "variant_policy"
    assert config.num_envs == 14
    assert config.worker_mode == "subproc"
    assert config.env_overrides["observation_mode"] == "socnav_struct"
    assert config.env_overrides["use_occupancy_grid"] is True
    assert config.env_overrides["include_grid_in_observation"] is True
    assert config.tracking["wandb"]["project"] == "robot_sf"
    assert config.tracking["wandb"]["tags"] == ["base", "num-envs-14"]


# Issue #6490: the issue_576_br06 PPO family was migrated to inherit shared
# settings from a single base config. The constants below pin that contract.
_ISSUE_576_BR06_FAMILY_DIR = Path("configs/training/ppo")
_ISSUE_576_BR06_BASE_NAME = "expert_ppo_issue_576_br06_base.yaml"
_ISSUE_576_BR06_VARIANTS = [
    "expert_ppo_issue_576_br06_v2_15m_all_maps.yaml",
    "expert_ppo_issue_576_br06_v2_15m_all_maps_randomized.yaml",
    "expert_ppo_issue_576_br06_v2_sanity_500k_all_maps.yaml",
    "expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml",
    "expert_ppo_issue_576_br06_v4_overnight_safety_warmstart.yaml",
    "expert_ppo_issue_576_br06_v4_validation_120k.yaml",
    "expert_ppo_issue_576_br06_v5_predictive_foresight.yaml",
    "expert_ppo_issue_576_br06_v6_predictive_foresight_success_aligned.yaml",
    "expert_ppo_issue_576_br06_v6_predictive_foresight_success_aligned_auto_envs.yaml",
    "expert_ppo_issue_576_br06_v7_predictive_foresight_xl_ego_success_aligned.yaml",
    "expert_ppo_issue_576_br06_v7_predictive_foresight_xl_ego_success_aligned_auto_envs.yaml",
    "expert_ppo_issue_576_br06_v8_predictive_foresight_success_priority.yaml",
    "expert_ppo_issue_576_br06_v9_predictive_foresight_xl_ego_success_priority.yaml",
    "expert_ppo_issue_576_br06_v9_predictive_foresight_xl_ego_success_priority_auto_envs.yaml",
    "expert_ppo_issue_576_br06_v10_predictive_foresight_success_priority_policy_analysis_select.yaml",
]
# The five variants that previously carried the deprecated
# evaluation.frequency_episodes field (ignored by train_ppo.py in favor of
# step_schedule). The migration drops it from exactly these files.
_ISSUE_576_BR06_DROPPED_FREQUENCY_EPISODES = {
    "expert_ppo_issue_576_br06_v2_15m_all_maps.yaml",
    "expert_ppo_issue_576_br06_v2_15m_all_maps_randomized.yaml",
    "expert_ppo_issue_576_br06_v2_sanity_500k_all_maps.yaml",
    "expert_ppo_issue_576_br06_v4_overnight_safety_warmstart.yaml",
    "expert_ppo_issue_576_br06_v4_validation_120k.yaml",
}

# Frozen pre-inheritance resolved-config baseline. It is the canonical JSON
# (sorted keys, compact separators) of `_load_expert_training_config_mapping`
# for every variant on origin/main BEFORE the base_config refactor, compressed
# with zlib and base64-encoded so the regression oracle stays self-contained
# (no git dependency, no opaque external fixture). Regenerate against a clean
# origin/main checkout with:
#   uv run python -c "import json,zlib,base64,pathlib;from scripts.training.train_ppo \
#     import _load_expert_training_config_mapping as L;b=pathlib.Path('configs/training/ppo'); \
#     d={p.name:L(b/p) for p in sorted(b.glob('expert_ppo_issue_576_br06_v*.yaml'))}; \
#     print(base64.b64encode(zlib.compress(json.dumps(d,sort_keys=True,separators=(',',':')).encode(),9)).decode())"
_ISSUE_576_BR06_PRECHANGE_BASELINE_B64 = (
    "eNrtXdtyozgQ/ReejZeLb8l37NvUlEoG2dYGJFaAE+9U/n2PBLbxDI4vcWYTb08lNbGurab79Gkk"
    "8A9PvBTCVKwoNJNlWQs2nk7Y3AQTtg4DVhiRyqSSa8EW2ohSLlcVK+skEWWJSqmNrDas0JlMNowr"
    "nm1KWbJSZCKphhueZ97jD28uyoolK5E8FVqqiuWiMjLxHr3tQIZXwht4iVZrYZZCJcJ2S3SWyVJq"
    "1dQ/BsNgPPCKDB94zZ6lSvWz9xgHQTA4HAktH14HnlBrtuBJpc2GPT1zsyztqEbgz7RT8CzsosqD"
    "Gb1HPxwPAytSjnVX+BwMMXkm11It3acgxGcluGE59Iai2LYvjF5CTfgcDkOIlWtdrZQrQJ9whqKK"
    "LxWvmkki26cSJpdQHZtrVaNhFLhSmQtd25knduaqSpiR5ZPrNHvF8tqFKJ5jxZ5BW8EgbpEJOzhb"
    "x16rAw2lGpkKt8Ql/kIztZBLt2KhMD2zOtZzjdkqUwsse8WVEhl6fPP0HCInGbpjeQLD4OJxVbrr"
    "lc+lEqn3feCtnBZxOdySoAKd1c0ig2E08OpSMLHUbGGcuM0szzKtVk0XiCpVktWpYE5CqRjmFWbd"
    "aqrp0CliuU7tukudKL5mEKpOqkbCkq047HUrtzXcxC6+GaPXolOxltbovKSovSNthOLzDIt9a5yF"
    "0bDvRFt9awM5sgVrVxk25nG6UybU0qnlaIcVvO4fqKCsRIF1zY40y/kL4/Ama9vh5FgjqDFjEsvq"
    "1sOMXzatG4uUrSO2qLPsmGacF6QS6naeGwynRxoa+BfslKVVaxbO6DrmyFH/zOY8se6Z7q6ZW4la"
    "1hnmKQthL4JTpy3PpJu+LXZKqzaFvZapXCyEwfolfCs1EMV6RCnzzoR2BGcyAEFYnq4LjGGlX3Qa"
    "AdG4W0ojj3U+a886SeoCS944k23r4HJrntWt2f7ofGKikKV2bhhDSKvQX4ATrZSAAuBk27X3tlvD"
    "nTUGWvCsRJuVzlJmFVvCnzlQ2Trud4s1okDZSqR1BuG/WXEE0LA1nDBw/16/Q+qF4FVt4KQvlXGY"
    "CQU6T2wczOtp0cHQ1OjCzr+FX0BfAzR7HImjwWSEn+9t1ZMwqGGl/Mdq5Nt4EA9iK3LjzyuZpkLB"
    "qHJbGUazAX6tnKrO4YlrlHq8rjSrVrhoy1VRO+dv1NRYMwJaG9tuEs32wytRMW4SuOi3aDwZ4Bdy"
    "2+lWsDtTcECcU8qcV8nKLRCYjoZekskCKlLLrY6E83uxcLEtgB9nMGWFCNMqcjocC98GPWs8Ollh"
    "2BHMG1qHCE+Z6xbZUMARC3NMBFnF3m2wwjoXBy7+s1JiFo5zBrcDWhQ2fLYDpSwKokkQB6M/g+k4"
    "CgIsf2tcO8/whsM/8LMzuj+SjJelTIDeCCrWZbQqLeSrRJYYL24YQWekkiNeuYj6w4bOhQRcA8qx"
    "+OXGovEG+G3b74saCVHWDdvttMCSCugJngH7WToIHQ/6anOYQZ23aLGtT4yGFWCmou09OlLb2zvV"
    "2jzzTV/XbVXm6Eq3xgFOM7S96Fuhj7fYTh11F7YSPLXIsYsg+1adkXIwq6NDWIpQ8aej9Ubw7Fkb"
    "oEyqa0TBX1V9sOKqsYBSJE2o3stjsbO1UXh1PIjicDCdTgcPD9EgjOMpPKnSFfDa0p8WpuIGpiyw"
    "w6ieWnOphCq1mWvEiR2fgG3MHegehuoW1z3Hb33wW98avw9E8PeI4O8QwW/83N8CgL8DgL/0nLXB"
    "pfEiHw7VPwYGd+FS/2W7PraRrlx41n+XjljtxEHZ3PiB/R/jHUTZ/YjejuT6W6TyGwqImna2Y3I7"
    "cujvyaEPcthBDRA4e11wgQHKO2JVzyF94kjk8RQhOsCPHeO/islHFzL5k1TvFEOOiCH3qu3/ScpC"
    "S8qM+LsWtvdh+ZUEa9ziVo3lZq5wh2avg/sjY6cY2CFYnCBUN2QbF8abcPwh8eZo/DAC80iAcnR9"
    "xDgY41e4j/oQ3l2yS+C9Qw8J6QnpCekJ6c9C+g5unAD9I4ns3cUC/0Aj/2FY6Kjcb3RzbaAoubI3"
    "T6CvJ8oHKEpQlDh6w/W+gb8PB/4Lqt+H5uPfyusbTbwLv5shbkrp4zuh9LS9SWDdBWXi7jeC8Ji4"
    "+4l9g/gi+h7fgL7Hp2/iv4u+jxyKqmY/lC8EAjgMNQfCmE96jOfc+HH6nE/03nM+oyvP+YwOj/mM"
    "6JgPxcGbnCy5bdISjc8Pj32h9OEuQ+nsRBR9C1HpKAkdJaGjJHd1lGTk79z9KCPc+T+aX88JuyjS"
    "TwkbuPEbq3YM9YCu7gW9+tTHiCH2yLQJPmEUPP0ejjgmikgUkSjiOyni5OYMMf6V9IXjuJ/1Tb4w"
    "5xud5Hw/wSIRvXa4UtfwvJ1tRKPGYogGEg28MQ3svxEYT65kfe4Bi2tp0rj3gQO6n0ZkiR6bo8fm"
    "6LG5a25u0k3M33gTsz+AEasl3kq89a5uX479I4+cnf/E282fdvu4h9Ymbz8IjBx+qTrnoe6Up7/7"
    "/RbBlTw9/On9FjP3kYg6EXUi6kTUiahfQdTPi2hE3Im4E3G/K+I+6X/7xPZNEXvHv+TVFZMPe3NF"
    "K0/nxRWTT8D0mXuTkEVa4vzE+YnzE+cnzk+c/0s8rHNpgKMUgFIASgEoBfhyKcDAwaHvMOy6Z9ym"
    "/VD5kjkSRzf/KRH4PycCSxcQU2mnd1bwxoCfNmsIozPShmj0zrShQQxKHChx+JqbBRcFQkoYKGGg"
    "hOGuEoZpP+d/yXz4/zvzhunN84ZGrDcTiOmH7iFcBJe0lUAZBGUQlEFQBkEZxJ1vPVwZFymhoISC"
    "EgpKKL58QvH+HYnZeV9IRpkEfdEmHUqiQ0mUGVBm8Nn3Fs4MaZQEUBJAScBdJQGzt48hdTz/Evo/"
    "+51fojn70H2Eh7PulxDpJ9JP2we0fUDbB5Qk3GuScFkkpFyBcgXKFe4qV3g4a8PgypTh4eN3DHpy"
    "h4dPlDvQISTKIiiLoCyCsgjKIu78ENK1gZGSCkoqKKmgpOIOkoqzziG9/guuttSc"
)
_ISSUE_576_BR06_PRECHANGE_BASELINE_SHA256 = (
    "a4a235f03e3eda8301d5194c9f201f713b1aa139020060c67b6ba508b88d4d7f"
)

# Per-variant family-specific overrides (captured from origin/main). Documents
# that the inheritance refactor leaves every launch/reward/foresight/warm-start/
# tracking override explicit. Tuple order:
# (policy_id, num_envs, worker_mode, seeds, total_timesteps, reward_name,
#  foresight_model_id, resume_model_id, wandb_group, wandb_tags)
_ISSUE_576_BR06_EXPECTED_OVERRIDES = {
    "expert_ppo_issue_576_br06_v2_15m_all_maps.yaml": (
        "ppo_expert_br06_v2_15m_all_maps",
        "auto",
        "auto",
        (123, 231, 777, 992, 1337),
        15000000,
        "route_completion_v2",
        None,
        None,
        "issue-576-br06",
        ["issue-576", "br-06", "ppo", "retrain-v2", "route-completion-v2"],
    ),
    "expert_ppo_issue_576_br06_v2_15m_all_maps_randomized.yaml": (
        "ppo_expert_br06_v2_15m_all_maps_randomized",
        "auto",
        "auto",
        (123, 231, 777, 992, 1337),
        15000000,
        "route_completion_v2",
        None,
        None,
        "issue-576-br06-randomized",
        ["issue-576", "br-06", "ppo", "retrain-v2", "route-completion-v2", "randomize-seeds"],
    ),
    "expert_ppo_issue_576_br06_v2_sanity_500k_all_maps.yaml": (
        "ppo_expert_br06_v2_sanity_500k_all_maps",
        "auto",
        "auto",
        (123,),
        500000,
        "route_completion_v2",
        None,
        None,
        "issue-576-br06",
        ["issue-576", "br-06", "ppo", "sanity", "route-completion-v2"],
    ),
    "expert_ppo_issue_576_br06_v3_15m_all_maps_randomized.yaml": (
        "ppo_expert_br06_v3_15m_all_maps_randomized",
        "auto",
        "auto",
        (123, 231, 777, 992, 1337),
        15000000,
        "route_completion_v3",
        None,
        None,
        "issue-576-br06-v3-randomized",
        ["issue-576", "br-06", "ppo", "retrain-v3", "route-completion-v3", "randomize-seeds"],
    ),
    "expert_ppo_issue_576_br06_v4_overnight_safety_warmstart.yaml": (
        "ppo_expert_br06_v4_overnight_safety_warmstart",
        8,
        "subproc",
        (123, 231, 777, 992, 1337),
        30000000,
        "route_completion_v3",
        None,
        "ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200",
        "issue-576-br06-v4-overnight",
        [
            "issue-576",
            "br-06",
            "ppo",
            "warmstart",
            "route-completion-v3",
            "safety-weighted",
            "randomized",
            "overnight",
        ],
    ),
    "expert_ppo_issue_576_br06_v4_validation_120k.yaml": (
        "ppo_expert_br06_v4_validation_120k",
        4,
        "subproc",
        (123,),
        15360000,
        "route_completion_v3",
        None,
        "ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200",
        None,
        None,
    ),
    "expert_ppo_issue_576_br06_v5_predictive_foresight.yaml": (
        "ppo_expert_br06_v5_predictive_foresight",
        8,
        "subproc",
        (123, 231, 777, 992, 1337),
        30000000,
        "route_completion_v3",
        "predictive_proxy_selected_v2_full",
        "ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200",
        "issue-576-br06-v5-predictive-foresight",
        ["issue-576", "br-06", "ppo", "predictive-foresight", "route-completion-v3", "randomized"],
    ),
    "expert_ppo_issue_576_br06_v6_predictive_foresight_success_aligned.yaml": (
        "ppo_expert_br06_v6_predictive_foresight_success_aligned",
        8,
        "subproc",
        (123, 231, 777, 992, 1337),
        30000000,
        "route_completion_v3",
        "predictive_proxy_selected_v2_full",
        "ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200",
        "issue-576-br06-v6-predictive-foresight-success-aligned",
        [
            "issue-576",
            "br-06",
            "ppo",
            "predictive-foresight",
            "success-aligned-reward",
            "v6",
            "route-completion-v3",
            "randomized",
        ],
    ),
    "expert_ppo_issue_576_br06_v6_predictive_foresight_success_aligned_auto_envs.yaml": (
        "ppo_expert_br06_v6_predictive_foresight_success_aligned_auto_envs",
        "auto",
        "auto",
        (123, 231, 777, 992, 1337),
        30000000,
        "route_completion_v3",
        "predictive_proxy_selected_v2_full",
        "ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200",
        "issue-576-br06-v6-predictive-foresight-success-aligned",
        [
            "issue-576",
            "br-06",
            "ppo",
            "predictive-foresight",
            "success-aligned-reward",
            "v6",
            "route-completion-v3",
            "randomized",
            "auto-envs",
        ],
    ),
    "expert_ppo_issue_576_br06_v7_predictive_foresight_xl_ego_success_aligned.yaml": (
        "ppo_expert_br06_v7_predictive_foresight_xl_ego_success_aligned",
        8,
        "subproc",
        (123, 231, 777, 992, 1337),
        30000000,
        "route_completion_v3",
        "predictive_proxy_selected_v2_xl_ego",
        "ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200",
        "issue-576-br06-v7-predictive-foresight-xl-ego-success-aligned",
        [
            "issue-576",
            "br-06",
            "ppo",
            "predictive-foresight",
            "xl-ego",
            "success-aligned-reward",
            "v7",
            "route-completion-v3",
            "randomized",
        ],
    ),
    "expert_ppo_issue_576_br06_v7_predictive_foresight_xl_ego_success_aligned_auto_envs.yaml": (
        "ppo_expert_br06_v7_predictive_foresight_xl_ego_success_aligned_auto_envs",
        "auto",
        "auto",
        (123, 231, 777, 992, 1337),
        30000000,
        "route_completion_v3",
        "predictive_proxy_selected_v2_xl_ego",
        "ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200",
        "issue-576-br06-v7-predictive-foresight-xl-ego-success-aligned",
        [
            "issue-576",
            "br-06",
            "ppo",
            "predictive-foresight",
            "xl-ego",
            "success-aligned-reward",
            "v7",
            "route-completion-v3",
            "randomized",
            "auto-envs",
        ],
    ),
    "expert_ppo_issue_576_br06_v8_predictive_foresight_success_priority.yaml": (
        "ppo_expert_br06_v8_predictive_foresight_success_priority",
        8,
        "subproc",
        (123, 231, 777, 992, 1337),
        30000000,
        "route_completion_v3",
        "predictive_proxy_selected_v2_full",
        "ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200",
        "issue-576-br06-v8-predictive-foresight-success-priority",
        [
            "issue-576",
            "br-06",
            "ppo",
            "predictive-foresight",
            "success-priority-reward",
            "v8",
            "route-completion-v3",
            "randomized",
        ],
    ),
    "expert_ppo_issue_576_br06_v9_predictive_foresight_xl_ego_success_priority.yaml": (
        "ppo_expert_br06_v9_predictive_foresight_xl_ego_success_priority",
        8,
        "subproc",
        (123, 231, 777, 992, 1337),
        30000000,
        "route_completion_v3",
        "predictive_proxy_selected_v2_xl_ego",
        "ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200",
        "issue-576-br06-v9-predictive-foresight-xl-ego-success-priority",
        [
            "issue-576",
            "br-06",
            "ppo",
            "predictive-foresight",
            "xl-ego",
            "success-priority-reward",
            "v9",
            "route-completion-v3",
            "randomized",
        ],
    ),
    "expert_ppo_issue_576_br06_v9_predictive_foresight_xl_ego_success_priority_auto_envs.yaml": (
        "ppo_expert_br06_v9_predictive_foresight_xl_ego_success_priority_auto_envs",
        "auto",
        "auto",
        (123, 231, 777, 992, 1337),
        30000000,
        "route_completion_v3",
        "predictive_proxy_selected_v2_xl_ego",
        "ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200",
        "issue-576-br06-v9-predictive-foresight-xl-ego-success-priority",
        [
            "issue-576",
            "br-06",
            "ppo",
            "predictive-foresight",
            "xl-ego",
            "success-priority-reward",
            "v9",
            "route-completion-v3",
            "randomized",
            "auto-envs",
        ],
    ),
    "expert_ppo_issue_576_br06_v10_predictive_foresight_success_priority_policy_analysis_select.yaml": (
        "ppo_expert_br06_v10_predictive_foresight_success_priority_policy_analysis_select",
        "auto_throughput",
        "subproc",
        (123, 231, 777, 992, 1337),
        30000000,
        "route_completion_v3",
        "predictive_proxy_selected_v2_full",
        "ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200",
        "issue-576-br06-v10-predictive-foresight-policy-analysis-select",
        [
            "issue-576",
            "br-06",
            "ppo",
            "predictive-foresight",
            "success-priority-reward",
            "v10",
            "policy-analysis-select",
            "route-completion-v3",
            "randomized",
        ],
    ),
}


def _issue_576_br06_prechange_baseline() -> dict:
    """Decode and integrity-check the frozen pre-refactor resolved-config baseline."""
    blob = zlib.decompress(base64.b64decode(_ISSUE_576_BR06_PRECHANGE_BASELINE_B64))
    assert hashlib.sha256(blob).hexdigest() == _ISSUE_576_BR06_PRECHANGE_BASELINE_SHA256
    return json.loads(blob)


def _strip_frequency_episodes(mapping: dict) -> dict:
    """Return a copy of ``mapping`` without the deprecated evaluation.frequency_episodes."""
    cleaned = copy.deepcopy(mapping)
    evaluation = cleaned.get("evaluation")
    if isinstance(evaluation, dict):
        evaluation.pop("frequency_episodes", None)
    return cleaned


@pytest.mark.parametrize("variant", _ISSUE_576_BR06_VARIANTS)
def test_issue_576_br06_family_resolves_to_prechange_values(variant: str) -> None:
    """Each migrated variant must resolve to its pre-refactor effective values.

    The base_config deep-merge must preserve every resolved value. The only
    intentional change is dropping the deprecated, ignored
    evaluation.frequency_episodes field from the five variants that carried it
    (it resolves to 0 everywhere now; cadence is controlled by step_schedule), so
    it is stripped from both sides before the exhaustive mapping comparison.
    """
    path = (_ISSUE_576_BR06_FAMILY_DIR / variant).resolve()

    resolved_mapping = _load_expert_training_config_mapping(path)
    # Exercise the full construction path to confirm load_expert_training_config
    # does not diverge from the resolver output.
    config = load_expert_training_config(path)

    expected_mapping = _issue_576_br06_prechange_baseline()[variant]
    assert _strip_frequency_episodes(resolved_mapping) == _strip_frequency_episodes(
        expected_mapping
    )
    # Cadence source is unchanged and the deprecated field now resolves to 0.
    assert config.evaluation.step_schedule
    assert config.evaluation.frequency_episodes == 0


@pytest.mark.parametrize("variant", _ISSUE_576_BR06_VARIANTS)
def test_issue_576_br06_family_overrides_remain_explicit(variant: str) -> None:
    """Family-specific launch/reward/foresight/warm-start/tracking overrides stay explicit."""
    (
        policy_id,
        num_envs,
        worker_mode,
        seeds,
        total_timesteps,
        reward_name,
        foresight_model_id,
        resume_model_id,
        wandb_group,
        wandb_tags,
    ) = _ISSUE_576_BR06_EXPECTED_OVERRIDES[variant]

    config = load_expert_training_config((_ISSUE_576_BR06_FAMILY_DIR / variant).resolve())
    variant_yaml = yaml.safe_load(
        (_ISSUE_576_BR06_FAMILY_DIR / variant).read_text(encoding="utf-8")
    )

    assert config.policy_id == policy_id
    # num_envs is normalized by the loader (e.g. "auto" -> "auto_throughput"),
    # so compare against the parsed form rather than the raw token.
    assert config.num_envs == _parse_num_envs(num_envs)
    assert config.worker_mode == worker_mode
    assert config.seeds == seeds
    assert config.total_timesteps == total_timesteps
    assert config.env_factory_kwargs["reward_name"] == reward_name
    assert config.env_overrides.get("predictive_foresight_model_id") == foresight_model_id
    assert config.resume_model_id == resume_model_id
    assert config.tracking["wandb"].get("group") == wandb_group
    assert config.tracking["wandb"].get("tags") == wandb_tags
    # Plateau window remains an explicit per-variant override (not in the base).
    assert "plateau_window" in variant_yaml["convergence"]


def test_issue_576_br06_base_inheritance_and_frequency_episodes_drop() -> None:
    """Every variant inherits the base and no longer carries the deprecated field."""
    base_path = (_ISSUE_576_BR06_FAMILY_DIR / _ISSUE_576_BR06_BASE_NAME).resolve()
    base_text = base_path.read_text(encoding="utf-8")
    base_yaml = yaml.safe_load(base_text)
    # The base holds shared settings and must not self-inherit or carry the
    # deprecated field.
    assert "base_config" not in base_yaml
    assert "frequency_episodes" not in base_yaml.get("evaluation", {})

    for variant in _ISSUE_576_BR06_VARIANTS:
        variant_path = (_ISSUE_576_BR06_FAMILY_DIR / variant).resolve()
        variant_yaml = yaml.safe_load(variant_path.read_text(encoding="utf-8"))
        assert variant_yaml["base_config"] == _ISSUE_576_BR06_BASE_NAME
        # The deprecated field is dropped everywhere (it was only ever present
        # in the five variants listed below, and ignored in favor of step_schedule).
        assert "frequency_episodes" not in variant_yaml.get("evaluation", {})
        # Cadence source remains explicit per variant.
        assert variant_yaml["evaluation"]["step_schedule"]
    assert len(_ISSUE_576_BR06_DROPPED_FREQUENCY_EPISODES) == 5


def test_load_expert_training_config_base_config_cycle_raises_value_error(
    tmp_path,
) -> None:
    """A base_config cycle (self-reference or mutual) must raise ValueError."""
    scenario_config = Path("configs/scenarios/classic_interactions_francis2023.yaml").resolve()
    self_ref = tmp_path / "self_cycle.yaml"
    self_ref.write_text(
        yaml.safe_dump(
            {
                "base_config": "self_cycle.yaml",
                "scenario_config": str(scenario_config),
                "total_timesteps": 1000,
                "seeds": [1],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="base_config cycle detected"):
        _load_expert_training_config_mapping(self_ref)

    mutual_a = tmp_path / "mutual_a.yaml"
    mutual_b = tmp_path / "mutual_b.yaml"
    mutual_a.write_text(
        yaml.safe_dump({"base_config": "mutual_b.yaml", "policy_id": "a"}),
        encoding="utf-8",
    )
    mutual_b.write_text(
        yaml.safe_dump({"base_config": "mutual_a.yaml", "policy_id": "b"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="base_config cycle detected"):
        _load_expert_training_config_mapping(mutual_a)


def test_load_expert_training_config_missing_base_config_raises(tmp_path) -> None:
    """A missing/nonexistent base_config must fail closed.

    The resolver raises FileNotFoundError (an OSError) for a missing base_config;
    a cycle raises ValueError (covered above). Issue #6490 expected a ValueError
    here as well, but the byte-frozen resolver in scripts/training/train_ppo.py
    surfaces missing-base as FileNotFoundError. The run still fails closed
    (no silent wrong values), so this asserts the actual fail-closed behavior;
    narrowing missing-base to ValueError is tracked as a follow-up because
    scripts/training/train_ppo.py is out of scope for this refactor.
    """
    scenario_config = Path("configs/scenarios/classic_interactions_francis2023.yaml").resolve()
    config_path = tmp_path / "missing_base.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "base_config": "does_not_exist_base.yaml",
                "scenario_config": str(scenario_config),
                "total_timesteps": 1000,
                "seeds": [1],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(FileNotFoundError):
        _load_expert_training_config_mapping(config_path)


@pytest.mark.parametrize("num_envs", [8, 14, 16, 30, 32])
def test_num_envs_benchmark_variants_inherit_shared_config(num_envs: int) -> None:
    """The issue-576 num_envs variants should keep only launch-specific overrides."""
    config_path = Path(
        "configs/training/ppo/benchmark_num_envs/"
        f"expert_ppo_issue_576_br06_v8_num_envs_benchmark_{num_envs:02d}_1m.yaml"
    ).resolve()

    config = load_expert_training_config(config_path)

    assert config.policy_id == f"ppo_expert_br06_v8_num_envs_benchmark_{num_envs:02d}_1m"
    assert config.num_envs == num_envs
    assert config.worker_mode == "subproc"
    assert config.resume_model_id == "ppo_expert_br06_v3_15m_all_maps_randomized_20260304T075200"
    assert config.total_timesteps == 16000000
    assert config.best_checkpoint_metric == "success_rate"
    assert config.feature_extractor == "grid_socnav"
    assert config.env_overrides["predictive_foresight_model_id"] == (
        "predictive_proxy_selected_v2_full"
    )
    assert config.env_overrides["robot_config"]["type"] == "differential_drive"
    assert config.env_factory_kwargs["reward_name"] == "route_completion_v3"
    assert config.scenario_sampling["weights"]["classic_cross_trap_high"] == 4.0
    assert config.evaluation.step_schedule == ((16000000, 500000),)
    assert config.tracking["wandb"]["group"] == "benchmark-num-envs-imech156u-v8"
    assert config.tracking["wandb"]["tags"] == [
        "issue-576",
        "br-06",
        "ppo",
        "predictive-foresight",
        "success-priority-reward",
        "num-envs-benchmark",
        "imech156-u",
        f"num-envs-{num_envs}",
        "1m-continuation",
    ]


def test_resolve_num_envs_auto_modes_use_cpu_and_memory_caps(monkeypatch) -> None:
    """Auto env modes should resolve to throughput and stable host-aware counts."""

    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    monkeypatch.delenv("SLURM_CPUS_ON_NODE", raising=False)
    monkeypatch.delenv("SLURM_JOB_CPUS_PER_NODE", raising=False)
    monkeypatch.setattr("scripts.training.train_ppo.os.cpu_count", lambda: 32)
    monkeypatch.setattr(
        "scripts.training.train_ppo._host_memory_gib",
        lambda: 36.8,
    )

    base_kwargs = {
        "scenario_config": Path(
            "configs/scenarios/classic_interactions_francis2023.yaml"
        ).resolve(),
        "seeds": (123,),
        "total_timesteps": 1000,
        "policy_id": "ppo_auto_test",
        "convergence": ConvergenceCriteria(
            success_rate=0.9,
            collision_rate=0.05,
            plateau_window=100,
        ),
        "evaluation": EvaluationSchedule(
            frequency_episodes=0,
            evaluation_episodes=4,
            hold_out_scenarios=(),
            step_schedule=((1000, 1000),),
        ),
    }

    throughput_cfg = ExpertTrainingConfig(**base_kwargs, num_envs="auto_throughput")
    stable_cfg = ExpertTrainingConfig(**base_kwargs, num_envs="auto_stable")

    assert _resolve_num_envs(throughput_cfg) == 27
    assert _resolve_num_envs(stable_cfg) == 13

    throughput_details = _describe_num_envs_resolution(throughput_cfg)
    stable_details = _describe_num_envs_resolution(stable_cfg)
    assert throughput_details["mode"] == "auto_throughput"
    assert throughput_details["decision"] == "throughput heuristic; limited by memory cap"
    assert throughput_details["memory_cap"] == 27
    assert stable_details["mode"] == "auto_stable"
    assert stable_details["decision"] == "stable headroom heuristic; limited by memory cap"
    assert stable_details["memory_cap"] == 13


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
        """Return the expected registry path while recording resolver inputs."""
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
        """W&B run stub that records logged training metrics."""

        def log(self, payload: dict[str, float | int], *, step: int) -> None:
            """Record a direct W&B log payload and step."""
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
        """W&B run stub storing metric payloads."""

        def __init__(self) -> None:
            self.payloads: list[tuple[dict[str, float | int], int]] = []

        def log(self, payload, step):
            """Record a metric payload at the given step."""
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
        """W&B artifact stub that records metadata and attached files."""

        def __init__(self, name, artifact_type=None, metadata=None, **kwargs):
            if artifact_type is None:
                artifact_type = kwargs.get("type")
            self.name = name
            self.type = artifact_type
            self.metadata = metadata
            self.description = None
            self.files: list[tuple[str, str | None]] = []

        def add_file(self, path, name=None):
            """Record an artifact file attachment."""
            self.files.append((str(path), name))

    class _Run:
        """W&B run stub that records logged artifacts and aliases."""

        def __init__(self) -> None:
            self.logged: list[tuple[object, list[str] | None]] = []

        def log_artifact(self, artifact, aliases=None):
            """Record an artifact upload."""
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
        """W&B artifact stub for immediate checkpoint upload tests."""

        def __init__(self, name, artifact_type=None, metadata=None, **kwargs):
            if artifact_type is None:
                artifact_type = kwargs.get("type")
            self.name = name
            self.type = artifact_type
            self.metadata = metadata
            self.description = None
            self.files: list[tuple[str, str | None]] = []

        def add_file(self, path, name=None):
            """Record an artifact file attachment."""
            self.files.append((str(path), name))

    class _Run:
        """W&B run stub with summary and artifact logging state."""

        def __init__(self) -> None:
            self.summary: dict[str, object] = {}
            self.logged: list[tuple[object, list[str] | None]] = []

        def log_artifact(self, artifact, aliases=None):
            """Record an artifact upload."""
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


def test_issue_2557_base_config_inheritance_equivalence() -> None:
    """All issue-2557 variants must match their pre-refactor resolved-config fingerprints."""
    baseline_path = Path("tests/integration/_baseline_issue_2557_resolved.json").resolve()
    assert baseline_path.exists(), (
        "Pre-change baseline missing; re-run capture before changing configs"
    )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert baseline["schema_version"] == "resolved-config-fingerprint.v1"
    assert isinstance(baseline["source_revision"], str)
    fingerprints = baseline["variants"]
    assert isinstance(fingerprints, dict)

    ablate_dir = Path("configs/training/ppo/ablations")
    variant_paths = sorted(ablate_dir.glob("expert_ppo_issue_2557_*_seed*_fixed.yaml"))
    assert len(variant_paths) == 24
    assert {path.name for path in variant_paths} == set(fingerprints)

    for config_path in variant_paths:
        resolved = _load_expert_training_config_mapping(config_path)
        canonical = json.dumps(resolved, default=str, sort_keys=True, separators=(",", ":"))
        actual_fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
        assert actual_fingerprint == fingerprints[config_path.name], (
            f"Resolved config {config_path.name} differs from the baseline at "
            f"{baseline['source_revision']}."
        )


def test_issue_6679_single_factor_base_config_inheritance_equivalence() -> None:
    """All 18 single_factor ablation variants must match pre-refactor resolved fingerprints."""
    baseline_path = Path("tests/integration/_baseline_issue_6679_resolved.json").resolve()
    assert baseline_path.exists(), (
        "Pre-change baseline missing; re-run capture before changing configs"
    )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert baseline["schema_version"] == "resolved-config-fingerprint.v1"
    assert isinstance(baseline["source_revision"], str)
    fingerprints = baseline["variants"]
    assert isinstance(fingerprints, dict)

    single_factor_dir = Path("configs/training/ppo/ablations/single_factor")
    variant_paths = [
        path
        for path in sorted(single_factor_dir.glob("*.yaml"))
        if not path.name.endswith("_base.yaml")
    ]
    assert len(variant_paths) == 18
    assert {path.name for path in variant_paths} == set(fingerprints)

    for config_path in variant_paths:
        resolved = _load_expert_training_config_mapping(config_path)
        canonical = json.dumps(resolved, default=str, sort_keys=True, separators=(",", ":"))
        actual_fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
        assert actual_fingerprint == fingerprints[config_path.name], (
            f"Resolved config {config_path.name} differs from the baseline at "
            f"{baseline['source_revision']}."
        )


_SINGLE_FACTOR_VARIANTS = [
    "asymmetric_critic_only_10m_env22_seed123.yaml",
    "asymmetric_critic_only_10m_env22_seed1337.yaml",
    "asymmetric_critic_only_10m_env22_seed231.yaml",
    "asymmetric_critic_only_1m_env22_seed123.yaml",
    "asymmetric_critic_only_1m_env22_seed1337.yaml",
    "asymmetric_critic_only_1m_env22_seed231.yaml",
    "attention_only_10m_env22_seed123.yaml",
    "attention_only_10m_env22_seed1337.yaml",
    "attention_only_10m_env22_seed231.yaml",
    "attention_only_1m_env22_seed123.yaml",
    "attention_only_1m_env22_seed1337.yaml",
    "attention_only_1m_env22_seed231.yaml",
    "reward_curriculum_only_10m_env22_seed123.yaml",
    "reward_curriculum_only_10m_env22_seed1337.yaml",
    "reward_curriculum_only_10m_env22_seed231.yaml",
    "reward_curriculum_only_1m_env22_seed123.yaml",
    "reward_curriculum_only_1m_env22_seed1337.yaml",
    "reward_curriculum_only_1m_env22_seed231.yaml",
]


@pytest.mark.parametrize("variant", _SINGLE_FACTOR_VARIANTS)
def test_issue_6679_single_factor_variant_equivalence(variant: str) -> None:
    """Parametrized equivalence test covering every migrated single_factor ablation variant."""
    baseline_path = Path("tests/integration/_baseline_issue_6679_resolved.json").resolve()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    expected_fingerprint = baseline["variants"][variant]

    config_path = Path("configs/training/ppo/ablations/single_factor") / variant
    resolved = _load_expert_training_config_mapping(config_path)
    canonical = json.dumps(resolved, default=str, sort_keys=True, separators=(",", ":"))
    actual_fingerprint = hashlib.sha256(canonical.encode()).hexdigest()

    assert actual_fingerprint == expected_fingerprint, (
        f"Variant {variant} resolved fingerprint {actual_fingerprint} "
        f"does not match baseline {expected_fingerprint}"
    )

    config = load_expert_training_config(config_path)
    assert config.policy_id
    assert len(config.seeds) == 1


_ISSUE_739_VARIANTS = [
    "configs/training/ppo/ablations/expert_ppo_issue_739_stage1_baseline.yaml",
    "configs/training/ppo/ablations/expert_ppo_issue_739_stage1_obs_grid_goal.yaml",
    "configs/training/ppo/ablations/expert_ppo_issue_739_stage1_obs_selective.yaml",
    "configs/training/ppo/ablations/expert_ppo_issue_739_stage1_reward_core.yaml",
    "configs/training/ppo/ablations/expert_ppo_issue_739_stage1_reward_tuned.yaml",
    "configs/training/ppo/ablations/expert_ppo_issue_739_stage2_opt_scale.yaml",
    "configs/training/ppo/ablations/expert_ppo_issue_739_stage2_sampling.yaml",
    "configs/training/ppo/expert_ppo_issue_739_12m_baseline_retrain.yaml",
]


@pytest.mark.parametrize("config_name", _ISSUE_739_VARIANTS)
def test_issue_6682_issue_739_variant_equivalence(config_name: str) -> None:
    """Every migrated issue-739 config must retain its frozen resolved mapping."""
    baseline_path = Path("tests/integration/_baseline_issue_6682_resolved.json").resolve()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    config_path = Path(config_name)
    expected_fingerprint = baseline["variants"][config_path.name]

    resolved = _load_expert_training_config_mapping(config_path)
    canonical = json.dumps(resolved, default=str, sort_keys=True, separators=(",", ":"))
    actual_fingerprint = hashlib.sha256(canonical.encode()).hexdigest()

    assert actual_fingerprint == expected_fingerprint, (
        f"Variant {config_path.name} resolved fingerprint {actual_fingerprint} "
        f"does not match baseline {baseline['source_revision']}"
    )
