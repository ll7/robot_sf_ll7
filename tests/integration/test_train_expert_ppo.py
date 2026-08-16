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


@pytest.mark.parametrize(
    ("config_name", "policy_id", "total_timesteps", "every_steps", "job_type", "tags"),
    [
        (
            "expert_ppo_issue_791_attention_head_promotion_128k.yaml",
            "ppo_expert_issue_791_attention_head_promotion_128k",
            131072,
            65536,
            "expert-ppo-128k-promotion",
            ["issue-791", "attention-head", "ppo", "promotion-128k"],
        ),
        (
            "expert_ppo_issue_791_attention_head_promotion_256k.yaml",
            "ppo_expert_issue_791_attention_head_promotion_256k",
            262144,
            32768,
            "expert-ppo-256k-promotion",
            ["issue-791", "attention-head", "ppo", "promotion-256k"],
        ),
    ],
)
def test_issue_6484_base_config_preserves_resolved_variants(
    config_name: str,
    policy_id: str,
    total_timesteps: int,
    every_steps: int,
    job_type: str,
    tags: list[str],
) -> None:
    """Base inheritance must preserve the frozen pre-refactor config mappings."""
    config_path = (Path("configs/training/ppo/ablations") / config_name).resolve()
    baseline_path = Path(__file__).with_name(
        "_baseline_issue_6484_attention_head_promotion_resolved.json"
    )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    resolved_mapping = _load_expert_training_config_mapping(config_path)
    resolved_fingerprint = hashlib.sha256(
        json.dumps(resolved_mapping, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()

    assert baseline["schema_version"] == "resolved-config-fingerprint.v1"
    assert resolved_fingerprint == baseline["variants"][config_name]

    config = load_expert_training_config(config_path)
    assert config.policy_id == policy_id
    assert config.total_timesteps == total_timesteps
    assert config.evaluation.step_schedule == ((None, every_steps),)
    assert resolved_mapping["tracking"]["wandb"]["job_type"] == job_type
    assert resolved_mapping["tracking"]["wandb"]["tags"] == tags


def test_issue_6484_short_budget_variants_keep_distinct_overrides() -> None:
    """The two inherited variants retain different budgets and evaluation cadence."""
    config_dir = Path("configs/training/ppo/ablations")
    short = _load_expert_training_config_mapping(
        (config_dir / "expert_ppo_issue_791_attention_head_promotion_128k.yaml").resolve()
    )
    long = _load_expert_training_config_mapping(
        (config_dir / "expert_ppo_issue_791_attention_head_promotion_256k.yaml").resolve()
    )

    assert short["total_timesteps"] == 131072
    assert long["total_timesteps"] == 262144
    assert short["evaluation"]["step_schedule"] == [{"every_steps": 65536}]
    assert long["evaluation"]["step_schedule"] == [{"every_steps": 32768}]
    assert short["policy_id"] != long["policy_id"]
    assert short["tracking"]["wandb"]["job_type"] != long["tracking"]["wandb"]["job_type"]


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
# Issue #6680: predictive variants (v5-v10) and the issue-708 from-scratch
# run (v11) now additionally chain through a predictive sub-base that holds
# the byte-identical shared predictive keys.
_ISSUE_576_BR06_FAMILY_DIR = Path("configs/training/ppo")
_ISSUE_576_BR06_BASE_NAME = "expert_ppo_issue_576_br06_base.yaml"
_ISSUE_576_BR06_PREDICTIVE_SUB_BASE_NAME = "expert_ppo_issue_576_br06_predictive_sub_base.yaml"
# Set of variants whose base_config points at the predictive sub-base rather
# than directly at the family base (two-level chain).
_ISSUE_576_BR06_PREDICTIVE_VARIANTS = frozenset(
    {
        "expert_ppo_issue_576_br06_v5_predictive_foresight.yaml",
        "expert_ppo_issue_576_br06_v6_predictive_foresight_success_aligned.yaml",
        "expert_ppo_issue_576_br06_v6_predictive_foresight_success_aligned_auto_envs.yaml",
        "expert_ppo_issue_576_br06_v7_predictive_foresight_xl_ego_success_aligned.yaml",
        "expert_ppo_issue_576_br06_v7_predictive_foresight_xl_ego_success_aligned_auto_envs.yaml",
        "expert_ppo_issue_576_br06_v8_predictive_foresight_success_priority.yaml",
        "expert_ppo_issue_576_br06_v9_predictive_foresight_xl_ego_success_priority.yaml",
        "expert_ppo_issue_576_br06_v9_predictive_foresight_xl_ego_success_priority_auto_envs.yaml",
        "expert_ppo_issue_576_br06_v10_predictive_foresight_success_priority_policy_analysis_select.yaml",
        "expert_ppo_issue_708_br06_v11_predictive_foresight_success_priority_from_scratch.yaml",
    }
)
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
    "expert_ppo_issue_708_br06_v11_predictive_foresight_success_priority_from_scratch.yaml",
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
# Frozen pre-inheritance resolved-config baseline. It is the canonical JSON
# (sorted keys, compact separators) of `_load_expert_training_config_mapping`
# for every variant on origin/main BEFORE the base_config refactor (plus v11
# added in issue #6680 using post-migration resolved values; the v2–v10 slice
# is byte-identical to the original baseline), compressed with zlib and
# base64-encoded so the regression oracle stays self-contained. Regenerate
# against a clean post-migration checkout with:
#   uv run python -c "import json,zlib,base64,hashlib,pathlib; \
#     from scripts.training.train_ppo import _load_expert_training_config_mapping as L; \
#     b=pathlib.Path('configs/training/ppo'); \
#     d={p.name:L(b/p) for p in sorted(b.glob('expert_ppo_issue_576_br06_v*.yaml'))}; \
#     v11=b/'expert_ppo_issue_708_br06_v11_predictive_foresight_success_priority_from_scratch.yaml'; \
#     d[v11.name]=L(v11); \
#     blob=json.dumps(d,sort_keys=True,separators=(',',':')).encode(); \
#     print(hashlib.sha256(blob).hexdigest()); \
#     print(base64.b64encode(zlib.compress(blob,9)).decode())"
_ISSUE_576_BR06_PRECHANGE_BASELINE_B64 = (
    "eNrtXdtyo7oS/ReejYeLb8l3nLepKZUMMtYOIEqAE++p/PtZEtjGMxBf4sxOPD2V1MQgpFbTvXot"
    "EPinI14KoStWFIrJsqwFm85nbKm9Gdv4Hiu0iGVUyY1gK6VFKZN1xco6ikRZYqdUWlZbVqhURlvG"
    "c55uS1myUqQiqsZbnqXO409nKcqKRWsRPRVK5hXLRKVl5Dw6u440r4QzciKVb4RORB4Jc1ik0lSW"
    "UuXN/kdv7E1HTpHiA6/Zs8xj9ew8hp7njY57QsuH15Ej8g1b8ahSesuenrlOStOrFvgz7mx4FmZS"
    "5dGIzqPrT8eeMSnDvCt89sYYPJUbmSf2k+fjcy64Zhn8hk2haV9olcBN+OyPfZiVKVWtc7sBx/gL"
    "bKp4kvOqGSQwx1RCZxKuY0uV12gYeHarzISqzcgzM3JVRUzL8sketHjF9NqJ5DzDjB2NtoLB3CIV"
    "pnO2CZ3WBwpO1TIWdooJ/kKzfCUTO2ORY3hmfKyWCqNVuhaY9prnuUhxxHdHLWFylOJwTE+gG5w8"
    "npf2fGVLmYvY+TFy1taLOB12SnCBSutmkt44GDl1KZhIFFtpa24zyrOMq3VzCEyVeZTWsWDWQpkz"
    "jCv0pvVUc0BnE8tUbOZdqijnGwaj6qhqLCzZmiNed3abwI3M5Js+eiM6Fhtpgs6JitoZaCNyvkwx"
    "2bf6WWmF+I6U8bfSsCNdsXaWfhMepw9KRZ5YtwwesEbW/QsXlJUoMK/FQLOMvzCObDKx7c+GGsGN"
    "KZOYVnc/wvhl26axiNkmYKs6TYc8Y7MglnC3zVxvPB9oqJFfiFMWV21Y2KDrhCPH/me25JFJz3h/"
    "zuxM8qROMU5ZCHMSrDvN9lTa4dvN1mnVtjDnMparldCYv0RuxRqmmIwoZdYZ0PRgQwYgiMhTdYE+"
    "jPWrTiMgGrdTaewxyWfiWUVRXWDKWxuy7T6k3IandRu2PzufmChkqWwahjDSOPQ34ESrXMABSLLd"
    "3HvbbZDOCh2teFqizVqlMTOOLZHPHKhsEveHwRpRYNtaxHUK478bcwTQsA0c37P/Xn/A6pXgVa2R"
    "pC+VtpgJB9pMbBLM6WnRwdBYq8KMv4NfQF8DNAccCYPRbIKfH+2uJ6Gxh5XyX+OR79NROAqNyU0+"
    "r2UcixxBlZmdfrAY4dfYmdcZMnGDrQ6vK8WqNU5asi5qm/yNm5poRkFra9tNqtmh+1xUjOsIKfo9"
    "mM5G+IXdZrg14k4XHBBnnbLkVbS2EwSmo6ETpbKAi/Jk5yNh816sbG3zkMcpQjlHhWkdOR9PhWuK"
    "ngkeFa3R7QThDa/DhKfUHhaYUsBRCzMMBFvFIW0wwzoTRyn+q1NC5k8zhrQDWhSmfLYdxSzwgpkX"
    "epP/efNp4HmY/i649pnhjMff8LMPum9RystSRkBvFBWTMiovDeTnkSzRX9gwgk5PJUe9shX1pymd"
    "Kwm4BpRj8snWoPEW+G3aHzY1FmJbt2y3wwJLKqAneAbiJ7EQOh317c0QBnXWosVuf6QVogAjFe3R"
    "k4G9vUfHSulnvu07dLcrtXSlu8cCTtO1Oek7o4db7IYOuhNbCx4b5NhXkEOrTk8ZmNVgF4YiVPxp"
    "cL8WPH1WGigTqxpV8HdXH824aiKgFFFTqg/2GOxsYxRZHY6C0B/N5/PRw0Mw8sNwjkyqVAW8NvSn"
    "hamwgSkD7AiqpzZcKpGXSi8V6sSeTyA2lhZ0j0t1i+uO5bcu+K1rgt8FIrgHRHD3iOA2ee7uAMDd"
    "A8A/asna4tJkkYuE6u8Dndtyqf4xhz62la5cOSZ/E0us9uZg21K7nvkf/R1V2UOPzp7kujukchsK"
    "iD3taEN2W3LoHsihC3LYQQ0QOHNecIIByntiVS9hfWRJ5LBECI7wY8/4r2LywYVM/iTVO8WQA2LI"
    "vW77O0mZ711NpKYtPtWYVmo37lHrdXR/pOsU0zoGhRPE6Yas4sK64k8/pK4M1gktMI4E+AbXV4aj"
    "Pn6H9aAPye0puwTGOzSQEJ0QnRCdEH0IH06A+4AwvTvMd4888h/Cf8flbuObawtCyXNzMQT+eiJ+"
    "T9Xgr64G/l8A8H35/l9Q9z7Unv5Rnt544l043XRxU4oe3glFp9uSBMpdUCaOfiMID4mjn7jeH15E"
    "08Mb0PTw9MX3d9H0iUXRvLmPyVcCBRyBmgFh9CddfnNu/Ti9Pid47/qcyZXrcybHy3MmtDyH6uAt"
    "VoRcWQaD6fllsK9kPtxlyVycqJZvISct9aClHrTU466WekzcfboPMr99/qP59dyviyL91K+BG7eJ"
    "astEj2jpwdCrV2VMGGqMjJsi4wfe05/hglOigkQFiQq+kwrOrmaC4e/kzp+G/exu9oW53eQkt/sF"
    "/ojQtd2VqkaG7WMjmDQRQ3SP6N6N6V7/hb1wdiW7sw86XEuHpr0L/+n6GJEienyNHl+jx9foYuUn"
    "v1jZX8CI1RJvJd56V5cpp+7Ao1/nP3l286fOPu7hsdnbD+RCwyd5Z33TnfL0d79nwruSp/u/vGdi"
    "YT8SUSeiTkSdiDoR9SuI+nkVjYg7EXci7ndF3Gf9b4HYvbHhkPiXvEJi9mFvkGjt6bxAYvYJmD6z"
    "b/QxSEucnzg/cX7i/MT5ifN/iYdvLi1wJAFIApAEIAnw5STAyMKhazHsumfW5v1Q+ZJaEkcX/0kI"
    "/M1CILEFMZZmeBsFb3T4aVWDH5whG4LJO2VDgxgkHEg4fM2bBRcVQhIMJBhIMNyVYJj3c/6X1EX+"
    "v1M3zG+uGxqz3hQQ8w+9h3ARXNKtBFIQpCBIQZCCIAVx57cerqyLJChIUJCgIEHx5QXF++9ILM77"
    "YjBSEvSFl7QoiRYlkTIgZfDZ7y2cWdJIBJAIIBFwVyJg8fYypE7mX0L/F3/yyywXH3of4eGs6yVE"
    "+on00+0Dun1Atw9IJNyrSLisEpJWIK1AWuGutMLDWTcMrpQMDx9/x6BHOzx8Iu1Ai5BIRZCKIBVB"
    "KoJUxJ0vQrq2MJKoIFFBooJExR2IimvXIc29RQsbvn/eXVtDbDLUMW2gkKQFSQtalUSrkkgqXC4V"
    "prDSeJS1HJQj/bclxkGzXMADyLLd5HvbbZDPRlzYr6cZ1h2/Mcm2/WnuV4qq/GZqhR0+4wB2bnKd"
    "mQmhYuy539vK5g4UCKvWiItkXdTVsBi5SUX9gpqERASJiDsQEUhcy/1NNrqHbOwXCPtDzpMEbUMr"
    "BX7p/+IVSu+6wfD6f4Fj+Bg="
)
_ISSUE_576_BR06_PRECHANGE_BASELINE_SHA256 = (
    "5d51cc77d1a4b7946bc8de53e7fd8c4eedcf3c39d3e53be692c5163379fbb719"
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
    "expert_ppo_issue_708_br06_v11_predictive_foresight_success_priority_from_scratch.yaml": (
        "ppo_expert_issue_708_br06_v11_predictive_foresight_success_priority_from_scratch",
        "auto_throughput",
        "subproc",
        (123, 231, 777, 992, 1337),
        30000000,
        "route_completion_v3",
        "predictive_proxy_selected_v2_full",
        None,  # from-scratch run: no resume_model_id
        "issue-708-ppo-from-scratch",
        [
            "issue-708",
            "ppo",
            "from-scratch",
            "predictive-foresight",
            "success-priority-reward",
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
    """Every variant inherits the base (directly or via predictive sub-base) and
    no longer carries the deprecated evaluation.frequency_episodes field.

    Issue #6680: predictive variants (v5-v10 and v11) now chain through
    expert_ppo_issue_576_br06_predictive_sub_base.yaml which itself inherits
    from expert_ppo_issue_576_br06_base.yaml. Non-predictive variants still
    inherit the family base directly.
    """
    base_path = (_ISSUE_576_BR06_FAMILY_DIR / _ISSUE_576_BR06_BASE_NAME).resolve()
    base_text = base_path.read_text(encoding="utf-8")
    base_yaml = yaml.safe_load(base_text)
    # The family base must not self-inherit or carry the deprecated field.
    assert "base_config" not in base_yaml
    assert "frequency_episodes" not in base_yaml.get("evaluation", {})

    sub_base_path = (
        _ISSUE_576_BR06_FAMILY_DIR / _ISSUE_576_BR06_PREDICTIVE_SUB_BASE_NAME
    ).resolve()
    sub_base_yaml = yaml.safe_load(sub_base_path.read_text(encoding="utf-8"))
    # The predictive sub-base must chain on the family base (one level up) and
    # must not carry the deprecated field.
    assert sub_base_yaml.get("base_config") == _ISSUE_576_BR06_BASE_NAME
    assert "frequency_episodes" not in sub_base_yaml.get("evaluation", {})

    for variant in _ISSUE_576_BR06_VARIANTS:
        variant_path = (_ISSUE_576_BR06_FAMILY_DIR / variant).resolve()
        variant_yaml = yaml.safe_load(variant_path.read_text(encoding="utf-8"))
        # Predictive variants chain through the sub-base; non-predictive
        # variants still inherit the family base directly.
        if variant in _ISSUE_576_BR06_PREDICTIVE_VARIANTS:
            assert variant_yaml["base_config"] == _ISSUE_576_BR06_PREDICTIVE_SUB_BASE_NAME
        else:
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
    """A missing/nonexistent base_config must fail closed with a ValueError."""
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
    with pytest.raises(ValueError, match="does not exist"):
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


# Issue #6681: the three issue_791 seed-replica groups were migrated to shared
# base configs via the existing base_config resolver. The constants below pin
# that contract and the frozen pre-change resolved-config baseline.
_ISSUE_791_ABLATE_DIR = Path("configs/training/ppo/ablations")
_ISSUE_791_BASELINE_PATH = Path("tests/integration/_baseline_issue_791_seed_replicas_resolved.json")

# Every migrated issue_791 seed-replica variant and its shared base config.
# Frozen in the pre-change baseline at origin/main c8a10c04; the resolver must
# reconstruct each variant's mapping byte-identically from the base + overrides.
_ISSUE_791_MIGRATED_VARIANTS = {
    # Group (a): all_scenarios_10m_env22_large_capacity leader + seed replicas.
    "expert_ppo_issue_791_all_scenarios_10m_env22_large_capacity.yaml": (
        "expert_ppo_issue_791_all_scenarios_10m_env22_large_capacity_base.yaml"
    ),
    "expert_ppo_issue_791_all_scenarios_10m_env22_large_capacity_seed1337.yaml": (
        "expert_ppo_issue_791_all_scenarios_10m_env22_large_capacity_base.yaml"
    ),
    "expert_ppo_issue_791_all_scenarios_10m_env22_large_capacity_seed231.yaml": (
        "expert_ppo_issue_791_all_scenarios_10m_env22_large_capacity_base.yaml"
    ),
    # Group (b): reward_curriculum_promotion_10m eval_aligned_large_capacity
    # leader + seed231/seed1337 variants and their _fixed counterparts.
    "expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity.yaml": (
        "expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity_base.yaml"
    ),
    "expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity_seed1337.yaml": (
        "expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity_base.yaml"
    ),
    "expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity_seed1337_fixed.yaml": (
        "expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity_base.yaml"
    ),
    "expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity_seed231.yaml": (
        "expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity_base.yaml"
    ),
    "expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity_seed231_fixed.yaml": (
        "expert_ppo_issue_791_reward_curriculum_promotion_10m_env22_eval_aligned_large_capacity_base.yaml"
    ),
    # Group (c): reward_curriculum_promotion_3m eval_aligned seed replicas.
    "expert_ppo_issue_791_reward_curriculum_promotion_3m_env22_eval_aligned_seed1337.yaml": (
        "expert_ppo_issue_791_reward_curriculum_promotion_3m_env22_eval_aligned_base.yaml"
    ),
    "expert_ppo_issue_791_reward_curriculum_promotion_3m_env22_eval_aligned_seed231.yaml": (
        "expert_ppo_issue_791_reward_curriculum_promotion_3m_env22_eval_aligned_base.yaml"
    ),
    "expert_ppo_issue_791_reward_curriculum_promotion_3m_env22_eval_aligned_seed992.yaml": (
        "expert_ppo_issue_791_reward_curriculum_promotion_3m_env22_eval_aligned_base.yaml"
    ),
}

# Known silent divergence pinned by the migration: only the all_scenarios leader
# carries env_factory_kwargs.peds_have_obstacle_forces; the seed replicas must
# keep its absence. Every resolved mapping stays byte-identical to origin/main.
_ISSUE_791_ALL_SCENARIOS_PEDS_OVERRIDE = {
    "expert_ppo_issue_791_all_scenarios_10m_env22_large_capacity.yaml": True,
    "expert_ppo_issue_791_all_scenarios_10m_env22_large_capacity_seed1337.yaml": False,
    "expert_ppo_issue_791_all_scenarios_10m_env22_large_capacity_seed231.yaml": False,
}

# Not migrated, but inherits the group (a) leader via base_config; its resolved
# mapping must stay unchanged too.
_ISSUE_791_BEST_CKPT_GUARD = (
    "expert_ppo_issue_791_best_ckpt_all_scenarios_horizon500_20m_env22.yaml"
)


def _issue_791_fingerprint(config_path: Path) -> str:
    """Return the canonical resolved-config fingerprint for ``config_path``."""
    resolved = _load_expert_training_config_mapping(config_path)
    canonical = json.dumps(resolved, default=str, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def _issue_791_baseline() -> dict:
    """Load the frozen pre-change resolved-config baseline for the migration."""
    assert _ISSUE_791_BASELINE_PATH.exists(), (
        "Pre-change baseline missing; re-run capture before changing configs"
    )
    baseline = json.loads(_ISSUE_791_BASELINE_PATH.read_text(encoding="utf-8"))
    assert baseline["schema_version"] == "resolved-config-fingerprint.v1"
    return baseline


@pytest.mark.parametrize("variant", sorted(_ISSUE_791_MIGRATED_VARIANTS))
def test_issue_791_seed_replicas_resolve_to_prechange_values(variant: str) -> None:
    """Each migrated issue_791 seed-replica variant matches its pre-refactor mapping.

    The base_config deep-merge must reconstruct the exact pre-change resolved
    mapping for every migrated variant, including the preserved
    peds_have_obstacle_forces divergence.
    """
    path = (_ISSUE_791_ABLATE_DIR / variant).resolve()
    baseline = _issue_791_baseline()

    actual_fingerprint = _issue_791_fingerprint(path)
    assert actual_fingerprint == baseline["variants"][variant], (
        f"Resolved config {variant} differs from the baseline at {baseline['source_revision']}."
    )
    # Exercise the full construction path so the loader does not diverge from the
    # resolver output after inheritance.
    config = load_expert_training_config(path)
    assert config.policy_id


@pytest.mark.parametrize("variant", sorted(_ISSUE_791_ALL_SCENARIOS_PEDS_OVERRIDE))
def test_issue_791_peds_have_obstacle_forces_divergence_preserved(variant: str) -> None:
    """The all_scenarios leader keeps env_factory_kwargs.peds_have_obstacle_forces.

    The replicas keep its absence; the shared env_overrides value stays common.
    This pins the silent divergence the migration must preserve exactly.
    """
    resolved = _load_expert_training_config_mapping((_ISSUE_791_ABLATE_DIR / variant).resolve())
    expects_override = _ISSUE_791_ALL_SCENARIOS_PEDS_OVERRIDE[variant]

    assert resolved["env_overrides"]["peds_have_obstacle_forces"] is True
    if expects_override:
        assert resolved["env_factory_kwargs"]["peds_have_obstacle_forces"] is True
    else:
        assert "peds_have_obstacle_forces" not in resolved["env_factory_kwargs"]


def test_issue_791_bases_inheritance_and_no_self_reference() -> None:
    """Every migrated variant inherits its shared base and the bases stay lean."""
    for variant, base_name in _ISSUE_791_MIGRATED_VARIANTS.items():
        variant_path = (_ISSUE_791_ABLATE_DIR / variant).resolve()
        variant_yaml = yaml.safe_load(variant_path.read_text(encoding="utf-8"))
        assert variant_yaml["base_config"] == base_name

        base_path = (_ISSUE_791_ABLATE_DIR / base_name).resolve()
        base_yaml = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        # A base must not self-inherit and carries no launch identity.
        assert "base_config" not in base_yaml
        assert "policy_id" not in base_yaml


def test_issue_791_best_ckpt_inheriting_leader_guard_unchanged() -> None:
    """A config inheriting the group (a) leader must resolve unchanged too.

    expert_ppo_issue_791_best_ckpt_all_scenarios_horizon500_20m_env22.yaml uses
    the all_scenarios leader as its base_config, so converting the leader to
    inherit from a shared base must leave its resolved mapping identical.
    """
    baseline = _issue_791_baseline()
    path = (_ISSUE_791_ABLATE_DIR / _ISSUE_791_BEST_CKPT_GUARD).resolve()
    assert (
        _issue_791_fingerprint(path) == baseline["non_migrated_guards"][_ISSUE_791_BEST_CKPT_GUARD]
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


_ISSUE_739_BASE_NAME = "expert_ppo_issue_739_base.yaml"

_ISSUE_739_VARIANT_PATHS = [
    "configs/training/ppo/ablations/expert_ppo_issue_739_stage1_baseline.yaml",
    "configs/training/ppo/ablations/expert_ppo_issue_739_stage1_obs_grid_goal.yaml",
    "configs/training/ppo/ablations/expert_ppo_issue_739_stage1_obs_selective.yaml",
    "configs/training/ppo/ablations/expert_ppo_issue_739_stage1_reward_core.yaml",
    "configs/training/ppo/ablations/expert_ppo_issue_739_stage1_reward_tuned.yaml",
    "configs/training/ppo/ablations/expert_ppo_issue_739_stage2_opt_scale.yaml",
    "configs/training/ppo/ablations/expert_ppo_issue_739_stage2_sampling.yaml",
    "configs/training/ppo/expert_ppo_issue_739_12m_baseline_retrain.yaml",
]


def test_issue_6682_issue_739_base_config_inheritance_equivalence() -> None:
    """All issue-739 family variants must match their pre-refactor resolved fingerprints."""
    baseline_path = Path("tests/integration/_baseline_issue_6682_resolved.json").resolve()
    assert baseline_path.exists(), (
        "Pre-change baseline missing; re-run capture before changing configs"
    )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert baseline["schema_version"] == "resolved-config-fingerprint.v1"
    assert isinstance(baseline["source_revision"], str)
    fingerprints = baseline["variants"]
    assert isinstance(fingerprints, dict)

    ablate_dir = Path("configs/training/ppo/ablations")
    stage_variant_paths = sorted(ablate_dir.glob("expert_ppo_issue_739_stage*.yaml"))
    assert len(stage_variant_paths) == 7
    variant_paths = [str(path) for path in stage_variant_paths]
    variant_paths.append("configs/training/ppo/expert_ppo_issue_739_12m_baseline_retrain.yaml")
    assert set(variant_paths) == set(fingerprints)

    base_path = ablate_dir / _ISSUE_739_BASE_NAME
    assert base_path.exists()
    base_yaml = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    assert "base_config" not in base_yaml

    for rel_path in variant_paths:
        resolved = _load_expert_training_config_mapping(Path(rel_path))
        canonical = json.dumps(resolved, default=str, sort_keys=True, separators=(",", ":"))
        actual_fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
        assert actual_fingerprint == fingerprints[rel_path], (
            f"Resolved config {rel_path} differs from the baseline at "
            f"{baseline['source_revision']}."
        )


@pytest.mark.parametrize("rel_path", _ISSUE_739_VARIANT_PATHS)
def test_issue_6682_issue_739_variant_equivalence(rel_path: str) -> None:
    """Parametrized equivalence test covering every migrated issue-739 family variant."""
    baseline_path = Path("tests/integration/_baseline_issue_6682_resolved.json").resolve()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    expected_fingerprint = baseline["variants"][rel_path]

    config_path = Path(rel_path)
    resolved = _load_expert_training_config_mapping(config_path)
    canonical = json.dumps(resolved, default=str, sort_keys=True, separators=(",", ":"))
    actual_fingerprint = hashlib.sha256(canonical.encode()).hexdigest()

    assert actual_fingerprint == expected_fingerprint, (
        f"Variant {rel_path} resolved fingerprint {actual_fingerprint} "
        f"does not match baseline {expected_fingerprint}"
    )

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base_path = (config_path.parent / raw["base_config"]).resolve()
    assert base_path.name == _ISSUE_739_BASE_NAME
    assert base_path.exists()

    config = load_expert_training_config(config_path)
    assert config.policy_id


_ISSUE_6683_BASELINE_PATH = Path("tests/integration/_baseline_issue_6683_resolved.json")
_ISSUE_6683_MATRIX_PATH = "configs/training/ppo/feature_extractor_candidates_12m_issue193.yaml"

# The eight PPO training configs that carried the deprecated, ignored
# evaluation.frequency_episodes field on origin/main before issue #6683 removed
# it, plus the candidate matrix that inherits feature_extractor_sweep_base.yaml.
_ISSUE_6683_VARIANT_PATHS = [
    "configs/training/ppo/ablations/issue_4018_density_curriculum_smoke.yaml",
    "configs/training/ppo/ablations/issue_4018_fixed_density_smoke.yaml",
    _ISSUE_6683_MATRIX_PATH,
    "configs/training/ppo/feature_extractor_sweep_base.yaml",
    "configs/training/ppo/issue_4014_ppo_lstm_recurrent_smoke.yaml",
    "configs/training/ppo/issue_4014_ppo_mamba_smoke.yaml",
    "configs/training/ppo/issue_4014_ppo_mamba_smoke_matched.yaml",
    "configs/training/ppo/issue_4014_ppo_smoke_matched.yaml",
    "configs/training/ppo/issue_4014_recurrent_ppo_lstm_smoke_matched.yaml",
]

# Lineages resolving through feature_extractor_sweep_base.yaml, which carried
# frequency_episodes: 10 before issue #6683. Removal falls back to the loader
# default 0 there — the single intentional resolved-value change, mirroring the
# #6513 precedent (the field was always ignored in favor of step_schedule).
_ISSUE_6683_SWEEP_BASE_LINEAGE = frozenset(
    {
        "configs/training/ppo/feature_extractor_sweep_base.yaml",
        _ISSUE_6683_MATRIX_PATH,
    }
)

_ISSUE_6683_EXPERT_VARIANT_PATHS = [
    path for path in _ISSUE_6683_VARIANT_PATHS if path != _ISSUE_6683_MATRIX_PATH
]


def _issue_6683_prechange_baseline() -> dict:
    """Load and integrity-check the frozen pre-change resolved-config baseline."""
    baseline_path = _ISSUE_6683_BASELINE_PATH.resolve()
    assert baseline_path.exists(), (
        "Pre-change baseline missing; re-run capture before changing configs"
    )
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert baseline["schema_version"] == "resolved-config-mapping.v1"
    assert isinstance(baseline["source_revision"], str)
    assert set(baseline["variants"]) == set(_ISSUE_6683_VARIANT_PATHS)
    return baseline


def _issue_6683_resolved_mapping(rel_path: str) -> dict:
    """Resolve a variant mapping the way its consumer does.

    Expert training configs load directly. The candidate matrix resolves through
    its base_config pointer relative to the repository root, mirroring
    scripts/training/fixed_feature_extractor_candidates.py.
    """
    if rel_path != _ISSUE_6683_MATRIX_PATH:
        return _load_expert_training_config_mapping(Path(rel_path))
    matrix = yaml.safe_load(Path(rel_path).read_text(encoding="utf-8"))
    return _load_expert_training_config_mapping(Path(matrix["base_config"]))


def test_issue_6683_baseline_records_removed_field_values() -> None:
    """The frozen baseline records the removed values: seven 0s and two 10s."""
    baseline = _issue_6683_prechange_baseline()
    recorded = baseline["pre_change_frequency_episodes"]
    assert set(recorded) == set(_ISSUE_6683_VARIANT_PATHS)
    for rel_path in _ISSUE_6683_SWEEP_BASE_LINEAGE:
        assert recorded[rel_path] == 10
    for rel_path in set(_ISSUE_6683_VARIANT_PATHS) - _ISSUE_6683_SWEEP_BASE_LINEAGE:
        assert recorded[rel_path] == 0


@pytest.mark.parametrize("rel_path", _ISSUE_6683_VARIANT_PATHS)
def test_issue_6683_frequency_episodes_drop_preserves_resolution(rel_path: str) -> None:
    """Resolved configs are identical except for the intentionally dropped field."""
    expected_mapping = _issue_6683_prechange_baseline()["variants"][rel_path]

    resolved = _issue_6683_resolved_mapping(rel_path)
    # The deprecated field is gone from the config side in every lineage.
    assert "frequency_episodes" not in resolved.get("evaluation", {})

    assert _strip_frequency_episodes(resolved) == _strip_frequency_episodes(expected_mapping), (
        f"Resolved config {rel_path} differs from the baseline beyond the "
        f"dropped evaluation.frequency_episodes field."
    )


@pytest.mark.parametrize("rel_path", _ISSUE_6683_EXPERT_VARIANT_PATHS)
def test_issue_6683_expert_configs_keep_step_schedule_cadence(rel_path: str) -> None:
    """All eight expert configs load with step_schedule as the cadence source."""
    config = load_expert_training_config(Path(rel_path))
    assert config.policy_id
    assert config.evaluation.step_schedule
    # The deprecated field resolves to the loader default everywhere now; for
    # the sweep-base lineage that is the intentional ignored-field drop 10 -> 0.
    assert config.evaluation.frequency_episodes == 0


def test_issue_6683_candidate_matrix_lineage_keeps_runner_semantics() -> None:
    """The inheriting matrix keeps its pointer, overrides, and rebuilt schedule."""
    baseline = _issue_6683_prechange_baseline()
    expected_overrides = baseline["matrix_overrides"][_ISSUE_6683_MATRIX_PATH]

    matrix = yaml.safe_load(Path(_ISSUE_6683_MATRIX_PATH).read_text(encoding="utf-8"))
    for key, value in expected_overrides.items():
        assert matrix[key] == value

    base_config = load_expert_training_config(Path(matrix["base_config"]))
    assert base_config.evaluation.step_schedule
    # Mirror the _configure_candidate EvaluationSchedule rebuild: cadence comes
    # from the matrix eval_every and the deprecated field falls to default 0.
    rebuilt = EvaluationSchedule(
        frequency_episodes=base_config.evaluation.frequency_episodes,
        evaluation_episodes=int(matrix["eval_episodes"]),
        hold_out_scenarios=base_config.evaluation.hold_out_scenarios,
        step_schedule=((None, int(matrix["eval_every"])),),
        randomize_seeds=False,
        scenario_config=base_config.evaluation.scenario_config,
    )
    assert rebuilt.frequency_episodes == 0
    assert rebuilt.evaluation_episodes == 5
    assert rebuilt.hold_out_scenarios == ()
    assert rebuilt.step_schedule == ((None, 48000),)


_ISSUE_6904_CONFIG_PATHS = [
    "configs/training/benchmark_orca_classic_cross_trap_subset.yaml",
    "configs/training/benchmark_orca_classic_crossing_subset.yaml",
    "configs/training/lidar/lidar_ppo_mlp_smoke_issue_1662.yaml",
    "configs/training/ppo_imitation/expert_ppo.yaml",
    "configs/training/ppo_imitation/expert_ppo_issue_403_grid.yaml",
    "configs/training/ppo_imitation/expert_ppo_issue_403_grid_diffdrive.yaml",
    "configs/training/ppo_imitation/expert_ppo_issue_403_grid_diffdrive_reverse.yaml",
    "configs/training/ppo_imitation/expert_ppo_issue_403_grid_diffdrive_reverse_no_holdout_15m.yaml",
    "configs/training/ppo_imitation/expert_ppo_issue_403_grid_diffdrive_reverse_no_holdout_15m_random.yaml",
    "configs/training/ppo_imitation/expert_ppo_issue_403_grid_radius_0_6.yaml",
]


def _issue_6904_baseline() -> dict:
    """Load and validate the frozen pre-change resolved-config baseline."""
    baseline_path = Path("tests/integration/_baseline_issue_6904_resolved.json").resolve()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert baseline["schema_version"] == "resolved-config-fingerprint.v1"
    assert baseline["field_removed"] == "evaluation.frequency_episodes"
    assert set(baseline["variants"]) == set(_ISSUE_6904_CONFIG_PATHS)
    assert set(baseline["pre_change_frequency_episodes"]) == set(_ISSUE_6904_CONFIG_PATHS)
    return baseline


def test_issue_6904_baseline_records_removed_frequency_values() -> None:
    """The baseline records the ignored values removed from all ten configs."""
    baseline = _issue_6904_baseline()
    values = baseline["pre_change_frequency_episodes"]
    assert values["configs/training/benchmark_orca_classic_cross_trap_subset.yaml"] == 1
    assert values["configs/training/benchmark_orca_classic_crossing_subset.yaml"] == 1
    assert {values[path] for path in _ISSUE_6904_CONFIG_PATHS[2:]} == {10}


@pytest.mark.parametrize("rel_path", _ISSUE_6904_CONFIG_PATHS)
def test_issue_6904_frequency_episodes_drop_preserves_resolved_behavior(rel_path: str) -> None:
    """Removing the ignored field preserves every other resolved config value."""
    baseline = _issue_6904_baseline()
    config_path = Path(rel_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert raw["evaluation"].get("step_schedule")

    resolved = _load_expert_training_config_mapping(config_path)
    assert "frequency_episodes" not in resolved.get("evaluation", {})
    canonical = json.dumps(resolved, default=str, sort_keys=True, separators=(",", ":"))
    actual_fingerprint = hashlib.sha256(canonical.encode()).hexdigest()
    assert actual_fingerprint == baseline["variants"][rel_path]

    config = load_expert_training_config(config_path)
    assert config.evaluation.step_schedule
    assert config.evaluation.frequency_episodes == 0


# Issue #6484: the issue_791 baseline_promotion short-budget pair was migrated to
# inherit shared settings from a single base config. The constants below pin
# that contract and the frozen pre-change resolved-config baseline.
_ISSUE_6484_BASELINE_PROMOTION_DIR = Path("configs/training/ppo/ablations")
_ISSUE_6484_BASELINE_PROMOTION_BASE_NAME = "expert_ppo_issue_791_baseline_promotion_base.yaml"
_ISSUE_6484_BASELINE_PROMOTION_BASELINE_PATH = Path(
    "tests/integration/_baseline_issue_6484_baseline_promotion_resolved.json"
)
_ISSUE_6484_BASELINE_PROMOTION_VARIANTS = [
    "expert_ppo_issue_791_baseline_promotion_128k.yaml",
    "expert_ppo_issue_791_baseline_promotion_256k.yaml",
]


def _issue_6484_baseline_promotion_baseline() -> dict:
    """Load and integrity-check the frozen pre-change resolved-config baseline."""
    assert _ISSUE_6484_BASELINE_PROMOTION_BASELINE_PATH.exists(), (
        "Pre-change baseline missing; re-run capture before changing configs"
    )
    baseline = json.loads(_ISSUE_6484_BASELINE_PROMOTION_BASELINE_PATH.read_text(encoding="utf-8"))
    assert baseline["schema_version"] == "resolved-config-fingerprint.v1"
    return baseline


def _issue_6484_baseline_promotion_fingerprint(config_path: Path) -> str:
    """Return the canonical resolved-config fingerprint for ``config_path``."""
    resolved = _load_expert_training_config_mapping(config_path)
    canonical = json.dumps(resolved, default=str, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


@pytest.mark.parametrize("variant", _ISSUE_6484_BASELINE_PROMOTION_VARIANTS)
def test_issue_6484_baseline_promotion_resolves_to_prechange_values(variant: str) -> None:
    """Each migrated baseline_promotion variant matches its pre-refactor mapping.

    The base_config deep-merge must reconstruct the exact pre-change resolved
    mapping for every migrated variant.
    """
    path = (_ISSUE_6484_BASELINE_PROMOTION_DIR / variant).resolve()
    baseline = _issue_6484_baseline_promotion_baseline()

    actual_fingerprint = _issue_6484_baseline_promotion_fingerprint(path)
    assert actual_fingerprint == baseline["variants"][variant], (
        f"Resolved config {variant} differs from the baseline at {baseline['source_revision']}."
    )
    config = load_expert_training_config(path)
    assert config.policy_id


@pytest.mark.parametrize("variant", _ISSUE_6484_BASELINE_PROMOTION_VARIANTS)
def test_issue_6484_baseline_promotion_loads_through_loader(variant: str) -> None:
    """Both migrated variants load through load_expert_training_config."""
    path = (_ISSUE_6484_BASELINE_PROMOTION_DIR / variant).resolve()
    config = load_expert_training_config(path)
    assert config.policy_id
    assert config.total_timesteps > 0
    assert config.evaluation.step_schedule


def test_issue_6484_baseline_promotion_base_inheritance_and_no_launch_identity() -> None:
    """Every variant inherits its shared base and the base stays lean."""
    for variant in _ISSUE_6484_BASELINE_PROMOTION_VARIANTS:
        variant_path = (_ISSUE_6484_BASELINE_PROMOTION_DIR / variant).resolve()
        variant_yaml = yaml.safe_load(variant_path.read_text(encoding="utf-8"))
        assert variant_yaml["base_config"] == _ISSUE_6484_BASELINE_PROMOTION_BASE_NAME

        base_path = (
            _ISSUE_6484_BASELINE_PROMOTION_DIR / _ISSUE_6484_BASELINE_PROMOTION_BASE_NAME
        ).resolve()
        base_yaml = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        # A base must not self-inherit and carries no launch identity.
        assert "base_config" not in base_yaml
        assert "policy_id" not in base_yaml
        assert "job_type" not in base_yaml.get("tracking", {}).get("wandb", {})
        assert "tags" not in base_yaml.get("tracking", {}).get("wandb", {})
        assert set(variant_yaml) == {
            "base_config",
            "policy_id",
            "total_timesteps",
            "evaluation",
            "tracking",
        }
        assert set(variant_yaml["evaluation"]) == {"step_schedule"}
        assert set(variant_yaml["tracking"]) == {"wandb"}
        assert set(variant_yaml["tracking"]["wandb"]) == {"job_type", "tags"}


def test_issue_6484_baseline_promotion_variants_keep_distinct_overrides() -> None:
    """The two inherited variants retain different budgets and evaluation cadence."""
    short = load_expert_training_config(
        (
            _ISSUE_6484_BASELINE_PROMOTION_DIR / "expert_ppo_issue_791_baseline_promotion_128k.yaml"
        ).resolve()
    )
    long = load_expert_training_config(
        (
            _ISSUE_6484_BASELINE_PROMOTION_DIR / "expert_ppo_issue_791_baseline_promotion_256k.yaml"
        ).resolve()
    )
    short_mapping = _load_expert_training_config_mapping(
        (
            _ISSUE_6484_BASELINE_PROMOTION_DIR / "expert_ppo_issue_791_baseline_promotion_128k.yaml"
        ).resolve()
    )
    long_mapping = _load_expert_training_config_mapping(
        (
            _ISSUE_6484_BASELINE_PROMOTION_DIR / "expert_ppo_issue_791_baseline_promotion_256k.yaml"
        ).resolve()
    )

    assert short.total_timesteps == 131072
    assert long.total_timesteps == 262144
    assert short.evaluation.step_schedule == ((None, 65536),)
    assert long.evaluation.step_schedule == ((None, 32768),)
    assert short.policy_id != long.policy_id
    assert (
        short_mapping["tracking"]["wandb"]["job_type"]
        != long_mapping["tracking"]["wandb"]["job_type"]
    )
    assert short_mapping["tracking"]["wandb"]["tags"] != long_mapping["tracking"]["wandb"]["tags"]


# Issue #6484: the issue-791 reward-curriculum short-budget pair shares one
# base config while preserving frozen resolved mappings and explicit launch identity.
_ISSUE_6484_REWARD_CURRICULUM_DIR = Path("configs/training/ppo/ablations")
_ISSUE_6484_REWARD_CURRICULUM_BASE_NAME = (
    "expert_ppo_issue_791_reward_curriculum_promotion_base.yaml"
)
_ISSUE_6484_REWARD_CURRICULUM_BASELINE_PATH = Path(
    "tests/integration/_baseline_issue_6484_reward_curriculum_promotion_resolved.json"
)
_ISSUE_6484_REWARD_CURRICULUM_VARIANTS = [
    "expert_ppo_issue_791_reward_curriculum_promotion_128k.yaml",
    "expert_ppo_issue_791_reward_curriculum_promotion_256k.yaml",
]


def _issue_6484_reward_curriculum_baseline() -> dict:
    """Load and integrity-check the frozen pre-change config baseline."""
    assert _ISSUE_6484_REWARD_CURRICULUM_BASELINE_PATH.exists(), (
        "Pre-change reward-curriculum baseline missing; re-run capture before changing configs"
    )
    baseline = json.loads(_ISSUE_6484_REWARD_CURRICULUM_BASELINE_PATH.read_text(encoding="utf-8"))
    assert baseline["schema_version"] == "resolved-config-fingerprint.v1"
    assert set(baseline["variants"]) == set(_ISSUE_6484_REWARD_CURRICULUM_VARIANTS)
    return baseline


def _issue_6484_reward_curriculum_fingerprint(config_path: Path) -> str:
    """Return the canonical resolved-config fingerprint for one variant."""
    resolved = _load_expert_training_config_mapping(config_path)
    canonical = json.dumps(resolved, default=str, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


@pytest.mark.parametrize("variant", _ISSUE_6484_REWARD_CURRICULUM_VARIANTS)
def test_issue_6484_reward_curriculum_resolves_to_prechange_values(variant: str) -> None:
    """Each inherited reward-curriculum variant matches its pre-refactor mapping."""
    path = (_ISSUE_6484_REWARD_CURRICULUM_DIR / variant).resolve()
    baseline = _issue_6484_reward_curriculum_baseline()

    assert _issue_6484_reward_curriculum_fingerprint(path) == baseline["variants"][variant], (
        f"Resolved config {variant} differs from the baseline at {baseline['source_revision']}."
    )
    config = load_expert_training_config(path)
    assert config.policy_id
    assert config.num_envs == 2
    assert config.evaluation.step_schedule


def test_issue_6484_reward_curriculum_base_reuse_and_explicit_launch_identity() -> None:
    """The variants reuse one lean base and keep launch identity explicit."""
    base_path = (
        _ISSUE_6484_REWARD_CURRICULUM_DIR / _ISSUE_6484_REWARD_CURRICULUM_BASE_NAME
    ).resolve()
    base_yaml = yaml.safe_load(base_path.read_text(encoding="utf-8"))

    assert "base_config" not in base_yaml
    assert "policy_id" not in base_yaml
    assert "num_envs" not in base_yaml
    assert "total_timesteps" not in base_yaml
    assert "step_schedule" not in base_yaml.get("evaluation", {})
    assert "job_type" not in base_yaml["tracking"]["wandb"]
    assert "tags" not in base_yaml["tracking"]["wandb"]

    for variant in _ISSUE_6484_REWARD_CURRICULUM_VARIANTS:
        variant_yaml = yaml.safe_load(
            (_ISSUE_6484_REWARD_CURRICULUM_DIR / variant).read_text(encoding="utf-8")
        )
        assert variant_yaml["base_config"] == _ISSUE_6484_REWARD_CURRICULUM_BASE_NAME
        assert set(variant_yaml) == {
            "base_config",
            "policy_id",
            "num_envs",
            "total_timesteps",
            "evaluation",
            "tracking",
        }
        assert variant_yaml["num_envs"] == 2
        assert set(variant_yaml["evaluation"]) == {"step_schedule"}
        assert set(variant_yaml["tracking"]) == {"wandb"}
        assert set(variant_yaml["tracking"]["wandb"]) == {"job_type", "tags"}


def test_issue_6484_reward_curriculum_variants_keep_distinct_overrides() -> None:
    """The inherited variants retain distinct budgets, cadence, and W&B metadata."""
    short_path = (
        _ISSUE_6484_REWARD_CURRICULUM_DIR
        / "expert_ppo_issue_791_reward_curriculum_promotion_128k.yaml"
    ).resolve()
    long_path = (
        _ISSUE_6484_REWARD_CURRICULUM_DIR
        / "expert_ppo_issue_791_reward_curriculum_promotion_256k.yaml"
    ).resolve()
    short = load_expert_training_config(short_path)
    long = load_expert_training_config(long_path)
    short_mapping = _load_expert_training_config_mapping(short_path)
    long_mapping = _load_expert_training_config_mapping(long_path)

    assert short.num_envs == long.num_envs == 2
    assert short.total_timesteps == 131072
    assert long.total_timesteps == 262144
    assert short.evaluation.step_schedule == ((None, 65536),)
    assert long.evaluation.step_schedule == ((None, 32768),)
    assert short.policy_id != long.policy_id
    assert short_mapping["tracking"]["wandb"]["job_type"] == "expert-ppo-128k-promotion"
    assert long_mapping["tracking"]["wandb"]["job_type"] == "expert-ppo-256k-promotion"
    assert short_mapping["tracking"]["wandb"]["tags"] != long_mapping["tracking"]["wandb"]["tags"]


# Issue #6484: the issue-791 baseline env22 pair shares a dedicated base while
# preserving explicit launch identity and frozen resolved mappings.
_ISSUE_6484_BASELINE_ENV22_DIR = Path("configs/training/ppo/ablations")
_ISSUE_6484_BASELINE_ENV22_BASE_NAME = "expert_ppo_issue_791_baseline_promotion_env22_base.yaml"
_ISSUE_6484_BASELINE_ENV22_BASELINE_PATH = Path(
    "tests/integration/_baseline_issue_6484_baseline_promotion_env22_resolved.json"
)
_ISSUE_6484_BASELINE_ENV22_VARIANTS = [
    "expert_ppo_issue_791_baseline_promotion_1m_env22.yaml",
    "expert_ppo_issue_791_baseline_promotion_3m_env22.yaml",
]


def _issue_6484_baseline_env22_baseline() -> dict:
    """Load and integrity-check the frozen env22 baseline."""
    assert _ISSUE_6484_BASELINE_ENV22_BASELINE_PATH.exists(), (
        "Pre-change env22 baseline missing; re-run capture before changing configs"
    )
    baseline = json.loads(_ISSUE_6484_BASELINE_ENV22_BASELINE_PATH.read_text(encoding="utf-8"))
    assert baseline["schema_version"] == "resolved-config-fingerprint.v1"
    assert set(baseline["variants"]) == set(_ISSUE_6484_BASELINE_ENV22_VARIANTS)
    return baseline


@pytest.mark.parametrize("variant", _ISSUE_6484_BASELINE_ENV22_VARIANTS)
def test_issue_6484_baseline_env22_resolves_to_prechange_values(variant: str) -> None:
    """Each env22 baseline variant matches its frozen pre-refactor mapping."""
    path = (_ISSUE_6484_BASELINE_ENV22_DIR / variant).resolve()
    baseline = _issue_6484_baseline_env22_baseline()

    assert _issue_6484_baseline_promotion_fingerprint(path) == baseline["variants"][variant], (
        f"Resolved config {variant} differs from the baseline at {baseline['source_revision']}."
    )
    config = load_expert_training_config(path)
    assert config.policy_id
    assert config.num_envs == 22


def test_issue_6484_baseline_env22_base_reuse_and_explicit_launch_identity() -> None:
    """The env22 variants reuse one base and keep launch identity explicit."""
    base_path = (_ISSUE_6484_BASELINE_ENV22_DIR / _ISSUE_6484_BASELINE_ENV22_BASE_NAME).resolve()
    base_yaml = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    assert "base_config" not in base_yaml
    assert "policy_id" not in base_yaml
    assert "total_timesteps" not in base_yaml
    assert "step_schedule" not in base_yaml.get("evaluation", {})
    assert "job_type" not in base_yaml["tracking"]["wandb"]
    assert "tags" not in base_yaml["tracking"]["wandb"]

    for variant in _ISSUE_6484_BASELINE_ENV22_VARIANTS:
        variant_yaml = yaml.safe_load(
            (_ISSUE_6484_BASELINE_ENV22_DIR / variant).read_text(encoding="utf-8")
        )
        assert variant_yaml["base_config"] == _ISSUE_6484_BASELINE_ENV22_BASE_NAME
        assert set(variant_yaml) == {
            "base_config",
            "policy_id",
            "num_envs",
            "total_timesteps",
            "evaluation",
            "tracking",
        }
        assert set(variant_yaml["evaluation"]) == {"step_schedule"}
        assert set(variant_yaml["tracking"]) == {"wandb"}
        assert set(variant_yaml["tracking"]["wandb"]) == {"job_type", "tags"}


def test_issue_6484_baseline_env22_variants_keep_distinct_overrides() -> None:
    """The env22 variants retain distinct budgets, cadence, and metadata."""
    one_m = load_expert_training_config(
        (
            _ISSUE_6484_BASELINE_ENV22_DIR / "expert_ppo_issue_791_baseline_promotion_1m_env22.yaml"
        ).resolve()
    )
    three_m = load_expert_training_config(
        (
            _ISSUE_6484_BASELINE_ENV22_DIR / "expert_ppo_issue_791_baseline_promotion_3m_env22.yaml"
        ).resolve()
    )
    assert one_m.num_envs == three_m.num_envs == 22
    assert one_m.total_timesteps == 1_000_000
    assert three_m.total_timesteps == 3_000_000
    assert one_m.evaluation.step_schedule == ((None, 131072),)
    assert three_m.evaluation.step_schedule == ((None, 262144),)
    assert one_m.policy_id != three_m.policy_id


# Issue #6484: the issue-791 attention-head env22 pair shares a dedicated base
# while preserving explicit launch identity and frozen resolved mappings.
_ISSUE_6484_ATTENTION_HEAD_ENV22_DIR = Path("configs/training/ppo/ablations")
_ISSUE_6484_ATTENTION_HEAD_ENV22_BASE_NAME = (
    "expert_ppo_issue_791_attention_head_promotion_env22_base.yaml"
)
_ISSUE_6484_ATTENTION_HEAD_ENV22_BASELINE_PATH = Path(
    "tests/integration/_baseline_issue_6484_attention_head_promotion_env22_resolved.json"
)
_ISSUE_6484_ATTENTION_HEAD_ENV22_VARIANTS = [
    "expert_ppo_issue_791_attention_head_promotion_1m_env22.yaml",
    "expert_ppo_issue_791_attention_head_promotion_3m_env22.yaml",
]


def _issue_6484_attention_head_env22_baseline() -> dict:
    """Load and integrity-check the frozen attention-head env22 baseline."""
    assert _ISSUE_6484_ATTENTION_HEAD_ENV22_BASELINE_PATH.exists(), (
        "Pre-change attention-head env22 baseline missing; re-run capture before changing configs"
    )
    baseline = json.loads(
        _ISSUE_6484_ATTENTION_HEAD_ENV22_BASELINE_PATH.read_text(encoding="utf-8")
    )
    assert baseline["schema_version"] == "resolved-config-fingerprint.v1"
    assert set(baseline["variants"]) == set(_ISSUE_6484_ATTENTION_HEAD_ENV22_VARIANTS)
    return baseline


@pytest.mark.parametrize("variant", _ISSUE_6484_ATTENTION_HEAD_ENV22_VARIANTS)
def test_issue_6484_attention_head_env22_resolves_to_prechange_values(variant: str) -> None:
    """Each attention-head env22 variant matches its frozen pre-refactor mapping."""
    path = (_ISSUE_6484_ATTENTION_HEAD_ENV22_DIR / variant).resolve()
    baseline = _issue_6484_attention_head_env22_baseline()

    assert _issue_6484_baseline_promotion_fingerprint(path) == baseline["variants"][variant], (
        f"Resolved config {variant} differs from the baseline at {baseline['source_revision']}."
    )
    config = load_expert_training_config(path)
    assert config.policy_id
    assert config.num_envs == 22


def test_issue_6484_attention_head_env22_base_reuse_and_explicit_launch_identity() -> None:
    """The attention-head env22 variants reuse one base and keep launch identity explicit."""
    base_path = (
        _ISSUE_6484_ATTENTION_HEAD_ENV22_DIR / _ISSUE_6484_ATTENTION_HEAD_ENV22_BASE_NAME
    ).resolve()
    base_yaml = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    assert "base_config" not in base_yaml
    assert "policy_id" not in base_yaml
    assert "num_envs" not in base_yaml
    assert "total_timesteps" not in base_yaml
    assert "step_schedule" not in base_yaml.get("evaluation", {})
    assert "job_type" not in base_yaml["tracking"]["wandb"]
    assert "tags" not in base_yaml["tracking"]["wandb"]

    for variant in _ISSUE_6484_ATTENTION_HEAD_ENV22_VARIANTS:
        variant_yaml = yaml.safe_load(
            (_ISSUE_6484_ATTENTION_HEAD_ENV22_DIR / variant).read_text(encoding="utf-8")
        )
        assert variant_yaml["base_config"] == _ISSUE_6484_ATTENTION_HEAD_ENV22_BASE_NAME
        assert set(variant_yaml) == {
            "base_config",
            "policy_id",
            "num_envs",
            "total_timesteps",
            "evaluation",
            "tracking",
        }
        assert set(variant_yaml["evaluation"]) == {"step_schedule"}
        assert set(variant_yaml["tracking"]) == {"wandb"}
        assert set(variant_yaml["tracking"]["wandb"]) == {"job_type", "tags"}


def test_issue_6484_attention_head_env22_variants_keep_distinct_overrides() -> None:
    """The attention-head env22 variants retain distinct budgets and metadata."""
    one_m_path = (
        _ISSUE_6484_ATTENTION_HEAD_ENV22_DIR
        / "expert_ppo_issue_791_attention_head_promotion_1m_env22.yaml"
    ).resolve()
    three_m_path = (
        _ISSUE_6484_ATTENTION_HEAD_ENV22_DIR
        / "expert_ppo_issue_791_attention_head_promotion_3m_env22.yaml"
    ).resolve()
    one_m = load_expert_training_config(one_m_path)
    three_m = load_expert_training_config(three_m_path)
    one_m_mapping = _load_expert_training_config_mapping(one_m_path)
    three_m_mapping = _load_expert_training_config_mapping(three_m_path)

    assert one_m.num_envs == three_m.num_envs == 22
    assert one_m.total_timesteps == 1_000_000
    assert three_m.total_timesteps == 3_000_000
    assert one_m.evaluation.step_schedule == ((None, 131072),)
    assert three_m.evaluation.step_schedule == ((None, 262144),)
    assert one_m.policy_id != three_m.policy_id
    assert one_m_mapping["feature_extractor_kwargs"]["use_pedestrian_attention"] is True
    assert three_m_mapping["feature_extractor_kwargs"]["use_pedestrian_attention"] is True
    assert one_m_mapping["env_factory_kwargs"]["asymmetric_critic"] is True
    assert three_m_mapping["env_factory_kwargs"]["asymmetric_critic"] is True
    assert one_m_mapping["tracking"]["wandb"]["job_type"] == "expert-ppo-1m-promotion-env22"
    assert three_m_mapping["tracking"]["wandb"]["job_type"] == "expert-ppo-3m-promotion-env22"
    assert (
        one_m_mapping["tracking"]["wandb"]["tags"] != three_m_mapping["tracking"]["wandb"]["tags"]
    )


# Issue #6484: the issue-791 attention-head long-horizon pair shares only its
# byte-equivalent settings. Run identity and intentional 10M differences stay
# explicit in both variants.
_ISSUE_6484_ATTENTION_HEAD_10M_ENV22_DIR = Path("configs/training/ppo/ablations")
_ISSUE_6484_ATTENTION_HEAD_10M_ENV22_BASE_NAME = (
    "expert_ppo_issue_791_attention_head_promotion_10m_env22_base.yaml"
)
_ISSUE_6484_ATTENTION_HEAD_10M_ENV22_BASELINE_PATH = Path(
    "tests/integration/_baseline_issue_6484_attention_head_promotion_10m_env22_resolved.json"
)
_ISSUE_6484_ATTENTION_HEAD_10M_ENV22_VARIANTS = [
    "expert_ppo_issue_791_attention_head_promotion_10m_env22.yaml",
    "expert_ppo_issue_791_attention_head_promotion_10m_env22_eval_aligned.yaml",
]


def _issue_6484_attention_head_10m_env22_baseline() -> dict:
    """Load and integrity-check the frozen long-horizon baseline."""
    assert _ISSUE_6484_ATTENTION_HEAD_10M_ENV22_BASELINE_PATH.exists(), (
        "Pre-change long-horizon baseline missing; re-run capture before changing configs"
    )
    baseline = json.loads(
        _ISSUE_6484_ATTENTION_HEAD_10M_ENV22_BASELINE_PATH.read_text(encoding="utf-8")
    )
    assert baseline["schema_version"] == "resolved-config-fingerprint.v1"
    assert set(baseline["variants"]) == set(_ISSUE_6484_ATTENTION_HEAD_10M_ENV22_VARIANTS)
    return baseline


@pytest.mark.parametrize("variant", _ISSUE_6484_ATTENTION_HEAD_10M_ENV22_VARIANTS)
def test_issue_6484_attention_head_10m_env22_resolves_to_prechange_values(
    variant: str,
) -> None:
    """Each inherited long-horizon variant matches its frozen resolved mapping."""
    path = (_ISSUE_6484_ATTENTION_HEAD_10M_ENV22_DIR / variant).resolve()
    baseline = _issue_6484_attention_head_10m_env22_baseline()

    assert _issue_6484_baseline_promotion_fingerprint(path) == baseline["variants"][variant], (
        f"Resolved config {variant} differs from the baseline at {baseline['source_revision']}."
    )
    config = load_expert_training_config(path)
    assert config.policy_id
    assert config.num_envs == 22
    assert config.total_timesteps == 10_000_000
    assert config.evaluation.step_schedule == ((None, 524288),)


def test_issue_6484_attention_head_10m_env22_base_reuse_keeps_run_identity_explicit() -> None:
    """The long-horizon base contains only common values and no launch metadata."""
    base_path = (
        _ISSUE_6484_ATTENTION_HEAD_10M_ENV22_DIR / _ISSUE_6484_ATTENTION_HEAD_10M_ENV22_BASE_NAME
    ).resolve()
    base_yaml = yaml.safe_load(base_path.read_text(encoding="utf-8"))

    assert "base_config" not in base_yaml
    assert "policy_id" not in base_yaml
    assert "scenario_config" not in base_yaml
    assert "num_envs" not in base_yaml
    assert "total_timesteps" not in base_yaml
    assert "predictive_foresight_device" not in base_yaml["env_overrides"]
    assert "scenario_sampling" not in base_yaml
    assert "step_schedule" not in base_yaml["evaluation"]
    assert "job_type" not in base_yaml["tracking"]["wandb"]
    assert "tags" not in base_yaml["tracking"]["wandb"]

    expected_keys = {
        "base_config",
        "policy_id",
        "scenario_config",
        "num_envs",
        "total_timesteps",
        "env_overrides",
        "scenario_sampling",
        "evaluation",
        "tracking",
    }
    for variant in _ISSUE_6484_ATTENTION_HEAD_10M_ENV22_VARIANTS:
        variant_yaml = yaml.safe_load(
            (_ISSUE_6484_ATTENTION_HEAD_10M_ENV22_DIR / variant).read_text(encoding="utf-8")
        )
        assert variant_yaml["base_config"] == _ISSUE_6484_ATTENTION_HEAD_10M_ENV22_BASE_NAME
        assert set(variant_yaml) == expected_keys
        assert set(variant_yaml["env_overrides"]) == {"predictive_foresight_device"}
        assert variant_yaml["env_overrides"]["predictive_foresight_device"] == "cuda"
        assert set(variant_yaml["evaluation"]) == {"step_schedule"}
        assert set(variant_yaml["tracking"]) == {"wandb"}
        assert set(variant_yaml["tracking"]["wandb"]) == {"job_type", "tags"}


def test_issue_6484_attention_head_10m_env22_preserves_intentional_differences() -> None:
    """Scenario/sampler identity and W&B metadata remain distinct after inheritance."""
    classic_path = (
        _ISSUE_6484_ATTENTION_HEAD_10M_ENV22_DIR
        / "expert_ppo_issue_791_attention_head_promotion_10m_env22.yaml"
    ).resolve()
    eval_aligned_path = (
        _ISSUE_6484_ATTENTION_HEAD_10M_ENV22_DIR
        / "expert_ppo_issue_791_attention_head_promotion_10m_env22_eval_aligned.yaml"
    ).resolve()
    classic = _load_expert_training_config_mapping(classic_path)
    eval_aligned = _load_expert_training_config_mapping(eval_aligned_path)

    assert classic["scenario_config"] != eval_aligned["scenario_config"]
    assert "weights" in classic["scenario_sampling"]
    assert "weights" not in eval_aligned["scenario_sampling"]
    assert classic["env_overrides"]["predictive_foresight_device"] == "cuda"
    assert eval_aligned["env_overrides"]["predictive_foresight_device"] == "cuda"
    assert classic["total_timesteps"] == eval_aligned["total_timesteps"] == 10_000_000
    assert classic["evaluation"]["step_schedule"] == eval_aligned["evaluation"]["step_schedule"]
    assert classic["policy_id"] != eval_aligned["policy_id"]
    assert classic["tracking"]["wandb"]["job_type"] != eval_aligned["tracking"]["wandb"]["job_type"]
    assert classic["tracking"]["wandb"]["tags"] != eval_aligned["tracking"]["wandb"]["tags"]


# Issue #6484: the issue-791 asymmetric-critic short-budget pair shares one
# base config. The frozen fingerprints pin the pre-refactor resolved mappings.
_ISSUE_6484_ASYMMETRIC_CRITIC_DIR = Path("configs/training/ppo/ablations")
_ISSUE_6484_ASYMMETRIC_CRITIC_BASE_NAME = (
    "expert_ppo_issue_791_asymmetric_critic_promotion_base.yaml"
)
_ISSUE_6484_ASYMMETRIC_CRITIC_BASELINE_PATH = Path(
    "tests/integration/_baseline_issue_6484_asymmetric_critic_promotion_resolved.json"
)
_ISSUE_6484_ASYMMETRIC_CRITIC_VARIANTS = [
    "expert_ppo_issue_791_asymmetric_critic_promotion_128k.yaml",
    "expert_ppo_issue_791_asymmetric_critic_promotion_256k.yaml",
]


def _issue_6484_asymmetric_critic_baseline() -> dict:
    """Load and integrity-check the frozen pre-change config baseline."""
    assert _ISSUE_6484_ASYMMETRIC_CRITIC_BASELINE_PATH.exists(), (
        "Pre-change baseline missing; re-run capture before changing configs"
    )
    baseline = json.loads(_ISSUE_6484_ASYMMETRIC_CRITIC_BASELINE_PATH.read_text(encoding="utf-8"))
    assert baseline["schema_version"] == "resolved-config-fingerprint.v1"
    assert set(baseline["variants"]) == set(_ISSUE_6484_ASYMMETRIC_CRITIC_VARIANTS)
    return baseline


def _issue_6484_asymmetric_critic_fingerprint(config_path: Path) -> str:
    """Return the canonical resolved-config fingerprint for one variant."""
    resolved = _load_expert_training_config_mapping(config_path)
    canonical = json.dumps(resolved, default=str, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


@pytest.mark.parametrize("variant", _ISSUE_6484_ASYMMETRIC_CRITIC_VARIANTS)
def test_issue_6484_asymmetric_critic_resolves_to_prechange_values(variant: str) -> None:
    """Each inherited variant matches its frozen pre-refactor mapping."""
    path = (_ISSUE_6484_ASYMMETRIC_CRITIC_DIR / variant).resolve()
    baseline = _issue_6484_asymmetric_critic_baseline()

    assert _issue_6484_asymmetric_critic_fingerprint(path) == baseline["variants"][variant], (
        f"Resolved config {variant} differs from the baseline at {baseline['source_revision']}."
    )
    config = load_expert_training_config(path)
    assert config.policy_id
    assert config.evaluation.step_schedule


def test_issue_6484_asymmetric_critic_base_inheritance_and_no_launch_identity() -> None:
    """The variants inherit one lean base and keep launch identity explicit."""
    base_path = (
        _ISSUE_6484_ASYMMETRIC_CRITIC_DIR / _ISSUE_6484_ASYMMETRIC_CRITIC_BASE_NAME
    ).resolve()
    base_yaml = yaml.safe_load(base_path.read_text(encoding="utf-8"))

    assert "base_config" not in base_yaml
    assert "policy_id" not in base_yaml
    assert "total_timesteps" not in base_yaml
    assert "step_schedule" not in base_yaml.get("evaluation", {})
    assert "job_type" not in base_yaml.get("tracking", {}).get("wandb", {})
    assert "tags" not in base_yaml.get("tracking", {}).get("wandb", {})

    for variant in _ISSUE_6484_ASYMMETRIC_CRITIC_VARIANTS:
        variant_yaml = yaml.safe_load(
            (_ISSUE_6484_ASYMMETRIC_CRITIC_DIR / variant).read_text(encoding="utf-8")
        )
        assert variant_yaml["base_config"] == _ISSUE_6484_ASYMMETRIC_CRITIC_BASE_NAME
        assert set(variant_yaml) == {
            "base_config",
            "policy_id",
            "total_timesteps",
            "evaluation",
            "tracking",
        }
        assert set(variant_yaml["evaluation"]) == {"step_schedule"}
        assert set(variant_yaml["tracking"]) == {"wandb"}
        assert set(variant_yaml["tracking"]["wandb"]) == {"job_type", "tags"}


def test_issue_6484_asymmetric_critic_variants_keep_distinct_overrides() -> None:
    """The two inherited variants retain distinct budgets and cadences."""
    short = load_expert_training_config(
        (
            _ISSUE_6484_ASYMMETRIC_CRITIC_DIR
            / "expert_ppo_issue_791_asymmetric_critic_promotion_128k.yaml"
        ).resolve()
    )
    long = load_expert_training_config(
        (
            _ISSUE_6484_ASYMMETRIC_CRITIC_DIR
            / "expert_ppo_issue_791_asymmetric_critic_promotion_256k.yaml"
        ).resolve()
    )
    assert short.total_timesteps == 131072
    assert long.total_timesteps == 262144
    assert short.evaluation.step_schedule == ((None, 65536),)
    assert long.evaluation.step_schedule == ((None, 32768),)
    assert short.policy_id != long.policy_id


# Issue #6484: the issue-791 asymmetric-critic env22 1m/3m variants reuse
# the existing asymmetric-critic base while retaining their larger launch width.
_ISSUE_6484_ASYMMETRIC_ENV22_DIR = Path("configs/training/ppo/ablations")
_ISSUE_6484_ASYMMETRIC_ENV22_BASE_NAME = (
    "expert_ppo_issue_791_asymmetric_critic_promotion_base.yaml"
)
_ISSUE_6484_ASYMMETRIC_ENV22_BASELINE_PATH = Path(
    "tests/integration/_baseline_issue_6484_asymmetric_critic_env22_resolved.json"
)
_ISSUE_6484_ASYMMETRIC_ENV22_VARIANTS = [
    "expert_ppo_issue_791_asymmetric_critic_promotion_1m_env22.yaml",
    "expert_ppo_issue_791_asymmetric_critic_promotion_3m_env22.yaml",
]


def _issue_6484_asymmetric_env22_baseline() -> dict:
    """Load and integrity-check the frozen env22 config baseline."""
    assert _ISSUE_6484_ASYMMETRIC_ENV22_BASELINE_PATH.exists(), (
        "Pre-change baseline missing; re-run capture before changing configs"
    )
    baseline = json.loads(_ISSUE_6484_ASYMMETRIC_ENV22_BASELINE_PATH.read_text(encoding="utf-8"))
    assert baseline["schema_version"] == "resolved-config-fingerprint.v1"
    assert set(baseline["variants"]) == set(_ISSUE_6484_ASYMMETRIC_ENV22_VARIANTS)
    return baseline


def _issue_6484_asymmetric_env22_fingerprint(config_path: Path) -> str:
    """Return the canonical resolved-config fingerprint for one env22 variant."""
    resolved = _load_expert_training_config_mapping(config_path)
    canonical = json.dumps(resolved, default=str, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


@pytest.mark.parametrize("variant", _ISSUE_6484_ASYMMETRIC_ENV22_VARIANTS)
def test_issue_6484_asymmetric_env22_resolves_to_prechange_values(variant: str) -> None:
    """Each env22 variant matches its frozen pre-refactor mapping."""
    path = (_ISSUE_6484_ASYMMETRIC_ENV22_DIR / variant).resolve()
    baseline = _issue_6484_asymmetric_env22_baseline()

    assert _issue_6484_asymmetric_env22_fingerprint(path) == baseline["variants"][variant], (
        f"Resolved config {variant} differs from the baseline at {baseline['source_revision']}."
    )
    config = load_expert_training_config(path)
    assert config.policy_id
    assert config.num_envs == 22


def test_issue_6484_asymmetric_env22_base_reuse_and_explicit_launch_identity() -> None:
    """The env22 variants reuse one base and keep launch identity explicit."""
    base_path = (
        _ISSUE_6484_ASYMMETRIC_ENV22_DIR / _ISSUE_6484_ASYMMETRIC_ENV22_BASE_NAME
    ).resolve()
    base_yaml = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    assert "base_config" not in base_yaml
    assert "policy_id" not in base_yaml
    assert "total_timesteps" not in base_yaml
    assert "step_schedule" not in base_yaml.get("evaluation", {})
    assert "job_type" not in base_yaml["tracking"]["wandb"]
    assert "tags" not in base_yaml["tracking"]["wandb"]

    for variant in _ISSUE_6484_ASYMMETRIC_ENV22_VARIANTS:
        variant_yaml = yaml.safe_load(
            (_ISSUE_6484_ASYMMETRIC_ENV22_DIR / variant).read_text(encoding="utf-8")
        )
        assert variant_yaml["base_config"] == _ISSUE_6484_ASYMMETRIC_ENV22_BASE_NAME
        assert set(variant_yaml) == {
            "base_config",
            "policy_id",
            "num_envs",
            "total_timesteps",
            "evaluation",
            "tracking",
        }
        assert set(variant_yaml["evaluation"]) == {"step_schedule"}
        assert set(variant_yaml["tracking"]) == {"wandb"}
        assert set(variant_yaml["tracking"]["wandb"]) == {"job_type", "tags"}


def test_issue_6484_asymmetric_env22_variants_keep_distinct_overrides() -> None:
    """The env22 variants retain distinct budgets, cadence, and launch metadata."""
    one_m = load_expert_training_config(
        (
            _ISSUE_6484_ASYMMETRIC_ENV22_DIR
            / "expert_ppo_issue_791_asymmetric_critic_promotion_1m_env22.yaml"
        ).resolve()
    )
    three_m = load_expert_training_config(
        (
            _ISSUE_6484_ASYMMETRIC_ENV22_DIR
            / "expert_ppo_issue_791_asymmetric_critic_promotion_3m_env22.yaml"
        ).resolve()
    )
    assert one_m.num_envs == three_m.num_envs == 22
    assert one_m.total_timesteps == 1_000_000
    assert three_m.total_timesteps == 3_000_000
    assert one_m.evaluation.step_schedule == ((None, 131072),)
    assert three_m.evaluation.step_schedule == ((None, 262144),)
    assert one_m.policy_id != three_m.policy_id


# Issue #6484: the issue-1024 h500 schedule retrain variants share one base
# config. The constants below pin the family and its frozen pre-change
# resolved-config fingerprints.
_ISSUE_6484_ISSUE_1024_DIR = Path("configs/training/ppo/ablations")
_ISSUE_6484_ISSUE_1024_BASE_NAME = (
    "expert_ppo_issue_1024_reward_curriculum_all_available_h500_schedule_12m_base.yaml"
)
_ISSUE_6484_ISSUE_1024_BASELINE_PATH = Path(
    "tests/integration/_baseline_issue_6484_issue_1024_resolved.json"
)
_ISSUE_6484_ISSUE_1024_VARIANTS = [
    "expert_ppo_issue_1024_reward_curriculum_all_available_h500_schedule_12m_env22.yaml",
    "expert_ppo_issue_1024_reward_curriculum_all_available_h500_schedule_12m_env30_l40s.yaml",
]


def _issue_6484_issue_1024_baseline() -> dict:
    """Load and integrity-check the frozen issue-1024 config baseline."""
    assert _ISSUE_6484_ISSUE_1024_BASELINE_PATH.exists(), (
        "Pre-change baseline missing; re-run capture before changing configs"
    )
    baseline = json.loads(_ISSUE_6484_ISSUE_1024_BASELINE_PATH.read_text(encoding="utf-8"))
    assert baseline["schema_version"] == "resolved-config-fingerprint.v1"
    return baseline


def _issue_6484_issue_1024_fingerprint(config_path: Path) -> str:
    """Return the canonical resolved-config fingerprint for an issue-1024 variant."""
    resolved = _load_expert_training_config_mapping(config_path)
    canonical = json.dumps(resolved, default=str, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


@pytest.mark.parametrize("variant", _ISSUE_6484_ISSUE_1024_VARIANTS)
def test_issue_6484_issue_1024_resolves_to_prechange_values(variant: str) -> None:
    """Each migrated issue-1024 variant matches its pre-refactor mapping."""
    path = (_ISSUE_6484_ISSUE_1024_DIR / variant).resolve()
    baseline = _issue_6484_issue_1024_baseline()

    assert _issue_6484_issue_1024_fingerprint(path) == baseline["variants"][variant]
    config = load_expert_training_config(path)
    assert config.policy_id
    assert config.total_timesteps == 12000000
    assert config.evaluation.step_schedule


def test_issue_6484_issue_1024_base_inheritance_and_no_launch_identity() -> None:
    """The shared base carries common settings but no launch identity."""
    base_path = (_ISSUE_6484_ISSUE_1024_DIR / _ISSUE_6484_ISSUE_1024_BASE_NAME).resolve()
    base_yaml = yaml.safe_load(base_path.read_text(encoding="utf-8"))

    assert "base_config" not in base_yaml
    assert "policy_id" not in base_yaml
    assert "num_envs" not in base_yaml
    assert "job_type" not in base_yaml["tracking"]["wandb"]
    assert "tags" not in base_yaml["tracking"]["wandb"]

    for variant in _ISSUE_6484_ISSUE_1024_VARIANTS:
        variant_yaml = yaml.safe_load(
            (_ISSUE_6484_ISSUE_1024_DIR / variant).read_text(encoding="utf-8")
        )
        assert variant_yaml["base_config"] == _ISSUE_6484_ISSUE_1024_BASE_NAME
        assert set(variant_yaml) == {"base_config", "policy_id", "num_envs", "tracking"}
        assert set(variant_yaml["tracking"]) == {"wandb"}
        assert set(variant_yaml["tracking"]["wandb"]) == {"job_type", "tags"}


def test_issue_6484_issue_1024_variants_keep_distinct_launch_metadata() -> None:
    """The env22 and env30-l40s launch identities remain explicit and distinct."""
    env22 = load_expert_training_config(
        (
            _ISSUE_6484_ISSUE_1024_DIR
            / "expert_ppo_issue_1024_reward_curriculum_all_available_h500_schedule_12m_env22.yaml"
        ).resolve()
    )
    env30 = load_expert_training_config(
        (
            _ISSUE_6484_ISSUE_1024_DIR
            / "expert_ppo_issue_1024_reward_curriculum_all_available_h500_schedule_12m_env30_l40s.yaml"
        ).resolve()
    )
    env22_mapping = _load_expert_training_config_mapping(
        (
            _ISSUE_6484_ISSUE_1024_DIR
            / "expert_ppo_issue_1024_reward_curriculum_all_available_h500_schedule_12m_env22.yaml"
        ).resolve()
    )
    env30_mapping = _load_expert_training_config_mapping(
        (
            _ISSUE_6484_ISSUE_1024_DIR
            / "expert_ppo_issue_1024_reward_curriculum_all_available_h500_schedule_12m_env30_l40s.yaml"
        ).resolve()
    )

    assert env22.num_envs == 22
    assert env30.num_envs == 30
    assert env22.policy_id.endswith("_env22")
    assert env30.policy_id.endswith("_env30_l40s")
    assert env22_mapping["tracking"]["wandb"]["job_type"] == (
        "expert-ppo-12m-all-available-h500-env22"
    )
    assert (
        env30_mapping["tracking"]["wandb"]["job_type"]
        == "expert-ppo-12m-all-available-h500-env30-l40s"
    )
    assert env22_mapping["tracking"]["wandb"]["tags"] != env30_mapping["tracking"]["wandb"]["tags"]


# Issue #4014: the issue_4014 PPO smoke config family was migrated to inherit
# shared settings from a single base config. The constants below pin that
# contract and the frozen pre-change resolved-config baseline.
_ISSUE_4014_FAMILY_DIR = Path("configs/training/ppo")
_ISSUE_4014_BASE_NAME = "issue_4014_ppo_smoke_base.yaml"
_ISSUE_4014_BASELINE_PATH = Path("tests/integration/_baseline_issue_4014_ppo_smoke_resolved.json")
_ISSUE_4014_VARIANTS = [
    "issue_4014_ppo_smoke_matched.yaml",
    "issue_4014_ppo_mamba_smoke.yaml",
    "issue_4014_ppo_mamba_smoke_matched.yaml",
]


def _issue_4014_baseline() -> dict:
    """Load and integrity-check the frozen pre-change resolved-config baseline."""
    assert _ISSUE_4014_BASELINE_PATH.exists(), (
        "Pre-change baseline missing; re-run capture before changing configs"
    )
    baseline = json.loads(_ISSUE_4014_BASELINE_PATH.read_text(encoding="utf-8"))
    assert baseline["schema_version"] == "resolved-config-fingerprint.v1"
    return baseline


def _issue_4014_fingerprint(config_path: Path) -> str:
    """Return the canonical resolved-config fingerprint for ``config_path``."""
    resolved = _load_expert_training_config_mapping(config_path)
    canonical = json.dumps(resolved, default=str, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


@pytest.mark.parametrize("variant", _ISSUE_4014_VARIANTS)
def test_issue_4014_ppo_smoke_resolves_to_prechange_values(variant: str) -> None:
    """Each migrated variant must resolve byte-identically to the frozen pre-change mapping.

    The base_config deep-merge must reconstruct the exact pre-change resolved
    mapping for every migrated variant.
    """
    path = (_ISSUE_4014_FAMILY_DIR / variant).resolve()
    baseline = _issue_4014_baseline()

    actual_fingerprint = _issue_4014_fingerprint(path)
    assert actual_fingerprint == baseline["variants"][variant], (
        f"Resolved config {variant} differs from the baseline at {baseline['source_revision']}."
    )


def test_issue_4014_ppo_smoke_base_inherits_and_no_launch_identity() -> None:
    """Every variant inherits its shared base and the base stays lean."""
    for variant in _ISSUE_4014_VARIANTS:
        variant_path = (_ISSUE_4014_FAMILY_DIR / variant).resolve()
        variant_yaml = yaml.safe_load(variant_path.read_text(encoding="utf-8"))
        assert variant_yaml["base_config"] == _ISSUE_4014_BASE_NAME

        base_path = (_ISSUE_4014_FAMILY_DIR / _ISSUE_4014_BASE_NAME).resolve()
        base_yaml = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        assert "base_config" not in base_yaml
        assert "policy_id" not in base_yaml


@pytest.mark.parametrize("variant", _ISSUE_4014_VARIANTS)
def test_issue_4014_ppo_smoke_loads_through_loader(variant: str) -> None:
    """Each migrated variant loads through load_expert_training_config successfully."""
    path = (_ISSUE_4014_FAMILY_DIR / variant).resolve()
    config = load_expert_training_config(path)
    assert config.policy_id
    assert config.total_timesteps == 2048
    assert config.seeds == (4014,)
    assert config.evaluation.step_schedule


def test_issue_4014_ppo_smoke_variants_keep_distinct_overrides() -> None:
    """The three inherited variants retain distinct policy_id, feature_extractor, and observation_mode."""
    matched = load_expert_training_config(
        (_ISSUE_4014_FAMILY_DIR / "issue_4014_ppo_smoke_matched.yaml").resolve()
    )
    mamba = load_expert_training_config(
        (_ISSUE_4014_FAMILY_DIR / "issue_4014_ppo_mamba_smoke.yaml").resolve()
    )
    mamba_matched = load_expert_training_config(
        (_ISSUE_4014_FAMILY_DIR / "issue_4014_ppo_mamba_smoke_matched.yaml").resolve()
    )

    assert matched.policy_id == "ppo_issue_4014_smoke_matched"
    assert mamba.policy_id == "ppo_mamba_issue_4014_smoke"
    assert mamba_matched.policy_id == "ppo_mamba_issue_4014_smoke_matched"

    assert matched.feature_extractor == "default"
    assert mamba.feature_extractor == "mamba"
    assert mamba_matched.feature_extractor == "mamba"

    assert matched.env_overrides["observation_mode"] == "default_gym"
    assert mamba.env_overrides["observation_mode"] == "default"
    assert mamba_matched.env_overrides["observation_mode"] == "default_gym"

    assert matched.num_envs == 1
    assert matched.worker_mode == "dummy"
    assert mamba.num_envs is None
    assert mamba.worker_mode == "auto"
    assert mamba_matched.num_envs == 1
    assert mamba_matched.worker_mode == "dummy"

    matched_mapping = _load_expert_training_config_mapping(
        (_ISSUE_4014_FAMILY_DIR / "issue_4014_ppo_smoke_matched.yaml").resolve()
    )
    mamba_mapping = _load_expert_training_config_mapping(
        (_ISSUE_4014_FAMILY_DIR / "issue_4014_ppo_mamba_smoke.yaml").resolve()
    )
    mamba_matched_mapping = _load_expert_training_config_mapping(
        (_ISSUE_4014_FAMILY_DIR / "issue_4014_ppo_mamba_smoke_matched.yaml").resolve()
    )

    assert matched_mapping["tracking"]["tensorboard"] is False
    assert "tracking" not in mamba_mapping
    assert mamba_matched_mapping["tracking"]["tensorboard"] is False


# Issue #4018: the issue_4018 density-curriculum / fixed-density smoke pair was
# migrated to inherit shared settings from a single base config. The constants
# below pin that contract and the frozen pre-change resolved-config baseline.
_ISSUE_4018_FAMILY_DIR = Path("configs/training/ppo/ablations")
_ISSUE_4018_BASE_NAME = "issue_4018_smoke_base.yaml"
_ISSUE_4018_BASELINE_PATH = Path("tests/integration/_baseline_issue_4018_smoke_resolved.json")
_ISSUE_4018_VARIANTS = [
    "issue_4018_density_curriculum_smoke.yaml",
    "issue_4018_fixed_density_smoke.yaml",
]


def _issue_4018_baseline() -> dict:
    """Load and integrity-check the frozen pre-change resolved-config baseline."""
    assert _ISSUE_4018_BASELINE_PATH.exists(), (
        "Pre-change baseline missing; re-run capture before changing configs"
    )
    baseline = json.loads(_ISSUE_4018_BASELINE_PATH.read_text(encoding="utf-8"))
    assert baseline["schema_version"] == "resolved-config-fingerprint.v1"
    return baseline


def _issue_4018_fingerprint(config_path: Path) -> str:
    """Return the canonical resolved-config fingerprint for ``config_path``."""
    resolved = _load_expert_training_config_mapping(config_path)
    canonical = json.dumps(resolved, default=str, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


@pytest.mark.parametrize("variant", _ISSUE_4018_VARIANTS)
def test_issue_4018_density_smoke_resolves_to_prechange_values(variant: str) -> None:
    """Each migrated variant must resolve byte-identically to the frozen pre-change mapping.

    The base_config deep-merge must reconstruct the exact pre-change resolved
    mapping for every migrated variant.
    """
    path = (_ISSUE_4018_FAMILY_DIR / variant).resolve()
    baseline = _issue_4018_baseline()

    actual_fingerprint = _issue_4018_fingerprint(path)
    assert actual_fingerprint == baseline["variants"][variant], (
        f"Resolved config {variant} differs from the baseline at {baseline['source_revision']}."
    )


def test_issue_4018_density_smoke_base_inherits_and_no_launch_identity() -> None:
    """Every variant inherits its shared base and the base stays lean."""
    for variant in _ISSUE_4018_VARIANTS:
        variant_path = (_ISSUE_4018_FAMILY_DIR / variant).resolve()
        variant_yaml = yaml.safe_load(variant_path.read_text(encoding="utf-8"))
        assert variant_yaml["base_config"] == _ISSUE_4018_BASE_NAME

        base_path = (_ISSUE_4018_FAMILY_DIR / _ISSUE_4018_BASE_NAME).resolve()
        base_yaml = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        # A base must not self-inherit and carries no launch identity.
        assert "base_config" not in base_yaml
        assert "policy_id" not in base_yaml


@pytest.mark.parametrize("variant", _ISSUE_4018_VARIANTS)
def test_issue_4018_density_smoke_loads_through_loader(variant: str) -> None:
    """Each migrated variant loads through load_expert_training_config successfully."""
    path = (_ISSUE_4018_FAMILY_DIR / variant).resolve()
    config = load_expert_training_config(path)
    assert config.policy_id
    assert config.total_timesteps == 96
    assert config.seeds == (4018,)
    assert config.evaluation.step_schedule


def test_issue_4018_density_smoke_variants_keep_distinct_overrides() -> None:
    """The density-curriculum and fixed-density variants retain distinct density settings."""
    density = load_expert_training_config(
        (_ISSUE_4018_FAMILY_DIR / "issue_4018_density_curriculum_smoke.yaml").resolve()
    )
    fixed = load_expert_training_config(
        (_ISSUE_4018_FAMILY_DIR / "issue_4018_fixed_density_smoke.yaml").resolve()
    )
    density_mapping = _load_expert_training_config_mapping(
        (_ISSUE_4018_FAMILY_DIR / "issue_4018_density_curriculum_smoke.yaml").resolve()
    )
    fixed_mapping = _load_expert_training_config_mapping(
        (_ISSUE_4018_FAMILY_DIR / "issue_4018_fixed_density_smoke.yaml").resolve()
    )

    assert density.policy_id == "issue_4018_density_curriculum_smoke"
    assert fixed.policy_id == "issue_4018_fixed_density_smoke"
    assert density.policy_id != fixed.policy_id

    # Density curriculum is enabled in one, disabled in the other.
    assert density_mapping["density_curriculum"]["enabled"] is True
    assert fixed_mapping["density_curriculum"]["enabled"] is False

    # The density curriculum variant has stages; the fixed-density variant does not.
    assert "stages" in density_mapping["density_curriculum"]
    assert "stages" not in fixed_mapping["density_curriculum"]

    # Tracking tags are distinct per variant.
    assert density_mapping["tracking"]["wandb"]["tags"] == [
        "issue-4018",
        "density-curriculum-smoke",
    ]
    assert fixed_mapping["tracking"]["wandb"]["tags"] == [
        "issue-4018",
        "fixed-density-smoke",
    ]
