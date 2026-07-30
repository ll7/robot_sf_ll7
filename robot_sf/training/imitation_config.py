"""Dataclasses for PPO training, fine-tuning, and imitation pipeline workflows."""

from __future__ import annotations

import difflib
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from robot_sf.common import ensure_seed_tuple
from robot_sf.telemetry.progress import PipelineStepDefinition

if TYPE_CHECKING:
    from robot_sf.training.multi_map_protocol import DomainRandomization, MultiMapTrainTestProtocol

_ALLOWED_PPO_HYPERPARAMS: frozenset[str] = frozenset(
    {
        "learning_rate",
        "batch_size",
        "n_epochs",
        "ent_coef",
        "clip_range",
        "target_kl",
        "n_steps",
        "gamma",
        "gae_lambda",
        "vf_coef",
        "max_grad_norm",
    }
)

_ALLOWED_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "algorithm",
        "base_config",
        "best_checkpoint_metric",
        "candidates",
        "convergence",
        "density_curriculum",
        "device",
        "domain_randomization",
        "env_factory_kwargs",
        "env_overrides",
        "eval_episodes",
        "eval_every",
        "evaluation",
        "feature_extractor",
        "feature_extractor_kwargs",
        "metric",
        "multi_map_protocol",
        "num_envs",
        "num_envs_reserve_cores",
        "output_dir",
        "policy",
        "policy_id",
        "policy_net_arch",
        "ppo_hyperparams",
        "randomize_seeds",
        "recurrent_policy",
        "recurrent_ppo_hyperparams",
        "resume_from",
        "resume_model_id",
        "resume_source_step",
        "safety_constraints",
        "scenario_config",
        "scenario_id",
        "scenario_sampling",
        "seed",
        "seeds",
        "snqi_baseline",
        "snqi_weights",
        "socnav_orca",
        "socnav_orca_neighbor_dist",
        "socnav_orca_time_horizon",
        "total_timesteps",
        "tracking",
        "worker_mode",
    }
)

_ALLOWED_CONVERGENCE_KEYS: frozenset[str] = frozenset(
    {
        "success_rate",
        "collision_rate",
        "plateau_window",
    }
)

_ALLOWED_EVALUATION_KEYS: frozenset[str] = frozenset(
    {
        "evaluation_episodes",
        "frequency_episodes",
        "full_policy_analysis_on_new_best",
        "full_policy_analysis_videos",
        "hold_out_scenarios",
        "randomize_seeds",
        "scenario_config",
        "step_schedule",
    }
)


def _suggest_close_matches(
    unknown_key: str,
    allowed: frozenset[str],
    *,
    max_suggestions: int = 3,
) -> list[str]:
    return difflib.get_close_matches(unknown_key, allowed, n=max_suggestions, cutoff=0.4)


def _validate_config_section_keys(
    section_name: str,
    section_data: object,
    allowed_keys: frozenset[str],
    dotted_prefix: str,
) -> None:
    if not isinstance(section_data, dict):
        return
    unknown = set(section_data) - allowed_keys
    if not unknown:
        return
    parts: list[str] = []
    for key in sorted(unknown):
        close = _suggest_close_matches(key, allowed_keys)
        hint = f" (did you mean {close}?)" if close else ""
        parts.append(f"    {dotted_prefix}.{key}{hint}")
    raise ValueError(f"{section_name} has unsupported keys:\n" + "\n".join(parts))


def validate_expert_training_config_keys(config_data: dict[str, object]) -> None:
    """Validate config keys against the allow-list, rejecting unknown keys.

    Raises:
        ValueError: With the full dotted key path and nearest valid alternatives
            when an unknown key is found.
    """
    unknown_top = set(config_data) - _ALLOWED_TOP_LEVEL_KEYS
    if unknown_top:
        parts: list[str] = []
        for key in sorted(unknown_top):
            close = _suggest_close_matches(key, _ALLOWED_TOP_LEVEL_KEYS)
            hint = f" (did you mean {close}?)" if close else ""
            parts.append(f"    {key}{hint}")
        raise ValueError(
            "ExpertTrainingConfig has unsupported top-level keys:\n" + "\n".join(parts)
        )

    convergence = config_data.get("convergence")
    if isinstance(convergence, dict):
        _validate_config_section_keys(
            "convergence",
            convergence,
            _ALLOWED_CONVERGENCE_KEYS,
            "convergence",
        )

    evaluation = config_data.get("evaluation")
    if isinstance(evaluation, dict):
        _validate_config_section_keys(
            "evaluation",
            evaluation,
            _ALLOWED_EVALUATION_KEYS,
            "evaluation",
        )

    ppo_raw = config_data.get("ppo_hyperparams")
    if isinstance(ppo_raw, dict):
        unknown_ppo = set(ppo_raw) - _ALLOWED_PPO_HYPERPARAMS
        if unknown_ppo:
            pko_parts: list[str] = []
            for key in sorted(unknown_ppo):
                close = _suggest_close_matches(key, _ALLOWED_PPO_HYPERPARAMS)
                hint = f" (did you mean {close}?)" if close else ""
                pko_parts.append(f"    ppo_hyperparams.{key}{hint}")
            raise ValueError("ppo_hyperparams has unsupported keys:\n" + "\n".join(pko_parts))


@dataclass(slots=True)
class ConvergenceCriteria:
    """Thresholds that qualify a PPO policy as an expert."""

    success_rate: float
    collision_rate: float
    plateau_window: int


@dataclass(slots=True)
class EvaluationSchedule:
    """Parameters governing periodic evaluation during PPO training workflows."""

    frequency_episodes: int
    evaluation_episodes: int
    hold_out_scenarios: tuple[str, ...] = ()
    step_schedule: tuple[tuple[int | None, int], ...] = ()
    randomize_seeds: bool = False
    scenario_config: Path | None = None


@dataclass(slots=True)
class ExpertTrainingConfig:
    """Configuration inputs for expert PPO training runs."""

    scenario_config: Path
    seeds: tuple[int, ...]
    total_timesteps: int
    policy_id: str
    convergence: ConvergenceCriteria
    evaluation: EvaluationSchedule
    ppo_hyperparams: dict[str, object] = field(default_factory=dict)
    best_checkpoint_metric: str = "success_rate"
    snqi_weights_path: Path | None = None
    snqi_baseline_path: Path | None = None
    randomize_seeds: bool = False
    scenario_id: str | None = None
    feature_extractor: str = "default"
    feature_extractor_kwargs: dict[str, object] = field(default_factory=dict)
    policy_net_arch: tuple[int, ...] = (64, 64)
    tracking: dict[str, object] = field(default_factory=dict)
    env_overrides: dict[str, object] = field(default_factory=dict)
    env_factory_kwargs: dict[str, object] = field(default_factory=dict)
    scenario_sampling: dict[str, object] = field(default_factory=dict)
    multi_map_protocol: MultiMapTrainTestProtocol | None = None
    domain_randomization: DomainRandomization | None = None
    density_curriculum: dict[str, object] = field(default_factory=dict)
    num_envs: int | str | None = None
    num_envs_reserve_cores: int = 0
    worker_mode: str = "auto"
    socnav_orca_time_horizon: float | None = None
    socnav_orca_neighbor_dist: float | None = None
    resume_from: Path | None = None
    resume_model_id: str | None = None
    resume_source_step: int | None = None

    @classmethod
    def from_raw(  # noqa: PLR0913
        cls,
        *,
        scenario_config: Path,
        seeds: tuple[int, ...] | list[int],
        randomize_seeds: bool = False,
        total_timesteps: int,
        policy_id: str,
        convergence: ConvergenceCriteria,
        evaluation: EvaluationSchedule,
        ppo_hyperparams: dict[str, object] | None = None,
        best_checkpoint_metric: str = "success_rate",
        snqi_weights_path: Path | None = None,
        snqi_baseline_path: Path | None = None,
        scenario_id: str | None = None,
        feature_extractor: str = "default",
        feature_extractor_kwargs: dict[str, object] | None = None,
        policy_net_arch: tuple[int, ...] | list[int] = (64, 64),
        tracking: dict[str, object] | None = None,
        env_overrides: dict[str, object] | None = None,
        env_factory_kwargs: dict[str, object] | None = None,
        scenario_sampling: dict[str, object] | None = None,
        multi_map_protocol: MultiMapTrainTestProtocol | None = None,
        domain_randomization: DomainRandomization | None = None,
        density_curriculum: dict[str, object] | None = None,
        num_envs: int | str | None = None,
        num_envs_reserve_cores: int = 0,
        worker_mode: str = "auto",
        socnav_orca_time_horizon: float | None = None,
        socnav_orca_neighbor_dist: float | None = None,
        resume_from: Path | None = None,
        resume_model_id: str | None = None,
        resume_source_step: int | None = None,
    ) -> ExpertTrainingConfig:
        """Create a config while coercing seeds to a canonical tuple.

        Returns:
            ExpertTrainingConfig: Constructed configuration instance.
        """

        resolved_env_factory_kwargs = dict(env_factory_kwargs or {})
        if (
            "reward_func" not in resolved_env_factory_kwargs
            and "reward_name" not in resolved_env_factory_kwargs
        ):
            resolved_env_factory_kwargs["reward_name"] = "route_completion_v2"

        return cls(
            scenario_config=scenario_config,
            seeds=ensure_seed_tuple(seeds),
            randomize_seeds=bool(randomize_seeds),
            total_timesteps=total_timesteps,
            policy_id=policy_id,
            convergence=convergence,
            evaluation=evaluation,
            ppo_hyperparams=dict(ppo_hyperparams or {}),
            best_checkpoint_metric=str(best_checkpoint_metric),
            snqi_weights_path=snqi_weights_path.resolve() if snqi_weights_path else None,
            snqi_baseline_path=snqi_baseline_path.resolve() if snqi_baseline_path else None,
            scenario_id=scenario_id,
            feature_extractor=str(feature_extractor),
            feature_extractor_kwargs=dict(feature_extractor_kwargs or {}),
            policy_net_arch=tuple(int(dim) for dim in policy_net_arch),
            tracking=dict(tracking or {}),
            env_overrides=dict(env_overrides or {}),
            env_factory_kwargs=resolved_env_factory_kwargs,
            scenario_sampling=dict(scenario_sampling or {}),
            multi_map_protocol=multi_map_protocol,
            domain_randomization=domain_randomization,
            density_curriculum=dict(density_curriculum or {}),
            num_envs=num_envs,
            num_envs_reserve_cores=int(num_envs_reserve_cores),
            worker_mode=str(worker_mode),
            socnav_orca_time_horizon=socnav_orca_time_horizon,
            socnav_orca_neighbor_dist=socnav_orca_neighbor_dist,
            resume_from=resume_from.resolve() if resume_from else None,
            resume_model_id=str(resume_model_id).strip() if resume_model_id else None,
            resume_source_step=(
                int(resume_source_step) if resume_source_step is not None else None
            ),
        )


@dataclass(slots=True)
class TrajectoryCollectionConfig:
    """Configuration controlling expert trajectory capture."""

    dataset_id: str
    source_policy_id: str
    episodes: int
    scenario_config: Path
    scenario_overrides: tuple[str, ...]
    output_format: str
    random_seeds: tuple[int, ...]
    scenario_id: str | None = None
    env_overrides: dict[str, object] = field(default_factory=dict)
    env_factory_kwargs: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_raw(  # noqa: PLR0913
        cls,
        *,
        dataset_id: str,
        source_policy_id: str,
        episodes: int,
        scenario_config: Path | str,
        scenario_overrides: tuple[str, ...] | list[str],
        output_format: str,
        random_seeds: tuple[int, ...] | list[int],
        scenario_id: str | None = None,
        env_overrides: dict[str, object] | None = None,
        env_factory_kwargs: dict[str, object] | None = None,
    ) -> TrajectoryCollectionConfig:
        """Create a config while coercing sequences to canonical tuples.

        Returns:
            TrajectoryCollectionConfig: Constructed configuration instance.
        """

        return cls(
            dataset_id=dataset_id,
            source_policy_id=source_policy_id,
            episodes=episodes,
            scenario_config=Path(scenario_config).resolve(),
            scenario_overrides=tuple(scenario_overrides),
            output_format=output_format,
            random_seeds=ensure_seed_tuple(random_seeds),
            scenario_id=scenario_id,
            env_overrides=dict(env_overrides or {}),
            env_factory_kwargs=dict(env_factory_kwargs or {}),
        )


@dataclass(slots=True)
class BehaviouralCloningConfig:
    """Hyperparameters for the offline pre-training phase."""

    dataset_id: str
    epochs: int
    batch_size: int
    learning_rate: float
    evaluation: EvaluationSchedule


@dataclass(slots=True)
class BCPretrainingConfig:
    """Configuration for behavioural cloning pre-training from expert trajectories."""

    run_id: str
    dataset_id: str
    policy_output_id: str
    bc_epochs: int
    batch_size: int
    learning_rate: float
    random_seeds: tuple[int, ...]
    training_config_path: Path | None = None
    scenario_config_path: Path | None = None
    scenario_id: str | None = None
    env_overrides: dict[str, object] = field(default_factory=dict)
    env_factory_kwargs: dict[str, object] = field(default_factory=dict)
    device: str = "auto"

    @classmethod
    def from_raw(  # noqa: PLR0913
        cls,
        *,
        run_id: str,
        dataset_id: str,
        policy_output_id: str,
        bc_epochs: int,
        batch_size: int,
        learning_rate: float,
        random_seeds: tuple[int, ...] | list[int],
        training_config_path: Path | None = None,
        scenario_config_path: Path | None = None,
        scenario_id: str | None = None,
        env_overrides: dict[str, object] | None = None,
        env_factory_kwargs: dict[str, object] | None = None,
        device: str | None = "auto",
    ) -> BCPretrainingConfig:
        """Create a config while coercing seeds to a canonical tuple.

        Returns:
            BCPretrainingConfig: Constructed configuration instance.
        """

        return cls(
            run_id=run_id,
            dataset_id=dataset_id,
            policy_output_id=policy_output_id,
            bc_epochs=bc_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            random_seeds=ensure_seed_tuple(random_seeds),
            training_config_path=training_config_path.resolve() if training_config_path else None,
            scenario_config_path=scenario_config_path.resolve() if scenario_config_path else None,
            scenario_id=str(scenario_id) if scenario_id else None,
            env_overrides=dict(env_overrides or {}),
            env_factory_kwargs=dict(env_factory_kwargs or {}),
            device=str(device or "auto").strip() or "auto",
        )


@dataclass(slots=True)
class PPOFineTuningConfig:
    """Configuration for PPO fine-tuning from a pre-trained policy."""

    run_id: str
    pretrained_policy_id: str
    total_timesteps: int
    random_seeds: tuple[int, ...]
    learning_rate: float
    snqi_weights_path: Path | None = None
    snqi_baseline_path: Path | None = None
    dataset_id: str | None = None
    training_config_path: Path | None = None
    scenario_config_path: Path | None = None
    scenario_id: str | None = None
    env_overrides: dict[str, object] = field(default_factory=dict)
    env_factory_kwargs: dict[str, object] = field(default_factory=dict)
    num_envs: int | str | None = None
    num_envs_reserve_cores: int = 0
    worker_mode: str = "auto"
    device: str = "auto"
    ppo_hyperparams: dict[str, object] = field(default_factory=dict)
    checkpoint_freq: int | None = None
    checkpoint_dir: Path | None = None

    @classmethod
    def from_raw(  # noqa: PLR0913
        cls,
        *,
        run_id: str,
        pretrained_policy_id: str,
        total_timesteps: int,
        random_seeds: tuple[int, ...] | list[int],
        learning_rate: float = 0.0003,
        snqi_weights_path: Path | None = None,
        snqi_baseline_path: Path | None = None,
        dataset_id: str | None = None,
        training_config_path: Path | None = None,
        scenario_config_path: Path | None = None,
        scenario_id: str | None = None,
        env_overrides: dict[str, object] | None = None,
        env_factory_kwargs: dict[str, object] | None = None,
        num_envs: int | str | None = None,
        num_envs_reserve_cores: int = 0,
        worker_mode: str | None = "auto",
        device: str | None = "auto",
        ppo_hyperparams: dict[str, object] | None = None,
        checkpoint_freq: int | None = None,
        checkpoint_dir: Path | None = None,
    ) -> PPOFineTuningConfig:
        """Create a config while coercing seeds to a canonical tuple.

        Returns:
            PPOFineTuningConfig: Constructed configuration instance.
        """

        return cls(
            run_id=run_id,
            pretrained_policy_id=pretrained_policy_id,
            total_timesteps=total_timesteps,
            random_seeds=ensure_seed_tuple(random_seeds),
            learning_rate=learning_rate,
            snqi_weights_path=snqi_weights_path.resolve() if snqi_weights_path else None,
            snqi_baseline_path=snqi_baseline_path.resolve() if snqi_baseline_path else None,
            dataset_id=str(dataset_id).strip() if dataset_id else None,
            training_config_path=training_config_path.resolve() if training_config_path else None,
            scenario_config_path=scenario_config_path.resolve() if scenario_config_path else None,
            scenario_id=str(scenario_id) if scenario_id else None,
            env_overrides=dict(env_overrides or {}),
            env_factory_kwargs=dict(env_factory_kwargs or {}),
            num_envs=num_envs,
            num_envs_reserve_cores=int(num_envs_reserve_cores),
            worker_mode=str(worker_mode or "auto").strip() or "auto",
            device=str(device or "auto").strip() or "auto",
            ppo_hyperparams=dict(ppo_hyperparams or {}),
            checkpoint_freq=int(checkpoint_freq) if checkpoint_freq is not None else None,
            checkpoint_dir=checkpoint_dir.resolve() if checkpoint_dir else None,
        )


@dataclass(slots=True)
class PPOFineTuneConfig:
    """Parameters for PPO fine-tuning that starts from a pre-trained policy."""

    scenario_config: Path
    seeds: tuple[int, ...]
    total_timesteps: int
    run_group_id: str
    comparison_baseline_id: str
    evaluation: EvaluationSchedule

    @classmethod
    def from_raw(
        cls,
        *,
        scenario_config: Path,
        seeds: tuple[int, ...] | list[int],
        total_timesteps: int,
        run_group_id: str,
        comparison_baseline_id: str,
        evaluation: EvaluationSchedule,
    ) -> PPOFineTuneConfig:
        """Create a config while coercing seeds to a canonical tuple.

        Returns:
            PPOFineTuneConfig: Constructed configuration instance.
        """

        return cls(
            scenario_config=scenario_config,
            seeds=ensure_seed_tuple(seeds),
            total_timesteps=total_timesteps,
            run_group_id=run_group_id,
            comparison_baseline_id=comparison_baseline_id,
            evaluation=evaluation,
        )


__all__ = [
    "_ALLOWED_CONVERGENCE_KEYS",
    "_ALLOWED_EVALUATION_KEYS",
    "_ALLOWED_PPO_HYPERPARAMS",
    "_ALLOWED_TOP_LEVEL_KEYS",
    "BCPretrainingConfig",
    "BehaviouralCloningConfig",
    "ConvergenceCriteria",
    "EvaluationSchedule",
    "ExpertTrainingConfig",
    "PPOFineTuneConfig",
    "PPOFineTuningConfig",
    "TrajectoryCollectionConfig",
    "build_imitation_pipeline_steps",
    "validate_expert_training_config_keys",
]


_DEFAULT_PIPELINE_STEPS: tuple[tuple[str, str, float], ...] = (
    ("train_expert", "Train Expert PPO Policy", 1800.0),
    ("collect_trajectories", "Collect Expert Trajectories", 900.0),
    ("bc_pretrain", "Behavioral Cloning Pre-training", 600.0),
    ("ppo_finetune", "PPO Fine-tuning", 1200.0),
    ("compare_runs", "Performance Comparison", 300.0),
)


def build_imitation_pipeline_steps(
    *,
    skip_expert: bool,
    include_comparison: bool,
) -> list[PipelineStepDefinition]:
    """Return pipeline step definitions honoring CLI toggles."""

    result: list[PipelineStepDefinition] = []
    for step_id, display_name, expected in _DEFAULT_PIPELINE_STEPS:
        if step_id == "train_expert" and skip_expert:
            continue
        if step_id == "compare_runs" and not include_comparison:
            continue
        result.append(
            PipelineStepDefinition(
                step_id=step_id,
                display_name=display_name,
                expected_duration_seconds=expected,
            )
        )
    if not result:
        raise ValueError("At least one pipeline step must remain enabled")
    return result
