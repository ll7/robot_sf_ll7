"""Dataclasses for PPO training, fine-tuning, and imitation pipeline workflows."""

from __future__ import annotations

import difflib
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING

from robot_sf.common import ensure_seed_tuple
from robot_sf.telemetry.progress import PipelineStepDefinition
from robot_sf.training.progress_weighted_bc import ProgressWeightedObjectiveConfig

if TYPE_CHECKING:
    from robot_sf.training.multi_map_protocol import DomainRandomization, MultiMapTrainTestProtocol


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


_EXPERT_CONFIG_FIELD_ALIASES: dict[str, str] = {
    "snqi_baseline_path": "snqi_baseline",
    "snqi_weights_path": "snqi_weights",
}
_EXPERT_CONFIG_COMPATIBILITY_KEYS: frozenset[str] = frozenset(
    {
        # The recursive loader consumes this before final validation.
        "base_config",
        # The recurrent PPO loader validates these extensions before reusing
        # ExpertTrainingConfig for its shared base fields.
        "algorithm",
        "recurrent_policy",
        "recurrent_ppo_hyperparams",
        # Backward-compatible grouped alias for the two flat dataclass fields.
        "socnav_orca",
    }
)
_CONVERGENCE_KEYS: frozenset[str] = frozenset(
    field_info.name for field_info in fields(ConvergenceCriteria)
)
_EVALUATION_KEYS: frozenset[str] = frozenset(
    {
        *(field_info.name for field_info in fields(EvaluationSchedule)),
        # Legacy no-op keys remain in tracked canonical configs. Keep them
        # recognized until those configs are migrated explicitly.
        "full_policy_analysis_on_new_best",
        "full_policy_analysis_videos",
    }
)
_EVALUATION_STEP_KEYS: frozenset[str] = frozenset({"every_steps", "until_step"})
_SOCNAV_ORCA_KEYS: frozenset[str] = frozenset({"neighbor_dist", "time_horizon"})
_MAPPING_SECTIONS: frozenset[str] = frozenset(
    {
        "convergence",
        "density_curriculum",
        "domain_randomization",
        "env_factory_kwargs",
        "env_overrides",
        "evaluation",
        "feature_extractor_kwargs",
        "multi_map_protocol",
        "ppo_hyperparams",
        "recurrent_ppo_hyperparams",
        "scenario_sampling",
        "socnav_orca",
        "tracking",
    }
)


def _expert_training_config_keys() -> frozenset[str]:
    """Return YAML keys owned by ExpertTrainingConfig and compatibility loaders."""
    field_keys = {
        _EXPERT_CONFIG_FIELD_ALIASES.get(field_info.name, field_info.name)
        for field_info in fields(ExpertTrainingConfig)
    }
    return frozenset(field_keys) | _EXPERT_CONFIG_COMPATIBILITY_KEYS


def _suggest_key(unknown_key: str, allowed_keys: Collection[str]) -> str | None:
    """Return the deterministic nearest key suggestion, when one is useful."""
    matches = difflib.get_close_matches(
        unknown_key,
        sorted(allowed_keys),
        n=1,
        cutoff=0.6,
    )
    return matches[0] if matches else None


def _unknown_key_errors(
    section_data: Mapping[object, object],
    *,
    allowed_keys: Collection[str],
    dotted_prefix: str,
) -> list[tuple[str, str]]:
    """Return deterministic full-path errors for unsupported mapping keys."""
    errors: list[tuple[str, str]] = []
    unknown_keys = set(section_data) - set(allowed_keys)
    for key in sorted(unknown_keys, key=lambda value: (type(value).__name__, repr(value))):
        if isinstance(key, str):
            dotted_path = f"{dotted_prefix}.{key}"
            suggestion = _suggest_key(key, allowed_keys)
            hint = f"; did you mean '{dotted_prefix}.{suggestion}'?" if suggestion else ""
        else:
            dotted_path = f"{dotted_prefix}[{key!r}]"
            hint = ""
        errors.append((dotted_path, f"{dotted_path}: unsupported key{hint}"))
    return errors


def _collect_mapping_sections(
    config_data: Mapping[object, object],
) -> tuple[dict[str, Mapping[object, object]], list[tuple[str, str]]]:
    """Collect mapping-valued sections and report malformed section values.

    Returns:
        Valid mapping sections and deterministic validation errors.
    """
    mappings: dict[str, Mapping[object, object]] = {}
    errors: list[tuple[str, str]] = []
    for section_name in sorted(_MAPPING_SECTIONS):
        if section_name not in config_data:
            continue
        section_data = config_data[section_name]
        dotted_path = f"config.{section_name}"
        if isinstance(section_data, Mapping):
            mappings[section_name] = section_data
        else:
            errors.append(
                (
                    dotted_path,
                    f"{dotted_path}: expected a mapping, got {type(section_data).__name__}",
                )
            )
    return mappings, errors


def _step_schedule_errors(
    evaluation: Mapping[object, object] | None,
) -> list[tuple[str, str]]:
    """Return full-path errors for malformed evaluation schedule entries."""
    if evaluation is None or "step_schedule" not in evaluation:
        return []

    errors: list[tuple[str, str]] = []
    step_schedule = evaluation["step_schedule"]
    step_path = "config.evaluation.step_schedule"
    if isinstance(step_schedule, (str, bytes)) or not isinstance(step_schedule, Sequence):
        return [
            (
                step_path,
                f"{step_path}: expected a sequence of mappings, got {type(step_schedule).__name__}",
            )
        ]

    for index, entry in enumerate(step_schedule):
        entry_path = f"{step_path}[{index}]"
        if not isinstance(entry, Mapping):
            errors.append(
                (
                    entry_path,
                    f"{entry_path}: expected a mapping, got {type(entry).__name__}",
                )
            )
            continue
        errors.extend(
            _unknown_key_errors(
                entry,
                allowed_keys=_EVALUATION_STEP_KEYS,
                dotted_prefix=entry_path,
            )
        )
    return errors


def validate_expert_training_config_keys(
    config_data: object,
    *,
    allowed_ppo_hyperparams: Collection[str],
) -> None:
    """Reject unsupported keys and malformed nested sections before training.

    The dataclass owns top-level YAML field names. The PPO launcher supplies its
    coercion-table keys so runtime parsing and validation cannot drift.

    Raises:
        ValueError: With deterministic full dotted paths for every discovered
            unsupported key or non-mapping section.
    """
    if not isinstance(config_data, Mapping):
        raise ValueError(
            f"config: expected a mapping for ExpertTrainingConfig, got {type(config_data).__name__}"
        )

    errors = _unknown_key_errors(
        config_data,
        allowed_keys=_expert_training_config_keys(),
        dotted_prefix="config",
    )
    mappings, mapping_errors = _collect_mapping_sections(config_data)
    errors.extend(mapping_errors)

    closed_sections = (
        ("convergence", _CONVERGENCE_KEYS),
        ("evaluation", _EVALUATION_KEYS),
        ("ppo_hyperparams", frozenset(allowed_ppo_hyperparams)),
        ("socnav_orca", _SOCNAV_ORCA_KEYS),
    )
    for section_name, allowed_keys in closed_sections:
        section_data = mappings.get(section_name)
        if section_data is None:
            continue
        errors.extend(
            _unknown_key_errors(
                section_data,
                allowed_keys=allowed_keys,
                dotted_prefix=f"config.{section_name}",
            )
        )

    errors.extend(_step_schedule_errors(mappings.get("evaluation")))

    if errors:
        lines = "\n".join(f"- {message}" for _, message in sorted(errors))
        raise ValueError(f"Invalid ExpertTrainingConfig keys or sections:\n{lines}")


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
    progress_weighted_objective: ProgressWeightedObjectiveConfig | None = None

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
        progress_weighted_objective: (
            ProgressWeightedObjectiveConfig | Mapping[str, object] | None
        ) = None,
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
            progress_weighted_objective=(
                progress_weighted_objective
                if isinstance(progress_weighted_objective, ProgressWeightedObjectiveConfig)
                else ProgressWeightedObjectiveConfig.from_mapping(progress_weighted_objective)
                if progress_weighted_objective
                else None
            ),
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
