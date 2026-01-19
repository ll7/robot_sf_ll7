"""Dataclasses describing configuration for PPO imitation workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from robot_sf.common import ensure_seed_tuple
from robot_sf.telemetry.progress import PipelineStepDefinition


@dataclass(slots=True)
class ConvergenceCriteria:
    """Thresholds that qualify a PPO policy as an expert."""

    success_rate: float
    collision_rate: float
    plateau_window: int


@dataclass(slots=True)
class EvaluationSchedule:
    """Parameters governing periodic evaluation during training."""

    frequency_episodes: int
    evaluation_episodes: int
    hold_out_scenarios: tuple[str, ...] = ()
    step_schedule: tuple[tuple[int | None, int], ...] = ()


@dataclass(slots=True)
class ExpertTrainingConfig:
    """Configuration inputs for training an expert PPO policy."""

    scenario_config: Path
    seeds: tuple[int, ...]
    total_timesteps: int
    policy_id: str
    convergence: ConvergenceCriteria
    evaluation: EvaluationSchedule
    scenario_id: str | None = None
    feature_extractor: str = "default"
    feature_extractor_kwargs: dict[str, object] = field(default_factory=dict)
    policy_net_arch: tuple[int, ...] = (64, 64)
    tracking: dict[str, object] = field(default_factory=dict)
    env_overrides: dict[str, object] = field(default_factory=dict)
    env_factory_kwargs: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_raw(
        cls,
        *,
        scenario_config: Path,
        seeds: tuple[int, ...] | list[int],
        total_timesteps: int,
        policy_id: str,
        convergence: ConvergenceCriteria,
        evaluation: EvaluationSchedule,
        scenario_id: str | None = None,
        feature_extractor: str = "default",
        feature_extractor_kwargs: dict[str, object] | None = None,
        policy_net_arch: tuple[int, ...] | list[int] = (64, 64),
        tracking: dict[str, object] | None = None,
        env_overrides: dict[str, object] | None = None,
        env_factory_kwargs: dict[str, object] | None = None,
    ) -> ExpertTrainingConfig:
        """Create a config while coercing seeds to a canonical tuple.

        Returns:
            ExpertTrainingConfig: Constructed configuration instance.
        """

        return cls(
            scenario_config=scenario_config,
            seeds=ensure_seed_tuple(seeds),
            total_timesteps=total_timesteps,
            policy_id=policy_id,
            convergence=convergence,
            evaluation=evaluation,
            scenario_id=scenario_id,
            feature_extractor=str(feature_extractor),
            feature_extractor_kwargs=dict(feature_extractor_kwargs or {}),
            policy_net_arch=tuple(int(dim) for dim in policy_net_arch),
            tracking=dict(tracking or {}),
            env_overrides=dict(env_overrides or {}),
            env_factory_kwargs=dict(env_factory_kwargs or {}),
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

    @classmethod
    def from_raw(
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

    @classmethod
    def from_raw(
        cls,
        *,
        run_id: str,
        dataset_id: str,
        policy_output_id: str,
        bc_epochs: int,
        batch_size: int,
        learning_rate: float,
        random_seeds: tuple[int, ...] | list[int],
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
        )


@dataclass(slots=True)
class PPOFineTuningConfig:
    """Configuration for PPO fine-tuning from a pre-trained policy."""

    run_id: str
    pretrained_policy_id: str
    total_timesteps: int
    random_seeds: tuple[int, ...]
    learning_rate: float

    @classmethod
    def from_raw(
        cls,
        *,
        run_id: str,
        pretrained_policy_id: str,
        total_timesteps: int,
        random_seeds: tuple[int, ...] | list[int],
        learning_rate: float = 0.0003,
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
