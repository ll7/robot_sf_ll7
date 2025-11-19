"""Dataclasses describing configuration for PPO imitation workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from robot_sf.common import ensure_seed_tuple


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


@dataclass(slots=True)
class ExpertTrainingConfig:
    """Configuration inputs for training an expert PPO policy."""

    scenario_config: Path
    seeds: tuple[int, ...]
    total_timesteps: int
    policy_id: str
    convergence: ConvergenceCriteria
    evaluation: EvaluationSchedule

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
    ) -> ExpertTrainingConfig:
        """Create a config while coercing seeds to a canonical tuple."""

        return cls(
            scenario_config=scenario_config,
            seeds=ensure_seed_tuple(seeds),
            total_timesteps=total_timesteps,
            policy_id=policy_id,
            convergence=convergence,
            evaluation=evaluation,
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
    ) -> TrajectoryCollectionConfig:
        """Create a config while coercing sequences to canonical tuples."""

        return cls(
            dataset_id=dataset_id,
            source_policy_id=source_policy_id,
            episodes=episodes,
            scenario_config=Path(scenario_config).resolve(),
            scenario_overrides=tuple(scenario_overrides),
            output_format=output_format,
            random_seeds=ensure_seed_tuple(random_seeds),
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
        """Create a config while coercing seeds to a canonical tuple."""

        return cls(
            scenario_config=scenario_config,
            seeds=ensure_seed_tuple(seeds),
            total_timesteps=total_timesteps,
            run_group_id=run_group_id,
            comparison_baseline_id=comparison_baseline_id,
            evaluation=evaluation,
        )


__all__ = [
    "BehaviouralCloningConfig",
    "ConvergenceCriteria",
    "EvaluationSchedule",
    "ExpertTrainingConfig",
    "PPOFineTuneConfig",
    "TrajectoryCollectionConfig",
]
