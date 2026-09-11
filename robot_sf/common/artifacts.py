"""Dataclasses capturing reproducible artefact metadata for PPO imitation workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path


class ExpertValidationState(StrEnum):
    """Lifecycle state for an expert policy artefact.

    Attributes:
        DRAFT: Recorded but not yet reviewed for reuse.
        APPROVED: Reviewed and cleared for downstream training or imitation.
        SUPERSEDED: Replaced by a newer policy version; kept for provenance.
        SYNTHETIC: Produced from synthetic data and not a validated expert.
    """

    DRAFT = "draft"
    APPROVED = "approved"
    SUPERSEDED = "superseded"
    SYNTHETIC = "synthetic"


class TrajectoryQuality(StrEnum):
    """Validation status for a recorded trajectory dataset.

    Attributes:
        DRAFT: Recorded but not yet validated for training use.
        VALIDATED: Passed the declared integrity checks and may be consumed.
        QUARANTINED: Known-defective and excluded from training use.
    """

    DRAFT = "draft"
    VALIDATED = "validated"
    QUARANTINED = "quarantined"


class TrainingRunStatus(StrEnum):
    """Execution outcome for a recorded training run.

    Attributes:
        COMPLETED: Run finished all declared epochs or episodes.
        FAILED: Run terminated with an error and is not usable evidence.
        PARTIAL: Run stopped early; metrics may be incomplete.
    """

    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class TrainingRunType(StrEnum):
    """Tagged type for registered training workflows.

    Attributes:
        EXPERT_TRAINING: Expert policy training run.
        TRAJECTORY_COLLECTION: Dataset collection run.
        BEHAVIOURAL_CLONING: Behavioural-cloning pretraining run.
        PPO_FINETUNE: PPO fine-tuning run from a prior policy.
        BASELINE_PPO: Baseline PPO run for comparison.
    """

    EXPERT_TRAINING = "expert_training"
    TRAJECTORY_COLLECTION = "trajectory_collection"
    BEHAVIOURAL_CLONING = "bc_pretrain"
    PPO_FINETUNE = "ppo_finetune"
    BASELINE_PPO = "baseline_ppo"


@dataclass(slots=True)
class MetricAggregate:
    """Aggregate statistics for a single metric value.

    Attributes:
        mean: Arithmetic mean across the recorded samples.
        median: Median across the recorded samples.
        p95: 95th percentile across the recorded samples.
        ci95: Optional two-sided 95 percent confidence interval.
    """

    mean: float
    median: float
    p95: float
    ci95: tuple[float, float] | None = None


@dataclass(slots=True)
class ExpertPolicyArtifact:
    """Reproducible snapshot describing a vetted PPO expert policy.

    Attributes:
        policy_id: Stable logical policy identifier.
        version: Policy version string owned by the training workflow.
        seeds: Training seeds that produced the policy.
        scenario_profile: Scenario families the policy was trained on.
        metrics: Aggregate metrics keyed by metric name.
        checkpoint_path: Path to the saved policy checkpoint.
        config_manifest: Path to the training configuration manifest.
        validation_state: Review state of the policy record.
        created_at: Creation timestamp recorded by the producer.
        metrics_synthetic: True when metrics came from synthetic fixtures.
        notes: Free-form reviewer notes kept with the record.
    """

    policy_id: str
    version: str
    seeds: tuple[int, ...]
    scenario_profile: tuple[str, ...]
    metrics: dict[str, MetricAggregate]
    checkpoint_path: Path
    config_manifest: Path
    validation_state: ExpertValidationState
    created_at: datetime
    metrics_synthetic: bool = False
    notes: tuple[str, ...] = field(default_factory=tuple)


@dataclass(slots=True)
class TrajectoryDatasetArtifact:
    """Metadata describing a curated expert trajectory dataset.

    Attributes:
        dataset_id: Stable logical dataset identifier.
        source_policy_id: Policy identifier that generated the trajectories.
        episode_count: Number of recorded episodes in the dataset.
        storage_path: Path to the dataset storage file.
        format: Declared storage format such as ``npz``.
        scenario_coverage: Episode counts keyed by scenario family.
        integrity_report: Producer-owned integrity details for the dataset.
        metadata: Additional producer metadata; not a claim surface.
        quality_status: Validation state of the dataset.
        created_at: Creation timestamp recorded by the producer.
    """

    dataset_id: str
    source_policy_id: str
    episode_count: int
    storage_path: Path
    format: str
    scenario_coverage: dict[str, int]
    integrity_report: dict[str, object]
    metadata: dict[str, object]
    quality_status: TrajectoryQuality
    created_at: datetime


@dataclass(slots=True)
class TrainingRunArtifact:
    """Manifest entry describing a training, collection, or evaluation run.

    Attributes:
        run_id: Stable logical run identifier.
        run_type: Registered workflow type for the run.
        input_artefacts: Logical identifiers of consumed input artefacts.
        seeds: Random seeds used by the run.
        metrics: Aggregate metrics keyed by metric name.
        episode_log_path: Path to the episode log produced by the run.
        wall_clock_hours: Measured wall-clock duration in hours.
        status: Terminal execution outcome of the run.
        eval_timeline_path: Optional path to the evaluation timeline output.
        eval_per_scenario_path: Optional path to per-scenario evaluation output.
        perf_summary_path: Optional path to the performance summary output.
        evaluation_scenario_config: Optional path to the evaluation scenario
            configuration.
        scenario_coverage: Episode counts keyed by scenario family.
        notes: Free-form producer notes kept with the record.
        metadata: Additional producer metadata; not a claim surface.
    """

    run_id: str
    run_type: TrainingRunType
    input_artefacts: tuple[str, ...]
    seeds: tuple[int, ...]
    metrics: dict[str, MetricAggregate]
    episode_log_path: Path
    wall_clock_hours: float
    status: TrainingRunStatus
    eval_timeline_path: Path | None = None
    eval_per_scenario_path: Path | None = None
    perf_summary_path: Path | None = None
    evaluation_scenario_config: Path | None = None
    scenario_coverage: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)


def ensure_seed_tuple(seeds: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Return a canonical tuple of integer seeds for manifest storage.

    Args:
        seeds: Seed values as a tuple or list.

    Returns:
        tuple[int, ...]: Seeds coerced with ``int`` in input order.
    """

    return tuple(int(seed) for seed in seeds)
