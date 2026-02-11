"""Dataclasses capturing reproducible artefact metadata for PPO imitation workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path


class ExpertValidationState(StrEnum):
    """Lifecycle state for an expert policy artefact."""

    DRAFT = "draft"
    APPROVED = "approved"
    SUPERSEDED = "superseded"
    SYNTHETIC = "synthetic"


class TrajectoryQuality(StrEnum):
    """Validation status for a recorded trajectory dataset."""

    DRAFT = "draft"
    VALIDATED = "validated"
    QUARANTINED = "quarantined"


class TrainingRunStatus(StrEnum):
    """Execution outcome for a recorded training run."""

    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class TrainingRunType(StrEnum):
    """Tagged type for registered training workflows."""

    EXPERT_TRAINING = "expert_training"
    TRAJECTORY_COLLECTION = "trajectory_collection"
    BEHAVIOURAL_CLONING = "bc_pretrain"
    PPO_FINETUNE = "ppo_finetune"
    BASELINE_PPO = "baseline_ppo"


@dataclass(slots=True)
class MetricAggregate:
    """Aggregate statistics for a single metric value."""

    mean: float
    median: float
    p95: float
    ci95: tuple[float, float] | None = None


@dataclass(slots=True)
class ExpertPolicyArtifact:
    """Reproducible snapshot describing a vetted PPO expert policy."""

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
    """Metadata describing a curated expert trajectory dataset."""

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
    """Manifest entry describing a training, collection, or evaluation run."""

    run_id: str
    run_type: TrainingRunType
    input_artefacts: tuple[str, ...]
    seeds: tuple[int, ...]
    metrics: dict[str, MetricAggregate]
    episode_log_path: Path
    wall_clock_hours: float
    status: TrainingRunStatus
    scenario_coverage: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def ensure_seed_tuple(seeds: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    """Return a canonical tuple of integer seeds for manifest storage."""

    return tuple(int(seed) for seed in seeds)
