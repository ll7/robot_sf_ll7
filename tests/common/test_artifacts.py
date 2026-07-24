"""Tests for artefact metadata dataclasses and utility functions."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from robot_sf.common.artifacts import (
    ExpertPolicyArtifact,
    ExpertValidationState,
    MetricAggregate,
    TrainingRunArtifact,
    TrainingRunStatus,
    TrainingRunType,
    TrajectoryDatasetArtifact,
    TrajectoryQuality,
    ensure_seed_tuple,
)


def test_ensure_seed_tuple_preserves_tuple():
    """Tuple of ints should be preserved as-is."""
    result = ensure_seed_tuple((1, 2, 3))
    assert result == (1, 2, 3)
    assert isinstance(result, tuple)


def test_ensure_seed_tuple_converts_list():
    """List of ints should be converted to tuple."""
    result = ensure_seed_tuple([4, 5, 6])
    assert result == (4, 5, 6)
    assert isinstance(result, tuple)


def test_ensure_seed_tuple_coerces_strings():
    """String integer tokens should be coerced via int()."""
    result = ensure_seed_tuple(("7", "8", "9"))
    assert result == (7, 8, 9)


def test_ensure_seed_tuple_coerces_numpy_ints():
    """Numpy integer types should be coerced to native Python int."""
    result = ensure_seed_tuple((np.int64(10), np.int32(20)))
    assert result == (10, 20)
    assert all(type(v) is int for v in result)


def test_ensure_seed_tuple_empty_tuple():
    """Empty tuple should return empty tuple."""
    result = ensure_seed_tuple(())
    assert result == ()


def test_ensure_seed_tuple_empty_list():
    """Empty list should return empty tuple."""
    result = ensure_seed_tuple([])
    assert result == ()
    assert isinstance(result, tuple)


def test_expert_validation_state_values():
    """ExpertValidationState enum values should be stable strings."""
    assert ExpertValidationState.DRAFT.value == "draft"
    assert ExpertValidationState.APPROVED.value == "approved"
    assert ExpertValidationState.SUPERSEDED.value == "superseded"
    assert ExpertValidationState.SYNTHETIC.value == "synthetic"


def test_expert_validation_state_from_string():
    """ExpertValidationState should be constructable from its string value."""
    assert ExpertValidationState("draft") is ExpertValidationState.DRAFT
    assert ExpertValidationState("approved") is ExpertValidationState.APPROVED


def test_trajectory_quality_values():
    """TrajectoryQuality enum values should be stable strings."""
    assert TrajectoryQuality.DRAFT.value == "draft"
    assert TrajectoryQuality.VALIDATED.value == "validated"
    assert TrajectoryQuality.QUARANTINED.value == "quarantined"


def test_trajectory_quality_from_string():
    """TrajectoryQuality should be constructable from its string value."""
    assert TrajectoryQuality("validated") is TrajectoryQuality.VALIDATED


def test_training_run_status_values():
    """TrainingRunStatus enum values should be stable strings."""
    assert TrainingRunStatus.COMPLETED.value == "completed"
    assert TrainingRunStatus.FAILED.value == "failed"
    assert TrainingRunStatus.PARTIAL.value == "partial"


def test_training_run_status_from_string():
    """TrainingRunStatus should be constructable from its string value."""
    assert TrainingRunStatus("failed") is TrainingRunStatus.FAILED


def test_training_run_type_values():
    """TrainingRunType enum values should be stable strings."""
    assert TrainingRunType.EXPERT_TRAINING.value == "expert_training"
    assert TrainingRunType.TRAJECTORY_COLLECTION.value == "trajectory_collection"
    assert TrainingRunType.BEHAVIOURAL_CLONING.value == "bc_pretrain"
    assert TrainingRunType.PPO_FINETUNE.value == "ppo_finetune"
    assert TrainingRunType.BASELINE_PPO.value == "baseline_ppo"


def test_training_run_type_from_string():
    """TrainingRunType should be constructable from its string value."""
    assert TrainingRunType("bc_pretrain") is TrainingRunType.BEHAVIOURAL_CLONING


def test_metric_aggregate_required_fields():
    """MetricAggregate should instantiate with only required fields."""
    m = MetricAggregate(mean=0.5, median=0.6, p95=0.9)
    assert m.mean == 0.5
    assert m.median == 0.6
    assert m.p95 == 0.9
    assert m.ci95 is None


def test_metric_aggregate_with_ci95():
    """MetricAggregate should accept optional ci95 field."""
    m = MetricAggregate(mean=1.0, median=1.5, p95=2.0, ci95=(0.5, 1.5))
    assert m.ci95 == (0.5, 1.5)


def test_expert_policy_artifact_defaults():
    """ExpertPolicyArtifact should apply default values for optional fields."""
    now = datetime.now()
    ckpt = Path("/ckpt")
    cfg = Path("/cfg")
    metrics = {"acc": MetricAggregate(mean=0.9, median=0.9, p95=0.95)}
    art = ExpertPolicyArtifact(
        policy_id="p1",
        version="v1",
        seeds=(1, 2),
        scenario_profile=("scenario_a",),
        metrics=metrics,
        checkpoint_path=ckpt,
        config_manifest=cfg,
        validation_state=ExpertValidationState.DRAFT,
        created_at=now,
    )
    assert art.policy_id == "p1"
    assert art.version == "v1"
    assert art.seeds == (1, 2)
    assert art.scenario_profile == ("scenario_a",)
    assert art.metrics == metrics
    assert art.checkpoint_path == ckpt
    assert art.config_manifest == cfg
    assert art.validation_state is ExpertValidationState.DRAFT
    assert art.created_at == now
    assert art.metrics_synthetic is False
    assert art.notes == ()


def test_expert_policy_artifact_with_optional():
    """ExpertPolicyArtifact should accept explicit values for optional fields."""
    now = datetime.now()
    ckpt = Path("/ckpt")
    cfg = Path("/cfg")
    metrics = {"acc": MetricAggregate(mean=0.9, median=0.9, p95=0.95)}
    art = ExpertPolicyArtifact(
        policy_id="p2",
        version="v2",
        seeds=(3,),
        scenario_profile=("scenario_b", "scenario_c"),
        metrics=metrics,
        checkpoint_path=ckpt,
        config_manifest=cfg,
        validation_state=ExpertValidationState.APPROVED,
        created_at=now,
        metrics_synthetic=True,
        notes=("note1", "note2"),
    )
    assert art.metrics_synthetic is True
    assert art.notes == ("note1", "note2")


def test_trajectory_dataset_artifact():
    """TrajectoryDatasetArtifact should instantiate with all required fields."""
    now = datetime.now()
    storage = Path("/data")
    coverage = {"scenario_a": 10}
    integrity = {"checksum": "abc123"}
    metadata = {"source": "test"}
    art = TrajectoryDatasetArtifact(
        dataset_id="ds1",
        source_policy_id="p1",
        episode_count=100,
        storage_path=storage,
        format="jsonl",
        scenario_coverage=coverage,
        integrity_report=integrity,
        metadata=metadata,
        quality_status=TrajectoryQuality.DRAFT,
        created_at=now,
    )
    assert art.dataset_id == "ds1"
    assert art.source_policy_id == "p1"
    assert art.episode_count == 100
    assert art.storage_path == storage
    assert art.format == "jsonl"
    assert art.scenario_coverage == coverage
    assert art.integrity_report == integrity
    assert art.metadata == metadata
    assert art.quality_status is TrajectoryQuality.DRAFT
    assert art.created_at == now


def test_training_run_artifact_defaults():
    """TrainingRunArtifact should apply default values for optional fields."""
    log_path = Path("/log")
    metrics = {"reward": MetricAggregate(mean=10.0, median=9.5, p95=12.0)}
    art = TrainingRunArtifact(
        run_id="r1",
        run_type=TrainingRunType.PPO_FINETUNE,
        input_artefacts=("art1",),
        seeds=(42,),
        metrics=metrics,
        episode_log_path=log_path,
        wall_clock_hours=1.5,
        status=TrainingRunStatus.COMPLETED,
    )
    assert art.run_id == "r1"
    assert art.run_type is TrainingRunType.PPO_FINETUNE
    assert art.input_artefacts == ("art1",)
    assert art.seeds == (42,)
    assert art.metrics == metrics
    assert art.episode_log_path == log_path
    assert art.wall_clock_hours == 1.5
    assert art.status is TrainingRunStatus.COMPLETED
    assert art.eval_timeline_path is None
    assert art.eval_per_scenario_path is None
    assert art.perf_summary_path is None
    assert art.evaluation_scenario_config is None
    assert art.scenario_coverage == {}
    assert art.notes == []


def test_training_run_artifact_with_optional():
    """TrainingRunArtifact should accept explicit values for optional fields."""
    log_path = Path("/log")
    eval_path = Path("/eval")
    metrics = {"reward": MetricAggregate(mean=10.0, median=9.5, p95=12.0)}
    art = TrainingRunArtifact(
        run_id="r2",
        run_type=TrainingRunType.BASELINE_PPO,
        input_artefacts=("art2", "art3"),
        seeds=(1, 2, 3),
        metrics=metrics,
        episode_log_path=log_path,
        wall_clock_hours=2.0,
        status=TrainingRunStatus.PARTIAL,
        eval_timeline_path=eval_path,
        eval_per_scenario_path=eval_path,
        perf_summary_path=eval_path,
        evaluation_scenario_config=eval_path,
        scenario_coverage={"test": 5},
        notes=["warning"],
    )
    assert art.eval_timeline_path == eval_path
    assert art.scenario_coverage == {"test": 5}
    assert art.notes == ["warning"]
