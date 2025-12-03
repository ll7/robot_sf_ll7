"""Unit tests for imitation manifest helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from robot_sf.benchmark import imitation_manifest
from robot_sf.common import (
    ExpertPolicyArtifact,
    ExpertValidationState,
    MetricAggregate,
    TrainingRunArtifact,
    TrainingRunStatus,
    TrainingRunType,
    TrajectoryDatasetArtifact,
    TrajectoryQuality,
)
from robot_sf.common.artifact_paths import (
    get_expert_policy_dir,
    get_imitation_report_dir,
    get_trajectory_dataset_dir,
)


@pytest.fixture(name="metrics_summary")
def _metrics_summary() -> dict[str, MetricAggregate]:
    """TODO docstring. Document this function.


    Returns:
        TODO docstring.
    """
    return {
        "success_rate": MetricAggregate(mean=0.92, median=0.93, p95=0.98, ci95=(0.90, 0.99)),
        "collision_rate": MetricAggregate(mean=0.03, median=0.02, p95=0.05, ci95=(0.01, 0.06)),
    }


def _sample_expert_artifact(metrics: dict[str, MetricAggregate]) -> ExpertPolicyArtifact:
    """TODO docstring. Document this function.

    Args:
        metrics: TODO docstring.

    Returns:
        TODO docstring.
    """
    checkpoint = get_expert_policy_dir() / "ppo_expert_v1.zip"
    config_manifest = Path("configs/training/expert_ppo.yaml")
    return ExpertPolicyArtifact(
        policy_id="ppo_expert_v1",
        version="2025-11-14",
        seeds=(11, 17, 29),
        scenario_profile=("benchmark/default.yaml",),
        metrics=metrics,
        checkpoint_path=checkpoint,
        config_manifest=config_manifest,
        validation_state=ExpertValidationState.APPROVED,
        created_at=datetime(2025, 11, 14, 9, 0, tzinfo=UTC),
    )


def test_serialize_expert_policy_makes_paths_relative(
    metrics_summary: dict[str, MetricAggregate],
) -> None:
    """TODO docstring. Document this function.

    Args:
        metrics_summary: TODO docstring.
    """
    artifact = _sample_expert_artifact(metrics_summary)
    record = imitation_manifest.serialize_expert_policy(artifact)

    assert record["policy_id"] == artifact.policy_id
    assert record["checkpoint_path"].startswith("benchmarks/expert_policies/")
    assert record["config_manifest"] == "configs/training/expert_ppo.yaml"
    assert record["metrics"]["success_rate"]["ci95"] == [0.90, 0.99]
    assert record["validation_state"] == ExpertValidationState.APPROVED.value


def test_write_expert_policy_manifest_round_trip(
    metrics_summary: dict[str, MetricAggregate],
) -> None:
    """TODO docstring. Document this function.

    Args:
        metrics_summary: TODO docstring.
    """
    artifact = _sample_expert_artifact(metrics_summary)
    expected = imitation_manifest.serialize_expert_policy(artifact)

    manifest_path = imitation_manifest.write_expert_policy_manifest(artifact)
    with manifest_path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)

    assert loaded == expected


def test_trajectory_manifest_serialization_handles_metadata() -> None:
    """TODO docstring. Document this function."""
    dataset_dir = get_trajectory_dataset_dir()
    data_path = dataset_dir / "traj_v1.npz"
    metadata = {
        "collected_at": datetime(2025, 11, 14, 12, 30, tzinfo=UTC),
        "source_path": Path("logs/collector.log"),
        "notes": ["steady", Path("calibration.csv")],
    }
    artifact = TrajectoryDatasetArtifact(
        dataset_id="traj_v1",
        source_policy_id="ppo_expert_v1",
        episode_count=240,
        storage_path=data_path,
        format="npz",
        scenario_coverage={"benchmark/default.yaml": 200, "benchmark/crowd.yaml": 40},
        integrity_report={"status": "ok", "warnings": []},
        metadata=metadata,
        quality_status=TrajectoryQuality.VALIDATED,
        created_at=datetime(2025, 11, 14, 12, 45, tzinfo=UTC),
    )

    record = imitation_manifest.serialize_trajectory_dataset(artifact)

    assert record["storage_path"].startswith("benchmarks/expert_trajectories/")
    assert record["metadata"]["collected_at"].startswith("2025-11-14T12:30")
    assert record["metadata"]["source_path"] == "logs/collector.log"
    assert record["metadata"]["notes"][1] == "calibration.csv"


def test_training_run_manifest_writes_to_runs_folder(
    metrics_summary: dict[str, MetricAggregate],
) -> None:
    """TODO docstring. Document this function.

    Args:
        metrics_summary: TODO docstring.
    """
    episode_log = get_imitation_report_dir() / "ppo_imitation" / "episodes.jsonl"
    artifact = TrainingRunArtifact(
        run_id="run_001",
        run_type=TrainingRunType.BEHAVIOURAL_CLONING,
        input_artefacts=("policy:ppo_expert_v1", "dataset:traj_v1"),
        seeds=(11, 17, 29),
        metrics=metrics_summary,
        episode_log_path=episode_log,
        wall_clock_hours=5.25,
        status=TrainingRunStatus.COMPLETED,
        notes=["warm start converged"],
    )

    output_path = imitation_manifest.write_training_run_manifest(artifact)
    with output_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert output_path.parent.name == "runs"
    assert payload["run_id"] == artifact.run_id
    assert payload["episode_log_path"].startswith("benchmarks/ppo_imitation/")
    assert payload["status"] == TrainingRunStatus.COMPLETED.value
    assert payload["notes"] == ["warm start converged"]


@pytest.mark.parametrize(
    "manifest_path",
    [None, Path("custom/output/expert.json")],
)
def test_write_trajectory_manifest_allows_custom_path(manifest_path: Path | None) -> None:
    """TODO docstring. Document this function.

    Args:
        manifest_path: TODO docstring.
    """
    artifact = TrajectoryDatasetArtifact(
        dataset_id="traj_custom",
        source_policy_id="ppo_expert_v1",
        episode_count=200,
        storage_path=get_trajectory_dataset_dir() / "traj_custom.npz",
        format="npz",
        scenario_coverage={"benchmark/default.yaml": 200},
        integrity_report={"status": "ok"},
        metadata={},
        quality_status=TrajectoryQuality.DRAFT,
        created_at=datetime.now(UTC),
    )

    path = imitation_manifest.write_trajectory_dataset_manifest(
        artifact, manifest_path=manifest_path
    )
    assert path.exists()
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    assert data["dataset_id"] == "traj_custom"
