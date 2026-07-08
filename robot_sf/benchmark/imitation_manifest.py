"""Manifest serialization helpers for imitation learning workflows.

This module converts imitation artefact dataclasses into JSON-friendly records and persists
those manifests under the governed ``output/`` hierarchy. Serialisation keeps paths relative
where possible so artefacts remain portable across machines and CI runners.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from robot_sf.common.artifact_paths import (
    get_artifact_root,
    get_expert_policy_manifest_path,
    get_imitation_report_dir,
    get_repository_root,
    get_trajectory_dataset_dir,
)
from robot_sf.common.atomic_io import atomic_write_json

if TYPE_CHECKING:  # pragma: no cover - import surface for type checkers only
    from collections.abc import Mapping

    from robot_sf.common import (
        ExpertPolicyArtifact,
        MetricAggregate,
        TrainingRunArtifact,
        TrajectoryDatasetArtifact,
    )


def _artifact_root() -> Path:
    """Return the resolved artifact root path."""
    return get_artifact_root().resolve(strict=False)


def _path_to_manifest(path: Path) -> str:
    """Deserialise ``path`` to a portable string, relative to the artefact root when possible.

    Returns:
        Portable path string relative to artifact root or repository root.

    Raises:
        ValueError: If ``path`` is absolute and outside both the artifact root and repository root.
    """

    path_obj = Path(path)
    if not path_obj.is_absolute():
        return str(path_obj)
    root = _artifact_root()
    try:
        return str(path_obj.resolve(strict=False).relative_to(root))
    except ValueError:
        repo_root = get_repository_root().resolve(strict=False)
        try:
            return str(path_obj.resolve(strict=False).relative_to(repo_root))
        except ValueError:
            raise ValueError(
                f"Path '{path_obj}' is outside allowed roots: '{root}' and '{repo_root}'"
            ) from None


def _path_to_manifest_lenient(path: Path, *, field: str) -> str:
    """Serialise ``path`` portably, degrading to the basename for foreign paths.

    Unlike :func:`_path_to_manifest`, this never raises when ``path`` resolves outside the
    artefact and repository roots. That situation is legitimate when a run executes from one
    git worktree but references a config or evaluation artefact materialised under a sibling
    worktree: the strict helper would raise and abort the manifest write *after* training has
    already completed, discarding the entire run's evidence.

    Instead, foreign absolute paths degrade to their basename — portable and free of any
    host-specific directory leakage — and the loss of the full path is logged so it remains
    traceable.

    Returns:
        Portable path string, or the basename when the path is outside the allowed roots.
    """

    try:
        return _path_to_manifest(path)
    except ValueError:
        basename = Path(path).name
        logger.warning(
            "Manifest field '{}' path {} is outside allowed roots; recording basename '{}' "
            "to preserve the completed run's evidence.",
            field,
            path,
            basename,
        )
        return basename


def _serialize_metric(metric: MetricAggregate) -> dict[str, Any]:
    """Serialize a metric aggregate into a JSON-friendly dict.

    Returns:
        Serialized metric mapping.
    """
    payload: dict[str, Any] = {
        "mean": metric.mean,
        "median": metric.median,
        "p95": metric.p95,
    }
    if metric.ci95 is not None:
        payload["ci95"] = list(metric.ci95)
    return payload


def _to_json_ready(value: Any) -> Any:
    """Convert arbitrary values into JSON-friendly representations.

    Returns:
        JSON-ready representation of the value.
    """
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Path):
        return _path_to_manifest(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _to_json_ready(v) for k, v in value.items()}
    if isinstance(value, list | tuple | set):
        return [_to_json_ready(v) for v in value]
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()  # type: ignore[return-value]
        except (TypeError, ValueError, AttributeError):  # pragma: no cover - best effort fallback
            pass
    return str(value)


def _serialize_metrics_map(metrics: Mapping[str, MetricAggregate]) -> dict[str, Any]:
    """Serialize metric aggregates keyed by name.

    Returns:
        Mapping of metric names to serialized metric dicts.
    """
    return {name: _serialize_metric(metric) for name, metric in sorted(metrics.items())}


def serialize_expert_policy(artifact: ExpertPolicyArtifact) -> dict[str, Any]:
    """Serialise an expert policy artefact into a manifest-friendly dictionary.

    Returns:
        Dictionary representation of the expert policy artifact.
    """

    return {
        "policy_id": artifact.policy_id,
        "version": artifact.version,
        "seeds": list(artifact.seeds),
        "scenario_profile": list(artifact.scenario_profile),
        "metrics": _serialize_metrics_map(artifact.metrics),
        "checkpoint_path": _path_to_manifest(artifact.checkpoint_path),
        "config_manifest": _path_to_manifest(artifact.config_manifest),
        "validation_state": artifact.validation_state.value,
        "created_at": artifact.created_at.isoformat(),
    }


def serialize_trajectory_dataset(artifact: TrajectoryDatasetArtifact) -> dict[str, Any]:
    """Serialise a trajectory dataset artefact.

    Returns:
        Dictionary representation of the trajectory dataset artifact.
    """

    return {
        "dataset_id": artifact.dataset_id,
        "source_policy_id": artifact.source_policy_id,
        "episode_count": artifact.episode_count,
        "storage_path": _path_to_manifest(artifact.storage_path),
        "format": artifact.format,
        "scenario_coverage": {
            key: int(value) for key, value in sorted(artifact.scenario_coverage.items())
        },
        "integrity_report": _to_json_ready(artifact.integrity_report),
        "metadata": _to_json_ready(artifact.metadata),
        "quality_status": artifact.quality_status.value,
        "created_at": artifact.created_at.isoformat(),
    }


def serialize_training_run(artifact: TrainingRunArtifact) -> dict[str, Any]:
    """Serialise a training run artefact.

    Returns:
        Dictionary representation of the training run artifact.
    """

    return {
        "run_id": artifact.run_id,
        "run_type": artifact.run_type.value,
        "input_artefacts": list(artifact.input_artefacts),
        "seeds": list(artifact.seeds),
        "metrics": _serialize_metrics_map(artifact.metrics),
        "episode_log_path": _path_to_manifest(artifact.episode_log_path),
        "eval_timeline_path": (
            _path_to_manifest_lenient(artifact.eval_timeline_path, field="eval_timeline_path")
            if artifact.eval_timeline_path is not None
            else None
        ),
        "eval_per_scenario_path": (
            _path_to_manifest_lenient(
                artifact.eval_per_scenario_path, field="eval_per_scenario_path"
            )
            if artifact.eval_per_scenario_path is not None
            else None
        ),
        "perf_summary_path": (
            _path_to_manifest_lenient(artifact.perf_summary_path, field="perf_summary_path")
            if artifact.perf_summary_path is not None
            else None
        ),
        "evaluation_scenario_config": (
            _path_to_manifest_lenient(
                artifact.evaluation_scenario_config, field="evaluation_scenario_config"
            )
            if artifact.evaluation_scenario_config is not None
            else None
        ),
        "wall_clock_hours": artifact.wall_clock_hours,
        "status": artifact.status.value,
        "scenario_coverage": {
            key: int(value) for key, value in sorted(artifact.scenario_coverage.items())
        },
        "notes": [str(note) for note in artifact.notes],
    }


def write_expert_policy_manifest(
    artifact: ExpertPolicyArtifact,
    *,
    manifest_path: Path | None = None,
) -> Path:
    """Persist an expert policy manifest and return the path written.

    Returns:
        Path to the written manifest file.
    """

    payload = serialize_expert_policy(artifact)
    target = (
        Path(manifest_path)
        if manifest_path is not None
        else get_expert_policy_manifest_path(artifact.policy_id)
    )
    atomic_write_json(target, payload)
    logger.debug("Expert policy manifest written to {}", target)
    return target


def write_trajectory_dataset_manifest(
    artifact: TrajectoryDatasetArtifact,
    *,
    manifest_path: Path | None = None,
) -> Path:
    """Persist a trajectory dataset manifest.

    Returns:
        Path to the written manifest file.
    """

    payload = serialize_trajectory_dataset(artifact)
    if manifest_path is None:
        base = get_trajectory_dataset_dir()
        target = base / f"{artifact.dataset_id}.json"
    else:
        target = Path(manifest_path)
    atomic_write_json(target, payload)
    logger.debug("Trajectory dataset manifest written to {}", target)
    return target


def get_training_run_manifest_path(run_id: str, extension: str = ".json") -> Path:
    """Return the canonical manifest path for an imitation training run.

    Returns:
        Path object pointing to the manifest file location.
    """

    base = get_imitation_report_dir() / "runs"
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{run_id}{extension}"


def write_training_run_manifest(
    artifact: TrainingRunArtifact,
    *,
    manifest_path: Path | None = None,
) -> Path:
    """Persist a training run manifest, ensuring atomic writes.

    Returns:
        Path to the written manifest file.
    """

    payload = serialize_training_run(artifact)
    target = (
        Path(manifest_path)
        if manifest_path is not None
        else get_training_run_manifest_path(artifact.run_id)
    )
    atomic_write_json(target, payload)
    logger.debug("Training run manifest written to {}", target)
    return target


__all__ = [
    "get_training_run_manifest_path",
    "serialize_expert_policy",
    "serialize_training_run",
    "serialize_trajectory_dataset",
    "write_expert_policy_manifest",
    "write_training_run_manifest",
    "write_trajectory_dataset_manifest",
]
