"""Manifest serialization helpers for imitation learning workflows.

This module converts imitation artefact dataclasses into JSON-friendly records and persists
those manifests under the governed ``output/`` hierarchy. Serialisation keeps paths relative
where possible so artefacts remain portable across machines and CI runners.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

from robot_sf.common.artifact_paths import (
    get_artifact_root,
    get_expert_policy_manifest_path,
    get_imitation_report_dir,
    get_trajectory_dataset_dir,
)

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
        Portable path string relative to artifact root if possible, otherwise absolute.
    """

    path_obj = Path(path)
    if not path_obj.is_absolute():
        return str(path_obj)
    root = _artifact_root()
    try:
        return str(path_obj.resolve(strict=False).relative_to(root))
    except ValueError:
        return str(path_obj)


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
        except Exception:  # pragma: no cover - best effort fallback
            pass
    return str(value)


def _serialize_metrics_map(metrics: Mapping[str, MetricAggregate]) -> dict[str, Any]:
    """Serialize metric aggregates keyed by name.

    Returns:
        Mapping of metric names to serialized metric dicts.
    """
    return {name: _serialize_metric(metric) for name, metric in sorted(metrics.items())}


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically write a JSON payload to disk."""
    path = path.resolve(strict=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=f"{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:  # pragma: no cover - defensive cleanup
                pass


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
        "wall_clock_hours": artifact.wall_clock_hours,
        "status": artifact.status.value,
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
    _atomic_write_json(target, payload)
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
    _atomic_write_json(target, payload)
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
    _atomic_write_json(target, payload)
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
