"""JSONL manifest persistence helpers for the run tracker."""

from __future__ import annotations

import json
from dataclasses import is_dataclass
from pathlib import Path
from threading import Lock
from typing import Any, cast

from robot_sf.telemetry.config import RunTrackerConfig
from robot_sf.telemetry.models import (
    PerformanceRecommendation,
    PerformanceTestResult,
    PipelineRunRecord,
    StepExecutionEntry,
    TelemetrySnapshot,
    serialize_many,
    serialize_payload,
)
from robot_sf.telemetry.run_registry import RunRegistry


class ManifestWriter:
    """Owns manifest + telemetry outputs for a single run."""

    def __init__(
        self,
        config: RunTrackerConfig,
        run_id: str,
        *,
        registry: RunRegistry | None = None,
    ) -> None:
        """Initialize the ManifestWriter with configuration and run context.

        Args:
            config: Configuration for the run tracker including filenames and settings.
            run_id: Unique identifier for this specific run.
            registry: Optional run registry for managing run directories. If not provided,
                a new RunRegistry will be created using the config.
        """
        if not isinstance(config, RunTrackerConfig):  # defensive gate for early adopters
            raise TypeError(f"config must be RunTrackerConfig, received {type(config)!r}")
        self._config = config
        self._run_id = run_id
        self._registry = registry or RunRegistry(config)
        self._registry.prune()
        self._run_dir = Path(self._registry.create_run_directory(run_id).path)
        self._manifest_path = self._run_dir / self._config.manifest_filename
        self._telemetry_path = self._run_dir / self._config.telemetry_filename
        self._steps_path = self._run_dir / self._config.steps_filename
        self._lock = Lock()

    @property
    def run_directory(self) -> Path:
        """Get the directory path where this run's outputs are stored.


        Returns:
            Path to the run directory containing manifest, telemetry, and steps files.
        """
        return self._run_dir

    def append_run_record(self, record: PipelineRunRecord | dict[str, object]) -> None:
        """Append a pipeline run record to the manifest file.

        Args:
            record: A PipelineRunRecord dataclass or dict containing run metadata and status.
        """
        payload = self._prepare_payload(record)
        with self._lock:
            self._append_json_line(self._manifest_path, payload)

    def append_telemetry_snapshot(self, snapshot: TelemetrySnapshot | dict[str, object]) -> None:
        """Append a telemetry snapshot to the telemetry file.

        Args:
            snapshot: A TelemetrySnapshot dataclass or dict containing performance metrics.
        """
        payload = self._prepare_payload(snapshot)
        with self._lock:
            self._append_json_line(self._telemetry_path, payload)

    def write_step_index(self, entries: list[StepExecutionEntry]) -> Path:
        """Write a step execution index file and return its path.

        Args:
            entries: List of StepExecutionEntry dataclasses or dicts describing executed steps.

        Returns:
            Path to the written steps index file.
        """
        payload = serialize_many(entries)
        with self._lock:
            self._steps_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self._steps_path

    def append_recommendations(
        self,
        recommendations: list[PerformanceRecommendation] | list[dict[str, object]],
    ) -> None:
        """Append performance recommendations to the manifest file.

        Args:
            recommendations: List of PerformanceRecommendation dataclasses or dicts.
        """
        serialized = [self._prepare_payload(item) for item in recommendations]
        with self._lock:
            for recommendation in serialized:
                self._append_json_line(self._manifest_path, {"recommendation": recommendation})

    def append_performance_test(self, result: PerformanceTestResult | dict[str, object]) -> None:
        """Append a performance test result to the manifest file.

        Args:
            result: A PerformanceTestResult dataclass or dict containing test results.
        """
        payload = self._prepare_payload(result)
        with self._lock:
            self._append_json_line(self._manifest_path, {"perf_test": payload})

    def iter_run_records(self) -> list[dict[str, object]]:
        """Get the directory path where this run's outputs are stored.


        Returns:
            Path to the run directory containing manifest, telemetry, and steps files.
        """
        if not self._manifest_path.exists():
            return []
        return [
            json.loads(line)
            for line in self._manifest_path.read_text(encoding="utf-8").splitlines()
            if line
        ]

    @staticmethod
    def _append_json_line(target: Path, payload: dict[str, Any]) -> None:
        """Append a JSON line to a file, creating directories as needed.

        Args:
            target: Path to the target file.
            payload: Dictionary to serialize as JSON and append.
        """
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    @staticmethod
    def _prepare_payload(payload: object) -> dict[str, Any]:
        """Convert a payload object to a dictionary for serialization.

        Args:
            payload: A dataclass, dict, or other object to prepare for serialization.

        Returns:
            Dictionary representation of the payload suitable for JSON serialization.

        Raises:
            TypeError: If the payload is neither a dict nor a dataclass.
        """
        if isinstance(payload, dict):
            return cast("dict[str, Any]", payload)
        if is_dataclass(payload):
            serialized = serialize_payload(payload)
            if not isinstance(serialized, dict):  # pragma: no cover - defensive guard
                raise TypeError("Serialized dataclass payload must produce a mapping")
            return serialized
        msg = (
            f"ManifestWriter expected a dataclass or dict payload, received type {type(payload)!r}"
        )
        raise TypeError(msg)
