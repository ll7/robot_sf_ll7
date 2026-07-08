"""Validation helpers for expert trajectory datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.benchmark.rl_trajectory_dataset import (
    RL_TRAJECTORY_DATASET_SCHEMA_VERSION,
    RL_TRAJECTORY_EPISODE_SCHEMA_VERSION,
    flatten_rl_trajectory_episodes,
    load_rl_trajectory_dataset,
)
from robot_sf.common import TrajectoryQuality

MAX_DATASET_SIZE_BYTES = 25 * 1024**3
BASE_REQUIRED_ARRAYS = (
    "positions",
    "actions",
    "observations",
)
DECISION_TRANSFORMER_SCHEMA = "trajectory_dataset.v2.decision_transformer_preflight"
DECISION_TRANSFORMER_REQUIRED_ARRAYS = (
    *BASE_REQUIRED_ARRAYS,
    "rewards",
    "terminated",
    "truncated",
    "return_to_go",
)
EXCLUDED_READINESS_STATUSES = {"fallback", "degraded"}
EXCLUDED_AVAILABILITY_STATUSES = {"not_available"}


@dataclass(slots=True)
class TrajectoryDatasetValidationResult:
    """Outcome of validating an expert trajectory dataset."""

    dataset_id: str
    dataset_path: Path
    episode_count: int
    scenario_coverage: dict[str, int]
    integrity_report: dict[str, Any]
    quality_status: TrajectoryQuality


class TrajectoryDatasetValidator:
    """Validate curated trajectory datasets for completeness and integrity."""

    def __init__(self, dataset_path: Path | str) -> None:
        """Initialize validator for a dataset path."""
        self.dataset_path = Path(dataset_path).resolve()
        self.dataset_id = self.dataset_path.stem

    def validate(
        self,
        *,
        minimum_episodes: int = 200,
        require_decision_transformer_fields: bool | None = None,
    ) -> TrajectoryDatasetValidationResult:
        """Validate dataset integrity and return a structured report.

        Returns:
            Validation result for the dataset.
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        if self.dataset_path.suffix.lower() not in {".npz", ".jsonl_frames", ".jsonl"}:
            raise ValueError(
                f"Unsupported trajectory dataset format: {self.dataset_path.suffix}",
            )

        file_size = self.dataset_path.stat().st_size
        if self.dataset_path.suffix.lower() == ".jsonl":
            result = self._validate_rl_jsonl(file_size, minimum_episodes)
            return result

        if self.dataset_path.suffix.lower() == ".npz":
            result = self._validate_npz(
                file_size,
                minimum_episodes,
                require_decision_transformer_fields=require_decision_transformer_fields,
            )
        else:
            result = self._validate_jsonl(file_size, minimum_episodes)
        return result

    def _validate_npz(
        self,
        file_size: int,
        minimum_episodes: int,
        *,
        require_decision_transformer_fields: bool | None,
    ) -> TrajectoryDatasetValidationResult:
        """Validate an `.npz` trajectory dataset payload.

        Returns:
            Validation result for the dataset.
        """
        with np.load(self.dataset_path, allow_pickle=True) as data:
            files = data.files
            arrays = {name: data[name] for name in files}

        episode_count = self._extract_episode_count(arrays)
        metadata = self._extract_metadata(arrays)
        coverage = metadata.get("scenario_coverage", {})
        require_dt = self._requires_decision_transformer_fields(
            arrays,
            metadata,
            require_decision_transformer_fields,
        )
        required_arrays = (
            DECISION_TRANSFORMER_REQUIRED_ARRAYS if require_dt else BASE_REQUIRED_ARRAYS
        )
        missing_arrays = [name for name in required_arrays if name not in arrays]
        alignment_issues = self._episode_alignment_issues(arrays, required_arrays)
        status_report = self._status_report(arrays, metadata)

        report = {
            "file_size": file_size,
            "present_arrays": sorted(arrays.keys()),
            "required_arrays": list(required_arrays),
            "decision_transformer_preflight": require_dt,
            "missing_arrays": missing_arrays,
            "episode_count": episode_count,
            "metadata": metadata,
            "alignment_issues": alignment_issues,
            "status_report": status_report,
        }

        quality = self._determine_quality(
            episode_count,
            missing_arrays,
            file_size,
            minimum_episodes,
            alignment_issues=alignment_issues,
            unlabeled_excluded_rows=int(status_report["unlabeled_excluded_rows"]),
        )
        return TrajectoryDatasetValidationResult(
            dataset_id=self.dataset_id,
            dataset_path=self.dataset_path,
            episode_count=episode_count,
            scenario_coverage=coverage,
            integrity_report=report,
            quality_status=quality,
        )

    def _validate_jsonl(
        self,
        file_size: int,
        minimum_episodes: int,
    ) -> TrajectoryDatasetValidationResult:
        """Validate a JSONL trajectory dataset payload.

        Returns:
            Validation result for the dataset.
        """
        lines = self.dataset_path.read_text(encoding="utf-8").strip().splitlines()
        episode_count = len(lines)
        report = {
            "file_size": file_size,
            "line_count": episode_count,
            "metadata": {},
        }
        quality = self._determine_quality(episode_count, [], file_size, minimum_episodes)
        return TrajectoryDatasetValidationResult(
            dataset_id=self.dataset_id,
            dataset_path=self.dataset_path,
            episode_count=episode_count,
            scenario_coverage={},
            integrity_report=report,
            quality_status=quality,
        )

    def _validate_rl_jsonl(
        self,
        file_size: int,
        minimum_episodes: int,
    ) -> TrajectoryDatasetValidationResult:
        """Validate an episode-major `RLTrajectoryDataset.v1` JSONL payload.

        Returns:
            Validation result for the dataset.
        """
        episodes = load_rl_trajectory_dataset(self.dataset_path)
        batch = flatten_rl_trajectory_episodes(episodes)
        scenario_coverage: dict[str, int] = {}
        for episode in episodes:
            scenario_coverage[episode.scenario_id] = (
                scenario_coverage.get(episode.scenario_id, 0) + 1
            )

        report = {
            "file_size": file_size,
            "line_count": len(episodes),
            "dataset_schema": RL_TRAJECTORY_DATASET_SCHEMA_VERSION,
            "episode_schema": RL_TRAJECTORY_EPISODE_SCHEMA_VERSION,
            "required_arrays": [
                "observations",
                "actions",
                "rewards",
                "return_to_go",
                "terminated",
                "truncated",
                "pedestrians",
                "robot_states",
            ],
            "missing_arrays": [],
            "step_count": len(batch["rewards"]),
            "metadata": {
                "splits": sorted({episode.split for episode in episodes}),
                "source_policy_ids": sorted({episode.source_policy_id for episode in episodes}),
            },
        }
        quality = self._determine_quality(len(episodes), [], file_size, minimum_episodes)
        return TrajectoryDatasetValidationResult(
            dataset_id=self.dataset_id,
            dataset_path=self.dataset_path,
            episode_count=len(episodes),
            scenario_coverage=scenario_coverage,
            integrity_report=report,
            quality_status=quality,
        )

    def _determine_quality(
        self,
        episode_count: int,
        missing_arrays: list[str],
        file_size: int,
        minimum_episodes: int,
        *,
        alignment_issues: list[dict[str, Any]] | None = None,
        unlabeled_excluded_rows: int = 0,
    ) -> TrajectoryQuality:
        """Determine dataset quality based on size and completeness.

        Returns:
            Quality classification for the dataset.
        """
        if file_size > MAX_DATASET_SIZE_BYTES:
            return TrajectoryQuality.QUARANTINED
        if missing_arrays or alignment_issues or unlabeled_excluded_rows:
            return TrajectoryQuality.QUARANTINED
        if episode_count >= minimum_episodes:
            return TrajectoryQuality.VALIDATED
        return TrajectoryQuality.DRAFT

    @staticmethod
    def _requires_decision_transformer_fields(
        arrays: dict[str, Any],
        metadata: dict[str, Any],
        explicit: bool | None,
    ) -> bool:
        """Return whether validation should enforce the Decision Transformer schema.

        Returns:
            True when explicit mode, metadata, or DT-specific arrays request the stricter schema.
        """
        if explicit is not None:
            return bool(explicit)
        if metadata.get("dataset_schema") == DECISION_TRANSFORMER_SCHEMA:
            return True
        return any(name in arrays for name in DECISION_TRANSFORMER_REQUIRED_ARRAYS[3:])

    @staticmethod
    def _episode_alignment_issues(
        arrays: dict[str, Any],
        required_arrays: tuple[str, ...],
    ) -> list[dict[str, Any]]:
        """Return per-episode length mismatches across trajectory arrays."""
        if any(name not in arrays for name in required_arrays):
            return []

        issues: list[dict[str, Any]] = []
        top_level_lengths = {name: len(arrays[name]) for name in required_arrays}
        episode_count = int(arrays.get("episode_count", np.array(min(top_level_lengths.values()))))
        top_level_mismatches = {
            name: length for name, length in top_level_lengths.items() if length != episode_count
        }
        if top_level_mismatches:
            issues.append(
                {
                    "episode_index": None,
                    "expected_episodes": episode_count,
                    "array_episode_counts": top_level_lengths,
                }
            )

        comparable_episode_count = min(top_level_lengths.values())
        for episode_index in range(episode_count):
            if episode_index >= comparable_episode_count:
                break
            lengths: dict[str, int] = {}
            for name in required_arrays:
                value = arrays[name][episode_index]
                try:
                    lengths[name] = len(value)
                except TypeError:
                    lengths[name] = 1
            expected = lengths["actions"]
            mismatched = {name: length for name, length in lengths.items() if length != expected}
            if mismatched:
                issues.append(
                    {
                        "episode_index": episode_index,
                        "expected_steps": expected,
                        "lengths": lengths,
                    }
                )
        return issues

    @staticmethod
    def _status_report(arrays: dict[str, Any], metadata: dict[str, Any]) -> dict[str, Any]:
        """Summarize fallback/degraded/not-available rows for fail-closed preflights.

        Returns:
            Status counters and samples used to quarantine unlabeled excluded rows.
        """
        readiness_values = TrajectoryDatasetValidator._status_values(arrays.get("readiness_status"))
        availability_values = TrajectoryDatasetValidator._status_values(
            arrays.get("availability_status")
        )
        excluded_samples: list[dict[str, Any]] = []
        excluded_rows = 0
        row_count = max(len(readiness_values), len(availability_values))
        for index in range(row_count):
            readiness = readiness_values[index] if index < len(readiness_values) else ""
            availability = availability_values[index] if index < len(availability_values) else ""
            if (
                readiness in EXCLUDED_READINESS_STATUSES
                or availability in EXCLUDED_AVAILABILITY_STATUSES
            ):
                excluded_rows += 1
                if len(excluded_samples) < 10:
                    excluded_samples.append(
                        {
                            "row_index": index,
                            "readiness_status": readiness,
                            "availability_status": availability,
                        }
                    )

        status_policy = metadata.get("status_policy")
        has_label_policy = isinstance(status_policy, dict) and bool(status_policy.get("handling"))
        return {
            "excluded_rows": excluded_rows,
            "excluded_samples": excluded_samples,
            "unlabeled_excluded_rows": 0 if has_label_policy else excluded_rows,
            "status_policy_present": has_label_policy,
        }

    @staticmethod
    def _status_values(raw: Any) -> list[str]:
        """Flatten optional status arrays into normalized lowercase strings.

        Returns:
            Normalized status values from the array payload.
        """
        if raw is None:
            return []
        array = np.asarray(raw, dtype=object).reshape(-1)
        return [str(value).strip().lower() for value in array]

    @staticmethod
    def _extract_episode_count(arrays: dict[str, Any]) -> int:
        """Extract episode count from metadata or array shapes.

        Returns:
            Episode count inferred from the arrays.
        """
        count_value = arrays.get("episode_count")
        if isinstance(count_value, np.ndarray):
            if count_value.size == 1:
                return int(count_value.item())
            return int(count_value.reshape(-1)[0])
        if count_value is not None:
            return int(count_value)

        positions = arrays.get("positions")
        if isinstance(positions, np.ndarray) and positions.ndim > 0:
            return int(positions.shape[0])

        return 0

    @staticmethod
    def _extract_metadata(arrays: dict[str, Any]) -> dict[str, Any]:
        """Extract metadata dict from mixed `.npz` representations.

        Returns:
            Metadata mapping extracted from the arrays.
        """
        raw = arrays.get("metadata")
        if isinstance(raw, np.ndarray):
            if raw.shape == ():
                raw = raw.tolist()
        if isinstance(raw, dict):
            return raw
        if hasattr(raw, "item"):
            try:
                return raw.item()
            except (ValueError, AttributeError, TypeError):
                pass
        return {}
