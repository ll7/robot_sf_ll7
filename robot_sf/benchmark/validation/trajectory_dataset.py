"""Validation helpers for expert trajectory datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_sf.common import TrajectoryQuality

MAX_DATASET_SIZE_BYTES = 25 * 1024**3
REQUIRED_ARRAYS = ("positions", "actions", "observations")


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
        """TODO docstring. Document this function.

        Args:
            dataset_path: TODO docstring.
        """
        self.dataset_path = Path(dataset_path).resolve()
        self.dataset_id = self.dataset_path.stem

    def validate(self, *, minimum_episodes: int = 200) -> TrajectoryDatasetValidationResult:
        """TODO docstring. Document this function.

        Args:
            minimum_episodes: TODO docstring.

        Returns:
            TODO docstring.
        """
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_path}")

        if self.dataset_path.suffix.lower() not in {".npz", ".jsonl_frames"}:
            raise ValueError(
                f"Unsupported trajectory dataset format: {self.dataset_path.suffix}",
            )

        file_size = self.dataset_path.stat().st_size
        if self.dataset_path.suffix.lower() == ".npz":
            result = self._validate_npz(file_size, minimum_episodes)
        else:
            result = self._validate_jsonl(file_size, minimum_episodes)
        return result

    def _validate_npz(
        self,
        file_size: int,
        minimum_episodes: int,
    ) -> TrajectoryDatasetValidationResult:
        """TODO docstring. Document this function.

        Args:
            file_size: TODO docstring.
            minimum_episodes: TODO docstring.

        Returns:
            TODO docstring.
        """
        with np.load(self.dataset_path, allow_pickle=True) as data:
            files = data.files
            arrays = {name: data[name] for name in files}

        missing_arrays = [name for name in REQUIRED_ARRAYS if name not in arrays]
        episode_count = self._extract_episode_count(arrays)
        metadata = self._extract_metadata(arrays)
        coverage = metadata.get("scenario_coverage", {})

        report = {
            "file_size": file_size,
            "present_arrays": sorted(arrays.keys()),
            "missing_arrays": missing_arrays,
            "episode_count": episode_count,
            "metadata": metadata,
        }

        quality = self._determine_quality(
            episode_count, missing_arrays, file_size, minimum_episodes
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
        """TODO docstring. Document this function.

        Args:
            file_size: TODO docstring.
            minimum_episodes: TODO docstring.

        Returns:
            TODO docstring.
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

    def _determine_quality(
        self,
        episode_count: int,
        missing_arrays: list[str],
        file_size: int,
        minimum_episodes: int,
    ) -> TrajectoryQuality:
        """TODO docstring. Document this function.

        Args:
            episode_count: TODO docstring.
            missing_arrays: TODO docstring.
            file_size: TODO docstring.
            minimum_episodes: TODO docstring.

        Returns:
            TODO docstring.
        """
        if file_size > MAX_DATASET_SIZE_BYTES:
            return TrajectoryQuality.QUARANTINED
        if missing_arrays:
            return TrajectoryQuality.QUARANTINED
        if episode_count >= minimum_episodes:
            return TrajectoryQuality.VALIDATED
        return TrajectoryQuality.DRAFT

    @staticmethod
    def _extract_episode_count(arrays: dict[str, Any]) -> int:
        """TODO docstring. Document this function.

        Args:
            arrays: TODO docstring.

        Returns:
            TODO docstring.
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
        """TODO docstring. Document this function.

        Args:
            arrays: TODO docstring.

        Returns:
            TODO docstring.
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
            except Exception:
                pass
        return {}
