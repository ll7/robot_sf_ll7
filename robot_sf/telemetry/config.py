"""Configuration helpers for the run-tracking telemetry subsystem."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from robot_sf.common.artifact_paths import ensure_run_tracker_tree, get_artifact_root


@dataclass(slots=True)
class RunTrackerConfig:
    """Runtime configuration for manifest + telemetry persistence."""

    artifact_root: Path | None = None
    manifest_filename: str = "manifest.jsonl"
    steps_filename: str = "steps.json"
    telemetry_filename: str = "telemetry.jsonl"
    retain_runs: int = 20
    telemetry_interval_seconds: float = 1.0

    def __post_init__(self) -> None:
        """Post init.

        Returns:
            None: Auto-generated placeholder description.
        """
        base_root = self.artifact_root or get_artifact_root()
        self.artifact_root = Path(base_root).expanduser().resolve()

    @property
    def run_tracker_root(self) -> Path:
        """Return the `run-tracker` directory and ensure it exists."""

        return ensure_run_tracker_tree(base_root=self.artifact_root)

    def get_run_directory(self, run_id: str) -> Path:
        """Return (and create) the directory dedicated to ``run_id``."""

        return ensure_run_tracker_tree(run_id=run_id, base_root=self.artifact_root)

    def manifest_path(self, run_id: str) -> Path:
        """Manifest path.

        Args:
            run_id: Auto-generated placeholder description.

        Returns:
            Path: Auto-generated placeholder description.
        """
        return self.get_run_directory(run_id) / self.manifest_filename

    def telemetry_path(self, run_id: str) -> Path:
        """Telemetry path.

        Args:
            run_id: Auto-generated placeholder description.

        Returns:
            Path: Auto-generated placeholder description.
        """
        return self.get_run_directory(run_id) / self.telemetry_filename

    def steps_path(self, run_id: str) -> Path:
        """Steps path.

        Args:
            run_id: Auto-generated placeholder description.

        Returns:
            Path: Auto-generated placeholder description.
        """
        return self.get_run_directory(run_id) / self.steps_filename
