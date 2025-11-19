"""Utilities for allocating run-tracker directories and enforcing rotation."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from robot_sf.telemetry.config import RunTrackerConfig


@dataclass(slots=True)
class RunDirectory:
    """Metadata returned whenever a run directory is allocated."""

    run_id: str
    path: Path


class RunRegistry:
    """Create per-run directories and prune historical runs when necessary."""

    def __init__(self, config: RunTrackerConfig) -> None:
        if not isinstance(config, RunTrackerConfig):  # pragma: no cover - runtime guard
            raise TypeError(f"config must be RunTrackerConfig, received {type(config)!r}")
        self._config = config

    @property
    def base_dir(self) -> Path:
        return Path(self._config.run_tracker_root)

    def create_run_directory(self, run_id: str, *, allow_existing: bool = False) -> RunDirectory:
        """Create the directory for ``run_id`` and materialize a lock file."""

        target = self.base_dir / run_id
        if target.exists() and not allow_existing:
            raise FileExistsError(f"Run directory already exists: {run_id}")
        target.mkdir(parents=True, exist_ok=True)
        lock_path = target / "run.lock"
        if not lock_path.exists():
            payload = {
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            lock_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return RunDirectory(run_id=run_id, path=target)

    def prune(self) -> None:
        """Keep only the most recent ``retain_runs`` directories."""

        max_runs = max(self._config.retain_runs, 1)
        run_dirs = [Path(item) for item in self.base_dir.iterdir() if item.is_dir()]
        if len(run_dirs) <= max_runs:
            return
        run_dirs.sort(key=lambda item: item.stat().st_mtime)
        for doomed in run_dirs[:-max_runs]:
            shutil.rmtree(doomed, ignore_errors=True)

    def list_run_directories(self) -> list[Path]:
        """Return currently materialized run directories (newest last)."""

        run_dirs = [Path(item) for item in self.base_dir.iterdir() if item.is_dir()]
        run_dirs.sort(key=lambda item: item.stat().st_mtime)
        return run_dirs


def generate_run_id(prefix: str = "run") -> str:
    """Return a deterministic-by-prefix UUID-backed identifier."""

    return f"{prefix}-{uuid.uuid4().hex}"
