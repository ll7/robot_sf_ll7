"""Helpers for querying run-tracker manifests and building history views."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robot_sf.telemetry.models import PipelineRunStatus

if TYPE_CHECKING:  # pragma: no cover - import hints only
    from robot_sf.telemetry.config import RunTrackerConfig


@dataclass(slots=True)
class RunHistoryEntry:
    """Materialized manifest snapshot for a single run."""

    run_id: str
    status: PipelineRunStatus
    created_at: datetime | None
    completed_at: datetime | None
    manifest_path: Path
    artifact_dir: Path
    enabled_steps: tuple[str, ...]
    summary: dict[str, Any]
    steps: list[dict[str, Any]]
    raw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the entry."""

        return {
            "run_id": self.run_id,
            "status": self.status.value,
            "created_at": _format_datetime(self.created_at),
            "completed_at": _format_datetime(self.completed_at),
            "manifest_path": str(self.manifest_path),
            "artifact_dir": str(self.artifact_dir),
            "enabled_steps": list(self.enabled_steps),
            "summary": self.summary,
            "steps": self.steps,
            "raw": self.raw,
        }


def list_runs(
    config: RunTrackerConfig,
    *,
    limit: int = 20,
    status: str | PipelineRunStatus | None = None,
    since: datetime | None = None,
) -> list[RunHistoryEntry]:
    """Return the most recent run entries honoring optional filters."""

    desired_status = _normalize_status(status)
    entries: list[RunHistoryEntry] = []
    for run_dir in _iter_run_directories(config):
        manifest_path = run_dir / config.manifest_filename
        if not manifest_path.is_file():
            continue
        record = _load_manifest_tail(manifest_path)
        if record is None:
            continue
        entry = _build_entry(record, manifest_path, run_dir)
        if desired_status and entry.status is not desired_status:
            continue
        if since and entry.created_at and entry.created_at < since:
            continue
        entries.append(entry)
    entries.sort(key=_history_sort_key, reverse=True)
    if limit > 0:
        return entries[:limit]
    return entries


def load_run(config: RunTrackerConfig, run_hint: str) -> RunHistoryEntry:
    """Load a specific run, resolving either a run ID or explicit path."""

    run_dir = _resolve_run_directory(config, run_hint)
    manifest_path = run_dir / config.manifest_filename
    record = _load_manifest_tail(manifest_path)
    if record is None:
        msg = f"Manifest missing or empty for run: {run_dir}"
        raise FileNotFoundError(msg)
    return _build_entry(record, manifest_path, run_dir)


def _iter_run_directories(config: RunTrackerConfig) -> list[Path]:
    root = config.run_tracker_root
    if not root.exists():
        return []
    return [child for child in root.iterdir() if child.is_dir()]


def _history_sort_key(entry: RunHistoryEntry) -> tuple[datetime, str]:
    tzinfo = (entry.completed_at or entry.created_at or datetime.now().astimezone()).tzinfo
    if tzinfo is None:
        tzinfo = datetime.now().astimezone().tzinfo
    fallback = datetime.fromtimestamp(0, tzinfo)
    completed = entry.completed_at or entry.created_at or fallback
    return completed, entry.run_id


def _build_entry(record: dict[str, Any], manifest_path: Path, run_dir: Path) -> RunHistoryEntry:
    run_id = str(record.get("run_id") or run_dir.name)
    status = _normalize_status(record.get("status")) or PipelineRunStatus.PENDING
    created_at = _parse_datetime(record.get("created_at"))
    completed_at = _parse_datetime(record.get("completed_at"))
    enabled_steps = tuple(str(step) for step in record.get("enabled_steps", ()))
    summary = record.get("summary") if isinstance(record.get("summary"), dict) else {}
    steps = record.get("steps") if isinstance(record.get("steps"), list) else []
    artifact_dir = Path(record.get("artifact_dir") or run_dir)
    return RunHistoryEntry(
        run_id=run_id,
        status=status,
        created_at=created_at,
        completed_at=completed_at,
        manifest_path=manifest_path,
        artifact_dir=artifact_dir,
        enabled_steps=enabled_steps,
        summary=summary,
        steps=steps,
        raw=record,
    )


def _load_manifest_tail(manifest_path: Path) -> dict[str, Any] | None:
    if not manifest_path.is_file():
        return None
    content = manifest_path.read_text(encoding="utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if not lines:
        return None
    data = json.loads(lines[-1])
    if not isinstance(data, dict):  # pragma: no cover - defensive guard
        return None
    return data


def _parse_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _format_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _normalize_status(value: str | PipelineRunStatus | None) -> PipelineRunStatus | None:
    if value is None:
        return None
    if isinstance(value, PipelineRunStatus):
        return value
    try:
        return PipelineRunStatus(str(value))
    except ValueError:
        return None


def _resolve_run_directory(config: RunTrackerConfig, run_hint: str) -> Path:
    candidate = Path(run_hint).expanduser()
    if candidate.is_dir():
        return candidate
    run_dir = config.run_tracker_root / run_hint
    if run_dir.is_dir():
        return run_dir
    msg = f"Unable to locate tracker artifacts for run: {run_hint}"
    raise FileNotFoundError(msg)
