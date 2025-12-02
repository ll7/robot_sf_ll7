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
    recommendations: tuple[dict[str, Any], ...] = ()
    perf_tests: tuple[dict[str, Any], ...] = ()
    raw: dict[str, Any] | None = None

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
            "recommendations": list(self.recommendations),
            "perf_tests": list(self.perf_tests),
            "raw": self.raw or {},
        }


def list_runs(
    config: RunTrackerConfig,
    *,
    limit: int = 20,
    status: str | PipelineRunStatus | None = None,
    since: datetime | None = None,
    scenario: str | None = None,
) -> list[RunHistoryEntry]:
    """Return the most recent run entries honoring optional filters.

    Args:
        config: Run tracker configuration
        limit: Maximum number of entries to return (0 for all)
        status: Filter by run status
        since: Filter runs created after this datetime
        scenario: Filter by scenario identifier (matches record.scenario_id)
    """

    desired_status = _normalize_status(status)
    entries: list[RunHistoryEntry] = []
    for run_dir in _iter_run_directories(config):
        manifest_path = run_dir / config.manifest_filename
        if not manifest_path.is_file():
            continue
        record, recommendations, perf_tests = _load_manifest_bundle(manifest_path)
        if record is None:
            continue
        entry = _build_entry(
            record,
            manifest_path,
            run_dir,
            recommendations=recommendations,
            perf_tests=perf_tests,
        )
        if desired_status and entry.status is not desired_status:
            continue
        if since and entry.created_at and entry.created_at < since:
            continue
        if scenario:
            # Try scenario_config_path stem, then summary.scenario_id, then initiator
            config_path = record.get("scenario_config_path")
            scenario_stem = Path(config_path).stem if config_path else None
            summary = record.get("summary") if isinstance(record.get("summary"), dict) else {}
            scenario_id = summary.get("scenario_id") or summary.get("scenario")
            if scenario not in (scenario_stem, scenario_id, record.get("scenario_id")):
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
    record, recommendations, perf_tests = _load_manifest_bundle(manifest_path)
    if record is None:
        msg = f"Manifest missing or empty for run: {run_dir}"
        raise FileNotFoundError(msg)
    return _build_entry(
        record,
        manifest_path,
        run_dir,
        recommendations=recommendations,
        perf_tests=perf_tests,
    )


def _iter_run_directories(config: RunTrackerConfig) -> list[Path]:
    """Recursively discover all run directories under the tracker root.

    Scans both direct children and nested subdirectories (e.g., perf-tests/latest)
    to ensure runs written with nested output hints are discoverable.
    """
    root = config.run_tracker_root
    if not root.exists():
        return []

    directories = []
    # Use rglob to recursively find all directories that contain manifest files
    manifest_pattern = f"**/{config.manifest_filename}"
    for manifest_path in root.glob(manifest_pattern):
        run_dir = manifest_path.parent
        directories.append(run_dir)

    return directories


def _history_sort_key(entry: RunHistoryEntry) -> tuple[datetime, str]:
    """History sort key.

    Args:
        entry: Auto-generated placeholder description.

    Returns:
        tuple[datetime, str]: Auto-generated placeholder description.
    """
    tzinfo = (entry.completed_at or entry.created_at or datetime.now().astimezone()).tzinfo
    if tzinfo is None:
        tzinfo = datetime.now().astimezone().tzinfo
    fallback = datetime.fromtimestamp(0, tzinfo)
    completed = entry.completed_at or entry.created_at or fallback
    return completed, entry.run_id


def _build_entry(
    record: dict[str, Any],
    manifest_path: Path,
    run_dir: Path,
    *,
    recommendations: list[dict[str, Any]] | None = None,
    perf_tests: list[dict[str, Any]] | None = None,
) -> RunHistoryEntry:
    """Build entry.

    Args:
        record: Auto-generated placeholder description.
        manifest_path: Auto-generated placeholder description.
        run_dir: Auto-generated placeholder description.
        recommendations: Auto-generated placeholder description.
        perf_tests: Auto-generated placeholder description.

    Returns:
        RunHistoryEntry: Auto-generated placeholder description.
    """
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
        recommendations=tuple(recommendations or ()),
        perf_tests=tuple(perf_tests or ()),
        raw=record,
    )


def _load_manifest_bundle(
    manifest_path: Path,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]:
    """Load manifest bundle.

    Args:
        manifest_path: Auto-generated placeholder description.

    Returns:
        tuple[dict[str, Any] | None, list[dict[str, Any]], list[dict[str, Any]]]: Auto-generated placeholder description.
    """
    if not manifest_path.is_file():
        return None, [], []
    content = manifest_path.read_text(encoding="utf-8")
    latest_record: dict[str, Any] | None = None
    recommendations: list[dict[str, Any]] = []
    perf_tests: list[dict[str, Any]] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:  # pragma: no cover - defensive guard
            continue
        if not isinstance(payload, dict):
            continue
        if "recommendation" in payload:
            recommendation = payload.get("recommendation")
            if isinstance(recommendation, dict):
                recommendations.append(recommendation)
            continue
        if "perf_test" in payload:
            perf = payload.get("perf_test")
            if isinstance(perf, dict):
                perf_tests.append(perf)
            continue
        latest_record = payload
    return latest_record, recommendations, perf_tests


def _parse_datetime(value: Any) -> datetime | None:
    """Parse datetime.

    Args:
        value: Auto-generated placeholder description.

    Returns:
        datetime | None: Auto-generated placeholder description.
    """
    if not value:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None


def _format_datetime(value: datetime | None) -> str | None:
    """Format datetime.

    Args:
        value: Auto-generated placeholder description.

    Returns:
        str | None: Auto-generated placeholder description.
    """
    if value is None:
        return None
    return value.isoformat()


def _normalize_status(value: str | PipelineRunStatus | None) -> PipelineRunStatus | None:
    """Normalize status.

    Args:
        value: Auto-generated placeholder description.

    Returns:
        PipelineRunStatus | None: Auto-generated placeholder description.
    """
    if value is None:
        return None
    if isinstance(value, PipelineRunStatus):
        return value
    try:
        return PipelineRunStatus(str(value))
    except ValueError:
        return None


def _resolve_run_directory(config: RunTrackerConfig, run_hint: str) -> Path:
    """Resolve run directory.

    Args:
        config: Auto-generated placeholder description.
        run_hint: Auto-generated placeholder description.

    Returns:
        Path: Auto-generated placeholder description.
    """
    candidate = Path(run_hint).expanduser()
    if candidate.is_dir():
        return candidate
    run_dir = config.run_tracker_root / run_hint
    if run_dir.is_dir():
        return run_dir
    manifest_match = _match_run_directory_by_manifest(config, run_hint)
    if manifest_match is not None:
        return manifest_match
    msg = f"Unable to locate tracker artifacts for run: {run_hint}"
    raise FileNotFoundError(msg)


def _match_run_directory_by_manifest(config: RunTrackerConfig, run_hint: str) -> Path | None:
    """Fallback: scan manifests to match by recorded run_id."""

    for run_dir in _iter_run_directories(config):
        manifest_path = run_dir / config.manifest_filename
        if not manifest_path.is_file():
            continue
        record, _, _ = _load_manifest_bundle(manifest_path)
        if not record:
            continue
        recorded_id = str(record.get("run_id") or "").strip()
        if recorded_id == run_hint:
            return run_dir
    return None
