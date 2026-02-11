"""Canonical telemetry data models for run tracking.

These dataclasses codify the schema described in
`specs/001-performance-tracking/data-model.md` so that both the tracker runtime
and downstream tooling share the same serialization contract.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime
from enum import Enum, StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import hints only during type checking
    from collections.abc import Iterable, Sequence


class PipelineRunStatus(StrEnum):
    """States for overall pipeline execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepStatus(StrEnum):
    """Lifecycle states for individual pipeline steps."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class RecommendationSeverity(StrEnum):
    """Severity levels for generated performance recommendations."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class PerformanceTestStatus(StrEnum):
    """Classification for scripted performance smoke tests."""

    PASSED = "passed"
    SOFT_BREACH = "soft-breach"
    FAILED = "failed"


@dataclass(slots=True)
class StepExecutionEntry:
    """Per-step timing and status metadata."""

    step_id: str
    display_name: str
    order: int
    status: StepStatus = StepStatus.PENDING
    started_at: datetime | None = None
    ended_at: datetime | None = None
    duration_seconds: float | None = None
    eta_snapshot_seconds: float | None = None
    artifacts: tuple[str, ...] = ()


@dataclass(slots=True)
class TelemetrySnapshot:
    """Time-series resource metrics captured during a run."""

    timestamp_ms: int
    frame_idx: int | None = None
    status: str | None = None
    step_id: str | None = None
    steps_per_sec: float | None = None
    fps: float | None = None
    cpu_percent_process: float | None = None
    cpu_percent_system: float | None = None
    memory_rss_mb: float | None = None
    gpu_util_percent: float | None = None
    gpu_mem_used_mb: float | None = None
    notes: str | None = None


@dataclass(slots=True)
class PerformanceRecommendation:
    """Structured remediation guidance emitted by the rule engine."""

    trigger: str
    severity: RecommendationSeverity
    message: str
    suggested_actions: tuple[str, ...]
    evidence: dict[str, Any] = field(default_factory=dict)
    timestamp_ms: int | None = None


@dataclass(slots=True)
class PerformanceTestResult:
    """Result metadata for a standalone performance smoke test."""

    test_id: str
    matrix: str
    throughput_baseline: float
    throughput_measured: float
    duration_seconds: float
    status: PerformanceTestStatus
    recommendations_ref: tuple[int, ...] = ()


@dataclass(slots=True)
class PipelineRunRecord:
    """Top-level manifest entry for a pipeline execution."""

    run_id: str
    created_at: datetime
    status: PipelineRunStatus
    enabled_steps: Sequence[str]
    artifact_dir: Path
    initiator: str | None = None
    scenario_config_path: Path | None = None
    completed_at: datetime | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    steps: list[StepExecutionEntry] = field(default_factory=list)
    telemetry: list[TelemetrySnapshot] = field(default_factory=list)
    recommendations: list[PerformanceRecommendation] = field(default_factory=list)
    perf_tests: list[PerformanceTestResult] = field(default_factory=list)


def serialize_payload(payload: Any) -> Any:
    """Return a JSON-serializable structure for arbitrary telemetry objects."""

    if payload is None:
        return None
    if isinstance(payload, datetime):
        return payload.isoformat(timespec="milliseconds")
    if isinstance(payload, Path):
        return str(payload)
    if isinstance(payload, Enum):
        return payload.value
    if is_dataclass(payload):
        return {key: serialize_payload(value) for key, value in asdict(payload).items()}
    if isinstance(payload, dict):
        return {key: serialize_payload(value) for key, value in payload.items()}
    if isinstance(payload, list | tuple | set):
        return [serialize_payload(item) for item in payload]
    return payload


def serialize_many(items: Iterable[Any]) -> list[Any]:
    """Serialize an iterable of payloads for JSON emission.

    Returns:
        list[Any]: Serialized payloads ready for JSON encoding.
    """

    return [serialize_payload(item) for item in items]
