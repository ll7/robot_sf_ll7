"""Run-tracking telemetry helpers for the imitation-learning pipeline.

The telemetry package centralizes the canonical data models, configuration
objects, and persistence helpers required by `specs/001-performance-tracking`.
It intentionally stays lightweight so both CLI tooling and training scripts can
reuse the same utilities without reimplementing artifact or locking logic.
"""

from .config import RunTrackerConfig
from .manifest_writer import ManifestWriter
from .models import (
    PerformanceRecommendation,
    PerformanceTestResult,
    PerformanceTestStatus,
    PipelineRunRecord,
    PipelineRunStatus,
    RecommendationSeverity,
    StepExecutionEntry,
    StepStatus,
    TelemetrySnapshot,
)
from .progress import PipelineStepDefinition, ProgressTracker
from .run_registry import RunRegistry, generate_run_id

__all__ = [
    "ManifestWriter",
    "PerformanceRecommendation",
    "PerformanceTestResult",
    "PerformanceTestStatus",
    "PipelineRunRecord",
    "PipelineRunStatus",
    "PipelineStepDefinition",
    "ProgressTracker",
    "RecommendationSeverity",
    "RunRegistry",
    "RunTrackerConfig",
    "StepExecutionEntry",
    "StepStatus",
    "TelemetrySnapshot",
    "generate_run_id",
]
