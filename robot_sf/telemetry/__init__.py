"""Run-tracking telemetry helpers for the imitation-learning pipeline.

The telemetry package centralizes the canonical data models, configuration
objects, and persistence helpers required by `specs/001-performance-tracking`.
It intentionally stays lightweight so both CLI tooling and training scripts can
reuse the same utilities without reimplementing artifact or locking logic.
"""

from .config import RunTrackerConfig
from .history import RunHistoryEntry, list_runs, load_run
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
from .recommendations import RecommendationEngine, RecommendationRules
from .run_registry import RunRegistry, generate_run_id
from .sampler import TelemetrySampler
from .tensorboard_adapter import TensorBoardAdapter, iter_telemetry_snapshots
from .visualization import (
    DEFAULT_TELEMETRY_METRICS,
    export_combined_image,
    make_surface_from_rgba,
    render_metric_panel,
)

__all__ = [
    "DEFAULT_TELEMETRY_METRICS",
    "ManifestWriter",
    "PerformanceRecommendation",
    "PerformanceTestResult",
    "PerformanceTestStatus",
    "PipelineRunRecord",
    "PipelineRunStatus",
    "PipelineStepDefinition",
    "ProgressTracker",
    "RecommendationEngine",
    "RecommendationRules",
    "RecommendationSeverity",
    "RunHistoryEntry",
    "RunRegistry",
    "RunTrackerConfig",
    "StepExecutionEntry",
    "StepStatus",
    "TelemetrySampler",
    "TelemetrySnapshot",
    "TensorBoardAdapter",
    "export_combined_image",
    "generate_run_id",
    "iter_telemetry_snapshots",
    "list_runs",
    "load_run",
    "make_surface_from_rgba",
    "render_metric_panel",
]
