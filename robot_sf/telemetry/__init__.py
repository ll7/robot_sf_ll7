"""Run-tracking telemetry helpers for the imitation-learning pipeline.

The telemetry package centralizes the canonical data models, configuration
objects, and persistence helpers required by `specs/001-performance-tracking`.
It intentionally stays lightweight so both CLI tooling and training scripts can
reuse the same utilities without reimplementing artifact or locking logic.

Exports are resolved lazily so importing a narrow telemetry helper does not
initialize optional TensorBoard or visualization dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - static type information only
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
        save_rgba_png,
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
    "save_rgba_png",
]

_EXPORT_MODULES = {
    "DEFAULT_TELEMETRY_METRICS": "visualization",
    "ManifestWriter": "manifest_writer",
    "PerformanceRecommendation": "models",
    "PerformanceTestResult": "models",
    "PerformanceTestStatus": "models",
    "PipelineRunRecord": "models",
    "PipelineRunStatus": "models",
    "PipelineStepDefinition": "progress",
    "ProgressTracker": "progress",
    "RecommendationEngine": "recommendations",
    "RecommendationRules": "recommendations",
    "RecommendationSeverity": "models",
    "RunHistoryEntry": "history",
    "RunRegistry": "run_registry",
    "RunTrackerConfig": "config",
    "StepExecutionEntry": "models",
    "StepStatus": "models",
    "TelemetrySampler": "sampler",
    "TelemetrySnapshot": "models",
    "TensorBoardAdapter": "tensorboard_adapter",
    "export_combined_image": "visualization",
    "generate_run_id": "run_registry",
    "iter_telemetry_snapshots": "tensorboard_adapter",
    "list_runs": "history",
    "load_run": "history",
    "make_surface_from_rgba": "visualization",
    "render_metric_panel": "visualization",
    "save_rgba_png": "visualization",
}
_SUBMODULES = frozenset(
    {
        "config",
        "gpu",
        "history",
        "manifest_writer",
        "models",
        "pane",
        "progress",
        "recommendations",
        "run_registry",
        "sampler",
        "tensorboard_adapter",
        "visualization",
    }
)


def __getattr__(name: str) -> Any:
    """Load a public telemetry export or submodule only when it is requested.

    Returns:
        The requested telemetry module or export.
    """
    module_name = _EXPORT_MODULES.get(name)
    if module_name is not None:
        value = getattr(import_module(f".{module_name}", __name__), name)
    elif name in _SUBMODULES:
        value = import_module(f".{name}", __name__)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Include lazily exported telemetry names in interactive discovery.

    Returns:
        Available telemetry attribute names.
    """
    return sorted(set(globals()) | set(__all__) | _SUBMODULES)
