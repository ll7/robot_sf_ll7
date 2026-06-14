"""Benchmark module for robot social navigation evaluation.

This module provides tools for running benchmarks, collecting metrics,
and analyzing robot navigation performance in social environments.
"""

from robot_sf.benchmark.errors import AggregationMetadataError
from robot_sf.benchmark.forecast_batch import (
    FORECAST_BATCH_SCHEMA_VERSION,
    ActorForecast,
    CoordinateFrame,
    ForecastBatch,
    ForecastBatchProvenance,
    load_forecast_batch,
    save_forecast_batch,
    validate_forecast_batch,
)
from robot_sf.benchmark.forecast_metrics import (
    FORECAST_METRICS_SCHEMA_VERSION,
    ForecastMetricRow,
    evaluate_forecast_batch,
    format_forecast_metrics_markdown,
)
from robot_sf.benchmark.helper_catalog import (
    load_trained_policy,
    prepare_classic_env,
    run_episodes_with_recording,
)
from robot_sf.benchmark.helper_registry import (
    ExampleOrchestrator,
    HelperCapability,
    HelperCategory,
    OrchestratorUsage,
    RegressionCheck,
)

__all__ = [
    "FORECAST_BATCH_SCHEMA_VERSION",
    "FORECAST_METRICS_SCHEMA_VERSION",
    "ActorForecast",
    "AggregationMetadataError",
    "CoordinateFrame",
    "ExampleOrchestrator",
    "ForecastBatch",
    "ForecastBatchProvenance",
    "ForecastMetricRow",
    "HelperCapability",
    "HelperCategory",
    "OrchestratorUsage",
    "RegressionCheck",
    "evaluate_forecast_batch",
    "format_forecast_metrics_markdown",
    "load_forecast_batch",
    "load_trained_policy",
    "prepare_classic_env",
    "run_episodes_with_recording",
    "save_forecast_batch",
    "validate_forecast_batch",
]
