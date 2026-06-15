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
from robot_sf.benchmark.forecast_dataset_recorder import (
    DEFAULT_FORECAST_DATASET_ID,
    FORECAST_DATASET_SCHEMA_VERSION,
    ForecastDatasetRecordResult,
    record_forecast_dataset_from_trace_exports,
    validate_forecast_dataset_manifest,
)
from robot_sf.benchmark.forecast_metrics import (
    FORECAST_METRICS_SCHEMA_VERSION,
    ForecastMetricRow,
    evaluate_forecast_batch,
    format_forecast_metrics_markdown,
)
from robot_sf.benchmark.forecast_observation_adapters import (
    ForecastActorObservation,
    ForecastObservationAdapter,
    ForecastObservationBatch,
    OracleFullStateForecastAdapter,
    TrackedAgentsForecastAdapter,
    build_constant_velocity_forecast_batch,
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
    "DEFAULT_FORECAST_DATASET_ID",
    "FORECAST_BATCH_SCHEMA_VERSION",
    "FORECAST_DATASET_SCHEMA_VERSION",
    "FORECAST_METRICS_SCHEMA_VERSION",
    "ActorForecast",
    "AggregationMetadataError",
    "CoordinateFrame",
    "ExampleOrchestrator",
    "ForecastActorObservation",
    "ForecastBatch",
    "ForecastBatchProvenance",
    "ForecastDatasetRecordResult",
    "ForecastMetricRow",
    "ForecastObservationAdapter",
    "ForecastObservationBatch",
    "HelperCapability",
    "HelperCategory",
    "OracleFullStateForecastAdapter",
    "OrchestratorUsage",
    "RegressionCheck",
    "TrackedAgentsForecastAdapter",
    "build_constant_velocity_forecast_batch",
    "evaluate_forecast_batch",
    "format_forecast_metrics_markdown",
    "load_forecast_batch",
    "load_trained_policy",
    "prepare_classic_env",
    "record_forecast_dataset_from_trace_exports",
    "run_episodes_with_recording",
    "save_forecast_batch",
    "validate_forecast_batch",
    "validate_forecast_dataset_manifest",
]
