"""Benchmark module for robot social navigation evaluation.

This module provides tools for running benchmarks, collecting metrics,
and analyzing robot navigation performance in social environments.

Exports are resolved lazily so that importing a lightweight sub-module
(e.g. ``robot_sf.benchmark.errors``) does not trigger TensorFlow,
simulator-registry, or other heavy stacks.  The public API surface is
unchanged; all names in ``__all__`` remain accessible via attribute lookup
on the package.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - static type information only
    from robot_sf.benchmark.assurance_fragment import (
        build_assurance_fragment,
        render_assurance_fragment_to_markdown,
        render_assurance_fragment_to_svg,
        validate_assurance_fragment,
        write_assurance_fragment,
    )
    from robot_sf.benchmark.benchmark_protocol import (
        AMMV_BENCHMARK_PROTOCOL_PATH,
        BenchmarkProtocolError,
        BenchmarkProtocolManifest,
        ClaimRules,
        load_benchmark_protocol,
        validate_benchmark_protocol_payload,
    )
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
    from robot_sf.benchmark.forecast_calibration_report import (
        FORECAST_CALIBRATION_REPORT_SCHEMA_VERSION,
        build_forecast_calibration_report,
        format_forecast_calibration_markdown,
        write_forecast_calibration_report,
    )
    from robot_sf.benchmark.forecast_conformal_pilot import (
        FORECAST_CONFORMAL_PILOT_SCHEMA_VERSION,
        build_forecast_conformal_pilot_report,
        format_forecast_conformal_pilot_markdown,
        write_forecast_conformal_pilot_report,
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
    from robot_sf.benchmark.forecast_transferability_stress_matrix import (
        DEFAULT_TRANSFER_DIMENSIONS,
        FORECAST_TRANSFERABILITY_STRESS_MATRIX_SCHEMA_VERSION,
        build_forecast_transferability_stress_matrix,
        format_forecast_transferability_stress_markdown,
        write_forecast_transferability_stress_matrix,
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
    from robot_sf.benchmark.metric_layers import (
        CANONICAL_METRIC_LAYERS,
        CANONICAL_METRICS,
        LAYER_ORDER,
        METRIC_LAYER_SCHEMA_VERSION,
        MetricDefinition,
        MetricLayerDefinition,
        build_metric_layer_summary,
    )
    from robot_sf.benchmark.scenario_failure_cause import (
        SCENARIO_FAILURE_CAUSE_SCHEMA_VERSION,
        VERDICT_DYNAMIC_BLOCKING_OR_DEADLOCK,
        VERDICT_INDETERMINATE,
        VERDICT_INFEASIBLE_ROUTE,
        VERDICT_PLANNER_LIMITED,
        VERDICT_TIME_LIMITED,
        VERDICT_VEHICLE_INFEASIBLE,
        ScenarioFailureCause,
        ScenarioFailureDiagnostics,
        classify_scenario_failure_cause,
        diagnostics_from_mapping,
    )

# Maps each public name to its source sub-module (relative to this package).
_LAZY: dict[str, str] = {
    # assurance_fragment
    "build_assurance_fragment": "assurance_fragment",
    "render_assurance_fragment_to_markdown": "assurance_fragment",
    "render_assurance_fragment_to_svg": "assurance_fragment",
    "validate_assurance_fragment": "assurance_fragment",
    "write_assurance_fragment": "assurance_fragment",
    # benchmark_protocol
    "AMMV_BENCHMARK_PROTOCOL_PATH": "benchmark_protocol",
    "BenchmarkProtocolError": "benchmark_protocol",
    "BenchmarkProtocolManifest": "benchmark_protocol",
    "ClaimRules": "benchmark_protocol",
    "load_benchmark_protocol": "benchmark_protocol",
    "validate_benchmark_protocol_payload": "benchmark_protocol",
    # errors
    "AggregationMetadataError": "errors",
    # forecast_batch
    "FORECAST_BATCH_SCHEMA_VERSION": "forecast_batch",
    "ActorForecast": "forecast_batch",
    "CoordinateFrame": "forecast_batch",
    "ForecastBatch": "forecast_batch",
    "ForecastBatchProvenance": "forecast_batch",
    "load_forecast_batch": "forecast_batch",
    "save_forecast_batch": "forecast_batch",
    "validate_forecast_batch": "forecast_batch",
    # forecast_calibration_report
    "FORECAST_CALIBRATION_REPORT_SCHEMA_VERSION": "forecast_calibration_report",
    "build_forecast_calibration_report": "forecast_calibration_report",
    "format_forecast_calibration_markdown": "forecast_calibration_report",
    "write_forecast_calibration_report": "forecast_calibration_report",
    # forecast_conformal_pilot
    "FORECAST_CONFORMAL_PILOT_SCHEMA_VERSION": "forecast_conformal_pilot",
    "build_forecast_conformal_pilot_report": "forecast_conformal_pilot",
    "format_forecast_conformal_pilot_markdown": "forecast_conformal_pilot",
    "write_forecast_conformal_pilot_report": "forecast_conformal_pilot",
    # forecast_dataset_recorder
    "DEFAULT_FORECAST_DATASET_ID": "forecast_dataset_recorder",
    "FORECAST_DATASET_SCHEMA_VERSION": "forecast_dataset_recorder",
    "ForecastDatasetRecordResult": "forecast_dataset_recorder",
    "record_forecast_dataset_from_trace_exports": "forecast_dataset_recorder",
    "validate_forecast_dataset_manifest": "forecast_dataset_recorder",
    # forecast_metrics
    "FORECAST_METRICS_SCHEMA_VERSION": "forecast_metrics",
    "ForecastMetricRow": "forecast_metrics",
    "evaluate_forecast_batch": "forecast_metrics",
    "format_forecast_metrics_markdown": "forecast_metrics",
    # forecast_observation_adapters
    "ForecastActorObservation": "forecast_observation_adapters",
    "ForecastObservationAdapter": "forecast_observation_adapters",
    "ForecastObservationBatch": "forecast_observation_adapters",
    "OracleFullStateForecastAdapter": "forecast_observation_adapters",
    "TrackedAgentsForecastAdapter": "forecast_observation_adapters",
    "build_constant_velocity_forecast_batch": "forecast_observation_adapters",
    # forecast_transferability_stress_matrix
    "DEFAULT_TRANSFER_DIMENSIONS": "forecast_transferability_stress_matrix",
    "FORECAST_TRANSFERABILITY_STRESS_MATRIX_SCHEMA_VERSION": "forecast_transferability_stress_matrix",
    "build_forecast_transferability_stress_matrix": "forecast_transferability_stress_matrix",
    "format_forecast_transferability_stress_markdown": "forecast_transferability_stress_matrix",
    "write_forecast_transferability_stress_matrix": "forecast_transferability_stress_matrix",
    # helper_catalog
    "load_trained_policy": "helper_catalog",
    "prepare_classic_env": "helper_catalog",
    "run_episodes_with_recording": "helper_catalog",
    # helper_registry
    "ExampleOrchestrator": "helper_registry",
    "HelperCapability": "helper_registry",
    "HelperCategory": "helper_registry",
    "OrchestratorUsage": "helper_registry",
    "RegressionCheck": "helper_registry",
    # metric_layers
    "CANONICAL_METRIC_LAYERS": "metric_layers",
    "CANONICAL_METRICS": "metric_layers",
    "LAYER_ORDER": "metric_layers",
    "METRIC_LAYER_SCHEMA_VERSION": "metric_layers",
    "MetricDefinition": "metric_layers",
    "MetricLayerDefinition": "metric_layers",
    "build_metric_layer_summary": "metric_layers",
    # scenario_failure_cause
    "SCENARIO_FAILURE_CAUSE_SCHEMA_VERSION": "scenario_failure_cause",
    "VERDICT_DYNAMIC_BLOCKING_OR_DEADLOCK": "scenario_failure_cause",
    "VERDICT_INDETERMINATE": "scenario_failure_cause",
    "VERDICT_INFEASIBLE_ROUTE": "scenario_failure_cause",
    "VERDICT_PLANNER_LIMITED": "scenario_failure_cause",
    "VERDICT_TIME_LIMITED": "scenario_failure_cause",
    "VERDICT_VEHICLE_INFEASIBLE": "scenario_failure_cause",
    "ScenarioFailureCause": "scenario_failure_cause",
    "ScenarioFailureDiagnostics": "scenario_failure_cause",
    "classify_scenario_failure_cause": "scenario_failure_cause",
    "diagnostics_from_mapping": "scenario_failure_cause",
}

__all__ = [
    "AMMV_BENCHMARK_PROTOCOL_PATH",
    "CANONICAL_METRICS",
    "CANONICAL_METRIC_LAYERS",
    "DEFAULT_FORECAST_DATASET_ID",
    "DEFAULT_TRANSFER_DIMENSIONS",
    "FORECAST_BATCH_SCHEMA_VERSION",
    "FORECAST_CALIBRATION_REPORT_SCHEMA_VERSION",
    "FORECAST_CONFORMAL_PILOT_SCHEMA_VERSION",
    "FORECAST_DATASET_SCHEMA_VERSION",
    "FORECAST_METRICS_SCHEMA_VERSION",
    "FORECAST_TRANSFERABILITY_STRESS_MATRIX_SCHEMA_VERSION",
    "LAYER_ORDER",
    "METRIC_LAYER_SCHEMA_VERSION",
    "SCENARIO_FAILURE_CAUSE_SCHEMA_VERSION",
    "VERDICT_DYNAMIC_BLOCKING_OR_DEADLOCK",
    "VERDICT_INDETERMINATE",
    "VERDICT_INFEASIBLE_ROUTE",
    "VERDICT_PLANNER_LIMITED",
    "VERDICT_TIME_LIMITED",
    "VERDICT_VEHICLE_INFEASIBLE",
    "ActorForecast",
    "AggregationMetadataError",
    "BenchmarkProtocolError",
    "BenchmarkProtocolManifest",
    "ClaimRules",
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
    "MetricDefinition",
    "MetricLayerDefinition",
    "OracleFullStateForecastAdapter",
    "OrchestratorUsage",
    "RegressionCheck",
    "ScenarioFailureCause",
    "ScenarioFailureDiagnostics",
    "TrackedAgentsForecastAdapter",
    "build_assurance_fragment",
    "build_constant_velocity_forecast_batch",
    "build_forecast_calibration_report",
    "build_forecast_conformal_pilot_report",
    "build_forecast_transferability_stress_matrix",
    "build_metric_layer_summary",
    "classify_scenario_failure_cause",
    "diagnostics_from_mapping",
    "evaluate_forecast_batch",
    "format_forecast_calibration_markdown",
    "format_forecast_conformal_pilot_markdown",
    "format_forecast_metrics_markdown",
    "format_forecast_transferability_stress_markdown",
    "load_benchmark_protocol",
    "load_forecast_batch",
    "load_trained_policy",
    "prepare_classic_env",
    "record_forecast_dataset_from_trace_exports",
    "render_assurance_fragment_to_markdown",
    "render_assurance_fragment_to_svg",
    "run_episodes_with_recording",
    "save_forecast_batch",
    "validate_assurance_fragment",
    "validate_benchmark_protocol_payload",
    "validate_forecast_batch",
    "validate_forecast_dataset_manifest",
    "write_assurance_fragment",
    "write_forecast_calibration_report",
    "write_forecast_conformal_pilot_report",
    "write_forecast_transferability_stress_matrix",
]


def __getattr__(name: str) -> Any:
    """Resolve public benchmark exports on first access.

    Returns:
        The requested attribute from its source sub-module.

    Raises:
        AttributeError: If ``name`` is not a known public export.
    """
    if name in _LAZY:
        module = import_module(f".{_LAZY[name]}", __name__)
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazily exported names in interactive discovery.

    Returns:
        Available package attribute names.
    """
    return sorted(set(globals()) | set(__all__))
