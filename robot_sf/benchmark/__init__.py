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
    from robot_sf.benchmark.case_dossier_figure import (
        CASE_DOSSIER_INPUT_SCHEMA_VERSION,
        CASE_DOSSIER_MANIFEST_SCHEMA_VERSION,
        CASE_DOSSIER_RENDERER_VERSION,
        CaseDossierBundle,
        CaseDossierError,
        render_case_dossier,
        validate_case_dossier_manifest,
    )
    from robot_sf.benchmark.errors import AggregationMetadataError
    from robot_sf.benchmark.forecast.forecast_batch import (
        FORECAST_BATCH_SCHEMA_VERSION,
        ActorForecast,
        CoordinateFrame,
        ForecastBatch,
        ForecastBatchProvenance,
        load_forecast_batch,
        save_forecast_batch,
        validate_forecast_batch,
    )
    from robot_sf.benchmark.forecast.forecast_calibration_report import (
        FORECAST_CALIBRATION_REPORT_SCHEMA_VERSION,
        build_forecast_calibration_report,
        format_forecast_calibration_markdown,
        write_forecast_calibration_report,
    )
    from robot_sf.benchmark.forecast.forecast_conformal_pilot import (
        FORECAST_CONFORMAL_PILOT_SCHEMA_VERSION,
        build_forecast_conformal_pilot_report,
        format_forecast_conformal_pilot_markdown,
        write_forecast_conformal_pilot_report,
    )
    from robot_sf.benchmark.forecast.forecast_dataset_recorder import (
        DEFAULT_FORECAST_DATASET_ID,
        FORECAST_DATASET_SCHEMA_VERSION,
        ForecastDatasetRecordResult,
        record_forecast_dataset_from_trace_exports,
        validate_forecast_dataset_manifest,
    )
    from robot_sf.benchmark.forecast.forecast_metrics import (
        FORECAST_METRICS_SCHEMA_VERSION,
        ForecastMetricRow,
        evaluate_forecast_batch,
        format_forecast_metrics_markdown,
    )
    from robot_sf.benchmark.forecast.forecast_observation_adapters import (
        ForecastActorObservation,
        ForecastObservationAdapter,
        ForecastObservationBatch,
        OracleFullStateForecastAdapter,
        TrackedAgentsForecastAdapter,
        build_constant_velocity_forecast_batch,
    )
    from robot_sf.benchmark.forecast.forecast_transferability_stress_matrix import (
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
    from robot_sf.benchmark.result_interpretation_packet import (
        SCHEMA_VERSION as RESULT_INTERPRETATION_PACKET_SCHEMA_VERSION,
    )
    from robot_sf.benchmark.result_interpretation_packet import (
        Evidence,
        FigureVisualContract,
        ResultInterpretationPacket,
        ResultInterpretationPacketError,
        build_and_validate_packet,
        compute_packet_digest,
        compute_post_review_digest,
        load_result_interpretation_packet,
        render_caption,
        validate_packet,
        validate_review_binding,
        write_caption,
        write_checksum_manifest,
        write_deterministic_json,
        write_review_report,
    )
    from robot_sf.benchmark.scenario.scenario_failure_cause import (
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
    # case_dossier_figure
    "CASE_DOSSIER_INPUT_SCHEMA_VERSION": "case_dossier_figure",
    "CASE_DOSSIER_MANIFEST_SCHEMA_VERSION": "case_dossier_figure",
    "CASE_DOSSIER_RENDERER_VERSION": "case_dossier_figure",
    "CaseDossierBundle": "case_dossier_figure",
    "CaseDossierError": "case_dossier_figure",
    "render_case_dossier": "case_dossier_figure",
    "validate_case_dossier_manifest": "case_dossier_figure",
    # errors
    "AggregationMetadataError": "errors",
    # forecast_batch
    "FORECAST_BATCH_SCHEMA_VERSION": "forecast.forecast_batch",
    "ActorForecast": "forecast.forecast_batch",
    "CoordinateFrame": "forecast.forecast_batch",
    "ForecastBatch": "forecast.forecast_batch",
    "ForecastBatchProvenance": "forecast.forecast_batch",
    "load_forecast_batch": "forecast.forecast_batch",
    "save_forecast_batch": "forecast.forecast_batch",
    "validate_forecast_batch": "forecast.forecast_batch",
    # forecast_calibration_report
    "FORECAST_CALIBRATION_REPORT_SCHEMA_VERSION": "forecast.forecast_calibration_report",
    "build_forecast_calibration_report": "forecast.forecast_calibration_report",
    "format_forecast_calibration_markdown": "forecast.forecast_calibration_report",
    "write_forecast_calibration_report": "forecast.forecast_calibration_report",
    # forecast_conformal_pilot
    "FORECAST_CONFORMAL_PILOT_SCHEMA_VERSION": "forecast.forecast_conformal_pilot",
    "build_forecast_conformal_pilot_report": "forecast.forecast_conformal_pilot",
    "format_forecast_conformal_pilot_markdown": "forecast.forecast_conformal_pilot",
    "write_forecast_conformal_pilot_report": "forecast.forecast_conformal_pilot",
    # forecast_dataset_recorder
    "DEFAULT_FORECAST_DATASET_ID": "forecast.forecast_dataset_recorder",
    "FORECAST_DATASET_SCHEMA_VERSION": "forecast.forecast_dataset_recorder",
    "ForecastDatasetRecordResult": "forecast.forecast_dataset_recorder",
    "record_forecast_dataset_from_trace_exports": "forecast.forecast_dataset_recorder",
    "validate_forecast_dataset_manifest": "forecast.forecast_dataset_recorder",
    # forecast_metrics
    "FORECAST_METRICS_SCHEMA_VERSION": "forecast.forecast_metrics",
    "ForecastMetricRow": "forecast.forecast_metrics",
    "evaluate_forecast_batch": "forecast.forecast_metrics",
    "format_forecast_metrics_markdown": "forecast.forecast_metrics",
    # forecast_observation_adapters
    "ForecastActorObservation": "forecast.forecast_observation_adapters",
    "ForecastObservationAdapter": "forecast.forecast_observation_adapters",
    "ForecastObservationBatch": "forecast.forecast_observation_adapters",
    "OracleFullStateForecastAdapter": "forecast.forecast_observation_adapters",
    "TrackedAgentsForecastAdapter": "forecast.forecast_observation_adapters",
    "build_constant_velocity_forecast_batch": "forecast.forecast_observation_adapters",
    # forecast_transferability_stress_matrix
    "DEFAULT_TRANSFER_DIMENSIONS": "forecast.forecast_transferability_stress_matrix",
    "FORECAST_TRANSFERABILITY_STRESS_MATRIX_SCHEMA_VERSION": "forecast.forecast_transferability_stress_matrix",
    "build_forecast_transferability_stress_matrix": "forecast.forecast_transferability_stress_matrix",
    "format_forecast_transferability_stress_markdown": "forecast.forecast_transferability_stress_matrix",
    "write_forecast_transferability_stress_matrix": "forecast.forecast_transferability_stress_matrix",
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
    # result_interpretation_packet
    "RESULT_INTERPRETATION_PACKET_SCHEMA_VERSION": "result_interpretation_packet",
    "Evidence": "result_interpretation_packet",
    "FigureVisualContract": "result_interpretation_packet",
    "ResultInterpretationPacket": "result_interpretation_packet",
    "ResultInterpretationPacketError": "result_interpretation_packet",
    "build_and_validate_packet": "result_interpretation_packet",
    "compute_packet_digest": "result_interpretation_packet",
    "compute_post_review_digest": "result_interpretation_packet",
    "load_result_interpretation_packet": "result_interpretation_packet",
    "render_caption": "result_interpretation_packet",
    "validate_packet": "result_interpretation_packet",
    "validate_review_binding": "result_interpretation_packet",
    "write_caption": "result_interpretation_packet",
    "write_checksum_manifest": "result_interpretation_packet",
    "write_deterministic_json": "result_interpretation_packet",
    "write_review_report": "result_interpretation_packet",
    # scenario_failure_cause
    "SCENARIO_FAILURE_CAUSE_SCHEMA_VERSION": "scenario.scenario_failure_cause",
    "VERDICT_DYNAMIC_BLOCKING_OR_DEADLOCK": "scenario.scenario_failure_cause",
    "VERDICT_INDETERMINATE": "scenario.scenario_failure_cause",
    "VERDICT_INFEASIBLE_ROUTE": "scenario.scenario_failure_cause",
    "VERDICT_PLANNER_LIMITED": "scenario.scenario_failure_cause",
    "VERDICT_TIME_LIMITED": "scenario.scenario_failure_cause",
    "VERDICT_VEHICLE_INFEASIBLE": "scenario.scenario_failure_cause",
    "ScenarioFailureCause": "scenario.scenario_failure_cause",
    "ScenarioFailureDiagnostics": "scenario.scenario_failure_cause",
    "classify_scenario_failure_cause": "scenario.scenario_failure_cause",
    "diagnostics_from_mapping": "scenario.scenario_failure_cause",
}

__all__ = [
    "AMMV_BENCHMARK_PROTOCOL_PATH",
    "CANONICAL_METRICS",
    "CANONICAL_METRIC_LAYERS",
    "CASE_DOSSIER_INPUT_SCHEMA_VERSION",
    "CASE_DOSSIER_MANIFEST_SCHEMA_VERSION",
    "CASE_DOSSIER_RENDERER_VERSION",
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
    "RESULT_INTERPRETATION_PACKET_SCHEMA_VERSION",
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
    "CaseDossierBundle",
    "CaseDossierError",
    "ClaimRules",
    "CoordinateFrame",
    "Evidence",
    "ExampleOrchestrator",
    "FigureVisualContract",
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
    "ResultInterpretationPacket",
    "ResultInterpretationPacketError",
    "ScenarioFailureCause",
    "ScenarioFailureDiagnostics",
    "TrackedAgentsForecastAdapter",
    "build_and_validate_packet",
    "build_assurance_fragment",
    "build_constant_velocity_forecast_batch",
    "build_forecast_calibration_report",
    "build_forecast_conformal_pilot_report",
    "build_forecast_transferability_stress_matrix",
    "build_metric_layer_summary",
    "classify_scenario_failure_cause",
    "compute_packet_digest",
    "compute_post_review_digest",
    "diagnostics_from_mapping",
    "evaluate_forecast_batch",
    "format_forecast_calibration_markdown",
    "format_forecast_conformal_pilot_markdown",
    "format_forecast_metrics_markdown",
    "format_forecast_transferability_stress_markdown",
    "load_benchmark_protocol",
    "load_forecast_batch",
    "load_result_interpretation_packet",
    "load_trained_policy",
    "prepare_classic_env",
    "record_forecast_dataset_from_trace_exports",
    "render_assurance_fragment_to_markdown",
    "render_assurance_fragment_to_svg",
    "render_caption",
    "render_case_dossier",
    "run_episodes_with_recording",
    "save_forecast_batch",
    "validate_assurance_fragment",
    "validate_benchmark_protocol_payload",
    "validate_case_dossier_manifest",
    "validate_forecast_batch",
    "validate_forecast_dataset_manifest",
    "validate_packet",
    "validate_review_binding",
    "write_assurance_fragment",
    "write_caption",
    "write_checksum_manifest",
    "write_deterministic_json",
    "write_forecast_calibration_report",
    "write_forecast_conformal_pilot_report",
    "write_forecast_transferability_stress_matrix",
    "write_review_report",
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
