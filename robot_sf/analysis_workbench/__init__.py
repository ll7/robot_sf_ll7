"""Analysis-workbench contracts and helpers."""

from robot_sf.analysis_workbench.real_trace_validation_contract import (
    CONTRACT_EVIDENCE_BOUNDARY,
    REAL_TRACE_VALIDATION_CONTRACT_SCHEMA_VERSION,
    PredicateCompatibility,
    RealTraceValidationContractError,
    RealTraceValidationContractReport,
    check_real_trace_validation_contract,
    load_real_trace_validation_contract,
)
from robot_sf.analysis_workbench.simulation_trace_export import (
    SIMULATION_TRACE_EXPORT_SCHEMA_VERSION,
    SimulationTraceExport,
    SimulationTraceExportValidationError,
    load_simulation_trace_export,
    simulation_trace_export_from_dict,
)
from robot_sf.analysis_workbench.trace_annotation import (
    TRACE_ANNOTATION_SET_SCHEMA_VERSION,
    TraceAnnotationSet,
    TraceAnnotationSetValidationError,
    load_trace_annotation_set,
    trace_annotation_set_from_dict,
)
from robot_sf.analysis_workbench.trace_failure_predicates import (
    TRACE_FAILURE_PREDICATE_SCHEMA_VERSION,
    TraceFailurePredicate,
    TraceFailurePredicateDefinition,
    aggregate_trace_failure_predicate_tables,
    build_trace_failure_predicate_definitions,
    extract_trace_failure_predicates,
    render_trace_failure_predicate_markdown,
)

__all__ = [
    "CONTRACT_EVIDENCE_BOUNDARY",
    "REAL_TRACE_VALIDATION_CONTRACT_SCHEMA_VERSION",
    "SIMULATION_TRACE_EXPORT_SCHEMA_VERSION",
    "TRACE_ANNOTATION_SET_SCHEMA_VERSION",
    "TRACE_FAILURE_PREDICATE_SCHEMA_VERSION",
    "PredicateCompatibility",
    "RealTraceValidationContractError",
    "RealTraceValidationContractReport",
    "SimulationTraceExport",
    "SimulationTraceExportValidationError",
    "TraceAnnotationSet",
    "TraceAnnotationSetValidationError",
    "TraceFailurePredicate",
    "TraceFailurePredicateDefinition",
    "aggregate_trace_failure_predicate_tables",
    "build_trace_failure_predicate_definitions",
    "check_real_trace_validation_contract",
    "extract_trace_failure_predicates",
    "load_real_trace_validation_contract",
    "load_simulation_trace_export",
    "load_trace_annotation_set",
    "render_trace_failure_predicate_markdown",
    "simulation_trace_export_from_dict",
    "trace_annotation_set_from_dict",
]
