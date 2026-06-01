"""Scenario certification public API."""

from robot_sf.scenario_certification.criticality_summary import (
    CRITICALITY_SUMMARY_SCHEMA_VERSION,
    CriticalitySummaryV1,
    build_criticality_summary_from_compact_evidence,
    build_criticality_summary_from_pilot,
    criticality_summary_to_dict,
    validate_criticality_summary,
)
from robot_sf.scenario_certification.perturbation_family_registry import (
    PerturbationFamily,
    perturbation_families,
    perturbation_family,
    supported_perturbation_families,
    validate_perturbation_family_parameters,
)
from robot_sf.scenario_certification.perturbation_preflight import (
    PERTURBATION_MANIFEST_SCHEMA_VERSION,
    PILOT_MATRIX_SCHEMA_VERSION,
    PREFLIGHT_SCHEMA_VERSION,
    PerturbationPilotMaterialization,
    PerturbationPreflightReport,
    PerturbationPreflightResult,
    materialize_perturbation_pilot_matrix,
    preflight_perturbation_manifest,
    preflight_to_dict,
)
from robot_sf.scenario_certification.v1 import (
    CERT_SCHEMA_VERSION,
    CertificationSettings,
    ScenarioCertificate,
    certificate_to_dict,
    certify_map_definition,
    certify_scenario,
    certify_scenario_file,
)

__all__ = [
    "CERT_SCHEMA_VERSION",
    "CRITICALITY_SUMMARY_SCHEMA_VERSION",
    "PERTURBATION_MANIFEST_SCHEMA_VERSION",
    "PILOT_MATRIX_SCHEMA_VERSION",
    "PREFLIGHT_SCHEMA_VERSION",
    "CertificationSettings",
    "CriticalitySummaryV1",
    "PerturbationFamily",
    "PerturbationPilotMaterialization",
    "PerturbationPreflightReport",
    "PerturbationPreflightResult",
    "ScenarioCertificate",
    "build_criticality_summary_from_compact_evidence",
    "build_criticality_summary_from_pilot",
    "certificate_to_dict",
    "certify_map_definition",
    "certify_scenario",
    "certify_scenario_file",
    "criticality_summary_to_dict",
    "materialize_perturbation_pilot_matrix",
    "perturbation_families",
    "perturbation_family",
    "preflight_perturbation_manifest",
    "preflight_to_dict",
    "supported_perturbation_families",
    "validate_criticality_summary",
    "validate_perturbation_family_parameters",
]
