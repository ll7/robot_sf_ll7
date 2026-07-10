"""Scenario certification public API.

Exports are resolved lazily so that importing a lightweight sub-module
(e.g. ``robot_sf.scenario_certification.failure_cause``) does not trigger
shapely, simulator-registry, or other heavy stacks.  The public API
surface is unchanged; all names in ``__all__`` remain accessible via
attribute lookup on the package.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - static type information only
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

_LAZY: dict[str, str] = {
    # criticality_summary
    "CRITICALITY_SUMMARY_SCHEMA_VERSION": "criticality_summary",
    "CriticalitySummaryV1": "criticality_summary",
    "build_criticality_summary_from_compact_evidence": "criticality_summary",
    "build_criticality_summary_from_pilot": "criticality_summary",
    "criticality_summary_to_dict": "criticality_summary",
    "validate_criticality_summary": "criticality_summary",
    # perturbation_family_registry
    "PerturbationFamily": "perturbation_family_registry",
    "perturbation_families": "perturbation_family_registry",
    "perturbation_family": "perturbation_family_registry",
    "supported_perturbation_families": "perturbation_family_registry",
    "validate_perturbation_family_parameters": "perturbation_family_registry",
    # perturbation_preflight
    "PERTURBATION_MANIFEST_SCHEMA_VERSION": "perturbation_preflight",
    "PILOT_MATRIX_SCHEMA_VERSION": "perturbation_preflight",
    "PREFLIGHT_SCHEMA_VERSION": "perturbation_preflight",
    "PerturbationPilotMaterialization": "perturbation_preflight",
    "PerturbationPreflightReport": "perturbation_preflight",
    "PerturbationPreflightResult": "perturbation_preflight",
    "materialize_perturbation_pilot_matrix": "perturbation_preflight",
    "preflight_perturbation_manifest": "perturbation_preflight",
    "preflight_to_dict": "perturbation_preflight",
    # v1
    "CERT_SCHEMA_VERSION": "v1",
    "CertificationSettings": "v1",
    "ScenarioCertificate": "v1",
    "certificate_to_dict": "v1",
    "certify_map_definition": "v1",
    "certify_scenario": "v1",
    "certify_scenario_file": "v1",
}

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


def __getattr__(name: str) -> Any:
    """Resolve public scenario-certification exports on first access.

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
