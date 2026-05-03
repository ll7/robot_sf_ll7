"""Scenario-certification adapter for adversarial candidates."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robot_sf.adversarial.config import CandidateSpec


_CLASSIFICATION_TO_ELIGIBILITY = {
    "invalid": "excluded",
    "geometrically_infeasible": "excluded",
    "kinodynamically_infeasible": "excluded",
    "dynamically_overconstrained": "excluded",
    "knife_edge": "stress_only",
    "hard_but_solvable": "eligible",
    "valid": "eligible",
}
_ELIGIBILITY_SEVERITY = {
    "eligible": 0,
    "stress_only": 1,
    "excluded": 2,
}


@dataclass(frozen=True)
class CertificationStatus:
    """Certification outcome for a generated candidate."""

    schema_version: str
    status: str
    reason: str
    details: dict[str, Any]

    @property
    def passed(self) -> bool:
        """Return True when the candidate is certified valid."""
        return self.status == "passed"

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable payload."""
        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "reason": self.reason,
            "details": dict(self.details),
        }


def passed_status(reason: str = "certification not required") -> CertificationStatus:
    """Return a passing advisory status."""
    return CertificationStatus(
        schema_version="scenario_cert.v1",
        status="passed",
        reason=reason,
        details={},
    )


def failed_status(reason: str, *, details: dict[str, Any] | None = None) -> CertificationStatus:
    """Return a failed certification status."""
    return CertificationStatus(
        schema_version="scenario_cert.v1",
        status="failed",
        reason=reason,
        details=details or {},
    )


def not_available_status(reason: str) -> CertificationStatus:
    """Return a not-available certification status."""
    return CertificationStatus(
        schema_version="scenario_cert.v1",
        status="not_available",
        reason=reason,
        details={},
    )


def certify_candidate(
    candidate: CandidateSpec,
    *,
    scenario_yaml_path: Path,
    require_certification: bool,
) -> CertificationStatus:
    """Certify a generated candidate using ``scenario_cert.v1`` when available.

    The scenario-certification package is a planned prerequisite, not yet a
    stable in-repo API. When strict certification is requested and no adapter is
    available, this function fails closed with a ``not_available`` status.
    """
    try:
        scenario_certification = import_module("robot_sf.scenario_certification")
    except ImportError:
        if require_certification:
            return not_available_status("scenario_cert.v1 adapter is not available")
        return passed_status("scenario_cert.v1 adapter not available; advisory mode")

    if not _adapter_supports_candidate(scenario_certification):
        if require_certification:
            return not_available_status("scenario_cert.v1 adapter is not available")
        return passed_status("scenario_cert.v1 adapter not available; advisory mode")

    try:
        payload = _run_scenario_certification_adapter(
            scenario_certification,
            candidate=candidate,
            scenario_yaml_path=scenario_yaml_path,
        )
    except Exception as exc:  # pragma: no cover - defensive against future adapter errors
        return failed_status(
            "scenario_cert.v1 raised during certification", details={"error": repr(exc)}
        )

    if not isinstance(payload, dict):
        return failed_status("scenario_cert.v1 returned a non-mapping payload")
    status = str(payload.get("status", "")).strip().lower()
    reason = str(payload.get("reason", status or "unknown"))
    details = payload.get("details", {})
    if not isinstance(details, dict):
        details = {"raw_details": details}
    if status in {"passed", "valid", "ok"}:
        return CertificationStatus("scenario_cert.v1", "passed", reason, details)
    if status in {"not_available", "unavailable"}:
        return CertificationStatus("scenario_cert.v1", "not_available", reason, details)
    return CertificationStatus("scenario_cert.v1", "failed", reason, details)


def _adapter_supports_candidate(scenario_certification: Any) -> bool:
    """Return whether the available adapter can certify a generated candidate."""
    certify_scenario_file = getattr(scenario_certification, "certify_scenario_file", None)
    certificate_to_dict = getattr(scenario_certification, "certificate_to_dict", None)
    if callable(certify_scenario_file) and callable(certificate_to_dict):
        return True

    certify_scenario = getattr(scenario_certification, "certify_scenario", None)
    if not callable(certify_scenario):
        return True

    signature = inspect.signature(certify_scenario)
    return "candidate" in signature.parameters or any(
        param.kind is inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()
    )


def _run_scenario_certification_adapter(
    scenario_certification: Any,
    *,
    candidate: CandidateSpec,
    scenario_yaml_path: Path,
) -> dict[str, Any]:
    """Run the available scenario certification API and normalize its payload."""
    certify_scenario_file = getattr(scenario_certification, "certify_scenario_file", None)
    certificate_to_dict = getattr(scenario_certification, "certificate_to_dict", None)
    if callable(certify_scenario_file) and callable(certificate_to_dict):
        certificates = certify_scenario_file(scenario_yaml_path)
        payloads = [certificate_to_dict(certificate) for certificate in certificates]
        if not payloads:
            return {"status": "failed", "reason": "scenario_cert.v1 returned no certificates"}
        worst = max(payloads, key=_certificate_payload_severity)
        eligibility = _certificate_payload_eligibility(worst)
        status = "failed" if eligibility in {"", "excluded"} else "passed"
        reasons = worst.get("reasons", [])
        reason = "; ".join(str(reason) for reason in reasons) if isinstance(reasons, list) else ""
        return {
            "status": status,
            "reason": reason or str(worst.get("classification", status)),
            "details": {"certificates": payloads},
        }

    certify_scenario = getattr(scenario_certification, "certify_scenario", None)
    if callable(certify_scenario):
        return certify_scenario(scenario_yaml_path, candidate=candidate)
    return {"status": "not_available", "reason": "scenario_cert.v1 adapter is not available"}


def _normalize_certificate_text(value: Any) -> str:
    """Normalize optional certificate enum fields without stringifying null values."""
    if not isinstance(value, str):
        return ""
    return value.strip().lower()


def _certificate_payload_eligibility(payload: dict[str, Any]) -> str:
    """Return benchmark eligibility, inferring it from classification when omitted."""
    eligibility = _normalize_certificate_text(payload.get("benchmark_eligibility"))
    if eligibility in _ELIGIBILITY_SEVERITY:
        return eligibility
    classification = _normalize_certificate_text(payload.get("classification"))
    return _CLASSIFICATION_TO_ELIGIBILITY.get(classification, "")


def _certificate_payload_severity(payload: dict[str, Any]) -> tuple[int, str]:
    """Sort certificates by benchmark eligibility severity, failing closed on unknown values."""
    eligibility = _certificate_payload_eligibility(payload)
    return (_ELIGIBILITY_SEVERITY.get(eligibility, 3), eligibility)


def candidate_allowed(status: CertificationStatus, *, require_certification: bool) -> bool:
    """Return whether a candidate may proceed to policy evaluation."""
    if status.passed:
        return True
    return not require_certification and status.status == "not_available"
