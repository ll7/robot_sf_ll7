"""Scenario-certification adapter for adversarial candidates."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from robot_sf.adversarial.config import CandidateSpec


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
        from robot_sf.scenario_certification import certify_scenario  # type: ignore
    except ImportError:
        if require_certification:
            return not_available_status("scenario_cert.v1 adapter is not available")
        return passed_status("scenario_cert.v1 adapter not available; advisory mode")

    try:
        payload = certify_scenario(scenario_yaml_path, candidate=candidate)
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


def candidate_allowed(status: CertificationStatus, *, require_certification: bool) -> bool:
    """Return whether a candidate may proceed to policy evaluation."""
    if status.passed:
        return True
    return not require_certification and status.status == "not_available"
