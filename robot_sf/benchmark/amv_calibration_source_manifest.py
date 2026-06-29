"""Metadata-only AMV actuation calibration source provenance manifest.

Issue #1585 is blocked on an accepted source for calibrated autonomous
micromobility vehicle (AMV) actuation evidence. This module checks the local
manifest contract for source identification only. It does not collect data,
ingest traces, calibrate actuation values, or update benchmark claims.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from robot_sf.benchmark.synthetic_actuation import actuation_variability_fields

AMV_CALIBRATION_SOURCE_MANIFEST_SCHEMA_VERSION = "amv_calibration_source_manifest.v1"
AMV_CALIBRATION_SOURCE_EVIDENCE_BOUNDARY = (
    "source_identification_only_no_data_collection_no_calibration_no_claim_update"
)

SOURCE_STATUS_READY = "ready"
SOURCE_STATUS_BLOCKED_EXTERNAL = "blocked-external-input"
SOURCE_STATUS_MISSING = "missing"

SOURCE_TYPE_HARDWARE_TRACE = "hardware_trace"
SOURCE_TYPE_OFFICIAL_SPEC = "official_spec"
SOURCE_TYPE_PLATFORM_CLASS_PROXY = "platform_class_proxy"
SOURCE_TYPE_UNKNOWN = "unknown"

LICENSE_STATUS_ACCEPTED = "accepted"
LICENSE_STATUS_MANUAL_REVIEW_REQUIRED = "manual_license_review_required"
LICENSE_STATUS_BLOCKED = "blocked"
LICENSE_STATUS_UNKNOWN = "unknown"

FIELD_STATUS_SUPPORTED = "supported"
FIELD_STATUS_MISSING = "missing"
FIELD_STATUS_UNSUPPORTED = "unsupported"

_SOURCE_STATUSES = {
    SOURCE_STATUS_READY,
    SOURCE_STATUS_BLOCKED_EXTERNAL,
    SOURCE_STATUS_MISSING,
}
_SOURCE_TYPES = {
    SOURCE_TYPE_HARDWARE_TRACE,
    SOURCE_TYPE_OFFICIAL_SPEC,
    SOURCE_TYPE_PLATFORM_CLASS_PROXY,
    SOURCE_TYPE_UNKNOWN,
}
_LICENSE_STATUSES = {
    LICENSE_STATUS_ACCEPTED,
    LICENSE_STATUS_MANUAL_REVIEW_REQUIRED,
    LICENSE_STATUS_BLOCKED,
    LICENSE_STATUS_UNKNOWN,
}
_FIELD_STATUSES = {
    FIELD_STATUS_SUPPORTED,
    FIELD_STATUS_MISSING,
    FIELD_STATUS_UNSUPPORTED,
}


class AmvCalibrationSourceManifestError(ValueError):
    """Raised when an AMV calibration source manifest cannot be parsed."""


@dataclass(frozen=True)
class AmvCalibrationFieldReport:
    """Per-field source-support status."""

    name: str
    support_status: str
    units: str | None = None
    blockers: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe report payload."""
        return asdict(self)


@dataclass(frozen=True)
class AmvCalibrationSourceReport:
    """Checked AMV calibration source-identification manifest.

    ``source_status == "ready"`` means only that a source candidate is identified for the
    declared boundary. ``hardware_calibration_claim_allowed`` is stricter and remains false for
    platform-class proxy sources.
    """

    schema_version: str
    manifest_id: str | None
    issue: int | None
    source_status: str
    source_type: str
    source_uri: str | None
    license_status: str | None
    claim_boundary: str | None
    evidence_boundary: str
    hardware_calibration_claim_allowed: bool
    fields: tuple[AmvCalibrationFieldReport, ...] = field(default_factory=tuple)
    blockers: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_ready(self) -> bool:
        """Whether the manifest has a usable source candidate for its boundary."""
        return self.source_status == SOURCE_STATUS_READY and not self.blockers

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-safe report payload."""
        payload = asdict(self)
        payload["fields"] = [field_report.to_dict() for field_report in self.fields]
        return payload


def load_amv_calibration_source_manifest(path: str | Path) -> dict[str, Any]:
    """Load an AMV calibration source manifest from YAML or JSON.

    Returns:
        Parsed manifest mapping.
    """
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise AmvCalibrationSourceManifestError(f"manifest file not found: {manifest_path}")
    try:
        payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise AmvCalibrationSourceManifestError(f"invalid YAML/JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise AmvCalibrationSourceManifestError("manifest must be a mapping")
    return dict(payload)


def check_amv_calibration_source_manifest(
    manifest: Mapping[str, Any],
    *,
    allowed_calibration_fields: set[str] | None = None,
) -> AmvCalibrationSourceReport:
    """Check source-identification readiness for AMV actuation calibration.

    The checker is fail-closed: incomplete metadata, license review blockers,
    missing supported fields, or source URIs that only point at tracking issues
    keep the manifest out of the ready state.

    Returns:
        Structured source-readiness report.
    """
    allowed_fields = allowed_calibration_fields or set(actuation_variability_fields())
    metadata, blockers = _extract_manifest_metadata(manifest)
    blockers.extend(_source_metadata_blockers(metadata))
    blockers.extend(_blocked_state_blockers(manifest, metadata["source_status"]))

    field_reports = _check_calibration_fields(manifest.get("calibration_fields"), allowed_fields)
    blockers.extend(_field_blockers(field_reports))
    if metadata["source_status"] == SOURCE_STATUS_READY and not _supported_fields(field_reports):
        blockers.append("ready source requires at least one supported calibration field")

    resolved_status = _resolve_source_status(metadata["source_status"], blockers)
    return AmvCalibrationSourceReport(
        schema_version=AMV_CALIBRATION_SOURCE_MANIFEST_SCHEMA_VERSION,
        manifest_id=metadata["manifest_id"],
        issue=metadata["issue"],
        source_status=resolved_status,
        source_type=metadata["source_type"],
        source_uri=metadata["source_uri"],
        license_status=metadata["license_status"],
        claim_boundary=metadata["claim_boundary"],
        evidence_boundary=AMV_CALIBRATION_SOURCE_EVIDENCE_BOUNDARY,
        hardware_calibration_claim_allowed=_hardware_claim_allowed(
            metadata,
            resolved_status,
            blockers,
        ),
        fields=tuple(field_reports),
        blockers=tuple(blockers),
    )


def _extract_manifest_metadata(manifest: Mapping[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Extract top-level manifest metadata and structural blockers.

    Returns:
        Metadata dictionary plus blocker messages.
    """
    blockers: list[str] = []
    schema_version = _optional_text(manifest.get("schema_version"))
    if schema_version != AMV_CALIBRATION_SOURCE_MANIFEST_SCHEMA_VERSION:
        blockers.append(
            f"schema_version must be {AMV_CALIBRATION_SOURCE_MANIFEST_SCHEMA_VERSION!r}"
        )

    manifest_id = _optional_text(manifest.get("manifest_id"))
    if manifest_id is None:
        blockers.append("manifest_id is required")

    issue = manifest.get("issue")
    if not isinstance(issue, int):
        blockers.append("issue must be an integer")
        issue = None

    metadata = {
        "manifest_id": manifest_id,
        "issue": issue,
        "source_status": _optional_text(manifest.get("source_status")) or SOURCE_STATUS_MISSING,
        "source_type": _optional_text(manifest.get("source_type")) or SOURCE_TYPE_UNKNOWN,
        "source_uri": _optional_text(manifest.get("source_uri")),
        "license_status": _optional_text(manifest.get("license_status")),
        "license": _optional_text(manifest.get("license")),
        "claim_boundary": _optional_text(manifest.get("claim_boundary")),
    }
    return metadata, blockers


def _source_metadata_blockers(metadata: Mapping[str, Any]) -> list[str]:
    """Return blockers for source URI, type, and license metadata."""
    blockers: list[str] = []
    source_status = metadata["source_status"]
    source_type = metadata["source_type"]
    source_uri = metadata["source_uri"]

    if source_status not in _SOURCE_STATUSES:
        blockers.append(f"source_status must be one of {sorted(_SOURCE_STATUSES)}")
    if source_type not in _SOURCE_TYPES:
        blockers.append(f"source_type must be one of {sorted(_SOURCE_TYPES)}")
    if source_status == SOURCE_STATUS_READY and source_uri is None:
        blockers.append("ready source requires source_uri")
    if source_status == SOURCE_STATUS_READY and _source_uri_is_tracking_issue(source_uri):
        blockers.append("ready source_uri must point at a source artifact, not a tracking issue")
    if source_status == SOURCE_STATUS_READY and source_type == SOURCE_TYPE_UNKNOWN:
        blockers.append("ready source requires source_type other than unknown")
    blockers.extend(_license_blockers(metadata))
    return blockers


def _license_blockers(metadata: Mapping[str, Any]) -> list[str]:
    """Return blockers for license and access status."""
    source_status = metadata["source_status"]
    license_status = metadata["license_status"]
    blockers: list[str] = []
    if source_status == SOURCE_STATUS_READY and license_status != LICENSE_STATUS_ACCEPTED:
        blockers.append("ready source requires license_status accepted")
    if license_status is None:
        blockers.append("license_status is required")
    elif license_status not in _LICENSE_STATUSES:
        blockers.append(f"license_status must be one of {sorted(_LICENSE_STATUSES)}")
    if source_status == SOURCE_STATUS_READY and metadata["license"] is None:
        blockers.append("ready source requires license")
    if license_status == LICENSE_STATUS_MANUAL_REVIEW_REQUIRED:
        blockers.append("license requires maintainer review before source can be ready")
    if license_status == LICENSE_STATUS_BLOCKED:
        blockers.append("license/access blocks use of this source")
    return blockers


def _blocked_state_blockers(manifest: Mapping[str, Any], source_status: str) -> list[str]:
    """Return blockers for explicit blocked-external-input manifests."""
    if source_status != SOURCE_STATUS_BLOCKED_EXTERNAL:
        return []
    blockers: list[str] = []
    if not _sequence_of_ints(manifest.get("blocker_issues", [])):
        blockers.append("blocked-external-input source requires blocker_issues")
    if _optional_text(manifest.get("blocked_reason")) is None:
        blockers.append("blocked-external-input source requires blocked_reason")
    return blockers


def _check_calibration_fields(
    raw_fields: Any,
    allowed_fields: set[str],
) -> tuple[AmvCalibrationFieldReport, ...]:
    """Validate declared source support for actuation calibration fields.

    Returns:
        Per-field support reports.
    """
    if not isinstance(raw_fields, Sequence) or isinstance(raw_fields, (str, bytes)):
        return (
            AmvCalibrationFieldReport(
                name="<manifest>",
                support_status=FIELD_STATUS_MISSING,
                blockers=("calibration_fields must be a non-empty list",),
            ),
        )
    if not raw_fields:
        return (
            AmvCalibrationFieldReport(
                name="<manifest>",
                support_status=FIELD_STATUS_MISSING,
                blockers=("calibration_fields must be a non-empty list",),
            ),
        )

    reports: list[AmvCalibrationFieldReport] = []
    seen: set[str] = set()
    for index, raw_field in enumerate(raw_fields):
        reports.append(_check_calibration_field(index, raw_field, allowed_fields, seen))
    return tuple(reports)


def _check_calibration_field(
    index: int,
    raw_field: Any,
    allowed_fields: set[str],
    seen: set[str],
) -> AmvCalibrationFieldReport:
    """Validate one calibration-field entry.

    Returns:
        Field support report.
    """
    if not isinstance(raw_field, Mapping):
        return AmvCalibrationFieldReport(
            name=f"<index:{index}>",
            support_status=FIELD_STATUS_MISSING,
            blockers=("field entry must be a mapping",),
        )

    name = _optional_text(raw_field.get("name")) or f"<index:{index}>"
    support_status = _optional_text(raw_field.get("support_status")) or FIELD_STATUS_MISSING
    units = _optional_text(raw_field.get("units"))
    blockers = _calibration_field_blockers(name, support_status, units, raw_field, seen)
    if name not in allowed_fields:
        blockers.append("not a canonical synthetic-actuation calibration field")
    return AmvCalibrationFieldReport(
        name=name,
        support_status=support_status,
        units=units,
        blockers=tuple(blockers),
    )


def _calibration_field_blockers(
    name: str,
    support_status: str,
    units: str | None,
    raw_field: Mapping[str, Any],
    seen: set[str],
) -> list[str]:
    """Return blockers for one calibration field."""
    blockers: list[str] = []
    if name in seen:
        blockers.append("duplicate calibration field")
    seen.add(name)
    if support_status not in _FIELD_STATUSES:
        blockers.append(f"support_status must be one of {sorted(_FIELD_STATUSES)}")
    if support_status == FIELD_STATUS_SUPPORTED and units is None:
        blockers.append("supported field requires units")
    if support_status != FIELD_STATUS_SUPPORTED and _optional_text(raw_field.get("reason")) is None:
        blockers.append("missing/unsupported field requires reason")
    return blockers


def _field_blockers(field_reports: Sequence[AmvCalibrationFieldReport]) -> list[str]:
    """Return field blockers with field names prefixed."""
    return [
        f"calibration_fields[{field_report.name}]: {field_blocker}"
        for field_report in field_reports
        for field_blocker in field_report.blockers
    ]


def _supported_fields(
    field_reports: Sequence[AmvCalibrationFieldReport],
) -> list[AmvCalibrationFieldReport]:
    """Return supported calibration fields without field-level blockers."""
    return [
        field_report
        for field_report in field_reports
        if field_report.support_status == FIELD_STATUS_SUPPORTED and not field_report.blockers
    ]


def _resolve_source_status(source_status: str, blockers: Sequence[str]) -> str:
    """Return final manifest status after blockers are applied."""
    if blockers and source_status == SOURCE_STATUS_READY:
        return SOURCE_STATUS_BLOCKED_EXTERNAL
    return source_status


def _hardware_claim_allowed(
    metadata: Mapping[str, Any],
    resolved_status: str,
    blockers: Sequence[str],
) -> bool:
    """Return whether metadata permits a hardware-calibrated claim boundary."""
    return (
        resolved_status == SOURCE_STATUS_READY
        and not blockers
        and metadata["source_type"] in {SOURCE_TYPE_HARDWARE_TRACE, SOURCE_TYPE_OFFICIAL_SPEC}
        and metadata["claim_boundary"] == "hardware-calibrated"
    )


def _optional_text(value: Any) -> str | None:
    """Return stripped non-empty string, otherwise ``None``."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _sequence_of_ints(value: Any) -> bool:
    """Return whether value is a non-empty sequence of integers."""
    return (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and bool(value)
        and all(isinstance(item, int) for item in value)
    )


def _source_uri_is_tracking_issue(source_uri: str | None) -> bool:
    """Return whether a URI points at a GitHub issue instead of a source artifact."""
    if source_uri is None:
        return False
    parsed = urlparse(source_uri.strip().lower())
    host = parsed.netloc.rsplit("@", 1)[-1].split(":", 1)[0]
    is_github = host == "github.com" or host.endswith(".github.com")
    return is_github and "/issues/" in parsed.path
