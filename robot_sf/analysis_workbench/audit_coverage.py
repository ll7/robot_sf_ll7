"""Release-bound coverage accounting for the benchmark-audit workbench.

BA-04 is deliberately a pure evaluator.  It consumes the typed BA-01 scan
report (or a small compatible mapping) and BA-03 ``ReviewRecord`` values; it
does not create a second journal, run a simulator, or infer benchmark
validity.  The protocol is an operational stopping rule, not a statistical
confidence certificate.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

from jsonschema import Draft202012Validator

from robot_sf.analysis_workbench.audit_contracts import (
    AuditContractError,
    Finding,
    ReviewRecord,
    canonical_json,
    record_from_dict,
)
from robot_sf.analysis_workbench.audit_detectors import DetectorRegistry
from robot_sf.analysis_workbench.review_report import (
    render_html as render_review_html,
)
from robot_sf.analysis_workbench.review_report import (
    render_markdown as render_review_markdown,
)

AUDIT_PROTOCOL_VERSION = "audit-protocol.v1"
AUDIT_COVERAGE_SCHEMA_VERSION = "audit-coverage.v1"
AUDIT_COVERAGE_SCHEMA_FILE = Path(__file__).with_name("schemas") / "audit_coverage.v1.json"

STATUS_INCOMPLETE = "incomplete"
STATUS_COMPLETE_WITH_DECLARED_EXCEPTIONS = "complete_with_declared_exceptions"
STATUS_COMPLETE_UNDER_PROTOCOL = "complete_under_protocol"
AUDIT_STATUSES = (
    STATUS_INCOMPLETE,
    STATUS_COMPLETE_WITH_DECLARED_EXCEPTIONS,
    STATUS_COMPLETE_UNDER_PROTOCOL,
)

DEFICIT_UNDER_REVIEW = "under_review"
DEFICIT_UNAVAILABLE = "unavailable"
DEFICIT_ERROR = "error"
DEFICIT_MET = "met"
DEFICIT_WAIVED = "waived"
DEFICIT_STATUSES = (
    DEFICIT_UNDER_REVIEW,
    DEFICIT_UNAVAILABLE,
    DEFICIT_ERROR,
    DEFICIT_MET,
    DEFICIT_WAIVED,
)

_INVENTORY_STATUSES = ("readable", "missing", "duplicate", "invalid", "unsupported")
_SIGNAL_STATUSES = ("flagged", "clear", "unavailable", "error")
_SCENARIO_GROUP_FIELDS = ("scenario_group", "scenario_group_id", "group", "scenario_id")
_OUTCOME_FIELDS = ("outcome", "result", "termination")
_ORDINARY_CONTROL_FIELDS = (
    "ordinary_control",
    "control_candidate",
    "is_control",
    "ordinary_candidate",
)
_HIGH_PRIORITY_VALUES = frozenset({"high", "critical", "priority", "high_priority"})
_RESOLVED_FINDING_STATUSES = frozenset({"resolved", "refuted", "waived", "closed"})
_EXCEPTION_STATUSES = frozenset({"waived", "exception", "declared"})
_EXCEPTION_FIELDS = frozenset({"requirement", "reason", "status", "stratum_id"})


class AuditCoverageError(ValueError):
    """Raised when a protocol or health report cannot be represented safely."""


def _text(value: Any, *, name: str, required: bool = True) -> str:
    if not isinstance(value, str):
        raise AuditCoverageError(f"{name} must be a string")
    if required and not value.strip():
        raise AuditCoverageError(f"{name} must be a non-empty string")
    return value.strip()


def _bool(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise AuditCoverageError(f"{name} must be a boolean")
    return value


def _count(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AuditCoverageError(f"{name} must be a non-negative integer")
    return value


def _strict_json(value: Any, *, path: str = "$") -> None:
    """Reject values that could make an identity or report non-deterministic."""

    if isinstance(value, float) and not math.isfinite(value):
        raise AuditCoverageError(f"{path} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise AuditCoverageError(f"{path} contains a non-string key")
            _strict_json(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _strict_json(item, path=f"{path}[{index}]")


def _freeze_json(value: Any) -> Any:
    """Recursively freeze JSON-like report state before storing it.

    Returns:
        An immutable mapping/sequence tree for the supplied value.
    """

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    return value


def _thaw_json(value: Any) -> Any:
    """Return a fresh mutable JSON-like copy for a report export."""

    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in value]
    return value


def _digest(value: Any) -> str:
    """Return a deterministic SHA-256 token for a strict JSON value."""

    _strict_json(value)
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if isinstance(value, Mapping):
        result = dict(value)
    elif is_dataclass(value):
        result = {item.name: getattr(value, item.name) for item in fields(value)}
    elif hasattr(value, "to_dict"):
        candidate = value.to_dict()
        if not isinstance(candidate, Mapping):
            raise AuditCoverageError(f"{name} to_dict() must return a mapping")
        result = dict(candidate)
    else:
        result = {}
        for key in (
            "episode_id",
            "planner_id",
            "scenario_group",
            "scenario_group_id",
            "group",
            "scenario_id",
            "outcome",
            "status",
            "row_id",
            "indexed",
            "readable",
            "valid",
            "supported",
            "expected",
            "duplicate",
            "ordinary_control",
            "control_candidate",
            "detector_id",
            "detector_version",
            "signal_id",
            "reason_code",
            "author_kind",
            "scope",
            "review_scope",
            "review_id",
        ):
            if hasattr(value, key):
                result[key] = getattr(value, key)
    if not result:
        raise AuditCoverageError(f"{name} must be a mapping or typed contract")
    return result


def _field(value: Any, *names: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        for name in names:
            if name in value:
                return value[name]
        return default
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return default


def _token(value: Any, *, default: str = "unknown") -> str:
    if isinstance(value, Mapping):
        value = value.get("label") or value.get("status") or value.get("value")
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    return str(value).strip()


def _sequence(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(value)
    return (value,)


def _exception_entry(value: Any, *, name: str) -> dict[str, Any]:
    """Normalize one closed, non-empty declared-exception envelope.

    Returns:
        A canonical exception mapping.
    """

    raw = _mapping(value, name=name)
    if any(not isinstance(key, str) for key in raw):
        raise AuditCoverageError(f"{name} must use string field names")
    unknown = set(raw) - _EXCEPTION_FIELDS - {"rationale", "reason_code"}
    if unknown:
        raise AuditCoverageError(f"{name} contains unknown fields: {', '.join(sorted(unknown))}")
    requirement = _text(
        raw.get("requirement") or raw.get("reason_code"),
        name=f"{name}.requirement",
    )
    reason = _text(
        raw.get("reason") or raw.get("rationale"),
        name=f"{name}.reason",
    )
    status = _token(raw.get("status"), default="waived").lower()
    if status not in _EXCEPTION_STATUSES:
        raise AuditCoverageError(f"{name}.status must be one of {sorted(_EXCEPTION_STATUSES)}")
    result = {"requirement": requirement, "reason": reason, "status": status}
    if raw.get("stratum_id") is not None:
        result["stratum_id"] = _text(raw["stratum_id"], name=f"{name}.stratum_id")
    return result


@dataclass(frozen=True, slots=True)
class AuditProtocol:
    """Configurable operational coverage targets for ``audit-protocol.v1``."""

    version: str = AUDIT_PROTOCOL_VERSION
    account_all_expected_rows: bool = True
    require_detector_attempts: bool = True
    full_human_review_target: int = 1
    ordinary_control_review_target: int = 1
    anomaly_representative_target: int = 1
    anomaly_context_control_target: int = 1
    require_integrity_disposition: bool = True
    scenario_group_fields: tuple[str, ...] = _SCENARIO_GROUP_FIELDS
    ordinary_candidate_fields: tuple[str, ...] = _ORDINARY_CONTROL_FIELDS
    declared_exceptions: tuple[Mapping[str, Any], ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate explicit targets and freeze mutable configuration values."""

        _text(self.version, name="protocol.version")
        for name in (
            "account_all_expected_rows",
            "require_detector_attempts",
            "require_integrity_disposition",
        ):
            _bool(getattr(self, name), name=f"protocol.{name}")
        for name in (
            "full_human_review_target",
            "ordinary_control_review_target",
            "anomaly_representative_target",
            "anomaly_context_control_target",
        ):
            _count(getattr(self, name), name=f"protocol.{name}")
        group_fields = tuple(
            _text(item, name="protocol.scenario_group_fields[]")
            for item in self.scenario_group_fields
        )
        control_fields = tuple(
            _text(item, name="protocol.ordinary_candidate_fields[]")
            for item in self.ordinary_candidate_fields
        )
        exceptions = tuple(
            _exception_entry(item, name="protocol.declared_exceptions[]")
            for item in self.declared_exceptions
        )
        metadata = _mapping(self.metadata, name="protocol.metadata") if self.metadata else {}
        _strict_json(exceptions, path="protocol.declared_exceptions")
        _strict_json(metadata, path="protocol.metadata")
        object.__setattr__(self, "scenario_group_fields", group_fields)
        object.__setattr__(self, "ordinary_candidate_fields", control_fields)
        object.__setattr__(self, "declared_exceptions", _freeze_json(exceptions))
        object.__setattr__(self, "metadata", _freeze_json(metadata))

    @classmethod
    def default(cls) -> AuditProtocol:
        """Return the shipped, non-statistical operational default."""

        return cls()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AuditProtocol:
        """Build a protocol from an explicit versioned mapping.

        Returns:
            Validated protocol configuration.
        """

        if not isinstance(payload, Mapping):
            raise AuditCoverageError("protocol must be a mapping")
        aliases = dict(payload)
        aliases.pop("protocol_digest", None)
        version = aliases.pop("version", None)
        for alias in ("protocol_version", "schema_version"):
            candidate = aliases.pop(alias, None)
            if candidate is None:
                continue
            if version is not None and candidate != version:
                raise AuditCoverageError("protocol version aliases disagree")
            version = candidate
        if version is None:
            version = AUDIT_PROTOCOL_VERSION
        allowed = {item.name for item in fields(cls)} - {"version"}
        unknown = set(aliases) - allowed
        if unknown:
            raise AuditCoverageError(
                f"protocol contains unknown fields: {', '.join(sorted(unknown))}"
            )
        return cls(version=version, **aliases)

    @property
    def digest(self) -> str:
        """Return the digest that binds this exact protocol configuration."""

        return _digest(self.to_dict(include_digest=False))

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        """Return the deterministic protocol document."""

        result: dict[str, Any] = {
            "version": self.version,
            "account_all_expected_rows": self.account_all_expected_rows,
            "require_detector_attempts": self.require_detector_attempts,
            "full_human_review_target": self.full_human_review_target,
            "ordinary_control_review_target": self.ordinary_control_review_target,
            "anomaly_representative_target": self.anomaly_representative_target,
            "anomaly_context_control_target": self.anomaly_context_control_target,
            "require_integrity_disposition": self.require_integrity_disposition,
            "scenario_group_fields": list(self.scenario_group_fields),
            "ordinary_candidate_fields": list(self.ordinary_candidate_fields),
            "declared_exceptions": _thaw_json(self.declared_exceptions),
            "metadata": _thaw_json(self.metadata),
        }
        if include_digest:
            result["protocol_digest"] = self.digest
        return result


@dataclass(frozen=True, slots=True)
class AuditIdentity:
    """Identity tuple that prevents a receipt crossing a release boundary."""

    campaign_digest: str = ""
    source_digest: str = ""
    release_digest: str = ""
    protocol_version: str = AUDIT_PROTOCOL_VERSION
    protocol_digest: str = ""
    detector_registry_digest: str = ""
    detector_config_digest: str = ""
    source_revision: str = ""
    scan_revision: str = ""
    review_revision: str = ""
    coverage_receipt_revision: str = ""

    def __post_init__(self) -> None:
        """Validate text identity fields while allowing fail-closed omissions."""

        names = (
            "campaign_digest",
            "source_digest",
            "release_digest",
            "protocol_version",
            "protocol_digest",
            "detector_registry_digest",
            "detector_config_digest",
            "source_revision",
            "scan_revision",
            "review_revision",
            "coverage_receipt_revision",
        )
        for name in names:
            value = getattr(self, name)
            if not isinstance(value, str):
                raise AuditCoverageError(f"identity.{name} must be a string")
            object.__setattr__(self, name, value.strip())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AuditIdentity:
        """Build an identity from canonical names and compatibility aliases.

        Returns:
            Release-bound identity value.
        """

        if not isinstance(payload, Mapping):
            raise AuditCoverageError("identity must be a mapping")
        aliases = {
            "campaign_digest": ("campaign_digest", "campaign_id"),
            "source_digest": ("source_digest", "source_id"),
            "release_digest": ("release_digest", "release_id", "release"),
            "protocol_version": ("protocol_version", "protocol_id"),
            "protocol_digest": ("protocol_digest", "protocol_revision"),
            "detector_registry_digest": ("detector_registry_digest", "registry_digest"),
            "detector_config_digest": (
                "detector_config_digest",
                "detector_configuration_digest",
                "config_digest",
                "config_identity",
            ),
            "source_revision": ("source_revision",),
            "scan_revision": ("scan_revision", "scan_digest", "cache_key"),
            "review_revision": ("review_revision",),
            "coverage_receipt_revision": ("coverage_receipt_revision", "receipt_revision"),
        }
        values: dict[str, str] = {}
        for target, names in aliases.items():
            value = next((payload[name] for name in names if name in payload), "")
            if value is None:
                value = ""
            if not isinstance(value, str):
                value = str(value)
            values[target] = value
        return cls(**values)

    @property
    def complete(self) -> bool:
        """Return whether all release/provenance identity components are present."""

        return all(
            (
                self.campaign_digest,
                self.source_digest,
                self.release_digest,
                self.protocol_version,
                self.protocol_digest,
                self.detector_registry_digest,
                self.detector_config_digest,
            )
        )

    @property
    def digest(self) -> str:
        """Return a deterministic receipt identity digest."""

        return _digest(self.to_dict(include_digest=False))

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        """Return the identity and its optional computed digest."""

        result = {
            "campaign_digest": self.campaign_digest,
            "source_digest": self.source_digest,
            "release_digest": self.release_digest,
            "protocol_version": self.protocol_version,
            "protocol_digest": self.protocol_digest,
            "detector_registry_digest": self.detector_registry_digest,
            "detector_config_digest": self.detector_config_digest,
            "source_revision": self.source_revision,
            "scan_revision": self.scan_revision,
            "review_revision": self.review_revision,
            "coverage_receipt_revision": self.coverage_receipt_revision,
        }
        if include_digest:
            result["identity_digest"] = self.digest
        return result


@dataclass(frozen=True, slots=True)
class ExpectedEpisode:
    """Small compatibility row used when a BA-01 report is not available."""

    episode_id: str
    planner_id: str = "unknown"
    scenario_group: str = "unknown"
    outcome: str = "unknown"
    status: str = "readable"
    expected: bool = True
    ordinary_control: bool = False
    row_id: str = ""
    reason: str = ""
    row: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the normalized row boundary."""

        _text(self.episode_id, name="episode_id")
        _text(self.planner_id, name="planner_id")
        _text(self.scenario_group, name="scenario_group")
        _text(self.outcome, name="outcome")
        if self.status not in _INVENTORY_STATUSES:
            raise AuditCoverageError(f"row status must be one of {_INVENTORY_STATUSES}")
        _bool(self.expected, name="expected")
        _bool(self.ordinary_control, name="ordinary_control")
        _strict_json(self.row, path="row")

    @property
    def readable(self) -> bool:
        """Return whether this row is eligible for coverage review."""

        return self.expected and self.status == "readable"

    def to_dict(self) -> dict[str, Any]:
        """Return a normalized row mapping."""

        return {
            "episode_id": self.episode_id,
            "planner_id": self.planner_id,
            "scenario_group": self.scenario_group,
            "outcome": self.outcome,
            "status": self.status,
            "expected": self.expected,
            "ordinary_control": self.ordinary_control,
            "row_id": self.row_id,
            "reason": self.reason,
            "row": dict(self.row),
        }


ExpectedRow = ExpectedEpisode


@dataclass(frozen=True, slots=True)
class DetectorAttempt:
    """One typed detector attempt after duplicate/replay collapse."""

    episode_id: str
    detector_id: str
    status: str = "error"
    detector_version: str = ""
    reason_code: str = ""
    message: str = ""
    attempt_id: str = ""

    def __post_init__(self) -> None:
        """Validate detector result status and identity."""

        _text(self.episode_id, name="detector.episode_id")
        _text(self.detector_id, name="detector.detector_id")
        if self.status not in _SIGNAL_STATUSES:
            raise AuditCoverageError(f"detector status must be one of {_SIGNAL_STATUSES}")
        if self.status in {"unavailable", "error"} and not (self.reason_code or self.message):
            raise AuditCoverageError("unavailable/error detector attempts require a reason")

    def to_dict(self) -> dict[str, Any]:
        """Return the typed detector attempt."""

        return {
            "episode_id": self.episode_id,
            "detector_id": self.detector_id,
            "status": self.status,
            "detector_version": self.detector_version,
            "reason_code": self.reason_code,
            "message": self.message,
            "attempt_id": self.attempt_id,
        }


@dataclass(frozen=True, slots=True)
class CoverageDeficit:
    """BA-02-compatible actionable coverage gap.

    The first nine fields intentionally match the BA-02 handoff shape.  The
    trailing identity fields carry the release/source/protocol context so a
    queue consumer cannot mistake a stale deficit for current evidence.
    """

    stratum_id: str
    observed: int
    target: int
    source_revision: str
    protocol_revision: str
    reason: str
    status: str
    source_id: str
    protocol_id: str
    dimensions: Mapping[str, str] = field(default_factory=dict)
    denominator: int = 0
    campaign_id: str = ""
    campaign_digest: str = ""
    source_digest: str = ""
    release_digest: str = ""
    detector_registry_digest: str = ""
    detector_config_digest: str = ""
    coverage_receipt_revision: str = ""
    evidence_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CoverageDeficit:
        """Restore a BA-04 or BA-02-compatible deficit mapping.

        Returns:
            Validated coverage deficit.
        """

        if not isinstance(payload, Mapping):
            raise AuditCoverageError("deficit must be a mapping")
        names = {item.name for item in fields(cls)}
        values = {name: payload[name] for name in names if name in payload}
        values["dimensions"] = payload.get("dimensions", {})
        values["evidence_ids"] = tuple(payload.get("evidence_ids", ()))
        return cls(**values)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> CoverageDeficit:
        """Compatibility alias for queue consumers using ``from_mapping``.

        Returns:
            Validated coverage deficit.
        """

        return cls.from_dict(payload)

    def __post_init__(self) -> None:
        """Validate the BA-02-compatible deficit contract."""

        for name in (
            "stratum_id",
            "status",
        ):
            _text(getattr(self, name), name=f"deficit.{name}")
        for name in (
            "source_revision",
            "protocol_revision",
            "reason",
            "source_id",
            "protocol_id",
        ):
            _text(getattr(self, name), name=f"deficit.{name}", required=False)
        if self.status not in DEFICIT_STATUSES:
            raise AuditCoverageError(f"deficit status must be one of {DEFICIT_STATUSES}")
        _count(self.observed, name="deficit.observed")
        _count(self.target, name="deficit.target")
        _count(self.denominator, name="deficit.denominator")
        dimensions = {str(key): _token(value) for key, value in self.dimensions.items()}
        _strict_json(dimensions, path="deficit.dimensions")
        evidence_ids = tuple(
            _text(item, name="deficit.evidence_ids[]") for item in self.evidence_ids
        )
        object.__setattr__(self, "dimensions", _freeze_json(dimensions))
        object.__setattr__(self, "evidence_ids", evidence_ids)

    def to_dict(self) -> dict[str, Any]:
        """Return the version-compatible deficit payload."""

        return {
            "schema_version": "coverage-deficit.v1",
            "stratum_id": self.stratum_id,
            "observed": self.observed,
            "target": self.target,
            "source_revision": self.source_revision,
            "protocol_revision": self.protocol_revision,
            "reason": self.reason,
            "status": self.status,
            "source_id": self.source_id,
            "protocol_id": self.protocol_id,
            "dimensions": _thaw_json(self.dimensions),
            "denominator": self.denominator,
            "campaign_id": self.campaign_id,
            "campaign_digest": self.campaign_digest,
            "source_digest": self.source_digest,
            "release_digest": self.release_digest,
            "detector_registry_digest": self.detector_registry_digest,
            "detector_config_digest": self.detector_config_digest,
            "coverage_receipt_revision": self.coverage_receipt_revision,
            "evidence_ids": list(self.evidence_ids),
        }

    def to_ba02_dict(self) -> dict[str, Any]:
        """Return the nine-field BA-02 queue handoff without BA-04 extras.

        BA-04 keeps ``waived`` visible in its own report.  BA-02's current
        control vocabulary represents an explicitly waived gap as ``met``;
        the reason remains present so the queue cannot mistake it for a
        missing calculation.
        """

        return {
            "stratum_id": self.stratum_id,
            "observed": self.observed,
            "target": self.target,
            "source_revision": self.source_revision,
            "protocol_revision": self.protocol_revision,
            "reason": self.reason,
            "status": "met" if self.status == DEFICIT_WAIVED else self.status,
            "source_id": self.source_id,
            "protocol_id": self.protocol_id,
        }


@dataclass(frozen=True, slots=True)
class AuditHealthReport:
    """Deterministic machine/readable audit health result."""

    identity: AuditIdentity
    protocol: AuditProtocol
    status: str
    counts: Mapping[str, Any]
    strata: tuple[Mapping[str, Any], ...] = ()
    deficits: tuple[CoverageDeficit, ...] = ()
    exceptions: tuple[Mapping[str, Any], ...] = ()
    findings: Mapping[str, Any] = field(default_factory=dict)
    materialization: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()
    report_digest: str = ""

    def __post_init__(self) -> None:  # noqa: C901
        """Validate and freeze report components, including the digest."""

        if not isinstance(self.identity, AuditIdentity):
            raise AuditCoverageError("report.identity must be AuditIdentity")
        if not isinstance(self.protocol, AuditProtocol):
            raise AuditCoverageError("report.protocol must be AuditProtocol")
        if self.status not in AUDIT_STATUSES:
            raise AuditCoverageError(f"report status must be one of {AUDIT_STATUSES}")
        for name, value in (
            ("counts", self.counts),
            ("findings", self.findings),
            ("materialization", self.materialization),
            ("diagnostics", self.diagnostics),
        ):
            if not isinstance(value, Mapping):
                raise AuditCoverageError(f"report.{name} must be a mapping")
            _strict_json(value, path=f"report.{name}")
        strata = tuple(dict(item) for item in self.strata)
        exceptions = tuple(
            _exception_entry(item, name="report.exceptions[]") for item in self.exceptions
        )
        _strict_json(strata, path="report.strata")
        _strict_json(exceptions, path="report.exceptions")
        object.__setattr__(self, "counts", _freeze_json(self.counts))
        object.__setattr__(self, "strata", _freeze_json(strata))
        object.__setattr__(self, "exceptions", _freeze_json(exceptions))
        object.__setattr__(self, "findings", _freeze_json(self.findings))
        object.__setattr__(self, "materialization", _freeze_json(self.materialization))
        object.__setattr__(self, "diagnostics", _freeze_json(self.diagnostics))
        object.__setattr__(
            self,
            "limitations",
            tuple(_text(item, name="limitations[]") for item in self.limitations),
        )
        if any(not isinstance(item, CoverageDeficit) for item in self.deficits):
            raise AuditCoverageError("report.deficits must contain CoverageDeficit values")
        if self.identity.protocol_version != self.protocol.version:
            raise AuditCoverageError("report identity protocol_version does not match protocol")
        if self.identity.protocol_digest != self.protocol.digest:
            raise AuditCoverageError("report identity protocol_digest does not match protocol")
        unwaived = tuple(item for item in self.deficits if item.status != DEFICIT_WAIVED)
        if self.status != STATUS_INCOMPLETE and not self.identity.complete:
            raise AuditCoverageError("complete report requires a complete identity")
        if self.status != STATUS_INCOMPLETE and unwaived:
            raise AuditCoverageError("complete report contains unwaived deficits")
        if self.status == STATUS_INCOMPLETE and not unwaived:
            raise AuditCoverageError("incomplete report requires an unwaived deficit")
        if self.status == STATUS_COMPLETE_UNDER_PROTOCOL and (self.deficits or self.exceptions):
            raise AuditCoverageError(
                "complete_under_protocol report cannot contain deficits or exceptions"
            )
        if self.status == STATUS_COMPLETE_WITH_DECLARED_EXCEPTIONS and not self.exceptions:
            raise AuditCoverageError("complete_with_declared_exceptions requires exceptions")
        digest = self._compute_digest()
        if self.report_digest and self.report_digest != digest:
            raise AuditCoverageError("report_digest does not match report contents")
        object.__setattr__(self, "report_digest", digest)
        _validate_report_semantics(self._payload(include_digest=True))

    @property
    def identity_digest(self) -> str:
        """Return the release-bound identity digest used by the receipt."""

        return self.identity.digest

    @property
    def coverage_deficits(self) -> tuple[CoverageDeficit, ...]:
        """Return deficits under the BA-02 handoff name."""

        return self.deficits

    @property
    def complete(self) -> bool:
        """Return whether the report has no unwaived deficit."""

        return self.status != STATUS_INCOMPLETE

    @property
    def completion_receipt(self) -> dict[str, Any]:
        """Return a compact identity-bound completion receipt."""

        return {
            "schema_version": "audit-completion-receipt.v1",
            "status": self.status,
            "identity_digest": self.identity.digest,
            "report_digest": self.report_digest,
            "campaign_digest": self.identity.campaign_digest,
            "source_digest": self.identity.source_digest,
            "release_digest": self.identity.release_digest,
            "protocol_version": self.identity.protocol_version,
            "protocol_digest": self.identity.protocol_digest,
            "detector_registry_digest": self.identity.detector_registry_digest,
            "detector_config_digest": self.identity.detector_config_digest,
            "source_revision": self.identity.source_revision,
            "scan_revision": self.identity.scan_revision,
            "review_revision": self.identity.review_revision,
            "coverage_receipt_revision": self.identity.coverage_receipt_revision,
        }

    def _payload(self, *, include_digest: bool) -> dict[str, Any]:
        """Build the canonical report payload used for digesting and export.

        Returns:
            JSON-compatible report mapping.
        """

        result: dict[str, Any] = {
            "schema_version": AUDIT_COVERAGE_SCHEMA_VERSION,
            "status": self.status,
            "identity": self.identity.to_dict(),
            "protocol": self.protocol.to_dict(),
            "counts": _thaw_json(self.counts),
            "strata": _thaw_json(self.strata),
            "deficits": [item.to_dict() for item in self.deficits],
            "exceptions": _thaw_json(self.exceptions),
            "findings": _thaw_json(self.findings),
            "materialization": _thaw_json(self.materialization),
            "diagnostics": _thaw_json(self.diagnostics),
            "limitations": list(self.limitations),
        }
        if include_digest:
            result["report_digest"] = self.report_digest
        return result

    def _compute_digest(self) -> str:
        return _digest(self._payload(include_digest=False))

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned machine-readable report."""

        return self._payload(include_digest=True)

    def to_json(self) -> str:
        """Return canonical JSON suitable for a durable report artifact."""

        return canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, payload: str) -> AuditHealthReport:
        """Restore a report from canonical JSON and verify its digest.

        Returns:
            Validated deterministic health report.
        """

        if not isinstance(payload, str):
            raise AuditCoverageError("health report JSON must be a string")
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise AuditCoverageError(f"invalid health report JSON: {exc}") from exc
        return cls.from_dict(value)

    def to_markdown(self) -> str:
        """Render through the existing SREV report renderer adapter.

        Returns:
            Deterministic Markdown report.
        """

        return render_markdown(self)

    def to_html(self) -> str:
        """Render through the existing SREV report renderer adapter.

        Returns:
            Deterministic standalone HTML report.
        """

        return render_html(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> AuditHealthReport:
        """Restore and verify a deterministic report document.

        Returns:
            Validated health report.
        """

        if not isinstance(payload, Mapping):
            raise AuditCoverageError("health report must be a mapping")
        validate_audit_coverage(payload)
        identity = AuditIdentity.from_dict(payload["identity"])
        protocol = AuditProtocol.from_dict(payload["protocol"])
        deficits = tuple(_deficit_from_dict(item) for item in payload["deficits"])
        report = cls(
            identity=identity,
            protocol=protocol,
            status=payload["status"],
            counts=payload["counts"],
            strata=tuple(payload["strata"]),
            deficits=deficits,
            exceptions=tuple(payload["exceptions"]),
            findings=payload["findings"],
            materialization=payload["materialization"],
            diagnostics=payload["diagnostics"],
            limitations=tuple(payload["limitations"]),
            report_digest=payload["report_digest"],
        )
        return report


def _deficit_from_dict(value: Mapping[str, Any]) -> CoverageDeficit:
    return CoverageDeficit.from_dict(value)


def default_audit_protocol() -> AuditProtocol:
    """Return the shipped ``audit-protocol.v1`` configuration."""

    return AuditProtocol.default()


def load_audit_protocol(value: Mapping[str, Any] | None = None) -> AuditProtocol:
    """Load the default protocol or a strict explicit mapping.

    Returns:
        Validated protocol configuration.
    """

    return default_audit_protocol() if value is None else AuditProtocol.from_dict(value)


def _coerce_protocol(value: AuditProtocol | Mapping[str, Any] | None) -> AuditProtocol:
    if value is None:
        return default_audit_protocol()
    if isinstance(value, AuditProtocol):
        return value
    return AuditProtocol.from_dict(value)


def _coerce_identity(value: AuditIdentity | Mapping[str, Any] | None) -> AuditIdentity:
    if value is None:
        return AuditIdentity()
    if isinstance(value, AuditIdentity):
        return value
    return AuditIdentity.from_dict(value)


def _scan_mapping(scan: Any) -> dict[str, Any]:
    if scan is None:
        return {}
    if isinstance(scan, Sequence) and not isinstance(scan, (str, bytes, bytearray)):
        return {}
    if isinstance(scan, Mapping):
        return dict(scan)
    if hasattr(scan, "to_dict"):
        value = scan.to_dict()
        if isinstance(value, Mapping):
            return dict(value)
    return _mapping(scan, name="scan_summary")


def _scan_inventory(scan: Any) -> tuple[Any, ...]:
    if scan is None:
        return ()
    if isinstance(scan, Sequence) and not isinstance(scan, (str, bytes, bytearray)):
        return tuple(scan)
    value = _field(scan, "inventory", "expected_rows", "rows", "episode_rows", default=None)
    return _sequence(value)


def _scan_signals(scan: Any) -> tuple[Any, ...]:
    if scan is None:
        return ()
    value = _field(scan, "signals", "detector_attempts", "detector_results", default=None)
    return _sequence(value)


def _scan_counts(scan: Any) -> Mapping[str, Any]:
    value = _field(scan, "counts", default={})
    return value if isinstance(value, Mapping) else {}


def _row_mapping(value: Any, index: int) -> dict[str, Any]:
    raw = _mapping(value, name=f"expected_rows[{index}]")
    nested = raw.get("row")
    if isinstance(nested, Mapping):
        result = dict(nested)
        result.update({key: item for key, item in raw.items() if key != "row"})
        return result
    return raw


def _normalize_row(  # noqa: C901
    value: Any,
    index: int,
    *,
    scenario_group_fields: Sequence[str] = _SCENARIO_GROUP_FIELDS,
    ordinary_candidate_fields: Sequence[str] = _ORDINARY_CONTROL_FIELDS,
) -> ExpectedEpisode:
    raw = _row_mapping(value, index)
    episode_id = _token(
        raw.get("episode_id") or raw.get("id") or raw.get("row_id"), default=f"row-{index}"
    )
    planner = _token(raw.get("planner_id") or raw.get("planner") or raw.get("algorithm"))
    group = "unknown"
    for key in scenario_group_fields:
        if raw.get(key) not in (None, ""):
            group = _token(raw[key])
            break
    outcome = "unknown"
    for key in _OUTCOME_FIELDS:
        if raw.get(key) not in (None, ""):
            outcome = _token(raw[key])
            break
    status = _token(raw.get("status"), default="readable").lower().replace("-", "_")
    if status in {"available", "valid", "indexed", "ok", "complete"}:
        status = "readable"
    if status not in _INVENTORY_STATUSES:
        if raw.get("readable") is False:
            status = "missing"
        elif raw.get("valid") is False:
            status = "invalid"
        elif raw.get("supported") is False:
            status = "unsupported"
        else:
            status = "invalid"
    expected = raw.get("expected", True)
    if not isinstance(expected, bool):
        expected = bool(expected)
    control = any(raw.get(key) is True for key in ordinary_candidate_fields)
    control = control or raw.get("role") in {"control", "ordinary", "normal_control"}
    if "row" in raw and raw["row"] is None and status == "readable":
        status = "missing"
    return ExpectedEpisode(
        episode_id=episode_id,
        planner_id=planner,
        scenario_group=group,
        outcome=outcome,
        status=status,
        expected=expected,
        ordinary_control=control,
        row_id=_token(raw.get("row_id"), default=episode_id),
        reason=_token(raw.get("reason"), default=""),
        row=raw,
    )


def _normalize_rows(
    scan: Any,
    expected_rows: Sequence[Any] | None,
    *,
    scenario_group_fields: Sequence[str] = _SCENARIO_GROUP_FIELDS,
    ordinary_candidate_fields: Sequence[str] = _ORDINARY_CONTROL_FIELDS,
) -> tuple[tuple[ExpectedEpisode, ...], dict[str, int]]:
    source_values = tuple(expected_rows) if expected_rows is not None else _scan_inventory(scan)
    rows: list[ExpectedEpisode] = []
    duplicate_count = 0
    seen: set[str] = set()
    for index, value in enumerate(source_values, start=1):
        row = _normalize_row(
            value,
            index,
            scenario_group_fields=scenario_group_fields,
            ordinary_candidate_fields=ordinary_candidate_fields,
        )
        key = row.episode_id
        if key in seen or row.status == "duplicate":
            duplicate_count += 1
            rows.append(
                replace(
                    row,
                    status="duplicate",
                    row_id=row.row_id or f"duplicate-{index}",
                    reason=row.reason or "duplicate logical episode row",
                )
            )
            continue
        seen.add(key)
        rows.append(row)
    counts = _inventory_counts(rows, duplicate_count)
    return tuple(sorted(rows, key=lambda item: (item.episode_id, item.row_id))), counts


def _inventory_counts(rows: Sequence[ExpectedEpisode], duplicate_count: int = 0) -> dict[str, int]:
    counter = Counter(row.status for row in rows if row.expected)
    indexed = sum(
        counter.get(status, 0) for status in ("readable", "duplicate", "invalid", "unsupported")
    )
    return {
        "expected": sum(1 for row in rows if row.expected),
        "indexed": indexed,
        "readable": counter.get("readable", 0),
        "missing": counter.get("missing", 0),
        "duplicate": counter.get("duplicate", 0),
        "invalid": counter.get("invalid", 0),
        "unsupported": counter.get("unsupported", 0),
        "unexpected": sum(1 for row in rows if not row.expected),
        "replay_or_duplicate_rows": duplicate_count,
    }


def _detector_ids(scan: Any, detector_registry: Any) -> tuple[str, ...]:
    value = detector_registry
    if value is None:
        value = _field(scan, "detector_registry", default=None)
    # A BA-01 detector schedule is authoritative only after it has crossed the
    # typed registry boundary.  In particular, detector IDs copied from an
    # arbitrary mapping or inferred from observed attempts cannot define the
    # denominator: doing so lets an unexpected clear attempt look complete.
    if isinstance(value, DetectorRegistry):
        return value.ids
    return ()


def _typed_detector_registry(scan: Any, detector_registry: Any) -> DetectorRegistry | None:
    """Return the admitted BA-01 registry, if one is present.

    Serialized registry-shaped mappings deliberately do not count here.  The
    BA-01 owner must deserialize and validate those documents before BA-04 can
    use their detector IDs as a schedule.
    """

    value = detector_registry
    if value is None:
        value = _field(scan, "detector_registry", default=None)
    return value if isinstance(value, DetectorRegistry) else None


def _normalize_signal(value: Any, index: int) -> DetectorAttempt:
    raw = _mapping(value, name=f"detector_attempts[{index}]")
    episode_id = _token(raw.get("episode_id") or raw.get("row_id"), default=f"unknown-{index}")
    detector_id = _token(raw.get("detector_id") or raw.get("detector"), default=f"detector-{index}")
    status = _token(raw.get("status"), default="error").lower().replace("-", "_")
    if status not in _SIGNAL_STATUSES:
        status = "error"
    reason = _token(raw.get("reason_code") or raw.get("reason"), default="missing_typed_result")
    message = _token(raw.get("message"), default="")
    return DetectorAttempt(
        episode_id=episode_id,
        detector_id=detector_id,
        status=status,
        detector_version=_token(raw.get("detector_version") or raw.get("version"), default=""),
        reason_code=reason
        if status in {"error", "unavailable"}
        else _token(raw.get("reason_code"), default=""),
        message=message,
        attempt_id=_token(
            raw.get("attempt_id") or raw.get("signal_id"), default=f"attempt-{index}"
        ),
    )


def _normalize_signals(  # noqa: C901
    scan: Any,
    detector_attempts: Sequence[Any] | None,
    rows: Sequence[ExpectedEpisode],
    detector_registry: Any,
) -> tuple[DetectorAttempt, ...]:
    values = tuple(detector_attempts) if detector_attempts is not None else _scan_signals(scan)
    normalized = [_normalize_signal(value, index) for index, value in enumerate(values, start=1)]
    detector_ids = _detector_ids(scan, detector_registry)
    readable_ids = {row.episode_id for row in rows if row.readable}
    typed_registry = _typed_detector_registry(scan, detector_registry)
    # An attempt is positive evidence only when it belongs to the active
    # readable-episode x typed-registry schedule.  Preserve the attempt ID but
    # turn all out-of-schedule values into explicit errors so they remain
    # visible without satisfying the detector gate.
    admitted: list[DetectorAttempt] = []
    for item in normalized:
        unexpected_episode = item.episode_id not in readable_ids
        unexpected_detector = not detector_ids or item.detector_id not in detector_ids
        missing_registry = typed_registry is None
        if unexpected_episode or unexpected_detector or missing_registry:
            reasons = []
            if missing_registry:
                reasons.append("missing_typed_registry")
            if unexpected_episode:
                reasons.append("unexpected_episode")
            if unexpected_detector:
                reasons.append("unexpected_detector")
            admitted.append(
                replace(
                    item,
                    status="error",
                    reason_code="unexpected_detector_attempt",
                    message=";".join(reasons),
                )
            )
        else:
            admitted.append(item)
    normalized = admitted
    observed_keys = {(item.episode_id, item.detector_id) for item in normalized}
    # A typed registry defines the expected schedule.  Do not synthesize a
    # schedule from untyped attempts or registry-like mappings.
    for episode_id in readable_ids:
        for detector_id in detector_ids:
            if (episode_id, detector_id) in observed_keys:
                continue
            normalized.append(
                DetectorAttempt(
                    episode_id=episode_id,
                    detector_id=detector_id,
                    status="error",
                    reason_code="missing_typed_result",
                    message="scheduled detector attempt has no typed result",
                    attempt_id=f"missing:{episode_id}:{detector_id}",
                )
            )
    collapsed: dict[tuple[str, str], DetectorAttempt] = {}
    severity = {"clear": 0, "flagged": 1, "unavailable": 2, "error": 3}
    for item in normalized:
        key = (item.episode_id, item.detector_id)
        existing = collapsed.get(key)
        if existing is None or severity[item.status] > severity[existing.status]:
            collapsed[key] = item
    return tuple(collapsed[key] for key in sorted(collapsed))


def _review_values(
    scan: Any, records: Sequence[Any], review_records: Sequence[Any] | None
) -> tuple[Any, ...]:
    if review_records is not None:
        return tuple(review_records)
    if records:
        return tuple(records)
    mapping = _scan_mapping(scan)
    return _sequence(mapping.get("review_records"))


def _review_identity_matches(value: Any, identity: AuditIdentity) -> bool:
    if not isinstance(value, ReviewRecord):
        return False
    declared = _field(value, "identity_digest", default="")
    if declared and declared != identity.digest:
        return False
    source_digest = _field(value, "source_digest", "source_id", default="")
    if source_digest and identity.source_digest and str(source_digest) != identity.source_digest:
        return False
    release_digest = _field(value, "release_digest", "release_id", default="")
    if (
        release_digest
        and identity.release_digest
        and str(release_digest) != identity.release_digest
    ):
        return False
    # BA-03 ReviewRecord carries the editor/selection revision as an integer.
    # If the active audit advertises a source, scan, or review revision, an
    # unavailable (zero) or stale receipt must not earn human coverage.  The
    # revision token is intentionally compared as text because BA-04 identity
    # values may be opaque hashes while BA-03's field is numeric.  If more
    # than one active revision is declared, every token must agree; a bare
    # BA-03 receipt cannot prove only one side of a source/scan boundary.
    expected_revisions = {
        str(item)
        for item in (
            identity.source_revision,
            identity.scan_revision,
            identity.review_revision,
        )
        if item
    }
    if expected_revisions:
        observed_revision = _field(value, "source_revision", default=0)
        if observed_revision in (None, "", 0):
            return False
        if any(str(observed_revision) != expected for expected in expected_revisions):
            return False
    return True


def _typed_review(value: Any) -> ReviewRecord | None:
    """Admit only a validated BA-03 ReviewRecord.

    Mapping inputs are accepted solely when they are canonical BA-03 record
    envelopes and deserialize successfully through the BA-03 record owner.
    A convenient ``episode_id`` mapping is not a review receipt and can never
    contribute to the human denominator.

    Returns:
        A validated BA-03 review, or ``None`` for an untrusted value.
    """

    if isinstance(value, ReviewRecord):
        return value
    if not isinstance(value, Mapping):
        return None
    try:
        record = record_from_dict(value)
    except (AuditContractError, KeyError, TypeError, ValueError):
        return None
    return record if isinstance(record, ReviewRecord) else None


def _review_is_contaminated(value: Any) -> bool:
    """Classify rejected replay-like mappings for visible accounting only.

    Returns:
        Whether the value carries an explicit replay/duplicate marker.
    """

    if not isinstance(value, Mapping):
        return False
    evidence_kind = _token(
        value.get("evidence_kind")
        or value.get("review_kind")
        or value.get("origin_kind")
        or value.get("record_kind"),
        default="",
    ).lower()
    return evidence_kind in {
        "agent",
        "clip",
        "duplicate",
        "interval",
        "one_click",
        "replay",
        "regenerated",
        "similarity",
    } or any(
        value.get(key) is True
        for key in (
            "is_duplicate",
            "is_replay",
            "regenerated",
            "replayed",
            "similarity_membership",
        )
    )


def _normalize_reviews(  # noqa: C901
    values: Sequence[Any], rows: Sequence[ExpectedEpisode], identity: AuditIdentity
) -> tuple[dict[str, Any], set[str]]:
    readable_ids = {row.episode_id for row in rows if row.readable}
    seen_records: set[str] = set()
    human: set[str] = set()
    agent: set[str] = set()
    interval: set[str] = set()
    full_human: set[str] = set()
    full_agent: set[str] = set()
    duplicate_count = 0
    stale_count = 0
    rejected_count = 0
    for index, value in enumerate(values, start=1):
        typed = _typed_review(value)
        if typed is None:
            if _review_is_contaminated(value):
                duplicate_count += 1
            else:
                rejected_count += 1
            continue
        value = typed
        episode_id = _field(value, "episode_id", default="")
        if not isinstance(episode_id, str) or episode_id not in readable_ids:
            continue
        if not _review_identity_matches(value, identity):
            stale_count += 1
            continue
        evidence_kind = _token(
            _field(
                value,
                "evidence_kind",
                "review_kind",
                "origin_kind",
                "record_kind",
                default="",
            ),
            default="",
        ).lower()
        contaminated = evidence_kind in {
            "agent",
            "clip",
            "duplicate",
            "interval",
            "one_click",
            "replay",
            "regenerated",
            "similarity",
        } or any(
            _field(value, key, default=False) is True
            for key in (
                "is_duplicate",
                "is_replay",
                "regenerated",
                "replayed",
                "similarity_membership",
            )
        )
        if contaminated:
            duplicate_count += 1
            continue
        record_id = _field(value, "review_id", default=f"review-{index}")
        if record_id in seen_records:
            duplicate_count += 1
            continue
        seen_records.add(record_id)
        author = _token(_field(value, "author_kind", default=""), default="")
        scope = _token(_field(value, "scope", "review_scope", default=""), default="")
        if author == "human":
            human.add(episode_id)
        elif author == "agent":
            agent.add(episode_id)
        if scope == "full_episode":
            if author == "human":
                full_human.add(episode_id)
            elif author == "agent":
                full_agent.add(episode_id)
        elif scope == "interval":
            interval.add(episode_id)
    reviewed_union = human | agent
    return (
        {
            "human_reviewed": len(human),
            "agent_reviewed": len(agent),
            "full_episode_human": len(full_human),
            "full_episode_agent": len(full_agent),
            "interval_reviewed": len(interval),
            "reviewed_episode_union": len(reviewed_union),
            "duplicate_or_replayed": duplicate_count,
            "stale_or_mismatched": stale_count,
            "untyped_or_invalid": rejected_count,
            "full_human_ids": sorted(full_human),
            "human_ids": sorted(human),
            "agent_ids": sorted(agent),
        },
        full_human,
    )


def _row_stratum(row: ExpectedEpisode) -> tuple[str, str, str]:
    return row.planner_id, row.scenario_group, row.outcome


def _stratum_id(dimensions: Mapping[str, Any]) -> str:
    return ":".join(f"{key}={_token(dimensions.get(key))}" for key in sorted(dimensions))


def _exception_entries(
    protocol: AuditProtocol, values: Sequence[Any] | Mapping[str, Any] | None
) -> tuple[dict[str, Any], ...]:
    entries: list[dict[str, Any]] = [
        _exception_entry(item, name="protocol.declared_exceptions[]")
        for item in protocol.declared_exceptions
    ]
    if isinstance(values, Mapping):
        entries.extend(
            _exception_entry(
                {"requirement": key, "reason": value, "status": "waived"},
                name="exceptions[]",
            )
            for key, value in values.items()
        )
    else:
        entries.extend(_exception_entry(item, name="exceptions[]") for item in _sequence(values))
    return tuple(sorted(entries, key=canonical_json))


def _exception_matches(deficit: CoverageDeficit, exception: Mapping[str, Any]) -> bool:
    """Return whether one closed exception names one specific deficit."""

    requirement = exception.get("requirement")
    requirement_values = {
        deficit.reason,
        deficit.stratum_id,
        deficit.dimensions.get("requirement"),
        "*",
    }
    if requirement not in requirement_values:
        return False
    stratum = exception.get("stratum_id")
    return stratum in {None, "*", deficit.stratum_id}


def _exception_for(
    deficit: CoverageDeficit, exceptions: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any] | None:
    for item in exceptions:
        if item.get("status") not in _EXCEPTION_STATUSES:
            continue
        if _exception_matches(deficit, item):
            return item
    return None


def _build_deficit(
    identity: AuditIdentity,
    *,
    dimensions: Mapping[str, Any],
    observed: int,
    target: int,
    reason: str,
    status: str = DEFICIT_UNDER_REVIEW,
    denominator: int = 0,
    evidence_ids: Sequence[str] = (),
) -> CoverageDeficit:
    stratum_id = _stratum_id(dimensions)
    return CoverageDeficit(
        stratum_id=stratum_id,
        observed=observed,
        target=target,
        # BA-02 calls these fields *revisions*, not digests.  The scan
        # revision is the queue's source revision/summary identity; when BA-04
        # was not given one, leave it unavailable so BA-02 can surface the
        # missing provenance instead of manufacturing a source-revision match
        # from the source digest.
        source_revision=identity.scan_revision or "",
        protocol_revision=identity.protocol_version or "",
        reason=reason,
        status=status,
        source_id=identity.source_digest or "",
        protocol_id=identity.protocol_digest or "",
        dimensions={key: _token(value) for key, value in dimensions.items()},
        denominator=denominator,
        campaign_id=identity.campaign_digest,
        campaign_digest=identity.campaign_digest,
        source_digest=identity.source_digest,
        release_digest=identity.release_digest,
        detector_registry_digest=identity.detector_registry_digest,
        detector_config_digest=identity.detector_config_digest,
        coverage_receipt_revision=identity.coverage_receipt_revision,
        evidence_ids=tuple(sorted(set(evidence_ids))),
    )


def _finding_mapping(value: Any) -> dict[str, Any]:
    raw = _mapping(value, name="findings[]")
    if isinstance(value, Finding):
        raw.setdefault("finding_id", value.finding_id)
        raw.setdefault("candidate_members", list(value.candidate_members))
        raw.setdefault("confirmed_members", list(value.confirmed_members))
        raw.setdefault("status", value.status)
        raw.setdefault("tags", list(value.tags))
    return raw


def _finding_summary(  # noqa: C901
    findings: Sequence[Any],
    full_human_ids: set[str],
    readable_ids: set[str],
    identity: AuditIdentity,
    protocol: AuditProtocol,
    exceptions: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[CoverageDeficit]]:
    deficits: list[CoverageDeficit] = []
    candidate_count = 0
    confirmed_count = 0
    high_priority = 0
    representatives = 0
    controls = 0
    unresolved_integrity = 0
    finding_rows: list[dict[str, Any]] = []
    for index, value in enumerate(findings, start=1):
        raw = _finding_mapping(value)
        finding_id = _token(
            raw.get("finding_id") or raw.get("cluster_id") or raw.get("id"),
            default=f"finding-{index}",
        )
        candidates = tuple(
            sorted(
                {
                    str(item)
                    for item in _sequence(raw.get("candidate_members") or raw.get("members"))
                    if str(item) in readable_ids
                }
            )
        )
        confirmed = tuple(
            sorted(
                {
                    str(item)
                    for item in _sequence(raw.get("confirmed_members") or raw.get("confirmations"))
                    if str(item) in readable_ids
                }
            )
        )
        candidate_count += len(candidates)
        confirmed_count += len(confirmed)
        tags = {str(item).lower() for item in _sequence(raw.get("tags"))}
        priority = _token(raw.get("priority") or raw.get("severity"), default="").lower()
        is_high = bool(
            raw.get("high_priority") is True
            or priority in _HIGH_PRIORITY_VALUES
            or tags & _HIGH_PRIORITY_VALUES
        )
        kind = _token(raw.get("kind") or raw.get("finding_kind"), default="").lower()
        is_integrity = bool(
            raw.get("integrity") is True or "integrity" in kind or "integrity" in tags
        )
        status = _token(raw.get("status"), default="proposed").lower()
        if is_high:
            high_priority += 1
        representative = _token(
            raw.get("representative_episode_id") or raw.get("representative"), default=""
        )
        if not representative and candidates:
            representative = candidates[0]
        representative_reviewed = bool(representative and representative in full_human_ids)
        if representative_reviewed:
            representatives += 1
        elif is_high and protocol.anomaly_representative_target:
            deficits.append(
                _build_deficit(
                    identity,
                    dimensions={"requirement": "finding_representative", "finding_id": finding_id},
                    observed=0,
                    target=protocol.anomaly_representative_target,
                    reason="missing_finding_representative"
                    if representative
                    else "finding_representative_unavailable",
                    status=DEFICIT_UNAVAILABLE if not representative else DEFICIT_UNDER_REVIEW,
                    denominator=len(candidates),
                    evidence_ids=candidates,
                )
            )
        controls_raw = (
            raw.get("control_episode_id")
            or raw.get("different_context_episode_id")
            or raw.get("negative_controls")
        )
        control_ids = tuple(
            sorted({str(item) for item in _sequence(controls_raw) if str(item) in readable_ids})
        )
        control_reviewed = False
        if control_ids:
            control_reviewed = any(item in full_human_ids for item in control_ids)
            if control_reviewed:
                controls += 1
            elif is_high and protocol.anomaly_context_control_target:
                deficits.append(
                    _build_deficit(
                        identity,
                        dimensions={
                            "requirement": "finding_context_control",
                            "finding_id": finding_id,
                        },
                        observed=0,
                        target=protocol.anomaly_context_control_target,
                        reason="missing_finding_context_control",
                        denominator=len(control_ids),
                        evidence_ids=control_ids,
                    )
                )
        if is_integrity and status not in _RESOLVED_FINDING_STATUSES:
            unresolved_integrity += 1
            if protocol.require_integrity_disposition:
                deficits.append(
                    _build_deficit(
                        identity,
                        dimensions={
                            "requirement": "integrity_disposition",
                            "finding_id": finding_id,
                        },
                        observed=0,
                        target=1,
                        reason="unresolved_critical_integrity_finding",
                        denominator=1,
                        evidence_ids=(finding_id,),
                    )
                )
        finding_rows.append(
            {
                "finding_id": finding_id,
                "candidate_members": list(candidates),
                "confirmed_members": list(confirmed),
                "high_priority": is_high,
                "integrity": is_integrity,
                "status": status,
                "representative_episode_id": representative,
                "representative_reviewed": representative_reviewed,
                "control_episode_ids": list(control_ids),
                "control_reviewed": bool(control_ids and control_reviewed),
            }
        )
    finding_rows.sort(key=lambda item: (item["finding_id"], canonical_json(item)))
    return (
        {
            "candidate_clusters": len(finding_rows),
            "candidate_members": candidate_count,
            "confirmed_members": confirmed_count,
            "high_priority_clusters": high_priority,
            "representatives_reviewed": representatives,
            "context_controls_reviewed": controls,
            "unresolved_critical_integrity": unresolved_integrity,
            "rows": finding_rows,
        },
        deficits,
    )


def _materialization_summary(value: Any) -> tuple[dict[str, int], list[Mapping[str, Any]]]:
    values = _sequence(value)
    if isinstance(value, Mapping) and not any(
        key in value for key in ("status", "fidelity", "materialization")
    ):
        values = ()
    statuses = Counter()
    rows: list[Mapping[str, Any]] = []
    for item in values:
        raw = _mapping(item, name="materialization[]")
        status = (
            _token(raw.get("status") or raw.get("fidelity"), default="unavailable")
            .lower()
            .replace("-", "_")
        )
        if status in {"verified", "complete", "matched", "ok", "success"}:
            status = "verified"
        elif status in {"diverged", "mismatch", "failed", "error"}:
            status = "diverged"
        elif status not in {"unverifiable", "unavailable"}:
            status = "unavailable"
        statuses[status] += 1
        rows.append({**raw, "status": status})
    return (
        {
            "total": sum(statuses.values()),
            "verified": statuses["verified"],
            "diverged": statuses["diverged"],
            "unverifiable": statuses["unverifiable"],
            "unavailable": statuses["unavailable"],
        },
        rows,
    )


def _identity_from_scan(  # noqa: C901
    scan: Any,
    protocol: AuditProtocol,
    supplied: AuditIdentity | Mapping[str, Any] | None,
    release_digest: str | None,
    *,
    detector_registry: Any = None,
    protocol_is_explicit: bool = False,
) -> AuditIdentity:
    scan_map = _scan_mapping(scan)
    audit = _field(scan, "audit", default=None)
    audit_map = _mapping(audit, name="scan.audit") if audit is not None else {}
    provenance = _field(scan, "provenance", default={})
    provenance = provenance if isinstance(provenance, Mapping) else {}
    source = provenance.get("source") if isinstance(provenance.get("source"), Mapping) else {}
    config = provenance.get("config") if isinstance(provenance.get("config"), Mapping) else {}
    audit_metadata = audit_map.get("metadata")
    audit_metadata = audit_metadata if isinstance(audit_metadata, Mapping) else {}
    registry = detector_registry
    if registry is None:
        registry = _field(scan, "detector_registry", default=None)
    registry_digest = _field(registry, "digest", default="")
    if not registry_digest and registry is not None:
        try:
            if isinstance(registry, Mapping):
                registry_digest = _digest(dict(registry))
            elif isinstance(registry, Sequence) and not isinstance(
                registry, (str, bytes, bytearray)
            ):
                registry_digest = _digest(tuple(registry))
            else:
                registry_digest = _digest(_mapping(registry, name="detector_registry"))
        except AuditCoverageError:
            registry_digest = ""
    base = {
        "campaign_digest": audit_map.get("campaign_digest")
        or scan_map.get("campaign_digest")
        or scan_map.get("campaign_id", ""),
        "source_digest": audit_map.get("source_digest")
        or scan_map.get("source_digest")
        or source.get("sha256_observed", ""),
        "release_digest": release_digest
        or audit_map.get("release_digest")
        or audit_map.get("release_id")
        or audit_metadata.get("release_digest")
        or audit_metadata.get("release_id")
        or scan_map.get("release_digest")
        or scan_map.get("release_id", ""),
        "protocol_version": protocol.version,
        "protocol_digest": protocol.digest,
        "detector_registry_digest": registry_digest or scan_map.get("detector_registry_digest", ""),
        "detector_config_digest": scan_map.get("detector_config_digest")
        or provenance.get("detector_config_digest", "")
        or config.get("detector_config_digest")
        or config.get("config_identity")
        or source.get("config_identity")
        or audit_metadata.get("config_identity")
        or audit_map.get("config_identity", "")
        or (_digest(config) if config else ""),
        "source_revision": scan_map.get("source_revision") or source.get("source_commit") or "",
        "scan_revision": scan_map.get("scan_revision")
        or _field(scan, "cache_key", "report_digest", default=""),
        "review_revision": scan_map.get("review_revision", ""),
        "coverage_receipt_revision": scan_map.get("coverage_receipt_revision", ""),
    }
    if supplied is not None:
        given = _coerce_identity(supplied).to_dict(include_digest=False)
        for key, value in given.items():
            if not value:
                continue
            if key in {"protocol_version", "protocol_digest"}:
                # The serialized protocol is the local BA-04 authority.  A
                # caller-supplied digest cannot silently describe a different
                # configuration than the one evaluated here.
                continue
            elif not base.get(key):
                base[key] = value
    base["protocol_version"] = protocol.version
    base["protocol_digest"] = protocol.digest
    return AuditIdentity(**{key: str(value or "") for key, value in base.items()})


def _prior_receipt_matches(receipt: Any, identity: AuditIdentity) -> bool:
    if receipt is None:
        return True
    return receipt_matches_identity(receipt, identity)


def _diagnostic_summary(
    scan: Any, diagnostics: Sequence[Any] | Mapping[str, Any] | None
) -> dict[str, int]:
    if diagnostics is None:
        counts = _scan_counts(scan).get("diagnostic", {})
        if isinstance(counts, Mapping):
            return {str(key): int(value) for key, value in counts.items() if isinstance(value, int)}
        values: tuple[Any, ...] = ()
    elif isinstance(diagnostics, Mapping):
        return {
            str(key): int(value) for key, value in diagnostics.items() if isinstance(value, int)
        }
    else:
        values = _sequence(diagnostics)
    statuses = Counter(
        _token(_field(item, "status", default="unknown"), default="unknown") for item in values
    )
    return {
        "requested": len(values),
        "executed": sum(
            count for status, count in statuses.items() if status not in {"requested", "unknown"}
        ),
        "successful": sum(
            count for status, count in statuses.items() if status in {"complete", "success"}
        ),
        "failed": sum(count for status, count in statuses.items() if status in {"failed", "error"}),
        "original_recorded_evidence": 0,
        "new_diagnostic_executions": len(values),
    }


def evaluate_coverage(  # noqa: C901, PLR0912, PLR0913, PLR0915
    scan_summary: Any = None,
    records: Sequence[Any] = (),
    *,
    protocol: AuditProtocol | Mapping[str, Any] | None = None,
    identity: AuditIdentity | Mapping[str, Any] | None = None,
    scan: Any = None,
    review_records: Sequence[Any] | None = None,
    expected_rows: Sequence[Any] | None = None,
    detector_attempts: Sequence[Any] | None = None,
    detector_registry: Any = None,
    findings: Sequence[Any] | None = None,
    integrity_findings: Sequence[Any] | None = None,
    materialization: Any = None,
    exceptions: Sequence[Any] | Mapping[str, Any] | None = None,
    diagnostics: Sequence[Any] | Mapping[str, Any] | None = None,
    prior_receipt: Any = None,
    previous_receipt: Any = None,
    release_digest: str | None = None,
) -> AuditHealthReport:
    """Evaluate release-bound coverage over BA-01 and BA-03 evidence.

    ``scan_summary`` may be an :class:`AuditScanReport`, a serialized report,
    or a compact mapping.  ``records``/``review_records`` are intentionally
    consumed as typed BA-03 review receipts; agent and interval records never
    satisfy a full human requirement.
    Returns:
        Deterministic release-bound health report.
    """

    source_scan = scan if scan is not None else scan_summary
    active_protocol = _coerce_protocol(protocol)
    active_identity = _identity_from_scan(
        source_scan,
        active_protocol,
        identity,
        release_digest,
        detector_registry=detector_registry,
        protocol_is_explicit=protocol is not None,
    )
    rows, inventory = _normalize_rows(
        source_scan,
        expected_rows,
        scenario_group_fields=active_protocol.scenario_group_fields,
        ordinary_candidate_fields=active_protocol.ordinary_candidate_fields,
    )
    attempts = _normalize_signals(source_scan, detector_attempts, rows, detector_registry)
    review_summary, full_human_ids = _normalize_reviews(
        _review_values(source_scan, records, review_records), rows, active_identity
    )
    readable_ids = {row.episode_id for row in rows if row.readable}
    exceptions_normalized = _exception_entries(active_protocol, exceptions)
    deficits: list[CoverageDeficit] = []
    if not active_identity.complete:
        missing = [
            key
            for key, value in active_identity.to_dict(include_digest=False).items()
            if key
            in {
                "campaign_digest",
                "source_digest",
                "release_digest",
                "protocol_digest",
                "detector_registry_digest",
                "detector_config_digest",
            }
            and not value
        ]
        deficits.append(
            _build_deficit(
                active_identity,
                dimensions={"requirement": "identity"},
                observed=0,
                target=1,
                reason="missing_identity:" + ",".join(missing),
                status=DEFICIT_ERROR,
            )
        )
    if active_protocol.account_all_expected_rows:
        for status in ("missing", "duplicate", "invalid", "unsupported"):
            amount = inventory[status]
            if amount:
                deficit_status = (
                    DEFICIT_ERROR
                    if status in {"duplicate", "invalid", "unsupported"}
                    else DEFICIT_UNAVAILABLE
                )
                deficits.append(
                    _build_deficit(
                        active_identity,
                        dimensions={"requirement": "expected_rows", "row_status": status},
                        observed=inventory["expected"] - amount,
                        target=inventory["expected"],
                        reason=f"expected_rows_{status}",
                        status=deficit_status,
                        denominator=inventory["expected"],
                    )
                )
        if not inventory["expected"]:
            deficits.append(
                _build_deficit(
                    active_identity,
                    dimensions={"requirement": "expected_rows"},
                    observed=0,
                    target=1,
                    reason="empty_expected_rows",
                    status=DEFICIT_UNAVAILABLE,
                )
            )
    detector_counts = Counter(item.status for item in attempts)
    by_detector: dict[str, dict[str, int]] = {}
    for detector_id in sorted(
        {item.detector_id for item in attempts} | set(_detector_ids(source_scan, detector_registry))
    ):
        detector_counts_for_id = Counter(
            item.status for item in attempts if item.detector_id == detector_id
        )
        by_detector[detector_id] = {
            "scheduled": sum(detector_counts_for_id.values()),
            "evaluable": detector_counts_for_id["flagged"] + detector_counts_for_id["clear"],
            "flagged": detector_counts_for_id["flagged"],
            "clear": detector_counts_for_id["clear"],
            "unavailable": detector_counts_for_id["unavailable"],
            "error": detector_counts_for_id["error"],
        }
    detector_summary = {
        "scheduled": len(attempts),
        "evaluable": detector_counts["flagged"] + detector_counts["clear"],
        "flagged": detector_counts["flagged"],
        "clear": detector_counts["clear"],
        "unavailable": detector_counts["unavailable"],
        "error": detector_counts["error"],
        "by_detector": by_detector,
    }
    if active_protocol.require_detector_attempts:
        typed_registry = _typed_detector_registry(source_scan, detector_registry)
        detector_ids = _detector_ids(source_scan, detector_registry)
        if readable_ids and (typed_registry is None or not detector_ids):
            deficits.append(
                _build_deficit(
                    active_identity,
                    dimensions={"requirement": "detector_attempts"},
                    observed=0,
                    target=len(readable_ids),
                    reason="missing_detector_registry_or_attempts",
                    status=DEFICIT_ERROR,
                    denominator=len(readable_ids),
                )
            )
        for status in ("unavailable", "error"):
            if detector_counts[status]:
                deficits.append(
                    _build_deficit(
                        active_identity,
                        dimensions={"requirement": "detector_attempts", "result_status": status},
                        observed=detector_summary["evaluable"],
                        target=detector_summary["scheduled"],
                        reason=f"detector_attempt_{status}",
                        status=DEFICIT_UNAVAILABLE if status == "unavailable" else DEFICIT_ERROR,
                        denominator=detector_summary["scheduled"],
                    )
                )
    stratum_rows: dict[tuple[str, str, str], list[ExpectedEpisode]] = defaultdict(list)
    for row in rows:
        if row.readable:
            stratum_rows[_row_stratum(row)].append(row)
    strata: list[dict[str, Any]] = []
    for planner, group, outcome in sorted(stratum_rows):
        members = stratum_rows[(planner, group, outcome)]
        reviewed = sorted({row.episode_id for row in members} & full_human_ids)
        target = active_protocol.full_human_review_target
        row = {
            "stratum_id": _stratum_id(
                {"planner_id": planner, "scenario_group": group, "outcome": outcome}
            ),
            "planner_id": planner,
            "scenario_group": group,
            "outcome": outcome,
            "denominator": len(members),
            "full_human_reviewed": len(reviewed),
            "target": target,
            "status": "met" if len(reviewed) >= target else "under_review",
            "reviewed_episode_ids": reviewed,
        }
        strata.append(row)
        if len(reviewed) < target:
            deficits.append(
                _build_deficit(
                    active_identity,
                    dimensions={
                        "requirement": "full_human_review",
                        "planner_id": planner,
                        "scenario_group": group,
                        "outcome": outcome,
                    },
                    observed=len(reviewed),
                    target=target,
                    reason="missing_full_human_review",
                    denominator=len(members),
                    evidence_ids=reviewed,
                )
            )
    control_groups: dict[tuple[str, str], list[ExpectedEpisode]] = defaultdict(list)
    for row in rows:
        if row.readable and row.ordinary_control:
            control_groups[(row.planner_id, row.scenario_group)].append(row)
    control_gaps: list[dict[str, Any]] = []
    control_reviewed = 0
    for (planner, group), members in sorted(control_groups.items()):
        reviewed = sorted({row.episode_id for row in members} & full_human_ids)
        control_reviewed += min(len(reviewed), active_protocol.ordinary_control_review_target)
        if len(reviewed) < active_protocol.ordinary_control_review_target:
            control_gaps.append(
                {
                    "planner_id": planner,
                    "scenario_group": group,
                    "candidate_ids": [row.episode_id for row in members],
                }
            )
            deficits.append(
                _build_deficit(
                    active_identity,
                    dimensions={
                        "requirement": "ordinary_control",
                        "planner_id": planner,
                        "scenario_group": group,
                    },
                    observed=len(reviewed),
                    target=active_protocol.ordinary_control_review_target,
                    reason="missing_ordinary_control_review",
                    denominator=len(members),
                    evidence_ids=reviewed,
                )
            )
    review_summary.update(
        {
            "full_human_strata": sum(1 for row in strata if row["status"] == "met"),
            "observed_strata": len(strata),
            "ordinary_control_candidates": sum(len(item) for item in control_groups.values()),
            "ordinary_control_groups": len(control_groups),
            "ordinary_control_required": (
                len(control_groups) * active_protocol.ordinary_control_review_target
            ),
            "ordinary_control_reviewed": control_reviewed,
            "ordinary_control_gap_groups": control_gaps,
        }
    )
    all_findings: tuple[Any, ...] = tuple(findings or ()) + tuple(integrity_findings or ())
    finding_summary, finding_deficits = _finding_summary(
        all_findings,
        full_human_ids,
        readable_ids,
        active_identity,
        active_protocol,
        exceptions_normalized,
    )
    deficits.extend(finding_deficits)
    materialization_counts, materialization_rows = _materialization_summary(materialization)
    if materialization is None:
        materialization_counts, materialization_rows = _materialization_summary(
            _scan_mapping(source_scan).get("materialization")
        )
    for item in materialization_rows:
        if item["status"] in {"diverged", "unverifiable"}:
            deficits.append(
                _build_deficit(
                    active_identity,
                    dimensions={
                        "requirement": "materialization_fidelity",
                        "episode_id": _token(item.get("episode_id"), default="unknown"),
                    },
                    observed=0,
                    target=1,
                    reason="materialization_mismatch"
                    if item["status"] == "diverged"
                    else "materialization_unverifiable",
                    status=DEFICIT_ERROR if item["status"] == "diverged" else DEFICIT_UNAVAILABLE,
                )
            )
    diagnostic_summary = _diagnostic_summary(source_scan, diagnostics)
    if not _prior_receipt_matches(
        prior_receipt if prior_receipt is not None else previous_receipt, active_identity
    ):
        deficits.append(
            _build_deficit(
                active_identity,
                dimensions={"requirement": "completion_receipt"},
                observed=0,
                target=1,
                reason="stale_completion_receipt",
                status=DEFICIT_ERROR,
            )
        )
    final_deficits: list[CoverageDeficit] = []
    applied_exception_indexes: set[int] = set()
    for deficit in sorted(deficits, key=lambda item: (item.stratum_id, item.reason, item.status)):
        exception_match = next(
            (
                (index, item)
                for index, item in enumerate(exceptions_normalized)
                if index not in applied_exception_indexes
                and item.get("status") in _EXCEPTION_STATUSES
                and _exception_matches(deficit, item)
            ),
            None,
        )
        if exception_match is not None:
            exception_index, _ = exception_match
            applied_exception_indexes.add(exception_index)
            final_deficits.append(replace(deficit, status=DEFICIT_WAIVED))
        else:
            final_deficits.append(deficit)
    applied_exceptions = [
        item
        for index, item in enumerate(exceptions_normalized)
        if index in applied_exception_indexes
    ]
    unwaived = [item for item in final_deficits if item.status != DEFICIT_WAIVED]
    status = STATUS_INCOMPLETE
    if not unwaived:
        status = (
            STATUS_COMPLETE_WITH_DECLARED_EXCEPTIONS
            if final_deficits or applied_exceptions
            else STATUS_COMPLETE_UNDER_PROTOCOL
        )
    counts = {
        "coverage": dict(inventory),
        "inventory": dict(inventory),
        "detectors": detector_summary,
        "reviews": review_summary,
        "review": review_summary,
        "findings": {key: value for key, value in finding_summary.items() if key != "rows"},
        "materialization": materialization_counts,
        "diagnostics": diagnostic_summary,
        "status": status,
    }
    limitations = (
        "Operational coverage accounting is not a statistical confidence bound.",
        "Coverage does not prove benchmark validity, absence of bugs, or causal explanations.",
        "Only typed BA-03 full_episode human ReviewRecord values satisfy full human coverage.",
        "Agent, interval, replay, duplicate, similarity, and regenerated execution evidence cannot inflate original episode denominators.",
    )
    return AuditHealthReport(
        identity=active_identity,
        protocol=active_protocol,
        status=status,
        counts=counts,
        strata=tuple(strata),
        deficits=tuple(final_deficits),
        exceptions=tuple(applied_exceptions),
        findings=finding_summary,
        materialization=materialization_counts,
        diagnostics=diagnostic_summary,
        limitations=limitations,
    )


evaluate_audit = evaluate_coverage
build_audit_health_report = evaluate_coverage


def receipt_matches_identity(
    receipt: Mapping[str, Any] | AuditHealthReport, identity: AuditIdentity | Mapping[str, Any]
) -> bool:
    """Return whether a complete-shape receipt belongs to the exact identity.

    The digest is an unkeyed integrity token, not a signature or proof that the
    referenced report was produced by a trusted evaluator.  Callers needing
    admission evidence must retain and validate the report itself as well.
    """

    active_identity = _coerce_identity(identity)
    candidate = receipt.completion_receipt if isinstance(receipt, AuditHealthReport) else receipt
    return _valid_completion_receipt(candidate, active_identity)


def is_completion_current(
    receipt: Mapping[str, Any] | AuditHealthReport, identity: AuditIdentity | Mapping[str, Any]
) -> bool:
    """Compatibility alias for release/protocol completion checks.

    Returns:
        ``True`` only when a completed receipt belongs to the exact identity.
    """

    active_identity = _coerce_identity(identity)
    if isinstance(receipt, AuditHealthReport):
        return receipt.complete and _valid_completion_receipt(
            receipt.completion_receipt, active_identity, require_complete=True
        )
    return _valid_completion_receipt(receipt, active_identity, require_complete=True)


_RECEIPT_IDENTITY_FIELDS = (
    "campaign_digest",
    "source_digest",
    "release_digest",
    "protocol_version",
    "protocol_digest",
    "detector_registry_digest",
    "detector_config_digest",
    "source_revision",
    "scan_revision",
    "review_revision",
    "coverage_receipt_revision",
)
_RECEIPT_FIELDS = frozenset(
    {"schema_version", "status", "identity_digest", "report_digest"} | set(_RECEIPT_IDENTITY_FIELDS)
)


def _is_hex_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _valid_completion_receipt(
    receipt: Any,
    identity: AuditIdentity,
    *,
    require_complete: bool = False,
) -> bool:
    """Validate the closed v1 receipt envelope and identity directions.

    Returns:
        Whether the receipt has the required shape and identity values.
    """

    if not isinstance(receipt, Mapping) or set(receipt) != _RECEIPT_FIELDS:
        return False
    if receipt.get("schema_version") != "audit-completion-receipt.v1":
        return False
    status = receipt.get("status")
    if status not in AUDIT_STATUSES:
        return False
    if require_complete and status not in {
        STATUS_COMPLETE_WITH_DECLARED_EXCEPTIONS,
        STATUS_COMPLETE_UNDER_PROTOCOL,
    }:
        return False
    if not _is_hex_digest(receipt.get("identity_digest")):
        return False
    if not _is_hex_digest(receipt.get("report_digest")):
        return False
    if receipt.get("identity_digest") != identity.digest:
        return False
    return all(receipt.get(field) == getattr(identity, field) for field in _RECEIPT_IDENTITY_FIELDS)


def _report_document_for_renderer(report: AuditHealthReport) -> dict[str, Any]:
    """Adapt health fields to the existing SREV renderer without new metrics.

    Returns:
        Renderer-compatible scenario report mapping.
    """

    claims: list[dict[str, Any]] = [
        {
            "claim_id": "health-status",
            "category": "observation",
            "statement": f"Audit status: {report.status}.",
            "status": "valid" if report.status != STATUS_INCOMPLETE else "partial",
            "citation_ids": (),
        },
        {
            "claim_id": "health-counts",
            "category": "observation",
            "statement": f"Coverage counts: {canonical_json(report.counts)}.",
            "status": "valid",
            "citation_ids": (),
        },
    ]
    for index, deficit in enumerate(report.deficits, start=1):
        claims.append(
            {
                "claim_id": f"deficit-{index}",
                "category": "diagnostic_hypothesis",
                "statement": f"{deficit.reason} ({deficit.stratum_id}).",
                "status": "valid" if deficit.status == DEFICIT_WAIVED else "failed",
                "citation_ids": (),
                "reason": deficit.status,
            }
        )
    missing = [
        {"measurement": item.reason, "unavailable_reason": item.status}
        for item in report.deficits
        if item.status in {DEFICIT_UNAVAILABLE, DEFICIT_ERROR}
    ]
    # BA-04 emits a health document, not an episode-media excerpt.  Give the
    # generic SREV renderer an explicit unavailable marker so it cannot infer
    # that a zero-length excerpt means the full episode was displayed or that
    # all telemetry measurements were present.
    missing.append(
        {
            "measurement": "episode_media",
            "unavailable_reason": "not materialized by the BA-04 health report",
        }
    )
    return {
        "schema_version": "review-report.v1",
        "report_id": f"audit-health-{report.report_digest}",
        "request_id": report.identity.digest,
        "title": "Benchmark Audit Health Report",
        "evidence_boundary": "Operational coverage report; not benchmark validity or statistical confidence evidence.",
        "provenance": {
            "component_id": "ba04-audit-coverage",
            "component_version": AUDIT_COVERAGE_SCHEMA_VERSION,
            "source_digests": {
                "campaign": report.identity.campaign_digest,
                "source": report.identity.source_digest,
                "release": report.identity.release_digest,
                "protocol": report.identity.protocol_digest,
                "detector_registry": report.identity.detector_registry_digest,
                "detector_config": report.identity.detector_config_digest,
            },
        },
        "excerpt": {
            "source_interval": {"start_s": 0.0, "end_s": 0.0},
            "omitted_intervals": [{"start_s": 0.0, "end_s": 0.0}],
            "full_episode_link": "not_available",
        },
        "claims": claims,
        "citations": [],
        "missing_measurements": missing,
        "limitations": list(report.limitations),
    }


def render_markdown(report: AuditHealthReport) -> str:
    """Render a health report using the canonical scenario-review renderer.

    Returns:
        Deterministic Markdown health report.
    """

    return render_review_markdown(_report_document_for_renderer(report))


def render_html(report: AuditHealthReport) -> str:
    """Render a health report as escaped standalone HTML.

    Returns:
        Deterministic standalone HTML health report.
    """

    return render_review_html(_report_document_for_renderer(report))


@lru_cache(maxsize=1)
def load_audit_coverage_schema() -> dict[str, Any]:
    """Load the versioned BA-04 JSON schema.

    Returns:
        Parsed JSON Schema mapping.
    """

    return json.loads(AUDIT_COVERAGE_SCHEMA_FILE.read_text(encoding="utf-8"))


def _canonical_equal(left: Any, right: Any) -> bool:
    """Compare report subdocuments using the same canonical JSON rules.

    Returns:
        Whether both values have identical canonical JSON representations.
    """

    return canonical_json(left) == canonical_json(right)


def _serialized_exception_matches(deficit: Mapping[str, Any], exception: Mapping[str, Any]) -> bool:
    """Apply the evaluator's one-deficit exception matching rule to JSON.

    Returns:
        Whether the exception identifies the serialized deficit.
    """

    requirement = exception.get("requirement")
    dimensions = deficit.get("dimensions", {})
    requirement_values = {
        deficit.get("reason"),
        deficit.get("stratum_id"),
        dimensions.get("requirement") if isinstance(dimensions, Mapping) else None,
        "*",
    }
    if requirement not in requirement_values:
        return False
    return exception.get("stratum_id") in {None, "*", deficit.get("stratum_id")}


def _validate_exception_links(
    deficits: Sequence[Mapping[str, Any]], exceptions: Sequence[Mapping[str, Any]]
) -> None:
    """Require a one-to-one link between exceptions and waived deficits."""

    waived_indexes = {
        index for index, deficit in enumerate(deficits) if deficit.get("status") == DEFICIT_WAIVED
    }
    matched_indexes: set[int] = set()
    for exception in exceptions:
        matches = [
            index
            for index, deficit in enumerate(deficits)
            if index in waived_indexes and _serialized_exception_matches(deficit, exception)
        ]
        if len(matches) != 1:
            raise AuditCoverageError(
                "each declared exception must match exactly one waived deficit"
            )
        if matches[0] in matched_indexes:
            raise AuditCoverageError("multiple declared exceptions match one waived deficit")
        matched_indexes.add(matches[0])
    if matched_indexes != waived_indexes:
        raise AuditCoverageError("each waived deficit requires one declared exception")


def _validate_report_accounting(payload: Mapping[str, Any], protocol: AuditProtocol) -> None:  # noqa: C901, PLR0912
    """Validate inventory, stratum, and review numerator/denominator equations."""

    counts = payload["counts"]
    coverage = counts["coverage"]
    expected_components = sum(
        coverage[key] for key in ("readable", "missing", "duplicate", "invalid", "unsupported")
    )
    indexed_components = sum(
        coverage[key] for key in ("readable", "duplicate", "invalid", "unsupported")
    )
    if coverage["expected"] != expected_components:
        raise AuditCoverageError("inventory expected count is inconsistent with row statuses")
    if coverage["indexed"] != indexed_components:
        raise AuditCoverageError("inventory indexed count is inconsistent with row statuses")
    if coverage["expected"] != coverage["indexed"] + coverage["missing"]:
        raise AuditCoverageError("inventory expected/indexed/missing equation is inconsistent")

    strata = payload["strata"]
    review_counts = counts["reviews"]
    if review_counts["observed_strata"] != len(strata):
        raise AuditCoverageError("review observed_strata does not match strata")
    if review_counts["full_human_strata"] != sum(item["status"] == "met" for item in strata):
        raise AuditCoverageError("review full_human_strata does not match strata")
    if sum(item["denominator"] for item in strata) != coverage["readable"]:
        raise AuditCoverageError("stratum denominators do not partition readable rows")

    full_human_ids = review_counts["full_human_ids"]
    human_ids = review_counts["human_ids"]
    if len(full_human_ids) != len(set(full_human_ids)):
        raise AuditCoverageError("full_human_ids contains duplicates")
    if len(human_ids) != len(set(human_ids)):
        raise AuditCoverageError("human_ids contains duplicates")
    if review_counts["full_episode_human"] != len(full_human_ids):
        raise AuditCoverageError("full_episode_human does not match full_human_ids")
    if not set(full_human_ids).issubset(human_ids):
        raise AuditCoverageError("full_human_ids must be a subset of human_ids")

    stratum_ids: set[str] = set()
    reviewed_ids: list[str] = []
    for stratum in strata:
        expected_id = _stratum_id(
            {
                "planner_id": stratum["planner_id"],
                "scenario_group": stratum["scenario_group"],
                "outcome": stratum["outcome"],
            }
        )
        if stratum["stratum_id"] != expected_id:
            raise AuditCoverageError("stratum_id does not match stratum dimensions")
        if stratum["stratum_id"] in stratum_ids:
            raise AuditCoverageError("strata contain duplicate stratum_id values")
        stratum_ids.add(stratum["stratum_id"])
        if stratum["denominator"] <= 0:
            raise AuditCoverageError("observed strata require a positive denominator")
        reviewed = stratum["reviewed_episode_ids"]
        if len(reviewed) != len(set(reviewed)):
            raise AuditCoverageError("stratum reviewed_episode_ids contains duplicates")
        if stratum["full_human_reviewed"] != len(reviewed):
            raise AuditCoverageError("stratum review numerator does not match reviewed IDs")
        if not 0 <= stratum["full_human_reviewed"] <= stratum["denominator"]:
            raise AuditCoverageError("stratum review numerator exceeds its denominator")
        if stratum["target"] != protocol.full_human_review_target:
            raise AuditCoverageError("stratum review target does not match protocol")
        expected_status = (
            "met" if stratum["full_human_reviewed"] >= stratum["target"] else "under_review"
        )
        if stratum["status"] != expected_status:
            raise AuditCoverageError("stratum status does not match review numerator and target")
        reviewed_ids.extend(reviewed)
    if len(reviewed_ids) != len(set(reviewed_ids)):
        raise AuditCoverageError("reviewed episode IDs occur in multiple strata")
    if set(reviewed_ids) != set(full_human_ids):
        raise AuditCoverageError("strata reviewed IDs do not match full_human_ids")


def _validate_complete_matrix(  # noqa: C901, PLR0912
    payload: Mapping[str, Any], protocol: AuditProtocol, *, allow_exceptions: bool
) -> None:
    """Validate the derived evidence matrix behind a complete status."""

    counts = payload["counts"]
    coverage = counts["coverage"]
    deficits = payload["deficits"]

    def covered(reason: str, dimensions: Mapping[str, Any] | None = None) -> bool:
        if not allow_exceptions:
            return False
        expected = {str(key): str(value) for key, value in (dimensions or {}).items()}
        return any(
            item.get("status") == DEFICIT_WAIVED
            and item.get("reason") == reason
            and all(
                str(item.get("dimensions", {}).get(key)) == value for key, value in expected.items()
            )
            for item in deficits
        )

    def require_gap(
        condition: bool,
        reason: str,
        dimensions: Mapping[str, Any] | None = None,
    ) -> None:
        if condition and not covered(reason, dimensions):
            raise AuditCoverageError(f"complete report has unaccounted {reason}")

    if protocol.account_all_expected_rows:
        for row_status in ("missing", "duplicate", "invalid", "unsupported"):
            require_gap(
                coverage[row_status] > 0,
                f"expected_rows_{row_status}",
                {"requirement": "expected_rows", "row_status": row_status},
            )
        require_gap(
            coverage["expected"] == 0,
            "empty_expected_rows",
            {"requirement": "expected_rows"},
        )

    reviews = counts["reviews"]
    strata = payload["strata"]
    if reviews["full_human_strata"] != sum(item["status"] == "met" for item in strata):
        raise AuditCoverageError("review full_human_strata does not match strata")
    for stratum in strata:
        if stratum["target"] != protocol.full_human_review_target:
            raise AuditCoverageError("stratum review target does not match protocol")
        unmet = stratum["status"] != "met" or (stratum["full_human_reviewed"] < stratum["target"])
        require_gap(
            unmet,
            "missing_full_human_review",
            {
                "requirement": "full_human_review",
                "planner_id": stratum["planner_id"],
                "scenario_group": stratum["scenario_group"],
                "outcome": stratum["outcome"],
            },
        )

    groups = reviews["ordinary_control_groups"]
    target = protocol.ordinary_control_review_target
    required = groups * target
    if reviews["ordinary_control_required"] != required:
        raise AuditCoverageError("ordinary control requirement is inconsistent")
    if groups > reviews["ordinary_control_candidates"]:
        raise AuditCoverageError("ordinary control groups exceed candidates")
    if reviews["ordinary_control_reviewed"] > required:
        raise AuditCoverageError("ordinary control reviews exceed the target")
    gaps = reviews["ordinary_control_gap_groups"]
    if target == 0 and gaps:
        raise AuditCoverageError("ordinary control gaps exist despite a zero target")
    if reviews["ordinary_control_reviewed"] < required and not gaps:
        raise AuditCoverageError("ordinary control review count is below target")
    for gap in gaps:
        require_gap(
            True,
            "missing_ordinary_control_review",
            {
                "requirement": "ordinary_control",
                "planner_id": gap.get("planner_id"),
                "scenario_group": gap.get("scenario_group"),
            },
        )

    detectors = counts["detectors"]
    if protocol.require_detector_attempts:
        require_gap(
            coverage["readable"] > 0 and not detectors["by_detector"],
            "missing_detector_registry_or_attempts",
            {"requirement": "detector_attempts"},
        )
        for signal_status in ("unavailable", "error"):
            require_gap(
                detectors[signal_status] > 0,
                f"detector_attempt_{signal_status}",
                {"requirement": "detector_attempts", "result_status": signal_status},
            )
        if detectors["evaluable"] != detectors["scheduled"] and not allow_exceptions:
            raise AuditCoverageError("complete report contains non-evaluable detector attempts")

    materialization = payload["materialization"]
    require_gap(
        materialization["diverged"] > 0,
        "materialization_mismatch",
    )
    require_gap(
        materialization["unverifiable"] > 0,
        "materialization_unverifiable",
    )

    finding_counts = payload["findings"]
    if protocol.require_integrity_disposition:
        require_gap(
            finding_counts["unresolved_critical_integrity"] > 0,
            "unresolved_critical_integrity_finding",
        )
    finding_rows = finding_counts["rows"]
    if finding_counts["representatives_reviewed"] != sum(
        bool(item["representative_reviewed"]) for item in finding_rows
    ):
        raise AuditCoverageError("finding representative count is inconsistent")
    if finding_counts["context_controls_reviewed"] != sum(
        bool(item["control_reviewed"]) for item in finding_rows
    ):
        raise AuditCoverageError("finding context-control count is inconsistent")
    for finding in finding_rows:
        if finding["high_priority"] and protocol.anomaly_representative_target:
            require_gap(
                not finding["representative_reviewed"],
                "missing_finding_representative",
                {"requirement": "finding_representative", "finding_id": finding["finding_id"]},
            )
        if (
            finding["high_priority"]
            and finding["control_episode_ids"]
            and protocol.anomaly_context_control_target
        ):
            require_gap(
                not finding["control_reviewed"],
                "missing_finding_context_control",
                {"requirement": "finding_context_control", "finding_id": finding["finding_id"]},
            )


def _validate_report_semantics(  # noqa: C901, PLR0912, PLR0915
    payload: Mapping[str, Any],
) -> None:
    """Validate derived fields and cross-object status invariants."""

    identity_payload = payload["identity"]
    protocol_payload = payload["protocol"]
    identity = AuditIdentity.from_dict(identity_payload)
    protocol = AuditProtocol.from_dict(protocol_payload)
    if identity_payload["identity_digest"] != identity.digest:
        raise AuditCoverageError("identity_digest does not match identity contents")
    if protocol_payload["protocol_digest"] != protocol.digest:
        raise AuditCoverageError("protocol_digest does not match protocol contents")
    if identity.protocol_version != protocol.version:
        raise AuditCoverageError("identity protocol_version does not match protocol")
    if identity.protocol_digest != protocol.digest:
        raise AuditCoverageError("identity protocol_digest does not match protocol")

    declared_digest = payload["report_digest"]
    digest_payload = dict(payload)
    digest_payload.pop("report_digest")
    if declared_digest != _digest(digest_payload):
        raise AuditCoverageError("report_digest does not match report contents")

    counts = payload["counts"]
    if counts["status"] != payload["status"]:
        raise AuditCoverageError("counts.status does not match report status")
    if not _canonical_equal(counts["coverage"], counts["inventory"]):
        raise AuditCoverageError("counts.coverage and counts.inventory disagree")
    if not _canonical_equal(counts["reviews"], counts["review"]):
        raise AuditCoverageError("counts.reviews and counts.review disagree")
    if not _canonical_equal(counts["materialization"], payload["materialization"]):
        raise AuditCoverageError("materialization counts disagree")
    if not _canonical_equal(counts["diagnostics"], payload["diagnostics"]):
        raise AuditCoverageError("diagnostic counts disagree")
    finding_aggregate = {key: value for key, value in payload["findings"].items() if key != "rows"}
    if not _canonical_equal(counts["findings"], finding_aggregate):
        raise AuditCoverageError("finding counts disagree")

    strata = payload["strata"]
    review_counts = counts["reviews"]
    if review_counts["human_reviewed"] != len(review_counts["human_ids"]):
        raise AuditCoverageError("review human_reviewed does not match human_ids")
    if review_counts["agent_reviewed"] != len(review_counts["agent_ids"]):
        raise AuditCoverageError("review agent_reviewed does not match agent_ids")
    if review_counts["reviewed_episode_union"] != len(
        set(review_counts["human_ids"]) | set(review_counts["agent_ids"])
    ):
        raise AuditCoverageError("reviewed_episode_union does not match review ids")
    if review_counts["full_episode_human"] > review_counts["human_reviewed"]:
        raise AuditCoverageError("full_episode_human exceeds human_reviewed")
    if review_counts["full_episode_agent"] > review_counts["agent_reviewed"]:
        raise AuditCoverageError("full_episode_agent exceeds agent_reviewed")
    if review_counts["observed_strata"] != len(strata):
        raise AuditCoverageError("review observed_strata does not match strata")
    if review_counts["full_human_strata"] != sum(item["status"] == "met" for item in strata):
        raise AuditCoverageError("review full_human_strata does not match strata")
    _validate_report_accounting(payload, protocol)

    detectors = counts["detectors"]
    by_detector = detectors["by_detector"]
    totals = {
        key: sum(item[key] for item in by_detector.values())
        for key in ("scheduled", "evaluable", "flagged", "clear", "unavailable", "error")
    }
    if any(detectors[key] != value for key, value in totals.items()):
        raise AuditCoverageError("detector totals do not match by_detector")
    if detectors["evaluable"] != detectors["flagged"] + detectors["clear"]:
        raise AuditCoverageError("detector evaluable count is inconsistent")

    materialization = payload["materialization"]
    if materialization["total"] != sum(
        materialization[key] for key in ("verified", "diverged", "unverifiable", "unavailable")
    ):
        raise AuditCoverageError("materialization total is inconsistent")

    deficits = payload["deficits"]
    unwaived = [item for item in deficits if item["status"] != DEFICIT_WAIVED]
    status = payload["status"]
    if status != STATUS_INCOMPLETE and not identity.complete:
        raise AuditCoverageError("complete report requires a complete identity")
    if status != STATUS_INCOMPLETE and unwaived:
        raise AuditCoverageError("complete report contains unwaived deficits")
    if status == STATUS_INCOMPLETE and not unwaived:
        raise AuditCoverageError("incomplete report requires an unwaived deficit")
    if status == STATUS_COMPLETE_UNDER_PROTOCOL and (deficits or payload["exceptions"]):
        raise AuditCoverageError(
            "complete_under_protocol report cannot contain deficits or exceptions"
        )
    if status == STATUS_COMPLETE_WITH_DECLARED_EXCEPTIONS and not payload["exceptions"]:
        raise AuditCoverageError("complete_with_declared_exceptions requires exceptions")
    _validate_exception_links(deficits, payload["exceptions"])
    if status in {
        STATUS_COMPLETE_WITH_DECLARED_EXCEPTIONS,
        STATUS_COMPLETE_UNDER_PROTOCOL,
    }:
        _validate_complete_matrix(
            payload,
            protocol,
            allow_exceptions=status == STATUS_COMPLETE_WITH_DECLARED_EXCEPTIONS,
        )

    for deficit in deficits:
        for key in (
            "campaign_digest",
            "source_digest",
            "release_digest",
            "detector_registry_digest",
            "detector_config_digest",
            "coverage_receipt_revision",
        ):
            value = deficit[key]
            expected = getattr(identity, key)
            if value and expected and value != expected:
                raise AuditCoverageError(f"deficit.{key} does not match report identity")
        if deficit["protocol_revision"] and deficit["protocol_revision"] != protocol.version:
            raise AuditCoverageError("deficit.protocol_revision does not match protocol version")
        if deficit["protocol_id"] and deficit["protocol_id"] != protocol.digest:
            raise AuditCoverageError("deficit.protocol_id does not match protocol digest")
        if deficit["source_id"] and identity.source_digest:
            if deficit["source_id"] != identity.source_digest:
                raise AuditCoverageError("deficit.source_id does not match source digest")


def validate_audit_coverage(payload: Mapping[str, Any]) -> None:
    """Validate a serialized BA-04 report against its schema."""

    if not isinstance(payload, Mapping):
        raise AuditCoverageError("health report must be a mapping")
    _strict_json(payload)
    # canonical_json additionally rejects unsupported Python objects and is
    # configured with allow_nan=False, so direct Python callers receive the
    # same finite/deterministic contract as JSON deserializers.
    try:
        canonical_json(payload)
    except (AuditContractError, TypeError, ValueError, RecursionError) as exc:
        raise AuditCoverageError(f"health report is not strict JSON: {exc}") from exc
    errors = sorted(
        Draft202012Validator(load_audit_coverage_schema()).iter_errors(payload),
        key=lambda item: list(item.absolute_path),
    )
    if errors:
        detail = "; ".join(
            f"/{'/'.join(map(str, item.absolute_path))}: {item.message}" for item in errors
        )
        raise AuditCoverageError(detail)
    _validate_report_semantics(payload)


__all__ = [
    "AUDIT_COVERAGE_SCHEMA_FILE",
    "AUDIT_COVERAGE_SCHEMA_VERSION",
    "AUDIT_PROTOCOL_VERSION",
    "AUDIT_STATUSES",
    "DEFICIT_ERROR",
    "DEFICIT_MET",
    "DEFICIT_STATUSES",
    "DEFICIT_UNAVAILABLE",
    "DEFICIT_UNDER_REVIEW",
    "DEFICIT_WAIVED",
    "STATUS_COMPLETE_UNDER_PROTOCOL",
    "STATUS_COMPLETE_WITH_DECLARED_EXCEPTIONS",
    "STATUS_INCOMPLETE",
    "AuditCoverageError",
    "AuditHealthReport",
    "AuditIdentity",
    "AuditProtocol",
    "CoverageDeficit",
    "DetectorAttempt",
    "ExpectedEpisode",
    "ExpectedRow",
    "build_audit_health_report",
    "default_audit_protocol",
    "evaluate_audit",
    "evaluate_coverage",
    "is_completion_current",
    "load_audit_coverage_schema",
    "load_audit_protocol",
    "receipt_matches_identity",
    "render_html",
    "render_markdown",
    "validate_audit_coverage",
]
