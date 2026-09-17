"""Versioned contracts shared by the benchmark-audit workbench.

The audit package stores small, reviewable records.  It deliberately reuses
``SourceRef`` from :mod:`review_contracts` for source identity and keeps
timeline/trace references as links rather than copying raw telemetry.  The
classes in this module are plain dataclasses so the offline store, a browser
client, and an agent can use the same JSON representation without an ORM.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

from jsonschema import Draft202012Validator

from robot_sf.analysis_workbench.review_contracts import SourceRef

AUDIT_RECORD_SCHEMA_VERSION = "audit-record.v1"
AUDIT_SCHEMA_VERSION = AUDIT_RECORD_SCHEMA_VERSION
AUDIT_EXPORT_SCHEMA_VERSION = "audit-export.v1"
AUDIT_JOURNAL_SCHEMA_VERSION = "audit-journal.v1"
SUPPORTED_AUDIT_SCHEMA_VERSIONS = frozenset({AUDIT_RECORD_SCHEMA_VERSION})

ANNOTATION_CLASSIFICATIONS = (
    "normal",
    "interesting_valid",
    "planner_defect",
    "benchmark_defect",
    "scenario_defect",
    "instrumentation_defect",
    "unclear",
)
ANNOTATION_MODES = ("one_click", "quick", "full")
AUTHOR_KINDS = ("human", "detector", "agent")
REVIEW_SCOPES = ("interval", "full_episode")
FINDING_STATUSES = (
    "proposed",
    "under_investigation",
    "supported",
    "refuted",
    "resolved",
)
SIGNAL_STATUSES = ("flagged", "clear", "unavailable", "error")
COORDINATE_FRAMES = ("world", "image")

_DIGEST_LENGTHS = {40, 64}
_T = TypeVar("_T")
AUDIT_RECORD_SCHEMA_FILE = Path(__file__).with_name("schemas") / "audit_record.v1.json"


class AuditContractError(ValueError):
    """Raised when an audit record is not safe to persist."""


class UnsupportedAuditSchemaError(AuditContractError):
    """Raised when a record uses an unknown major/schema version."""


class AuditIdentityError(AuditContractError):
    """Raised when an identity is too weak to bind evidence."""


def utc_now() -> str:
    """Return a stable UTC timestamp for record metadata."""

    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _text(value: Any, *, name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise AuditContractError(f"{name} must be a non-empty string")
    return value


def _optional_text(value: Any, *, name: str) -> str:
    if value is None:
        return ""
    return _text(value, name=name, allow_empty=True)


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AuditContractError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise AuditContractError(f"{name} must be a finite number")
    return result


def _digest(value: Any, *, name: str, required: bool = False) -> str:
    if value is None:
        value = ""
    if not isinstance(value, str):
        raise AuditContractError(f"{name} must be a hexadecimal digest")
    if not value:
        if required:
            raise AuditIdentityError(f"{name} is required for evidence identity")
        return ""
    if len(value) not in _DIGEST_LENGTHS or any(
        char not in "0123456789abcdefABCDEF" for char in value
    ):
        raise AuditContractError(f"{name} must be a 40- or 64-character hexadecimal digest")
    return value.lower()


def _jsonable(value: Any) -> Any:
    """Convert supported contract values to strict JSON-compatible values.

    Returns:
        A JSON-compatible primitive, list, or mapping.
    """

    if is_dataclass(value):
        return {key: _jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        raise AuditContractError("records cannot contain non-finite numbers")
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise AuditContractError(f"unsupported value in audit record: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize a record deterministically for digests and NDJSON writes.

    Returns:
        Canonical strict-JSON text.
    """

    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


@lru_cache(maxsize=1)
def load_audit_record_schema() -> dict[str, Any]:
    """Load the public audit-record JSON schema.

    Returns:
        Parsed JSON Schema mapping.
    """

    return json.loads(AUDIT_RECORD_SCHEMA_FILE.read_text(encoding="utf-8"))


def validate_record(payload: Mapping[str, Any]) -> None:
    """Validate a canonical record envelope and its typed semantics.

    Args:
        payload: Mapping containing ``schema_version``, ``record_type``, and
            the record-specific fields.

    Raises:
        AuditContractError: If JSON Schema or typed validation fails.
    """

    if not isinstance(payload, Mapping):
        raise AuditContractError("record payload must be a mapping")
    errors = sorted(
        Draft202012Validator(load_audit_record_schema()).iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        details = "; ".join(
            f"/{'/'.join(map(str, error.absolute_path))}: {error.message}" for error in errors
        )
        raise AuditContractError(details)
    record_from_dict(payload)


def _coerce_source(value: SourceRef | Mapping[str, Any] | None) -> SourceRef | None:
    if value is None:
        return None
    if isinstance(value, SourceRef):
        return value
    if not isinstance(value, Mapping):
        raise AuditContractError("source must be a SourceRef or mapping")
    required = {"artifact_id", "uri", "format"}
    missing = sorted(required - set(value))
    if missing:
        raise AuditContractError(f"source is missing required fields: {', '.join(missing)}")
    known = {item.name for item in fields(SourceRef)}
    unknown = set(value) - known
    if unknown:
        raise AuditContractError(f"source contains unknown fields: {', '.join(sorted(unknown))}")
    return SourceRef(**{name: value.get(name, "") for name in known})


@dataclass(frozen=True, slots=True)
class TimeInterval:
    """A simulation-time interval, including a point when ``end_s`` is omitted."""

    start_s: float
    end_s: float | None = None

    def __post_init__(self) -> None:
        """Validate and normalize the interval endpoints."""

        start = _finite(self.start_s, name="interval.start_s")
        end = start if self.end_s is None else _finite(self.end_s, name="interval.end_s")
        if end < start:
            raise AuditContractError("interval.end_s must be >= interval.start_s")
        object.__setattr__(self, "start_s", start)
        object.__setattr__(self, "end_s", end)

    def to_dict(self) -> dict[str, float]:
        """Return the interval as JSON-compatible fields.

        Returns:
            A mapping with finite ``start_s`` and ``end_s`` values.
        """

        return {
            "start_s": self.start_s,
            "end_s": self.end_s if self.end_s is not None else self.start_s,
        }


@dataclass(frozen=True, slots=True)
class EpisodeRef:
    """Immutable identity of one recorded execution.

    Planner, scenario and seed are descriptive lookup fields only.  The
    generated ``episode_id`` includes campaign/source digests and execution
    identity so a rerun or changed checkpoint cannot silently collide.
    """

    campaign_digest: str = ""
    source_digest: str = ""
    execution_id: str = ""
    planner_id: str = ""
    scenario_id: str = ""
    seed: int | str | None = None
    attempt: int = 0
    config_digest: str = ""
    checkpoint_digest: str = ""
    environment_digest: str = ""
    source: SourceRef | None = None
    episode_id: str = ""
    campaign_id: str = ""
    source_id: str = ""

    def __post_init__(self) -> None:
        """Validate evidence identity and derive a collision-resistant ID."""

        campaign_digest = self.campaign_digest or self.campaign_id
        source_digest = self.source_digest or self.source_id
        execution_id = _text(self.execution_id, name="execution_id")
        campaign_digest = _digest(campaign_digest, name="campaign_digest", required=True)
        source_digest = _digest(source_digest, name="source_digest", required=True)
        config_digest = _digest(self.config_digest, name="config_digest")
        checkpoint_digest = _digest(self.checkpoint_digest, name="checkpoint_digest")
        environment_digest = _digest(self.environment_digest, name="environment_digest")
        if not isinstance(self.attempt, int) or isinstance(self.attempt, bool) or self.attempt < 0:
            raise AuditContractError("attempt must be a non-negative integer")
        for name, value in (
            ("planner_id", self.planner_id),
            ("scenario_id", self.scenario_id),
        ):
            _optional_text(value, name=name)
        if self.seed is not None and not isinstance(self.seed, (int, str)):
            raise AuditContractError("seed must be an integer, string, or null")
        source = _coerce_source(self.source)
        identity = {
            "campaign_digest": campaign_digest,
            "source_digest": source_digest,
            "execution_id": execution_id,
            "planner_id": self.planner_id,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "attempt": self.attempt,
            "config_digest": config_digest,
            "checkpoint_digest": checkpoint_digest,
            "environment_digest": environment_digest,
        }
        generated_id = "episode-" + hashlib.sha256(canonical_json(identity).encode()).hexdigest()
        episode_id = self.episode_id or generated_id
        _text(episode_id, name="episode_id")
        object.__setattr__(self, "campaign_digest", campaign_digest)
        object.__setattr__(self, "source_digest", source_digest)
        object.__setattr__(self, "execution_id", execution_id)
        object.__setattr__(self, "config_digest", config_digest)
        object.__setattr__(self, "checkpoint_digest", checkpoint_digest)
        object.__setattr__(self, "environment_digest", environment_digest)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(self, "campaign_id", campaign_digest)
        object.__setattr__(self, "source_id", source_digest)

    @property
    def identity_key(self) -> str:
        """Return a canonical key suitable for joins and deduplication."""

        payload = {
            "campaign_digest": self.campaign_digest,
            "source_digest": self.source_digest,
            "execution_id": self.execution_id,
            "planner_id": self.planner_id,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "attempt": self.attempt,
            "config_digest": self.config_digest,
            "checkpoint_digest": self.checkpoint_digest,
            "environment_digest": self.environment_digest,
        }
        return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class CampaignAudit:
    """Campaign-level audit metadata and source/protocol identity."""

    audit_id: str
    campaign_digest: str
    source_digest: str
    title: str = ""
    schema_version: str = AUDIT_RECORD_SCHEMA_VERSION
    protocol_version: str = ""
    protocol_digest: str = ""
    source: SourceRef | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate campaign-level identity and source metadata."""

        _text(self.audit_id, name="audit_id")
        _digest(self.campaign_digest, name="campaign_digest", required=True)
        _digest(self.source_digest, name="source_digest", required=True)
        _check_record_version(self.schema_version)
        _coerce_source(self.source)


@dataclass(frozen=True, slots=True)
class Signal:
    """Detector output with explicit unavailable/error states."""

    signal_id: str
    detector_id: str
    detector_version: str = ""
    status: str = "flagged"
    reason_code: str = ""
    episode_id: str = ""
    evidence: tuple[Mapping[str, Any], ...] = ()
    measured: Mapping[str, Any] = field(default_factory=dict)
    interval: TimeInterval | None = None
    threshold: Mapping[str, Any] | None = None
    missingness: tuple[str, ...] = ()
    message: str = ""

    def __post_init__(self) -> None:
        """Validate detector status and missingness semantics."""

        _text(self.signal_id, name="signal_id")
        _text(self.detector_id, name="detector_id")
        if self.status not in SIGNAL_STATUSES:
            raise AuditContractError(f"signal status must be one of {SIGNAL_STATUSES}")
        if self.status in {"unavailable", "error"} and not self.message and not self.missingness:
            raise AuditContractError("unavailable/error signals require message or missingness")
        if self.interval is not None and not isinstance(self.interval, TimeInterval):
            object.__setattr__(self, "interval", _interval(self.interval))


@dataclass(frozen=True, slots=True)
class Reference:
    """A source-bound spatial/metric/event reference.

    ``image`` points are always allowed.  ``world`` points from video require
    an explicit calibration marker; source traces that already declare world
    coordinates remain valid without an invented transform.
    """

    reference_id: str
    coordinate_frame: str
    point: tuple[float, float]
    source: SourceRef | None = None
    timestamp_s: float | None = None
    source_revision: str = ""
    calibration: Mapping[str, Any] | None = None
    actor_id: str = ""
    object_id: str = ""
    goal_id: str = ""
    waypoint_id: str = ""
    metric_id: str = ""
    event_id: str = ""

    def __post_init__(self) -> None:
        """Validate coordinates and reject uncalibrated video-world rebinding."""

        _text(self.reference_id, name="reference_id")
        if self.coordinate_frame not in COORDINATE_FRAMES:
            raise AuditContractError(f"coordinate_frame must be one of {COORDINATE_FRAMES}")
        if len(self.point) != 2:
            raise AuditContractError("reference point must contain exactly two coordinates")
        point = (
            _finite(self.point[0], name="reference.point[0]"),
            _finite(self.point[1], name="reference.point[1]"),
        )
        source = _coerce_source(self.source)
        if self.timestamp_s is not None:
            _finite(self.timestamp_s, name="reference.timestamp_s")
        if self.coordinate_frame == "world" and source is not None:
            video_format = source.format.lower()
            if video_format in {"video", "mp4", "webm", "image", "png", "jpg", "jpeg"}:
                if not self.calibration:
                    raise AuditContractError(
                        "world coordinates on uncalibrated video are not allowed; provide calibration"
                    )
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "source", source)


@dataclass(frozen=True, slots=True)
class Annotation:
    """Human/detector/agent annotation with explicit review scope."""

    annotation_id: str
    episode_id: str
    classification: str
    mode: str = "quick"
    author_kind: str = "human"
    author_id: str = ""
    interval: TimeInterval | None = None
    observed_behavior: str = ""
    suspected_cause: str = ""
    evidence: tuple[Mapping[str, Any], ...] = ()
    confidence: float | None = None
    references: tuple[Reference, ...] = ()
    tags: tuple[str, ...] = ()
    review_scope: str = "interval"
    source_revision: int = 0
    source_identity: str = ""
    created_at: str = field(default_factory=utc_now)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source_ref: SourceRef | None = None

    def __post_init__(self) -> None:
        """Validate annotation mode, author, scope, and optional evidence."""

        _text(self.annotation_id, name="annotation_id")
        _text(self.episode_id, name="episode_id")
        if self.classification not in ANNOTATION_CLASSIFICATIONS:
            raise AuditContractError(
                f"annotation classification must be one of {ANNOTATION_CLASSIFICATIONS}"
            )
        if self.mode not in ANNOTATION_MODES:
            raise AuditContractError(f"annotation mode must be one of {ANNOTATION_MODES}")
        if self.author_kind not in AUTHOR_KINDS:
            raise AuditContractError(f"author_kind must be one of {AUTHOR_KINDS}")
        if self.review_scope not in REVIEW_SCOPES:
            raise AuditContractError(f"review_scope must be one of {REVIEW_SCOPES}")
        if not isinstance(self.source_revision, int) or self.source_revision < 0:
            raise AuditContractError("source_revision must be a non-negative integer")
        if self.confidence is not None:
            confidence = _finite(self.confidence, name="confidence")
            if not 0.0 <= confidence <= 1.0:
                raise AuditContractError("confidence must be between 0 and 1")
            object.__setattr__(self, "confidence", confidence)
        if self.interval is not None and not isinstance(self.interval, TimeInterval):
            object.__setattr__(self, "interval", _interval(self.interval))
        references = tuple(
            reference if isinstance(reference, Reference) else reference_from_dict(reference)
            for reference in self.references
        )
        object.__setattr__(self, "references", references)
        object.__setattr__(self, "source_ref", _coerce_source(self.source_ref))
        if self.mode == "one_click" and self.evidence:
            raise AuditContractError("one-click annotations cannot carry detailed evidence")

    @property
    def is_full_episode_review(self) -> bool:
        """Return whether this annotation explicitly reviews the full episode."""

        return self.review_scope == "full_episode"

    @property
    def category(self) -> str:
        """Return the taxonomy label under the trace-annotation vocabulary."""

        return self.classification


@dataclass(frozen=True, slots=True)
class ReviewPacket:
    """A selected review unit and its compatibility/missingness context."""

    packet_id: str
    primary: EpisodeRef
    peers: tuple[EpisodeRef, ...] = ()
    signals: tuple[Signal, ...] = ()
    selection_reasons: tuple[str, ...] = ()
    missingness: tuple[str, ...] = ()
    policy_version: str = ""
    input_revision: int = 0

    def __post_init__(self) -> None:
        """Validate the selected primary episode and peer context."""

        _text(self.packet_id, name="packet_id")
        primary = (
            self.primary
            if isinstance(self.primary, EpisodeRef)
            else episode_ref_from_dict(self.primary)
        )
        peers = tuple(
            peer if isinstance(peer, EpisodeRef) else episode_ref_from_dict(peer)
            for peer in self.peers
        )
        signals = tuple(
            signal if isinstance(signal, Signal) else signal_from_dict(signal)
            for signal in self.signals
        )
        object.__setattr__(self, "primary", primary)
        object.__setattr__(self, "peers", peers)
        object.__setattr__(self, "signals", signals)
        if not isinstance(self.input_revision, int) or self.input_revision < 0:
            raise AuditContractError("input_revision must be a non-negative integer")

    @property
    def primary_episode(self) -> EpisodeRef:
        """Return the selected primary episode."""

        return self.primary

    @property
    def compatible_peers(self) -> tuple[EpisodeRef, ...]:
        """Return peer candidates retained with the packet."""

        return self.peers


@dataclass(frozen=True, slots=True)
class Finding:
    """A durable symptom/defect grouping with separate membership evidence."""

    finding_id: str
    title: str
    status: str = "proposed"
    candidate_members: tuple[str, ...] = ()
    confirmed_members: tuple[str, ...] = ()
    negative_controls: tuple[str, ...] = ()
    observations: tuple[str, ...] = ()
    hypotheses: tuple[str, ...] = ()
    evidence: tuple[Mapping[str, Any], ...] = ()
    negative_evidence: tuple[Mapping[str, Any], ...] = ()
    tags: tuple[str, ...] = ()
    diagnostic_results: tuple[Mapping[str, Any], ...] = ()
    github_issue: Mapping[str, Any] | None = None
    source_revision: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Validate lifecycle state and independent membership lists."""

        _text(self.finding_id, name="finding_id")
        _text(self.title, name="title")
        if self.status not in FINDING_STATUSES:
            raise AuditContractError(f"finding status must be one of {FINDING_STATUSES}")
        for name, members in (
            ("candidate_members", self.candidate_members),
            ("confirmed_members", self.confirmed_members),
            ("negative_controls", self.negative_controls),
        ):
            if any(not isinstance(member, str) or not member for member in members):
                raise AuditContractError(f"{name} must contain non-empty member IDs")

    @property
    def candidate_count(self) -> int:
        """Return the number of current candidate manifestations."""

        return len(self.candidate_members)

    @property
    def confirmed_count(self) -> int:
        """Return the number of explicitly confirmed manifestations."""

        return len(self.confirmed_members)

    @property
    def negative_control_count(self) -> int:
        """Return the number of recorded negative controls."""

        return len(self.negative_controls)

    @property
    def candidates(self) -> tuple[str, ...]:
        """Return candidate manifestation IDs."""

        return self.candidate_members

    @property
    def confirmations(self) -> tuple[str, ...]:
        """Return confirmed manifestation IDs."""

        return self.confirmed_members


@dataclass(frozen=True, slots=True)
class ActionRecord:
    """An auditable user/agent action, including failed or undone actions."""

    action_id: str
    action_type: str
    actor_kind: str = "human"
    actor_id: str = ""
    target_id: str = ""
    status: str = "committed"
    details: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        """Validate action identity and author kind."""

        _text(self.action_id, name="action_id")
        _text(self.action_type, name="action_type")
        if self.actor_kind not in AUTHOR_KINDS:
            raise AuditContractError(f"actor_kind must be one of {AUTHOR_KINDS}")


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    """Explicit review receipt; opening an interval is not full review credit."""

    review_id: str
    episode_id: str
    scope: str
    outcome: str = ""
    author_kind: str = "human"
    author_id: str = ""
    source_revision: int = 0
    annotation_ids: tuple[str, ...] = ()
    created_at: str = field(default_factory=utc_now)
    notes: str = ""

    def __post_init__(self) -> None:
        """Validate explicit review scope and author identity."""

        _text(self.review_id, name="review_id")
        _text(self.episode_id, name="episode_id")
        if self.scope not in REVIEW_SCOPES:
            raise AuditContractError(f"review scope must be one of {REVIEW_SCOPES}")
        if self.author_kind not in AUTHOR_KINDS:
            raise AuditContractError(f"author_kind must be one of {AUTHOR_KINDS}")
        if not isinstance(self.source_revision, int) or self.source_revision < 0:
            raise AuditContractError("source_revision must be a non-negative integer")


_RECORD_TYPES: dict[str, type[Any]] = {
    "campaign_audit": CampaignAudit,
    "episode_ref": EpisodeRef,
    "review_packet": ReviewPacket,
    "signal": Signal,
    "reference": Reference,
    "annotation": Annotation,
    "finding": Finding,
    "action_record": ActionRecord,
    "review_record": ReviewRecord,
}


def _check_record_version(version: str) -> None:
    if version not in SUPPORTED_AUDIT_SCHEMA_VERSIONS:
        raise UnsupportedAuditSchemaError(f"unsupported audit schema version: {version}")


def record_type(record: Any) -> str:
    """Return the stable record type name for a contract instance."""

    for name, cls in _RECORD_TYPES.items():
        if isinstance(record, cls):
            return name
    raise AuditContractError(f"unsupported audit record type: {type(record).__name__}")


def record_id(record: Any) -> str:
    """Return the stable primary key used by the projection."""

    kind = record_type(record)
    field_name = {
        "campaign_audit": "audit_id",
        "episode_ref": "episode_id",
        "review_packet": "packet_id",
        "signal": "signal_id",
        "reference": "reference_id",
        "annotation": "annotation_id",
        "finding": "finding_id",
        "action_record": "action_id",
        "review_record": "review_id",
    }[kind]
    if hasattr(record, field_name):
        return str(getattr(record, field_name))
    raise AuditContractError(f"record type {kind} has no stable ID")


def record_to_dict(record: Any, *, include_type: bool = True) -> dict[str, Any]:
    """Return one validated record as a JSON-compatible mapping."""

    kind = record_type(record)
    payload = _jsonable(record)
    if not isinstance(payload, dict):  # pragma: no cover - dataclasses always map
        raise AuditContractError("record payload must be a mapping")
    payload["schema_version"] = AUDIT_RECORD_SCHEMA_VERSION
    if include_type:
        payload["record_type"] = kind
    payload["record_id"] = record_id(record)
    return payload


def _tuple_of_mappings(value: Any, *, name: str) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise AuditContractError(f"{name} must be a sequence")
    return tuple(
        dict(item) if isinstance(item, Mapping) else _mapping_error(name) for item in value
    )


def _mapping_error(name: str) -> Mapping[str, Any]:
    raise AuditContractError(f"{name} must contain mappings")


def _tuple_of_strings(value: Any, *, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise AuditContractError(f"{name} must be a sequence")
    values = tuple(str(item) for item in value)
    if any(not item for item in values):
        raise AuditContractError(f"{name} must contain non-empty strings")
    return values


def _interval(value: Any) -> TimeInterval | None:
    if value is None:
        return None
    if isinstance(value, TimeInterval):
        return value
    if not isinstance(value, Mapping):
        raise AuditContractError("interval must be a mapping")
    return TimeInterval(float(value["start_s"]), float(value.get("end_s", value["start_s"])))


def _source_from_payload(value: Any) -> SourceRef | None:
    return _coerce_source(value)


def episode_ref_from_dict(payload: Mapping[str, Any]) -> EpisodeRef:
    """Build an :class:`EpisodeRef` from a versioned mapping.

    Returns:
        Validated episode identity.
    """

    return EpisodeRef(
        campaign_digest=str(payload.get("campaign_digest", payload.get("campaign_id", ""))),
        source_digest=str(payload.get("source_digest", payload.get("source_id", ""))),
        execution_id=str(payload.get("execution_id", "")),
        planner_id=str(payload.get("planner_id", "")),
        scenario_id=str(payload.get("scenario_id", "")),
        seed=payload.get("seed"),
        attempt=int(payload.get("attempt", 0)),
        config_digest=str(payload.get("config_digest", "")),
        checkpoint_digest=str(payload.get("checkpoint_digest", "")),
        environment_digest=str(payload.get("environment_digest", "")),
        source=_source_from_payload(payload.get("source")),
        episode_id=str(payload.get("episode_id", "")),
    )


def reference_from_dict(payload: Mapping[str, Any]) -> Reference:
    return Reference(
        reference_id=str(payload["reference_id"]),
        coordinate_frame=str(payload["coordinate_frame"]),
        point=tuple(payload["point"]),
        source=_source_from_payload(payload.get("source")),
        timestamp_s=payload.get("timestamp_s"),
        source_revision=str(payload.get("source_revision", "")),
        calibration=payload.get("calibration"),
        actor_id=str(payload.get("actor_id", "")),
        object_id=str(payload.get("object_id", "")),
        goal_id=str(payload.get("goal_id", "")),
        waypoint_id=str(payload.get("waypoint_id", "")),
        metric_id=str(payload.get("metric_id", "")),
        event_id=str(payload.get("event_id", "")),
    )


def signal_from_dict(payload: Mapping[str, Any]) -> Signal:
    """Build a detector signal from a canonical mapping.

    Returns:
        Validated signal record.
    """

    return Signal(
        signal_id=str(payload["signal_id"]),
        detector_id=str(payload["detector_id"]),
        detector_version=str(payload.get("detector_version", "")),
        status=str(payload.get("status", "flagged")),
        reason_code=str(payload.get("reason_code", "")),
        episode_id=str(payload.get("episode_id", "")),
        evidence=_tuple_of_mappings(payload.get("evidence", ()), name="evidence"),
        measured=dict(payload.get("measured", {})),
        interval=_interval(payload.get("interval")),
        threshold=(dict(payload["threshold"]) if payload.get("threshold") is not None else None),
        missingness=_tuple_of_strings(payload.get("missingness", ()), name="missingness"),
        message=str(payload.get("message", "")),
    )


def annotation_from_dict(payload: Mapping[str, Any]) -> Annotation:
    """Build an annotation from a canonical mapping.

    Returns:
        Validated annotation record.
    """

    return Annotation(
        annotation_id=str(payload["annotation_id"]),
        episode_id=str(payload["episode_id"]),
        classification=str(payload["classification"]),
        mode=str(payload.get("mode", "quick")),
        author_kind=str(payload.get("author_kind", "human")),
        author_id=str(payload.get("author_id", "")),
        interval=_interval(payload.get("interval")),
        observed_behavior=str(payload.get("observed_behavior", "")),
        suspected_cause=str(payload.get("suspected_cause", "")),
        evidence=_tuple_of_mappings(payload.get("evidence", ()), name="evidence"),
        confidence=payload.get("confidence"),
        references=tuple(reference_from_dict(item) for item in payload.get("references", ())),
        tags=_tuple_of_strings(payload.get("tags", ()), name="tags"),
        review_scope=str(payload.get("review_scope", "interval")),
        source_revision=int(payload.get("source_revision", 0)),
        source_identity=str(payload.get("source_identity", "")),
        created_at=str(payload.get("created_at", utc_now())),
        metadata=dict(payload.get("metadata", {})),
        source_ref=_source_from_payload(payload.get("source_ref")),
    )


def finding_from_dict(payload: Mapping[str, Any]) -> Finding:
    """Build a finding from a canonical mapping.

    Returns:
        Validated finding record.
    """

    return Finding(
        finding_id=str(payload["finding_id"]),
        title=str(payload["title"]),
        status=str(payload.get("status", "proposed")),
        candidate_members=_tuple_of_strings(
            payload.get("candidate_members", ()), name="candidate_members"
        ),
        confirmed_members=_tuple_of_strings(
            payload.get("confirmed_members", ()), name="confirmed_members"
        ),
        negative_controls=_tuple_of_strings(
            payload.get("negative_controls", ()), name="negative_controls"
        ),
        observations=_tuple_of_strings(payload.get("observations", ()), name="observations"),
        hypotheses=_tuple_of_strings(payload.get("hypotheses", ()), name="hypotheses"),
        evidence=_tuple_of_mappings(payload.get("evidence", ()), name="evidence"),
        negative_evidence=_tuple_of_mappings(
            payload.get("negative_evidence", ()), name="negative_evidence"
        ),
        tags=_tuple_of_strings(payload.get("tags", ()), name="tags"),
        diagnostic_results=_tuple_of_mappings(
            payload.get("diagnostic_results", ()), name="diagnostic_results"
        ),
        github_issue=(
            dict(payload["github_issue"]) if payload.get("github_issue") is not None else None
        ),
        source_revision=str(payload.get("source_revision", "")),
        created_at=str(payload.get("created_at", utc_now())),
        updated_at=str(payload.get("updated_at", utc_now())),
    )


def review_packet_from_dict(payload: Mapping[str, Any]) -> ReviewPacket:
    """Build a review packet from a canonical mapping.

    Returns:
        Validated review packet.
    """

    return ReviewPacket(
        packet_id=str(payload["packet_id"]),
        primary=episode_ref_from_dict(payload["primary"]),
        peers=tuple(episode_ref_from_dict(item) for item in payload.get("peers", ())),
        signals=tuple(signal_from_dict(item) for item in payload.get("signals", ())),
        selection_reasons=_tuple_of_strings(
            payload.get("selection_reasons", ()), name="selection_reasons"
        ),
        missingness=_tuple_of_strings(payload.get("missingness", ()), name="missingness"),
        policy_version=str(payload.get("policy_version", "")),
        input_revision=int(payload.get("input_revision", 0)),
    )


def record_from_dict(payload: Mapping[str, Any]) -> Any:  # noqa: C901
    """Validate and deserialize a canonical record mapping.

    Returns:
        The typed audit-contract instance represented by ``payload``.
    """

    if not isinstance(payload, Mapping):
        raise AuditContractError("record payload must be a mapping")
    version = str(payload.get("schema_version", AUDIT_RECORD_SCHEMA_VERSION))
    _check_record_version(version)
    kind = payload.get("record_type")
    if not isinstance(kind, str) or kind not in _RECORD_TYPES:
        raise AuditContractError(f"unknown audit record_type: {kind!r}")
    if kind == "episode_ref":
        record = episode_ref_from_dict(payload)
    elif kind == "signal":
        record = signal_from_dict(payload)
    elif kind == "reference":
        record = reference_from_dict(payload)
    elif kind == "annotation":
        record = annotation_from_dict(payload)
    elif kind == "finding":
        record = finding_from_dict(payload)
    elif kind == "review_packet":
        record = review_packet_from_dict(payload)
    else:
        cls = _RECORD_TYPES[kind]
        names = {item.name for item in fields(cls)}
        values = {name: payload[name] for name in names if name in payload}
        if kind == "campaign_audit":
            values["source"] = _source_from_payload(values.get("source"))
        if kind in {"action_record", "review_record"}:
            if kind == "action_record":
                values["details"] = dict(values.get("details", {}))
            else:
                values["annotation_ids"] = _tuple_of_strings(
                    values.get("annotation_ids", ()), name="annotation_ids"
                )
        record = cls(**values)
    declared_id = payload.get("record_id")
    if declared_id is not None and declared_id != record_id(record):
        raise AuditContractError(
            f"record_id {declared_id!r} does not match {record_type(record)} identity"
        )
    return record


def serialize_record(record: Any) -> str:
    """Serialize one record as canonical JSON.

    Returns:
        Deterministic JSON text.
    """

    return canonical_json(record_to_dict(record))


serialize_audit_record = serialize_record


def deserialize_record(value: str | bytes | Mapping[str, Any]) -> Any:
    """Deserialize one canonical JSON record.

    Returns:
        The validated typed record.
    """

    payload = json.loads(value) if isinstance(value, (str, bytes)) else value
    return record_from_dict(payload)


deserialize_audit_record = deserialize_record


def write_ndjson(records: Sequence[Any], path: str | Path) -> Path:
    """Write records as deterministic newline-delimited JSON.

    Returns:
        The path written.
    """

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(serialize_record(record) + "\n" for record in records)
    output.write_text(data, encoding="utf-8")
    return output


def read_ndjson(path: str | Path) -> list[Any]:
    """Read and validate every complete canonical record in an NDJSON file.

    Returns:
        Typed records in file order.
    """

    records: list[Any] = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            records.append(deserialize_record(line))
        except (AuditContractError, json.JSONDecodeError) as exc:
            raise AuditContractError(f"invalid NDJSON record at line {line_number}: {exc}") from exc
    return records


def migrate_record(
    payload: Mapping[str, Any], *, target_version: str = AUDIT_RECORD_SCHEMA_VERSION
) -> dict[str, Any]:
    """Migrate one record to a supported version without dropping provenance.

    Returns:
        A canonical mapping using ``target_version``.
    """

    _check_record_version(target_version)
    record = record_from_dict(payload)
    migrated = record_to_dict(record)
    migrated["schema_version"] = target_version
    return migrated


__all__ = [
    "ANNOTATION_CLASSIFICATIONS",
    "ANNOTATION_MODES",
    "AUDIT_EXPORT_SCHEMA_VERSION",
    "AUDIT_JOURNAL_SCHEMA_VERSION",
    "AUDIT_RECORD_SCHEMA_FILE",
    "AUDIT_RECORD_SCHEMA_VERSION",
    "AUDIT_SCHEMA_VERSION",
    "AUTHOR_KINDS",
    "COORDINATE_FRAMES",
    "FINDING_STATUSES",
    "REVIEW_SCOPES",
    "SIGNAL_STATUSES",
    "ActionRecord",
    "Annotation",
    "AuditContractError",
    "AuditIdentityError",
    "CampaignAudit",
    "EpisodeRef",
    "Finding",
    "Reference",
    "ReviewPacket",
    "ReviewRecord",
    "Signal",
    "TimeInterval",
    "UnsupportedAuditSchemaError",
    "annotation_from_dict",
    "canonical_json",
    "deserialize_audit_record",
    "deserialize_record",
    "episode_ref_from_dict",
    "finding_from_dict",
    "load_audit_record_schema",
    "migrate_record",
    "read_ndjson",
    "record_from_dict",
    "record_id",
    "record_to_dict",
    "record_type",
    "review_packet_from_dict",
    "serialize_audit_record",
    "serialize_record",
    "signal_from_dict",
    "utc_now",
    "validate_record",
    "write_ndjson",
]
