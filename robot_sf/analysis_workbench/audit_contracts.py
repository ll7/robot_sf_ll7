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
SOURCE_PROVENANCE_STATUSES = ("verified", "stale", "mutated", "unavailable")
SOURCE_PROVENANCE_VERIFIED = "verified"
SOURCE_PROVENANCE_STALE = "stale"
SOURCE_PROVENANCE_MUTATED = "mutated"
SOURCE_PROVENANCE_UNAVAILABLE = "unavailable"

_DIGEST_LENGTHS = {40, 64}
_IMAGE_DIMENSION_LIMIT = 100_000_000
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


def _dimension(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AuditContractError(f"{name} must be a positive integer")
    if value < 1 or value > _IMAGE_DIMENSION_LIMIT:
        raise AuditContractError(f"{name} must be between 1 and {_IMAGE_DIMENSION_LIMIT}")
    return value


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


def _source_sha256(source: SourceRef | None) -> str:
    """Return a validated source SHA-256, or empty when it is unavailable."""

    if source is None or not source.sha256:
        return ""
    value = source.sha256
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in value)
    ):
        raise AuditContractError("source.sha256 must be a 64-character hexadecimal digest")
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


def _strict_json_values(value: Any, *, path: str = "$") -> None:
    """Reject non-finite values before schema validation or typed construction."""

    if isinstance(value, float) and not math.isfinite(value):
        raise AuditContractError(f"{path} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise AuditContractError(f"{path} contains a non-string field name")
            _strict_json_values(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _strict_json_values(item, path=f"{path}[{index}]")


def _reject_json_constant(token: str) -> Any:
    raise AuditContractError(f"invalid audit JSON constant: {token}")


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

    record_from_dict(payload)


def _validate_record_payload(payload: Mapping[str, Any]) -> None:
    """Validate the envelope and record-type schema before construction."""

    if not isinstance(payload, Mapping):
        raise AuditContractError("record payload must be a mapping")
    _strict_json_values(payload)
    if "schema_version" not in payload:
        raise AuditContractError("record schema_version is required")
    version = payload["schema_version"]
    if not isinstance(version, str):
        raise AuditContractError("record schema_version must be a string")
    _check_record_version(version)
    if "record_type" not in payload:
        raise AuditContractError("record record_type is required")
    if "record_id" not in payload:
        raise AuditContractError("record record_id is required")
    errors = sorted(
        Draft202012Validator(load_audit_record_schema()).iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    if errors:
        details = "; ".join(
            f"/{'/'.join(map(str, error.absolute_path))}: {error.message}" for error in errors
        )
        raise AuditContractError(details)


def _coerce_source(value: SourceRef | Mapping[str, Any] | None) -> SourceRef | None:
    if value is None:
        return None
    if isinstance(value, SourceRef):
        _source_sha256(value)
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
    source = SourceRef(**{name: value.get(name, "") for name in known})
    _source_sha256(source)
    return source


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
        if self.episode_id and self.episode_id != generated_id:
            raise AuditIdentityError(
                "episode_id must equal the collision-resistant identity derived from its fields"
            )
        episode_id = generated_id
        source_sha256 = _source_sha256(source)
        if source_sha256 and source_sha256 != source_digest:
            raise AuditIdentityError("source_digest does not match source.sha256")
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

    @property
    def source_provenance_status(self) -> str:
        """Return whether the source pointer is hash-bound or unavailable."""

        return (
            SOURCE_PROVENANCE_VERIFIED
            if _source_sha256(self.source)
            else SOURCE_PROVENANCE_UNAVAILABLE
        )


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
        campaign_digest = _digest(self.campaign_digest, name="campaign_digest", required=True)
        source_digest = _digest(self.source_digest, name="source_digest", required=True)
        _check_record_version(self.schema_version)
        source = _coerce_source(self.source)
        source_sha256 = _source_sha256(source)
        if source_sha256 and source_sha256 != source_digest:
            raise AuditIdentityError("source_digest does not match source.sha256")
        object.__setattr__(self, "campaign_digest", campaign_digest)
        object.__setattr__(self, "source_digest", source_digest)
        object.__setattr__(self, "source", source)


@dataclass(frozen=True, slots=True)
class ImageDisplayTransform:
    """Closed transform between source-image and displayed crop coordinates.

    Coordinates are kept as exact floating-point affine values; no pixel
    rounding is applied.  This makes a source/display/source round trip
    reversible while the dimensions and crop are still bounded and typed.
    """

    source_width: int
    source_height: int
    display_width: int
    display_height: int
    crop_x: float = 0.0
    crop_y: float = 0.0
    crop_width: float | None = None
    crop_height: float | None = None

    def __post_init__(self) -> None:
        """Validate dimensions and the crop rectangle."""

        source_width = _dimension(self.source_width, name="source_width")
        source_height = _dimension(self.source_height, name="source_height")
        display_width = _dimension(self.display_width, name="display_width")
        display_height = _dimension(self.display_height, name="display_height")
        crop_x = _finite(self.crop_x, name="crop_x")
        crop_y = _finite(self.crop_y, name="crop_y")
        crop_width = (
            float(source_width)
            if self.crop_width is None
            else _finite(self.crop_width, name="crop_width")
        )
        crop_height = (
            float(source_height)
            if self.crop_height is None
            else _finite(self.crop_height, name="crop_height")
        )
        if crop_x < 0 or crop_y < 0 or crop_width <= 0 or crop_height <= 0:
            raise AuditContractError(
                "crop rectangle must have finite non-negative origin and positive size"
            )
        if crop_x + crop_width > source_width or crop_y + crop_height > source_height:
            raise AuditContractError("crop rectangle must be contained in the source image")
        object.__setattr__(self, "source_width", source_width)
        object.__setattr__(self, "source_height", source_height)
        object.__setattr__(self, "display_width", display_width)
        object.__setattr__(self, "display_height", display_height)
        object.__setattr__(self, "crop_x", crop_x)
        object.__setattr__(self, "crop_y", crop_y)
        object.__setattr__(self, "crop_width", crop_width)
        object.__setattr__(self, "crop_height", crop_height)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> ImageDisplayTransform:
        """Build a transform while rejecting opaque or unknown fields.

        Returns:
            A validated closed transform.
        """

        if not isinstance(value, Mapping):
            raise AuditContractError("image calibration must be a transform mapping")
        allowed = {
            "source_width",
            "source_height",
            "display_width",
            "display_height",
            "crop_x",
            "crop_y",
            "crop_width",
            "crop_height",
        }
        unknown = set(value) - allowed
        if unknown:
            raise AuditContractError(
                f"image calibration transform contains unknown fields: {', '.join(sorted(unknown))}"
            )
        required = {"source_width", "source_height", "display_width", "display_height"}
        missing = required - set(value)
        if missing:
            raise AuditContractError(
                f"image calibration transform is missing fields: {', '.join(sorted(missing))}"
            )
        try:
            return cls(**{name: value[name] for name in allowed if name in value})
        except (TypeError, ValueError, KeyError) as exc:
            raise AuditContractError(f"invalid image calibration transform: {exc}") from exc

    def to_dict(self) -> dict[str, Any]:
        """Return the closed transform mapping used in canonical records."""

        return {
            "source_width": self.source_width,
            "source_height": self.source_height,
            "display_width": self.display_width,
            "display_height": self.display_height,
            "crop_x": self.crop_x,
            "crop_y": self.crop_y,
            "crop_width": self.crop_width,
            "crop_height": self.crop_height,
        }

    @staticmethod
    def _point(point: Sequence[Any], *, name: str) -> tuple[float, float]:
        if isinstance(point, (str, bytes)) or len(point) != 2:
            raise AuditContractError(f"{name} must contain exactly two coordinates")
        return (_finite(point[0], name=f"{name}[0]"), _finite(point[1], name=f"{name}[1]"))

    def source_to_display(self, point: Sequence[Any]) -> tuple[float, float]:
        """Map a source-image point into the displayed crop.

        Returns:
            The displayed coordinates.
        """

        x, y = self._point(point, name="source point")
        return (
            (x - self.crop_x) * self.display_width / self.crop_width,
            (y - self.crop_y) * self.display_height / self.crop_height,
        )

    def display_to_source(self, point: Sequence[Any]) -> tuple[float, float]:
        """Map a displayed crop point back to source-image coordinates.

        Returns:
            The source-image coordinates.
        """

        x, y = self._point(point, name="display point")
        return (
            x * self.crop_width / self.display_width + self.crop_x,
            y * self.crop_height / self.display_height + self.crop_y,
        )


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


def _world_calibration(value: Mapping[str, Any] | ImageDisplayTransform | None) -> dict[str, Any]:
    """Validate the explicit mapping shape required for media-world coordinates.

    Returns:
        A shallow copy of the validated calibration mapping.
    """

    if not isinstance(value, Mapping):
        raise AuditContractError(
            "media-world calibration must be a mapping with kind, version, and parameters"
        )
    allowed = {"kind", "version", "parameters"}
    unknown = set(value) - allowed
    if unknown:
        raise AuditContractError(
            f"media-world calibration contains unknown fields: {', '.join(sorted(unknown))}"
        )
    if not isinstance(value.get("kind"), str) or not value["kind"].strip():
        raise AuditContractError("media-world calibration kind is required")
    version = value.get("version", 1)
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise AuditContractError("media-world calibration version must be a positive integer")
    parameters = value.get("parameters")
    if not isinstance(parameters, Mapping) or not parameters:
        raise AuditContractError("media-world calibration parameters must be a non-empty mapping")
    _strict_json_values(parameters)
    return dict(value)


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
    calibration: Mapping[str, Any] | ImageDisplayTransform | None = None
    actor_id: str = ""
    object_id: str = ""
    goal_id: str = ""
    waypoint_id: str = ""
    metric_id: str = ""
    event_id: str = ""
    source_point: tuple[float, float] | None = None
    seek_identity: str = ""

    def __post_init__(self) -> None:  # noqa: C901
        """Validate coordinates and reject uncalibrated video-world rebinding."""

        _text(self.reference_id, name="reference_id")
        if self.coordinate_frame not in COORDINATE_FRAMES:
            raise AuditContractError(f"coordinate_frame must be one of {COORDINATE_FRAMES}")
        if isinstance(self.point, (str, bytes)) or len(self.point) != 2:
            raise AuditContractError("reference point must contain exactly two coordinates")
        point = (
            _finite(self.point[0], name="reference.point[0]"),
            _finite(self.point[1], name="reference.point[1]"),
        )
        source = _coerce_source(self.source)
        timestamp = (
            None
            if self.timestamp_s is None
            else _finite(self.timestamp_s, name="reference.timestamp_s")
        )
        source_revision = _optional_text(self.source_revision, name="reference.source_revision")
        if source is not None and source.source_commit and source_revision:
            if source_revision != source.source_commit:
                raise AuditIdentityError("reference source_revision does not match source_ref")
        seek_identity = _optional_text(self.seek_identity, name="reference.seek_identity")
        source_point = self.source_point
        if source_point is None:
            source_point = point if self.coordinate_frame == "image" else None
        elif isinstance(source_point, (str, bytes)) or len(source_point) != 2:
            raise AuditContractError("reference source_point must contain exactly two coordinates")
        if source_point is not None:
            source_point = (
                _finite(source_point[0], name="reference.source_point[0]"),
                _finite(source_point[1], name="reference.source_point[1]"),
            )
        calibration = self.calibration
        if calibration is not None and not isinstance(
            calibration, (Mapping, ImageDisplayTransform)
        ):
            raise AuditContractError("reference calibration must be a typed transform mapping")
        if self.coordinate_frame == "image" and calibration is not None:
            calibration = (
                calibration
                if isinstance(calibration, ImageDisplayTransform)
                else ImageDisplayTransform.from_mapping(calibration)
            )
        if self.coordinate_frame == "world" and source is not None:
            media_format = source.format.lower().strip()
            is_media = (
                media_format.startswith("video/")
                or media_format.startswith("image/")
                or media_format.startswith("video-")
                or media_format
                in {
                    "video",
                    "mp4",
                    "webm",
                    "image",
                    "png",
                    "jpg",
                    "jpeg",
                    "video-mp4.v1",
                    "image/jpeg",
                }
            )
            if is_media:
                if not self.calibration:
                    raise AuditContractError(
                        "world coordinates on uncalibrated video are not allowed; provide calibration"
                    )
                calibration = _world_calibration(calibration)
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "timestamp_s", timestamp)
        object.__setattr__(self, "source_revision", source_revision)
        object.__setattr__(self, "source_point", source_point)
        object.__setattr__(self, "seek_identity", seek_identity)
        object.__setattr__(self, "calibration", calibration)

    @property
    def source_coordinates(self) -> tuple[float, float] | None:
        """Return source-image coordinates without replacing them by display coordinates."""

        return self.source_point


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
    source_revision: int | str = 0
    source_identity: str = ""
    created_at: str = field(default_factory=utc_now)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    source_ref: SourceRef | None = None
    provenance_status: str = SOURCE_PROVENANCE_UNAVAILABLE
    provenance_reason: str = ""

    def __post_init__(self) -> None:  # noqa: C901, PLR0912, PLR0915
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
        if isinstance(self.source_revision, bool) or not isinstance(
            self.source_revision, (int, str)
        ):
            raise AuditContractError("source_revision must be a non-negative integer or token")
        if isinstance(self.source_revision, int) and self.source_revision < 0:
            raise AuditContractError("source_revision must be a non-negative integer")
        revision_token = (
            "" if self.source_revision in (0, "") else str(self.source_revision).strip()
        )
        if not revision_token and self.source_revision not in (0, ""):
            raise AuditContractError("source_revision token must be non-empty")
        if self.provenance_status not in SOURCE_PROVENANCE_STATUSES:
            raise AuditContractError(
                f"provenance_status must be one of {SOURCE_PROVENANCE_STATUSES}"
            )
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
        source = _coerce_source(self.source_ref)
        source_sha256 = _source_sha256(source)
        source_identity = (
            self.source_identity.strip() if isinstance(self.source_identity, str) else ""
        )
        if not isinstance(self.source_identity, str):
            raise AuditContractError("source_identity must be a string")
        status = self.provenance_status
        identity_matches_hash = bool(source_sha256 and source_identity.lower() == source_sha256)
        identity_matches_artifact = bool(
            source is not None and source_identity == source.artifact_id
        )
        if (
            source_sha256
            and source_identity
            and not (identity_matches_hash or identity_matches_artifact)
        ):
            if status not in {SOURCE_PROVENANCE_STALE, SOURCE_PROVENANCE_MUTATED}:
                raise AuditIdentityError(
                    "annotation source_identity does not match source_ref.sha256"
                )
        if source is not None and source.source_commit:
            if not revision_token:
                if status == SOURCE_PROVENANCE_VERIFIED:
                    raise AuditIdentityError(
                        "annotation source_revision is unavailable for source_ref.source_commit"
                    )
                status = SOURCE_PROVENANCE_UNAVAILABLE
            elif revision_token != source.source_commit:
                if status not in {SOURCE_PROVENANCE_STALE, SOURCE_PROVENANCE_MUTATED}:
                    raise AuditIdentityError("annotation source_revision does not match source_ref")
        if status == SOURCE_PROVENANCE_VERIFIED and (
            source is None or not source_sha256 or not identity_matches_hash
        ):
            raise AuditIdentityError(
                "verified annotation provenance requires source_ref, source_identity, and source hash"
            )
        revision_unavailable = bool(
            source is not None and source.source_commit and not revision_token
        )
        if (
            status == SOURCE_PROVENANCE_UNAVAILABLE
            and source_sha256
            and identity_matches_hash
            and not revision_unavailable
        ):
            status = SOURCE_PROVENANCE_VERIFIED
        object.__setattr__(self, "source_ref", source)
        object.__setattr__(self, "source_identity", source_identity)
        object.__setattr__(self, "provenance_status", status)
        if self.provenance_reason:
            reason = self.provenance_reason
        else:
            reason = {
                SOURCE_PROVENANCE_VERIFIED: "source identity and revision are bound",
                SOURCE_PROVENANCE_STALE: "source changed after annotation; attachment is stale",
                SOURCE_PROVENANCE_MUTATED: "source identity changed or was mutated",
                SOURCE_PROVENANCE_UNAVAILABLE: "source identity or revision is unavailable",
            }[status]
        object.__setattr__(self, "provenance_reason", reason)
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

    @property
    def source_provenance_status(self) -> str:
        """Return the explicit source attachment provenance status."""

        return self.provenance_status


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
        source_revision=payload.get("source_revision", ""),
        calibration=payload.get("calibration"),
        actor_id=str(payload.get("actor_id", "")),
        object_id=str(payload.get("object_id", "")),
        goal_id=str(payload.get("goal_id", "")),
        waypoint_id=str(payload.get("waypoint_id", "")),
        metric_id=str(payload.get("metric_id", "")),
        event_id=str(payload.get("event_id", "")),
        source_point=(
            tuple(payload["source_point"]) if payload.get("source_point") is not None else None
        ),
        seek_identity=str(payload.get("seek_identity", "")),
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
        source_revision=payload.get("source_revision", 0),
        source_identity=str(payload.get("source_identity", "")),
        created_at=str(payload.get("created_at", utc_now())),
        metadata=dict(payload.get("metadata", {})),
        source_ref=_source_from_payload(payload.get("source_ref")),
        provenance_status=str(payload.get("provenance_status", SOURCE_PROVENANCE_UNAVAILABLE)),
        provenance_reason=str(payload.get("provenance_reason", "")),
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

    _validate_record_payload(payload)
    try:
        kind = payload["record_type"]
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
        declared_id = payload["record_id"]
        if declared_id != record_id(record):
            raise AuditContractError(
                f"record_id {declared_id!r} does not match {record_type(record)} identity"
            )
        return record
    except AuditContractError:
        raise
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
        raise AuditContractError(
            f"invalid {payload.get('record_type', 'audit')} record: {exc}"
        ) from exc


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

    try:
        payload = (
            json.loads(value, parse_constant=_reject_json_constant)
            if isinstance(value, (str, bytes))
            else value
        )
        _strict_json_values(payload)
    except (TypeError, ValueError, UnicodeDecodeError) as exc:
        raise AuditContractError(f"invalid audit JSON: {exc}") from exc
    except AuditContractError:
        raise
    try:
        return record_from_dict(payload)
    except AuditContractError:
        raise
    except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
        raise AuditContractError(f"invalid audit record: {exc}") from exc


deserialize_audit_record = deserialize_record


def write_ndjson(records: Sequence[Any], path: str | Path) -> Path:
    """Write records through :class:`AuditStore`'s canonical writer.

    ``audit.ndjson`` is a transaction journal, not a record-only file.  Keep
    this compatibility helper restricted to that filename and route writes
    through the serialized store so callers cannot create an unreadable
    journal by writing bare records.

    Returns:
        The path written.
    """

    output = Path(path)
    if output.name != "audit.ndjson":
        raise AuditContractError(
            "standalone record NDJSON is not a canonical audit journal; use AuditStore"
        )
    from robot_sf.analysis_workbench.audit_store import AuditStore  # noqa: PLC0415

    record_list = list(records)
    if not record_list:
        output.parent.mkdir(parents=True, exist_ok=True)
        with AuditStore(output.parent):
            pass
        return output
    payloads = [record_to_dict(record) for record in record_list]
    record_ids = [record_id(record) for record in record_list]
    if len(set(record_ids)) != len(record_ids):
        raise AuditContractError("NDJSON writes cannot contain duplicate record IDs")
    with AuditStore(output.parent) as store:
        current = {rid: store.get(rid, include_deleted=True) for rid in record_ids}
        if all(
            item is not None
            and not item.deleted
            and item.record is not None
            and canonical_json(record_to_dict(item.record)) == canonical_json(payload)
            for item, payload in zip(current.values(), payloads, strict=True)
        ) and len(current) == len(payloads):
            return output
        expected = {
            rid: (stored.revision if stored is not None else 0) for rid, stored in current.items()
        }
        operation_payload = {"records": payloads, "expected_revisions": expected}
        operation_id = (
            "ndjson-" + hashlib.sha256(canonical_json(operation_payload).encode()).hexdigest()
        )
        store.commit(record_list, operation_id=operation_id, expected_revisions=expected)
    return output


def read_ndjson(path: str | Path) -> list[Any]:  # noqa: C901
    """Read and validate every complete canonical record in an NDJSON file.

    Returns:
        Typed records in file order.
    """

    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise AuditContractError(f"cannot read NDJSON: {exc}") from exc
    records: list[Any] = []
    first_payload: Mapping[str, Any] | None = None
    for line in lines:
        if line.strip():
            try:
                candidate = json.loads(line, parse_constant=_reject_json_constant)
            except (TypeError, ValueError, UnicodeDecodeError, AuditContractError) as exc:
                raise AuditContractError(f"invalid NDJSON record at line 1: {exc}") from exc
            if isinstance(candidate, Mapping):
                first_payload = candidate
            break
    if first_payload is not None and first_payload.get("kind") == "header":
        if first_payload.get("schema_version") != AUDIT_JOURNAL_SCHEMA_VERSION:
            raise AuditContractError("unsupported audit journal schema")
        from robot_sf.analysis_workbench.audit_store import AuditStore, AuditStoreError  # noqa: PLC0415

        try:
            with AuditStore(Path(path).parent) as store:
                latest: dict[str, Any | None] = {}
                order: list[str] = []
                for transaction in store.journal():
                    for change in transaction["changes"]:
                        rid = str(change["record_id"])
                        if rid not in order:
                            order.append(rid)
                        latest[rid] = (
                            record_from_dict(change["record"])
                            if change["record"] is not None
                            else None
                        )
                return [latest[rid] for rid in order if latest.get(rid) is not None]
        except (AuditStoreError, OSError, AuditContractError) as exc:
            raise AuditContractError(f"invalid audit journal: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            records.append(deserialize_record(line))
        except (AuditContractError, json.JSONDecodeError, UnicodeDecodeError) as exc:
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
    "SOURCE_PROVENANCE_MUTATED",
    "SOURCE_PROVENANCE_STALE",
    "SOURCE_PROVENANCE_STATUSES",
    "SOURCE_PROVENANCE_UNAVAILABLE",
    "SOURCE_PROVENANCE_VERIFIED",
    "ActionRecord",
    "Annotation",
    "AuditContractError",
    "AuditIdentityError",
    "CampaignAudit",
    "EpisodeRef",
    "Finding",
    "ImageDisplayTransform",
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
