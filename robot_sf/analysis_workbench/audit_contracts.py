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
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from types import MappingProxyType
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
DETECTOR_RULE_PROPOSAL_KINDS = (
    "new_rule",
    "threshold_change",
    "parameter_change",
    "cohort_change",
)
DETECTOR_RULE_PROPOSAL_STATUSES = ("proposed", "approved", "rejected", "withdrawn")
DETECTOR_RULE_PROPOSAL_ACTIVATION_STATUSES = ("inactive",)
DETECTOR_RULE_PROPOSAL_IMMUTABLE_FIELDS = (
    "proposal_id",
    "proposal_kind",
    "target_detector_id",
    "candidate_rule",
    "detector_registry_version",
    "detector_registry_digest",
    "campaign_digest",
    "source_identity",
    "source_revision",
    "annotation_ids",
    "finding_ids",
    "episode_ids",
    "rationale",
    "metadata",
    "activation_status",
    "proposer_kind",
    "proposer_id",
    "created_at",
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
_PROPOSAL_TEXT_LIMIT = 4096
_PROPOSAL_RULE_DEPTH_LIMIT = 8
_PROPOSAL_RULE_MAPPING_LIMIT = 64
_PROPOSAL_RULE_SEQUENCE_LIMIT = 128
_PROPOSAL_RULE_NODE_LIMIT = 512
_PROPOSAL_METADATA_LIMIT = 32
_PROPOSAL_ID_LIMIT = 256
_PROPOSAL_DECLARATIVE_KEY_RE = re.compile(r"[a-z][a-z0-9_]{0,127}")
_PROPOSAL_EXECUTION_KEY_PARTS = frozenset(
    {
        "callable",
        "callback",
        "class",
        "command",
        "commands",
        "code",
        "entrypoint",
        "eval",
        "exec",
        "executable",
        "expression",
        "function",
        "import",
        "module",
        "python",
        "runtime",
        "script",
        "shell",
    }
)
_PROPOSAL_EXECUTION_KEY_FRAGMENTS = frozenset(
    item for item in _PROPOSAL_EXECUTION_KEY_PARTS if item not in {"class", "code"}
)
_PROPOSAL_SPLIT_CLASS_CODE_RE = re.compile(r"(?:c_*l_*a_*s_*s|c_*o_*d_*e)(?:_|$)")
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
        return {item.name: _jsonable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise AuditContractError("audit mappings must use string field names")
        return {key: _jsonable(item) for key, item in value.items()}
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


def _closed_mapping(value: Any, *, name: str) -> dict[str, Any]:
    """Return a strict JSON mapping without silently coercing its fields."""

    if not isinstance(value, Mapping):
        raise AuditContractError(f"{name} must be a mapping")
    result = dict(value)
    _jsonable(result)
    _strict_json_values(result, path=name)
    return result


def _proposal_key_is_executable(key: str) -> bool:
    """Return whether a proposal key names an execution mechanism."""

    normalized = "".join(char for char in key.casefold() if char.isalnum())
    parts = {
        part
        for part in "".join(char if char.isalnum() else " " for char in key.casefold()).split()
        if part
    }
    if normalized in _PROPOSAL_EXECUTION_KEY_PARTS or parts & {"class", "code"}:
        return True
    if _PROPOSAL_SPLIT_CLASS_CODE_RE.search(key):
        return True
    return any(fragment in normalized for fragment in _PROPOSAL_EXECUTION_KEY_FRAGMENTS)


def _bounded_proposal_value(  # noqa: C901, PLR0912
    value: Any,
    *,
    name: str,
    depth: int = 0,
    nodes: list[int] | None = None,
    reject_executable_keys: bool = False,
) -> Any:
    """Validate a bounded JSON value used by a detector proposal.

    Proposal data is declarative input.  It is intentionally copied into
    ordinary JSON values, bounded by depth/node/collection limits, and never
    accepted as a callable, module, command, or expression.

    Returns:
        A copied JSON-compatible value.
    """

    if nodes is None:
        nodes = [0]
    nodes[0] += 1
    if nodes[0] > _PROPOSAL_RULE_NODE_LIMIT:
        raise AuditContractError(f"{name} exceeds the proposal value size limit")
    if depth > _PROPOSAL_RULE_DEPTH_LIMIT:
        raise AuditContractError(f"{name} exceeds the proposal nesting limit")
    if isinstance(value, str):
        if len(value) > _PROPOSAL_TEXT_LIMIT:
            raise AuditContractError(f"{name} exceeds the proposal text limit")
        return value
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise AuditContractError(f"{name} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        if len(value) > _PROPOSAL_RULE_MAPPING_LIMIT:
            raise AuditContractError(f"{name} exceeds the proposal mapping limit")
        result: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise AuditContractError(f"{name} keys must be non-empty strings")
            if len(key) > 128:
                raise AuditContractError(f"{name}.{key} exceeds the proposal key limit")
            if reject_executable_keys:
                if _PROPOSAL_DECLARATIVE_KEY_RE.fullmatch(key) is None:
                    raise AuditContractError(
                        f"{name}.{key} is not a declarative lower_snake_case key"
                    )
                if _proposal_key_is_executable(key):
                    raise AuditContractError(
                        f"{name}.{key} is executable proposal data; use declarative values only"
                    )
            result[key] = _bounded_proposal_value(
                item,
                name=f"{name}.{key}",
                depth=depth + 1,
                nodes=nodes,
                reject_executable_keys=reject_executable_keys,
            )
        return result
    if isinstance(value, (tuple, list)):
        if len(value) > _PROPOSAL_RULE_SEQUENCE_LIMIT:
            raise AuditContractError(f"{name} exceeds the proposal sequence limit")
        return [
            _bounded_proposal_value(
                item,
                name=f"{name}[{index}]",
                depth=depth + 1,
                nodes=nodes,
                reject_executable_keys=reject_executable_keys,
            )
            for index, item in enumerate(value)
        ]
    raise AuditContractError(f"{name} contains unsupported value {type(value).__name__}")


def _bounded_proposal_mapping(
    value: Any, *, name: str, reject_executable_keys: bool
) -> dict[str, Any]:
    """Return a bounded, strict JSON mapping for proposal fields."""

    if not isinstance(value, Mapping):
        raise AuditContractError(f"{name} must be a mapping")
    result = _bounded_proposal_value(
        value,
        name=name,
        reject_executable_keys=reject_executable_keys,
    )
    if not isinstance(result, dict):  # pragma: no cover - mapping input guarantees this.
        raise AuditContractError(f"{name} must be a mapping")
    return result


def _freeze_proposal_value(value: Any) -> Any:
    """Return an immutable copy of a proposal JSON value."""

    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_proposal_value(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_proposal_value(item) for item in value)
    return value


def _proposal_text(value: Any, *, name: str, allow_empty: bool = False) -> str:
    """Validate a bounded proposal text field.

    Returns:
        The validated text.
    """

    result = _text(value, name=name, allow_empty=allow_empty)
    if len(result) > _PROPOSAL_TEXT_LIMIT:
        raise AuditContractError(f"{name} exceeds the proposal text limit")
    return result


def _proposal_ids(value: Any, *, name: str) -> tuple[str, ...]:
    """Validate bounded, deterministic evidence ID lists.

    Returns:
        A normalized tuple of evidence IDs.
    """

    result = _tuple_of_strings(value, name=name)
    if len(result) > _PROPOSAL_RULE_SEQUENCE_LIMIT:
        raise AuditContractError(f"{name} exceeds the proposal sequence limit")
    if any(len(item) > _PROPOSAL_ID_LIMIT for item in result):
        raise AuditContractError(f"{name} contains an ID exceeding the proposal key limit")
    return result


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
        _text(value.artifact_id, name="source.artifact_id")
        _text(value.uri, name="source.uri")
        _text(value.format, name="source.format")
        for name in {item.name for item in fields(SourceRef)} - {
            "artifact_id",
            "uri",
            "format",
        }:
            _optional_text(getattr(value, name), name=f"source.{name}")
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
    _text(value["artifact_id"], name="source.artifact_id")
    _text(value["uri"], name="source.uri")
    _text(value["format"], name="source.format")
    for name in known - {"artifact_id", "uri", "format"}:
        _optional_text(value.get(name, ""), name=f"source.{name}")
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
            ("episode_id", self.episode_id),
            ("campaign_id", self.campaign_id),
            ("source_id", self.source_id),
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
        _optional_text(self.title, name="title")
        campaign_digest = _digest(self.campaign_digest, name="campaign_digest", required=True)
        source_digest = _digest(self.source_digest, name="source_digest", required=True)
        _check_record_version(self.schema_version)
        _optional_text(self.protocol_version, name="protocol_version")
        _optional_text(self.protocol_digest, name="protocol_digest")
        source = _coerce_source(self.source)
        source_sha256 = _source_sha256(source)
        if source_sha256 and source_sha256 != source_digest:
            raise AuditIdentityError("source_digest does not match source.sha256")
        metadata = _closed_mapping(self.metadata, name="metadata")
        object.__setattr__(self, "campaign_digest", campaign_digest)
        object.__setattr__(self, "source_digest", source_digest)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "metadata", metadata)


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
        _optional_text(self.detector_version, name="detector_version")
        _optional_text(self.reason_code, name="reason_code")
        _optional_text(self.episode_id, name="episode_id")
        _optional_text(self.message, name="message")
        if self.status not in SIGNAL_STATUSES:
            raise AuditContractError(f"signal status must be one of {SIGNAL_STATUSES}")
        if self.status in {"unavailable", "error"} and not self.message and not self.missingness:
            raise AuditContractError("unavailable/error signals require message or missingness")
        if self.interval is not None and not isinstance(self.interval, TimeInterval):
            object.__setattr__(self, "interval", _interval(self.interval))
        evidence = _tuple_of_mappings(self.evidence, name="evidence")
        measured = _closed_mapping(self.measured, name="measured")
        threshold = (
            None if self.threshold is None else _closed_mapping(self.threshold, name="threshold")
        )
        missingness = _tuple_of_strings(self.missingness, name="missingness")
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "measured", measured)
        object.__setattr__(self, "threshold", threshold)
        object.__setattr__(self, "missingness", missingness)


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
        for name, value in (
            ("actor_id", self.actor_id),
            ("object_id", self.object_id),
            ("goal_id", self.goal_id),
            ("waypoint_id", self.waypoint_id),
            ("metric_id", self.metric_id),
            ("event_id", self.event_id),
        ):
            _optional_text(value, name=f"reference.{name}")
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
        for name, value in (
            ("author_id", self.author_id),
            ("observed_behavior", self.observed_behavior),
            ("suspected_cause", self.suspected_cause),
            ("provenance_reason", self.provenance_reason),
        ):
            _optional_text(value, name=name)
        _text(self.created_at, name="created_at")
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
        evidence = _tuple_of_mappings(self.evidence, name="evidence")
        tags = _tuple_of_strings(self.tags, name="tags")
        metadata = _closed_mapping(self.metadata, name="metadata")
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "tags", tags)
        object.__setattr__(self, "metadata", metadata)

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
        _optional_text(self.policy_version, name="policy_version")
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
        if (
            isinstance(self.input_revision, bool)
            or not isinstance(self.input_revision, int)
            or self.input_revision < 0
        ):
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
        _optional_text(self.source_revision, name="source_revision")
        _text(self.created_at, name="created_at")
        _text(self.updated_at, name="updated_at")
        if self.status not in FINDING_STATUSES:
            raise AuditContractError(f"finding status must be one of {FINDING_STATUSES}")
        for name, members in (
            ("candidate_members", self.candidate_members),
            ("confirmed_members", self.confirmed_members),
            ("negative_controls", self.negative_controls),
        ):
            if any(not isinstance(member, str) or not member for member in members):
                raise AuditContractError(f"{name} must contain non-empty member IDs")
        for name, values in (
            ("evidence", self.evidence),
            ("negative_evidence", self.negative_evidence),
            ("diagnostic_results", self.diagnostic_results),
        ):
            object.__setattr__(self, name, _tuple_of_mappings(values, name=name))
        object.__setattr__(
            self, "observations", _tuple_of_strings(self.observations, name="observations")
        )
        object.__setattr__(
            self, "hypotheses", _tuple_of_strings(self.hypotheses, name="hypotheses")
        )
        object.__setattr__(self, "tags", _tuple_of_strings(self.tags, name="tags"))
        if self.github_issue is not None:
            object.__setattr__(
                self, "github_issue", _closed_mapping(self.github_issue, name="github_issue")
            )

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
class DetectorRuleProposal:
    """A declarative detector-rule candidate awaiting human governance.

    This record is deliberately separate from :class:`Finding`.  A proposal
    describes data that a future detector *could* consume; it is never an
    executable detector implementation or an implicit registry update.  V1
    stores the proposal and its decision while ``activation_status`` remains
    permanently ``inactive``.
    """

    proposal_id: str
    proposal_kind: str
    target_detector_id: str
    candidate_rule: Mapping[str, Any]
    detector_registry_version: str = ""
    detector_registry_digest: str = ""
    campaign_digest: str = ""
    source_identity: str = ""
    source_revision: int | str = ""
    annotation_ids: tuple[str, ...] = ()
    finding_ids: tuple[str, ...] = ()
    episode_ids: tuple[str, ...] = ()
    rationale: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    lifecycle_status: str = "proposed"
    activation_status: str = "inactive"
    proposer_kind: str = "agent"
    proposer_id: str = ""
    author_kind: str = "agent"
    author_id: str = ""
    decided_by_kind: str = ""
    decided_by_id: str = ""
    decided_at: str = ""
    decision_reason: str = ""
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:  # noqa: C901, PLR0912, PLR0915
        """Validate provenance, declarative data, and the human decision gate."""

        _proposal_text(self.proposal_id, name="proposal_id")
        _proposal_text(self.proposal_kind, name="proposal_kind")
        _proposal_text(self.target_detector_id, name="target_detector_id")
        _proposal_text(
            self.detector_registry_version,
            name="detector_registry_version",
            allow_empty=True,
        )
        detector_registry_digest = _digest(
            self.detector_registry_digest,
            name="detector_registry_digest",
        )
        campaign_digest = _digest(self.campaign_digest, name="campaign_digest")
        source_identity = _proposal_text(
            self.source_identity,
            name="source_identity",
            allow_empty=True,
        ).strip()
        if isinstance(self.source_revision, bool) or not isinstance(
            self.source_revision, (int, str)
        ):
            raise AuditContractError("source_revision must be a non-negative integer or token")
        if isinstance(self.source_revision, int) and self.source_revision < 0:
            raise AuditContractError("source_revision must be a non-negative integer")
        source_revision: int | str = (
            ""
            if self.source_revision == ""
            else self.source_revision
            if isinstance(self.source_revision, int)
            else _proposal_text(
                self.source_revision,
                name="source_revision",
                allow_empty=True,
            ).strip()
        )
        if self.source_revision not in (0, "") and not source_revision:
            raise AuditContractError("source_revision token must be non-empty")

        candidate_rule = _bounded_proposal_mapping(
            self.candidate_rule,
            name="candidate_rule",
            reject_executable_keys=True,
        )
        if not candidate_rule:
            raise AuditContractError("candidate_rule must contain declarative data")
        metadata = _bounded_proposal_mapping(
            self.metadata,
            name="metadata",
            reject_executable_keys=False,
        )
        if len(metadata) > _PROPOSAL_METADATA_LIMIT:
            raise AuditContractError("metadata exceeds the proposal field limit")

        annotation_ids = _proposal_ids(self.annotation_ids, name="annotation_ids")
        finding_ids = _proposal_ids(self.finding_ids, name="finding_ids")
        episode_ids = _proposal_ids(self.episode_ids, name="episode_ids")
        rationale = _proposal_text(self.rationale, name="rationale")
        for name, value in (
            ("proposer_id", self.proposer_id),
            ("author_id", self.author_id),
            ("decided_by_id", self.decided_by_id),
        ):
            _proposal_text(value, name=name, allow_empty=True)
        _proposal_text(self.created_at, name="created_at")
        _proposal_text(self.updated_at, name="updated_at")
        decided_at = _proposal_text(self.decided_at, name="decided_at", allow_empty=True)
        decided_by_kind = _proposal_text(
            self.decided_by_kind,
            name="decided_by_kind",
            allow_empty=True,
        )
        decision_reason = _proposal_text(
            self.decision_reason,
            name="decision_reason",
            allow_empty=True,
        )

        if self.proposal_kind not in DETECTOR_RULE_PROPOSAL_KINDS:
            raise AuditContractError(f"proposal_kind must be one of {DETECTOR_RULE_PROPOSAL_KINDS}")
        if self.lifecycle_status not in DETECTOR_RULE_PROPOSAL_STATUSES:
            raise AuditContractError(
                f"lifecycle_status must be one of {DETECTOR_RULE_PROPOSAL_STATUSES}"
            )
        if self.activation_status not in DETECTOR_RULE_PROPOSAL_ACTIVATION_STATUSES:
            raise AuditContractError("detector rule proposals are inactive in audit-record.v1")
        for name, value in (
            ("proposer_kind", self.proposer_kind),
            ("author_kind", self.author_kind),
        ):
            if value not in AUTHOR_KINDS:
                raise AuditContractError(f"{name} must be one of {AUTHOR_KINDS}")

        decision_fields = (
            decided_by_kind,
            self.decided_by_id,
            decided_at,
            decision_reason,
        )
        if self.lifecycle_status == "proposed":
            if any(decision_fields):
                raise AuditContractError(
                    "proposed detector rule proposals cannot carry decision fields"
                )
        else:
            if self.author_kind != "human":
                raise AuditContractError(
                    "approved, rejected, or withdrawn proposals require a human author"
                )
            if decided_by_kind != "human":
                raise AuditContractError("proposal decisions must be made by a human")
            if not self.decided_by_id.strip():
                raise AuditContractError("decided_by_id is required for a proposal decision")
            if not self.decided_at.strip():
                raise AuditContractError("decided_at is required for a proposal decision")
            if not decision_reason.strip():
                raise AuditContractError("decision_reason is required for a proposal decision")

        object.__setattr__(self, "candidate_rule", _freeze_proposal_value(candidate_rule))
        object.__setattr__(self, "metadata", _freeze_proposal_value(metadata))
        object.__setattr__(self, "detector_registry_digest", detector_registry_digest)
        object.__setattr__(self, "campaign_digest", campaign_digest)
        object.__setattr__(self, "source_identity", source_identity)
        object.__setattr__(self, "source_revision", source_revision)
        object.__setattr__(self, "annotation_ids", annotation_ids)
        object.__setattr__(self, "finding_ids", finding_ids)
        object.__setattr__(self, "episode_ids", episode_ids)
        object.__setattr__(self, "rationale", rationale)
        object.__setattr__(self, "decided_by_kind", decided_by_kind)
        object.__setattr__(self, "decided_at", decided_at)
        object.__setattr__(self, "decision_reason", decision_reason)


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
        _optional_text(self.actor_id, name="actor_id")
        _optional_text(self.target_id, name="target_id")
        _text(self.status, name="status")
        _text(self.created_at, name="created_at")
        object.__setattr__(self, "details", _closed_mapping(self.details, name="details"))


@dataclass(frozen=True, slots=True)
class ReviewRecord:
    """Explicit review receipt; opening an interval is not full review credit.

    ``source_identity`` is the opaque BA-04 source-revision token (usually the
    admitted source commit), not the source digest used by annotation records.
    ``scan_identity`` is the opaque BA-01 scan token (usually its cache key).
    """

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
    # ``source_revision`` is intentionally retained as the integer editor
    # selection/CAS revision.  These optional opaque tokens carry the distinct
    # BA-04 source and scan identities without changing that existing meaning.
    source_identity: str = ""
    scan_identity: str = ""

    def __post_init__(self) -> None:
        """Validate explicit review scope and author identity."""

        _text(self.review_id, name="review_id")
        _text(self.episode_id, name="episode_id")
        if self.scope not in REVIEW_SCOPES:
            raise AuditContractError(f"review scope must be one of {REVIEW_SCOPES}")
        if self.author_kind not in AUTHOR_KINDS:
            raise AuditContractError(f"author_kind must be one of {AUTHOR_KINDS}")
        if (
            isinstance(self.source_revision, bool)
            or not isinstance(self.source_revision, int)
            or self.source_revision < 0
        ):
            raise AuditContractError("source_revision must be a non-negative integer")
        object.__setattr__(
            self, "annotation_ids", _tuple_of_strings(self.annotation_ids, name="annotation_ids")
        )
        _optional_text(self.outcome, name="outcome")
        _optional_text(self.author_id, name="author_id")
        _optional_text(self.notes, name="notes")
        source_identity = _optional_text(self.source_identity, name="source_identity").strip()
        scan_identity = _optional_text(self.scan_identity, name="scan_identity").strip()
        _text(self.created_at, name="created_at")
        object.__setattr__(self, "source_identity", source_identity)
        object.__setattr__(self, "scan_identity", scan_identity)


_RECORD_TYPES: dict[str, type[Any]] = {
    "campaign_audit": CampaignAudit,
    "episode_ref": EpisodeRef,
    "review_packet": ReviewPacket,
    "signal": Signal,
    "reference": Reference,
    "annotation": Annotation,
    "finding": Finding,
    "detector_rule_proposal": DetectorRuleProposal,
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
        "detector_rule_proposal": "proposal_id",
        "action_record": "action_id",
        "review_record": "review_id",
    }[kind]
    if hasattr(record, field_name):
        return _text(getattr(record, field_name), name=f"{kind}.{field_name}")
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
    if include_type:
        _validate_record_payload(payload)
    return payload


def _tuple_of_mappings(value: Any, *, name: str) -> tuple[Mapping[str, Any], ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise AuditContractError(f"{name} must be a sequence")
    return tuple(_closed_mapping(item, name=f"{name}[{index}]") for index, item in enumerate(value))


def _mapping_error(name: str) -> Mapping[str, Any]:
    raise AuditContractError(f"{name} must contain mappings")


def _tuple_of_strings(value: Any, *, name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise AuditContractError(f"{name} must be a sequence")
    if any(not isinstance(item, str) or not item for item in value):
        raise AuditContractError(f"{name} must contain non-empty strings")
    return tuple(value)


def _interval(value: Any) -> TimeInterval | None:
    if value is None:
        return None
    if isinstance(value, TimeInterval):
        return value
    if not isinstance(value, Mapping):
        raise AuditContractError("interval must be a mapping")
    return TimeInterval(value["start_s"], value.get("end_s", value["start_s"]))


def _source_from_payload(value: Any) -> SourceRef | None:
    return _coerce_source(value)


def episode_ref_from_dict(payload: Mapping[str, Any]) -> EpisodeRef:
    """Build an :class:`EpisodeRef` from a versioned mapping.

    Returns:
        Validated episode identity.
    """

    return EpisodeRef(
        campaign_digest=payload.get("campaign_digest", payload.get("campaign_id", "")),
        source_digest=payload.get("source_digest", payload.get("source_id", "")),
        execution_id=payload.get("execution_id", ""),
        planner_id=payload.get("planner_id", ""),
        scenario_id=payload.get("scenario_id", ""),
        seed=payload.get("seed"),
        attempt=payload.get("attempt", 0),
        config_digest=payload.get("config_digest", ""),
        checkpoint_digest=payload.get("checkpoint_digest", ""),
        environment_digest=payload.get("environment_digest", ""),
        source=_source_from_payload(payload.get("source")),
        episode_id=payload.get("episode_id", ""),
    )


def reference_from_dict(payload: Mapping[str, Any]) -> Reference:
    return Reference(
        reference_id=payload["reference_id"],
        coordinate_frame=payload["coordinate_frame"],
        point=tuple(payload["point"]),
        source=_source_from_payload(payload.get("source")),
        timestamp_s=payload.get("timestamp_s"),
        source_revision=payload.get("source_revision", ""),
        calibration=payload.get("calibration"),
        actor_id=payload.get("actor_id", ""),
        object_id=payload.get("object_id", ""),
        goal_id=payload.get("goal_id", ""),
        waypoint_id=payload.get("waypoint_id", ""),
        metric_id=payload.get("metric_id", ""),
        event_id=payload.get("event_id", ""),
        source_point=(
            tuple(payload["source_point"]) if payload.get("source_point") is not None else None
        ),
        seek_identity=payload.get("seek_identity", ""),
    )


def signal_from_dict(payload: Mapping[str, Any]) -> Signal:
    """Build a detector signal from a canonical mapping.

    Returns:
        Validated signal record.
    """

    return Signal(
        signal_id=payload["signal_id"],
        detector_id=payload["detector_id"],
        detector_version=payload.get("detector_version", ""),
        status=payload.get("status", "flagged"),
        reason_code=payload.get("reason_code", ""),
        episode_id=payload.get("episode_id", ""),
        evidence=_tuple_of_mappings(payload.get("evidence", ()), name="evidence"),
        measured=payload.get("measured", {}),
        interval=_interval(payload.get("interval")),
        threshold=payload.get("threshold"),
        missingness=_tuple_of_strings(payload.get("missingness", ()), name="missingness"),
        message=payload.get("message", ""),
    )


def annotation_from_dict(payload: Mapping[str, Any]) -> Annotation:
    """Build an annotation from a canonical mapping.

    Returns:
        Validated annotation record.
    """

    return Annotation(
        annotation_id=payload["annotation_id"],
        episode_id=payload["episode_id"],
        classification=payload["classification"],
        mode=payload.get("mode", "quick"),
        author_kind=payload.get("author_kind", "human"),
        author_id=payload.get("author_id", ""),
        interval=_interval(payload.get("interval")),
        observed_behavior=payload.get("observed_behavior", ""),
        suspected_cause=payload.get("suspected_cause", ""),
        evidence=_tuple_of_mappings(payload.get("evidence", ()), name="evidence"),
        confidence=payload.get("confidence"),
        references=tuple(reference_from_dict(item) for item in payload.get("references", ())),
        tags=_tuple_of_strings(payload.get("tags", ()), name="tags"),
        review_scope=payload.get("review_scope", "interval"),
        source_revision=payload.get("source_revision", 0),
        source_identity=payload.get("source_identity", ""),
        created_at=payload.get("created_at", utc_now()),
        metadata=payload.get("metadata", {}),
        source_ref=_source_from_payload(payload.get("source_ref")),
        provenance_status=payload.get("provenance_status", SOURCE_PROVENANCE_UNAVAILABLE),
        provenance_reason=payload.get("provenance_reason", ""),
    )


def finding_from_dict(payload: Mapping[str, Any]) -> Finding:
    """Build a finding from a canonical mapping.

    Returns:
        Validated finding record.
    """

    return Finding(
        finding_id=payload["finding_id"],
        title=payload["title"],
        status=payload.get("status", "proposed"),
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
        github_issue=payload.get("github_issue"),
        source_revision=payload.get("source_revision", ""),
        created_at=payload.get("created_at", utc_now()),
        updated_at=payload.get("updated_at", utc_now()),
    )


def detector_rule_proposal_from_dict(payload: Mapping[str, Any]) -> DetectorRuleProposal:
    """Build a detector-rule proposal from a canonical mapping.

    Returns:
        A validated :class:`DetectorRuleProposal`.
    """

    return DetectorRuleProposal(
        proposal_id=payload["proposal_id"],
        proposal_kind=payload["proposal_kind"],
        target_detector_id=payload["target_detector_id"],
        candidate_rule=payload["candidate_rule"],
        detector_registry_version=payload.get("detector_registry_version", ""),
        detector_registry_digest=payload.get("detector_registry_digest", ""),
        campaign_digest=payload.get("campaign_digest", ""),
        source_identity=payload.get("source_identity", ""),
        source_revision=payload.get("source_revision", ""),
        annotation_ids=_proposal_ids(payload.get("annotation_ids", ()), name="annotation_ids"),
        finding_ids=_proposal_ids(payload.get("finding_ids", ()), name="finding_ids"),
        episode_ids=_proposal_ids(payload.get("episode_ids", ()), name="episode_ids"),
        rationale=payload.get("rationale", ""),
        metadata=payload.get("metadata", {}),
        lifecycle_status=payload.get("lifecycle_status", "proposed"),
        activation_status=payload.get("activation_status", "inactive"),
        proposer_kind=payload.get("proposer_kind", "agent"),
        proposer_id=payload.get("proposer_id", ""),
        author_kind=payload.get("author_kind", "agent"),
        author_id=payload.get("author_id", ""),
        decided_by_kind=payload.get("decided_by_kind", ""),
        decided_by_id=payload.get("decided_by_id", ""),
        decided_at=payload.get("decided_at", ""),
        decision_reason=payload.get("decision_reason", ""),
        created_at=payload.get("created_at", utc_now()),
        updated_at=payload.get("updated_at", utc_now()),
    )


def review_packet_from_dict(payload: Mapping[str, Any]) -> ReviewPacket:
    """Build a review packet from a canonical mapping.

    Returns:
        Validated review packet.
    """

    return ReviewPacket(
        packet_id=payload["packet_id"],
        primary=episode_ref_from_dict(payload["primary"]),
        peers=tuple(episode_ref_from_dict(item) for item in payload.get("peers", ())),
        signals=tuple(signal_from_dict(item) for item in payload.get("signals", ())),
        selection_reasons=_tuple_of_strings(
            payload.get("selection_reasons", ()), name="selection_reasons"
        ),
        missingness=_tuple_of_strings(payload.get("missingness", ()), name="missingness"),
        policy_version=payload.get("policy_version", ""),
        input_revision=payload.get("input_revision", 0),
    )


def record_from_dict(payload: Mapping[str, Any]) -> Any:  # noqa: C901, PLR0912
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
        elif kind == "detector_rule_proposal":
            record = detector_rule_proposal_from_dict(payload)
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
                    values["details"] = values.get("details", {})
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
    "DETECTOR_RULE_PROPOSAL_ACTIVATION_STATUSES",
    "DETECTOR_RULE_PROPOSAL_IMMUTABLE_FIELDS",
    "DETECTOR_RULE_PROPOSAL_KINDS",
    "DETECTOR_RULE_PROPOSAL_STATUSES",
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
    "DetectorRuleProposal",
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
    "detector_rule_proposal_from_dict",
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
