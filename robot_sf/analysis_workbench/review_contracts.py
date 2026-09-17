"""Typed review contracts and the fail-closed admitted-source resolver.

This module owns the SREV-01 shared-contract surface: versioned interfaces,
JSON schemas, validation with stable reason codes, canonical content digests,
and the frozen ``run(request)`` invocation with inspect and capability-report
behavior. It reuses the canonical trace/timeline/annotation owners for
computation and never replaces them.

The ``admitted-source-receipt.v1`` contract is deliberately separate from
scientific evidence admission. It proves that a fixture source is the exact
local byte sequence bound to a request and recipe; it does not authorize a
benchmark, planner, simulator, or paper-facing claim.
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import json
import math
import os
import re
import stat
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from functools import cache
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import urlsplit

from jsonschema import Draft202012Validator

from robot_sf.errors import RobotSfError

SCHEMA_DIR = Path(__file__).with_name("schemas")

REVIEW_BUNDLE_SCHEMA_VERSION = "review-bundle.v1"
VISUALIZATION_SPEC_SCHEMA_VERSION = "visualization-spec.v1"
COMPONENT_DESCRIPTOR_SCHEMA_VERSION = "component-descriptor.v1"
COMPONENT_REQUEST_SCHEMA_VERSION = "component-request.v1"
COMPONENT_RESULT_SCHEMA_VERSION = "component-result.v1"
EXPERIMENT_RECIPE_SCHEMA_VERSION = "experiment-recipe.v1"
ADMITTED_SOURCE_RECEIPT_SCHEMA_VERSION = "admitted-source-receipt.v1"

SCHEMA_FILES = {
    REVIEW_BUNDLE_SCHEMA_VERSION: "review_bundle.v1.json",
    VISUALIZATION_SPEC_SCHEMA_VERSION: "visualization_spec.v1.json",
    COMPONENT_DESCRIPTOR_SCHEMA_VERSION: "component_descriptor.v1.json",
    COMPONENT_REQUEST_SCHEMA_VERSION: "component_request.v1.json",
    COMPONENT_RESULT_SCHEMA_VERSION: "component_result.v1.json",
    EXPERIMENT_RECIPE_SCHEMA_VERSION: "experiment_recipe.v1.json",
    ADMITTED_SOURCE_RECEIPT_SCHEMA_VERSION: "admitted_source_receipt.v1.json",
}

SUPPORTED_MAJOR_VERSIONS = {"v1"}
RESULT_STATUSES = ("complete", "partial", "unavailable", "failed", "cancelled")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_SHA40_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_TEST_PRESET = {"width": 320, "height": 180, "fps": 10.0, "speed": 1.0}
_ADMITTED_SOURCE_SUPPORTS_DIR_FD = os.open in os.supports_dir_fd

MAX_REVIEW_CONTRACT_DIAGNOSTIC_CHARS = 200
MAX_REVIEW_CONTRACT_VALIDATION_ERRORS = 32
MAX_REVIEW_CONTRACT_ID_CHARS = 256
MAX_REVIEW_CONTRACT_CLI_INPUT_BYTES = 256 * 1024
MAX_REVIEW_CONTRACT_REASON_CHARS = MAX_REVIEW_CONTRACT_DIAGNOSTIC_CHARS
MAX_ADMITTED_SOURCE_RECEIPT_BYTES = 256 * 1024
MAX_ADMITTED_SOURCE_BYTES = 16 * 1024 * 1024
MAX_ADMITTED_SOURCE_RECEIPT_ID_CHARS = MAX_REVIEW_CONTRACT_ID_CHARS
MAX_ADMITTED_SOURCE_URI_CHARS = 4_096
MAX_ADMITTED_SOURCE_METADATA_CHARS = 256

ADMITTED_SOURCE_REASON_RECEIPT_MISSING = "receipt_missing"
ADMITTED_SOURCE_REASON_RECEIPT_UNREADABLE = "receipt_unreadable"
ADMITTED_SOURCE_REASON_RECEIPT_MALFORMED = "receipt_malformed"
ADMITTED_SOURCE_REASON_RECEIPT_UNSUPPORTED = "receipt_unsupported"
ADMITTED_SOURCE_REASON_RECEIPT_STALE = "receipt_stale"
ADMITTED_SOURCE_REASON_SOURCE_MISSING = "source_missing"
ADMITTED_SOURCE_REASON_SOURCE_MUTATED = "source_mutated"
ADMITTED_SOURCE_REASON_SOURCE_ESCAPED_ROOT = "source_escaped_root"
ADMITTED_SOURCE_REASON_SOURCE_NOT_REGULAR = "source_not_regular"
ADMITTED_SOURCE_REASON_SOURCE_TOO_LARGE = "source_too_large"
ADMITTED_SOURCE_REASON_SOURCE_PROTECTION_UNAVAILABLE = "source_protection_unavailable"
ADMITTED_SOURCE_REASON_ALLOWED_ROOT_MISSING = "allowed_root_missing"
ADMITTED_SOURCE_REASON_ALLOWED_ROOT_INVALID = "allowed_root_invalid"

ADMITTED_SOURCE_STATUS = "admitted"
ADMITTED_SOURCE_KINDS = frozenset({"fixture", "diagnostic"})
ADMITTED_SOURCE_EVIDENCE_BOUNDARY = "diagnostic_only"
ADMITTED_SOURCE_DEPENDENT_FAMILY_STATUS = "standalone_fixture_only"


def _bounded_text_detail(value: Any, *, limit: int = MAX_REVIEW_CONTRACT_DIAGNOSTIC_CHARS) -> str:
    """Return a compact bounded representation of arbitrary diagnostic text."""
    detail = value if isinstance(value, str) else str(value)
    detail = " ".join(detail.split())
    if not detail:
        return type(value).__name__
    return detail[:limit]


_VALIDATION_ERRORS_OMISSION_MARKER = "additional validation errors omitted"


def _bounded_validation_errors(errors: list[str]) -> tuple[str, ...]:
    """Return public validation errors within the cap, including its marker.

    The omission marker occupies one entry in the public tuple.  Therefore an
    overflowing input retains at most ``cap - 1`` concrete messages before
    appending the marker.
    """
    overflowing = len(errors) > MAX_REVIEW_CONTRACT_VALIDATION_ERRORS
    concrete_limit = (
        MAX_REVIEW_CONTRACT_VALIDATION_ERRORS - 1
        if overflowing
        else MAX_REVIEW_CONTRACT_VALIDATION_ERRORS
    )
    bounded_errors = [_bounded_text_detail(error) for error in errors[:concrete_limit]]
    if overflowing:
        bounded_errors.append(_VALIDATION_ERRORS_OMISSION_MARKER)
    return tuple(bounded_errors)


class _AdmittedSourceInputLimitError(ValueError):
    """Identify a receipt or source that exceeds the resolver input ceiling."""


class _AdmittedSourceFileRejectedError(ValueError):
    """Identify a path rejected before any potentially blocking read."""

    def __init__(self, detail: str):
        """Store the stable rejection detail used by loader boundaries."""
        self.detail = detail
        super().__init__(detail)


class ReviewContractsValidationError(RobotSfError, ValueError):
    """Raised when a review-contract payload fails schema or semantic validation.

    The public ``errors`` tuple is capped at
    ``MAX_REVIEW_CONTRACT_VALIDATION_ERRORS`` entries.  When violations exceed
    that cap, the final entry is the omission marker, so it replaces one
    concrete message rather than adding a 33rd entry to the configured limit.
    """

    def __init__(self, errors: list[str], *, source: str | Path | None = None):
        """Build an actionable validation error."""

        self.errors = _bounded_validation_errors(errors)
        self.source = _bounded_text_detail(source) if source is not None else None
        prefix = f"{self.source}: " if self.source else ""
        super().__init__(prefix + "; ".join(self.errors))


@cache
def load_review_contracts_schema(version: str) -> dict[str, Any]:
    """Load one public review-contract JSON schema.

    Returns:
        Parsed JSON Schema document.


    Args:
        version: Schema version such as ``review-bundle.v1``.

    Returns:
        Parsed JSON Schema document.

    Raises:
        ReviewContractsValidationError: For an unknown schema version.
    """

    try:
        filename = SCHEMA_FILES[version]
    except KeyError:
        raise ReviewContractsValidationError([f"unknown schema version: {version}"]) from None
    return json.loads((SCHEMA_DIR / filename).read_text(encoding="utf-8"))


def _schema_errors(version: str, payload: Mapping[str, Any]) -> list[str]:
    validator = Draft202012Validator(load_review_contracts_schema(version))
    violations = []
    for error in validator.iter_errors(payload):
        violations.append(error)
        if len(violations) > MAX_REVIEW_CONTRACT_VALIDATION_ERRORS:
            break
    violations.sort(key=lambda err: list(err.absolute_path))
    errors = [
        f"{_pointer(error.absolute_path)}: {_bounded_text_detail(error.message)}"
        for error in violations
    ]
    if len(violations) > MAX_REVIEW_CONTRACT_VALIDATION_ERRORS:
        errors.append(f"/: {_VALIDATION_ERRORS_OMISSION_MARKER}")
    return list(_bounded_validation_errors(errors))


def _pointer(path: Any) -> str:
    return _bounded_text_detail("/" + "/".join(str(token) for token in path))


def _require_schema(version: str, payload: Mapping[str, Any], *, source: Any = None) -> None:
    errors = _schema_errors(version, payload)
    if errors:
        raise ReviewContractsValidationError(errors, source=source)


def _check_version_supported(version: str, *, source: Any = None) -> None:
    major = version.rsplit("v", 1)[-1].split(".", 1)[0]
    if f"v{major}" not in SUPPORTED_MAJOR_VERSIONS:
        raise ReviewContractsValidationError(
            [f"unsupported major version: {version}"], source=source
        )


def _check_finite(value: Any, *, path: str, errors: list[str]) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return
    if not math.isfinite(value):
        errors.append(f"{path}: non-finite number is not strict-JSON safe")


def _check_no_traversal(value: str, *, path: str, errors: list[str]) -> None:
    pure = Path(value)
    if pure.is_absolute() or ".." in pure.parts:
        errors.append(
            f"{path}: path traversal or absolute path is rejected: {_bounded_text_detail(value)}"
        )


def _check_safe_artifact_id(value: Any, *, path: str, errors: list[str]) -> None:
    """Reject source identifiers that cannot be used as one output filename.

    ``artifact_id`` is an identifier rather than a relative path.  Rejecting
    both POSIX and Windows separators keeps the contract safe when a request
    created on one platform is consumed on another.
    """

    if not isinstance(value, str):
        return
    windows = PureWindowsPath(value)
    if (
        value in {".", ".."}
        or "/" in value
        or "\\" in value
        or windows.is_absolute()
        or bool(windows.drive)
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        errors.append(f"{path}: unsafe artifact id for a filename: {value!r}")


def _check_sha256(value: Any, *, path: str, errors: list[str]) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        errors.append(f"{path}: expected 64-hex SHA-256")


def _check_sha40(value: Any, *, path: str, errors: list[str]) -> None:
    if not isinstance(value, str) or _SHA40_RE.fullmatch(value) is None:
        errors.append(f"{path}: expected 40-hex commit SHA")


def _contains_lone_unicode_surrogate(value: str) -> bool:
    """Return whether *value* contains an unpaired UTF-16 surrogate."""
    index = 0
    while index < len(value):
        codepoint = ord(value[index])
        if 0xD800 <= codepoint <= 0xDBFF:
            if index + 1 < len(value) and 0xDC00 <= ord(value[index + 1]) <= 0xDFFF:
                index += 2
                continue
            return True
        if 0xDC00 <= codepoint <= 0xDFFF:
            return True
        index += 1
    return False


def _contains_unicode_surrogate(value: Any) -> bool:
    """Return whether a contract payload contains a lone UTF-16 surrogate.

    Contract payloads can include retained configuration and other nested
    JSON values.  Walk those values iteratively so maliciously deep input
    cannot turn this boundary check into a recursion failure.  Mapping keys
    are checked as well because they participate in canonical identities.
    """
    pending: list[Any] = [value]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if isinstance(current, str):
            if _contains_lone_unicode_surrogate(current):
                return True
            continue
        if isinstance(current, Mapping):
            marker = id(current)
            if marker in visited:
                continue
            visited.add(marker)
            for key, nested in current.items():
                pending.extend((key, nested))
        elif isinstance(current, (list, tuple)):
            marker = id(current)
            if marker in visited:
                continue
            visited.add(marker)
            pending.extend(current)
    return False


def _reject_unicode_surrogates(payload: Any, *, source: Any = None) -> None:
    """Reject Unicode surrogate code points before schema or digest handling."""
    if _contains_unicode_surrogate(payload):
        raise ReviewContractsValidationError(
            ["payload contains unsupported Unicode surrogate"], source=source
        )


def _contains_non_finite(value: Any) -> bool:
    """Return whether nested JSON-like values contain NaN or infinity."""
    pending: list[Any] = [value]
    visited: set[int] = set()
    while pending:
        current = pending.pop()
        if isinstance(current, float) and not math.isfinite(current):
            return True
        if isinstance(current, Mapping):
            marker = id(current)
            if marker in visited:
                continue
            visited.add(marker)
            for key, nested in current.items():
                pending.extend((key, nested))
        elif isinstance(current, (list, tuple)):
            marker = id(current)
            if marker in visited:
                continue
            visited.add(marker)
            pending.extend(current)
    return False


def _reject_non_finite(payload: Any, *, source: Any = None) -> None:
    """Reject non-finite numbers before direct API schema or digest handling."""
    if _contains_non_finite(payload):
        raise ReviewContractsValidationError(["payload contains non-finite number"], source=source)


def _request_finite_validation_payload(
    payload: Mapping[str, Any], *, source: Any = None
) -> Mapping[str, Any]:
    """Select request fields whose finite values this shared boundary owns.

    Source-bound requests are treated as untrusted persisted or CLI data and
    are checked recursively in full.  An in-memory request without a source
    marker has its component-owned config deferred to the component runner;
    envelope fields remain subject to this boundary's checks.

    Returns:
        The request mapping to use for recursive finite-value validation.
    """
    if source is not None:
        return payload
    return {key: value for key, value in payload.items() if key != "config"}


def _result_validation_payload(
    payload: Mapping[str, Any], *, source: Any = None
) -> Mapping[str, Any]:
    """Bound an internal result reason before strict result validation.

    Source-bound result documents remain strict.  Internal component results
    may assemble long diagnostic details, so their leading reason code is
    retained while the detail is bounded for the result envelope.

    Returns:
        The result mapping to validate.
    """
    if source is not None:
        return payload
    reason = payload.get("reason")
    if isinstance(reason, str) and len(reason) > MAX_REVIEW_CONTRACT_REASON_CHARS:
        return {
            **payload,
            "reason": _bounded_text_detail(reason, limit=MAX_REVIEW_CONTRACT_REASON_CHARS),
        }
    return payload


def _check_envelope_identity_limits(
    payload: Mapping[str, Any], *, source: Any = None, include_reason: bool = False
) -> None:
    """Reject oversized envelope identities and failure details before construction."""
    errors: list[str] = []
    for field_name in ("request_id", "component_id"):
        value = payload.get(field_name)
        if isinstance(value, str) and len(value) > MAX_REVIEW_CONTRACT_ID_CHARS:
            errors.append(
                f"/{field_name}: exceeds maximum length of {MAX_REVIEW_CONTRACT_ID_CHARS} characters"
            )
    if include_reason:
        reason = payload.get("reason")
        if isinstance(reason, str) and len(reason) > MAX_REVIEW_CONTRACT_REASON_CHARS:
            errors.append(
                f"/reason: exceeds maximum length of {MAX_REVIEW_CONTRACT_REASON_CHARS} characters"
            )
    if errors:
        raise ReviewContractsValidationError(errors, source=source)


def _safe_identity(value: Any) -> str:
    """Return a bounded identity suitable for a failure result envelope."""
    if isinstance(value, str) and value and len(value) <= MAX_REVIEW_CONTRACT_ID_CHARS:
        return value
    return "unknown"


def _bounded_join(values: Any, *, separator: str = ", ") -> str:
    """Join diagnostic values without constructing an unbounded detail string.

    Returns:
        A bounded joined diagnostic string.
    """
    parts: list[str] = []
    total = 0
    omitted = "additional details omitted"
    for value in values:
        part = _bounded_text_detail(value)
        added = len(part) + (len(separator) if parts else 0)
        if total + added > MAX_REVIEW_CONTRACT_DIAGNOSTIC_CHARS:
            remaining = (
                MAX_REVIEW_CONTRACT_DIAGNOSTIC_CHARS - total - (len(separator) if parts else 0)
            )
            if remaining > 0:
                parts.append(omitted[:remaining])
            break
        parts.append(part)
        total += added
    return separator.join(parts)


@dataclass(frozen=True, slots=True)
class SourceRef:
    """Identity and integrity pointer to one source artifact."""

    artifact_id: str
    uri: str
    format: str
    schema: str = ""
    sha256: str = ""
    source_commit: str = ""
    config_identity: str = ""
    units: str = ""
    coordinate_frame: str = ""


@dataclass(frozen=True, slots=True)
class ReviewBundle:
    """Index of episode/trace/geometry/media/diagnostic references."""

    bundle_id: str
    episodes: tuple[dict[str, Any], ...]
    extensions: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VisualizationSpec:
    """Renderer-neutral visualization specification."""

    spec_id: str
    sources: tuple[dict[str, Any], ...]
    document: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ComponentDescriptor:
    """Self-contained capability descriptor for one component."""

    component_id: str
    component_version: str
    supported_input_versions: tuple[str, ...]
    output_types: tuple[str, ...]
    required_capabilities: tuple[str, ...] = ()
    optional_capabilities: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ComponentRequest:
    """Fixture request envelope for a component invocation."""

    request_id: str
    component_id: str
    sources: tuple[SourceRef, ...]
    output_directory: str
    config: dict[str, Any] = field(default_factory=dict)
    required_capabilities: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ComponentResult:
    """Result envelope for a component invocation."""

    request_id: str
    component_id: str
    status: str
    artifacts: tuple[dict[str, Any], ...] = ()
    diagnostics: tuple[dict[str, Any], ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ExperimentRecipe:
    """Finite candidate-intervention recipe."""

    recipe_id: str
    document: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AdmittedSourceRef:
    """Integrity and provenance identity for one locally resolved source."""

    uri: str
    format: str
    schema: str
    sha256: str
    source_commit: str
    config_identity: str
    units: str = ""
    coordinate_frame: str = ""

    def to_dict(self) -> dict[str, str]:
        """Return the source identity as a JSON-safe mapping."""

        payload = {
            "uri": self.uri,
            "format": self.format,
            "schema": self.schema,
            "sha256": self.sha256,
            "source_commit": self.source_commit,
            "config_identity": self.config_identity,
        }
        if self.units:
            payload["units"] = self.units
        if self.coordinate_frame:
            payload["coordinate_frame"] = self.coordinate_frame
        return payload


@dataclass(frozen=True, slots=True)
class AdmittedSourceReceipt:
    """Versioned receipt binding a fixture source to request and recipe identities."""

    receipt_id: str
    source: AdmittedSourceRef
    request_sha256: str
    recipe_sha256: str
    status: str
    source_kind: str
    evidence_boundary: str
    scientific_claim_allowed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the receipt as a JSON-safe mapping."""

        return {
            "schema_version": ADMITTED_SOURCE_RECEIPT_SCHEMA_VERSION,
            "receipt_id": self.receipt_id,
            "source": self.source.to_dict(),
            "request_sha256": self.request_sha256,
            "recipe_sha256": self.recipe_sha256,
            "status": self.status,
            "source_kind": self.source_kind,
            "evidence_boundary": self.evidence_boundary,
            "scientific_claim_allowed": self.scientific_claim_allowed,
        }


@dataclass(frozen=True, slots=True)
class AdmittedSourceResolution:
    """Result of resolving and protected-reading one admitted-source receipt."""

    status: str
    reason: str
    source_path: Path | None = None
    receipt: AdmittedSourceReceipt | None = None
    detail: str = ""
    source_bytes: bytes | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a compact JSON-safe resolution report."""

        return {
            "status": self.status,
            "reason": self.reason,
            "source_path": str(self.source_path) if self.source_path is not None else None,
            "receipt": self.receipt.to_dict() if self.receipt is not None else None,
            "detail": self.detail,
        }


def _resolver_value_or_rejection(
    value: Any,
    early: AdmittedSourceResolution | None,
    *,
    status: str,
    reason: str,
    detail: str,
    receipt: AdmittedSourceReceipt | None = None,
) -> Any:
    """Return a helper result or a bounded rejection for an impossible empty value."""
    if early is not None:
        return early
    if value is None:
        return _resolution(status, reason, receipt=receipt, detail=detail)
    return value


def review_bundle_from_dict(payload: Mapping[str, Any], *, source: Any = None) -> ReviewBundle:
    """Validate and build a review bundle, checking semantic boundaries.

    Returns:
        Validated review bundle.
    """
    _reject_non_finite(payload, source=source)
    _require_schema(REVIEW_BUNDLE_SCHEMA_VERSION, payload, source=source)
    errors: list[str] = []
    seen: set[str] = set()
    for episode in payload["episodes"]:
        for ref in episode["references"]:
            artifact_id = ref["artifact_id"]
            if artifact_id in seen:
                errors.append(f"/episodes: duplicate scoped artifact id: {artifact_id}")
            seen.add(artifact_id)
            _check_sha256(ref["sha256"], path=f"/episodes/{artifact_id}/sha256", errors=errors)
            _check_sha40(
                ref["source_commit"], path=f"/episodes/{artifact_id}/source_commit", errors=errors
            )
            _check_no_traversal(ref["uri"], path=f"/episodes/{artifact_id}/uri", errors=errors)
    if errors:
        raise ReviewContractsValidationError(errors, source=source)
    known = {"schema_version", "bundle_id", "episodes"}
    return ReviewBundle(
        bundle_id=str(payload["bundle_id"]),
        episodes=tuple(dict(episode) for episode in payload["episodes"]),
        extensions={k: payload[k] for k in payload if k not in known},
    )


def visualization_spec_from_dict(
    payload: Mapping[str, Any], *, source: Any = None
) -> VisualizationSpec:
    """Validate and build a visualization spec, checking time and unit boundaries.

    Returns:
        Validated visualization spec.
    """
    _reject_non_finite(payload, source=source)
    _require_schema(VISUALIZATION_SPEC_SCHEMA_VERSION, payload, source=source)
    errors: list[str] = []
    for index, interval in enumerate(payload.get("source_intervals", [])):
        start, end = interval["start_s"], interval["end_s"]
        _check_finite(start, path=f"/source_intervals/{index}/start_s", errors=errors)
        _check_finite(end, path=f"/source_intervals/{index}/end_s", errors=errors)
        if isinstance(start, (int, float)) and isinstance(end, (int, float)):
            if not end > start:
                errors.append(f"/source_intervals/{index}: end_s must exceed start_s")
    units = payload.get("units", "")
    if units and units != "seconds/metres/radians" and not payload.get("unit_transforms"):
        errors.append("/units: non-canonical units require explicit unit_transforms")
    if errors:
        raise ReviewContractsValidationError(errors, source=source)
    known = {"schema_version", "spec_id", "sources"}
    return VisualizationSpec(
        spec_id=str(payload["spec_id"]),
        sources=tuple(dict(source_ref) for source_ref in payload["sources"]),
        document={k: payload[k] for k in payload if k not in known},
    )


def component_descriptor_from_dict(
    payload: Mapping[str, Any], *, source: Any = None
) -> ComponentDescriptor:
    """Validate and build a component descriptor.

    Returns:
        Validated component descriptor.
    """
    _reject_non_finite(payload, source=source)
    _check_envelope_identity_limits(payload, source=source)
    _require_schema(COMPONENT_DESCRIPTOR_SCHEMA_VERSION, payload, source=source)
    return ComponentDescriptor(
        component_id=str(payload["component_id"]),
        component_version=str(payload["component_version"]),
        supported_input_versions=tuple(payload["supported_input_versions"]),
        output_types=tuple(payload["output_types"]),
        required_capabilities=tuple(payload.get("required_capabilities", [])),
        optional_capabilities=tuple(payload.get("optional_capabilities", [])),
    )


def component_request_from_dict(
    payload: Mapping[str, Any], *, source: Any = None
) -> ComponentRequest:
    """Validate and build a component request, rejecting unsafe output paths.

    A source marker identifies persisted or CLI input and enables recursive
    validation of component-owned config.  Source-less in-memory requests
    defer config value validation to the component runner, while envelope
    fields and all source declarations remain strictly validated.

    Returns:
        Validated component request.
    """
    _reject_unicode_surrogates(payload, source=source)
    _reject_non_finite(_request_finite_validation_payload(payload, source=source), source=source)
    _check_envelope_identity_limits(payload, source=source)
    _require_schema(COMPONENT_REQUEST_SCHEMA_VERSION, payload, source=source)
    errors: list[str] = []
    _check_no_traversal(str(payload["output_directory"]), path="/output_directory", errors=errors)
    seen_artifact_ids: set[str] = set()
    for index, ref in enumerate(payload["sources"]):
        artifact_id = ref["artifact_id"]
        _check_safe_artifact_id(artifact_id, path=f"/sources/{index}/artifact_id", errors=errors)
        if artifact_id in seen_artifact_ids:
            errors.append(f"/sources: duplicate scoped artifact id: {artifact_id}")
        seen_artifact_ids.add(artifact_id)
        _check_no_traversal(ref["uri"], path=f"/sources/{index}/uri", errors=errors)
    if errors:
        raise ReviewContractsValidationError(errors, source=source)
    return ComponentRequest(
        request_id=str(payload["request_id"]),
        component_id=str(payload["component_id"]),
        sources=tuple(
            SourceRef(
                artifact_id=str(ref["artifact_id"]),
                uri=str(ref["uri"]),
                format=str(ref["format"]),
                schema=str(ref.get("schema", "")),
                sha256=str(ref.get("sha256", "")),
                source_commit=str(ref.get("source_commit", "")),
                config_identity=str(ref.get("config_identity", "")),
                units=str(ref.get("units", "")),
                coordinate_frame=str(ref.get("coordinate_frame", "")),
            )
            for ref in payload["sources"]
        ),
        output_directory=str(payload["output_directory"]),
        config=dict(payload.get("config", {})),
        required_capabilities=tuple(payload.get("required_capabilities", [])),
    )


def component_result_from_dict(
    payload: Mapping[str, Any], *, source: Any = None
) -> ComponentResult:
    """Validate and build a component result, enforcing status semantics.

    Source-bound result documents are strict.  Source-less internal results
    have oversized diagnostic reasons bounded before validation so stable
    leading reason codes survive the shared result boundary.

    Returns:
        Validated component result.
    """
    _reject_non_finite(payload, source=source)
    payload = _result_validation_payload(payload, source=source)
    _check_envelope_identity_limits(payload, source=source, include_reason=True)
    _require_schema(COMPONENT_RESULT_SCHEMA_VERSION, payload, source=source)
    errors: list[str] = []
    if payload["status"] != "complete" and payload.get("artifacts"):
        errors.append(f"/status: {payload['status']} outputs cannot carry complete status")
    for index, artifact in enumerate(payload.get("artifacts", [])):
        _check_sha256(artifact["sha256"], path=f"/artifacts/{index}/sha256", errors=errors)
        _check_no_traversal(artifact["uri"], path=f"/artifacts/{index}/uri", errors=errors)
    if errors:
        raise ReviewContractsValidationError(errors, source=source)
    return ComponentResult(
        request_id=str(payload["request_id"]),
        component_id=str(payload["component_id"]),
        status=str(payload["status"]),
        artifacts=tuple(dict(artifact) for artifact in payload.get("artifacts", [])),
        diagnostics=tuple(dict(item) for item in payload.get("diagnostics", [])),
        provenance=dict(payload.get("provenance", {})),
        reason=str(payload.get("reason", "")),
    )


def experiment_recipe_from_dict(
    payload: Mapping[str, Any], *, source: Any = None
) -> ExperimentRecipe:
    """Validate and build an experiment recipe.

    Returns:
        Validated experiment recipe.
    """
    _reject_unicode_surrogates(payload, source=source)
    _reject_non_finite(payload, source=source)
    _require_schema(EXPERIMENT_RECIPE_SCHEMA_VERSION, payload, source=source)
    errors: list[str] = []
    seen: set[str] = set()
    for intervention in payload["interventions"]:
        intervention_id = intervention["intervention_id"]
        if intervention_id in seen:
            errors.append(f"/interventions: duplicate scoped id: {intervention_id}")
        seen.add(intervention_id)
    if errors:
        raise ReviewContractsValidationError(errors, source=source)
    return ExperimentRecipe(recipe_id=str(payload["recipe_id"]), document=dict(payload))


def admitted_source_receipt_from_dict(
    payload: Mapping[str, Any], *, source: Any = None
) -> AdmittedSourceReceipt:
    """Validate and build an admitted-source receipt.

    The receipt describes a fixture or diagnostic source only.  It is not an
    evidence-admission decision, and parsing it does not read or trust the
    referenced source bytes.

    Returns:
        Validated admitted-source receipt.

    Raises:
        ReviewContractsValidationError: If the receipt is malformed or does
            not carry the explicit diagnostic-only boundary.
    """
    if not isinstance(payload, Mapping):
        raise ReviewContractsValidationError(["expected a mapping payload"], source=source)
    _reject_unicode_surrogates(payload, source=source)
    _reject_non_finite(payload, source=source)
    _check_admitted_source_input_limits(payload, source=source)
    _require_schema(ADMITTED_SOURCE_RECEIPT_SCHEMA_VERSION, payload, source=source)
    source_payload = payload["source"]
    errors: list[str] = []
    _check_sha256(source_payload["sha256"], path="/source/sha256", errors=errors)
    _check_sha40(source_payload["source_commit"], path="/source/source_commit", errors=errors)
    uri = source_payload["uri"]
    if "\x00" in uri:
        errors.append("/source/uri: NUL bytes are rejected")
    if not uri.strip():
        errors.append("/source/uri: URI must contain non-whitespace text")
    if errors:
        raise ReviewContractsValidationError(errors, source=source)
    return AdmittedSourceReceipt(
        receipt_id=str(payload["receipt_id"]),
        source=AdmittedSourceRef(
            uri=uri,
            format=str(source_payload["format"]),
            schema=str(source_payload["schema"]),
            sha256=str(source_payload["sha256"]).lower(),
            source_commit=str(source_payload["source_commit"]).lower(),
            config_identity=str(source_payload["config_identity"]),
            units=str(source_payload.get("units", "")),
            coordinate_frame=str(source_payload.get("coordinate_frame", "")),
        ),
        request_sha256=str(payload["request_sha256"]).lower(),
        recipe_sha256=str(payload["recipe_sha256"]).lower(),
        status=str(payload["status"]),
        source_kind=str(payload["source_kind"]),
        evidence_boundary=str(payload["evidence_boundary"]),
        scientific_claim_allowed=payload["scientific_claim_allowed"],
    )


def _check_admitted_source_input_limits(payload: Mapping[str, Any], *, source: Any = None) -> None:
    """Reject oversized receipt identity fields before schema traversal."""
    errors: list[str] = []
    receipt_id = payload.get("receipt_id")
    if isinstance(receipt_id, str) and len(receipt_id) > MAX_ADMITTED_SOURCE_RECEIPT_ID_CHARS:
        errors.append(
            f"/receipt_id: exceeds maximum length of {MAX_ADMITTED_SOURCE_RECEIPT_ID_CHARS} characters"
        )
    source_payload = payload.get("source")
    if isinstance(source_payload, Mapping):
        for key, maximum in (
            ("uri", MAX_ADMITTED_SOURCE_URI_CHARS),
            ("format", MAX_ADMITTED_SOURCE_METADATA_CHARS),
            ("schema", MAX_ADMITTED_SOURCE_METADATA_CHARS),
            ("config_identity", MAX_ADMITTED_SOURCE_METADATA_CHARS),
            ("units", MAX_ADMITTED_SOURCE_METADATA_CHARS),
            ("coordinate_frame", MAX_ADMITTED_SOURCE_METADATA_CHARS),
        ):
            value = source_payload.get(key)
            if isinstance(value, str) and len(value) > maximum:
                errors.append(f"/source/{key}: exceeds maximum length of {maximum} characters")
    if errors:
        raise ReviewContractsValidationError(errors, source=source)


def _bounded_error_detail(
    error: BaseException, *, limit: int = MAX_REVIEW_CONTRACT_DIAGNOSTIC_CHARS
) -> str:
    """Return a compact parse or filesystem detail for a stable rejection."""
    if isinstance(error, ReviewContractsValidationError):
        return _bounded_text_detail("; ".join(error.errors), limit=limit)
    return _bounded_text_detail(error, limit=limit)


def _open_regular_file_no_follow(path: Path, *, label: str) -> int:
    """Open a path without following its final component and require a regular file.

    ``O_NONBLOCK`` prevents special files from turning the open into a blocking
    operation.  The descriptor is type-checked before any read, so the bytes
    are obtained from the same file object that was checked.

    Returns:
        An open descriptor owned by the caller.

    Raises:
        _AdmittedSourceFileRejectedError: If safe regular-file support is absent
            or the opened path is not a regular file.
    """
    nofollow = getattr(os, "O_NOFOLLOW", None)
    nonblocking = getattr(os, "O_NONBLOCK", None)
    if not isinstance(nofollow, int) or not isinstance(nonblocking, int):
        raise _AdmittedSourceFileRejectedError(f"{label} cannot be read safely")
    flags = os.O_RDONLY | nofollow | nonblocking | getattr(os, "O_CLOEXEC", 0)
    file_fd = os.open(path, flags)
    try:
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise _AdmittedSourceFileRejectedError(f"{label} must be a regular file")
    except BaseException:
        os.close(file_fd)
        raise
    return file_fd


def _read_bounded_text(path: Path, *, maximum_bytes: int, label: str) -> str:
    """Read UTF-8 text from a bounded, no-follow regular-file descriptor.

    Returns:
        The decoded UTF-8 content.
    """
    file_fd = _open_regular_file_no_follow(path, label=label)
    try:
        content = os.read(file_fd, maximum_bytes + 1)
    finally:
        os.close(file_fd)
    if len(content) > maximum_bytes:
        raise _AdmittedSourceInputLimitError(
            f"{label} exceeds maximum size of {maximum_bytes} bytes"
        )
    return content.decode("utf-8")


def _receipt_read_error_detail(error: BaseException) -> str:
    """Classify parser-limit failures without exposing unbounded input detail.

    Returns:
        A bounded, stable error detail.
    """
    if isinstance(error, _AdmittedSourceInputLimitError):
        return _bounded_error_detail(error)
    if isinstance(error, _AdmittedSourceFileRejectedError):
        return error.detail
    if (
        isinstance(error, (BlockingIOError, OSError))
        and getattr(error, "errno", None) == errno.ELOOP
    ):
        return "path cannot be read safely"
    if isinstance(error, (RecursionError, ValueError)):
        return "receipt JSON is invalid or exceeds parser limits"
    return _bounded_error_detail(error)


def _reject_nonstandard_json_constant(value: str) -> Any:
    """Reject NaN/Infinity tokens accepted by Python's permissive JSON decoder."""
    raise ValueError(f"non-standard JSON constant is not allowed: {value}")


def load_admitted_source_receipt(path: str | Path) -> AdmittedSourceReceipt:
    """Load and validate one JSON admitted-source receipt from *path*.

    Returns:
        Validated admitted-source receipt.

    Raises:
        ReviewContractsValidationError: If the file is unreadable or invalid.
    """
    try:
        receipt_path = Path(path)
    except (OSError, TypeError, ValueError) as error:
        raise ReviewContractsValidationError(
            [f"cannot read admitted-source receipt: {_bounded_error_detail(error)}"]
        ) from error
    try:
        payload = json.loads(
            _read_bounded_text(
                receipt_path,
                maximum_bytes=MAX_ADMITTED_SOURCE_RECEIPT_BYTES,
                label="admitted-source receipt",
            ),
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (OSError, RecursionError, UnicodeError, ValueError) as error:
        raise ReviewContractsValidationError(
            [f"cannot read admitted-source receipt: {_receipt_read_error_detail(error)}"],
            source=receipt_path,
        ) from error
    if not isinstance(payload, Mapping):
        raise ReviewContractsValidationError(
            ["admitted-source receipt must be a JSON object"], source=receipt_path
        )
    try:
        return admitted_source_receipt_from_dict(payload, source=receipt_path)
    except RecursionError as error:
        raise ReviewContractsValidationError(
            [f"cannot validate admitted-source receipt: {_receipt_read_error_detail(error)}"],
            source=receipt_path,
        ) from error


def component_request_canonical_digest(
    request: ComponentRequest | Mapping[str, Any],
) -> str:
    """Hash the location-independent identity of a validated component request.

    Returns:
        Hex SHA-256 digest of the request identity.
    """
    return _canonical_digest(_request_identity_document(request))


def experiment_recipe_canonical_digest(
    recipe: ExperimentRecipe | Mapping[str, Any],
) -> str:
    """Hash the canonical JSON document of a validated experiment recipe.

    Returns:
        Hex SHA-256 digest of the recipe document.
    """
    if isinstance(recipe, ExperimentRecipe):
        document = recipe.document
        _reject_non_finite(document)
    elif isinstance(recipe, Mapping):
        document = experiment_recipe_from_dict(recipe).document
    else:
        raise TypeError("recipe must be an experiment recipe mapping or ExperimentRecipe")
    return _canonical_digest(document)


def _recipe_identity_value(
    recipe: ExperimentRecipe | Mapping[str, Any] | None,
    key: str,
) -> Any:
    """Read an optional expected source identity from a recipe document.

    Returns:
        The requested identity value, or ``None`` when it is absent.
    """
    if recipe is None:
        return None
    document = recipe.document if isinstance(recipe, ExperimentRecipe) else recipe
    if not isinstance(document, Mapping):
        return None
    source_identity = document.get("source_identity")
    if isinstance(source_identity, Mapping):
        return source_identity.get(key)
    return None


def _resolution(
    status: str,
    reason: str,
    *,
    receipt: AdmittedSourceReceipt | None = None,
    source_path: Path | None = None,
    detail: str = "",
    source_bytes: bytes | None = None,
) -> AdmittedSourceResolution:
    """Build a resolution result with a stable reason and optional detail.

    Returns:
        Resolution with no source path or bytes unless explicitly supplied.
    """
    return AdmittedSourceResolution(
        status=status,
        reason=reason,
        source_path=source_path,
        receipt=receipt,
        detail=_bounded_text_detail(detail) if detail else "",
        source_bytes=source_bytes,
    )


def _receipt_payload(
    receipt: AdmittedSourceReceipt | Mapping[str, Any] | str | Path,
) -> tuple[Mapping[str, Any] | None, AdmittedSourceResolution | None]:
    """Load a receipt input without validating its contract.

    Returns:
        A raw mapping and no early result, or a fail-closed result for a path
        that cannot be loaded.
    """
    if isinstance(receipt, AdmittedSourceReceipt):
        return receipt.to_dict(), None
    if isinstance(receipt, Mapping):
        return receipt, None
    try:
        receipt_path = Path(receipt)
    except (OSError, TypeError, ValueError) as error:
        return None, _resolution(
            "unavailable",
            ADMITTED_SOURCE_REASON_RECEIPT_UNREADABLE,
            detail=_bounded_error_detail(error),
        )
    try:
        raw_payload = json.loads(
            _read_bounded_text(
                receipt_path,
                maximum_bytes=MAX_ADMITTED_SOURCE_RECEIPT_BYTES,
                label="admitted-source receipt",
            ),
            parse_constant=_reject_nonstandard_json_constant,
        )
    except FileNotFoundError:
        return None, _resolution("unavailable", ADMITTED_SOURCE_REASON_RECEIPT_MISSING)
    except (OSError, RecursionError, UnicodeError, ValueError) as error:
        return None, _resolution(
            "unavailable",
            ADMITTED_SOURCE_REASON_RECEIPT_UNREADABLE,
            detail=_receipt_read_error_detail(error),
        )
    if not isinstance(raw_payload, Mapping):
        return None, _resolution(
            "failed",
            ADMITTED_SOURCE_REASON_RECEIPT_MALFORMED,
            detail="receipt must be a JSON object",
        )
    return raw_payload, None


def _parse_admitted_source_receipt(
    payload: Mapping[str, Any],
) -> tuple[AdmittedSourceReceipt | None, AdmittedSourceResolution | None]:
    """Validate the receipt schema and its explicit diagnostic boundary.

    Returns:
        A parsed receipt and no early result, or a stable rejection result.
    """
    schema_version = payload.get("schema_version")
    if schema_version is None:
        return None, _resolution(
            "failed",
            ADMITTED_SOURCE_REASON_RECEIPT_MALFORMED,
            detail="schema_version is missing",
        )
    if not isinstance(schema_version, str):
        return None, _resolution(
            "failed",
            ADMITTED_SOURCE_REASON_RECEIPT_MALFORMED,
            detail="schema_version must be a string",
        )
    if schema_version != ADMITTED_SOURCE_RECEIPT_SCHEMA_VERSION:
        return None, _resolution(
            "unavailable",
            ADMITTED_SOURCE_REASON_RECEIPT_UNSUPPORTED,
            detail="schema version is unsupported",
        )
    boundary_fields = (
        "status",
        "source_kind",
        "evidence_boundary",
        "scientific_claim_allowed",
    )
    missing_boundary_fields = [field for field in boundary_fields if field not in payload]
    if missing_boundary_fields:
        return None, _resolution(
            "failed",
            ADMITTED_SOURCE_REASON_RECEIPT_MALFORMED,
            detail="required boundary fields are missing: " + ", ".join(missing_boundary_fields),
        )
    source_kind = payload.get("source_kind")
    if not isinstance(source_kind, str):
        return None, _resolution(
            "failed",
            ADMITTED_SOURCE_REASON_RECEIPT_MALFORMED,
            detail="source_kind must be a string",
        )
    if (
        any(
            payload.get(field) != expected
            for field, expected in (
                ("status", ADMITTED_SOURCE_STATUS),
                ("evidence_boundary", ADMITTED_SOURCE_EVIDENCE_BOUNDARY),
                ("scientific_claim_allowed", False),
            )
        )
        or source_kind not in ADMITTED_SOURCE_KINDS
    ):
        return None, _resolution(
            "unavailable",
            ADMITTED_SOURCE_REASON_RECEIPT_UNSUPPORTED,
            detail="receipt is outside the fixture/diagnostic-only boundary",
        )
    try:
        return admitted_source_receipt_from_dict(payload), None
    except ReviewContractsValidationError as error:
        return None, _resolution(
            "failed",
            ADMITTED_SOURCE_REASON_RECEIPT_MALFORMED,
            detail=_bounded_error_detail(error),
        )


def _parse_admitted_source_receipt_safely(
    payload: Mapping[str, Any],
) -> tuple[AdmittedSourceReceipt | None, AdmittedSourceResolution | None]:
    """Convert parser recursion into the stable malformed-receipt result.

    Returns:
        A parsed receipt and no early result, or a bounded malformed result.
    """
    try:
        return _parse_admitted_source_receipt(payload)
    except RecursionError as error:
        return None, _resolution(
            "failed",
            ADMITTED_SOURCE_REASON_RECEIPT_MALFORMED,
            detail=_receipt_read_error_detail(error),
        )


def _validated_request(
    request: ComponentRequest | Mapping[str, Any],
) -> ComponentRequest:
    """Return a validated request object for identity and source binding.

    Returns:
        Validated component request.
    """
    if isinstance(request, ComponentRequest):
        _reject_non_finite(asdict(request))
        _check_envelope_identity_limits(
            {"request_id": request.request_id, "component_id": request.component_id}
        )
        return request
    if isinstance(request, Mapping):
        return component_request_from_dict(request)
    raise TypeError("request must be a component request mapping or ComponentRequest")


_COMPONENT_REQUEST_SOURCE_DECLARATION_FIELDS = (
    "artifact_id",
    "uri",
    "format",
    "schema",
    "sha256",
    "source_commit",
    "config_identity",
    "units",
    "coordinate_frame",
)
_REQUEST_SOURCE_RECEIPT_BINDING_FIELDS = (
    "uri",
    "format",
    "schema",
    "sha256",
    "source_commit",
    "config_identity",
    "units",
    "coordinate_frame",
)
_SOURCE_SEMANTIC_BINDING_FIELDS = ("units", "coordinate_frame")


def _normalize_v1_source_declaration(value: Any) -> str:
    """Normalize an optional v1 source declaration to its canonical text form.

    Returns:
        The source declaration or an empty string for an absent/non-text value.
    """
    return value if isinstance(value, str) else ""


def _component_request_source_identity(source: SourceRef) -> dict[str, str]:
    """Return every v1 source declaration, normalizing absent optionals."""
    return {
        field: _normalize_v1_source_declaration(getattr(source, field, ""))
        for field in _COMPONENT_REQUEST_SOURCE_DECLARATION_FIELDS
    }


def _request_source_binding_mismatch(
    source: SourceRef,
    receipt_source: AdmittedSourceRef,
) -> str | None:
    """Return the first supplied request declaration that differs from a receipt.

    Returns:
        The mismatching declaration name, or ``None`` when all supplied values match.
    """
    for field_name in _REQUEST_SOURCE_RECEIPT_BINDING_FIELDS:
        current = _normalize_v1_source_declaration(getattr(source, field_name, ""))
        expected = _normalize_v1_source_declaration(getattr(receipt_source, field_name, ""))
        if field_name in {"sha256", "source_commit"}:
            current = current.lower()
            expected = expected.lower()
        if field_name in _SOURCE_SEMANTIC_BINDING_FIELDS:
            if current != expected:
                return field_name
        elif (field_name in {"uri", "format"} or current) and current != expected:
            return field_name
    return None


def _request_identity_document(request: ComponentRequest | Mapping[str, Any]) -> dict[str, Any]:
    """Return the location-independent request identity used by SREV-22."""
    validated = _validated_request(request)
    return {
        "schema_version": COMPONENT_REQUEST_SCHEMA_VERSION,
        "request_id": validated.request_id,
        "component_id": validated.component_id,
        "sources": [_component_request_source_identity(source) for source in validated.sources],
        "config": dict(validated.config),
        "required_capabilities": list(validated.required_capabilities),
    }


def _check_one_digest_binding(
    receipt: AdmittedSourceReceipt,
    *,
    name: str,
    receipt_digest: str,
    expected_digest: str | None,
    payload: Any,
    compute: Callable[[Any], str],
) -> tuple[str | None, AdmittedSourceResolution | None]:
    """Compute and compare one request or recipe digest.

    Returns:
        The effective digest and no early result, or a stable stale result.
    """
    effective_digest = expected_digest
    try:
        if payload is not None:
            computed_digest = compute(payload)
            if expected_digest is not None and (
                not isinstance(expected_digest, str) or expected_digest.lower() != computed_digest
            ):
                return None, _resolution(
                    "unavailable",
                    ADMITTED_SOURCE_REASON_RECEIPT_STALE,
                    receipt=receipt,
                    detail=f"supplied {name} digest does not match the {name} payload",
                )
            effective_digest = computed_digest
    except (RecursionError, ReviewContractsValidationError, TypeError, ValueError) as error:
        return None, _resolution(
            "unavailable",
            ADMITTED_SOURCE_REASON_RECEIPT_STALE,
            receipt=receipt,
            detail=f"{name} identity is unusable: {_bounded_error_detail(error)}",
        )
    if effective_digest is None:
        return None, _resolution(
            "unavailable",
            ADMITTED_SOURCE_REASON_RECEIPT_STALE,
            receipt=receipt,
            detail=f"{name} identity is required",
        )
    if (
        not isinstance(effective_digest, str)
        or _SHA256_RE.fullmatch(effective_digest) is None
        or effective_digest.lower() != receipt_digest
    ):
        return None, _resolution(
            "unavailable",
            ADMITTED_SOURCE_REASON_RECEIPT_STALE,
            receipt=receipt,
            detail=f"{name} SHA-256 does not match the receipt",
        )
    return effective_digest, None


def _check_receipt_digest_bindings(
    receipt: AdmittedSourceReceipt,
    *,
    request: ComponentRequest | Mapping[str, Any] | None,
    recipe: ExperimentRecipe | Mapping[str, Any] | None,
    expected_request_sha256: str | None,
    expected_recipe_sha256: str | None,
) -> AdmittedSourceResolution | None:
    """Check request and recipe digests and the request source binding.

    Returns:
        A stale resolution on mismatch, or ``None`` when all bindings pass.
    """
    if request is None or recipe is None:
        return _resolution(
            "unavailable",
            ADMITTED_SOURCE_REASON_RECEIPT_STALE,
            receipt=receipt,
            detail="current request and recipe context are required for admission",
        )
    _, early = _check_one_digest_binding(
        receipt,
        name="request",
        receipt_digest=receipt.request_sha256,
        expected_digest=expected_request_sha256,
        payload=request,
        compute=component_request_canonical_digest,
    )
    if early is not None:
        return early
    _, early = _check_one_digest_binding(
        receipt,
        name="recipe",
        receipt_digest=receipt.recipe_sha256,
        expected_digest=expected_recipe_sha256,
        payload=recipe,
        compute=experiment_recipe_canonical_digest,
    )
    if early is not None:
        return early
    validated_request = _validated_request(request)
    source_mismatch: str | None = None
    matching_source_found = False
    for source in validated_request.sources:
        if source.uri != receipt.source.uri or source.format != receipt.source.format:
            continue
        matching_source_found = True
        current_mismatch = _request_source_binding_mismatch(source, receipt.source)
        if current_mismatch is None:
            return None
        if source_mismatch is None:
            source_mismatch = current_mismatch
    if source_mismatch is not None:
        return _resolution(
            "unavailable",
            ADMITTED_SOURCE_REASON_RECEIPT_STALE,
            receipt=receipt,
            detail=f"request source {source_mismatch} does not match the receipt",
        )
    if not matching_source_found:
        return _resolution(
            "unavailable",
            ADMITTED_SOURCE_REASON_RECEIPT_STALE,
            receipt=receipt,
            detail="receipt URI and format are not present in the request",
        )
    return None


def _check_source_identity_bindings(
    receipt: AdmittedSourceReceipt,
    *,
    recipe: ExperimentRecipe | Mapping[str, Any] | None,
    expected_source_commit: str | None,
    expected_config_identity: str | None,
) -> AdmittedSourceResolution | None:
    """Require source commit and config identities to match the receipt.

    Returns:
        A stale resolution on mismatch or missing expectations, or ``None``.
    """
    recipe_source = _recipe_identity_value(recipe, "source_commit")
    if expected_source_commit is not None and (
        not isinstance(expected_source_commit, str)
        or not isinstance(recipe_source, str)
        or expected_source_commit.lower() != recipe_source.lower()
    ):
        return _resolution(
            "unavailable",
            ADMITTED_SOURCE_REASON_RECEIPT_STALE,
            receipt=receipt,
            detail="explicit source commit identity does not match the recipe",
        )
    expected_source = recipe_source
    if not isinstance(expected_source, str) or _SHA40_RE.fullmatch(expected_source) is None:
        return _resolution(
            "unavailable",
            ADMITTED_SOURCE_REASON_RECEIPT_STALE,
            receipt=receipt,
            detail="source commit identity is missing or not a commit SHA; migrate the v1 recipe",
        )
    if expected_source.lower() != receipt.source.source_commit:
        return _resolution(
            "unavailable",
            ADMITTED_SOURCE_REASON_RECEIPT_STALE,
            receipt=receipt,
            detail="source commit identity does not match the receipt",
        )
    recipe_config = _recipe_identity_value(recipe, "config_identity")
    if expected_config_identity is not None and expected_config_identity != recipe_config:
        return _resolution(
            "unavailable",
            ADMITTED_SOURCE_REASON_RECEIPT_STALE,
            receipt=receipt,
            detail="explicit config identity does not match the recipe",
        )
    expected_config = recipe_config
    if not isinstance(expected_config, str) or not expected_config.strip():
        return _resolution(
            "unavailable",
            ADMITTED_SOURCE_REASON_RECEIPT_STALE,
            receipt=receipt,
            detail="config identity is missing or empty; migrate the v1 recipe",
        )
    if expected_config != receipt.source.config_identity:
        return _resolution(
            "unavailable",
            ADMITTED_SOURCE_REASON_RECEIPT_STALE,
            receipt=receipt,
            detail="config identity does not match the receipt",
        )
    return None


def _check_recipe_semantic_bindings(
    receipt: AdmittedSourceReceipt,
    *,
    recipe: ExperimentRecipe | Mapping[str, Any],
) -> AdmittedSourceResolution | None:
    """Require exact recipe bindings for units and coordinate frame.

    Returns:
        A stale resolution on a missing, non-text, or conflicting binding.
    """
    for key in _SOURCE_SEMANTIC_BINDING_FIELDS:
        expected = _recipe_identity_value(recipe, key)
        if expected is None:
            expected = ""
        elif not isinstance(expected, str):
            return _resolution(
                "unavailable",
                ADMITTED_SOURCE_REASON_RECEIPT_STALE,
                receipt=receipt,
                detail=f"recipe source_identity.{key} must be a string",
            )
        if expected != getattr(receipt.source, key):
            return _resolution(
                "unavailable",
                ADMITTED_SOURCE_REASON_RECEIPT_STALE,
                receipt=receipt,
                detail=f"recipe source_identity.{key} does not match the receipt",
            )
    return None


def _check_recipe_source_bindings(
    receipt: AdmittedSourceReceipt,
    *,
    recipe: ExperimentRecipe | Mapping[str, Any] | None,
) -> AdmittedSourceResolution | None:
    """Check source metadata and the recipe's admission reference.

    Returns:
        A stale resolution on missing or conflicting recipe metadata, or
        ``None`` when the recipe carries no conflicting value.
    """
    if recipe is None:
        return _resolution(
            "unavailable",
            ADMITTED_SOURCE_REASON_RECEIPT_STALE,
            receipt=receipt,
            detail="current recipe context is required for admission",
        )
    document = recipe.document if isinstance(recipe, ExperimentRecipe) else recipe
    if not isinstance(document, Mapping):
        return _resolution(
            "unavailable",
            ADMITTED_SOURCE_REASON_RECEIPT_STALE,
            receipt=receipt,
            detail="recipe document is not a mapping",
        )
    admission_reference = document.get("admission_reference")
    if not isinstance(admission_reference, str) or not admission_reference.strip():
        return _resolution(
            "unavailable",
            ADMITTED_SOURCE_REASON_RECEIPT_STALE,
            receipt=receipt,
            detail="recipe admission_reference is required; migrate the v1 recipe",
        )
    if admission_reference != receipt.receipt_id:
        return _resolution(
            "unavailable",
            ADMITTED_SOURCE_REASON_RECEIPT_STALE,
            receipt=receipt,
            detail="receipt ID does not match recipe admission_reference",
        )
    source_identity = document.get("source_identity")
    if not isinstance(source_identity, Mapping):
        return _resolution(
            "unavailable",
            ADMITTED_SOURCE_REASON_RECEIPT_STALE,
            receipt=receipt,
            detail="recipe source_identity is required for admission",
        )
    boundary_result = _check_recipe_boundary_bindings(receipt, source_identity)
    if boundary_result is not None:
        return boundary_result
    for key, actual in (
        ("source_uri", receipt.source.uri),
        ("source_format", receipt.source.format),
        ("source_schema", receipt.source.schema),
    ):
        expected = _recipe_identity_value(recipe, key)
        if expected is not None and expected != actual:
            return _resolution(
                "unavailable",
                ADMITTED_SOURCE_REASON_RECEIPT_STALE,
                receipt=receipt,
                detail=f"recipe {key} does not match the receipt",
            )
    return _check_recipe_semantic_bindings(receipt, recipe=recipe)


def _check_recipe_boundary_bindings(
    receipt: AdmittedSourceReceipt,
    source_identity: Mapping[str, Any],
) -> AdmittedSourceResolution | None:
    """Require a recipe to preserve the receipt's diagnostic-only boundary.

    Returns:
        A stale resolution on a boundary mismatch, or ``None``.
    """
    recipe_boundary = (
        ("kind", receipt.source_kind),
        ("evidence_boundary", ADMITTED_SOURCE_EVIDENCE_BOUNDARY),
        ("scientific_claim_allowed", False),
        ("dependent_family_status", ADMITTED_SOURCE_DEPENDENT_FAMILY_STATUS),
    )
    for key, expected in recipe_boundary:
        actual = source_identity.get(key)
        if key == "kind":
            boundary_matches = isinstance(actual, str) and actual in ADMITTED_SOURCE_KINDS
            boundary_matches = boundary_matches and actual == expected
        elif key == "scientific_claim_allowed":
            boundary_matches = actual is expected
        else:
            boundary_matches = actual == expected
        if not boundary_matches:
            return _resolution(
                "unavailable",
                ADMITTED_SOURCE_REASON_RECEIPT_STALE,
                receipt=receipt,
                detail=f"recipe source_identity.{key} does not preserve the diagnostic-only boundary",
            )
    return None


def _source_open_flags(*, directory: bool) -> int | None:
    """Return no-follow descriptor flags or ``None`` when safe open is unavailable."""
    nofollow = getattr(os, "O_NOFOLLOW", None)
    directory_flag = getattr(os, "O_DIRECTORY", None)
    nonblocking = getattr(os, "O_NONBLOCK", None)
    if (
        not isinstance(nofollow, int)
        or not isinstance(nonblocking, int)
        or (directory and not isinstance(directory_flag, int))
        or not _ADMITTED_SOURCE_SUPPORTS_DIR_FD
    ):
        return None
    flags = os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0)
    if directory:
        flags |= directory_flag
    else:
        flags |= nonblocking
    return flags


def _source_open_rejection(
    receipt: AdmittedSourceReceipt,
    error: BaseException,
) -> AdmittedSourceResolution:
    """Convert a protected-open failure into a stable source resolution.

    Returns:
        A bounded unavailable resolution classified by the open failure.
    """
    if getattr(error, "errno", None) == errno.ELOOP:
        reason = ADMITTED_SOURCE_REASON_SOURCE_ESCAPED_ROOT
    elif getattr(error, "errno", None) in {errno.ENOTDIR, errno.EISDIR}:
        reason = ADMITTED_SOURCE_REASON_SOURCE_NOT_REGULAR
    else:
        reason = ADMITTED_SOURCE_REASON_SOURCE_MISSING
    return _resolution(
        "unavailable",
        reason,
        receipt=receipt,
        detail=_bounded_error_detail(error),
    )


def _read_source_fd_bounded(fd: int) -> tuple[str, bytes]:
    """Hash and retain bounded bytes from one already-open regular-file descriptor.

    Returns:
        The lower-case SHA-256 digest and the exact bytes read from ``fd``.
    """
    digest = hashlib.sha256()
    content = bytearray()
    bytes_read = 0
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        bytes_read += len(chunk)
        if bytes_read > MAX_ADMITTED_SOURCE_BYTES:
            raise _AdmittedSourceInputLimitError(
                f"source exceeds maximum size of {MAX_ADMITTED_SOURCE_BYTES} bytes"
            )
        digest.update(chunk)
        content.extend(chunk)
    return digest.hexdigest(), bytes(content)


def _close_source_fds(fds: list[int]) -> None:
    """Close source descriptors while suppressing cleanup-only operating errors."""
    for fd in reversed(fds):
        try:
            os.close(fd)
        except OSError:
            pass


def _open_source_fds(
    root: Path,
    parts: tuple[str, ...],
    *,
    directory_flags: int,
    file_flags: int,
    root_fd: int | None = None,
) -> tuple[int, list[int]]:
    """Open every source component relative to no-follow directory descriptors.

    Returns:
        The source descriptor and all descriptors that the caller must close.
    """
    opened_fds: list[int] = []
    try:
        if root_fd is None:
            opened_root_fd = os.open(root, directory_flags)
        else:
            opened_root_fd = os.dup(root_fd)
            if not stat.S_ISDIR(os.fstat(opened_root_fd).st_mode):
                raise OSError(errno.ENOTDIR, "pinned allowed root is not a directory")
        opened_fds.append(opened_root_fd)
        parent_fd = opened_root_fd
        for component in parts[:-1]:
            parent_fd = os.open(component, directory_flags, dir_fd=parent_fd)
            opened_fds.append(parent_fd)
        source_fd = os.open(parts[-1], file_flags, dir_fd=parent_fd)
        opened_fds.append(source_fd)
        return source_fd, opened_fds
    except BaseException:
        _close_source_fds(opened_fds)
        raise


def _read_source_or_rejection(
    receipt: AdmittedSourceReceipt,
    root: Path,
    *,
    root_fd: int | None = None,
) -> tuple[str, bytes] | AdmittedSourceResolution:
    """Open, type-check, hash, and retain one source under a descriptor boundary.

    Every path component is opened relative to a no-follow directory descriptor.
    The returned bytes and digest therefore come from the same regular-file
    descriptor, not from a pathname that can be replaced between checks.

    Returns:
        A ``(sha256, bytes)`` pair or a stable unavailable resolution.
    """
    directory_flags = _source_open_flags(directory=True)
    file_flags = _source_open_flags(directory=False)
    if directory_flags is None or file_flags is None:
        return _resolution(
            "unavailable",
            ADMITTED_SOURCE_REASON_SOURCE_PROTECTION_UNAVAILABLE,
            receipt=receipt,
            detail="platform lacks required no-follow descriptor support",
        )
    parts = Path(receipt.source.uri).parts
    if not parts:
        return _resolution(
            "unavailable",
            ADMITTED_SOURCE_REASON_SOURCE_NOT_REGULAR,
            receipt=receipt,
            detail="source URI does not identify a regular file",
        )

    opened_fds: list[int] = []
    source_fd: int | None = None
    try:
        source_fd, opened_fds = _open_source_fds(
            root,
            parts,
            directory_flags=directory_flags,
            file_flags=file_flags,
            root_fd=root_fd,
        )
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode):
            return _resolution(
                "unavailable",
                ADMITTED_SOURCE_REASON_SOURCE_NOT_REGULAR,
                receipt=receipt,
            )
        return _read_source_fd_bounded(source_fd)
    except _AdmittedSourceInputLimitError as error:
        return _resolution(
            "unavailable",
            ADMITTED_SOURCE_REASON_SOURCE_TOO_LARGE,
            receipt=receipt,
            detail=_bounded_error_detail(error),
        )
    except (OSError, ValueError, TypeError) as error:
        return _source_open_rejection(receipt, error)
    finally:
        _close_source_fds(opened_fds)


def _resolve_allowed_root(
    allowed_root: str | Path,
    *,
    allowed_root_fd: int | None = None,
) -> tuple[Path | None, AdmittedSourceResolution | None]:
    """Resolve and require an existing directory trust boundary.

    Returns:
        The canonical root and no early result, or a stable root rejection.
    """
    if allowed_root_fd is not None:
        try:
            if not stat.S_ISDIR(os.fstat(allowed_root_fd).st_mode):
                raise OSError(errno.ENOTDIR, "pinned allowed root is not a directory")
            root = Path(allowed_root)
        except (OSError, TypeError, ValueError) as error:
            return None, _resolution(
                "unavailable",
                ADMITTED_SOURCE_REASON_ALLOWED_ROOT_INVALID,
                detail=_bounded_error_detail(error),
            )
        return root, None
    try:
        if isinstance(allowed_root, str) and not allowed_root.strip():
            raise ValueError("allowed root must contain non-whitespace text")
        root = Path(allowed_root).resolve(strict=True)
    except FileNotFoundError:
        return None, _resolution("unavailable", ADMITTED_SOURCE_REASON_ALLOWED_ROOT_MISSING)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        return None, _resolution(
            "unavailable",
            ADMITTED_SOURCE_REASON_ALLOWED_ROOT_INVALID,
            detail=_bounded_error_detail(error),
        )
    if not root.is_dir():
        return None, _resolution("unavailable", ADMITTED_SOURCE_REASON_ALLOWED_ROOT_INVALID)
    return root, None


def _resolve_source_path(
    receipt: AdmittedSourceReceipt,
    root: Path,
    *,
    descriptor_bound: bool = False,
) -> tuple[Path | None, AdmittedSourceResolution | None]:
    """Validate a receipt URI beneath *root* before descriptor-bound source use.

    Returns:
        A canonical display path and no early result, or a stable path rejection.
    """
    uri = receipt.source.uri
    try:
        uri_path = Path(uri)
        uri_info = urlsplit(uri)
    except (OSError, TypeError, ValueError) as error:
        return None, _resolution(
            "unavailable",
            ADMITTED_SOURCE_REASON_RECEIPT_UNSUPPORTED,
            receipt=receipt,
            detail=(f"source URI cannot be interpreted locally: {_bounded_error_detail(error)}"),
        )
    if uri_info.scheme or uri_info.netloc or uri_info.query or uri_info.fragment:
        return None, _resolution(
            "unavailable",
            ADMITTED_SOURCE_REASON_RECEIPT_UNSUPPORTED,
            receipt=receipt,
            detail="source URI must be a relative path without a URI scheme",
        )
    if "\\" in uri or uri_path.is_absolute() or ".." in uri_path.parts:
        return None, _resolution(
            "unavailable",
            ADMITTED_SOURCE_REASON_SOURCE_ESCAPED_ROOT,
            receipt=receipt,
            detail="source URI is absolute, traverses its root, or uses a platform separator",
        )
    candidate = root.joinpath(uri_path)
    if descriptor_bound:
        return candidate, None
    try:
        resolved = candidate.resolve(strict=False)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as error:
        return None, _resolution(
            "unavailable",
            ADMITTED_SOURCE_REASON_SOURCE_ESCAPED_ROOT,
            receipt=receipt,
            detail=_bounded_error_detail(error),
        )
    return resolved, None


def resolve_admitted_source(  # noqa: C901, PLR0913
    receipt: AdmittedSourceReceipt | Mapping[str, Any] | str | Path,
    *,
    allowed_root: str | Path,
    request: ComponentRequest | Mapping[str, Any] | None = None,
    recipe: ExperimentRecipe | Mapping[str, Any] | None = None,
    expected_request_sha256: str | None = None,
    expected_recipe_sha256: str | None = None,
    expected_source_commit: str | None = None,
    expected_config_identity: str | None = None,
    allowed_root_fd: int | None = None,
) -> AdmittedSourceResolution:
    """Resolve a receipt only after all source and identity checks pass.

    ``allowed_root`` is an explicit caller-owned trust boundary.  The receipt
    URI is interpreted as a relative local path beneath that root; remote URIs,
    absolute paths, traversal, and symlink escapes are rejected.  Current
    request and recipe documents are required.  Their canonical digests,
    URI/format binding, admission reference, and source commit/config identity
    are checked against the receipt.  Optional expected identity arguments may
    corroborate those documents but cannot replace them.

    Returns:
        ``status='admitted'`` with protected source bytes and a compatibility
        display path only after the current source bytes match the receipt.
        All rejection results omit ``source_path`` and ``source_bytes``.
    """
    raw_receipt, early = _receipt_payload(receipt)
    raw_or_result = _resolver_value_or_rejection(
        raw_receipt,
        early,
        status="failed",
        reason=ADMITTED_SOURCE_REASON_RECEIPT_MALFORMED,
        detail="receipt payload resolver returned no payload",
    )
    if isinstance(raw_or_result, AdmittedSourceResolution):
        return raw_or_result
    parsed_receipt, early = _parse_admitted_source_receipt_safely(raw_or_result)
    parsed_or_result = _resolver_value_or_rejection(
        parsed_receipt,
        early,
        status="failed",
        reason=ADMITTED_SOURCE_REASON_RECEIPT_MALFORMED,
        detail="receipt parser returned no receipt",
    )
    if isinstance(parsed_or_result, AdmittedSourceResolution):
        return parsed_or_result
    parsed_receipt = parsed_or_result
    early = _check_receipt_digest_bindings(
        parsed_receipt,
        request=request,
        recipe=recipe,
        expected_request_sha256=expected_request_sha256,
        expected_recipe_sha256=expected_recipe_sha256,
    )
    if early is not None:
        return early
    early = _check_recipe_source_bindings(parsed_receipt, recipe=recipe)
    if early is not None:
        return early
    early = _check_source_identity_bindings(
        parsed_receipt,
        recipe=recipe,
        expected_source_commit=expected_source_commit,
        expected_config_identity=expected_config_identity,
    )
    if early is not None:
        return early
    if allowed_root_fd is None:
        root, early = _resolve_allowed_root(allowed_root)
    else:
        root, early = _resolve_allowed_root(allowed_root, allowed_root_fd=allowed_root_fd)
    root_or_result = _resolver_value_or_rejection(
        root,
        early,
        status="unavailable",
        reason=ADMITTED_SOURCE_REASON_ALLOWED_ROOT_INVALID,
        detail="allowed-root resolver returned no path",
    )
    if isinstance(root_or_result, AdmittedSourceResolution):
        return root_or_result
    if allowed_root_fd is None:
        source_path, early = _resolve_source_path(parsed_receipt, root_or_result)
    else:
        source_path, early = _resolve_source_path(
            parsed_receipt,
            root_or_result,
            descriptor_bound=True,
        )
    source_or_result = _resolver_value_or_rejection(
        source_path,
        early,
        status="unavailable",
        reason=ADMITTED_SOURCE_REASON_SOURCE_MISSING,
        receipt=parsed_receipt,
        detail="source-path resolver returned no path",
    )
    if isinstance(source_or_result, AdmittedSourceResolution):
        return source_or_result
    source_path = source_or_result
    if allowed_root_fd is None:
        observed_or_result = _read_source_or_rejection(parsed_receipt, root_or_result)
    else:
        observed_or_result = _read_source_or_rejection(
            parsed_receipt,
            root_or_result,
            root_fd=allowed_root_fd,
        )
    if isinstance(observed_or_result, AdmittedSourceResolution):
        return observed_or_result
    observed_sha256, source_bytes = observed_or_result
    if observed_sha256 != parsed_receipt.source.sha256:
        return _resolution(
            "failed",
            ADMITTED_SOURCE_REASON_SOURCE_MUTATED,
            receipt=parsed_receipt,
            detail=f"expected {parsed_receipt.source.sha256}, observed {observed_sha256}",
        )
    return _resolution(
        ADMITTED_SOURCE_STATUS,
        ADMITTED_SOURCE_STATUS,
        receipt=parsed_receipt,
        source_path=source_path,
        source_bytes=source_bytes,
    )


def review_bundle_canonical_digest(bundle: ReviewBundle) -> str:
    """Bind logical bundle content with a stable digest (location-independent).

    Returns:
        Hex SHA-256 digest of the canonical encoding.
    """
    return _canonical_digest(
        {
            "schema_version": REVIEW_BUNDLE_SCHEMA_VERSION,
            "bundle_id": bundle.bundle_id,
            "episodes": [dict(episode) for episode in bundle.episodes],
        }
    )


def _canonical_digest(value: Any) -> str:
    _reject_non_finite(value)
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


INSPECT_COMPONENT_ID = "srev01-inspect"
CAPABILITY_COMPONENT_ID = "srev01-capability-report"

_INSPECT_DESCRIPTOR = ComponentDescriptor(
    component_id=INSPECT_COMPONENT_ID,
    component_version="1.0.0",
    supported_input_versions=(COMPONENT_REQUEST_SCHEMA_VERSION,),
    output_types=("inspect-report.v1",),
)

_CAPABILITY_DESCRIPTOR = ComponentDescriptor(
    component_id=CAPABILITY_COMPONENT_ID,
    component_version="1.0.0",
    supported_input_versions=(COMPONENT_REQUEST_SCHEMA_VERSION,),
    output_types=("capability-report.v1",),
    optional_capabilities=("video-frames",),
)


def capability_report() -> dict[str, Any]:
    """Describe the components this module executes offline without AI or Rerun.

    Returns:
        Capability report document.
    """
    return {
        "schema_version": "capability-report.v1",
        "components": [asdict(_INSPECT_DESCRIPTOR), asdict(_CAPABILITY_DESCRIPTOR)],
    }


def _resolve_output_dir(request: ComponentRequest, base: Path) -> Path:
    output_dir = base / request.output_directory
    if output_dir.exists():
        raise ReviewContractsValidationError(
            [
                "/output_directory: output collision, already exists: "
                f"{_bounded_text_detail(request.output_directory)}"
            ]
        )
    return output_dir


def _unsupported_capabilities(
    request: ComponentRequest, descriptor: ComponentDescriptor
) -> list[str]:
    supported = set(descriptor.required_capabilities) | set(descriptor.optional_capabilities)
    return [name for name in request.required_capabilities if name not in supported]


def _write_json(path: Path, payload: Any) -> str:
    text = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run(request: ComponentRequest, *, base: Path | None = None) -> ComponentResult:
    """Execute one inspect or capability-report request.

    Args:
        request: Validated component request.
        base: Base directory the request output directory resolves under.

    Returns:
        Component result with artifacts, diagnostics, and provenance.
    """
    root = base if base is not None else Path.cwd()
    request_id = _safe_identity(request.request_id)
    component_id = _safe_identity(request.component_id)
    try:
        output_dir = _resolve_output_dir(request, root)
        if request.component_id == INSPECT_COMPONENT_ID:
            descriptor = _INSPECT_DESCRIPTOR
        elif request.component_id == CAPABILITY_COMPONENT_ID:
            descriptor = _CAPABILITY_DESCRIPTOR
        else:
            return ComponentResult(
                request_id=request_id,
                component_id=component_id,
                status="unavailable",
                reason=_bounded_text_detail(f"unsupported component: {component_id}"),
            )
        missing = _unsupported_capabilities(request, descriptor)
        if missing:
            return ComponentResult(
                request_id=request_id,
                component_id=component_id,
                status="unavailable",
                reason=_bounded_text_detail(
                    f"missing capabilities: {_bounded_join(sorted(missing))}"
                ),
            )
        if request.component_id == INSPECT_COMPONENT_ID:
            report = {
                "schema_version": "inspect-report.v1",
                "request_id": request_id,
                "sources": [asdict(ref) for ref in request.sources],
                "config": dict(request.config),
            }
            filename = "inspect-report.json"
        else:
            report = capability_report()
            filename = "capability-report.json"
        digest = _write_json(output_dir / filename, report)
        return ComponentResult(
            request_id=request_id,
            component_id=component_id,
            status="complete",
            artifacts=(
                {
                    "artifact_id": filename,
                    "uri": str(Path(request.output_directory) / filename),
                    "sha256": digest,
                },
            ),
            provenance={"output_directory": request.output_directory},
        )
    except ReviewContractsValidationError as error:
        reason = _bounded_error_detail(error)
        return ComponentResult(
            request_id=request_id,
            component_id=component_id,
            status="failed",
            reason=reason,
        )
    except (OSError, TypeError, ValueError, RecursionError) as error:
        return ComponentResult(
            request_id=request_id,
            component_id=component_id,
            status="failed",
            reason=_bounded_error_detail(error),
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute SREV-01 review-contract components.")
    parser.add_argument("--input", required=True, help="Component request JSON file.")
    parser.add_argument("--config", required=False, default=None, help="Optional config JSON file.")
    parser.add_argument("--output", required=True, help="Output directory (must not exist).")
    parser.add_argument("--base", required=False, default=None, help="Base directory for output.")
    return parser


def _cli_identity(payload: Any) -> tuple[str, str]:
    """Return safe identity fields for a failure emitted before request validation."""
    if not isinstance(payload, Mapping):
        return "unknown", "unknown"
    request_id = payload.get("request_id")
    component_id = payload.get("component_id")
    return _safe_identity(request_id), _safe_identity(component_id)


def _result_document(result: ComponentResult) -> dict[str, Any]:
    """Serialize a component result with the shared versioned envelope.

    Returns:
        JSON-safe ``component-result.v1`` payload.
    """
    document = {"schema_version": COMPONENT_RESULT_SCHEMA_VERSION, **asdict(result)}
    document["request_id"] = _safe_identity(document.get("request_id"))
    document["component_id"] = _safe_identity(document.get("component_id"))
    document["reason"] = _bounded_text_detail(document.get("reason", ""))
    return document


def _print_cli_failure(reason: str, *, payload: Any = None) -> int:
    """Print a contract-valid failed result for pre-request CLI errors.

    Returns:
        The CLI failure exit code.
    """
    request_id, component_id = _cli_identity(payload)
    result = ComponentResult(
        request_id=request_id,
        component_id=component_id,
        status="failed",
        reason=_bounded_text_detail(reason),
    )
    print(  # noqa: T201
        json.dumps(_result_document(result), sort_keys=True, indent=2, allow_nan=False)
    )
    return 1


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for inspect and capability-report components.

    Returns:
        Process exit code (0 when the result status is complete).
    """
    args = _build_parser().parse_args(argv)
    payload: Any = None
    try:
        payload = json.loads(
            _read_bounded_text(
                Path(args.input),
                maximum_bytes=MAX_REVIEW_CONTRACT_CLI_INPUT_BYTES,
                label="request JSON",
            ),
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (OSError, ValueError, RecursionError):
        return _print_cli_failure("invalid_input: request JSON cannot be parsed safely")
    if not isinstance(payload, dict):
        return _print_cli_failure("invalid_input: request must be a JSON object")
    request_id, component_id = _cli_identity(payload)
    if args.config is not None:
        try:
            config = json.loads(
                _read_bounded_text(
                    Path(args.config),
                    maximum_bytes=MAX_REVIEW_CONTRACT_CLI_INPUT_BYTES,
                    label="config JSON",
                ),
                parse_constant=_reject_nonstandard_json_constant,
            )
        except (OSError, ValueError, RecursionError):
            return _print_cli_failure(
                "invalid_input: config JSON cannot be parsed safely", payload=payload
            )
        if not isinstance(config, dict):
            return _print_cli_failure(
                "invalid_input: config must be a JSON object", payload=payload
            )
        request_config = payload.get("config", {})
        if not isinstance(request_config, dict):
            return _print_cli_failure(
                "invalid_input: request config must be a JSON object", payload=payload
            )
        payload = {**payload, "config": {**request_config, **config}}
    payload = {**payload, "output_directory": args.output}
    try:
        request = component_request_from_dict(payload, source=args.input)
    except (ReviewContractsValidationError, RecursionError):
        return _print_cli_failure(
            "invalid_input: request does not satisfy component-request.v1",
            payload={"request_id": request_id, "component_id": component_id},
        )
    result = run(request, base=Path(args.base) if args.base is not None else None)
    print(  # noqa: T201 - CLI output
        json.dumps(_result_document(result), sort_keys=True, indent=2, allow_nan=False)
    )
    return 0 if result.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
