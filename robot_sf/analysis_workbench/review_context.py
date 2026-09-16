"""SREV-06 review-context component: diagnostic cohort context reports.

This module is a deliberately narrow contract consumer.  It reads an explicitly
selected, integrity-checked campaign-result document and optionally an
episode-selection document, then writes a diagnostic-only JSON/HTML context
report.  It never executes a simulator, planner, benchmark, or learned model.

The shared SREV-01 component request/result envelopes remain authoritative.  The
leaf-owned ``OUTPUT_SCHEMAS`` registry documents the two payload schemas that
this component advertises without changing the shared contract owner.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import os
import re
import secrets
import stat
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path, PureWindowsPath
from typing import Any

from jsonschema import Draft202012Validator

from robot_sf.analysis_workbench.review_contracts import (
    COMPONENT_DESCRIPTOR_SCHEMA_VERSION,
    COMPONENT_REQUEST_SCHEMA_VERSION,
    COMPONENT_RESULT_SCHEMA_VERSION,
    ComponentRequest,
    ComponentResult,
    ReviewContractsValidationError,
    SourceRef,
    component_descriptor_from_dict,
    component_request_from_dict,
)
from robot_sf.benchmark.case_workbench import _load_records, _v2_integrity_errors

COMPONENT_ID = "srev06-review-context"
COMPONENT_VERSION = "1.0.0"

CAMPAIGN_RESULT_FORMAT = "campaign-result"
CAMPAIGN_RESULT_STORE_FORMAT = "campaign-result-store"
EPISODE_SELECTION_FORMAT = "episode-selection"
CAMPAIGN_RESULT_SCHEMA = "campaign-result.v1"
CAMPAIGN_RESULT_STORE_SCHEMA = "campaign-result-store.v2"
EPISODE_SELECTION_SCHEMA = "episode-selection.v1"

REQUIRED_CAPABILITIES = (CAMPAIGN_RESULT_FORMAT,)
OPTIONAL_CAPABILITIES = (EPISODE_SELECTION_FORMAT,)
_CAMPAIGN_RESULT_FORMATS = frozenset({CAMPAIGN_RESULT_FORMAT, CAMPAIGN_RESULT_STORE_FORMAT})
_CAMPAIGN_RESULT_SCHEMAS = frozenset({CAMPAIGN_RESULT_SCHEMA, CAMPAIGN_RESULT_STORE_SCHEMA})

OUTPUT_REPORT_FILENAME = "context-report.json"
OUTPUT_HTML_FILENAME = "context-report.html"
OUTPUT_CAPABILITY_FILENAME = "missing-capability-report.json"

STATUS_COMPLETE = "complete"
STATUS_PARTIAL = "partial"
STATUS_UNAVAILABLE = "unavailable"
STATUS_FAILED = "failed"

PERCENTILE_METHOD = "linear-interpolation-on-sorted-values"
CLAIM_BOUNDARY = "diagnostic_only_recorded_cohort_context"
EVIDENCE_STATUS = "diagnostic_only"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_SHA40_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_SEMVER_RE = re.compile(r"^(?:v)?(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)$")
_EXECUTION_STATUSES = frozenset(
    {"native", "adapter", "fallback", "degraded", "unavailable", "failed"}
)
_NON_ADMISSIBLE_STATUSES = frozenset({"fallback", "degraded", "unavailable", "failed"})
_MAX_SOURCE_BYTES = 64 * 1024 * 1024
_MAX_CONTROL_BYTES = 1 * 1024 * 1024
_MAX_CONTROL_DEPTH = 32
_MAX_CONTROL_NODES = 8192
_MAX_CONTROL_COLLECTION_ITEMS = 4096
_MAX_CONTROL_STRING_BYTES = 256 * 1024
_MAX_SOURCE_FILES = 4096
_MAX_OUTPUT_SOURCES = 16
_METRIC_NAME_RE = r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$"
_IDENTITY_PROVENANCE_KEYS = ("source_commit", "config_identity")
_CANONICAL_CAMPAIGN_ID_KEYS = ("campaign_id", "study_id")
_CANONICAL_SOURCE_COMMIT_KEYS = ("source_commit", "git_hash", "commit_sha", "commit")
_CANONICAL_CONFIG_IDENTITY_KEYS = ("config_identity", "config_hash")
_CANONICAL_CONFIG_DIGEST_KEYS = ("config_digest",)
_MALFORMED_EPISODE_DIAGNOSTICS = frozenset(
    {
        "episode_row_malformed",
        "episode_row_missing_id",
        "episode_row_seed_invalid",
        "episode_row_config_invalid",
        "episode_row_metrics_invalid",
        "episode_row_outcome_invalid",
        "episode_row_execution_status_invalid",
        "episode_row_execution_status_conflict",
    }
)


# These schemas belong to this leaf's payloads.  The shared request/result
# schemas intentionally remain in review_contracts.py, which is owned by SREV-01.
OUTPUT_SCHEMAS: dict[str, dict[str, Any]] = {
    "review-context.v1": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://robot-sf.local/schemas/review-context.v1.json",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "evidence_status",
            "claim_boundary",
            "grain",
            "denominator",
            "outcomes",
            "metrics",
            "selection_coverage",
            "campaign",
            "exclusions",
            "source_provenance",
        ],
        "properties": {
            "schema_version": {"const": "review-context.v1"},
            "evidence_status": {"const": EVIDENCE_STATUS},
            "claim_boundary": {"type": "string", "minLength": 1},
            "grain": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "planner_ids",
                    "scenario_ids",
                    "seeds",
                    "config_ids",
                    "episodes",
                ],
                "properties": {
                    "planner_ids": {
                        "type": "array",
                        "maxItems": _MAX_CONTROL_COLLECTION_ITEMS,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "scenario_ids": {
                        "type": "array",
                        "maxItems": _MAX_CONTROL_COLLECTION_ITEMS,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "seeds": {
                        "type": "array",
                        "maxItems": _MAX_CONTROL_COLLECTION_ITEMS,
                        "items": {"type": "integer"},
                    },
                    "config_ids": {
                        "type": "array",
                        "maxItems": _MAX_CONTROL_COLLECTION_ITEMS,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "episodes": {"type": "integer", "minimum": 0},
                },
            },
            "denominator": {"type": "integer", "minimum": 0},
            "outcomes": {
                "type": "object",
                "maxProperties": _MAX_CONTROL_COLLECTION_ITEMS,
                "propertyNames": {"pattern": _METRIC_NAME_RE},
                "additionalProperties": {"type": "integer", "minimum": 0},
            },
            "metrics": {
                "type": "object",
                "maxProperties": _MAX_CONTROL_COLLECTION_ITEMS,
                "propertyNames": {"pattern": _METRIC_NAME_RE},
                "additionalProperties": {"$ref": "#/$defs/metric_summary"},
            },
            "selection_coverage": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "status",
                    "selected",
                    "denominator",
                    "unknown_selected_ids",
                    "source_artifact_id",
                ],
                "properties": {
                    "status": {
                        "enum": ["used", "not_supplied", "skipped", "unavailable", "invalid"]
                    },
                    "selected": {"type": "integer", "minimum": 0},
                    "denominator": {"type": "integer", "minimum": 0},
                    "unknown_selected_ids": {
                        "type": "array",
                        "maxItems": _MAX_CONTROL_COLLECTION_ITEMS,
                        "items": {"type": "string"},
                    },
                    "source_artifact_id": {"type": ["string", "null"]},
                },
            },
            "campaign": {
                "type": "object",
                "additionalProperties": False,
                "required": ["campaign_id", "source_artifact_id", "availability"],
                "properties": {
                    "campaign_id": {"type": "string", "minLength": 1},
                    "source_artifact_id": {"type": "string", "minLength": 1},
                    "availability": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["status", "reason"],
                        "properties": {
                            "status": {"const": STATUS_COMPLETE},
                            "reason": {"type": "string"},
                        },
                    },
                },
            },
            "exclusions": {
                "type": "object",
                "additionalProperties": False,
                "required": ["count", "by_status", "rows"],
                "properties": {
                    "count": {"type": "integer", "minimum": 0},
                    "by_status": {
                        "type": "object",
                        "maxProperties": _MAX_CONTROL_COLLECTION_ITEMS,
                        "additionalProperties": {"type": "integer", "minimum": 0},
                    },
                    "rows": {
                        "type": "array",
                        "maxItems": _MAX_CONTROL_COLLECTION_ITEMS,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["episode_id", "status", "reason"],
                            "properties": {
                                "episode_id": {"type": "string", "minLength": 1},
                                "status": {"type": "string", "minLength": 1},
                                "reason": {"type": "string", "minLength": 1},
                            },
                        },
                    },
                },
            },
            "source_provenance": {
                "type": "array",
                "maxItems": _MAX_OUTPUT_SOURCES,
                "items": {"$ref": "#/$defs/source_provenance"},
            },
        },
        "$defs": {
            "metric_summary": {
                "type": "object",
                "additionalProperties": False,
                "required": ["count", "missing", "denominator"],
                "properties": {
                    "count": {"type": "integer", "minimum": 0},
                    "missing": {"type": "integer", "minimum": 0},
                    "denominator": {"type": "integer", "minimum": 0},
                    "min": {"type": "number"},
                    "max": {"type": "number"},
                    "mean": {"type": "number"},
                    "p25": {"type": "number"},
                    "p50": {"type": "number"},
                    "p75": {"type": "number"},
                    "percentile_method": {"const": PERCENTILE_METHOD},
                },
            },
            "source_provenance": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "artifact_id",
                    "uri",
                    "format",
                    "schema_declared",
                    "sha256_declared",
                    "source_commit",
                    "config_identity",
                    "config_digest",
                    "config_digests",
                    "units",
                    "coordinate_frame",
                    "sha256_observed",
                    "integrity_status",
                    "execution_status",
                    "canonical_source",
                ],
                "properties": {
                    "artifact_id": {"type": "string", "minLength": 1},
                    "uri": {"type": "string", "minLength": 1},
                    "format": {
                        "enum": [
                            CAMPAIGN_RESULT_FORMAT,
                            CAMPAIGN_RESULT_STORE_FORMAT,
                            EPISODE_SELECTION_FORMAT,
                        ]
                    },
                    "schema_declared": {"type": "string", "minLength": 1},
                    "sha256_declared": {"type": ["string", "null"], "pattern": _SHA256_RE.pattern},
                    "source_commit": {"type": ["string", "null"], "pattern": _SHA40_RE.pattern},
                    "config_identity": {"type": ["string", "null"], "minLength": 1},
                    "config_digest": {"type": ["string", "null"]},
                    "config_digests": {
                        "type": "array",
                        "maxItems": _MAX_CONTROL_COLLECTION_ITEMS,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "units": {"type": ["string", "null"]},
                    "coordinate_frame": {"type": ["string", "null"]},
                    "sha256_observed": {
                        "type": ["string", "null"],
                        "pattern": _SHA256_RE.pattern,
                    },
                    "integrity_status": {"type": "string", "minLength": 1},
                    "execution_status": {
                        "type": ["string", "null"],
                        "enum": [*sorted(_EXECUTION_STATUSES), None],
                    },
                    "canonical_source": {"type": "boolean"},
                    "canonical_manifest_schema": {"type": ["string", "null"]},
                    "payload_provenance": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "source_commit": {
                                "type": ["string", "null"],
                                "pattern": _SHA40_RE.pattern,
                            },
                            "config_identity": {"type": ["string", "null"]},
                            "config_digest": {"type": ["string", "null"]},
                            "execution_status": {
                                "type": ["string", "null"],
                                "enum": [*sorted(_EXECUTION_STATUSES), None],
                            },
                            "producer": {"type": ["string", "null"]},
                        },
                    },
                },
            },
        },
    },
    "missing-capability-report.v1": {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://robot-sf.local/schemas/missing-capability-report.v1.json",
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version",
            "missing_capabilities",
            "skipped_optional_streams",
            "diagnostics",
        ],
        "properties": {
            "schema_version": {"const": "missing-capability-report.v1"},
            "missing_capabilities": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
            "skipped_optional_streams": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
            },
            "diagnostics": {
                "type": "array",
                "maxItems": _MAX_CONTROL_COLLECTION_ITEMS,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["code", "severity"],
                    "properties": {
                        "code": {"type": "string", "minLength": 1},
                        "severity": {"enum": ["error", "info", "warning"]},
                        "detail": {"type": "string"},
                    },
                },
            },
        },
    },
}

# Explicit alias for callers looking for a registry-shaped name.
OUTPUT_SCHEMA_REGISTRY = OUTPUT_SCHEMAS

_DESCRIPTOR_DOC: dict[str, Any] = {
    "schema_version": COMPONENT_DESCRIPTOR_SCHEMA_VERSION,
    "component_id": COMPONENT_ID,
    "component_version": COMPONENT_VERSION,
    "supported_input_versions": [COMPONENT_REQUEST_SCHEMA_VERSION],
    "output_types": list(OUTPUT_SCHEMAS),
    "required_capabilities": list(REQUIRED_CAPABILITIES),
    "optional_capabilities": list(OPTIONAL_CAPABILITIES),
}
DESCRIPTOR = component_descriptor_from_dict(_DESCRIPTOR_DOC)


@dataclass(frozen=True, slots=True)
class _Diagnostic:
    """One structured diagnostic with a stable machine-readable code."""

    code: str
    severity: str = "error"
    detail: str = ""

    def as_dict(self) -> dict[str, str]:
        """Return the JSON representation of this diagnostic."""
        document = {"code": self.code, "severity": self.severity}
        if self.detail:
            document["detail"] = self.detail
        return document


@dataclass(frozen=True, slots=True)
class _LoadedSource:
    """One source document after integrity and schema admission."""

    ref: SourceRef
    payload: dict[str, Any]
    provenance: dict[str, Any]
    execution_status: str


class _CanonicalIdentityError(ReviewContractsValidationError):
    """Reject a canonical directory whose identity is missing or contradictory."""

    def __init__(self, code: str, detail: str):
        """Build an error carrying the leaf diagnostic code."""
        self.code = code
        super().__init__([detail])


@dataclass(slots=True)
class _ReservedOutputDirectory:
    """Retain trusted descriptors for one reserved output directory."""

    path: Path
    parent_fd: int
    directory_fd: int
    name: str
    device: int
    inode: int
    published_names: set[str] = field(default_factory=set)

    def close(self) -> None:
        """Close retained descriptors without turning cleanup into a new failure."""
        for attribute in ("directory_fd", "parent_fd"):
            descriptor = getattr(self, attribute)
            if descriptor < 0:
                continue
            try:
                os.close(descriptor)
            except OSError:
                pass
            finally:
                setattr(self, attribute, -1)


@dataclass
class _Episode:
    """One admitted episode at the canonical episode grain."""

    episode_id: str
    planner_id: str = "unknown"
    scenario_id: str = "unknown"
    seed: int | None = None
    config_id: str = "unknown"
    outcome: str = "unknown"
    metrics: dict[str, Any] = field(default_factory=dict)
    execution_status: str = "native"
    source_artifact_id: str = ""

    @property
    def identity(self) -> tuple[Any, ...]:
        """Return the full identity used for duplicate detection."""
        return (
            self.episode_id,
            self.planner_id,
            self.scenario_id,
            self.seed,
            self.config_id,
        )

    def fingerprint(self) -> str:
        """Return a deterministic fingerprint for exact duplicate detection."""
        payload = {
            "identity": self.identity,
            "outcome": self.outcome,
            "metrics": self.metrics,
            "execution_status": self.execution_status,
        }
        try:
            encoded = json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode("utf-8")
        except (TypeError, ValueError, OverflowError):
            encoded = repr(payload).encode("utf-8", errors="backslashreplace")
        return _sha256_bytes(encoded)


def descriptor() -> dict[str, Any]:
    """Return a JSON-safe, schema-valid component descriptor."""
    return json.loads(json.dumps(_DESCRIPTOR_DOC))


def output_schemas() -> dict[str, dict[str, Any]]:
    """Return a JSON-safe copy of the leaf-owned output schema registry."""
    return json.loads(json.dumps(OUTPUT_SCHEMAS))


def _sha256_bytes(payload: bytes) -> str:
    """Return the hexadecimal SHA-256 digest of raw bytes."""
    return hashlib.sha256(payload).hexdigest()


def _safe_detail(value: Any, *, limit: int = 160) -> str:
    """Render bounded diagnostic detail without letting input control policy.

    Returns:
        A bounded, NUL-escaped diagnostic string.
    """
    text = str(value).replace("\x00", "\\x00")
    return text[:limit]


def _add_diagnostic(
    diagnostics: list[_Diagnostic], code: str, *, severity: str = "error", detail: Any = ""
) -> None:
    """Append one bounded structured diagnostic."""
    diagnostics.append(_Diagnostic(code, severity, _safe_detail(detail) if detail else ""))


def _reject_json_constant(value: str) -> None:
    """Reject non-standard JSON constants before they reach a leaf payload."""
    raise ValueError(f"non-standard JSON constant: {value}")


def _reject_non_finite_number(value: Any, *, label: str, path: str) -> None:
    """Reject an in-memory floating-point value that JSON cannot persist."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ReviewContractsValidationError(
            [f"{label}_non_finite_number: non-finite number at {path}"]
        )


def _validate_bounded_document(value: Any, *, label: str) -> None:
    """Bound nested control/source documents before expensive leaf processing.

    The shared request schema owns field semantics.  This leaf-owned guard
    bounds depth, node count, collection size, and string size so a valid
    envelope cannot turn into an unbounded control or resource operation.
    """
    pending: list[tuple[Any, int, str]] = [(value, 0, label)]
    nodes = 0
    while pending:
        current, depth, path = pending.pop()
        nodes += 1
        if depth > _MAX_CONTROL_DEPTH:
            raise ReviewContractsValidationError(
                [f"{label}_depth_exceeded: maximum {_MAX_CONTROL_DEPTH} levels"]
            )
        if nodes > _MAX_CONTROL_NODES:
            raise ReviewContractsValidationError(
                [f"{label}_nodes_exceeded: maximum {_MAX_CONTROL_NODES} values"]
            )
        if isinstance(current, str):
            if len(current.encode("utf-8", errors="surrogatepass")) > _MAX_CONTROL_STRING_BYTES:
                raise ReviewContractsValidationError(
                    [
                        f"{label}_string_too_large: maximum "
                        f"{_MAX_CONTROL_STRING_BYTES} bytes at {path}"
                    ]
                )
            continue
        _reject_non_finite_number(current, label=label, path=path)
        if isinstance(current, Mapping):
            if len(current) > _MAX_CONTROL_COLLECTION_ITEMS:
                raise ReviewContractsValidationError(
                    [
                        f"{label}_properties_exceeded: maximum "
                        f"{_MAX_CONTROL_COLLECTION_ITEMS} properties at {path}"
                    ]
                )
            pending.extend((item, depth + 1, f"{path}.{key}") for key, item in current.items())
        elif isinstance(current, (list, tuple)):
            if len(current) > _MAX_CONTROL_COLLECTION_ITEMS:
                raise ReviewContractsValidationError(
                    [
                        f"{label}_items_exceeded: maximum "
                        f"{_MAX_CONTROL_COLLECTION_ITEMS} items at {path}"
                    ]
                )
            pending.extend(
                (item, depth + 1, f"{path}[{index}]") for index, item in enumerate(current)
            )


def _diagnostic_documents(diagnostics: list[_Diagnostic]) -> tuple[dict[str, str], ...]:
    """Return sorted, deduplicated diagnostic documents."""
    unique = {
        (
            item.code,
            item.severity,
            item.detail,
        ): item.as_dict()
        for item in diagnostics
    }
    return tuple(unique[key] for key in sorted(unique))


def _reason_with_diagnostics(prefix: str, diagnostics: list[_Diagnostic]) -> str:
    """Return a bounded reason that retains stable diagnostic codes."""
    codes = sorted({item.code for item in diagnostics})[:8]
    return prefix if not codes else f"{prefix}; {', '.join(codes)}"


def _result(
    request_id: str,
    component_id: str,
    status: str,
    *,
    reason: str = "",
    diagnostics: list[_Diagnostic] | None = None,
    provenance: dict[str, Any] | None = None,
    artifacts: tuple[dict[str, Any], ...] = (),
) -> ComponentResult:
    """Build a typed result envelope with stable structured diagnostics.

    Returns:
        A component result containing JSON-safe diagnostic mappings.
    """
    return ComponentResult(
        request_id=request_id,
        component_id=component_id,
        status=status,
        artifacts=artifacts,
        diagnostics=_diagnostic_documents(diagnostics or []),
        provenance=provenance or {},
        reason=reason,
    )


def _request_identity(request: Any) -> tuple[str, str]:
    """Extract safe identity fields for malformed direct API calls.

    Returns:
        The request and component identifiers safe for a failure envelope.
    """
    if isinstance(request, ComponentRequest):
        return request.request_id, request.component_id
    if isinstance(request, dict):
        request_id = request.get("request_id")
        component_id = request.get("component_id")
        return (
            request_id if isinstance(request_id, str) and request_id else "unknown",
            component_id if isinstance(component_id, str) and component_id else COMPONENT_ID,
        )
    return "unknown", COMPONENT_ID


def _check_version_compatible(config: dict[str, Any]) -> str | None:
    """Return a stable failure reason for an incompatible component version."""
    minimum = config.get("min_component_version")
    if minimum is None:
        return None
    wanted_match = _SEMVER_RE.fullmatch(str(minimum))
    ours_match = _SEMVER_RE.fullmatch(COMPONENT_VERSION)
    if wanted_match is None or ours_match is None:
        return f"incompatible_component_version: malformed min_component_version: {_safe_detail(minimum)!r}"
    wanted = tuple(int(value) for value in wanted_match.groups())
    ours = tuple(int(value) for value in ours_match.groups())
    if wanted > ours:
        return (
            "incompatible_component_version: "
            f"request needs v{str(minimum).lstrip('v')}, component is v{COMPONENT_VERSION}"
        )
    return None


def _reject_not_applicable(request: ComponentRequest) -> ComponentResult | None:
    """Reject unsupported components, capabilities, and versions.

    Returns:
        An unavailable/failed result, or ``None`` when the request applies.
    """
    if request.component_id != COMPONENT_ID:
        return _result(
            request.request_id,
            request.component_id,
            STATUS_UNAVAILABLE,
            reason=f"unsupported_component: {_safe_detail(request.component_id)}",
        )
    supported = set(REQUIRED_CAPABILITIES) | set(OPTIONAL_CAPABILITIES)
    missing = sorted(set(request.required_capabilities) - supported)
    if missing:
        return _result(
            request.request_id,
            request.component_id,
            STATUS_UNAVAILABLE,
            reason=f"missing_required_capabilities: {', '.join(missing)}",
        )
    version_error = _check_version_compatible(request.config)
    if version_error is not None:
        return _result(
            request.request_id,
            request.component_id,
            STATUS_FAILED,
            reason=version_error,
        )
    return None


def _path_parts(path_value: str, *, kind: str) -> tuple[str, ...]:
    """Validate a relative POSIX path and return normalized parts.

    Returns:
        Normalized path components.
    """
    try:
        candidate = Path(path_value)
        windows = PureWindowsPath(path_value)
    except (TypeError, ValueError) as error:
        raise ReviewContractsValidationError(
            [f"{kind} rejected: malformed path: {type(error).__name__}"]
        ) from error
    if (
        not isinstance(path_value, str)
        or not candidate.parts
        or candidate.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or ".." in candidate.parts
        or ".." in windows.parts
    ):
        raise ReviewContractsValidationError(
            [f"{kind} rejected (absolute or traversal): {_safe_detail(path_value)}"]
        )
    return candidate.parts


def _resolve_source(path_value: str, base: Path) -> Path:
    """Resolve a source path and reject lexical or real-path escapes.

    Returns:
        The contained regular-file or canonical-store directory path.
    """
    parts = _path_parts(path_value, kind="source uri")
    try:
        root = base.resolve(strict=True)
        if not root.is_dir():
            raise ReviewContractsValidationError([f"source base is not a directory: {base}"])
        current = root
        for part in parts:
            current = current / part
            if current.is_symlink():
                raise ReviewContractsValidationError(
                    [f"source uri rejected (symlink component): {_safe_detail(path_value)}"]
                )
        resolved = current.resolve(strict=True)
    except ReviewContractsValidationError:
        raise
    except (OSError, RuntimeError) as error:
        raise ReviewContractsValidationError(
            [f"source uri cannot be resolved safely: {type(error).__name__}"]
        ) from error
    if not resolved.is_relative_to(root) or not (resolved.is_file() or resolved.is_dir()):
        raise ReviewContractsValidationError(
            [
                "source uri rejected (outside base or not a regular file/directory): "
                f"{_safe_detail(path_value)}"
            ]
        )
    return resolved


def _read_regular_file_no_follow(path: Path, *, limit: int, kind: str) -> bytes:
    """Read one already-contained regular file without following its final link.

    Returns:
        The bounded file bytes.
    """
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    non_blocking = getattr(os, "O_NONBLOCK", 0)
    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    file_fd = os.open(path, os.O_RDONLY | close_on_exec | no_follow | non_blocking)
    try:
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise ReviewContractsValidationError([f"{kind} is not a regular file: {path.name}"])
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise ReviewContractsValidationError([f"{kind} too large: maximum {limit} bytes"])
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(file_fd)


def _directory_files(path: Path) -> list[Path]:
    """Inventory a contained source directory without admitting links/special files.

    Returns:
        Sorted regular files in the directory tree.
    """
    files: list[Path] = []
    total_bytes = 0
    pending = [path]
    while pending:
        current = pending.pop()
        try:
            entries = sorted(os.scandir(current), key=lambda entry: entry.name)
        except OSError as error:
            raise ReviewContractsValidationError(
                [f"source directory cannot be read: {type(error).__name__}"]
            ) from error
        for entry in entries:
            if entry.is_symlink():
                raise ReviewContractsValidationError(
                    [f"source directory rejected (symlink entry): {_safe_detail(entry.name)}"]
                )
            try:
                mode = entry.stat(follow_symlinks=False).st_mode
            except OSError as error:
                raise ReviewContractsValidationError(
                    [f"source directory entry cannot be inspected: {type(error).__name__}"]
                ) from error
            entry_path = Path(entry.path)
            if stat.S_ISDIR(mode):
                pending.append(entry_path)
            elif stat.S_ISREG(mode):
                files.append(entry_path)
                total_bytes += entry.stat(follow_symlinks=False).st_size
                if len(files) > _MAX_SOURCE_FILES:
                    raise ReviewContractsValidationError(
                        [f"source directory has too many files: maximum {_MAX_SOURCE_FILES}"]
                    )
                if total_bytes > _MAX_SOURCE_BYTES:
                    raise ReviewContractsValidationError(
                        [f"source directory too large: maximum {_MAX_SOURCE_BYTES} bytes"]
                    )
            else:
                raise ReviewContractsValidationError(
                    [f"source directory rejected (special entry): {_safe_detail(entry.name)}"]
                )
    return sorted(files, key=lambda item: str(item.relative_to(path)))


def _read_descriptor_backed_file(
    relative_parts: tuple[str, ...], root: Path, *, limit: int, kind: str
) -> bytes:
    """Read one bounded regular file from a directory descriptor walk.

    The directory descriptor is opened before walking any attacker-controlled
    component.  Every component uses ``O_NOFOLLOW`` and the final file also
    uses ``O_NONBLOCK`` so a path replaced by a link, FIFO, or special file
    cannot redirect or block the snapshot.

    Returns:
        The bounded file bytes.
    """
    if not relative_parts:
        raise ReviewContractsValidationError([f"{kind} path is empty"])
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    non_blocking = getattr(os, "O_NONBLOCK", 0)
    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    root_fd = os.open(
        root,
        os.O_RDONLY | close_on_exec | directory_flag | no_follow,
    )
    current_fd = root_fd
    file_fd = -1
    try:
        for part in relative_parts[:-1]:
            next_fd = os.open(
                part,
                os.O_RDONLY | close_on_exec | directory_flag | no_follow,
                dir_fd=current_fd,
            )
            os.close(current_fd)
            current_fd = next_fd
        file_fd = os.open(
            relative_parts[-1],
            os.O_RDONLY | close_on_exec | no_follow | non_blocking,
            dir_fd=current_fd,
        )
        if not stat.S_ISREG(os.fstat(file_fd).st_mode):
            raise ReviewContractsValidationError([f"{kind} is not a regular file"])
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(file_fd, 1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > limit:
                raise ReviewContractsValidationError([f"{kind} too large: maximum {limit} bytes"])
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        if file_fd >= 0:
            os.close(file_fd)
        if current_fd >= 0:
            os.close(current_fd)


def _read_contained_bytes(path_value: str, base: Path) -> bytes:
    """Read one regular file through no-follow directory descriptors.

    The descriptor walk binds each path component to a directory file
    descriptor.  This keeps the containment check paired with the subsequent
    read instead of resolving a path and reopening it by name later.

    Returns:
        The source bytes.
    """
    resolved = _resolve_source(path_value, base)
    if not resolved.is_file():
        raise ReviewContractsValidationError(
            [f"source uri is a directory; canonical adapter required: {_safe_detail(path_value)}"]
        )
    return _read_descriptor_backed_file(
        _path_parts(path_value, kind="source uri"),
        base.resolve(strict=True),
        limit=_MAX_SOURCE_BYTES,
        kind=f"source uri: {_safe_detail(path_value)}",
    )


def _snapshot_directory(
    directory: Path, files: list[Path], snapshot_root: Path
) -> tuple[list[Path], str]:
    """Snapshot a canonical source once before handing it to an owner loader.

    The owner loader accepts paths, so passing the original directory would
    reopen attacker-controlled JSONL/receipt files after the leaf's digest
    check.  This function performs one descriptor-backed, bounded read of each
    inventoried file and materializes those bytes in a private temporary tree.
    All subsequent owner reads use that immutable-in-process snapshot.

    Returns:
        Snapshot file paths and the digest of the snapshotted directory bytes.
    """
    snapshot_files: list[Path] = []
    digest = hashlib.sha256()
    total_bytes = 0
    for source_path in files:
        relative = source_path.relative_to(directory)
        raw = _read_descriptor_backed_file(
            relative.parts,
            directory,
            limit=_MAX_SOURCE_BYTES,
            kind=f"canonical source: {relative}",
        )
        total_bytes += len(raw)
        if total_bytes > _MAX_SOURCE_BYTES:
            raise ReviewContractsValidationError(
                [f"source directory too large: maximum {_MAX_SOURCE_BYTES} bytes"]
            )
        target = snapshot_root.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(raw)
        snapshot_files.append(target)
    return snapshot_files, digest.hexdigest()


def _output_directory_flags() -> int:
    """Return no-follow flags suitable for opening a trusted directory."""
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )


def _output_directory_matches_entry(output: _ReservedOutputDirectory) -> bool:
    """Return whether the reserved name still names the retained directory."""
    if output.parent_fd < 0 or output.directory_fd < 0:
        return False
    try:
        visible = os.stat(output.name, dir_fd=output.parent_fd, follow_symlinks=False)
        trusted = os.fstat(output.directory_fd)
    except (OSError, ValueError):
        return False
    return (
        stat.S_ISDIR(visible.st_mode)
        and visible.st_dev == output.device
        and visible.st_ino == output.inode
        and trusted.st_dev == output.device
        and trusted.st_ino == output.inode
    )


def _assert_output_directory_current(output: _ReservedOutputDirectory) -> None:
    """Fail closed when the reserved output directory entry was replaced."""
    if not _output_directory_matches_entry(output):
        raise ReviewContractsValidationError(["output directory replaced during publication"])


def _open_output_parent(root: Path, parts: tuple[str, ...]) -> int:
    """Open the reserved output parent through a no-follow descriptor walk.

    Returns:
        The descriptor for the parent of the final output directory.
    """
    parent_fd = -1
    try:
        parent_fd = os.open(root, _output_directory_flags())
        if not stat.S_ISDIR(os.fstat(parent_fd).st_mode):
            raise ReviewContractsValidationError([f"output base is not a directory: {root}"])
        for part in parts[:-1]:
            try:
                os.mkdir(part, dir_fd=parent_fd)
            except FileExistsError:
                pass
            next_fd = os.open(part, _output_directory_flags(), dir_fd=parent_fd)
            os.close(parent_fd)
            parent_fd = next_fd
        return parent_fd
    except BaseException:
        if parent_fd >= 0:
            os.close(parent_fd)
        raise


def _create_output_directory(
    parent_fd: int, output_name: str, output_directory: str
) -> tuple[int, os.stat_result]:
    """Create and open the final output directory without following links.

    Returns:
        The retained directory descriptor and its stat result.
    """
    try:
        os.mkdir(output_name, dir_fd=parent_fd)
    except FileExistsError as error:
        raise ReviewContractsValidationError(
            [f"output_collision: already exists: {output_directory}"]
        ) from error
    output_fd = -1
    try:
        output_fd = os.open(output_name, _output_directory_flags(), dir_fd=parent_fd)
        output_stat = os.fstat(output_fd)
        if not stat.S_ISDIR(output_stat.st_mode):
            raise ReviewContractsValidationError(
                [f"output directory is not a real directory: {output_directory}"]
            )
        return output_fd, output_stat
    except BaseException:
        if output_fd >= 0:
            os.close(output_fd)
        raise


def _reserve_output_directory(output_directory: str, base: Path) -> _ReservedOutputDirectory:
    """Reserve an output directory and retain descriptors for its publication parent.

    Returns:
        The reserved directory and its retained parent/directory descriptors.
    """
    parts = _path_parts(output_directory, kind="output directory")
    root = base.resolve(strict=True)
    if not root.is_dir():
        raise ReviewContractsValidationError([f"output base is not a directory: {base}"])
    parent_fd = -1
    output_fd = -1
    reserved: _ReservedOutputDirectory | None = None
    try:
        parent_fd = _open_output_parent(root, parts)
        output_name = parts[-1]
        output_fd, output_stat = _create_output_directory(parent_fd, output_name, output_directory)
        reserved = _ReservedOutputDirectory(
            path=root.joinpath(*parts),
            parent_fd=parent_fd,
            directory_fd=output_fd,
            name=output_name,
            device=output_stat.st_dev,
            inode=output_stat.st_ino,
        )
        parent_fd = -1
        output_fd = -1
        return reserved
    except ReviewContractsValidationError:
        raise
    except (OSError, RuntimeError) as error:
        raise ReviewContractsValidationError(
            [
                "output directory cannot be reserved safely: "
                f"{_safe_detail(output_directory)}: {type(error).__name__}"
            ]
        ) from error
    finally:
        if reserved is None:
            if output_fd >= 0:
                os.close(output_fd)
            if parent_fd >= 0:
                os.close(parent_fd)


def _release_empty_output(output_directory: _ReservedOutputDirectory | None) -> None:
    """Remove files from, then safely release, a reserved output directory."""
    if output_directory is None:
        return
    try:
        if output_directory.directory_fd >= 0:
            for name in tuple(output_directory.published_names):
                try:
                    os.unlink(name, dir_fd=output_directory.directory_fd)
                except FileNotFoundError:
                    pass
            output_directory.published_names.clear()
        if output_directory.parent_fd >= 0 and _output_directory_matches_entry(output_directory):
            try:
                os.rmdir(output_directory.name, dir_fd=output_directory.parent_fd)
            except OSError:
                pass
    finally:
        output_directory.close()


def _finite_number(value: Any) -> float | None:
    """Return a finite float, or ``None`` for every malformed numeric value."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _percentile(sorted_values: list[float], fraction: float) -> float:
    """Compute one deterministic linearly interpolated percentile.

    Returns:
        The interpolated percentile value.
    """
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = fraction * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    weight = rank - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def _normalize_execution_status(value: Any) -> str | None:
    """Normalize one declared execution status, without guessing unknown values.

    Returns:
        A recognized normalized status, or ``None``.
    """
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower().replace("-", "_")
    return normalized if normalized in _EXECUTION_STATUSES else None


def _validate_source_declarations(
    ref: SourceRef,
    expected_schema: str | tuple[str, ...],
    diagnostics: list[_Diagnostic],
    *,
    expected_config_identity: str | None = None,
) -> bool:
    """Validate the declarations required before source computation.

    Returns:
        ``True`` when all required declarations are present and well formed.
    """
    valid = True
    expected_schemas = (
        {expected_schema} if isinstance(expected_schema, str) else set(expected_schema)
    )
    if ref.schema not in expected_schemas:
        _add_diagnostic(
            diagnostics,
            "source_schema_mismatch",
            detail=f"{ref.artifact_id}:{','.join(sorted(expected_schemas))}",
        )
        valid = False
    if not ref.sha256:
        _add_diagnostic(diagnostics, "source_digest_missing", detail=ref.artifact_id)
        valid = False
    elif _SHA256_RE.fullmatch(ref.sha256) is None:
        _add_diagnostic(diagnostics, "source_digest_malformed", detail=ref.artifact_id)
        valid = False
    if not ref.source_commit:
        _add_diagnostic(diagnostics, "source_commit_missing", detail=ref.artifact_id)
        valid = False
    elif _SHA40_RE.fullmatch(ref.source_commit) is None:
        _add_diagnostic(diagnostics, "source_commit_malformed", detail=ref.artifact_id)
        valid = False
    if not ref.config_identity.strip():
        _add_diagnostic(diagnostics, "config_identity_missing", detail=ref.artifact_id)
        valid = False
    if expected_config_identity is not None and ref.config_identity != expected_config_identity:
        _add_diagnostic(diagnostics, "config_identity_mismatch", detail=ref.artifact_id)
        valid = False
    return valid


def _source_provenance(ref: SourceRef) -> dict[str, Any]:
    """Create a provenance record before attempting to read a source.

    Returns:
        A mutable provenance record that can be completed after reading.
    """
    return {
        "artifact_id": ref.artifact_id,
        "uri": ref.uri,
        "format": ref.format,
        "schema_declared": ref.schema,
        "sha256_declared": ref.sha256.lower() if ref.sha256 else None,
        "source_commit": ref.source_commit or None,
        "config_identity": ref.config_identity or None,
        "config_digest": None,
        "config_digests": [],
        "units": ref.units or None,
        "coordinate_frame": ref.coordinate_frame or None,
        "sha256_observed": None,
        "integrity_status": "unverified",
        "execution_status": None,
        "canonical_source": False,
    }


def _source_format_matches(ref: SourceRef, expected_format: str) -> bool:
    """Return whether a source reference belongs to the expected leaf family."""
    if expected_format == CAMPAIGN_RESULT_FORMAT:
        return ref.format in _CAMPAIGN_RESULT_FORMATS
    return ref.format == expected_format


def _source_schema_values(expected_schema: str) -> frozenset[str]:
    """Return accepted schemas for one source family."""
    if expected_schema == CAMPAIGN_RESULT_SCHEMA:
        return frozenset(_CAMPAIGN_RESULT_SCHEMAS)
    return frozenset({expected_schema})


def _load_json_bytes(raw: bytes, *, label: str) -> Any:
    """Decode strict JSON and apply the leaf's bounded-document guard.

    Returns:
        The bounded decoded document.
    """
    try:
        payload = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ReviewContractsValidationError([f"{label} is not strict JSON"]) from error
    except ValueError as error:
        if str(error).startswith("non-standard JSON constant:"):
            raise ReviewContractsValidationError(
                [f"{label}_non_finite_number: non-finite JSON constant"]
            ) from error
        raise ReviewContractsValidationError([f"{label} is not strict JSON"]) from error
    _validate_bounded_document(payload, label=label)
    return payload


def _canonical_manifest(path: Path, files: set[Path]) -> dict[str, Any] | None:
    """Read an optional manifest from a safely inventoried directory.

    Returns:
        The manifest object, or ``None`` when the directory has no manifest.
    """
    manifest_path = path / "manifest.json"
    if manifest_path not in files:
        return None
    payload = _load_json_bytes(
        _read_regular_file_no_follow(manifest_path, limit=_MAX_CONTROL_BYTES, kind="manifest"),
        label="canonical manifest",
    )
    if not isinstance(payload, dict):
        raise ReviewContractsValidationError(["canonical manifest must be a JSON object"])
    return payload


def _validate_strict_jsonl(path: Path) -> None:
    """Reject non-standard JSON constants before the owner JSONL loader runs."""
    raw = _read_regular_file_no_follow(path, limit=_MAX_SOURCE_BYTES, kind="canonical JSONL")
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            _load_json_bytes(line, label=f"canonical JSONL line {line_number}")
        except ReviewContractsValidationError as error:
            raise ReviewContractsValidationError(
                [f"canonical JSONL line {line_number}: {'; '.join(error.errors)}"]
            ) from error


def _identity_values(container: Mapping[str, Any], keys: tuple[str, ...]) -> tuple[list[str], bool]:
    """Collect non-empty identity values and flag non-string declarations.

    Returns:
        The collected values and whether a non-string declaration was found.
    """
    values: list[str] = []
    invalid = False
    for key in keys:
        value = container.get(key)
        if value in (None, ""):
            continue
        if not isinstance(value, str) or not value.strip():
            invalid = True
            continue
        values.append(value.strip())
    return values, invalid


def _record_identity_containers(record: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return owner row mappings that can carry canonical identity metadata."""
    containers: list[Mapping[str, Any]] = [record]
    for key in ("provenance", "result_provenance", "cell_context"):
        nested = record.get(key)
        if isinstance(nested, Mapping):
            containers.append(nested)
    metadata = record.get("algorithm_metadata")
    if isinstance(metadata, Mapping):
        trace = metadata.get("analysis_trace")
        if isinstance(trace, Mapping):
            containers.append(trace)
    return tuple(containers)


def _identity_values_from_containers(
    containers: tuple[Mapping[str, Any], ...], keys: tuple[str, ...]
) -> tuple[list[str], bool]:
    """Collect identity aliases from all supported nested provenance containers.

    Returns:
        The collected values and whether any declaration was malformed.
    """
    values: list[str] = []
    invalid = False
    for container in containers:
        candidates, container_invalid = _identity_values(container, keys)
        values.extend(candidates)
        invalid = invalid or container_invalid
    return values, invalid


def _validate_payload_campaign_identity(
    payload: Mapping[str, Any], *, expected_campaign_id: str, diagnostics: list[_Diagnostic]
) -> bool:
    """Reject foreign or contradictory campaign aliases in regular result rows.

    Returns:
        ``True`` when every mapped episode row binds to the requested campaign.
    """
    entries = payload.get("episodes")
    if not isinstance(entries, list):
        return True
    valid = True
    for index, record in enumerate(entries):
        if not isinstance(record, Mapping):
            continue
        values, invalid = _identity_values_from_containers(
            _record_identity_containers(record), _CANONICAL_CAMPAIGN_ID_KEYS
        )
        if (
            invalid
            or len(set(values)) > 1
            or any(value != expected_campaign_id for value in values)
        ):
            _add_diagnostic(diagnostics, "mixed_campaign_row", detail=index)
            valid = False
    return valid


def _canonical_config_digest(
    manifest: Mapping[str, Any] | None, records: list[dict[str, Any]]
) -> list[str]:
    """Collect independent owner ``config_digest`` provenance values.

    Returns:
        Sorted unique digests, or an empty list when the owner provided none.
    """
    containers: list[Mapping[str, Any]] = []
    if isinstance(manifest, Mapping):
        containers.append(manifest)
    for record in records:
        if isinstance(record, Mapping):
            containers.extend(_record_identity_containers(record))
    values, invalid = _identity_values_from_containers(
        tuple(containers), _CANONICAL_CONFIG_DIGEST_KEYS
    )
    unique = sorted(set(values))
    if invalid:
        raise _CanonicalIdentityError(
            "canonical_config_digest_invalid",
            "canonical config_digest provenance is not a non-empty string",
        )
    return unique


def _canonical_identity(
    manifest: Mapping[str, Any] | None,
    records: list[dict[str, Any]],
    *,
    ref: SourceRef,
) -> tuple[str, str, str]:
    """Bind campaign, commit, and configuration identity to canonical owner bytes.

    A caller-supplied source reference is not evidence of the identity of a
    canonical directory.  The owner manifest may bind the whole directory;
    otherwise every row must carry the corresponding identity field.

    Returns:
        The bound campaign ID, source commit, and configuration identity.
    """
    manifest_mapping = manifest if isinstance(manifest, Mapping) else {}
    specs = (
        (
            "campaign",
            _CANONICAL_CAMPAIGN_ID_KEYS,
            None,
            "canonical campaign identity",
        ),
        (
            "source_commit",
            _CANONICAL_SOURCE_COMMIT_KEYS,
            ref.source_commit,
            "canonical source commit",
        ),
        (
            "config_identity",
            _CANONICAL_CONFIG_IDENTITY_KEYS,
            ref.config_identity,
            "canonical configuration identity",
        ),
    )
    bound: dict[str, str] = {}
    for name, keys, expected, label in specs:
        manifest_values, manifest_invalid = _identity_values(manifest_mapping, keys)
        values = list(manifest_values)
        rows_missing_value = False
        rows_invalid = False
        for record in records:
            row_values: list[str] = []
            row_invalid = False
            for container in _record_identity_containers(record):
                candidates, invalid = _identity_values(container, keys)
                row_values.extend(candidates)
                row_invalid = row_invalid or invalid
            if not row_values and not manifest_values:
                rows_missing_value = True
            values.extend(row_values)
            rows_invalid = rows_invalid or row_invalid
        unique = set(values)
        if manifest_invalid or rows_invalid or not unique or rows_missing_value:
            raise _CanonicalIdentityError(
                "canonical_identity_unbound",
                f"{label} is missing from the canonical manifest/rows",
            )
        if len(unique) != 1:
            raise _CanonicalIdentityError(
                "canonical_identity_mismatch",
                f"{label} conflicts across the canonical manifest/rows",
            )
        actual = next(iter(unique))
        if expected is not None and actual != expected:
            raise _CanonicalIdentityError(
                "canonical_identity_mismatch",
                f"{label} does not match the source declaration",
            )
        bound[name] = actual
    return bound["campaign"], bound["source_commit"], bound["config_identity"]


def _load_canonical_directory(  # noqa: C901
    directory: Path,
    *,
    ref: SourceRef,
    expected_campaign_id: str,
    files: list[Path],
) -> tuple[dict[str, Any], str, str | None]:
    """Adapt a case-workbench/campaign-result-store directory to leaf rows.

    ``case_workbench._load_records`` remains the canonical row loader.  This
    adapter only verifies the bounded, no-follow input boundary and wraps the
    owner's rows in the leaf's campaign-result envelope; it does not redefine
    the result-store schema or central registry.

    Returns:
        The wrapped payload, source execution status, and manifest schema.
    """
    file_set = set(files)
    root_manifest = _canonical_manifest(directory, file_set)
    store = directory
    manifest = root_manifest
    nested_store = directory / "campaign-result-store.v2"
    if nested_store.is_dir():
        nested_files = [item for item in files if nested_store in item.parents]
        nested_manifest = _canonical_manifest(nested_store, set(nested_files))
        if nested_manifest is None:
            raise ReviewContractsValidationError(["canonical result store manifest is missing"])
        store = nested_store
        manifest = nested_manifest
    manifest_schema = manifest.get("schema_version") if isinstance(manifest, dict) else None
    has_parquet = (store / "episodes.parquet") in file_set
    has_jsonl = any((store / name) in file_set for name in ("episodes.jsonl", "records.jsonl"))
    if has_parquet:
        if manifest_schema != CAMPAIGN_RESULT_STORE_SCHEMA:
            raise ReviewContractsValidationError(
                ["canonical result store manifest schema mismatch"]
            )
        integrity_errors = _v2_integrity_errors(store)
        if integrity_errors:
            raise ReviewContractsValidationError(
                ["canonical result store integrity invalid: " + ",".join(integrity_errors[:8])]
            )
    elif not has_jsonl:
        raise ReviewContractsValidationError(
            ["canonical result store has no episodes.jsonl, records.jsonl, or episodes.parquet"]
        )
    jsonl_path = next(
        (
            store / name
            for name in ("episodes.jsonl", "records.jsonl")
            if (store / name) in file_set
        ),
        None,
    )
    if jsonl_path is not None:
        _validate_strict_jsonl(jsonl_path)
    try:
        records = _load_records(store)
    except (OSError, RuntimeError, ValueError) as error:
        raise ReviewContractsValidationError(
            [f"canonical result store rows unavailable: {type(error).__name__}"]
        ) from error
    if not isinstance(records, list):
        raise ReviewContractsValidationError(["canonical result store rows must be a list"])
    _validate_bounded_document(records, label="canonical source")
    payload_campaign_id, payload_source_commit, payload_config_identity = _canonical_identity(
        manifest,
        records,
        ref=ref,
    )
    payload_config_digests = _canonical_config_digest(manifest, records)
    source_status = (
        _normalize_execution_status(manifest.get("execution_status"))
        if isinstance(manifest, dict) and "execution_status" in manifest
        else "native"
    )
    if source_status is None:
        raise ReviewContractsValidationError(["canonical result store execution status is invalid"])
    payload = {
        "schema_version": ref.schema,
        "campaign_id": payload_campaign_id,
        "source_commit": payload_source_commit,
        "config_identity": payload_config_identity,
        "config_digest": payload_config_digests[0] if len(payload_config_digests) == 1 else None,
        "config_digests": payload_config_digests,
        "execution_status": source_status,
        "episodes": records,
    }
    return payload, source_status, manifest_schema if isinstance(manifest_schema, str) else None


def _validate_payload_identity(
    payload: Mapping[str, Any],
    *,
    ref: SourceRef,
    diagnostics: list[_Diagnostic],
    required: bool,
    include_episode_rows: bool = False,
) -> bool:
    """Require payload identity fields to agree with the declared source ref.

    Returns:
        ``True`` when every present/required identity agrees.
    """
    valid = True
    containers = list(_record_identity_containers(payload))
    if include_episode_rows:
        entries = payload.get("episodes")
        if isinstance(entries, list):
            containers.extend(
                container
                for record in entries
                if isinstance(record, Mapping)
                for container in _record_identity_containers(record)
            )
    for key, aliases in (
        ("source_commit", _CANONICAL_SOURCE_COMMIT_KEYS),
        ("config_identity", _CANONICAL_CONFIG_IDENTITY_KEYS),
    ):
        values, invalid = _identity_values_from_containers(tuple(containers), aliases)
        unique = set(values)
        if invalid or len(unique) > 1:
            _add_diagnostic(diagnostics, f"{key}_mismatch", detail=ref.artifact_id)
            valid = False
            continue
        if not unique:
            if required:
                _add_diagnostic(
                    diagnostics, f"source_payload_{key}_missing", detail=ref.artifact_id
                )
                valid = False
            continue
        if next(iter(unique)) != getattr(ref, key):
            _add_diagnostic(diagnostics, f"{key}_mismatch", detail=ref.artifact_id)
            valid = False
    return valid


def _payload_config_digest(
    payload: Mapping[str, Any], diagnostics: list[_Diagnostic], artifact_id: str
) -> tuple[list[str], bool]:
    """Read independent config-digest provenance without using it as identity.

    Returns:
        Sorted unique digests and whether their declarations were valid.
    """
    containers = list(_record_identity_containers(payload))
    records = payload.get("episodes")
    if isinstance(records, list):
        for record in records:
            if isinstance(record, Mapping):
                containers.extend(_record_identity_containers(record))
    values, invalid = _identity_values_from_containers(
        tuple(containers), _CANONICAL_CONFIG_DIGEST_KEYS
    )
    unique = sorted(set(values))
    if invalid:
        _add_diagnostic(diagnostics, "config_digest_mismatch", detail=artifact_id)
        return [], False
    return unique, True


def _load_source(  # noqa: C901, PLR0912, PLR0915
    ref: SourceRef,
    *,
    expected_format: str,
    expected_schema: str,
    root: Path,
    diagnostics: list[_Diagnostic],
    provenance: dict[str, Any] | None = None,
    expected_config_identity: str | None = None,
    expected_campaign_id: str | None = None,
) -> _LoadedSource | None:
    """Read, hash, schema-check, and admit one canonical source document.

    Returns:
        An admitted source or ``None`` when the source is unavailable.
    """
    provenance = provenance if provenance is not None else _source_provenance(ref)
    if not _validate_source_declarations(
        ref,
        tuple(sorted(_source_schema_values(expected_schema))),
        diagnostics,
        expected_config_identity=expected_config_identity,
    ):
        provenance["integrity_status"] = "declaration_invalid"
        return None
    if not _source_format_matches(ref, expected_format):
        _add_diagnostic(diagnostics, "source_format_mismatch", detail=ref.artifact_id)
        provenance["integrity_status"] = "format_mismatch"
        return None
    try:
        resolved = _resolve_source(ref.uri, root)
        canonical_manifest_schema: str | None = None
        if resolved.is_dir():
            files = _directory_files(resolved)
            with tempfile.TemporaryDirectory(prefix="srev06-source-snapshot-") as snapshot_dir:
                snapshot_files, observed_sha = _snapshot_directory(
                    resolved, files, Path(snapshot_dir)
                )
                if observed_sha.lower() != ref.sha256.lower():
                    _add_diagnostic(diagnostics, "source_digest_mismatch", detail=ref.artifact_id)
                    provenance["sha256_observed"] = observed_sha
                    provenance["integrity_status"] = "digest_mismatch"
                    return None
                if expected_format != CAMPAIGN_RESULT_FORMAT:
                    raise ReviewContractsValidationError(
                        [f"directory source is not supported for {expected_format}"]
                    )
                if expected_campaign_id is None:
                    raise ReviewContractsValidationError(["canonical campaign id is required"])
                payload, source_status, canonical_manifest_schema = _load_canonical_directory(
                    Path(snapshot_dir),
                    ref=ref,
                    expected_campaign_id=expected_campaign_id,
                    files=snapshot_files,
                )
            provenance["sha256_observed"] = observed_sha
            provenance["canonical_source"] = True
            provenance["canonical_manifest_schema"] = canonical_manifest_schema
        else:
            raw = _read_contained_bytes(ref.uri, root)
            observed_sha = _sha256_bytes(raw)
            provenance["sha256_observed"] = observed_sha
            if observed_sha.lower() != ref.sha256.lower():
                _add_diagnostic(diagnostics, "source_digest_mismatch", detail=ref.artifact_id)
                provenance["integrity_status"] = "digest_mismatch"
                return None
            try:
                payload = _load_json_bytes(raw, label=f"source {ref.artifact_id}")
            except ReviewContractsValidationError as error:
                _add_diagnostic(
                    diagnostics,
                    "source_not_json",
                    detail=f"{ref.artifact_id}:{'; '.join(error.errors)}",
                )
                provenance["integrity_status"] = "invalid_json"
                return None
            if not isinstance(payload, dict):
                _add_diagnostic(diagnostics, "source_not_json_object", detail=ref.artifact_id)
                provenance["integrity_status"] = "source_shape_invalid"
                return None
            source_status = _normalize_execution_status(payload.get("execution_status"))
            if source_status is None:
                _add_diagnostic(
                    diagnostics,
                    "source_execution_status_missing_or_invalid",
                    detail=ref.artifact_id,
                )
                provenance["integrity_status"] = "execution_status_invalid"
                return None
    except _CanonicalIdentityError as error:
        _add_diagnostic(
            diagnostics,
            error.code,
            detail=f"{ref.artifact_id}:{'; '.join(error.errors)}",
        )
        provenance["integrity_status"] = "canonical_identity_invalid"
        return None
    except ReviewContractsValidationError as error:
        if any("non_finite_number" in item for item in error.errors):
            _add_diagnostic(
                diagnostics,
                "source_non_finite_number",
                detail=f"{ref.artifact_id}:{'; '.join(error.errors)}",
            )
            provenance["integrity_status"] = "non_finite_json"
            return None
        _add_diagnostic(
            diagnostics,
            "source_unreadable_or_unsafe",
            detail=f"{ref.artifact_id}:{'; '.join(error.errors)}",
        )
        provenance["integrity_status"] = "unavailable"
        return None
    except (OSError, RuntimeError, ValueError) as error:
        _add_diagnostic(
            diagnostics,
            "source_unreadable_or_unsafe",
            detail=f"{ref.artifact_id}:{type(error).__name__}",
        )
        provenance["integrity_status"] = "unavailable"
        return None
    if payload.get("schema_version") not in _source_schema_values(expected_schema):
        _add_diagnostic(diagnostics, "source_payload_schema_mismatch", detail=ref.artifact_id)
        provenance["integrity_status"] = "schema_mismatch"
        return None
    if payload.get("schema_version") != ref.schema:
        _add_diagnostic(diagnostics, "source_payload_schema_mismatch", detail=ref.artifact_id)
        provenance["integrity_status"] = "provenance_mismatch"
        return None
    if (
        expected_format == CAMPAIGN_RESULT_FORMAT
        and expected_campaign_id is not None
        and not _validate_payload_campaign_identity(
            payload,
            expected_campaign_id=expected_campaign_id,
            diagnostics=diagnostics,
        )
    ):
        provenance["integrity_status"] = "provenance_mismatch"
        return None
    if not _validate_payload_identity(
        payload,
        ref=ref,
        diagnostics=diagnostics,
        required=not provenance.get("canonical_source", False),
        include_episode_rows=expected_format == CAMPAIGN_RESULT_FORMAT,
    ):
        provenance["integrity_status"] = "provenance_mismatch"
        return None
    config_digests, config_digest_valid = _payload_config_digest(
        payload, diagnostics, ref.artifact_id
    )
    if not config_digest_valid:
        provenance["integrity_status"] = "provenance_mismatch"
        return None
    provenance["integrity_status"] = "digest_and_schema_verified"
    provenance["execution_status"] = source_status
    provenance["config_digest"] = config_digests[0] if len(config_digests) == 1 else None
    provenance["config_digests"] = config_digests
    payload_provenance = payload.get("provenance")
    if isinstance(payload_provenance, dict):
        selected_provenance: dict[str, str | None] = {}
        for key in (
            "source_commit",
            "config_identity",
            "config_digest",
            "execution_status",
            "producer",
        ):
            if key not in payload_provenance:
                continue
            value = payload_provenance[key]
            if value is not None and not isinstance(value, str):
                _add_diagnostic(
                    diagnostics, "source_payload_provenance_invalid", detail=ref.artifact_id
                )
                provenance["integrity_status"] = "provenance_mismatch"
                return None
            selected_provenance[key] = value
        provenance["payload_provenance"] = selected_provenance
    return _LoadedSource(ref, payload, provenance, source_status)


def _canonical_ref(
    refs: list[SourceRef], *, family: str, config: dict[str, Any], diagnostics: list[_Diagnostic]
) -> SourceRef | None:
    """Select exactly one canonical source, never silently merge candidates.

    Returns:
        The selected source reference, or ``None`` on ambiguity.
    """
    if len(refs) == 1:
        return refs[0]
    selector_key = (
        "canonical_campaign_artifact_id"
        if family == CAMPAIGN_RESULT_FORMAT
        else "canonical_selection_artifact_id"
    )
    selector = config.get(selector_key)
    if not isinstance(selector, str) or not selector.strip():
        _add_diagnostic(diagnostics, "ambiguous_canonical_sources", detail=family)
        return None
    selected = [ref for ref in refs if ref.artifact_id == selector]
    if len(selected) != 1:
        _add_diagnostic(diagnostics, "canonical_source_not_found", detail=f"{family}:{selector}")
        return None
    return selected[0]


def _collect_sources(
    request: ComponentRequest, diagnostics: list[_Diagnostic]
) -> tuple[dict[str, list[SourceRef]], set[str], set[str]]:
    """Group only requested/required source families and report omissions.

    Returns:
        Grouped refs, missing required families, and skipped optional families.
    """
    required_families = set(REQUIRED_CAPABILITIES) | set(request.required_capabilities)
    by_format: dict[str, list[SourceRef]] = {}
    skipped_optional: set[str] = set()
    configured_skips = request.config.get("skip_optional_capabilities", [])
    if configured_skips is None:
        configured_skips = []
    if not isinstance(configured_skips, list) or not all(
        isinstance(item, str) and item in OPTIONAL_CAPABILITIES for item in configured_skips
    ):
        _add_diagnostic(
            diagnostics,
            "optional_skip_config_invalid",
            detail="skip_optional_capabilities must list known optional capabilities",
        )
        configured_skips = []
    configured_skips_set = set(configured_skips)
    for ref in request.sources:
        if ref.format not in _CAMPAIGN_RESULT_FORMATS | set(OPTIONAL_CAPABILITIES):
            _add_diagnostic(
                diagnostics,
                "unknown_source_format",
                detail=f"{ref.artifact_id}:{ref.format}",
            )
            continue
        family = CAMPAIGN_RESULT_FORMAT if ref.format in _CAMPAIGN_RESULT_FORMATS else ref.format
        if family in OPTIONAL_CAPABILITIES and family in configured_skips_set:
            skipped_optional.add(family)
            _add_diagnostic(
                diagnostics,
                "optional_stream_skipped",
                detail=f"{ref.artifact_id}:{family}",
            )
            continue
        by_format.setdefault(family, []).append(ref)
    for family in sorted(configured_skips_set - set(by_format)):
        skipped_optional.add(family)
        _add_diagnostic(diagnostics, "optional_stream_skipped", detail=family)
    missing = required_families - set(by_format)
    for family in sorted(missing):
        _add_diagnostic(diagnostics, "required_source_family_missing", detail=family)
    return by_format, missing, skipped_optional


def _row_status(
    item: dict[str, Any], source_status: str, index: int, diagnostics: list[_Diagnostic]
) -> str | None:
    """Resolve row execution status without treating descriptive text as status.

    Returns:
        The effective status, or ``None`` for malformed/conflicting status fields.
    """
    row_status = _normalize_execution_status(item["row_status"]) if "row_status" in item else None
    if "row_status" in item and row_status is None:
        _add_diagnostic(diagnostics, "episode_row_execution_status_invalid", detail=index)
        return None
    declared = [row_status] if row_status is not None else []
    for key in ("execution_status", "status"):
        if key not in item:
            continue
        normalized = _normalize_execution_status(item[key])
        if row_status is None:
            if normalized is None:
                _add_diagnostic(diagnostics, "episode_row_execution_status_invalid", detail=index)
                return None
            declared.append(normalized)
        elif normalized is not None and normalized != row_status:
            declared.append(normalized)
    if len(set(declared)) > 1:
        _add_diagnostic(diagnostics, "episode_row_execution_status_conflict", detail=index)
        return None
    if source_status in _NON_ADMISSIBLE_STATUSES:
        if declared and declared[0] != source_status:
            _add_diagnostic(
                diagnostics,
                "source_row_execution_status_conflict",
                detail=index,
            )
        return source_status
    return declared[0] if declared else source_status


def _text_field(item: dict[str, Any], *keys: str) -> str:
    """Return a non-empty text field or an explicit unknown identity value."""
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _outcome_label(item: Mapping[str, Any]) -> str:
    """Map a structured canonical outcome to a stable diagnostic label.

    Returns:
        A non-empty label suitable for the output outcome map.
    """
    outcome = item.get("outcome", "unknown")
    if isinstance(outcome, str) and outcome.strip():
        return outcome.strip()
    if isinstance(outcome, Mapping):
        label = outcome.get("label") or outcome.get("status") or outcome.get("termination_reason")
        if isinstance(label, str) and label.strip():
            return label.strip()
        if outcome.get("collision") is True:
            return "collision"
        if outcome.get("success") is True or outcome.get("reached_goal") is True:
            return "success"
    termination_reason = item.get("termination_reason")
    if isinstance(termination_reason, str) and termination_reason.strip():
        return termination_reason.strip()
    return "unknown"


def _parse_episodes(  # noqa: C901, PLR0912, PLR0915
    raw: dict[str, Any],
    *,
    campaign_id: str,
    source: _LoadedSource,
    diagnostics: list[_Diagnostic],
) -> tuple[list[_Episode], list[dict[str, str]], bool]:
    """Parse rows, exclude non-admissible execution modes, and reject conflicts.

    Returns:
        Admitted episodes, excluded-row records, and a conflict flag.
    """
    entries = raw.get("episodes")
    if not isinstance(entries, list) or not entries:
        _add_diagnostic(diagnostics, "episodes_missing_or_empty")
        return [], [], False
    episodes: list[_Episode] = []
    exclusions: list[dict[str, str]] = []
    seen_rows: dict[str, _Episode] = {}
    by_episode_id: dict[str, _Episode] = {}
    by_fingerprint: set[str] = set()
    conflicting = False
    for index, item in enumerate(entries):
        if not isinstance(item, dict):
            _add_diagnostic(diagnostics, "episode_row_malformed", detail=index)
            continue
        row_campaign = item.get("campaign_id")
        if row_campaign is not None and row_campaign != campaign_id:
            _add_diagnostic(diagnostics, "mixed_campaign_row", detail=index)
            conflicting = True
            continue
        episode_id = item.get("episode_id")
        if not isinstance(episode_id, str) or not episode_id.strip():
            _add_diagnostic(diagnostics, "episode_row_missing_id", detail=index)
            continue
        episode_id = episode_id.strip()
        seed = item.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            _add_diagnostic(diagnostics, "episode_row_seed_invalid", detail=episode_id)
            continue
        config = item.get("config", {})
        if config is not None and not isinstance(config, dict):
            _add_diagnostic(diagnostics, "episode_row_config_invalid", detail=episode_id)
            continue
        config_id = _text_field(config or {}, "config_id", "config_identity", "config_hash")
        if config_id == "unknown":
            config_id = _text_field(item, "config_id", "config_identity", "config_hash")
        metrics = item.get("metrics", {})
        if metrics is not None and not isinstance(metrics, dict):
            _add_diagnostic(diagnostics, "episode_row_metrics_invalid", detail=episode_id)
            continue
        outcome_value = _outcome_label(item)
        if not isinstance(outcome_value, str) or not outcome_value.strip():
            _add_diagnostic(diagnostics, "episode_row_outcome_invalid", detail=episode_id)
            continue
        execution_status = _row_status(item, source.execution_status, index, diagnostics)
        if execution_status is None:
            continue
        episode = _Episode(
            episode_id=episode_id,
            planner_id=_text_field(item, "planner_id", "planner", "algo"),
            scenario_id=_text_field(item, "scenario_id", "scenario"),
            seed=seed,
            config_id=config_id,
            outcome=outcome_value.strip(),
            metrics=dict(metrics or {}),
            execution_status=execution_status,
            source_artifact_id=source.ref.artifact_id,
        )
        previous = seen_rows.get(episode_id)
        if previous is not None:
            if (
                previous.identity != episode.identity
                or previous.fingerprint() != episode.fingerprint()
            ):
                _add_diagnostic(diagnostics, "conflicting_duplicate_episode", detail=episode_id)
                conflicting = True
            else:
                _add_diagnostic(
                    diagnostics,
                    "duplicate_episode_excerpt",
                    severity="info",
                    detail=episode_id,
                )
            continue
        seen_rows[episode_id] = episode
        if execution_status in _NON_ADMISSIBLE_STATUSES:
            exclusions.append(
                {
                    "episode_id": episode_id,
                    "status": execution_status,
                    "reason": "non_admissible_execution_status",
                }
            )
            _add_diagnostic(
                diagnostics,
                "row_excluded_non_admissible_status",
                detail=f"{episode_id}:{execution_status}",
            )
            continue
        fingerprint = episode.fingerprint()
        if fingerprint in by_fingerprint:
            _add_diagnostic(
                diagnostics,
                "duplicate_episode_excerpt",
                severity="info",
                detail=episode_id,
            )
            continue
        by_episode_id[episode_id] = episode
        by_fingerprint.add(fingerprint)
        episodes.append(episode)
    return episodes, exclusions, conflicting


def _parse_selection(
    source: _LoadedSource | None,
    *,
    episodes: list[_Episode],
    campaign_id: str,
    diagnostics: list[_Diagnostic],
) -> tuple[list[str], list[str], bool, str]:
    """Validate a selection document and split known and unknown episode IDs.

    Returns:
        Known IDs, unknown IDs, malformed-selection flag, and coverage status.
    """
    if source is None:
        return [], [], False, "not_supplied"
    if source.execution_status in _NON_ADMISSIBLE_STATUSES:
        _add_diagnostic(
            diagnostics,
            "required_selection_source_non_admissible",
            detail=source.execution_status,
        )
        return [], [], True, "unavailable"
    payload = source.payload
    supplied_campaign_id = payload.get("campaign_id")
    if not isinstance(supplied_campaign_id, str) or not supplied_campaign_id.strip():
        _add_diagnostic(diagnostics, "selection_campaign_unbound", detail=source.ref.artifact_id)
        return [], [], True, "invalid"
    if supplied_campaign_id.strip() != campaign_id:
        _add_diagnostic(diagnostics, "selection_campaign_mismatch", detail=source.ref.artifact_id)
        return [], [], True, "invalid"
    raw = payload.get("selected_episode_ids")
    if not isinstance(raw, list) or not all(isinstance(item, str) and item.strip() for item in raw):
        _add_diagnostic(diagnostics, "selection_ids_malformed", detail=source.ref.artifact_id)
        return [], [], True, "invalid"
    selected = [item.strip() for item in raw]
    if len(set(selected)) != len(selected):
        _add_diagnostic(diagnostics, "duplicate_selection_id", detail=source.ref.artifact_id)
    known_ids = {episode.episode_id for episode in episodes}
    unique_selected = list(dict.fromkeys(selected))
    known = [item for item in unique_selected if item in known_ids]
    unknown = [item for item in unique_selected if item not in known_ids]
    if unknown:
        _add_diagnostic(
            diagnostics,
            "unknown_selected_ids",
            detail=",".join(sorted(unknown)),
        )
    return known, unknown, False, "used"


def _bind_source_identities(
    campaign_source: _LoadedSource,
    selection_source: _LoadedSource | None,
    diagnostics: list[_Diagnostic],
) -> dict[str, Any]:
    """Bind loaded sources to one shared commit/config identity.

    Returns:
        The shared identity and the observed source digests.
    """
    identity = {
        "source_commit": campaign_source.ref.source_commit,
        "config_identity": campaign_source.ref.config_identity,
        "source_digests": [campaign_source.provenance["sha256_observed"]],
        "config_digests": sorted(
            {
                digest
                for source in (campaign_source, selection_source)
                if source is not None
                for digest in source.provenance.get("config_digests", [])
                if isinstance(digest, str)
            }
        ),
    }
    if selection_source is None:
        return identity
    for key in _IDENTITY_PROVENANCE_KEYS:
        if getattr(campaign_source.ref, key) != getattr(selection_source.ref, key):
            _add_diagnostic(diagnostics, "cross_source_identity_mismatch", detail=key)
    selection_digest = selection_source.provenance.get("sha256_observed")
    if not isinstance(selection_digest, str):
        _add_diagnostic(
            diagnostics,
            "cross_source_digest_unverified",
            detail=selection_source.ref.artifact_id,
        )
    else:
        identity["source_digests"].append(selection_digest)
    identity["source_digests"] = sorted(set(identity["source_digests"]))
    return identity


def _summarize_metric(
    name: str, episodes: list[_Episode], diagnostics: list[_Diagnostic]
) -> dict[str, Any]:
    """Summarize one metric while preserving denominator and invalid values.

    Returns:
        A missing-aware metric summary.
    """
    observed: list[float] = []
    invalid = 0
    for episode in episodes:
        if name not in episode.metrics:
            continue
        value = _finite_number(episode.metrics[name])
        if value is None:
            invalid += 1
        else:
            observed.append(value)
    if invalid:
        _add_diagnostic(diagnostics, "metric_value_invalid", detail=name)
    missing = len(episodes) - len(observed)
    summary: dict[str, Any] = {
        "count": len(observed),
        "missing": missing,
        "denominator": len(episodes),
    }
    if observed:
        ordered = sorted(observed)
        summary.update(
            {
                "min": ordered[0],
                "max": ordered[-1],
                "mean": sum(ordered) / len(ordered),
                "p25": _percentile(ordered, 0.25),
                "p50": _percentile(ordered, 0.50),
                "p75": _percentile(ordered, 0.75),
                "percentile_method": PERCENTILE_METHOD,
            }
        )
    return summary


def _build_report(  # noqa: PLR0913
    episodes: list[_Episode],
    selection: list[str],
    unknown_selection: list[str],
    *,
    selection_status: str,
    selection_source_artifact_id: str | None,
    campaign_id: str,
    campaign_source_artifact_id: str,
    exclusions: list[dict[str, str]],
    diagnostics: list[_Diagnostic],
    source_provenance: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the schema-validated diagnostic context document.

    Returns:
        A report document matching ``review-context.v1``.
    """
    seeds = sorted({episode.seed for episode in episodes if episode.seed is not None})
    planner_ids = sorted({episode.planner_id for episode in episodes})
    scenario_ids = sorted({episode.scenario_id for episode in episodes})
    config_ids = sorted({episode.config_id for episode in episodes})
    outcomes: dict[str, int] = {}
    for episode in episodes:
        outcomes[episode.outcome] = outcomes.get(episode.outcome, 0) + 1
    metric_names = sorted({name for episode in episodes for name in episode.metrics})
    metrics = {name: _summarize_metric(name, episodes, diagnostics) for name in metric_names}
    known_ids = {episode.episode_id for episode in episodes}
    covered = sorted(set(selection) & known_ids)
    by_status: dict[str, int] = {}
    for exclusion in exclusions:
        status = exclusion["status"]
        by_status[status] = by_status.get(status, 0) + 1
    return {
        "schema_version": "review-context.v1",
        "evidence_status": EVIDENCE_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "grain": {
            "planner_ids": planner_ids,
            "scenario_ids": scenario_ids,
            "seeds": seeds,
            "config_ids": config_ids,
            "episodes": len(episodes),
        },
        "denominator": len(episodes),
        "outcomes": dict(sorted(outcomes.items())),
        "metrics": metrics,
        "selection_coverage": {
            "status": selection_status,
            "selected": len(covered),
            "denominator": len(episodes),
            "unknown_selected_ids": sorted(unknown_selection),
            "source_artifact_id": selection_source_artifact_id,
        },
        "campaign": {
            "campaign_id": campaign_id,
            "source_artifact_id": campaign_source_artifact_id,
            "availability": {"status": STATUS_COMPLETE, "reason": ""},
        },
        "exclusions": {
            "count": len(exclusions),
            "by_status": dict(sorted(by_status.items())),
            "rows": exclusions,
        },
        "source_provenance": source_provenance,
    }


def _render_html(document: dict[str, Any]) -> str:
    """Render deterministic standalone HTML from a validated report.

    Returns:
        A standalone HTML document string.
    """
    rows = "\n".join(
        f"<tr><td>{html.escape(name)}</td>"
        f"<td>{metric.get('count', 0)}</td><td>{metric.get('missing', 0)}</td>"
        f"<td>{metric.get('min', '')}</td><td>{metric.get('p50', '')}</td>"
        f"<td>{metric.get('max', '')}</td></tr>"
        for name, metric in sorted(document["metrics"].items())
    )
    outcomes = "\n".join(
        f"<tr><td>{html.escape(str(outcome))}</td><td>{count}</td></tr>"
        for outcome, count in sorted(document["outcomes"].items())
    )
    exclusions = "\n".join(
        f"<tr><td>{html.escape(row['episode_id'])}</td>"
        f"<td>{html.escape(row['status'])}</td>"
        f"<td>{html.escape(row['reason'])}</td></tr>"
        for row in document["exclusions"]["rows"]
    )
    selection = document["selection_coverage"]
    unknown_selection = ", ".join(selection["unknown_selected_ids"]) or "none"
    campaign = document["campaign"]
    provenance_rows = "\n".join(
        "<tr>"
        f"<td>{html.escape(str(source['artifact_id']))}</td>"
        f"<td>{html.escape(str(source['format']))}</td>"
        f"<td>{html.escape(str(source['schema_declared']))}</td>"
        f"<td>{html.escape(str(source['source_commit'] or 'unavailable'))}</td>"
        f"<td>{html.escape(str(source['config_identity'] or 'unavailable'))}</td>"
        f"<td>{html.escape(str(source['config_digest'] or ', '.join(source['config_digests']) or 'unavailable'))}</td>"
        f"<td>{html.escape(str(source['sha256_observed'] or 'unavailable'))}</td>"
        f"<td>{html.escape(str(source['integrity_status']))}</td>"
        "</tr>"
        for source in document["source_provenance"]
    )
    grain = document["grain"]
    return (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        "<title>Review context report</title></head><body>"
        f"<h1>Review context report</h1><p>Diagnostic-only denominator: "
        f"{document['denominator']} admitted episodes; planners: "
        f"{html.escape(str(grain['planner_ids']))}; scenarios: "
        f"{html.escape(str(grain['scenario_ids']))}.</p>"
        "<h2>Campaign and selection</h2>"
        f"<p>Campaign: <code>{html.escape(campaign['campaign_id'])}</code>; "
        f"campaign source: <code>{html.escape(campaign['source_artifact_id'])}</code>; "
        f"availability: <code>{html.escape(campaign['availability']['status'])}</code>.</p>"
        f"<p>Selection status: <code>{html.escape(selection['status'])}</code>; "
        f"source: <code>{html.escape(str(selection['source_artifact_id'] or 'none'))}</code>; "
        f"coverage: {selection['selected']}/{selection['denominator']}; "
        f"unknown IDs: {html.escape(unknown_selection)}.</p>"
        "<h2>Outcomes</h2><table><tr><th>Outcome</th><th>Count</th></tr>"
        f"{outcomes}</table>"
        "<h2>Metrics</h2><table><tr><th>Metric</th><th>n</th><th>missing</th>"
        f"<th>min</th><th>p50</th><th>max</th></tr>{rows}</table>"
        "<h2>Source provenance</h2><table><tr><th>Artifact</th><th>Format</th>"
        "<th>Schema</th><th>Source commit</th><th>Config identity</th>"
        "<th>Config digest(s)</th>"
        f"<th>Observed digest</th><th>Integrity</th></tr>{provenance_rows}</table>"
        "<h2>Exclusions</h2><table><tr><th>Episode</th><th>Status</th>"
        f"<th>Reason</th></tr>{exclusions}</table>"
        f"<p>Excluded rows: {document['exclusions']['count']}; evidence status: "
        f"{html.escape(document['evidence_status'])}.</p>"
        "</body></html>\n"
    )


def _validate_output_document(document: dict[str, Any]) -> None:  # noqa: C901
    """Validate one leaf output against its registered machine-readable schema."""
    _validate_bounded_document(document, label="output")
    version = document.get("schema_version")
    schema = OUTPUT_SCHEMAS.get(version)
    if schema is None:
        raise ReviewContractsValidationError([f"unknown output schema: {version}"])
    errors = [
        error.message
        for error in sorted(
            Draft202012Validator(schema).iter_errors(document),
            key=lambda item: list(item.absolute_path),
        )
    ]
    if errors:
        raise ReviewContractsValidationError(errors)
    if version == "review-context.v1":
        denominator = document["denominator"]
        if sum(document["outcomes"].values()) != denominator:
            errors.append("outcomes must account for the admitted denominator")
        for name, summary in document["metrics"].items():
            if summary["count"] + summary["missing"] != summary["denominator"]:
                errors.append(f"metrics.{name} count plus missing must equal denominator")
        selection = document["selection_coverage"]
        if selection["denominator"] != denominator:
            errors.append("selection_coverage.denominator must equal report denominator")
        if selection["selected"] > selection["denominator"]:
            errors.append("selection_coverage.selected cannot exceed denominator")
        if selection["status"] == "used" and not selection["source_artifact_id"]:
            errors.append("used selection coverage requires source_artifact_id")
        if selection["status"] != "used" and selection["selected"]:
            errors.append("non-used selection coverage cannot report selected episodes")
        exclusions = document["exclusions"]
        if exclusions["count"] != len(exclusions["rows"]):
            errors.append("exclusions.count must equal the number of rows")
        if sum(exclusions["by_status"].values()) != exclusions["count"]:
            errors.append("exclusions.by_status must account for exclusions.count")
        for source in document["source_provenance"]:
            if (
                source["sha256_declared"] is not None
                and source["sha256_observed"] is not None
                and source["sha256_declared"].lower() != source["sha256_observed"].lower()
            ):
                errors.append(f"source digest mismatch: {source['artifact_id']}")
    if errors:
        raise ReviewContractsValidationError(errors)


def _open_output_temporary(directory_fd: int, prefix: str) -> tuple[int, str]:
    """Create a private temporary file through a trusted directory descriptor.

    Returns:
        The open temporary-file descriptor and its directory-relative name.
    """
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    for _ in range(128):
        name = f".{prefix}.{secrets.token_hex(16)}.partial"
        try:
            return os.open(name, flags, 0o600, dir_fd=directory_fd), name
        except FileExistsError:
            continue
    raise ReviewContractsValidationError(["output temporary name allocation failed"])


def _publish_output(
    directory_fd: int,
    temporary_name: str,
    final_name: str,
    output_directory: _ReservedOutputDirectory | None,
    published_names: set[str],
) -> None:
    """Publish one fsync'd temporary file without replacing an existing name."""
    try:
        # A hard-link publication is atomic on the same filesystem and
        # fails with EEXIST instead of replacing a final name or symlink.
        os.link(
            temporary_name,
            final_name,
            src_dir_fd=directory_fd,
            dst_dir_fd=directory_fd,
            follow_symlinks=False,
        )
    except FileExistsError as error:
        raise ReviewContractsValidationError(
            [f"output_collision: already exists: {final_name}"]
        ) from error
    except OSError as error:
        raise ReviewContractsValidationError(
            [f"output_atomic_materialization_failed: {type(error).__name__}"]
        ) from error

    published_names.add(final_name)
    try:
        os.unlink(temporary_name, dir_fd=directory_fd)
        if output_directory is not None:
            _assert_output_directory_current(output_directory)
        os.fsync(directory_fd)
    except BaseException:
        try:
            os.unlink(final_name, dir_fd=directory_fd)
        except OSError:
            pass
        published_names.discard(final_name)
        raise


def _atomic_materialize_no_replace(
    path: Path,
    text: str,
    *,
    output_directory: _ReservedOutputDirectory | None = None,
) -> str:
    """Publish UTF-8 text atomically through a retained no-follow directory fd.

    Returns:
        The SHA-256 digest of the published bytes.
    """
    encoded = text.encode("utf-8")
    owns_directory_fd = output_directory is None
    directory_fd = -1
    temporary_name: str | None = None
    temporary_fd = -1
    published_names = output_directory.published_names if output_directory is not None else set()
    try:
        if output_directory is None:
            directory_fd = os.open(path.parent, _output_directory_flags())
            if not stat.S_ISDIR(os.fstat(directory_fd).st_mode):
                raise ReviewContractsValidationError(
                    [f"output parent is not a real directory: {path.parent}"]
                )
        else:
            directory_fd = output_directory.directory_fd
            _assert_output_directory_current(output_directory)
        temporary_fd, temporary_name = _open_output_temporary(directory_fd, path.name)
        handle = os.fdopen(temporary_fd, "wb")
        temporary_fd = -1
        with handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        _publish_output(
            directory_fd,
            temporary_name,
            path.name,
            output_directory,
            published_names,
        )
        temporary_name = None
        return _sha256_bytes(encoded)
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        if temporary_name is not None and directory_fd >= 0:
            try:
                os.unlink(temporary_name, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
        if owns_directory_fd and directory_fd >= 0:
            os.close(directory_fd)


def _strict_json_text(payload: Any, *, label: str) -> str:
    """Return bounded JSON text while rejecting non-finite or non-JSON values."""
    _validate_bounded_document(payload, label=label)
    try:
        return json.dumps(payload, sort_keys=True, indent=2, allow_nan=False)
    except (TypeError, ValueError, OverflowError, UnicodeError) as error:
        raise ReviewContractsValidationError([f"{label} is not strict JSON"]) from error


def _write_json(
    path: Path,
    payload: dict[str, Any],
    *,
    output_directory: _ReservedOutputDirectory | None = None,
) -> str:
    """Write JSON to an atomic, exclusive final name and return its byte digest.

    Returns:
        The SHA-256 digest of the written bytes.
    """
    text = _strict_json_text(payload, label="output") + "\n"
    return _atomic_materialize_no_replace(path, text, output_directory=output_directory)


def _write_text_exclusive(
    path: Path,
    text: str,
    *,
    output_directory: _ReservedOutputDirectory | None = None,
) -> str:
    """Write text to an atomic, exclusive final name and return its byte digest.

    Returns:
        The SHA-256 digest of the written bytes.
    """
    return _atomic_materialize_no_replace(path, text, output_directory=output_directory)


def run(request: ComponentRequest, *, base: Path | None = None) -> ComponentResult:  # noqa: C901, PLR0912, PLR0915
    """Build one diagnostic-only cohort context report.

    The public API expects a validated :class:`ComponentRequest`; malformed
    direct calls still receive a typed failed result so callers never need to
    catch an implementation ``AttributeError``.

    Returns:
        A typed result with no complete artifacts unless all admission gates pass.
    """
    request_id, component_id = _request_identity(request)
    if not isinstance(request, ComponentRequest):
        return _result(
            request_id,
            component_id,
            STATUS_FAILED,
            reason="invalid_request: expected ComponentRequest",
        )
    if not isinstance(request.config, dict):
        return _result(
            request_id,
            component_id,
            STATUS_FAILED,
            reason="invalid_request: config must be an object",
            diagnostics=[_Diagnostic("invalid_request")],
        )
    rejected = _reject_not_applicable(request)
    if rejected is not None:
        return rejected
    diagnostics: list[_Diagnostic] = []
    source_provenance: list[dict[str, Any]] = []
    output_dir: _ReservedOutputDirectory | None = None
    root = base if base is not None else Path.cwd()
    base_provenance: dict[str, Any] = {
        "evidence_status": EVIDENCE_STATUS,
        "claim_boundary": CLAIM_BOUNDARY,
        "source_integrity": "unverified",
        "sources": source_provenance,
    }
    try:
        _validate_bounded_document(request.config, label="request config")
        if len(request.sources) > _MAX_OUTPUT_SOURCES:
            _add_diagnostic(
                diagnostics,
                "source_references_exceeded",
                detail=f"maximum {_MAX_OUTPUT_SOURCES}",
            )
            return _result(
                request.request_id,
                request.component_id,
                STATUS_FAILED,
                reason="source_references_exceeded",
                diagnostics=diagnostics,
                provenance=base_provenance,
            )
        campaign_id = request.config.get("campaign_id")
        if not isinstance(campaign_id, str) or not campaign_id.strip():
            return _result(
                request.request_id,
                request.component_id,
                STATUS_UNAVAILABLE,
                reason="campaign_context_unavailable: campaign_id is required",
                diagnostics=[
                    _Diagnostic("campaign_context_unavailable", "error", "campaign_id missing")
                ],
                provenance=base_provenance,
            )
        campaign_id = campaign_id.strip()
        config_identity = request.config.get("config_identity")
        if config_identity is not None and (
            not isinstance(config_identity, str) or not config_identity.strip()
        ):
            return _result(
                request.request_id,
                request.component_id,
                STATUS_FAILED,
                reason="invalid_config: config_identity must be a non-empty string",
                diagnostics=[_Diagnostic("config_identity_invalid")],
                provenance=base_provenance,
            )
        by_format, missing, skipped_optional = _collect_sources(request, diagnostics)
        if missing:
            _release_empty_output(output_dir)
            return _result(
                request.request_id,
                request.component_id,
                STATUS_UNAVAILABLE,
                reason=_reason_with_diagnostics(
                    "required_source_family_missing: " + ", ".join(sorted(missing)),
                    diagnostics,
                ),
                diagnostics=diagnostics,
                provenance=base_provenance,
            )
        output_dir = _reserve_output_directory(request.output_directory, root)
        campaign_ref = _canonical_ref(
            by_format[CAMPAIGN_RESULT_FORMAT],
            family=CAMPAIGN_RESULT_FORMAT,
            config=request.config,
            diagnostics=diagnostics,
        )
        selection_ref = None
        if EPISODE_SELECTION_FORMAT in by_format:
            selection_ref = _canonical_ref(
                by_format[EPISODE_SELECTION_FORMAT],
                family=EPISODE_SELECTION_FORMAT,
                config=request.config,
                diagnostics=diagnostics,
            )
        if campaign_ref is None:
            _release_empty_output(output_dir)
            return _result(
                request.request_id,
                request.component_id,
                STATUS_FAILED,
                reason=_reason_with_diagnostics(
                    "canonical_campaign_source_unavailable", diagnostics
                ),
                diagnostics=diagnostics,
                provenance=base_provenance,
            )
        campaign_provenance = _source_provenance(campaign_ref)
        source_provenance.append(campaign_provenance)
        campaign_source = _load_source(
            campaign_ref,
            expected_format=CAMPAIGN_RESULT_FORMAT,
            expected_schema=CAMPAIGN_RESULT_SCHEMA,
            root=root,
            diagnostics=diagnostics,
            provenance=campaign_provenance,
            expected_config_identity=config_identity,
            expected_campaign_id=campaign_id,
        )
        if campaign_source is None:
            _release_empty_output(output_dir)
            canonical_identity_failure = any(
                item.code.startswith("canonical_identity_") for item in diagnostics
            )
            return _result(
                request.request_id,
                request.component_id,
                STATUS_UNAVAILABLE if canonical_identity_failure else STATUS_FAILED,
                reason=_reason_with_diagnostics(
                    "canonical_campaign_source_unavailable"
                    if canonical_identity_failure
                    else "required_source_unavailable: campaign-result",
                    diagnostics,
                ),
                diagnostics=diagnostics,
                provenance=base_provenance,
            )
        source_campaign_id = campaign_source.payload.get("campaign_id")
        if source_campaign_id != campaign_id:
            _release_empty_output(output_dir)
            _add_diagnostic(
                diagnostics,
                "campaign_context_unavailable",
                detail=f"requested:{campaign_id}:source:{source_campaign_id}",
            )
            return _result(
                request.request_id,
                request.component_id,
                STATUS_UNAVAILABLE,
                reason="campaign_context_unavailable: requested campaign is not in canonical source",
                diagnostics=diagnostics,
                provenance=base_provenance,
            )
        selection_source: _LoadedSource | None = None
        if selection_ref is not None:
            selection_provenance = _source_provenance(selection_ref)
            source_provenance.append(selection_provenance)
            selection_source = _load_source(
                selection_ref,
                expected_format=EPISODE_SELECTION_FORMAT,
                expected_schema=EPISODE_SELECTION_SCHEMA,
                root=root,
                diagnostics=diagnostics,
                provenance=selection_provenance,
                expected_config_identity=config_identity,
            )
            if (
                selection_source is None
                and EPISODE_SELECTION_FORMAT in request.required_capabilities
            ):
                _release_empty_output(output_dir)
                return _result(
                    request.request_id,
                    request.component_id,
                    STATUS_UNAVAILABLE,
                    reason="required_source_unavailable: episode-selection",
                    diagnostics=diagnostics,
                    provenance=base_provenance,
                )
        source_identity = _bind_source_identities(campaign_source, selection_source, diagnostics)
        base_provenance["source_identity"] = source_identity
        if any(item.code == "cross_source_identity_mismatch" for item in diagnostics):
            _release_empty_output(output_dir)
            return _result(
                request.request_id,
                request.component_id,
                STATUS_FAILED,
                reason="cross_source_identity_mismatch",
                diagnostics=diagnostics,
                provenance=base_provenance,
            )
        episodes, exclusions, conflicting = _parse_episodes(
            campaign_source.payload,
            campaign_id=campaign_id,
            source=campaign_source,
            diagnostics=diagnostics,
        )
        if conflicting:
            _release_empty_output(output_dir)
            return _result(
                request.request_id,
                request.component_id,
                STATUS_FAILED,
                reason=_reason_with_diagnostics(
                    "conflicting_duplicate_episode_or_mixed_campaign", diagnostics
                ),
                diagnostics=diagnostics,
                provenance=base_provenance,
            )
        if any(item.code in _MALFORMED_EPISODE_DIAGNOSTICS for item in diagnostics):
            _release_empty_output(output_dir)
            return _result(
                request.request_id,
                request.component_id,
                STATUS_FAILED,
                reason=_reason_with_diagnostics("malformed_campaign_source", diagnostics),
                diagnostics=diagnostics,
                provenance=base_provenance,
            )
        if not episodes:
            _release_empty_output(output_dir)
            if exclusions:
                reason = "no_admissible_episodes: all rows excluded by execution status"
            else:
                reason = "no_episodes_indexed"
            return _result(
                request.request_id,
                request.component_id,
                STATUS_UNAVAILABLE,
                reason=reason,
                diagnostics=diagnostics,
                provenance=base_provenance,
            )
        selected, unknown, selection_invalid, selection_status = _parse_selection(
            selection_source,
            episodes=episodes,
            campaign_id=campaign_id,
            diagnostics=diagnostics,
        )
        if selection_ref is not None and selection_source is None:
            selection_status = "unavailable"
        elif selection_ref is None and EPISODE_SELECTION_FORMAT in skipped_optional:
            selection_status = "skipped"
        elif selection_ref is None and EPISODE_SELECTION_FORMAT in by_format:
            selection_status = "unavailable"
        if EPISODE_SELECTION_FORMAT in request.required_capabilities and selection_source is None:
            _release_empty_output(output_dir)
            _add_diagnostic(
                diagnostics, "required_source_unavailable", detail=EPISODE_SELECTION_FORMAT
            )
            return _result(
                request.request_id,
                request.component_id,
                STATUS_UNAVAILABLE,
                reason="required_source_unavailable: episode-selection",
                diagnostics=diagnostics,
                provenance=base_provenance,
            )
        if selection_invalid and EPISODE_SELECTION_FORMAT in request.required_capabilities:
            _release_empty_output(output_dir)
            return _result(
                request.request_id,
                request.component_id,
                STATUS_FAILED,
                reason=_reason_with_diagnostics("invalid_episode_selection", diagnostics),
                diagnostics=diagnostics,
                provenance=base_provenance,
            )
        selection_campaign_identity_invalid = any(
            item.code in {"selection_campaign_unbound", "selection_campaign_mismatch"}
            and selection_source is not None
            and item.detail == selection_source.ref.artifact_id
            for item in diagnostics
        )
        if selection_campaign_identity_invalid:
            _release_empty_output(output_dir)
            return _result(
                request.request_id,
                request.component_id,
                STATUS_FAILED,
                reason=_reason_with_diagnostics("invalid_episode_selection", diagnostics),
                diagnostics=diagnostics,
                provenance=base_provenance,
            )
        document = _build_report(
            episodes,
            selected,
            unknown,
            selection_status=selection_status,
            selection_source_artifact_id=(
                selection_source.ref.artifact_id if selection_source is not None else None
            ),
            campaign_id=campaign_id,
            campaign_source_artifact_id=campaign_source.ref.artifact_id,
            exclusions=exclusions,
            diagnostics=diagnostics,
            source_provenance=source_provenance,
        )
        capability_payload = {
            "schema_version": "missing-capability-report.v1",
            "missing_capabilities": sorted(
                set(OPTIONAL_CAPABILITIES) - set(by_format) - set(skipped_optional)
            ),
            "skipped_optional_streams": sorted(skipped_optional),
            "diagnostics": list(_diagnostic_documents(diagnostics)),
        }
        _validate_output_document(document)
        _validate_output_document(capability_payload)
        report_digest = _write_json(
            output_dir.path / OUTPUT_REPORT_FILENAME,
            document,
            output_directory=output_dir,
        )
        html_digest = _write_text_exclusive(
            output_dir.path / OUTPUT_HTML_FILENAME,
            _render_html(document),
            output_directory=output_dir,
        )
        capability_digest = _write_json(
            output_dir.path / OUTPUT_CAPABILITY_FILENAME,
            capability_payload,
            output_directory=output_dir,
        )
        _assert_output_directory_current(output_dir)
        base_provenance.update(
            {
                "output_directory": request.output_directory,
                "episodes": len(episodes),
                "denominator": len(episodes),
                "excluded_rows": len(exclusions),
                "output_artifacts": {
                    OUTPUT_REPORT_FILENAME: report_digest,
                    OUTPUT_HTML_FILENAME: html_digest,
                    OUTPUT_CAPABILITY_FILENAME: capability_digest,
                },
                "source_integrity": "digest_and_schema_verified",
                "source_identity": source_identity,
                "source_execution_status": {
                    source.ref.artifact_id: source.execution_status
                    for source in (campaign_source, selection_source)
                    if source is not None
                },
            }
        )
        blocking = [item for item in diagnostics if item.severity == "error"]
        status = STATUS_PARTIAL if blocking else STATUS_COMPLETE
        reason = "; ".join(sorted({item.code for item in blocking})[:8])
        artifacts: tuple[dict[str, Any], ...] = ()
        if status == STATUS_COMPLETE:
            artifacts = (
                {
                    "artifact_id": OUTPUT_REPORT_FILENAME,
                    "uri": str(Path(request.output_directory) / OUTPUT_REPORT_FILENAME),
                    "sha256": report_digest,
                },
                {
                    "artifact_id": OUTPUT_HTML_FILENAME,
                    "uri": str(Path(request.output_directory) / OUTPUT_HTML_FILENAME),
                    "sha256": html_digest,
                },
                {
                    "artifact_id": OUTPUT_CAPABILITY_FILENAME,
                    "uri": str(Path(request.output_directory) / OUTPUT_CAPABILITY_FILENAME),
                    "sha256": capability_digest,
                },
            )
        return _result(
            request.request_id,
            request.component_id,
            status,
            reason=reason,
            diagnostics=diagnostics,
            provenance=base_provenance,
            artifacts=artifacts,
        )
    except ReviewContractsValidationError as error:
        _release_empty_output(output_dir)
        for item in error.errors:
            _add_diagnostic(diagnostics, "contract_validation_error", detail=item)
        return _result(
            request_id,
            component_id,
            STATUS_FAILED,
            reason="; ".join(error.errors),
            diagnostics=diagnostics,
            provenance=base_provenance,
        )
    except (OSError, UnicodeError, TypeError, ValueError, OverflowError) as error:
        _release_empty_output(output_dir)
        _add_diagnostic(diagnostics, "internal_boundary_error", detail=type(error).__name__)
        return _result(
            request_id,
            component_id,
            STATUS_FAILED,
            reason=f"internal_boundary_error: {type(error).__name__}",
            diagnostics=diagnostics,
            provenance=base_provenance,
        )
    except Exception as error:  # noqa: BLE001
        _release_empty_output(output_dir)
        _add_diagnostic(diagnostics, "internal_boundary_error", detail=type(error).__name__)
        return _result(
            request_id,
            component_id,
            STATUS_FAILED,
            reason=f"internal_boundary_error: {type(error).__name__}",
            diagnostics=diagnostics,
            provenance=base_provenance,
        )
    finally:
        if output_dir is not None:
            output_dir.close()


def _result_document(result: ComponentResult) -> dict[str, Any]:
    """Serialize a result as a JSON-safe shared component-result.v1 document.

    Returns:
        A JSON-schema-compatible result mapping.
    """
    document = asdict(result)
    document["artifacts"] = [dict(item) for item in result.artifacts]
    document["diagnostics"] = [dict(item) for item in result.diagnostics]
    result_document = {"schema_version": COMPONENT_RESULT_SCHEMA_VERSION, **document}
    _validate_bounded_document(result_document, label="result")
    return result_document


def _cli_failure_result(payload: Any, reason: str) -> ComponentResult:
    """Build a safe typed failure result for malformed CLI input.

    Returns:
        A failed component result with safe identity fields.
    """
    request_id, component_id = _request_identity(payload)
    return _result(request_id, component_id, STATUS_FAILED, reason=reason)


def _serialized_cli_result(result: ComponentResult, payload: Any) -> tuple[ComponentResult, str]:
    """Serialize a CLI result strictly, replacing an invalid result with failure.

    Returns:
        The result that was serialized and its strict JSON representation.
    """
    try:
        return result, _strict_json_text(_result_document(result), label="result")
    except (
        ReviewContractsValidationError,
        TypeError,
        ValueError,
        OverflowError,
        UnicodeError,
    ) as error:
        fallback = _cli_failure_result(
            payload, f"result_serialization_error: {type(error).__name__}"
        )
        return fallback, _strict_json_text(_result_document(fallback), label="result")


def _read_control_document(path_value: str, *, label: str) -> Any:
    """Read one bounded, no-follow CLI control document.

    Returns:
        The strict, bounded decoded document.
    """
    raw = _read_regular_file_no_follow(
        Path(path_value), limit=_MAX_CONTROL_BYTES, kind=f"{label} control document"
    )
    return _load_json_bytes(raw, label=label)


def _build_parser() -> argparse.ArgumentParser:
    """Return the standalone CLI argument parser."""
    parser = argparse.ArgumentParser(description="Build diagnostic cohort context reports.")
    parser.add_argument("--input", required=True, help="Component request JSON file.")
    parser.add_argument("--config", required=False, default=None, help="Optional config JSON.")
    parser.add_argument("--output", required=True, help="Output directory (must not exist).")
    parser.add_argument("--base", required=False, default=None, help="Base dir for resolution.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and always print a schema-versioned result envelope.

    Returns:
        Process exit code, zero only for a complete result.
    """
    try:
        args = _build_parser().parse_args(argv)
    except SystemExit as error:
        if error.code == 0:
            raise
        result = _cli_failure_result(None, "invalid_cli_arguments")
        _, serialized = _serialized_cli_result(result, None)
        print(serialized)  # noqa: T201
        return 1
    payload: Any = None
    try:
        payload = _read_control_document(args.input, label="request")
        if not isinstance(payload, dict):
            raise ReviewContractsValidationError(["request must be a JSON object"])
        if args.config is not None:
            config = _read_control_document(args.config, label="config")
            if not isinstance(config, dict):
                raise ReviewContractsValidationError(["config must be a JSON object"])
            request_config = payload.get("config", {})
            if not isinstance(request_config, dict):
                raise ReviewContractsValidationError(["request config must be a JSON object"])
            payload = {**payload, "config": {**request_config, **config}}
        _validate_bounded_document(payload, label="request")
        payload = {**payload, "output_directory": args.output}
        request = component_request_from_dict(payload, source=args.input)
        result = run(request, base=Path(args.base) if args.base is not None else None)
    except ReviewContractsValidationError as error:
        result = _cli_failure_result(payload, "invalid_request: " + "; ".join(error.errors))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        result = _cli_failure_result(payload, f"invalid_request: {type(error).__name__}")
    except Exception as error:  # noqa: BLE001 - CLI must preserve the result contract
        result = _cli_failure_result(payload, f"internal_error: {type(error).__name__}")
    result, serialized = _serialized_cli_result(result, payload)
    print(serialized)  # noqa: T201
    return 0 if result.status == STATUS_COMPLETE else 1


if __name__ == "__main__":
    raise SystemExit(main())
