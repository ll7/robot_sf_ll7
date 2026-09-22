"""BA-06 audit workbench fixture and injectable BA-05 service adapter.

This module is deliberately a small UI seam, not a second audit engine.  The
browser receives a queue-shaped snapshot from an injected facade and sends
annotation/finding mutations back through that facade.  ``FixtureAuditService``
is explicitly diagnostic-only.  ``ServiceAuditWorkbenchFacade`` forwards the
same operations to a server-held BA-05 service session without importing or
duplicating service authority, queue selection, persistence, or source checks.

The emitted HTML is self-contained with respect to repository assets.  It
copies the existing SREV-16 panel and SREV-17 editor modules alongside the
new shell and uses relative imports only, so an offline launch never reaches a
remote CDN or silently falls back to a native/live service.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import threading
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import fields, is_dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

from robot_sf.analysis_workbench.review_contracts import SourceRef
from robot_sf.render.audit_trace_projection import project_retained_native_trace

AUDIT_WORKBENCH_SCHEMA_VERSION = "audit-workbench.v1"
AUDIT_WORKBENCH_COMPONENT_ID = "ba06-audit-workbench"
AUDIT_WORKBENCH_COMPONENT_VERSION = "0.1.0-fixture"
AUDIT_SERVICE_WORKBENCH_SCHEMA_VERSION = "audit-workbench-service.v1"
AUDIT_SERVICE_FACADE_ID = "ba06-audit-service-facade"
OUTPUT_MODEL_FILENAME = AUDIT_WORKBENCH_SCHEMA_VERSION + ".json"
OUTPUT_HTML_FILENAME = AUDIT_WORKBENCH_SCHEMA_VERSION + ".html"
FIXTURE_SERVICE_ID = "ba06-fixture-facade"


class AuditWorkbenchError(RuntimeError):
    """Base error for the narrow UI/facade contract."""


class AuditWorkbenchConflictError(AuditWorkbenchError):
    """Raised when a UI mutation targets a stale selection or record revision."""

    status = "conflict"

    def __init__(self, reason: str, *, expected: int | None = None, actual: int | None = None):
        """Create a conflict carrying the expected and observed revisions."""

        super().__init__(reason)
        self.reason = reason
        self.expected_revision = expected
        self.actual_revision = actual


class AuditWorkbenchUnavailableError(AuditWorkbenchError):
    """Raised when the selected case or requested service operation is unavailable."""

    status = "unavailable"


class AuditWorkbenchFacade(Protocol):
    """Small BA-05-shaped boundary consumed by the UI shell.

    A future service implementation can replace the fixture without changing
    the browser flow.  Methods return JSON-compatible mappings so a transport
    adapter can preserve service status and conflict details verbatim.
    """

    def next(self, *, expected_selection_revision: int) -> Mapping[str, Any]:
        """Select the next packet using compare-and-swap selection context."""

    def save_annotation(
        self,
        annotation: Mapping[str, Any],
        *,
        expected_selection_revision: int,
        expected_revision: int,
        operation_id: str,
    ) -> Mapping[str, Any]:
        """Persist one editor-owned annotation through the injected service."""

    def persist_finding(  # noqa: PLR0913
        self,
        annotation: Mapping[str, Any],
        *,
        expected_selection_revision: int,
        operation_id: str,
        expected_context: Any | None = None,
        expected_context_revision: int | None = None,
        expected_source_revision: int | str | None = None,
        expected_queue_state_revision: int | None = None,
        expected_queue_input_revision: int | None = None,
        expected_queue_input_identity: Any | None = None,
    ) -> Mapping[str, Any]:
        """Persist a finding derived from an already saved annotation."""

    def read_saved_records(
        self,
        episode_id: str | None = None,
        *,
        context: Any | None = None,
        operation_id: str | None = None,
    ) -> Mapping[str, Any]:
        """Read source-bound annotations/findings for a selected episode."""

    def snapshot(self) -> Mapping[str, Any]:
        """Return a durable/reopenable UI snapshot."""

    def read_selected_artifact_status(self) -> Mapping[str, Any]:
        """Return read-only selected artifact/native capability status."""

    def sync_finding(
        self,
        *,
        finding_id: str,
        repository: str,
        expected_finding_revision: int,
        expected_selection_revision: int,
        expected_context_revision: int,
        expected_source_revision: int | str,
        operation_id: str,
        retry_ambiguous: bool = False,
    ) -> Mapping[str, Any]:
        """Publish one canonical finding through the server-held service."""


class AuditCodexClientProtocol(Protocol):
    """Narrow server-held BA-05 Codex lifecycle seam."""

    def start(
        self,
        audit_session: Any,
        *,
        prompt: str,
        context: Any,
        operation_id: str,
        token_budget: int,
        compute_budget: float,
        audit_token: str,
    ) -> Any:
        """Admit one context-bound Codex turn."""

    def cancel(
        self,
        audit_session: Any,
        *,
        operation_id: str,
        reason: str,
        audit_token: str,
    ) -> Any:
        """Cancel one already-returned Codex session."""

    def reconnect(self, codex_session_id: str, *, operation_id: str, audit_token: str) -> Any:
        """Reconnect one durable provider session through BA-05."""


_SENSITIVE_KEYS = frozenset(
    {
        "api_key",
        "apikey",
        "auth",
        "authorization",
        "bearer",
        "client_secret",
        "clientsecret",
        "credential",
        "credentials",
        "oauth",
        "password",
        "provider_key",
        "refresh_token",
        "filesystem_path",
        "output_directory",
        "output_root",
        "path",
        "private_root",
        "secret",
        "source_root",
        "session_token",
        "token",
    }
)
_SAFE_URI_SCHEMES = frozenset({"artifact", "http", "https", "s3", "urn"})
_REDACTED_LOCAL_URI = "<redacted-local-uri>"
_SOURCE_REF_FIELD_NAMES = tuple(field.name for field in fields(SourceRef))
_SOURCE_REF_REQUIRED_FIELD_NAMES = ("artifact_id", "uri", "format")


def _sensitive_key(value: Any) -> bool:
    """Return whether a mapping key can carry a secret or authority."""

    if not isinstance(value, str):
        return False
    normalized = value.casefold().replace("-", "_")
    compact = normalized.replace("_", "")
    return (
        normalized in _SENSITIVE_KEYS
        or normalized.endswith("_token")
        or normalized.endswith("_credential")
        or normalized.endswith(("_path", "_root", "_directory"))
        or compact in {"accesstoken", "apikey", "clientsecret", "providertoken", "sessiontoken"}
    )


def _safe_service_key(value: Any, *, secret: str | None = None) -> str:
    """Redact a server secret if it appears in an otherwise safe key.

    Returns:
        A string key with any known server secret replaced.
    """

    key = str(value)
    return key.replace(secret, "<redacted>") if secret and secret in key else key


def _uri_field(value: Any) -> bool:
    """Return whether a field carries a URI/path-like browser reference."""

    normalized = str(value).casefold().replace("-", "_")
    return normalized == "uri" or normalized.endswith("_uri")


def _uri_is_local_path(value: str) -> bool:
    """Return whether a URI string is shaped like a local filesystem path."""

    return value.startswith(("/", "\\", "~/", "~\\")) or (
        len(value) >= 2 and value[0].isalpha() and value[1] == ":"
    )


def _uri_has_traversal(value: str) -> bool:
    """Return whether a decoded URI contains path separators or traversal."""

    return "\\" in value or any(part in {".", ".."} for part in value.split("/"))


def _uri_is_local_or_traversal(value: str) -> bool:
    """Return whether URI text names local filesystem state or traversal."""

    return _uri_is_local_path(value) or _uri_has_traversal(value)


def _uri_has_ambiguous_text(value: str) -> bool:
    """Return whether URI text needs normalization before it can be trusted."""

    return any(
        character.isspace() or ord(character) < 0x20 or ord(character) == 0x7F
        for character in value
    )


def _uri_has_sensitive_parts(parsed: Any) -> bool:
    """Return whether a parsed URI carries credentials or hidden components."""

    try:
        return bool(parsed.query or parsed.fragment or parsed.username or parsed.password)
    except ValueError:
        return True


def _uri_is_local_host(parsed: Any) -> bool:
    """Return whether an HTTP URI targets a loopback host."""

    if parsed.scheme.casefold() not in {"http", "https"}:
        return False
    try:
        hostname = (parsed.hostname or "").casefold().rstrip(".")
    except ValueError:
        return True
    return hostname in {"localhost", "127.0.0.1", "::1"}


def _safe_service_uri(value: Any, *, secret: str | None = None) -> Any:
    """Keep logical/non-local identifiers while redacting local or uncertain URIs.

    ``SourceRef.uri`` is a valid contract field even when it contains an
    absolute local path.  Browser payloads may retain relative logical IDs and
    credential-free non-local URLs, but local paths, traversal, credentials,
    query-bearing URLs, and unknown URI schemes are not useful durable UI
    authority and are redacted.

    Returns:
        A safe URI string or a recursively sanitized non-string value.
    """

    if not isinstance(value, str):
        return _safe_service_value(value, secret=secret)
    safe = value.replace(secret, "<redacted>") if secret and secret in value else value
    if not safe:
        return safe
    if _uri_has_ambiguous_text(safe) or "%" in safe:
        # Percent-encoded delimiters, paths, credentials, and secrets cannot
        # be distinguished safely after this boundary.  Reject all encoded
        # URI text rather than echoing a value whose browser interpretation
        # differs from the server-side spelling.
        return _REDACTED_LOCAL_URI
    if _uri_is_local_or_traversal(safe):
        return _REDACTED_LOCAL_URI
    try:
        parsed = urlsplit(safe)
    except ValueError:
        return _REDACTED_LOCAL_URI
    scheme = parsed.scheme.casefold()
    if _uri_has_sensitive_parts(parsed):
        return _REDACTED_LOCAL_URI
    if scheme == "file" or (scheme and scheme not in _SAFE_URI_SCHEMES):
        return _REDACTED_LOCAL_URI
    if _uri_is_local_host(parsed):
        return _REDACTED_LOCAL_URI
    if scheme in {"artifact", "http", "https", "s3"} and not parsed.netloc:
        return _REDACTED_LOCAL_URI
    return safe


def _safe_service_value(value: Any, *, secret: str | None = None, depth: int = 0) -> Any:
    """Convert a service value to bounded JSON without serializing authority.

    BA-05 result envelopes are intentionally accepted without importing BA-05
    types.  This keeps the BA-06 render package usable on the pre-integration
    branch while still accepting dataclasses and ``to_dict`` implementations
    from the service branch.  Unknown objects are omitted rather than rendered
    through ``repr``; exception/session reprs are a common accidental secret
    disclosure path.

    Returns:
        A bounded JSON-compatible value, with sensitive fields omitted.
    """

    if depth > 16:
        return None
    if value is None or isinstance(value, (bool, int, float, str)):
        if secret and isinstance(value, str) and secret in value:
            return value.replace(secret, "<redacted>")
        return value
    if isinstance(value, Mapping):
        return {
            _safe_service_key(key, secret=secret): (
                _safe_service_uri(item, secret=secret)
                if _uri_field(key)
                else _safe_service_value(item, secret=secret, depth=depth + 1)
            )
            for key, item in value.items()
            if not _sensitive_key(key)
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_safe_service_value(item, secret=secret, depth=depth + 1) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: (
                _safe_service_uri(getattr(value, field.name), secret=secret)
                if _uri_field(field.name)
                else _safe_service_value(getattr(value, field.name), secret=secret, depth=depth + 1)
            )
            for field in fields(value)
            if not _sensitive_key(field.name)
        }
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            converted = to_dict()
        except (AttributeError, TypeError, ValueError, RuntimeError):
            converted = None
        if converted is not None and converted is not value:
            return _safe_service_value(converted, secret=secret, depth=depth + 1)
    return None


def _service_attr(value: Any, name: str, default: Any = None) -> Any:
    """Read a field from either a typed service envelope or a mapping.

    Returns:
        The field value or ``default`` when the field is absent.
    """

    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _service_result_mapping(result: Any, *, secret: str | None = None) -> dict[str, Any]:
    """Project one BA-05 ``ServiceResult`` while keeping its envelope fields.

    Returns:
        A safe mapping containing status, reason, value, operation, and context.
    """

    raw: Any
    if isinstance(result, Mapping):
        raw = dict(result)
    else:
        to_dict = getattr(result, "to_dict", None)
        if callable(to_dict):
            try:
                raw = to_dict()
            except (AttributeError, TypeError, ValueError, RuntimeError):
                raw = {}
        else:
            raw = {}
    mapped = _safe_service_value(raw, secret=secret)
    if not isinstance(mapped, dict):
        mapped = {}
    status = _service_attr(result, "status", mapped.get("status", "unavailable"))
    reason = _service_attr(result, "reason", mapped.get("reason", ""))
    value = _service_attr(result, "value", mapped.get("value"))
    operation = _service_attr(result, "operation", mapped.get("operation"))
    context = _service_attr(result, "context", mapped.get("context"))
    mapped["schema_version"] = mapped.get("schema_version", AUDIT_SERVICE_WORKBENCH_SCHEMA_VERSION)
    safe_status = _safe_service_value(status, secret=secret)
    safe_reason = _safe_service_value(reason, secret=secret)
    mapped["status"] = str(safe_status if safe_status is not None else "unavailable")
    if isinstance(safe_reason, str):
        mapped["reason"] = safe_reason
    else:
        # Keep typed and mapping envelopes on the same redaction path.  In
        # particular, never overwrite the already-safe ``to_dict`` projection
        # with a raw typed ``ServiceResult.reason`` value.
        mapped["reason"] = str(_safe_service_value(str(safe_reason or ""), secret=secret))
    mapped["value"] = _safe_service_value(value, secret=secret)
    mapped["operation"] = _safe_service_value(operation, secret=secret)
    mapped["context"] = _safe_service_value(context, secret=secret)
    return mapped


def _result_succeeded(result: Any) -> bool:
    """Return whether a service envelope completed without flattening status."""

    return str(_service_attr(result, "status", "")) in {
        "complete",
        "committed",
        "ok",
        "saved",
        "selected",
    }


def _is_revision(value: Any) -> bool:
    """Return whether a service revision is a non-negative integer."""

    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _context_matches_request(requested: Any, returned: Any) -> bool:
    """Check returned context revision and stable selection identity.

    Returns:
        ``True`` only when the returned context matches the requested
        revision and identity fields.
    """

    requested_revision = _service_attr(requested, "context_revision")
    returned_revision = _service_attr(returned, "context_revision")
    if not _is_revision(requested_revision) or returned_revision != requested_revision:
        return False
    for field_name in (
        "campaign_id",
        "execution_id",
        "episode_id",
        "interval_id",
        "reference_id",
        "actor_id",
        "source_identity",
        "source_revision",
    ):
        expected = _service_attr(requested, field_name)
        actual = _service_attr(returned, field_name)
        if expected is None:
            continue
        if expected == "" and actual in (None, ""):
            # Typed BA-05 contexts use empty strings for optional fields while
            # their JSON projection uses null.  Both are the same absent
            # value; do not reject an otherwise exact Codex result binding.
            continue
        if field_name == "source_revision" and expected in (0, ""):
            continue
        if actual != expected:
            return False
    return True


_CONTEXT_UNSET = object()


def _validated_commit_result(
    result: Any,
    *,
    requested_record_type: str,
    requested_record_id: str,
    requested_operation_id: str | None,
    requested_operation_type: str,
) -> Any | None:
    """Return a canonical commit receipt only when all write identities agree.

    The BA-05 service result and its operation receipt are separate authority
    records.  Both must say ``committed`` and identify the same operation; the
    durable ``CommitResult`` must also be non-deleted, well-typed, and match
    the record requested by this facade method.
    """

    if str(_service_attr(result, "status", "")) != "committed":
        return None
    value = _service_attr(result, "value")
    operation = _service_attr(result, "operation")
    if value is None or operation is None:
        return None
    if _service_attr(value, "committed", False) is not True:
        return None
    if _service_attr(value, "deleted", None) is not False:
        return None
    operation_id = _service_attr(value, "operation_id")
    record_id = _service_attr(value, "record_id")
    record_type = _service_attr(value, "record_type")
    revision = _service_attr(value, "revision")
    global_revision = _service_attr(value, "global_revision")
    if (
        not isinstance(operation_id, str)
        or not operation_id
        or not isinstance(record_id, str)
        or not record_id
        or not isinstance(record_type, str)
        or not record_type
        or not _is_revision(revision)
        or not _is_revision(global_revision)
    ):
        return None
    if record_type != requested_record_type or record_id != requested_record_id:
        return None
    if requested_operation_id is not None and operation_id != requested_operation_id:
        return None
    operation_receipt_id = _service_attr(operation, "operation_id")
    operation_status = _service_attr(operation, "status")
    operation_type = _service_attr(operation, "operation_type")
    if (
        operation_receipt_id != operation_id
        or operation_status != "committed"
        or operation_type != requested_operation_type
    ):
        return None
    return value


def _write_result_mapping(
    result: Any,
    *,
    secret: str | None = None,
    operation: str,
    requested_record_type: str,
    requested_record_id: str,
    requested_operation_id: str | None,
    requested_operation_type: str,
    expected_context: Any = _CONTEXT_UNSET,
) -> dict[str, Any]:
    """Project a write result and fail closed when its receipt is not canonical.

    Returns:
        A safe service envelope whose presentation status is derived solely
        from the validated commit receipt.
    """

    envelope = _service_result_mapping(result, secret=secret)
    envelope.pop("presentation_status", None)
    # A service mapping may contain stale or forged browser-facing receipt
    # fields alongside an otherwise valid typed ``CommitResult``.  Remove all
    # write-authority projections before adding the receipt proved below.
    receipt_fields = (
        "receipt",
        "operation_id",
        "record_id",
        "record_type",
        "revision",
        "global_revision",
        "checkpoint_revision",
        "committed",
        "deleted",
        "replayed",
    )
    for field_name in receipt_fields:
        envelope.pop(field_name, None)
    commit = _validated_commit_result(
        result,
        requested_record_type=requested_record_type,
        requested_record_id=requested_record_id,
        requested_operation_id=requested_operation_id,
        requested_operation_type=requested_operation_type,
    )
    if commit is not None:
        receipt = _value_mapping(commit, secret=secret)
        required_receipt_fields = (
            "committed",
            "deleted",
            "operation_id",
            "record_id",
            "record_type",
            "revision",
            "global_revision",
        )
        if not all(field_name in receipt for field_name in required_receipt_fields):
            commit = None
        else:
            envelope["receipt"] = receipt
            for field_name in receipt_fields:
                if field_name != "receipt" and field_name in receipt:
                    envelope[field_name] = receipt[field_name]
    context_matches = True
    returned_context = _service_attr(result, "context")
    if expected_context is not _CONTEXT_UNSET:
        context_matches = returned_context is not None and _context_matches_request(
            expected_context, returned_context
        )
    if commit is not None and not context_matches:
        envelope["service_status"] = envelope.get("status")
        envelope["status"] = "unavailable"
        envelope["reason"] = f"{operation} returned a context that does not match the request"
        envelope["conflict"] = {
            "expected_context": _safe_service_value(expected_context, secret=secret),
            "actual_context": _safe_service_value(returned_context, secret=secret),
        }
        envelope["presentation_status"] = "unavailable"
        return envelope
    if commit is not None:
        envelope["presentation_status"] = "saved"
        return envelope
    if _result_succeeded(result):
        envelope["service_status"] = envelope.get("status")
        envelope["status"] = "unavailable"
        envelope["reason"] = f"{operation} did not return a committed CommitResult"
    envelope["presentation_status"] = {
        "conflict": "conflict",
        "denied": "denied",
        "failed": "failed",
        "unavailable": "unavailable",
    }.get(str(envelope.get("status")), "unavailable")
    return envelope


def _inspection_available(result: Any) -> bool:
    """Return whether a successful episode envelope carries readable data."""

    if not _result_succeeded(result):
        return False
    value_status = _service_attr(_service_attr(result, "value"), "status")
    return value_status not in {"missing", "unavailable", "error"}


def _value_mapping(value: Any, *, secret: str | None = None) -> dict[str, Any]:
    """Return a safe mapping for a typed service value."""

    mapped = _safe_service_value(value, secret=secret)
    return mapped if isinstance(mapped, dict) else {}


def _local_service_result(
    status: str,
    reason: str,
    *,
    context: Any = None,
    conflict: Mapping[str, Any] | None = None,
    secret: str | None = None,
) -> dict[str, Any]:
    """Build a local envelope for authority/request failures.

    Returns:
        A service-shaped failure envelope.
    """

    safe_reason = _safe_service_value(reason, secret=secret)
    result: dict[str, Any] = {
        "schema_version": AUDIT_SERVICE_WORKBENCH_SCHEMA_VERSION,
        "status": _safe_service_value(status, secret=secret),
        "reason": safe_reason if isinstance(safe_reason, str) else "",
        "value": None,
        "operation": None,
        "context": _safe_service_value(context, secret=secret),
    }
    if conflict is not None:
        result["conflict"] = _safe_service_value(conflict, secret=secret)
    return result


_ARTIFACT_STATUS_SCHEMA_VERSION = "audit-artifact-status.v1"
_ARTIFACT_RESULT_STATUSES = frozenset(
    {
        "complete",
        "committed",
        "ok",
        "selected",
        "conflict",
        "unavailable",
        "denied",
        "failed",
        "cancelled",
    }
)
_ARTIFACT_SUCCESS_RESULT_STATUSES = frozenset({"complete", "committed", "ok", "selected"})
_ARTIFACT_CAPABILITY_STATUSES = frozenset(
    {"available", "unavailable", "not_configured", "no_selection"}
)
_ARTIFACT_CLASSIFICATIONS = frozenset({"historical_original", "derived_render", "unavailable"})
_ARTIFACT_FIDELITIES = frozenset({"verified", "diverged", "unverifiable", "unavailable"})


def _artifact_safe_reference(
    value: Any, *, limit: int = 256, secret: str | None = None
) -> str | int | None:
    """Keep only bounded opaque status identifiers, never paths or roots.

    Returns:
        A safe integer/string reference or ``None``.
    """

    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if not isinstance(value, str):
        return None
    value = value.strip()
    if (
        not value
        or len(value) > limit
        or (secret is not None and secret in value)
        or "%" in value
        or "/" in value
        or "\\" in value
    ):
        return None
    return value


def _artifact_reason(value: Any, *, secret: str | None = None) -> str:
    """Keep a bounded human reason without echoing path-like text.

    Returns:
        A safe reason suitable for a browser status line.
    """

    if not isinstance(value, str):
        return "artifact status is unavailable"
    value = value.strip()
    if (
        not value
        or len(value) > 256
        or (secret is not None and secret in value)
        or "%" in value
        or "/" in value
        or "\\" in value
    ):
        return "artifact status is unavailable"
    return value


def _artifact_reason_is_unsafe(value: Any, *, secret: str | None = None) -> bool:
    """Return whether a non-empty reason is not safe for a status envelope."""

    if value in (None, ""):
        return False
    return _artifact_reason(value, secret=secret) != str(value).strip()


def _artifact_status_mapping(value: Any) -> Mapping[str, Any]:
    """Unwrap one status value without retaining its generic service envelope.

    Returns:
        The status mapping, or an empty mapping for unsupported input.
    """

    candidate = value
    if isinstance(candidate, Mapping) and isinstance(candidate.get("value"), Mapping):
        candidate = candidate["value"]
    return candidate if isinstance(candidate, Mapping) else {}


def _artifact_status_source_identity_token(value: Any) -> str | None:
    """Extract one opaque source identity from an editor-model identity map.

    Only digest and artifact-id fields are eligible.  URI/path fields are
    intentionally ignored so fixture binding cannot turn source metadata into
    a rendered or serialized path reference.

    Returns:
        One digest/artifact identifier, or ``None`` when no safe candidate exists.
    """

    if isinstance(value, str):
        return value
    if not isinstance(value, Mapping):
        return None
    sources = value.get("sources")
    if isinstance(sources, Mapping):
        for item in sources.values():
            if isinstance(item, Mapping):
                for field_name in ("sha256", "artifact_id"):
                    identity = item.get(field_name)
                    if identity:
                        return str(identity)
    for field_name in ("sha256", "artifact_id"):
        identity = value.get(field_name)
        if identity:
            return str(identity)
    return None


def _artifact_binding_consensus(field_name: str, *values: Any) -> tuple[Any, bool]:
    """Select one safe binding value only when all supplied aliases agree.

    Returns:
        The normalized value and whether supplied aliases disagree or are unsafe.
    """

    present = [value for value in values if value not in (None, "")]
    if not present:
        return None, False
    normalized = [
        _artifact_status_source_identity_token(value) if field_name == "source_digest" else value
        for value in present
    ]
    safe_values = [_artifact_safe_reference(value) for value in normalized]
    first = safe_values[0]
    conflict = first is None or any(value is None or value != first for value in safe_values[1:])
    return first, conflict


def _artifact_status_binding_mismatch(
    value: Any,
    *,
    episode_id: Any = None,
    context_revision: Any = None,
    source_revision: Any = None,
    source_digest: Any = None,
    require_complete_binding: bool = False,
    binding_conflict: bool = False,
    secret: str | None = None,
) -> bool:
    """Reject status values that cannot be proven to describe this selection.

    Returns:
        Whether the value is missing or mismatched against the expected binding.
    """

    if binding_conflict:
        return True
    candidate = _artifact_status_mapping(value)
    expected = {
        "episode_id": episode_id,
        "context_revision": context_revision,
        "source_revision": source_revision,
        "source_digest": source_digest,
    }
    if require_complete_binding and any(
        _artifact_safe_reference(expected_value, secret=secret) is None
        for expected_value in expected.values()
    ):
        return True
    for field_name, expected_value in expected.items():
        safe_expected = _artifact_safe_reference(expected_value, secret=secret)
        raw_value = candidate.get(field_name)
        if expected_value not in (None, "") and safe_expected is None:
            return True
        if raw_value is not None and _artifact_safe_reference(raw_value, secret=secret) is None:
            return True
        if safe_expected is not None:
            if (
                raw_value is None
                or _artifact_safe_reference(raw_value, secret=secret) != safe_expected
            ):
                return True
    return False


def _artifact_status_projection(
    value: Any,
    *,
    episode_id: Any = None,
    context_revision: Any = None,
    source_revision: Any = None,
    source_digest: Any = None,
    require_complete_binding: bool = False,
    binding_conflict: bool = False,
    secret: str | None = None,
) -> dict[str, Any]:
    """Normalize service-owned artifact status to the browser-safe vocabulary.

    Returns:
        A redacted, diagnostic-only status projection.
    """

    candidate = _artifact_status_mapping(value)
    expected_binding = {
        "episode_id": _artifact_safe_reference(episode_id, secret=secret),
        "context_revision": _artifact_safe_reference(context_revision, secret=secret),
        "source_revision": _artifact_safe_reference(source_revision, secret=secret),
        "source_digest": _artifact_safe_reference(source_digest, secret=secret),
    }
    binding_mismatch = _artifact_status_binding_mismatch(
        value,
        episode_id=episode_id,
        context_revision=context_revision,
        source_revision=source_revision,
        source_digest=source_digest,
        require_complete_binding=require_complete_binding,
        binding_conflict=binding_conflict,
        secret=secret,
    )
    if binding_mismatch:
        candidate = {}
    selected_episode = _artifact_safe_reference(candidate.get("episode_id"), secret=secret)
    if expected_binding["episode_id"] is not None:
        selected_episode = expected_binding["episode_id"]
    has_selection = selected_episode is not None
    fallback_status = "unavailable" if has_selection else "no_selection"
    fallback_reason = (
        "selected artifact status is unavailable" if has_selection else "no selected episode"
    )

    def capability(raw: Any, *, native: bool) -> dict[str, Any]:
        raw = raw if isinstance(raw, Mapping) else {}
        status = raw.get("status")
        raw_reason = raw.get("reason")
        safe_reason = _artifact_reason(raw_reason, secret=secret)
        unsafe_reason = raw_reason is not None and safe_reason != str(raw_reason).strip()
        raw_result_status = raw.get("result_status")
        safe_result_status = _artifact_safe_reference(raw_result_status, limit=64, secret=secret)
        unsafe_result_status = raw_result_status is not None and safe_result_status is None
        invalid_status = not isinstance(status, str) or status not in _ARTIFACT_CAPABILITY_STATUSES
        classification = raw.get("classification")
        invalid_classification = classification is not None and (
            not isinstance(classification, str) or classification not in _ARTIFACT_CLASSIFICATIONS
        )
        fidelity = raw.get("fidelity")
        invalid_fidelity = fidelity is not None and (
            not isinstance(fidelity, str) or fidelity not in _ARTIFACT_FIDELITIES
        )
        unsafe_value = (
            invalid_status
            or unsafe_reason
            or unsafe_result_status
            or invalid_classification
            or invalid_fidelity
        )
        if not isinstance(status, str) or status not in _ARTIFACT_CAPABILITY_STATUSES:
            status = fallback_status
        if unsafe_value:
            status = fallback_status
        projected: dict[str, Any] = {
            "status": status,
            "reason": fallback_reason if unsafe_value else (safe_reason or fallback_reason),
            "diagnostic_only": True,
        }
        if not native:
            projected["classification"] = (
                classification
                if not unsafe_value and classification in _ARTIFACT_CLASSIFICATIONS
                else None
            )
            projected["fidelity"] = (
                fidelity if not unsafe_value and fidelity in _ARTIFACT_FIDELITIES else None
            )
        else:
            projected["evidence_boundary"] = "diagnostic_only"
            projected["scientific_claim_allowed"] = False
        if safe_result_status is not None and not unsafe_value:
            projected["result_status"] = safe_result_status
        if not unsafe_value and isinstance(raw.get("simulation_executed"), bool):
            projected["simulation_executed"] = raw["simulation_executed"]
        return projected

    projected_status = {
        "schema_version": _ARTIFACT_STATUS_SCHEMA_VERSION,
        "episode_id": selected_episode,
        "context_revision": expected_binding["context_revision"]
        if expected_binding["context_revision"] is not None
        else _artifact_safe_reference(candidate.get("context_revision"), secret=secret),
        "source_revision": expected_binding["source_revision"]
        if expected_binding["source_revision"] is not None
        else _artifact_safe_reference(candidate.get("source_revision"), secret=secret),
        "source_digest": expected_binding["source_digest"]
        if expected_binding["source_digest"] is not None
        else _artifact_safe_reference(candidate.get("source_digest"), secret=secret),
    }
    projected_status["materialization"] = capability(
        candidate.get("materialization") if isinstance(candidate, Mapping) else None,
        native=False,
    )
    projected_status["native_diagnostic"] = capability(
        candidate.get("native_diagnostic") if isinstance(candidate, Mapping) else None,
        native=True,
    )
    return projected_status


def _artifact_status_context_mismatch(
    actual: Any, expected: Mapping[str, Any], *, secret: str | None = None
) -> bool:
    """Reject a service envelope whose context is foreign to the remembered selection.

    Returns:
        Whether the returned context is absent, unsafe, or foreign.
    """

    if actual is None:
        return True
    source_aliases = [
        _artifact_status_source_identity_token(_service_attr(actual, "source_identity")),
        _artifact_status_source_identity_token(_service_attr(actual, "source_digest")),
    ]
    source_aliases = [value for value in source_aliases if value not in (None, "")]
    safe_source_aliases = [
        _artifact_safe_reference(value, secret=secret) for value in source_aliases
    ]
    if safe_source_aliases and (
        any(value is None for value in safe_source_aliases)
        or any(value != safe_source_aliases[0] for value in safe_source_aliases[1:])
    ):
        return True
    fields = {
        "episode_id": _service_attr(actual, "episode_id"),
        "context_revision": _service_attr(actual, "context_revision"),
        "source_revision": _service_attr(actual, "source_revision"),
        "source_digest": source_aliases[0] if source_aliases else None,
    }
    for field_name, expected_value in expected.items():
        if expected_value in (None, ""):
            continue
        safe_expected = _artifact_safe_reference(expected_value, secret=secret)
        actual_value = fields.get(field_name)
        safe_actual = _artifact_safe_reference(actual_value, secret=secret)
        if safe_expected is None or actual_value is None or safe_actual != safe_expected:
            return True
    return any(
        actual_value is not None and _artifact_safe_reference(actual_value, secret=secret) is None
        for actual_value in fields.values()
    )


_CODEX_MAX_PROMPT_LENGTH = 8192
_CODEX_MAX_OPERATION_ID_LENGTH = 128
_CODEX_MAX_REASON_LENGTH = 512
_CODEX_MAX_TOKEN_BUDGET = 1_000_000
_CODEX_MAX_COMPUTE_BUDGET = 1_000_000.0
_CODEX_OPERATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_CODEX_PATH_PATTERN = re.compile(
    r"(?:[A-Za-z]:[\\/]|/|(?:^|(?<=[\s(]))(?:\.\.?/|[A-Za-z0-9_.-]+/))[^\s<>\"']*",
    re.IGNORECASE,
)
_CODEX_CONTEXT_FIELDS = frozenset(
    {
        "context_revision",
        "selection_revision",
        "campaign_id",
        "execution_id",
        "episode_id",
        "scenario_id",
        "interval_id",
        "reference_id",
        "actor_id",
    }
)


def _codex_raw_mapping(value: Any) -> dict[str, Any]:
    """Read a Codex result through an explicit mapping/dataclass projection.

    Returns:
        A detached mapping, or an empty mapping for an unsupported result.
    """

    if isinstance(value, Mapping):
        return dict(value)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            converted = to_dict()
        except Exception:  # noqa: BLE001 - malformed provider results are unavailable
            return {}
        return dict(converted) if isinstance(converted, Mapping) else {}
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: getattr(value, field.name) for field in fields(value)}
    return {}


def _codex_text(value: Any, *, secret: str | None = None, maximum: int = 1024) -> str | None:
    """Return bounded diagnostic text with token and path material removed."""

    if not isinstance(value, str):
        return None
    if secret and secret in value:
        value = value.replace(secret, "<redacted>")
    value = _CODEX_PATH_PATTERN.sub("<path redacted>", value.replace("\x00", "")[:maximum])
    # The positive projection does not need to preserve path-shaped prose.
    # Redact any remaining slash/backslash-bearing token as a conservative
    # fallback for relative paths, URI-like values, and roots not covered by
    # the anchored expression above.
    return " ".join(
        "<path redacted>" if "/" in token or "\\" in token else token for token in value.split()
    )


def _codex_reference(value: Any, *, secret: str | None = None) -> str | int | float | None:
    """Return one non-authority scalar reference."""

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value if not isinstance(value, float) or math.isfinite(value) else None
    if not isinstance(value, str) or "/" in value or "\\" in value:
        return None
    if secret and secret in value:
        return None
    return value[:256]


def _codex_context(value: Any, *, secret: str | None = None) -> dict[str, Any]:
    raw = _codex_raw_mapping(value)
    return {
        key: reference
        for key in _CODEX_CONTEXT_FIELDS
        if (reference := _codex_reference(raw.get(key), secret=secret)) is not None
    }


def _codex_source(value: Any, *, secret: str | None = None) -> dict[str, Any]:
    raw = _codex_raw_mapping(value)
    return {
        key: reference
        for key in ("source_revision", "source_digest", "digest")
        if (reference := _codex_reference(raw.get(key), secret=secret)) is not None
    }


def _codex_queue_projection(
    value: Mapping[str, Any] | None, *, secret: str | None = None
) -> dict[str, Any]:
    """Project queue identity fields without authority-like strings.

    Returns:
        A scalar queue identity projection.
    """

    if not isinstance(value, Mapping):
        return {}
    return {
        key: reference
        for key, item in value.items()
        if (reference := _codex_reference(item, secret=secret)) is not None
    }


def _codex_evidence(value: Any, *, secret: str | None = None) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        return []
    safe: list[Any] = []
    for item in value[:64]:
        if isinstance(item, Mapping) or is_dataclass(item):
            raw = _codex_raw_mapping(item)
            projected = {
                key: reference
                for key in (
                    "evidence_id",
                    "id",
                    "reference",
                    "artifact_id",
                    "episode_id",
                    "context_revision",
                    "source_revision",
                    "source_digest",
                )
                if (reference := _codex_reference(raw.get(key), secret=secret)) is not None
            }
            if projected:
                safe.append(projected)
        else:
            reference = _codex_reference(item, secret=secret)
            if reference is not None:
                safe.append(reference)
    return safe


def _codex_evidence_ids(value: Any, *, secret: str | None = None) -> list[Any]:
    safe_ids: list[Any] = []
    for item in _codex_evidence(value, secret=secret):
        if isinstance(item, Mapping):
            reference = item.get("evidence_id", item.get("id", item.get("reference")))
        else:
            reference = item
        if reference is not None:
            safe_ids.append(reference)
    return safe_ids[:64]


def _codex_usage(value: Any) -> dict[str, int | float]:
    raw = _codex_raw_mapping(value)
    safe: dict[str, int | float] = {}
    for key in (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "reserved_compute",
        "measured_compute",
        "token_budget",
        "compute_budget",
    ):
        item = raw.get(key)
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            continue
        if isinstance(item, float) and not math.isfinite(item):
            continue
        safe[key] = item
    tokens = raw.get("tokens")
    if "total_tokens" not in safe and isinstance(tokens, int) and not isinstance(tokens, bool):
        safe["total_tokens"] = tokens
    compute = raw.get("compute")
    if (
        "measured_compute" not in safe
        and isinstance(compute, (int, float))
        and not isinstance(compute, bool)
    ):
        try:
            finite_compute = math.isfinite(float(compute))
        except (OverflowError, ValueError):
            finite_compute = False
        if finite_compute:
            safe["measured_compute"] = compute
    return safe


def _codex_activity(value: Any, *, secret: str | None = None) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    safe: list[dict[str, Any]] = []
    for event in value[:64]:
        if isinstance(event, Mapping) or is_dataclass(event):
            raw = _codex_raw_mapping(event)
            item: dict[str, Any] = {}
            message = _codex_text(raw.get("message", raw.get("text")), secret=secret, maximum=512)
            if message is not None:
                item["message"] = message
            evidence = raw.get("evidence_ids", raw.get("evidence"))
            if evidence is not None:
                item["evidence_ids"] = _codex_evidence_ids(evidence, secret=secret)
            operation_id = _codex_reference(raw.get("operation_id"), secret=secret)
            if operation_id is not None:
                item["operation_id"] = operation_id
            timestamp = _codex_reference(raw.get("timestamp", raw.get("created_at")))
            if timestamp is not None:
                item["timestamp"] = timestamp
            if item:
                safe.append(item)
        else:
            message = _codex_text(event, secret=secret, maximum=512)
            if message is not None:
                safe.append({"message": message})
    return safe


def _codex_result_projection(  # noqa: C901, PLR0912, PLR0915
    value: Any, *, secret: str | None = None
) -> dict[str, Any]:
    """Project a typed/mapping Codex result into the browser-safe shape.

    Returns:
        A positive browser-safe diagnostic projection.
    """

    raw = _codex_raw_mapping(value)
    session = _codex_raw_mapping(raw.get("session"))
    receipt = _codex_raw_mapping(raw.get("receipt"))
    operation = _codex_raw_mapping(raw.get("operation"))
    session_context = _codex_raw_mapping(session.get("context"))
    context_value = raw.get("context") or raw.get("current_context") or session_context
    source_value = raw.get("source") or raw.get("current_source")
    if not source_value:
        source_value = {
            "source_revision": raw.get(
                "source_revision", session.get("source_revision", receipt.get("source_revision"))
            ),
            "source_digest": raw.get(
                "source_digest", session.get("source_digest", receipt.get("source_digest"))
            ),
        }
    projected: dict[str, Any] = {}
    status = _codex_text(raw.get("status", _service_attr(value, "status")), maximum=64)
    if status is not None:
        projected["status"] = status
    reason = _codex_text(
        raw.get("reason", _service_attr(value, "reason")), secret=secret, maximum=1024
    )
    if reason is not None:
        projected["reason"] = reason
    operation_id = raw.get(
        "operation_id", receipt.get("operation_id", operation.get("operation_id"))
    )
    if operation_id is None:
        operation_id = session.get("last_operation_id")
    operation_reference = _codex_reference(operation_id, secret=secret)
    if operation_reference is not None:
        projected["operation_id"] = operation_reference
    codex_session_id = session.get("session_id")
    if codex_session_id is None:
        codex_session_id = raw.get("codex_session_id")
    session_reference = _codex_reference(codex_session_id, secret=secret)
    if session_reference is not None:
        projected["codex_session_id"] = session_reference
    context = _codex_context(context_value, secret=secret)
    if context:
        projected["context"] = context
        for key in ("context_revision", "selection_revision"):
            if key in context:
                projected[key] = context[key]
    source = _codex_source(source_value, secret=secret)
    if source:
        projected["source"] = source
        for key in ("source_revision", "source_digest"):
            if key in source:
                projected[key] = source[key]
    route = raw.get("route") or session.get("route") or receipt.get("route")
    route_id = raw.get("route_id")
    if route_id is None:
        route_id = _codex_raw_mapping(route).get("route_id")
    route_reference = _codex_reference(route_id, secret=secret)
    if route_reference is not None:
        projected["route_id"] = route_reference
    evidence = raw.get("evidence_ids", raw.get("evidence_references"))
    if evidence is None:
        evidence = raw.get("evidence", session.get("evidence", receipt.get("evidence_ids")))
    evidence_references = _codex_evidence(evidence, secret=secret)
    if evidence_references:
        projected["evidence_references"] = evidence_references
        projected["evidence_ids"] = _codex_evidence_ids(evidence_references, secret=secret)
    usage = raw.get("usage", receipt.get("usage", operation.get("usage")))
    projected_usage = _codex_usage(usage)
    if projected_usage:
        projected["usage"] = projected_usage
    activity = raw.get("activity", raw.get("events"))
    if activity is None:
        activity = session.get("events")
    if activity is None and operation.get("message") is not None:
        activity = [operation]
    projected_activity = _codex_activity(activity, secret=secret)
    if projected_activity:
        projected["activity"] = projected_activity
    activity_scope = _codex_text(raw.get("activity_scope"), maximum=64)
    if activity_scope is not None:
        projected["activity_scope"] = activity_scope
    elif projected_activity:
        projected["activity_scope"] = "process"
    events_reason = _codex_text(raw.get("events_reason"), secret=secret, maximum=1024)
    if events_reason is not None:
        projected["events_reason"] = events_reason
    return projected


def _codex_result_binding_mismatch(  # noqa: C901, PLR0912
    value: Any,
    *,
    operation_id: str,
    expected_context: Any,
    expected_source_revision: Any,
    expected_source_digest: Any,
) -> str | None:
    """Reject a Codex result whose explicit authority binding is foreign.

    The injected client is deliberately treated as untrusted at this edge.
    Successful and failed results may omit binding fields, but any operation,
    context, or source identity they do return must agree with the request.

    Returns:
        A bounded conflict reason, or ``None`` when no explicit binding drift
        is present.
    """

    raw = _codex_raw_mapping(value)
    session = _codex_raw_mapping(raw.get("session"))
    receipt = _codex_raw_mapping(raw.get("receipt"))
    operation = _codex_raw_mapping(raw.get("operation"))
    mappings = [raw, session, receipt, operation]
    for candidate in (session, receipt):
        nested_operation = _codex_raw_mapping(candidate.get("operation"))
        if nested_operation:
            mappings.append(nested_operation)

    for candidate in mappings:
        for key in ("operation_id", "last_operation_id"):
            if key in candidate and candidate[key] is not None and candidate[key] != operation_id:
                return "Codex result operation binding does not match the request"

    for candidate in mappings:
        for key in ("context", "current_context"):
            result_context = candidate.get(key)
            if result_context is not None and not _context_matches_request(
                expected_context, result_context
            ):
                return "Codex result context binding does not match the request"
    expected_context_revision = _service_attr(expected_context, "context_revision")
    for candidate in mappings:
        if (
            "context_revision" in candidate
            and candidate["context_revision"] is not None
            and candidate["context_revision"] != expected_context_revision
        ):
            return "Codex result context revision does not match the request"

    for candidate in tuple(mappings):
        for key in ("source", "current_source"):
            source = candidate.get(key)
            if source is None:
                continue
            source_mapping = _codex_raw_mapping(source)
            if not source_mapping:
                return "Codex result source binding is malformed"
            mappings.append(source_mapping)
    for candidate in mappings:
        if (
            "source_revision" in candidate
            and candidate["source_revision"] is not None
            and candidate["source_revision"] != expected_source_revision
        ):
            return "Codex result source revision does not match the request"
        for key in ("source_digest", "digest"):
            if (
                key in candidate
                and candidate[key] is not None
                and candidate[key] != expected_source_digest
            ):
                return "Codex result source identity does not match the request"
    return None


def _codex_result_context_binding_mismatch(  # noqa: C901, PLR0912
    value: Any,
    *,
    expected_operation_id: str,
    expected_context: Any,
    expected_source_revision: Any,
    expected_source_digest: Any,
) -> str | None:
    """Validate reconnect context without treating its old operation as current.

    Reconnect returns the durable session's last operation identity, which is
    intentionally different from the new browser reconnect operation.  The
    source/context binding still has to be explicit and match exactly before
    the returned handle is retained by the server facade.

    Returns:
        A bounded conflict reason, or ``None`` when the binding matches.
    """

    if not isinstance(expected_source_digest, str) or not expected_source_digest.strip():
        return "recovered Codex session has no authoritative source identity"

    raw = _codex_raw_mapping(value)
    mappings: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    seen: set[int] = set()

    def add_mapping(candidate: Any) -> None:
        mapping = _codex_raw_mapping(candidate)
        if not mapping or id(mapping) in seen:
            return
        seen.add(id(mapping))
        mappings.append(mapping)
        pending.append(mapping)

    add_mapping(raw)
    saw_context = False
    saw_source = False
    saw_source_revision = False
    saw_source_digest = False
    while pending:
        candidate = pending.pop(0)
        for nested_key in ("session", "receipt", "operation", "result"):
            nested = candidate.get(nested_key)
            if nested is not None:
                add_mapping(nested)
        for key in ("context", "current_context"):
            if key not in candidate or candidate[key] is None:
                continue
            saw_context = True
            result_context = candidate[key]
            if not _codex_raw_mapping(result_context):
                return "recovered Codex session context binding is malformed"
            if not _context_matches_request(expected_context, result_context):
                return "recovered Codex session context does not match the selected case"
        for key in ("source", "current_source"):
            if key not in candidate or candidate[key] is None:
                continue
            saw_source = True
            source_mapping = _codex_raw_mapping(candidate[key])
            if not source_mapping:
                return "recovered Codex session source binding is malformed"
            add_mapping(source_mapping)
        for key in ("operation_id",):
            if key in candidate and candidate[key] is not None:
                if candidate[key] != expected_operation_id:
                    return "recovered Codex session operation binding does not match"
        if any(
            key in candidate and candidate[key] is not None
            for key in ("source_revision", "source_digest", "digest")
        ):
            saw_source = True

    expected_context_revision = _service_attr(expected_context, "context_revision")
    for candidate in mappings:
        if (
            candidate.get("context_revision") is not None
            and candidate.get("context_revision") != expected_context_revision
        ):
            return "recovered Codex session context revision does not match the selection"
        if (
            candidate.get("source_revision") is not None
            and candidate.get("source_revision") != expected_source_revision
        ):
            return "recovered Codex session source revision does not match the selection"
        if candidate.get("source_revision") is not None:
            saw_source_revision = True
        for key in ("source_digest", "digest"):
            if candidate.get(key) is not None and candidate.get(key) != expected_source_digest:
                return "recovered Codex session source identity does not match the selection"
            if candidate.get(key) is not None:
                saw_source_digest = True
    if not saw_context:
        return "recovered Codex session result has no authoritative context binding"
    if not saw_source:
        return "recovered Codex session result has no authoritative source binding"
    if not saw_source_revision or not saw_source_digest:
        return "recovered Codex session source binding is incomplete"
    return None


class ServiceAuditWorkbenchFacade:
    """Server-held BA-05 service adapter for the audit workbench.

    The facade is deliberately injectable: the BA-06 checkout predates the
    BA-05 service module, so it depends on the frozen method/envelope contract
    rather than importing an unaccepted implementation.  ``service`` owns all
    selection, persistence, policy, and source checks.  The session and token
    remain private to this object and are never included in a returned model.
    """

    def __init__(
        self,
        service: Any | None = None,
        session: Any | None = None,
        *,
        token: str | None = None,
        audit_service: Any | None = None,
        audit_token: str | None = None,
        codex_client: AuditCodexClientProtocol | None = None,
        audit_codex_client: AuditCodexClientProtocol | None = None,
    ):
        """Bind one server-owned service session without exposing its token."""

        service = service if service is not None else audit_service
        token = token if token is not None else audit_token
        if (
            codex_client is not None
            and audit_codex_client is not None
            and codex_client is not audit_codex_client
        ):
            raise AuditWorkbenchError("conflicting Codex clients were supplied")
        codex_client = codex_client if codex_client is not None else audit_codex_client
        if service is None:
            raise AuditWorkbenchError("a BA-05 service is required")
        if session is None:
            raise AuditWorkbenchError("a server-held audit session is required")
        self._service = service
        self._session = session
        self._token = token if isinstance(token, str) and token else None
        self._codex_client = codex_client
        self._codex_state_lock = threading.RLock()
        self._codex_start_in_flight = False
        self._codex_session_handle: Any = None
        self._codex_operation_id: str | None = None
        self._codex_session_context: Any = None
        self._codex_session_source_revision: Any = None
        self._codex_session_source_digest: Any = None
        self._context = _service_attr(session, "context")
        self._selected_packet: dict[str, Any] | None = None
        self._selected_episode_id: str | None = None
        self._last_episode: Any = None
        self._last_next: Any = None
        self._last_queue: Any = None
        self._last_coverage: Any = None
        self._last_annotation: Mapping[str, Any] | None = None
        self._last_annotation_record: dict[str, Any] | None = None
        self._last_annotation_receipt: dict[str, Any] | None = None
        self._last_annotation_context_revision: Any = None
        self._last_annotation_episode_id: str | None = None
        self._last_annotation_revision: int | None = None
        self._last_annotation_context: Any = None
        self._last_finding: dict[str, Any] | None = None
        self._last_finding_receipt: dict[str, Any] | None = None
        self._annotation_records: dict[str, dict[str, Any]] = {}
        self._annotation_receipts: dict[str, dict[str, Any]] = {}
        self._finding_records: dict[str, dict[str, Any]] = {}
        self._finding_receipts: dict[str, dict[str, Any]] = {}
        # This opaque scope binds the legacy in-memory projection to one facade
        # instance.  It is never used for a durable service projection.
        self._record_projection_scope = hashlib.sha256(
            f"ba06-record-projection:{id(self)}".encode()
        ).hexdigest()[:24]
        self._selection_epoch = 0
        self._queue_revisions: dict[str, Any] = {}
        selected_episode = _service_attr(self._context, "episode_id")
        self._selected_episode_id = str(selected_episode) if selected_episode else None
        if self._codex_client is not None:
            # Start/cancel are explicit injected capabilities.  A durable
            # service-backed activity read is independent of that client and
            # is exposed below when BA-05 provides its authenticated method.
            self.codex_start = self._codex_start  # type: ignore[attr-defined]
            self.codex_read = self._codex_read  # type: ignore[attr-defined]
            self.codex_cancel = self._codex_cancel  # type: ignore[attr-defined]
            self.codex_reconnect = self._codex_reconnect  # type: ignore[attr-defined]
        if callable(getattr(self._service, "read_codex_activity", None)):
            self.codex_read = self._codex_read  # type: ignore[attr-defined]

    @property
    def service_metadata(self) -> Mapping[str, Any]:
        """Return non-secret metadata suitable for a browser-visible model."""

        session_id = _service_attr(self._session, "session_id")
        if session_id is None and isinstance(self._session, str):
            session_id = self._session
        actor = _service_attr(self._session, "actor")
        actor_id = _service_attr(actor, "actor_id")
        metadata = {
            "id": AUDIT_SERVICE_FACADE_ID,
            "authority": "server_held_audit_service",
            "session_id": str(session_id) if session_id else None,
            "actor_id": str(actor_id) if actor_id else None,
            "token_transport": "server_only",
            "evidence_status": "service_backed_not_benchmark_evidence",
            "diagnostic_only": True,
            "fixture": False,
        }
        safe_metadata = _safe_service_value(metadata, secret=self._token)
        return safe_metadata if isinstance(safe_metadata, Mapping) else {}

    def _local_result(
        self,
        status: str,
        reason: str,
        *,
        context: Any = None,
        conflict: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a local envelope with the facade's secret-aware sanitizer.

        Returns:
            A safe local service envelope.
        """

        return _local_service_result(
            status,
            reason,
            context=context,
            conflict=conflict,
            secret=self._token,
        )

    def _denied(self) -> dict[str, Any]:
        return self._local_result(
            "denied",
            "server-held audit session token is unavailable",
            context=self._context,
        )

    def _call_service(self, method_name: str, *args: Any, **kwargs: Any) -> Any:
        """Call BA-05 with the private token, or return an explicit status.

        Returns:
            The typed service result or a bounded local failure envelope.
        """

        if self._token is None:
            return self._denied()
        method = getattr(self._service, method_name, None)
        if not callable(method):
            aliases = {
                "write_annotation": ("save_annotation",),
                "write_finding": ("save_finding",),
                "related_cases": ("find_related_cases", "compatible_peers"),
                "read_selected_artifact_status": ("selected_artifact_status",),
            }
            method = next(
                (
                    getattr(self._service, alias, None)
                    for alias in aliases.get(method_name, ())
                    if callable(getattr(self._service, alias, None))
                ),
                None,
            )
        if not callable(method):
            return self._local_result(
                "unavailable", f"BA-05 service operation {method_name!r} is unavailable"
            )
        call_kwargs = dict(kwargs)
        call_kwargs["token"] = self._token
        try:
            return method(self._session, *args, **call_kwargs)
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError) as exc:
            reason = str(exc)
            if self._token and self._token in reason:
                reason = reason.replace(self._token, "<redacted>")
            return self._local_result("failed", reason or "BA-05 service operation failed")

    def _codex_local_result(
        self,
        status: str,
        reason: str,
        *,
        conflict: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build a Codex-safe local result without the generic authority envelope.

        Returns:
            A bounded diagnostic result without authority fields.
        """

        result: dict[str, Any] = {
            "status": _codex_text(status, maximum=64) or "unavailable",
            "reason": _codex_text(reason, secret=self._token, maximum=1024) or "",
        }
        context = _codex_context(self._context, secret=self._token)
        if context:
            result["context"] = context
        if conflict is not None:
            safe_conflict = _safe_service_value(conflict, secret=self._token)
            if isinstance(safe_conflict, Mapping):
                result["conflict"] = dict(safe_conflict)
        return result

    @staticmethod
    def _codex_queue_revisions(
        value: Any, *, require_complete: bool = False
    ) -> dict[str, Any] | None:
        """Read only queue CAS fields needed to bind one Codex turn.

        Returns:
            The selected queue revision identity, or ``None`` when a complete
            authoritative identity is required but missing or malformed.
        """

        current = value
        visited: set[int] = set()
        for _ in range(4):
            if current is None or id(current) in visited:
                break
            visited.add(id(current))
            revisions: dict[str, Any] = {}
            for key, source in (
                ("queue_state_revision", "state_revision"),
                ("queue_input_revision", "input_revision"),
                ("queue_input_identity", "input_identity"),
            ):
                item = _service_attr(current, key, _CONTEXT_UNSET)
                if item is _CONTEXT_UNSET:
                    item = _service_attr(current, source, _CONTEXT_UNSET)
                if item is not _CONTEXT_UNSET and item is not None:
                    revisions[key] = item
            if revisions:
                complete = (
                    set(revisions)
                    == {
                        "queue_state_revision",
                        "queue_input_revision",
                        "queue_input_identity",
                    }
                    and _is_revision(revisions["queue_state_revision"])
                    and _is_revision(revisions["queue_input_revision"])
                    and isinstance(revisions["queue_input_identity"], str)
                    and bool(revisions["queue_input_identity"].strip())
                )
                if require_complete:
                    return revisions if complete else None
                return revisions
            nested = _service_attr(current, "value", _CONTEXT_UNSET)
            if nested is _CONTEXT_UNSET or nested is current:
                break
            current = nested
        return None if require_complete else {}

    def _codex_binding_conflict(
        self,
        reason: str,
        *,
        expected_context: Any,
        actual_context: Any = None,
        expected_queue: Mapping[str, Any] | None = None,
        actual_queue: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        conflict: dict[str, Any] = {
            "expected_context": _codex_context(expected_context, secret=self._token),
            "actual_context": _codex_context(
                actual_context if actual_context is not None else self._context,
                secret=self._token,
            ),
        }
        if expected_queue is not None:
            conflict["expected_queue"] = _codex_queue_projection(expected_queue, secret=self._token)
        if actual_queue is not None:
            conflict["actual_queue"] = _codex_queue_projection(actual_queue, secret=self._token)
        return self._codex_local_result("conflict", reason, conflict=conflict)

    def _codex_session_binding_failure(self, context: Any) -> str | None:
        """Check the retained post-turn handle against current authority.

        Returns:
            A bounded stale-binding reason, or ``None`` when it still matches.
        """

        if self._codex_session_context is not None and not _context_matches_request(
            self._codex_session_context, context
        ):
            return "Codex session context is stale"
        current_source_revision = self._source_revision()
        if (
            self._codex_session_source_revision is not None
            and current_source_revision != self._codex_session_source_revision
        ):
            return "Codex session source revision is stale"
        current_source_digest = _service_attr(context, "source_identity")
        if (
            self._codex_session_source_digest is not None
            and current_source_digest != self._codex_session_source_digest
        ):
            return "Codex session source identity is stale"
        handle_context = _service_attr(self._codex_session_handle, "context")
        if handle_context is not None and not _context_matches_request(context, handle_context):
            return "Codex session handle context is stale"
        for field_name, expected in (
            ("source_revision", current_source_revision),
            ("source_digest", current_source_digest),
        ):
            actual = _service_attr(self._codex_session_handle, field_name, _CONTEXT_UNSET)
            if actual is not _CONTEXT_UNSET and actual != expected:
                return f"Codex session handle {field_name} is stale"
        return None

    def _codex_start_binding(
        self,
        *,
        operation_id: str,
        expected_selection_revision: int | None,
    ) -> tuple[Any | None, dict[str, Any] | None]:
        """Re-read selection/source/queue state before Codex admission.

        Returns:
            The authoritative context and an optional fail-closed result.
        """

        selection_guard = self._selection_guard(expected_selection_revision)
        if selection_guard is not None:
            return None, selection_guard
        if self._selected_packet is None or not self._selected_episode_id:
            return None, self._codex_local_result(
                "unavailable", "Codex start requires a selected audit episode"
            )
        initial_selection = self._selection_epoch
        initial_context = deepcopy(self._context)
        initial_source_revision = self._source_revision()
        initial_queue = self._codex_queue_revisions(self._queue_revisions, require_complete=True)
        if not _is_revision(self._context_revision()) or initial_source_revision is None:
            return None, self._codex_local_result(
                "unavailable", "Codex start requires an authoritative context and source"
            )
        if initial_queue is None:
            return None, self._codex_local_result(
                "unavailable", "Codex start requires a complete queue CAS identity"
            )
        context_result = self._call_service("read_context", operation_id=f"{operation_id}:context")
        self._remember_result(context_result)
        returned_context = _service_attr(context_result, "context")
        candidate_context = _service_attr(context_result, "value")
        if (
            returned_context is None
            and _service_attr(candidate_context, "context_revision") is not None
        ):
            returned_context = candidate_context
        if not _result_succeeded(context_result) or returned_context is None:
            return None, self._codex_local_result(
                "unavailable", "Codex start could not revalidate the audit context"
            )
        if not _context_matches_request(initial_context, returned_context):
            return None, self._codex_binding_conflict(
                "Codex start selection context is stale",
                expected_context=initial_context,
                actual_context=returned_context,
            )
        queue_result = self._call_service(
            "read_queue",
            limit=1,
            context=deepcopy(initial_context),
            operation_id=f"{operation_id}:queue",
        )
        self._remember_result(queue_result)
        queue_value = _service_attr(queue_result, "value")
        current_queue = self._codex_queue_revisions(queue_value, require_complete=True)
        if not _result_succeeded(queue_result) or current_queue != initial_queue:
            return None, self._codex_binding_conflict(
                "Codex start queue selection is stale",
                expected_context=initial_context,
                expected_queue=initial_queue,
                actual_queue=current_queue or {},
            )
        if (
            self._selection_epoch != initial_selection
            or not _context_matches_request(initial_context, self._context)
            or self._source_revision() != initial_source_revision
        ):
            return None, self._codex_binding_conflict(
                "Codex start selection changed before admission",
                expected_context=initial_context,
                expected_queue=initial_queue,
                actual_queue=current_queue,
            )
        return deepcopy(self._context), None

    def _codex_client_is_bound(self) -> bool:
        """Require an injected client to use this exact service instance.

        Returns:
            Whether the client's optional service binding matches this facade.
        """

        client_service = getattr(self._codex_client, "service", _CONTEXT_UNSET)
        return client_service is self._service

    @staticmethod
    def _codex_signature(method: Any) -> tuple[Mapping[str, inspect.Parameter], bool]:
        try:
            parameters = inspect.signature(method).parameters
        except (TypeError, ValueError):
            return {}, False
        accepts_kwargs = any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
        )
        return parameters, accepts_kwargs

    def _codex_start(  # noqa: C901, PLR0912
        self,
        *,
        prompt: str,
        operation_id: str,
        token_budget: int,
        compute_budget: float,
        expected_selection_revision: int | None = None,
    ) -> Mapping[str, Any]:
        """Start one post-selection diagnostic Codex turn through BA-05.

        Returns:
            A safe diagnostic result or an explicit conflict/unavailable state.
        """

        if self._token is None:
            return self._denied()
        if self._codex_client is None or not self._codex_client_is_bound():
            return self._codex_local_result("unavailable", "BA-05 Codex capability is unavailable")
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or len(prompt) > _CODEX_MAX_PROMPT_LENGTH
            or "\x00" in prompt
        ):
            return self._codex_local_result("failed", "prompt is empty or exceeds the limit")
        if self._token in prompt:
            return self._codex_local_result("denied", "Codex prompt contains server authority")
        if (
            not isinstance(operation_id, str)
            or not _CODEX_OPERATION_ID.fullmatch(operation_id)
            or self._token in operation_id
        ):
            return self._codex_local_result("failed", "operation_id is invalid")
        if (
            isinstance(token_budget, bool)
            or not isinstance(token_budget, int)
            or not 1 <= token_budget <= _CODEX_MAX_TOKEN_BUDGET
        ):
            return self._codex_local_result("failed", "token_budget is outside the permitted bound")
        try:
            compute_value = float(compute_budget)
        except (OverflowError, ValueError):
            compute_value = math.nan
        if (
            isinstance(compute_budget, bool)
            or not isinstance(compute_budget, (int, float))
            or not math.isfinite(compute_value)
            or not 1.0 <= compute_value <= _CODEX_MAX_COMPUTE_BUDGET
        ):
            return self._codex_local_result(
                "failed", "compute_budget must be at least 1.0 and within the permitted bound"
            )
        with self._codex_state_lock:
            if self._codex_start_in_flight:
                return self._codex_local_result(
                    "unavailable", "Codex start is already in flight; active cancel is unavailable"
                )
            self._codex_start_in_flight = True
        try:
            authoritative_context, failure = self._codex_start_binding(
                operation_id=operation_id,
                expected_selection_revision=expected_selection_revision,
            )
            if failure is not None:
                return failure
            try:
                raw_result = self._codex_client.start(
                    self._session,
                    prompt=prompt,
                    context=authoritative_context,
                    operation_id=operation_id,
                    token_budget=token_budget,
                    compute_budget=compute_value,
                    audit_token=self._token,
                )
            except Exception:  # noqa: BLE001 - provider details stay server-side
                return self._codex_local_result("unavailable", "Codex start failed")
            binding_failure = _codex_result_binding_mismatch(
                raw_result,
                operation_id=operation_id,
                expected_context=authoritative_context,
                expected_source_revision=self._source_revision(),
                expected_source_digest=_service_attr(authoritative_context, "source_identity"),
            )
            if binding_failure is not None:
                return self._codex_local_result("conflict", binding_failure)
            projected = _codex_result_projection(raw_result, secret=self._token)
            if "status" not in projected:
                return self._codex_local_result("unavailable", "Codex start returned no status")
            projected.setdefault("operation_id", operation_id)
            projected.setdefault(
                "context", _codex_context(authoritative_context, secret=self._token)
            )
            source = _codex_source(
                {
                    "source_revision": self._source_revision(),
                    "source_digest": _service_attr(authoritative_context, "source_identity"),
                },
                secret=self._token,
            )
            if source:
                projected.setdefault("source", source)
            if projected["status"] in {"complete", "committed", "ok", "saved"}:
                session_handle = _service_attr(raw_result, "session")
                if session_handle is not None:
                    self._codex_session_handle = session_handle
                self._codex_operation_id = operation_id
                self._codex_session_context = deepcopy(authoritative_context)
                self._codex_session_source_revision = self._source_revision()
                self._codex_session_source_digest = _service_attr(
                    authoritative_context, "source_identity"
                )
            return projected
        finally:
            with self._codex_state_lock:
                self._codex_start_in_flight = False

    def _codex_read(self, *, operation_id: str | None = None) -> Mapping[str, Any]:  # noqa: C901
        """Read only the authenticated BA-05 activity projection, when accepted.

        Returns:
            A safe activity projection or an explicit unavailable state.
        """

        if self._token is None:
            return self._denied()
        method = getattr(self._service, "read_codex_activity", None)
        if not callable(method):
            return self._codex_local_result(
                "unavailable", "BA-05 Codex activity read is unavailable"
            )
        method_owner = getattr(method, "__self__", None)
        if method_owner is not None and method_owner is not self._service:
            return self._codex_local_result(
                "unavailable", "BA-05 Codex activity read is bound to another service"
            )
        if operation_id is not None and (
            not isinstance(operation_id, str)
            or not _CODEX_OPERATION_ID.fullmatch(operation_id)
            or self._token in operation_id
        ):
            return self._codex_local_result("failed", "operation_id is invalid")
        parameters, _accepts_kwargs = self._codex_signature(method)
        token_name = (
            "audit_token" if "audit_token" in parameters and "token" not in parameters else "token"
        )
        if token_name not in parameters:
            return self._codex_local_result(
                "unavailable", "BA-05 Codex activity read has no authenticated token boundary"
            )
        kwargs: dict[str, Any] = {token_name: self._token}
        if operation_id is not None and "operation_id" in parameters:
            kwargs["operation_id"] = operation_id
        session_id = _service_attr(self._codex_session_handle, "session_id")
        if session_id is not None and "codex_session_id" in parameters:
            kwargs["codex_session_id"] = session_id
        if "context" in parameters:
            kwargs["context"] = deepcopy(self._context)
        if "limit" in parameters:
            kwargs["limit"] = 64
        try:
            raw_result = method(self._session, **kwargs)
        except Exception:  # noqa: BLE001 - activity-read details stay server-side
            return self._codex_local_result("unavailable", "BA-05 Codex activity read failed")
        projected = _codex_result_projection(raw_result, secret=self._token)
        return (
            projected
            if "status" in projected
            else self._codex_local_result(
                "unavailable", "BA-05 Codex activity read returned no status"
            )
        )

    def _codex_reconnect(  # noqa: C901, PLR0912
        self,
        *,
        codex_session_id: str,
        operation_id: str,
        expected_selection_revision: int,
        expected_context_revision: int,
    ) -> Mapping[str, Any]:
        """Recover a durable Codex session without widening its authority.

        The browser supplies only the opaque durable Codex session ID and
        compare-and-swap selection revisions.  The server-held client supplies
        the audit token and revalidates the persisted route/source through the
        BA-05 authority ledger.

        Returns:
            A projected recovered-session result or an explicit conflict,
            unavailable, or failed status.
        """

        if self._token is None:
            return self._denied()
        if self._codex_client is None or not self._codex_client_is_bound():
            return self._codex_local_result("unavailable", "BA-05 Codex capability is unavailable")
        if (
            not isinstance(codex_session_id, str)
            or not _CODEX_OPERATION_ID.fullmatch(codex_session_id)
            or self._token in codex_session_id
        ):
            return self._codex_local_result("failed", "Codex session ID is invalid")
        if (
            not isinstance(operation_id, str)
            or not _CODEX_OPERATION_ID.fullmatch(operation_id)
            or self._token in operation_id
        ):
            return self._codex_local_result("failed", "operation_id is invalid")
        if (
            isinstance(expected_selection_revision, bool)
            or not isinstance(expected_selection_revision, int)
            or expected_selection_revision < 0
            or isinstance(expected_context_revision, bool)
            or not isinstance(expected_context_revision, int)
            or expected_context_revision < 0
        ):
            return self._codex_local_result("failed", "Codex reconnect revisions are invalid")
        if expected_selection_revision != self._selection_epoch:
            return self._codex_local_result(
                "conflict",
                "Codex reconnect selection revision is stale",
                conflict={
                    "expected_revision": expected_selection_revision,
                    "actual_revision": self._selection_epoch,
                },
            )
        if expected_context_revision != self._context_revision():
            return self._codex_local_result(
                "conflict",
                "Codex reconnect context revision is stale",
                conflict={
                    "expected_revision": expected_context_revision,
                    "actual_revision": self._context_revision(),
                },
            )
        if self._codex_start_in_flight:
            return self._codex_local_result(
                "unavailable", "Codex start is in flight; reconnect is deferred"
            )
        authoritative_context, failure = self._codex_start_binding(
            operation_id=f"{operation_id}:bind",
            expected_selection_revision=expected_selection_revision,
        )
        if failure is not None or authoritative_context is None:
            return failure or self._codex_local_result(
                "conflict", "Codex reconnect has no authoritative context"
            )
        expected_source_revision = self._source_revision()
        expected_source_digest = _service_attr(authoritative_context, "source_identity")
        if (
            not isinstance(expected_source_digest, str)
            or not expected_source_digest.strip()
            or expected_source_revision is None
        ):
            return self._codex_local_result(
                "unavailable", "BA-05 Codex reconnect has no authoritative source binding"
            )
        method = getattr(self._codex_client, "reconnect", None)
        if not callable(method):
            return self._codex_local_result(
                "unavailable", "BA-05 Codex provider reconnect is unavailable"
            )
        parameters, accepts_kwargs = self._codex_signature(method)
        if "operation_id" not in parameters and not accepts_kwargs:
            return self._codex_local_result(
                "unavailable", "BA-05 Codex reconnect has no durable operation boundary"
            )
        if "audit_token" in parameters:
            token_name = "audit_token"
        elif "token" in parameters:
            token_name = "token"
        elif accepts_kwargs:
            token_name = "audit_token"
        else:
            token_name = ""
        if token_name not in parameters and not accepts_kwargs:
            return self._codex_local_result(
                "unavailable", "BA-05 Codex reconnect has no authenticated token boundary"
            )
        kwargs: dict[str, Any] = {
            "operation_id": operation_id,
            token_name: self._token,
        }
        try:
            raw_result = method(codex_session_id, **kwargs)
        except Exception:  # noqa: BLE001 - provider/authority details stay server-side
            return self._codex_local_result("unavailable", "Codex provider reconnect failed")
        binding_failure = _codex_result_context_binding_mismatch(
            raw_result,
            expected_operation_id=operation_id,
            expected_context=authoritative_context,
            expected_source_revision=expected_source_revision,
            expected_source_digest=expected_source_digest,
        )
        if binding_failure is not None:
            return self._codex_local_result("conflict", binding_failure)
        projected = _codex_result_projection(raw_result, secret=self._token)
        status = projected.get("status")
        if status not in {"complete", "committed", "ok", "saved", "cancelled"}:
            return projected or self._codex_local_result(
                "unavailable", "Codex session recovery returned no status"
            )
        session_handle = _service_attr(raw_result, "session")
        if session_handle is None:
            return self._codex_local_result(
                "unavailable", "Codex provider reconnect returned no durable session"
            )
        returned_session_id = _service_attr(session_handle, "session_id")
        if returned_session_id != codex_session_id:
            return self._codex_local_result(
                "conflict", "Codex provider reconnect returned a foreign session"
            )
        returned_result_session_id = _service_attr(raw_result, "codex_session_id")
        if (
            returned_result_session_id is not None
            and returned_result_session_id != codex_session_id
        ):
            return self._codex_local_result(
                "conflict", "Codex provider reconnect returned a foreign session binding"
            )
        with self._codex_state_lock:
            self._codex_session_handle = session_handle
            self._codex_operation_id = operation_id
            self._codex_session_context = deepcopy(authoritative_context)
            self._codex_session_source_revision = expected_source_revision
            self._codex_session_source_digest = expected_source_digest
        projected.setdefault("operation_id", operation_id)
        projected.setdefault("context", _codex_context(authoritative_context, secret=self._token))
        projected.setdefault(
            "codex_session_id", _service_attr(session_handle, "session_id", codex_session_id)
        )
        return projected

    def _codex_cancel(  # noqa: C901
        self,
        *,
        reason: str,
        operation_id: str,
    ) -> Mapping[str, Any]:
        """Cancel a completed-turn session; never fake an in-flight interrupt.

        Returns:
            The client status, preserving ambiguous and kill-switch outcomes.
        """

        if self._token is None:
            return self._denied()
        if self._codex_client is None or not self._codex_client_is_bound():
            return self._codex_local_result("unavailable", "BA-05 Codex capability is unavailable")
        if self._codex_start_in_flight:
            return self._codex_local_result(
                "unavailable", "Codex start is in flight; active cancel is unavailable"
            )
        if (
            not isinstance(operation_id, str)
            or not _CODEX_OPERATION_ID.fullmatch(operation_id)
            or self._token in operation_id
        ):
            return self._codex_local_result("failed", "operation_id is invalid")
        if (
            not isinstance(reason, str)
            or not reason.strip()
            or len(reason) > _CODEX_MAX_REASON_LENGTH
        ):
            return self._codex_local_result("failed", "reason is empty or exceeds the limit")
        if self._token in reason:
            return self._codex_local_result(
                "denied", "Codex cancel reason contains server authority"
            )
        if self._codex_session_handle is None or self._codex_operation_id != operation_id:
            return self._codex_local_result(
                "unavailable", "no completed Codex session matches the operation"
            )
        authoritative_context, failure = self._codex_start_binding(
            operation_id=f"{operation_id}:cancel-bind",
            expected_selection_revision=self._selection_epoch,
        )
        if failure is not None:
            return failure
        if authoritative_context is None:
            return self._codex_local_result(
                "unavailable", "Codex cancellation has no authoritative context"
            )
        binding_failure = self._codex_session_binding_failure(authoritative_context)
        if binding_failure is not None:
            return self._codex_local_result("conflict", binding_failure)
        try:
            raw_result = self._codex_client.cancel(
                self._codex_session_handle,
                operation_id=operation_id,
                reason=reason,
                audit_token=self._token,
            )
        except Exception:  # noqa: BLE001 - ambiguous provider details stay server-side
            return self._codex_local_result("unavailable", "Codex cancellation is ambiguous")
        result_binding_failure = _codex_result_binding_mismatch(
            raw_result,
            operation_id=operation_id,
            expected_context=authoritative_context,
            expected_source_revision=self._source_revision(),
            expected_source_digest=_service_attr(authoritative_context, "source_identity"),
        )
        if result_binding_failure is not None:
            return self._codex_local_result("conflict", result_binding_failure)
        projected = _codex_result_projection(raw_result, secret=self._token)
        projected.setdefault("operation_id", operation_id)
        context = _codex_context(authoritative_context, secret=self._token)
        if context:
            projected.setdefault("context", context)
        source = _codex_source(
            {
                "source_revision": self._source_revision(),
                "source_digest": _service_attr(authoritative_context, "source_identity"),
            },
            secret=self._token,
        )
        if source:
            projected.setdefault("source", source)
        return (
            projected
            if "status" in projected
            else self._codex_local_result("unavailable", "Codex cancellation returned no status")
        )

    def _remember_result(self, result: Any) -> None:
        context = _service_attr(result, "context")
        if context is not None:
            if self._last_annotation_context is not None and not self._context_matches_request(
                self._last_annotation_context, context
            ):
                self._clear_retained_annotation()
            self._context = context

    def _clear_retained_annotation(self) -> None:
        """Discard a receipt that can no longer be proven fresh or exact."""

        self._last_annotation = None
        self._last_annotation_record = None
        self._last_annotation_receipt = None
        self._last_annotation_context_revision = None
        self._last_annotation_episode_id = None
        self._last_annotation_revision = None
        self._last_annotation_context = None
        self._last_finding = None
        self._last_finding_receipt = None
        self._annotation_records.clear()
        self._annotation_receipts.clear()
        self._finding_records.clear()
        self._finding_receipts.clear()

    @staticmethod
    def _context_matches_request(requested: Any, returned: Any) -> bool:
        """Check returned context revision and stable selection identity.

        Returns:
            ``True`` only when the returned context matches the requested
            revision and identity fields.
        """

        return _context_matches_request(requested, returned)

    def _annotation_freshness_guard(self) -> dict[str, Any] | None:
        """Reject a finding derived from an annotation in an older context.

        Returns:
            A conflict/unavailable envelope when freshness is not provable,
            otherwise ``None``.
        """

        saved_revision = self._last_annotation_context_revision
        current_revision = self._context_revision()
        if (
            not _is_revision(saved_revision)
            or not _is_revision(self._last_annotation_revision)
            or not _is_revision(current_revision)
        ):
            return self._local_result(
                "unavailable",
                "finding requires an authoritative annotation context revision",
                context=self._context,
            )
        if saved_revision != current_revision:
            return self._local_result(
                "conflict",
                "annotation receipt is stale for the current service context revision",
                context=self._context,
                conflict={
                    "annotation_context_revision": saved_revision,
                    "actual_context_revision": current_revision,
                },
            )
        if self._last_annotation_context is None:
            return self._local_result(
                "unavailable",
                "finding requires an authoritative annotation context identity",
                context=self._context,
            )
        if not self._context_matches_request(self._last_annotation_context, self._context):
            return self._local_result(
                "conflict",
                "annotation receipt is stale for the current service context identity",
                context=self._context,
                conflict={
                    "annotation_context": _safe_service_value(
                        self._last_annotation_context, secret=self._token
                    ),
                    "actual_context": _safe_service_value(self._context, secret=self._token),
                },
            )
        if self._last_annotation_episode_id != self._selected_episode_id:
            return self._local_result(
                "conflict",
                "annotation receipt is stale for the selected episode",
                context=self._context,
                conflict={
                    "annotation_episode_id": self._last_annotation_episode_id,
                    "selected_episode_id": self._selected_episode_id,
                },
            )
        context_episode_id = _service_attr(self._context, "episode_id")
        if context_episode_id and str(context_episode_id) != str(self._selected_episode_id):
            return self._local_result(
                "conflict",
                "annotation receipt does not match the current service episode",
                context=self._context,
                conflict={
                    "annotation_episode_id": self._last_annotation_episode_id,
                    "context_episode_id": context_episode_id,
                },
            )
        return None

    def _context_revision(self) -> Any:
        return _service_attr(self._context, "context_revision")

    def _source_revision(self) -> Any:
        value = _service_attr(self._context, "source_revision")
        if value is not None:
            return value
        return _service_attr(self._session, "source_revision")

    def _record_context_identity(self) -> dict[str, Any]:
        """Project only stable context identity for local record reopen.

        Returns:
            A JSON-compatible stable identity projection.
        """

        identity: dict[str, Any] = {}
        for field_name in (
            "context_revision",
            "campaign_id",
            "execution_id",
            "episode_id",
            "interval_id",
            "reference_id",
            "actor_id",
            "source_identity",
            "source_revision",
        ):
            value = _service_attr(self._context, field_name)
            if value is not None:
                identity[field_name] = _safe_service_value(value, secret=self._token)
        return identity

    def _record_projection(self) -> dict[str, Any]:
        """Describe the complete projection held by this facade instance.

        ``complete`` is intentionally scoped to this legacy in-memory facade.
        When BA-05 exposes ``read_saved_records``, :meth:`snapshot` emits a
        separate authoritative service-store projection instead.

        Returns:
            A scoped, non-authoritative projection descriptor.
        """

        return {
            "scope": "facade_instance",
            "scope_id": self._record_projection_scope,
            "completeness": "complete",
            "authoritative": False,
            "durability": "facade_instance_only",
            "fresh_process": "unknown",
            "context_identity": self._record_context_identity(),
            "annotation_ids": list(self._annotation_records),
            "finding_ids": list(self._finding_records),
        }

    def _service_cas_guard(
        self,
        *,
        expected_context: Any | None = None,
        expected_context_revision: int | None = None,
        expected_source_revision: int | str | None = None,
        expected_queue_state_revision: int | None = None,
        expected_queue_input_revision: int | None = None,
        expected_queue_input_identity: Any | None = None,
    ) -> dict[str, Any] | None:
        """Reject a browser write whose service context has moved.

        The browser's selection epoch is only a UI ordering guard.  A durable
        finding write also carries the service context, source, and queue CAS
        values observed by the editor.  Missing current values are conflicts,
        not evidence that the write is safe.

        Returns:
            A conflict envelope when any supplied CAS value is stale, otherwise
            ``None``.
        """

        checks: tuple[tuple[str, Any, Any], ...] = (
            ("expected_context_revision", expected_context_revision, self._context_revision()),
            ("expected_source_revision", expected_source_revision, self._source_revision()),
            (
                "expected_queue_state_revision",
                expected_queue_state_revision,
                self._queue_revisions.get("queue_state_revision"),
            ),
            (
                "expected_queue_input_revision",
                expected_queue_input_revision,
                self._queue_revisions.get("queue_input_revision"),
            ),
            (
                "expected_queue_input_identity",
                expected_queue_input_identity,
                self._queue_revisions.get("queue_input_identity"),
            ),
        )
        conflict: dict[str, Any] = {}
        revision_fields = {
            "expected_context_revision",
            "expected_queue_state_revision",
            "expected_queue_input_revision",
        }
        for field_name, expected, actual in checks:
            if expected is not None and (
                expected != actual or (field_name in revision_fields and not _is_revision(expected))
            ):
                conflict[field_name] = expected
                conflict[field_name.removeprefix("expected_")] = actual
        if expected_context is not None and not self._context_matches_request(
            expected_context, self._context
        ):
            conflict["expected_context"] = _safe_service_value(expected_context, secret=self._token)
            conflict["actual_context"] = _safe_service_value(self._context, secret=self._token)
        if not conflict:
            return None
        return self._local_result(
            "conflict",
            "stale service context, source, or queue binding",
            context=self._context,
            conflict=conflict,
        )

    def _selection_guard(self, expected: int | None) -> dict[str, Any] | None:
        """Check the browser epoch without treating it as BA-02/BA-05 CAS.

        Returns:
            A conflict envelope when stale, otherwise ``None``.
        """

        if expected is None:
            return None
        if isinstance(expected, bool) or not isinstance(expected, int):
            return self._local_result(
                "conflict",
                "expected selection epoch must be an integer",
                context=self._context,
                conflict={"expected_selection_epoch": expected},
            )
        if expected != self._selection_epoch:
            return self._local_result(
                "conflict",
                f"stale selection epoch: expected {expected}, current {self._selection_epoch}",
                context=self._context,
                conflict={
                    "expected_selection_epoch": expected,
                    "actual_selection_epoch": self._selection_epoch,
                    "context_revision": self._context_revision(),
                    **self._queue_revisions,
                },
            )
        return None

    @staticmethod
    def _primary_episode_id(packet: Any) -> str | None:
        primary = _service_attr(packet, "primary")
        if primary is None:
            primary = _service_attr(packet, "primary_episode")
        episode_id = _service_attr(primary, "episode_id")
        if episode_id:
            return str(episode_id)
        direct = _service_attr(packet, "episode_id")
        return str(direct) if direct else None

    def _attach_retained_scene(
        self, packet: dict[str, Any], episode_result: Any, *, operation_id: str | None
    ) -> None:
        """Attach only a scanner-admitted, service-read retained native scene."""

        if not _inspection_available(episode_result):
            return
        admitted_row = _service_attr(_service_attr(episode_result, "value"), "row")
        if not isinstance(admitted_row, Mapping):
            return
        scan_defaults = admitted_row.get("_audit_scan_identity_defaults")
        projection = project_retained_native_trace(
            admitted_row,
            request_id=operation_id,
            trusted_scan_identity_defaults=(
                scan_defaults if isinstance(scan_defaults, Mapping) else None
            ),
        )
        if projection["status"] not in {"complete", "partial"}:
            return
        editor_model = _safe_service_value(projection["editor_model"], secret=self._token)
        if not isinstance(editor_model, dict):
            return
        packet["editor_model"] = editor_model
        packet["projection_status"] = projection["status"]
        packet["cursor"] = {"time_s": projection["panel_model"]["time"]["origin_s"]}
        self._selected_packet = deepcopy(packet)

    def _next_projection(  # noqa: PLR0915
        self, result: Any, *, operation_id: str | None
    ) -> dict[str, Any]:
        """Project a successful queue result only after binding its episode.

        Returns:
            A service envelope with explicit queue, packet, and episode fields.
        """

        envelope = _service_result_mapping(result, secret=self._token)
        self._remember_result(result)
        if not _result_succeeded(result):
            envelope.pop("presentation_status", None)
            envelope["presentation_status"] = {
                "conflict": "conflict",
                "denied": "denied",
                "failed": "failed",
                "unavailable": "unavailable",
            }.get(str(envelope.get("status")), "unavailable")
            return envelope
        value = _service_attr(result, "value")
        if value is None:
            envelope["presentation_status"] = "empty"
            return envelope
        queue_value = _value_mapping(value, secret=self._token)
        packet_object = _service_attr(value, "packet")
        if packet_object is None and isinstance(value, Mapping):
            packet_object = value.get("packet")
        packet = _value_mapping(packet_object, secret=self._token)
        episode_id = self._primary_episode_id(packet_object)
        if episode_id is None:
            envelope["status"] = "unavailable"
            envelope["reason"] = "BA-02 queue result has no digest-bound primary episode"
            envelope["selection_status"] = str(_service_attr(result, "status", ""))
            envelope["presentation_status"] = "unavailable"
            return envelope
        primary = _service_attr(packet_object, "primary")
        packet["episode_id"] = episode_id
        for field_name in ("execution_id", "planner_id", "scenario_id", "seed"):
            field_value = _service_attr(primary, field_name)
            if field_value is not None:
                packet.setdefault(field_name, _safe_service_value(field_value, secret=self._token))

        request_context = deepcopy(self._context)
        episode_result = self._call_service(
            "read_episode",
            episode_id,
            context=request_context,
            operation_id=f"{operation_id}:episode" if operation_id else None,
        )
        episode_envelope = _service_result_mapping(episode_result, secret=self._token)
        returned_context = _service_attr(episode_result, "context")
        if returned_context is None or not self._context_matches_request(
            request_context, returned_context
        ):
            # Do not let a successful read for a different episode/context
            # replace the queue selection.  Clearing the retained write state
            # also makes a subsequent browser write fail closed.
            self._selected_packet = None
            self._selected_episode_id = None
            self._last_episode = None
            self._last_next = None
            self._last_queue = None
            self._clear_retained_annotation()
            envelope.pop("packet", None)
            envelope.pop("value", None)
            envelope["service_status"] = envelope.get("status")
            envelope["status"] = "unavailable"
            envelope["reason"] = (
                "read_episode returned a context that does not match the queue selection"
            )
            envelope["conflict"] = {
                "expected_context": _safe_service_value(request_context, secret=self._token),
                "actual_context": _safe_service_value(returned_context, secret=self._token),
            }
            envelope["queue"] = queue_value
            envelope["context"] = _safe_service_value(request_context, secret=self._token)
            envelope["episode_result"] = episode_envelope
            envelope["selection_status"] = str(_service_attr(result, "status", ""))
            envelope["inspection_status"] = episode_envelope["status"]
            envelope["presentation_status"] = "unavailable"
            return envelope
        self._remember_result(episode_result)
        self._last_next = result
        self._last_episode = episode_result
        self._last_queue = value
        self._selected_episode_id = episode_id
        self._selected_packet = deepcopy(packet)
        self._queue_revisions = self._codex_queue_revisions(value)
        self._queue_revisions = {
            key: _safe_service_value(item, secret=self._token)
            for key, item in self._queue_revisions.items()
            if item is not None
        }
        # Queue/browser payloads are selection hints, not trace authority.
        self._attach_retained_scene(packet, episode_result, operation_id=operation_id)
        envelope["value"] = _safe_service_value(value, secret=self._token)
        envelope["queue"] = queue_value
        envelope["packet"] = deepcopy(self._selected_packet)
        envelope["episode"] = episode_envelope.get("value")
        envelope["episode_result"] = episode_envelope
        envelope["context"] = _safe_service_value(self._context, secret=self._token)
        artifact_status_result = self.read_selected_artifact_status()
        envelope["artifact_status_result"] = artifact_status_result
        status_binding = self._artifact_status_binding()
        envelope["artifact_status"] = _artifact_status_projection(
            artifact_status_result.get("value"),
            require_complete_binding=True,
            secret=self._token,
            **status_binding,
        )
        envelope.update(self._queue_revisions)
        envelope["context_revision"] = _safe_service_value(
            self._context_revision(), secret=self._token
        )
        envelope["selection_epoch"] = self._selection_epoch + 1
        envelope["selection_revision"] = self._selection_epoch + 1
        envelope["selection_revision_authority"] = "ui_async_epoch_only"
        envelope["selection_status"] = str(_service_attr(result, "status", ""))
        envelope["inspection_status"] = episode_envelope["status"]
        envelope["presentation_status"] = (
            "selected"
            if _inspection_available(episode_result)
            else (
                "unavailable"
                if episode_envelope["status"] in {"complete", "committed"}
                else episode_envelope["status"]
            )
        )
        if not _inspection_available(episode_result):
            envelope["inspection_reason"] = episode_envelope["reason"]
        self._selection_epoch += 1
        return envelope

    def next(
        self,
        *,
        expected_selection_revision: int | None = None,
        context: Any = None,
        expected_context_revision: int | None = None,
        expected_queue_state_revision: int | None = None,
        expected_queue_input_revision: int | None = None,
        force_current: bool = False,
        operation_id: str | None = None,
    ) -> Mapping[str, Any]:
        """Select BA-02 Next and bind the typed primary episode for inspection.

        Returns:
            The original service envelope plus a safe presentation projection.
        """

        guard = self._selection_guard(expected_selection_revision)
        if guard is not None:
            return guard
        selected_context = self._context if context is None else context
        result = self._call_service(
            "next",
            context=selected_context,
            expected_context_revision=expected_context_revision,
            expected_queue_state_revision=expected_queue_state_revision,
            expected_queue_input_revision=expected_queue_input_revision,
            force_current=force_current,
            operation_id=operation_id,
        )
        return self._next_projection(result, operation_id=operation_id)

    def record_human_review(
        self,
        *,
        outcome: str,
        expected_selection_revision: int,
        expected_context_revision: int,
        expected_queue_state_revision: int,
        expected_queue_input_revision: int,
        operation_id: str,
    ) -> Mapping[str, Any]:
        """Forward an explicit human full-episode review under exact CAS guards.

        Returns:
            The source-bound BA-05 result; selection itself never grants credit.
        """

        guard = self._selection_guard(expected_selection_revision)
        if guard is not None:
            return guard
        if not self._selected_episode_id or not self._selected_packet:
            return self._local_result("unavailable", "no selected review packet")
        if expected_context_revision != self._context_revision():
            return self._local_result("conflict", "stale service context revision")
        result = self._call_service(
            "record_human_review",
            outcome=outcome,
            context=self._context,
            expected_context_revision=expected_context_revision,
            expected_queue_state_revision=expected_queue_state_revision,
            expected_queue_input_revision=expected_queue_input_revision,
            operation_id=operation_id,
        )
        self._remember_result(result)
        envelope = _service_result_mapping(result, secret=self._token)
        if _result_succeeded(result):
            coverage = self.read_coverage(operation_id=f"{operation_id}:coverage")
            envelope["coverage_result"] = coverage
        return envelope

    def read_context(self, *, operation_id: str | None = None) -> Mapping[str, Any]:
        """Read the service-owned selection context.

        Returns:
            The BA-05 service result envelope.
        """

        result = self._call_service("read_context", operation_id=operation_id)
        self._remember_result(result)
        return _service_result_mapping(result, secret=self._token)

    def update_context(
        self,
        context: Any,
        *,
        expected_context_revision: int,
        operation_id: str | None = None,
    ) -> Mapping[str, Any]:
        """CAS-update context through BA-05 without treating UI epoch as CAS.

        Returns:
            The BA-05 service result envelope.
        """

        result = self._call_service(
            "update_context",
            context,
            expected_context_revision=expected_context_revision,
            operation_id=operation_id,
        )
        envelope = _service_result_mapping(result, secret=self._token)
        if _result_succeeded(result):
            authoritative_context = _service_attr(result, "context")
            if authoritative_context is None:
                candidate = _service_attr(result, "value")
                if _service_attr(candidate, "context_revision") is not None:
                    authoritative_context = candidate
            expected_next_revision = (
                expected_context_revision + 1 if _is_revision(expected_context_revision) else None
            )
            requested_revision = _service_attr(context, "context_revision")
            valid_update = (
                expected_next_revision is not None
                and requested_revision == expected_next_revision
                and self._context_matches_request(context, authoritative_context)
            )
            if not valid_update:
                self._clear_retained_annotation()
                envelope["service_status"] = envelope.get("status")
                envelope["status"] = "unavailable"
                envelope["reason"] = (
                    "update_context did not return the requested authoritative context"
                )
            else:
                self._context = authoritative_context
        else:
            self._remember_result(result)
        return envelope

    def read_episode(
        self, episode_id: str | None = None, *, operation_id: str | None = None
    ) -> Mapping[str, Any]:
        """Read one typed episode and preserve service missingness verbatim.

        Returns:
            The BA-05 service result envelope without an invented editor model.
        """

        selected = episode_id or self._selected_episode_id
        if not selected:
            return self._local_result("unavailable", "no digest-bound episode is selected")
        result = self._call_service(
            "read_episode",
            str(selected),
            context=self._context,
            operation_id=operation_id,
        )
        self._remember_result(result)
        return _service_result_mapping(result, secret=self._token)

    def _saved_record_projection(self, stored: Any) -> tuple[str, str, dict[str, Any]] | None:
        """Project one BA-03 ``StoredRecord`` without manufacturing review state.

        BA-05 owns source, episode, and revision filtering.  This adapter only
        unwraps the already-filtered record and carries its durable revision
        metadata into the browser model.  Missing or contradictory metadata is
        rejected instead of being presented as a current record.

        Returns:
            ``(record_type, record_id, record)`` for a valid stored record, or
            ``None`` when the service value cannot be safely projected.
        """

        record = _service_attr(stored, "record")
        if record is None:
            record = _service_attr(stored, "payload")
        if record is None and isinstance(stored, Mapping):
            record = stored.get("value")
        record_type = _service_attr(stored, "record_type")
        record_id = _service_attr(stored, "record_id")
        deleted = _service_attr(stored, "deleted")
        revision = _service_attr(stored, "revision")
        global_revision = _service_attr(stored, "global_revision")
        if (
            deleted is not False
            or not isinstance(record_type, str)
            or record_type not in {"annotation", "finding"}
            or not isinstance(record_id, str)
            or not record_id
            or not _is_revision(revision)
            or revision <= 0
            or not _is_revision(global_revision)
            or global_revision <= 0
            or record is None
        ):
            return None
        projected = _value_mapping(record, secret=self._token)
        if not projected:
            return None
        embedded_type = projected.get("record_type")
        if embedded_type is not None and str(embedded_type) != record_type:
            return None
        embedded_id = projected.get("record_id")
        if embedded_id is not None and str(embedded_id) != record_id:
            return None
        id_field = "annotation_id" if record_type == "annotation" else "finding_id"
        embedded_record_id = projected.get(id_field)
        if embedded_record_id is not None and str(embedded_record_id) != record_id:
            return None
        projected["record_type"] = record_type
        projected["record_id"] = record_id
        projected[id_field] = record_id
        projected["revision"] = revision
        projected["global_revision"] = global_revision
        for field_name in ("operation_id", "committed_at"):
            value = _service_attr(stored, field_name)
            if value is not None:
                projected[field_name] = _safe_service_value(value, secret=self._token)
        return record_type, record_id, projected

    def _saved_record_context_matches(
        self, requested: Any, returned: Any, *, episode_id: str
    ) -> bool:
        """Require a saved-record read to return the exact requested binding.

        Returns:
            ``True`` only when all stable source and selection fields match.
        """

        if returned is None or not self._context_matches_request(requested, returned):
            return False
        if str(_service_attr(returned, "episode_id", "")) != str(episode_id):
            return False
        # ``source_revision=0`` is a real digest-only binding for durable
        # records, not an unknown wildcard.  The generic context helper keeps
        # older read paths compatible, so this read path checks it explicitly.
        for field_name in (
            "campaign_id",
            "execution_id",
            "episode_id",
            "interval_id",
            "reference_id",
            "actor_id",
            "source_identity",
            "source_revision",
        ):
            expected = _service_attr(requested, field_name)
            if expected is None:
                continue
            if _service_attr(returned, field_name) != expected:
                return False
        return True

    def read_saved_records(
        self,
        episode_id: str | None = None,
        *,
        context: Any = None,
        operation_id: str | None = None,
    ) -> Mapping[str, Any]:
        """Read source-bound BA-03 annotations/findings for the selected episode.

        The BA-05 service remains authoritative for authentication, source
        identity, generated episode membership, and source revision.  The
        facade does not synthesize rows when that read is unavailable and does
        not convert agent annotations or proposed findings into human review.

        Returns:
            A safe BA-05 result envelope with a flattened ``records`` list on
            successful reads, or the service's explicit unavailable/conflict
            status when the binding cannot be proved.
        """

        selected = self._selected_episode_id if episode_id is None else episode_id
        if not selected:
            return self._local_result("unavailable", "no digest-bound episode is selected")
        requested_context = self._context if context is None else context
        result = self._call_service(
            "read_saved_records",
            str(selected),
            context=requested_context,
            operation_id=operation_id,
        )
        envelope = _service_result_mapping(result, secret=self._token)
        if not _result_succeeded(result):
            self._remember_result(result)
            return envelope
        returned_context = _service_attr(result, "context")
        if not self._saved_record_context_matches(
            requested_context, returned_context, episode_id=str(selected)
        ):
            envelope["service_status"] = envelope.get("status")
            envelope["status"] = "unavailable"
            envelope["reason"] = "read_saved_records returned a context that does not match"
            envelope["conflict"] = {
                "expected_context": _safe_service_value(requested_context, secret=self._token),
                "actual_context": _safe_service_value(returned_context, secret=self._token),
            }
            return envelope
        raw_records = _service_attr(result, "value")
        if isinstance(raw_records, Mapping):
            raw_records = raw_records.get("records", raw_records.get("value"))
        if not isinstance(raw_records, Sequence) or isinstance(raw_records, (str, bytes)):
            envelope["service_status"] = envelope.get("status")
            envelope["status"] = "unavailable"
            envelope["reason"] = "read_saved_records returned malformed durable records"
            return envelope
        projections: list[dict[str, Any]] = []
        seen_ids: set[tuple[str, str]] = set()
        for stored in raw_records:
            projected = self._saved_record_projection(stored)
            if projected is None:
                envelope["service_status"] = envelope.get("status")
                envelope["status"] = "unavailable"
                envelope["reason"] = "read_saved_records returned malformed durable records"
                return envelope
            record_type, record_id, record = projected
            if (record_type, record_id) in seen_ids:
                envelope["service_status"] = envelope.get("status")
                envelope["status"] = "unavailable"
                envelope["reason"] = "read_saved_records returned duplicate durable records"
                return envelope
            seen_ids.add((record_type, record_id))
            projections.append(record)
        self._remember_result(result)
        envelope["records"] = projections
        envelope["episode_id"] = str(selected)
        return envelope

    # Frozen BA-05 callers use both names for this read operation.
    read_records = read_saved_records
    list_saved_records = read_saved_records

    def _read_episode_field(
        self, operation: str, episode_id: str | None, *, operation_id: str | None
    ) -> Mapping[str, Any]:
        selected = episode_id or self._selected_episode_id
        if not selected:
            return self._local_result("unavailable", "no digest-bound episode is selected")
        result = self._call_service(
            operation,
            str(selected),
            context=self._context,
            operation_id=operation_id,
        )
        self._remember_result(result)
        return _service_result_mapping(result, secret=self._token)

    def read_metrics(
        self, episode_id: str | None = None, *, operation_id: str | None = None
    ) -> Mapping[str, Any]:
        """Read source-bound metrics without manufacturing a panel stream.

        Returns:
            The BA-05 service result envelope.
        """

        return self._read_episode_field("read_metrics", episode_id, operation_id=operation_id)

    def read_events(
        self, episode_id: str | None = None, *, operation_id: str | None = None
    ) -> Mapping[str, Any]:
        """Read retained events and preserve unavailable status.

        Returns:
            The BA-05 service result envelope.
        """

        return self._read_episode_field("read_events", episode_id, operation_id=operation_id)

    def read_geometry(
        self, episode_id: str | None = None, *, operation_id: str | None = None
    ) -> Mapping[str, Any]:
        """Read retained geometry and preserve unavailable status.

        Returns:
            The BA-05 service result envelope.
        """

        return self._read_episode_field("read_geometry", episode_id, operation_id=operation_id)

    def read_signals(
        self, episode_id: str | None = None, *, operation_id: str | None = None
    ) -> Mapping[str, Any]:
        """Read detector signals without promoting them to findings.

        Returns:
            The BA-05 service result envelope.
        """

        selected = episode_id or self._selected_episode_id
        result = self._call_service(
            "read_signals",
            selected,
            context=self._context,
            operation_id=operation_id,
        )
        self._remember_result(result)
        return _service_result_mapping(result, secret=self._token)

    def related_cases(
        self,
        episode_id: str | None = None,
        *,
        mode: str = "same_scenario_across_planners",
        limit: int = 20,
        operation_id: str | None = None,
    ) -> Mapping[str, Any]:
        """Read explained peer candidates; never confirm finding membership.

        Returns:
            The BA-05 service result envelope.
        """

        selected = episode_id or self._selected_episode_id
        if not selected:
            return self._local_result("unavailable", "no digest-bound episode is selected")
        result = self._call_service(
            "related_cases",
            str(selected),
            mode=mode,
            limit=limit,
            context=self._context,
            operation_id=operation_id,
        )
        self._remember_result(result)
        return _service_result_mapping(result, secret=self._token)

    def read_queue(self, *, limit: int = 20, operation_id: str | None = None) -> Mapping[str, Any]:
        """Read a queue snapshot; this method never advances selection.

        Returns:
            The BA-05 service result envelope.
        """

        result = self._call_service(
            "read_queue", limit=limit, context=self._context, operation_id=operation_id
        )
        self._remember_result(result)
        return _service_result_mapping(result, secret=self._token)

    def read_coverage(self, *, operation_id: str | None = None) -> Mapping[str, Any]:
        """Read service-owned coverage without synthesizing review credit.

        Returns:
            The BA-05 service result envelope.
        """

        result = self._call_service(
            "read_coverage", context=self._context, operation_id=operation_id
        )
        self._remember_result(result)
        self._last_coverage = result
        return _service_result_mapping(result, secret=self._token)

    @staticmethod
    def _source_identity_token(value: Any) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, Mapping):
            sources = value.get("sources")
            if isinstance(sources, Mapping):
                for item in sources.values():
                    if isinstance(item, Mapping):
                        for key in ("sha256", "artifact_id"):
                            if item.get(key):
                                return str(item[key])
            for key in ("sha256", "artifact_id"):
                if value.get(key):
                    return str(value[key])
        return ""

    @staticmethod
    def _canonical_source_ref(value: Any) -> dict[str, str] | None:
        """Project a source identity to the closed BA-03 ``SourceRef`` shape.

        The browser intentionally sends only the public nine-field ``SourceRef``
        projection.  Native retained-trace identities may carry scanner and
        integrity evidence beside those fields, so matching the raw mappings
        would reject an otherwise admitted source.  Digest aliases are handled
        the same way as the browser projection for native identities that only
        expose a declared or computed digest.

        Returns:
            The browser-compatible SourceRef projection, or ``None`` when the
            required artifact identity fields are unavailable.
        """

        if isinstance(value, SourceRef):
            value = {
                field_name: getattr(value, field_name) for field_name in _SOURCE_REF_FIELD_NAMES
            }
        if not isinstance(value, Mapping):
            return None
        projected: dict[str, str] = {}
        for field_name in _SOURCE_REF_FIELD_NAMES:
            candidate = value.get(field_name)
            if field_name == "sha256" and not candidate:
                candidate = value.get("declared_sha256") or value.get("computed_sha256") or ""
            if isinstance(candidate, str) and candidate:
                projected[field_name] = candidate
        if any(field_name not in projected for field_name in _SOURCE_REF_REQUIRED_FIELD_NAMES):
            return None
        return projected

    @classmethod
    def _match_visual_source(
        cls, sources: Mapping[Any, Any], supplied_ref: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], dict[str, str]]:
        """Find the unique admitted source matching a browser SourceRef.

        Returns:
            The admitted source mapping and its canonical SourceRef projection.
        """

        supplied_projection = cls._canonical_source_ref(supplied_ref)
        if supplied_projection is None:
            raise AuditWorkbenchConflictError("annotation visual source is not selected")
        matches = [
            (item, projection)
            for item in sources.values()
            if isinstance(item, Mapping)
            for projection in (cls._canonical_source_ref(item),)
            if projection is not None and projection == supplied_projection
        ]
        if not matches:
            raise AuditWorkbenchConflictError("annotation visual source is not selected")
        if len(matches) > 1:
            raise AuditWorkbenchConflictError("annotation visual source is ambiguous")
        return matches[0]

    def _bind_visual_annotation_reference(
        self,
        raw_reference: Any,
        *,
        index: int,
        sources: Mapping[Any, Any],
        campaign_projection: dict[str, str] | None,
        campaign_ref: Mapping[str, Any],
        current_revision: Any,
        authoritative_revision: str,
        bindings: dict[str, Any],
    ) -> Any:
        """Bind one nested visual reference and retain its original identity.

        Returns:
            The detached reference mapping with authoritative source fields, or
            the original value when it is not a source-bearing mapping.
        """

        if not isinstance(raw_reference, Mapping):
            return raw_reference
        reference = dict(raw_reference)
        supplied_ref = reference.get("source")
        if not isinstance(supplied_ref, Mapping):
            return reference
        supplied_projection = self._canonical_source_ref(supplied_ref)
        if supplied_projection is None:
            raise AuditWorkbenchConflictError("annotation reference visual source is not selected")
        if campaign_projection is not None and supplied_projection == campaign_projection:
            return reference
        admitted, admitted_projection = self._match_visual_source(sources, supplied_ref)
        nested_revision = reference.get("source_revision")
        admitted_revision = admitted_projection.get("source_commit")
        if nested_revision not in (None, "", 0, "0", admitted_revision, current_revision):
            if current_revision is None or str(nested_revision) != str(current_revision):
                raise AuditWorkbenchConflictError(
                    "annotation reference visual source revision is stale"
                )
        reference_id = str(reference.get("reference_id") or index + 1)
        binding_key = reference_id
        if binding_key in bindings:
            binding_key = f"{reference_id}#{index + 1}"
        bindings[binding_key] = {
            "source_ref": _safe_service_value(admitted, secret=self._token),
            "source_identity": admitted_projection.get("sha256")
            or admitted_projection.get("artifact_id"),
            "source_revision": nested_revision,
            "source_provenance": _safe_service_value(
                {
                    key: value
                    for key, value in admitted.items()
                    if key not in _SOURCE_REF_FIELD_NAMES
                },
                secret=self._token,
            ),
        }
        reference["source"] = {**campaign_ref, "source_commit": ""}
        reference["source_revision"] = authoritative_revision
        return reference

    def _bind_visual_annotation_references(
        self,
        payload: dict[str, Any],
        *,
        sources: Mapping[Any, Any],
        campaign_ref: Mapping[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        """Rebind nested visual references to the authoritative campaign source.

        BA-03 validates every ``Reference.source`` against the session source,
        while SREV-17 references intentionally point at the selected retained
        trace. Validate each nested trace source against the admitted source
        set before replacing it, and retain its visual binding as metadata.
        """

        raw_references = payload.get("references")
        if not isinstance(raw_references, (list, tuple)):
            return
        campaign_projection = self._canonical_source_ref(campaign_ref)
        bound_references: list[Any] = []
        bindings: dict[str, Any] = {}
        existing_bindings = metadata.get("reference_source_bindings")
        if isinstance(existing_bindings, Mapping):
            bindings.update(existing_bindings)
        current_revision = self._source_revision()
        authoritative_revision = "0" if current_revision in (None, "") else str(current_revision)
        for index, raw_reference in enumerate(raw_references):
            bound_references.append(
                self._bind_visual_annotation_reference(
                    raw_reference,
                    index=index,
                    sources=sources,
                    campaign_projection=campaign_projection,
                    campaign_ref=campaign_ref,
                    current_revision=current_revision,
                    authoritative_revision=authoritative_revision,
                    bindings=bindings,
                )
            )
        payload["references"] = bound_references
        if bindings:
            metadata["reference_source_bindings"] = bindings

    def _bind_visual_annotation_source(self, payload: dict[str, Any]) -> bool:
        """Translate an exact selected SREV trace source into BA-05 source authority.

        The editor's trace digest identifies the visual evidence, not the
        campaign file whose revision BA-05 stores. Preserve both identities
        without accepting a browser-nominated replacement trace or source.

        Returns:
            Whether an exact visual source was rebound to campaign authority.
        """

        supplied_ref = payload.get("source_ref")
        editor = _service_attr(self._selected_packet, "editor_model")
        sources = _service_attr(_service_attr(editor, "source_identity"), "sources")
        if not isinstance(supplied_ref, Mapping) or not isinstance(sources, Mapping):
            return False
        admitted, admitted_projection = self._match_visual_source(sources, supplied_ref)
        visual_identity = payload.get("source_identity")
        if visual_identity not in (
            admitted_projection.get("sha256"),
            admitted_projection.get("artifact_id"),
        ):
            raise AuditWorkbenchConflictError("annotation visual source identity is stale")
        visual_revision = payload.get("source_revision")
        if visual_revision not in (
            None,
            admitted_projection.get("source_commit"),
            self._source_revision(),
        ):
            raise AuditWorkbenchConflictError("annotation visual source revision is stale")
        campaign_ref = _service_attr(self._session, "source_ref")
        campaign_digest = _service_attr(self._session, "source_digest")
        if not isinstance(campaign_digest, str) or not campaign_digest or campaign_ref is None:
            raise AuditWorkbenchUnavailableError("authoritative campaign source is unavailable")
        if is_dataclass(campaign_ref) and not isinstance(campaign_ref, type):
            campaign_ref = {
                field.name: getattr(campaign_ref, field.name) for field in fields(campaign_ref)
            }
        if not isinstance(campaign_ref, Mapping):
            raise AuditWorkbenchUnavailableError("authoritative campaign source is unavailable")
        metadata = payload.get("metadata")
        metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        metadata["visual_source_ref"] = _safe_service_value(admitted, secret=self._token)
        metadata["visual_source_provenance"] = _safe_service_value(
            {key: value for key, value in admitted.items() if key not in _SOURCE_REF_FIELD_NAMES},
            secret=self._token,
        )
        self._bind_visual_annotation_references(
            payload,
            sources=sources,
            campaign_ref=campaign_ref,
            metadata=metadata,
        )
        metadata["visual_source_identity"] = visual_identity
        if visual_revision is not None:
            metadata["visual_source_revision"] = visual_revision
        if campaign_ref.get("source_commit"):
            metadata["campaign_source_commit"] = campaign_ref["source_commit"]
        payload["metadata"] = metadata
        # BA-03's annotation contract compares source_ref.source_commit to
        # source_revision, although the service's source revision may be a
        # scan version (or digest-only zero), not a Git commit. The digest is
        # authoritative here; retain the commit separately without asserting
        # that these distinct revisions are interchangeable.
        payload["source_ref"] = {**campaign_ref, "source_commit": ""}
        payload["source_identity"] = campaign_digest
        payload["source_revision"] = self._source_revision()
        return True

    def _bind_ordinary_annotation_source(self, payload: dict[str, Any]) -> None:
        """Keep the existing campaign-source binding for nonvisual records."""

        source_identity = self._source_identity_token(payload.get("source_identity"))
        if not source_identity:
            source_identity = self._source_identity_token(
                _service_attr(self._context, "source_identity")
            )
        if source_identity:
            original_identity = payload.get("source_identity")
            payload["source_identity"] = source_identity
            if isinstance(original_identity, Mapping):
                metadata = payload.get("metadata")
                metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
                metadata.setdefault("source_identity", _safe_service_value(original_identity))
                payload["metadata"] = metadata
        if "source_revision" not in payload:
            source_revision = self._source_revision()
            if source_revision is not None:
                payload["source_revision"] = source_revision

    def _annotation_payload(self, annotation: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(annotation, Mapping):
            raise AuditWorkbenchError("annotation must be a mapping")
        if not self._selected_episode_id:
            raise AuditWorkbenchUnavailableError("no digest-bound episode is selected")
        payload = dict(annotation)
        annotation_id = str(payload.get("annotation_id") or payload.get("record_id") or "")
        if not annotation_id:
            raise AuditWorkbenchError("annotation_id is required")
        supplied_episode = payload.get("episode_id")
        if supplied_episode and str(supplied_episode) != self._selected_episode_id:
            raise AuditWorkbenchConflictError(
                "annotation episode does not match selected service context"
            )
        payload.update(
            {
                "schema_version": payload.get("schema_version", "audit-record.v1"),
                "record_type": "annotation",
                "record_id": annotation_id,
                "annotation_id": annotation_id,
                "episode_id": self._selected_episode_id,
            }
        )
        if not self._bind_visual_annotation_source(payload):
            self._bind_ordinary_annotation_source(payload)
        return payload

    def _write_kwargs(
        self,
        *,
        expected_revision: int | None,
        expected_source_revision: int | str | None,
        operation_id: str | None,
    ) -> dict[str, Any]:
        source_revision = (
            expected_source_revision
            if expected_source_revision is not None
            else self._source_revision()
        )
        kwargs: dict[str, Any] = {
            "context": self._context,
            "expected_revision": expected_revision,
            "operation_id": operation_id,
        }
        if source_revision is not None:
            kwargs["expected_source_revision"] = source_revision
        return kwargs

    def save_annotation(
        self,
        annotation: Mapping[str, Any],
        *,
        expected_selection_revision: int | None = None,
        expected_revision: int | None = None,
        expected_source_revision: int | str | None = None,
        expected_context_revision: int | None = None,
        operation_id: str | None = None,
    ) -> Mapping[str, Any]:
        """Persist a typed annotation through BA-05's source/CAS boundary.

        Returns:
            The original BA-05 status, receipt, context, and CAS fields.
        """

        guard = self._selection_guard(expected_selection_revision)
        if guard is not None:
            return guard
        if (
            expected_context_revision is not None
            and expected_context_revision != self._context_revision()
        ):
            return self._local_result(
                "conflict",
                "stale service context revision",
                context=self._context,
                conflict={
                    "expected_context_revision": expected_context_revision,
                    "actual_context_revision": self._context_revision(),
                },
            )
        try:
            payload = self._annotation_payload(annotation)
        except AuditWorkbenchConflictError as exc:
            return self._local_result("conflict", str(exc), context=self._context)
        except AuditWorkbenchUnavailableError as exc:
            return self._local_result("unavailable", str(exc), context=self._context)
        except AuditWorkbenchError as exc:
            return self._local_result("failed", str(exc), context=self._context)
        request_context = deepcopy(self._context)
        result = self._call_service(
            "write_annotation",
            payload,
            **self._write_kwargs(
                expected_revision=expected_revision,
                expected_source_revision=expected_source_revision,
                operation_id=operation_id,
            ),
        )
        self._remember_result(result)
        envelope = _write_result_mapping(
            result,
            secret=self._token,
            operation="write_annotation",
            requested_record_type="annotation",
            requested_record_id=str(payload["annotation_id"]),
            requested_operation_id=operation_id,
            requested_operation_type="write.annotation",
            expected_context=request_context,
        )
        if envelope.get("presentation_status") == "saved":
            self._last_annotation = deepcopy(payload)
            self._last_annotation_context_revision = self._context_revision()
            self._last_annotation_episode_id = self._selected_episode_id
            self._last_annotation_context = deepcopy(self._context)
            revision = _service_attr(_service_attr(result, "value"), "revision")
            self._last_annotation_revision = revision if _is_revision(revision) else None
            self._last_annotation_receipt = deepcopy(envelope)
            self._last_annotation_record = deepcopy(payload)
            receipt = envelope.get("receipt")
            if isinstance(receipt, Mapping):
                for field_name in (
                    "revision",
                    "record_revision",
                    "global_revision",
                    "operation_id",
                    "replayed",
                ):
                    if field_name in receipt:
                        self._last_annotation_record[field_name] = deepcopy(receipt[field_name])
            annotation_id = str(self._last_annotation_record["annotation_id"])
            self._annotation_records[annotation_id] = deepcopy(self._last_annotation_record)
            self._annotation_receipts[annotation_id] = deepcopy(envelope)
        return envelope

    save_annotation_record = save_annotation

    def _retained_annotation_payload(
        self, annotation: Mapping[str, Any] | None
    ) -> dict[str, Any] | None:
        """Return the exact committed annotation after rejecting edits."""

        if self._last_annotation is None:
            return None
        retained = self._last_annotation
        if annotation is None:
            return deepcopy(retained)
        candidate = self._annotation_payload(annotation)
        for field_name in ("record_type", "schema_version", "annotation_id", "record_id"):
            if field_name in annotation and annotation[field_name] != retained.get(field_name):
                raise AuditWorkbenchConflictError(
                    "finding annotation changes the committed annotation identity"
                )
        if candidate != retained:
            raise AuditWorkbenchConflictError(
                "finding annotation changes the committed annotation content"
            )
        return deepcopy(retained)

    def _finding_record_projection(
        self,
        result: Any,
        envelope: Mapping[str, Any],
        payload: Mapping[str, Any],
        *,
        finding_id: str,
        title: str | None,
    ) -> dict[str, Any]:
        """Project a committed proposed finding for the browser/reopen seam.

        BA-05's ``finding_from_annotation`` receipt is authoritative for the
        write, while the operation contract supplies the proposed/candidate
        shape when the typed result does not carry the full record.

        Returns:
            A safe finding record projection carrying the commit revision.
        """

        value = _service_attr(result, "value")
        record = _value_mapping(_service_attr(value, "record"), secret=self._token)
        required_fields = {
            "finding_id",
            "title",
            "status",
            "candidate_members",
            "confirmed_members",
            "negative_controls",
        }
        if (
            not record
            or record.get("finding_id") != finding_id
            or not required_fields <= set(record)
        ):
            record = {
                "record_type": "finding",
                "record_id": finding_id,
                "finding_id": finding_id,
                "title": title or str(payload.get("classification", "review finding")),
                "status": "proposed",
                "candidate_members": [str(payload["episode_id"])],
                "confirmed_members": [],
                "negative_controls": [],
                "annotation_ids": [str(payload["annotation_id"])],
            }
        else:
            record.setdefault("record_type", "finding")
            record.setdefault("record_id", finding_id)
            record["finding_id"] = finding_id
        receipt = envelope.get("receipt")
        if isinstance(receipt, Mapping):
            for field_name in (
                "revision",
                "record_revision",
                "global_revision",
                "operation_id",
                "replayed",
            ):
                if field_name in receipt:
                    record[field_name] = deepcopy(receipt[field_name])
        projected = _safe_service_value(record, secret=self._token)
        return projected if isinstance(projected, dict) else {}

    def persist_finding(  # noqa: PLR0913
        self,
        annotation: Mapping[str, Any] | None = None,
        *,
        expected_selection_revision: int | None = None,
        expected_context: Any | None = None,
        expected_context_revision: int | None = None,
        expected_source_revision: int | str | None = None,
        expected_queue_state_revision: int | None = None,
        expected_queue_input_revision: int | None = None,
        expected_queue_input_identity: Any | None = None,
        operation_id: str | None = None,
        finding_id: str | None = None,
        title: str | None = None,
    ) -> Mapping[str, Any]:
        """Create a proposed finding only from the preceding committed annotation.

        Returns:
            The original BA-05 status and finding receipt.
        """

        guard = self._selection_guard(expected_selection_revision)
        if guard is not None:
            return guard
        service_cas_guard = self._service_cas_guard(
            expected_context=expected_context,
            expected_context_revision=expected_context_revision,
            expected_source_revision=expected_source_revision,
            expected_queue_state_revision=expected_queue_state_revision,
            expected_queue_input_revision=expected_queue_input_revision,
            expected_queue_input_identity=expected_queue_input_identity,
        )
        if service_cas_guard is not None:
            return service_cas_guard
        if self._last_annotation is None or self._last_annotation_receipt is None:
            return self._local_result(
                "unavailable",
                "finding requires a durably committed annotation receipt",
                context=self._context,
            )
        freshness_guard = self._annotation_freshness_guard()
        if freshness_guard is not None:
            return freshness_guard
        try:
            payload = self._retained_annotation_payload(annotation)
            if payload is None:
                return self._local_result(
                    "unavailable",
                    "finding requires a durably committed annotation receipt",
                    context=self._context,
                )
        except (
            AuditWorkbenchConflictError,
            AuditWorkbenchUnavailableError,
            AuditWorkbenchError,
        ) as exc:
            status = "conflict" if isinstance(exc, AuditWorkbenchConflictError) else "unavailable"
            if isinstance(exc, AuditWorkbenchError) and not isinstance(
                exc, (AuditWorkbenchConflictError, AuditWorkbenchUnavailableError)
            ):
                status = "failed"
            return self._local_result(status, str(exc), context=self._context)
        selected_id = self._selected_episode_id
        target_finding_id = finding_id or f"finding-{selected_id}"
        request_context = deepcopy(self._context)
        result = self._call_service(
            "finding_from_annotation",
            payload,
            finding_id=target_finding_id,
            title=title or str(payload.get("classification", "review finding")),
            context=self._context,
            operation_id=operation_id,
        )
        self._remember_result(result)
        envelope = _write_result_mapping(
            result,
            secret=self._token,
            operation="finding_from_annotation",
            requested_record_type="finding",
            requested_record_id=str(target_finding_id),
            requested_operation_id=operation_id,
            requested_operation_type="write.finding",
            expected_context=request_context,
        )
        envelope["annotation_receipt"] = _safe_service_value(
            self._last_annotation_receipt, secret=self._token
        )
        if envelope.get("presentation_status") == "saved":
            record = self._finding_record_projection(
                result,
                envelope,
                payload,
                finding_id=target_finding_id,
                title=title,
            )
            envelope["finding"] = record
            self._last_finding = deepcopy(record)
            self._last_finding_receipt = deepcopy(envelope)
            finding_record_id = str(record.get("finding_id") or target_finding_id)
            self._finding_records[finding_record_id] = deepcopy(record)
            self._finding_receipts[finding_record_id] = deepcopy(envelope)
        return envelope

    persist_finding_record = persist_finding

    def write_finding(
        self,
        finding: Mapping[str, Any],
        *,
        expected_revision: int | None = None,
        expected_source_revision: int | str | None = None,
        expected_selection_revision: int | None = None,
        operation_id: str | None = None,
    ) -> Mapping[str, Any]:
        """Persist an already typed finding while retaining candidate status.

        Returns:
            The original BA-05 status and commit receipt.
        """

        guard = self._selection_guard(expected_selection_revision)
        if guard is not None:
            return guard
        if not isinstance(finding, Mapping):
            return self._local_result("failed", "finding must be a mapping", context=self._context)
        payload = dict(finding)
        finding_id = str(payload.get("finding_id") or payload.get("record_id") or "")
        if not finding_id:
            return self._local_result("failed", "finding_id is required", context=self._context)
        payload.setdefault("schema_version", "audit-record.v1")
        payload["record_type"] = "finding"
        payload["record_id"] = finding_id
        payload["finding_id"] = finding_id
        request_context = deepcopy(self._context)
        result = self._call_service(
            "write_finding",
            payload,
            **self._write_kwargs(
                expected_revision=expected_revision,
                expected_source_revision=expected_source_revision,
                operation_id=operation_id,
            ),
        )
        self._remember_result(result)
        return _write_result_mapping(
            result,
            secret=self._token,
            operation="write_finding",
            requested_record_type="finding",
            requested_record_id=finding_id,
            requested_operation_id=operation_id,
            requested_operation_type="write.finding",
            expected_context=request_context,
        )

    def run_native_diagnostic(
        self, *, operation_id: str | None = None, **kwargs: Any
    ) -> Mapping[str, Any]:
        """Forward the server-only diagnostic call without claiming benchmark evidence.

        Returns:
            The diagnostic-only BA-05 result envelope.
        """

        result = self._call_service("run_native_diagnostic", operation_id=operation_id, **kwargs)
        self._remember_result(result)
        envelope = _service_result_mapping(result, secret=self._token)
        envelope.setdefault("evidence_boundary", "diagnostic_only")
        envelope.setdefault("scientific_claim_allowed", False)
        return envelope

    def materialize_selected(
        self, *, operation_id: str | None = None, **kwargs: Any
    ) -> Mapping[str, Any]:
        """Forward server-side materialization and expose only its result envelope.

        Returns:
            The BA-05 materialization result envelope.
        """

        result = self._call_service("materialize_selected", operation_id=operation_id, **kwargs)
        self._remember_result(result)
        envelope = _service_result_mapping(result, secret=self._token)
        value = _service_attr(result, "value")
        to_dict = getattr(value, "to_dict", None)
        if callable(to_dict):
            try:
                envelope["value"] = _safe_service_value(to_dict(), secret=self._token)
            except (AttributeError, TypeError, ValueError, RuntimeError):
                pass
        return envelope

    def _artifact_status_binding(self) -> dict[str, Any]:
        """Return the current service-held binding used to validate one status read."""

        episode_id = self._selected_episode_id
        if episode_id is None:
            episode_id = _service_attr(self._context, "episode_id")
        source_digest = _service_attr(self._context, "source_identity")
        if not source_digest:
            source_digest = _service_attr(self._session, "source_digest")
        if not episode_id:
            source_digest = None
        return {
            "episode_id": episode_id,
            "context_revision": self._context_revision(),
            "source_revision": self._source_revision(),
            "source_digest": source_digest,
        }

    def _artifact_status_envelope(
        self, result: Any, *, binding: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Project a status result without generic operation or receipt fields.

        Returns:
            A bounded artifact-status envelope tied to ``binding``.
        """

        value = _service_attr(result, "value")
        raw_result_status = str(_service_attr(result, "status", "unavailable"))
        result_status = (
            raw_result_status if raw_result_status in _ARTIFACT_RESULT_STATUSES else "unavailable"
        )
        binding_mismatch = _artifact_status_binding_mismatch(value, secret=self._token, **binding)
        context_mismatch = _artifact_status_context_mismatch(
            _service_attr(result, "context"), binding, secret=self._token
        )
        result_succeeded = raw_result_status in _ARTIFACT_SUCCESS_RESULT_STATUSES
        unsafe_reason = _artifact_reason_is_unsafe(
            _service_attr(result, "reason", ""), secret=self._token
        )
        if not result_succeeded:
            # A failed/conflict/denied envelope is not a capability result.
            # Never let an adapter smuggle a positive value through its error
            # status, even when the nested fields happen to match.
            value = None
            reason = _artifact_reason(_service_attr(result, "reason", ""), secret=self._token)
        elif binding_mismatch or context_mismatch or unsafe_reason:
            result_status = "conflict"
            reason = (
                "selected artifact status is stale"
                if binding_mismatch or context_mismatch
                else "selected artifact status is unavailable"
            )
            value = None
        else:
            reason = _artifact_reason(_service_attr(result, "reason", ""), secret=self._token)
        projected = _artifact_status_projection(
            value, require_complete_binding=True, secret=self._token, **binding
        )
        safe_context = {
            field: reference
            for field in ("context_revision", "episode_id", "source_revision")
            if (reference := _artifact_safe_reference(binding.get(field), secret=self._token))
            is not None
        }
        return {
            "schema_version": _ARTIFACT_STATUS_SCHEMA_VERSION,
            "status": result_status,
            "reason": reason,
            "value": projected,
            "context": safe_context,
        }

    def read_selected_artifact_status(self) -> Mapping[str, Any]:
        """Read selected artifact/native capability without starting work.

        Returns:
            A status-specific, source-bound envelope without operation receipts.
        """

        binding = self._artifact_status_binding()
        result = self._call_service(
            "read_selected_artifact_status", context=deepcopy(self._context)
        )
        # A status read never advances the remembered context.  This prevents a
        # delayed response from moving the selection shown by the workbench.
        return self._artifact_status_envelope(result, binding=binding)

    selected_artifact_status = read_selected_artifact_status

    def sync_finding(  # noqa: C901, PLR0913
        self,
        finding: Any | None = None,
        *,
        finding_id: str | None = None,
        repository: str | None = None,
        expected_finding_revision: int | None = None,
        expected_selection_revision: int | None = None,
        expected_context_revision: int | None = None,
        expected_source_revision: int | str | None = None,
        retry_ambiguous: bool = False,
        operation_id: str | None = None,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        """Publish one retained finding through the accepted BA-05 service.

        BA-06 supplies only the finding/repository identity and compare-and-swap
        coordinates.  Provider credentials, evidence, outbox state, and the
        canonical finding load remain inside BA-05.  The positional ``finding``
        form is retained for fixture callers; live requests use ``finding_id``.

        Returns:
            The service sync receipt or an explicit unavailable envelope.
        """

        guard = self._selection_guard(expected_selection_revision)
        if guard is not None:
            return guard
        service_cas_guard = self._service_cas_guard(
            expected_context_revision=expected_context_revision,
            expected_source_revision=expected_source_revision,
        )
        if service_cas_guard is not None:
            return service_cas_guard

        candidate: Mapping[str, Any] | None = None
        if isinstance(finding, Mapping):
            candidate = finding
        elif finding is not None:
            return self._local_result(
                "failed", "sync_finding finding must be a mapping", context=self._context
            )
        if candidate is None and finding_id:
            candidate = self._finding_records.get(str(finding_id))
        if candidate is None and self._last_finding is not None:
            candidate = self._last_finding
        if candidate is not None:
            candidate_id = candidate.get("finding_id") or candidate.get("record_id")
            if finding_id is None and candidate_id:
                finding_id = str(candidate_id)
            if expected_finding_revision is None:
                candidate_revision = candidate.get("revision", candidate.get("record_revision"))
                if _is_revision(candidate_revision):
                    expected_finding_revision = candidate_revision
        if not isinstance(finding_id, str) or not finding_id.strip():
            return self._local_result(
                "unavailable",
                "sync_finding requires a retained canonical finding_id",
                context=self._context,
            )
        if not isinstance(repository, str) or not repository.strip():
            return self._local_result(
                "unavailable",
                "sync_finding requires an explicit allowlisted repository",
                context=self._context,
            )
        if not _is_revision(expected_finding_revision):
            return self._local_result(
                "unavailable",
                "sync_finding requires a canonical finding revision",
                context=self._context,
            )
        request_context = deepcopy(self._context)
        call_kwargs = dict(kwargs)
        call_kwargs.update(
            {
                "finding_id": finding_id,
                "repository": repository,
                "context": request_context,
                "expected_finding_revision": expected_finding_revision,
                "expected_source_revision": expected_source_revision,
                "retry_ambiguous": retry_ambiguous,
            }
        )
        result = self._call_service("sync_finding", operation_id=operation_id, **call_kwargs)
        self._remember_result(result)
        envelope = _service_result_mapping(result, secret=self._token)
        envelope.setdefault("evidence_boundary", "diagnostic_only")
        envelope.setdefault("scientific_claim_allowed", False)
        return envelope

    sync_github = sync_finding

    def snapshot(self) -> Mapping[str, Any]:  # noqa: C901, PLR0915
        """Reconstruct a service snapshot from reads; never fall back to fixtures.

        Returns:
            A service-backed snapshot containing sub-operation envelopes.
        """

        if self._token is None:
            denied = self._denied()
            return {
                "schema_version": AUDIT_SERVICE_WORKBENCH_SCHEMA_VERSION,
                "service": dict(self.service_metadata),
                **denied,
            }
        # A snapshot is a fresh read. Reusing operation IDs here replays the
        # first values after a review or Next mutates durable queue state.
        context = self.read_context()
        queue = self.read_queue()
        coverage = self.read_coverage()
        episode = self.read_episode() if self._selected_episode_id else None
        service_records_available = callable(getattr(self._service, "read_saved_records", None))
        records = (
            self.read_saved_records(
                episode_id=self._selected_episode_id,
            )
            if service_records_available
            else None
        )
        artifact_status_result = self.read_selected_artifact_status()
        status_binding = self._artifact_status_binding()
        artifact_status = _artifact_status_projection(
            artifact_status_result.get("value"),
            require_complete_binding=True,
            secret=self._token,
            **status_binding,
        )
        statuses = [context.get("status"), queue.get("status"), coverage.get("status")]
        if episode is not None:
            statuses.append(episode.get("status"))
        if records is not None:
            statuses.append(records.get("status"))
        status = (
            "complete"
            if all(item in {"complete", "committed", "ok"} for item in statuses)
            else next(
                (item for item in statuses if item not in {"complete", "committed", "ok"}),
                "unavailable",
            )
        )
        context_value = context.get("value")
        context_revision = _service_attr(context_value, "context_revision")
        if context_revision is None:
            context_revision = _service_attr(self._context, "context_revision")
        queue_value = queue.get("value")
        queue_revisions = self._codex_queue_revisions(queue_value)
        queue_revisions = {
            key: _safe_service_value(item, secret=self._token)
            for key, item in queue_revisions.items()
            if item is not None
        }
        self._queue_revisions.update(queue_revisions)
        snapshot: dict[str, Any] = {
            "schema_version": AUDIT_SERVICE_WORKBENCH_SCHEMA_VERSION,
            "status": status,
            "reason": "" if status == "complete" else "one or more service reads are unavailable",
            "service": dict(self.service_metadata),
            "context_result": context,
            "queue_result": queue,
            "coverage_result": coverage,
            "context": context.get("value"),
            "queue": queue.get("value"),
            "coverage": coverage.get("value"),
            "packet": deepcopy(self._selected_packet),
            "episode_result": episode,
            "episode": episode.get("value") if episode else None,
            "records_result": records,
            "artifact_status_result": artifact_status_result,
            "artifact_status": artifact_status,
            "selection_epoch": self._selection_epoch,
            "selection_revision": self._selection_epoch,
            "selection_revision_authority": "ui_async_epoch_only",
            "context_revision": _safe_service_value(context_revision, secret=self._token),
            **queue_revisions,
            **self._queue_revisions,
        }
        service_records_failed = False
        if records is not None and records.get("status") in {"complete", "committed", "ok"}:
            projected_records = records.get("records")
            if not isinstance(projected_records, list):
                # ``read_saved_records`` validates this before returning a
                # successful envelope.  Keep this guard at the snapshot edge
                # so future service adapters cannot turn a malformed payload
                # into an empty-success browser model.
                snapshot["records_result"] = self._local_result(
                    "unavailable",
                    "read_saved_records returned malformed durable records",
                    context=self._context,
                )
                snapshot["status"] = "unavailable"
                snapshot["reason"] = "one or more service reads are unavailable"
                projected_records = None
            if projected_records is not None:
                self._annotation_records = {
                    str(record["annotation_id"]): deepcopy(record)
                    for record in projected_records
                    if isinstance(record, Mapping)
                    and record.get("record_type") == "annotation"
                    and record.get("annotation_id")
                }
                self._finding_records = {
                    str(record["finding_id"]): deepcopy(record)
                    for record in projected_records
                    if isinstance(record, Mapping)
                    and record.get("record_type") == "finding"
                    and record.get("finding_id")
                }
                snapshot["annotations"] = [
                    _safe_service_value(record, secret=self._token)
                    for record in self._annotation_records.values()
                ]
                snapshot["findings"] = [
                    _safe_service_value(record, secret=self._token)
                    for record in self._finding_records.values()
                ]
                snapshot["annotation_records"] = {
                    record_id: _safe_service_value(record, secret=self._token)
                    for record_id, record in self._annotation_records.items()
                }
                snapshot["finding_records"] = {
                    record_id: _safe_service_value(record, secret=self._token)
                    for record_id, record in self._finding_records.items()
                }
                snapshot["record_projection"] = {
                    "scope": "service_store",
                    "scope_id": "ba05-audit-store",
                    "completeness": "complete",
                    "authoritative": True,
                    "durability": "service_store",
                    "fresh_process": "reconnectable",
                    "record_authority": "ba03_durable_store",
                    "human_review": "not_inferred",
                    "context_identity": self._record_context_identity(),
                    "annotation_ids": list(self._annotation_records),
                    "finding_ids": list(self._finding_records),
                }
            else:
                service_records_failed = True
        if records is not None and records.get("status") not in {"complete", "committed", "ok"}:
            snapshot["records_status"] = str(records.get("status") or "unknown")
            snapshot["records_reason"] = str(
                records.get("reason") or "durable record read is unavailable"
            )
        elif records is not None:
            snapshot["records_status"] = "complete"
            snapshot["records_reason"] = ""
        elif not service_records_failed and (self._annotation_records or self._finding_records):
            # Compatibility for the pre-read BA-05 branch: a non-empty
            # projection is explicitly scoped to this live facade instance.
            snapshot["annotations"] = [
                _safe_service_value(record, secret=self._token)
                for record in self._annotation_records.values()
            ]
            snapshot["findings"] = [
                _safe_service_value(record, secret=self._token)
                for record in self._finding_records.values()
            ]
            snapshot["annotation_records"] = {
                record_id: _safe_service_value(record, secret=self._token)
                for record_id, record in self._annotation_records.items()
            }
            snapshot["finding_records"] = {
                record_id: _safe_service_value(record, secret=self._token)
                for record_id, record in self._finding_records.items()
            }
            snapshot["annotation_receipts"] = {
                record_id: _safe_service_value(receipt, secret=self._token)
                for record_id, receipt in self._annotation_receipts.items()
            }
            snapshot["finding_receipts"] = {
                record_id: _safe_service_value(receipt, secret=self._token)
                for record_id, receipt in self._finding_receipts.items()
            }
            snapshot["record_projection"] = self._record_projection()
            snapshot["records_status"] = "same_facade_only"
            snapshot["records_reason"] = "durable record reads are unavailable"
        if self._last_annotation_receipt is not None:
            snapshot["annotation_receipt"] = _safe_service_value(
                self._last_annotation_receipt, secret=self._token
            )
        if self._last_finding_receipt is not None:
            snapshot["finding_receipt"] = _safe_service_value(
                self._last_finding_receipt, secret=self._token
            )
        return snapshot

    # Compatibility aliases for the existing browser/controller vocabulary.
    saveAnnotation = save_annotation
    persistFinding = persist_finding
    select_next = next  # noqa: A003
    queue_next = next  # noqa: A003


def _clone(value: Any) -> Any:
    """Clone JSON-compatible fixture data without sharing mutable state.

    Returns:
        A detached JSON-compatible value.
    """

    return json.loads(json.dumps(value, allow_nan=False))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_identity(case_id: str) -> dict[str, Any]:
    digest = _sha256(f"ba06-fixture:{case_id}")
    return {
        "sources": {
            "scene": {
                "artifact_id": f"{case_id}-scene",
                "uri": f"{case_id}/scene.json",
                "format": "threejs-viewer.v1",
                "schema": "threejs-viewer.v1",
                "sha256": digest,
                "source_commit": "b" * 40,
                "config_identity": "ba06-fixture-config",
                "units": "m",
                "coordinate_frame": "map",
            }
        }
    }


def _editor_model(case_id: str, *, missing_media: bool) -> dict[str, Any]:
    """Build the minimum SREV-16/SREV-17-compatible model for one fixture.

    Returns:
        A review-editor model with simulation-time cursor authority.
    """

    source_identity = _source_identity(case_id)
    source = source_identity["sources"]["scene"]
    scene_samples = [
        {"time_s": 0.0, "value": {"robot": {"position": [0.0, 0.0]}}},
        {"time_s": 1.0, "value": {"robot": {"position": [0.8, 0.0]}}},
        {"time_s": 2.5, "value": {"robot": {"position": [1.8, 0.3]}}},
        {"time_s": 4.0, "value": {"robot": {"position": [3.0, 0.7]}}},
    ]
    video_stream: dict[str, Any]
    if missing_media:
        video_stream = {
            "status": "unavailable",
            "reason": "recording_not_present",
            "resolution_s": 1.0,
            "samples": [],
        }
    else:
        # ``time_s`` is simulation time; ``pts_s`` is declared media time.
        # Deliberately uneven PTS values catch accidental fps arithmetic.
        video_stream = {
            "status": "available",
            "resolution_s": 1.5,
            "media_uri": f"{case_id}/recording.mp4",
            "samples": [
                {"time_s": 0.0, "value": {"pts_s": 0.00}},
                {"time_s": 1.0, "value": {"pts_s": 0.041}},
                {"time_s": 2.5, "value": {"pts_s": 0.113}},
                {"time_s": 4.0, "value": {"pts_s": 0.176}},
            ],
        }
    metric_samples = [
        {"time_s": 0.0, "value": 1.2},
        {"time_s": 1.0, "value": 0.9},
        {"time_s": 2.5, "value": 0.7},
        {"time_s": 4.0, "value": 1.1},
    ]
    panel_model = {
        "schema_version": "review-panels.v1",
        "context": {
            "campaign_id": "ba06-fixture-campaign",
            "execution_id": f"{case_id}-execution",
            "episode_id": case_id,
            "actor_id": "robot",
            "selection_revision": 0,
        },
        "source_identity": source_identity,
        "time": {
            "origin_s": 0.0,
            "terminal_s": 4.0,
            "cursor": {"time_s": 0.0},
        },
        "streams": {
            "scene": {
                "status": "available",
                "resolution_s": 1.5,
                "samples": scene_samples,
            },
            "video": video_stream,
            "metric:clearance": {
                "status": "available",
                "resolution_s": 1.5,
                "samples": metric_samples,
            },
        },
        "scene_surface": {
            "schema_version": "threejs-viewer.v1",
            "map": {"width": 8.0, "height": 6.0},
        },
        "goal_geometry": {
            "point": {"status": "available", "value": [3.0, 0.7]},
            "completion_boundary": {"status": "available", "value": {"radius_m": 0.35}},
        },
        "metrics": {
            "clearance": {
                "metric_id": "clearance",
                "label": "Obstacle clearance",
                "unit": "m",
                "visible": True,
                "stream": {"samples": metric_samples},
                "current": {"sample_time_s": 0.0},
            }
        },
        "events": [],
        "intervals": [],
        "controls": {"default_speed": 1.0, "speeds": [0.5, 1.0, 2.0]},
    }
    return {
        "schema_version": "review-editor.v1",
        "context": dict(panel_model["context"]),
        "source_identity": source_identity,
        "source_identity_status": {
            source["artifact_id"]: {"status": "verified", "sha256": source["sha256"]}
        },
        "panel_model": panel_model,
        "streams": panel_model["streams"],
        "scene_surface": panel_model["scene_surface"],
        "goal_geometry": panel_model["goal_geometry"],
        "metrics": panel_model["metrics"],
        "events": [],
        "time": panel_model["time"],
        "annotations": [],
        "storyboard": {
            "schema_version": "review-storyboard-edit.v1",
            "source_identity": source_identity,
            "source_revision": "fixture-revision-1",
            "intervals": [],
            "order": [],
            "captions": {},
        },
        "overlay_state": {
            "numbered": True,
            "highlights": True,
            "arrows": False,
            "rings": False,
            "distances": False,
        },
        "controls": {
            "typing_suppresses_shortcuts": True,
            "source_time_authority": "simulation_time",
        },
    }


def fixture_document() -> dict[str, Any]:
    """Return normal and missing-media cases used by the offline UI slice."""

    return {
        "schema_version": AUDIT_WORKBENCH_SCHEMA_VERSION,
        "service": {
            "id": FIXTURE_SERVICE_ID,
            "authority": "injected_fixture_facade",
            "native": False,
            "evidence_status": "diagnostic_only",
            "ba05_integration": "pending_service_contract_freeze",
        },
        "queue": [
            {
                "episode_id": "fixture-normal-control",
                "execution_id": "fixture-normal-control-execution",
                "planner_id": "fixture-planner",
                "scenario_id": "corridor-control",
                "seed": 7,
                "selection_reasons": [
                    {"code": "coverage_deficit", "label": "ordinary control is under-reviewed"},
                    {"code": "control_case", "label": "normal control"},
                ],
                "cursor": {"time_s": 0.0, "authority": "simulation_time"},
                "metric_units": {"time": "s", "distance": "m", "clearance": "m"},
                "metrics": [
                    {"id": "clearance", "label": "Obstacle clearance", "value": 1.2, "unit": "m"},
                    {"id": "outcome", "label": "Outcome", "value": "success", "unit": ""},
                ],
                "media": {
                    "status": "available",
                    "resolution_s": 1.5,
                    "samples": _editor_model("fixture-normal-control", missing_media=False)[
                        "streams"
                    ]["video"]["samples"],
                },
                "editor_model": _editor_model("fixture-normal-control", missing_media=False),
            },
            {
                "episode_id": "fixture-missing-media",
                "execution_id": "fixture-missing-media-execution",
                "planner_id": "fixture-planner",
                "scenario_id": "crossing-anomaly",
                "seed": 11,
                "selection_reasons": [
                    {"code": "detector_signal", "label": "planner disagreement signal"},
                    {
                        "code": "media_unavailable",
                        "label": "recording is not present; scene remains reviewable",
                    },
                ],
                "cursor": {"time_s": 1.0, "authority": "simulation_time"},
                "metric_units": {"time": "s", "distance": "m", "clearance": "m"},
                "metrics": [
                    {"id": "clearance", "label": "Obstacle clearance", "value": 0.7, "unit": "m"},
                    {"id": "outcome", "label": "Outcome", "value": "failure", "unit": ""},
                ],
                "media": {
                    "status": "unavailable",
                    "reason": "recording_not_present",
                    "resolution_s": 1.5,
                    "samples": [],
                },
                "editor_model": _editor_model("fixture-missing-media", missing_media=True),
            },
        ],
        "coverage": {
            "schema_version": "audit-coverage.v1",
            "reviewed": 0,
            "total": 2,
            "remaining": 2,
            "reviewed_episode_ids": [],
            "status": "under_review",
        },
    }


def _coverage(total: int, reviewed_ids: Sequence[str]) -> dict[str, Any]:
    unique_ids = list(dict.fromkeys(str(item) for item in reviewed_ids))
    return {
        "schema_version": "audit-coverage.v1",
        "reviewed": len(unique_ids),
        "total": total,
        "remaining": max(0, total - len(unique_ids)),
        "reviewed_episode_ids": unique_ids,
        "status": "complete" if len(unique_ids) >= total else "under_review",
    }


class FixtureAuditService:
    """In-memory service-shaped facade for the first UI-only vertical slice.

    The state deliberately follows the BA-03/BA-05 vocabulary (selection
    revision, expected record revision, operation ID and durable receipt), but
    the implementation is not a store.  It is deterministic, disposable test
    state that lets browser tests prove wiring before BA-05 is integrated.
    """

    def __init__(self, document: Mapping[str, Any] | None = None):
        """Initialize disposable state from the supplied fixture document."""

        payload = fixture_document() if document is None else _clone(document)
        if payload.get("schema_version") != AUDIT_WORKBENCH_SCHEMA_VERSION:
            raise AuditWorkbenchError("fixture schema_version must be audit-workbench.v1")
        queue = payload.get("queue")
        if not isinstance(queue, list) or not queue:
            raise AuditWorkbenchError("fixture queue must contain at least one case")
        self._queue = queue
        self._coverage = _clone(payload.get("coverage", {}))
        self._index = int(payload.get("queue_index", -1))
        self._selection_revision = int(payload.get("selection_revision", 0))
        initial_annotations = payload.get("annotations", [])
        if not isinstance(initial_annotations, list):
            initial_annotations = []
        self._annotations = {
            str(item.get("annotation_id") or item.get("record_id")): _clone(item)
            for item in initial_annotations
            if isinstance(item, Mapping) and (item.get("annotation_id") or item.get("record_id"))
        }
        self._annotation_revisions = {
            annotation_id: int(record.get("revision", record.get("record_revision", 0)))
            for annotation_id, record in self._annotations.items()
        }
        initial_annotation_operations = payload.get("annotation_operation_receipts", {})
        self._annotation_operation_receipts = (
            {
                str(operation_id): _clone(receipt)
                for operation_id, receipt in initial_annotation_operations.items()
                if isinstance(receipt, Mapping)
            }
            if isinstance(initial_annotation_operations, Mapping)
            else {}
        )
        initial_findings = payload.get("findings", [])
        if not isinstance(initial_findings, list):
            initial_findings = []
        self._findings = {
            str(item.get("finding_id") or item.get("record_id")): _clone(item)
            for item in initial_findings
            if isinstance(item, Mapping) and (item.get("finding_id") or item.get("record_id"))
        }
        initial_finding_operations = payload.get("finding_operation_ids", {})
        self._finding_operation_ids = (
            {
                str(finding_id): str(operation_id)
                for finding_id, operation_id in initial_finding_operations.items()
            }
            if isinstance(initial_finding_operations, Mapping)
            else {}
        )
        for finding_id, finding in self._findings.items():
            operation_id = finding.get("operation_id")
            if operation_id and finding_id not in self._finding_operation_ids:
                self._finding_operation_ids[finding_id] = str(operation_id)
        packet = payload.get("packet")
        self._last_packet = _clone(packet) if isinstance(packet, Mapping) else None
        self._artifact_status = _clone(payload.get("artifact_status"))
        self.service_metadata = _clone(payload.get("service", {}))

    @property
    def selection_revision(self) -> int:
        """Return the current selection compare-and-swap revision."""

        return self._selection_revision

    def _check_selection(self, expected: int) -> None:
        if expected != self._selection_revision:
            raise AuditWorkbenchConflictError(
                f"stale selection revision: expected {expected}, current {self._selection_revision}",
                expected=expected,
                actual=self._selection_revision,
            )

    def _selected_case(self) -> dict[str, Any]:
        if self._last_packet is None:
            raise AuditWorkbenchUnavailableError("no audit packet is selected")
        return self._last_packet

    def _packet(self, case: Mapping[str, Any]) -> dict[str, Any]:
        packet = _clone(case)
        packet["selection_revision"] = self._selection_revision
        packet["cursor"] = dict(packet.get("cursor", {}))
        packet["cursor"]["selection_revision"] = self._selection_revision
        editor_model = packet.get("editor_model")
        if isinstance(editor_model, dict):
            editor_model.setdefault("context", {})["selection_revision"] = self._selection_revision
            editor_model["context"]["context_revision"] = self._selection_revision
            editor_model["context"]["queue_selection_revision"] = self._selection_revision
            editor_model["context"]["cursor"] = dict(packet["cursor"])
            panel_model = editor_model.get("panel_model")
            if isinstance(panel_model, dict):
                panel_context = panel_model.setdefault("context", {})
                panel_context["selection_revision"] = self._selection_revision
                panel_context["context_revision"] = self._selection_revision
                panel_context["queue_selection_revision"] = self._selection_revision
                panel_context["cursor"] = dict(packet["cursor"])
        return packet

    def _sync_selected_context(self, annotation: Mapping[str, Any]) -> None:
        """Reflect editor cursor/context in the selected packet snapshot."""

        if self._last_packet is None:
            return
        interval = annotation.get("interval")
        cursor_time = interval.get("start_s") if isinstance(interval, Mapping) else None
        if isinstance(cursor_time, (int, float)) and not isinstance(cursor_time, bool):
            self._last_packet.setdefault("cursor", {})["time_s"] = float(cursor_time)
            editor_model = self._last_packet.get("editor_model")
            if isinstance(editor_model, dict):
                context = editor_model.setdefault("context", {})
                context["cursor"] = {"time_s": float(cursor_time)}
                metadata = annotation.get("metadata")
                editor_revision = (
                    metadata.get("selection_revision") if isinstance(metadata, Mapping) else None
                )
                if isinstance(editor_revision, int) and not isinstance(editor_revision, bool):
                    context["selection_revision"] = editor_revision
                    context["context_revision"] = editor_revision
                    context["queue_selection_revision"] = self._selection_revision
                    panel_model = editor_model.get("panel_model")
                    if isinstance(panel_model, dict):
                        panel_context = panel_model.setdefault("context", {})
                        panel_context["selection_revision"] = editor_revision
                        panel_context["context_revision"] = editor_revision
                        panel_context["queue_selection_revision"] = self._selection_revision
                        panel_context["cursor"] = {"time_s": float(cursor_time)}

    def next(self, *, expected_selection_revision: int) -> Mapping[str, Any]:
        """Select the next fixture case and expose reasons/units/media status.

        Returns:
            A selected packet, or an explicit empty-queue response.
        """

        self._check_selection(expected_selection_revision)
        next_index = self._index + 1
        if next_index >= len(self._queue):
            return {
                "status": "empty",
                "reason": "queue_exhausted",
                "selection_revision": self._selection_revision,
                "coverage": _clone(self._coverage),
                "next": None,
            }
        self._index = next_index
        self._selection_revision += 1
        self._last_packet = self._packet(self._queue[self._index])
        return {
            "status": "selected",
            "packet": _clone(self._last_packet),
            "selection_revision": self._selection_revision,
            "coverage": _clone(self._coverage),
            "next": self.peek_next(),
        }

    def save_annotation(
        self,
        annotation: Mapping[str, Any],
        *,
        expected_selection_revision: int,
        expected_revision: int,
        operation_id: str,
    ) -> Mapping[str, Any]:
        """Apply a fixture annotation with selection and record CAS checks.

        Returns:
            A save receipt containing the incremented record revision.
        """

        self._check_selection(expected_selection_revision)
        if operation_id in self._annotation_operation_receipts:
            return _clone(self._annotation_operation_receipts[operation_id])
        packet = self._selected_case()
        episode_id = str(packet["episode_id"])
        annotation_id = str(annotation.get("annotation_id") or annotation.get("record_id") or "")
        if not annotation_id:
            raise AuditWorkbenchError("annotation_id is required")
        actual_revision = self._annotation_revisions.get(annotation_id, 0)
        if expected_revision != actual_revision:
            raise AuditWorkbenchConflictError(
                f"revision conflict for {annotation_id!r}: expected {expected_revision}, current {actual_revision}",
                expected=expected_revision,
                actual=actual_revision,
            )
        stored = _clone(annotation)
        stored["episode_id"] = episode_id
        stored["selection_revision"] = self._selection_revision
        stored["service_boundary"] = "injected_fixture_facade"
        revision = actual_revision + 1
        stored["revision"] = revision
        self._annotations[annotation_id] = stored
        self._annotation_revisions[annotation_id] = revision
        self._sync_selected_context(annotation)
        receipt = {
            "status": "saved",
            "record_id": annotation_id,
            "revision": revision,
            "selection_revision": self._selection_revision,
            "operation_id": operation_id,
            "record": _clone(stored),
        }
        self._annotation_operation_receipts[operation_id] = _clone(receipt)
        return receipt

    def persist_finding(
        self,
        annotation: Mapping[str, Any],
        *,
        expected_selection_revision: int,
        operation_id: str,
    ) -> Mapping[str, Any]:
        """Create one proposed finding and update fixture coverage.

        Returns:
            A finding receipt with updated coverage and the next queue item.
        """

        self._check_selection(expected_selection_revision)
        packet = self._selected_case()
        episode_id = str(packet["episode_id"])
        annotation_id = str(annotation.get("annotation_id") or annotation.get("record_id") or "")
        stored_annotation = self._annotations.get(annotation_id)
        if stored_annotation is None:
            raise AuditWorkbenchUnavailableError("finding requires a durably saved annotation")
        if (
            str(stored_annotation.get("episode_id")) != episode_id
            or int(stored_annotation.get("selection_revision", -1)) != self._selection_revision
        ):
            raise AuditWorkbenchConflictError(
                "saved annotation does not belong to the selected episode/context",
                expected=self._selection_revision,
                actual=int(stored_annotation.get("selection_revision", -1)),
            )
        finding_id = f"finding-{episode_id}"
        previous_operation = self._finding_operation_ids.get(finding_id)
        if previous_operation == operation_id:
            finding = self._findings[finding_id]
        else:
            finding = {
                "schema_version": "audit-record.v1",
                "record_type": "finding",
                "record_id": finding_id,
                "finding_id": finding_id,
                "status": "proposed",
                "title": str(stored_annotation.get("classification", "review finding")),
                "candidate_members": [episode_id],
                "confirmed_members": [],
                "negative_controls": [],
                "annotation_ids": [annotation_id],
                "operation_id": operation_id,
                "service_boundary": "injected_fixture_facade",
            }
            self._findings[finding_id] = finding
            self._finding_operation_ids[finding_id] = operation_id
            reviewed_ids = list(self._coverage.get("reviewed_episode_ids", []))
            reviewed_ids.append(episode_id)
            self._coverage = _coverage(
                int(self._coverage.get("total", len(self._queue))), reviewed_ids
            )
        return {
            "status": "saved",
            "finding": _clone(finding),
            "coverage": _clone(self._coverage),
            "selection_revision": self._selection_revision,
            "operation_id": operation_id,
            "next": self.peek_next(),
        }

    def peek_next(self) -> Mapping[str, Any] | None:
        """Return the next case identity without changing selection state."""

        next_index = self._index + 1
        if next_index >= len(self._queue):
            return None
        case = self._queue[next_index]
        return {
            "episode_id": case.get("episode_id"),
            "scenario_id": case.get("scenario_id"),
            "selection_reasons": _clone(case.get("selection_reasons", [])),
        }

    def _artifact_status_binding(self) -> dict[str, Any]:
        """Derive a complete status binding from the selected fixture packet.

        Returns:
            The selected episode, context revision, source revision, and digest.
        """

        packet = self._last_packet if isinstance(self._last_packet, Mapping) else {}
        editor_model = packet.get("editor_model")
        editor_context = _service_attr(editor_model, "context", {})
        storyboard = _service_attr(editor_model, "storyboard", {})
        panel_model = _service_attr(editor_model, "panel_model", {})
        panel_context = _service_attr(panel_model, "context", {})
        episode_id, episode_conflict = _artifact_binding_consensus(
            "episode_id",
            packet.get("episode_id"),
            _service_attr(editor_context, "episode_id"),
            _service_attr(editor_model, "episode_id"),
            _service_attr(panel_context, "episode_id"),
            _service_attr(panel_model, "episode_id"),
        )
        context_revision, context_conflict = _artifact_binding_consensus(
            "context_revision",
            packet.get("context_revision"),
            packet.get("selection_revision"),
            _service_attr(editor_context, "context_revision"),
            _service_attr(editor_context, "selection_revision"),
            _service_attr(panel_context, "context_revision"),
            _service_attr(panel_context, "selection_revision"),
        )
        source_revision, source_revision_conflict = _artifact_binding_consensus(
            "source_revision",
            packet.get("source_revision"),
            _service_attr(editor_context, "source_revision"),
            _service_attr(editor_model, "source_revision"),
            _service_attr(storyboard, "source_revision"),
            _service_attr(panel_context, "source_revision"),
            _service_attr(panel_model, "source_revision"),
        )
        source_digest, source_digest_conflict = _artifact_binding_consensus(
            "source_digest",
            packet.get("source_digest"),
            packet.get("source_identity"),
            _service_attr(editor_context, "source_digest"),
            _service_attr(editor_context, "source_identity"),
            _service_attr(editor_model, "source_digest"),
            _service_attr(editor_model, "source_identity"),
            _service_attr(storyboard, "source_digest"),
            _service_attr(storyboard, "source_identity"),
            _service_attr(panel_context, "source_digest"),
            _service_attr(panel_context, "source_identity"),
            _service_attr(panel_model, "source_digest"),
            _service_attr(panel_model, "source_identity"),
        )
        return {
            "episode_id": episode_id,
            "context_revision": context_revision,
            "source_revision": source_revision,
            "source_digest": source_digest,
            "binding_conflict": any(
                (
                    episode_conflict,
                    context_conflict,
                    source_revision_conflict,
                    source_digest_conflict,
                )
            ),
        }

    def snapshot(self) -> Mapping[str, Any]:
        """Return state sufficient to render or reopen the fixture session."""

        binding = self._artifact_status_binding()
        artifact_status = _artifact_status_projection(
            self._artifact_status,
            require_complete_binding=True,
            **binding,
        )
        return {
            "schema_version": AUDIT_WORKBENCH_SCHEMA_VERSION,
            "service": _clone(self.service_metadata),
            "queue": _clone(self._queue),
            "queue_index": self._index,
            "selection_revision": self._selection_revision,
            "packet": _clone(self._last_packet),
            "coverage": _clone(self._coverage),
            "annotations": _clone(list(self._annotations.values())),
            "annotation_operation_receipts": _clone(self._annotation_operation_receipts),
            "findings": _clone(list(self._findings.values())),
            "finding_operation_ids": _clone(self._finding_operation_ids),
            "artifact_status": artifact_status,
            "next": _clone(self.peek_next()),
        }


def build_audit_workbench_document(
    facade: AuditWorkbenchFacade | ServiceAuditWorkbenchFacade | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the JSON model consumed by the browser shell.

    ``facade`` may be a real injected implementation or a fixture document;
    passing a document constructs the disposable fixture service explicitly so
    tests cannot accidentally imply a live/native BA-05 connection.

    Returns:
        A JSON-compatible workbench snapshot for the browser shell.
    """

    service: AuditWorkbenchFacade | ServiceAuditWorkbenchFacade
    if facade is None or isinstance(facade, Mapping):
        service = FixtureAuditService(facade if isinstance(facade, Mapping) else None)
    else:
        service = facade
    snapshot = _clone(service.snapshot())
    if isinstance(service, ServiceAuditWorkbenchFacade):
        # A service snapshot is intentionally not reshaped into the fixture
        # queue/editor model.  The server route that owns this facade can
        # consume ``service_snapshot`` without granting the browser authority.
        return {
            "schema_version": AUDIT_WORKBENCH_SCHEMA_VERSION,
            "component_id": AUDIT_WORKBENCH_COMPONENT_ID,
            "component_version": "0.2.0-service-facade",
            "service": snapshot.get("service", dict(service.service_metadata)),
            "service_snapshot": snapshot,
            "status": snapshot.get("status", "unavailable"),
            "context": snapshot.get("context"),
            "packet": snapshot.get("packet"),
            "episode": snapshot.get("episode"),
            "coverage": snapshot.get("coverage"),
            "artifact_status": snapshot.get("artifact_status"),
            "queue": snapshot.get("queue"),
            "persistence": {
                "writes": "audit_service",
                "browser_local_storage": False,
                "native_or_live_claim": False,
                "service_token_transport": "server_only",
                "service_context_cas": True,
                "queue_revision_cas": True,
            },
        }
    return {
        "schema_version": AUDIT_WORKBENCH_SCHEMA_VERSION,
        "component_id": AUDIT_WORKBENCH_COMPONENT_ID,
        "component_version": AUDIT_WORKBENCH_COMPONENT_VERSION,
        "service": snapshot.get("service", {"id": "injected-facade"}),
        "queue": snapshot.get("queue", []),
        "queue_index": snapshot.get("queue_index", -1),
        "selection_revision": snapshot.get("selection_revision", 0),
        "packet": snapshot.get("packet"),
        "coverage": snapshot.get("coverage", {}),
        "artifact_status": snapshot.get("artifact_status"),
        "annotations": snapshot.get("annotations", []),
        "findings": snapshot.get("findings", []),
        "next": snapshot.get("next"),
        "persistence": {
            "writes": "injected_facade_only",
            "browser_local_storage": False,
            "native_or_live_claim": False,
            "selection_revision_cas": True,
        },
    }


def render_audit_workbench_html(document: Mapping[str, Any]) -> str:
    """Render an offline shell that instantiates the injected fixture facade.

    Returns:
        An HTML document with only repository-local module references.
    """

    service = document.get("service") if isinstance(document, Mapping) else None
    if isinstance(service, Mapping) and service.get("id") == AUDIT_SERVICE_FACADE_ID:
        raise AuditWorkbenchError(
            "service-backed workbench requires its server route; fixture HTML cannot mount it"
        )
    payload = json.dumps(document, sort_keys=True, allow_nan=False).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Robot SF benchmark audit workbench (fixture)</title>
    <link rel="stylesheet" href="./components/audit_workbench/audit_workbench.css" />
  </head>
  <body>
    <main id="audit-workbench-root" aria-labelledby="audit-workbench-title"></main>
    <script type="application/json" id="audit-workbench-data">{payload}</script>
    <script type="module">
      import {{ createFixtureFacade, mountAuditWorkbench }} from "./components/audit_workbench/audit_workbench.js";
      const data = JSON.parse(document.getElementById("audit-workbench-data").textContent || "{{}}");
      mountAuditWorkbench(data, document.getElementById("audit-workbench-root"), {{
        facade: createFixtureFacade(data),
      }});
    </script>
  </body>
</html>
"""


def _copy_asset(output_dir: Path, relative: str) -> Path:
    destination = output_dir / "components" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    asset = resources.files("robot_sf.render.web_assets").joinpath("components", relative)
    destination.write_bytes(asset.read_bytes())
    return destination


def write_fixture_workbench(
    output_dir: str | Path,
    *,
    facade: AuditWorkbenchFacade | ServiceAuditWorkbenchFacade | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the model, HTML, and local component assets for an offline demo.

    Returns:
        A completion receipt with the generated model and artifact digests.
    """

    destination = Path(output_dir)
    if destination.exists():
        raise AuditWorkbenchError(f"output collision: {destination}")
    destination.mkdir(parents=True)
    document = build_audit_workbench_document(facade)
    model_path = destination / OUTPUT_MODEL_FILENAME
    html_path = destination / OUTPUT_HTML_FILENAME
    model_path.write_text(json.dumps(document, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    html_path.write_text(render_audit_workbench_html(document), encoding="utf-8")
    assets = [
        _copy_asset(destination, "audit_workbench/audit_workbench.js"),
        _copy_asset(destination, "audit_workbench/audit_workbench.css"),
        _copy_asset(destination, "review_editor/review_editor.js"),
        _copy_asset(destination, "review_panels/review_panels.js"),
    ]
    return {
        "status": "complete",
        "document": document,
        "artifacts": [
            {
                "artifact_id": OUTPUT_MODEL_FILENAME,
                "sha256": _sha256(model_path.read_text("utf-8")),
            },
            {"artifact_id": OUTPUT_HTML_FILENAME, "sha256": _sha256(html_path.read_text("utf-8"))},
            *[
                {
                    "artifact_id": str(path.relative_to(destination)),
                    "sha256": _sha256(path.read_text("utf-8")),
                }
                for path in assets
            ],
        ],
    }


# Concise aliases mirror the existing SREV component vocabulary.  The aliases
# keep the injectable boundary discoverable while BA05/BA06 integration settles
# on one public class spelling.
AuditServiceWorkbenchFacade = ServiceAuditWorkbenchFacade
ServiceBackedAuditWorkbenchFacade = ServiceAuditWorkbenchFacade
LiveAuditWorkbenchFacade = ServiceAuditWorkbenchFacade
build_model = build_audit_workbench_document
render_html = render_audit_workbench_html


__all__ = [
    "AUDIT_SERVICE_FACADE_ID",
    "AUDIT_SERVICE_WORKBENCH_SCHEMA_VERSION",
    "AUDIT_WORKBENCH_COMPONENT_ID",
    "AUDIT_WORKBENCH_COMPONENT_VERSION",
    "AUDIT_WORKBENCH_SCHEMA_VERSION",
    "FIXTURE_SERVICE_ID",
    "OUTPUT_HTML_FILENAME",
    "OUTPUT_MODEL_FILENAME",
    "AuditServiceWorkbenchFacade",
    "AuditWorkbenchConflictError",
    "AuditWorkbenchError",
    "AuditWorkbenchFacade",
    "AuditWorkbenchUnavailableError",
    "FixtureAuditService",
    "LiveAuditWorkbenchFacade",
    "ServiceAuditWorkbenchFacade",
    "ServiceBackedAuditWorkbenchFacade",
    "build_audit_workbench_document",
    "build_model",
    "fixture_document",
    "render_audit_workbench_html",
    "render_html",
    "write_fixture_workbench",
]
