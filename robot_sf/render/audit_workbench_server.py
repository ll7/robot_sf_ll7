"""Loopback-only transport for a server-held Benchmark Auditor facade.

The browser receives operation results, never the audit session credential.  This
module does not create authority: the caller must supply an admitted, policy-bound
``ServiceAuditWorkbenchFacade`` and a SREV-15 workbench document.
"""

from __future__ import annotations

import json
import math
import re
import secrets
import threading
import time
from collections.abc import Mapping
from copy import deepcopy
from http import HTTPStatus
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from robot_sf.render.audit_workbench import ServiceAuditWorkbenchFacade
from robot_sf.render.review_sessions import is_loopback_origin
from robot_sf.render.review_workbench import AUDIT_WORKBENCH_ASSETS, _render_html

MAX_REQUEST_BYTES = 64 * 1024
MAX_JSON_DEPTH = 64
SESSION_LIFETIME_SECONDS = 3600
SESSION_COOKIE = "audit_workbench_session"
_FORBIDDEN_KEYS = frozenset(
    {
        "token",
        "session_token",
        "audit_token",
        "authorization",
        "access_token",
        "api_key",
        "credential",
        "credentials",
        "secret",
        "session_id",
        "audit_session_id",
        "provider_session_id",
        "route",
        "route_id",
        "source_path",
        "provider_path",
        "trace_path",
        "recording_path",
        "campaign_root",
        "source_uri",
        "file_path",
        "path",
        "cwd",
        "working_directory",
        "executable",
        "command",
        "provider",
        "model",
        "environment",
        "env",
    }
)
_OPERATIONS = frozenset(
    {
        "snapshot",
        "next",
        "related_cases",
        "save_annotation",
        "persist_finding",
        "sync_finding",
        "record_human_review",
        "materialize_selected",
        "run_native_diagnostic",
        "codex_start",
        "codex_read",
        "codex_cancel",
    }
)
_CODEX_START_ARGUMENTS = frozenset({"prompt", "operation_id", "token_budget", "compute_budget"})
_CODEX_READ_ARGUMENTS = frozenset({"operation_id"})
_CODEX_CANCEL_ARGUMENTS = frozenset({"reason", "operation_id"})
_HUMAN_REVIEW_OUTCOMES = frozenset({"pass", "fail", "uncertain"})
_MATERIALIZATION_STATUSES = frozenset({"complete", "partial", "unavailable", "failed"})
_MATERIALIZATION_SERVICE_STATUSES = _MATERIALIZATION_STATUSES | frozenset(
    {"committed", "cancelled", "conflict", "denied"}
)
_MATERIALIZATION_KINDS = frozenset({"historical_original", "derived_render", "unavailable"})
_MATERIALIZATION_FIDELITIES = frozenset({"verified", "diverged", "unverifiable", "unavailable"})
_NATIVE_DIAGNOSTIC_SERVICE_STATUSES = frozenset(
    {"complete", "failed", "unavailable", "cancelled", "conflict", "denied"}
)
_MAX_CODEX_PROMPT_LENGTH = 8192
_MAX_CODEX_OPERATION_ID_LENGTH = 128
_MAX_CODEX_REASON_LENGTH = 512
_MAX_CODEX_TOKEN_BUDGET = 1_000_000
_MAX_CODEX_COMPUTE_BUDGET = 1_000_000.0
_OPAQUE_OPERATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_MAX_CODEX_PROJECTION_DEPTH = 64
_MAX_CODEX_PROJECTION_NODES = 512
_RELATED_CASE_MODES = frozenset(
    {
        "same_scenario_across_planners",
        "same_scenario",
        "same_planner_across_seeds",
        "same_planner",
        "symptom",
        "anomaly_signature",
        "anomaly",
        "geometry",
        "outcome",
        "metric_behaviour",
        "metric",
        "existing_finding",
        "finding",
    }
)
_RELATED_CASE_ARGUMENTS = frozenset(
    {"episode_id", "mode", "limit", "operation_id", "expected_context_revision"}
)
_SYNC_FINDING_ARGUMENTS = frozenset(
    {
        "finding_id",
        "repository",
        "expected_finding_revision",
        "expected_selection_revision",
        "expected_context_revision",
        "expected_source_revision",
        "retry_ambiguous",
        "operation_id",
    }
)
_GITHUB_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_MAX_SYNC_FINDING_ID_LENGTH = 256
_MAX_SYNC_REPOSITORY_LENGTH = 256
_MAX_RELATED_CASE_ID_LENGTH = 256
_MAX_RELATED_CASE_MODE_LENGTH = 64
_MAX_RELATED_CASE_LIMIT = 100
_RELATED_CASE_COMPATIBILITY = frozenset({"compatible", "unknown", "incompatible"})
_RELATED_CASE_FORBIDDEN_KEYS = _FORBIDDEN_KEYS | frozenset(
    {
        "candidate_members",
        "confirmed_members",
        "negative_controls",
        "finding",
        "finding_id",
        "membership",
        "provenance",
        "client_secret",
        "session_token",
        "authorization",
        "auth",
    }
)
_ARGUMENTS = {
    "snapshot": frozenset(),
    "next": frozenset(
        {
            "expected_selection_revision",
            "expected_context_revision",
            "expected_queue_state_revision",
            "expected_queue_input_revision",
            "force_current",
            "operation_id",
        }
    ),
    "related_cases": _RELATED_CASE_ARGUMENTS,
    "save_annotation": frozenset(
        {
            "annotation",
            "expected_selection_revision",
            "expected_revision",
            "expected_source_revision",
            "expected_context_revision",
            "operation_id",
        }
    ),
    "persist_finding": frozenset(
        {
            "annotation",
            "expected_selection_revision",
            "expected_context",
            "expected_context_revision",
            "expected_source_revision",
            "expected_queue_state_revision",
            "expected_queue_input_revision",
            "expected_queue_input_identity",
            "operation_id",
            "finding_id",
            "title",
        }
    ),
    "sync_finding": _SYNC_FINDING_ARGUMENTS,
    "record_human_review": frozenset(
        {
            "outcome",
            "expected_selection_revision",
            "expected_context_revision",
            "expected_queue_state_revision",
            "expected_queue_input_revision",
            "operation_id",
        }
    ),
    "materialize_selected": frozenset(
        {"operation_id", "expected_selection_revision", "expected_context_revision"}
    ),
    "run_native_diagnostic": frozenset(
        {
            "operation_id",
            "expected_selection_revision",
            "expected_context_revision",
            "intervention_id",
            "robot_goal",
            "activation_epsilon_m",
            "deadline_s",
        }
    ),
    "codex_start": _CODEX_START_ARGUMENTS,
    "codex_read": _CODEX_READ_ARGUMENTS,
    "codex_cancel": _CODEX_CANCEL_ARGUMENTS,
}

_CODEX_RESULT_FIELDS = frozenset(
    {
        "status",
        "reason",
        "operation_id",
        "context",
        "current_context",
        "source",
        "current_source",
        "route_id",
        "evidence_ids",
        "evidence_references",
        "usage",
        "activity",
        "events",
        "activity_scope",
        "context_revision",
        "selection_revision",
        "source_revision",
        "source_digest",
        "session",
        "receipt",
        "snapshot",
        "current",
        "result",
    }
)
_CODEX_CONTEXT_FIELDS = frozenset(
    {"context_revision", "selection_revision", "episode_id", "scenario_id", "execution_id"}
)
_CODEX_SOURCE_FIELDS = frozenset({"source_revision", "source_digest", "digest"})
_CODEX_USAGE_FIELDS = frozenset(
    {
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "reserved_compute",
        "measured_compute",
        "token_budget",
        "compute_budget",
    }
)
_CODEX_ACTIVITY_FIELDS = frozenset({"message", "text", "evidence_ids", "operation_id", "timestamp"})
_CODEX_REFERENCE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,255}$")


class _CodexCapabilityUnavailable(Exception):
    """Raised when the server-held facade has not admitted Codex support."""


class _CodexProjectionError(ValueError):
    """Raised when a Codex result exceeds the bounded browser projection."""


def _dispatch_facade_operation(  # noqa: C901
    facade: ServiceAuditWorkbenchFacade, operation: str, arguments: dict[str, Any]
) -> Any:
    """Dispatch only the closed server-facade operation set.

    Codex methods are optional while BA05 integration is being staged.  The
    explicit branches are intentional: a browser request can never turn an
    arbitrary attribute name into a facade call.

    Returns:
        The admitted facade operation result.
    """

    if operation == "snapshot":
        return facade.snapshot(**arguments)
    if operation == "next":
        return facade.next(**arguments)
    if operation == "related_cases":
        # The browser may carry a stale context revision as a read guard, but
        # context remains server-held and is never forwarded as authority.
        related_arguments = {
            key: value for key, value in arguments.items() if key != "expected_context_revision"
        }
        return facade.related_cases(**related_arguments)
    if operation == "save_annotation":
        return facade.save_annotation(**arguments)
    if operation == "persist_finding":
        return facade.persist_finding(**arguments)
    if operation == "sync_finding":
        return facade.sync_finding(**arguments)
    if operation == "record_human_review":
        return facade.record_human_review(**arguments)
    if operation == "codex_start":
        method = getattr(facade, "codex_start", None)
        if not callable(method):
            raise _CodexCapabilityUnavailable
        return method(**arguments)
    if operation == "codex_read":
        method = getattr(facade, "codex_read", None)
        if not callable(method):
            raise _CodexCapabilityUnavailable
        return method(**arguments)
    if operation == "codex_cancel":
        method = getattr(facade, "codex_cancel", None)
        if not callable(method):
            raise _CodexCapabilityUnavailable
        return method(**arguments)
    raise AssertionError(f"unhandled audit workbench operation: {operation}")


def _bounded_text(value: Any, *, field: str, maximum: int, required: bool = True) -> str | None:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be text")
    if required and not value.strip():
        raise ValueError(f"{field} must not be empty")
    if len(value) > maximum or "\x00" in value:
        raise ValueError(f"{field} is too long")
    return value


def _validate_human_review_arguments(arguments: dict[str, Any]) -> None:
    """Admit only a bounded, fully CAS-bound explicit human review request."""

    if set(arguments) != _ARGUMENTS["record_human_review"]:
        raise ValueError("human review requires all closed fields")
    outcome = arguments["outcome"]
    if not isinstance(outcome, str) or outcome not in _HUMAN_REVIEW_OUTCOMES:
        raise ValueError("human review outcome is invalid")
    for field in (
        "expected_selection_revision",
        "expected_context_revision",
        "expected_queue_state_revision",
        "expected_queue_input_revision",
    ):
        value = arguments[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"human review {field} is invalid")
    operation_id = arguments["operation_id"]
    if not isinstance(operation_id, str) or not _OPAQUE_OPERATION_ID.fullmatch(operation_id):
        raise ValueError("human review operation_id is invalid")


def _validate_sync_finding_arguments(arguments: dict[str, Any]) -> None:  # noqa: C901
    """Admit one explicit repository publication request with full CAS."""

    if set(arguments) != _SYNC_FINDING_ARGUMENTS:
        raise ValueError("GitHub sync requires all closed fields")
    finding_id = arguments["finding_id"]
    if (
        not isinstance(finding_id, str)
        or not finding_id.strip()
        or len(finding_id) > _MAX_SYNC_FINDING_ID_LENGTH
        or not re.fullmatch(r"[A-Za-z0-9_.:-]+", finding_id)
    ):
        raise ValueError("GitHub sync finding_id is invalid")
    repository = arguments["repository"]
    if (
        not isinstance(repository, str)
        or len(repository) > _MAX_SYNC_REPOSITORY_LENGTH
        or not _GITHUB_REPOSITORY.fullmatch(repository)
    ):
        raise ValueError("GitHub sync repository is invalid")
    for field in (
        "expected_finding_revision",
        "expected_selection_revision",
        "expected_context_revision",
    ):
        value = arguments[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"GitHub sync {field} is invalid")
    source_revision = arguments["expected_source_revision"]
    if isinstance(source_revision, bool) or not isinstance(source_revision, (int, str)):
        raise ValueError("GitHub sync expected_source_revision is invalid")
    if isinstance(source_revision, int) and source_revision < 0:
        raise ValueError("GitHub sync expected_source_revision is invalid")
    if isinstance(source_revision, str) and (
        not source_revision or len(source_revision) > _MAX_SYNC_REPOSITORY_LENGTH
    ):
        raise ValueError("GitHub sync expected_source_revision is invalid")
    operation_id = arguments["operation_id"]
    if not isinstance(operation_id, str) or not _OPAQUE_OPERATION_ID.fullmatch(operation_id):
        raise ValueError("GitHub sync operation_id is invalid")
    if not isinstance(arguments["retry_ambiguous"], bool):
        raise ValueError("GitHub sync retry_ambiguous is invalid")


def _validate_materialization_arguments(arguments: dict[str, Any]) -> None:
    """Accept only selection-bound, path-free browser action arguments."""

    if set(arguments) != _ARGUMENTS["materialize_selected"]:
        raise ValueError("materialization requires all closed fields")
    operation_id = arguments["operation_id"]
    if not isinstance(operation_id, str) or not _OPAQUE_OPERATION_ID.fullmatch(operation_id):
        raise ValueError("materialization operation_id is invalid")
    for field in ("expected_selection_revision", "expected_context_revision"):
        value = arguments[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"materialization {field} is invalid")


def _validate_native_diagnostic_arguments(  # noqa: C901 - closed request validation
    arguments: dict[str, Any],
) -> None:
    """Admit only a bounded goal intervention bound to the current selection."""

    def finite_number(value: Any) -> bool:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return False
        try:
            return math.isfinite(value)
        except OverflowError:
            return False

    if set(arguments) != _ARGUMENTS["run_native_diagnostic"]:
        raise ValueError("native diagnostic requires all closed fields")
    for field in ("operation_id", "intervention_id"):
        value = arguments[field]
        if not isinstance(value, str) or not _OPAQUE_OPERATION_ID.fullmatch(value):
            raise ValueError(f"native diagnostic {field} is invalid")
    for field in ("expected_selection_revision", "expected_context_revision"):
        value = arguments[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"native diagnostic {field} is invalid")
    goal = arguments["robot_goal"]
    if not isinstance(goal, list) or len(goal) != 2 or not all(map(finite_number, goal)):
        raise ValueError("native diagnostic robot_goal is invalid")
    for field in ("activation_epsilon_m", "deadline_s"):
        value = arguments[field]
        if not finite_number(value):
            raise ValueError(f"native diagnostic {field} is invalid")
    if arguments["activation_epsilon_m"] < 0 or not 0 < arguments["deadline_s"] <= 60:
        raise ValueError("native diagnostic limits are invalid")


def _sanitize_native_diagnostic_result(
    value: Any, *, episode_id: str, source_episode_id: str | None = None
) -> dict[str, Any]:
    """Project only selected, measured diagnostic status; never raw runner data.

    Returns:
        A browser-safe, diagnostic-only result without runner inputs or paths.
    """

    projected: dict[str, Any] = {
        "status": "failed",
        "diagnostic_only": True,
        "scientific_claim_allowed": False,
    }
    if not isinstance(value, Mapping):
        return projected
    status = value.get("status")
    if not isinstance(status, str) or status not in _NATIVE_DIAGNOSTIC_SERVICE_STATUSES:
        return projected
    projected["status"] = status
    if status != "complete":
        return projected
    result = value.get("value")
    if not isinstance(result, Mapping) or result.get("status") != "complete":
        projected["status"] = "failed"
        return projected
    if (
        result.get("evidence_boundary") != "diagnostic_only"
        or result.get("scientific_claim_allowed") is not False
    ):
        projected["status"] = "failed"
        return projected
    original = result.get("original_identity")
    original_episode_id = original.get("episode_id") if isinstance(original, Mapping) else None
    if not isinstance(original_episode_id, str) or original_episode_id not in {
        episode_id,
        source_episode_id,
    }:
        projected["status"] = "conflict"
        return projected
    fidelity = result.get("fidelity")
    activation = result.get("activation")
    if (
        not isinstance(fidelity, Mapping)
        or fidelity.get("status") != "verified"
        or not isinstance(activation, Mapping)
        or activation.get("status") != "verified"
        or activation.get("input_changed") is not True
        or activation.get("trace_changed") is not True
    ):
        projected["status"] = "failed"
        return projected
    projected.update(control_fidelity="verified", activation="verified")
    return projected


def _materialization_source_alias(
    facade: ServiceAuditWorkbenchFacade, *, selected_episode_id: str
) -> str | None:
    """Return only the literal row ID from the current admitted episode read.

    Returns:
        A verified BA-05 row alias, or ``None`` when the cached read is stale.
    """

    episode_result = getattr(facade, "_last_episode", None)
    if getattr(episode_result, "status", None) != "complete":
        return None
    context = getattr(episode_result, "context", None)
    current_context = getattr(facade, "_context", None)
    if not facade._context_matches_request(current_context, context):
        return None
    context_episode_id = (
        current_context.get("episode_id")
        if isinstance(current_context, Mapping)
        else getattr(current_context, "episode_id", None)
    )
    if context_episode_id != selected_episode_id:
        return None
    episode = getattr(episode_result, "value", None)
    row = getattr(episode, "row", None)
    if not isinstance(row, Mapping):
        return None
    alias = row.get("episode_id")
    return alias if isinstance(alias, str) and alias else None


def _safe_materialization_failure_details(value: Any) -> dict[str, Any]:
    """Expose bounded non-path diagnostics for an unavailable render.

    Returns:
        A safe reason/diagnostic projection, or an empty mapping.
    """

    if not isinstance(value, Mapping):
        return {}
    details: dict[str, Any] = {}
    reason = value.get("reason")
    if isinstance(reason, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", reason):
        details["reason"] = reason
    diagnostics = value.get("diagnostics")
    if isinstance(diagnostics, (list, tuple)):
        safe_diagnostics = [
            item
            for item in diagnostics
            if isinstance(item, str) and re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", item)
        ][:8]
        if safe_diagnostics:
            details["diagnostics"] = safe_diagnostics
    return details


def _sanitize_materialization_result(
    value: Any, *, episode_id: str, source_episode_id: str | None = None
) -> dict[str, Any]:
    """Project only claim-bounded diagnostic status, never paths or artifacts.

    Returns:
        A bounded browser-safe status without source/output authority.
    """

    projected: dict[str, Any] = {
        "status": "failed",
        "diagnostic_only": True,
        "scientific_claim_allowed": False,
    }
    if not isinstance(value, Mapping):
        projected["reason"] = "materialization result is unavailable"
        return projected
    status = value.get("status")
    if not isinstance(status, str) or status not in _MATERIALIZATION_SERVICE_STATUSES:
        projected["reason"] = "materialization status is invalid"
        return projected
    result = value.get("value")
    projected["status"] = status
    if status not in {"complete", "partial"}:
        projected.update(_safe_materialization_failure_details(value.get("value")))
        return projected
    result_episode_id = result.get("episode_id") if isinstance(result, Mapping) else None
    if not isinstance(result_episode_id, str) or result_episode_id not in {
        episode_id,
        source_episode_id,
    }:
        projected.update(status="conflict", reason="materialization result is stale")
        return projected
    result_status = result.get("status")
    kind = result.get("materialization_kind")
    fidelity = result.get("fidelity")
    if (
        not isinstance(result_status, str)
        or result_status not in _MATERIALIZATION_STATUSES
        or not isinstance(kind, str)
        or kind not in _MATERIALIZATION_KINDS
        or not isinstance(fidelity, str)
        or fidelity not in _MATERIALIZATION_FIDELITIES
        or result.get("diagnostic_only") is not True
    ):
        projected.update(status="failed", reason="materialization result is invalid")
        return projected
    projected["status"] = result_status
    if result_status not in {"complete", "partial"} or kind == "unavailable":
        if kind == "unavailable":
            projected["status"] = "unavailable"
        return projected
    projected["classification"] = kind
    projected["fidelity"] = fidelity
    return projected


def _validate_codex_arguments(operation: str, arguments: dict[str, Any]) -> None:  # noqa: C901
    """Validate browser-facing Codex arguments before facade dispatch."""

    if operation == "codex_start":
        _bounded_text(arguments.get("prompt"), field="prompt", maximum=_MAX_CODEX_PROMPT_LENGTH)
        operation_id = _bounded_text(
            arguments.get("operation_id"),
            field="operation_id",
            maximum=_MAX_CODEX_OPERATION_ID_LENGTH,
        )
        if not _OPAQUE_OPERATION_ID.fullmatch(operation_id or ""):
            raise ValueError("operation_id must be an opaque identifier")
        token_budget = arguments.get("token_budget")
        if isinstance(token_budget, bool) or not isinstance(token_budget, int):
            raise ValueError("token_budget must be an integer")
        if not 1 <= token_budget <= _MAX_CODEX_TOKEN_BUDGET:
            raise ValueError("token_budget is outside the permitted bound")
        compute_budget = arguments.get("compute_budget")
        if isinstance(compute_budget, bool) or not isinstance(compute_budget, (int, float)):
            raise ValueError("compute_budget must be numeric")
        try:
            compute_budget_value = float(compute_budget)
        except (OverflowError, ValueError) as exc:
            raise ValueError("compute_budget must be finite") from exc
        if (
            not math.isfinite(compute_budget_value)
            or not 1.0 <= compute_budget_value <= _MAX_CODEX_COMPUTE_BUDGET
        ):
            raise ValueError("compute_budget is outside the permitted bound")
        return
    if operation == "codex_read":
        if "operation_id" in arguments:
            operation_id = _bounded_text(
                arguments["operation_id"],
                field="operation_id",
                maximum=_MAX_CODEX_OPERATION_ID_LENGTH,
            )
            if not _OPAQUE_OPERATION_ID.fullmatch(operation_id or ""):
                raise ValueError("operation_id must be an opaque identifier")
        return
    if operation == "codex_cancel":
        operation_id = _bounded_text(
            arguments.get("operation_id"),
            field="operation_id",
            maximum=_MAX_CODEX_OPERATION_ID_LENGTH,
        )
        if not _OPAQUE_OPERATION_ID.fullmatch(operation_id or ""):
            raise ValueError("operation_id must be an opaque identifier")
        _bounded_text(arguments.get("reason"), field="reason", maximum=_MAX_CODEX_REASON_LENGTH)


def _validate_related_case_arguments(arguments: dict[str, Any]) -> None:  # noqa: C901
    """Validate the small browser read contract for related-case retrieval."""

    if "episode_id" in arguments:
        episode_id = _bounded_text(
            arguments["episode_id"],
            field="episode_id",
            maximum=_MAX_RELATED_CASE_ID_LENGTH,
        )
        if not _OPAQUE_OPERATION_ID.fullmatch(episode_id or ""):
            raise ValueError("episode_id must be an opaque identifier")
    if "mode" in arguments:
        mode = _bounded_text(arguments["mode"], field="mode", maximum=_MAX_RELATED_CASE_MODE_LENGTH)
        if mode not in _RELATED_CASE_MODES:
            raise ValueError("unsupported related-case mode")
    if "limit" in arguments:
        limit = arguments["limit"]
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise ValueError("related-case limit must be an integer")
        if not 0 <= limit <= _MAX_RELATED_CASE_LIMIT:
            raise ValueError("related-case limit is outside the permitted bound")
    if "operation_id" in arguments:
        operation_id = _bounded_text(
            arguments["operation_id"],
            field="operation_id",
            maximum=_MAX_CODEX_OPERATION_ID_LENGTH,
        )
        if not _OPAQUE_OPERATION_ID.fullmatch(operation_id or ""):
            raise ValueError("operation_id must be an opaque identifier")
    if "expected_context_revision" in arguments:
        revision = arguments["expected_context_revision"]
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise ValueError("expected_context_revision must be a non-negative integer")


def _normalize_codex_text(value: str) -> str:
    """Decode bounded percent escapes for path detection only.

    Codex text is untrusted provider output.  The normalized spelling is never
    returned directly when it differs from the input: otherwise a percent-
    encoded credential could become a literal credential during projection.

    Returns:
        A bounded text spelling suitable only for path detection.
    """

    normalized = value
    for _ in range(3):
        decoded = unquote(normalized)
        if decoded == normalized:
            break
        normalized = decoded
    return normalized


def _redact_path_like_text(value: str) -> str:
    """Redact path-shaped tokens with a bounded linear scan.

    Provider text is untrusted and can be arbitrarily repetitive.  Avoid a
    backtracking regular expression here so sanitization remains predictable
    even for adversarial input.

    Returns:
        Text with path-shaped tokens replaced by a fixed marker.
    """

    output: list[str] = []
    token: list[str] = []

    def flush_token() -> None:
        if token:
            token_text = "".join(token)
            output.append(
                "<path redacted>" if "/" in token_text or "\\" in token_text else token_text
            )
            token.clear()

    for character in value:
        if character.isspace():
            flush_token()
            output.append(character)
        else:
            token.append(character)
    flush_token()
    return "".join(output)


def _sanitize_codex_text(value: Any, *, maximum: int = 8192) -> str | None:
    if not isinstance(value, str):
        return None
    bounded = value.replace("\x00", "")[:maximum]
    normalized = _normalize_codex_text(bounded)
    if normalized != bounded:
        # Partial encoding can leave the credential itself in the unencoded
        # fragments, so never preserve any part of an encoded provider field.
        return "<encoded redacted>"
    return _redact_path_like_text(bounded)


def _sanitize_codex_reference(value: Any) -> str | int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value if not isinstance(value, float) or math.isfinite(value) else None
    if not isinstance(value, str):
        return None
    candidate = value[:256]
    if "%" in value or not _CODEX_REFERENCE_PATTERN.fullmatch(candidate):
        return None
    return candidate


def _guard_codex_container(value: Any, *, depth: int, seen: set[int]) -> None:
    """Bound recursive Codex result traversal before inspecting a container."""

    if depth > _MAX_CODEX_PROJECTION_DEPTH:
        raise _CodexProjectionError("Codex result nesting exceeds the browser bound")
    if not isinstance(value, (Mapping, list)):
        return
    identity = id(value)
    if identity in seen:
        raise _CodexProjectionError("Codex result contains a repeated or cyclic container")
    if len(seen) >= _MAX_CODEX_PROJECTION_NODES:
        raise _CodexProjectionError("Codex result exceeds the browser resource bound")
    seen.add(identity)


def _sanitize_codex_evidence(value: Any, *, depth: int, seen: set[int]) -> list[Any]:
    if not isinstance(value, list):
        return []
    _guard_codex_container(value, depth=depth, seen=seen)
    sanitized: list[Any] = []
    for item in value[:64]:
        if isinstance(item, Mapping):
            _guard_codex_container(item, depth=depth + 1, seen=seen)
            reference = {
                key: _sanitize_codex_reference(item[key])
                for key in ("evidence_id", "id", "reference")
                if key in item and _sanitize_codex_reference(item[key]) is not None
            }
            if reference:
                sanitized.append(reference)
        else:
            reference = _sanitize_codex_reference(item)
            if reference is not None:
                sanitized.append(reference)
    return sanitized


def _sanitize_codex_usage(value: Any, *, depth: int, seen: set[int]) -> dict[str, int | float]:
    if not isinstance(value, Mapping):
        return {}
    _guard_codex_container(value, depth=depth, seen=seen)
    sanitized: dict[str, int | float] = {}
    for key in _CODEX_USAGE_FIELDS:
        item = value.get(key)
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            continue
        if isinstance(item, float) and not math.isfinite(item):
            continue
        sanitized[key] = item
    return sanitized


def _sanitize_codex_activity(  # noqa: C901
    value: Any, *, depth: int, seen: set[int]
) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    _guard_codex_container(value, depth=depth, seen=seen)
    sanitized: list[dict[str, Any]] = []
    for event in value[:64]:
        if isinstance(event, Mapping):
            _guard_codex_container(event, depth=depth + 1, seen=seen)
            item: dict[str, Any] = {}
            for key in _CODEX_ACTIVITY_FIELDS:
                if key not in event:
                    continue
                if key in {"message", "text"}:
                    message = _sanitize_codex_text(event[key], maximum=512)
                    if message is not None:
                        item[key] = message
                elif key == "evidence_ids":
                    item[key] = _sanitize_codex_evidence(event[key], depth=depth + 2, seen=seen)
                elif key == "operation_id":
                    reference = _sanitize_codex_reference(event[key])
                    if reference is not None:
                        item[key] = reference
                elif key == "timestamp":
                    timestamp = _sanitize_codex_reference(event[key])
                    if timestamp is not None:
                        item[key] = timestamp
            if item:
                sanitized.append(item)
        else:
            message = _sanitize_codex_text(event, maximum=512)
            if message is not None:
                sanitized.append({"message": message})
    return sanitized


def _sanitize_codex_mapping(  # noqa: C901
    value: Any,
    fields: frozenset[str],
    *,
    depth: int = 0,
    seen: set[int] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return {}
    if seen is None:
        seen = set()
    _guard_codex_container(value, depth=depth, seen=seen)
    sanitized: dict[str, Any] = {}
    for key in fields:
        if key not in value:
            continue
        item = value[key]
        if key in {"context", "current_context"}:
            sanitized[key] = _sanitize_codex_mapping(
                item, _CODEX_CONTEXT_FIELDS, depth=depth + 1, seen=seen
            )
        elif key in {"source", "current_source"}:
            sanitized[key] = _sanitize_codex_mapping(
                item, _CODEX_SOURCE_FIELDS, depth=depth + 1, seen=seen
            )
        elif key in {"session", "receipt", "snapshot", "current", "result"}:
            sanitized[key] = _sanitize_codex_mapping(
                item, _CODEX_RESULT_FIELDS, depth=depth + 1, seen=seen
            )
        elif key in {"evidence_ids", "evidence_references"}:
            sanitized[key] = _sanitize_codex_evidence(item, depth=depth + 1, seen=seen)
        elif key == "usage":
            sanitized[key] = _sanitize_codex_usage(item, depth=depth + 1, seen=seen)
        elif key in {"activity", "events"}:
            sanitized[key] = _sanitize_codex_activity(item, depth=depth + 1, seen=seen)
        elif key in {
            "operation_id",
            "route_id",
            "context_revision",
            "selection_revision",
            "episode_id",
            "scenario_id",
            "execution_id",
            "source_revision",
            "source_digest",
        }:
            reference = _sanitize_codex_reference(item)
            if reference is not None:
                sanitized[key] = reference
        elif key in {"status", "reason", "activity_scope"}:
            text = _sanitize_codex_text(item)
            if text is not None:
                sanitized[key] = text
    return sanitized


def _sanitize_codex_result(value: Any) -> dict[str, Any]:
    """Project Codex results into the positive browser-safe response shape.

    Returns:
        A bounded JSON-compatible projection containing only status, current
        context/source references, usage, and activity fields.
    """

    return _sanitize_codex_mapping(value, _CODEX_RESULT_FIELDS)


class _RelatedCaseProjectionError(ValueError):
    """Raised when a related-case result cannot satisfy its read contract."""


def _sanitize_related_text(value: Any, *, maximum: int = 512) -> str | None:
    """Keep bounded explanation text while hiding encoded/path-like values.

    Returns:
        Sanitized text or ``None`` for non-text values.
    """

    if not isinstance(value, str):
        return None
    bounded = value.replace("\x00", "")[:maximum]
    normalized = _normalize_codex_text(bounded)
    if normalized != bounded:
        return "<encoded redacted>"
    return _redact_path_like_text(bounded)


def _sanitize_related_reference(value: Any) -> str | int | float | None:
    """Project one related-case ID or small scalar reference.

    Returns:
        A bounded opaque scalar or ``None`` when unsafe.
    """

    return _sanitize_codex_reference(value)


def _sanitize_related_features(  # noqa: C901
    value: Any, *, depth: int = 0, nodes: list[int] | None = None
) -> Any:
    """Project similarity features without carrying authority or membership.

    Returns:
        A bounded JSON-compatible feature projection.
    """

    if nodes is None:
        nodes = [0]
    if depth > 4 or nodes[0] >= 128:
        raise _RelatedCaseProjectionError("related-case features exceed the browser bound")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        nodes[0] += 1
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise _RelatedCaseProjectionError("related-case feature is not finite")
        nodes[0] += 1
        return value
    if isinstance(value, str):
        nodes[0] += 1
        return _sanitize_related_text(value, maximum=256)
    if isinstance(value, Mapping):
        nodes[0] += 1
        projected: dict[str, Any] = {}
        for raw_key, item in list(value.items())[:32]:
            key = str(raw_key)
            normalized = key.casefold().replace("-", "_")
            if normalized in _RELATED_CASE_FORBIDDEN_KEYS or "path" in normalized:
                continue
            projected[key[:128]] = _sanitize_related_features(item, depth=depth + 1, nodes=nodes)
        return projected
    if isinstance(value, (list, tuple)):
        nodes[0] += 1
        return [
            _sanitize_related_features(item, depth=depth + 1, nodes=nodes) for item in value[:32]
        ]
    raise _RelatedCaseProjectionError("related-case feature has an unsupported type")


def _sanitize_related_context(value: Any) -> dict[str, Any]:
    """Expose only revision and opaque identity fields from service context.

    Returns:
        A safe context identity projection.
    """

    if not isinstance(value, Mapping):
        return {}
    projected: dict[str, Any] = {}
    for key in (
        "context_revision",
        "source_revision",
        "episode_id",
        "scenario_id",
        "execution_id",
    ):
        if key not in value:
            continue
        reference = _sanitize_related_reference(value[key])
        if reference is not None:
            projected[key] = reference
    return projected


def _related_case_items(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, Mapping):
        for key in ("candidates", "related_cases", "results", "items"):
            candidate = value.get(key)
            if isinstance(candidate, (list, tuple)):
                return list(candidate)
    raise _RelatedCaseProjectionError("related-case service result has no bounded candidate list")


def _sanitize_related_case_item(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _RelatedCaseProjectionError("related-case candidate is not an object")
    projected: dict[str, Any] = {}
    query_id = _sanitize_related_reference(value.get("query_id"))
    candidate_id = _sanitize_related_reference(value.get("candidate_id"))
    mode = _sanitize_related_reference(value.get("mode"))
    compatibility = value.get("compatibility")
    score = value.get("score")
    if query_id is None or candidate_id is None or mode is None:
        raise _RelatedCaseProjectionError("related-case candidate identity is incomplete")
    if mode not in _RELATED_CASE_MODES:
        raise _RelatedCaseProjectionError("related-case candidate mode is unsupported")
    if compatibility not in _RELATED_CASE_COMPATIBILITY:
        raise _RelatedCaseProjectionError("related-case candidate compatibility is invalid")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
        raise _RelatedCaseProjectionError("related-case candidate score is invalid")
    if not 0.0 <= float(score) <= 1.0:
        raise _RelatedCaseProjectionError("related-case candidate score is outside the bound")
    projected.update(
        query_id=query_id,
        candidate_id=candidate_id,
        mode=mode,
        score=float(score),
        compatibility=compatibility,
    )
    for key in ("reasons", "missingness", "warnings"):
        raw_values = value.get(key, ())
        if not isinstance(raw_values, (list, tuple)):
            raise _RelatedCaseProjectionError(f"related-case {key} is malformed")
        projected[key] = [
            text
            for item in list(raw_values)[:16]
            if (text := _sanitize_related_text(item, maximum=256)) is not None
        ]
    raw_features = value.get("features", {})
    projected["features"] = _sanitize_related_features(raw_features)
    # Similarity rows are retrieval candidates only.  Membership fields from
    # a malformed provider result are deliberately not copied across the seam.
    return projected


def _related_context_revision(facade: ServiceAuditWorkbenchFacade) -> Any:
    context = getattr(facade, "_context", None)
    if isinstance(context, Mapping):
        return context.get("context_revision")
    return getattr(context, "context_revision", None)


def _related_scope_conflict(
    facade: ServiceAuditWorkbenchFacade, arguments: Mapping[str, Any]
) -> str | None:
    """Reject browser reads that are stale or point outside the selected case.

    Returns:
        A conflict reason, or ``None`` when the request is current.
    """

    selected = getattr(facade, "_selected_episode_id", None)
    requested = arguments.get("episode_id")
    if not selected:
        return "select an audit case before loading related cases"
    if requested and str(selected) != str(requested):
        return "related-case request targets a foreign selected episode"
    expected_revision = arguments.get("expected_context_revision")
    actual_revision = _related_context_revision(facade)
    if (
        expected_revision is not None
        and actual_revision is not None
        and str(expected_revision) != str(actual_revision)
    ):
        return "related-case request targets a stale context revision"
    return None


def _selected_source_query_id(facade: ServiceAuditWorkbenchFacade) -> str | None:
    """Read the selected BA-05 episode's literal row ID as a verified alias.

    Returns:
        The scanner-admitted source row ID, or ``None`` when unavailable.
    """

    episode_result = getattr(facade, "_last_episode", None)
    episode = getattr(episode_result, "value", None)
    if isinstance(episode, Mapping):
        row = episode.get("row")
    else:
        row = getattr(episode, "row", None)
    if not isinstance(row, Mapping):
        return None
    source_id = row.get("episode_id")
    return source_id if isinstance(source_id, str) and source_id else None


def _related_case_envelope_identity(value: Any) -> tuple[str, str]:
    """Validate status/schema before either can enter browser-visible data.

    Returns:
        The supported status and schema version.
    """

    if not isinstance(value, Mapping):
        raise _RelatedCaseProjectionError("related-case service result is not an object")
    status = value.get("status", "unavailable")
    if not isinstance(status, str) or status not in {
        "complete",
        "committed",
        "ok",
        "denied",
        "conflict",
        "failed",
        "unavailable",
        "cancelled",
    }:
        raise _RelatedCaseProjectionError("related-case service status is malformed")
    schema_version = value.get("schema_version", "audit-service.v1")
    if not isinstance(schema_version, str) or schema_version not in {
        "audit-service.v1",
        "audit-workbench-service.v1",
    }:
        raise _RelatedCaseProjectionError("related-case service schema is unsupported")
    return status, schema_version


def _sanitize_related_cases_result(
    value: Any, *, expected_query_id: str | None = None, source_query_id: str | None = None
) -> dict[str, Any]:
    """Project one BA-05 related-case envelope into a read-only browser shape.

    Returns:
        A safe service envelope with unconfirmed retrieval candidates only.
    """

    status, schema_version = _related_case_envelope_identity(value)
    result: dict[str, Any] = {
        "schema_version": schema_version,
        "status": status,
        "reason": _sanitize_related_text(value.get("reason", ""), maximum=512) or "",
        "value": None,
        "membership_boundary": "candidates_are_unconfirmed",
        "claim_boundary": "retrieval_only_not_benchmark_evidence",
    }
    operation = _sanitize_related_reference(value.get("operation_id"))
    if operation is not None:
        result["operation_id"] = operation
    context = _sanitize_related_context(value.get("context"))
    if context:
        result["context"] = context
    if status not in {"complete", "committed", "ok"}:
        return result
    candidates = [
        _sanitize_related_case_item(item) for item in _related_case_items(value.get("value"))
    ]
    query_ids = {str(item["query_id"]) for item in candidates}
    accepted_query_ids = {str(expected_query_id)}
    if source_query_id is not None:
        accepted_query_ids.add(source_query_id)
    if expected_query_id is not None and not query_ids <= accepted_query_ids:
        result["status"] = "conflict"
        result["reason"] = "related-case result belongs to a foreign selected episode"
        return result
    if len(query_ids) > 1:
        result["status"] = "conflict"
        result["reason"] = "related-case result contains multiple query episodes"
        return result
    if expected_query_id is not None:
        # BA-02 selects a generated EpisodeRef ID; BA-03 similarity reports
        # the selected row's literal ID. Verify that alias against the
        # service-read episode before presenting one canonical queue ID.
        for candidate in candidates:
            candidate["query_id"] = str(expected_query_id)
    result["value"] = candidates
    result["candidate_count"] = len(candidates)
    return result


def make_audit_workbench_server(  # noqa: C901 - bounded HTTP verb guards live here
    document: dict[str, Any],
    facade: ServiceAuditWorkbenchFacade,
    *,
    materialization_output_root: Path | None = None,
) -> ThreadingHTTPServer:
    """Bind one exact-origin server on an ephemeral IPv4 loopback port.

    The caller owns ``serve_forever`` and ``server_close``.  Only the generated
    SREV-15 HTML and its declared local component assets are served.

    Returns:
        A loopback server with no browser-visible audit credential.
    """

    extension = document.get("extensions", {}).get("audit_workbench")
    if not isinstance(extension, dict) or extension.get("mode") != "service_live":
        raise ValueError("live audit extension is required")
    if not isinstance(facade, ServiceAuditWorkbenchFacade):
        raise TypeError("a server-held service facade is required")
    html = _render_html(document).encode("utf-8")
    secret = getattr(facade, "_token", None)
    if isinstance(secret, str) and secret:
        json_secret = json.dumps(secret)[1:-1]
        # ``_render_html`` escapes ``</`` to ``<\\/`` inside JSON script data.
        # Check every final serialized spelling, including that browser-parse
        # round-trip form, before exposing the document.
        serialized_forms = {
            secret.encode("utf-8"),
            json_secret.encode("utf-8"),
            json_secret.replace("</", "<\\/").encode("utf-8"),
        }
        if any(form and form in html for form in serialized_forms):
            raise ValueError("audit credential leaked into the SREV-15 document")
    allowed_assets = frozenset(f"/components/{item}" for item in AUDIT_WORKBENCH_ASSETS)
    asset_root = Path(__file__).parent / "web_assets" / "components"
    call_lock = threading.RLock()
    browser_session = secrets.token_urlsafe(32)
    session_deadline = time.monotonic() + SESSION_LIFETIME_SECONDS

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *args: Any) -> None:
            # Operation IDs and service reasons are deliberately never logged.
            pass

        def _origin(self) -> str:
            return f"http://127.0.0.1:{self.server.server_port}"

        def _host_valid(self) -> bool:
            return self.headers.get("Host", "") == f"127.0.0.1:{self.server.server_port}"

        def _session_valid(self) -> bool:
            if time.monotonic() >= session_deadline:
                return False
            cookie_headers = self.headers.get_all("Cookie", [])
            if len(cookie_headers) != 1:
                return False
            cookie_parts = [part.strip() for part in cookie_headers[0].split(";")]
            if sum(part.startswith(f"{SESSION_COOKIE}=") for part in cookie_parts) != 1:
                return False
            try:
                cookie = SimpleCookie(cookie_headers[0])
                candidate = cookie[SESSION_COOKIE].value
            except (CookieError, KeyError):
                return False
            return secrets.compare_digest(candidate, browser_session)

        def _send(
            self, status: HTTPStatus, body: bytes, content_type: str, *, issue_cookie: bool = False
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            if issue_cookie:
                self.send_header(
                    "Set-Cookie",
                    f"{SESSION_COOKIE}={browser_session}; HttpOnly; SameSite=Strict; "
                    f"Path=/api/audit; Max-Age={SESSION_LIFETIME_SECONDS}",
                )
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: HTTPStatus, value: dict[str, Any]) -> None:
            encoded = json.dumps(value, allow_nan=False, sort_keys=True).encode("utf-8")
            secret = getattr(facade, "_token", None)
            if isinstance(secret, str) and secret:
                escaped_secret = json.dumps(secret)[1:-1].encode("utf-8")
                encoded = encoded.replace(escaped_secret, b"<redacted>")
                encoded = encoded.replace(secret.encode("utf-8"), b"<redacted>")
            self._send(status, encoded, "application/json; charset=utf-8")

        def do_GET(self) -> None:
            if not self._host_valid():
                self._json(HTTPStatus.FORBIDDEN, {"status": "denied"})
                return
            path = urlsplit(self.path)
            if path.query or path.fragment:
                self._json(HTTPStatus.NOT_FOUND, {"status": "unavailable"})
                return
            if path.path in {"/", "/review-workbench.v1.html"}:
                self._send(HTTPStatus.OK, html, "text/html; charset=utf-8", issue_cookie=True)
                return
            if path.path not in allowed_assets:
                self._json(HTTPStatus.NOT_FOUND, {"status": "unavailable"})
                return
            relative = path.path.removeprefix("/components/")
            content = (asset_root / relative).read_bytes()
            mime = "text/javascript" if relative.endswith(".js") else "text/css"
            self._send(HTTPStatus.OK, content, f"{mime}; charset=utf-8")

        def do_POST(self) -> None:  # noqa: C901, PLR0912, PLR0915 - fail-closed request validation
            if not self._host_valid():
                self._json(HTTPStatus.FORBIDDEN, {"status": "denied"})
                return
            if self.path != "/api/audit":
                self._json(HTTPStatus.NOT_FOUND, {"status": "unavailable"})
                return
            origin = self.headers.get("Origin", "")
            if not is_loopback_origin(origin) or origin != self._origin():
                self._json(HTTPStatus.FORBIDDEN, {"status": "denied"})
                return
            if not self._session_valid():
                self._json(HTTPStatus.FORBIDDEN, {"status": "denied"})
                return
            if self.headers.get_content_type() != "application/json":
                self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"status": "failed"})
                return
            try:
                size = int(self.headers.get("Content-Length", ""))
            except ValueError:
                size = 0
            if not 0 < size <= MAX_REQUEST_BYTES:
                self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"status": "failed"})
                return
            try:
                payload = json.loads(self.rfile.read(size))
            except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
                self._json(HTTPStatus.BAD_REQUEST, {"status": "failed"})
                return
            if not isinstance(payload, dict) or set(payload) != {"operation", "arguments"}:
                self._json(HTTPStatus.BAD_REQUEST, {"status": "failed"})
                return
            operation, arguments = payload["operation"], payload["arguments"]
            if (
                not isinstance(operation, str)
                or operation not in _OPERATIONS
                or not isinstance(arguments, dict)
            ):
                self._json(HTTPStatus.BAD_REQUEST, {"status": "failed"})
                return
            if _unsafe_payload(payload):
                self._json(HTTPStatus.FORBIDDEN, {"status": "denied"})
                return
            if not set(arguments) <= _ARGUMENTS[operation]:
                self._json(HTTPStatus.BAD_REQUEST, {"status": "failed"})
                return
            try:
                _validate_codex_arguments(operation, arguments)
                if operation == "related_cases":
                    _validate_related_case_arguments(arguments)
                if operation == "record_human_review":
                    _validate_human_review_arguments(arguments)
                if operation == "sync_finding":
                    _validate_sync_finding_arguments(arguments)
                if operation == "materialize_selected":
                    _validate_materialization_arguments(arguments)
                if operation == "run_native_diagnostic":
                    _validate_native_diagnostic_arguments(arguments)
            except ValueError:
                self._json(HTTPStatus.BAD_REQUEST, {"status": "failed"})
                return
            try:
                with call_lock:
                    if operation == "run_native_diagnostic":
                        config = getattr(
                            getattr(facade, "_service", None), "native_diagnostic_config", None
                        )
                        if not getattr(config, "bindings", ()):
                            self._json(
                                HTTPStatus.NOT_IMPLEMENTED,
                                {
                                    "status": "unavailable",
                                    "reason": "native diagnostic is not configured",
                                },
                            )
                            return
                        selected = getattr(facade, "_selected_episode_id", None)
                        if (
                            not isinstance(selected, str)
                            or not selected
                            or arguments["expected_selection_revision"]
                            != getattr(facade, "_selection_epoch", None)
                            or arguments["expected_context_revision"] != facade._context_revision()
                        ):
                            self._json(
                                HTTPStatus.CONFLICT,
                                {
                                    "status": "conflict",
                                    "reason": "native diagnostic selection is stale",
                                },
                            )
                            return
                        result = facade.run_native_diagnostic(
                            intervention_id=arguments["intervention_id"],
                            robot_goal=arguments["robot_goal"],
                            activation_epsilon_m=arguments["activation_epsilon_m"],
                            deadline_s=arguments["deadline_s"],
                            context=deepcopy(getattr(facade, "_context", None)),
                            operation_id=arguments["operation_id"],
                        )
                        self._json(
                            HTTPStatus.OK,
                            _sanitize_native_diagnostic_result(
                                result,
                                episode_id=selected,
                                source_episode_id=_materialization_source_alias(
                                    facade, selected_episode_id=selected
                                ),
                            ),
                        )
                        return
                    if operation == "materialize_selected":
                        if materialization_output_root is None:
                            self._json(
                                HTTPStatus.NOT_IMPLEMENTED,
                                {
                                    "status": "unavailable",
                                    "reason": "materialization is not configured",
                                },
                            )
                            return
                        selected = getattr(facade, "_selected_episode_id", None)
                        if (
                            not isinstance(selected, str)
                            or not selected
                            or arguments["expected_selection_revision"]
                            != getattr(facade, "_selection_epoch", None)
                            or arguments["expected_context_revision"] != facade._context_revision()
                        ):
                            self._json(
                                HTTPStatus.CONFLICT,
                                {
                                    "status": "conflict",
                                    "reason": "materialization selection is stale",
                                },
                            )
                            return
                        result = facade.materialize_selected(
                            output_root=materialization_output_root,
                            context=deepcopy(getattr(facade, "_context", None)),
                            operation_id=arguments["operation_id"],
                        )
                        self._json(
                            HTTPStatus.OK,
                            _sanitize_materialization_result(
                                result,
                                episode_id=selected,
                                source_episode_id=_materialization_source_alias(
                                    facade, selected_episode_id=selected
                                ),
                            ),
                        )
                        return
                    if operation == "sync_finding":
                        selected = getattr(facade, "_selected_episode_id", None)
                        if (
                            not isinstance(selected, str)
                            or not selected
                            or arguments["expected_selection_revision"]
                            != getattr(facade, "_selection_epoch", None)
                            or arguments["expected_context_revision"] != facade._context_revision()
                        ):
                            self._json(
                                HTTPStatus.CONFLICT,
                                {
                                    "status": "conflict",
                                    "reason": "GitHub sync selection or context is stale",
                                },
                            )
                            return
                        result = facade.sync_finding(**arguments)
                        self._json(HTTPStatus.OK, result)
                        return
                    if operation == "related_cases":
                        scope_reason = _related_scope_conflict(facade, arguments)
                        if scope_reason is not None:
                            self._json(
                                HTTPStatus.CONFLICT,
                                {"status": "conflict", "reason": scope_reason},
                            )
                            return
                    result = _dispatch_facade_operation(facade, operation, arguments)
            except _CodexCapabilityUnavailable:
                self._json(
                    HTTPStatus.NOT_IMPLEMENTED,
                    {"status": "unavailable", "reason": "Codex capability is unavailable"},
                )
                return
            except (TypeError, ValueError):
                self._json(HTTPStatus.BAD_REQUEST, {"status": "failed"})
                return
            except Exception:  # noqa: BLE001 - never send service exceptions to browser
                self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"status": "failed"})
                return
            if operation == "related_cases":
                expected_query_id = arguments.get("episode_id") or getattr(
                    facade, "_selected_episode_id", None
                )
                try:
                    result = _sanitize_related_cases_result(
                        result,
                        expected_query_id=(
                            str(expected_query_id) if expected_query_id is not None else None
                        ),
                        source_query_id=_selected_source_query_id(facade),
                    )
                except _RelatedCaseProjectionError:
                    self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"status": "failed"})
                    return
            if operation.startswith("codex_"):
                try:
                    result = _sanitize_codex_result(result)
                except Exception:  # noqa: BLE001 - never expose malformed provider results
                    self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"status": "failed"})
                    return
            if not isinstance(result, dict):
                result = dict(result)
            self._json(HTTPStatus.OK, result)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    # Do not let a caller close the SQLite-backed service while an in-flight
    # request still owns the facade. ``ThreadingHTTPServer`` defaults to
    # daemon request threads, which makes ``server_close()`` return while a
    # handler can still be reading or checkpointing the audit store.
    server.daemon_threads = False
    server.block_on_close = True
    return server


def _unsafe_payload(value: Any) -> bool:
    """Detect forbidden authority fields and excessive nesting without recursion.

    Returns:
        Whether the client payload must be denied before facade dispatch.
    """

    pending = [(value, 0)]
    while pending:
        current, depth = pending.pop()
        if depth > MAX_JSON_DEPTH:
            return True
        if isinstance(current, dict):
            for key, item in current.items():
                if str(key).lower() in _FORBIDDEN_KEYS:
                    return True
                pending.append((item, depth + 1))
        elif isinstance(current, list):
            pending.extend((item, depth + 1) for item in current)
    return False
