# ruff: noqa: DOC201

"""Validated offline MCP-shaped adapter for :mod:`audit_service`.

The repository does not currently depend on an MCP SDK.  This module therefore
owns only a small, versioned request/response envelope and a dispatcher that
forwards every operation to one :class:`AuditService` instance.  It contains
no storage, provider, renderer, or authorization policy of its own.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any

from robot_sf.analysis_workbench.audit_contracts import record_from_dict
from robot_sf.analysis_workbench.audit_service import (
    AUDIT_SELECTION_CONTEXT_SCHEMA_VERSION,
    AuditSelectionContext,
    AuditService,
    AuditServiceError,
    AuditValidationError,
    ServiceResult,
)

AUDIT_MCP_REQUEST_SCHEMA_VERSION = "audit-mcp-request.v1"
AUDIT_MCP_RESPONSE_SCHEMA_VERSION = "audit-mcp-response.v1"
MAX_MCP_REQUEST_BYTES = 256 * 1024
MAX_MCP_PAYLOAD_BYTES = 128 * 1024
MAX_MCP_DEPTH = 20
MAX_MCP_NODES = 4_096
REDACTED_MCP_REQUEST_ID = "<redacted-request-id>"
LOOPBACK_ORIGINS = frozenset(
    {
        "http://127.0.0.1",
        "https://127.0.0.1",
        "http://localhost",
        "https://localhost",
        "http://[::1]",
        "https://[::1]",
    }
)


def _walk(value: Any, *, depth: int = 0, nodes: list[int] | None = None) -> None:
    """Apply a bounded JSON-like walk before dispatching untrusted payload data."""

    counters = nodes if nodes is not None else [0]
    counters[0] += 1
    if counters[0] > MAX_MCP_NODES:
        raise AuditValidationError("MCP payload contains too many values")
    if depth > MAX_MCP_DEPTH:
        raise AuditValidationError("MCP payload is too deeply nested")
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise AuditValidationError("MCP object keys must be strings")
            _walk(item, depth=depth + 1, nodes=counters)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _walk(item, depth=depth + 1, nodes=counters)


def _strict_payload(value: Mapping[str, Any], *, limit: int, name: str) -> dict[str, Any]:
    """Copy and bound a request mapping without interpreting text as authority."""

    if not isinstance(value, Mapping):
        raise AuditValidationError(f"{name} must be a mapping")
    copied = dict(value)
    _walk(copied)
    try:
        encoded = json.dumps(copied, allow_nan=False, separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise AuditValidationError(f"{name} must be strict JSON: {exc}") from exc
    if len(encoded) > limit:
        raise AuditValidationError(f"{name} exceeds {limit} bytes")
    return copied


def _json_safe(value: Any) -> Any:
    """Project service values into strict MCP JSON without changing authority."""

    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class AuditMCPRequest:
    """Closed MCP request envelope bound to one authenticated service session."""

    request_id: str
    session_id: str
    session_token: str = field(repr=False)
    origin: str
    operation: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = AUDIT_MCP_REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate identity, origin text, operation name, and payload bounds."""

        for name, value in (
            ("request_id", self.request_id),
            ("session_id", self.session_id),
            ("session_token", self.session_token),
            ("origin", self.origin),
            ("operation", self.operation),
        ):
            if not isinstance(value, str) or not value.strip():
                raise AuditValidationError(f"MCP {name} must be non-empty")
            if len(value) > 4_096:
                raise AuditValidationError(f"MCP {name} exceeds the text limit")
        if self.schema_version != AUDIT_MCP_REQUEST_SCHEMA_VERSION:
            raise AuditValidationError("unsupported MCP request schema")
        object.__setattr__(
            self,
            "payload",
            _strict_payload(self.payload, limit=MAX_MCP_PAYLOAD_BYTES, name="MCP payload"),
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> AuditMCPRequest:
        """Build a closed request from an untrusted mapping."""

        payload = _strict_payload(value, limit=MAX_MCP_REQUEST_BYTES, name="MCP request")
        allowed = {
            "schema_version",
            "request_id",
            "session_id",
            "session_token",
            "origin",
            "operation",
            "payload",
        }
        unknown = set(payload) - allowed
        if unknown:
            raise AuditValidationError(
                f"MCP request contains unknown fields: {', '.join(sorted(unknown))}"
            )
        required = {
            "schema_version",
            "request_id",
            "session_id",
            "session_token",
            "origin",
            "operation",
        }
        missing = required - set(payload)
        if missing:
            raise AuditValidationError(
                f"MCP request is missing required fields: {', '.join(sorted(missing))}"
            )
        return cls(**payload)

    def to_dict(self, *, include_token: bool = False) -> dict[str, Any]:
        """Return a safe request envelope; token inclusion is privileged."""

        result = {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "session_token": self.session_token if include_token else "<redacted>",
            "origin": self.origin,
            "operation": self.operation,
            "payload": dict(self.payload),
        }
        return result


@dataclass(frozen=True, slots=True)
class AuditMCPResponse:
    """Closed MCP response envelope carrying the shared service result."""

    request_id: str
    status: str
    result: Mapping[str, Any] = field(default_factory=dict)
    reason: str = ""
    schema_version: str = AUDIT_MCP_RESPONSE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the response envelope."""

        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "status": self.status,
            "reason": self.reason,
            "result": _json_safe(self.result),
        }


class AuditMCPDispatcher:
    """Thin validated dispatcher; all domain work remains in ``AuditService``."""

    def __init__(
        self, service: AuditService, *, allowed_origins: set[str] | frozenset[str] | None = None
    ) -> None:
        """Bind one service authority and an explicit loopback origin set."""

        self.service = service
        self.allowed_origins = frozenset(allowed_origins or LOOPBACK_ORIGINS)

    def _request(self, request: AuditMCPRequest | Mapping[str, Any]) -> AuditMCPRequest:
        return (
            request
            if isinstance(request, AuditMCPRequest)
            else AuditMCPRequest.from_mapping(request)
        )

    @staticmethod
    def _context(payload: Mapping[str, Any]) -> AuditSelectionContext | None:
        value = payload.get("context")
        if value is None:
            return None
        if not isinstance(value, Mapping):
            raise AuditValidationError("MCP context must be a mapping")
        if value.get("schema_version") != AUDIT_SELECTION_CONTEXT_SCHEMA_VERSION:
            raise AuditValidationError("MCP context must use the versioned selection schema")
        return AuditSelectionContext.from_mapping(value)

    @staticmethod
    def _operation_id(
        envelope: AuditMCPRequest,
        payload: Mapping[str, Any],
        response_request_id: str,
        *,
        label: str = "",
    ) -> str:
        """Resolve one closed operation ID without accepting explicit nulls."""

        candidate = payload.get("operation_id", response_request_id)
        if not isinstance(candidate, str) or not candidate.strip() or len(candidate) > 4_096:
            raise AuditValidationError(
                f"MCP {label + ' ' if label else ''}operation_id must be a non-empty bounded string"
            )
        if envelope.session_token in candidate:
            raise AuditValidationError(
                f"MCP {label + ' ' if label else ''}operation_id cannot contain session credentials"
            )
        return candidate

    @staticmethod
    def _next_arguments(envelope: AuditMCPRequest, payload: Mapping[str, Any]) -> tuple[str, bool]:
        """Validate mutation-sensitive Next arguments before service admission."""

        operation_id = AuditMCPDispatcher._operation_id(
            envelope,
            payload,
            envelope.request_id,
            label="next",
        )
        force_current = payload.get("force_current", False)
        if not isinstance(force_current, bool):
            raise AuditValidationError("MCP next force_current must be a boolean")
        return operation_id, force_current

    @staticmethod
    def _safe_request_id(request_id: str, session_token: str | None) -> str:
        """Redact bearer-containing correlation IDs before response serialization."""

        if isinstance(session_token, str) and session_token and session_token in request_id:
            return REDACTED_MCP_REQUEST_ID
        return request_id

    @staticmethod
    def _response(request_id: str, result: ServiceResult[Any]) -> AuditMCPResponse:
        return AuditMCPResponse(
            request_id,
            result.status,
            _json_safe(result.to_dict()),
            result.reason,
        )

    def dispatch(  # noqa: C901, PLR0912, PLR0915
        self, request: AuditMCPRequest | Mapping[str, Any]
    ) -> AuditMCPResponse:
        """Validate origin/token and dispatch one bounded operation."""

        try:
            envelope = self._request(request)
            response_request_id = self._safe_request_id(envelope.request_id, envelope.session_token)
            if envelope.origin not in self.allowed_origins:
                return AuditMCPResponse(
                    response_request_id, "denied", reason="origin is not allowed"
                )
            if not self.service.validate_session_token(envelope.session_id, envelope.session_token):
                return AuditMCPResponse(
                    response_request_id, "denied", reason="session token is invalid"
                )
            payload = envelope.payload
            context = self._context(payload)
            operation = envelope.operation
            common_operation_id = (
                response_request_id
                if operation == "next"
                else self._operation_id(envelope, payload, response_request_id)
            )
            common = {
                "context": context,
                "operation_id": common_operation_id,
                "token": envelope.session_token,
            }
            if operation == "read_campaign":
                result = self.service.read_campaign(
                    envelope.session_id,
                    config=payload.get("config"),
                    detector_ids=payload.get("detector_ids"),
                    repository=payload.get("repository"),
                    recipe_id=payload.get("recipe_id"),
                    **common,
                )
            elif operation == "read_episode":
                result = self.service.read_episode(
                    envelope.session_id, payload["episode_id"], **common
                )
            elif operation in {"read_metrics", "read_events", "read_geometry"}:
                method = getattr(self.service, operation)
                result = method(envelope.session_id, payload["episode_id"], **common)
            elif operation == "read_signals":
                result = self.service.read_signals(
                    envelope.session_id, payload.get("episode_id"), **common
                )
            elif operation in {"related_cases", "find_related_cases"}:
                result = self.service.related_cases(
                    envelope.session_id,
                    payload["episode_id"],
                    mode=payload.get("mode", "same_scenario_across_planners"),
                    limit=payload.get("limit", 20),
                    **common,
                )
            elif operation in {"read_queue", "queue"}:
                result = self.service.read_queue(
                    envelope.session_id, limit=payload.get("limit", 20), **common
                )
            elif operation == "next":
                operation_id, force_current = self._next_arguments(envelope, payload)
                result = self.service.next(
                    envelope.session_id,
                    context=context,
                    expected_context_revision=payload.get("expected_context_revision"),
                    expected_queue_state_revision=payload.get("expected_queue_state_revision"),
                    expected_queue_input_revision=payload.get("expected_queue_input_revision"),
                    force_current=force_current,
                    operation_id=operation_id,
                    token=envelope.session_token,
                )
            elif operation in {"read_coverage", "coverage"}:
                result = self.service.read_coverage(envelope.session_id, **common)
            elif operation == "run_native_diagnostic":
                allowed = {
                    "context",
                    "operation_id",
                    "intervention_id",
                    "robot_goal",
                    "activation_epsilon_m",
                    "deadline_s",
                }
                unknown = set(payload) - allowed
                if unknown:
                    raise AuditValidationError(
                        "native diagnostic tool contains forbidden fields: "
                        + ", ".join(sorted(unknown))
                    )
                intervention = {
                    key: payload[key]
                    for key in ("intervention_id", "robot_goal", "activation_epsilon_m")
                    if key in payload
                }
                result = self.service.run_native_diagnostic(
                    envelope.session_id,
                    intervention=intervention,
                    deadline_s=payload.get("deadline_s", 30.0),
                    **common,
                )
            elif operation == "materialize_selected":
                result = self.service.materialize_selected(
                    envelope.session_id,
                    output_root=payload["output_root"],
                    output_directory=payload.get("output_directory"),
                    render_config=payload.get("render_config"),
                    **common,
                )
            elif operation == "read_detector_rule_proposal":
                allowed = {"context", "operation_id", "proposal_id"}
                unknown = set(payload) - allowed
                if unknown:
                    raise AuditValidationError(
                        "detector rule proposal read contains forbidden fields: "
                        + ", ".join(sorted(unknown))
                    )
                proposal_id = payload.get("proposal_id")
                if not isinstance(proposal_id, str) or not proposal_id.strip():
                    raise AuditValidationError("MCP proposal read requires proposal_id")
                result = self.service.read_detector_rule_proposal(
                    envelope.session_id,
                    proposal_id,
                    **common,
                )
            elif operation == "write_detector_rule_proposal":
                allowed = {
                    "context",
                    "operation_id",
                    "record",
                    "expected_revision",
                    "expected_source_revision",
                }
                unknown = set(payload) - allowed
                if unknown:
                    raise AuditValidationError(
                        "detector rule proposal write contains forbidden fields: "
                        + ", ".join(sorted(unknown))
                    )
                record_payload = payload.get("record")
                if not isinstance(record_payload, Mapping):
                    raise AuditValidationError(
                        "MCP proposal write requires a typed DetectorRuleProposal mapping"
                    )
                result = self.service.write_detector_rule_proposal(
                    envelope.session_id,
                    record_from_dict(record_payload),
                    expected_revision=payload.get("expected_revision"),
                    expected_source_revision=payload.get("expected_source_revision"),
                    **common,
                )
            elif operation in {"write_annotation", "write_reference", "write_finding"}:
                record_payload = payload.get(
                    "record", payload.get(operation.removeprefix("write_"))
                )
                if not isinstance(record_payload, Mapping):
                    raise AuditValidationError("MCP write requires a typed BA-03 record mapping")
                method = getattr(self.service, operation)
                result = method(
                    envelope.session_id,
                    record_from_dict(record_payload),
                    expected_revision=payload.get("expected_revision"),
                    expected_source_revision=payload.get("expected_source_revision"),
                    **common,
                )
            elif operation == "sync_finding":
                result = self.service.sync_finding(
                    envelope.session_id,
                    finding_id=payload["finding_id"],
                    repository=payload["repository"],
                    evidence=payload.get("evidence"),
                    expected_finding_revision=payload.get("expected_finding_revision"),
                    expected_source_revision=payload.get("expected_source_revision"),
                    retry_ambiguous=payload.get("retry_ambiguous", False),
                    worker_id=payload.get("worker_id", ""),
                    **common,
                )
            elif operation in {"cancel", "cancel_session", "kill_switch"}:
                result = self.service.kill_switch(
                    envelope.session_id,
                    reason=payload.get("reason", "MCP cancellation"),
                    operation_id=common["operation_id"],
                    token=envelope.session_token,
                )
            else:
                return AuditMCPResponse(
                    response_request_id, "unavailable", reason="unknown audit operation"
                )
            return self._response(response_request_id, result)
        except (AuditServiceError, KeyError, TypeError, ValueError) as exc:
            if isinstance(request, AuditMCPRequest):
                request_id = self._safe_request_id(request.request_id, request.session_token)
            elif isinstance(request, Mapping):
                raw_request_id = str(request.get("request_id", "unknown"))
                raw_token = request.get("session_token")
                request_id = self._safe_request_id(
                    raw_request_id, raw_token if isinstance(raw_token, str) else None
                )
            else:
                request_id = "unknown"
            return AuditMCPResponse(request_id, "failed", reason=str(exc))

    handle = dispatch


class FakeMCPTransport:
    """Offline transport used by tests and local client development."""

    def __init__(self, dispatcher: AuditMCPDispatcher) -> None:
        """Bind the fake transport to the same dispatcher instance."""

        self.dispatcher = dispatcher
        self.requests: list[AuditMCPRequest] = []

    def send(self, request: AuditMCPRequest | Mapping[str, Any]) -> AuditMCPResponse:
        """Dispatch one request without opening a socket or invoking an SDK."""

        envelope = (
            request
            if isinstance(request, AuditMCPRequest)
            else AuditMCPRequest.from_mapping(request)
        )
        self.requests.append(envelope)
        return self.dispatcher.dispatch(envelope)

    call = send


OfflineFakeTransport = FakeMCPTransport


__all__ = [
    "AUDIT_MCP_REQUEST_SCHEMA_VERSION",
    "AUDIT_MCP_RESPONSE_SCHEMA_VERSION",
    "LOOPBACK_ORIGINS",
    "AuditMCPDispatcher",
    "AuditMCPRequest",
    "AuditMCPResponse",
    "FakeMCPTransport",
    "OfflineFakeTransport",
]
