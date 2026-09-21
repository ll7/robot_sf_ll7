# ruff: noqa: DOC201

"""Capability-gated Codex client/session protocol for audit investigations.

The client deliberately does not contain a model table and does not call the
tool-free ``review_ai`` narrative adapter.  A route is usable only when an
installed-capability inspector returns a complete, discovered route receipt.
Offline tests can inject :class:`FakeCodexProvider`; a fake receipt is never
reported as a live route.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import secrets
import shutil
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Protocol

from robot_sf.analysis_workbench.audit_authority import (
    AuditAuthorityError,
    AuthorityOperationConflict,
)
from robot_sf.analysis_workbench.audit_service import (
    AuditBudgetExceeded,
    AuditCancelled,
    AuditContextConflict,
    AuditSelectionContext,
    AuditService,
    AuditServiceError,
    AuditSession,
    CapabilityUnavailable,
    CodexOperationLease,
    CodexOperationSnapshot,
    CodexSessionSnapshot,
    ServiceResult,
)

AUDIT_CODEX_SCHEMA_VERSION = "audit-codex.v1"
AUDIT_CODEX_CAPABILITY_SCHEMA_VERSION = "audit-codex-capability.v1"
AUDIT_CODEX_ROUTE_SCHEMA_VERSION = "audit-codex-route.v1"
AUDIT_CODEX_EVENT_SCHEMA_VERSION = "audit-codex-event.v1"
CODEX_STATUSES = ("complete", "unavailable", "failed", "cancelled", "conflict")
MAX_PROMPT_CHARS = 32_768
MAX_EVENTS = 256
MAX_EVIDENCE = 256


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _text(value: Any, *, name: str, limit: int = 4_096) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")
    if len(value) > limit:
        raise ValueError(f"{name} exceeds {limit} characters")
    return value


def _optional(value: Any, *, name: str, limit: int = 4_096) -> str:
    if value is None:
        return ""
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"{name} must be a bounded string")
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class CodexRouteReceipt:
    """Exact route metadata returned by installed-capability discovery."""

    route_id: str
    provider: str
    model_id: str
    client_version: str
    protocol: str
    discovered: bool = True
    capability_digest: str = ""
    source: str = "installed-capability-inspection"
    schema_version: str = AUDIT_CODEX_ROUTE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Reject incomplete or guessed route identities."""

        if self.schema_version != AUDIT_CODEX_ROUTE_SCHEMA_VERSION:
            raise ValueError("unsupported Codex route schema")
        for name in ("route_id", "provider", "model_id", "client_version", "protocol"):
            _text(getattr(self, name), name=name)
        if not isinstance(self.discovered, bool):
            raise ValueError("Codex route discovered must be boolean")
        if not self.discovered:
            raise ValueError("a Codex route must be discovered before use")
        object.__setattr__(self, "source", _text(self.source, name="route.source"))
        digest = self.capability_digest or _digest(self.to_dict(include_digest=False))
        object.__setattr__(self, "capability_digest", digest)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CodexRouteReceipt:
        """Build a route only from complete inspector-provided metadata."""

        if not isinstance(value, Mapping):
            raise ValueError("Codex route must be a mapping")
        allowed = {
            "schema_version",
            "route_id",
            "provider",
            "model_id",
            "client_version",
            "protocol",
            "discovered",
            "capability_digest",
            "source",
        }
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"Codex route contains unknown fields: {', '.join(sorted(unknown))}")
        if "schema_version" not in value:
            raise ValueError("Codex route schema_version is required and versioned")
        return cls(**dict(value))

    def to_dict(self, *, include_digest: bool = True) -> dict[str, Any]:
        """Return a route receipt suitable for audit evidence."""

        result = {
            "schema_version": self.schema_version,
            "route_id": self.route_id,
            "provider": self.provider,
            "model_id": self.model_id,
            "client_version": self.client_version,
            "protocol": self.protocol,
            "discovered": self.discovered,
            "source": self.source,
        }
        if include_digest:
            result["capability_digest"] = self.capability_digest
        return result


@dataclass(frozen=True, slots=True)
class CodexCapabilityInspection:
    """Installed capability result; unavailable is a first-class outcome."""

    status: str
    routes: tuple[CodexRouteReceipt, ...] = ()
    inspector: str = ""
    reason: str = ""
    installed: bool = False
    schema_version: str = AUDIT_CODEX_CAPABILITY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate route status and preserve explicit inspection limits."""

        if self.schema_version != AUDIT_CODEX_CAPABILITY_SCHEMA_VERSION:
            raise ValueError("unsupported Codex capability schema")
        if self.status not in {"available", "unavailable"}:
            raise ValueError("capability status must be available or unavailable")
        if not isinstance(self.installed, bool):
            raise ValueError("capability installed must be boolean")
        routes = tuple(self.routes)
        if any(not isinstance(route, CodexRouteReceipt) for route in routes):
            raise ValueError("capability routes must be typed receipts")
        if self.status == "available" and not routes:
            raise ValueError("available capability inspection requires a discovered route")
        object.__setattr__(self, "routes", routes)
        object.__setattr__(self, "inspector", _optional(self.inspector, name="inspector"))
        object.__setattr__(self, "reason", _optional(self.reason, name="reason", limit=2_048))

    @classmethod
    def unavailable(cls, reason: str, *, inspector: str = "") -> CodexCapabilityInspection:
        """Build an explicit unavailable result without guessing a route."""

        return cls("unavailable", reason=reason, inspector=inspector)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> CodexCapabilityInspection:
        """Parse installed capability metadata from an injected inspector."""

        if not isinstance(value, Mapping):
            raise ValueError("capability inspection must be a mapping")
        allowed = {"schema_version", "status", "routes", "inspector", "reason", "installed"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(
                f"capability inspection contains unknown fields: {', '.join(sorted(unknown))}"
            )
        routes_payload = value.get("routes", ())
        if not isinstance(routes_payload, (tuple, list)):
            raise ValueError("capability routes must be an array")
        routes = tuple(
            CodexRouteReceipt.from_mapping(item) if isinstance(item, Mapping) else item
            for item in routes_payload
        )
        return cls(
            routes=routes,
            **{
                key: value[key]
                for key in ("schema_version", "status", "inspector", "reason", "installed")
                if key in value
            },
        )

    def choose(self, route_id: str | None = None) -> CodexRouteReceipt | None:
        """Select only one of the routes actually discovered by the inspector."""

        if self.status != "available":
            return None
        if route_id is None:
            return self.routes[0] if len(self.routes) == 1 else None
        return next((route for route in self.routes if route.route_id == route_id), None)

    def to_dict(self) -> dict[str, Any]:
        """Return capability evidence."""

        return {
            "schema_version": self.schema_version,
            "status": self.status,
            "routes": [route.to_dict() for route in self.routes],
            "inspector": self.inspector,
            "reason": self.reason,
            "installed": self.installed,
        }


def inspect_installed_capabilities(
    inspector: Callable[[], CodexCapabilityInspection | Mapping[str, Any]] | Any | None = None,
) -> CodexCapabilityInspection:
    """Inspect an installed client through an explicit adapter.

    Without an injected inspector this function intentionally returns
    ``unavailable``.  Discovering an executable named ``codex`` is not enough
    to establish a supported App Server route or model identity.
    """

    if inspector is not None:
        try:
            raw = inspector() if callable(inspector) else inspector.inspect()
            if isinstance(raw, CodexCapabilityInspection):
                return raw
            return CodexCapabilityInspection.from_mapping(raw)
        except (OSError, TypeError, ValueError, AttributeError) as exc:
            return CodexCapabilityInspection.unavailable(f"capability inspection failed: {exc}")
    executable = shutil.which("codex")
    if executable:
        return CodexCapabilityInspection.unavailable(
            "installed executable found but supported route metadata was not discovered",
            inspector="PATH",
        )
    return CodexCapabilityInspection.unavailable("Codex client/App Server is not installed")


@dataclass(frozen=True, slots=True)
class CodexEvidenceRef:
    """Clickable, source-bound evidence reference without local private paths."""

    evidence_id: str
    artifact_id: str
    locator: str
    source_digest: str
    source_revision: int | str
    episode_id: str = ""
    context_revision: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Return evidence identity and locator."""

        return {
            "evidence_id": self.evidence_id,
            "artifact_id": self.artifact_id,
            "locator": self.locator,
            "source_digest": self.source_digest,
            "source_revision": self.source_revision,
            "episode_id": self.episode_id or None,
            "context_revision": self.context_revision,
        }


@dataclass(frozen=True, slots=True)
class CodexActivityEvent:
    """Conversation/activity event linked to an audit operation."""

    event_id: str
    kind: str
    status: str
    message: str = ""
    evidence_ids: tuple[str, ...] = ()
    operation_id: str = ""
    created_at: str = field(default_factory=_now)
    schema_version: str = AUDIT_CODEX_EVENT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return one bounded event."""

        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "kind": self.kind,
            "status": self.status,
            "message": self.message,
            "evidence_ids": list(self.evidence_ids),
            "operation_id": self.operation_id,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class CodexOperationReceipt:
    """Receipt for start/resume/reconnect/cancel activity."""

    operation_id: str
    action: str
    status: str
    session_id: str
    route: CodexRouteReceipt | None
    context_revision: int
    source_revision: int | str
    source_digest: str = ""
    evidence_ids: tuple[str, ...] = ()
    provider_session_id: str = ""
    reason: str = ""
    replayed: bool = False
    request_digest: str = ""
    usage: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        """Return route/evidence/usage receipt metadata."""

        return {
            "schema_version": AUDIT_CODEX_SCHEMA_VERSION,
            "operation_id": self.operation_id,
            "action": self.action,
            "status": self.status,
            "session_id": self.session_id,
            "route": self.route.to_dict() if self.route is not None else None,
            "context_revision": self.context_revision,
            "source_revision": self.source_revision,
            "source_digest": self.source_digest,
            "evidence_ids": list(self.evidence_ids),
            "provider_session_id": self.provider_session_id,
            "reason": self.reason,
            "replayed": self.replayed,
            "request_digest": self.request_digest,
            "usage": dict(self.usage),
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class CodexResult:
    """Typed Codex client result."""

    status: str
    session: CodexSession | None = None
    receipt: CodexOperationReceipt | None = None
    events: tuple[CodexActivityEvent, ...] = ()
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe result envelope."""

        return {
            "schema_version": AUDIT_CODEX_SCHEMA_VERSION,
            "status": self.status,
            "session": self.session.to_dict() if self.session is not None else None,
            "receipt": self.receipt.to_dict() if self.receipt is not None else None,
            "events": [event.to_dict() for event in self.events],
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class CodexSession:
    """Agent session bound to exact service context/evidence/route."""

    session_id: str
    audit_session_id: str
    context: AuditSelectionContext
    route: CodexRouteReceipt
    evidence: tuple[CodexEvidenceRef, ...]
    provider_session_id: str
    source_digest: str = ""
    source_revision: int | str = 0
    status: str = "active"
    events: tuple[CodexActivityEvent, ...] = ()
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        """Return a session receipt with no provider credentials."""

        return {
            "schema_version": AUDIT_CODEX_SCHEMA_VERSION,
            "session_id": self.session_id,
            "audit_session_id": self.audit_session_id,
            "context": self.context.to_dict(),
            "route": self.route.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "provider_session_id": self.provider_session_id,
            "source_digest": self.source_digest,
            "source_revision": self.source_revision,
            "status": self.status,
            "events": [event.to_dict() for event in self.events],
            "created_at": self.created_at,
        }


class CodexProvider(Protocol):
    """Minimal supported-provider seam used by the client."""

    supports_tools: bool

    def start(
        self, *, route: CodexRouteReceipt, prompt: str, evidence: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        """Start a provider session with the shared audit tools."""

    def resume(
        self,
        *,
        provider_session_id: str,
        route: CodexRouteReceipt,
        evidence: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Resume one provider session."""

    def reconnect(self, *, provider_session_id: str, route: CodexRouteReceipt) -> Mapping[str, Any]:
        """Reconnect a provider transport."""

    def cancel(self, *, provider_session_id: str, route: CodexRouteReceipt) -> Mapping[str, Any]:
        """Cancel provider work."""


class FakeCodexProvider:
    """Deterministic offline provider; its route must still come from inspection."""

    supports_tools = True

    def __init__(self, *, response: Mapping[str, Any] | None = None) -> None:
        """Configure bounded fake response data."""

        self.response = dict(response or {})
        self.calls: list[dict[str, Any]] = []

    def start(
        self, *, route: CodexRouteReceipt, prompt: str, evidence: Sequence[Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        """Record a fake start and return a stable provider session ID."""

        self.calls.append(
            {
                "action": "start",
                "route": route.to_dict(),
                "prompt": prompt,
                "evidence": list(evidence),
            }
        )
        result = dict(self.response)
        result.setdefault("provider_session_id", f"fake-{secrets.token_hex(8)}")
        result.setdefault("message", "offline fake provider response")
        result.setdefault("usage", {"tokens": 0, "compute": 0.0})
        return result

    def resume(
        self,
        *,
        provider_session_id: str,
        route: CodexRouteReceipt,
        evidence: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Record a fake resume."""

        self.calls.append(
            {
                "action": "resume",
                "provider_session_id": provider_session_id,
                "route": route.to_dict(),
                "evidence": list(evidence),
            }
        )
        return {
            "provider_session_id": provider_session_id,
            "message": "offline fake provider resumed",
            "usage": {"tokens": 0, "compute": 0.0},
        }

    def reconnect(self, *, provider_session_id: str, route: CodexRouteReceipt) -> Mapping[str, Any]:
        """Record a fake reconnect."""

        self.calls.append(
            {
                "action": "reconnect",
                "provider_session_id": provider_session_id,
                "route": route.to_dict(),
            }
        )
        return {
            "provider_session_id": provider_session_id,
            "message": "offline fake provider reconnected",
            "usage": {"tokens": 0, "compute": 0.0, "issue_writes": 0},
        }

    def cancel(self, *, provider_session_id: str, route: CodexRouteReceipt) -> Mapping[str, Any]:
        """Record a fake cancellation."""

        self.calls.append(
            {
                "action": "cancel",
                "provider_session_id": provider_session_id,
                "route": route.to_dict(),
            }
        )
        return {
            "provider_session_id": provider_session_id,
            "message": "offline fake provider cancelled",
            "usage": {"tokens": 0, "compute": 0.0, "issue_writes": 0},
        }


class AuditCodexClient:
    """Bind an installed route/provider to one audit service session."""

    def __init__(
        self,
        service: AuditService,
        *,
        inspector: Callable[[], Any] | Any | None = None,
        provider: CodexProvider | None = None,
    ) -> None:
        """Configure capability discovery and an optional provider implementation."""

        self.service = service
        self.inspector = inspector
        self.provider = provider
        self._sessions: dict[str, CodexSession] = {}
        self._audit_handles: dict[str, AuditSession] = {}
        self._cancel_handle_aliases: dict[str, CodexSession] = {}
        self._receipts: dict[str, CodexOperationReceipt] = {}
        self._request_digests: dict[str, str] = {}
        self._lock = threading.RLock()

    @staticmethod
    def _policy_digest(audit: AuditSession) -> str:
        """Return the exact policy identity bound into authority snapshots."""

        return _digest(audit.policy.to_dict())

    @staticmethod
    def _source_ref(audit: AuditSession) -> dict[str, str] | None:
        """Serialize the complete source identity without dropping fields."""

        if audit.source_ref is None:
            return None
        return {
            name: str(getattr(audit.source_ref, name))
            for name in audit.source_ref.__dataclass_fields__
        }

    @staticmethod
    def _valid_usage(result: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
        """Return usage only when all provider meter fields are present."""

        if not isinstance(result, Mapping) or not isinstance(result.get("usage"), Mapping):
            return None
        usage = result["usage"]
        tokens = usage.get("tokens")
        compute = usage.get("compute")
        issue_writes = usage.get("issue_writes", 0)
        if (
            isinstance(tokens, bool)
            or not isinstance(tokens, int)
            or tokens < 0
            or isinstance(compute, bool)
            or not isinstance(compute, (int, float))
            or not math.isfinite(float(compute))
            or compute < 0
            or isinstance(issue_writes, bool)
            or not isinstance(issue_writes, int)
            or issue_writes < 0
        ):
            return None
        return {"tokens": tokens, "compute": float(compute), "issue_writes": issue_writes}

    @staticmethod
    def _authority_session_snapshot(
        session: CodexSession,
        audit: AuditSession,
        *,
        updated_at: str | None = None,
        last_operation_id: str,
    ) -> CodexSessionSnapshot:
        """Build the token-free authority snapshot sent to the service."""

        return CodexSessionSnapshot(
            codex_session_id=session.session_id,
            audit_session_id=session.audit_session_id,
            actor=audit.actor.to_dict(),
            policy_id=audit.policy.policy_id,
            policy_revision=audit.policy.policy_revision,
            policy_digest=AuditCodexClient._policy_digest(audit),
            context=session.context.to_dict(),
            source_ref=AuditCodexClient._source_ref(audit),
            source_digest=session.source_digest,
            source_revision=session.source_revision,
            route=session.route.to_dict(),
            evidence=tuple(item.to_dict() for item in session.evidence),
            provider_session_id=session.provider_session_id,
            status="cancelled" if session.status == "cancelled" else "active",
            created_at=session.created_at,
            updated_at=updated_at or _now(),
            last_operation_id=last_operation_id,
        )

    @staticmethod
    def _route_from_snapshot(snapshot: CodexSessionSnapshot) -> CodexRouteReceipt:
        """Hydrate a route only after service-side source/actor checks."""

        return CodexRouteReceipt.from_mapping(snapshot.route)

    @staticmethod
    def _session_from_snapshot(snapshot: CodexSessionSnapshot) -> CodexSession:
        """Hydrate a local session from an authenticated authority snapshot."""

        evidence = tuple(
            CodexEvidenceRef(
                evidence_id=str(item["evidence_id"]),
                artifact_id=str(item["artifact_id"]),
                locator=str(item["locator"]),
                source_digest=str(item["source_digest"]),
                source_revision=item["source_revision"],
                episode_id=str(item.get("episode_id") or ""),
                context_revision=int(item["context_revision"]),
            )
            for item in snapshot.evidence
        )
        return CodexSession(
            session_id=snapshot.codex_session_id,
            audit_session_id=snapshot.audit_session_id,
            context=AuditSelectionContext.from_mapping(snapshot.context),
            route=AuditCodexClient._route_from_snapshot(snapshot),
            evidence=evidence,
            provider_session_id=snapshot.provider_session_id,
            source_digest=snapshot.source_digest,
            source_revision=snapshot.source_revision,
            status=snapshot.status,
            created_at=snapshot.created_at,
        )

    def _rediscovered_route(self, snapshot: CodexSessionSnapshot) -> CodexRouteReceipt:
        """Require current capability discovery to reproduce the stored route."""

        route = self.inspect_capabilities().choose(str(snapshot.route.get("route_id", "")))
        if route is None:
            raise CapabilityUnavailable("persisted Codex route is unavailable or ambiguous")
        if route.to_dict() != dict(snapshot.route):
            raise CapabilityUnavailable("persisted Codex route metadata changed")
        return route

    def _receipt_from_lease(
        self,
        lease: CodexOperationLease,
        session: CodexSession | None,
        *,
        replayed: bool | None = None,
    ) -> CodexOperationReceipt | None:
        """Rebuild a public receipt from durable operation state."""

        if session is None or lease.operation is None:
            return None
        operation = lease.operation
        return self._receipt(
            session,
            operation.action,
            operation.result_status or lease.status,
            operation.operation_id,
            reason=lease.reason,
            usage=operation.result.get("usage") if isinstance(operation.result, Mapping) else None,
            replayed=lease.replayed if replayed is None else replayed,
            request_digest=operation.request_digest,
        )

    def _recovered_result(
        self, lease: CodexOperationLease, *, fallback_session: CodexSession | None = None
    ) -> CodexResult:
        """Map a terminal/ambiguous durable lease into a client result."""

        session = fallback_session
        if session is None and lease.session is not None:
            with self._lock:
                session = self._sessions.get(lease.session.codex_session_id)
            if session is None:
                session = self._session_from_snapshot(lease.session)
        if lease.status == "ambiguous":
            return CodexResult("unavailable", session=session, reason=lease.reason)
        status = lease.operation.result_status if lease.operation is not None else lease.status
        status = status or "unavailable"
        receipt = self._receipt_from_lease(lease, session)
        if session is not None:
            with self._lock:
                self._sessions[session.session_id] = session
        return CodexResult(status, session=session, receipt=receipt, reason=lease.reason)

    @staticmethod
    def _require_app_server_reservation(
        provider: CodexProvider,
        *,
        token_budget: int,
        compute_budget: float,
        action: str,
    ) -> None:
        """Apply the App Server's explicit conservative admission floor."""

        if not AuditCodexClient._is_app_server_provider(provider):
            return
        try:
            compute = float(compute_budget)
        except (TypeError, ValueError, OverflowError):
            compute = math.nan
        if not math.isfinite(compute) or compute < 1.0:
            raise ValueError(f"App Server {action} requires a finite reserved_compute >= 1.0")
        if isinstance(token_budget, bool) or not isinstance(token_budget, int) or token_budget <= 0:
            raise ValueError(
                f"App Server {action} requires a finite positive reserved_tokens value"
            )

    def _finish_durable(
        self,
        audit: AuditSession,
        *,
        lease: CodexOperationLease,
        provider: CodexProvider,
        request_digest: str,
        result_status: str,
        provider_result: Mapping[str, Any],
        session: CodexSession | None,
        provider_session_id: str,
    ) -> CodexResult:
        """Publish a measured provider result through the authority CAS."""

        action = lease.operation.action if lease.operation is not None else ""
        provider_result, result_status, session = self._durable_provider_result_for_finish(
            provider,
            action=action,
            provider_result=provider_result,
            result_status=result_status,
            session=session,
            reserved_tokens=lease.reserved_tokens,
            reserved_compute=lease.reserved_compute,
            reserved_issue_writes=lease.reserved_issue_writes,
        )
        usage = self._valid_usage(provider_result)
        if usage is None:
            return CodexResult(
                "unavailable",
                session=session,
                reason="provider usage is missing or malformed; Codex operation is ambiguous",
            )
        snapshot = (
            self._authority_session_snapshot(
                session,
                audit,
                last_operation_id=lease.operation.operation_id if lease.operation else "",
            )
            if session is not None
            else None
        )
        try:
            finished = self.service.finish_codex_operation(
                audit,
                operation_id=lease.operation.operation_id if lease.operation else "",
                request_digest=request_digest,
                result_status=result_status,
                result=provider_result,
                provider_session_id=provider_session_id,
                usage=usage,
                session_snapshot=snapshot,
                route_digest=lease.operation.route_digest if lease.operation else None,
            )
        except AuthorityOperationConflict as exc:
            return CodexResult(
                "conflict",
                session=session,
                reason=f"Codex operation finish conflicts with authority state: {exc}",
            )
        except (AuditAuthorityError, AuditServiceError, TypeError, ValueError, KeyError) as exc:
            # Once provider work was admitted, a failed finish cannot be
            # converted into a retry: its inflight row remains ambiguous.
            return CodexResult(
                "unavailable",
                session=session,
                reason=f"Codex operation finish is ambiguous: {exc}",
            )
        if finished.status == "replay" or finished.replayed:
            return self._recovered_result(finished, fallback_session=session)
        if finished.status != "replay" and finished.operation is not None:
            updated = session
            if finished.session is not None:
                updated = self._session_from_snapshot(finished.session)
                if session is not None:
                    updated = self._replace(updated, events=session.events)
            if updated is not None:
                with self._lock:
                    self._sessions[updated.session_id] = updated
            receipt = self._receipt_from_lease(finished, updated, replayed=False)
            final_status = finished.operation.result_status or result_status
            return CodexResult(final_status, updated, receipt, reason=finished.reason)
        final_status = (
            finished.operation.result_status
            if finished.operation is not None and finished.operation.result_status
            else result_status
        )
        return CodexResult(final_status, session=session, reason=finished.reason)

    def inspect_capabilities(self) -> CodexCapabilityInspection:
        """Return the current installed-capability evidence."""

        return inspect_installed_capabilities(self.inspector)

    def recover_session(self, codex_session_id: str, *, audit_token: str) -> CodexResult:
        """Re-authenticate and hydrate one durable Codex session after restart."""

        try:
            snapshot = self.service.load_codex_session(codex_session_id, audit_token=audit_token)
            route = self._rediscovered_route(snapshot)
            # ``load_codex_session`` already performs the source/policy/evidence
            # checks.  Reconnect again only to recover the opaque local handle;
            # the token is deliberately supplied explicitly after restart.
            audit = self.service.reconnect_session(snapshot.audit_session_id, audit_token)
            session = self._session_from_snapshot(snapshot)
            if session.route != route:
                raise CapabilityUnavailable("persisted Codex route changed during recovery")
            with self._lock:
                self._sessions[session.session_id] = session
                self._audit_handles[session.session_id] = audit
            return CodexResult(
                "cancelled" if session.status == "cancelled" else "complete",
                session=session,
            )
        except AuditCancelled as exc:
            return CodexResult("cancelled", reason=str(exc))
        except CapabilityUnavailable as exc:
            return CodexResult("unavailable", reason=str(exc))
        except (AuditContextConflict, AuditServiceError, TypeError, ValueError, KeyError) as exc:
            return CodexResult("conflict", reason=str(exc))

    reconnect_session = recover_session

    def _evidence(
        self, audit_session: AuditSession, context: AuditSelectionContext
    ) -> tuple[CodexEvidenceRef, ...]:
        values = self.service.evidence_refs(audit_session, context=context)
        result = []
        for index, item in enumerate(values[:MAX_EVIDENCE]):
            result.append(
                CodexEvidenceRef(
                    evidence_id=f"evidence-{index + 1}",
                    artifact_id=str(item.get("artifact_id", "audit-source")),
                    locator=str(item.get("locator", "")),
                    source_digest=str(item.get("source_digest", audit_session.source_digest)),
                    source_revision=item.get("source_revision", audit_session.source_revision),
                    episode_id=str(item.get("episode_id") or ""),
                    context_revision=context.context_revision,
                )
            )
        return tuple(result)

    def _receipt(
        self,
        session: CodexSession,
        action: str,
        status: str,
        operation_id: str,
        *,
        reason: str = "",
        usage: Mapping[str, Any] | None = None,
        replayed: bool = False,
        request_digest: str = "",
    ) -> CodexOperationReceipt:
        receipt = CodexOperationReceipt(
            operation_id=operation_id,
            action=action,
            status=status,
            session_id=session.session_id,
            route=session.route,
            context_revision=session.context.context_revision,
            source_revision=session.source_revision,
            source_digest=session.source_digest,
            evidence_ids=tuple(item.evidence_id for item in session.evidence),
            provider_session_id=session.provider_session_id,
            reason=reason,
            replayed=replayed,
            request_digest=request_digest,
            usage=dict(usage or {}),
        )
        with self._lock:
            self._receipts[operation_id] = receipt
            self._request_digests[operation_id] = request_digest
        return receipt

    def _replayed_operation(
        self, operation_id: str, request_digest: str, *, session: CodexSession | None = None
    ) -> CodexResult | None:
        """Reject conflicting Codex operation replays before provider work."""

        with self._lock:
            receipt = self._receipts.get(operation_id)
            if receipt is None:
                return None
            if self._request_digests.get(operation_id, "") != request_digest:
                return CodexResult(
                    "conflict",
                    session=session,
                    reason="Codex operation ID is bound to another request",
                )
            known = self._sessions.get(receipt.session_id, session)
        return CodexResult(
            receipt.status,
            session=known,
            receipt=replace(receipt, replayed=True),
            reason=receipt.reason,
        )

    @staticmethod
    def _provider_usage(
        result: Mapping[str, Any] | None,
    ) -> tuple[int | None, float | None, int | None]:
        """Return finite usage fields, or ``None`` fields for fail-closed metering."""

        if not isinstance(result, Mapping) or not isinstance(result.get("usage"), Mapping):
            return None, None, None
        usage = result["usage"]
        return usage.get("tokens"), usage.get("compute"), usage.get("issue_writes", 0)

    def _settle_provider_usage(
        self,
        audit: AuditSession,
        *,
        operation_id: str,
        reservation_id: str,
        reservation_operation_id: str,
        reserved_tokens: int,
        reserved_compute: float,
        provider_result: Mapping[str, Any] | None,
    ) -> ServiceResult[Any]:
        """Settle one provider action with actual usage before promotion."""

        actual_tokens, actual_compute, actual_issue_writes = self._provider_usage(provider_result)
        return self.service.settle_budget(
            audit,
            reserved_tokens=reserved_tokens,
            reserved_compute=reserved_compute,
            actual_tokens=actual_tokens,
            actual_compute=actual_compute,
            actual_issue_writes=actual_issue_writes,
            reservation_id=reservation_id,
            reservation_operation_id=reservation_operation_id,
            operation_id=f"{operation_id}-meter",
        )

    @staticmethod
    def _reservation_id(result: ServiceResult[Any]) -> str | None:
        """Extract a typed opaque reservation capability from a service result."""

        if not isinstance(result.value, Mapping):
            return None
        value = result.value.get("reservation_id")
        return value if isinstance(value, str) and value else None

    def _release_lost_reservation_reply(
        self,
        audit: AuditSession,
        *,
        operation_id: str,
        reservation_operation_id: str,
    ) -> bool:
        """Settle a durable reservation to zero when no provider call was sent."""

        recovered = self.service.recover_budget_reservation(
            audit, operation_id=reservation_operation_id
        )
        if not isinstance(recovered, Mapping):
            return False
        reservation_id = recovered.get("reservation_id")
        reserved = recovered.get("reserved")
        if not isinstance(reservation_id, str) or not isinstance(reserved, Mapping):
            return False
        tokens = reserved.get("tokens")
        compute = reserved.get("compute")
        issue_writes = reserved.get("issue_writes")
        if (
            isinstance(tokens, bool)
            or not isinstance(tokens, int)
            or tokens < 0
            or isinstance(compute, bool)
            or not isinstance(compute, (int, float))
            or not math.isfinite(float(compute))
            or compute < 0
            or issue_writes != 0
        ):
            return False
        settled = self._settle_provider_usage(
            audit,
            operation_id=operation_id,
            reservation_id=reservation_id,
            reservation_operation_id=reservation_operation_id,
            reserved_tokens=tokens,
            reserved_compute=float(compute),
            provider_result={"usage": {"tokens": 0, "compute": 0.0, "issue_writes": 0}},
        )
        return settled.ok

    @staticmethod
    def _is_app_server_provider(provider: CodexProvider) -> bool:
        """Identify the experimental App Server without widening generic providers."""

        try:
            from robot_sf.analysis_workbench.audit_codex_app_server import (  # noqa: PLC0415
                CodexAppServerProvider,
            )
        except ImportError:
            return False
        return isinstance(provider, CodexAppServerProvider)

    @staticmethod
    def _provider_reserved_tokens(
        provider: CodexProvider,
        reservation: ServiceResult[Any],
        *,
        action: str,
    ) -> int | None:
        """Require a finite positive token reservation for a live model turn."""

        if not AuditCodexClient._is_app_server_provider(provider):
            return None
        value = reservation.value
        reserved = value.get("reserved") if isinstance(value, Mapping) else None
        raw_tokens = reserved.get("tokens") if isinstance(reserved, Mapping) else None
        if isinstance(raw_tokens, bool) or not isinstance(raw_tokens, int) or raw_tokens <= 0:
            raise ValueError(
                f"App Server {action} requires a finite positive reserved_tokens value"
            )
        return raw_tokens

    @staticmethod
    def _provider_reserved_compute(
        provider: CodexProvider,
        reservation: ServiceResult[Any] | CodexOperationLease,
        *,
        action: str,
    ) -> float | None:
        """Return an App Server turn charge from the durable service reservation.

        The ordinary ``CodexProvider`` contract intentionally remains unchanged:
        generic providers do not receive provider-specific keyword arguments.  The
        experimental App Server is the one narrow exception, and only when its
        concrete lifecycle method advertises the explicit ``reserved_compute``
        parameter.  The latter keeps older test adapters/subclasses that own their
        own pre-admission seam compatible while preventing an accidental kwarg
        change for generic providers.
        """

        try:
            from robot_sf.analysis_workbench.audit_codex_app_server import (  # noqa: PLC0415
                UNMEASURED_TURN_COMPUTE_CHARGE,
            )
        except ImportError:
            return None
        if not AuditCodexClient._is_app_server_provider(provider):
            return None
        try:
            method = getattr(type(provider), action)
            parameters = inspect.signature(method).parameters
        except (AttributeError, TypeError, ValueError):
            parameters = {}

        if isinstance(reservation, CodexOperationLease):
            raw_compute = reservation.reserved_compute
        else:
            value = reservation.value
            reserved = value.get("reserved") if isinstance(value, Mapping) else None
            raw_compute = reserved.get("compute") if isinstance(reserved, Mapping) else None
        try:
            compute = float(raw_compute)
        except (OverflowError, TypeError, ValueError):
            compute = math.nan
        if (
            isinstance(raw_compute, bool)
            or not isinstance(raw_compute, (int, float))
            or not math.isfinite(compute)
            or compute < float(UNMEASURED_TURN_COMPUTE_CHARGE)
        ):
            raise ValueError(
                f"App Server {action} requires a finite reserved_compute >= "
                f"{UNMEASURED_TURN_COMPUTE_CHARGE}"
            )
        return compute if "reserved_compute" in parameters else None

    def _durable_admission_capability_valid(  # noqa: C901
        self,
        lease: CodexOperationLease,
        *,
        provider: CodexProvider,
        expected: Mapping[str, Any],
    ) -> bool:
        """Require the authority lease needed before calling a provider.

        ``begin_codex_operation`` commits the reservation before returning.  A
        lost or malformed response therefore cannot be repaired by calling a
        legacy reserve/release hook: doing so could leave the provider turn
        unaccounted for or admit it twice.  Keep the committed hold
        fail-closed until a later process can recover the exact operation.  The
        returned lease is untrusted until its complete identity matches both
        the submitted request and the authority state.
        """

        required = {
            "action",
            "operation_id",
            "request_digest",
            "audit_session_id",
            "actor",
            "policy_digest",
            "context_revision",
            "source_revision",
            "source_digest",
            "route_digest",
            "codex_session_id",
            "reserved_tokens",
            "reserved_compute",
            "reserved_issue_writes",
        }
        if not isinstance(expected, Mapping) or not required.issubset(expected):
            return False
        operation = lease.operation
        if (
            lease.status != "admitted"
            or not isinstance(operation, CodexOperationSnapshot)
            or operation.status != "inflight"
            or not isinstance(lease.reservation_id, str)
            or not lease.reservation_id
            or not isinstance(lease.reservation_operation_id, str)
            or not lease.reservation_operation_id
            or lease.reservation_id != operation.reservation_id
            or lease.reservation_operation_id != operation.reservation_operation_id
        ):
            return False
        for name in (
            "action",
            "operation_id",
            "request_digest",
            "audit_session_id",
            "actor",
            "policy_digest",
            "context_revision",
            "source_revision",
            "source_digest",
            "route_digest",
            "codex_session_id",
        ):
            if getattr(operation, name) != expected[name]:
                return False
        if (
            isinstance(lease.reserved_tokens, bool)
            or not isinstance(lease.reserved_tokens, int)
            or lease.reserved_tokens < 0
            or isinstance(lease.reserved_compute, bool)
            or not isinstance(lease.reserved_compute, (int, float))
            or not math.isfinite(float(lease.reserved_compute))
            or lease.reserved_compute < 0
            or isinstance(lease.reserved_issue_writes, bool)
            or not isinstance(lease.reserved_issue_writes, int)
            or lease.reserved_issue_writes < 0
        ):
            return False
        if (
            lease.reserved_tokens != expected["reserved_tokens"]
            or lease.reserved_compute != expected["reserved_compute"]
            or lease.reserved_issue_writes != expected["reserved_issue_writes"]
        ):
            return False
        action = expected["action"]
        if action in {"start", "resume"} and AuditCodexClient._is_app_server_provider(provider):
            try:
                from robot_sf.analysis_workbench.audit_codex_app_server import (  # noqa: PLC0415
                    UNMEASURED_TURN_COMPUTE_CHARGE,
                )
            except ImportError:
                return False
            if lease.reserved_tokens <= 0 or lease.reserved_compute < float(
                UNMEASURED_TURN_COMPUTE_CHARGE
            ):
                return False
        try:
            state = self.service.authority.snapshot()
            extensions = state.get("extensions")
            codex = extensions.get("codex") if isinstance(extensions, Mapping) else None
            operations = codex.get("operations") if isinstance(codex, Mapping) else None
            raw_operation = (
                operations.get(expected["operation_id"])
                if isinstance(operations, Mapping)
                else None
            )
            authoritative_operation = self.service._codex_operation_snapshot(raw_operation)
            reservations = state.get("reservations")
            reservation = (
                reservations.get(operation.reservation_id)
                if isinstance(reservations, Mapping)
                else None
            )
            budget_operations = state.get("operations")
            reservation_operation = (
                budget_operations.get(operation.reservation_operation_id)
                if isinstance(budget_operations, Mapping)
                else None
            )
        except (AuditServiceError, TypeError, ValueError, KeyError, AttributeError):
            return False
        if (
            authoritative_operation.to_dict() != operation.to_dict()
            or authoritative_operation.status != "inflight"
            or not isinstance(reservation, Mapping)
            or not isinstance(reservation_operation, Mapping)
        ):
            return False
        if (
            reservation.get("reservation_id") != operation.reservation_id
            or reservation.get("session_id") != expected["audit_session_id"]
            or reservation.get("operation_id") != operation.reservation_operation_id
            or reservation.get("tokens") != expected["reserved_tokens"]
            or reservation.get("compute") != expected["reserved_compute"]
            or reservation.get("issue_writes") != expected["reserved_issue_writes"]
            or reservation_operation.get("operation_id") != operation.reservation_operation_id
            or reservation_operation.get("operation_type") != "budget.reserve.codex"
            or reservation_operation.get("session_id") != expected["audit_session_id"]
            or reservation_operation.get("status") != "finished"
            or reservation_operation.get("result_status") != "committed"
        ):
            return False
        return True

    @staticmethod
    def _durable_admission_expectation(
        audit: AuditSession,
        *,
        request: Mapping[str, Any],
        context: AuditSelectionContext,
        route_digest: str,
        codex_session_id: str | None,
        reservation: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Build the request identity a returned admission must reproduce."""

        return {
            **dict(request),
            "audit_session_id": audit.session_id,
            "actor": audit.actor.to_dict(),
            "policy_digest": AuditCodexClient._policy_digest(audit),
            "context_revision": context.context_revision,
            "source_revision": audit.source_revision,
            "source_digest": audit.source_digest,
            "route_digest": route_digest,
            "codex_session_id": codex_session_id,
            "reserved_tokens": reservation.get("tokens"),
            "reserved_compute": reservation.get("compute"),
            "reserved_issue_writes": reservation.get("issue_writes", 0),
        }

    @staticmethod
    def _durable_provider_result_for_finish(
        provider: CodexProvider,
        *,
        action: str,
        provider_result: Mapping[str, Any],
        result_status: str,
        session: CodexSession | None,
        reserved_tokens: int,
        reserved_compute: float,
        reserved_issue_writes: int,
    ) -> tuple[Mapping[str, Any], str, CodexSession | None]:
        """Burn an App Server reservation when a turn is not safely measured.

        The App Server route emits token telemetry but does not provide a
        physical compute meter.  Its provider adapter uses one bounded,
        explicitly unmeasured per-turn accounting charge.  A successful
        durable callback that reports zero must not release the reservation as
        if no work occurred.  A failed turn with valid bounded telemetry keeps
        that actual settlement; missing or unsafe telemetry burns the ceiling.
        Upper-bound overruns remain untouched so authority can derive a
        terminal ``denied`` result from the authoritative budget state.
        """

        if not AuditCodexClient._is_app_server_provider(provider) or action not in {
            "start",
            "resume",
        }:
            return provider_result, result_status, session
        try:
            from robot_sf.analysis_workbench.audit_codex_app_server import (  # noqa: PLC0415
                UNMEASURED_TURN_COMPUTE_CHARGE,
            )
        except ImportError:
            return provider_result, result_status, session
        usage = AuditCodexClient._valid_usage(provider_result)
        status = str(provider_result.get("status", "complete"))
        accepted_statuses = {
            "start": {"complete", "success", "started"},
            "resume": {"complete", "success", "resumed", "started"},
        }[action]
        if (
            usage is not None
            and usage["compute"] >= float(UNMEASURED_TURN_COMPUTE_CHARGE)
            and usage["issue_writes"] == 0
            and (status in accepted_statuses or status == "failed")
        ):
            return provider_result, result_status, session

        normalized = dict(provider_result)
        normalized["status"] = "failed"
        normalized["usage"] = {
            "tokens": reserved_tokens,
            "compute": reserved_compute,
            "issue_writes": reserved_issue_writes,
        }
        normalized.setdefault(
            "reason",
            "App Server turn telemetry was not safely measured; the reservation ceiling was burned",
        )
        return normalized, "failed", None

    @staticmethod
    def _provider_settlement_result(
        provider: CodexProvider,
        provider_result: Mapping[str, Any] | None,
        *,
        action: str,
        reserved_tokens: int,
        reserved_compute: float,
    ) -> Mapping[str, Any] | None:
        """Burn a live reservation when provider telemetry is ambiguous."""

        if not AuditCodexClient._is_app_server_provider(provider):
            return provider_result
        from robot_sf.analysis_workbench.audit_codex_app_server import (  # noqa: PLC0415
            UNMEASURED_TURN_COMPUTE_CHARGE,
        )

        actual_tokens, actual_compute, actual_issue_writes = AuditCodexClient._provider_usage(
            provider_result
        )
        try:
            finite_compute = float(actual_compute)
        except (OverflowError, TypeError, ValueError):
            finite_compute = math.nan
        accepted_statuses = {
            "start": {"complete", "success", "started"},
            "resume": {"complete", "success", "resumed", "started"},
        }.get(action, set())
        status = (
            str(provider_result["status"])
            if isinstance(provider_result, Mapping) and "status" in provider_result
            else ""
        )
        if (
            isinstance(actual_tokens, int)
            and not isinstance(actual_tokens, bool)
            and actual_tokens >= 0
            and isinstance(actual_compute, (int, float))
            and not isinstance(actual_compute, bool)
            and math.isfinite(finite_compute)
            and actual_compute >= float(UNMEASURED_TURN_COMPUTE_CHARGE)
            and isinstance(actual_issue_writes, int)
            and not isinstance(actual_issue_writes, bool)
            and actual_issue_writes >= 0
            and actual_tokens <= reserved_tokens
            and finite_compute <= reserved_compute
            and actual_issue_writes == 0
            and status in accepted_statuses
        ):
            return provider_result
        return {
            "usage": {
                "tokens": reserved_tokens,
                "compute": reserved_compute,
                "issue_writes": 0,
            }
        }

    def _blocked_cancel(
        self,
        session: CodexSession,
        *,
        operation_id: str,
        request_digest: str,
        reason: str,
    ) -> CodexResult:
        """Record a safety cancellation blocked from provider authority."""

        event = self._event(session, "cancel", "conflict", reason, operation_id)
        updated = self._replace(
            session,
            status="cancelled",
            events=(*session.events, event),
        )
        with self._lock:
            self._sessions[updated.session_id] = updated
        receipt = self._receipt(
            updated,
            "cancel",
            "conflict",
            operation_id,
            reason=reason,
            request_digest=request_digest,
        )
        return CodexResult("conflict", updated, receipt, (event,), reason)

    @staticmethod
    def _cancel_request_digest(session: CodexSession, reason: str) -> str:
        """Derive the stable, token-free request binding for a cancel transition."""

        return _digest(
            {
                "action": "cancel",
                "session_id": session.session_id,
                "provider_session_id": session.provider_session_id,
                "route": session.route.to_dict(),
                "context": session.context.to_dict(),
                "source_digest": session.source_digest,
                "source_revision": session.source_revision,
                "reason": reason,
            }
        )

    def _durable_blocked_cancel(
        self,
        audit: AuditSession,
        session: CodexSession,
        *,
        operation_id: str,
        request_digest: str,
        reason: str,
        allow_context_stale: bool = False,
        _collision_fallback: bool = False,
    ) -> CodexResult:
        """Persist a provider-free cancellation blocked by stale source bytes."""

        updated = session
        admitted = False
        try:
            route = session.route
            lease = self.service.begin_codex_operation(
                audit,
                operation_id=operation_id,
                action="cancel",
                codex_session_id=session.session_id,
                request_digest=request_digest,
                context=session.context,
                route_digest=route.capability_digest,
                allow_cancelled=True,
                allow_source_stale=True,
                allow_context_stale=allow_context_stale,
                allow_reservation_collision=True,
            )
            if lease.status != "admitted":
                return self._recovered_result(lease, fallback_session=session)
            else:
                admitted = True
                event = self._event(session, "cancel", "conflict", reason, operation_id)
                updated = self._replace(
                    session, status="cancelled", events=(*session.events, event)
                )
                snapshot = self._authority_session_snapshot(
                    updated,
                    audit,
                    last_operation_id=operation_id,
                )
                finished = self.service.finish_codex_operation(
                    audit,
                    operation_id=operation_id,
                    request_digest=request_digest,
                    result_status="conflict",
                    result={
                        "status": "conflict",
                        "reason": reason,
                        "usage": {"tokens": 0, "compute": 0.0, "issue_writes": 0},
                    },
                    provider_session_id=updated.provider_session_id,
                    usage={"tokens": 0, "compute": 0.0, "issue_writes": 0},
                    session_snapshot=snapshot,
                    route_digest=route.capability_digest,
                    allow_source_stale=True,
                )
                result = self._recovered_result(finished, fallback_session=updated)
            with self._lock:
                self._sessions[updated.session_id] = updated
            return result
        except AuditAuthorityError as exc:
            if not admitted and not _collision_fallback:
                fallback_operation_id = (
                    f"codex-cancel-conflict-"
                    f"{_digest({'session_id': session.session_id, 'operation_id': operation_id, 'reason': reason})[:32]}"
                )
                return self._durable_blocked_cancel(
                    audit,
                    session,
                    operation_id=fallback_operation_id,
                    request_digest=request_digest,
                    reason=f"{reason}; cancellation operation collision: {exc}",
                    allow_context_stale=allow_context_stale,
                    _collision_fallback=True,
                )
            return CodexResult(
                "unavailable",
                session=updated,
                reason=f"{reason}; durable cancellation record unavailable: {exc}",
            )
        except (AuditServiceError, TypeError, ValueError, KeyError) as exc:
            return CodexResult(
                "unavailable",
                session=updated,
                reason=f"{reason}; durable cancellation record unavailable: {exc}",
            )

    def _durable_cancel_conflict(
        self,
        current: CodexSession,
        *,
        operation_id: str,
        reason: str,
        audit_token: str | None,
        request_reason: str | None = None,
    ) -> CodexResult:
        """Persist a provider-free cancel conflict after a binding failure."""

        audit = self._audit_for(current)
        owned = self.service.get_session(audit, token=audit_token)
        rebound = current
        allow_context_stale = False
        if owned.context != current.context:
            rebound = self._replace(
                current,
                context=owned.context,
                evidence=self._evidence(audit, owned.context),
                source_digest=owned.source_digest,
                source_revision=owned.source_revision,
            )
            allow_context_stale = True
        return self._durable_blocked_cancel(
            audit,
            rebound,
            operation_id=operation_id,
            request_digest=self._cancel_request_digest(
                rebound, request_reason if request_reason is not None else reason
            ),
            reason=reason,
            allow_context_stale=allow_context_stale,
        )

    def _terminal_cancel(
        self,
        session: CodexSession,
        *,
        operation_id: str,
        request_digest: str,
        status: str,
        reason: str,
    ) -> CodexResult:
        """Retain a terminal local cancellation after service kill authority."""

        event = self._event(session, "cancel", status, reason, operation_id)
        updated = self._replace(
            session,
            status="cancelled",
            events=(*session.events, event),
        )
        with self._lock:
            self._sessions[updated.session_id] = updated
        receipt = self._receipt(
            updated,
            "cancel",
            status,
            operation_id,
            reason=reason,
            request_digest=request_digest,
        )
        return CodexResult(status, updated, receipt, (event,), reason)

    def _event(
        self, session: CodexSession, kind: str, status: str, message: str, operation_id: str
    ) -> CodexActivityEvent:
        return CodexActivityEvent(
            event_id=f"event-{uuid.uuid4().hex}",
            kind=kind,
            status=status,
            message=message[:4_096],
            evidence_ids=tuple(item.evidence_id for item in session.evidence),
            operation_id=operation_id,
        )

    @staticmethod
    def _provider_ok(result: Mapping[str, Any], *accepted: str) -> bool:
        """Accept only an explicit successful provider status."""

        if not isinstance(result, Mapping):
            raise TypeError("Codex provider result must be a mapping")
        return str(result.get("status", "complete")) in set(accepted)

    def _provider_failure(
        self,
        session: CodexSession,
        *,
        action: str,
        operation_id: str,
        result: Mapping[str, Any],
        request_digest: str,
        usage: Mapping[str, Any] | None = None,
    ) -> CodexResult:
        """Retain a failed activity and receipt without changing session authority."""

        reason = str(result.get("reason", f"provider {action} failed"))
        event = self._event(session, action, "failed", reason, operation_id)
        updated = self._replace(session, events=(*session.events, event))
        with self._lock:
            self._sessions[updated.session_id] = updated
        receipt = self._receipt(
            updated,
            action,
            "failed",
            operation_id,
            reason=reason,
            usage=usage,
            request_digest=request_digest,
        )
        return CodexResult("failed", updated, receipt, (event,), reason)

    def _start_durable(  # noqa: C901, PLR0912
        self,
        audit_session: AuditSession | str,
        *,
        prompt: str,
        context: AuditSelectionContext | Mapping[str, Any] | None,
        route_id: str | None,
        operation_id: str | None,
        token_budget: int,
        compute_budget: float,
        audit_token: str | None,
    ) -> CodexResult:
        """Run start through the authority-owned operation state machine."""

        opid = operation_id or f"codex-op-{uuid.uuid4().hex}"
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
            return CodexResult("failed", reason="prompt is empty or exceeds the limit")
        try:
            audit = self.service.get_session(audit_session, token=audit_token)
            selected = (
                audit.context
                if context is None
                else context
                if isinstance(context, AuditSelectionContext)
                else AuditSelectionContext.from_mapping(context)
            )
            if selected != audit.context:
                return CodexResult("conflict", reason="selection context is stale")
            self.service._check_session(audit)
            self.service._assert_source(audit)
            capabilities = self.inspect_capabilities()
            route = capabilities.choose(route_id)
            if route is None:
                return CodexResult(
                    "unavailable",
                    reason=capabilities.reason or "no uniquely discovered Codex route",
                )
            provider = self.provider
            if provider is None:
                return CodexResult(
                    "unavailable", reason="no supported Codex provider is configured"
                )
            if not getattr(provider, "supports_tools", False):
                return CodexResult(
                    "unavailable", reason="provider route does not expose audit tools"
                )
            self._require_app_server_reservation(
                provider,
                token_budget=token_budget,
                compute_budget=compute_budget,
                action="start",
            )
            request_digest = _digest(
                {
                    "action": "start",
                    "audit_session_id": audit.session_id,
                    "prompt": prompt,
                    "context": selected.to_dict(),
                    "route": route.to_dict(),
                    "token_budget": token_budget,
                    "compute_budget": compute_budget,
                }
            )
            admission_expected = self._durable_admission_expectation(
                audit,
                request={
                    "action": "start",
                    "operation_id": opid,
                    "request_digest": request_digest,
                },
                context=selected,
                route_digest=route.capability_digest,
                codex_session_id=None,
                reservation={
                    "tokens": token_budget,
                    "compute": compute_budget,
                    "issue_writes": 0,
                },
            )
            lease = self.service.begin_codex_operation(
                audit,
                operation_id=opid,
                action="start",
                request_digest=request_digest,
                context=selected,
                route_digest=route.capability_digest,
                reserved_tokens=token_budget,
                reserved_compute=compute_budget,
                audit_token=audit_token,
            )
            if lease.status != "admitted":
                recovered = self._recovered_result(lease)
                if recovered.session is not None:
                    with self._lock:
                        self._audit_handles[recovered.session.session_id] = audit
                return recovered
            if not self._durable_admission_capability_valid(
                lease, provider=provider, expected=admission_expected
            ):
                return CodexResult(
                    "unavailable",
                    reason=(
                        "durable Codex budget admission capability is missing or malformed; "
                        "the operation remains ambiguous"
                    ),
                )
            evidence = self._evidence(audit, selected)
            provider_kwargs: dict[str, Any] = {}
            reserved_compute = self._provider_reserved_compute(provider, lease, action="start")
            if reserved_compute is not None:
                provider_kwargs["reserved_compute"] = reserved_compute
            try:
                provider_result = provider.start(
                    route=route,
                    prompt=prompt,
                    evidence=[item.to_dict() for item in evidence],
                    **provider_kwargs,
                )
            except Exception as exc:  # noqa: BLE001 - accepted work is ambiguous.
                return CodexResult(
                    "unavailable",
                    reason=f"Codex provider start is ambiguous after admission: {exc}",
                )
            if not isinstance(provider_result, Mapping):
                return CodexResult(
                    "unavailable",
                    reason="Codex provider returned a malformed result; operation is ambiguous",
                )
            usage = self._valid_usage(provider_result)
            if usage is None:
                return CodexResult(
                    "unavailable",
                    reason="provider usage is missing or malformed; Codex operation is ambiguous",
                )
            accepted = self._provider_ok(provider_result, "complete", "success", "started")
            provider_session_id = str(provider_result.get("provider_session_id", ""))
            session: CodexSession | None = None
            if accepted:
                provider_session_id = _text(
                    provider_result.get("provider_session_id"), name="provider_session_id"
                )
                session = CodexSession(
                    session_id=f"codex-session-{uuid.uuid4().hex}",
                    audit_session_id=audit.session_id,
                    context=selected,
                    route=route,
                    evidence=evidence,
                    provider_session_id=provider_session_id,
                    source_digest=audit.source_digest,
                    source_revision=audit.source_revision,
                )
                event = self._event(
                    session,
                    "activity",
                    "complete",
                    str(provider_result.get("message", "Codex investigation started")),
                    opid,
                )
                session = self._replace(session, events=(event,))
            result_status = "complete" if accepted else "failed"
            finished = self._finish_durable(
                audit,
                lease=lease,
                provider=provider,
                request_digest=request_digest,
                result_status=result_status,
                provider_result=provider_result,
                session=session,
                provider_session_id=provider_session_id,
            )
            if finished.session is not None:
                with self._lock:
                    self._audit_handles[finished.session.session_id] = audit
            return finished
        except AuditCancelled as exc:
            return CodexResult("cancelled", reason=str(exc))
        except AuditContextConflict as exc:
            return CodexResult("conflict", reason=str(exc))
        except CapabilityUnavailable as exc:
            return CodexResult("unavailable", reason=str(exc))
        except AuditBudgetExceeded as exc:
            return CodexResult("denied", reason=str(exc))
        except (AuthorityOperationConflict, AuditServiceError, TypeError, KeyError) as exc:
            return CodexResult("conflict", reason=str(exc))
        except ValueError as exc:
            return CodexResult("failed", reason=str(exc))

    def start(  # noqa: C901, PLR0912, PLR0915
        self,
        audit_session: AuditSession | str,
        *,
        prompt: str,
        context: AuditSelectionContext | Mapping[str, Any] | None = None,
        route_id: str | None = None,
        operation_id: str | None = None,
        token_budget: int = 0,
        compute_budget: float = 0.0,
        audit_token: str | None = None,
    ) -> CodexResult:
        """Start an evidence-bound investigation on a discovered route only."""

        return self._start_durable(
            audit_session,
            prompt=prompt,
            context=context,
            route_id=route_id,
            operation_id=operation_id,
            token_budget=token_budget,
            compute_budget=compute_budget,
            audit_token=audit_token,
        )

        opid = operation_id or f"codex-op-{uuid.uuid4().hex}"
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > MAX_PROMPT_CHARS:
            return CodexResult("failed", reason="prompt is empty or exceeds the limit")
        try:
            audit = self.service.get_session(audit_session, token=audit_token)
            selected = (
                audit.context
                if context is None
                else context
                if isinstance(context, AuditSelectionContext)
                else AuditSelectionContext.from_mapping(context)
            )
            if selected != audit.context:
                return CodexResult("conflict", reason="selection context is stale")
            self.service._check_session(audit)
            self.service._assert_source(audit)
            capabilities = self.inspect_capabilities()
            route = capabilities.choose(route_id)
            if route is None:
                return CodexResult(
                    "unavailable",
                    reason=capabilities.reason or "no uniquely discovered Codex route",
                )
            provider = self.provider
            if provider is None:
                return CodexResult(
                    "unavailable", reason="no supported Codex provider is configured"
                )
            if not getattr(provider, "supports_tools", False):
                return CodexResult(
                    "unavailable", reason="provider route does not expose audit tools"
                )
            request_digest = _digest(
                {
                    "action": "start",
                    "audit_session_id": audit.session_id,
                    "prompt": prompt,
                    "context": selected.to_dict(),
                    "route": route.to_dict(),
                    "token_budget": token_budget,
                    "compute_budget": compute_budget,
                }
            )
            replay = self._replayed_operation(opid, request_digest)
            if replay is not None:
                return replay
            evidence = self._evidence(audit, selected)
            # Reserve caller-declared limits before provider work.  The
            # returned provider meter is settled below; a missing/oversized
            # meter never becomes a successful Codex result.
            reservation_operation_id = f"{opid}-reserve"
            reservation = self.service.reserve_budget(
                audit,
                tokens=token_budget,
                compute=compute_budget,
                operation_id=reservation_operation_id,
            )
            if not reservation.ok:
                return CodexResult(reservation.status, reason=reservation.reason)
            reservation_id = self._reservation_id(reservation)
            if reservation_id is None:
                released = self._release_lost_reservation_reply(
                    audit,
                    operation_id=opid,
                    reservation_operation_id=reservation_operation_id,
                )
                return CodexResult(
                    "unavailable",
                    reason="budget reservation capability is missing"
                    if released
                    else "budget reservation capability is missing; hold remains unresolved",
                )
            provider_result: Mapping[str, Any] | None = None
            settlement_tokens = token_budget
            settlement_compute = compute_budget
            provider_started = False
            try:
                reserved_compute = self._provider_reserved_compute(
                    provider,
                    reservation,
                    action="start",
                )
                if reserved_compute is not None:
                    settlement_compute = reserved_compute
                    provider_kwargs = {"reserved_compute": reserved_compute}
                else:
                    provider_kwargs = {}
                reserved_tokens = self._provider_reserved_tokens(
                    provider,
                    reservation,
                    action="start",
                )
                if reserved_tokens is not None:
                    settlement_tokens = reserved_tokens
                provider_started = True
                provider_result = provider.start(
                    route=route,
                    prompt=prompt,
                    evidence=[item.to_dict() for item in evidence],
                    **provider_kwargs,
                )
            except Exception as exc:  # noqa: BLE001 - conservatively settle provider failures.
                self._settle_provider_usage(
                    audit,
                    operation_id=opid,
                    reservation_id=reservation_id,
                    reservation_operation_id=reservation_operation_id,
                    reserved_tokens=settlement_tokens,
                    reserved_compute=settlement_compute,
                    provider_result=(
                        self._provider_settlement_result(
                            provider,
                            None,
                            action="start",
                            reserved_tokens=settlement_tokens,
                            reserved_compute=settlement_compute,
                        )
                        if provider_started
                        else None
                    ),
                )
                return CodexResult("failed", reason=str(exc))
            settlement_result = self._provider_settlement_result(
                provider,
                provider_result,
                action="start",
                reserved_tokens=settlement_tokens,
                reserved_compute=settlement_compute,
            )
            meter = self._settle_provider_usage(
                audit,
                operation_id=opid,
                reservation_id=reservation_id,
                reservation_operation_id=reservation_operation_id,
                reserved_tokens=settlement_tokens,
                reserved_compute=settlement_compute,
                provider_result=settlement_result,
            )
            if not meter.ok:
                return CodexResult(meter.status, reason=meter.reason)
            if settlement_result is not provider_result:
                return CodexResult(
                    "failed",
                    reason="live App Server usage telemetry is missing or malformed",
                )
            if not self._provider_ok(provider_result, "complete", "success", "started"):
                return CodexResult(
                    "failed", reason=str(provider_result.get("reason", "provider start failed"))
                )
            provider_session_id = _text(
                provider_result.get("provider_session_id"), name="provider_session_id"
            )
            session = CodexSession(
                session_id=f"codex-session-{uuid.uuid4().hex}",
                audit_session_id=audit.session_id,
                context=selected,
                route=route,
                evidence=evidence,
                provider_session_id=provider_session_id,
                source_digest=audit.source_digest,
                source_revision=audit.source_revision,
            )
            message = str(provider_result.get("message", "Codex investigation started"))
            event = self._event(session, "activity", "complete", message, opid)
            session = (
                CodexSession(**{**session.__dict__, "events": (event,)})
                if hasattr(session, "__dict__")
                else CodexSession(
                    session_id=session.session_id,
                    audit_session_id=session.audit_session_id,
                    context=session.context,
                    route=session.route,
                    evidence=session.evidence,
                    provider_session_id=session.provider_session_id,
                    source_digest=session.source_digest,
                    source_revision=session.source_revision,
                    status=session.status,
                    events=(event,),
                    created_at=session.created_at,
                )
            )
            with self._lock:
                self._sessions[session.session_id] = session
                self._audit_handles[session.session_id] = audit
            receipt = self._receipt(
                session,
                "start",
                "complete",
                opid,
                usage=provider_result.get("usage"),
                request_digest=request_digest,
            )
            return CodexResult("complete", session=session, receipt=receipt, events=(event,))
        except AuditCancelled as exc:
            return CodexResult("cancelled", reason=str(exc))
        except AuditContextConflict as exc:
            return CodexResult("conflict", reason=str(exc))
        except CapabilityUnavailable as exc:
            return CodexResult("unavailable", reason=str(exc))
        except (AuditServiceError, TypeError, ValueError, KeyError) as exc:
            return CodexResult("failed", reason=str(exc))

    start_session = start
    investigate = start

    def _codex_session(
        self, session: CodexSession | str, *, audit_token: str | None = None
    ) -> CodexSession:
        with self._lock:
            if isinstance(session, CodexSession):
                known = self._sessions.get(session.session_id)
                if known is not session:
                    raise ValueError("Codex session is not owned by this client")
                return known
            known = self._sessions.get(session)
        if known is not None:
            return known
        if not isinstance(audit_token, str) or not audit_token.strip():
            raise ValueError("audit token is required for Codex session recovery")
        recovered = self.recover_session(session, audit_token=audit_token)
        if recovered.session is None:
            raise ValueError(recovered.reason or "unknown Codex session")
        return recovered.session

    def _cancel_retry_session(
        self,
        submitted: CodexSession,
        *,
        operation_id: str,
        reason: str,
    ) -> CodexSession | None:
        """Accept one exact terminal cancel replay from an older local handle.

        Durable cancellation replaces the local immutable session object.  A
        caller may still retry the same operation with the original opaque
        handle, but only when the currently owned handle has the same provider,
        route, source, context, and evidence binding and the durable terminal
        operation matches the submitted request digest exactly.
        """

        with self._lock:
            current = self._sessions.get(submitted.session_id)
            audit = self._audit_handles.get(submitted.session_id)
            cancel_handle = self._cancel_handle_aliases.get(submitted.session_id)
        if current is None or current is submitted or current.status != "cancelled":
            return None
        if cancel_handle is not submitted:
            return None
        if not self._same_session_binding(submitted, current):
            return None
        if audit is None:
            return None
        raw = (
            self.service.authority.snapshot()
            .get("extensions", {})
            .get("codex", {})
            .get("operations", {})
            .get(operation_id)
        )
        if not isinstance(raw, Mapping):
            return None
        if (
            raw.get("action") != "cancel"
            or raw.get("status") != "finished"
            or raw.get("audit_session_id") != submitted.audit_session_id
            or raw.get("codex_session_id") != submitted.session_id
            or raw.get("request_digest") != self._cancel_request_digest(submitted, reason)
        ):
            return None
        return current

    @staticmethod
    def _same_session_binding(left: CodexSession, right: CodexSession) -> bool:
        """Compare immutable provider/session identity, excluding lifecycle events."""

        return (
            left.session_id == right.session_id
            and left.audit_session_id == right.audit_session_id
            and left.context == right.context
            and left.route == right.route
            and left.evidence == right.evidence
            and left.provider_session_id == right.provider_session_id
            and left.source_digest == right.source_digest
            and left.source_revision == right.source_revision
        )

    def _audit_for(self, session: CodexSession) -> AuditSession:
        """Resolve the exact private audit handle bound at Codex start."""

        with self._lock:
            try:
                return self._audit_handles[session.session_id]
            except KeyError as exc:
                raise ValueError("Codex session has no owned audit handle") from exc

    def _check_audit_lifecycle(
        self, session: CodexSession, *, audit_token: str | None = None
    ) -> AuditSession:
        """Revalidate audit ownership, kill switch, context, and source bytes."""

        audit = self._audit_for(session)
        owned = self.service.get_session(audit, token=audit_token)
        if owned.context != session.context:
            raise AuditContextConflict("audit selection context changed")
        self.service._check_session(owned)
        self.service._assert_source(owned)
        if (
            owned.source_digest != session.source_digest
            or owned.source_revision != session.source_revision
        ):
            raise AuditContextConflict("audit source identity or revision changed")
        return owned

    def _reconnect_durable(  # noqa: C901
        self,
        session: CodexSession | str,
        *,
        operation_id: str | None,
        audit_token: str | None,
    ) -> CodexResult:
        """Reconnect through one durable provider operation."""

        try:
            current = self._codex_session(session, audit_token=audit_token)
            audit = self._check_audit_lifecycle(current, audit_token=audit_token)
            if self.provider is None:
                return CodexResult(
                    "unavailable",
                    session=current,
                    reason="no supported Codex provider is configured",
                )
            route = self.inspect_capabilities().choose(current.route.route_id)
            if route is None or route.to_dict() != current.route.to_dict():
                return CodexResult(
                    "unavailable",
                    session=current,
                    reason="current Codex route does not match session",
                )
            opid = operation_id or f"codex-op-{uuid.uuid4().hex}"
            request_digest = _digest(
                {
                    "action": "reconnect",
                    "session_id": current.session_id,
                    "provider_session_id": current.provider_session_id,
                    "route": route.to_dict(),
                    "context": current.context.to_dict(),
                    "source_digest": current.source_digest,
                    "source_revision": current.source_revision,
                }
            )
            admission_expected = self._durable_admission_expectation(
                audit,
                request={
                    "action": "reconnect",
                    "operation_id": opid,
                    "request_digest": request_digest,
                },
                context=current.context,
                route_digest=route.capability_digest,
                codex_session_id=current.session_id,
                reservation={"tokens": 0, "compute": 0.0, "issue_writes": 0},
            )
            lease = self.service.begin_codex_operation(
                audit,
                audit_token=audit_token,
                operation_id=opid,
                action="reconnect",
                codex_session_id=current.session_id,
                request_digest=request_digest,
                context=current.context,
                route_digest=route.capability_digest,
            )
            if lease.status != "admitted":
                recovered = self._recovered_result(lease, fallback_session=current)
                if recovered.session is not None:
                    with self._lock:
                        self._audit_handles[recovered.session.session_id] = audit
                return recovered
            if not self._durable_admission_capability_valid(
                lease, provider=self.provider, expected=admission_expected
            ):
                return CodexResult(
                    "unavailable",
                    session=current,
                    reason=(
                        "durable Codex budget admission capability is missing or malformed; "
                        "the operation remains ambiguous"
                    ),
                )
            try:
                provider_result = self.provider.reconnect(
                    provider_session_id=current.provider_session_id, route=route
                )
            except Exception as exc:  # noqa: BLE001 - admitted work is ambiguous.
                return CodexResult(
                    "unavailable",
                    session=current,
                    reason=f"Codex provider reconnect is ambiguous after admission: {exc}",
                )
            if (
                not isinstance(provider_result, Mapping)
                or self._valid_usage(provider_result) is None
            ):
                return CodexResult(
                    "unavailable",
                    session=current,
                    reason="provider usage is missing or malformed; Codex operation is ambiguous",
                )
            accepted = self._provider_ok(
                provider_result, "complete", "success", "reconnected", "started"
            )
            if accepted:
                event = self._event(
                    current,
                    "reconnect",
                    "complete",
                    str(provider_result.get("message", "Codex session reconnected")),
                    opid,
                )
                updated = self._replace(current, events=(*current.events, event), status="active")
            else:
                updated = None
            finished = self._finish_durable(
                audit,
                lease=lease,
                provider=self.provider,
                request_digest=request_digest,
                result_status="complete" if accepted else "failed",
                provider_result=provider_result,
                session=updated,
                provider_session_id=str(
                    provider_result.get("provider_session_id", current.provider_session_id)
                ),
            )
            if finished.session is not None:
                with self._lock:
                    self._audit_handles[finished.session.session_id] = audit
            return finished
        except AuditCancelled as exc:
            return CodexResult("cancelled", session=locals().get("current"), reason=str(exc))
        except AuditContextConflict as exc:
            return CodexResult("conflict", session=locals().get("current"), reason=str(exc))
        except CapabilityUnavailable as exc:
            return CodexResult("unavailable", session=locals().get("current"), reason=str(exc))
        except (AuditServiceError, TypeError, ValueError, KeyError) as exc:
            return CodexResult("conflict", session=locals().get("current"), reason=str(exc))

    def reconnect(  # noqa: C901
        self,
        session: CodexSession | str,
        *,
        operation_id: str | None = None,
        audit_token: str | None = None,
    ) -> CodexResult:
        """Reconnect a provider session while preserving route/evidence receipt."""

        return self._reconnect_durable(session, operation_id=operation_id, audit_token=audit_token)

        try:
            current = self._codex_session(session, audit_token=audit_token)
        except ValueError as exc:
            return CodexResult("conflict", reason=str(exc))
        opid = operation_id or f"codex-op-{uuid.uuid4().hex}"
        request_digest = _digest(
            {
                "action": "reconnect",
                "session_id": current.session_id,
                "provider_session_id": current.provider_session_id,
                "route": current.route.to_dict(),
                "context": current.context.to_dict(),
                "source_digest": current.source_digest,
                "source_revision": current.source_revision,
            }
        )
        try:
            audit = self._check_audit_lifecycle(current)
            replay = self._replayed_operation(opid, request_digest, session=current)
            if replay is not None:
                return replay
            if self.provider is None:
                return CodexResult(
                    "unavailable",
                    session=current,
                    reason="no supported Codex provider is configured",
                )
            reservation_operation_id = f"{opid}-reserve"
            reservation = self.service.reserve_budget(audit, operation_id=reservation_operation_id)
            if not reservation.ok:
                return CodexResult(reservation.status, session=current, reason=reservation.reason)
            reservation_id = self._reservation_id(reservation)
            if reservation_id is None:
                released = self._release_lost_reservation_reply(
                    audit,
                    operation_id=opid,
                    reservation_operation_id=reservation_operation_id,
                )
                return CodexResult(
                    "unavailable",
                    session=current,
                    reason="budget reservation capability is missing"
                    if released
                    else "budget reservation capability is missing; hold remains unresolved",
                )
            try:
                provider_result = self.provider.reconnect(
                    provider_session_id=current.provider_session_id, route=current.route
                )
            except Exception as exc:  # noqa: BLE001 - provider failures are typed and receipted.
                self._settle_provider_usage(
                    audit,
                    operation_id=opid,
                    reservation_id=reservation_id,
                    reservation_operation_id=reservation_operation_id,
                    reserved_tokens=0,
                    reserved_compute=0.0,
                    provider_result=None,
                )
                return self._provider_failure(
                    current,
                    action="reconnect",
                    operation_id=opid,
                    result={"reason": f"provider reconnect failed: {exc}"},
                    request_digest=request_digest,
                )
            meter = self._settle_provider_usage(
                audit,
                operation_id=opid,
                reservation_id=reservation_id,
                reservation_operation_id=reservation_operation_id,
                reserved_tokens=0,
                reserved_compute=0.0,
                provider_result=provider_result,
            )
            if not meter.ok:
                return CodexResult(meter.status, session=current, reason=meter.reason)
            if not self._provider_ok(
                provider_result, "complete", "success", "reconnected", "started"
            ):
                return self._provider_failure(
                    current,
                    action="reconnect",
                    operation_id=opid,
                    result=provider_result,
                    request_digest=request_digest,
                    usage=provider_result.get("usage"),
                )
            event = self._event(
                current,
                "reconnect",
                "complete",
                str(provider_result.get("message", "Codex session reconnected")),
                opid,
            )
            updated = self._replace(current, events=(*current.events, event), status="active")
            with self._lock:
                self._sessions[updated.session_id] = updated
            receipt = self._receipt(
                updated,
                "reconnect",
                "complete",
                opid,
                usage=provider_result.get("usage"),
                request_digest=request_digest,
            )
            return CodexResult("complete", updated, receipt, (event,))
        except AuditCancelled as exc:
            return CodexResult("cancelled", session=current, reason=str(exc))
        except AuditContextConflict as exc:
            return CodexResult("conflict", session=current, reason=str(exc))
        except CapabilityUnavailable as exc:
            return CodexResult("unavailable", session=current, reason=str(exc))
        except (AuditServiceError, TypeError, ValueError, KeyError) as exc:
            return CodexResult("failed", session=current, reason=str(exc))

    def _resume_durable(  # noqa: C901, PLR0912
        self,
        session: CodexSession | str,
        *,
        operation_id: str | None,
        token_budget: int,
        compute_budget: float,
        audit_token: str | None,
    ) -> CodexResult:
        """Resume a recovered session through durable admission and finish."""

        current: CodexSession | None = None
        try:
            current = self._codex_session(session, audit_token=audit_token)
            if current.status == "cancelled":
                return CodexResult(
                    "cancelled", session=current, reason="Codex session is cancelled"
                )
            audit = self._check_audit_lifecycle(current, audit_token=audit_token)
            if self.provider is None:
                return CodexResult(
                    "unavailable",
                    session=current,
                    reason="no supported Codex provider is configured",
                )
            route = self.inspect_capabilities().choose(current.route.route_id)
            if route is None or route.to_dict() != current.route.to_dict():
                return CodexResult(
                    "unavailable",
                    session=current,
                    reason="current Codex route does not match session",
                )
            self._require_app_server_reservation(
                self.provider,
                token_budget=token_budget,
                compute_budget=compute_budget,
                action="resume",
            )
            opid = operation_id or f"codex-op-{uuid.uuid4().hex}"
            request_digest = _digest(
                {
                    "action": "resume",
                    "session_id": current.session_id,
                    "provider_session_id": current.provider_session_id,
                    "route": route.to_dict(),
                    "context": current.context.to_dict(),
                    "source_digest": current.source_digest,
                    "source_revision": current.source_revision,
                    "token_budget": token_budget,
                    "compute_budget": compute_budget,
                }
            )
            admission_expected = self._durable_admission_expectation(
                audit,
                request={
                    "action": "resume",
                    "operation_id": opid,
                    "request_digest": request_digest,
                },
                context=current.context,
                route_digest=route.capability_digest,
                codex_session_id=current.session_id,
                reservation={
                    "tokens": token_budget,
                    "compute": compute_budget,
                    "issue_writes": 0,
                },
            )
            lease = self.service.begin_codex_operation(
                audit,
                audit_token=audit_token,
                operation_id=opid,
                action="resume",
                codex_session_id=current.session_id,
                request_digest=request_digest,
                context=current.context,
                route_digest=route.capability_digest,
                reserved_tokens=token_budget,
                reserved_compute=compute_budget,
            )
            if lease.status != "admitted":
                recovered = self._recovered_result(lease, fallback_session=current)
                if recovered.session is not None:
                    with self._lock:
                        self._audit_handles[recovered.session.session_id] = audit
                return recovered
            if not self._durable_admission_capability_valid(
                lease, provider=self.provider, expected=admission_expected
            ):
                return CodexResult(
                    "unavailable",
                    session=current,
                    reason=(
                        "durable Codex budget admission capability is missing or malformed; "
                        "the operation remains ambiguous"
                    ),
                )
            provider_kwargs: dict[str, Any] = {}
            reserved_compute = self._provider_reserved_compute(
                self.provider, lease, action="resume"
            )
            if reserved_compute is not None:
                provider_kwargs["reserved_compute"] = reserved_compute
            try:
                provider_result = self.provider.resume(
                    provider_session_id=current.provider_session_id,
                    route=route,
                    evidence=[item.to_dict() for item in current.evidence],
                    **provider_kwargs,
                )
            except Exception as exc:  # noqa: BLE001 - admitted work is ambiguous.
                return CodexResult(
                    "unavailable",
                    session=current,
                    reason=f"Codex provider resume is ambiguous after admission: {exc}",
                )
            if (
                not isinstance(provider_result, Mapping)
                or self._valid_usage(provider_result) is None
            ):
                return CodexResult(
                    "unavailable",
                    session=current,
                    reason="provider usage is missing or malformed; Codex operation is ambiguous",
                )
            accepted = self._provider_ok(
                provider_result, "complete", "success", "resumed", "started"
            )
            if accepted:
                event = self._event(
                    current,
                    "resume",
                    "complete",
                    str(provider_result.get("message", "Codex session resumed")),
                    opid,
                )
                updated = self._replace(current, events=(*current.events, event), status="active")
            else:
                updated = None
            return self._finish_durable(
                audit,
                lease=lease,
                provider=self.provider,
                request_digest=request_digest,
                result_status="complete" if accepted else "failed",
                provider_result=provider_result,
                session=updated,
                provider_session_id=str(
                    provider_result.get("provider_session_id", current.provider_session_id)
                ),
            )
        except AuditCancelled as exc:
            return CodexResult("cancelled", session=current, reason=str(exc))
        except AuditContextConflict as exc:
            return CodexResult("conflict", session=current, reason=str(exc))
        except CapabilityUnavailable as exc:
            return CodexResult("unavailable", session=current, reason=str(exc))
        except AuditBudgetExceeded as exc:
            return CodexResult("denied", session=current, reason=str(exc))
        except (AuthorityOperationConflict, AuditServiceError, TypeError, KeyError) as exc:
            return CodexResult("conflict", session=current, reason=str(exc))
        except ValueError as exc:
            return CodexResult("failed", session=current, reason=str(exc))

    def resume(  # noqa: C901, PLR0912
        self,
        session: CodexSession | str,
        *,
        operation_id: str | None = None,
        token_budget: int = 0,
        compute_budget: float = 0.0,
        audit_token: str | None = None,
    ) -> CodexResult:
        """Resume the same provider route with one explicitly reserved turn."""

        return self._resume_durable(
            session,
            operation_id=operation_id,
            token_budget=token_budget,
            compute_budget=compute_budget,
            audit_token=audit_token,
        )

        try:
            current = self._codex_session(session, audit_token=audit_token)
        except ValueError as exc:
            return CodexResult("conflict", reason=str(exc))
        if current.status == "cancelled":
            return CodexResult("cancelled", session=current, reason="Codex session is cancelled")
        opid = operation_id or f"codex-op-{uuid.uuid4().hex}"
        try:
            request_digest = _digest(
                {
                    "action": "resume",
                    "session_id": current.session_id,
                    "provider_session_id": current.provider_session_id,
                    "route": current.route.to_dict(),
                    "context": current.context.to_dict(),
                    "source_digest": current.source_digest,
                    "source_revision": current.source_revision,
                    "token_budget": token_budget,
                    "compute_budget": compute_budget,
                }
            )
            audit = self._check_audit_lifecycle(current)
            replay = self._replayed_operation(opid, request_digest, session=current)
            if replay is not None:
                return replay
            if self.provider is None:
                return CodexResult(
                    "unavailable",
                    session=current,
                    reason="no supported Codex provider is configured",
                )
            reservation_operation_id = f"{opid}-reserve"
            reservation = self.service.reserve_budget(
                audit,
                tokens=token_budget,
                compute=compute_budget,
                operation_id=reservation_operation_id,
            )
            if not reservation.ok:
                return CodexResult(reservation.status, session=current, reason=reservation.reason)
            reservation_id = self._reservation_id(reservation)
            if reservation_id is None:
                released = self._release_lost_reservation_reply(
                    audit,
                    operation_id=opid,
                    reservation_operation_id=reservation_operation_id,
                )
                return CodexResult(
                    "unavailable",
                    session=current,
                    reason="budget reservation capability is missing"
                    if released
                    else "budget reservation capability is missing; hold remains unresolved",
                )
            settlement_tokens = token_budget
            settlement_compute = compute_budget
            provider_started = False
            try:
                reserved_compute = self._provider_reserved_compute(
                    self.provider,
                    reservation,
                    action="resume",
                )
                if reserved_compute is not None:
                    settlement_compute = reserved_compute
                    provider_kwargs = {"reserved_compute": reserved_compute}
                else:
                    provider_kwargs = {}
                reserved_tokens = self._provider_reserved_tokens(
                    self.provider,
                    reservation,
                    action="resume",
                )
                if reserved_tokens is not None:
                    settlement_tokens = reserved_tokens
                provider_started = True
                provider_result = self.provider.resume(
                    provider_session_id=current.provider_session_id,
                    route=current.route,
                    evidence=[item.to_dict() for item in current.evidence],
                    **provider_kwargs,
                )
            except Exception as exc:  # noqa: BLE001 - provider failures are typed and receipted.
                failure_meter = (
                    self._provider_settlement_result(
                        self.provider,
                        None,
                        action="resume",
                        reserved_tokens=settlement_tokens,
                        reserved_compute=settlement_compute,
                    )
                    if provider_started
                    else None
                )
                self._settle_provider_usage(
                    audit,
                    operation_id=opid,
                    reservation_id=reservation_id,
                    reservation_operation_id=reservation_operation_id,
                    reserved_tokens=settlement_tokens,
                    reserved_compute=settlement_compute,
                    provider_result=failure_meter,
                )
                return self._provider_failure(
                    current,
                    action="resume",
                    operation_id=opid,
                    result={"reason": f"provider resume failed: {exc}"},
                    request_digest=request_digest,
                    usage=failure_meter.get("usage")
                    if isinstance(failure_meter, Mapping)
                    else None,
                )
            settlement_result = self._provider_settlement_result(
                self.provider,
                provider_result,
                action="resume",
                reserved_tokens=settlement_tokens,
                reserved_compute=settlement_compute,
            )
            meter = self._settle_provider_usage(
                audit,
                operation_id=opid,
                reservation_id=reservation_id,
                reservation_operation_id=reservation_operation_id,
                reserved_tokens=settlement_tokens,
                reserved_compute=settlement_compute,
                provider_result=settlement_result,
            )
            if not meter.ok:
                return CodexResult(meter.status, session=current, reason=meter.reason)
            if settlement_result is not provider_result:
                return self._provider_failure(
                    current,
                    action="resume",
                    operation_id=opid,
                    result={"reason": "live App Server usage telemetry is missing or malformed"},
                    request_digest=request_digest,
                    usage=settlement_result.get("usage")
                    if isinstance(settlement_result, Mapping)
                    else None,
                )
            if not self._provider_ok(provider_result, "complete", "success", "resumed", "started"):
                return self._provider_failure(
                    current,
                    action="resume",
                    operation_id=opid,
                    result=provider_result,
                    request_digest=request_digest,
                    usage=provider_result.get("usage"),
                )
            event = self._event(
                current,
                "resume",
                "complete",
                str(provider_result.get("message", "Codex session resumed")),
                opid,
            )
            updated = self._replace(current, events=(*current.events, event), status="active")
            with self._lock:
                self._sessions[updated.session_id] = updated
            receipt = self._receipt(
                updated,
                "resume",
                "complete",
                opid,
                usage=provider_result.get("usage"),
                request_digest=request_digest,
            )
            return CodexResult("complete", updated, receipt, (event,))
        except AuditCancelled as exc:
            return CodexResult("cancelled", session=current, reason=str(exc))
        except AuditContextConflict as exc:
            return CodexResult("conflict", session=current, reason=str(exc))
        except CapabilityUnavailable as exc:
            return CodexResult("unavailable", session=current, reason=str(exc))
        except (AuditServiceError, TypeError, ValueError, KeyError) as exc:
            return CodexResult("failed", session=current, reason=str(exc))

    def _cancel_durable(  # noqa: C901, PLR0912
        self,
        session: CodexSession | str,
        *,
        operation_id: str | None,
        reason: str,
        audit_token: str | None,
    ) -> CodexResult:
        """Resolve service kill authority before one durable provider cancel."""

        current: CodexSession | None = None
        checked_reason = reason if isinstance(reason, str) else str(reason)
        try:
            try:
                current = self._codex_session(session, audit_token=audit_token)
            except ValueError as exc:
                if not isinstance(session, CodexSession) or operation_id is None:
                    return CodexResult("conflict", reason=str(exc))
                current = self._cancel_retry_session(
                    session,
                    operation_id=operation_id,
                    reason=reason if isinstance(reason, str) else str(reason),
                )
                if current is None:
                    return CodexResult("conflict", reason=str(exc))
            audit = self._audit_for(current)
            if isinstance(session, CodexSession) and current.status != "cancelled":
                with self._lock:
                    self._cancel_handle_aliases.setdefault(current.session_id, session)
            owned = self.service.get_session(audit, token=audit_token)
            opid = operation_id or f"codex-op-{uuid.uuid4().hex}"
            if audit.session_token in opid or audit.session_token in checked_reason:
                return CodexResult(
                    "conflict",
                    session=current,
                    reason="cancel operation contains a sensitive value",
                )
            if current.status == "cancelled" and operation_id is not None:
                raw_operation = (
                    self.service.authority.snapshot()
                    .get("extensions", {})
                    .get("codex", {})
                    .get("operations", {})
                    .get(opid)
                )
                if (
                    isinstance(raw_operation, Mapping)
                    and raw_operation.get("action") == "cancel"
                    and raw_operation.get("status") == "finished"
                    and isinstance(raw_operation.get("request_digest"), str)
                    and isinstance(raw_operation.get("reason"), str)
                ):
                    if raw_operation["request_digest"] != self._cancel_request_digest(
                        current, checked_reason
                    ):
                        return CodexResult(
                            "conflict",
                            session=current,
                            reason="Codex cancel operation ID is bound to another request",
                        )
                    return self._durable_blocked_cancel(
                        audit,
                        current,
                        operation_id=opid,
                        request_digest=raw_operation["request_digest"],
                        reason=raw_operation["reason"],
                    )
            if current.status == "cancelled":
                return self._durable_blocked_cancel(
                    audit,
                    current,
                    operation_id=opid,
                    request_digest=self._cancel_request_digest(current, checked_reason),
                    reason="Codex session is already cancelled",
                )
            request_digest = self._cancel_request_digest(current, checked_reason)
            killed = self.service.kill_switch(audit, reason=checked_reason)
            if not killed.ok and killed.status != "cancelled":
                return CodexResult(killed.status, session=current, reason=killed.reason)
            if owned.context != current.context:
                return self._durable_cancel_conflict(
                    current,
                    operation_id=opid,
                    reason="audit selection context changed",
                    audit_token=audit_token,
                    request_reason=checked_reason,
                )
            try:
                self.service._assert_source(owned)
            except AuditContextConflict as exc:
                return self._durable_blocked_cancel(
                    audit,
                    current,
                    operation_id=opid,
                    request_digest=request_digest,
                    reason=str(exc),
                )
            if self.provider is None:
                return self._terminal_cancel(
                    current,
                    operation_id=opid,
                    request_digest=request_digest,
                    status="unavailable",
                    reason="no supported Codex provider is configured",
                )
            route = self.inspect_capabilities().choose(current.route.route_id)
            if route is None or route.to_dict() != current.route.to_dict():
                return CodexResult(
                    "unavailable",
                    session=current,
                    reason="current Codex route does not match session",
                )
            admission_expected = self._durable_admission_expectation(
                audit,
                request={
                    "action": "cancel",
                    "operation_id": opid,
                    "request_digest": request_digest,
                },
                context=current.context,
                route_digest=route.capability_digest,
                codex_session_id=current.session_id,
                reservation={"tokens": 0, "compute": 0.0, "issue_writes": 0},
            )
            lease = self.service.begin_codex_operation(
                audit,
                audit_token=audit_token,
                operation_id=opid,
                action="cancel",
                codex_session_id=current.session_id,
                request_digest=request_digest,
                context=current.context,
                route_digest=route.capability_digest,
                allow_cancelled=True,
            )
            if lease.status != "admitted":
                return self._recovered_result(lease, fallback_session=current)
            if not self._durable_admission_capability_valid(
                lease, provider=self.provider, expected=admission_expected
            ):
                return CodexResult(
                    "unavailable",
                    session=current,
                    reason=(
                        "durable Codex budget admission capability is missing or malformed; "
                        "the operation remains ambiguous"
                    ),
                )
            try:
                provider_result = self.provider.cancel(
                    provider_session_id=current.provider_session_id, route=route
                )
            except Exception as exc:  # noqa: BLE001 - admitted cancellation is ambiguous.
                return CodexResult(
                    "unavailable",
                    session=current,
                    reason=f"Codex provider cancel is ambiguous after admission: {exc}",
                )
            if (
                not isinstance(provider_result, Mapping)
                or self._valid_usage(provider_result) is None
            ):
                return CodexResult(
                    "unavailable",
                    session=current,
                    reason="provider usage is missing or malformed; Codex operation is ambiguous",
                )
            accepted = self._provider_ok(
                provider_result, "complete", "success", "cancelled", "started"
            )
            event = self._event(
                current,
                "cancel",
                "cancelled" if accepted else "failed",
                str(provider_result.get("message", checked_reason)),
                opid,
            )
            updated = self._replace(
                current,
                events=(*current.events, event),
                status="cancelled",
            )
            return self._finish_durable(
                audit,
                lease=lease,
                provider=self.provider,
                request_digest=request_digest,
                result_status="cancelled" if accepted else "failed",
                provider_result=provider_result,
                session=updated,
                provider_session_id=str(
                    provider_result.get("provider_session_id", current.provider_session_id)
                ),
            )
        except AuditCancelled as exc:
            return CodexResult("cancelled", session=current, reason=str(exc))
        except AuditContextConflict as exc:
            if current is None:
                return CodexResult("conflict", reason=str(exc))
            try:
                return self._durable_cancel_conflict(
                    current,
                    operation_id=operation_id or f"codex-op-{uuid.uuid4().hex}",
                    reason=str(exc),
                    audit_token=audit_token,
                    request_reason=checked_reason,
                )
            except (AuditServiceError, TypeError, ValueError, KeyError) as conflict_exc:
                return CodexResult("unavailable", session=current, reason=str(conflict_exc))
        except CapabilityUnavailable as exc:
            return CodexResult("unavailable", session=current, reason=str(exc))
        except AuditBudgetExceeded as exc:
            return CodexResult("denied", session=current, reason=str(exc))
        except AuditAuthorityError as exc:
            if current is None:
                return CodexResult("conflict", reason=str(exc))
            try:
                return self._durable_cancel_conflict(
                    current,
                    audit_token=audit_token,
                    operation_id=operation_id or f"codex-op-{uuid.uuid4().hex}",
                    reason=f"cancellation authority conflict: {exc}",
                    request_reason=checked_reason,
                )
            except (AuditServiceError, TypeError, ValueError, KeyError) as conflict_exc:
                return CodexResult("unavailable", session=current, reason=str(conflict_exc))
        except (AuditServiceError, TypeError, ValueError, KeyError) as exc:
            return CodexResult("conflict", session=current, reason=str(exc))

    def cancel(  # noqa: C901
        self,
        session: CodexSession | str,
        *,
        operation_id: str | None = None,
        reason: str = "Codex cancellation requested",
        audit_token: str | None = None,
    ) -> CodexResult:
        """Cancel provider work and retain the cancellation activity receipt."""

        return self._cancel_durable(
            session, operation_id=operation_id, reason=reason, audit_token=audit_token
        )

        try:
            current = self._codex_session(session, audit_token=audit_token)
        except ValueError as exc:
            return CodexResult("conflict", reason=str(exc))
        opid = operation_id or f"codex-op-{uuid.uuid4().hex}"
        request_digest = _digest(
            {
                "action": "cancel",
                "session_id": current.session_id,
                "provider_session_id": current.provider_session_id,
                "route": current.route.to_dict(),
                "context": current.context.to_dict(),
                "source_digest": current.source_digest,
                "source_revision": current.source_revision,
                "reason": reason,
            }
        )
        try:
            audit = self._audit_for(current)
            owned = self.service.get_session(audit)
        except (AuditServiceError, TypeError, ValueError, KeyError) as exc:
            return CodexResult("failed", session=current, reason=str(exc))
        replay = self._replayed_operation(opid, request_digest, session=current)
        if replay is not None:
            return replay
        try:
            # The service kill switch is authoritative and must run even when
            # the admitted source has gone stale.  No provider call follows
            # until both context and source are revalidated.
            # Let the service mint its own opaque kill operation identity.  The
            # caller's operation ID remains the Codex replay key, but it must
            # not be able to pre-bind or poison the authoritative kill record.
            killed = self.service.kill_switch(audit, reason=reason)
            if not killed.ok and killed.status != "cancelled":
                return CodexResult(killed.status, session=current, reason=killed.reason)
            if owned.context != current.context:
                return self._blocked_cancel(
                    current,
                    operation_id=opid,
                    request_digest=request_digest,
                    reason="audit selection context changed",
                )
            self.service._assert_source(owned)
            if self.provider is None:
                return self._terminal_cancel(
                    current,
                    operation_id=opid,
                    request_digest=request_digest,
                    status="unavailable",
                    reason="no supported Codex provider is configured",
                )
            reservation_operation_id = f"{opid}-reserve"
            reservation = self.service.reserve_budget(
                audit,
                operation_id=reservation_operation_id,
                enforce_active=False,
            )
            if not reservation.ok:
                return CodexResult(reservation.status, session=current, reason=reservation.reason)
            reservation_id = self._reservation_id(reservation)
            if reservation_id is None:
                released = self._release_lost_reservation_reply(
                    audit,
                    operation_id=opid,
                    reservation_operation_id=reservation_operation_id,
                )
                return CodexResult(
                    "unavailable",
                    session=current,
                    reason="budget reservation capability is missing"
                    if released
                    else "budget reservation capability is missing; hold remains unresolved",
                )
            try:
                provider_result = self.provider.cancel(
                    provider_session_id=current.provider_session_id, route=current.route
                )
            except Exception as exc:  # noqa: BLE001 - provider failures are typed and receipted.
                self._settle_provider_usage(
                    audit,
                    operation_id=opid,
                    reservation_id=reservation_id,
                    reservation_operation_id=reservation_operation_id,
                    reserved_tokens=0,
                    reserved_compute=0.0,
                    provider_result=None,
                )
                return self._terminal_cancel(
                    current,
                    operation_id=opid,
                    request_digest=request_digest,
                    status="failed",
                    reason=f"provider cancel failed: {exc}",
                )
            meter = self._settle_provider_usage(
                audit,
                operation_id=opid,
                reservation_id=reservation_id,
                reservation_operation_id=reservation_operation_id,
                reserved_tokens=0,
                reserved_compute=0.0,
                provider_result=provider_result,
            )
            if not meter.ok:
                return self._terminal_cancel(
                    current,
                    operation_id=opid,
                    request_digest=request_digest,
                    status=meter.status,
                    reason=meter.reason,
                )
            if not self._provider_ok(
                provider_result, "complete", "success", "cancelled", "started"
            ):
                return self._terminal_cancel(
                    current,
                    operation_id=opid,
                    request_digest=request_digest,
                    status="failed",
                    reason=str(provider_result.get("reason", "provider cancel failed")),
                )
            event = self._event(
                current, "cancel", "cancelled", str(provider_result.get("message", reason)), opid
            )
            updated = self._replace(current, events=(*current.events, event), status="cancelled")
            with self._lock:
                self._sessions[updated.session_id] = updated
            receipt = self._receipt(
                updated,
                "cancel",
                "cancelled",
                opid,
                reason=reason,
                usage=provider_result.get("usage"),
                request_digest=request_digest,
            )
            return CodexResult("cancelled", updated, receipt, (event,), reason)
        except AuditCancelled as exc:
            return CodexResult("cancelled", session=current, reason=str(exc))
        except AuditContextConflict as exc:
            return self._blocked_cancel(
                current,
                operation_id=opid,
                request_digest=request_digest,
                reason=str(exc),
            )
        except CapabilityUnavailable as exc:
            return CodexResult("unavailable", session=current, reason=str(exc))
        except (AuditServiceError, TypeError, ValueError, KeyError) as exc:
            return CodexResult("failed", session=current, reason=str(exc))

    @staticmethod
    def _replace(session: CodexSession, **changes: Any) -> CodexSession:
        """Rebuild a frozen session while retaining exact route/evidence identity."""

        values = {
            "session_id": session.session_id,
            "audit_session_id": session.audit_session_id,
            "context": session.context,
            "route": session.route,
            "evidence": session.evidence,
            "provider_session_id": session.provider_session_id,
            "source_digest": session.source_digest,
            "source_revision": session.source_revision,
            "status": session.status,
            "events": session.events,
            "created_at": session.created_at,
        }
        values.update(changes)
        return CodexSession(**values)

    resume_session = resume
    reconnect_session = reconnect
    cancel_session = cancel


# Friendly aliases used by thin clients and tests.
CodexClient = AuditCodexClient
FakeProvider = FakeCodexProvider
inspect_capabilities = inspect_installed_capabilities


__all__ = [
    "AUDIT_CODEX_CAPABILITY_SCHEMA_VERSION",
    "AUDIT_CODEX_EVENT_SCHEMA_VERSION",
    "AUDIT_CODEX_ROUTE_SCHEMA_VERSION",
    "AUDIT_CODEX_SCHEMA_VERSION",
    "AuditCodexClient",
    "CodexActivityEvent",
    "CodexCapabilityInspection",
    "CodexClient",
    "CodexEvidenceRef",
    "CodexOperationReceipt",
    "CodexProvider",
    "CodexResult",
    "CodexRouteReceipt",
    "CodexSession",
    "FakeCodexProvider",
    "FakeProvider",
    "inspect_capabilities",
    "inspect_installed_capabilities",
]
