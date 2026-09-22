"""Focused BA-06 fixture/UI adapter tests (diagnostic-only, not native evidence)."""

from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from robot_sf.analysis_workbench.audit_contracts import Annotation
from robot_sf.analysis_workbench.audit_store import StoredRecord
from robot_sf.analysis_workbench.review_contracts import SourceRef
from robot_sf.render.audit_workbench import (
    AuditWorkbenchConflictError,
    AuditWorkbenchError,
    AuditWorkbenchUnavailableError,
    FixtureAuditService,
    ServiceAuditWorkbenchFacade,
    _safe_service_value,
    build_audit_workbench_document,
    fixture_document,
    write_fixture_workbench,
)


@dataclass(frozen=True)
class _FakeContext:
    """Small BA-05-shaped context used without importing the pending service."""

    context_revision: int = 4
    source_revision: str = "source-revision-1"
    source_identity: str = "source-digest-1"
    execution_id: str = "execution-real"
    scenario_id: str = "scenario-real"
    episode_id: str = ""


@dataclass(frozen=True)
class _FakeOperation:
    operation_id: str
    operation_type: str
    status: str = "committed"
    replayed: bool = False


@dataclass(frozen=True)
class _FakeResult:
    status: str
    value: Any = None
    reason: str = ""
    operation: Any = None
    context: Any = None

    def to_dict(self) -> dict[str, Any]:
        """Match the safe BA-05 ServiceResult envelope shape."""

        return {
            "schema_version": "audit-service.v1",
            "status": self.status,
            "reason": self.reason,
            "value": self.value,
            "operation": self.operation,
            "context": self.context,
        }


@dataclass(frozen=True)
class _FakeCapabilityResult:
    """BA-05-shaped capability envelope nested inside ``ServiceResult.value``."""

    capability: str
    status: str
    value: Any = None
    reason: str = ""


@dataclass(frozen=True)
class _FakePacket:
    packet_id: str
    primary: Any
    selection_reasons: tuple[str, ...] = ("detector_signal",)
    missingness: tuple[str, ...] = ()


@dataclass(frozen=True)
class _FakePrimary:
    episode_id: str
    execution_id: str = "execution-real"
    source_digest: str = "source-digest-1"


@dataclass(frozen=True)
class _FakeQueueResult:
    packet: _FakePacket
    state_revision: int = 17
    input_revision: int = 23
    input_identity: str = "queue-input-23"


class _FakeSession:
    """Server-side handle with a deliberately searchable secret token."""

    session_id = "session-real"
    session_token = "do-not-serialize-session-token"
    actor = type("Actor", (), {"actor_id": "reviewer"})()
    context = _FakeContext()


class _FakeAuditService:
    """Typed-envelope fake that records the exact authority/CAS calls."""

    def __init__(self, *, episode_status: str = "complete") -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.episode_status = episode_status
        self.context = _FakeContext()
        self.next_result: Any = None
        self.annotation_result: Any = None
        self.finding_result: Any = None
        self.update_context_result: Any = None
        self.episode_context: Any = None
        self.queue_state_revision = 17
        self.queue_input_revision = 23
        self.queue_input_identity = "queue-input-23"
        self.nested_queue_result = False
        self.read_queue_fields: dict[str, Any] | None = None

    def _record(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.calls.append((name, args, kwargs))

    def next(self, session: Any, **kwargs: Any) -> Any:
        self._record("next", session, **kwargs)
        if self.next_result is not None:
            return self.next_result
        packet = _FakePacket("packet-real", _FakePrimary("episode-real"))
        return _FakeResult(
            "complete",
            _FakeQueueResult(packet),
            operation=_FakeOperation("next-1", "queue.next", "complete"),
            context=self.context,
        )

    def read_episode(self, session: Any, episode_id: str, **kwargs: Any) -> _FakeResult:
        self._record("read_episode", session, episode_id, **kwargs)
        returned_context = self.episode_context or self.context
        if self.episode_status != "complete":
            return _FakeResult(
                self.episode_status,
                {
                    "episode_id": episode_id,
                    "status": "missing",
                    "reason": "recording_not_present",
                    "row": None,
                },
                reason="recording_not_present",
                operation=_FakeOperation("episode-1", "read.episode", self.episode_status),
                context=returned_context,
            )
        return _FakeResult(
            "complete",
            {
                "episode_id": episode_id,
                "status": "readable",
                "row": {"metrics": {"clearance": 0.4}, "events": []},
            },
            operation=_FakeOperation("episode-1", "read.episode", "complete"),
            context=returned_context,
        )

    def write_annotation(self, session: Any, annotation: Any, **kwargs: Any) -> _FakeResult:
        self._record("write_annotation", session, annotation, **kwargs)
        if self.annotation_result is not None:
            return self.annotation_result
        operation_id = kwargs.get("operation_id") or "annotation-1"
        return _FakeResult(
            "committed",
            {
                "committed": True,
                "operation_id": operation_id,
                "record_id": annotation["annotation_id"],
                "record_type": "annotation",
                "revision": 1,
                "global_revision": 9,
                "deleted": False,
                "replayed": False,
            },
            operation=_FakeOperation(operation_id, "write.annotation"),
            context=self.context,
        )

    def finding_from_annotation(self, session: Any, annotation: Any, **kwargs: Any) -> _FakeResult:
        self._record("finding_from_annotation", session, annotation, **kwargs)
        if self.finding_result is not None:
            return self.finding_result
        operation_id = kwargs.get("operation_id") or "finding-1"
        return _FakeResult(
            "committed",
            {
                "committed": True,
                "operation_id": operation_id,
                "record_id": kwargs["finding_id"],
                "record_type": "finding",
                "revision": 1,
                "global_revision": 10,
                "deleted": False,
                "candidate_members": [annotation["episode_id"]],
                "confirmed_members": [],
            },
            operation=_FakeOperation(operation_id, "write.finding"),
            context=self.context,
        )

    def read_context(self, session: Any, **kwargs: Any) -> _FakeResult:
        self._record("read_context", session, **kwargs)
        return _FakeResult("complete", self.context, context=self.context)

    def update_context(self, session: Any, context: Any, **kwargs: Any) -> _FakeResult:
        self._record("update_context", session, context, **kwargs)
        if self.update_context_result is not None:
            return self.update_context_result
        self.context = context
        return _FakeResult("complete", context, context=context)

    def read_queue(self, session: Any, **kwargs: Any) -> _FakeResult:
        self._record("read_queue", session, **kwargs)
        queue_value: Any = {
            "items": [],
            "state_revision": self.queue_state_revision,
            "input_revision": self.queue_input_revision,
            "input_identity": self.queue_input_identity,
        }
        if self.read_queue_fields is not None:
            queue_value = {"items": [], **self.read_queue_fields}
        if self.nested_queue_result:
            queue_value = _FakeCapabilityResult(
                "ba-02.queue",
                "complete",
                queue_value,
            )
        return _FakeResult(
            "complete",
            queue_value,
            context=self.context,
        )

    def read_coverage(self, session: Any, **kwargs: Any) -> _FakeResult:
        self._record("read_coverage", session, **kwargs)
        return _FakeResult("complete", {"reviewed": 0, "remaining": 1}, context=self.context)


@dataclass(frozen=True)
class _FakeCodexSession:
    """Private Codex handle retained only by the server facade."""

    session_id: str = "codex-session-private"
    provider_session_id: str = "provider-session-private"


class _FakeCodexClient:
    """Recording narrow client for the facade lifecycle tests."""

    def __init__(self, service: Any) -> None:
        self.service = service
        self.calls: list[tuple[str, Any, dict[str, Any]]] = []
        self.session = _FakeCodexSession()
        self.start_entered = threading.Event()
        self.release_start = threading.Event()
        self.block_start = False
        self.foreign_start_binding = False
        self.foreign_cancel_binding = False
        self.foreign_reconnect_binding = False
        self.foreign_reconnect_operation_binding = False
        self.missing_reconnect_binding = False
        self.partial_reconnect_source_binding = False
        self.conflicting_nested_start_binding = False
        self.conflicting_nested_cancel_binding = False
        self.conflicting_nested_reconnect_binding = False
        self.reason_text: str | None = None
        self.activity_text: str | None = None

    def start(self, session: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("start", session, kwargs))
        if self.block_start:
            self.start_entered.set()
            assert self.release_start.wait(timeout=5)
        result: dict[str, Any] = {
            "status": "complete",
            "operation_id": (
                "foreign-operation" if self.foreign_start_binding else kwargs["operation_id"]
            ),
            "session": self.session,
            "context": (
                _FakeContext(context_revision=99, episode_id="foreign-episode")
                if self.foreign_start_binding
                else kwargs["context"]
            ),
            "source": {
                "source_revision": (
                    "foreign-source-revision" if self.foreign_start_binding else "source-revision-1"
                ),
                "source_digest": (
                    "foreign-source-digest" if self.foreign_start_binding else "source-digest-1"
                ),
            },
            "route": {
                "route_id": "route-observed",
                "provider": "private-provider",
                "model_id": "private-model",
            },
            "provider_session_id": self.session.provider_session_id,
            "source_path": "/private/codex/source.json",
            "receipt": {
                "provider_session_id": self.session.provider_session_id,
                "request_digest": "private-request-digest",
                "usage": {"tokens": 4, "compute": 1.0},
            },
            "evidence": [
                {
                    "evidence_id": "evidence-safe",
                    "locator": "/private/codex/source.json#episode-real",
                }
            ],
            "usage": {"tokens": 4, "compute": 1.0},
            "events": [
                {
                    "message": self.activity_text or "Codex diagnostic completed",
                    "operation_id": kwargs["operation_id"],
                }
            ],
        }
        if self.conflicting_nested_start_binding:
            result["session"] = {
                "context": _FakeContext(context_revision=99, episode_id="foreign-episode"),
                "source_revision": "foreign-source-revision",
                "source_digest": "foreign-source-digest",
                "last_operation_id": "foreign-operation",
            }
        if self.reason_text is not None:
            result["reason"] = self.reason_text
        return result

    def reconnect(
        self, codex_session_id: str, *, operation_id: str, audit_token: str
    ) -> dict[str, Any]:
        self.calls.append(
            (
                "reconnect",
                codex_session_id,
                {"operation_id": operation_id, "audit_token": audit_token},
            )
        )
        result: dict[str, Any] = {
            "status": "complete",
            "session": self.session,
            "codex_session_id": (
                "foreign-session" if self.foreign_reconnect_binding else codex_session_id
            ),
            "operation_id": operation_id,
            "context": self.service.context,
            "source": {
                "source_revision": "source-revision-1",
                "source_digest": "source-digest-1",
            },
            "receipt": {
                "operation_id": (
                    "foreign-operation"
                    if self.foreign_reconnect_operation_binding
                    else operation_id
                ),
                "source_revision": "source-revision-1",
                "source_digest": "source-digest-1",
            },
            "provider_session_id": self.session.provider_session_id,
        }
        if self.missing_reconnect_binding:
            result.pop("context")
            result.pop("source")
        if self.partial_reconnect_source_binding:
            result["source"] = {"source_digest": "source-digest-1"}
            result["receipt"].pop("source_revision")
        if self.conflicting_nested_reconnect_binding:
            result["session"] = {
                "session_id": codex_session_id,
                "context": _FakeContext(context_revision=99, episode_id="foreign-episode"),
                "source_revision": "foreign-source-revision",
                "source_digest": "foreign-source-digest",
            }
        return result

    def cancel(self, session: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(("cancel", session, kwargs))
        result: dict[str, Any] = {
            "status": "cancelled",
            "operation_id": (
                "foreign-operation" if self.foreign_cancel_binding else kwargs["operation_id"]
            ),
            "session": self.session,
            "provider_session_id": self.session.provider_session_id,
            "reason": kwargs["reason"],
            "usage": {"tokens": 0, "compute": 0.0},
        }
        if self.foreign_cancel_binding:
            result["context"] = _FakeContext(context_revision=99, episode_id="foreign-episode")
            result["source"] = {
                "source_revision": "foreign-source-revision",
                "source_digest": "foreign-source-digest",
            }
        if self.conflicting_nested_cancel_binding:
            result["context"] = self.service.context
            result["source"] = {
                "source_revision": "source-revision-1",
                "source_digest": "source-digest-1",
            }
            result["session"] = {
                "context": _FakeContext(context_revision=99, episode_id="foreign-episode"),
                "source_revision": "foreign-source-revision",
                "source_digest": "foreign-source-digest",
                "last_operation_id": "foreign-operation",
            }
        return result


def test_service_facade_projects_typed_next_and_preserves_missingness() -> None:
    """A typed queue packet is bound to read_episode without fixture media."""
    service = _FakeAuditService(episode_status="unavailable")
    facade = ServiceAuditWorkbenchFacade(service, _FakeSession(), token=_FakeSession.session_token)

    result = facade.next(
        expected_context_revision=4,
        expected_queue_state_revision=16,
        expected_queue_input_revision=22,
        operation_id="next-1",
    )

    assert result["status"] == "complete"
    assert result["selection_status"] == "complete"
    assert result["inspection_status"] == "unavailable"
    assert result["presentation_status"] == "unavailable"
    assert result["episode"]["status"] == "missing"
    assert result["episode"]["row"] is None
    assert result["queue_state_revision"] == 17
    assert result["queue_input_revision"] == 23
    assert result["context_revision"] == 4
    assert result["packet"]["episode_id"] == "episode-real"
    assert "editor_model" not in result["packet"]
    assert "media" not in result["packet"]
    assert _FakeSession.session_token not in json.dumps(result, sort_keys=True)

    next_call = service.calls[0]
    assert next_call[0] == "next"
    assert next_call[2]["expected_context_revision"] == 4
    assert next_call[2]["expected_queue_state_revision"] == 16
    assert next_call[2]["expected_queue_input_revision"] == 22
    assert next_call[2]["token"] == _FakeSession.session_token
    assert service.calls[1][0] == "read_episode"
    assert service.calls[1][1][1] == "episode-real"


def test_service_facade_projects_selected_artifact_status_without_authority_leakage() -> None:
    """Status is service-owned, diagnostic-only, and path-safe at the facade."""

    service = _FakeAuditService()
    service.context = _FakeContext(episode_id="episode-real")
    status_context = {
        "context_revision": 4,
        "episode_id": "episode-real",
        "source_revision": "source-revision-1",
        "source_identity": "/private/source-identity",
    }

    def read_status(session: Any, *, token: str, context: Any) -> _FakeResult:
        service._record("read_selected_artifact_status", session, token=token, context=context)
        return _FakeResult(
            "complete",
            {
                "schema_version": "audit-artifact-status.v1",
                "episode_id": "episode-real",
                "context_revision": 4,
                "source_revision": "source-revision-1",
                "source_digest": "/private/source-digest",
                "receipt": {"operation_id": "private-receipt"},
                "materialization": {
                    "status": "not_configured",
                    "classification": "derived_render",
                    "fidelity": "unverifiable",
                    "reason": "/private/output-root is not configured",
                    "result_status": "/private/result",
                    "diagnostic_only": True,
                },
                "native_diagnostic": {
                    "status": "not_configured",
                    "reason": "native binding is not configured",
                    "evidence_boundary": "diagnostic_only",
                    "scientific_claim_allowed": False,
                },
            },
            context=status_context,
        )

    service.read_selected_artifact_status = read_status  # type: ignore[attr-defined]
    session = _FakeSession()
    session.context = service.context
    facade = ServiceAuditWorkbenchFacade(service, session, token=_FakeSession.session_token)

    result = facade.read_selected_artifact_status()

    assert result["status"] == "conflict"
    assert result["value"]["episode_id"] == "episode-real"
    assert result["value"]["materialization"]["status"] == "unavailable"
    assert result["value"]["materialization"]["classification"] is None
    assert result["value"]["materialization"]["fidelity"] is None
    assert result["value"]["materialization"]["reason"] == "selected artifact status is unavailable"
    assert "result_status" not in result["value"]["materialization"]
    assert result["value"]["native_diagnostic"]["scientific_claim_allowed"] is False
    assert result["context"] == {
        "context_revision": 4,
        "episode_id": "episode-real",
        "source_revision": "source-revision-1",
    }
    serialized = json.dumps(result, sort_keys=True)
    assert "/private" not in serialized
    assert "operation" not in result
    assert "receipt" not in serialized
    status_call = next(call for call in service.calls if call[0] == "read_selected_artifact_status")
    assert status_call[2]["token"] == _FakeSession.session_token
    assert status_call[2]["context"].episode_id == "episode-real"
    snapshot = facade.snapshot()
    assert snapshot["artifact_status"]["materialization"]["status"] == "unavailable"
    assert snapshot["artifact_status_result"]["status"] == "conflict"


def test_service_facade_rejects_token_in_artifact_status_reason_and_nested_fields() -> None:
    """Status projection must not echo a server-held token from any status field."""

    service = _FakeAuditService()
    service.context = _FakeContext(episode_id="episode-real")
    token = _FakeSession.session_token

    def read_status(session: Any, *, token: str, context: Any) -> _FakeResult:
        return _FakeResult(
            "complete",
            {
                "episode_id": "episode-real",
                "context_revision": 4,
                "source_revision": "source-revision-1",
                "source_digest": "source-digest-1",
                "materialization": {
                    "status": "available",
                    "reason": token,
                    "result_status": token,
                    "classification": "historical_original",
                    "fidelity": "verified",
                },
                "native_diagnostic": {"status": "available", "reason": "configured"},
            },
            reason=token,
            context=service.context,
        )

    service.read_selected_artifact_status = read_status  # type: ignore[attr-defined]
    session = _FakeSession()
    session.context = service.context
    facade = ServiceAuditWorkbenchFacade(service, session, token=token)

    result = facade.read_selected_artifact_status()

    assert result["status"] == "conflict"
    assert result["value"]["materialization"]["status"] == "unavailable"
    assert result["value"]["materialization"]["classification"] is None
    assert result["value"]["materialization"]["fidelity"] is None
    assert token not in json.dumps(result, sort_keys=True)

    snapshot = facade.snapshot()
    assert snapshot["artifact_status"]["materialization"]["status"] == "unavailable"
    assert token not in json.dumps(snapshot, sort_keys=True)


def test_service_facade_stale_artifact_status_does_not_claim_capability() -> None:
    """A stale status envelope remains unavailable and carries no action path."""

    service = _FakeAuditService()
    service.context = _FakeContext(episode_id="episode-real")
    service.read_selected_artifact_status = lambda session, *, token, context: _FakeResult(
        "conflict",
        reason="selected context is stale",
        context=service.context,
    )  # type: ignore[attr-defined]
    session = _FakeSession()
    session.context = service.context
    facade = ServiceAuditWorkbenchFacade(service, session, token=_FakeSession.session_token)

    result = facade.read_selected_artifact_status()

    assert result["status"] == "conflict"
    assert result["value"]["materialization"]["status"] == "unavailable"
    assert "run" not in json.dumps(result, sort_keys=True).lower()


def test_service_facade_rejects_foreign_delayed_artifact_status_without_context_drift() -> None:
    """A delayed status cannot replace the selected episode or source binding."""

    service = _FakeAuditService()
    service.context = _FakeContext(episode_id="episode-real")

    def read_status(session: Any, *, token: str, context: Any) -> _FakeResult:
        return _FakeResult(
            "complete",
            {
                "episode_id": "foreign-episode",
                "context_revision": 99,
                "source_revision": "foreign-source-revision",
                "source_digest": "foreign-source-digest",
                "materialization": {
                    "status": "available",
                    "reason": "retained trace available",
                    "classification": "historical_original",
                    "fidelity": "verified",
                },
                "native_diagnostic": {
                    "status": "available",
                    "reason": "native binding is configured",
                },
            },
            context=_FakeContext(
                context_revision=99,
                episode_id="foreign-episode",
                source_revision="foreign-source-revision",
                source_identity="foreign-source-digest",
            ),
        )

    service.read_selected_artifact_status = read_status  # type: ignore[attr-defined]
    session = _FakeSession()
    session.context = service.context
    facade = ServiceAuditWorkbenchFacade(service, session, token=_FakeSession.session_token)

    result = facade.read_selected_artifact_status()

    assert result["status"] == "conflict"
    assert result["value"]["episode_id"] == "episode-real"
    assert result["value"]["materialization"]["status"] == "unavailable"
    assert result["value"]["native_diagnostic"]["status"] == "unavailable"
    assert result["value"]["materialization"]["classification"] is None
    assert facade._context.episode_id == "episode-real"


def test_service_facade_rejects_conflicting_source_identity_aliases() -> None:
    """A foreign source_digest cannot be hidden behind a matching source_identity."""

    service = _FakeAuditService()
    service.context = _FakeContext(episode_id="episode-real")
    service.read_selected_artifact_status = lambda session, *, token, context: _FakeResult(
        "complete",
        {
            "episode_id": "episode-real",
            "context_revision": 4,
            "source_revision": "source-revision-1",
            "source_digest": "source-digest-1",
            "materialization": {
                "status": "available",
                "reason": "retained trace available",
                "classification": "historical_original",
                "fidelity": "verified",
            },
            "native_diagnostic": {"status": "available", "reason": "configured"},
        },
        context={
            "context_revision": 4,
            "episode_id": "episode-real",
            "source_revision": "source-revision-1",
            "source_identity": "source-digest-1",
            "source_digest": "foreign-digest",
        },
    )  # type: ignore[attr-defined]
    session = _FakeSession()
    session.context = service.context
    facade = ServiceAuditWorkbenchFacade(service, session, token=_FakeSession.session_token)

    result = facade.read_selected_artifact_status()

    assert result["status"] == "conflict"
    assert result["value"]["materialization"]["status"] == "unavailable"
    assert result["value"]["materialization"]["classification"] is None


def test_fixture_snapshot_rejects_foreign_artifact_status() -> None:
    """Fixture reopen/snapshot paths cannot replay status for another episode."""

    document = fixture_document()
    document["artifact_status"] = {
        "episode_id": "foreign-episode",
        "context_revision": 99,
        "source_revision": "foreign-source",
        "source_digest": "foreign-digest",
        "materialization": {
            "status": "available",
            "reason": "retained trace available",
            "classification": "historical_original",
            "fidelity": "verified",
        },
        "native_diagnostic": {"status": "available", "reason": "configured"},
    }
    service = FixtureAuditService(document)
    selected = service.next(expected_selection_revision=0)
    assert selected["status"] == "selected"

    status = service.snapshot()["artifact_status"]

    assert status["episode_id"] == selected["packet"]["episode_id"]
    assert status["materialization"]["status"] == "unavailable"
    assert status["materialization"]["classification"] is None
    assert status["native_diagnostic"]["status"] == "unavailable"


def test_fixture_artifact_status_requires_nested_source_binding() -> None:
    """Fixture status cannot claim availability without the selected source binding."""

    document = fixture_document()
    document["artifact_status"] = {
        "episode_id": "fixture-normal-control",
        "context_revision": 1,
        "source_revision": "foreign-source",
        "source_digest": "foreign-digest",
        "materialization": {
            "status": "available",
            "reason": "retained trace available",
            "classification": "historical_original",
            "fidelity": "verified",
        },
        "native_diagnostic": {"status": "available", "reason": "configured"},
    }
    service = FixtureAuditService(document)

    before_selection = service.snapshot()["artifact_status"]
    assert before_selection["episode_id"] is None
    assert before_selection["materialization"]["status"] == "no_selection"
    assert before_selection["materialization"]["classification"] is None
    assert before_selection["native_diagnostic"]["status"] == "no_selection"

    service.next(expected_selection_revision=0)
    selected = service.snapshot()["artifact_status"]
    assert selected["episode_id"] == "fixture-normal-control"
    assert selected["materialization"]["status"] == "unavailable"
    assert selected["materialization"]["classification"] is None
    assert selected["materialization"]["fidelity"] is None
    assert selected["native_diagnostic"]["status"] == "unavailable"

    missing_source_document = fixture_document()
    missing_source_document["artifact_status"] = {
        "episode_id": "fixture-normal-control",
        "context_revision": 1,
        "materialization": {
            "status": "available",
            "reason": "retained trace available",
            "classification": "historical_original",
            "fidelity": "verified",
        },
        "native_diagnostic": {"status": "available", "reason": "configured"},
    }
    missing_source_service = FixtureAuditService(missing_source_document)
    missing_source_service.next(expected_selection_revision=0)
    missing_source = missing_source_service.snapshot()["artifact_status"]
    assert missing_source["materialization"]["status"] == "unavailable"
    assert missing_source["materialization"]["classification"] is None
    assert missing_source["native_diagnostic"]["status"] == "unavailable"


def test_fixture_artifact_status_accepts_matching_nested_source_binding() -> None:
    """A status matching the editor model's nested source identity remains readable."""

    document = fixture_document()
    source = document["queue"][0]["editor_model"]
    document["artifact_status"] = {
        "episode_id": "fixture-normal-control",
        "context_revision": 1,
        "source_revision": source["storyboard"]["source_revision"],
        "source_digest": source["source_identity"]["sources"]["scene"]["sha256"],
        "materialization": {
            "status": "available",
            "reason": "retained trace available",
            "classification": "historical_original",
            "fidelity": "verified",
        },
        "native_diagnostic": {"status": "available", "reason": "configured"},
    }
    service = FixtureAuditService(document)
    service.next(expected_selection_revision=0)

    status = service.snapshot()["artifact_status"]
    assert status["materialization"]["status"] == "available"
    assert status["materialization"]["classification"] == "historical_original"
    assert status["materialization"]["fidelity"] == "verified"
    assert status["native_diagnostic"]["status"] == "available"


@pytest.mark.parametrize(
    "conflicting_field", ["context_revision", "source_revision", "source_digest"]
)
def test_fixture_artifact_status_rejects_conflicting_binding_aliases(
    conflicting_field: str,
) -> None:
    """Fixture binding refuses packet aliases that disagree with nested editor authority."""

    document = fixture_document()
    packet = document["queue"][0]
    source = packet["editor_model"]
    packet[conflicting_field] = {
        "context_revision": 99,
        "source_revision": "packet-source-revision",
        "source_digest": "packet-source-digest",
    }[conflicting_field]
    document["artifact_status"] = {
        "episode_id": "fixture-normal-control",
        "context_revision": packet.get("context_revision", 1),
        "source_revision": packet.get("source_revision", source["storyboard"]["source_revision"]),
        "source_digest": packet.get(
            "source_digest", source["source_identity"]["sources"]["scene"]["sha256"]
        ),
        "materialization": {
            "status": "available",
            "reason": "retained trace available",
            "classification": "historical_original",
            "fidelity": "verified",
        },
        "native_diagnostic": {"status": "available", "reason": "configured"},
    }
    service = FixtureAuditService(document)
    service.next(expected_selection_revision=0)

    status = service.snapshot()["artifact_status"]

    assert status["materialization"]["status"] == "unavailable"
    assert status["materialization"]["classification"] is None
    assert status["materialization"]["fidelity"] is None
    assert status["native_diagnostic"]["status"] == "unavailable"


@pytest.mark.parametrize("status", ["conflict", "unavailable", "failed"])
def test_service_facade_artifact_status_drops_non_success_values(status: str) -> None:
    """Non-success envelopes cannot carry a positive artifact capability value."""

    service = _FakeAuditService()
    service.context = _FakeContext(episode_id="episode-real")
    value = {
        "episode_id": "episode-real",
        "context_revision": 4,
        "source_revision": "source-revision-1",
        "source_digest": "source-digest-1",
        "materialization": {
            "status": "available",
            "reason": "configured",
            "classification": "historical_original",
            "fidelity": "verified",
        },
        "native_diagnostic": {"status": "available", "reason": "configured"},
    }
    service.read_selected_artifact_status = lambda session, *, token, context: _FakeResult(
        status, value=value, reason="status is not available", context=service.context
    )  # type: ignore[attr-defined]
    session = _FakeSession()
    session.context = service.context
    facade = ServiceAuditWorkbenchFacade(service, session, token=_FakeSession.session_token)

    result = facade.read_selected_artifact_status()

    assert result["status"] == status
    assert result["value"]["materialization"]["status"] == "unavailable"
    assert result["value"]["materialization"]["classification"] is None
    assert result["value"]["materialization"]["fidelity"] is None
    assert result["value"]["native_diagnostic"]["status"] == "unavailable"


def test_service_facade_artifact_status_rejects_generic_saved_result() -> None:
    """A generic write-result status cannot admit an artifact capability value."""

    service = _FakeAuditService()
    service.context = _FakeContext(episode_id="episode-real")
    value = {
        "episode_id": "episode-real",
        "context_revision": 4,
        "source_revision": "source-revision-1",
        "source_digest": "source-digest-1",
        "materialization": {
            "status": "available",
            "reason": "configured",
            "classification": "historical_original",
            "fidelity": "verified",
        },
        "native_diagnostic": {"status": "available", "reason": "configured"},
    }
    service.read_selected_artifact_status = lambda session, *, token, context: _FakeResult(
        "saved", value=value, reason="saved by a generic write operation", context=service.context
    )  # type: ignore[attr-defined]
    session = _FakeSession()
    session.context = service.context
    facade = ServiceAuditWorkbenchFacade(service, session, token=_FakeSession.session_token)

    result = facade.read_selected_artifact_status()

    assert result["status"] == "unavailable"
    assert result["value"]["materialization"]["status"] == "unavailable"
    assert result["value"]["materialization"]["classification"] is None
    assert result["value"]["native_diagnostic"]["status"] == "unavailable"


def test_service_facade_artifact_status_rejects_encoded_reason() -> None:
    """Percent-encoded path separators cannot preserve a positive status value."""

    service = _FakeAuditService()
    service.context = _FakeContext(episode_id="episode-real")
    value = {
        "episode_id": "episode-real",
        "context_revision": 4,
        "source_revision": "source-revision-1",
        "source_digest": "source-digest-1",
        "materialization": {
            "status": "available",
            "reason": "configured",
            "classification": "historical_original",
            "fidelity": "verified",
        },
        "native_diagnostic": {"status": "available", "reason": "configured"},
    }
    service.read_selected_artifact_status = lambda session, *, token, context: _FakeResult(
        "complete",
        value=value,
        reason="file:%2Fprivate%2Fstatus",
        context=service.context,
    )  # type: ignore[attr-defined]
    session = _FakeSession()
    session.context = service.context
    facade = ServiceAuditWorkbenchFacade(service, session, token=_FakeSession.session_token)

    result = facade.read_selected_artifact_status()

    assert result["status"] == "conflict"
    assert result["value"]["materialization"]["status"] == "unavailable"
    assert result["value"]["materialization"]["classification"] is None
    assert "%2F" not in json.dumps(result, sort_keys=True)


def test_service_facade_artifact_status_rejects_encoded_source_reference() -> None:
    """Percent-encoded source identities cannot satisfy the selected binding."""

    service = _FakeAuditService()
    service.context = _FakeContext(episode_id="episode-real")
    value = {
        "episode_id": "episode-real",
        "context_revision": 4,
        "source_revision": "source-revision-1",
        "source_digest": "source-digest-%2Fprivate",
        "materialization": {
            "status": "available",
            "reason": "configured",
            "classification": "historical_original",
            "fidelity": "verified",
        },
        "native_diagnostic": {"status": "available", "reason": "configured"},
    }
    service.read_selected_artifact_status = lambda session, *, token, context: _FakeResult(
        "complete", value=value, reason="safe", context=service.context
    )  # type: ignore[attr-defined]
    session = _FakeSession()
    session.context = service.context
    facade = ServiceAuditWorkbenchFacade(service, session, token=_FakeSession.session_token)

    result = facade.read_selected_artifact_status()

    assert result["status"] == "conflict"
    assert result["value"]["materialization"]["status"] == "unavailable"
    assert result["value"]["materialization"]["classification"] is None
    assert "%2F" not in json.dumps(result, sort_keys=True)


@pytest.mark.parametrize(
    ("field", "malformed"),
    [("status", []), ("status", {}), ("classification", []), ("fidelity", {})],
)
def test_service_facade_artifact_status_projects_malformed_capability_fields(
    field: str, malformed: Any
) -> None:
    """Malformed capability fields downgrade safely instead of raising."""

    service = _FakeAuditService()
    service.context = _FakeContext(episode_id="episode-real")
    materialization = {
        "status": "available",
        "reason": "configured",
        "classification": "historical_original",
        "fidelity": "verified",
    }
    materialization[field] = malformed
    value = {
        "episode_id": "episode-real",
        "context_revision": 4,
        "source_revision": "source-revision-1",
        "source_digest": "source-digest-1",
        "materialization": materialization,
        "native_diagnostic": {"status": "available", "reason": "configured"},
    }
    service.read_selected_artifact_status = lambda session, *, token, context: _FakeResult(
        "complete", value=value, reason="safe", context=service.context
    )  # type: ignore[attr-defined]
    session = _FakeSession()
    session.context = service.context
    facade = ServiceAuditWorkbenchFacade(service, session, token=_FakeSession.session_token)

    result = facade.read_selected_artifact_status()

    materialization_result = result["value"]["materialization"]
    assert materialization_result["status"] == "unavailable"
    assert materialization_result["classification"] is None
    assert materialization_result["fidelity"] is None


def test_service_facade_artifact_status_drops_unsafe_envelope_reason() -> None:
    """A path-like envelope reason cannot preserve a positive nested value."""

    service = _FakeAuditService()
    service.context = _FakeContext(episode_id="episode-real")
    value = {
        "episode_id": "episode-real",
        "context_revision": 4,
        "source_revision": "source-revision-1",
        "source_digest": "source-digest-1",
        "materialization": {
            "status": "available",
            "reason": "configured",
            "classification": "historical_original",
            "fidelity": "verified",
        },
        "native_diagnostic": {"status": "available", "reason": "configured"},
    }
    service.read_selected_artifact_status = lambda session, *, token, context: _FakeResult(
        "complete",
        value=value,
        reason="/private/status-result",
        context=service.context,
    )  # type: ignore[attr-defined]
    session = _FakeSession()
    session.context = service.context
    facade = ServiceAuditWorkbenchFacade(service, session, token=_FakeSession.session_token)

    result = facade.read_selected_artifact_status()

    assert result["status"] == "conflict"
    assert result["value"]["materialization"]["status"] == "unavailable"
    assert result["value"]["materialization"]["classification"] is None
    assert result["value"]["native_diagnostic"]["status"] == "unavailable"
    assert "/private" not in json.dumps(result, sort_keys=True)


def test_service_facade_artifact_status_drops_missing_result_context() -> None:
    """A successful value without its returned context is not capability evidence."""

    service = _FakeAuditService()
    service.context = _FakeContext(episode_id="episode-real")
    value = {
        "episode_id": "episode-real",
        "context_revision": 4,
        "source_revision": "source-revision-1",
        "source_digest": "source-digest-1",
        "materialization": {
            "status": "available",
            "reason": "configured",
            "classification": "historical_original",
            "fidelity": "verified",
        },
        "native_diagnostic": {"status": "available", "reason": "configured"},
    }
    service.read_selected_artifact_status = lambda session, *, token, context: _FakeResult(
        "complete", value=value, reason="safe", context=None
    )  # type: ignore[attr-defined]
    session = _FakeSession()
    session.context = service.context
    facade = ServiceAuditWorkbenchFacade(service, session, token=_FakeSession.session_token)

    result = facade.read_selected_artifact_status()

    assert result["status"] == "conflict"
    assert result["value"]["materialization"]["status"] == "unavailable"
    assert result["value"]["materialization"]["classification"] is None
    assert result["value"]["native_diagnostic"]["status"] == "unavailable"


def test_service_facade_codex_start_binds_authority_and_stays_diagnostic() -> None:
    service = _FakeAuditService()
    client = _FakeCodexClient(service)
    token = _FakeSession.session_token
    facade = ServiceAuditWorkbenchFacade(service, _FakeSession(), token=token, codex_client=client)
    selected = facade.next(operation_id="next-codex-authority")
    calls_before = len(service.calls)

    result = facade.codex_start(
        prompt="inspect the selected episode",
        operation_id="codex-authority-1",
        token_budget=128,
        compute_budget=1.0,
        expected_selection_revision=selected["selection_revision"],
    )

    assert result["status"] == "complete"
    assert result["operation_id"] == "codex-authority-1"
    assert result["context"]["context_revision"] == 4
    assert result["context"]["execution_id"] == "execution-real"
    assert result["context"]["scenario_id"] == "scenario-real"
    assert result["source"]["source_revision"] == "source-revision-1"
    assert result["route_id"] == "route-observed"
    assert result["evidence_ids"] == ["evidence-safe"]
    assert result["usage"] == {"measured_compute": 1.0, "total_tokens": 4}
    assert result["activity"][0]["message"] == "Codex diagnostic completed"
    result_json = json.dumps(result, sort_keys=True)
    for forbidden in (
        token,
        "private-provider",
        "private-model",
        "provider-session-private",
        "private-request-digest",
        "/private/codex/source.json",
        "provider_session_id",
        "source_path",
    ):
        assert forbidden not in result_json
    assert len(client.calls) == 1
    action, session, kwargs = client.calls[0]
    assert action == "start"
    assert session is facade._session
    assert kwargs["audit_token"] == token
    assert kwargs["context"] == service.context
    assert kwargs["token_budget"] == 128
    assert kwargs["compute_budget"] == 1.0
    assert facade._selection_epoch == selected["selection_revision"]
    assert len(service.calls) == calls_before + 2
    assert [call[0] for call in service.calls[-2:]] == ["read_context", "read_queue"]
    assert not any(
        call[0] in {"write_annotation", "finding_from_annotation"} for call in service.calls
    )


def test_service_facade_codex_start_unwraps_nested_ba05_queue_capability() -> None:
    """The BA-05 read-queue capability envelope still binds all CAS fields."""
    service = _FakeAuditService()
    service.nested_queue_result = True
    client = _FakeCodexClient(service)
    facade = ServiceAuditWorkbenchFacade(
        service, _FakeSession(), token=_FakeSession.session_token, codex_client=client
    )
    selected = facade.next(operation_id="next-codex-nested-queue")

    result = facade.codex_start(
        prompt="inspect",
        operation_id="codex-nested-queue",
        token_budget=128,
        compute_budget=1.0,
        expected_selection_revision=selected["selection_revision"],
    )

    assert result["status"] == "complete"
    assert [call[0] for call in client.calls] == ["start"]


@pytest.mark.parametrize(
    "queue_fields",
    [
        pytest.param({}, id="empty"),
        pytest.param({"state_revision": 17}, id="partial-state"),
        pytest.param({"state_revision": 17, "input_revision": 23}, id="partial-input"),
        pytest.param(
            {
                "state_revision": True,
                "input_revision": 23,
                "input_identity": "queue-input-23",
            },
            id="invalid-state-type",
        ),
    ],
)
def test_service_facade_codex_start_requires_complete_queue_cas(
    queue_fields: dict[str, Any],
) -> None:
    """Empty, partial, or malformed queue identity cannot admit Codex."""
    service = _FakeAuditService()
    service.read_queue_fields = queue_fields
    service.next_result = _FakeResult(
        "complete",
        {"packet": _FakePacket("packet-real", _FakePrimary("episode-real")), **queue_fields},
        context=service.context,
    )
    client = _FakeCodexClient(service)
    facade = ServiceAuditWorkbenchFacade(
        service, _FakeSession(), token=_FakeSession.session_token, codex_client=client
    )
    facade.next(operation_id="next-codex-incomplete-queue")

    result = facade.codex_start(
        prompt="inspect",
        operation_id="codex-incomplete-queue",
        token_budget=128,
        compute_budget=1.0,
    )

    assert result["status"] == "unavailable"
    assert "complete queue CAS" in result["reason"]
    assert not client.calls


def test_service_facade_codex_start_rejects_unbound_client() -> None:
    """A client without the exact owning service cannot receive the token."""
    service = _FakeAuditService()
    client = _FakeCodexClient(service)
    del client.service
    facade = ServiceAuditWorkbenchFacade(
        service, _FakeSession(), token=_FakeSession.session_token, codex_client=client
    )
    facade.next(operation_id="next-codex-unbound")

    result = facade.codex_start(
        prompt="inspect",
        operation_id="codex-unbound",
        token_budget=128,
        compute_budget=1.0,
    )

    assert result["status"] == "unavailable"
    assert not client.calls
    assert _FakeSession.session_token not in json.dumps(result, sort_keys=True)


def test_service_facade_codex_start_rejects_foreign_return_binding() -> None:
    """A provider result cannot relabel a turn as another operation/context."""
    service = _FakeAuditService()
    client = _FakeCodexClient(service)
    client.foreign_start_binding = True
    facade = ServiceAuditWorkbenchFacade(
        service, _FakeSession(), token=_FakeSession.session_token, codex_client=client
    )
    facade.next(operation_id="next-codex-foreign-result")

    result = facade.codex_start(
        prompt="inspect",
        operation_id="codex-foreign-result",
        token_budget=128,
        compute_budget=1.0,
    )

    assert result["status"] == "conflict"
    assert "foreign-operation" not in json.dumps(result, sort_keys=True)
    assert facade._codex_operation_id is None
    assert facade._codex_session_handle is None


def test_service_facade_codex_start_rejects_conflicting_nested_binding() -> None:
    """A matching top-level binding cannot mask a foreign nested session."""
    service = _FakeAuditService()
    client = _FakeCodexClient(service)
    client.conflicting_nested_start_binding = True
    facade = ServiceAuditWorkbenchFacade(
        service, _FakeSession(), token=_FakeSession.session_token, codex_client=client
    )
    facade.next(operation_id="next-codex-conflicting-start")

    result = facade.codex_start(
        prompt="inspect",
        operation_id="codex-conflicting-start",
        token_budget=128,
        compute_budget=1.0,
    )

    assert result["status"] == "conflict"
    assert facade._codex_operation_id is None
    assert facade._codex_session_handle is None


def test_service_facade_codex_projection_redacts_unlisted_paths() -> None:
    """Diagnostic prose redacts arbitrary absolute and relative path forms."""
    service = _FakeAuditService()
    client = _FakeCodexClient(service)
    client.reason_text = (
        "/srv/data/secret.json /opt/tenant/config /root/secret relative/private/secret"
    )
    client.activity_text = "see C:\\tenant\\secret.json and ./relative/private/secret"
    facade = ServiceAuditWorkbenchFacade(
        service, _FakeSession(), token=_FakeSession.session_token, codex_client=client
    )
    facade.next(operation_id="next-codex-paths")

    result = facade.codex_start(
        prompt="inspect",
        operation_id="codex-paths",
        token_budget=128,
        compute_budget=1.0,
    )
    result_json = json.dumps(result, sort_keys=True)

    for forbidden in (
        "/srv/data/secret.json",
        "/opt/tenant/config",
        "/root/secret",
        "relative/private/secret",
        "C:\\tenant\\secret.json",
        "./relative/private/secret",
    ):
        assert forbidden not in result_json
    assert "<path redacted>" in result_json


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param(
            lambda service: setattr(service, "context", _FakeContext(source_identity="source-new")),
            id="source-change",
        ),
        pytest.param(
            lambda service: setattr(service, "context", _FakeContext(context_revision=5)),
            id="context-change",
        ),
        pytest.param(
            lambda service: setattr(service, "queue_state_revision", 18),
            id="queue-change",
        ),
    ],
)
def test_service_facade_codex_start_rejects_stale_binding_before_client(
    mutation,
) -> None:
    service = _FakeAuditService()
    client = _FakeCodexClient(service)
    facade = ServiceAuditWorkbenchFacade(
        service, _FakeSession(), token=_FakeSession.session_token, codex_client=client
    )
    facade.next(operation_id="next-codex-stale")
    mutation(service)

    result = facade.codex_start(
        prompt="inspect",
        operation_id="codex-stale",
        token_budget=128,
        compute_budget=1.0,
        expected_selection_revision=1,
    )

    assert result["status"] == "conflict"
    assert not client.calls


def test_service_facade_codex_start_rejects_stale_selection_revision() -> None:
    service = _FakeAuditService()
    client = _FakeCodexClient(service)
    facade = ServiceAuditWorkbenchFacade(
        service, _FakeSession(), token=_FakeSession.session_token, codex_client=client
    )
    facade.next(operation_id="next-codex-selection")

    result = facade.codex_start(
        prompt="inspect",
        operation_id="codex-selection-stale",
        token_budget=128,
        compute_budget=1.0,
        expected_selection_revision=0,
    )

    assert result["status"] == "conflict"
    assert not client.calls
    assert not any(call[0] == "read_context" for call in service.calls)


def test_service_facade_codex_reconnect_rehydrates_durable_session() -> None:
    """Recovery retains only the opaque session handle after authority revalidation."""
    service = _FakeAuditService()
    service.context = _FakeContext(episode_id="episode-real")
    client = _FakeCodexClient(service)
    token = _FakeSession.session_token
    facade = ServiceAuditWorkbenchFacade(service, _FakeSession(), token=token, codex_client=client)
    selected = facade.next(operation_id="next-codex-reconnect")

    result = facade.codex_reconnect(
        codex_session_id="codex-session-private",
        operation_id="codex-reconnect-1",
        expected_selection_revision=selected["selection_revision"],
        expected_context_revision=4,
    )

    assert result["status"] == "complete"
    assert result["operation_id"] == "codex-reconnect-1"
    assert result["codex_session_id"] == "codex-session-private"
    assert result["context"]["episode_id"] == "episode-real"
    assert result["source"] == {
        "source_revision": "source-revision-1",
        "source_digest": "source-digest-1",
    }
    result_json = json.dumps(result, sort_keys=True)
    assert token not in result_json
    assert "provider-session-private" not in result_json
    assert [call[0] for call in client.calls] == ["reconnect"]
    action, session_id, kwargs = client.calls[0]
    assert action == "reconnect"
    assert session_id == "codex-session-private"
    assert kwargs["operation_id"] == "codex-reconnect-1"
    assert kwargs["audit_token"] == token
    assert facade._codex_session_handle is client.session
    assert facade._codex_operation_id == "codex-reconnect-1"


def test_service_facade_codex_reconnect_rejects_stale_or_foreign_binding() -> None:
    """A stale selection or foreign recovery receipt never hydrates a handle."""
    service = _FakeAuditService()
    client = _FakeCodexClient(service)
    facade = ServiceAuditWorkbenchFacade(
        service, _FakeSession(), token=_FakeSession.session_token, codex_client=client
    )
    selected = facade.next(operation_id="next-codex-reconnect-stale")
    stale = facade.codex_reconnect(
        codex_session_id="codex-session-private",
        operation_id="codex-reconnect-stale",
        expected_selection_revision=selected["selection_revision"],
        expected_context_revision=3,
    )
    assert stale["status"] == "conflict"
    assert not client.calls

    client.foreign_reconnect_binding = True
    foreign = facade.codex_reconnect(
        codex_session_id="codex-session-private",
        operation_id="codex-reconnect-foreign",
        expected_selection_revision=selected["selection_revision"],
        expected_context_revision=4,
    )
    assert foreign["status"] == "conflict"
    assert facade._codex_session_handle is None
    assert facade._codex_operation_id is None
    assert len(client.calls) == 1


def test_service_facade_codex_reconnect_requires_explicit_bindings() -> None:
    """A provider result without context/source proof cannot hydrate a handle."""
    service = _FakeAuditService()
    client = _FakeCodexClient(service)
    facade = ServiceAuditWorkbenchFacade(
        service, _FakeSession(), token=_FakeSession.session_token, codex_client=client
    )
    selected = facade.next(operation_id="next-codex-reconnect-missing")
    client.missing_reconnect_binding = True

    result = facade.codex_reconnect(
        codex_session_id="codex-session-private",
        operation_id="codex-reconnect-missing",
        expected_selection_revision=selected["selection_revision"],
        expected_context_revision=4,
    )

    assert result["status"] == "conflict"
    assert "authoritative context binding" in result["reason"]
    assert facade._codex_session_handle is None
    assert facade._codex_operation_id is None


def test_service_facade_codex_reconnect_rejects_foreign_operation_and_partial_source() -> None:
    """A reconnect receipt must bind the new operation and both source identities."""
    service = _FakeAuditService()
    client = _FakeCodexClient(service)
    facade = ServiceAuditWorkbenchFacade(
        service, _FakeSession(), token=_FakeSession.session_token, codex_client=client
    )
    selected = facade.next(operation_id="next-codex-reconnect-receipt")
    client.foreign_reconnect_operation_binding = True

    foreign_operation = facade.codex_reconnect(
        codex_session_id="codex-session-private",
        operation_id="codex-reconnect-foreign-operation",
        expected_selection_revision=selected["selection_revision"],
        expected_context_revision=4,
    )

    assert foreign_operation["status"] == "conflict"
    assert "operation binding" in foreign_operation["reason"]
    assert facade._codex_session_handle is None

    client.foreign_reconnect_operation_binding = False
    client.partial_reconnect_source_binding = True
    partial_source = facade.codex_reconnect(
        codex_session_id="codex-session-private",
        operation_id="codex-reconnect-partial-source",
        expected_selection_revision=selected["selection_revision"],
        expected_context_revision=4,
    )

    assert partial_source["status"] == "conflict"
    assert "source binding is incomplete" in partial_source["reason"]
    assert facade._codex_session_handle is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"prompt": "", "token_budget": 128, "compute_budget": 1.0},
        {"prompt": "x" * 8193, "token_budget": 128, "compute_budget": 1.0},
        {"prompt": "inspect", "token_budget": 0, "compute_budget": 1.0},
        {"prompt": "inspect", "token_budget": 128, "compute_budget": 0.5},
        {"prompt": "inspect", "token_budget": 128, "compute_budget": 10**1000},
    ],
)
def test_service_facade_codex_start_requires_bounded_positive_budgets(
    kwargs: dict[str, Any],
) -> None:
    service = _FakeAuditService()
    client = _FakeCodexClient(service)
    facade = ServiceAuditWorkbenchFacade(
        service, _FakeSession(), token=_FakeSession.session_token, codex_client=client
    )
    facade.next(operation_id="next-codex-budget")

    result = facade.codex_start(
        operation_id="codex-budget-invalid",
        **kwargs,
    )

    assert result["status"] == "failed"
    assert not client.calls


def test_service_facade_codex_cancel_is_post_turn_and_preserves_result_status() -> None:
    service = _FakeAuditService()
    client = _FakeCodexClient(service)
    facade = ServiceAuditWorkbenchFacade(
        service, _FakeSession(), token=_FakeSession.session_token, codex_client=client
    )
    facade.next(operation_id="next-codex-cancel")
    started = facade.codex_start(
        prompt="inspect",
        operation_id="codex-cancel-1",
        token_budget=128,
        compute_budget=1.0,
    )
    assert started["status"] == "complete"

    cancelled = facade.codex_cancel(reason="post-turn review", operation_id="codex-cancel-1")

    assert cancelled["status"] == "cancelled"
    cancel_calls = [call for call in client.calls if call[0] == "cancel"]
    assert len(cancel_calls) == 1
    _, session, kwargs = cancel_calls[0]
    assert session is client.session
    assert kwargs["audit_token"] == _FakeSession.session_token
    assert kwargs["reason"] == "post-turn review"
    assert _FakeSession.session_token not in json.dumps(cancelled, sort_keys=True)
    assert "provider-session-private" not in json.dumps(cancelled, sort_keys=True)


def test_service_facade_codex_cancel_rejects_stale_post_turn_binding() -> None:
    """A context/source move after start cannot forward the old handle."""
    service = _FakeAuditService()
    client = _FakeCodexClient(service)
    facade = ServiceAuditWorkbenchFacade(
        service, _FakeSession(), token=_FakeSession.session_token, codex_client=client
    )
    facade.next(operation_id="next-codex-cancel-stale")
    started = facade.codex_start(
        prompt="inspect",
        operation_id="codex-cancel-stale",
        token_budget=128,
        compute_budget=1.0,
    )
    assert started["status"] == "complete"
    service.context = _FakeContext(context_revision=5, episode_id="episode-new")

    result = facade.codex_cancel(reason="post-turn review", operation_id="codex-cancel-stale")

    assert result["status"] == "conflict"
    assert not any(call[0] == "cancel" for call in client.calls)


def test_service_facade_codex_cancel_rejects_foreign_return_binding() -> None:
    """A cancel receipt with another operation/context is not accepted."""
    service = _FakeAuditService()
    client = _FakeCodexClient(service)
    facade = ServiceAuditWorkbenchFacade(
        service, _FakeSession(), token=_FakeSession.session_token, codex_client=client
    )
    facade.next(operation_id="next-codex-cancel-foreign")
    started = facade.codex_start(
        prompt="inspect",
        operation_id="codex-cancel-foreign",
        token_budget=128,
        compute_budget=1.0,
    )
    assert started["status"] == "complete"
    client.foreign_cancel_binding = True

    result = facade.codex_cancel(reason="post-turn review", operation_id="codex-cancel-foreign")

    assert result["status"] == "conflict"
    assert "foreign-operation" not in json.dumps(result, sort_keys=True)


def test_service_facade_codex_cancel_rejects_conflicting_nested_binding() -> None:
    """A matching top-level cancel binding cannot mask a foreign session."""
    service = _FakeAuditService()
    client = _FakeCodexClient(service)
    facade = ServiceAuditWorkbenchFacade(
        service, _FakeSession(), token=_FakeSession.session_token, codex_client=client
    )
    facade.next(operation_id="next-codex-conflicting-cancel")
    started = facade.codex_start(
        prompt="inspect",
        operation_id="codex-conflicting-cancel",
        token_budget=128,
        compute_budget=1.0,
    )
    assert started["status"] == "complete"
    client.conflicting_nested_cancel_binding = True

    result = facade.codex_cancel(reason="post-turn review", operation_id="codex-conflicting-cancel")

    assert result["status"] == "conflict"
    assert "foreign-operation" not in json.dumps(result, sort_keys=True)


def test_service_facade_codex_cancel_is_unavailable_while_start_is_in_flight() -> None:
    service = _FakeAuditService()
    client = _FakeCodexClient(service)
    client.block_start = True
    facade = ServiceAuditWorkbenchFacade(
        service, _FakeSession(), token=_FakeSession.session_token, codex_client=client
    )
    facade.next(operation_id="next-codex-in-flight")
    start_result: list[dict[str, Any]] = []

    def run_start() -> None:
        start_result.append(
            facade.codex_start(
                prompt="inspect",
                operation_id="codex-in-flight",
                token_budget=128,
                compute_budget=1.0,
            )
        )

    thread = threading.Thread(target=run_start)
    thread.start()
    assert client.start_entered.wait(timeout=5)
    unavailable = facade.codex_cancel(reason="interrupt", operation_id="codex-in-flight")
    assert unavailable["status"] == "unavailable"
    assert not any(call[0] == "cancel" for call in client.calls)
    client.release_start.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert start_result[0]["status"] == "complete"


def test_service_facade_codex_read_requires_explicit_authenticated_ba05_api() -> None:
    service = _FakeAuditService()
    client = _FakeCodexClient(service)
    facade = ServiceAuditWorkbenchFacade(
        service, _FakeSession(), token=_FakeSession.session_token, codex_client=client
    )
    facade.next(operation_id="next-codex-read-unavailable")

    result = facade.codex_read(operation_id="codex-read-unavailable")

    assert result["status"] == "unavailable"
    assert "activity read is unavailable" in result["reason"]
    assert not any(call[0] == "read_codex_activity" for call in service.calls)


def test_service_facade_codex_read_rejects_kwargs_only_authentication_boundary() -> None:
    """A variadic read method cannot prove that the server token was used."""
    service = _FakeAuditService()

    def read_codex_activity(session: Any, **kwargs: Any) -> dict[str, Any]:
        service._record("read_codex_activity", session, **kwargs)
        return {"status": "complete"}

    service.read_codex_activity = read_codex_activity
    facade = ServiceAuditWorkbenchFacade(service, _FakeSession(), token=_FakeSession.session_token)
    facade.next(operation_id="next-codex-read-kwargs-only")

    result = facade.codex_read(operation_id="codex-read-kwargs-only")

    assert result["status"] == "unavailable"
    assert not any(call[0] == "read_codex_activity" for call in service.calls)


def test_service_facade_codex_read_uses_authenticated_service_projection() -> None:
    service = _FakeAuditService()

    def read_codex_activity(
        session: Any,
        *,
        token: str,
        operation_id: str | None = None,
        codex_session_id: str | None = None,
        context: Any = None,
    ) -> dict[str, Any]:
        service._record(
            "read_codex_activity",
            session,
            token=token,
            operation_id=operation_id,
            codex_session_id=codex_session_id,
            context=context,
        )
        return {
            "status": "complete",
            "reason": "",
            "events_reason": "events are not persisted",
            "session": {
                "codex_session_id": "private-codex-session",
                "provider_session_id": "private-provider-session",
                "context": service.context,
                "route": {"route_id": "route-read", "provider": "private-provider"},
                "evidence": [{"evidence_id": "evidence-read", "locator": "/private/path"}],
            },
            "operation": {
                "operation_id": "codex-read-1",
                "message": "read activity",
                "usage": {"tokens": 2, "compute": 1.0},
            },
            "source_path": "/private/path",
        }

    service.read_codex_activity = read_codex_activity
    facade = ServiceAuditWorkbenchFacade(service, _FakeSession(), token=_FakeSession.session_token)
    facade.next(operation_id="next-codex-read")

    result = facade.codex_read(operation_id="codex-read-1")

    assert result["status"] == "complete"
    assert result["route_id"] == "route-read"
    assert result["activity"][0]["message"] == "read activity"
    assert result["events_reason"] == "events are not persisted"
    result_json = json.dumps(result, sort_keys=True)
    for forbidden in (
        "private-codex-session",
        "private-provider-session",
        "private-provider",
        "/private/path",
        "source_path",
    ):
        assert forbidden not in result_json
    read_call = next(call for call in service.calls if call[0] == "read_codex_activity")
    assert read_call[1][0] is facade._session
    assert read_call[2]["token"] == _FakeSession.session_token
    assert read_call[2]["context"] == service.context


def test_service_facade_rejects_episode_read_context_drift_before_writes() -> None:
    """A queue packet cannot select an episode read from another context."""
    service = _FakeAuditService()
    service.context = _FakeContext(episode_id="episode-old")
    service.episode_context = _FakeContext(episode_id="episode-new")
    facade = ServiceAuditWorkbenchFacade(service, _FakeSession(), token=_FakeSession.session_token)

    result = facade.next(expected_selection_revision=0, operation_id="next-context-drift")

    assert result["status"] == "unavailable"
    assert result["presentation_status"] == "unavailable"
    assert "does not match" in result["reason"]
    assert result["conflict"]["expected_context"]["episode_id"] == "episode-old"
    assert result["conflict"]["actual_context"]["episode_id"] == "episode-new"
    assert facade._selected_episode_id is None
    assert not any(call[0] == "write_annotation" for call in service.calls)
    failed_write = facade.save_annotation(
        {"annotation_id": "annotation-after-context-drift"},
        expected_selection_revision=0,
        expected_revision=0,
        operation_id="annotation-after-context-drift",
    )
    assert failed_write["status"] == "unavailable"
    assert not any(call[0] == "write_annotation" for call in service.calls)


def test_service_facade_preserves_stale_cas_without_read_or_fixture_fallback() -> None:
    """A service conflict is returned unchanged and cannot select fixture data."""
    service = _FakeAuditService()
    service.next_result = _FakeResult(
        "conflict",
        reason="next context revision CAS failed",
        operation=_FakeOperation("next-stale", "queue.next", "conflict"),
        context=service.context,
    )
    service_token = "secret"
    facade = ServiceAuditWorkbenchFacade(service, _FakeSession(), token=service_token)

    result = facade.next(expected_context_revision=3, operation_id="next-stale")

    assert result["status"] == "conflict"
    assert result["reason"] == "next context revision CAS failed"
    assert result["operation"]["operation_id"] == "next-stale"
    assert "packet" not in result
    assert [call[0] for call in service.calls] == ["next"]


@pytest.mark.parametrize("typed", [True, False], ids=["typed", "mapping"])
@pytest.mark.parametrize("status", ["complete", "failed"])
def test_service_facade_redacts_typed_and_mapping_reasons(typed: bool, status: str) -> None:
    """Typed and mapping envelopes cannot reintroduce a token through ``reason``."""
    service = _FakeAuditService()
    service_token = "reason-private-token"
    reason = f"{status} {service_token}"
    if typed:
        service.next_result = _FakeResult(status, reason=reason, context=service.context)
    else:
        service.next_result = {
            "schema_version": "audit-service.v1",
            "status": status,
            "reason": reason,
            "value": None,
            "operation": None,
            "context": service.context,
        }
    facade = ServiceAuditWorkbenchFacade(service, _FakeSession(), token=service_token)

    result = facade.next(operation_id=f"reason-{status}")

    assert service_token not in json.dumps(result, sort_keys=True)
    assert result["reason"] == f"{status} <redacted>"


def test_service_facade_redacts_local_conflicts_context_and_metadata() -> None:
    """Local envelopes and metadata use the same server-secret sanitizer."""
    service = _FakeAuditService()
    service_token = "local-private-token"
    session = _FakeSession()
    session.session_id = service_token
    session.actor = type("Actor", (), {"actor_id": service_token})()
    session.context = {
        "context_revision": 4,
        "source_revision": "source-revision-1",
        "source_identity": "source-digest-1",
        "extension_detail": service_token,
        "source_path": "/private/secret/source.json",
        "trace_path": "/private/secret/trace.json",
        "recording_path": "/private/secret/recording.mp4",
        "campaign_root": "/private/secret/campaign",
    }
    facade = ServiceAuditWorkbenchFacade(service, session, token=service_token)

    result = facade.next(expected_selection_revision=service_token)
    metadata = facade.service_metadata

    assert service_token not in json.dumps(result, sort_keys=True)
    assert service_token not in json.dumps(metadata, sort_keys=True)
    assert result["conflict"]["expected_selection_epoch"] == "<redacted>"
    assert result["context"]["extension_detail"] == "<redacted>"
    for key in ("source_path", "trace_path", "recording_path", "campaign_root"):
        assert key not in result["context"]
    assert metadata["session_id"] == "<redacted>"
    assert metadata["actor_id"] == "<redacted>"


def test_service_facade_drops_path_like_fields_from_service_envelopes() -> None:
    """Campaign/trace path metadata cannot reach a browser service response."""
    service = _FakeAuditService()
    service.next_result = {
        "status": "failed",
        "reason": "queue unavailable",
        "value": None,
        "operation": None,
        "context": {
            "context_revision": 4,
            "source_path": "/private/secret/source.json",
            "trace_path": "/private/secret/trace.json",
            "recording_path": "/private/secret/recording.mp4",
            "campaign_root": "/private/secret/campaign",
        },
    }
    facade = ServiceAuditWorkbenchFacade(service, _FakeSession(), token=_FakeSession.session_token)

    result = facade.next(operation_id="path-like-envelope")

    assert result["context"] == {"context_revision": 4}
    assert "/private/secret" not in json.dumps(result, sort_keys=True)


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("/private/allowed/campaign.json", "<redacted-local-uri>"),
        ("file:///private/allowed/campaign.json", "<redacted-local-uri>"),
        (r"C:\\private\\allowed\\campaign.json", "<redacted-local-uri>"),
        (r"\\\\server\\share\\campaign.json", "<redacted-local-uri>"),
        ("../private/campaign.json", "<redacted-local-uri>"),
        (" /private/allowed/campaign.json", "<redacted-local-uri>"),
        ("%2Fprivate%2Fallowed%2Fcampaign.json", "<redacted-local-uri>"),
        ("artifact/scene.json", "artifact/scene.json"),
        ("artifact://campaign/scene.json", "artifact://campaign/scene.json"),
        ("https://example.invalid/campaign.json", "https://example.invalid/campaign.json"),
        ("urn:example:campaign", "urn:example:campaign"),
        ("s3://bucket/campaign.json", "s3://bucket/campaign.json"),
        ("https://example.invalid/campaign.json?token=secret", "<redacted-local-uri>"),
        (
            "https://example.invalid/campaign.json%3Ftoken=unbound-secret",
            "<redacted-local-uri>",
        ),
    ],
)
def test_service_facade_uri_policy_is_conservative(uri: str, expected: str) -> None:
    """URI serialization keeps logical IDs and redacts local/uncertain values."""

    secret = "uri-test-secret"
    result = _safe_service_value({"uri": uri}, secret=secret)

    assert result["uri"] == expected


def test_service_facade_rejects_percent_encoded_server_secret() -> None:
    """A percent-encoded token cannot survive URI serialization."""

    uri = "https://example.invalid/%73%65%72%76%65%72%2D%73%65%63%72%65%74"
    server_secret = "server-secret"
    result = _safe_service_value({"uri": uri}, secret=server_secret)

    assert result["uri"] == "<redacted-local-uri>"
    assert server_secret not in json.dumps(result, sort_keys=True)


def test_service_facade_never_reports_saved_without_commit_result() -> None:
    """A status-only replay cannot be presented as a durable write."""
    service = _FakeAuditService()
    service_token = "private-token"
    facade = ServiceAuditWorkbenchFacade(service, _FakeSession(), token=service_token)
    facade.next(operation_id="next-empty-commit")
    annotation = {
        "annotation_id": "annotation-empty-commit",
        "episode_id": "episode-real",
        "classification": "unclear",
    }
    service.annotation_result = _FakeResult(
        "committed", reason="replayed without receipt", context=service.context
    )

    saved = facade.save_annotation(
        annotation,
        expected_selection_revision=1,
        expected_revision=0,
        operation_id="annotation-empty-commit",
    )

    assert saved["status"] == "unavailable"
    assert saved["service_status"] == "committed"
    assert saved["presentation_status"] == "unavailable"
    assert "saved" not in saved.values()

    service.annotation_result = None
    committed = facade.save_annotation(
        annotation,
        expected_selection_revision=1,
        expected_revision=0,
        operation_id="annotation-commit",
    )
    assert committed["presentation_status"] == "saved"
    service.finding_result = {
        "status": "committed",
        "reason": "replayed without receipt",
        "value": None,
        "operation": None,
        "context": service.context,
    }

    finding = facade.persist_finding(
        annotation,
        expected_selection_revision=1,
        operation_id="finding-empty-commit",
    )

    assert finding["status"] == "unavailable"
    assert finding["service_status"] == "committed"
    assert finding["presentation_status"] == "unavailable"
    assert "saved" not in finding.values()


@pytest.mark.parametrize(
    "case",
    [
        "missing_operation",
        "operation_id_mismatch",
        "operation_not_committed",
        "operation_type_mismatch",
        "wrong_record_type",
        "wrong_record_id",
        "deleted",
        "wrong_service_status",
    ],
)
def test_service_facade_requires_matching_canonical_commit_receipt(case: str) -> None:
    """A plausible commit value alone cannot create a browser ``saved`` state."""
    service = _FakeAuditService()
    service_token = "private-token"
    facade = ServiceAuditWorkbenchFacade(service, _FakeSession(), token=service_token)
    facade.next(operation_id="next-canonical-receipt")
    requested_operation_id = "annotation-request"
    annotation_id = "annotation-canonical"
    value = {
        "committed": True,
        "deleted": False,
        "operation_id": requested_operation_id,
        "record_id": annotation_id,
        "record_type": "annotation",
        "revision": 1,
        "global_revision": 9,
    }
    result_status = "committed"
    operation: Any = _FakeOperation(requested_operation_id, "write.annotation")
    if case == "missing_operation":
        operation = None
    elif case == "operation_id_mismatch":
        value["operation_id"] = "other-operation"
        operation = _FakeOperation("other-operation", "write.annotation")
    elif case == "operation_not_committed":
        operation = _FakeOperation(requested_operation_id, "write.annotation", "failed")
    elif case == "operation_type_mismatch":
        operation = _FakeOperation(requested_operation_id, "write.finding")
    elif case == "wrong_record_type":
        value["record_type"] = "finding"
    elif case == "wrong_record_id":
        value["record_id"] = "other-annotation"
    elif case == "deleted":
        value["deleted"] = True
    elif case == "wrong_service_status":
        result_status = "saved"
    service.annotation_result = _FakeResult(
        result_status,
        value,
        operation=operation,
        context=service.context,
    )

    result = facade.save_annotation(
        {
            "annotation_id": annotation_id,
            "episode_id": "episode-real",
            "classification": "unclear",
        },
        expected_selection_revision=1,
        expected_revision=0,
        operation_id=requested_operation_id,
    )

    assert result["status"] == "unavailable"
    assert result["presentation_status"] == "unavailable"
    assert result.get("service_status") == result_status
    assert facade._last_annotation is None


def test_service_facade_derives_failed_write_presentation_status() -> None:
    """An untrusted failed-envelope marker cannot claim ``saved``."""
    service = _FakeAuditService()
    service.annotation_result = {
        "status": "failed",
        "reason": "store unavailable",
        "value": None,
        "operation": None,
        "context": service.context,
        "presentation_status": "saved",
    }
    service_token = "private-token"
    facade = ServiceAuditWorkbenchFacade(service, _FakeSession(), token=service_token)
    facade.next(operation_id="next-failed-presentation")

    result = facade.save_annotation(
        {
            "annotation_id": "annotation-failed-presentation",
            "episode_id": "episode-real",
            "classification": "unclear",
        },
        expected_selection_revision=1,
        expected_revision=0,
        operation_id="annotation-failed-presentation",
    )

    assert result["status"] == "failed"
    assert result["presentation_status"] == "failed"
    assert result["presentation_status"] != "saved"


def test_service_facade_failed_write_context_drift_discards_annotation() -> None:
    """A failed write with a changed source identity cannot feed a finding."""
    service = _FakeAuditService()
    service_token = "private-token"
    facade = ServiceAuditWorkbenchFacade(service, _FakeSession(), token=service_token)
    facade.next(operation_id="next-failed-context-drift")
    annotation = {
        "annotation_id": "annotation-failed-context-drift",
        "episode_id": "episode-real",
        "classification": "unclear",
    }
    saved = facade.save_annotation(
        annotation,
        expected_selection_revision=1,
        expected_revision=0,
        operation_id="annotation-before-context-drift",
    )
    assert saved["presentation_status"] == "saved"

    drifted_context = _FakeContext(source_identity="different-source")
    service.annotation_result = _FakeResult(
        "failed",
        reason="write context changed",
        context=drifted_context,
        operation=_FakeOperation("annotation-context-drift", "write.annotation", "failed"),
    )
    failed = facade.save_annotation(
        annotation,
        expected_selection_revision=1,
        expected_revision=1,
        operation_id="annotation-context-drift",
    )
    assert failed["status"] == "failed"
    assert failed["presentation_status"] == "failed"

    finding = facade.persist_finding(
        annotation,
        expected_selection_revision=1,
        operation_id="finding-after-failed-context-drift",
    )

    assert finding["status"] == "unavailable"
    assert "durably committed annotation" in finding["reason"]
    assert not any(call[0] == "finding_from_annotation" for call in service.calls)


@pytest.mark.parametrize(
    "returned_context",
    [
        pytest.param(
            _FakeContext(source_identity="different-finding-source"),
            id="source-identity-drift",
        ),
        pytest.param(_FakeContext(context_revision=5), id="context-revision-drift"),
        pytest.param(_FakeContext(episode_id="different-episode"), id="episode-identity-drift"),
    ],
)
def test_service_facade_downgrades_committed_finding_context_drift(
    returned_context: _FakeContext,
) -> None:
    """A committed finding with a mismatched result context cannot claim saved."""
    service = _FakeAuditService()
    service_token = "private-token"
    facade = ServiceAuditWorkbenchFacade(service, _FakeSession(), token=service_token)
    facade.next(operation_id="next-finding-context-drift")
    annotation = {
        "annotation_id": "annotation-finding-context-drift",
        "episode_id": "episode-real",
        "classification": "unclear",
    }
    saved = facade.save_annotation(
        annotation,
        expected_selection_revision=1,
        expected_revision=0,
        operation_id="annotation-finding-context-drift",
    )
    assert saved["presentation_status"] == "saved"
    previous_finding = facade.persist_finding(
        None,
        expected_selection_revision=1,
        operation_id="finding-before-context-drift",
        finding_id="finding-before-context-drift",
    )
    assert previous_finding["presentation_status"] == "saved"
    assert facade.snapshot()["finding_receipt"]["receipt"]["record_id"] == (
        "finding-before-context-drift"
    )

    finding_id = "finding-context-drift"
    finding_operation_id = "finding-context-drift-operation"
    service.finding_result = _FakeResult(
        "committed",
        {
            "committed": True,
            "deleted": False,
            "operation_id": finding_operation_id,
            "record_id": finding_id,
            "record_type": "finding",
            "revision": 1,
            "global_revision": 10,
        },
        operation=_FakeOperation(finding_operation_id, "write.finding"),
        context=returned_context,
    )

    finding = facade.persist_finding(
        None,
        expected_selection_revision=1,
        operation_id=finding_operation_id,
        finding_id=finding_id,
    )

    assert finding["status"] == "unavailable"
    assert finding["service_status"] == "committed"
    assert finding["presentation_status"] == "unavailable"
    assert finding["receipt"]["record_id"] == finding_id
    assert finding["receipt"]["record_type"] == "finding"
    assert finding["receipt"]["deleted"] is False
    assert facade._last_annotation is None
    assert facade._last_finding_receipt is None
    assert "finding_receipt" not in facade.snapshot()


def test_service_facade_replaces_forged_top_level_receipt_with_commit_receipt() -> None:
    """Valid commits cannot inherit browser-facing receipt fields from a mapping."""
    service = _FakeAuditService()
    service_token = "private-token"
    facade = ServiceAuditWorkbenchFacade(service, _FakeSession(), token=service_token)
    facade.next(operation_id="next-forged-top-level-receipt")
    annotation_id = "annotation-canonical-top-level"
    operation_id = "annotation-canonical-top-level-op"
    commit = {
        "committed": True,
        "deleted": False,
        "operation_id": operation_id,
        "record_id": annotation_id,
        "record_type": "annotation",
        "revision": 3,
        "global_revision": 12,
        "replayed": True,
    }
    service.annotation_result = {
        "status": "committed",
        "reason": "",
        "value": commit,
        "operation": _FakeOperation(operation_id, "write.annotation"),
        "context": service.context,
        "receipt": {
            "committed": True,
            "deleted": True,
            "operation_id": "forged-operation",
            "record_id": "forged-record",
            "record_type": "finding",
            "revision": 999,
            "global_revision": 999,
        },
        "operation_id": "forged-operation",
        "record_id": "forged-record",
        "record_type": "finding",
        "revision": 999,
        "global_revision": 999,
        "deleted": True,
    }

    saved = facade.save_annotation(
        {
            "annotation_id": annotation_id,
            "episode_id": "episode-real",
            "classification": "unclear",
        },
        expected_selection_revision=1,
        expected_revision=0,
        operation_id=operation_id,
    )

    assert saved["presentation_status"] == "saved"
    assert saved["receipt"] == commit
    assert saved["record_id"] == annotation_id
    assert saved["record_type"] == "annotation"
    assert saved["revision"] == 3
    assert saved["global_revision"] == 12
    assert saved["deleted"] is False
    assert saved["operation_id"] == operation_id
    assert saved["receipt"]["record_id"] != "forged-record"

    finding = facade.persist_finding(
        None,
        expected_selection_revision=1,
        operation_id="finding-after-canonical-receipt",
    )
    assert finding["presentation_status"] == "saved"
    assert finding["annotation_receipt"]["receipt"] == commit


def test_service_facade_detaches_selected_packet_from_returned_and_snapshot_state() -> None:
    """Consumers cannot mutate the facade's selected packet through a response."""
    service = _FakeAuditService()
    service_token = "private-token"
    facade = ServiceAuditWorkbenchFacade(service, _FakeSession(), token=service_token)

    selected = facade.next(operation_id="next-detached-packet")
    selected["packet"]["episode_id"] = "tampered-episode"
    selected["packet"]["primary"]["episode_id"] = "tampered-primary"

    first_snapshot = facade.snapshot()
    assert first_snapshot["packet"]["episode_id"] == "episode-real"
    assert first_snapshot["packet"]["primary"]["episode_id"] == "episode-real"

    first_snapshot["packet"]["episode_id"] = "tampered-snapshot"
    second_snapshot = facade.snapshot()
    assert second_snapshot["packet"]["episode_id"] == "episode-real"


def test_service_facade_failed_next_derives_presentation_status() -> None:
    """A failed Next envelope cannot preserve a forged selected marker."""
    service = _FakeAuditService()
    service.next_result = {
        "status": "failed",
        "reason": "queue unavailable",
        "value": None,
        "operation": _FakeOperation("next-failed-marker", "queue.next", "failed"),
        "context": service.context,
        "presentation_status": "selected",
    }
    service_token = "private-token"
    facade = ServiceAuditWorkbenchFacade(service, _FakeSession(), token=service_token)

    result = facade.next(operation_id="next-failed-marker")

    assert result["status"] == "failed"
    assert result["presentation_status"] == "failed"
    assert result["presentation_status"] != "selected"
    assert "packet" not in result


def test_service_facade_rejects_finding_after_context_revision_changes() -> None:
    """A finding cannot reuse an annotation receipt under a newer context."""
    service = _FakeAuditService()
    service_token = "private-token"
    facade = ServiceAuditWorkbenchFacade(service, _FakeSession(), token=service_token)
    facade.next(operation_id="next-context-freshness")
    annotation = {
        "annotation_id": "annotation-context-freshness",
        "episode_id": "episode-real",
        "classification": "unclear",
    }
    saved = facade.save_annotation(
        annotation,
        expected_selection_revision=1,
        expected_revision=0,
        operation_id="annotation-context-freshness",
    )
    assert saved["presentation_status"] == "saved"

    updated_context = _FakeContext(context_revision=5)
    updated = facade.update_context(
        updated_context,
        expected_context_revision=4,
        operation_id="context-update",
    )
    assert updated["status"] == "complete"
    finding = facade.persist_finding(
        annotation,
        expected_selection_revision=1,
        operation_id="finding-stale-context",
    )

    assert finding["status"] == "conflict"
    assert "stale" in finding["reason"]
    assert finding["conflict"] == {
        "annotation_context_revision": 4,
        "actual_context_revision": 5,
    }
    assert not any(call[0] == "finding_from_annotation" for call in service.calls)


def test_service_facade_finding_uses_exact_committed_annotation_content() -> None:
    """Finding creation rejects caller edits instead of forwarding them."""
    service = _FakeAuditService()
    service_token = "private-token"
    facade = ServiceAuditWorkbenchFacade(service, _FakeSession(), token=service_token)
    facade.next(operation_id="next-retained-annotation")
    annotation = {
        "annotation_id": "annotation-retained",
        "episode_id": "episode-real",
        "classification": "unclear",
        "metadata": {"editor_note": "committed content"},
    }
    saved = facade.save_annotation(
        annotation,
        expected_selection_revision=1,
        expected_revision=0,
        operation_id="annotation-retained",
    )
    assert saved["presentation_status"] == "saved"
    saved["receipt"]["revision"] = 999
    retained_finding = facade.persist_finding(
        annotation,
        expected_selection_revision=1,
        operation_id="finding-retained-original",
    )
    assert retained_finding["annotation_receipt"]["receipt"]["revision"] == 1
    service.calls.clear()

    edited = dict(annotation)
    edited["classification"] = "collision"
    edited["metadata"] = {"editor_note": "UNSAVED EDIT"}
    finding = facade.persist_finding(
        edited,
        expected_selection_revision=1,
        operation_id="finding-retained",
    )

    assert finding["status"] == "conflict"
    assert "committed annotation content" in finding["reason"]
    assert not any(call[0] == "finding_from_annotation" for call in service.calls)


@pytest.mark.parametrize("identity_mismatch", [False, True])
def test_service_facade_rejects_untrusted_successful_context_update(
    identity_mismatch: bool,
) -> None:
    """A stale or identity-mismatched update cannot preserve finding freshness."""
    service = _FakeAuditService()
    service_token = "private-token"
    facade = ServiceAuditWorkbenchFacade(service, _FakeSession(), token=service_token)
    facade.next(operation_id="next-context-update-integrity")
    annotation = {
        "annotation_id": "annotation-context-update-integrity",
        "episode_id": "episode-real",
        "classification": "unclear",
    }
    saved = facade.save_annotation(
        annotation,
        expected_selection_revision=1,
        expected_revision=0,
        operation_id="annotation-context-update-integrity",
    )
    assert saved["presentation_status"] == "saved"
    returned_context = _FakeContext(
        context_revision=5 if identity_mismatch else 4,
        source_identity="other-source" if identity_mismatch else "source-digest-1",
    )
    service.update_context_result = _FakeResult(
        "complete", returned_context, context=returned_context
    )
    requested_context = _FakeContext(
        context_revision=5,
        source_identity="requested-source" if identity_mismatch else "source-digest-1",
    )

    updated = facade.update_context(
        requested_context,
        expected_context_revision=4,
        operation_id="context-update-integrity",
    )
    finding = facade.persist_finding(
        annotation,
        expected_selection_revision=1,
        operation_id="finding-after-untrusted-context",
    )

    assert updated["status"] == "unavailable"
    assert updated["reason"] == "update_context did not return the requested authoritative context"
    assert finding["status"] == "unavailable"
    assert not any(call[0] == "finding_from_annotation" for call in service.calls)


def test_service_facade_annotation_and_finding_keep_receipts_and_source_cas() -> None:
    """Writes use the selected service context and retain typed commit receipts."""
    service = _FakeAuditService()
    service_token = "private-token"
    facade = ServiceAuditWorkbenchFacade(service, _FakeSession(), token=service_token)
    selected = facade.next(operation_id="next-annotation")
    assert selected["presentation_status"] == "selected"

    annotation = {
        "record_type": "annotation",
        "annotation_id": "annotation-real",
        "episode_id": "episode-real",
        "classification": "unclear",
        "mode": "quick",
        "source_identity": {"sources": {"scene": {"sha256": "source-digest-1"}}},
    }
    saved = facade.save_annotation(
        annotation,
        expected_selection_revision=1,
        expected_revision=0,
        expected_source_revision="source-revision-1",
        operation_id="annotation-1",
    )
    assert saved["status"] == "committed"
    assert saved["presentation_status"] == "saved"
    assert saved["receipt"]["global_revision"] == 9
    write_call = next(call for call in service.calls if call[0] == "write_annotation")
    assert write_call[1][1]["episode_id"] == "episode-real"
    assert write_call[2]["expected_revision"] == 0
    assert write_call[2]["expected_source_revision"] == "source-revision-1"
    assert write_call[2]["token"] == "private-token"

    finding = facade.persist_finding(
        annotation,
        expected_selection_revision=1,
        operation_id="finding-1",
    )
    assert finding["status"] == "committed"
    assert finding["annotation_receipt"]["status"] == "committed"
    assert finding["value"]["confirmed_members"] == []

    stale = facade.save_annotation(
        annotation,
        expected_selection_revision=0,
        expected_revision=1,
        operation_id="annotation-stale",
    )
    assert stale["status"] == "conflict"
    assert len([call for call in service.calls if call[0] == "write_annotation"]) == 1


def test_service_facade_native_visual_annotation_rebinds_canonical_source_and_commits() -> None:
    """A browser SourceRef reaches the native service CAS despite provenance extras."""

    service = _FakeAuditService()
    session = _FakeSession()
    session.source_digest = "campaign-source-digest"
    session.source_ref = SourceRef(
        artifact_id="campaign",
        uri="artifact://campaign.json",
        format="campaign-json",
    )
    facade = ServiceAuditWorkbenchFacade(service, session, token=session.session_token)
    selected = facade.next(operation_id="next-native-visual")
    assert selected["presentation_status"] == "selected"

    native_source = {
        "artifact_id": "retained-native-analysis-trace",
        "uri": "retained://analysis-trace.v1",
        "format": "analysis-trace.v1",
        "schema": "analysis-trace.v1",
        "sha256": "native-trace-digest",
        "declared_sha256": "native-trace-digest",
        "computed_sha256": "native-trace-digest",
        "integrity": "verified",
        "availability": "retained",
        "admission": "not_evaluated",
        "source_commit": "native-git-commit",
        "config_identity": "native-config-digest",
        "coordinate_frame": "map",
        "units": "m",
    }
    assert facade._selected_packet is not None
    facade._selected_packet["editor_model"] = {
        "source_identity": {"sources": {"native": native_source}}
    }
    canonical_source = {
        key: value
        for key, value in native_source.items()
        if key
        in {
            "artifact_id",
            "uri",
            "format",
            "schema",
            "sha256",
            "source_commit",
            "config_identity",
            "units",
            "coordinate_frame",
        }
    }
    annotation = {
        "annotation_id": "native-visual-note",
        "episode_id": "episode-real",
        "classification": "unclear",
        "observation": "retained trace shows a crossing",
        "hypothesis": "planner yielded before clearance recovered",
        "evidence": "scene sample at t=1.0",
        "source_ref": canonical_source,
        "source_identity": native_source["sha256"],
        "source_revision": native_source["source_commit"],
        "references": [
            {
                "reference_id": "scene-1",
                "source": canonical_source,
                "source_revision": native_source["source_commit"],
            }
        ],
    }

    saved = facade.save_annotation(
        annotation,
        expected_selection_revision=selected["selection_revision"],
        expected_revision=0,
        expected_context_revision=selected["context_revision"],
        expected_source_revision="source-revision-1",
        operation_id="native-visual-save",
    )

    assert saved["status"] == "committed"
    assert saved["presentation_status"] == "saved"
    write_call = next(call for call in service.calls if call[0] == "write_annotation")
    written = write_call[1][1]
    assert write_call[2]["expected_revision"] == 0
    assert write_call[2]["context"] == service.context
    assert write_call[2]["expected_source_revision"] == "source-revision-1"
    assert written["source_ref"] == {
        "artifact_id": "campaign",
        "uri": "artifact://campaign.json",
        "format": "campaign-json",
        "schema": "",
        "sha256": "",
        "source_commit": "",
        "config_identity": "",
        "units": "",
        "coordinate_frame": "",
    }
    assert written["source_identity"] == "campaign-source-digest"
    assert written["source_revision"] == "source-revision-1"
    assert written["references"][0]["source"] == written["source_ref"]
    assert written["references"][0]["source_revision"] == "source-revision-1"
    safe_visual_source = dict(native_source, uri="<redacted-local-uri>")
    assert written["metadata"]["visual_source_ref"] == safe_visual_source
    assert written["metadata"]["visual_source_provenance"] == {
        "declared_sha256": "native-trace-digest",
        "computed_sha256": "native-trace-digest",
        "integrity": "verified",
        "availability": "retained",
        "admission": "not_evaluated",
    }
    assert written["metadata"]["reference_source_bindings"]["scene-1"] == {
        "source_ref": safe_visual_source,
        "source_identity": "native-trace-digest",
        "source_revision": "native-git-commit",
        "source_provenance": {
            "declared_sha256": "native-trace-digest",
            "computed_sha256": "native-trace-digest",
            "integrity": "verified",
            "availability": "retained",
            "admission": "not_evaluated",
        },
    }
    stale_annotation = dict(annotation)
    stale_annotation["references"] = [
        {
            "reference_id": "scene-1",
            "source": canonical_source,
            "source_revision": "stale-native-revision",
        }
    ]
    stale = facade.save_annotation(
        stale_annotation,
        expected_selection_revision=selected["selection_revision"],
        expected_revision=1,
        expected_context_revision=selected["context_revision"],
        expected_source_revision="source-revision-1",
        operation_id="native-visual-save-stale-reference",
    )
    assert stale["status"] == "conflict"
    assert "visual source revision is stale" in stale["reason"]
    assert len([call for call in service.calls if call[0] == "write_annotation"]) == 1


@pytest.mark.parametrize(
    ("field_name", "stale_value"),
    [
        ("expected_context_revision", 3),
        ("expected_source_revision", "source-revision-old"),
        ("expected_queue_state_revision", 16),
        ("expected_queue_input_revision", 22),
        ("expected_queue_input_identity", "queue-input-old"),
    ],
)
def test_service_facade_finding_binds_browser_service_cas(
    field_name: str, stale_value: Any
) -> None:
    """Finding writes reject stale context/source/queue values before BA-05."""
    service = _FakeAuditService()
    facade = ServiceAuditWorkbenchFacade(service, _FakeSession(), token=_FakeSession.session_token)
    selected = facade.next(operation_id="next-finding-cas")
    annotation = {
        "annotation_id": "annotation-finding-cas",
        "episode_id": selected["packet"]["episode_id"],
        "classification": "unclear",
    }
    saved = facade.save_annotation(
        annotation,
        expected_selection_revision=1,
        expected_revision=0,
        operation_id="annotation-finding-cas",
    )
    assert saved["presentation_status"] == "saved"
    expected = {
        "expected_context": {
            "context_revision": 4,
            "source_revision": "source-revision-1",
            "source_identity": "source-digest-1",
            "episode_id": "",
        },
        "expected_context_revision": 4,
        "expected_source_revision": "source-revision-1",
        "expected_queue_state_revision": 17,
        "expected_queue_input_revision": 23,
        "expected_queue_input_identity": "queue-input-23",
    }
    service.calls.clear()

    matching = facade.persist_finding(
        annotation,
        expected_selection_revision=1,
        operation_id="finding-matching-service-cas",
        **expected,
    )
    assert matching["presentation_status"] == "saved"

    stale = dict(expected)
    stale[field_name] = stale_value
    service.calls.clear()

    result = facade.persist_finding(
        annotation,
        expected_selection_revision=1,
        operation_id="finding-stale-service-cas",
        **stale,
    )

    assert result["status"] == "conflict"
    assert "service context, source, or queue binding" in result["reason"]
    assert not any(call[0] == "finding_from_annotation" for call in service.calls)


def test_service_facade_finding_rejects_editor_context_identity_drift() -> None:
    """A finding cannot be written for a browser token from another episode."""
    service = _FakeAuditService()
    facade = ServiceAuditWorkbenchFacade(service, _FakeSession(), token=_FakeSession.session_token)
    selected = facade.next(operation_id="next-finding-context-identity")
    annotation = {
        "annotation_id": "annotation-finding-context-identity",
        "episode_id": selected["packet"]["episode_id"],
        "classification": "unclear",
    }
    assert (
        facade.save_annotation(
            annotation,
            expected_selection_revision=1,
            expected_revision=0,
            operation_id="annotation-finding-context-identity",
        )["presentation_status"]
        == "saved"
    )
    service.calls.clear()

    result = facade.persist_finding(
        annotation,
        expected_selection_revision=1,
        expected_context={
            "context_revision": 4,
            "source_revision": "source-revision-1",
            "source_identity": "source-digest-1",
            "episode_id": "episode-other",
        },
        expected_context_revision=4,
        expected_source_revision="source-revision-1",
        expected_queue_state_revision=17,
        expected_queue_input_revision=23,
        expected_queue_input_identity="queue-input-23",
        operation_id="finding-context-identity-drift",
    )

    assert result["status"] == "conflict"
    assert "expected_context" in result["conflict"]
    assert not any(call[0] == "finding_from_annotation" for call in service.calls)


def test_service_facade_snapshot_projects_committed_records_for_browser_reopen() -> None:
    """A committed service write remains presentable after the live snapshot round-trip."""
    service = _FakeAuditService()
    facade = ServiceAuditWorkbenchFacade(service, _FakeSession(), token=_FakeSession.session_token)
    selected = facade.next(operation_id="next-reopen-records")
    annotation = {
        "annotation_id": "annotation-reopen-records",
        "episode_id": selected["packet"]["episode_id"],
        "classification": "unclear",
    }

    saved = facade.save_annotation(
        annotation,
        expected_selection_revision=1,
        expected_revision=0,
        operation_id="annotation-reopen-records",
    )
    finding = facade.persist_finding(
        annotation,
        expected_selection_revision=1,
        operation_id="finding-reopen-records",
    )

    assert saved["presentation_status"] == "saved"
    assert finding["presentation_status"] == "saved"
    assert finding["finding"]["finding_id"] == "finding-episode-real"
    assert finding["finding"]["candidate_members"] == ["episode-real"]
    assert finding["finding"]["confirmed_members"] == []
    snapshot = facade.snapshot()
    assert snapshot["context_revision"] == 4
    assert snapshot["queue_state_revision"] == 17
    assert snapshot["queue_input_revision"] == 23
    assert snapshot["queue_input_identity"] == "queue-input-23"
    assert len(snapshot["annotations"]) == 1
    reopened_annotation = snapshot["annotations"][0]
    assert reopened_annotation["annotation_id"] == "annotation-reopen-records"
    assert reopened_annotation["record_id"] == "annotation-reopen-records"
    assert reopened_annotation["episode_id"] == "episode-real"
    assert reopened_annotation["revision"] == 1
    assert reopened_annotation["global_revision"] == 9
    assert reopened_annotation["operation_id"] == "annotation-reopen-records"
    assert snapshot["findings"][0]["finding_id"] == "finding-episode-real"
    assert snapshot["findings"][0]["revision"] == 1
    assert snapshot["finding_receipt"]["receipt"]["record_id"] == "finding-episode-real"
    assert _FakeSession.session_token not in json.dumps(snapshot, sort_keys=True)


def test_service_facade_snapshot_keeps_multiple_records_by_id_and_declares_scope() -> None:
    """Same-facade reopen keeps every committed record without claiming durability."""
    service = _FakeAuditService()
    facade = ServiceAuditWorkbenchFacade(service, _FakeSession(), token=_FakeSession.session_token)
    facade.next(operation_id="next-multiple-records")
    for index in (1, 2):
        annotation = {
            "annotation_id": f"annotation-multiple-{index}",
            "episode_id": "episode-real",
            "classification": "unclear",
        }
        saved = facade.save_annotation(
            annotation,
            expected_selection_revision=1,
            expected_revision=0,
            operation_id=f"annotation-multiple-{index}",
        )
        assert saved["presentation_status"] == "saved"
        finding = facade.persist_finding(
            annotation,
            expected_selection_revision=1,
            operation_id=f"finding-multiple-{index}",
            finding_id=f"finding-multiple-{index}",
        )
        assert finding["presentation_status"] == "saved"

    snapshot = facade.snapshot()

    assert set(snapshot["annotation_records"]) == {
        "annotation-multiple-1",
        "annotation-multiple-2",
    }
    assert set(snapshot["finding_records"]) == {"finding-multiple-1", "finding-multiple-2"}
    assert [record["annotation_id"] for record in snapshot["annotations"]] == [
        "annotation-multiple-1",
        "annotation-multiple-2",
    ]
    assert [record["finding_id"] for record in snapshot["findings"]] == [
        "finding-multiple-1",
        "finding-multiple-2",
    ]
    projection = snapshot["record_projection"]
    assert projection["scope"] == "facade_instance"
    assert projection["completeness"] == "complete"
    assert projection["authoritative"] is False
    assert projection["context_identity"]["context_revision"] == 4
    assert projection["annotation_ids"] == ["annotation-multiple-1", "annotation-multiple-2"]
    assert projection["finding_ids"] == ["finding-multiple-1", "finding-multiple-2"]


def test_service_facade_snapshot_reopens_durable_records_with_zero_source_revision() -> None:
    """BA-03 records are authoritative on a fresh facade, including revision zero."""
    session = _FakeSession()
    session.context = _FakeContext(source_revision=0, episode_id="episode-real")
    service = _FakeAuditService()
    service.context = session.context
    token = _FakeSession.session_token
    unsafe_uris = (
        "/private/allowed/campaign.json",
        " /private/allowed/campaign.json",
        "%2Fprivate%2Fallowed%2Fcampaign.json",
        "https://example.invalid/campaign.json%3Ftoken=unbound-secret",
        "https://example.invalid/%73%65%72%76%65%72%2D%73%65%63%72%65%74",
    )
    local_annotation = Annotation(
        annotation_id="annotation-durable",
        episode_id="episode-real",
        classification="unclear",
        author_kind="agent",
        source_identity="source-digest-1",
        source_revision=0,
        source_ref=SourceRef(
            artifact_id="campaign",
            uri="/private/allowed/campaign.json",
            format="campaign-json",
        ),
    )
    logical_annotation = Annotation(
        annotation_id="annotation-logical",
        episode_id="episode-real",
        classification="unclear",
        author_kind="agent",
        source_identity="source-digest-1",
        source_revision=0,
        source_ref=SourceRef(
            artifact_id="campaign",
            uri="artifact/scene.json",
            format="campaign-json",
        ),
    )
    encoded_annotations = tuple(
        StoredRecord(
            record_id=f"annotation-unsafe-{index}",
            record_type="annotation",
            revision=1,
            deleted=False,
            operation_id=f"annotation-unsafe-op-{index}",
            committed_at=f"2026-09-20T00:00:{index + 2:02d}Z",
            global_revision=index + 10,
            record=Annotation(
                annotation_id=f"annotation-unsafe-{index}",
                episode_id="episode-real",
                classification="unclear",
                author_kind="agent",
                source_identity="source-digest-1",
                source_revision=0,
                source_ref=SourceRef(
                    artifact_id="campaign",
                    uri=uri,
                    format="campaign-json",
                ),
            ),
        )
        for index, uri in enumerate(unsafe_uris)
    )
    stored = (
        StoredRecord(
            record_id="annotation-durable",
            record_type="annotation",
            revision=1,
            deleted=False,
            operation_id="annotation-op",
            committed_at="2026-09-20T00:00:00Z",
            global_revision=8,
            record=local_annotation,
        ),
        StoredRecord(
            record_id="annotation-logical",
            record_type="annotation",
            revision=1,
            deleted=False,
            operation_id="annotation-logical-op",
            committed_at="2026-09-20T00:00:01Z",
            global_revision=9,
            record=logical_annotation,
        ),
    ) + encoded_annotations

    def read_saved_records(_session: Any, episode_id: str, **kwargs: Any) -> _FakeResult:
        service._record("read_saved_records", _session, episode_id, **kwargs)
        return _FakeResult("complete", stored, context=service.context)

    service.read_saved_records = read_saved_records  # type: ignore[attr-defined]
    facade = ServiceAuditWorkbenchFacade(service, session, token=token)

    snapshot = facade.snapshot()

    assert snapshot["status"] == "complete"
    before_by_id = {item["annotation_id"]: item for item in facade.read_saved_records()["records"]}
    assert before_by_id["annotation-durable"]["author_kind"] == "agent"
    assert before_by_id["annotation-durable"]["source_revision"] == 0
    assert before_by_id["annotation-durable"]["source_ref"]["uri"] == "<redacted-local-uri>"
    assert before_by_id["annotation-logical"]["source_ref"]["uri"] == "artifact/scene.json"
    for index in range(len(unsafe_uris)):
        assert before_by_id[f"annotation-unsafe-{index}"]["source_ref"]["uri"] == (
            "<redacted-local-uri>"
        )
    snapshot_by_id = {item["annotation_id"]: item for item in snapshot["annotations"]}
    assert snapshot_by_id["annotation-durable"]["source_ref"]["uri"] == "<redacted-local-uri>"
    assert snapshot_by_id["annotation-logical"]["source_ref"]["uri"] == "artifact/scene.json"
    for index in range(len(unsafe_uris)):
        assert snapshot_by_id[f"annotation-unsafe-{index}"]["source_ref"]["uri"] == (
            "<redacted-local-uri>"
        )
    projection = snapshot["record_projection"]
    assert projection["scope"] == "service_store"
    assert projection["authoritative"] is True
    assert projection["durability"] == "service_store"
    assert projection["fresh_process"] == "reconnectable"
    assert projection["human_review"] == "not_inferred"
    assert snapshot["records_result"]["status"] == "complete"
    assert token not in json.dumps(snapshot, sort_keys=True)
    assert "/private/allowed/campaign.json" not in json.dumps(snapshot, sort_keys=True)
    snapshot_json = json.dumps(snapshot, sort_keys=True)
    assert all(uri not in snapshot_json for uri in unsafe_uris)
    read_call = next(call for call in service.calls if call[0] == "read_saved_records")
    assert read_call[1][1] == "episode-real"
    assert read_call[2]["token"] == token
    assert read_call[2]["context"].source_revision == 0
    assert read_call[2]["operation_id"] is None


def test_service_facade_saved_record_read_rejects_wrong_context_without_local_fallback() -> None:
    """A mismatched service context cannot become durable browser state."""
    session = _FakeSession()
    session.context = _FakeContext(source_revision=0, episode_id="episode-real")
    service = _FakeAuditService()
    service.context = _FakeContext(source_revision=0, episode_id="episode-other")

    def read_saved_records(_session: Any, episode_id: str, **kwargs: Any) -> _FakeResult:
        service._record("read_saved_records", _session, episode_id, **kwargs)
        return _FakeResult("complete", (), context=service.context)

    service.read_saved_records = read_saved_records  # type: ignore[attr-defined]
    facade = ServiceAuditWorkbenchFacade(service, session, token=_FakeSession.session_token)

    result = facade.read_saved_records(operation_id="read-wrong-context")

    assert result["status"] == "unavailable"
    assert result["service_status"] == "complete"
    assert "records" not in result
    assert result["conflict"]["actual_context"]["episode_id"] == "episode-other"
    assert facade._context is session.context


def test_service_facade_without_record_read_does_not_claim_empty_durable_arrays() -> None:
    """Before BA-05 read support, an empty facade projection remains unknown."""
    session = _FakeSession()
    session.context = _FakeContext(episode_id="episode-real")
    service = _FakeAuditService()
    service.context = session.context
    facade = ServiceAuditWorkbenchFacade(service, session, token=_FakeSession.session_token)

    snapshot = facade.snapshot()

    assert "annotations" not in snapshot
    assert "findings" not in snapshot
    assert "record_projection" not in snapshot


def test_service_facade_without_token_is_denied_and_never_calls_service() -> None:
    """A missing server authority is explicit and cannot silently use the fixture."""
    service = _FakeAuditService()
    facade = ServiceAuditWorkbenchFacade(service, _FakeSession())

    selected = facade.next(expected_selection_revision=0)
    snapshot = facade.snapshot()

    assert selected["status"] == "denied"
    assert snapshot["status"] == "denied"
    assert service.calls == []
    assert _FakeSession.session_token not in json.dumps(snapshot, sort_keys=True)


def test_service_document_does_not_mount_fixture_html() -> None:
    """The service model remains a server-route handoff, never a fixture mount."""
    service = _FakeAuditService()
    service_token = "private-token"
    facade = ServiceAuditWorkbenchFacade(service, _FakeSession(), token=service_token)

    document = build_audit_workbench_document(facade)

    assert document["service"]["id"] == "ba06-audit-service-facade"
    assert document["persistence"]["service_token_transport"] == "server_only"
    assert "ba06-fixture-facade" not in json.dumps(document, sort_keys=True)
    assert service_token not in json.dumps(document, sort_keys=True)
    with pytest.raises(AuditWorkbenchError, match="requires its server route"):
        from robot_sf.render.audit_workbench import render_audit_workbench_html

        render_audit_workbench_html(document)


def test_service_facade_keeps_unaccepted_github_capability_unavailable() -> None:
    """No direct provider call is inferred when BA05 has no sync wrapper."""
    service = _FakeAuditService()
    service_token = "provider-secret"
    facade = ServiceAuditWorkbenchFacade(service, _FakeSession(), token=service_token)

    result = facade.sync_finding({"finding_id": "finding-real"}, operation_id="github-1")

    assert result["status"] == "unavailable"
    assert "sync_finding" in result["reason"]
    assert service.calls == []
    assert service_token not in json.dumps(result, sort_keys=True)


def test_service_facade_sync_maps_keyword_only_service_contract_and_cas() -> None:
    """BA-06 forwards only canonical identity and server-held context to BA-05."""

    class SyncService(_FakeAuditService):
        def sync_finding(self, session: Any, **kwargs: Any) -> _FakeResult:
            self._record("sync_finding", session, **kwargs)
            return _FakeResult(
                "committed",
                {
                    "status": "created",
                    "repository": kwargs["repository"],
                    "finding_id": kwargs["finding_id"],
                    "issue": {"url": "https://github.com/ll7/robot_sf_ll7/issues/7"},
                    "source_path": "/private/source.json",
                    "token": _FakeSession.session_token,
                },
                operation=_FakeOperation(kwargs["operation_id"], "github.sync_finding"),
                context=self.context,
            )

    service = SyncService()
    session = _FakeSession()
    facade = ServiceAuditWorkbenchFacade(service, session, token=session.session_token)
    facade._selected_episode_id = "episode-real"
    facade._selection_epoch = 2
    facade._last_finding = {"finding_id": "finding-real", "revision": 7}
    result = facade.sync_finding(
        finding_id="finding-real",
        repository="ll7/robot_sf_ll7",
        expected_finding_revision=7,
        expected_selection_revision=2,
        expected_context_revision=4,
        expected_source_revision="source-revision-1",
        operation_id="github-sync-1",
    )

    assert result["status"] == "committed"
    assert result["value"]["status"] == "created"
    assert result["evidence_boundary"] == "diagnostic_only"
    assert result["scientific_claim_allowed"] is False
    assert session.session_token not in json.dumps(result, sort_keys=True)
    assert "/private/source.json" not in json.dumps(result, sort_keys=True)
    call = service.calls[-1]
    assert call[0] == "sync_finding"
    assert call[2]["finding_id"] == "finding-real"
    assert call[2]["expected_finding_revision"] == 7
    assert call[2]["expected_source_revision"] == "source-revision-1"
    assert call[2]["context"] == session.context
    assert call[2]["token"] == session.session_token


def test_service_facade_sync_rejects_stale_selection_before_service_call() -> None:
    """A stale browser epoch cannot reach the provider-bound service seam."""

    class SyncService(_FakeAuditService):
        def sync_finding(self, session: Any, **kwargs: Any) -> _FakeResult:
            self._record("sync_finding", session, **kwargs)
            return _FakeResult("committed", {}, context=self.context)

    service = SyncService()
    facade = ServiceAuditWorkbenchFacade(service, _FakeSession(), token=_FakeSession.session_token)
    facade._selected_episode_id = "episode-real"
    facade._selection_epoch = 3
    facade._last_finding = {"finding_id": "finding-real", "revision": 1}
    result = facade.sync_finding(
        finding_id="finding-real",
        repository="ll7/robot_sf_ll7",
        expected_finding_revision=1,
        expected_selection_revision=2,
        expected_context_revision=4,
        expected_source_revision="source-revision-1",
        operation_id="github-sync-stale",
    )

    assert result["status"] == "conflict"
    assert service.calls == []


def test_fixture_queue_selection_cas_and_missing_media_case() -> None:
    """Queue Next advances selection revision and keeps missing media explicit."""
    service = FixtureAuditService(fixture_document())

    first = service.next(expected_selection_revision=0)

    assert first["status"] == "selected"
    assert first["selection_revision"] == 1
    assert first["packet"]["episode_id"] == "fixture-normal-control"
    assert first["next"]["episode_id"] == "fixture-missing-media"
    assert first["packet"]["metric_units"]["clearance"] == "m"
    assert first["packet"]["media"]["status"] == "available"
    with pytest.raises(AuditWorkbenchConflictError, match="stale selection revision"):
        service.next(expected_selection_revision=0)

    second = service.next(expected_selection_revision=1)

    assert second["packet"]["episode_id"] == "fixture-missing-media"
    assert second["packet"]["media"] == {
        "status": "unavailable",
        "reason": "recording_not_present",
        "resolution_s": 1.5,
        "samples": [],
    }


def test_fixture_unavailable_state_is_explicit_before_selection() -> None:
    """The facade reports an unavailable mutation instead of fabricating a case."""
    service = FixtureAuditService(fixture_document())

    with pytest.raises(AuditWorkbenchUnavailableError, match="no audit packet is selected"):
        service.save_annotation(
            {"annotation_id": "annotation-before-selection"},
            expected_selection_revision=0,
            expected_revision=0,
            operation_id="operation-before-selection",
        )


def test_annotation_and_finding_update_coverage_and_next() -> None:
    """Editor-shaped annotation persistence produces a finding and coverage receipt."""
    service = FixtureAuditService(fixture_document())
    selected = service.next(expected_selection_revision=0)
    packet = selected["packet"]
    annotation = {
        "record_type": "annotation",
        "annotation_id": "annotation-fixture-normal",
        "classification": "normal",
        "mode": "quick",
        "episode_id": packet["episode_id"],
        "metadata": {"selection_revision": selected["selection_revision"]},
    }

    receipt = service.save_annotation(
        annotation,
        expected_selection_revision=1,
        expected_revision=0,
        operation_id="operation-annotation-1",
    )
    assert receipt["status"] == "saved"
    assert receipt["revision"] == 1
    assert receipt["record"]["revision"] == 1
    assert (
        service.save_annotation(
            annotation,
            expected_selection_revision=1,
            expected_revision=0,
            operation_id="operation-annotation-1",
        )
        == receipt
    )
    with pytest.raises(AuditWorkbenchConflictError, match="revision conflict"):
        service.save_annotation(
            annotation,
            expected_selection_revision=1,
            expected_revision=0,
            operation_id="operation-annotation-stale",
        )

    finding = service.persist_finding(
        annotation,
        expected_selection_revision=1,
        operation_id="operation-finding-1",
    )
    assert finding["status"] == "saved"
    assert finding["finding"]["candidate_members"] == [packet["episode_id"]]
    assert finding["coverage"]["reviewed_episode_ids"] == [packet["episode_id"]]
    assert finding["coverage"]["remaining"] == 1
    assert finding["next"]["episode_id"] == "fixture-missing-media"

    snapshot = service.snapshot()
    assert snapshot["finding_operation_ids"] == {
        "finding-fixture-normal-control": "operation-finding-1"
    }
    reopened = FixtureAuditService(snapshot)
    retry = reopened.persist_finding(
        annotation,
        expected_selection_revision=1,
        operation_id="operation-finding-1",
    )
    assert retry["finding"]["finding_id"] == finding["finding"]["finding_id"]
    assert len(reopened.snapshot()["findings"]) == 1
    rebuilt = build_audit_workbench_document(snapshot)
    assert len(rebuilt["findings"]) == 1
    assert rebuilt["coverage"]["reviewed"] == 1


def test_fixture_finding_rejects_cross_episode_annotation_and_reopen_keeps_revision() -> None:
    """Finding identity and annotation CAS remain bound across selection changes."""
    service = FixtureAuditService(fixture_document())
    first = service.next(expected_selection_revision=0)
    annotation = {
        "record_type": "annotation",
        "annotation_id": "annotation-cross-case",
        "classification": "normal",
        "episode_id": first["packet"]["episode_id"],
    }
    saved = service.save_annotation(
        annotation,
        expected_selection_revision=1,
        expected_revision=0,
        operation_id="operation-cross-case-save",
    )
    assert saved["record"]["revision"] == 1
    restored = FixtureAuditService(service.snapshot())
    restored_retry = restored.save_annotation(
        annotation,
        expected_selection_revision=1,
        expected_revision=0,
        operation_id="operation-cross-case-save",
    )
    assert restored_retry == saved
    restored_save = restored.save_annotation(
        annotation,
        expected_selection_revision=1,
        expected_revision=1,
        operation_id="operation-restored-save",
    )
    assert restored_save["revision"] == 2

    service.next(expected_selection_revision=1)
    with pytest.raises(AuditWorkbenchConflictError, match="does not belong"):
        service.persist_finding(
            annotation,
            expected_selection_revision=2,
            operation_id="operation-cross-case-finding",
        )
    assert service.snapshot()["findings"] == []


def test_document_and_offline_assets_mark_fixture_boundary(tmp_path: Path) -> None:
    """Generated HTML stays local and visibly records its non-native boundary."""
    document = build_audit_workbench_document()

    assert document["service"]["id"] == "ba06-fixture-facade"
    assert document["service"]["native"] is False
    assert document["persistence"]["native_or_live_claim"] is False
    result = write_fixture_workbench(tmp_path / "audit")
    html = (tmp_path / "audit" / "audit-workbench.v1.html").read_text(encoding="utf-8")
    assert result["status"] == "complete"
    assert "http://" not in html and "https://" not in html
    assert "audit_workbench.js" in html
    assert (tmp_path / "audit" / "components" / "review_editor" / "review_editor.js").is_file()
    assert (tmp_path / "audit" / "components" / "review_panels" / "review_panels.js").is_file()


def test_browser_controller_runtime_covers_normal_and_missing_media_cases() -> None:
    """Execute the dependency-free DOM harness used by the fixture UI slice."""
    completed = subprocess.run(
        ["node", str(Path(__file__).with_name("audit_workbench_runtime.mjs"))],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "audit_workbench_runtime: ok" in completed.stdout
