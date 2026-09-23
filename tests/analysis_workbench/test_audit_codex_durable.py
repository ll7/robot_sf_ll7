"""Focused restart/replay proofs for the durable Codex authority seam."""

from __future__ import annotations

import hashlib
import json
import shutil
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from robot_sf.analysis_workbench.audit_authority import (
    AuditAuthorityStore,
    AuthorityCorruptionError,
    AuthorityOperationConflict,
)
from robot_sf.analysis_workbench.audit_codex import (
    AuditCodexClient,
    CodexRouteReceipt,
    FakeCodexProvider,
    _digest,
)
from robot_sf.analysis_workbench.audit_codex_app_server import CodexAppServerProvider
from robot_sf.analysis_workbench.audit_service import (
    AuditContextConflict,
    AuditPolicyError,
    AuditSelectionContext,
    AuditService,
    AuditValidationError,
    SessionPolicy,
)

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "analysis_workbench"
    / "audit_campaign_v1"
    / "campaign.json"
)


def _setup(
    tmp_path: Path,
    *,
    token_budget: int = 5,
    compute_budget: float = 5.0,
    source_revision: int | None = None,
):
    root = tmp_path / "allowed"
    root.mkdir()
    source = root / "campaign.json"
    shutil.copyfile(FIXTURE, source)
    if source_revision is not None:
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["source_revision"] = source_revision
        source.write_text(json.dumps(payload), encoding="utf-8")
    service = AuditService(tmp_path / "store", campaign_source=source, source_root=root)

    session = service.open_session(
        AuditSelectionContext(campaign_id="audit-fixture-campaign"),
        actor="agent",
        actor_id="codex-agent",
        policy=SessionPolicy(
            allowed_roots=(str(root),),
            token_budget=token_budget,
            compute_budget=compute_budget,
        ),
    )
    return service, session


def _route() -> CodexRouteReceipt:
    return CodexRouteReceipt(
        "durable-route", "fixture-provider", "fixture-model", "1.0", "app-server"
    )


def _make_inspector(route: CodexRouteReceipt):
    def inspect() -> dict[str, object]:
        return {"status": "available", "routes": [route.to_dict()]}

    return inspect


def _resume_request_digest(session, route: CodexRouteReceipt) -> str:
    return _digest(
        {
            "action": "resume",
            "session_id": session.session_id,
            "provider_session_id": session.provider_session_id,
            "route": route.to_dict(),
            "context": session.context.to_dict(),
            "source_digest": session.source_digest,
            "source_revision": session.source_revision,
            "token_budget": 0,
            "compute_budget": 0.0,
        }
    )


class ZeroComputeAppServerProvider(CodexAppServerProvider):
    """No-process App Server seam with controllable turn accounting."""

    def __init__(self) -> None:
        """Configure deterministic lifecycle responses."""

        super().__init__()
        self.calls: list[str] = []
        self.start_compute = 1.0
        self.resume_compute = 1.0

    def start(self, *, route, prompt, evidence, reserved_compute=None):
        self.calls.append("start")
        return {
            "status": "complete",
            "provider_session_id": "app-server-session",
            "message": "recorded App Server turn",
            "usage": {"tokens": 0, "compute": self.start_compute},
        }

    def resume(self, *, provider_session_id, route, evidence, reserved_compute=None):
        self.calls.append("resume")
        return {
            "status": "complete",
            "provider_session_id": provider_session_id,
            "message": "recorded App Server resume",
            "usage": {"tokens": 0, "compute": self.resume_compute},
        }


def test_fresh_client_recovers_session_and_replays_terminal_operation_without_provider_work(
    tmp_path: Path,
) -> None:
    service, audit = _setup(tmp_path, token_budget=5, compute_budget=5.0)
    route = _route()
    inspector = _make_inspector(route)
    first_provider = FakeCodexProvider()
    first = AuditCodexClient(service, inspector=inspector, provider=first_provider)
    started = first.start(audit, prompt="inspect once", operation_id="durable-start")
    assert started.status == "complete"
    assert started.session is not None
    local_replay = first.start(audit, prompt="inspect once", operation_id="durable-start")
    assert local_replay.status == "complete"
    assert local_replay.session is started.session
    codex_session_id = started.session.session_id
    token = audit.session_token
    policy = audit.policy
    service.close()

    reopened = AuditService(
        tmp_path / "store",
        campaign_source=tmp_path / "allowed" / "campaign.json",
        source_root=tmp_path / "allowed",
        trusted_policy=policy,
    )
    second_provider = FakeCodexProvider()
    second = AuditCodexClient(reopened, inspector=inspector, provider=second_provider)
    recovered = second.recover_session(codex_session_id, audit_token=token)
    assert recovered.status == "complete"
    assert recovered.session is not None

    resumed = second.resume(
        codex_session_id,
        audit_token=token,
        operation_id="durable-resume",
    )
    assert resumed.status == "complete"
    assert len(second_provider.calls) == 1
    replay = second.resume(
        codex_session_id,
        audit_token=token,
        operation_id="durable-resume",
    )
    assert replay.status == "complete"
    assert replay.receipt is not None and replay.receipt.replayed
    assert len(second_provider.calls) == 1
    reopened.close()


def test_read_codex_activity_projects_durable_status_without_authority_internals(
    tmp_path: Path,
) -> None:
    service, audit = _setup(tmp_path)
    route = _route()
    client = AuditCodexClient(
        service, inspector=_make_inspector(route), provider=FakeCodexProvider()
    )
    started = client.start(audit, prompt="read the durable activity", operation_id="activity-start")
    assert started.status == "complete"
    assert started.session is not None

    activity = service.read_codex_activity(
        audit,
        codex_session_id=started.session.session_id,
        audit_token=audit.session_token,
    )
    assert activity.status == "complete"
    assert activity.session is not None
    assert activity.session.codex_session_id == started.session.session_id
    assert activity.operation is not None
    assert activity.operation.operation_id == "activity-start"
    assert activity.operation.status == "finished"
    assert activity.operation.result_status == "complete"
    assert activity.operation.usage == {"tokens": 0, "compute": 0.0, "issue_writes": 0}
    assert activity.evidence
    assert activity.events == ()
    assert activity.activity_scope == "unavailable"

    payload = activity.to_dict()
    serialized = json.dumps(payload)
    assert audit.session_token not in serialized
    assert "request_digest" not in serialized
    assert "reservation_id" not in serialized
    assert "reservation_operation_id" not in serialized
    assert "source_ref" not in serialized
    assert "provider_session_id" not in serialized


def test_read_codex_activity_recovers_durable_status_after_service_restart(
    tmp_path: Path,
) -> None:
    service, audit = _setup(tmp_path)
    route = _route()
    client = AuditCodexClient(
        service, inspector=_make_inspector(route), provider=FakeCodexProvider()
    )
    started = client.start(audit, prompt="restart-safe status", operation_id="restart-read")
    assert started.status == "complete"
    assert started.session is not None
    codex_session_id = started.session.session_id
    token = audit.session_token
    policy = audit.policy
    service.close()

    reopened = AuditService(
        tmp_path / "store",
        campaign_source=tmp_path / "allowed" / "campaign.json",
        source_root=tmp_path / "allowed",
        trusted_policy=policy,
    )
    try:
        activity = reopened.read_codex_activity(
            audit.session_id,
            codex_session_id=codex_session_id,
            audit_token=token,
        )
        assert activity.status == "complete"
        assert activity.operation is not None
        assert activity.operation.operation_id == "restart-read"
        assert activity.events == ()
        assert activity.activity_scope == "unavailable"
        assert "unavailable" in activity.events_reason
    finally:
        reopened.close()


def test_read_codex_activity_rejects_stale_foreign_and_unknown_selectors(
    tmp_path: Path,
) -> None:
    service, audit = _setup(tmp_path)
    route = _route()
    provider = FakeCodexProvider()
    client = AuditCodexClient(service, inspector=_make_inspector(route), provider=provider)
    first = client.start(audit, prompt="bound first activity", operation_id="bound-first")
    assert first.status == "complete"
    assert first.session is not None

    foreign_audit = service.open_session(
        audit.context,
        actor="agent",
        actor_id="other-agent",
        policy=audit.policy,
    )
    foreign = client.start(
        foreign_audit,
        prompt="bound foreign activity",
        operation_id="bound-foreign",
    )
    assert foreign.status == "complete"
    assert foreign.session is not None

    with pytest.raises(AuditPolicyError):
        service.read_codex_activity(
            audit,
            codex_session_id=foreign.session.session_id,
            audit_token=audit.session_token,
        )
    with pytest.raises(AuditPolicyError):
        service.read_codex_activity(
            audit,
            operation_id="bound-foreign",
            audit_token=audit.session_token,
        )
    with pytest.raises(AuditPolicyError):
        service.read_codex_activity(
            audit,
            codex_session_id="missing-codex-session",
            audit_token=audit.session_token,
        )
    with pytest.raises(AuditPolicyError):
        wrong_token = f"wrong-{audit.session_id}"
        service.read_codex_activity(
            audit,
            codex_session_id=first.session.session_id,
            audit_token=wrong_token,
        )

    moved = service.update_context(
        audit,
        audit.context.next_revision(time_s=1.0),
        expected_context_revision=audit.context.context_revision,
    )
    assert moved.status == "committed"
    with pytest.raises(AuditContextConflict):
        service.read_codex_activity(
            audit,
            codex_session_id=first.session.session_id,
            audit_token=audit.session_token,
        )


def test_read_codex_activity_reports_inflight_operation_without_reservation_details(
    tmp_path: Path,
) -> None:
    service, audit = _setup(tmp_path, token_budget=5, compute_budget=5.0)
    route = _route()
    lease = service.begin_codex_operation(
        audit,
        operation_id="activity-inflight",
        action="start",
        request_digest=_digest({"operation": "activity-inflight"}),
        context=audit.context,
        route_digest=route.capability_digest,
        reserved_tokens=1,
        reserved_compute=1.0,
        audit_token=audit.session_token,
    )
    assert lease.status == "admitted"

    activity = service.read_codex_activity(
        audit,
        operation_id="activity-inflight",
        audit_token=audit.session_token,
    )
    assert activity.session is None
    assert activity.status == "inflight"
    assert activity.operation is not None
    assert activity.operation.status == "inflight"
    assert activity.operation.result_status == ""
    assert activity.usage.to_dict() == {"tokens": 1, "compute": 1.0, "issue_writes": 0}
    serialized = json.dumps(activity.to_dict())
    assert "reservation_id" not in serialized
    assert "request_digest" not in serialized


def test_read_codex_activity_rejects_mutated_source(tmp_path: Path) -> None:
    service, audit = _setup(tmp_path)
    route = _route()
    client = AuditCodexClient(
        service, inspector=_make_inspector(route), provider=FakeCodexProvider()
    )
    started = client.start(audit, prompt="source-bound activity", operation_id="source-read")
    assert started.status == "complete"
    assert started.session is not None

    source = Path(service.campaign_source)
    source.write_bytes(source.read_bytes() + b"\nsource changed")
    with pytest.raises(AuditContextConflict):
        service.read_codex_activity(
            audit,
            codex_session_id=started.session.session_id,
            audit_token=audit.session_token,
        )


@pytest.mark.parametrize("mutation", ("invalid_usage", "forbidden_prompt"))
def test_read_codex_activity_rejects_malformed_nested_authority_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """A trusted malformed extension row cannot be promoted to activity truth."""

    service, audit = _setup(tmp_path)
    client = AuditCodexClient(
        service, inspector=_make_inspector(_route()), provider=FakeCodexProvider()
    )
    started = client.start(audit, prompt="inspect nested result", operation_id="nested-read")
    assert started.status == "complete"
    state = deepcopy(service.authority.snapshot())
    result = state["extensions"]["codex"]["operations"]["nested-read"]["result"]
    if mutation == "invalid_usage":
        result["usage"]["tokens"] = "not-an-integer"
    else:
        result["prompt"] = "should-never-be-public"
    monkeypatch.setattr(service.authority, "snapshot", lambda: state)

    with pytest.raises(AuditValidationError):
        service.read_codex_activity(
            audit,
            operation_id="nested-read",
            audit_token=audit.session_token,
        )


@pytest.mark.parametrize("action", ("start", "resume"))
@pytest.mark.parametrize(
    "mutation", ("empty-reservation", "wrong-operation", "unknown-reservation")
)
def test_lost_durable_admission_capability_stops_before_provider_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str, mutation: str
) -> None:
    """A committed hold is ambiguous when its returned lease is malformed."""

    service, audit = _setup(tmp_path, token_budget=5, compute_budget=5.0)
    route = _route()
    provider = FakeCodexProvider()
    client = AuditCodexClient(service, inspector=_make_inspector(route), provider=provider)
    original_begin = service.begin_codex_operation
    corrupt_admission = [action == "start"]

    def malformed_admission(*args, **kwargs):
        lease = original_begin(*args, **kwargs)
        if corrupt_admission[0] and lease.status == "admitted":
            if mutation == "empty-reservation":
                return replace(lease, reservation_id="")
            if mutation == "wrong-operation":
                assert lease.operation is not None
                return replace(
                    lease,
                    operation=replace(lease.operation, operation_id="wrong-operation"),
                )
            assert lease.operation is not None
            return replace(
                lease,
                operation=replace(
                    lease.operation,
                    reservation_id="unknown-reservation",
                    reservation_operation_id="unknown-reserve-operation",
                ),
                reservation_id="unknown-reservation",
                reservation_operation_id="unknown-reserve-operation",
            )
        return lease

    monkeypatch.setattr(service, "begin_codex_operation", malformed_admission)
    try:
        if action == "start":
            result = client.start(
                audit,
                prompt="malformed durable admission",
                token_budget=1,
                compute_budget=1.0,
                operation_id="malformed-start",
            )
        else:
            started = client.start(
                audit,
                prompt="prepare durable resume",
                token_budget=1,
                compute_budget=1.0,
                operation_id="prepare-resume",
            )
            assert started.status == "complete"
            assert started.session is not None
            corrupt_admission[0] = True
            result = client.resume(
                started.session,
                token_budget=1,
                compute_budget=1.0,
                operation_id="malformed-resume",
            )
        assert result.status == "unavailable"
        assert "admission capability" in result.reason
        assert [call["action"] for call in provider.calls] == (
            [] if action == "start" else ["start"]
        )
        assert len(service._reservations) == 1
        operation_id = "malformed-start" if action == "start" else "malformed-resume"
        state = service.authority.snapshot()
        operation = state["extensions"]["codex"]["operations"][operation_id]
        assert operation["status"] == "inflight"
        assert (
            state["reservations"][operation["reservation_id"]]["operation_id"]
            == operation["reservation_operation_id"]
        )

        retry = (
            client.start(
                audit,
                prompt="malformed durable admission",
                token_budget=1,
                compute_budget=1.0,
                operation_id=operation_id,
            )
            if action == "start"
            else client.resume(
                started.session,
                token_budget=1,
                compute_budget=1.0,
                operation_id="malformed-resume",
            )
        )
        assert retry.status == "unavailable"
        assert len(service._reservations) == 1
        assert [call["action"] for call in provider.calls] == (
            [] if action == "start" else ["start"]
        )
    finally:
        service.close()


def test_durable_app_server_zero_compute_success_burns_start_reservation(
    tmp_path: Path,
) -> None:
    service, audit = _setup(tmp_path, token_budget=7, compute_budget=1.0)
    route = _route()
    provider = ZeroComputeAppServerProvider()
    provider.start_compute = 0.0
    client = AuditCodexClient(service, inspector=_make_inspector(route), provider=provider)
    try:
        result = client.start(
            audit,
            prompt="zero telemetry must not refund a live turn",
            token_budget=7,
            compute_budget=1.0,
            operation_id="zero-compute-start",
        )
        assert result.status == "failed"
        assert provider.calls == ["start"]
        assert audit.usage.to_dict() == {"tokens": 7, "compute": 1.0, "issue_writes": 0}
        assert not service._reservations
        operation = service.authority.snapshot()["extensions"]["codex"]["operations"][
            "zero-compute-start"
        ]
        assert operation["result_status"] == "failed"
        assert operation["result"]["usage"] == {
            "tokens": 7,
            "compute": 1.0,
            "issue_writes": 0,
        }
    finally:
        provider.close()
        service.close()


def test_durable_app_server_zero_compute_success_burns_resume_reservation(
    tmp_path: Path,
) -> None:
    service, audit = _setup(tmp_path, token_budget=5, compute_budget=2.0)
    route = _route()
    provider = ZeroComputeAppServerProvider()
    provider.resume_compute = 0.0
    client = AuditCodexClient(service, inspector=_make_inspector(route), provider=provider)
    try:
        started = client.start(
            audit,
            prompt="prepare zero telemetry resume",
            token_budget=1,
            compute_budget=1.0,
            operation_id="zero-compute-resume-start",
        )
        assert started.status == "complete"
        assert started.session is not None
        resumed = client.resume(
            started.session,
            token_budget=1,
            compute_budget=1.0,
            operation_id="zero-compute-resume",
        )
        assert resumed.status == "failed"
        assert provider.calls == ["start", "resume"]
        assert audit.usage.to_dict() == {"tokens": 1, "compute": 2.0, "issue_writes": 0}
        assert not service._reservations
    finally:
        provider.close()
        service.close()


def test_new_cancel_operation_on_cancelled_session_is_provider_free(tmp_path: Path) -> None:
    service, audit = _setup(tmp_path)
    route = _route()
    provider = FakeCodexProvider()
    client = AuditCodexClient(service, inspector=_make_inspector(route), provider=provider)
    try:
        started = client.start(audit, prompt="cancel once", operation_id="cancel-start")
        assert started.status == "complete"
        assert started.session is not None
        first = client.cancel(
            started.session,
            operation_id="cancel-first",
            reason="stop once",
        )
        assert first.status == "cancelled"
        assert first.session is not None
        assert [call["action"] for call in provider.calls] == ["start", "cancel"]

        repeated = client.cancel(
            first.session,
            operation_id="cancel-second",
            reason="stop again",
        )
        assert repeated.status == "conflict"
        assert repeated.receipt is not None
        assert not repeated.receipt.replayed
        assert [call["action"] for call in provider.calls] == ["start", "cancel"]
        operation = service.authority.snapshot()["extensions"]["codex"]["operations"][
            "cancel-second"
        ]
        assert operation["status"] == "finished"
        assert operation["result_status"] == "conflict"

        replay = client.cancel(
            repeated.session,
            operation_id="cancel-second",
            reason="stop again",
        )
        assert replay.status == "conflict"
        assert replay.receipt is not None and replay.receipt.replayed
        assert [call["action"] for call in provider.calls] == ["start", "cancel"]
    finally:
        service.close()


def test_context_update_after_codex_session_preserves_historical_reopen(tmp_path: Path) -> None:
    service, audit = _setup(tmp_path)
    route = _route()
    client = AuditCodexClient(
        service, inspector=_make_inspector(route), provider=FakeCodexProvider()
    )
    started = client.start(audit, prompt="keep context bound", operation_id="context-bound-start")
    assert started.status == "complete"
    assert started.session is not None
    moved = service.update_context(
        audit,
        audit.context.next_revision(time_s=1.0),
        expected_context_revision=0,
    )
    assert moved.status == "committed"
    assert audit.context.context_revision == 1
    token = audit.session_token
    policy = audit.policy
    service.close()

    reopened = AuditService(
        tmp_path / "store",
        campaign_source=tmp_path / "allowed" / "campaign.json",
        source_root=tmp_path / "allowed",
        trusted_policy=policy,
    )
    recovered_provider = FakeCodexProvider()
    try:
        recovered_audit = reopened.reconnect_session(audit.session_id, token)
        recovered_client = AuditCodexClient(
            reopened,
            inspector=_make_inspector(route),
            provider=recovered_provider,
        )
        recovered = recovered_client.recover_session(started.session.session_id, audit_token=token)
        assert recovered_audit.context.context_revision == 1
        assert recovered.status == "complete"
        assert recovered.session is not None
        assert recovered.session.context.context_revision == 0
        stale_resume = recovered_client.resume(
            recovered.session,
            operation_id="historical-stale-resume",
            audit_token=token,
        )
        assert stale_resume.status == "conflict"
        assert recovered_provider.calls == []
    finally:
        reopened.close()


@pytest.mark.parametrize("queue_style", (False, True), ids=("context-cas", "next-style"))
def test_admitted_source_revision_preserves_historical_reopen(
    tmp_path: Path, queue_style: bool
) -> None:
    service, audit = _setup(tmp_path, source_revision=7)
    route = _route()
    provider = FakeCodexProvider()
    client = AuditCodexClient(
        service,
        inspector=_make_inspector(route),
        provider=provider,
    )
    started = client.start(audit, prompt="source revision history", operation_id="revision-start")
    assert started.status == "complete"
    assert started.session is not None
    assert audit.context.source_revision == 7
    updated = audit.context.next_revision(
        episode_id="fixture-readable",
        source_identity=audit.source_digest if queue_style else audit.context.source_identity,
        source_revision=audit.source_revision if queue_style else audit.context.source_revision,
    )
    if queue_style:

        def commit_selection(working, _state):
            service._set_context_locked(working, updated)
            return updated.to_dict()

        value, _replayed = service._mutate_session_authority(
            audit,
            transaction_id=f"authority.queue.next.context:{audit.session_id}:revision-next",
            request_digest=_digest({"operation_id": "revision-next", "queue_style": True}),
            callback=commit_selection,
        )
        assert value["source_revision"] == 7
    else:
        moved = service.update_context(audit, updated, expected_context_revision=0)
        assert moved.status == "committed"
    assert audit.context.context_revision == 1
    token = audit.session_token
    policy = audit.policy
    codex_session_id = started.session.session_id
    session_id = audit.session_id
    service.close()

    reopened = AuditService(
        tmp_path / "store",
        campaign_source=tmp_path / "allowed" / "campaign.json",
        source_root=tmp_path / "allowed",
        trusted_policy=policy,
    )
    recovered_provider = FakeCodexProvider()
    try:
        recovered_audit = reopened.reconnect_session(session_id, token)
        assert recovered_audit.context.source_revision == 7
        recovered_client = AuditCodexClient(
            reopened,
            inspector=_make_inspector(route),
            provider=recovered_provider,
        )
        recovered = recovered_client.recover_session(codex_session_id, audit_token=token)
        assert recovered.status == "complete"
        assert recovered.session is not None
        assert recovered.session.context.context_revision == 0
        assert recovered.session.context.source_revision == 7
        stale_resume = recovered_client.resume(
            recovered.session,
            operation_id="revision-stale-resume",
            audit_token=token,
        )
        assert stale_resume.status == "conflict"
        assert recovered_provider.calls == []
    finally:
        reopened.close()


def test_inflight_operation_is_ambiguous_after_restart_and_keeps_reservation(
    tmp_path: Path,
) -> None:
    service, audit = _setup(tmp_path, token_budget=2, compute_budget=2.0)
    route = _route()
    request_digest = hashlib.sha256(b"durable-inflight-request").hexdigest()
    admitted = service.begin_codex_operation(
        audit,
        operation_id="durable-inflight",
        action="start",
        request_digest=request_digest,
        context=audit.context,
        route_digest=route.capability_digest,
        reserved_tokens=1,
        reserved_compute=1.0,
    )
    assert admitted.status == "admitted"
    token = audit.session_token
    policy = audit.policy
    service.close()

    reopened = AuditService(
        tmp_path / "store",
        campaign_source=tmp_path / "allowed" / "campaign.json",
        source_root=tmp_path / "allowed",
        trusted_policy=policy,
    )
    recovered = reopened.reconnect_session(audit.session_id, token)
    ambiguous = reopened.begin_codex_operation(
        recovered,
        operation_id="durable-inflight",
        action="start",
        request_digest=request_digest,
        context=recovered.context,
        route_digest=route.capability_digest,
        reserved_tokens=1,
        reserved_compute=1.0,
    )
    assert ambiguous.status == "ambiguous"
    different = reopened.begin_codex_operation(
        recovered,
        operation_id="durable-inflight-retry",
        action="start",
        request_digest=hashlib.sha256(b"different-request").hexdigest(),
        context=recovered.context,
        route_digest=route.capability_digest,
        reserved_tokens=1,
        reserved_compute=1.0,
    )
    assert different.status == "ambiguous"
    assert len(reopened._reservations) == 1
    assert recovered.usage.tokens == 1
    reopened.close()


def test_recovery_requires_token_source_and_route_reauthentication(tmp_path: Path) -> None:
    service, audit = _setup(tmp_path)
    route = _route()
    inspector = _make_inspector(route)
    client = AuditCodexClient(service, inspector=inspector, provider=FakeCodexProvider())
    started = client.start(audit, prompt="inspect", operation_id="reauth-start")
    assert started.session is not None
    token = audit.session_token
    policy = audit.policy
    codex_session_id = started.session.session_id
    source = Path(service.campaign_source)
    service.close()

    source.write_bytes(source.read_bytes() + b" changed")
    reopened = AuditService(
        tmp_path / "store",
        campaign_source=source,
        source_root=tmp_path / "allowed",
        trusted_policy=policy,
    )
    second = AuditCodexClient(reopened, inspector=inspector, provider=FakeCodexProvider())
    wrong_token_value = token[:-1] + ("x" if token[-1] != "x" else "y")
    wrong_token = second.recover_session(codex_session_id, audit_token=wrong_token_value)
    assert wrong_token.status == "conflict"
    stale_source = second.recover_session(codex_session_id, audit_token=token)
    assert stale_source.status == "conflict"
    reopened.close()


def test_codex_authority_journal_never_contains_audit_token(tmp_path: Path) -> None:
    service, audit = _setup(tmp_path)
    route = _route()
    client = AuditCodexClient(
        service,
        inspector=_make_inspector(route),
        provider=FakeCodexProvider(),
    )
    assert (
        client.start(audit, prompt="redaction", operation_id="redaction-start").status == "complete"
    )
    journal = (tmp_path / "store" / "audit-authority.ndjson").read_text(encoding="utf-8")
    assert audit.session_token not in journal
    service.close()


def test_codex_reservation_operation_id_collision_fails_closed(tmp_path: Path) -> None:
    service, audit = _setup(tmp_path)
    route = _route()
    collision = service.consume_budget(
        audit,
        operation_id="codex-budget-reserve:collision",
    )
    assert collision.status == "committed"
    request_digest = hashlib.sha256(b"collision-request").hexdigest()
    with pytest.raises(AuthorityOperationConflict):
        service.begin_codex_operation(
            audit,
            operation_id="collision",
            action="start",
            request_digest=request_digest,
            context=audit.context,
            route_digest=route.capability_digest,
            reserved_tokens=0,
            reserved_compute=0.0,
        )
    state = service.authority.snapshot()
    assert state["operations"]["codex-budget-reserve:collision"]["operation_type"] == (
        "budget.consume"
    )
    service.close()


def _prepare_direct_finish(tmp_path: Path):
    service, audit = _setup(tmp_path)
    route = _route()
    client = AuditCodexClient(
        service,
        inspector=_make_inspector(route),
        provider=FakeCodexProvider(),
    )
    started = client.start(audit, prompt="prepare", operation_id="prepare-start")
    assert started.session is not None
    request_digest = _resume_request_digest(started.session, route)
    lease = service.begin_codex_operation(
        audit,
        operation_id="forged-finish",
        action="resume",
        codex_session_id=started.session.session_id,
        request_digest=request_digest,
        context=started.session.context,
        route_digest=route.capability_digest,
    )
    assert lease.status == "admitted"
    snapshot = client._authority_session_snapshot(
        started.session,
        audit,
        last_operation_id="forged-finish",
    )
    return service, audit, route, request_digest, snapshot


def test_generic_settlement_cannot_orphan_durable_codex_reservation(tmp_path: Path) -> None:
    service, audit, route, request_digest, snapshot = _prepare_direct_finish(tmp_path)
    try:
        operation = service.authority.snapshot()["extensions"]["codex"]["operations"][
            "forged-finish"
        ]
        reservation_id = operation["reservation_id"]
        generic = service.settle_budget(
            audit,
            reserved_tokens=0,
            reserved_compute=0.0,
            actual_tokens=0,
            actual_compute=0.0,
            reservation_id=reservation_id,
            reservation_operation_id=operation["reservation_operation_id"],
            operation_id="generic-codex-settlement",
        )
        assert generic.status == "denied"
        assert "Codex reservation" in generic.reason
        state = service.authority.snapshot()
        assert reservation_id in state["reservations"]
        assert state["extensions"]["codex"]["operations"]["forged-finish"]["status"] == ("inflight")
        finished = service.finish_codex_operation(
            audit,
            operation_id="forged-finish",
            request_digest=request_digest,
            result_status="complete",
            result={"status": "complete", "usage": {"tokens": 0, "compute": 0.0}},
            provider_session_id=snapshot.provider_session_id,
            usage={"tokens": 0, "compute": 0.0},
            session_snapshot=snapshot,
            route_digest=route.capability_digest,
        )
        assert finished.status == "replay" and not finished.replayed
        assert (
            service.authority.snapshot()["extensions"]["codex"]["operations"]["forged-finish"][
                "status"
            ]
            == "finished"
        )
        assert reservation_id not in service.authority.snapshot()["reservations"]
        assert service.kill_switch(audit, operation_id="stop-after-codex-finish").status == (
            "cancelled"
        )
        token = audit.session_token
        policy = audit.policy
    finally:
        service.close()

    reopened = AuditService(
        tmp_path / "store",
        campaign_source=tmp_path / "allowed" / "campaign.json",
        source_root=tmp_path / "allowed",
        trusted_policy=policy,
    )
    try:
        assert reopened.reconnect_session(audit.session_id, token).cancelled is True
        assert not reopened.authority.snapshot()["reservations"]
    finally:
        reopened.close()


def test_finish_rejects_forged_evidence_before_persisting(tmp_path: Path) -> None:
    service, audit, route, request_digest, snapshot = _prepare_direct_finish(tmp_path)
    forged_evidence = dict(snapshot.evidence[0])
    forged_evidence["artifact_id"] = "forged-artifact"
    forged = replace(snapshot, evidence=(forged_evidence,))
    with pytest.raises(AuditContextConflict, match="evidence"):
        service.finish_codex_operation(
            audit,
            operation_id="forged-finish",
            request_digest=request_digest,
            result_status="complete",
            result={"status": "complete", "usage": {"tokens": 0, "compute": 0.0}},
            provider_session_id=snapshot.provider_session_id,
            usage={"tokens": 0, "compute": 0.0},
            session_snapshot=forged,
            route_digest=route.capability_digest,
        )
    raw = service.authority.snapshot()["extensions"]["codex"]["operations"]["forged-finish"]
    assert raw["status"] == "inflight"
    service.close()


def test_finish_rejects_forged_route_before_persisting(tmp_path: Path) -> None:
    service, audit, route, request_digest, snapshot = _prepare_direct_finish(tmp_path)
    forged_route = dict(snapshot.route)
    forged_route["model_id"] = "forged-model"
    forged = replace(snapshot, route=forged_route)
    with pytest.raises(AuditContextConflict, match="route"):
        service.finish_codex_operation(
            audit,
            operation_id="forged-finish",
            request_digest=request_digest,
            result_status="complete",
            result={"status": "complete", "usage": {"tokens": 0, "compute": 0.0}},
            provider_session_id=snapshot.provider_session_id,
            usage={"tokens": 0, "compute": 0.0},
            session_snapshot=forged,
            route_digest=route.capability_digest,
        )
    raw = service.authority.snapshot()["extensions"]["codex"]["operations"]["forged-finish"]
    assert raw["status"] == "inflight"
    service.close()


def test_provider_result_values_cannot_persist_the_audit_token(tmp_path: Path) -> None:
    service, audit = _setup(tmp_path)
    route = _route()
    provider = FakeCodexProvider(
        response={
            "status": "failed",
            "reason": audit.session_token,
            "usage": {"tokens": 0, "compute": 0.0},
        }
    )
    client = AuditCodexClient(
        service,
        inspector=_make_inspector(route),
        provider=provider,
    )
    result = client.start(audit, prompt="secret result", operation_id="token-value")
    assert result.status == "unavailable"
    journal = (tmp_path / "store" / "audit-authority.ndjson").read_text(encoding="utf-8")
    assert audit.session_token not in journal
    service.close()


def test_empty_existing_authority_journal_fails_closed(tmp_path: Path) -> None:
    store = tmp_path / "store"
    store.mkdir()
    (store / "audit-authority.ndjson").write_text("", encoding="utf-8")
    with pytest.raises(AuthorityCorruptionError):
        AuditAuthorityStore(store)


def test_codex_authority_inputs_reject_audit_token_before_any_write(tmp_path: Path) -> None:
    service, audit = _setup(tmp_path)
    route = _route()
    provider = FakeCodexProvider()
    client = AuditCodexClient(service, inspector=_make_inspector(route), provider=provider)
    token = audit.session_token
    try:
        sensitive_start = client.start(audit, prompt="secret input", operation_id=token)
        assert sensitive_start.status == "conflict"
        assert provider.calls == []

        started = client.start(audit, prompt="safe input", operation_id="safe-start")
        assert started.status == "complete"
        sensitive_cancel = client.cancel(
            started.session,
            operation_id="safe-cancel",
            reason=token,
        )
        assert sensitive_cancel.status == "conflict"
        assert all(call["action"] != "cancel" for call in provider.calls)

        with pytest.raises(AuditPolicyError):
            service.kill_switch(audit, operation_id="generic-kill", reason=token)
        generic = service.consume_budget(audit, operation_id=token)
        assert generic.status == "conflict"
        journal = (tmp_path / "store" / "audit-authority.ndjson").read_text(encoding="utf-8")
        assert token not in journal
    finally:
        service.close()


def test_finish_rebinds_snapshot_against_authoritative_context(tmp_path: Path) -> None:
    service, audit = _setup(tmp_path)
    route = _route()
    provider = FakeCodexProvider()
    client = AuditCodexClient(service, inspector=_make_inspector(route), provider=provider)
    try:
        started = client.start(audit, prompt="context race", operation_id="context-start")
        assert started.status == "complete"
        original_mutate = service.authority.mutate
        interposed = [False]

        def mutate_with_context_change(*args, **kwargs):
            transaction_id = args[0] if args else kwargs.get("transaction_id", "")
            if (
                isinstance(transaction_id, str)
                and transaction_id.startswith("authority.codex.finish:")
                and not interposed[0]
            ):
                interposed[0] = True
                updated = audit.context.next_revision(time_s=1.0)
                assert (
                    service.update_context(
                        audit,
                        updated,
                        expected_context_revision=0,
                    ).status
                    == "committed"
                )
            return original_mutate(*args, **kwargs)

        service.authority.mutate = mutate_with_context_change  # type: ignore[method-assign]
        result = client.resume(started.session, operation_id="context-race-resume")
        assert result.status == "conflict"
        state = service.authority.snapshot()
        operation = state["extensions"]["codex"]["operations"]["context-race-resume"]
        assert operation["status"] == "inflight"
        stored = state["extensions"]["codex"]["sessions"][started.session.session_id]
        assert stored["context"]["context_revision"] == 0
        assert state["sessions"][audit.session_id]["context"]["context_revision"] == 1
        token = audit.session_token
        policy = audit.policy
    finally:
        service.close()

    reopened = AuditService(
        tmp_path / "store",
        campaign_source=tmp_path / "allowed" / "campaign.json",
        source_root=tmp_path / "allowed",
        trusted_policy=policy,
    )
    try:
        recovered = reopened.reconnect_session(audit.session_id, token)
        assert recovered.context.context_revision == 1
        recovered_provider = FakeCodexProvider()
        recovered_codex = AuditCodexClient(
            reopened,
            inspector=_make_inspector(route),
            provider=recovered_provider,
        )
        recovered_result = recovered_codex.recover_session(
            started.session.session_id, audit_token=token
        )
        assert recovered_result.status == "complete"
        assert recovered_result.session is not None
        stale_retry = recovered_codex.resume(
            recovered_result.session,
            operation_id="context-race-resume",
            audit_token=token,
        )
        assert stale_retry.status == "conflict"
        assert recovered_provider.calls == []
        assert len(reopened._reservations) == 1
        assert recovered_result.session.context.context_revision == 0
    finally:
        reopened.close()


def test_finished_operation_replays_after_historical_context_advance(tmp_path: Path) -> None:
    service, audit = _setup(tmp_path)
    route = _route()
    client = AuditCodexClient(
        service,
        inspector=_make_inspector(route),
        provider=FakeCodexProvider(),
    )
    try:
        started = client.start(audit, prompt="historical terminal", operation_id="terminal-start")
        assert started.status == "complete"
        assert started.session is not None
        state = service.authority.snapshot()
        operation = state["extensions"]["codex"]["operations"]["terminal-start"]
        snapshot = state["extensions"]["codex"]["sessions"][started.session.session_id]
        assert operation["status"] == "finished"
        assert snapshot["context"]["context_revision"] == 0
        assert (
            service.update_context(
                audit,
                audit.context.next_revision(time_s=1.0),
                expected_context_revision=0,
            ).status
            == "committed"
        )

        replay = service.finish_codex_operation(
            audit,
            operation_id="terminal-start",
            request_digest=operation["request_digest"],
            result_status=operation["result_status"],
            result=operation["result"],
            provider_session_id=operation["provider_session_id"],
            usage=operation["result"]["usage"],
            session_snapshot=snapshot,
            route_digest=operation["route_digest"],
        )
        assert replay.status == "replay"
        assert replay.replayed
        assert replay.session is not None
        assert replay.session.context["context_revision"] == 0
        assert not service._reservations
    finally:
        service.close()


def test_finish_derives_denial_from_authoritative_budget_state(tmp_path: Path) -> None:
    service, audit = _setup(tmp_path, token_budget=5, compute_budget=5.0)
    route = _route()
    provider = FakeCodexProvider(response={"usage": {"tokens": 2, "compute": 0.0}})
    client = AuditCodexClient(service, inspector=_make_inspector(route), provider=provider)
    try:
        original_mutate = service.authority.mutate
        interposed = [False]

        def mutate_with_budget_change(*args, **kwargs):
            transaction_id = args[0] if args else kwargs.get("transaction_id", "")
            if (
                isinstance(transaction_id, str)
                and transaction_id.startswith("authority.codex.finish:")
                and not interposed[0]
            ):
                interposed[0] = True
                assert service.consume_budget(audit, tokens=4, operation_id="budget-race").ok
            return original_mutate(*args, **kwargs)

        service.authority.mutate = mutate_with_budget_change  # type: ignore[method-assign]
        result = client.start(
            audit,
            prompt="budget race",
            operation_id="budget-race-start",
            token_budget=1,
        )
        assert result.status == "denied"
        assert result.receipt is not None
        state = service.authority.snapshot()
        operation = state["extensions"]["codex"]["operations"]["budget-race-start"]
        assert operation["result_status"] == "denied"
        assert "budget" in operation["reason"].lower()
        assert state["sessions"][audit.session_id]["usage"]["tokens"] == 5
    finally:
        service.close()


def test_stale_cancel_collision_is_durable_and_provider_free(tmp_path: Path) -> None:
    service, audit = _setup(tmp_path)
    route = _route()
    provider = FakeCodexProvider()
    client = AuditCodexClient(service, inspector=_make_inspector(route), provider=provider)
    source = Path(service.campaign_source)
    try:
        started = client.start(audit, prompt="stale collision", operation_id="stale-start")
        assert started.session is not None
        assert service.consume_budget(
            audit,
            operation_id="codex-budget-reserve:stale-cancel",
        ).ok
        source.write_bytes(source.read_bytes() + b" stale")
        cancelled = client.cancel(
            started.session,
            operation_id="stale-cancel",
            reason="stop on stale source",
        )
        assert cancelled.status == "conflict"
        assert cancelled.receipt is not None and not cancelled.receipt.replayed
        assert all(call["action"] != "cancel" for call in provider.calls)
        operation = service.authority.snapshot()["extensions"]["codex"]["operations"][
            "stale-cancel"
        ]
        assert operation["status"] == "finished"
        assert operation["result_status"] == "conflict"
        replay = client.cancel(
            started.session,
            operation_id="stale-cancel",
            reason="stop on stale source",
        )
        assert replay.status == "conflict"
        assert replay.receipt is not None and replay.receipt.replayed
    finally:
        service.close()


def test_historical_context_cancel_replay_is_durable_and_provider_free(tmp_path: Path) -> None:
    service, audit = _setup(tmp_path)
    route = _route()
    provider = FakeCodexProvider()
    client = AuditCodexClient(service, inspector=_make_inspector(route), provider=provider)
    try:
        started = client.start(audit, prompt="context cancellation", operation_id="context-start")
        assert started.session is not None
        updated = audit.context.next_revision(time_s=2.0)
        moved = service.update_context(audit, updated, expected_context_revision=0)
        assert moved.status == "committed"
        cancelled = client.cancel(
            started.session,
            operation_id="context-cancel",
            reason="stop after context change",
        )
        assert cancelled.status == "conflict"
        assert cancelled.receipt is not None and not cancelled.receipt.replayed
        replay = client.cancel(
            cancelled.session,
            operation_id="context-cancel",
            reason="stop after context change",
        )
        assert replay.status == "conflict"
        assert replay.receipt is not None and replay.receipt.replayed
        assert all(call["action"] != "cancel" for call in provider.calls)
        forged_handle = replace(started.session)
        forged_replay = client.cancel(
            forged_handle,
            operation_id="context-cancel",
            reason="stop after context change",
        )
        assert forged_replay.status == "conflict"
        assert forged_replay.receipt is None
        mismatch = client.cancel(
            cancelled.session,
            operation_id="context-cancel",
            reason="a different cancellation request",
        )
        assert mismatch.status == "conflict"
        assert mismatch.receipt is None
    finally:
        service.close()
