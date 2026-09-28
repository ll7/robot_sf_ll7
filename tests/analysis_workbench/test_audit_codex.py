"""Offline capability-gated Codex session protocol tests."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

import robot_sf.analysis_workbench.audit_codex as audit_codex_module
from robot_sf.analysis_workbench.audit_codex import (
    AuditCodexClient,
    CodexCapabilityInspection,
    CodexRouteReceipt,
    FakeCodexProvider,
    inspect_installed_capabilities,
)
from robot_sf.analysis_workbench.audit_codex_app_server import (
    APP_SERVER_PROTOCOL,
    CodexAppServerProvider,
)
from robot_sf.analysis_workbench.audit_service import (
    AuditSelectionContext,
    AuditService,
    ServiceResult,
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
):
    root = tmp_path / "allowed"
    root.mkdir()
    source = root / "campaign.json"
    shutil.copyfile(FIXTURE, source)
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


class RaisingProvider(FakeCodexProvider):
    """Offline provider that raises from one lifecycle operation."""

    def __init__(self, action: str) -> None:
        """Configure the lifecycle operation that should raise."""

        super().__init__()
        self.action = action

    def reconnect(self, *, provider_session_id: str, route: CodexRouteReceipt):
        if self.action == "reconnect":
            raise RuntimeError("transport exploded")
        return super().reconnect(provider_session_id=provider_session_id, route=route)

    def resume(
        self,
        *,
        provider_session_id: str,
        route: CodexRouteReceipt,
        evidence,
    ):
        if self.action == "resume":
            raise RuntimeError("transport exploded")
        return super().resume(
            provider_session_id=provider_session_id,
            route=route,
            evidence=evidence,
        )

    def cancel(self, *, provider_session_id: str, route: CodexRouteReceipt):
        if self.action == "cancel":
            raise RuntimeError("transport exploded")
        return super().cancel(provider_session_id=provider_session_id, route=route)


class RecordingAppServerProvider(CodexAppServerProvider):
    """No-process App Server seam that records the authority hook arguments."""

    def __init__(self) -> None:
        """Configure an in-memory provider without starting a subprocess."""

        super().__init__()
        self.calls: list[dict[str, object]] = []
        self.start_status = "complete"
        self.start_tokens = 0
        self.start_compute = 1.0
        self.resume_tokens = 7

    def start(self, *, route, prompt, evidence, reserved_compute=None):
        self.calls.append(
            {
                "action": "start",
                "route": route,
                "prompt": prompt,
                "evidence": list(evidence),
                "reserved_compute": reserved_compute,
            }
        )
        return {
            "status": self.start_status,
            "provider_session_id": "app-server-session",
            "message": "recorded App Server turn",
            "usage": {"tokens": self.start_tokens, "compute": self.start_compute},
        }

    def resume(self, *, provider_session_id, route, evidence, reserved_compute=None):
        self.calls.append(
            {
                "action": "resume",
                "provider_session_id": provider_session_id,
                "route": route,
                "evidence": list(evidence),
                "reserved_compute": reserved_compute,
            }
        )
        return {
            "status": "complete",
            "provider_session_id": provider_session_id,
            "message": "recorded App Server resume",
            "usage": {"tokens": self.resume_tokens, "compute": 1.0},
        }


class FailingAppServerProvider(RecordingAppServerProvider):
    """App Server seam that raises after a turn has been admitted."""

    def start(self, *, route, prompt, evidence, reserved_compute=None):
        """Record an admitted turn, then simulate ambiguous provider failure."""

        super().start(
            route=route,
            prompt=prompt,
            evidence=evidence,
            reserved_compute=reserved_compute,
        )
        raise RuntimeError("live App Server transport lost after turn admission")


def test_missing_route_is_unavailable_and_never_guessed() -> None:
    inspection = inspect_installed_capabilities(lambda: {"status": "available", "routes": []})
    assert inspection.status == "unavailable"
    assert inspection.routes == ()


def test_fake_provider_receipts_bind_route_context_evidence_and_lifecycle(tmp_path: Path) -> None:
    service, audit_session = _setup(tmp_path)
    route = CodexRouteReceipt(
        "route-fixture", "provider-fixture", "model-fixture", "1.0.0", "app-server"
    )
    provider = FakeCodexProvider()
    client = AuditCodexClient(
        service,
        inspector=lambda: {"status": "available", "routes": [route.to_dict()]},
        provider=provider,
    )
    try:
        started = client.start(
            audit_session, prompt="inspect selected evidence", operation_id="codex-start"
        )
        assert started.status == "complete"
        assert started.session is not None
        assert started.receipt is not None
        assert started.receipt.route == route
        assert started.session.evidence[0].source_digest == audit_session.source_digest
        assert started.session.evidence[0].locator.startswith("audit://")

        reconnected = client.reconnect(started.session, operation_id="codex-reconnect")
        resumed = client.resume(reconnected.session, operation_id="codex-resume")
        cancelled = client.cancel(resumed.session, operation_id="codex-cancel")
        assert reconnected.status == "complete"
        assert resumed.status == "complete"
        assert cancelled.status == "cancelled"
        assert {call["action"] for call in provider.calls} == {
            "start",
            "reconnect",
            "resume",
            "cancel",
        }
    finally:
        service.close()


def test_app_server_hook_forwards_the_durable_turn_reservation_and_preserves_binding(
    tmp_path: Path,
) -> None:
    service, audit_session = _setup(tmp_path, token_budget=10)
    route = CodexRouteReceipt(
        "provider-fixture:model-fixture",
        "provider-fixture",
        "model-fixture",
        "0.154.0",
        APP_SERVER_PROTOCOL,
        source="live-app-server-capability",
    )
    provider = RecordingAppServerProvider()
    client = AuditCodexClient(
        service,
        inspector=lambda: {"status": "available", "routes": [route.to_dict()]},
        provider=provider,
    )
    try:
        started = client.start(
            audit_session,
            prompt="inspect the admitted evidence",
            route_id=route.route_id,
            token_budget=1,
            compute_budget=1.0,
            operation_id="app-server-start",
        )
        assert started.status == "complete"
        assert started.session is not None
        assert started.receipt is not None
        assert provider.calls[0]["action"] == "start"
        assert provider.calls[0]["route"] == route
        assert provider.calls[0]["reserved_compute"] == 1.0
        assert provider.calls[0]["evidence"][0]["source_digest"] == audit_session.source_digest
        assert started.receipt.context_revision == audit_session.context.context_revision
        assert started.receipt.source_digest == audit_session.source_digest

        resumed = client.resume(
            started.session,
            token_budget=7,
            compute_budget=1.0,
            operation_id="app-server-resume",
        )
        assert resumed.status == "complete"
        assert provider.calls[1]["action"] == "resume"
        assert provider.calls[1]["route"] == route
        assert provider.calls[1]["reserved_compute"] == 1.0
        assert resumed.receipt is not None
        assert resumed.receipt.usage["tokens"] == 7
        assert audit_session.usage.compute == 2.0
        assert audit_session.usage.tokens == 7

        resume_replay = client.resume(
            resumed.session,
            token_budget=7,
            compute_budget=1.0,
            operation_id="app-server-resume",
        )
        assert resume_replay.status == "complete"
        assert resume_replay.receipt is not None and resume_replay.receipt.replayed

        conflicting_resume = client.resume(
            resumed.session,
            token_budget=6,
            compute_budget=1.0,
            operation_id="app-server-resume",
        )
        assert conflicting_resume.status == "conflict"
        assert len(provider.calls) == 2

        replay = client.start(
            audit_session,
            prompt="inspect the admitted evidence",
            route_id=route.route_id,
            token_budget=1,
            compute_budget=1.0,
            operation_id="app-server-start",
        )
        assert replay.status == "complete"
        assert replay.receipt is not None and replay.receipt.replayed
        assert len(provider.calls) == 2

        conflicting_replay = client.start(
            audit_session,
            prompt="inspect the admitted evidence",
            route_id=route.route_id,
            token_budget=1,
            compute_budget=2.0,
            operation_id="app-server-start",
        )
        assert conflicting_replay.status == "conflict"
        assert len(provider.calls) == 2
    finally:
        provider.close()
        service.close()


@pytest.mark.parametrize("compute_budget", [0.0, 0.5])
def test_app_server_hook_rejects_insufficient_start_reservation_before_provider_work(
    tmp_path: Path, compute_budget: float
) -> None:
    service, audit_session = _setup(tmp_path)
    route = CodexRouteReceipt(
        "provider-fixture:model-fixture",
        "provider-fixture",
        "model-fixture",
        "0.154.0",
        APP_SERVER_PROTOCOL,
        source="live-app-server-capability",
    )
    provider = RecordingAppServerProvider()
    client = AuditCodexClient(
        service,
        inspector=lambda: {"status": "available", "routes": [route.to_dict()]},
        provider=provider,
    )
    try:
        result = client.start(
            audit_session,
            prompt="must not start without one bounded compute unit",
            route_id=route.route_id,
            compute_budget=compute_budget,
            operation_id=f"app-server-insufficient-{compute_budget}",
        )
        assert result.status == "failed"
        assert "reserved_compute" in result.reason
        assert provider.calls == []
        assert audit_session.usage.to_dict() == {"tokens": 0, "compute": 0.0, "issue_writes": 0}
        assert not service._reservations
    finally:
        provider.close()
        service.close()


def test_app_server_resume_requires_an_explicit_per_turn_reservation(
    tmp_path: Path,
) -> None:
    service, audit_session = _setup(tmp_path)
    route = CodexRouteReceipt(
        "provider-fixture:model-fixture",
        "provider-fixture",
        "model-fixture",
        "0.154.0",
        APP_SERVER_PROTOCOL,
        source="live-app-server-capability",
    )
    provider = RecordingAppServerProvider()
    client = AuditCodexClient(
        service,
        inspector=lambda: {"status": "available", "routes": [route.to_dict()]},
        provider=provider,
    )
    try:
        started = client.start(
            audit_session,
            prompt="admit the first turn",
            route_id=route.route_id,
            token_budget=1,
            compute_budget=1.0,
            operation_id="app-server-resume-start",
        )
        assert started.status == "complete"
        assert started.session is not None

        rejected = client.resume(
            started.session,
            token_budget=0,
            compute_budget=1.0,
            operation_id="app-server-resume-without-budget",
        )
        assert rejected.status == "failed"
        assert "reserved_tokens" in rejected.reason
        assert [call["action"] for call in provider.calls] == ["start"]
        assert audit_session.usage.compute == 1.0
        assert not service._reservations
    finally:
        provider.close()
        service.close()


def test_app_server_start_rejects_zero_token_reservation_before_provider_work(
    tmp_path: Path,
) -> None:
    service, audit_session = _setup(tmp_path)
    route = CodexRouteReceipt(
        "provider-fixture:model-fixture",
        "provider-fixture",
        "model-fixture",
        "0.154.0",
        APP_SERVER_PROTOCOL,
        source="live-app-server-capability",
    )
    provider = RecordingAppServerProvider()
    client = AuditCodexClient(
        service,
        inspector=lambda: {"status": "available", "routes": [route.to_dict()]},
        provider=provider,
    )
    try:
        result = client.start(
            audit_session,
            prompt="must not start without a bounded token reservation",
            route_id=route.route_id,
            token_budget=0,
            compute_budget=1.0,
            operation_id="app-server-zero-token-start",
        )
        assert result.status == "failed"
        assert "reserved_tokens" in result.reason
        assert provider.calls == []
        assert audit_session.usage.to_dict() == {"tokens": 0, "compute": 0.0, "issue_writes": 0}
        assert not service._reservations
    finally:
        provider.close()
        service.close()


def test_app_server_start_rejects_policy_without_token_capacity_before_provider_work(
    tmp_path: Path,
) -> None:
    service, audit_session = _setup(tmp_path, token_budget=0)
    route = CodexRouteReceipt(
        "provider-fixture:model-fixture",
        "provider-fixture",
        "model-fixture",
        "0.154.0",
        APP_SERVER_PROTOCOL,
        source="live-app-server-capability",
    )
    provider = RecordingAppServerProvider()
    client = AuditCodexClient(
        service,
        inspector=lambda: {"status": "available", "routes": [route.to_dict()]},
        provider=provider,
    )
    try:
        result = client.start(
            audit_session,
            prompt="policy must deny an unbudgeted live turn",
            route_id=route.route_id,
            token_budget=1,
            compute_budget=1.0,
            operation_id="app-server-token-policy-denial",
        )
        assert result.status == "denied"
        assert provider.calls == []
        assert audit_session.usage.to_dict() == {"tokens": 0, "compute": 0.0, "issue_writes": 0}
        assert not service._reservations
    finally:
        provider.close()
        service.close()


def test_app_server_ambiguous_failure_burns_the_turn_reservation(
    tmp_path: Path,
) -> None:
    service, audit_session = _setup(tmp_path, token_budget=10)
    route = CodexRouteReceipt(
        "provider-fixture:model-fixture",
        "provider-fixture",
        "model-fixture",
        "0.154.0",
        APP_SERVER_PROTOCOL,
        source="live-app-server-capability",
    )
    provider = FailingAppServerProvider()
    client = AuditCodexClient(
        service,
        inspector=lambda: {"status": "available", "routes": [route.to_dict()]},
        provider=provider,
    )
    try:
        result = client.start(
            audit_session,
            prompt="admit one live turn before the transport fails",
            route_id=route.route_id,
            token_budget=7,
            compute_budget=1.0,
            operation_id="app-server-ambiguous-failure",
        )
        assert result.status == "unavailable"
        assert "transport lost" in result.reason
        assert len(provider.calls) == 1
        assert audit_session.usage.to_dict() == {"tokens": 7, "compute": 1.0, "issue_writes": 0}
        assert len(service._reservations) == 1
    finally:
        provider.close()
        service.close()


@pytest.mark.parametrize("action", ["start", "resume"])
def test_durable_codex_admission_does_not_use_legacy_reserve_budget(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    service, audit_session = _setup(tmp_path, token_budget=14, compute_budget=2.0)
    route = CodexRouteReceipt(
        "provider-fixture:model-fixture",
        "provider-fixture",
        "model-fixture",
        "0.154.0",
        APP_SERVER_PROTOCOL,
        source="live-app-server-capability",
    )
    provider = RecordingAppServerProvider()
    provider.start_tokens = 7
    client = AuditCodexClient(
        service,
        inspector=lambda: {"status": "available", "routes": [route.to_dict()]},
        provider=provider,
    )
    try:
        started = None
        if action == "resume":
            started = client.start(
                audit_session,
                prompt="first turn",
                route_id=route.route_id,
                token_budget=7,
                compute_budget=1.0,
                operation_id="lost-capability-prior-start",
            )
            assert started.status == "complete" and started.session is not None

        def legacy_reserve_is_forbidden(*args, **kwargs):
            raise AssertionError("durable Codex admission must not use legacy reserve_budget")

        monkeypatch.setattr(service, "reserve_budget", legacy_reserve_is_forbidden)
        previous_calls = len(provider.calls)
        if action == "start":
            result = client.start(
                audit_session,
                prompt="one durable provider turn",
                route_id=route.route_id,
                token_budget=7,
                compute_budget=1.0,
                operation_id="durable-admission-start",
            )
        else:
            result = client.resume(
                started.session,
                token_budget=7,
                compute_budget=1.0,
                operation_id="durable-admission-resume",
            )
        assert result.status == "complete", result.reason
        assert len(provider.calls) == previous_calls + 1
        assert not service.authority.snapshot()["reservations"]
        assert not service._reservations, result.reason
        expected_usage = (
            {"tokens": 14, "compute": 2.0, "issue_writes": 0}
            if started
            else {
                "tokens": 7,
                "compute": 1.0,
                "issue_writes": 0,
            }
        )
        assert audit_session.usage.to_dict() == expected_usage
    finally:
        provider.close()
        service.close()


def test_app_server_zero_compute_success_burns_reservation(tmp_path: Path) -> None:
    service, audit_session = _setup(tmp_path, token_budget=7, compute_budget=1.0)
    route = CodexRouteReceipt(
        "provider-fixture:model-fixture",
        "provider-fixture",
        "model-fixture",
        "0.154.0",
        APP_SERVER_PROTOCOL,
        source="live-app-server-capability",
    )
    provider = RecordingAppServerProvider()
    provider.start_compute = 0.0
    client = AuditCodexClient(
        service,
        inspector=lambda: {"status": "available", "routes": [route.to_dict()]},
        provider=provider,
    )
    try:
        result = client.start(
            audit_session,
            prompt="zero compute is malformed",
            route_id=route.route_id,
            token_budget=7,
            compute_budget=1.0,
            operation_id="app-server-zero-compute",
        )
        assert result.status == "failed"
        assert audit_session.usage.to_dict() == {"tokens": 7, "compute": 1.0, "issue_writes": 0}
        assert not service._reservations
    finally:
        provider.close()
        service.close()


@pytest.mark.parametrize(
    ("provider_status", "provider_tokens"),
    [("complete", 8), ("failed", 0)],
)
def test_app_server_overrun_or_failed_turn_burns_budget_before_repeated_work(
    tmp_path: Path, provider_status: str, provider_tokens: int
) -> None:
    service, audit_session = _setup(tmp_path, token_budget=7, compute_budget=1.0)
    route = CodexRouteReceipt(
        "provider-fixture:model-fixture",
        "provider-fixture",
        "model-fixture",
        "0.154.0",
        APP_SERVER_PROTOCOL,
        source="live-app-server-capability",
    )
    provider = RecordingAppServerProvider()
    provider.start_status = provider_status
    provider.start_tokens = provider_tokens
    client = AuditCodexClient(
        service,
        inspector=lambda: {"status": "available", "routes": [route.to_dict()]},
        provider=provider,
    )
    try:
        first = client.start(
            audit_session,
            prompt="admit one bounded live turn",
            route_id=route.route_id,
            token_budget=7,
            compute_budget=1.0,
            operation_id=f"app-server-burn-{provider_status}",
        )
        assert first.status == ("denied" if provider_status == "complete" else "failed")
        expected_tokens = 7 if provider_tokens > 7 else 0
        assert audit_session.usage.to_dict() == {
            "tokens": expected_tokens,
            "compute": 1.0,
            "issue_writes": 0,
        }
        assert not service._reservations

        repeated = client.start(
            audit_session,
            prompt="repeat only if another reservation is admitted",
            route_id=route.route_id,
            token_budget=1,
            compute_budget=1.0,
            operation_id=f"app-server-repeat-{provider_status}",
        )
        assert repeated.status == "denied"
        assert len(provider.calls) == 1
    finally:
        provider.close()
        service.close()


def test_context_change_denies_stale_resume_after_codex_start(tmp_path: Path) -> None:
    service, audit_session = _setup(tmp_path)
    route = CodexRouteReceipt(
        "route-fixture", "provider-fixture", "model-fixture", "1.0.0", "app-server"
    )
    client = AuditCodexClient(
        service,
        inspector=lambda: {"status": "available", "routes": [route.to_dict()]},
        provider=FakeCodexProvider(),
    )
    try:
        started = client.start(audit_session, prompt="inspect", operation_id="codex-context-start")
        assert started.session is not None
        updated_context = audit_session.context.next_revision(time_s=2.0)
        assert (
            service.update_context(
                audit_session, updated_context, expected_context_revision=0
            ).status
            == "committed"
        )
        resumed = client.resume(started.session, operation_id="codex-context-resume")
        assert resumed.status == "conflict"
    finally:
        service.close()


def test_provider_metering_ownership_and_operation_replay_fail_closed(tmp_path: Path) -> None:
    service, audit_session = _setup(tmp_path)
    route = CodexRouteReceipt(
        "route-fixture", "provider-fixture", "model-fixture", "1.0.0", "app-server"
    )
    try:
        overrun_provider = FakeCodexProvider(response={"usage": {"tokens": 100, "compute": 100.0}})
        overrun_client = AuditCodexClient(
            service,
            inspector=lambda: {"status": "available", "routes": [route.to_dict()]},
            provider=overrun_provider,
        )
        overrun = overrun_client.start(audit_session, prompt="overrun", operation_id="overrun")
        assert overrun.status == "denied"
        assert audit_session.usage.to_dict() == {"tokens": 0, "compute": 0.0, "issue_writes": 0}

        provider = FakeCodexProvider()
        client = AuditCodexClient(
            service,
            inspector=lambda: {"status": "available", "routes": [route.to_dict()]},
            provider=provider,
        )
        started = client.start(audit_session, prompt="first", operation_id="same-op")
        assert started.status == "complete"
        replay = client.start(audit_session, prompt="different", operation_id="same-op")
        assert replay.status == "conflict"
        same_request = client.start(audit_session, prompt="first", operation_id="same-op")
        assert same_request.status == "complete"
        assert same_request.receipt is not None and same_request.receipt.replayed
        source = Path(service.campaign_source)
        source.write_bytes(source.read_bytes() + b" ")
        stale_replay = client.start(audit_session, prompt="first", operation_id="same-op")
        assert stale_replay.status == "conflict"
        killed = service.kill_switch(audit_session, operation_id="kill-for-codex")
        assert killed.status == "cancelled"
        assert (
            client.resume(started.session, operation_id="resume-after-kill").status == "cancelled"
        )
        forged = replace(started.session, provider_session_id="forged-provider-session")
        assert client.cancel(forged, operation_id="cancel-forged").status == "conflict"
        assert all(
            call.get("provider_session_id") != "forged-provider-session" for call in provider.calls
        )
    finally:
        service.close()


def test_codex_resume_rechecks_source_and_receipt_revision(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    source = root / "campaign.json"
    shutil.copyfile(FIXTURE, source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["source_revision"] = 7
    source.write_text(json.dumps(payload), encoding="utf-8")
    service = AuditService(tmp_path / "store", campaign_source=source, source_root=root)
    audit_session = service.open_session(
        AuditSelectionContext(campaign_id="audit-fixture-campaign"),
        actor="agent",
        actor_id="codex-agent",
        policy=SessionPolicy(allowed_roots=(str(root),), token_budget=5, compute_budget=5.0),
    )
    route = CodexRouteReceipt(
        "route-fixture", "provider-fixture", "model-fixture", "1.0.0", "app-server"
    )
    client = AuditCodexClient(
        service,
        inspector=lambda: {"status": "available", "routes": [route.to_dict()]},
        provider=FakeCodexProvider(),
    )
    try:
        started = client.start(audit_session, prompt="inspect", operation_id="revision-start")
        assert started.receipt is not None and started.receipt.source_revision == 7
        assert started.session is not None and started.session.source_revision == 7
        source.write_bytes(source.read_bytes() + b" ")
        resumed = client.resume(started.session, operation_id="revision-resume")
        assert resumed.status == "conflict"
        cancelled = client.cancel(started.session, operation_id="revision-cancel")
        assert cancelled.status == "conflict"
        assert all(call["action"] != "cancel" for call in client.provider.calls)
    finally:
        service.close()


def test_stale_source_cancel_activates_kill_switch_without_provider_authority(
    tmp_path: Path,
) -> None:
    service, audit_session = _setup(tmp_path)
    route = CodexRouteReceipt(
        "route-fixture", "provider-fixture", "model-fixture", "1.0.0", "app-server"
    )
    provider = FakeCodexProvider()
    client = AuditCodexClient(
        service,
        inspector=lambda: {"status": "available", "routes": [route.to_dict()]},
        provider=provider,
    )
    source = Path(service.campaign_source)
    original = source.read_bytes()
    try:
        started = client.start(audit_session, prompt="inspect", operation_id="stale-cancel-start")
        assert started.status == "complete"
        assert started.session is not None

        source.write_bytes(original + b" stale-source")
        cancelled = client.cancel(
            started.session,
            operation_id="stale-cancel",
            reason="stop despite stale source",
        )
        assert cancelled.status == "conflict"
        assert cancelled.receipt is not None
        assert not cancelled.receipt.replayed
        owned = service.get_session(audit_session)
        assert owned.cancelled and not owned.active
        assert all(call["action"] != "cancel" for call in provider.calls)

        replay = client.cancel(
            cancelled.session,
            operation_id="stale-cancel",
            reason="stop despite stale source",
        )
        assert replay.status == "conflict"
        assert replay.receipt is not None and replay.receipt.replayed
        durable = service.authority.snapshot()["extensions"]["codex"]["operations"]["stale-cancel"]
        assert durable["status"] == "finished"
        assert durable["result_status"] == "conflict"

        source.write_bytes(original)
        resumed = client.resume(cancelled.session, operation_id="resume-after-stale-cancel")
        assert resumed.status == "cancelled"
        assert all(call["action"] != "resume" for call in provider.calls)
    finally:
        service.close()


@pytest.mark.parametrize("action", ("reconnect", "resume", "cancel"))
def test_provider_runtime_error_is_typed_and_settles_reservation(
    tmp_path: Path, action: str
) -> None:
    service, audit_session = _setup(tmp_path)
    route = CodexRouteReceipt(
        "route-fixture", "provider-fixture", "model-fixture", "1.0.0", "app-server"
    )
    provider = RaisingProvider(action)
    client = AuditCodexClient(
        service,
        inspector=lambda: {"status": "available", "routes": [route.to_dict()]},
        provider=provider,
    )
    try:
        started = client.start(audit_session, prompt="inspect", operation_id=f"{action}-start")
        assert started.status == "complete"
        assert started.session is not None
        result = getattr(client, action)(started.session, operation_id=f"{action}-failure")
        assert result.status == "unavailable"
        assert result.receipt is None
        assert "transport exploded" in result.reason
        assert result.session is not None
        assert result.session.status == "active"
        assert len(service._reservations) == 1
        if action == "cancel":
            assert service.get_session(audit_session).cancelled
        else:
            assert not service.get_session(audit_session).cancelled
    finally:
        service.close()


def test_cancel_without_provider_keeps_terminal_receipt_and_kill_state(tmp_path: Path) -> None:
    service, audit_session = _setup(tmp_path)
    route = CodexRouteReceipt(
        "route-fixture", "provider-fixture", "model-fixture", "1.0.0", "app-server"
    )
    provider = FakeCodexProvider()
    client = AuditCodexClient(
        service,
        inspector=lambda: {"status": "available", "routes": [route.to_dict()]},
        provider=provider,
    )
    try:
        started = client.start(audit_session, prompt="inspect", operation_id="no-provider-start")
        assert started.session is not None
        client.provider = None
        result = client.cancel(started.session, operation_id="no-provider-cancel")
        assert result.status == "unavailable"
        assert result.receipt is not None
        assert result.session is not None and result.session.status == "cancelled"
        assert service.get_session(audit_session).cancelled
        assert not service._reservations
    finally:
        service.close()


def test_cancel_kill_operation_is_service_owned_and_collision_resistant(tmp_path: Path) -> None:
    service, audit_session = _setup(tmp_path)
    route = CodexRouteReceipt(
        "route-fixture", "provider-fixture", "model-fixture", "1.0.0", "app-server"
    )
    provider = FakeCodexProvider()
    client = AuditCodexClient(
        service,
        inspector=lambda: {"status": "available", "routes": [route.to_dict()]},
        provider=provider,
    )
    try:
        started = client.start(audit_session, prompt="inspect", operation_id="poison-start")
        assert started.session is not None
        caller_operation_id = "poison-cancel"
        poisoned_kill_id = f"{caller_operation_id}-kill"
        prebound = service.consume_budget(audit_session, operation_id=poisoned_kill_id)
        assert prebound.status == "committed"
        source = Path(service.campaign_source)
        source.write_bytes(source.read_bytes() + b" stale-before-cancel")

        cancelled = client.cancel(
            started.session,
            operation_id=caller_operation_id,
            reason="stop now",
        )
        assert cancelled.status == "conflict"
        assert cancelled.receipt is not None
        assert service.get_session(audit_session).cancelled
        assert all(call["action"] != "cancel" for call in provider.calls)
        replay = client.cancel(
            cancelled.session,
            operation_id=caller_operation_id,
            reason="stop now",
        )
        assert replay.status == "conflict"
        assert replay.receipt is not None and replay.receipt.replayed
    finally:
        service.close()


def test_codex_start_recovers_an_active_budget_reservation_replay(tmp_path: Path) -> None:
    service, audit_session = _setup(tmp_path)
    route = CodexRouteReceipt(
        "route-fixture", "provider-fixture", "model-fixture", "1.0.0", "app-server"
    )
    provider = FakeCodexProvider()
    client = AuditCodexClient(
        service,
        inspector=lambda: {"status": "available", "routes": [route.to_dict()]},
        provider=provider,
    )
    try:
        pre_reserved = service.reserve_budget(
            audit_session,
            tokens=1,
            operation_id="replayed-start-reserve",
        )
        assert pre_reserved.status == "committed"
        started = client.start(
            audit_session,
            prompt="inspect",
            token_budget=1,
            operation_id="replayed-start",
        )
        assert started.status == "complete"
        assert len(provider.calls) == 1
        assert len(service._reservations) == 1
        assert audit_session.usage.tokens == 1
    finally:
        service.close()


def test_codex_route_and_capability_contracts_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        CodexRouteReceipt("", "provider", "model", "1", "app-server")
    with pytest.raises(ValueError, match="discovered"):
        CodexRouteReceipt("route", "provider", "model", "1", "app-server", discovered=False)
    route = CodexRouteReceipt("route", "provider", "model", "1", "app-server")
    with pytest.raises(ValueError, match="unknown fields"):
        CodexRouteReceipt.from_mapping({**route.to_dict(), "unexpected": True})
    with pytest.raises(ValueError, match="schema_version"):
        CodexRouteReceipt.from_mapping(
            {key: value for key, value in route.to_dict().items() if key != "schema_version"}
        )
    with pytest.raises(ValueError, match="available capability"):
        CodexCapabilityInspection("available")
    with pytest.raises(ValueError, match="installed"):
        CodexCapabilityInspection("unavailable", installed=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="capability routes"):
        CodexCapabilityInspection.from_mapping({"status": "available", "routes": {}})
    inspection = CodexCapabilityInspection.from_mapping(
        {"status": "available", "routes": [route.to_dict()], "installed": True}
    )
    assert inspection.choose() == route
    assert inspection.choose("missing") is None
    assert inspect_installed_capabilities(lambda: {"status": "invalid"}).status == "unavailable"
    monkeypatch.setattr(audit_codex_module.shutil, "which", lambda _name: None)
    assert inspect_installed_capabilities().reason == "Codex client/App Server is not installed"


def test_codex_provider_meter_helpers_reject_ambiguous_usage() -> None:
    client_type = AuditCodexClient
    assert client_type._valid_usage(None) is None
    assert client_type._valid_usage({"usage": {"tokens": -1, "compute": 1.0}}) is None
    assert client_type._valid_usage({"usage": {"tokens": 2, "compute": 1, "issue_writes": 0}}) == {
        "tokens": 2,
        "compute": 1.0,
        "issue_writes": 0,
    }
    assert client_type._provider_usage(None) == (None, None, None)
    assert client_type._provider_usage({"usage": {"tokens": 2, "compute": 1.0}}) == (
        2,
        1.0,
        0,
    )
    with pytest.raises(TypeError, match="mapping"):
        client_type._provider_ok([], "complete")  # type: ignore[arg-type]
    assert not client_type._provider_ok({"status": "failed"}, "complete")


def test_codex_app_server_metering_helpers_preserve_or_burn_the_bound(
    tmp_path: Path,
) -> None:
    del tmp_path
    from robot_sf.analysis_workbench.audit_codex_app_server import UNMEASURED_TURN_COMPUTE_CHARGE

    provider = RecordingAppServerProvider()
    valid = {
        "status": "complete",
        "usage": {"tokens": 1, "compute": UNMEASURED_TURN_COMPUTE_CHARGE, "issue_writes": 0},
    }
    assert (
        AuditCodexClient._provider_settlement_result(
            provider,
            valid,
            action="start",
            reserved_tokens=2,
            reserved_compute=1.0,
        )
        == valid
    )
    burned = AuditCodexClient._provider_settlement_result(
        provider,
        {"status": "unknown", "usage": {"tokens": 4, "compute": 0.0, "issue_writes": 0}},
        action="start",
        reserved_tokens=2,
        reserved_compute=1.0,
    )
    assert burned == {"usage": {"tokens": 2, "compute": 1.0, "issue_writes": 0}}

    normalized, status, session = AuditCodexClient._durable_provider_result_for_finish(
        provider,
        action="resume",
        provider_result={"status": "complete", "usage": {"tokens": 0, "compute": 0.0}},
        result_status="complete",
        session=None,
        reserved_tokens=3,
        reserved_compute=1.0,
        reserved_issue_writes=0,
    )
    assert status == "failed" and session is None
    assert normalized["usage"] == {"tokens": 3, "compute": 1.0, "issue_writes": 0}


def test_codex_provider_reservations_are_typed_and_provider_specific() -> None:
    provider = RecordingAppServerProvider()
    reservation = ServiceResult(
        "committed", value={"reserved": {"tokens": 2, "compute": 1.0, "issue_writes": 0}}
    )
    assert AuditCodexClient._provider_reserved_tokens(provider, reservation, action="start") == 2
    assert AuditCodexClient._provider_reserved_compute(provider, reservation, action="start") == 1.0
    assert (
        AuditCodexClient._provider_reserved_tokens(FakeCodexProvider(), reservation, action="start")
        is None
    )
    with pytest.raises(ValueError, match="reserved_tokens"):
        AuditCodexClient._provider_reserved_tokens(
            provider,
            ServiceResult("committed", value={"reserved": {"tokens": 0, "compute": 1.0}}),
            action="start",
        )
    with pytest.raises(ValueError, match="reserved_compute"):
        AuditCodexClient._require_app_server_reservation(
            provider, token_budget=1, compute_budget="bad", action="start"
        )


def test_codex_local_replay_rejects_conflicting_request_before_provider_work(
    tmp_path: Path,
) -> None:
    service, audit_session = _setup(tmp_path)
    route = CodexRouteReceipt(
        "route-fixture", "provider-fixture", "model-fixture", "1", "app-server"
    )
    provider = FakeCodexProvider()
    client = AuditCodexClient(
        service,
        inspector=lambda: {"status": "available", "routes": [route.to_dict()]},
        provider=provider,
    )
    try:
        started = client.start(audit_session, prompt="replay", operation_id="local-replay")
        assert started.receipt is not None and started.session is not None
        conflict = client._replayed_operation(
            "local-replay", "different-request", session=started.session
        )
        assert conflict is not None and conflict.status == "conflict"
        replay = client._replayed_operation(
            "local-replay", started.receipt.request_digest, session=started.session
        )
        assert replay is not None and replay.receipt is not None and replay.receipt.replayed
        assert len(provider.calls) == 1
    finally:
        service.close()
