"""BA-05 queue Next integration and two-store crash boundaries."""

from __future__ import annotations

import json
import os
import threading
from typing import TYPE_CHECKING, Any

import pytest

from robot_sf.analysis_workbench import audit_service_adapters
from robot_sf.analysis_workbench.audit_codex import (
    AuditCodexClient,
    CodexRouteReceipt,
    FakeCodexProvider,
)
from robot_sf.analysis_workbench.audit_contracts import EpisodeRef
from robot_sf.analysis_workbench.audit_queue import AuditQueue, QueueDataset
from robot_sf.analysis_workbench.audit_service import (
    AuditSelectionContext,
    AuditService,
    CapabilityUnavailable,
    SessionPolicy,
)
from robot_sf.analysis_workbench.audit_service_adapters import (
    AuditSourceBinding,
    QueueNextAdapter,
    QueueNextResult,
)

if TYPE_CHECKING:
    from pathlib import Path


CAMPAIGN_DIGEST = "a" * 64


def _make_service(
    root: Path,
    *,
    queue_path: Path,
    dataset: QueueDataset,
    policy: SessionPolicy,
) -> tuple[AuditService, Any, QueueNextAdapter]:
    service = AuditService(
        root / "store",
        campaign_source={"campaign_id": "audit-campaign", "episodes": []},
        trusted_policy=policy,
    )
    session = service.open_session(
        AuditSelectionContext(campaign_id="audit-campaign"), policy=policy
    )
    bound = service.update_context(
        session,
        session.context.next_revision(source_identity=session.source_digest),
        expected_context_revision=0,
    )
    assert bound.status == "committed"
    binding = AuditSourceBinding(
        "audit-campaign", CAMPAIGN_DIGEST, session.source_digest, session.source_revision
    )

    def queue_provider() -> AuditQueue:
        return AuditQueue(dataset, state_path=queue_path, store=service.store)

    adapter = QueueNextAdapter(binding, queue_provider)
    service.next_adapter = adapter
    service.queue_next_adapter = adapter
    return service, session, adapter


def _bind_session(service: AuditService, policy: SessionPolicy) -> Any:
    session = service.open_session(
        AuditSelectionContext(campaign_id="audit-campaign"), policy=policy
    )
    bound = service.update_context(
        session,
        session.context.next_revision(source_identity=session.source_digest),
        expected_context_revision=0,
    )
    assert bound.status == "committed"
    return session


def _dataset(source_digest: str, *, count: int = 2) -> QueueDataset:
    candidates = tuple(
        EpisodeRef(
            campaign_digest=CAMPAIGN_DIGEST,
            source_digest=source_digest,
            execution_id=f"execution-{index}",
            planner_id="simple_policy",
            scenario_id="corridor",
            seed=index,
            config_digest="c" * 64,
            environment_digest="d" * 64,
        )
        for index in range(count)
    )
    return QueueDataset(
        candidates=candidates,
        campaign_digest=CAMPAIGN_DIGEST,
        source_digest=source_digest,
    )


def test_next_codex_next_restart_rejects_stale_resume_without_provider_work(
    tmp_path: Path,
) -> None:
    """A real queue selection invalidates a durable Codex turn across restart."""

    policy = SessionPolicy(token_budget=5, compute_budget=5.0)
    queue_path = tmp_path / "queue.json"
    service, audit, _adapter = _make_service(
        tmp_path, queue_path=queue_path, dataset=_dataset("0" * 64), policy=policy
    )
    dataset = _dataset(audit.source_digest)
    binding = AuditSourceBinding(
        "audit-campaign", CAMPAIGN_DIGEST, audit.source_digest, audit.source_revision
    )
    service.next_adapter = QueueNextAdapter(
        binding,
        lambda: AuditQueue(dataset, state_path=queue_path, store=service.store),
    )
    service.queue_next_adapter = service.next_adapter
    first = service.next(audit, operation_id="first-selection")
    assert first.status == "complete", first.reason
    assert audit.context.context_revision == 2

    route = CodexRouteReceipt(
        "durable-route", "fixture-provider", "fixture-model", "1.0", "app-server"
    )

    def inspector() -> dict[str, Any]:
        return {"status": "available", "routes": [route.to_dict()]}

    provider = FakeCodexProvider()
    codex = AuditCodexClient(service, inspector=inspector, provider=provider)
    started = codex.start(audit, prompt="inspect selected case", operation_id="selected-start")
    assert started.status == "complete", started.reason
    assert started.session is not None
    assert len(provider.calls) == 1
    codex_session_id = started.session.session_id

    second = service.next(audit, operation_id="second-selection")
    assert second.status == "complete", second.reason
    assert audit.context.context_revision == 3
    assert AuditQueue(dataset, state_path=queue_path).state_revision == 2
    token = audit.session_token
    audit_session_id = audit.session_id
    service.close()

    reopened = AuditService(
        tmp_path / "store",
        campaign_source={"campaign_id": "audit-campaign", "episodes": []},
        trusted_policy=policy,
    )
    replay_provider = FakeCodexProvider()
    try:
        recovered_audit = reopened.reconnect_session(audit_session_id, token)
        assert recovered_audit.context.context_revision == 3
        replay_codex = AuditCodexClient(reopened, inspector=inspector, provider=replay_provider)
        recovered = replay_codex.recover_session(codex_session_id, audit_token=token)
        assert recovered.status == "complete", recovered.reason
        assert recovered.session is not None
        assert recovered.session.context.context_revision == 2
        stale = replay_codex.resume(
            recovered.session,
            operation_id="stale-selected-resume",
            audit_token=token,
        )
        assert stale.status == "conflict"
        assert replay_provider.calls == []
    finally:
        reopened.close()


def test_next_replay_and_restart_return_the_same_packet(tmp_path: Path) -> None:
    policy = SessionPolicy()
    queue_path = tmp_path / "queue.json"
    source_service, session, _adapter = _make_service(
        tmp_path / "first", queue_path=queue_path, dataset=_dataset("0" * 64), policy=policy
    )
    dataset = _dataset(session.source_digest)
    # Replace the fixture adapter with the source-bound queue input after source admission.
    source_service.next_adapter = QueueNextAdapter(
        AuditSourceBinding(
            "audit-campaign", CAMPAIGN_DIGEST, session.source_digest, session.source_revision
        ),
        lambda: AuditQueue(dataset, state_path=queue_path, store=source_service.store),
    )
    source_service.queue_next_adapter = source_service.next_adapter
    first = source_service.next(session, operation_id="next-replay")
    assert first.status == "complete", first.reason
    assert isinstance(first.value, QueueNextResult)
    packet_id = first.value.packet.packet_id
    context = first.context
    assert context is not None and context.reference_id == packet_id
    session_id = session.session_id
    session_token = session.session_token
    source_digest = session.source_digest
    source_revision = session.source_revision
    source_service.close()

    restarted = AuditService(
        tmp_path / "first" / "store",
        campaign_source={"campaign_id": "audit-campaign", "episodes": []},
        trusted_policy=policy,
    )
    restarted.next_adapter = QueueNextAdapter(
        AuditSourceBinding("audit-campaign", CAMPAIGN_DIGEST, source_digest, source_revision),
        lambda: AuditQueue(dataset, state_path=queue_path, store=restarted.store),
    )
    restarted.queue_next_adapter = restarted.next_adapter
    recovered = restarted.reconnect_session(session_id, session_token)
    replay = restarted.next(recovered, operation_id="next-replay")
    assert replay.status == "complete", replay.reason
    assert replay.operation is not None and replay.operation.replayed
    assert isinstance(replay.value, QueueNextResult)
    assert replay.value.packet.packet_id == packet_id
    assert AuditQueue(dataset, state_path=queue_path).state_revision == 1
    restarted.close()


def test_next_crash_after_queue_commit_fails_closed_on_restart(tmp_path: Path) -> None:
    policy = SessionPolicy()
    queue_path = tmp_path / "queue.json"
    service, session, adapter = _make_service(
        tmp_path, queue_path=queue_path, dataset=_dataset("0" * 64), policy=policy
    )
    dataset = _dataset(session.source_digest)
    adapter = QueueNextAdapter(
        AuditSourceBinding(
            "audit-campaign", CAMPAIGN_DIGEST, session.source_digest, session.source_revision
        ),
        lambda: AuditQueue(dataset, state_path=queue_path, store=service.store),
    )
    service.next_adapter = adapter
    service.queue_next_adapter = adapter
    original_mutate = service._mutate_session_authority

    def crash_after_queue_commit(*args: Any, **kwargs: Any) -> Any:
        if str(kwargs.get("transaction_id", "")).startswith("authority.queue.next.context:"):
            raise KeyboardInterrupt("simulated process loss after queue commit")
        return original_mutate(*args, **kwargs)

    service._mutate_session_authority = crash_after_queue_commit  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt, match="queue commit"):
        service.next(session, operation_id="next-crash")
    assert AuditQueue(dataset, state_path=queue_path).state_revision == 1
    service.close()

    restarted = AuditService(
        tmp_path / "store",
        campaign_source={"campaign_id": "audit-campaign", "episodes": []},
        trusted_policy=policy,
    )
    restarted.next_adapter = QueueNextAdapter(
        AuditSourceBinding("audit-campaign", CAMPAIGN_DIGEST, session.source_digest, 0),
        lambda: AuditQueue(dataset, state_path=queue_path, store=restarted.store),
    )
    restarted.queue_next_adapter = restarted.next_adapter
    recovered = restarted.reconnect_session(session.session_id, session.session_token)
    retry = restarted.next(recovered, operation_id="next-crash")
    assert retry.status == "conflict"
    assert "in progress" in retry.reason
    assert AuditQueue(dataset, state_path=queue_path).state_revision == 1
    restarted.close()


def test_next_finalizes_after_outer_receipt_on_exact_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = SessionPolicy()
    queue_path = tmp_path / "queue.json"
    service, session, _adapter = _make_service(
        tmp_path, queue_path=queue_path, dataset=_dataset("0" * 64), policy=policy
    )
    dataset = _dataset(session.source_digest)
    adapter = QueueNextAdapter(
        AuditSourceBinding("audit-campaign", CAMPAIGN_DIGEST, session.source_digest, 0),
        lambda: AuditQueue(dataset, state_path=queue_path, store=service.store),
    )
    service.next_adapter = adapter
    service.queue_next_adapter = adapter
    original_replay = QueueNextAdapter.replay
    crashed = True

    def crash_finalization(self: QueueNextAdapter, **kwargs: Any) -> QueueNextResult:
        nonlocal crashed
        if crashed and kwargs.get("operation_id") == "next-finalize-retry":
            crashed = False
            raise KeyboardInterrupt("simulated process loss after terminal receipt")
        return original_replay(self, **kwargs)

    monkeypatch.setattr(QueueNextAdapter, "replay", crash_finalization)
    with pytest.raises(KeyboardInterrupt, match="terminal receipt"):
        service.next(session, operation_id="next-finalize-retry")

    coordination_path = queue_path.with_name("queue.json.service-next.json")
    coordination = json.loads(coordination_path.read_text(encoding="utf-8"))
    assert coordination["active"]["operation_id"] == "next-finalize-retry"
    assert coordination["active"]["phase"] == "queue_committed"
    assert service.authority.snapshot()["operations"]["next-finalize-retry"]["status"] == "finished"

    replay = service.next(session, operation_id="next-finalize-retry")
    assert replay.status == "complete", replay.reason
    assert replay.operation is not None and replay.operation.replayed
    assert isinstance(replay.value, QueueNextResult)
    coordination = json.loads(coordination_path.read_text(encoding="utf-8"))
    assert coordination["active"] is None
    assert "next-finalize-retry" in coordination["completed"]
    assert AuditQueue(dataset, state_path=queue_path).state_revision == 1
    service.close()


def test_next_finalizes_explicit_original_context_and_competing_next_progresses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = SessionPolicy()
    queue_path = tmp_path / "queue.json"
    service, session, _adapter = _make_service(
        tmp_path, queue_path=queue_path, dataset=_dataset("0" * 64), policy=policy
    )
    dataset = _dataset(session.source_digest)
    adapter = QueueNextAdapter(
        AuditSourceBinding("audit-campaign", CAMPAIGN_DIGEST, session.source_digest, 0),
        lambda: AuditQueue(dataset, state_path=queue_path, store=service.store),
    )
    service.next_adapter = adapter
    service.queue_next_adapter = adapter
    original_context = session.context
    original_replay = QueueNextAdapter.replay
    crashed = True

    def crash_finalization(self: QueueNextAdapter, **kwargs: Any) -> QueueNextResult:
        nonlocal crashed
        if crashed and kwargs.get("operation_id") == "next-explicit-finalize-retry":
            crashed = False
            raise KeyboardInterrupt("simulated process loss after terminal receipt")
        return original_replay(self, **kwargs)

    monkeypatch.setattr(QueueNextAdapter, "replay", crash_finalization)
    with pytest.raises(KeyboardInterrupt, match="terminal receipt"):
        service.next(
            session,
            context=original_context,
            operation_id="next-explicit-finalize-retry",
        )

    replay = service.next(
        session,
        context=original_context,
        operation_id="next-explicit-finalize-retry",
    )
    assert replay.status == "complete", replay.reason
    assert replay.operation is not None and replay.operation.replayed
    assert isinstance(replay.value, QueueNextResult)

    followup = service.next(session, operation_id="next-after-explicit-finalize")
    assert followup.status == "complete", followup.reason
    assert isinstance(followup.value, QueueNextResult)
    assert followup.value.packet.packet_id != replay.value.packet.packet_id
    assert AuditQueue(dataset, state_path=queue_path).state_revision == 2
    service.close()


def test_next_context_race_before_transact_returns_is_historical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = SessionPolicy()
    queue_path = tmp_path / "queue.json"
    service, session, _adapter = _make_service(
        tmp_path, queue_path=queue_path, dataset=_dataset("0" * 64), policy=policy
    )
    dataset = _dataset(session.source_digest)
    adapter = QueueNextAdapter(
        AuditSourceBinding("audit-campaign", CAMPAIGN_DIGEST, session.source_digest, 0),
        lambda: AuditQueue(dataset, state_path=queue_path, store=service.store),
    )
    service.next_adapter = adapter
    service.queue_next_adapter = adapter
    original_transact = QueueNextAdapter.transact
    raced = False

    def transact_then_advance(self: QueueNextAdapter, **kwargs: Any) -> Any:
        nonlocal raced
        after_select = kwargs["after_select"]

        def callback(selected: Any) -> Any:
            nonlocal raced
            result = after_select(selected)
            if not raced and kwargs.get("operation_id") == "next-transact-race":
                raced = True
                changed = session.context.next_revision(episode_id="unrelated-context")
                moved = service.update_context(
                    session,
                    changed,
                    expected_context_revision=session.context.context_revision,
                    operation_id="next-transact-race-update",
                )
                assert moved.status == "committed", moved.reason
            return result

        request = dict(kwargs)
        request["after_select"] = callback
        return original_transact(self, **request)

    monkeypatch.setattr(QueueNextAdapter, "transact", transact_then_advance)
    first = service.next(session, operation_id="next-transact-race")
    assert first.status == "unavailable"
    assert first.value is None
    assert first.operation is not None and first.operation.status == "complete"
    assert first.operation.context_revision == 2
    assert session.context.context_revision == 3
    assert "historical" in first.reason

    retry = service.next(session, operation_id="next-transact-race")
    assert retry.status == "unavailable"
    assert retry.value is None
    assert retry.operation is not None and retry.operation.replayed
    assert "historical" in retry.reason

    coordination_path = queue_path.with_name("queue.json.service-next.json")
    coordination = json.loads(coordination_path.read_text(encoding="utf-8"))
    assert coordination["active"] is None
    assert AuditQueue(dataset, state_path=queue_path).state_revision == 1
    service.close()


def test_kill_switch_from_next_callback_does_not_deadlock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A same-thread cancellation callback can re-enter the durable queue lock."""

    policy = SessionPolicy()
    queue_path = tmp_path / "queue.json"
    service, session, _adapter = _make_service(
        tmp_path, queue_path=queue_path, dataset=_dataset("0" * 64), policy=policy
    )
    dataset = _dataset(session.source_digest)
    adapter = QueueNextAdapter(
        AuditSourceBinding("audit-campaign", CAMPAIGN_DIGEST, session.source_digest, 0),
        lambda: AuditQueue(dataset, state_path=queue_path, store=service.store),
    )
    service.next_adapter = adapter
    service.queue_next_adapter = adapter
    original_transact = QueueNextAdapter.transact
    callback_result: dict[str, Any] = {}
    worker_result: dict[str, Any] = {}

    def transact_then_cancel(self: QueueNextAdapter, **kwargs: Any) -> Any:
        after_select = kwargs["after_select"]

        def callback(selected: Any) -> Any:
            result = after_select(selected)
            callback_result["value"] = service.kill_switch(
                session,
                reason="cancel from next callback",
                operation_id="next-callback-cancel",
            )
            return result

        request = dict(kwargs)
        request["after_select"] = callback
        return original_transact(self, **request)

    monkeypatch.setattr(QueueNextAdapter, "transact", transact_then_cancel)

    def worker() -> None:
        worker_result["value"] = service.next(session, operation_id="next-callback-cancel-race")

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout=10)
    try:
        assert not thread.is_alive(), "Next callback re-entry deadlocked on the queue lock"
        assert callback_result["value"].status == "cancelled"
        assert worker_result["value"].status == "complete"
    finally:
        service.close()


def test_next_lock_setup_failure_allows_same_thread_retry_with_durable_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A setup failure must not let a same-thread retry bypass the file lock."""

    assert audit_service_adapters.fcntl is not None
    queue_path = tmp_path / "queue.json"
    original_open = audit_service_adapters.os.open
    open_failed = False

    def fail_open_once(*args: Any, **kwargs: Any) -> int:
        nonlocal open_failed
        raw_path = args[0] if args else kwargs.get("path")
        if not open_failed and raw_path is not None:
            if os.fspath(raw_path).endswith(".service-next.lock"):
                open_failed = True
                raise OSError("injected durable lock setup failure")
        return original_open(*args, **kwargs)

    monkeypatch.setattr(audit_service_adapters.os, "open", fail_open_once)
    flock_calls: list[int] = []
    original_flock = audit_service_adapters.fcntl.flock

    def record_flock(file_descriptor: int, operation: int) -> Any:
        flock_calls.append(operation)
        return original_flock(file_descriptor, operation)

    monkeypatch.setattr(audit_service_adapters.fcntl, "flock", record_flock)
    results: dict[str, Any] = {}
    finished = threading.Event()

    def retry_on_same_thread() -> None:
        try:
            with audit_service_adapters._next_lock(queue_path):
                results["first_entered"] = True
        except CapabilityUnavailable:
            results["first_failed"] = True
        except BaseException as exc:  # pragma: no cover - assertion reports the failure.
            results["first_error"] = exc
        try:
            with audit_service_adapters._next_lock(queue_path):
                results["retry_entered"] = True
        except BaseException as exc:  # pragma: no cover - assertion reports the failure.
            results["retry_error"] = exc
        finally:
            finished.set()

    thread = threading.Thread(target=retry_on_same_thread, daemon=True)
    thread.start()
    assert finished.wait(timeout=2), "same-thread retry did not complete after setup failure"
    thread.join(timeout=1)
    assert not thread.is_alive()
    assert results.get("first_failed") is True
    assert results.get("first_entered") is None
    assert results.get("retry_entered") is True
    assert "retry_error" not in results
    assert audit_service_adapters.fcntl.LOCK_EX in flock_calls
    assert audit_service_adapters.fcntl.LOCK_UN in flock_calls


def test_next_lock_setup_failure_releases_state_for_other_thread_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed owner must release the per-path mutex for another thread."""

    assert audit_service_adapters.fcntl is not None
    queue_path = tmp_path / "queue.json"
    original_open = audit_service_adapters.os.open
    open_failed = False

    def fail_open_once(*args: Any, **kwargs: Any) -> int:
        nonlocal open_failed
        raw_path = args[0] if args else kwargs.get("path")
        if not open_failed and raw_path is not None:
            if os.fspath(raw_path).endswith(".service-next.lock"):
                open_failed = True
                raise OSError("injected durable lock setup failure")
        return original_open(*args, **kwargs)

    monkeypatch.setattr(audit_service_adapters.os, "open", fail_open_once)
    first_finished = threading.Event()
    retry_entered = threading.Event()
    results: dict[str, Any] = {}

    def failing_owner() -> None:
        try:
            with audit_service_adapters._next_lock(queue_path):
                results["first_entered"] = True
        except CapabilityUnavailable:
            results["first_failed"] = True
        except BaseException as exc:  # pragma: no cover - assertion reports the failure.
            results["first_error"] = exc
        finally:
            first_finished.set()

    first = threading.Thread(target=failing_owner, daemon=True)
    first.start()
    assert first_finished.wait(timeout=2), "injected setup failure did not complete"
    first.join(timeout=1)
    assert not first.is_alive()

    def retry_on_other_thread() -> None:
        try:
            with audit_service_adapters._next_lock(queue_path):
                results["retry_entered"] = True
                retry_entered.set()
        except BaseException as exc:  # pragma: no cover - assertion reports the failure.
            results["retry_error"] = exc

    retry = threading.Thread(target=retry_on_other_thread, daemon=True)
    retry.start()
    assert retry_entered.wait(timeout=2), "other-thread retry remained blocked"
    retry.join(timeout=1)
    assert not retry.is_alive()
    assert results.get("first_failed") is True
    assert results.get("first_entered") is None
    assert results.get("retry_entered") is True
    assert "retry_error" not in results


def test_next_lock_fdopen_failure_closes_descriptor_and_resets_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed descriptor wrapper must close its raw descriptor before retry."""

    queue_path = tmp_path / "queue.json"
    original_fdopen = audit_service_adapters.os.fdopen
    original_close = audit_service_adapters.os.close
    failed_descriptor: int | None = None
    close_calls: list[int] = []

    def fail_fdopen_once(file_descriptor: int, *args: Any, **kwargs: Any) -> Any:
        nonlocal failed_descriptor
        if failed_descriptor is None:
            failed_descriptor = file_descriptor
            raise OSError("injected descriptor wrapping failure")
        return original_fdopen(file_descriptor, *args, **kwargs)

    def record_close(file_descriptor: int) -> None:
        close_calls.append(file_descriptor)
        original_close(file_descriptor)

    monkeypatch.setattr(audit_service_adapters.os, "fdopen", fail_fdopen_once)
    monkeypatch.setattr(audit_service_adapters.os, "close", record_close)
    with pytest.raises(CapabilityUnavailable, match="lock is unavailable"):
        with audit_service_adapters._next_lock(queue_path):
            pass

    lock_key = os.path.abspath(os.fspath(queue_path.with_name("queue.json.service-next.lock")))
    lock_state = audit_service_adapters._NEXT_LOCK_STATES[lock_key]
    assert lock_state.owner is None
    assert lock_state.depth == 0
    assert failed_descriptor is not None
    assert failed_descriptor in close_calls

    with audit_service_adapters._next_lock(queue_path):
        pass


def test_next_context_race_finalizes_historical_result_and_context_none_retry(
    tmp_path: Path,
) -> None:
    policy = SessionPolicy()
    queue_path = tmp_path / "queue.json"
    service, session, _adapter = _make_service(
        tmp_path, queue_path=queue_path, dataset=_dataset("0" * 64), policy=policy
    )
    dataset = _dataset(session.source_digest)
    adapter = QueueNextAdapter(
        AuditSourceBinding("audit-campaign", CAMPAIGN_DIGEST, session.source_digest, 0),
        lambda: AuditQueue(dataset, state_path=queue_path, store=service.store),
    )
    service.next_adapter = adapter
    service.queue_next_adapter = adapter
    original_record = service._record_action_result
    raced = False

    def record_then_advance_context(*args: Any, **kwargs: Any) -> Any:
        nonlocal raced
        receipt = original_record(*args, **kwargs)
        if not raced and args[1] == "next-context-race":
            raced = True
            changed = session.context.next_revision(episode_id="unrelated-context")
            moved = service.update_context(
                session,
                changed,
                expected_context_revision=session.context.context_revision,
                operation_id="next-context-race-update",
            )
            assert moved.status == "committed", moved.reason
        return receipt

    service._record_action_result = record_then_advance_context  # type: ignore[method-assign]
    first = service.next(session, operation_id="next-context-race")
    assert first.status == "unavailable"
    assert first.value is None
    assert first.operation is not None and first.operation.status == "complete"
    assert "historical" in first.reason

    coordination_path = queue_path.with_name("queue.json.service-next.json")
    coordination = json.loads(coordination_path.read_text(encoding="utf-8"))
    assert coordination["active"] is None
    assert "next-context-race" in coordination["completed"]
    assert AuditQueue(dataset, state_path=queue_path).state_revision == 1

    retry = service.next(session, operation_id="next-context-race")
    assert retry.status == "unavailable"
    assert retry.value is None
    assert retry.operation is not None and retry.operation.replayed
    assert "historical" in retry.reason
    assert AuditQueue(dataset, state_path=queue_path).state_revision == 1
    service.close()


def test_outer_receipt_crash_keeps_owner_lease_and_terminalizes_blocked_session(
    tmp_path: Path,
) -> None:
    policy = SessionPolicy()
    queue_path = tmp_path / "queue.json"
    first, session, _adapter = _make_service(
        tmp_path / "first", queue_path=queue_path, dataset=_dataset("0" * 64), policy=policy
    )
    dataset = _dataset(session.source_digest)
    first.next_adapter = QueueNextAdapter(
        AuditSourceBinding("audit-campaign", CAMPAIGN_DIGEST, session.source_digest, 0),
        lambda: AuditQueue(dataset, state_path=queue_path, store=first.store),
    )
    first.queue_next_adapter = first.next_adapter

    def crash_outer_receipt(*_args: Any, **_kwargs: Any) -> Any:
        raise KeyboardInterrupt("simulated process loss while recording terminal receipt")

    first._record_action_result = crash_outer_receipt  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt, match="terminal receipt"):
        first.next(session, operation_id="next-owner-crash")
    first.close()

    coordination_path = queue_path.with_name("queue.json.service-next.json")
    coordination = json.loads(coordination_path.read_text(encoding="utf-8"))
    assert coordination["active"]["operation_id"] == "next-owner-crash"
    assert coordination["active"]["phase"] == "queue_committed"

    second = AuditService(
        tmp_path / "first" / "store",
        campaign_source={"campaign_id": "audit-campaign", "episodes": []},
        trusted_policy=policy,
    )
    other = _bind_session(second, policy)
    second.next_adapter = QueueNextAdapter(
        AuditSourceBinding("audit-campaign", CAMPAIGN_DIGEST, other.source_digest, 0),
        lambda: AuditQueue(dataset, state_path=queue_path, store=second.store),
    )
    second.queue_next_adapter = second.next_adapter
    blocked = second.next(other, operation_id="next-blocked-by-owner")
    assert blocked.status == "conflict", blocked.reason
    blocked_operation = second.authority.snapshot()["operations"]["next-blocked-by-owner"]
    assert blocked_operation["status"] == "finished"
    assert blocked_operation["result_status"] == "conflict"

    coordination = json.loads(coordination_path.read_text(encoding="utf-8"))
    assert coordination["active"]["operation_id"] == "next-owner-crash"
    assert "next-blocked-by-owner" not in coordination["completed"]
    assert AuditQueue(dataset, state_path=queue_path).state_revision == 1
    second.close()


def test_unresolved_crash_blocks_a_second_authenticated_session(tmp_path: Path) -> None:
    policy = SessionPolicy()
    queue_path = tmp_path / "queue.json"
    first, session, _adapter = _make_service(
        tmp_path / "first", queue_path=queue_path, dataset=_dataset("0" * 64), policy=policy
    )
    dataset = _dataset(session.source_digest)
    first.next_adapter = QueueNextAdapter(
        AuditSourceBinding("audit-campaign", CAMPAIGN_DIGEST, session.source_digest, 0),
        lambda: AuditQueue(dataset, state_path=queue_path, store=first.store),
    )
    first.queue_next_adapter = first.next_adapter
    original_mutate = first._mutate_session_authority

    def crash_after_queue_commit(*args: Any, **kwargs: Any) -> Any:
        if str(kwargs.get("transaction_id", "")).startswith("authority.queue.next.context:"):
            raise KeyboardInterrupt("simulated process loss after queue commit")
        return original_mutate(*args, **kwargs)

    first._mutate_session_authority = crash_after_queue_commit  # type: ignore[method-assign]
    with pytest.raises(KeyboardInterrupt):
        first.next(session, operation_id="next-cross-session-crash")
    first.close()

    second = AuditService(
        tmp_path / "first" / "store",
        campaign_source={"campaign_id": "audit-campaign", "episodes": []},
        trusted_policy=policy,
    )
    other = _bind_session(second, policy)
    second.next_adapter = QueueNextAdapter(
        AuditSourceBinding("audit-campaign", CAMPAIGN_DIGEST, other.source_digest, 0),
        lambda: AuditQueue(dataset, state_path=queue_path, store=second.store),
    )
    second.queue_next_adapter = second.next_adapter
    blocked = second.next(other, operation_id="next-cross-session-followup")
    assert blocked.status in {"conflict", "unavailable"}
    assert AuditQueue(dataset, state_path=queue_path).state_revision == 1
    second.close()


def test_authority_race_after_queue_commit_stays_unresolved(tmp_path: Path) -> None:
    policy = SessionPolicy()
    queue_path = tmp_path / "queue.json"
    first, session, _adapter = _make_service(
        tmp_path / "first", queue_path=queue_path, dataset=_dataset("0" * 64), policy=policy
    )
    dataset = _dataset(session.source_digest)
    second = AuditService(
        tmp_path / "first" / "store",
        campaign_source={"campaign_id": "audit-campaign", "episodes": []},
        trusted_policy=policy,
    )
    other = second.reconnect_session(session.session_id, session.session_token)
    first.next_adapter = QueueNextAdapter(
        AuditSourceBinding("audit-campaign", CAMPAIGN_DIGEST, session.source_digest, 0),
        lambda: AuditQueue(dataset, state_path=queue_path, store=first.store),
    )
    first.queue_next_adapter = first.next_adapter
    original_refresh = first._refresh_session

    def refresh_then_race(*args: Any, **kwargs: Any) -> Any:
        current = original_refresh(*args, **kwargs)
        changed = current.context.next_revision(source_identity=current.source_digest)
        raced = second.update_context(
            other,
            changed,
            expected_context_revision=current.context.context_revision,
            operation_id="context-race-after-next-preflight",
        )
        assert raced.status == "committed"
        return current

    first._refresh_session = refresh_then_race  # type: ignore[method-assign]
    result = first.next(session, operation_id="next-authority-race")
    assert result.status == "unavailable"
    assert "queue commit" in result.reason or "reconcile" in result.reason
    assert AuditQueue(dataset, state_path=queue_path).state_revision == 1
    assert first.authority.snapshot()["operations"]["next-authority-race"]["status"] == "inflight"
    first.close()
    second.close()


def test_replay_uses_the_exact_historical_selection_envelope(tmp_path: Path) -> None:
    policy = SessionPolicy()
    queue_path = tmp_path / "queue.json"
    service, session, _adapter = _make_service(
        tmp_path, queue_path=queue_path, dataset=_dataset("0" * 64, count=1), policy=policy
    )
    dataset = _dataset(session.source_digest, count=1)
    service.next_adapter = QueueNextAdapter(
        AuditSourceBinding("audit-campaign", CAMPAIGN_DIGEST, session.source_digest, 0),
        lambda: AuditQueue(dataset, state_path=queue_path, store=service.store),
    )
    service.queue_next_adapter = service.next_adapter
    first = service.next(session, operation_id="next-exact-replay")
    assert first.status == "complete"
    assert isinstance(first.value, QueueNextResult)
    first_selection = first.value.context.selection_id
    first_explanation = first.value.selection_result.explanation
    AuditQueue(dataset, state_path=queue_path).select_next(force_current=True)

    replay = service.next(session, operation_id="next-exact-replay")
    assert replay.status == "complete"
    assert isinstance(replay.value, QueueNextResult)
    assert replay.value.context.selection_id == first_selection
    assert replay.value.selection_result.explanation == first_explanation
    service.close()


def test_two_services_serialize_next_and_stale_context_cas(tmp_path: Path) -> None:
    policy = SessionPolicy()
    queue_path = tmp_path / "queue.json"
    first, session, _adapter = _make_service(
        tmp_path / "first", queue_path=queue_path, dataset=_dataset("0" * 64), policy=policy
    )
    dataset = _dataset(session.source_digest)
    first.next_adapter = QueueNextAdapter(
        AuditSourceBinding("audit-campaign", CAMPAIGN_DIGEST, session.source_digest, 0),
        lambda: AuditQueue(dataset, state_path=queue_path, store=first.store),
    )
    first.queue_next_adapter = first.next_adapter
    second = AuditService(
        tmp_path / "first" / "store",
        campaign_source={"campaign_id": "audit-campaign", "episodes": []},
        trusted_policy=policy,
    )
    other = second.reconnect_session(session.session_id, session.session_token)
    second.next_adapter = QueueNextAdapter(
        AuditSourceBinding("audit-campaign", CAMPAIGN_DIGEST, other.source_digest, 0),
        lambda: AuditQueue(dataset, state_path=queue_path, store=second.store),
    )
    second.queue_next_adapter = second.next_adapter
    results: list[Any] = []
    barrier = threading.Barrier(2)

    def run(service: AuditService, handle: Any, operation_id: str) -> None:
        barrier.wait(timeout=5)
        results.append(service.next(handle, operation_id=operation_id))

    threads = (
        threading.Thread(target=run, args=(first, session, "next-race-a")),
        threading.Thread(target=run, args=(second, other, "next-race-b")),
    )
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
    assert len(results) == 2
    assert sorted(result.status for result in results) == ["complete", "conflict"]
    assert AuditQueue(dataset, state_path=queue_path).state_revision == 1
    first.close()
    second.close()


def test_next_rejects_stale_queue_source_and_operation_collision(tmp_path: Path) -> None:
    policy = SessionPolicy()
    queue_path = tmp_path / "queue.json"
    service, session, _adapter = _make_service(
        tmp_path, queue_path=queue_path, dataset=_dataset("0" * 64), policy=policy
    )
    stale_revision = service.next(
        session, expected_queue_state_revision=1, operation_id="next-stale-revision"
    )
    assert stale_revision.status == "conflict"
    stale_dataset = _dataset("e" * 64)
    service.next_adapter = QueueNextAdapter(
        AuditSourceBinding("audit-campaign", CAMPAIGN_DIGEST, session.source_digest, 0),
        lambda: AuditQueue(stale_dataset, state_path=queue_path),
    )
    service.queue_next_adapter = service.next_adapter
    stale = service.next(session, operation_id="next-stale-source")
    assert stale.status == "conflict"
    assert service.authority.snapshot()["operations"]["next-stale-source"]["status"] == "finished"
    assert service.store.get("audit-operation-next-stale-source-after") is not None
    assert AuditQueue(stale_dataset, state_path=queue_path).state_revision == 0

    dataset = _dataset(session.source_digest)
    service.next_adapter = QueueNextAdapter(
        AuditSourceBinding("audit-campaign", CAMPAIGN_DIGEST, session.source_digest, 0),
        lambda: AuditQueue(dataset, state_path=queue_path),
    )
    service.queue_next_adapter = service.next_adapter
    requested_context = session.context
    first = service.next(session, context=requested_context, operation_id="next-collision")
    assert first.status == "complete"
    changed_context = service.next(session, context=session.context, operation_id="next-collision")
    assert changed_context.status == "conflict"
    collision = service.next(session, operation_id="next-collision", force_current=True)
    assert collision.status == "conflict"
    assert AuditQueue(dataset, state_path=queue_path).state_revision == 1
    service.close()
