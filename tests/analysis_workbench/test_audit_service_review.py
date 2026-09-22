"""Authenticated BA-05 full-human review writes through the BA-02 queue."""

from __future__ import annotations

import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from robot_sf.analysis_workbench import audit_service_adapters
from robot_sf.analysis_workbench.audit_queue import AuditQueue, QueueDataset, ScanSummary
from robot_sf.analysis_workbench.audit_scan import scan_campaign
from robot_sf.analysis_workbench.audit_service import (
    AuditSelectionContext,
    AuditService,
    SessionPolicy,
)
from robot_sf.analysis_workbench.audit_service_adapters import AuditSourceBinding, QueueNextAdapter

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "analysis_workbench"
    / "audit_campaign_v1"
    / "campaign.json"
)
CAMPAIGN_ID = "audit-fixture-campaign"


@contextmanager
def _observed_lock(lock: Any, event: threading.Event, *, observe: bool):
    if observe:
        event.set()
    with lock:
        yield


def _service(tmp_path: Path, *, actor: str = "human"):
    report = scan_campaign(FIXTURE)
    service = AuditService(tmp_path / "store", campaign_source=FIXTURE, source_root=FIXTURE.parent)
    session = service.open_session(
        AuditSelectionContext(campaign_id=CAMPAIGN_ID),
        actor=actor,
        actor_id="reviewer-1",
        policy=SessionPolicy(allowed_roots=(str(FIXTURE.parent),)),
    )
    bound = service.update_context(
        session,
        session.context.next_revision(source_identity=session.source_digest),
        expected_context_revision=0,
    )
    assert bound.status == "committed"
    dataset = QueueDataset(
        report.episode_refs,
        scan_summary=ScanSummary.from_scan_report(report),
        campaign_digest=report.audit.campaign_digest,
        source_digest=report.audit.source_digest,
    )
    queue_path = tmp_path / "queue.json"
    adapter = QueueNextAdapter(
        AuditSourceBinding(
            CAMPAIGN_ID,
            report.audit.campaign_digest,
            session.source_digest,
            session.source_revision,
        ),
        lambda: AuditQueue(dataset, state_path=queue_path, store=service.store),
    )
    service.next_adapter = adapter
    service.queue_next_adapter = adapter
    selected = service.next(session, operation_id="select-for-review")
    assert selected.status == "complete", selected.reason
    return service, session, report, dataset, queue_path


def test_human_review_commits_only_selected_scan_bound_episode(tmp_path: Path) -> None:
    service, session, report, dataset, queue_path = _service(tmp_path)
    try:
        state_revision = AuditQueue(dataset, state_path=queue_path).state_revision
        committed = service.record_human_review(
            session,
            outcome="pass",
            expected_context_revision=session.context.context_revision,
            expected_queue_state_revision=state_revision,
            expected_queue_input_revision=dataset.input_revision,
            operation_id="explicit-full-human-review",
        )
        assert committed.status == "committed", committed.reason
        assert committed.value.episode_id == session.context.episode_id
        assert committed.value.scope == "full_episode"
        assert committed.value.author_kind == "human"
        assert committed.value.author_id == "reviewer-1"
        assert committed.value.source_identity == report.audit.source.source_commit
        assert committed.value.scan_identity == report.cache_key
        assert service.store.get(committed.value.review_id) is not None

        replay = service.record_human_review(
            session,
            outcome="pass",
            expected_context_revision=session.context.context_revision,
            expected_queue_state_revision=state_revision,
            expected_queue_input_revision=dataset.input_revision,
            operation_id="explicit-full-human-review",
        )
        assert replay.status == "committed"
        assert replay.operation.operation_id == committed.operation.operation_id
        assert len(service.store.list_records(record_type="review_record")) == 1

        changed_request = service.record_human_review(
            session,
            outcome="fail",
            expected_context_revision=session.context.context_revision,
            expected_queue_state_revision=state_revision,
            expected_queue_input_revision=dataset.input_revision,
            operation_id="explicit-full-human-review",
        )
        assert changed_request.status == "conflict"

        stale = service.record_human_review(
            session,
            outcome="fail",
            expected_context_revision=session.context.context_revision,
            expected_queue_state_revision=state_revision,
            expected_queue_input_revision=dataset.input_revision,
            operation_id="stale-review-attempt",
        )
        assert stale.status == "conflict"
    finally:
        service.close()


def test_agent_session_cannot_grant_full_human_review(tmp_path: Path) -> None:
    service, session, _report, dataset, queue_path = _service(tmp_path, actor="agent")
    try:
        state_revision = AuditQueue(dataset, state_path=queue_path).state_revision
        result = service.record_human_review(
            session,
            outcome="pass",
            expected_context_revision=session.context.context_revision,
            expected_queue_state_revision=state_revision,
            expected_queue_input_revision=dataset.input_revision,
            operation_id="agent-must-not-credit",
        )
        assert result.status == "denied"
        assert not service.store.list_records(record_type="review_record")
    finally:
        service.close()


def test_reloaded_mapping_cannot_grant_new_human_credit(tmp_path: Path) -> None:
    service, session, _report, dataset, queue_path = _service(tmp_path)
    try:
        restored = QueueDataset.from_mapping(dataset.to_dict())
        assert not restored._identity_admitted
        adapter = QueueNextAdapter(
            service.next_adapter.binding,
            lambda: AuditQueue(restored, state_path=queue_path, store=service.store),
        )
        service.next_adapter = adapter
        service.queue_next_adapter = adapter
        state_revision = AuditQueue(restored, state_path=queue_path).state_revision
        result = service.record_human_review(
            session,
            outcome="pass",
            expected_context_revision=session.context.context_revision,
            expected_queue_state_revision=state_revision,
            expected_queue_input_revision=restored.input_revision,
            operation_id="reloaded-mapping-review",
        )
        assert result.status == "unavailable"
        assert not service.store.list_records(record_type="review_record")
    finally:
        service.close()


def test_initial_context_bind_does_not_require_queue_lock(tmp_path: Path) -> None:
    """A configured queue adapter still permits the first source CAS bind."""

    report = scan_campaign(FIXTURE)
    service = AuditService(tmp_path / "store", campaign_source=FIXTURE, source_root=FIXTURE.parent)
    session = service.open_session(
        AuditSelectionContext(campaign_id=CAMPAIGN_ID),
        actor="human",
        actor_id="reviewer-1",
        policy=SessionPolicy(allowed_roots=(str(FIXTURE.parent),)),
    )
    dataset = QueueDataset(
        report.episode_refs,
        scan_summary=ScanSummary.from_scan_report(report),
        campaign_digest=report.audit.campaign_digest,
        source_digest=report.audit.source_digest,
    )
    queue_path = tmp_path / "queue.json"
    adapter = QueueNextAdapter(
        AuditSourceBinding(
            CAMPAIGN_ID,
            report.audit.campaign_digest,
            session.source_digest,
            session.source_revision,
        ),
        lambda: AuditQueue(dataset, state_path=queue_path, store=service.store),
    )
    service.next_adapter = adapter
    service.queue_next_adapter = adapter
    try:
        bound = service.update_context(
            session,
            session.context.next_revision(source_identity=session.source_digest),
            expected_context_revision=0,
            operation_id="initial-context-bind-with-adapter",
        )
        assert bound.status == "committed", bound.reason
    finally:
        service.close()


def test_context_cas_serializes_with_review_queue_commit(tmp_path: Path, monkeypatch: Any) -> None:
    """A context move cannot commit inside review's preflight-to-receipt gap."""

    service, session, _report, dataset, queue_path = _service(tmp_path)
    try:
        state_revision = AuditQueue(dataset, state_path=queue_path).state_revision
        original_review = QueueNextAdapter.record_human_review
        original_next_lock = audit_service_adapters._next_lock
        preflight_passed = threading.Event()
        update_lock_attempted = threading.Event()
        release_review = threading.Event()
        update_finished = threading.Event()
        review_result: dict[str, Any] = {}
        update_result: dict[str, Any] = {}

        def paused_review(self: QueueNextAdapter, **kwargs: Any) -> Any:
            before_review = kwargs["before_review"]

            def before(queue: AuditQueue) -> None:
                before_review(queue)
                preflight_passed.set()
                assert release_review.wait(10), "review preflight did not release"

            request = dict(kwargs)
            request["before_review"] = before
            return original_review(self, **request)

        monkeypatch.setattr(QueueNextAdapter, "record_human_review", paused_review)

        original_episode_id = session.context.episode_id
        original_context_revision = session.context.context_revision

        def review_worker() -> None:
            review_result["value"] = service.record_human_review(
                session,
                outcome="pass",
                expected_context_revision=original_context_revision,
                expected_queue_state_revision=state_revision,
                expected_queue_input_revision=dataset.input_revision,
                operation_id="review-context-lock-race",
            )

        moved_context = session.context.next_revision(episode_id="context-moved-after-review")

        def update_worker() -> None:
            try:
                update_result["value"] = service.update_context(
                    session,
                    moved_context,
                    expected_context_revision=session.context.context_revision,
                    operation_id="review-context-lock-race-update",
                )
            finally:
                update_finished.set()

        def observed_next_lock(path: Path):
            return _observed_lock(
                original_next_lock(path),
                update_lock_attempted,
                observe=threading.current_thread() is update_thread,
            )

        review_thread = threading.Thread(target=review_worker)
        update_thread = threading.Thread(target=update_worker)
        # Observe the inner file-lock boundary after transaction_lock has
        # snapshotted the queue; observing method entry races that snapshot.
        monkeypatch.setattr(audit_service_adapters, "_next_lock", observed_next_lock)
        review_thread.start()
        assert preflight_passed.wait(10), "review preflight did not run"
        update_thread.start()
        assert update_lock_attempted.wait(10), "context update did not request the queue lock"
        assert not update_finished.is_set(), "context CAS crossed the review queue lock"
        release_review.set()
        review_thread.join(timeout=10)
        update_thread.join(timeout=10)
        assert not review_thread.is_alive()
        assert not update_thread.is_alive()

        reviewed = review_result["value"]
        moved = update_result["value"]
        assert reviewed.status == "committed", reviewed.reason
        assert moved.status == "committed", moved.reason
        assert reviewed.value.episode_id == original_episode_id
        assert moved.value.episode_id == "context-moved-after-review"
        assert reviewed.operation.context_revision == original_context_revision
        assert moved.operation.context_revision == original_context_revision + 1
        assert session.context.episode_id == moved.value.episode_id
        assert len(service.store.list_records(record_type="review_record")) == 1
    finally:
        service.close()


def test_cancel_wins_before_review_preflight_when_it_owns_queue_lock(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Cancellation acquired first prevents a later review from granting credit."""

    service, session, _report, dataset, queue_path = _service(tmp_path)
    try:
        state_revision = AuditQueue(dataset, state_path=queue_path).state_revision
        original_lock = QueueNextAdapter.transaction_lock
        lock_acquired = threading.Event()
        release_cancel = threading.Event()
        cancel_result: dict[str, Any] = {}
        review_result: dict[str, Any] = {}

        @contextmanager
        def gated_lock(self: QueueNextAdapter, *, context: AuditSelectionContext):
            with original_lock(self, context=context):
                lock_acquired.set()
                assert release_cancel.wait(10), "cancel lock did not release"
                yield

        monkeypatch.setattr(QueueNextAdapter, "transaction_lock", gated_lock)

        def cancel_worker() -> None:
            cancel_result["value"] = service.kill_switch(
                session,
                reason="cancel-review-race",
                operation_id="cancel-review-lock-race",
            )

        def review_worker() -> None:
            review_result["value"] = service.record_human_review(
                session,
                outcome="pass",
                expected_context_revision=session.context.context_revision,
                expected_queue_state_revision=state_revision,
                expected_queue_input_revision=dataset.input_revision,
                operation_id="cancel-review-lock-race-review",
            )

        cancel_thread = threading.Thread(target=cancel_worker)
        review_thread = threading.Thread(target=review_worker)
        cancel_thread.start()
        assert lock_acquired.wait(10), "cancel did not acquire the queue lock"
        review_thread.start()
        assert not cancel_result and not review_result
        release_cancel.set()
        cancel_thread.join(timeout=10)
        review_thread.join(timeout=10)
        assert not cancel_thread.is_alive()
        assert not review_thread.is_alive()

        assert cancel_result["value"].status == "cancelled"
        assert review_result["value"].status == "cancelled"
        assert service.store.list_records(record_type="review_record") == []
    finally:
        service.close()
