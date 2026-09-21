"""BA-05 reads use BA-02/BA-04 owners under one admitted source revision."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

import robot_sf.analysis_workbench.audit_service_adapters as adapters_module
from robot_sf.analysis_workbench.audit_contracts import EpisodeRef, ReviewRecord
from robot_sf.analysis_workbench.audit_coverage import (
    STATUS_INCOMPLETE,
    AuditHealthReport,
    AuditIdentity,
    default_audit_protocol,
    evaluate_coverage,
)
from robot_sf.analysis_workbench.audit_queue import AuditQueue, QueueDataset, SelectionResult
from robot_sf.analysis_workbench.audit_scan import scan_campaign
from robot_sf.analysis_workbench.audit_service import (
    AuditContextConflict,
    AuditNextAmbiguous,
    AuditNextBlocked,
    AuditSelectionContext,
    AuditService,
    CapabilityUnavailable,
    SessionPolicy,
)
from robot_sf.analysis_workbench.audit_service_adapters import (
    AuditSourceBinding,
    CoverageReadAdapter,
    QueueNextAdapter,
    QueueReadAdapter,
    coverage_reviews_for_scan,
)

CAMPAIGN = "a" * 64
SOURCE = "b" * 64
FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "analysis_workbench"
    / "audit_campaign_v1"
    / "campaign.json"
)


def _binding() -> AuditSourceBinding:
    return AuditSourceBinding("audit-campaign", CAMPAIGN, SOURCE, 3)


def _context() -> AuditSelectionContext:
    return AuditSelectionContext(
        campaign_id="audit-campaign", source_identity=SOURCE, source_revision=3
    )


def _queue(*, source_digest: str = SOURCE) -> AuditQueue:
    episode = EpisodeRef(
        campaign_digest=CAMPAIGN,
        source_digest=source_digest,
        execution_id="execution-1",
        planner_id="simple_policy",
        scenario_id="corridor",
        seed=1,
        config_digest="c" * 64,
        environment_digest="d" * 64,
    )
    return AuditQueue(
        QueueDataset(
            candidates=(episode,),
            campaign_digest=CAMPAIGN,
            source_digest=source_digest,
        )
    )


def _report(*, source_digest: str = SOURCE, source_revision: str = "3") -> AuditHealthReport:
    protocol = default_audit_protocol()
    return evaluate_coverage(
        [],
        identity=AuditIdentity(
            campaign_digest=CAMPAIGN,
            source_digest=source_digest,
            source_revision=source_revision,
            protocol_digest=protocol.digest,
        ),
        protocol=protocol,
    )


def test_coverage_projection_joins_only_unique_full_episode_identity() -> None:
    scan = scan_campaign(FIXTURE)
    ref = scan.episode_refs[0]
    review = ReviewRecord(
        review_id="review-generated-id",
        episode_id=ref.episode_id,
        scope="full_episode",
        author_kind="human",
        source_identity=scan.audit.source.source_commit,
        scan_identity=scan.cache_key,
    )
    assert (
        evaluate_coverage(scan, review_records=(review,)).counts["reviews"]["full_episode_human"]
        == 0
    )
    projected = coverage_reviews_for_scan(scan, (review,))
    assert projected[0].episode_id == "fixture-readable"
    assert review.episode_id == ref.episode_id  # the durable receipt is never rewritten
    assert (
        evaluate_coverage(scan, review_records=projected).counts["reviews"]["full_episode_human"]
        == 1
    )

    readable = next(item for item in scan.inventory if item.episode_id == "fixture-readable")
    ambiguous = replace(
        scan,
        inventory=(*scan.inventory, replace(readable, episode_id="same-identity-second-row")),
    )
    assert coverage_reviews_for_scan(ambiguous, (review,)) == ()

    wrong_identity = replace(
        scan,
        inventory=tuple(
            replace(item, row={**item.row, "seed": 999}) if item is readable else item
            for item in scan.inventory
        ),
    )
    assert coverage_reviews_for_scan(wrong_identity, (review,)) == ()

    foreign_ref = replace(ref, campaign_digest="f" * 64, episode_id="")
    foreign_scan = replace(scan, episode_refs=(foreign_ref,))
    foreign_review = replace(review, episode_id=foreign_ref.episode_id)
    assert coverage_reviews_for_scan(foreign_scan, (foreign_review,)) == ()

    literal = replace(review, review_id="review-literal", episode_id="fixture-readable")
    assert coverage_reviews_for_scan(scan, (literal,)) == (literal,)
    stale = replace(review, source_identity="stale-source")
    assert (
        evaluate_coverage(scan, review_records=coverage_reviews_for_scan(scan, (stale,))).counts[
            "reviews"
        ]["full_episode_human"]
        == 0
    )


def test_queue_read_ranks_without_advancing_ba02_state() -> None:
    queue = _queue()
    adapter = QueueReadAdapter(_binding(), lambda: queue)
    before = queue.state_revision

    result = adapter.read(context=_context(), limit=1)

    assert result.status == "complete"
    assert result.value["ranked"][0]["episode_id"] == queue.dataset.candidates[0].episode_id
    assert result.value["current_packet"] is None
    assert queue.state_revision == before


def test_queue_read_rejects_stale_context_and_replaced_input() -> None:
    binding = _binding()
    adapter = QueueReadAdapter(binding, _queue)
    with pytest.raises(AuditContextConflict, match="source revision"):
        adapter.read(context=replace(_context(), source_revision=4), limit=1)
    with pytest.raises(AuditContextConflict, match="input identity"):
        QueueReadAdapter(binding, lambda: _queue(source_digest="e" * 64)).read(
            context=_context(), limit=1
        )


def test_coverage_read_preserves_incomplete_deficits_and_source_identity() -> None:
    adapter = CoverageReadAdapter(_binding(), _report)

    result = adapter.read(context=_context())

    assert result.status == "complete"
    assert result.value["status"] == STATUS_INCOMPLETE
    assert result.value["identity"]["source_revision"] == "3"
    with pytest.raises(AuditContextConflict, match="input identity"):
        CoverageReadAdapter(_binding(), lambda: _report(source_digest="e" * 64)).read(
            context=_context()
        )


def test_coverage_read_keeps_scan_commit_distinct_from_service_revision() -> None:
    adapter = CoverageReadAdapter(
        _binding(),
        lambda: _report(source_revision="source-commit-1"),
        report_source_revision="source-commit-1",
    )

    result = adapter.read(context=_context())

    assert result.status == "complete"
    assert result.value["identity"]["source_revision"] == "source-commit-1"
    with pytest.raises(AuditContextConflict, match="source revision"):
        adapter.read(context=replace(_context(), source_revision=4))
    with pytest.raises(AuditContextConflict, match="input identity"):
        CoverageReadAdapter(
            _binding(),
            lambda: _report(source_revision="different-commit"),
            report_source_revision="source-commit-1",
        ).read(context=_context())


def test_service_reads_real_queue_and_coverage_adapters_without_advancing_queue(
    tmp_path: Path,
) -> None:
    service = AuditService(tmp_path / "store", campaign_source=FIXTURE, source_root=FIXTURE.parent)
    try:
        session = service.open_session(
            AuditSelectionContext(campaign_id="audit-fixture-campaign"),
            policy=SessionPolicy(allowed_roots=(str(FIXTURE.parent),)),
        )
        bound = service.update_context(
            session,
            session.context.next_revision(source_identity=session.source_digest),
            expected_context_revision=0,
        )
        assert bound.status == "committed"
        binding = AuditSourceBinding(
            session.context.campaign_id, CAMPAIGN, session.source_digest, session.source_revision
        )
        queue = _queue(source_digest=session.source_digest)
        protocol = default_audit_protocol()
        report = evaluate_coverage(
            [],
            identity=AuditIdentity(
                campaign_digest=CAMPAIGN,
                source_digest=session.source_digest,
                source_revision=str(session.source_revision),
                protocol_digest=protocol.digest,
            ),
            protocol=protocol,
        )
        service.queue_adapter = QueueReadAdapter(binding, lambda: queue)
        service.coverage_adapter = CoverageReadAdapter(binding, lambda: report)
        before = queue.state_revision

        queue_result = service.read_queue(session, limit=1)
        coverage_result = service.read_coverage(session)

        assert queue_result.status == "complete", queue_result.reason
        assert queue_result.value.status == "complete"
        assert queue_result.value.value["ranked"]
        assert coverage_result.status == "complete"
        assert coverage_result.value.value["status"] == STATUS_INCOMPLETE
        assert queue.state_revision == before
    finally:
        service.close()


def test_queue_next_selects_and_binds_packet_under_durable_state(tmp_path: Path) -> None:
    state_path = tmp_path / "queue.json"
    queue = AuditQueue(_queue().dataset, state_path=state_path)
    adapter = QueueNextAdapter(_binding(), lambda: queue)

    result = adapter.select_next(context=_context())

    assert result.status == "complete"
    assert result.value.packet.packet_id.startswith("review-packet-")
    assert result.value.context.primary_episode_id == queue.dataset.candidates[0].episode_id
    assert result.value.state_revision == 1
    assert queue.current_packet == result.value.packet
    assert queue.state.selection_history[-1] == result.value.context


def test_queue_next_rejects_unbound_or_non_durable_queue(tmp_path: Path) -> None:
    with pytest.raises(AuditContextConflict, match="input identity"):
        QueueNextAdapter(
            _binding(),
            lambda: AuditQueue(
                _queue(source_digest="e" * 64).dataset,
                state_path=tmp_path / "stale.json",
            ),
        ).select_next(context=_context())
    with pytest.raises(CapabilityUnavailable, match="state persistence"):
        QueueNextAdapter(_binding(), _queue).select_next(context=_context())


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"not-json", "malformed"),
        (b'{"schema_version":"audit-queue-next.v1"}', "schema is invalid"),
        (
            b'{"schema_version":"other","active":null,"completed":{}}',
            "unsupported",
        ),
        (
            b'{"schema_version":"audit-queue-next.v1","active":[],"completed":{}}',
            "active lease is malformed",
        ),
        (
            b'{"schema_version":"audit-queue-next.v1","active":null,"completed":[]}',
            "replay index is malformed",
        ),
        (
            b'{"schema_version":"audit-queue-next.v1","active":null,"completed":{"op":[]}}',
            "replay entry is malformed",
        ),
        (
            b'{"schema_version":"audit-queue-next.v1","active":null,"completed":{},"value":NaN}',
            "malformed",
        ),
    ],
)
def test_next_coordination_rejects_malformed_or_unsupported_state(
    tmp_path: Path, payload: bytes, message: str
) -> None:
    path = tmp_path / "coordination.json"
    path.write_bytes(payload)
    with pytest.raises(CapabilityUnavailable, match=message):
        adapters_module._read_coordination(path)


def test_next_coordination_rejects_symlinks_and_oversized_writes(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "coordination.json"
    link.symlink_to(target)
    with pytest.raises(CapabilityUnavailable, match="must not be a symlink"):
        adapters_module._read_coordination(link)
    with pytest.raises(CapabilityUnavailable, match="must not be a symlink"):
        adapters_module._write_coordination(link, {})

    oversized = {"payload": "x" * (adapters_module._NEXT_COORDINATION_MAX_BYTES + 1)}
    with pytest.raises(CapabilityUnavailable, match="exceeds its bound"):
        adapters_module._write_coordination(tmp_path / "oversized.json", oversized)
    too_large = tmp_path / "too-large.json"
    too_large.write_bytes(b"x" * (adapters_module._NEXT_COORDINATION_MAX_BYTES + 1))
    with pytest.raises(CapabilityUnavailable, match="exceeds its bound"):
        adapters_module._read_coordination(too_large)


def test_next_coordination_write_failure_is_reported_and_temp_is_removed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "coordination.json"

    def fail_replace(*_args: object, **_kwargs: object) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(adapters_module.os, "replace", fail_replace)
    with pytest.raises(CapabilityUnavailable, match="cannot be committed"):
        adapters_module._write_coordination(path, adapters_module._empty_coordination())
    assert not list(tmp_path.glob("*.tmp"))


def test_next_adapters_fail_closed_on_unavailable_providers_and_contexts(
    tmp_path: Path,
) -> None:
    def invalid_provider() -> object:
        return object()

    def corrupt_queue() -> AuditQueue:
        raise ValueError("corrupt")

    def invalid_report() -> object:
        return object()

    with pytest.raises(AuditContextConflict, match="campaign"):
        QueueReadAdapter(_binding(), _queue).read(
            context=replace(_context(), campaign_id="other"), limit=1
        )
    with pytest.raises(AuditContextConflict, match="source identity"):
        QueueReadAdapter(_binding(), _queue).read(
            context=replace(_context(), source_identity="e" * 64), limit=1
        )
    with pytest.raises(CapabilityUnavailable, match="queue provider"):
        QueueReadAdapter(_binding(), invalid_provider).read(context=_context(), limit=1)
    with pytest.raises(CapabilityUnavailable, match="queue state is unavailable"):
        QueueNextAdapter(_binding(), corrupt_queue).select_next(context=_context())
    with pytest.raises(CapabilityUnavailable, match="queue provider"):
        QueueNextAdapter(_binding(), invalid_provider).select_next(context=_context())

    with pytest.raises(CapabilityUnavailable, match="report provider"):
        CoverageReadAdapter(_binding(), invalid_report).read(context=_context())

    empty = AuditQueue(
        QueueDataset(candidates=(), campaign_digest=CAMPAIGN, source_digest=SOURCE),
        state_path=tmp_path / "empty.json",
    )
    result = QueueNextAdapter(_binding(), lambda: empty).select_next(context=_context())
    assert result.status == "unavailable"


def test_next_transact_rejects_invalid_revisions_and_clears_preflight_lease(
    tmp_path: Path,
) -> None:
    queue = AuditQueue(_queue().dataset, state_path=tmp_path / "queue.json")
    adapter = QueueNextAdapter(_binding(), lambda: queue)
    with pytest.raises(AuditContextConflict, match="state revision is invalid"):
        adapter.transact(context=_context(), expected_state_revision=True)
    with pytest.raises(AuditContextConflict, match="input revision is invalid"):
        adapter.transact(context=_context(), expected_input_revision=-1)

    def reject(_queue: AuditQueue) -> None:
        raise AuditContextConflict("preflight rejected")

    with pytest.raises(AuditContextConflict, match="preflight rejected"):
        adapter.transact(context=_context(), before_select=reject, operation_id="preflight-op")
    sidecar = adapters_module._next_coordination_path(queue.state_path)
    state = json.loads(sidecar.read_text(encoding="utf-8"))
    assert state["active"] is None


def test_next_validation_and_lease_conflicts_are_explicit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    queue = AuditQueue(_queue().dataset, state_path=tmp_path / "queue.json")
    adapter = QueueNextAdapter(_binding(), lambda: queue)
    selection = queue.select_next()
    assert selection is not None

    for changed, message in (
        (replace(selection.context, campaign_digest="e" * 64), "campaign identity"),
        (replace(selection.context, source_digest="e" * 64), "source identity"),
        (replace(selection.context, input_revision=99), "input revision"),
        (replace(selection.context, input_identity="e" * 64), "input identity"),
    ):
        with pytest.raises(AuditContextConflict, match=message):
            adapter._validate_selection(
                queue,
                SelectionResult(packet=selection.packet, context=changed),
                expected_state_revision=None,
                expected_input_revision=None,
                require_current=False,
            )
    with pytest.raises(AuditContextConflict, match="current packet"):
        adapter._validate_selection(
            queue,
            SelectionResult(
                packet=replace(selection.packet, packet_id="other"), context=selection.context
            ),
            expected_state_revision=None,
            expected_input_revision=None,
        )
    with pytest.raises(CapabilityUnavailable, match="selection history"):
        adapter._validate_selection(
            queue,
            SelectionResult(
                packet=selection.packet,
                context=replace(selection.context, selection_id="other-selection"),
            ),
            expected_state_revision=None,
            expected_input_revision=None,
            require_current=False,
        )
    with pytest.raises(AuditContextConflict, match="state revision"):
        adapter._validate_selection(
            queue,
            selection,
            expected_state_revision=99,
            expected_input_revision=None,
        )
    with pytest.raises(AuditContextConflict, match="input revision"):
        adapter._validate_selection(
            queue,
            selection,
            expected_state_revision=None,
            expected_input_revision=99,
        )

    path = adapters_module._next_coordination_path(queue.state_path)
    candidate = adapter._lease_record(
        queue,
        context=_context(),
        operation_id="owner",
        session_id="session",
        expected_state_revision=None,
        expected_input_revision=None,
    )
    for phase, expected in (
        ("preflight", AuditContextConflict),
        ("queue_committed", AuditNextBlocked),
        ("unknown", AuditNextAmbiguous),
    ):
        active = {**candidate, "operation_id": "other", "phase": phase}
        adapters_module._write_coordination(
            path, {**adapters_module._empty_coordination(), "active": active}
        )
        with pytest.raises(expected):
            adapter._claim_lease(
                queue,
                context=_context(),
                operation_id="owner",
                session_id="session",
                expected_state_revision=None,
                expected_input_revision=None,
            )
    same_committed = {**candidate, "phase": "queue_committed"}
    adapters_module._write_coordination(
        path, {**adapters_module._empty_coordination(), "active": same_committed}
    )
    with pytest.raises(AuditNextAmbiguous, match="unresolved"):
        adapter._claim_lease(
            queue,
            context=_context(),
            operation_id="owner",
            session_id="session",
            expected_state_revision=None,
            expected_input_revision=None,
        )

    monkeypatch.setattr(adapters_module, "fcntl", None)
    with pytest.raises(CapabilityUnavailable, match="POSIX"):
        with adapters_module._next_lock(queue.state_path):
            pass
