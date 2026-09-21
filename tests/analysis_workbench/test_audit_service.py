"""Offline BA-05 service contracts: policy, CAS, records, and adapters."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest

from robot_sf.analysis_workbench import audit_authority as audit_authority_module
from robot_sf.analysis_workbench import audit_materialize as audit_materialize_module
from robot_sf.analysis_workbench import audit_scan as audit_scan_module
from robot_sf.analysis_workbench import audit_service as audit_service_module
from robot_sf.analysis_workbench.audit_authority import (
    AuditAuthorityStore,
    AuthorityCorruptionError,
    AuthorityOperationConflict,
    AuthorityPlatformError,
)
from robot_sf.analysis_workbench.audit_contracts import (
    SOURCE_PROVENANCE_MUTATED,
    SOURCE_PROVENANCE_STALE,
    Annotation,
    Finding,
    Reference,
    record_to_dict,
)
from robot_sf.analysis_workbench.audit_service import (
    AuditContextConflict,
    AuditPolicyError,
    AuditSelectionContext,
    AuditService,
    AuditValidationError,
    EpisodeView,
    SessionPolicy,
)
from robot_sf.analysis_workbench.review_contracts import SourceRef

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "analysis_workbench"
    / "audit_campaign_v1"
    / "campaign.json"
)


def _context(*, episode_id: str = "") -> AuditSelectionContext:
    return AuditSelectionContext(campaign_id="audit-fixture-campaign", episode_id=episode_id)


def _service(tmp_path: Path, *, actor: str = "agent") -> tuple[AuditService, object]:
    root = tmp_path / "allowed"
    root.mkdir()
    source = root / "campaign.json"
    shutil.copyfile(FIXTURE, source)
    service = AuditService(tmp_path / "audit-store", campaign_source=source, source_root=root)
    policy = SessionPolicy(allowed_roots=(str(root),), token_budget=4, compute_budget=4.0)
    session = service.open_session(_context(), actor=actor, actor_id=f"{actor}-1", policy=policy)
    return service, session


def _service_for_payload(tmp_path: Path, payload: dict[str, Any]) -> tuple[AuditService, object]:
    root = tmp_path / "allowed"
    root.mkdir()
    source = root / "campaign.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    service = AuditService(tmp_path / "audit-store", campaign_source=source, source_root=root)
    policy = SessionPolicy(allowed_roots=(str(root),), token_budget=4, compute_budget=4.0)
    session = service.open_session(
        AuditSelectionContext(campaign_id=str(payload["campaign_id"])),
        actor="agent",
        actor_id="agent-1",
        policy=policy,
    )
    return service, session


def _versioned_record_service(
    tmp_path: Path,
    *,
    source_revision: int | str | None = 7,
) -> tuple[AuditService, object, SessionPolicy, Path, str, AuditSelectionContext]:
    """Open a service/session pair with an explicit source revision."""

    root = tmp_path / "allowed"
    root.mkdir()
    source = root / "campaign.json"
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    if source_revision is not None:
        payload["source_revision"] = source_revision
    source.write_text(json.dumps(payload), encoding="utf-8")
    policy = SessionPolicy(allowed_roots=(str(root),), token_budget=12, compute_budget=12.0)
    service = AuditService(tmp_path / "audit-store", campaign_source=source, source_root=root)
    session = service.open_session(
        AuditSelectionContext(campaign_id="audit-fixture-campaign"),
        actor="agent",
        actor_id="agent-1",
        policy=policy,
    )
    campaign = service.read_campaign(session)
    assert campaign.value is not None
    episode = campaign.value.episode("fixture-readable")
    assert episode is not None and episode.episode_ref is not None
    generated_id = episode.episode_ref.episode_id
    previous_context = session.context
    updated = service.update_context(
        session,
        previous_context.next_revision(episode_id=generated_id),
        expected_context_revision=previous_context.context_revision,
    )
    assert updated.status == "committed" and updated.value is not None
    return service, session, policy, source, generated_id, previous_context


def test_selection_context_is_closed_and_cursor_revision_is_bound() -> None:
    context = _context()
    assert context.to_dict()["schema_version"] == "audit-selection-context.v1"
    next_context = context.next_revision(episode_id="fixture-readable", time_s=1.0)
    assert next_context.context_revision == 1
    assert next_context.cursor.context_revision == 1

    malformed = context.to_dict()
    malformed["unexpected"] = "data"
    try:
        AuditSelectionContext.from_mapping(malformed)
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("unknown context fields must be rejected")


def test_read_saved_records_reconnects_from_durable_projection(tmp_path: Path) -> None:
    """A fresh service can recover source-bound records without the token in its result."""

    service, session, policy, source, generated_id, _previous_context = _versioned_record_service(
        tmp_path
    )
    session_id = session.session_id
    session_token = session.session_token
    context = session.context
    annotation = Annotation(
        annotation_id="annotation-reconnect",
        episode_id=generated_id,
        classification="unclear",
        author_kind="agent",
        author_id="agent-1",
        source_revision=7,
        source_identity=session.source_digest,
    )
    try:
        assert (
            service.write_annotation(
                session,
                annotation,
                context=context,
                expected_source_revision=7,
                operation_id="annotation-reconnect",
            ).status
            == "committed"
        )
        assert (
            service.finding_from_annotation(
                session,
                annotation,
                finding_id="finding-reconnect",
                title="reconnect finding",
                context=context,
                operation_id="finding-reconnect",
            ).status
            == "committed"
        )
    finally:
        service.close()

    reopened = AuditService(
        tmp_path / "audit-store",
        campaign_source=source,
        source_root=source.parent,
        trusted_policy=policy,
    )
    try:
        reconnected = reopened.reconnect_session(session_id, session_token)
        result = reopened.read_saved_records(
            reconnected.session_id,
            episode_id=generated_id,
            token=session_token,
        )
        assert result.status == "complete", result.reason
        assert result.value is not None
        assert [item.record_type for item in result.value] == ["annotation", "finding"]
        assert [item.record_id for item in result.value] == [
            "annotation-reconnect",
            "finding-reconnect",
        ]
        assert all(item.revision == 1 and item.global_revision > 0 for item in result.value)
        serialized = result.to_dict()
        assert serialized["value"][0]["record"]["episode_id"] == generated_id
        assert serialized["value"][1]["record"]["candidate_members"] == [generated_id]
        assert serialized["value"][1]["record"]["source_revision"] == "7"
        assert session_token not in json.dumps(serialized, sort_keys=True)
    finally:
        reopened.close()


@pytest.mark.parametrize("source_revision", [None, ""])
def test_read_saved_records_supports_digest_bound_zero_revision_reconnect(
    tmp_path: Path,
    source_revision: int | str | None,
) -> None:
    """Explicit digest-bound revision-zero records survive a service restart."""

    service, session, policy, source, generated_id, _previous_context = _versioned_record_service(
        tmp_path,
        source_revision=source_revision,
    )
    assert session.source_revision in (0, "")
    context = session.context
    session_token = session.session_token
    annotation = Annotation(
        annotation_id="annotation-digest-only",
        episode_id=generated_id,
        classification="unclear",
        author_kind="agent",
        author_id="agent-1",
        source_revision=0,
        source_identity=session.source_digest,
    )
    try:
        assert (
            service.write_annotation(
                session,
                annotation,
                context=context,
                expected_source_revision=session.source_revision,
                operation_id="annotation-digest-only",
            ).status
            == "committed"
        )
        assert (
            service.finding_from_annotation(
                session,
                annotation,
                finding_id="finding-digest-only",
                context=context,
                operation_id="finding-digest-only",
            ).status
            == "committed"
        )
        for index, record in enumerate(
            (
                Annotation(
                    annotation_id="annotation-foreign-digest",
                    episode_id=generated_id,
                    classification="unclear",
                    source_revision=0,
                    source_identity="foreign-source",
                ),
                Annotation(
                    annotation_id="annotation-other-episode",
                    episode_id="episode-other",
                    classification="unclear",
                    source_revision=0,
                    source_identity=session.source_digest,
                ),
                Annotation(
                    annotation_id="annotation-blank-digest-revision",
                    episode_id=generated_id,
                    classification="unclear",
                    source_revision="",
                    source_identity=session.source_digest,
                ),
                Finding(
                    finding_id="finding-blank-digest-revision",
                    title="legacy blank finding",
                    candidate_members=(generated_id,),
                    source_revision="",
                ),
                Finding(
                    finding_id="finding-other-episode",
                    title="other episode finding",
                    candidate_members=("episode-other",),
                    source_revision="0",
                ),
            )
        ):
            service.store.save(
                record,
                operation_id=f"seed-digest-only-{index}",
                actor="human",
                actor_id="fixture",
            )

        before = service.read_saved_records(session, episode_id=generated_id)
        assert before.status == "complete", before.reason
        assert before.value is not None
        assert [item.record_id for item in before.value] == [
            "annotation-digest-only",
            "finding-digest-only",
        ]
        finding_before = next(item for item in before.value if item.record_type == "finding")
        assert finding_before.record.source_revision == "0"
    finally:
        service.close()

    reopened = AuditService(
        tmp_path / "audit-store",
        campaign_source=source,
        source_root=source.parent,
        trusted_policy=policy,
    )
    try:
        reconnected = reopened.reconnect_session(session.session_id, session_token)
        after = reopened.read_saved_records(
            reconnected.session_id,
            episode_id=generated_id,
            token=session_token,
        )
        assert after.status == "complete", after.reason
        assert after.value is not None
        assert [item.record_id for item in after.value] == [
            "annotation-digest-only",
            "finding-digest-only",
        ]
        finding_after = next(item for item in after.value if item.record_type == "finding")
        assert finding_after.record.source_revision == "0"
        assert session_token not in json.dumps(after.to_dict(), sort_keys=True)
    finally:
        reopened.close()


def test_read_saved_records_filters_wrong_source_episode_membership_and_revision(
    tmp_path: Path,
) -> None:
    """Stale or unrelated durable rows cannot be promoted into the selected view."""

    service, session, _policy, _source, generated_id, _previous_context = _versioned_record_service(
        tmp_path
    )
    try:
        records = (
            Annotation(
                annotation_id="annotation-valid",
                episode_id=generated_id,
                classification="unclear",
                source_revision=7,
                source_identity=session.source_digest,
            ),
            Annotation(
                annotation_id="annotation-foreign-source",
                episode_id=generated_id,
                classification="unclear",
                source_revision=7,
                source_identity="foreign-source",
            ),
            Annotation(
                annotation_id="annotation-wrong-episode",
                episode_id="episode-other",
                classification="unclear",
                source_revision=7,
                source_identity=session.source_digest,
            ),
            Finding(
                finding_id="finding-valid",
                title="valid finding",
                candidate_members=(generated_id,),
                source_revision="7",
            ),
            Finding(
                finding_id="finding-stale-revision",
                title="stale finding",
                candidate_members=(generated_id,),
                source_revision="6",
            ),
            Finding(
                finding_id="finding-wrong-member",
                title="unrelated finding",
                candidate_members=("episode-other",),
                source_revision="7",
            ),
        )
        for index, record in enumerate(records):
            service.store.save(
                record,
                operation_id=f"seed-read-filter-{index}",
                actor="human",
                actor_id="fixture",
            )

        result = service.read_saved_records(session, episode_id=generated_id)
        assert result.status == "complete", result.reason
        assert result.value is not None
        assert [item.record_id for item in result.value] == [
            "annotation-valid",
            "finding-valid",
        ]
    finally:
        service.close()


def test_finding_from_annotation_rejects_foreign_identity_before_and_after_reconnect(
    tmp_path: Path,
) -> None:
    """Foreign annotation identity cannot become a current-source finding."""

    service, session, policy, source, generated_id, _previous_context = _versioned_record_service(
        tmp_path
    )
    context = session.context
    session_id = session.session_id
    session_token = session.session_token
    annotation = Annotation(
        annotation_id="annotation-foreign-finding-source",
        episode_id=generated_id,
        classification="unclear",
        author_kind="agent",
        author_id="agent-1",
        source_revision=7,
        source_identity="foreign-source",
    )
    try:
        before = service.finding_from_annotation(
            session,
            annotation,
            finding_id="finding-foreign-before-reconnect",
            context=context,
            operation_id="finding-foreign-before-reconnect-op",
        )
        assert before.status == "conflict"
        assert service.store.get("finding-foreign-before-reconnect") is None
    finally:
        service.close()

    reopened = AuditService(
        tmp_path / "audit-store",
        campaign_source=source,
        source_root=source.parent,
        trusted_policy=policy,
    )
    try:
        reconnected = reopened.reconnect_session(session_id, session_token)
        after = reopened.finding_from_annotation(
            reconnected.session_id,
            annotation,
            finding_id="finding-foreign-after-reconnect",
            context=reconnected.context,
            operation_id="finding-foreign-after-reconnect-op",
            token=session_token,
        )
        assert after.status == "conflict"
        assert reopened.store.get("finding-foreign-after-reconnect") is None
        records = reopened.read_saved_records(
            reconnected.session_id,
            episode_id=generated_id,
            token=session_token,
        )
        assert records.status == "complete"
        assert records.value == ()
    finally:
        reopened.close()


def test_finding_from_annotation_rejects_conflicting_source_ref_before_and_after_reconnect(
    tmp_path: Path,
) -> None:
    """A stale/conflicting annotation SourceRef cannot become a finding."""

    service, session, policy, source, generated_id, _previous_context = _versioned_record_service(
        tmp_path,
        source_revision=None,
    )
    context = session.context
    session_id = session.session_id
    session_token = session.session_token
    foreign_source = SourceRef(
        artifact_id="foreign-artifact",
        uri="file:///foreign.json",
        format="application/json",
        sha256="a" * 64,
    )
    annotation = Annotation(
        annotation_id="annotation-conflicting-finding-source-ref",
        episode_id=generated_id,
        classification="unclear",
        author_kind="agent",
        author_id="agent-1",
        source_revision=0,
        source_identity=session.source_digest,
        source_ref=foreign_source,
        provenance_status=SOURCE_PROVENANCE_STALE,
    )
    try:
        before = service.finding_from_annotation(
            session,
            annotation,
            finding_id="finding-conflicting-ref-before-reconnect",
            context=context,
            operation_id="finding-conflicting-ref-before-reconnect-op",
        )
        assert before.status == "conflict"
        assert service.store.get("finding-conflicting-ref-before-reconnect") is None
    finally:
        service.close()

    reopened = AuditService(
        tmp_path / "audit-store",
        campaign_source=source,
        source_root=source.parent,
        trusted_policy=policy,
    )
    try:
        reconnected = reopened.reconnect_session(session_id, session_token)
        after = reopened.finding_from_annotation(
            reconnected.session_id,
            annotation,
            finding_id="finding-conflicting-ref-after-reconnect",
            context=reconnected.context,
            operation_id="finding-conflicting-ref-after-reconnect-op",
            token=session_token,
        )
        assert after.status == "conflict"
        assert reopened.store.get("finding-conflicting-ref-after-reconnect") is None
        records = reopened.read_saved_records(
            reconnected.session_id,
            episode_id=generated_id,
            token=session_token,
        )
        assert records.status == "complete"
        assert records.value == ()
    finally:
        reopened.close()


def test_finding_from_annotation_rejects_stale_mutated_provenance_dataclass_and_mapping(
    tmp_path: Path,
) -> None:
    """Explicit stale/mutated provenance cannot become a finding after restart."""

    service, session, policy, source, generated_id, _previous_context = _versioned_record_service(
        tmp_path,
        source_revision=None,
    )
    context = session.context
    session_id = session.session_id
    session_token = session.session_token
    inputs: list[Annotation | dict[str, Any]] = []
    for index, provenance_status in enumerate((SOURCE_PROVENANCE_STALE, SOURCE_PROVENANCE_MUTATED)):
        annotation = Annotation(
            annotation_id=f"annotation-{provenance_status}-{index}",
            episode_id=generated_id,
            classification="unclear",
            author_kind="agent",
            author_id="agent-1",
            source_revision=0,
            source_identity=session.source_digest,
            provenance_status=provenance_status,
        )
        inputs.extend((annotation, record_to_dict(annotation)))
    try:
        for index, annotation in enumerate(inputs):
            before = service.finding_from_annotation(
                session,
                annotation,
                finding_id=f"finding-provenance-before-{index}",
                context=context,
                operation_id=f"finding-provenance-before-{index}-op",
            )
            assert before.status == "conflict"
            assert service.store.get(f"finding-provenance-before-{index}") is None
    finally:
        service.close()

    reopened = AuditService(
        tmp_path / "audit-store",
        campaign_source=source,
        source_root=source.parent,
        trusted_policy=policy,
    )
    try:
        reconnected = reopened.reconnect_session(session_id, session_token)
        for index, annotation in enumerate(inputs):
            after = reopened.finding_from_annotation(
                reconnected.session_id,
                annotation,
                finding_id=f"finding-provenance-after-{index}",
                context=reconnected.context,
                operation_id=f"finding-provenance-after-{index}-op",
                token=session_token,
            )
            assert after.status == "conflict"
            assert reopened.store.get(f"finding-provenance-after-{index}") is None
        records = reopened.read_saved_records(
            reconnected.session_id,
            episode_id=generated_id,
            token=session_token,
        )
        assert records.status == "complete"
        assert records.value == ()
    finally:
        reopened.close()


def test_finding_from_annotation_rejects_unauthenticated_author_before_and_after_reconnect(
    tmp_path: Path,
) -> None:
    """Finding derivation requires authorship to match the authenticated actor."""

    service, session, policy, source, generated_id, _previous_context = _versioned_record_service(
        tmp_path,
        source_revision=None,
    )
    context = session.context
    session_id = session.session_id
    session_token = session.session_token
    inputs = (
        Annotation(
            annotation_id="annotation-human-author",
            episode_id=generated_id,
            classification="unclear",
            author_kind="human",
            author_id="human-1",
            source_revision=0,
            source_identity=session.source_digest,
        ),
        Annotation(
            annotation_id="annotation-foreign-agent-author",
            episode_id=generated_id,
            classification="unclear",
            author_kind="agent",
            author_id="foreign-agent",
            source_revision=0,
            source_identity=session.source_digest,
        ),
    )
    try:
        for index, annotation in enumerate(inputs):
            before = service.finding_from_annotation(
                session,
                annotation,
                finding_id=f"finding-author-before-{index}",
                context=context,
                operation_id=f"finding-author-before-{index}-op",
            )
            assert before.status == "denied"
            assert service.store.get(f"finding-author-before-{index}") is None
    finally:
        service.close()

    reopened = AuditService(
        tmp_path / "audit-store",
        campaign_source=source,
        source_root=source.parent,
        trusted_policy=policy,
    )
    try:
        reconnected = reopened.reconnect_session(session_id, session_token)
        for index, annotation in enumerate(inputs):
            after = reopened.finding_from_annotation(
                reconnected.session_id,
                annotation,
                finding_id=f"finding-author-after-{index}",
                context=reconnected.context,
                operation_id=f"finding-author-after-{index}-op",
                token=session_token,
            )
            assert after.status == "denied"
            assert reopened.store.get(f"finding-author-after-{index}") is None
        records = reopened.read_saved_records(
            reconnected.session_id,
            episode_id=generated_id,
            token=session_token,
        )
        assert records.status == "complete"
        assert records.value == ()
    finally:
        reopened.close()


@pytest.mark.parametrize("source_revision", [None, ""])
def test_finding_from_annotation_rejects_foreign_nested_reference_revision(
    tmp_path: Path,
    source_revision: int | str | None,
) -> None:
    """Digest-only sessions reject nested references with foreign revisions."""

    service, session, policy, source, generated_id, _previous_context = _versioned_record_service(
        tmp_path,
        source_revision=source_revision,
    )
    context = session.context
    session_id = session.session_id
    session_token = session.session_token
    annotation = Annotation(
        annotation_id="annotation-foreign-nested-revision",
        episode_id=generated_id,
        classification="unclear",
        author_kind="agent",
        author_id="agent-1",
        source_revision=0,
        source_identity=session.source_digest,
        references=(
            Reference(
                reference_id="reference-foreign-revision",
                coordinate_frame="image",
                point=(1.0, 2.0),
                source_revision="foreign-revision",
            ),
        ),
    )
    try:
        before = service.finding_from_annotation(
            session,
            annotation,
            finding_id="finding-foreign-nested-before",
            context=context,
            operation_id="finding-foreign-nested-before-op",
        )
        assert before.status == "conflict"
        assert service.store.get("finding-foreign-nested-before") is None
    finally:
        service.close()

    reopened = AuditService(
        tmp_path / "audit-store",
        campaign_source=source,
        source_root=source.parent,
        trusted_policy=policy,
    )
    try:
        reconnected = reopened.reconnect_session(session_id, session_token)
        after = reopened.finding_from_annotation(
            reconnected.session_id,
            annotation,
            finding_id="finding-foreign-nested-after",
            context=reconnected.context,
            operation_id="finding-foreign-nested-after-op",
            token=session_token,
        )
        assert after.status == "conflict"
        assert reopened.store.get("finding-foreign-nested-after") is None
        records = reopened.read_saved_records(
            reconnected.session_id,
            episode_id=generated_id,
            token=session_token,
        )
        assert records.status == "complete"
        assert records.value == ()
    finally:
        reopened.close()


def test_read_saved_records_abstains_on_ambiguous_generated_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A colliding generated ID never turns a durable row into selected evidence."""

    service, session, _policy, _source, generated_id, _previous_context = _versioned_record_service(
        tmp_path
    )
    try:
        campaign_result = service.read_campaign(session)
        assert campaign_result.value is not None
        ambiguous = replace(
            campaign_result.value,
            episodes=(
                *campaign_result.value.episodes,
                EpisodeView(generated_id, "readable", {"episode_id": generated_id}),
            ),
        )
        monkeypatch.setattr(service, "_scan", lambda *_args, **_kwargs: ambiguous)
        result = service.read_saved_records(session, episode_id=generated_id)
        assert result.status == "unavailable"
        assert result.value is None
    finally:
        service.close()


def test_read_saved_records_requires_authentication_and_rejects_stale_context(
    tmp_path: Path,
) -> None:
    """Session IDs, forged tokens, and old contexts cannot read durable records."""

    service, session, _policy, _source, generated_id, previous_context = _versioned_record_service(
        tmp_path
    )
    try:
        with pytest.raises(AuditPolicyError, match="token is required"):
            service.read_saved_records(session.session_id, episode_id=generated_id)
        with pytest.raises(AuditPolicyError, match="token is invalid"):
            forged = "forged-session-token"
            service.read_saved_records(
                session.session_id,
                episode_id=generated_id,
                token=forged,
            )

        stale = service.read_saved_records(
            session,
            episode_id=generated_id,
            context=previous_context,
        )
        assert stale.status == "conflict"
        assert stale.value is None
        assert "token" not in json.dumps(stale.to_dict(), sort_keys=True).lower()
    finally:
        service.close()


def test_read_episode_resolves_unique_generated_queue_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A BA-02 packet ID reads its literal BA-01 row without changing write identity."""

    service, session = _service(tmp_path)
    try:
        campaign = service.read_campaign(session)
        assert campaign.value is not None
        readable = campaign.value.episode("fixture-readable")
        assert readable is not None and readable.episode_ref is not None
        generated_id = readable.episode_ref.episode_id
        assert generated_id != readable.episode_id
        selected = service.read_episode(session, generated_id)
        assert selected.status == "complete"
        assert selected.value is not None
        assert selected.value.episode_id == "fixture-readable"
        assert selected.value.episode_ref == readable.episode_ref
        metrics = service.read_metrics(session, generated_id)
        assert metrics.status == "complete"
        assert metrics.value == {"clearance_m": 1.0, "speed_m_s": 0.5}
        generated_signals = service.read_signals(session, generated_id)
        literal_signals = service.read_signals(session, readable.episode_id)
        assert generated_signals.status == "complete"
        assert generated_signals.value == literal_signals.value
        assert generated_signals.value
        related = service.related_cases(session, generated_id)
        assert related.status == "complete"
        assert related.value is not None
        assert all(item.candidate_id != "fixture-readable" for item in related.value)

        colliding = EpisodeView(generated_id, "readable", {"episode_id": generated_id})
        ambiguous = replace(campaign.value, episodes=(*campaign.value.episodes, colliding))
        assert ambiguous.episode(generated_id) is None
        monkeypatch.setattr(service, "_scan", lambda *_args, **_kwargs: ambiguous)
        collision_signals = service.read_signals(session, generated_id)
        assert collision_signals.status == "complete"
        assert collision_signals.value == ()
    finally:
        service.close()


@pytest.mark.parametrize(
    ("identity_fields", "expected_config_digest"),
    [
        ({"config_digest": "", "config": {"config_digest": "a" * 64}}, "a" * 64),
        ({"config_digest": "", "config": {"config_hash": "b" * 64}}, "b" * 64),
        (
            {"config_digest": "", "config_identity": "nested-config-identity"},
            hashlib.sha256(b"nested-config-identity").hexdigest(),
        ),
    ],
)
def test_generated_alias_reads_use_nested_config_identity(
    tmp_path: Path,
    identity_fields: dict[str, object],
    expected_config_digest: str,
) -> None:
    """Generated reads retain scanner config-digest fallback semantics."""

    row = {
        "episode_id": "nested-config-row",
        "execution_id": "nested-config-execution",
        "planner_id": "nested-config-planner",
        "scenario_id": "nested-config-scenario",
        "seed": 3,
        "metrics": {"clearance_m": 1.0},
        **identity_fields,
    }
    payload = {
        "campaign_id": "nested-config-campaign",
        "schema_version": "campaign-result.v1",
        "execution_status": "native",
        "expected_episode_ids": ["nested-config-row"],
        "episodes": [row],
    }
    service, session = _service_for_payload(tmp_path, payload)
    try:
        campaign = service.read_campaign(session)
        assert campaign.status == "complete", campaign.reason
        assert campaign.value is not None
        readable = campaign.value.episode("nested-config-row")
        assert readable is not None and readable.episode_ref is not None
        assert readable.episode_ref.config_digest == expected_config_digest
        generated_id = readable.episode_ref.episode_id

        selected = service.read_episode(session, generated_id)
        assert selected.status == "complete"
        assert selected.value is not None
        assert selected.value.episode_id == "nested-config-row"
        generated_signals = service.read_signals(session, generated_id)
        literal_signals = service.read_signals(session, "nested-config-row")
        assert generated_signals.status == "complete"
        assert generated_signals.value == literal_signals.value
        assert generated_signals.value
    finally:
        service.close()


def test_scan_binds_refs_by_full_identity_when_execution_id_is_reused(tmp_path: Path) -> None:
    """Distinct planner/seed rows sharing an execution ID keep distinct refs."""

    payload = {
        "campaign_id": "identity-campaign",
        "schema_version": "campaign-result.v1",
        "execution_status": "native",
        "expected_episode_ids": ["literal-a", "literal-b"],
        "episodes": [
            {
                "episode_id": "literal-a",
                "execution_id": "shared-execution",
                "planner_id": "planner-a",
                "scenario_id": "shared-scenario",
                "seed": 1,
                "metrics": {"clearance_m": 1.0},
            },
            {
                "episode_id": "literal-b",
                "execution_id": "shared-execution",
                "planner_id": "planner-b",
                "scenario_id": "shared-scenario",
                "seed": 2,
                "metrics": {"clearance_m": 2.0},
            },
        ],
    }
    service, session = _service_for_payload(tmp_path, payload)
    try:
        result = service.read_campaign(session)
        assert result.status == "complete", result.reason
        assert result.value is not None
        episodes = {item.episode_id: item for item in result.value.episodes}
        first = episodes["literal-a"].episode_ref
        second = episodes["literal-b"].episode_ref
        assert first is not None and second is not None
        assert first.episode_id != second.episode_id
        assert first.planner_id == "planner-a"
        assert second.planner_id == "planner-b"
        assert first.seed == 1
        assert second.seed == 2
        for literal_id, reference in (("literal-a", first), ("literal-b", second)):
            selected = service.read_episode(session, reference.episode_id)
            assert selected.status == "complete"
            assert selected.value is not None
            assert selected.value.episode_id == literal_id
            generated_signals = service.read_signals(session, reference.episode_id)
            literal_signals = service.read_signals(session, literal_id)
            assert generated_signals.status == "complete"
            assert generated_signals.value == literal_signals.value
            assert generated_signals.value
    finally:
        service.close()


def test_scan_drops_ambiguous_duplicate_episode_refs(tmp_path: Path) -> None:
    """Identical generated identities never attach a ref to either row."""

    payload = {
        "campaign_id": "ambiguous-campaign",
        "schema_version": "campaign-result.v1",
        "execution_status": "native",
        "expected_episode_ids": ["literal-a", "literal-b"],
        "episodes": [
            {
                "episode_id": "literal-a",
                "execution_id": "same-execution",
                "planner_id": "same-planner",
                "scenario_id": "same-scenario",
                "seed": 1,
                "metrics": {"clearance_m": 1.0},
            },
            {
                "episode_id": "literal-b",
                "execution_id": "same-execution",
                "planner_id": "same-planner",
                "scenario_id": "same-scenario",
                "seed": 1,
                "metrics": {"clearance_m": 2.0},
            },
        ],
    }
    service, session = _service_for_payload(tmp_path, payload)
    try:
        result = service.read_campaign(session)
        assert result.status == "complete", result.reason
        assert result.value is not None
        episodes = {item.episode_id: item for item in result.value.episodes}
        assert episodes["literal-a"].episode_ref is None
        assert episodes["literal-b"].episode_ref is None
        refs = result.value.report.episode_refs
        assert len(refs) == 2
        assert refs[0].episode_id == refs[1].episode_id
        generated_id = refs[0].episode_id
        assert result.value.episode(generated_id) is None
        selected = service.read_episode(session, generated_id)
        assert selected.status == "unavailable"
        assert selected.value is not None
        assert selected.value.reason == "episode is not present"
        ambiguous_signals = service.read_signals(session, generated_id)
        assert ambiguous_signals.status == "complete"
        assert ambiguous_signals.value == ()
    finally:
        service.close()


def test_source_revision_identity_and_allowlist_checks_are_server_bound(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    source = root / "campaign.json"
    shutil.copyfile(FIXTURE, source)
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["source_revision"] = 7
    source.write_text(json.dumps(payload), encoding="utf-8")
    service = AuditService(tmp_path / "store", campaign_source=source, source_root=root)
    try:
        session = service.open_session(
            AuditSelectionContext(campaign_id="audit-fixture-campaign"),
            policy=SessionPolicy(
                allowed_roots=(str(root),),
                allowed_repositories=("robot-sf/audit",),
                allowed_recipes=("recipe-a",),
            ),
        )
        updated = service.update_context(
            session,
            session.context.next_revision(episode_id="fixture-readable", source_revision=7),
            expected_context_revision=0,
        )
        assert updated.status == "committed"
        assert updated.value is not None and updated.value.source_revision == 7
        allowed = service.read_campaign(session, repository="robot-sf/audit", recipe_id="recipe-a")
        denied = service.read_campaign(session, repository="other/repo", recipe_id="recipe-a")
        assert allowed.status == "partial"
        assert denied.status == "denied"
    finally:
        service.close()


def test_session_handles_are_immutable_and_id_calls_require_authentication(tmp_path: Path) -> None:
    service, session = _service(tmp_path)
    try:
        with pytest.raises(AuditPolicyError, match="token is required"):
            service.read_context(session.session_id)
        with pytest.raises(FrozenInstanceError):
            session.actor = session.actor
        with pytest.raises(FrozenInstanceError):
            session.usage.tokens = 0
        authenticated = service.read_context(session.session_id, token=session.session_token)
        assert authenticated.status == "complete"
    finally:
        service.close()


def test_materialization_requires_selected_episode_and_allowlisted_output(tmp_path: Path) -> None:
    service, session = _service(tmp_path)
    try:
        output = tmp_path / "allowed" / "derived"
        unselected = service.materialize_selected(
            session, output_root=output, operation_id="materialize-unselected"
        )
        assert unselected.status == "conflict"
        assert not output.exists()
        assert service.get_session(session).usage.compute == 0.0

        updated = service.update_context(
            session,
            session.context.next_revision(episode_id="fixture-readable"),
            expected_context_revision=0,
        )
        assert updated.status == "committed"
        denied = service.materialize_selected(
            session,
            output_root=tmp_path / "outside",
            operation_id="materialize-outside",
        )
        assert denied.status == "denied"
        assert not (tmp_path / "outside").exists()
        assert service.get_session(session).usage.compute == 0.0
    finally:
        service.close()


@pytest.mark.parametrize("existing_output_root", [False, True])
def test_materialization_output_root_swap_after_policy_check_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_output_root: bool,
) -> None:
    """A root replaced after policy admission cannot redirect materialization."""

    service, session = _service(tmp_path)
    output = tmp_path / "allowed" / "derived-race"
    admitted = tmp_path / "allowed" / "derived-race-admitted"
    outside = tmp_path / "outside"
    outside.mkdir()
    if existing_output_root:
        output.mkdir()
    try:
        updated = service.update_context(
            session,
            session.context.next_revision(episode_id="fixture-readable"),
            expected_context_revision=0,
        )
        assert updated.status == "committed"
        original_allows_path = SessionPolicy.allows_path
        swapped = False

        def replace_after_check(policy: SessionPolicy, path: str | Path) -> bool:
            nonlocal swapped
            allowed = original_allows_path(policy, path)
            if not swapped and allowed and Path(path) == output:
                if output.exists():
                    output.rename(admitted)
                output.symlink_to(outside, target_is_directory=True)
                swapped = True
            return allowed

        monkeypatch.setattr(SessionPolicy, "allows_path", replace_after_check)
        result = service.materialize_selected(
            session,
            output_root=output,
            operation_id=f"materialize-root-race-{existing_output_root}",
        )

        assert swapped
        assert result.status == "denied"
        assert result.value is None
        assert service.get_session(session).usage.compute == 0.0
        assert not any(outside.rglob("*"))
    finally:
        if output.is_symlink():
            output.unlink()
        if admitted.exists():
            admitted.rmdir()
        service.close()


def test_materialization_source_root_swap_after_policy_check_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source root replaced after policy admission cannot redirect reads."""

    allowed = tmp_path / "allowed"
    source = allowed / "source"
    source.mkdir(parents=True)
    campaign_source = source / "campaign.json"
    shutil.copyfile(FIXTURE, campaign_source)
    outside = tmp_path / "outside"
    outside.mkdir()
    admitted = allowed / "source-admitted"
    service = AuditService(
        tmp_path / "audit-store",
        campaign_source=campaign_source,
        source_root=source,
    )
    policy = SessionPolicy(allowed_roots=(str(allowed),), token_budget=4, compute_budget=4.0)
    session = service.open_session(_context(), actor="agent", actor_id="agent-1", policy=policy)
    try:
        updated = service.update_context(
            session,
            session.context.next_revision(episode_id="fixture-readable"),
            expected_context_revision=0,
        )
        assert updated.status == "committed"
        original_allows_path = SessionPolicy.allows_path
        swapped = False

        def replace_source_after_check(policy_arg: SessionPolicy, path: str | Path) -> bool:
            nonlocal swapped
            allowed_path = original_allows_path(policy_arg, path)
            if not swapped and allowed_path and Path(path) == source:
                source.rename(admitted)
                source.symlink_to(outside, target_is_directory=True)
                swapped = True
            return allowed_path

        monkeypatch.setattr(SessionPolicy, "allows_path", replace_source_after_check)
        result = service.materialize_selected(
            session,
            output_root=allowed / "derived",
            operation_id="materialize-source-root-race",
        )

        assert swapped
        assert result.status == "denied"
        assert result.value is None
        assert service.get_session(session).usage.compute == 0.0
        assert not any(outside.rglob("*"))
    finally:
        if source.is_symlink():
            source.unlink()
        if admitted.exists():
            admitted.rename(source)
        service.close()


def test_materialization_replay_never_reruns_source_adapter(tmp_path: Path) -> None:
    service, session = _service(tmp_path)
    try:
        updated = service.update_context(
            session,
            session.context.next_revision(episode_id="fixture-readable"),
            expected_context_revision=0,
        )
        assert updated.status == "committed"
        output = tmp_path / "allowed" / "derived"
        first = service.materialize_selected(
            session, output_root=output, operation_id="materialize-fixture"
        )
        assert first.operation is not None
        assert first.value is not None
        assert first.value.status == "unavailable"
        assert first.value.simulation_executed is False
        assert service.get_session(session).usage.compute == 1.0
        repeated = service.materialize_selected(
            session, output_root=output, operation_id="materialize-fixture"
        )
        assert repeated.operation is not None and repeated.operation.replayed
        assert repeated.status == "unavailable"
        assert repeated.value is None
        assert service.get_session(session).usage.compute == 1.0
        changed_request = service.materialize_selected(
            session,
            output_root=output,
            output_directory="another",
            operation_id="materialize-fixture",
        )
        assert changed_request.status == "conflict"
    finally:
        service.close()


def test_materialization_retained_render_uses_descriptor_bound_output_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retained render can consume a missing root through the capability."""

    service, session = _service(tmp_path)
    try:
        updated = service.update_context(
            session,
            session.context.next_revision(episode_id="fixture-readable"),
            expected_context_revision=0,
        )
        assert updated.status == "committed"
        original_episode_lookup = service._episode_from_campaign

        def retained_episode(session_arg: object, episode_id: str) -> EpisodeView:
            episode = original_episode_lookup(session_arg, episode_id)  # type: ignore[arg-type]
            row = {
                "episode_id": episode.episode_id,
                "retained_states": [
                    {"time_s": 0.0, "x": 0.0, "y": 0.0},
                    {"time_s": 1.0, "x": 1.0, "y": 0.0},
                ],
            }
            return EpisodeView(episode.episode_id, episode.status, row, episode.reason)

        monkeypatch.setattr(service, "_episode_from_campaign", retained_episode)
        output = tmp_path / "allowed" / "retained-derived"
        result = service.materialize_selected(
            session,
            output_root=output,
            output_directory="render",
            operation_id="materialize-retained-render",
        )

        assert result.status == "complete", result.reason
        assert result.value is not None
        assert result.value.materialization_kind == "derived_render"
        assert (output / "render" / "trajectory.png").is_file()
        assert (output / "render" / "audit-materialization.v1.json").is_file()
    finally:
        service.close()


@pytest.mark.parametrize("cancel_after_charge", [False, True])
def test_materialization_cancellation_never_reaches_adapter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancel_after_charge: bool
) -> None:
    service, session = _service(tmp_path)
    try:
        updated = service.update_context(
            session,
            session.context.next_revision(episode_id="fixture-readable"),
            expected_context_revision=0,
        )
        assert updated.status == "committed"
        original_charge = service.consume_budget

        def cancel_at_charge(*args, **kwargs):
            if not cancel_after_charge:
                killed = service.kill_switch(session, reason="operator cancellation")
                assert killed.status == "cancelled"
            result = original_charge(*args, **kwargs)
            if cancel_after_charge:
                assert result.ok
                killed = service.kill_switch(session, reason="operator cancellation")
                assert killed.status == "cancelled"
            return result

        monkeypatch.setattr(service, "consume_budget", cancel_at_charge)
        output = tmp_path / "allowed" / "derived"
        result = service.materialize_selected(
            session,
            output_root=output,
            operation_id=f"materialize-cancel-{cancel_after_charge}",
        )
        assert result.status == "cancelled"
        assert not output.exists()
        assert service.get_session(session).usage.compute == (1.0 if cancel_after_charge else 0.0)
    finally:
        service.close()


def test_materialization_cancellation_between_admission_and_adapter_is_diagnostic_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A late cancellation keeps artifacts diagnostic and blocks later work."""

    service, session = _service(tmp_path)
    try:
        updated = service.update_context(
            session,
            session.context.next_revision(episode_id="fixture-readable"),
            expected_context_revision=0,
        )
        assert updated.status == "committed"
        original_episode_lookup = service._episode_from_campaign

        def retained_episode(session_arg: object, episode_id: str) -> EpisodeView:
            episode = original_episode_lookup(session_arg, episode_id)  # type: ignore[arg-type]
            return EpisodeView(
                episode.episode_id,
                episode.status,
                {
                    "episode_id": episode.episode_id,
                    "retained_states": [
                        {"time_s": 0.0, "x": 0.0, "y": 0.0},
                        {"time_s": 1.0, "x": 1.0, "y": 0.0},
                    ],
                },
                episode.reason,
            )

        monkeypatch.setattr(service, "_episode_from_campaign", retained_episode)
        adapter_entered = threading.Event()
        release_adapter = threading.Event()
        real_materialize = audit_materialize_module.materialize_episode

        def hold_adapter(*args: Any, **kwargs: Any) -> Any:
            adapter_entered.set()
            assert release_adapter.wait(timeout=5)
            return real_materialize(*args, **kwargs)

        monkeypatch.setattr(audit_materialize_module, "materialize_episode", hold_adapter)
        output = tmp_path / "allowed" / "cancelled-derived"
        worker_result: dict[str, Any] = {}

        def run_materialization() -> None:
            worker_result["result"] = service.materialize_selected(
                session,
                output_root=output,
                output_directory="render",
                operation_id="materialize-cancelled-admission-race",
            )

        worker = threading.Thread(target=run_materialization)
        worker.start()
        assert adapter_entered.wait(timeout=5)
        killed = service.kill_switch(
            session,
            reason="cancel while adapter admission is held",
            operation_id="cancelled-admission-race-stop",
        )
        assert killed.status == "cancelled"
        release_adapter.set()
        worker.join(timeout=10)
        assert not worker.is_alive()

        result = worker_result["result"]
        assert result.status == "cancelled"
        assert result.value is None
        assert result.reason == "cancel while adapter admission is held"
        assert service.get_session(session).usage.compute == 1.0
        manifest = output / "render" / "audit-materialization.v1.json"
        assert manifest.is_file()
        assert json.loads(manifest.read_text(encoding="utf-8"))["materialization_kind"] == (
            "derived_render"
        )

        later = service.materialize_selected(
            session,
            output_root=tmp_path / "allowed" / "cancelled-later",
            operation_id="materialize-after-cancellation",
        )
        assert later.status == "cancelled"
        assert not (tmp_path / "allowed" / "cancelled-later").exists()
        assert service.get_session(session).usage.compute == 1.0
    finally:
        service.close()


def test_context_cas_cannot_clear_bound_source_identity(tmp_path: Path) -> None:
    service, session = _service(tmp_path)
    try:
        bound = service.update_context(
            session,
            session.context.next_revision(source_identity=session.source_digest),
            expected_context_revision=0,
        )
        assert bound.status == "committed"
        cleared = service.update_context(
            session,
            bound.value.next_revision(source_identity=""),
            expected_context_revision=1,
        )
        assert cleared.status == "conflict"
        assert session.context.source_identity == session.source_digest
    finally:
        service.close()


def test_replay_is_source_context_bound_and_read_value_is_explicitly_unavailable(
    tmp_path: Path,
) -> None:
    service, session = _service(tmp_path)
    try:
        first = service.read_episode(session, "fixture-readable", operation_id="read-replay")
        assert first.status == "complete"
        repeated = service.read_episode(session, "fixture-readable", operation_id="read-replay")
        assert repeated.status == "unavailable"
        assert "not durably retained" in repeated.reason

        source = Path(service.campaign_source)
        source.write_bytes(source.read_bytes() + b" ")
        stale_source = service.read_episode(session, "fixture-readable", operation_id="read-replay")
        assert stale_source.status == "conflict"
    finally:
        service.close()

    context_root = tmp_path / "context"
    context_root.mkdir()
    service, session = _service(context_root)
    try:
        first = service.read_episode(session, "fixture-readable", operation_id="context-replay")
        assert first.status == "complete"
        updated = service.update_context(
            session,
            session.context.next_revision(episode_id="fixture-readable"),
            expected_context_revision=0,
        )
        assert updated.status == "committed"
        stale_context = service.read_episode(
            session, "fixture-readable", operation_id="context-replay"
        )
        assert stale_context.status == "conflict"
    finally:
        service.close()


def test_nested_annotation_reference_and_budget_replays_fail_closed(tmp_path: Path) -> None:
    service, session = _service(tmp_path)
    context = session.context.next_revision(episode_id="fixture-readable")
    try:
        assert service.update_context(session, context, expected_context_revision=0).status == (
            "committed"
        )
        foreign = SourceRef(
            artifact_id="foreign-artifact",
            uri="file:///foreign.json",
            format="application/json",
            sha256="a" * 64,
        )
        annotation = Annotation(
            annotation_id="nested-foreign-source",
            episode_id="fixture-readable",
            classification="unclear",
            author_kind="agent",
            author_id="agent-1",
            references=(
                Reference(
                    reference_id="nested-reference",
                    coordinate_frame="image",
                    point=(1.0, 2.0),
                    source=foreign,
                ),
            ),
        )
        nested = service.write_annotation(session, annotation, context=context)
        assert nested.status == "conflict"

        first = service.consume_budget(session, tokens=1, operation_id="budget-replay")
        changed = service.consume_budget(session, tokens=2, operation_id="budget-replay")
        assert first.status == "committed"
        assert changed.status == "conflict"
        assert session.usage.tokens == 1

        entered = threading.Event()
        release = threading.Event()
        original_charge = service._charge

        def slow_charge(target, **kwargs):
            entered.set()
            assert release.wait(timeout=5)
            return original_charge(target, **kwargs)

        service._charge = slow_charge
        results: list[object] = []
        first_thread = threading.Thread(
            target=lambda: results.append(
                service.consume_budget(session, tokens=1, operation_id="concurrent-budget")
            )
        )
        second_thread = threading.Thread(
            target=lambda: results.append(
                service.consume_budget(session, tokens=1, operation_id="concurrent-budget")
            )
        )
        first_thread.start()
        assert entered.wait(timeout=5)
        second_thread.start()
        second_thread.join(timeout=5)
        release.set()
        first_thread.join(timeout=5)
        statuses = sorted(result.status for result in results)
        assert statuses == ["committed", "conflict"]
    finally:
        service.close()


def test_nested_source_reference_requires_all_supplied_identity_fields_to_match(
    tmp_path: Path,
) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    source = root / "campaign.json"
    shutil.copyfile(FIXTURE, source)
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    canonical = SourceRef(
        artifact_id="canonical-campaign",
        uri=source.name,
        format="campaign-result",
        schema="campaign-result.v1",
        sha256=source_digest,
    )
    service = AuditService(
        tmp_path / "store",
        campaign_source=source,
        source_ref=canonical,
        source_root=root,
    )
    session = service.open_session(
        _context(),
        actor="agent",
        actor_id="agent-1",
        policy=SessionPolicy(allowed_roots=(str(root),), token_budget=4, compute_budget=4.0),
    )
    context = session.context.next_revision(episode_id="fixture-readable")
    try:
        assert service.update_context(session, context, expected_context_revision=0).status == (
            "committed"
        )
        mixed = Annotation(
            annotation_id="nested-mixed-source",
            episode_id="fixture-readable",
            classification="unclear",
            author_kind="agent",
            author_id="agent-1",
            references=(
                Reference(
                    reference_id="nested-mixed-reference",
                    coordinate_frame="image",
                    point=(1.0, 2.0),
                    source=SourceRef(
                        artifact_id=canonical.artifact_id,
                        uri=canonical.uri,
                        format=canonical.format,
                        sha256="a" * 64,
                    ),
                ),
            ),
        )
        result = service.write_annotation(session, mixed, context=context)
        assert result.status == "conflict"
        assert "source" in result.reason
    finally:
        service.close()


def test_budget_reservation_settlement_requires_opaque_owner_capability(
    tmp_path: Path,
) -> None:
    service, session = _service(tmp_path)
    try:
        reserved = service.reserve_budget(session, tokens=3, operation_id="reserve-owner")
        assert reserved.status == "committed"
        assert isinstance(reserved.value, dict)
        reservation_id = reserved.value["reservation_id"]
        forged = service.settle_budget(
            session,
            reserved_tokens=3,
            reserved_compute=0.0,
            actual_tokens=0,
            actual_compute=0.0,
            reservation_id=reservation_id,
            reservation_operation_id="forged-owner",
            operation_id="settle-forged",
        )
        assert forged.status == "denied"
        assert session.usage.tokens == 3

        settled = service.settle_budget(
            session,
            reserved_tokens=3,
            reserved_compute=0.0,
            actual_tokens=0,
            actual_compute=0.0,
            reservation_id=reservation_id,
            reservation_operation_id="reserve-owner",
            operation_id="settle-owner",
        )
        assert settled.status == "committed"
        assert session.usage.to_dict() == {"tokens": 0, "compute": 0.0, "issue_writes": 0}
    finally:
        service.close()


def test_budget_reservation_replay_reconstructs_active_capability(tmp_path: Path) -> None:
    service, session = _service(tmp_path)
    try:
        first = service.reserve_budget(session, tokens=2, operation_id="reserve-replay")
        replay = service.reserve_budget(session, tokens=2, operation_id="reserve-replay")
        assert first.status == "committed"
        assert replay.status == "committed"
        assert replay.operation is not None and replay.operation.replayed
        assert isinstance(first.value, dict) and isinstance(replay.value, dict)
        assert replay.value["reservation_id"] == first.value["reservation_id"]

        settled = service.settle_budget(
            session,
            reserved_tokens=2,
            reserved_compute=0.0,
            actual_tokens=0,
            actual_compute=0.0,
            reservation_id=replay.value["reservation_id"],
            reservation_operation_id="reserve-replay",
            operation_id="settle-replayed-reservation",
        )
        assert settled.status == "committed"
        unavailable = service.reserve_budget(session, tokens=2, operation_id="reserve-replay")
        assert unavailable.status == "unavailable"
        assert "no longer available" in unavailable.reason
    finally:
        service.close()


def test_session_admission_rejects_canonical_source_digest_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    source = root / "campaign.json"
    shutil.copyfile(FIXTURE, source)
    service = AuditService(
        tmp_path / "store",
        campaign_source=source,
        source_root=root,
        source_ref=SourceRef(
            artifact_id="canonical-campaign",
            uri=str(source),
            format="application/json",
            sha256="b" * 64,
        ),
    )
    try:
        with pytest.raises(AuditContextConflict, match="digest"):
            service.open_session(
                _context(),
                actor="agent",
                actor_id="agent-1",
                policy=SessionPolicy(allowed_roots=(str(root),)),
            )
    finally:
        service.close()


def test_session_admission_rejects_canonical_source_path_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    source = root / "campaign.json"
    shutil.copyfile(FIXTURE, source)
    service = AuditService(
        tmp_path / "store",
        campaign_source=source,
        source_root=root,
        source_ref=SourceRef(
            artifact_id="canonical-campaign",
            uri="foreign.json",
            format="campaign-result",
            schema="campaign-result.v1",
            sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        ),
    )
    try:
        with pytest.raises(AuditContextConflict, match="scan contract|path"):
            service.open_session(
                _context(),
                actor="agent",
                actor_id="agent-1",
                policy=SessionPolicy(allowed_roots=(str(root),)),
            )
    finally:
        service.close()


def test_session_admission_rejects_canonical_source_format_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    source = root / "campaign.json"
    shutil.copyfile(FIXTURE, source)
    service = AuditService(
        tmp_path / "store",
        campaign_source=source,
        source_root=root,
        source_ref=SourceRef(
            artifact_id="canonical-campaign",
            uri=source.name,
            format="json",
            schema="campaign-result.v1",
            sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        ),
    )
    try:
        with pytest.raises(AuditContextConflict, match="path or format"):
            service.open_session(
                _context(),
                actor="agent",
                actor_id="agent-1",
                policy=SessionPolicy(allowed_roots=(str(root),)),
            )
    finally:
        service.close()


@pytest.mark.parametrize("configured", (False, True))
def test_source_reference_admission_is_detectorless(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, configured: bool
) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    source = root / "campaign.json"
    shutil.copyfile(FIXTURE, source)
    source_ref = (
        SourceRef(
            artifact_id="canonical-campaign",
            uri=source.name,
            format="campaign-result",
            schema="campaign-result.v1",
            sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        )
        if configured
        else None
    )
    service = AuditService(
        tmp_path / "store",
        campaign_source=source,
        source_root=root,
        source_ref=source_ref,
    )
    scan_detector_ids: list[tuple[str, ...] | None] = []
    detector_calls: list[str] = []
    original_scan = audit_service_module.scan_campaign
    original_detect = audit_scan_module.detect

    def count_scan(*args, **kwargs):
        scan_detector_ids.append(kwargs.get("detector_ids"))
        return original_scan(*args, **kwargs)

    def count_detect(detector_id, *args, **kwargs):
        detector_calls.append(detector_id)
        return original_detect(detector_id, *args, **kwargs)

    monkeypatch.setattr(audit_service_module, "scan_campaign", count_scan)
    monkeypatch.setattr(audit_scan_module, "detect", count_detect)
    try:
        session = service.open_session(
            _context(),
            actor="agent",
            actor_id="agent-1",
            policy=SessionPolicy(allowed_roots=(str(root),)),
        )
        assert session.source_digest == hashlib.sha256(source.read_bytes()).hexdigest()
        assert scan_detector_ids == [()] * (2 if configured else 1)
        assert detector_calls == []
    finally:
        service.close()


def test_kill_switch_replays_only_the_original_request(tmp_path: Path) -> None:
    service, session = _service(tmp_path)
    try:
        first = service.kill_switch(session, reason="first stop", operation_id="kill-replay")
        replay = service.kill_switch(session, reason="first stop", operation_id="kill-replay")
        changed = service.kill_switch(session, reason="changed stop", operation_id="kill-replay")
        assert first.status == "cancelled"
        assert replay.status == "cancelled"
        assert replay.operation is not None and replay.operation.replayed
        assert changed.status == "conflict"
        after = service.store.get("audit-operation-kill-replay-after")
        assert after is not None
        assert after.record.details["reason"] == "first stop"
    finally:
        service.close()


def test_kill_switch_releases_active_budget_reservations(tmp_path: Path) -> None:
    service, session = _service(tmp_path)
    try:
        reserved = service.reserve_budget(session, tokens=2, operation_id="kill-reserve")
        assert reserved.status == "committed"
        reservation_id = reserved.value["reservation_id"]
        killed = service.kill_switch(session, reason="release work", operation_id="kill-release")
        assert killed.status == "cancelled"
        assert not service._reservations
        assert session.usage.to_dict() == {"tokens": 0, "compute": 0.0, "issue_writes": 0}
        settled = service.settle_budget(
            session,
            reserved_tokens=2,
            reserved_compute=0.0,
            actual_tokens=0,
            actual_compute=0.0,
            reservation_id=reservation_id,
            reservation_operation_id="kill-reserve",
            operation_id="settle-after-kill",
        )
        assert settled.status == "denied"
    finally:
        service.close()


def test_generic_reservation_cannot_claim_codex_namespace(tmp_path: Path) -> None:
    service, session = _service(tmp_path)
    try:
        with pytest.raises(AuditValidationError, match="service-owned"):
            service.reserve_budget(
                session,
                tokens=1,
                operation_id="codex-budget-reserve:forged",
            )
        assert service.get_session(session).usage.tokens == 0
        assert not service._reservations
        stopped = service.kill_switch(session, operation_id="reject-forged-reserve")
        assert stopped.status == "cancelled"
    finally:
        service.close()


def test_record_write_rolls_back_when_source_changes_during_store_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, session = _service(tmp_path)
    source = Path(service.campaign_source)
    context = session.context.next_revision(episode_id="fixture-readable")
    annotation = Annotation(
        annotation_id="interleaved-source-write",
        episode_id="fixture-readable",
        classification="unclear",
        author_kind="agent",
        author_id="agent-1",
    )
    try:
        assert service.update_context(session, context, expected_context_revision=0).status == (
            "committed"
        )
        original_save = service.store.save

        def mutate_after_commit(record, **kwargs):
            committed = original_save(record, **kwargs)
            if getattr(record, "annotation_id", "") == annotation.annotation_id:
                source.write_bytes(source.read_bytes() + b" interleaved")
            return committed

        monkeypatch.setattr(service.store, "save", mutate_after_commit)
        result = service.write_annotation(session, annotation, context=context)
        assert result.status == "conflict"
        assert "during record commit" in result.reason
        assert service.store.get(annotation.annotation_id) is None
        tombstone = service.store.get(annotation.annotation_id, include_deleted=True)
        assert tombstone is not None and tombstone.deleted
    finally:
        service.close()


def test_source_mutation_after_write_linearization_fails_closed_on_next_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, session = _service(tmp_path)
    source = Path(service.campaign_source)
    context = session.context.next_revision(episode_id="fixture-readable")
    annotation = Annotation(
        annotation_id="post-linearization-source-write",
        episode_id="fixture-readable",
        classification="unclear",
        author_kind="agent",
        author_id="agent-1",
    )
    try:
        assert service.update_context(session, context, expected_context_revision=0).status == (
            "committed"
        )
        original_assert_source = service._assert_source
        assert_calls = 0

        def mutate_after_final_check(target, **kwargs):
            nonlocal assert_calls
            result = original_assert_source(target, **kwargs)
            assert_calls += 1
            if assert_calls == 4:
                source.write_bytes(source.read_bytes() + b" post-linearization")
            return result

        monkeypatch.setattr(service, "_assert_source", mutate_after_final_check)
        result = service.write_annotation(session, annotation, context=context)
        monkeypatch.setattr(service, "_assert_source", original_assert_source)
        assert assert_calls == 4
        assert result.status == "committed"
        assert service.store.get(annotation.annotation_id) is not None

        next_read = service.read_episode(
            session,
            "fixture-readable",
            operation_id="post-linearization-read",
        )
        assert next_read.status == "conflict"
        assert "source changed" in next_read.reason
    finally:
        service.close()


def test_service_reads_are_source_bound_and_optional_siblings_are_unavailable(
    tmp_path: Path,
) -> None:
    service, session = _service(tmp_path)
    try:
        campaign = service.read_campaign(session)
        assert campaign.status == "partial"
        assert campaign.value is not None
        assert campaign.value.episode("fixture-readable").metrics == {
            "clearance_m": 1.0,
            "speed_m_s": 0.5,
        }
        metrics = service.read_metrics(session, "fixture-readable")
        assert metrics.status == "complete"
        assert metrics.value["clearance_m"] == 1.0
        queue = service.read_queue(session)
        coverage = service.read_coverage(session)
        assert queue.status == "unavailable"
        assert queue.value.reason == "ba-02 adapter is not merged"
        assert coverage.status == "unavailable"
        assert coverage.value.reason == "ba-04 adapter is not merged"
    finally:
        service.close()


def test_related_case_lookup_is_scoped_and_bounded(tmp_path: Path) -> None:
    service, session = _service(tmp_path)
    try:
        found = service.related_cases(session, "fixture-readable", limit=1)
        missing = service.related_cases(session, "does-not-exist", limit=1)
        assert found.status == "complete"
        assert missing.status == "complete"
        assert missing.value == ()
        with pytest.raises(AuditValidationError, match="related-case limit"):
            service.related_cases(session, "fixture-readable", limit=1_001)
        with pytest.raises(AuditValidationError, match="related-case limit"):
            service.related_cases(session, "fixture-readable", limit=True)
    finally:
        service.close()


def test_agent_reference_and_finding_writes_preserve_candidate_boundary(tmp_path: Path) -> None:
    service, session = _service(tmp_path)
    context = session.context.next_revision(episode_id="fixture-readable")
    try:
        assert service.update_context(session, context, expected_context_revision=0).status == (
            "committed"
        )
        reference = Reference(reference_id="ba05-ref-1", coordinate_frame="image", point=(2.0, 3.0))
        saved_reference = service.write_reference(
            session, reference, context=context, operation_id="ba05-write-ref"
        )
        assert saved_reference.status == "committed"
        candidate = Finding(
            finding_id="ba05-candidate-1",
            title="Unconfirmed clearance concern",
            candidate_members=("fixture-readable",),
        )
        saved_finding = service.write_finding(
            session, candidate, context=context, operation_id="ba05-write-finding"
        )
        assert saved_finding.status == "committed"
        prohibited = Finding(
            finding_id="ba05-confirmed-by-agent",
            title="Unsupported confirmation",
            status="supported",
            confirmed_members=("fixture-readable",),
        )
        denied = service.write_finding(
            session, prohibited, context=context, operation_id="ba05-write-confirmed"
        )
        assert denied.status == "denied"
        proposed = service.finding_from_annotation(
            session,
            Annotation(
                annotation_id="ba05-annotation-for-finding",
                episode_id="fixture-readable",
                classification="unclear",
                author_kind="agent",
                author_id="agent-1",
                source_revision=0,
                source_identity=session.source_digest,
            ),
            finding_id="ba05-from-annotation",
            context=context,
            operation_id="ba05-from-annotation-op",
        )
        assert proposed.status == "committed"
        assert proposed.value is not None
    finally:
        service.close()


def test_annotation_write_uses_store_cas_actor_separation_and_before_after_records(
    tmp_path: Path,
) -> None:
    service, session = _service(tmp_path)
    try:
        annotation = Annotation(
            annotation_id="annotation-agent-1",
            episode_id="fixture-readable",
            classification="unclear",
            author_kind="agent",
            author_id="agent-1",
        )
        context = _context(episode_id="fixture-readable").next_revision(time_s=1.0)
        updated_context = service.update_context(
            session,
            context,
            expected_context_revision=0,
            operation_id="annotation-context-op",
        )
        assert updated_context.status == "committed"
        committed = service.write_annotation(
            session,
            annotation,
            context=context,
            operation_id="annotation-op-1",
        )
        assert committed.status == "committed"
        assert committed.value is not None
        assert committed.value.revision == 1

        replayed = service.write_annotation(
            session,
            annotation,
            context=context,
            operation_id="annotation-op-1",
        )
        assert replayed.status == "committed"
        assert replayed.operation is not None and replayed.operation.replayed

        conflicting_replay = service.write_annotation(
            session,
            Annotation(
                annotation_id="annotation-agent-different-payload",
                episode_id="fixture-readable",
                classification="normal",
                author_kind="agent",
                author_id="agent-1",
            ),
            context=context,
            operation_id="annotation-op-1",
        )
        assert conflicting_replay.status == "conflict"

        human_record = Annotation(
            annotation_id="annotation-human-impersonation",
            episode_id="fixture-readable",
            classification="unclear",
            author_kind="human",
        )
        denied = service.write_annotation(
            session,
            human_record,
            context=context,
            operation_id="annotation-op-impersonation",
        )
        assert denied.status == "denied"

        records = service.list_operation_records(session)
        statuses = {(record.action_type, record.status) for record in records}
        assert ("write.annotation", "started") in statuses
        assert ("write.annotation", "committed") in statuses
        assert all(record.actor_kind == "agent" for record in records)
    finally:
        service.close()


def test_stale_context_source_and_budget_failures_are_receipted(tmp_path: Path) -> None:
    service, session = _service(tmp_path)
    try:
        updated = service.update_context(
            session,
            _context(episode_id="fixture-readable").next_revision(time_s=1.0),
            expected_context_revision=0,
            operation_id="context-op-1",
        )
        assert updated.status == "committed"
        stale = service.read_episode(
            session, "fixture-readable", context=_context(), operation_id="stale-op"
        )
        assert stale.status == "conflict"
        assert stale.operation is not None

        budget = service.consume_budget(session, tokens=4)
        assert budget.status == "committed"
        exhausted = service.consume_budget(session, tokens=1)
        assert exhausted.status == "denied"

        source = Path(service.campaign_source)
        payload = json.loads(source.read_text(encoding="utf-8"))
        payload["source_revision"] = 2
        source.write_text(json.dumps(payload), encoding="utf-8")
        changed = service.read_campaign(session, operation_id="changed-source-op")
        assert changed.status == "conflict"

        killed = service.kill_switch(session, reason="test stop", operation_id="kill-op")
        assert killed.status == "cancelled"
        cancelled = service.read_campaign(session, operation_id="cancelled-op")
        assert cancelled.status == "cancelled"
        assert any(
            record.status == "cancelled" for record in service.list_operation_records(session)
        )
    finally:
        service.close()


def test_session_reconnect_persists_context_usage_reservation_and_redacts_token(
    tmp_path: Path,
) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    source = root / "campaign.json"
    shutil.copyfile(FIXTURE, source)
    policy = SessionPolicy(
        allowed_roots=(str(root),),
        token_budget=4,
        compute_budget=4.0,
        policy_id="launcher-audit",
    )
    service = AuditService(tmp_path / "store", campaign_source=source, source_root=root)
    session = service.open_session(
        _context(), policy=policy, session_id="durable-session", actor_id="agent-1"
    )
    token = session.session_token
    context = session.context.next_revision(episode_id="fixture-readable")
    assert service.update_context(session, context, expected_context_revision=0).status == (
        "committed"
    )
    reserved = service.reserve_budget(session, tokens=2, operation_id="durable-reservation")
    assert reserved.status == "committed"
    assert token not in (tmp_path / "store" / "audit-authority.ndjson").read_text()
    assert "session_token" not in session.to_dict()
    service.close()

    reopened = AuditService(
        tmp_path / "store",
        campaign_source=source,
        source_root=root,
        trusted_policy=policy,
    )
    recovered = reopened.reconnect_session("durable-session", token)
    assert recovered.context == context
    assert recovered.usage.tokens == 2
    assert reopened.validate_session_token("durable-session", token)
    assert len(reopened._reservations) == 1
    reservation_id = next(iter(reopened._reservations))
    settled = reopened.settle_budget(
        recovered,
        reserved_tokens=2,
        reserved_compute=0.0,
        actual_tokens=1,
        actual_compute=0.0,
        reservation_id=reservation_id,
        reservation_operation_id="durable-reservation",
        operation_id="durable-settlement",
    )
    assert settled.status == "committed"
    assert recovered.usage.tokens == 1
    reopened.close()


def test_cross_instance_budget_reservation_is_serialized_and_bounded(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    source = root / "campaign.json"
    shutil.copyfile(FIXTURE, source)
    policy = SessionPolicy(allowed_roots=(str(root),), token_budget=1, compute_budget=1.0)
    first = AuditService(tmp_path / "store", campaign_source=source, source_root=root)
    opened = first.open_session(_context(), policy=policy, session_id="race-session")
    token = opened.session_token
    first.close()
    left = AuditService(
        tmp_path / "store", campaign_source=source, source_root=root, trusted_policy=policy
    )
    right = AuditService(
        tmp_path / "store", campaign_source=source, source_root=root, trusted_policy=policy
    )
    left_session = left.reconnect_session("race-session", token)
    right_session = right.reconnect_session("race-session", token)
    results: list[str] = []

    def reserve(service: AuditService, session: object, operation_id: str) -> None:
        result = service.reserve_budget(session, tokens=1, operation_id=operation_id)  # type: ignore[arg-type]
        results.append(result.status)

    one = threading.Thread(target=reserve, args=(left, left_session, "race-left"))
    two = threading.Thread(target=reserve, args=(right, right_session, "race-right"))
    one.start()
    two.start()
    one.join(timeout=5)
    two.join(timeout=5)
    assert not one.is_alive() and not two.is_alive()
    assert sorted(results) == ["committed", "denied"]
    current = left.reconnect_session("race-session", token)
    assert current.usage.tokens == 1
    assert len(left._reservations) == 1
    left.close()
    right.close()


def test_external_write_operation_replays_once_after_reconnect(tmp_path: Path) -> None:
    service, session = _service(tmp_path)
    context = session.context.next_revision(episode_id="fixture-readable")
    annotation = Annotation(
        annotation_id="durable-write-replay",
        episode_id="fixture-readable",
        classification="unclear",
        author_kind="agent",
        author_id="agent-1",
    )
    assert service.update_context(session, context, expected_context_revision=0).status == (
        "committed"
    )
    first = service.write_annotation(
        session,
        annotation,
        context=context,
        operation_id="durable-write-operation",
    )
    assert first.status == "committed"
    token = session.session_token
    sid = session.session_id
    policy = session.policy
    service.close()

    reopened = AuditService(
        tmp_path / "audit-store",
        campaign_source=tmp_path / "allowed" / "campaign.json",
        source_root=tmp_path / "allowed",
        trusted_policy=policy,
    )
    recovered = reopened.reconnect_session(sid, token)
    replay = reopened.write_annotation(
        recovered,
        annotation,
        context=context,
        operation_id="durable-write-operation",
    )
    assert replay.status == "committed"
    assert replay.operation is not None and replay.operation.replayed
    assert [item.revision for item in reopened.store.history(annotation.annotation_id)] == [1]
    reopened.close()


def test_cross_instance_inflight_operation_replay_never_runs_second_callback(  # noqa: C901, PLR0915
    tmp_path: Path,
) -> None:
    """A replayed begin receipt must not authorize a second external write."""

    service, session = _service(tmp_path)
    token = session.session_token
    policy = session.policy
    service.close()
    left = AuditService(
        tmp_path / "audit-store",
        campaign_source=tmp_path / "allowed" / "campaign.json",
        source_root=tmp_path / "allowed",
        trusted_policy=policy,
    )
    right = AuditService(
        tmp_path / "audit-store",
        campaign_source=tmp_path / "allowed" / "campaign.json",
        source_root=tmp_path / "allowed",
        trusted_policy=policy,
    )
    left_session = left.reconnect_session(session.session_id, token)
    right_session = right.reconnect_session(session.session_id, token)
    operation_id = "cross-instance-inflight-write"
    prebegin = threading.Barrier(2)
    begin_return = threading.Barrier(2)
    first_callback = threading.Event()
    replay_checked = threading.Event()
    release_callback = threading.Event()
    writes: list[str] = []
    errors: list[BaseException] = []
    results: list[str] = []
    writes_lock = threading.Lock()

    for current in (left, right):
        original_existing = current._existing_receipt
        first_existing = [True]

        def gate_existing(*args, _original=original_existing, _first=first_existing, **kwargs):
            value = _original(*args, **kwargs)
            if _first[0]:
                _first[0] = False
                prebegin.wait(timeout=5)
            return value

        current._existing_receipt = gate_existing  # type: ignore[method-assign]
        original_mutate = current.authority.mutate

        def gate_begin(*args, _original=original_mutate, **kwargs):
            mutation = _original(*args, **kwargs)
            transaction_id = args[0] if args else kwargs.get("operation_id", "")
            if isinstance(transaction_id, str) and transaction_id.startswith(
                "authority.operation.begin:"
            ):
                begin_return.wait(timeout=5)
            return mutation

        current.authority.mutate = gate_begin  # type: ignore[method-assign]

    for current in (left, right):
        original_lookup = current._authority_operation

        def observe_replay(operation_id_value: str, _original=original_lookup):
            value = _original(operation_id_value)
            if operation_id_value == operation_id and value is not None:
                replay_checked.set()
            return value

        current._authority_operation = observe_replay  # type: ignore[method-assign]

    def external_write() -> None:
        with writes_lock:
            writes.append("write")
            callback_number = len(writes)
        if callback_number == 1:
            first_callback.set()
            assert release_callback.wait(timeout=5)
        else:
            # The old admission bug reaches this branch while the first
            # callback is deliberately held in-flight.
            replay_checked.set()

    def invoke(current: AuditService, target: object) -> None:
        try:
            result, _receipt = current._execute(
                target,  # type: ignore[arg-type]
                operation_type="write.external",
                operation_id=operation_id,
                context=target.context,  # type: ignore[attr-defined]
                request_digest="a" * 64,
                callback=external_write,
            )
            results.append(result.status)
        except BaseException as exc:  # pragma: no cover - regression guard
            errors.append(exc)

    one = threading.Thread(target=invoke, args=(left, left_session))
    two = threading.Thread(target=invoke, args=(right, right_session))
    one.start()
    two.start()
    assert first_callback.wait(timeout=5)
    assert replay_checked.wait(timeout=5)
    # The event is set by the replay branch, not by a second callback.  A
    # second external write would set it from ``external_write`` as well, but
    # only after the first callback has been held above.
    with writes_lock:
        assert writes == ["write"]
    release_callback.set()
    one.join(timeout=5)
    two.join(timeout=5)
    assert not one.is_alive() and not two.is_alive()
    assert errors == []
    assert sorted(results) == ["committed", "conflict"]
    left.close()
    right.close()


def test_cross_session_global_operation_id_rejects_second_callback(  # noqa: C901, PLR0915
    tmp_path: Path,
) -> None:
    """An operation ID remains owned when a different session races admission."""

    root = tmp_path / "allowed"
    root.mkdir()
    source = root / "campaign.json"
    shutil.copyfile(FIXTURE, source)
    policy = SessionPolicy(allowed_roots=(str(root),), token_budget=4, compute_budget=4.0)
    initial = AuditService(tmp_path / "audit-store", campaign_source=source, source_root=root)
    session_a = initial.open_session(
        _context(),
        actor_id="shared-agent",
        policy=policy,
        session_id="global-session-a",
    )
    session_b = initial.open_session(
        _context(),
        actor_id="shared-agent",
        policy=policy,
        session_id="global-session-b",
    )
    token_a = session_a.session_token
    token_b = session_b.session_token
    initial.close()
    left = AuditService(
        tmp_path / "audit-store",
        campaign_source=source,
        source_root=root,
        trusted_policy=policy,
    )
    right = AuditService(
        tmp_path / "audit-store",
        campaign_source=source,
        source_root=root,
        trusted_policy=policy,
    )
    left_session = left.reconnect_session("global-session-a", token_a)
    right_session = right.reconnect_session("global-session-b", token_b)
    operation_id = "cross-session-global-operation"
    after_existing = threading.Barrier(2)
    first_callback = threading.Event()
    admission_checked = threading.Event()
    release_callback = threading.Event()
    writes: list[str] = []
    errors: list[BaseException] = []
    results: list[str] = []
    writes_lock = threading.Lock()

    for current in (left, right):
        original_existing = current._existing_receipt
        first_existing = [True]

        def gate_existing(*args, _original=original_existing, _first=first_existing, **kwargs):
            value = _original(*args, **kwargs)
            if _first[0]:
                _first[0] = False
                after_existing.wait(timeout=5)
            return value

        current._existing_receipt = gate_existing  # type: ignore[method-assign]
        original_mutate = current.authority.mutate

        def observe_begin(*args, _original=original_mutate, **kwargs):
            transaction_id = args[0] if args else kwargs.get("operation_id", "")
            try:
                return _original(*args, **kwargs)
            except AuthorityOperationConflict:
                if isinstance(transaction_id, str) and transaction_id.startswith(
                    "authority.operation.begin:"
                ):
                    admission_checked.set()
                raise

        current.authority.mutate = observe_begin  # type: ignore[method-assign]

    def external_write() -> None:
        with writes_lock:
            writes.append("write")
            callback_number = len(writes)
        if callback_number == 1:
            first_callback.set()
            assert release_callback.wait(timeout=5)
        else:
            # The pre-fix cross-session admission bug reaches this branch.
            admission_checked.set()

    def invoke(current: AuditService, target: object) -> None:
        try:
            result, _receipt = current._execute(
                target,  # type: ignore[arg-type]
                operation_type="write.external",
                operation_id=operation_id,
                context=target.context,  # type: ignore[attr-defined]
                request_digest="a" * 64,
                callback=external_write,
            )
            results.append(result.status)
        except BaseException as exc:  # pragma: no cover - regression guard
            errors.append(exc)

    one = threading.Thread(target=invoke, args=(left, left_session))
    two = threading.Thread(target=invoke, args=(right, right_session))
    one.start()
    two.start()
    try:
        assert first_callback.wait(timeout=5)
        assert admission_checked.wait(timeout=5)
        with writes_lock:
            assert writes == ["write"]
    finally:
        release_callback.set()
    one.join(timeout=5)
    two.join(timeout=5)
    assert not one.is_alive() and not two.is_alive()
    assert errors == []
    assert sorted(results) == ["committed", "conflict"]
    left.close()
    right.close()


def test_cancel_race_cannot_charge_a_reservation_after_authority_cancel(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    source = root / "campaign.json"
    shutil.copyfile(FIXTURE, source)
    policy = SessionPolicy(allowed_roots=(str(root),), token_budget=1, compute_budget=1.0)
    initial = AuditService(tmp_path / "store", campaign_source=source, source_root=root)
    opened = initial.open_session(_context(), policy=policy, session_id="cancel-race")
    token = opened.session_token
    initial.close()
    left = AuditService(
        tmp_path / "store", campaign_source=source, source_root=root, trusted_policy=policy
    )
    right = AuditService(
        tmp_path / "store", campaign_source=source, source_root=root, trusted_policy=policy
    )
    left_session = left.reconnect_session("cancel-race", token)
    right_session = right.reconnect_session("cancel-race", token)
    entered = threading.Event()
    release = threading.Event()
    original_mutate = left._mutate_session_authority

    def delayed_mutate(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return original_mutate(*args, **kwargs)

    left._mutate_session_authority = delayed_mutate  # type: ignore[method-assign]
    reservation_result: list[object] = []
    reserve_thread = threading.Thread(
        target=lambda: reservation_result.append(
            left.reserve_budget(left_session, tokens=1, operation_id="cancel-race-reserve")
        )
    )
    reserve_thread.start()
    assert entered.wait(timeout=5)
    cancelled = right.kill_switch(right_session, reason="race stop", operation_id="race-cancel")
    release.set()
    reserve_thread.join(timeout=5)
    assert cancelled.status == "cancelled"
    assert not reserve_thread.is_alive()
    assert reservation_result[0].status == "cancelled"
    recovered = left.reconnect_session("cancel-race", token)
    assert recovered.cancelled
    assert recovered.usage.to_dict() == {"tokens": 0, "compute": 0.0, "issue_writes": 0}
    assert not left._reservations
    left.close()
    right.close()


def test_cancel_reconnect_and_replay_cannot_resurrect_session(tmp_path: Path) -> None:
    service, session = _service(tmp_path)
    token = session.session_token
    sid = session.session_id
    reserved = service.reserve_budget(session, tokens=2, operation_id="cancel-durable-reserve")
    assert reserved.status == "committed"
    cancelled = service.kill_switch(session, reason="operator stop", operation_id="cancel-durable")
    assert cancelled.status == "cancelled"
    service.close()

    policy = SessionPolicy(
        allowed_roots=(str(tmp_path / "allowed"),), token_budget=4, compute_budget=4.0
    )
    reopened = AuditService(
        tmp_path / "audit-store",
        campaign_source=tmp_path / "allowed" / "campaign.json",
        source_root=tmp_path / "allowed",
        trusted_policy=policy,
    )
    recovered = reopened.reconnect_session(sid, token)
    assert recovered.cancelled and recovered.cancel_reason == "operator stop"
    assert not reopened._reservations
    replay = reopened.kill_switch(recovered, reason="operator stop", operation_id="cancel-durable")
    assert replay.status == "cancelled"
    assert replay.operation is not None and replay.operation.replayed
    denied = reopened.consume_budget(recovered, tokens=1, operation_id="after-cancel")
    assert denied.status == "cancelled"
    reopened.close()


def test_reconnect_requires_trusted_policy_and_current_source(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    source = root / "campaign.json"
    shutil.copyfile(FIXTURE, source)
    policy = SessionPolicy(
        allowed_roots=(str(root),), token_budget=2, compute_budget=2.0, policy_id="policy-a"
    )
    service = AuditService(tmp_path / "store", campaign_source=source, source_root=root)
    session = service.open_session(_context(), policy=policy)
    token = session.session_token
    sid = session.session_id
    service.close()
    untrusted = AuditService(tmp_path / "store", campaign_source=source, source_root=root)
    with pytest.raises(AuditPolicyError, match="trusted launcher policy"):
        untrusted.reconnect_session(sid, token)
    with pytest.raises(AuditPolicyError, match="trusted launcher policy"):
        untrusted.read_context(sid, token=token)
    untrusted.close()
    narrower = SessionPolicy(
        allowed_roots=(str(root),), token_budget=1, compute_budget=2.0, policy_id="policy-a"
    )
    recovered_service = AuditService(
        tmp_path / "store",
        campaign_source=source,
        source_root=root,
        trusted_policy=narrower,
    )
    with pytest.raises(AuditPolicyError, match="exceeds trusted"):
        recovered_service.reconnect_session(sid, token)
    source.write_bytes(source.read_bytes() + b" changed")
    trusted = AuditService(
        tmp_path / "store", campaign_source=source, source_root=root, trusted_policy=policy
    )
    with pytest.raises(AuditPolicyError, match="source cannot be revalidated"):
        trusted.reconnect_session(sid, token)
    recovered_service.close()
    trusted.close()


def test_reconnect_rejects_changed_full_source_reference_identity(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    source = root / "campaign.json"
    shutil.copyfile(FIXTURE, source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    policy = SessionPolicy(allowed_roots=(str(root),), token_budget=2, compute_budget=2.0)
    source_a = SourceRef(
        artifact_id="artifact-a",
        uri=source.name,
        format="campaign-result",
        schema="campaign-result.v1",
        sha256=digest,
    )
    service = AuditService(
        tmp_path / "store",
        campaign_source=source,
        source_root=root,
        source_ref=source_a,
    )
    session = service.open_session(_context(), policy=policy, session_id="source-ref-session")
    token = session.session_token
    assert session.source_ref is not None
    assert session.source_ref.artifact_id == "artifact-a"
    assert service.evidence_refs(session)[0]["artifact_id"] == "artifact-a"
    service.close()

    source_b = replace(session.source_ref, artifact_id="artifact-b")
    changed = AuditService(
        tmp_path / "store",
        campaign_source=source,
        source_root=root,
        source_ref=source_b,
        trusted_policy=policy,
    )
    with pytest.raises(AuditPolicyError, match="full identity changed"):
        changed.reconnect_session("source-ref-session", token)
    changed.close()


def test_inflight_reservation_is_released_during_restart_reconciliation(tmp_path: Path) -> None:
    root = tmp_path / "allowed"
    root.mkdir()
    source = root / "campaign.json"
    shutil.copyfile(FIXTURE, source)
    policy = SessionPolicy(allowed_roots=(str(root),), token_budget=2, compute_budget=2.0)
    service = AuditService(tmp_path / "store", campaign_source=source, source_root=root)
    session = service.open_session(_context(), policy=policy)
    token = session.session_token
    state = service.authority.snapshot()
    state["sessions"][session.session_id]["usage"]["tokens"] = 1
    state["reservations"]["crashed-reservation"] = {
        "reservation_id": "crashed-reservation",
        "session_id": session.session_id,
        "operation_id": "crashed-reserve-op",
        "tokens": 1,
        "compute": 0.0,
        "issue_writes": 0,
    }
    state["operations"]["crashed-reserve-op"] = {
        "operation_id": "crashed-reserve-op",
        "operation_type": "budget.reserve",
        "session_id": session.session_id,
        "actor": session.actor.to_dict(),
        "request_digest": "a" * 64,
        "context_revision": session.context.context_revision,
        "source_revision": session.source_revision,
        "source_digest": session.source_digest,
        "status": "inflight",
        "result_status": "",
        "reason": "",
        "result_digest": "",
        "created_at": session.created_at,
    }
    service.authority.mutate(
        "test.seed-crashed-reservation",
        "b" * 64,
        lambda latest: latest.update(state) or {"seeded": True},
    )
    service.close()
    reopened = AuditService(
        tmp_path / "store",
        campaign_source=source,
        source_root=root,
        trusted_policy=policy,
    )
    recovered = reopened.reconnect_session(session.session_id, token)
    assert recovered.usage.tokens == 0
    assert not reopened._reservations
    assert (
        reopened.authority.snapshot()["operations"]["crashed-reserve-op"]["result_status"]
        == "unavailable"
    )
    reopened.close()


def test_authority_journal_tamper_fails_closed(tmp_path: Path) -> None:
    authority = AuditAuthorityStore(tmp_path / "authority")
    authority.mutate("one", "a" * 64, lambda state: state)
    journal = tmp_path / "authority" / authority.JOURNAL_FILENAME
    raw = journal.read_text(encoding="utf-8")
    journal.write_text(raw.replace('"sequence":1', '"sequence":2', 1), encoding="utf-8")
    with pytest.raises(AuthorityCorruptionError):
        AuditAuthorityStore(tmp_path / "authority")


def test_authority_fails_closed_without_cross_process_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(audit_authority_module, "fcntl", None)
    with pytest.raises(AuthorityPlatformError, match="fcntl.flock"):
        AuditAuthorityStore(tmp_path / "authority")


def test_authority_exposes_serialized_sequence_cas_for_combined_transitions(
    tmp_path: Path,
) -> None:
    authority = AuditAuthorityStore(tmp_path / "authority")
    _state, sequence = authority.snapshot_with_sequence()
    authority.mutate("one", "a" * 64, lambda state: state, expected_sequence=sequence)
    with pytest.raises(AuthorityOperationConflict, match="sequence conflict"):
        authority.mutate("two", "b" * 64, lambda state: state, expected_sequence=sequence)
