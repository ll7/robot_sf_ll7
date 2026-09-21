"""Executable BA-03 persistence contract examples."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import zipfile
from dataclasses import replace

import pytest

from robot_sf.analysis_workbench.audit_contracts import (
    ActionRecord,
    Annotation,
    AuditContractError,
    DetectorRuleProposal,
    EpisodeRef,
    ReviewRecord,
    Signal,
    TimeInterval,
    record_to_dict,
)
from robot_sf.analysis_workbench.audit_store import (
    AuditConflictError,
    AuditCorruptionError,
    AuditExportError,
    AuditStore,
    AuditStoreError,
    OperationConflictError,
    ProjectionError,
)


def _episode() -> EpisodeRef:
    return EpisodeRef(
        campaign_digest="a" * 64,
        source_digest="b" * 64,
        execution_id="exec-1",
        planner_id="ppo",
        scenario_id="corridor",
        seed=7,
        config_digest="c" * 64,
    )


def test_quick_annotation_round_trips_after_canonical_commit(tmp_path) -> None:
    annotation = Annotation(
        annotation_id="annotation-1",
        episode_id=_episode().episode_id,
        classification="interesting_valid",
        mode="quick",
        author_kind="human",
        interval=TimeInterval(1.0, 1.5),
    )
    with AuditStore(tmp_path) as store:
        result = store.save(annotation, operation_id="op-1", expected_revision=0)
        loaded = store.get(annotation.annotation_id)

    assert result.committed
    assert loaded is not None
    assert loaded.record == annotation
    assert loaded.revision == 1
    assert store.canonical_path.exists()


def _annotation(annotation_id: str, *, text: str = "") -> Annotation:
    return Annotation(
        annotation_id=annotation_id,
        episode_id=_episode().episode_id,
        classification="unclear",
        mode="quick",
        observed_behavior=text,
    )


def _proposal(proposal_id: str = "proposal-1") -> DetectorRuleProposal:
    return DetectorRuleProposal(
        proposal_id=proposal_id,
        proposal_kind="threshold_change",
        target_detector_id="goal_adjacent_timeout",
        candidate_rule={
            "predicate": "goal_adjacent_timeout.v1",
            "parameters": {"tail_steps": 120},
        },
        detector_registry_version="audit-detector-registry.v1",
        detector_registry_digest="d" * 64,
        campaign_digest="a" * 64,
        source_identity="source-1",
        source_revision="commit-1",
        annotation_ids=("annotation-1",),
        finding_ids=("finding-1",),
        episode_ids=("episode-1",),
        rationale="The observed tail behavior warrants a bounded threshold probe.",
        metadata={"evidence_boundary": "diagnostic_only"},
        proposer_kind="agent",
        proposer_id="agent-1",
        author_kind="agent",
        author_id="agent-1",
    )


def test_detector_rule_proposal_agent_create_human_decision_uses_cas(tmp_path) -> None:
    proposal = _proposal()
    with AuditStore(tmp_path) as store:
        created = store.save(
            proposal,
            operation_id="proposal-create",
            expected_revision=0,
            actor="agent",
            actor_id="agent-1",
        )
        approved = replace(
            proposal,
            lifecycle_status="approved",
            author_kind="human",
            author_id="reviewer-1",
            decided_by_kind="human",
            decided_by_id="reviewer-1",
            decided_at="2026-09-20T10:00:00Z",
            decision_reason="Human review accepted this diagnostic candidate.",
            updated_at="2026-09-20T10:00:01Z",
        )
        decision = store.save(
            approved,
            operation_id="proposal-approve",
            expected_revision=created.revision,
            actor="human",
            actor_id="reviewer-1",
        )
        loaded = store.get(proposal.proposal_id)
        assert loaded is not None
        assert loaded.record == approved
        assert loaded.revision == 2
        assert loaded.proposer_kind == "agent"
        assert loaded.activation_status == "inactive"
        assert decision.revision == 2
        assert [item.revision for item in store.history(proposal.proposal_id)] == [1, 2]
        changed_origin = replace(
            approved,
            candidate_rule={"predicate": "changed"},
            detector_registry_version="other-registry",
            detector_registry_digest="e" * 64,
            campaign_digest="f" * 64,
            source_identity="source-2",
            source_revision="commit-2",
            annotation_ids=("annotation-2",),
            finding_ids=("finding-2",),
            episode_ids=("episode-2",),
            rationale="changed origin",
            metadata={"changed": True},
            proposer_id="agent-2",
        )
        with pytest.raises(AuditStoreError, match="immutable fields"):
            store.save(
                changed_origin,
                operation_id="proposal-mutated",
                expected_revision=decision.revision,
                actor="human",
                actor_id="reviewer-1",
            )
        with pytest.raises(AuditStoreError, match="immutable fields"):
            store.force_save(
                changed_origin,
                operation_id="proposal-force-mutated",
                actor="human",
                actor_id="reviewer-1",
            )
        assert store.get(proposal.proposal_id).record == approved
        with pytest.raises(AuditConflictError, match="revision conflict"):
            store.save(
                replace(approved, decision_reason="stale decision"),
                operation_id="proposal-stale",
                expected_revision=created.revision,
                actor="human",
                actor_id="reviewer-1",
            )


def test_detector_rule_proposal_replay_and_projection_rebuild_preserve_origin(tmp_path) -> None:
    proposal = _proposal("proposal-rebuild")
    with AuditStore(tmp_path) as store:
        first = store.save(
            proposal,
            operation_id="proposal-replay",
            expected_revision=0,
            actor="agent",
            actor_id="agent-1",
        )
        replay = store.save(
            proposal,
            operation_id="proposal-replay",
            expected_revision=0,
            actor="agent",
            actor_id="agent-1",
        )
        assert first.revision == replay.revision == 1
        assert replay.replayed
        projection = store.projection_path
    projection.unlink()
    with AuditStore(tmp_path) as rebuilt:
        loaded = rebuilt.get(proposal.proposal_id)
        assert loaded is not None
        assert loaded.record == proposal


def test_record_id_type_is_stable_for_proposal_force_save_and_reopen(tmp_path) -> None:
    proposal = _proposal("proposal-type-stable")
    replacement = ActionRecord(
        action_id=proposal.proposal_id,
        action_type="unrelated-replacement",
        actor_kind="agent",
        actor_id="agent-1",
    )
    with AuditStore(tmp_path) as store:
        created = store.save(
            proposal,
            operation_id="proposal-type-stable-create",
            expected_revision=0,
            actor="agent",
            actor_id="agent-1",
        )
        with pytest.raises(AuditStoreError, match="record type is immutable"):
            store.force_save(
                replacement,
                operation_id="proposal-type-stable-replace",
                actor="agent",
                actor_id="agent-1",
            )
        assert store.get(proposal.proposal_id).record == proposal
        assert [
            (item.revision, item.record_type) for item in store.history(proposal.proposal_id)
        ] == [(created.revision, "detector_rule_proposal")]
        store.rebuild_projection()

    with AuditStore(tmp_path) as reopened:
        loaded = reopened.get(proposal.proposal_id)
        assert loaded is not None
        assert loaded.record == proposal
        assert [item.record_type for item in reopened.history(proposal.proposal_id)] == [
            "detector_rule_proposal"
        ]

    lines = tmp_path.joinpath("audit.ndjson").read_bytes().splitlines()
    transaction = json.loads(lines[1])
    tampered_transaction = dict(transaction)
    tampered_transaction["operation_id"] = "proposal-type-stable-tampered"
    tampered_transaction["global_revision"] = 2
    tampered_transaction["expected_revisions"] = {proposal.proposal_id: 1}
    tampered_change = dict(transaction["changes"][0])
    tampered_change["record_type"] = "action_record"
    tampered_change["revision"] = 2
    tampered_change["record"] = record_to_dict(replacement)
    tampered_transaction["changes"] = [tampered_change]
    changes = tuple(
        (
            change["record_id"],
            "tombstone" if change["deleted"] else change["record_type"],
            change["record"],
            change["deleted"],
        )
        for change in tampered_transaction["changes"]
    )
    tampered_transaction["request_digest"] = AuditStore._request_digest(
        tampered_transaction["operation_id"],
        changes,
        tampered_transaction["expected_revisions"],
        tampered_transaction["actor"],
        unconditional=tampered_transaction.get("unconditional", False),
    )
    lines.append(json.dumps(tampered_transaction, sort_keys=True).encode())
    tmp_path.joinpath("audit.ndjson").write_bytes(b"\n".join(lines) + b"\n")
    with pytest.raises(AuditCorruptionError, match="record type changed"):
        AuditStore(tmp_path)


def test_tombstone_record_type_is_stable_on_digest_valid_replay(tmp_path) -> None:
    proposal = _proposal("proposal-tombstone-type")
    with AuditStore(tmp_path) as store:
        created = store.save(
            proposal,
            operation_id="proposal-tombstone-create",
            expected_revision=0,
            actor="agent",
            actor_id="agent-1",
        )
        removed = store.delete(
            proposal.proposal_id,
            operation_id="proposal-tombstone-delete",
            expected_revision=created.revision,
        )
        assert removed.revision == 2

    lines = tmp_path.joinpath("audit.ndjson").read_bytes().splitlines()
    transaction = json.loads(lines[2])
    tampered_transaction = dict(transaction)
    tampered_transaction["operation_id"] = "proposal-tombstone-tampered"
    tampered_transaction["global_revision"] = 3
    tampered_transaction["expected_revisions"] = {proposal.proposal_id: 2}
    tampered_change = dict(transaction["changes"][0])
    tampered_change["record_type"] = "action_record"
    tampered_change["revision"] = 3
    tampered_transaction["changes"] = [tampered_change]
    changes = ((proposal.proposal_id, "tombstone", None, True),)
    tampered_transaction["request_digest"] = AuditStore._request_digest(
        tampered_transaction["operation_id"],
        changes,
        tampered_transaction["expected_revisions"],
        tampered_transaction["actor"],
        unconditional=False,
    )
    lines.append(json.dumps(tampered_transaction, sort_keys=True).encode())
    tmp_path.joinpath("audit.ndjson").write_bytes(b"\n".join(lines) + b"\n")
    with pytest.raises(AuditCorruptionError, match="record type changed"):
        AuditStore(tmp_path)


def test_first_unknown_tombstone_keeps_unknown_type(tmp_path) -> None:
    with AuditStore(tmp_path) as store:
        result = store.delete("unknown", operation_id="unknown-delete")
        assert result.revision == 1
        tombstone = store.get("unknown", include_deleted=True)
        assert tombstone is not None
        assert tombstone.deleted
        assert tombstone.record_type == "unknown"

    with AuditStore(tmp_path) as reopened:
        tombstone = reopened.get("unknown", include_deleted=True)
        assert tombstone is not None
        assert tombstone.record_type == "unknown"


def test_detector_rule_proposal_tampered_history_fails_reopen_before_rebuild(tmp_path) -> None:
    proposal = _proposal("proposal-tamper")
    with AuditStore(tmp_path) as store:
        created = store.save(
            proposal,
            operation_id="proposal-tamper-create",
            expected_revision=0,
            actor="agent",
            actor_id="agent-1",
        )
        approved = replace(
            proposal,
            lifecycle_status="approved",
            author_kind="human",
            author_id="reviewer-1",
            decided_by_kind="human",
            decided_by_id="reviewer-1",
            decided_at="2026-09-20T10:00:00Z",
            decision_reason="Human review accepted this diagnostic candidate.",
            updated_at="2026-09-20T10:00:01Z",
        )
        store.save(
            approved,
            operation_id="proposal-tamper-approve",
            expected_revision=created.revision,
            actor="human",
            actor_id="reviewer-1",
        )
        store.close()

    lines = tmp_path.joinpath("audit.ndjson").read_bytes().splitlines()
    transaction = json.loads(lines[2])
    transaction["changes"][0]["record"]["proposer_id"] = "agent-2"
    changes = tuple(
        (
            change["record_id"],
            "tombstone" if change["deleted"] else change["record_type"],
            change["record"],
            change["deleted"],
        )
        for change in transaction["changes"]
    )
    transaction["request_digest"] = AuditStore._request_digest(
        transaction["operation_id"],
        changes,
        transaction["expected_revisions"],
        transaction["actor"],
        unconditional=transaction.get("unconditional", False),
    )
    lines[2] = json.dumps(transaction, sort_keys=True).encode()
    tmp_path.joinpath("audit.ndjson").write_bytes(b"\n".join(lines) + b"\n")
    with pytest.raises(AuditCorruptionError, match="immutable fields"):
        AuditStore(tmp_path)


def test_stale_revision_conflict_and_replayed_operation_are_explicit(tmp_path) -> None:
    with AuditStore(tmp_path) as store:
        original = _annotation("a")
        first = store.save(original, operation_id="op-a", expected_revision=0)
        replay = store.save(original, operation_id="op-a", expected_revision=0)
        assert replay.replayed
        with pytest.raises(AuditConflictError):
            store.save(_annotation("a", text="new"), operation_id="op-b", expected_revision=0)
        with pytest.raises(OperationConflictError):
            store.save(
                _annotation("a", text="different"),
                operation_id="op-a",
                expected_revision=first.revision,
            )


def test_existing_record_write_requires_cas_or_explicit_force_operation(tmp_path) -> None:
    with AuditStore(tmp_path) as store:
        first = store.save(
            _annotation("a", text="before"), operation_id="op-a", expected_revision=0
        )
        with pytest.raises(AuditConflictError, match="expected_revision"):
            store.save(_annotation("a", text="silent overwrite"), operation_id="op-b")
        forced = store.force_save(
            _annotation("a", text="explicit overwrite"), operation_id="op-force"
        )
        assert forced.revision == first.revision + 1
        assert store.get("a").observed_behavior == "explicit overwrite"
        with pytest.raises(OperationConflictError):
            store.save(_annotation("a", text="explicit overwrite"), operation_id="op-force")


@pytest.mark.parametrize("rid", ["", None, 123])
def test_delete_rejects_invalid_record_ids_before_changing_journal(tmp_path, rid) -> None:
    with AuditStore(tmp_path) as store:
        before = store.canonical_path.read_bytes()
        with pytest.raises(AuditStoreError):
            store.delete(rid, operation_id="delete-invalid")
        assert store.canonical_path.read_bytes() == before


@pytest.mark.parametrize("operation_id", ["", None, 123])
def test_delete_rejects_invalid_operation_ids_before_changing_journal(
    tmp_path, operation_id
) -> None:
    with AuditStore(tmp_path) as store:
        before = store.canonical_path.read_bytes()
        with pytest.raises(AuditStoreError):
            store.delete("record", operation_id=operation_id)
        assert store.canonical_path.read_bytes() == before


def test_transaction_actor_must_match_annotation_and_review_author(tmp_path) -> None:
    episode_id = _episode().episode_id
    agent_annotation = Annotation(
        annotation_id="agent-annotation",
        episode_id=episode_id,
        classification="unclear",
        author_kind="agent",
    )
    agent_review = ReviewRecord(
        review_id="agent-review",
        episode_id=episode_id,
        scope="interval",
        author_kind="agent",
    )
    with AuditStore(tmp_path) as store:
        before = store.canonical_path.read_bytes()
        with pytest.raises(AuditStoreError, match="actor"):
            store.save(agent_annotation, operation_id="op-human-annotation", expected_revision=0)
        with pytest.raises(AuditStoreError, match="actor"):
            store.save(agent_review, operation_id="op-human-review", expected_revision=0)
        assert store.canonical_path.read_bytes() == before
        assert (
            store.save(
                agent_annotation,
                operation_id="op-agent-annotation",
                expected_revision=0,
                actor="agent",
            ).revision
            == 1
        )
        assert (
            store.save(
                agent_review,
                operation_id="op-agent-review",
                expected_revision=0,
                actor="agent",
            ).revision
            == 1
        )


def test_transaction_actor_must_match_action_actor_on_save_and_replay(tmp_path) -> None:
    action = ActionRecord(
        action_id="agent-action",
        action_type="review",
        actor_kind="agent",
    )
    with AuditStore(tmp_path) as store:
        before = store.canonical_path.read_bytes()
        with pytest.raises(AuditStoreError, match="actor"):
            store.save(action, operation_id="op-human-action", expected_revision=0)
        assert store.canonical_path.read_bytes() == before
        store.save(action, operation_id="op-agent-action", expected_revision=0, actor="agent")
        store.close()

    lines = tmp_path.joinpath("audit.ndjson").read_bytes().splitlines()
    transaction = json.loads(lines[1])
    transaction["actor"]["kind"] = "human"
    changes = tuple(
        (
            change["record_id"],
            "tombstone" if change["deleted"] else change["record_type"],
            change["record"],
            change["deleted"],
        )
        for change in transaction["changes"]
    )
    transaction["request_digest"] = AuditStore._request_digest(
        transaction["operation_id"],
        changes,
        transaction["expected_revisions"],
        transaction["actor"],
        unconditional=transaction.get("unconditional", False),
    )
    lines[1] = json.dumps(transaction, sort_keys=True).encode()
    tmp_path.joinpath("audit.ndjson").write_bytes(b"\n".join(lines) + b"\n")
    with pytest.raises(AuditCorruptionError, match="actor kind"):
        AuditStore(tmp_path)


def test_invalid_measured_payload_never_reaches_journal(tmp_path) -> None:
    with AuditStore(tmp_path) as store:
        before = store.canonical_path.read_bytes()
        with pytest.raises(AuditContractError):
            Signal(signal_id="invalid-measured", detector_id="telemetry", measured=[])
        assert store.canonical_path.read_bytes() == before


def test_export_requires_overwrite_and_rejects_overlapping_destinations(tmp_path) -> None:
    root = tmp_path / "store"
    with AuditStore(root) as store:
        store.save(_annotation("export"), operation_id="op-export", expected_revision=0)
        for destination in (root, root / "nested" / "backup.zip", tmp_path):
            before = store.canonical_path.read_bytes()
            with pytest.raises(AuditExportError, match="destination"):
                store.export(destination, overwrite=True)
            assert store.canonical_path.read_bytes() == before


def test_export_overwrite_stages_directory_output(tmp_path) -> None:
    root = tmp_path / "store"
    destination = tmp_path / "backup"
    with AuditStore(root) as store:
        store.save(_annotation("export-dir"), operation_id="op-export-dir", expected_revision=0)
        store.export(destination)
        (destination / "stale.txt").write_text("stale", encoding="utf-8")
        store.export(destination, overwrite=True)
    assert not (destination / "stale.txt").exists()
    assert (destination / "audit.ndjson").exists()


def test_tampered_digest_valid_backup_is_rejected_before_overwrite(tmp_path) -> None:
    source_root = tmp_path / "source"
    destination = tmp_path / "destination"
    with AuditStore(source_root) as source:
        source.save(_annotation("new"), operation_id="op-new", expected_revision=0)
        source.save(_annotation("second"), operation_id="op-second", expected_revision=0)
        backup = source.export(tmp_path / "backup.zip")
    with AuditStore(destination) as old:
        old.save(_annotation("old"), operation_id="op-old", expected_revision=0)

    with zipfile.ZipFile(backup) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        lines = archive.read("audit.ndjson").splitlines()
    transaction = json.loads(lines[1])
    transaction["request_digest"] = "0" * 64
    lines[1] = json.dumps(transaction, sort_keys=True).encode()
    tampered_journal = b"\n".join(lines) + b"\n"
    manifest["journal_digest"] = hashlib.sha256(tampered_journal).hexdigest()
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True))
        archive.writestr("audit.ndjson", tampered_journal)

    with pytest.raises(AuditStoreError, match="digest"):
        AuditStore.restore(tampered, destination, overwrite=True)
    with AuditStore(destination) as preserved:
        assert preserved.get("old") is not None
        assert preserved.get("new") is None


def test_restore_rejects_digest_valid_sensitive_backup_before_overwrite(tmp_path) -> None:
    source_root = tmp_path / "source"
    destination = tmp_path / "destination"
    with AuditStore(source_root) as source:
        source.save(_annotation("new"), operation_id="op-new", expected_revision=0)
        backup = source.export(tmp_path / "backup.zip")
    with AuditStore(destination) as old:
        old.save(_annotation("old"), operation_id="op-old", expected_revision=0)

    with zipfile.ZipFile(backup) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        lines = archive.read("audit.ndjson").splitlines()
    transaction = json.loads(lines[1])
    transaction["changes"][0]["record"]["metadata"] = {"password": "secret"}
    changes = tuple(
        (
            change["record_id"],
            "tombstone" if change["deleted"] else change["record_type"],
            change["record"],
            change["deleted"],
        )
        for change in transaction["changes"]
    )
    transaction["request_digest"] = AuditStore._request_digest(
        transaction["operation_id"],
        changes,
        transaction["expected_revisions"],
        transaction["actor"],
        unconditional=transaction.get("unconditional", False),
    )
    lines[1] = json.dumps(transaction, sort_keys=True).encode()
    tampered_journal = b"\n".join(lines) + b"\n"
    manifest["journal_digest"] = hashlib.sha256(tampered_journal).hexdigest()
    tampered = tmp_path / "sensitive.zip"
    with zipfile.ZipFile(tampered, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True))
        archive.writestr("audit.ndjson", tampered_journal)

    with pytest.raises(AuditExportError, match="sensitive"):
        AuditStore.restore(tampered, destination, overwrite=True)
    with AuditStore(destination) as preserved:
        assert preserved.get("old") is not None
        assert preserved.get("new") is None


def test_restore_requires_explicit_overwrite_for_empty_existing_destination(tmp_path) -> None:
    source_root = tmp_path / "source"
    destination = tmp_path / "destination"
    with AuditStore(source_root) as source:
        source.save(_annotation("new"), operation_id="op-new", expected_revision=0)
        backup = source.export(tmp_path / "backup.zip")
    destination.mkdir()
    with pytest.raises(AuditExportError, match="overwrite"):
        AuditStore.restore(backup, destination)
    assert not (destination / "audit.ndjson").exists()


def test_tombstone_history_and_undo_keep_all_revisions(tmp_path) -> None:
    with AuditStore(tmp_path) as store:
        first = store.save(
            _annotation("a", text="before"), operation_id="op-a", expected_revision=0
        )
        removed = store.delete("a", operation_id="op-delete", expected_revision=first.revision)
        assert store.get("a") is None
        tombstone = store.get("a", include_deleted=True)
        assert tombstone is not None and tombstone.deleted
        restored = store.undo("a", operation_id="op-undo", expected_revision=removed.revision)
        assert restored.revision == 3
        assert store.get("a").observed_behavior == "before"
        assert [item.revision for item in store.history("a")] == [1, 2, 3]


def test_projection_failure_after_journal_commit_recovers_on_reopen(tmp_path) -> None:
    with pytest.raises(ProjectionError):
        AuditStore(tmp_path, fail_after_journal_once=True).save(
            _annotation("a"), operation_id="op-a", expected_revision=0
        )
    with AuditStore(tmp_path) as recovered:
        loaded = recovered.get("a")
        assert loaded is not None
        assert recovered.checkpoint().revision == 1


def test_two_clients_serialize_writes_and_reject_stale_cas(tmp_path) -> None:
    stores = [AuditStore(tmp_path), AuditStore(tmp_path)]
    errors: list[Exception] = []

    def write(store: AuditStore, operation_id: str) -> None:
        try:
            store.save(
                _annotation("a", text=operation_id), operation_id=operation_id, expected_revision=0
            )
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=write, args=(store, f"op-{index}"))
        for index, store in enumerate(stores)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for store in stores:
        store.close()
    assert len(errors) == 1
    assert isinstance(errors[0], AuditConflictError)
    with AuditStore(tmp_path) as store:
        assert store.get_revision("a") == 1


def test_deleted_projection_rebuild_and_portable_restore(tmp_path) -> None:
    source_root = tmp_path / "source"
    with AuditStore(source_root) as store:
        store.save(_annotation("a"), operation_id="op-a", expected_revision=0)
        backup = store.export(tmp_path / "backup.zip")
        store.projection_path.unlink()
        assert store.rebuild_projection().revision == 1
    restored_root = tmp_path / "relocated"
    with AuditStore.restore(backup, restored_root) as restored:
        loaded = restored.get("a")
        assert loaded is not None
        assert loaded.record.annotation_id == "a"


def test_relocated_store_rebuilds_projection_from_canonical_journal(tmp_path) -> None:
    original = tmp_path / "original"
    with AuditStore(original) as store:
        store.save(_annotation("relocated"), operation_id="op-relocate", expected_revision=0)
    moved = tmp_path / "moved"
    shutil.move(original, moved)
    with AuditStore(moved) as relocated:
        assert relocated.get("relocated") is not None
        assert relocated.checkpoint().revision == 1


def test_overwrite_restore_discards_corrupt_projection(tmp_path) -> None:
    source_root = tmp_path / "source"
    destination = tmp_path / "destination"
    with AuditStore(source_root) as source:
        source.save(_annotation("a"), operation_id="op-a", expected_revision=0)
        backup = source.export(tmp_path / "backup.zip", include_projection=True)
    destination.mkdir()
    (destination / "audit.sqlite3").write_bytes(b"not sqlite")
    with AuditStore.restore(backup, destination, overwrite=True) as restored:
        assert restored.get("a") is not None


def test_overwrite_restore_preserves_old_store_on_digest_valid_malformed_journal(tmp_path) -> None:
    source_root = tmp_path / "source"
    destination = tmp_path / "destination"
    with AuditStore(source_root) as source:
        source.save(_annotation("new"), operation_id="op-new", expected_revision=0)
        backup = source.export(tmp_path / "backup.zip")
    with AuditStore(destination) as old:
        old.save(_annotation("old"), operation_id="op-old", expected_revision=0)

    with zipfile.ZipFile(backup) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        journal = archive.read("audit.ndjson")
    malformed = journal + b"{not-json}"
    manifest["journal_digest"] = hashlib.sha256(malformed).hexdigest()
    bad_backup = tmp_path / "malformed.zip"
    with zipfile.ZipFile(bad_backup, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, sort_keys=True))
        archive.writestr("audit.ndjson", malformed)

    with pytest.raises(AuditStoreError):
        AuditStore.restore(bad_backup, destination, overwrite=True)
    with AuditStore(destination) as preserved:
        assert preserved.get("old") is not None
        assert preserved.get("new") is None
