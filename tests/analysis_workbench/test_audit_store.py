"""Executable BA-03 persistence contract examples."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import zipfile

import pytest

from robot_sf.analysis_workbench.audit_contracts import Annotation, EpisodeRef, TimeInterval
from robot_sf.analysis_workbench.audit_store import (
    AuditConflictError,
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
