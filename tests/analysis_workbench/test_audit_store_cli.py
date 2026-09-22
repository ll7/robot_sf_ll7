"""Offline audit-store command-line contract tests."""

from __future__ import annotations

import json

from robot_sf.analysis_workbench.audit_contracts import Annotation, EpisodeRef
from robot_sf.analysis_workbench.audit_store import AuditStore, main


def _annotation(annotation_id: str) -> Annotation:
    episode = EpisodeRef(
        campaign_digest="a" * 64,
        source_digest="b" * 64,
        execution_id="cli-exec",
    )
    return Annotation(
        annotation_id=annotation_id,
        episode_id=episode.episode_id,
        classification="unclear",
    )


def test_cli_inspect_rebuild_export_restore_are_machine_readable(tmp_path, capsys) -> None:
    root = tmp_path / "store"
    with AuditStore(root) as store:
        store.save(_annotation("cli-a"), operation_id="op-cli", expected_revision=0)

    assert main(["inspect", str(root)]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["record_count"] == 1

    assert main(["rebuild", str(root)]) == 0
    rebuilt = json.loads(capsys.readouterr().out)
    assert rebuilt["revision"] == 1

    backup = tmp_path / "cli-backup.zip"
    assert main(["export", str(root), str(backup)]) == 0
    assert json.loads(capsys.readouterr().out)["backup"] == str(backup)
    assert main(["export", str(root), str(backup)]) == 2
    assert "exists" in json.loads(capsys.readouterr().err)["error"]

    restored = tmp_path / "restored"
    assert main(["restore", str(backup), str(restored)]) == 0
    assert json.loads(capsys.readouterr().out)["root"] == str(restored)


def test_cli_restore_requires_explicit_overwrite(tmp_path, capsys) -> None:
    root = tmp_path / "store"
    with AuditStore(root) as store:
        store.save(_annotation("cli-b"), operation_id="op-cli-b", expected_revision=0)
        backup = store.export(tmp_path / "backup.zip")
    destination = tmp_path / "destination"
    destination.mkdir()
    assert main(["restore", str(backup), str(destination)]) == 2
    error = json.loads(capsys.readouterr().err)
    assert "overwrite" in error["error"]
