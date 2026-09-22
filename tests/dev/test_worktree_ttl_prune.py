"""Focused tests for the recurring worktree TTL prune helper (issue #9257)."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from scripts.dev import worktree_ttl_prune as ttl

NOW = 1_800_000_000.0
DAY = 86_400.0


def _row(path: str, *, decision: str, **extra: object) -> dict[str, object]:
    """Build one hygiene row with the fields the selector consumes."""
    row: dict[str, object] = {
        "path": path,
        "branch": f"branch-{Path(path).name}",
        "head_sha": "0" * 40,
        "decision": decision,
        "active_claims": [],
        "ignored_artifacts": [],
        "tracked_durable_paths": [],
        "reasons": [],
    }
    row.update(extra)
    return row


def test_selection_keeps_only_old_clean_removeable_worktrees() -> None:
    """Eligibility requires removeable, clean, unclaimed, and beyond the TTL."""
    old = "/fleet/old-merged"
    young = "/fleet/young-merged"
    dirty = "/fleet/dirty"
    claimed = "/fleet/claimed"
    artifacts = "/fleet/with-artifacts"
    unknown = "/fleet/unreadable"
    rows = [
        _row(old, decision="removeable"),
        _row(young, decision="removeable"),
        _row(dirty, decision="preserve", reasons=["dirty"]),
        _row(claimed, decision="removeable", active_claims=["lease-1"]),
        _row(artifacts, decision="removeable", ignored_artifacts=["output/x"]),
        _row(unknown, decision="removeable"),
    ]
    epochs = {
        old: NOW - 30 * DAY,
        young: NOW - 1 * DAY,
        claimed: NOW - 30 * DAY,
        artifacts: NOW - 30 * DAY,
        unknown: None,
    }

    eligible, skipped = ttl.select_ttl_candidates(
        rows, ttl_days=7, now=NOW, activity_lookup=epochs.get
    )

    assert [entry["path"] for entry in eligible] == [old]
    assert eligible[0]["age_days"] == pytest.approx(30.0, abs=0.1)
    assert [entry["skip_reason"] for entry in skipped] == [
        "within_ttl",
        "hygiene_decision:preserve",
        "active_claims",
        "preservation_evidence",
        "activity_unavailable",
    ]
    assert skipped[0]["age_days"] == pytest.approx(1.0, abs=0.1)


def test_selection_is_fail_closed_on_malformed_rows() -> None:
    """A row that is not a mapping is reported, never selected."""
    eligible, skipped = ttl.select_ttl_candidates(
        ["not-a-row"],  # type: ignore[list-item]
        ttl_days=7,
        now=NOW,
        activity_lookup=lambda path: NOW - 30 * DAY,
    )

    assert eligible == []
    assert skipped == [{"path": "", "skip_reason": "malformed_row"}]


def test_activity_epoch_prefers_the_newer_worktree_metadata(tmp_path: Path) -> None:
    """The admin HEAD timestamp wins when it is newer than the directory."""
    worktree = tmp_path / "wt"
    worktree.mkdir()
    old = NOW - 40 * DAY
    os.utime(worktree, (old, old))
    git_dir = tmp_path / "admin"
    git_dir.mkdir()
    head_file = git_dir / "HEAD"
    head_file.write_text("ref: refs/heads/x\n", encoding="utf-8")
    newer = NOW - 2 * DAY
    os.utime(head_file, (newer, newer))

    def fake_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["git"], 0, stdout=f"{git_dir}\n")

    assert ttl.worktree_activity_epoch(str(worktree), run=fake_run) == pytest.approx(newer, abs=1.0)


def test_activity_epoch_is_none_for_missing_paths(tmp_path: Path) -> None:
    """An unreadable worktree path cannot be aged and is refused."""
    assert ttl.worktree_activity_epoch(str(tmp_path / "missing")) is None


def _write_hygiene(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    payload = {"schema": "worktree_hygiene_snapshot.v1", "worktrees": rows}
    path = tmp_path / "hygiene.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _age(path: Path, *, days: float) -> None:
    stamp = time.time() - days * DAY
    os.utime(path, (stamp, stamp))


def test_cli_dry_run_lists_exactly_the_eligible_set(tmp_path: Path, capsys, monkeypatch) -> None:
    """Dry-run selects the old clean worktree and never invokes the reaper."""
    stale = tmp_path / "stale"
    fresh = tmp_path / "fresh"
    stale.mkdir()
    fresh.mkdir()
    _age(stale, days=30)
    _age(fresh, days=1)
    hygiene = _write_hygiene(
        tmp_path,
        [
            _row(str(stale), decision="removeable"),
            _row(str(fresh), decision="removeable"),
            _row(str(tmp_path / "dirty"), decision="preserve", reasons=["dirty"]),
        ],
    )
    calls: list[str] = []
    monkeypatch.setattr(
        ttl, "_run_reaper", lambda path, *, repo_root: calls.append(path) or {"path": path}
    )

    exit_code = ttl.main(["--hygiene-json", str(hygiene), "--ttl-days", "7", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["mode"] == "dry_run"
    assert [entry["path"] for entry in payload["eligible"]] == [str(stale)]
    assert payload["reaper_results"] == []
    assert calls == []


def test_cli_apply_delegates_once_per_candidate(tmp_path: Path, capsys, monkeypatch) -> None:
    """Apply mode calls the reaper exactly once for each selected path."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _age(first, days=10)
    _age(second, days=20)
    hygiene = _write_hygiene(
        tmp_path,
        [_row(str(first), decision="removeable"), _row(str(second), decision="removeable")],
    )
    calls: list[str] = []

    def fake_reaper(path: str, *, repo_root: Path) -> dict[str, object]:
        calls.append(path)
        return {"path": path, "status": "applied", "returncode": 0}

    monkeypatch.setattr(ttl, "_run_reaper", fake_reaper)

    exit_code = ttl.main(["--hygiene-json", str(hygiene), "--apply", "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert payload["mode"] == "apply"
    assert sorted(calls) == sorted([str(first), str(second)])
    assert [item["status"] for item in payload["reaper_results"]] == ["applied", "applied"]


def test_cli_rejects_non_positive_ttl(tmp_path: Path, capsys) -> None:
    """A non-positive TTL is a usage error, not a silent full prune."""
    assert ttl.main(["--ttl-days", "0"]) == 2
    assert "--ttl-days must be positive" in capsys.readouterr().err


def test_cli_reports_unreadable_hygiene_snapshot(tmp_path: Path, capsys) -> None:
    """An unreadable fleet snapshot fails closed with a structured error."""
    exit_code = ttl.main(["--hygiene-json", str(tmp_path / "missing.json"), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["status"] == "error"
    assert "could not read hygiene snapshot" in payload["error"]
