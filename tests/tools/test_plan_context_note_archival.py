"""Tests for the context-note archival-sweep planner."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import yaml

from scripts.tools import plan_context_note_archival as planner

if TYPE_CHECKING:
    from pathlib import Path


def _run(repo: Path, *args: str, env: dict[str, str] | None = None) -> None:
    subprocess.run(list(args), cwd=repo, env=env, check=True, capture_output=True, text=True)


def _write(repo: Path, path: str, text: str) -> None:
    full_path = repo / path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    full_path.write_text(text, encoding="utf-8")


def _commit(repo: Path, message: str, *, iso_date: str) -> None:
    env = {
        "GIT_AUTHOR_DATE": iso_date,
        "GIT_COMMITTER_DATE": iso_date,
        "GIT_AUTHOR_NAME": "Test Author",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test Author",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    _run(repo, "git", "add", ".")
    _run(repo, "git", "commit", "-m", message, env=env)


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _run(repo, "git", "init")
    _run(repo, "git", "config", "user.name", "Test Author")
    _run(repo, "git", "config", "user.email", "test@example.com")
    _write(repo, "docs/context/INDEX.md", "# Index\n")
    _write(repo, "docs/context/README.md", "# Context\n")
    _commit(repo, "base", iso_date="2026-01-01T00:00:00+00:00")
    return repo


def _write_catalog(repo: Path, entries: list[dict[str, object]]) -> None:
    base_entries = [
        {"path": "docs/context/README.md", "status": "current", "freshness": "maintained"},
        {"path": "docs/context/INDEX.md", "status": "current", "freshness": "maintained"},
    ]
    _write(
        repo,
        "docs/context/catalog.yaml",
        yaml.safe_dump({"version": 1, "entries": [*base_entries, *entries]}, sort_keys=False),
    )


def test_superseded_with_existing_replacement_is_high_confidence_candidate(tmp_path: Path) -> None:
    """Superseded notes whose replacement exists are high-confidence archive moves."""

    repo = _repo(tmp_path)
    _write(repo, "docs/context/old.md", "# Old\n")
    _write(repo, "docs/context/new.md", "# New\n")
    _write_catalog(
        repo,
        [
            {
                "path": "docs/context/old.md",
                "status": "superseded",
                "freshness": "dated",
                "replacement": "docs/context/new.md",
            }
        ],
    )
    _commit(repo, "superseded with replacement", iso_date="2026-01-02T00:00:00+00:00")

    moves, conflicts = planner.plan_archival(repo_root=repo)

    assert conflicts == []
    assert len(moves) == 1
    move = moves[0]
    assert move.category == "superseded"
    assert move.confidence == "high"
    assert move.source == "docs/context/old.md"
    assert move.target == "docs/context/archive/old.md"
    assert move.replacement == "docs/context/new.md"


def test_superseded_without_valid_replacement_is_not_planned(tmp_path: Path) -> None:
    """Integrity violations stay with the checker (Rule A); the planner skips them."""

    repo = _repo(tmp_path)
    _write(repo, "docs/context/old.md", "# Old\n")
    _write_catalog(
        repo,
        [
            # Missing replacement -> checker error, not an archive candidate.
            {"path": "docs/context/old.md", "status": "superseded", "freshness": "dated"},
            # Replacement points at a nonexistent file -> not safe to archive.
            {
                "path": "docs/context/old.md",
                "status": "superseded",
                "freshness": "dated",
                "replacement": "docs/context/missing.md",
            },
        ],
    )
    _commit(repo, "superseded without valid replacement", iso_date="2026-01-02T00:00:00+00:00")

    moves, conflicts = planner.plan_archival(repo_root=repo)

    assert moves == []
    assert conflicts == []


def test_stale_current_dated_note_is_review_candidate(tmp_path: Path) -> None:
    """Stale dated notes surface as lower-confidence review candidates."""

    repo = _repo(tmp_path)
    _write(repo, "docs/context/dated.md", "# Dated\n")
    _write_catalog(
        repo,
        [{"path": "docs/context/dated.md", "status": "current", "freshness": "dated"}],
    )
    _commit(repo, "dated note", iso_date="2025-01-01T00:00:00+00:00")

    moves, conflicts = planner.plan_archival(
        repo_root=repo, max_age_days=180, now=datetime(2026, 1, 1, tzinfo=UTC)
    )

    assert conflicts == []
    assert len(moves) == 1
    move = moves[0]
    assert move.category == "stale"
    assert move.confidence == "review"
    assert move.source == "docs/context/dated.md"
    assert move.target == "docs/context/archive/dated.md"
    assert move.age_days == 365


def test_no_stale_flag_excludes_review_candidates(tmp_path: Path) -> None:
    """include_stale=False plans only high-confidence superseded candidates."""

    repo = _repo(tmp_path)
    _write(repo, "docs/context/dated.md", "# Dated\n")
    _write_catalog(
        repo,
        [{"path": "docs/context/dated.md", "status": "current", "freshness": "dated"}],
    )
    _commit(repo, "dated note", iso_date="2025-01-01T00:00:00+00:00")

    moves, conflicts = planner.plan_archival(
        repo_root=repo,
        max_age_days=180,
        include_stale=False,
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert moves == []
    assert conflicts == []


def test_duplicate_archive_target_is_conflict(tmp_path: Path) -> None:
    """Two notes with the same basename collide at one archive target."""

    repo = _repo(tmp_path)
    _write(repo, "docs/context/old.md", "# Old A\n")
    _write(repo, "docs/context/sub/old.md", "# Old B\n")
    _write(repo, "docs/context/new.md", "# New\n")
    _write_catalog(
        repo,
        [
            {
                "path": "docs/context/old.md",
                "status": "superseded",
                "freshness": "dated",
                "replacement": "docs/context/new.md",
            },
            {
                "path": "docs/context/sub/old.md",
                "status": "superseded",
                "freshness": "dated",
                "replacement": "docs/context/new.md",
            },
        ],
    )
    _commit(repo, "two old notes same basename", iso_date="2026-01-02T00:00:00+00:00")

    moves, conflicts = planner.plan_archival(repo_root=repo)

    assert len(moves) == 2
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict.kind == "duplicate_target"
    assert conflict.target == "docs/context/archive/old.md"
    assert set(conflict.sources) == {"docs/context/old.md", "docs/context/sub/old.md"}

    summary = planner._summary(moves, conflicts, archive_dir=planner.checker.ARCHIVE_DIR)
    assert summary["exit_code"] == 1


def test_existing_archive_target_is_conflict(tmp_path: Path) -> None:
    """An archive destination that already exists blocks an unconditional move."""

    repo = _repo(tmp_path)
    _write(repo, "docs/context/old.md", "# Old\n")
    _write(repo, "docs/context/new.md", "# New\n")
    _write(repo, "docs/context/archive/old.md", "# Pre-existing archive\n")
    _write_catalog(
        repo,
        [
            {
                "path": "docs/context/old.md",
                "status": "superseded",
                "freshness": "dated",
                "replacement": "docs/context/new.md",
            }
        ],
    )
    _commit(repo, "pre-existing archive target", iso_date="2026-01-02T00:00:00+00:00")

    moves, conflicts = planner.plan_archival(repo_root=repo)

    assert len(moves) == 1
    assert [c.kind for c in conflicts] == ["target_exists"]
    assert conflicts[0].target == "docs/context/archive/old.md"


def test_already_archived_note_is_not_replanned(tmp_path: Path) -> None:
    """Notes already under the archive dir are not re-proposed for moving."""

    repo = _repo(tmp_path)
    _write(repo, "docs/context/archive/old.md", "# Already archived\n")
    _write(repo, "docs/context/new.md", "# New\n")
    _write_catalog(
        repo,
        [
            {
                "path": "docs/context/archive/old.md",
                "status": "superseded",
                "freshness": "dated",
                "replacement": "docs/context/new.md",
            }
        ],
    )
    _commit(repo, "already archived note", iso_date="2026-01-02T00:00:00+00:00")

    moves, conflicts = planner.plan_archival(repo_root=repo)

    assert moves == []
    assert conflicts == []


def test_summary_reports_counts_and_plan_only_note(tmp_path: Path) -> None:
    """The JSON summary carries category counts, a clean exit, and the plan-only note."""

    repo = _repo(tmp_path)
    _write(repo, "docs/context/old.md", "# Old\n")
    _write(repo, "docs/context/new.md", "# New\n")
    _write_catalog(
        repo,
        [
            {
                "path": "docs/context/old.md",
                "status": "superseded",
                "freshness": "dated",
                "replacement": "docs/context/new.md",
            }
        ],
    )
    _commit(repo, "superseded", iso_date="2026-01-02T00:00:00+00:00")

    moves, conflicts = planner.plan_archival(repo_root=repo)
    summary = planner._summary(moves, conflicts, archive_dir=planner.checker.ARCHIVE_DIR)

    assert summary["schema_version"] == "context_note_archival_plan.v1"
    assert summary["counts"] == {"superseded": 1}
    assert summary["exit_code"] == 0
    assert summary["archive_dir"] == "docs/context/archive"
    assert "plan-only" in summary["note"]


def test_valid_approval_manifest_accepts_current_superseded_candidate(tmp_path: Path) -> None:
    """Approved moves are valid only when they match current generated candidates."""

    repo = _repo(tmp_path)
    _write(repo, "docs/context/old.md", "# Old\n")
    _write(repo, "docs/context/new.md", "# New\n")
    _write_catalog(
        repo,
        [
            {
                "path": "docs/context/old.md",
                "status": "superseded",
                "freshness": "dated",
                "replacement": "docs/context/new.md",
            }
        ],
    )
    _commit(repo, "superseded", iso_date="2026-01-02T00:00:00+00:00")
    manifest = repo / "approved_moves.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "context_archive_approved_moves.v1",
                "issue": 3190,
                "moves": [
                    {
                        "source": "docs/context/old.md",
                        "target": "docs/context/archive/old.md",
                        "category": "superseded",
                        "replacement": "docs/context/new.md",
                        "reason": "Maintainer approved superseded-note archive move.",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    approved_moves, findings, conflicts = planner.validate_approval_manifest(
        manifest,
        repo_root=repo,
    )

    assert len(approved_moves) == 1
    assert findings == []
    assert conflicts == []


def test_approval_manifest_fails_when_replacement_drifted_from_plan(tmp_path: Path) -> None:
    """A manifest cannot silently approve stale replacement metadata."""

    repo = _repo(tmp_path)
    _write(repo, "docs/context/old.md", "# Old\n")
    _write(repo, "docs/context/new.md", "# New\n")
    _write(repo, "docs/context/wrong.md", "# Wrong\n")
    _write_catalog(
        repo,
        [
            {
                "path": "docs/context/old.md",
                "status": "superseded",
                "freshness": "dated",
                "replacement": "docs/context/new.md",
            }
        ],
    )
    _commit(repo, "superseded", iso_date="2026-01-02T00:00:00+00:00")
    manifest = repo / "approved_moves.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "context_archive_approved_moves.v1",
                "issue": 3190,
                "moves": [
                    {
                        "source": "docs/context/old.md",
                        "target": "docs/context/archive/old.md",
                        "category": "superseded",
                        "replacement": "docs/context/wrong.md",
                        "reason": "Approved before catalog replacement changed.",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    _approved_moves, findings, _conflicts = planner.validate_approval_manifest(
        manifest,
        repo_root=repo,
    )

    assert [finding.rule for finding in findings] == ["replacement_mismatch"]


def test_approval_manifest_fails_for_move_not_in_current_plan(tmp_path: Path) -> None:
    """Approval manifests are checked against current candidates, not trusted by shape alone."""

    repo = _repo(tmp_path)
    _write(repo, "docs/context/current.md", "# Current\n")
    _write_catalog(
        repo,
        [{"path": "docs/context/current.md", "status": "current", "freshness": "maintained"}],
    )
    _commit(repo, "current note", iso_date="2026-01-02T00:00:00+00:00")
    manifest = repo / "approved_moves.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "context_archive_approved_moves.v1",
                "issue": 3190,
                "moves": [
                    {
                        "source": "docs/context/current.md",
                        "target": "docs/context/archive/current.md",
                        "category": "superseded",
                        "reason": "Invalid approval for a non-candidate.",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    _approved_moves, findings, _conflicts = planner.validate_approval_manifest(
        manifest,
        repo_root=repo,
    )

    assert [finding.rule for finding in findings] == ["not_in_current_plan"]


def test_approval_manifest_reports_duplicate_target_conflict(tmp_path: Path) -> None:
    """A reviewed subset still fails closed when approved targets collide."""

    repo = _repo(tmp_path)
    _write(repo, "docs/context/old.md", "# Old A\n")
    _write(repo, "docs/context/sub/old.md", "# Old B\n")
    _write(repo, "docs/context/new.md", "# New\n")
    _write_catalog(
        repo,
        [
            {
                "path": "docs/context/old.md",
                "status": "superseded",
                "freshness": "dated",
                "replacement": "docs/context/new.md",
            },
            {
                "path": "docs/context/sub/old.md",
                "status": "superseded",
                "freshness": "dated",
                "replacement": "docs/context/new.md",
            },
        ],
    )
    _commit(repo, "superseded collision", iso_date="2026-01-02T00:00:00+00:00")
    manifest = repo / "approved_moves.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "context_archive_approved_moves.v1",
                "issue": 3190,
                "moves": [
                    {
                        "source": "docs/context/old.md",
                        "target": "docs/context/archive/old.md",
                        "category": "superseded",
                        "replacement": "docs/context/new.md",
                        "reason": "First approved move.",
                    },
                    {
                        "source": "docs/context/sub/old.md",
                        "target": "docs/context/archive/old.md",
                        "category": "superseded",
                        "replacement": "docs/context/new.md",
                        "reason": "Second approved move.",
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    _approved_moves, findings, conflicts = planner.validate_approval_manifest(
        manifest,
        repo_root=repo,
    )

    assert [conflict.kind for conflict in conflicts] == ["duplicate_target"]
    assert "duplicate_target" in {finding.rule for finding in findings}
    assert "conflict_duplicate_target" in {finding.rule for finding in findings}


def test_cli_approval_manifest_json_reports_failure(tmp_path: Path) -> None:
    """The CLI exposes approval validation as a fail-closed JSON gate."""

    repo = _repo(tmp_path)
    _write(repo, "docs/context/current.md", "# Current\n")
    _write_catalog(
        repo,
        [{"path": "docs/context/current.md", "status": "current", "freshness": "maintained"}],
    )
    _commit(repo, "current note", iso_date="2026-01-02T00:00:00+00:00")
    manifest = repo / "approved_moves.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "schema_version": "context_archive_approved_moves.v1",
                "issue": 3190,
                "moves": [
                    {
                        "source": "docs/context/current.md",
                        "target": "docs/context/archive/current.md",
                        "category": "superseded",
                        "reason": "Invalid approval for a non-candidate.",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "python",
            "-m",
            "scripts.tools.plan_context_note_archival",
            "--approval-manifest",
            "approved_moves.yaml",
            "--json-output",
            "-",
        ],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert result.returncode == 1
    assert payload["schema_version"] == "context_archive_approved_moves.v1"
    assert payload["finding_count"] == 1
    assert payload["findings"][0]["rule"] == "not_in_current_plan"
