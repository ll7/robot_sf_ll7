"""Tests for check_context_note_freshness_decay.py."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import yaml

from scripts.tools import check_context_note_freshness_decay as checker

if TYPE_CHECKING:
    from pathlib import Path


def _run(repo: Path, *args: str, env: dict[str, str] | None = None) -> None:
    subprocess.run(
        list(args),
        cwd=repo,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


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
    _write(
        repo,
        "docs/context/catalog.yaml",
        yaml.safe_dump(
            {
                "version": 1,
                "entries": [
                    {
                        "path": "docs/context/README.md",
                        "status": "current",
                        "freshness": "maintained",
                    },
                    {
                        "path": "docs/context/INDEX.md",
                        "status": "current",
                        "freshness": "maintained",
                    },
                ],
            },
            sort_keys=False,
        ),
    )
    _commit(repo, "base", iso_date="2026-01-01T00:00:00+00:00")
    return repo


def _write_catalog(repo: Path, entries: list[dict[str, object]]) -> None:
    base_entries = [
        {
            "path": "docs/context/README.md",
            "status": "current",
            "freshness": "maintained",
        },
        {
            "path": "docs/context/INDEX.md",
            "status": "current",
            "freshness": "maintained",
        },
    ]
    _write(
        repo,
        "docs/context/catalog.yaml",
        yaml.safe_dump({"version": 1, "entries": [*base_entries, *entries]}, sort_keys=False),
    )


def test_rule_a_superseded_without_replacement(tmp_path: Path) -> None:
    """Superseded catalog entry without replacement is an error (Rule A)."""
    repo = _repo(tmp_path)
    _write(repo, "docs/context/old.md", "# Old\n")
    _write_catalog(
        repo,
        [
            {
                "path": "docs/context/old.md",
                "status": "superseded",
                "freshness": "dated",
            }
        ],
    )
    _commit(repo, "superseded note without replacement", iso_date="2026-01-02T00:00:00+00:00")

    findings, proposed_moves, _conflicts = checker.check_freshness_decay(repo_root=repo)
    assert len(findings) == 1
    assert findings[0].rule == "superseded_replacement"
    assert findings[0].severity == "error"
    assert "must name a replacement" in findings[0].message
    assert proposed_moves == []


def test_rule_a_superseded_with_missing_replacement(tmp_path: Path) -> None:
    """Superseded entry pointing to a non-existent replacement is an error (Rule A)."""
    repo = _repo(tmp_path)
    _write(repo, "docs/context/old.md", "# Old\n")
    _write_catalog(
        repo,
        [
            {
                "path": "docs/context/old.md",
                "status": "superseded",
                "freshness": "dated",
                "replacement": "docs/context/nonexistent.md",
            }
        ],
    )
    _commit(
        repo, "superseded note with nonexistent replacement", iso_date="2026-01-02T00:00:00+00:00"
    )

    findings, _proposed_moves, _conflicts = checker.check_freshness_decay(repo_root=repo)
    assert len(findings) == 1
    assert findings[0].rule == "superseded_replacement"
    assert findings[0].severity == "error"
    assert "replacement file does not exist" in findings[0].message


def test_rule_a_superseded_with_valid_replacement(tmp_path: Path) -> None:
    """Superseded entry with valid replacement has no findings and proposes high-confidence move."""
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
    _commit(repo, "superseded note with replacement", iso_date="2026-01-02T00:00:00+00:00")

    findings, proposed_moves, _conflicts = checker.check_freshness_decay(repo_root=repo)
    assert len(findings) == 0
    assert len(proposed_moves) == 1
    assert proposed_moves[0].category == "superseded"
    assert proposed_moves[0].confidence == "high"
    assert proposed_moves[0].source == "docs/context/old.md"
    assert proposed_moves[0].target == "docs/context/archive/old.md"


def test_rule_b_stale_current_dated_note(tmp_path: Path) -> None:
    """Current dated entries trigger Rule B findings if older than max-age-days and unreferenced."""
    repo = _repo(tmp_path)
    _write(repo, "docs/context/dated.md", "# Dated\n")
    _write_catalog(
        repo,
        [
            {
                "path": "docs/context/dated.md",
                "status": "current",
                "freshness": "dated",
            }
        ],
    )
    _commit(repo, "dated note", iso_date="2025-01-01T00:00:00+00:00")

    findings, proposed_moves, _conflicts = checker.check_freshness_decay(
        repo_root=repo,
        max_age_days=180,
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert len(findings) == 1
    assert findings[0].rule == "stale_current_dated"
    assert findings[0].severity == "warning"
    assert findings[0].age_days == 365
    assert len(proposed_moves) == 1
    assert proposed_moves[0].category == "stale"
    assert proposed_moves[0].confidence == "review"


def test_rule_b_stale_referenced_is_skipped(tmp_path: Path) -> None:
    """Current dated entries are not stale if referenced in INDEX.md or another file."""
    repo = _repo(tmp_path)
    _write(repo, "docs/context/INDEX.md", "[dated](dated.md)\n")
    _write(repo, "docs/context/dated.md", "# Dated\n")
    _write_catalog(
        repo,
        [
            {
                "path": "docs/context/dated.md",
                "status": "current",
                "freshness": "dated",
            }
        ],
    )
    _commit(repo, "dated note with reference", iso_date="2025-01-01T00:00:00+00:00")

    findings, proposed_moves, _conflicts = checker.check_freshness_decay(
        repo_root=repo,
        max_age_days=180,
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert len(findings) == 0
    assert proposed_moves == []


def test_rule_c_orphan_note(tmp_path: Path) -> None:
    """Orphan notes not in INDEX.md or catalog.yaml trigger Rule C warnings."""
    repo = _repo(tmp_path)
    _write(repo, "docs/context/orphan.md", "# Orphan\n")
    _commit(repo, "orphan note", iso_date="2026-01-02T00:00:00+00:00")

    findings, _proposed_moves, _conflicts = checker.check_freshness_decay(repo_root=repo)
    assert any(
        f.rule == "orphan_context_note"
        and f.path == "docs/context/orphan.md"
        and f.severity == "warning"
        for f in findings
    )


def test_conflict_duplicate_target(tmp_path: Path) -> None:
    """Proposing moves with the same basename causes a duplicate target conflict."""
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
    _commit(repo, "colliding moves", iso_date="2026-01-02T00:00:00+00:00")

    _findings, _proposed_moves, conflicts = checker.check_freshness_decay(repo_root=repo)
    assert len(conflicts) == 1
    assert "duplicate_target" in conflicts[0]


def test_conflict_target_exists(tmp_path: Path) -> None:
    """Proposing a move to a destination that already exists causes a conflict."""
    repo = _repo(tmp_path)
    _write(repo, "docs/context/old.md", "# Old\n")
    _write(repo, "docs/context/new.md", "# New\n")
    _write(repo, "docs/context/archive/old.md", "# Existing archive\n")
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
    _commit(repo, "existing target", iso_date="2026-01-02T00:00:00+00:00")

    _findings, _proposed_moves, conflicts = checker.check_freshness_decay(repo_root=repo)
    assert len(conflicts) == 1
    assert "target_exists" in conflicts[0]
