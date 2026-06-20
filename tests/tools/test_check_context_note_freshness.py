"""Tests for context-note freshness checks."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml

from scripts.tools import check_context_note_freshness as checker


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


def test_markdown_target_normalization_preserves_parent_segments() -> None:
    """Markdown link normalization must not silently rewrite parent-relative paths."""

    assert "../evidence/example.md" in checker._markdown_targets("[bad](../evidence/example.md)")


def test_superseded_without_replacement_is_hard_error(tmp_path: Path) -> None:
    """Superseded catalog rows must name their replacement."""

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
    _commit(repo, "superseded note", iso_date="2026-01-02T00:00:00+00:00")

    findings = checker.check_freshness(repo_root=repo)
    summary = checker._summary(findings, strict=False)

    assert summary["exit_code"] == 1
    assert findings[0].rule == "superseded_replacement"
    assert findings[0].severity == "error"


def test_custom_context_paths_drive_index_catalog_and_exclusions(tmp_path: Path) -> None:
    """Custom context directories must not fall back to docs/context constants."""

    repo = _repo(tmp_path)
    _write(repo, "custom/context/INDEX.md", "[active](active.md)\n")
    _write(repo, "custom/context/README.md", "# Custom\n")
    _write(repo, "custom/context/active.md", "# Active\n")
    _write(repo, "custom/context/orphan.md", "# Orphan\n")
    _write(repo, "custom/context/evidence/ignored.md", "# Ignored evidence\n")
    _write(
        repo,
        "custom/context/catalog.yaml",
        yaml.safe_dump(
            {
                "version": 1,
                "entries": [
                    {
                        "path": "custom/context/README.md",
                        "status": "current",
                        "freshness": "maintained",
                    },
                    {
                        "path": "custom/context/active.md",
                        "status": "current",
                        "freshness": "dated",
                    },
                    {
                        "status": "superseded",
                        "freshness": "dated",
                    },
                ],
            },
            sort_keys=False,
        ),
    )
    _commit(repo, "custom context", iso_date="2025-01-01T00:00:00+00:00")

    findings = checker.check_freshness(
        repo_root=repo,
        catalog_path=Path("custom/context/catalog.yaml"),
        context_dir=Path("custom/context"),
        index_path=Path("custom/context/INDEX.md"),
        max_age_days=180,
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )
    by_path = {finding.path: finding.rule for finding in findings}

    assert by_path["custom/context/catalog.yaml:entries[2]"] == "superseded_replacement"
    assert by_path["custom/context/orphan.md"] == "orphan_context_note"
    assert "custom/context/active.md" not in by_path
    assert "custom/context/evidence/ignored.md" not in by_path


def test_current_dated_note_uses_git_last_touch_for_staleness(tmp_path: Path) -> None:
    """Current dated entries use committed touch dates for stale-note warnings."""

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

    findings = checker.check_freshness(
        repo_root=repo,
        max_age_days=180,
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert [finding.rule for finding in findings] == ["stale_current_dated"]
    assert findings[0].age_days == 365
    assert findings[0].last_touched_at == "2025-01-01T00:00:00+00:00"


def test_index_reference_suppresses_current_dated_staleness_warning(tmp_path: Path) -> None:
    """Indexed dated notes remain active entry points even when old."""

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
    _commit(repo, "indexed dated note", iso_date="2025-01-01T00:00:00+00:00")

    findings = checker.check_freshness(
        repo_root=repo,
        max_age_days=180,
        now=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert findings == []


def test_orphan_context_note_warns_when_absent_from_index_and_catalog(tmp_path: Path) -> None:
    """Tracked context notes absent from both routing surfaces are warnings."""

    repo = _repo(tmp_path)
    _write(repo, "docs/context/orphan.md", "# Orphan\n")
    _commit(repo, "orphan note", iso_date="2026-01-02T00:00:00+00:00")

    findings = checker.check_freshness(repo_root=repo)

    assert any(
        finding.rule == "orphan_context_note"
        and finding.path == "docs/context/orphan.md"
        and finding.severity == "warning"
        for finding in findings
    )


def test_strict_mode_promotes_warnings_to_nonzero_exit(tmp_path: Path) -> None:
    """Strict mode turns warning-only reports into a failing exit code."""

    repo = _repo(tmp_path)
    _write(repo, "docs/context/orphan.md", "# Orphan\n")
    _commit(repo, "orphan note", iso_date="2026-01-02T00:00:00+00:00")

    findings = checker.check_freshness(repo_root=repo)

    assert checker._summary(findings, strict=False)["exit_code"] == 0
    assert checker._summary(findings, strict=True)["exit_code"] == 1
