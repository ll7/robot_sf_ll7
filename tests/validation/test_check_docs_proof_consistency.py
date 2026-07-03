"""Tests for the docs/proof consistency checker."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import pytest

from scripts.validation.check_docs_proof_consistency import (
    ChangedFile,
    _collect_diagnostics,
    _context_catalog_diagnostics,
    _context_index_link_diagnostics,
    _context_note_freshness_diagnostics,
    _context_readme_link_diagnostics,
    _evidence_catalog_coverage_diagnostics,
    _parse_name_status,
    _selected_files,
    _should_check_context_catalog,
)


def _run_git(repo_root: Path, *args: str, env: dict[str, str] | None = None) -> None:
    """Run a git command for a temporary repository fixture."""
    subprocess.run(
        ["git", *args],
        cwd=repo_root,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def _commit_all(repo_root: Path, message: str, *, iso_date: str) -> None:
    """Commit all fixture changes with a deterministic timestamp."""
    env = {
        "GIT_AUTHOR_DATE": iso_date,
        "GIT_COMMITTER_DATE": iso_date,
        "GIT_AUTHOR_NAME": "Test Author",
        "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "Test Author",
        "GIT_COMMITTER_EMAIL": "test@example.com",
    }
    _run_git(repo_root, "add", ".")
    _run_git(repo_root, "commit", "-m", message, env=env)


def _freshness_repo(tmp_path: Path, catalog_entries: str) -> Path:
    """Create minimal git repo for context-note freshness integration tests."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    _run_git(repo_root, "init")
    _run_git(repo_root, "config", "user.name", "Test Author")
    _run_git(repo_root, "config", "user.email", "test@example.com")
    context_dir = repo_root / "docs/context"
    context_dir.mkdir(parents=True)
    (context_dir / "README.md").write_text("# Context\n", encoding="utf-8")
    (context_dir / "INDEX.md").write_text("# Index\n", encoding="utf-8")
    (context_dir / "catalog.yaml").write_text(
        "version: 1\n"
        "entries:\n"
        "  - path: docs/context/README.md\n"
        "    status: current\n"
        "    freshness: maintained\n"
        "  - path: docs/context/INDEX.md\n"
        "    status: current\n"
        "    freshness: maintained\n"
        f"{catalog_entries}",
        encoding="utf-8",
    )
    return repo_root


def test_missing_context_note_index_link_is_reported() -> None:
    """New top-level context notes should be linked from the context README."""
    diagnostics = _context_readme_link_diagnostics(
        [ChangedFile(status="A", path=Path("docs/context/issue_999_example.md"))],
        context_readme_text="# Context Notes Workflow\n",
    )

    assert diagnostics == [
        type(diagnostics[0])(
            path=Path("docs/context/issue_999_example.md"),
            message="added context note is not linked from docs/context/README.md",
        )
    ]


def test_context_note_anchor_link_counts_as_index_link() -> None:
    """Anchored links should satisfy the context-note discoverability check."""
    diagnostics = _context_readme_link_diagnostics(
        [ChangedFile(status="A", path=Path("docs/context/issue_999_example.md"))],
        context_readme_text="- [Issue 999 Example](issue_999_example.md#validation)\n",
    )

    assert diagnostics == []


def test_context_note_requires_context_index_link() -> None:
    """New top-level context notes should be linked from the context index."""
    diagnostics = _context_index_link_diagnostics(
        [ChangedFile(status="A", path=Path("docs/context/issue_999_example.md"))],
        context_index_text="# Context Retrieval Index\n",
    )

    assert diagnostics == [
        type(diagnostics[0])(
            path=Path("docs/context/issue_999_example.md"),
            message="added context note is not linked from docs/context/INDEX.md",
        )
    ]


def test_context_index_link_check_skips_context_entrypoints() -> None:
    """README and INDEX are the context entrypoints, not notes requiring index links."""
    diagnostics = _context_index_link_diagnostics(
        [
            ChangedFile(status="A", path=Path("docs/context/README.md")),
            ChangedFile(status="A", path=Path("docs/context/INDEX.md")),
        ],
        context_index_text="# Context Retrieval Index\n",
    )

    assert diagnostics == []


def test_context_note_same_basename_in_subdir_does_not_count_as_index_link() -> None:
    """A link to another note with the same basename should not index a top-level note."""
    diagnostics = _context_readme_link_diagnostics(
        [ChangedFile(status="A", path=Path("docs/context/issue_999_example.md"))],
        context_readme_text="- [Archived Example](archive/issue_999_example.md)\n",
    )

    assert any(
        "not linked from docs/context/README.md" in diagnostic.message for diagnostic in diagnostics
    )


def test_copied_context_note_requires_index_link() -> None:
    """Copied top-level context notes should be handled like added notes."""
    diagnostics = _context_readme_link_diagnostics(
        [ChangedFile(status="C100", path=Path("docs/context/issue_999_example.md"))],
        context_readme_text="# Context Notes Workflow\n",
    )

    assert any(
        "not linked from docs/context/README.md" in diagnostic.message for diagnostic in diagnostics
    )


def test_parse_name_status_uses_new_path_for_rename() -> None:
    """Rename rows should track the current worktree path, not the old path pair."""
    changed = _parse_name_status("R100\tdocs/context/old.md\tdocs/context/new.md\n")

    assert changed == [ChangedFile(status="R100", path=Path("docs/context/new.md"))]


def test_absolute_local_path_in_tracked_evidence_is_reported(tmp_path: Path) -> None:
    """Tracked evidence should not preserve machine-local absolute paths."""
    repo_root = tmp_path
    (repo_root / "docs/context/evidence").mkdir(parents=True)
    (repo_root / "docs/context/README.md").write_text(
        "# Context Notes Workflow\n", encoding="utf-8"
    )
    evidence = repo_root / "docs/context/evidence/report.md"
    evidence.write_text("Artifact came from /home/user/output/report.json.\n", encoding="utf-8")

    diagnostics = _collect_diagnostics(
        [ChangedFile(status="M", path=Path("docs/context/evidence/report.md"))],
        repo_root=repo_root,
    )

    assert any(
        "absolute local filesystem paths" in diagnostic.message for diagnostic in diagnostics
    )


def test_output_pointer_in_tracked_evidence_is_reported(tmp_path: Path) -> None:
    """Tracked evidence should not link to ignored output artifacts."""
    repo_root = tmp_path
    (repo_root / "docs/context/evidence").mkdir(parents=True)
    (repo_root / "docs/context/README.md").write_text(
        "# Context Notes Workflow\n", encoding="utf-8"
    )
    evidence = repo_root / "docs/context/evidence/report.md"
    evidence.write_text("[artifact](output/reports/run.json)\n", encoding="utf-8")

    diagnostics = _collect_diagnostics(
        [ChangedFile(status="M", path=Path("docs/context/evidence/report.md"))],
        repo_root=repo_root,
    )

    assert any("ignored output/ artifacts" in diagnostic.message for diagnostic in diagnostics)


def test_inline_output_pointer_in_markdown_evidence_is_reported(tmp_path: Path) -> None:
    """Markdown evidence should flag inline ignored-output pointers, not only links."""
    repo_root = tmp_path
    (repo_root / "docs/context/evidence").mkdir(parents=True)
    (repo_root / "docs/context/README.md").write_text(
        "# Context Notes Workflow\n", encoding="utf-8"
    )
    evidence = repo_root / "docs/context/evidence/report.md"
    evidence.write_text("Manifest: `output/reports/run.json`\n", encoding="utf-8")

    diagnostics = _collect_diagnostics(
        [ChangedFile(status="M", path=Path("docs/context/evidence/report.md"))],
        repo_root=repo_root,
    )

    assert any("ignored output/ artifacts" in diagnostic.message for diagnostic in diagnostics)


def test_prefix_sibling_of_evidence_dir_is_not_treated_as_tracked_evidence(tmp_path: Path) -> None:
    """Only the real evidence directory should trigger durable-evidence checks."""
    repo_root = tmp_path
    (repo_root / "docs/context/evidence_backup").mkdir(parents=True)
    (repo_root / "docs/context/README.md").write_text(
        "# Context Notes Workflow\n", encoding="utf-8"
    )
    note = repo_root / "docs/context/evidence_backup/report.md"
    note.write_text("[artifact](output/reports/run.json)\n", encoding="utf-8")

    diagnostics = _collect_diagnostics(
        [ChangedFile(status="M", path=Path("docs/context/evidence_backup/report.md"))],
        repo_root=repo_root,
    )

    assert diagnostics == []


def test_context_note_output_command_is_allowed(tmp_path: Path) -> None:
    """Regular context notes may still mention output paths in reproducible commands."""
    repo_root = tmp_path
    (repo_root / "docs/context").mkdir(parents=True)
    (repo_root / "docs/context/README.md").write_text(
        "- [Issue 999 Example](issue_999_example.md)\n",
        encoding="utf-8",
    )
    note = repo_root / "docs/context/issue_999_example.md"
    note.write_text(
        "Run `BASE_REF=origin/main scripts/dev/pr_ready_check.sh` and inspect `output/coverage/coverage.json` locally.\n",
        encoding="utf-8",
    )

    diagnostics = _collect_diagnostics(
        [ChangedFile(status="M", path=Path("docs/context/issue_999_example.md"))],
        repo_root=repo_root,
    )

    assert diagnostics == []


def test_stale_no_validation_phrase_with_commands_is_reported(tmp_path: Path) -> None:
    """Notes should not claim no validation ran while listing executed commands."""
    repo_root = tmp_path
    (repo_root / "docs/context").mkdir(parents=True)
    (repo_root / "docs/context/README.md").write_text(
        "- [Issue 999 Example](issue_999_example.md)\n",
        encoding="utf-8",
    )
    note = repo_root / "docs/context/issue_999_example.md"
    note.write_text(
        "No validation commands were run during this pass.\n\n"
        "- `uv run pytest tests/test_cli.py -q`\n",
        encoding="utf-8",
    )

    diagnostics = _collect_diagnostics(
        [ChangedFile(status="M", path=Path("docs/context/issue_999_example.md"))],
        repo_root=repo_root,
    )

    assert any(
        "no validation commands were run" in diagnostic.message for diagnostic in diagnostics
    )


def test_line_start_issue_reference_is_reported(tmp_path: Path) -> None:
    """Markdown issue references should not start lines as bare #123 tokens."""
    repo_root = tmp_path
    (repo_root / "docs/context").mkdir(parents=True)
    (repo_root / "docs/context/README.md").write_text(
        "- [Issue 999 Example](issue_999_example.md)\n",
        encoding="utf-8",
    )
    note = repo_root / "docs/context/issue_999_example.md"
    note.write_text("#1108 needs a clearer reference style.\n", encoding="utf-8")

    diagnostics = _collect_diagnostics(
        [ChangedFile(status="M", path=Path("docs/context/issue_999_example.md"))],
        repo_root=repo_root,
    )

    assert any("Issue #1108" in diagnostic.message for diagnostic in diagnostics)


def test_preferred_issue_reference_form_is_allowed(tmp_path: Path) -> None:
    """The preferred prose form should satisfy the issue-reference style guard."""
    repo_root = tmp_path
    (repo_root / "docs/context").mkdir(parents=True)
    (repo_root / "docs/context/README.md").write_text(
        "- [Issue 999 Example](issue_999_example.md)\n",
        encoding="utf-8",
    )
    note = repo_root / "docs/context/issue_999_example.md"
    note.write_text("Issue #1108 keeps this link readable in Markdown.\n", encoding="utf-8")

    diagnostics = _collect_diagnostics(
        [ChangedFile(status="M", path=Path("docs/context/issue_999_example.md"))],
        repo_root=repo_root,
    )

    assert diagnostics == []


def test_issue_reference_guard_handles_bullets_tables_and_headings(tmp_path: Path) -> None:
    """Flag bare issue references in lists/tables without banning numeric headings."""
    repo_root = tmp_path
    (repo_root / "docs/context").mkdir(parents=True)
    (repo_root / "docs/context/README.md").write_text(
        "- [Issue 999 Example](issue_999_example.md)\n",
        encoding="utf-8",
    )
    note = repo_root / "docs/context/issue_999_example.md"
    note.write_text(
        "# 2026 plan\n\n"
        "- #1108 should use prose.\n"
        "| #1262 | missing prefix |\n"
        "```\n"
        "#999 is only an example inside a code block\n"
        "```\n",
        encoding="utf-8",
    )

    diagnostics = _collect_diagnostics(
        [ChangedFile(status="M", path=Path("docs/context/issue_999_example.md"))],
        repo_root=repo_root,
    )

    messages = [diagnostic.message for diagnostic in diagnostics]
    assert any("Issue #1108" in message for message in messages)
    assert any("Issue #1262" in message for message in messages)
    assert not any("Issue #999" in message for message in messages)


def test_issue_reference_guard_ignores_indented_code_blocks(tmp_path: Path) -> None:
    """Four-space indented Markdown code blocks should not be prose diagnostics."""
    repo_root = tmp_path
    (repo_root / "docs/context").mkdir(parents=True)
    (repo_root / "docs/context/README.md").write_text(
        "- [Issue 999 Example](issue_999_example.md)\n",
        encoding="utf-8",
    )
    note = repo_root / "docs/context/issue_999_example.md"
    note.write_text("    #1108 is literal code, not prose.\n", encoding="utf-8")

    diagnostics = _collect_diagnostics(
        [ChangedFile(status="M", path=Path("docs/context/issue_999_example.md"))],
        repo_root=repo_root,
    )

    assert diagnostics == []


def test_issue_reference_guard_handles_indented_long_fences(tmp_path: Path) -> None:
    """Markdown fences can be indented up to three spaces and longer than three ticks."""
    repo_root = tmp_path
    (repo_root / "docs/context").mkdir(parents=True)
    (repo_root / "docs/context/README.md").write_text(
        "- [Issue 999 Example](issue_999_example.md)\n",
        encoding="utf-8",
    )
    note = repo_root / "docs/context/issue_999_example.md"
    note.write_text(
        "   ````python\n"
        "#1108 is only sample code.\n"
        "   ````\n"
        "#1262 is prose and should be flagged.\n",
        encoding="utf-8",
    )

    diagnostics = _collect_diagnostics(
        [ChangedFile(status="M", path=Path("docs/context/issue_999_example.md"))],
        repo_root=repo_root,
    )

    messages = [diagnostic.message for diagnostic in diagnostics]
    assert not any("Issue #1108" in message for message in messages)
    assert any("Issue #1262" in message for message in messages)


def test_explicit_path_new_context_note_is_treated_as_added(tmp_path: Path) -> None:
    """Explicit path checks should still enforce added-note index-link validation."""
    repo_root = tmp_path
    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True, text=True)
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    (repo_root / "docs/context").mkdir(parents=True)
    (repo_root / "docs/context/README.md").write_text(
        "# Context Notes Workflow\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "add", "docs/context/README.md"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "test fixture"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )

    note = repo_root / "docs/context/issue_999_example.md"
    note.write_text("Fresh note body.\n", encoding="utf-8")

    selected = _selected_files(
        argparse.Namespace(path=[str(note)], base="HEAD"),
        repo_root=repo_root,
    )
    diagnostics = _collect_diagnostics(selected, repo_root=repo_root)

    assert selected == [ChangedFile(status="A", path=Path("docs/context/issue_999_example.md"))]
    assert any(
        "not linked from docs/context/README.md" in diagnostic.message for diagnostic in diagnostics
    )


def test_explicit_path_outside_repo_is_rejected(tmp_path: Path) -> None:
    """Explicit path checks should not read files outside the repository root."""
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    outside_path = tmp_path / "outside.md"

    with pytest.raises(ValueError, match="repository root"):
        _selected_files(
            argparse.Namespace(path=[str(outside_path)], base="HEAD"),
            repo_root=repo_root,
        )


def _write_catalog_with_output_pointer(repo_root: Path) -> None:
    """Create catalog debt that should only fail strict catalog modes."""
    (repo_root / "robot_sf").mkdir()
    (repo_root / "robot_sf/example.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo_root / "docs/context/evidence").mkdir(parents=True)
    evidence = repo_root / "docs/context/evidence/report.md"
    evidence.write_text("[artifact](output/reports/run.json)\n", encoding="utf-8")
    catalog = repo_root / "docs/context/catalog.yaml"
    catalog.write_text(
        """
version: 1
entries:
  - path: docs/context/evidence/report.md
    status: evidence
    freshness: evidence
""".strip()
        + "\n",
        encoding="utf-8",
    )


def test_code_only_diff_skips_context_catalog_debt(tmp_path: Path) -> None:
    """Code-only selected files should not surface unrelated catalog debt."""
    repo_root = tmp_path
    _write_catalog_with_output_pointer(repo_root)

    diagnostics = _collect_diagnostics(
        [ChangedFile(status="M", path=Path("robot_sf/example.py"))],
        repo_root=repo_root,
    )

    assert diagnostics == []
    assert not _should_check_context_catalog(
        [ChangedFile(status="M", path=Path("robot_sf/example.py"))]
    )


def test_docs_context_note_diff_with_anchors_skips_catalog_debt(tmp_path: Path) -> None:
    """Docs/context note PRs must not surface unrelated baseline catalog debt.

    Mirrors the ``check_docs_proof_consistency_diff.sh`` docs/context-only
    selection after issue #4178: a changed note plus the README/INDEX link
    anchors, but *not* ``catalog.yaml``.  Because the catalog is not selected,
    pre-existing catalog debt (here an evidence row pointing at ignored
    ``output/`` artifacts) must stay silent for the note-only PR.
    """
    repo_root = tmp_path
    _write_catalog_with_output_pointer(repo_root)
    (repo_root / "docs/context/example_note.md").write_text("Clean note body.\n", encoding="utf-8")

    selected = [
        ChangedFile(status="M", path=Path("docs/context/example_note.md")),
        ChangedFile(status="M", path=Path("docs/context/README.md")),
        ChangedFile(status="M", path=Path("docs/context/INDEX.md")),
    ]

    assert not _should_check_context_catalog(selected)
    assert _collect_diagnostics(selected, repo_root=repo_root) == []


def test_selected_context_catalog_keeps_strict_catalog_diagnostics(tmp_path: Path) -> None:
    """Selecting the context catalog keeps strict catalog provenance checks."""
    repo_root = tmp_path
    _write_catalog_with_output_pointer(repo_root)

    diagnostics = _collect_diagnostics(
        [ChangedFile(status="M", path=Path("docs/context/catalog.yaml"))],
        repo_root=repo_root,
    )

    assert _should_check_context_catalog(
        [ChangedFile(status="M", path=Path("docs/context/catalog.yaml"))]
    )
    assert any("ignored output/ artifacts" in diagnostic.message for diagnostic in diagnostics)


def test_force_context_catalog_keeps_strict_catalog_diagnostics(tmp_path: Path) -> None:
    """Explicit full context-catalog mode checks catalog debt even for code diffs."""
    repo_root = tmp_path
    _write_catalog_with_output_pointer(repo_root)

    diagnostics = _collect_diagnostics(
        [ChangedFile(status="M", path=Path("robot_sf/example.py"))],
        repo_root=repo_root,
        force_context_catalog=True,
    )

    assert any("ignored output/ artifacts" in diagnostic.message for diagnostic in diagnostics)


def test_context_catalog_reports_missing_status_and_path(tmp_path: Path) -> None:
    """Catalog entries need existing paths plus status and freshness metadata."""
    repo_root = tmp_path
    catalog = repo_root / "docs/context/catalog.yaml"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(
        """
version: 1
entries:
  - path: docs/context/missing.md
    freshness: maintained
""".strip()
        + "\n",
        encoding="utf-8",
    )

    diagnostics = _context_catalog_diagnostics(
        Path("docs/context/catalog.yaml"),
        repo_root=repo_root,
    )
    messages = [diagnostic.message for diagnostic in diagnostics]

    assert any("path does not exist" in message for message in messages)
    assert any("status must be one of" in message for message in messages)


def test_context_catalog_accepts_valid_entry(tmp_path: Path) -> None:
    """A complete catalog row with declared metadata should pass."""
    repo_root = tmp_path
    (repo_root / "docs/context").mkdir(parents=True)
    (repo_root / "docs/context/current.md").write_text("current\n", encoding="utf-8")
    catalog = repo_root / "docs/context/catalog.yaml"
    catalog.write_text(
        """
version: 1
status_values:
  current: Active source of truth.
  historical: Background.
  superseded: Replaced.
  evidence: Proof.
  proposal: Planned.
freshness_values:
  maintained: Update when changed.
  dated: Date-scoped.
  policy: Durable boundary.
  evidence: Evidence pointer.
entries:
  - path: docs/context/current.md
    status: current
    freshness: maintained
""".strip()
        + "\n",
        encoding="utf-8",
    )

    diagnostics = _context_catalog_diagnostics(
        Path("docs/context/catalog.yaml"),
        repo_root=repo_root,
    )

    assert diagnostics == []


def test_context_catalog_rejects_unknown_freshness(tmp_path: Path) -> None:
    """Freshness metadata should use the catalog vocabulary."""
    repo_root = tmp_path
    (repo_root / "docs/context").mkdir(parents=True)
    (repo_root / "docs/context/current.md").write_text("current\n", encoding="utf-8")
    catalog = repo_root / "docs/context/catalog.yaml"
    catalog.write_text(
        """
version: 1
freshness_values:
  maintained: Update when changed.
entries:
  - path: docs/context/current.md
    status: current
    freshness: stale
""".strip()
        + "\n",
        encoding="utf-8",
    )

    diagnostics = _context_catalog_diagnostics(
        Path("docs/context/catalog.yaml"),
        repo_root=repo_root,
    )

    assert any("freshness must be one of" in diagnostic.message for diagnostic in diagnostics)


def test_context_catalog_requires_superseded_replacement(tmp_path: Path) -> None:
    """Superseded catalog entries must name an existing replacement."""
    repo_root = tmp_path
    (repo_root / "docs/context").mkdir(parents=True)
    (repo_root / "docs/context/old.md").write_text("old\n", encoding="utf-8")
    catalog = repo_root / "docs/context/catalog.yaml"
    catalog.write_text(
        """
version: 1
entries:
  - path: docs/context/old.md
    status: superseded
    freshness: dated
""".strip()
        + "\n",
        encoding="utf-8",
    )

    diagnostics = _context_catalog_diagnostics(
        Path("docs/context/catalog.yaml"),
        repo_root=repo_root,
    )

    assert any("replacement is required" in diagnostic.message for diagnostic in diagnostics)


def test_context_note_freshness_reports_superseded_errors(tmp_path: Path) -> None:
    """Docs-proof freshness integration fails superseded rows without replacements."""
    repo_root = _freshness_repo(
        tmp_path,
        "  - path: docs/context/old.md\n    status: superseded\n    freshness: dated\n",
    )
    (repo_root / "docs/context/old.md").write_text("# Old\n", encoding="utf-8")
    _commit_all(repo_root, "superseded fixture", iso_date="2026-01-01T00:00:00+00:00")

    diagnostics = _context_note_freshness_diagnostics(
        repo_root=repo_root,
        max_age_days=180,
        strict=False,
    )

    assert any("superseded_replacement" in diagnostic.message for diagnostic in diagnostics)
    assert any(diagnostic.path == Path("docs/context/old.md") for diagnostic in diagnostics)


def test_context_note_freshness_non_strict_ignores_orphan_warnings(tmp_path: Path) -> None:
    """Docs-proof non-strict mode preserves freshness warning-only compatibility."""
    repo_root = _freshness_repo(tmp_path, "")
    (repo_root / "docs/context/orphan.md").write_text("# Orphan\n", encoding="utf-8")
    _commit_all(repo_root, "orphan fixture", iso_date="2026-01-01T00:00:00+00:00")

    diagnostics = _context_note_freshness_diagnostics(
        repo_root=repo_root,
        max_age_days=180,
        strict=False,
    )

    assert diagnostics == []


def test_context_note_freshness_strict_reports_orphan_warnings(tmp_path: Path) -> None:
    """Docs-proof strict freshness mode promotes orphan warnings to failures."""
    repo_root = _freshness_repo(tmp_path, "")
    (repo_root / "docs/context/orphan.md").write_text("# Orphan\n", encoding="utf-8")
    _commit_all(repo_root, "orphan fixture", iso_date="2026-01-01T00:00:00+00:00")

    diagnostics = _context_note_freshness_diagnostics(
        repo_root=repo_root,
        max_age_days=180,
        strict=True,
    )

    assert any("orphan_context_note" in diagnostic.message for diagnostic in diagnostics)
    assert any(diagnostic.path == Path("docs/context/orphan.md") for diagnostic in diagnostics)


def test_context_note_freshness_strict_reports_stale_current_notes(tmp_path: Path) -> None:
    """Docs-proof strict freshness mode promotes stale current-note warnings."""
    repo_root = _freshness_repo(
        tmp_path,
        "  - path: docs/context/dated.md\n    status: current\n    freshness: dated\n",
    )
    (repo_root / "docs/context/dated.md").write_text("# Dated\n", encoding="utf-8")
    _commit_all(repo_root, "dated fixture", iso_date="2025-01-01T00:00:00+00:00")

    diagnostics = _context_note_freshness_diagnostics(
        repo_root=repo_root,
        max_age_days=180,
        strict=True,
    )

    assert any("stale_current_dated" in diagnostic.message for diagnostic in diagnostics)
    assert any(diagnostic.path == Path("docs/context/dated.md") for diagnostic in diagnostics)


def test_freshness_diff_scope_ignores_unchanged_superseded(tmp_path: Path) -> None:
    """Diff scope must not fail on a superseded row when no relevant file changed."""
    repo_root = _freshness_repo(
        tmp_path,
        "  - path: docs/context/old.md\n    status: superseded\n    freshness: dated\n",
    )
    (repo_root / "docs/context/old.md").write_text("# Old\n", encoding="utf-8")
    _commit_all(repo_root, "superseded fixture", iso_date="2026-01-01T00:00:00+00:00")

    diagnostics = _context_note_freshness_diagnostics(
        repo_root=repo_root,
        max_age_days=180,
        strict=False,
        changed_note_paths=set(),
        catalog_changed=False,
    )

    assert diagnostics == []


def test_freshness_diff_scope_reports_changed_superseded_note(tmp_path: Path) -> None:
    """Diff scope fails when the diff touches the superseded note itself."""
    repo_root = _freshness_repo(
        tmp_path,
        "  - path: docs/context/old.md\n    status: superseded\n    freshness: dated\n",
    )
    (repo_root / "docs/context/old.md").write_text("# Old\n", encoding="utf-8")
    _commit_all(repo_root, "superseded fixture", iso_date="2026-01-01T00:00:00+00:00")

    diagnostics = _context_note_freshness_diagnostics(
        repo_root=repo_root,
        max_age_days=180,
        strict=False,
        changed_note_paths={Path("docs/context/old.md")},
        catalog_changed=False,
    )

    assert any("superseded_replacement" in diagnostic.message for diagnostic in diagnostics)


def test_freshness_diff_scope_catalog_change_reports_superseded(tmp_path: Path) -> None:
    """A catalog edit keeps catalog-driven superseded errors even if the note is unchanged."""
    repo_root = _freshness_repo(
        tmp_path,
        "  - path: docs/context/old.md\n    status: superseded\n    freshness: dated\n",
    )
    (repo_root / "docs/context/old.md").write_text("# Old\n", encoding="utf-8")
    _commit_all(repo_root, "superseded fixture", iso_date="2026-01-01T00:00:00+00:00")

    diagnostics = _context_note_freshness_diagnostics(
        repo_root=repo_root,
        max_age_days=180,
        strict=False,
        changed_note_paths=set(),
        catalog_changed=True,
    )

    assert any("superseded_replacement" in diagnostic.message for diagnostic in diagnostics)


def test_freshness_diff_scope_orphan_requires_changed_note(tmp_path: Path) -> None:
    """Orphan findings are note-driven: a catalog change alone must not resurrect them."""
    repo_root = _freshness_repo(tmp_path, "")
    (repo_root / "docs/context/orphan.md").write_text("# Orphan\n", encoding="utf-8")
    _commit_all(repo_root, "orphan fixture", iso_date="2026-01-01T00:00:00+00:00")

    # Catalog changed but the orphan note itself is not in the diff -> not reported.
    catalog_only = _context_note_freshness_diagnostics(
        repo_root=repo_root,
        max_age_days=180,
        strict=True,
        changed_note_paths=set(),
        catalog_changed=True,
    )
    assert catalog_only == []

    # The orphan note is in the diff -> reported under strict scope.
    note_changed = _context_note_freshness_diagnostics(
        repo_root=repo_root,
        max_age_days=180,
        strict=True,
        changed_note_paths={Path("docs/context/orphan.md")},
        catalog_changed=True,
    )
    assert any("orphan_context_note" in diagnostic.message for diagnostic in note_changed)


def test_context_catalog_evidence_rejects_output_pointer(tmp_path: Path) -> None:
    """Evidence catalog entries should not depend on ignored output paths."""
    repo_root = tmp_path
    (repo_root / "docs/context/evidence").mkdir(parents=True)
    evidence = repo_root / "docs/context/evidence/report.md"
    evidence.write_text("Durable evidence: output/local/report.json\n", encoding="utf-8")
    catalog = repo_root / "docs/context/catalog.yaml"
    catalog.write_text(
        """
version: 1
entries:
  - path: docs/context/evidence/report.md
    status: evidence
    freshness: evidence
""".strip()
        + "\n",
        encoding="utf-8",
    )

    diagnostics = _context_catalog_diagnostics(
        Path("docs/context/catalog.yaml"),
        repo_root=repo_root,
    )

    assert any("ignored output/ artifacts" in diagnostic.message for diagnostic in diagnostics)


def test_context_catalog_accepts_explicit_legacy_dirty_evidence(tmp_path: Path) -> None:
    """Legacy evidence entries can be cataloged without weakening the default guard."""
    repo_root = tmp_path
    (repo_root / "docs/context/evidence").mkdir(parents=True)
    evidence = repo_root / "docs/context/evidence/legacy_report.json"
    evidence.write_text('{"source": "output/local/report.json"}\n', encoding="utf-8")
    catalog = repo_root / "docs/context/catalog.yaml"
    catalog.write_text(
        """
version: 1
entries:
- path: docs/context/evidence/legacy_report.json
  status: evidence
  freshness: evidence
  legacy_dirty_evidence: true
""".strip()
        + "\n",
        encoding="utf-8",
    )

    diagnostics = _context_catalog_diagnostics(
        Path("docs/context/catalog.yaml"),
        repo_root=repo_root,
    )

    assert diagnostics == []


def test_context_catalog_skips_binary_evidence_scan(tmp_path: Path) -> None:
    """Binary evidence entries should not crash text-only provenance checks."""
    repo_root = tmp_path
    (repo_root / "docs/context/evidence").mkdir(parents=True)
    evidence = repo_root / "docs/context/evidence/frame.png"
    evidence.write_bytes(b"\x89PNG\r\n\x1a\n\x80\x81")
    catalog = repo_root / "docs/context/catalog.yaml"
    catalog.write_text(
        """
version: 1
entries:
  - path: docs/context/evidence/frame.png
    status: evidence
    freshness: evidence
""".strip()
        + "\n",
        encoding="utf-8",
    )

    diagnostics = _context_catalog_diagnostics(
        Path("docs/context/catalog.yaml"),
        repo_root=repo_root,
    )

    assert diagnostics == []


def test_context_catalog_reports_malformed_yaml(tmp_path: Path) -> None:
    """Malformed catalog YAML should return a diagnostic instead of a traceback."""
    repo_root = tmp_path
    catalog = repo_root / "docs/context/catalog.yaml"
    catalog.parent.mkdir(parents=True)
    catalog.write_text("version: [\n", encoding="utf-8")

    diagnostics = _context_catalog_diagnostics(
        Path("docs/context/catalog.yaml"),
        repo_root=repo_root,
    )

    assert len(diagnostics) == 1
    assert "context catalog is not a valid YAML file" in diagnostics[0].message


# ---------------------------------------------------------------------------
# Evidence catalog coverage checks  (--check-evidence-catalog mode)
# ---------------------------------------------------------------------------


def _make_git_repo(tmp_path: Path) -> Path:
    """Create a minimal git repo rooted at tmp_path and return it."""
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "t@t"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    return tmp_path


def _git_add_commit(repo_root: Path, *paths: Path) -> None:
    """Stage and commit the given paths in the tmp git repo."""
    for p in paths:
        subprocess.run(
            ["git", "add", str(p.relative_to(repo_root))],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
    subprocess.run(
        ["git", "commit", "-m", "fixture"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )


def test_evidence_catalog_reports_uncovered_bundle(tmp_path: Path) -> None:
    """An evidence bundle with no catalog entry should be reported."""
    repo_root = _make_git_repo(tmp_path)
    bundle = repo_root / "docs/context/evidence/issue_999_example"
    bundle.mkdir(parents=True)
    summary = bundle / "summary.json"
    summary.write_text('{"status": "ok"}\n', encoding="utf-8")

    # Catalog exists but has NO entry for the bundle.
    catalog = repo_root / "docs/context/catalog.yaml"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text(
        "version: 1\nentries: []\n",
        encoding="utf-8",
    )
    _git_add_commit(repo_root, summary, catalog)

    diagnostics = _evidence_catalog_coverage_diagnostics(repo_root=repo_root)

    assert any("docs/context/evidence/issue_999_example" in d.message for d in diagnostics), (
        diagnostics
    )


def test_evidence_catalog_passes_when_bundle_is_covered(tmp_path: Path) -> None:
    """A bundle whose file is cataloged should produce no diagnostic."""
    repo_root = _make_git_repo(tmp_path)
    bundle = repo_root / "docs/context/evidence/issue_999_covered"
    bundle.mkdir(parents=True)
    summary = bundle / "summary.json"
    summary.write_text('{"status": "ok"}\n', encoding="utf-8")

    catalog = repo_root / "docs/context/catalog.yaml"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text(
        "version: 1\nentries:\n"
        "  - path: docs/context/evidence/issue_999_covered/summary.json\n"
        "    status: evidence\n"
        "    freshness: evidence\n",
        encoding="utf-8",
    )
    _git_add_commit(repo_root, summary, catalog)

    diagnostics = _evidence_catalog_coverage_diagnostics(repo_root=repo_root)

    assert diagnostics == [], diagnostics


def test_evidence_catalog_reports_uncovered_standalone_file(tmp_path: Path) -> None:
    """A standalone tracked evidence file with no catalog entry should be reported."""
    repo_root = _make_git_repo(tmp_path)
    (repo_root / "docs/context/evidence").mkdir(parents=True)
    standalone = repo_root / "docs/context/evidence/issue_1662_lidar_ppo_smoke_summary.json"
    standalone.write_text('{"status": "ok"}\n', encoding="utf-8")

    catalog = repo_root / "docs/context/catalog.yaml"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text("version: 1\nentries: []\n", encoding="utf-8")
    _git_add_commit(repo_root, standalone, catalog)

    diagnostics = _evidence_catalog_coverage_diagnostics(repo_root=repo_root)

    assert any("issue_1662_lidar_ppo_smoke_summary.json" in d.message for d in diagnostics), (
        diagnostics
    )


def test_evidence_catalog_covered_by_bundle_file_entry(tmp_path: Path) -> None:
    """A catalog entry for one file in a bundle covers the bundle."""
    repo_root = _make_git_repo(tmp_path)
    bundle = repo_root / "docs/context/evidence/issue_999_root_entry"
    bundle.mkdir(parents=True)
    f1 = bundle / "a.json"
    f2 = bundle / "b.json"
    f1.write_text('{"a": 1}\n', encoding="utf-8")
    f2.write_text('{"b": 2}\n', encoding="utf-8")

    catalog = repo_root / "docs/context/catalog.yaml"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text(
        "version: 1\nentries:\n"
        "  - path: docs/context/evidence/issue_999_root_entry/a.json\n"
        "    status: evidence\n"
        "    freshness: evidence\n",
        encoding="utf-8",
    )
    _git_add_commit(repo_root, f1, f2, catalog)

    # Both files are in the same bundle; one entry suffices.
    diagnostics = _evidence_catalog_coverage_diagnostics(repo_root=repo_root)

    assert diagnostics == [], diagnostics


def test_evidence_catalog_empty_evidence_dir_passes(tmp_path: Path) -> None:
    """An empty evidence directory (no tracked files) should pass cleanly."""
    repo_root = _make_git_repo(tmp_path)
    (repo_root / "docs/context/evidence").mkdir(parents=True)

    catalog = repo_root / "docs/context/catalog.yaml"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    catalog.write_text("version: 1\nentries: []\n", encoding="utf-8")
    _git_add_commit(repo_root, catalog)

    diagnostics = _evidence_catalog_coverage_diagnostics(repo_root=repo_root)

    assert diagnostics == [], diagnostics


def test_evidence_catalog_missing_catalog_file_passes(tmp_path: Path) -> None:
    """When the catalog does not exist yet, evidence coverage check should not crash."""
    repo_root = _make_git_repo(tmp_path)
    bundle = repo_root / "docs/context/evidence/issue_999_no_catalog"
    bundle.mkdir(parents=True)
    summary = bundle / "summary.json"
    summary.write_text('{"status": "ok"}\n', encoding="utf-8")
    _git_add_commit(repo_root, summary)

    # No catalog file at all.
    diagnostics = _evidence_catalog_coverage_diagnostics(repo_root=repo_root)

    # Without a catalog, every bundle is uncovered — should still report, not crash.
    assert any("issue_999_no_catalog" in d.message for d in diagnostics), diagnostics
