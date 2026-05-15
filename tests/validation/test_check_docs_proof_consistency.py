"""Tests for the docs/proof consistency checker."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from scripts.validation.check_docs_proof_consistency import (
    ChangedFile,
    _collect_diagnostics,
    _context_readme_link_diagnostics,
    _parse_name_status,
    _selected_files,
)


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


def test_context_note_same_basename_in_subdir_does_not_count_as_index_link() -> None:
    """A link to another note with the same basename should not index a top-level note."""
    diagnostics = _context_readme_link_diagnostics(
        [ChangedFile(status="A", path=Path("docs/context/issue_999_example.md"))],
        context_readme_text="- [Archived Example](archive/issue_999_example.md)\n",
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
